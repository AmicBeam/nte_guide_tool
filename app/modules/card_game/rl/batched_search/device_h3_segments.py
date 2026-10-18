"""Deadline supervision for resumable CUDA H3 graphs in one live arena.

The device owns start/search/commit phases. The host replays the same bounded
chunk and reads aggregate progress/errors; it never chooses or advances roots.
All raw, policy, value, RNG and statistics archives remain device-resident until
natural completion. Partial roots, guard exits and deadlines export no labels.
"""
import time
from collections import deque


def _guard_failed(loop):
    # Result masks must not hide a real conditional-graph guard at a planned
    # phase boundary. These statuses are never cleared or retried here.
    for name in ('_outer', '_inner', '_chunk', '_start_gate', '_commit_gate'):
        runner = getattr(loop, name, None)
        errors = getattr(runner, 'errors', None)
        if errors is not None and bool(errors.any().cpu()):
            return True
    return False


def _progress(result, previous, max_decisions):
    errors = result.errors.cpu().tolist()
    steps = result.steps.cpu().tolist()
    complete = result.complete.cpu().tolist()
    truncated = result.truncated.cpu().tolist()
    if (not errors or len({len(errors), len(steps), len(complete), len(truncated)}) != 1
            or any(type(v) is not int or v != 0 for v in errors)
            or any(type(v) is not bool for v in complete + truncated)
            or any(truncated)):
        raise RuntimeError('H3 chunk rule/guard/truncation failure; no joint learning or resume')
    if any(type(v) is not int or not 0 <= v <= max_decisions for v in steps):
        raise ValueError('H3 chunk exceeded original physical decision ceiling')
    if previous is not None:
        if len(previous['steps']) != len(steps):
            raise ValueError('H3 persistent cohort size changed')
        for now, before, done, was_done in zip(steps, previous['steps'], complete, previous['complete_rows']):
            if now < before or now > before + 1 or (was_done and (now != before or not done)):
                raise ValueError('H3 chunk restarted or skipped a physical prefix')
    return dict(steps=steps, complete_rows=complete, errors=errors, truncated=truncated,
                natural_games=sum(complete))


def replay_h3_chunks(collector, loop, metadata, *, checkpoint=None, owned=None, initial=None):
    """Resume only intact device phases; return only after all natural games.

    A root can span many chunks without incrementing physical steps. No
    initialize/reset/restore is called after capture, and no partial tape export
    occurs here. Final full-prefix independent proof remains the collector's job.
    """
    absolute = min(job['deadline'] for job in metadata.jobs)
    stop = time.monotonic() + absolute - time.time()
    if _guard_failed(loop):
        raise RuntimeError('Prior H3 conditional-graph guard cannot be resumed')
    previous = _progress(loop.result_device(), None, collector.max_decisions)
    if (any(previous['steps']) or any(previous['complete_rows'])) and not (checkpoint and checkpoint.restored):
        raise ValueError('H3 collection must begin from its original captured initialization')
    if checkpoint is not None and checkpoint.restored and all(previous['complete_rows']):
        return loop.result_device()
    records = deque(maxlen=32)
    count, replay_seconds = 0, 0.0
    torch = collector.executor.torch
    while True:
        # Preserve the same original absolute deadline, plus a monotonic bound
        # so wall-clock corrections cannot extend this collection.
        remaining = min(absolute - time.time(), stop - time.monotonic())
        if remaining < collector.seconds + collector.proof_seconds:
            raise TimeoutError('Original H3 deadline exhausted; partial chunks have no learning labels')
        if checkpoint is not None:
            checkpoint.require_window(collector.seconds,margin=20,owned=owned,initial=initial,diag=previous)
        collector._memory()
        start = time.monotonic()
        loop.replay_chunk()
        torch.cuda.synchronize()
        count += 1
        elapsed = time.monotonic() - start
        replay_seconds += elapsed
        result = loop.result_device()
        diag = _progress(result, previous, collector.max_decisions)
        ticks = int(loop.chunk_search_ticks_device.cpu().item())
        if ticks < 0 or ticks > collector.chunk_search_ticks:
            raise ValueError('H3 chunk exceeded its declared search tick budget')
        diag.update(chunk_search_ticks=ticks, chunk_seconds=elapsed, device_used_bytes=collector._memory())
        records.append(diag)
        collector.last_diagnostics.update(diag, chunk_replays=count, replay_seconds=replay_seconds,
            chunk_search_tick_limit=collector.chunk_search_ticks, recent_chunks=list(records),
            complete=False, optimizer_steps=0)
        if elapsed > collector.seconds:
            collector.last_diagnostics['chunk_timed_out'] = True
            raise RuntimeError('H3 chunk exceeded original replay budget; no joint labels')
        if _guard_failed(loop):
            raise RuntimeError('Actual H3 conditional-graph guard failed; prefix cannot be resumed')
        if checkpoint is not None:
            path=checkpoint.save(owned,initial,diag)
            collector.last_diagnostics.update(checkpoint=path,persisted_chunks=checkpoint.total)
        if collector.on_chunk is not None:
            from copy import deepcopy
            collector.on_chunk(deepcopy(collector.last_diagnostics))
        if all(diag['complete_rows']):
            return result
        if checkpoint is not None and checkpoint.max_chunks is not None and checkpoint.written>=checkpoint.max_chunks:
            raise H3SegmentPaused(checkpoint.path)
        previous = diag


class H3SegmentPaused(RuntimeError):
    """A clean durable GPU chunk, including an unfinished root, was saved."""
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        super().__init__('H3 clean process boundary: ' + str(checkpoint))


def _owned_tensor_graph(roots, *, allow_cpu_reference=False):
    """Enumerate reviewed owned project objects, retaining storage/view topology.

    CUDA handles/code/streams and the stateless executor are never serialized.
    Inference wrappers ARE visited; only their immutable version-bound model
    banks are excluded. Keepalive, row-work and teacher objects remain included.
    """
    import torch
    from types import SimpleNamespace
    tensors, objects, topology, implementations = {}, {}, {}, {}
    skip = {'executor', 'module', 'alias_module', 'driver', 'runtime', 'graph',
            '_bootstrap_graph', '_abort_graph', 'capture_stream', 'body_stream', '_owned_streams',
            '_thread', '_setup_stream', '_capture_streams', 'schema', 't', 'torch'}
    def walk(value, path):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cuda' and not (allow_cpu_reference and value.device.type == 'cpu'):
                raise ValueError('Actual CUDA owned tensors required; CPU is a protocol fixture only')
            tensors[path] = value
            return
        if value is None or type(value)in(str,int,float,bool):
            topology[path]=dict(scalar=value,type=type(value).__name__);return
        if isinstance(value,(type,torch.device,torch.dtype))or callable(value):return
        permitted=type(value).__module__.startswith('app.modules.card_game.rl.') or (
            allow_cpu_reference and type(value).__module__.startswith('tests.'))
        if not isinstance(value,(dict,list,tuple,SimpleNamespace)) and not permitted:
            return
        if type(value).__name__ in ('CudaModule','CudaDriver','CudaRuntime'): return
        ident = id(value)
        if ident in objects:
            topology[path] = dict(alias=objects[ident]); return
        objects[ident] = path
        if isinstance(value,dict):
            topology[path] = dict(kind='dict', keys=[repr(k) for k in value])
            for key,item in value.items(): walk(item,path+'['+repr(key)+']')
        elif isinstance(value,(list,tuple)):
            topology[path] = dict(kind=type(value).__name__,length=len(value))
            for i,item in enumerate(value): walk(item,path+'['+str(i)+']')
        else:
            topology[path] = dict(kind=type(value).__module__+'.'+type(value).__qualname__)
            from hashlib import sha256
            from pathlib import Path
            import sys
            for cls in type(value).__mro__:
                module=sys.modules.get(cls.__module__)
                filename=getattr(module,'__file__',None)
                if filename and cls.__module__.startswith('app.modules.card_game.rl.'):
                    implementations[cls.__module__]=sha256(Path(filename).read_bytes()).hexdigest()
            inference = type(value).__module__.endswith(('device_inference','device_memory_bounded_inference','device_value_only_inference'))
            for name,item in vars(value).items():
                if name in skip: continue
                if type(value).__name__ == 'DeviceStatisticsPool' and name in ('_owner_thread','_stream'):
                    continue  # Rebind to the fresh process/thread; never restore an old owner identity.
                if inference and name in ('oracle','weights','legacy_weights'): continue
                # Exact conditional-guard clock exclusion. Every replay resets
                # this GPU-globaltimer clock; it is not search/game state.
                if type(value).__name__ == 'CudaConditionalWhile' and name == 'start': continue
                walk(item,path+'.'+name)
    for name,value in roots.items(): walk(value,name)
    topology['$implementations']=implementations
    groups, layout, storage = {}, {}, {}
    for path,value in tensors.items():
        owned = value.untyped_storage(); identity = owned._cdata
        group = groups.setdefault(identity,path)
        layout[path] = dict(storage=group, dtype=str(value.dtype),shape=list(value.shape),
            stride=list(value.stride()),offset=value.storage_offset(),bytes=owned.nbytes(),device=str(value.device))
        if group not in storage: storage[group] = owned
    return tensors, layout, topology, storage


def snapshot_h3_tensors(roots, *, allow_cpu_reference=False):
    """After synchronized clean chunk only. Does not grant any GPU qualification."""
    import torch
    loop=roots.get('loop')
    if loop is not None and _guard_failed(loop): raise RuntimeError('H3 guard prefix cannot be saved')
    for name,field in (('loop','_errors'),('search','errors'),('raw','_game_errors'),
                       ('tape','_game_errors'),('tape','_append_errors'),('audit','errors')):
        value=getattr(roots.get(name),field,None)
        if isinstance(value,torch.Tensor)and bool(value.any().cpu()):
            raise RuntimeError('Failed H3 semantic prefix cannot be checkpointed')
    tensors, layout, topology, storages = _owned_tensor_graph(roots,allow_cpu_reference=allow_cpu_reference)
    data = {}
    for path,storage in storages.items():
        value = tensors[path]
        raw = torch.empty(0,dtype=torch.uint8,device=value.device).set_(storage,0,(storage.nbytes(),),(1,))
        data[path] = raw.detach().cpu().clone()
    return dict(schema='h3_owned_tensor_storage_v1',layout=layout,topology=topology,storage=data)


def restore_h3_tensors(roots, saved, *, allow_cpu_reference=False):
    """Restore into newly captured buffers; never replace a Tensor or a lease."""
    import torch
    if saved.get('schema') != 'h3_owned_tensor_storage_v1': raise ValueError('Unknown H3 snapshot layout')
    tensors, layout, topology, storages = _owned_tensor_graph(roots,allow_cpu_reference=allow_cpu_reference)
    if saved['layout'] != layout or saved['topology'] != topology or set(saved['storage']) != set(storages):
        raise ValueError('H3 tensor path/layout/view/alias topology differs from frozen capture')
    for path,storage in storages.items():
        source = saved['storage'][path]
        if (not isinstance(source,torch.Tensor) or source.device.type!='cpu' or source.dtype!=torch.uint8
                or source.ndim!=1 or source.numel()!=storage.nbytes()):
            raise ValueError('H3 raw owned storage size/dtype differs')
    # Validate the entire graph before the first write, to avoid partial restore.
    for path,storage in storages.items():
        raw = torch.empty(0,dtype=torch.uint8,device=tensors[path].device).set_(storage,0,(storage.nbytes(),),(1,))
        raw.copy_(saved['storage'][path].to(raw.device))
    return True


def h3_proof_input_digest(value):
    from hashlib import sha256
    import json
    from .pure_checkpoint import _encode
    graph,arrays=_encode(value)
    digest=sha256(json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for name,array in sorted(arrays.items()):
        digest.update(name.encode());digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode());digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


class H3CheckpointSession:
    """Strict immutable process chunks. Failed semantic prefixes stay rejected."""
    def __init__(self, directory, resume, binding, *, continuation=None, max_chunks=None,process_stop=None):
        from pathlib import Path
        from .pure_checkpoint import read_payload
        if max_chunks is not None and (type(max_chunks)is not int or max_chunks<1):
            raise ValueError('Positive explicit H3 process chunk limit required')
        self.directory=(Path(directory)if directory is not None else Path(resume).parent).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        if (self.directory/'REJECTED.json').exists(): raise RuntimeError('Prior H3 failed prefix remains rejected')
        import json
        self.binding=json.loads(json.dumps(binding,allow_nan=False))
        self.continuation,self.max_chunks=continuation,max_chunks
        self.process_stop=process_stop
        self.proof_hits=self.proof_writes=0
        paths=sorted(self.directory.glob('chunk-*'))
        self.path=str(Path(resume).resolve())if resume else (str(paths[-1])if paths else None)
        self.payload=read_payload(self.path,self.binding,kind='h3_chunk',continuation=continuation)if self.path and not(
            self.directory/'completed-games').exists()else None
        self.restored=False;self.written=0
        self.total=0 if self.payload is None else self.payload['chunk_number']
    def restore(self, owned):
        if self.payload is not None:
            restore_h3_tensors(owned,self.payload['tensors'])
            self.restored=True
        return self.payload
    def save(self, owned, initial, diag):
        from .pure_checkpoint import write_payload
        self.total+=1;self.written+=1
        self.path=write_payload(self.directory/('chunk-%08d'%self.total),
            dict(tensors=snapshot_h3_tensors(owned), initial=initial,diagnostics=diag,chunk_number=self.total),
            self.binding,kind='h3_chunk',continuation=self.continuation)
        return self.path
    def reject(self,error):
        import json
        path=self.directory/'REJECTED.json'
        if not path.exists():
            with path.open('x')as out:json.dump(dict(error=repr(error),learning_rejected=True),out)
    def require_window(self,seconds,*,margin=10,owned=None,initial=None,diag=None):
        if self.process_stop is not None and time.monotonic()+seconds+margin>self.process_stop:
            if self.path is None and owned is not None:self.save(owned,initial,diag)
            raise H3SegmentPaused(self.path or str(self.directory))
    def proof(self,owner,step,inputs):
        from .pure_checkpoint import read_payload
        path=self.directory/('proof-%04d-%04d'%(owner,step))
        if not path.exists():return None
        data=read_payload(path,self.binding,kind='h3_root_proof',continuation=self.continuation)
        if data['input_digest']!=h3_proof_input_digest(inputs):
            raise ValueError('Cached H3 proof input/GPU statistics differ from original root')
        self.proof_hits+=1
        return data['reference']
    def save_proof(self,owner,step,inputs,reference):
        from .pure_checkpoint import write_payload
        write_payload(self.directory/('proof-%04d-%04d'%(owner,step)),
            dict(input_digest=h3_proof_input_digest(inputs),reference=reference),self.binding,
            kind='h3_root_proof',continuation=self.continuation)
        self.proof_writes+=1
    def completed_games(self):
        from .pure_checkpoint import read_payload
        path=self.directory/'completed-games'
        return read_payload(path,self.binding,kind='h3_verified_games',continuation=self.continuation)if path.exists()else None
    def save_games(self,games):
        from .pure_checkpoint import write_payload
        return write_payload(self.directory/'completed-games',games,self.binding,
            kind='h3_verified_games',continuation=self.continuation)
