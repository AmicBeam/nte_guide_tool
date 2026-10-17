"""CUDA-only batched native calls with authoritative persistent SoA columns.

Transient arguments/projections are separate from game storage. This module
never calls host game encoding, decoding, formal rules, or the CPU executor.
"""
from dataclasses import dataclass

from .cuda_vm import VmLimits
from .native_soa_source import emit_native_soa_source, source_identity
from .record_soa import TorchRecordSoABatch, MISSING
from .device_rule_protocol import DeviceMeta


_SIGNATURE = ['int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'int32',
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'int32', 'int32', 'int32', 'int32',
    'int32*', 'int32*', 'int32*', 'int64*', 'int32', 'int32', 'int32', 'int32', 'int32',
    'int32', 'int32*', 'int32', 'int32*', 'int32', 'int64*', 'int32*',
    'int64*', 'float64*', 'int32*', 'uint8*']+['int32*']*7+['uint8*']
_EXPORT_SIGNATURE = (['int32*','int32*','uint8*','int32*','int32*']
    + ['int32*','int32*','uint8*']+['int32']*3
    + ['int64*','float64*','int32*','uint8*']+['int32*']*5
    + ['int32*','int32*','uint8*','int32*','int32*','int32*','int32*']
    + ['int32*','uint8*']+['int32']*7)
_SELECT_SIGNATURE = (_EXPORT_SIGNATURE[:20]+['int32*','int32','int32*','int32*','int32*','uint8*']+['int32']*4)


@dataclass(frozen=True)
class CudaSoACall:
    state: TorchRecordSoABatch
    scratch: object
    outputs: object
    statuses: object
    roots: object
    event: object
    keepalive: tuple
    context: tuple

    def typed_result(self):
        """Device root selection; statuses must gate downstream consumption."""
        return TorchRecordSoABatch(self.state.schema, self.state.buffers, self.roots)


@dataclass(frozen=True)
class CudaSoAMeta(DeviceMeta):
    keepalive: tuple = ()


def _validate_soa_stack(stack_bytes, execute_only):
    if type(execute_only) is not bool:
        raise TypeError('Explicit diagnostic kernel selection required')
    if type(stack_bytes) is not int or not (
            131072 <= stack_bytes <= 1024**2 or (execute_only and stack_bytes == 65536)):
        raise ValueError('Full SoA requires 128 KiB to 1 MiB stack; execute-only diagnostics also allow 64 KiB')


class CudaSoARules:
    execution_mode = 'native_domain_soa_aot_cuda'
    interpreter_fallback = False
    runtime_instruction_fetch = False
    supports_full_duel_rules = False
    gpu_qualified = False

    def __init__(self, image, schema, *, device='cuda', limits=VmLimits(), threads=1,
                 stack_bytes=131072,stack_runs=False,native_functions=False,native_values=False,
                 execute_only=False, separate_compilation=False, indexed_attributes=False,
                 direct_runtime_calls=False, indexed_container_methods=False,
                 direct_type_queries=False):
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU fallback is forbidden')
        if type(threads) is not int or not 1 <= threads <= 1024:
            raise ValueError('Bounded CUDA block size required')
        _validate_soa_stack(stack_bytes, execute_only)
        if type(indexed_attributes) is not bool or (indexed_attributes and not separate_compilation):
            raise ValueError('Runtime attribute indexing requires separate compilation')
        for option in (direct_runtime_calls, indexed_container_methods, direct_type_queries):
            if type(option) is not bool or (option and not separate_compilation):
                raise ValueError('Runtime specialization requires separate compilation')
        if type(separate_compilation) is not bool or separate_compilation and not native_functions:
            raise ValueError('Separate compilation requires complete native functions')
        self.torch, self.device = torch, torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        image.verify()
        self.image, self.schema, self.limits, self.threads = image, schema, limits, threads
        source, self.native_plan = emit_native_soa_source(image, schema, limits,
            stack_runs=stack_runs, native_functions=native_functions, native_values=native_values)
        self.stack_runs=stack_runs
        self.native_functions=native_functions
        self.native_values=native_values
        self.rule_source_sha256 = source_identity(source)
        self.execute_only = execute_only
        self.separate_compilation = separate_compilation
        self.indexed_attributes = indexed_attributes
        self.direct_runtime_calls = direct_runtime_calls
        self.indexed_container_methods = indexed_container_methods
        self.direct_type_queries = direct_type_queries
        if execute_only:
            from .native_soa_source import execute_only_source
            source = execute_only_source(source)
        self.source_sha256 = source_identity(source)
        from .cuda_runtime import CudaModule
        from .device_rule_results import _SPEC, _VALUE_PREFIX
        signatures = {'vm_soa_execute': _SIGNATURE,
            'soa_export_projection': _EXPORT_SIGNATURE,
            'soa_decode_metadata': _SPEC+['int64*','float64*','int32*','uint8*']+['int32*']*6,
            'soa_decode_pair': _VALUE_PREFIX+['int32*']*3+['int32']*4,
            'soa_decode_float': _VALUE_PREFIX+['float64*','int32*']+['int32']*4,
            'soa_select_result': _SELECT_SIGNATURE,
            'vm_clock':['int64*'],
            'vm_compact': ['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
                'int32','int32','int32','int32*','int32*','int32*','uint8*','int32*','int32*',
                'int32*','int32*','int32*','int32','int32','int32','int32','int32']}
        if execute_only:
            signatures = {name: signatures[name] for name in ('vm_soa_execute', 'vm_clock')}
        units = None
        if separate_compilation:
            from .native_compile_units import split_native_source
            compiled_units = list(split_native_source(source))
            if indexed_attributes:
                from .native_runtime_index import index_runtime_attributes
                compiled_units[0] = index_runtime_attributes(compiled_units[0], image)
            if direct_runtime_calls or direct_type_queries:
                from .native_runtime_calls import transform_runtime_calls
                compiled_units[0] = transform_runtime_calls(compiled_units[0], image,
                    function_calls=direct_runtime_calls, scalar_builtins=direct_runtime_calls,
                    type_builtins=direct_type_queries, mapping_get=direct_type_queries)
            if indexed_container_methods:
                from .native_effect_queries import specialize_effect_queries
                compiled_units[0] = specialize_effect_queries(compiled_units[0], image)
            units = [(unit.name, unit.source) for unit in compiled_units]
        self.module = CudaModule(source, signatures, compilation_units=units,
            device=str(self.device), binary_format='cubin')
        self.module.ensure_stack(stack_bytes)
        self.constants = {key: torch.tensor(getattr(image, key).copy(), device=self.device)
                          for key in ('nodes', 'edges', 'payload', 'functions', 'classes', 'modules')}
        self.constants['instructions'] = torch.full((1, 4), MISSING, dtype=torch.int32, device=self.device)
        self._buffers = self.constants
        self._result_paths = {}
        for path in ((), (0,), (1,), (1,0,0), (1,0,1), (1,1,0), (1,1,1)):
            self.prepare_result_path(path)
        self.closed = False

    def prepare_result_path(self, path):
        """Upload static packet indices before capture, retaining their lifetime."""
        if type(path) not in (list,tuple) or len(path)>8 or any(type(i) is not int or not 0<=i<2**31 for i in path):
            raise ValueError('Bounded static packet path required')
        key=tuple(path)
        if key not in self._result_paths:
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Prepare static SoA result paths before CUDA graph capture')
            self._result_paths[key]=self.torch.tensor(key or (0,),dtype=self.torch.int32,device=self.device)
        return self._result_paths[key]

    def clock_ns(self):
        """Explicit setup/diagnostic clock boundary, never a captured search step."""
        out=self.torch.empty(1,dtype=self.torch.int64,device=self.device)
        event=self.module.launch('vm_clock',[out],grid=1,block=1)
        event.synchronize()
        return int(out.cpu()[0])

    def _tensor(self, value, dtype, shape, name):
        if (not isinstance(value, self.torch.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != tuple(shape) or not value.is_contiguous()):
            raise ValueError('Exact contiguous CUDA '+name+' required')

    def prepare_workspace(self,state,scratch):
        """One native language/context workspace for sequential captured sites."""
        torch,rows,schema=self.torch,state.rows,self.schema
        if getattr(scratch,'_soa_context',None) is not None:
            if scratch._soa_state is not state:
                raise ValueError('Scratch is already attached to different columns')
            return scratch._soa_workspace
        from types import SimpleNamespace
        from .vm_soa_views import value_cache_extent
        make=lambda shape,dtype=torch.int32:torch.zeros(shape,dtype=dtype,device=self.device)
        views=torch.full((schema.total_objects,rows),MISSING,dtype=torch.int32,device=self.device)
        cache=torch.full((value_cache_extent(schema),rows),MISSING,dtype=torch.int32,device=self.device)
        context=(views,cache,make(views.shape),make(views.shape),make((len(self.image.nodes),rows)))
        stride=14+self.limits.locals+self.limits.stack+self.limits.cells
        workspace=SimpleNamespace(context=context,
            frames=make((rows,self.limits.frames,stride)),
            marks=make((rows,scratch.layout.max_nodes+len(self.image.nodes)),torch.int64),
            queue=make((rows,scratch.layout.max_nodes)))
        scratch._soa_context=context;scratch._soa_state=state;scratch._soa_workspace=workspace
        return workspace

    def reset_workspace(self,state,scratch):
        workspace=self.prepare_workspace(state,scratch)
        views,cache,clones,queue,constants=workspace.context
        views.fill_(MISSING);cache.fill_(MISSING);clones.zero_();queue.zero_();constants.zero_()
        scratch.counts.zero_();scratch.counts[:,0]=1;scratch.roots.zero_();scratch.nodes[:,0].zero_()
        return workspace

    def launch(self, entry, state, scratch, argument_refs, typed_arguments, *, deadlines=None, active=None):
        """All state buffers and argument refs remain on device during execution.

        typed_arguments[B,nargs] holds declared global object refs or zero for
        transient arguments. scratch contains ONLY arguments and language data.
        """
        from .cuda_vm import DeviceGraph
        if not isinstance(state, TorchRecordSoABatch) or state.schema.identity != self.schema.identity:
            raise ValueError('Matching typed device state required')
        if not isinstance(scratch, DeviceGraph) or scratch.rows != state.rows or state.rows < 1:
            raise ValueError('Matching nonempty transient device scratch required')
        if entry not in self.image.entries:
            raise ValueError('Unknown native SoA entry')
        torch, rows = self.torch, state.rows
        if argument_refs.ndim != 2 or argument_refs.shape[1] > 256:
            raise ValueError('Bounded [B,nargs] references required')
        nargs = argument_refs.shape[1]
        self._tensor(argument_refs, torch.int32, (rows, nargs), 'argument references')
        self._tensor(typed_arguments, torch.int32, (rows, nargs), 'typed references')
        for key, dtype in (('i64', torch.int64), ('f64', torch.float64), ('i32', torch.int32), ('u8', torch.uint8)):
            self._tensor(state.buffers[key], dtype, (self.schema.lengths[key], rows), key+' columns')
        empty = lambda shape, dtype=torch.int32: torch.zeros(shape, dtype=dtype, device=self.device)
        workspace=self.prepare_workspace(state,scratch)
        frames=workspace.frames
        outputs = torch.full((rows,), MISSING, dtype=torch.int32, device=self.device)
        statuses, roots = empty((rows, 4)), empty((rows,))
        if deadlines is None:
            deadlines = empty((rows,), torch.int64)
        if active is None:
            active = torch.ones(rows, dtype=torch.uint8, device=self.device)
        self._tensor(active, torch.uint8, (rows,), 'active mask')
        self._tensor(deadlines, torch.int64, (rows,), 'deadlines')
        constants, schema = self.constants, self.schema
        cnc = len(constants['nodes'])
        gc_marks = workspace.marks
        gc_marks.zero_()
        gc_queue = workspace.queue
        context = workspace.context
        views,cache,clone_map,clone_queue,constant_adopt=context
        args = [scratch.nodes, scratch.edges, scratch.payload, scratch.counts, argument_refs, nargs,
            constants['nodes'], constants['edges'], constants['payload'], constants['instructions'],
            constants['functions'], len(constants['functions']), cnc, len(constants['edges']),
            len(constants['payload']), frames, outputs, statuses, deadlines, rows,
            scratch.layout.max_nodes, scratch.layout.max_edges, scratch.layout.max_payload_bytes,
            int(self.image.entries[entry]), self.limits.steps, constants['classes'], len(constants['classes']),
            constants['modules'], len(constants['modules']), gc_marks, gc_queue,
            *[state.buffers[k] for k in ('i64', 'f64', 'i32', 'u8')],
            views, cache, clone_map, clone_queue, constant_adopt, typed_arguments, roots, active]
        event = self.module.launch('vm_soa_execute', args,
                                   grid=(rows+self.threads-1)//self.threads, block=self.threads,
                                   capture_safe=True)
        return CudaSoACall(state, scratch, outputs, statuses, roots, event,
                           (frames, deadlines, gc_marks, gc_queue, views, cache,
                            clone_map, clone_queue, constant_adopt, argument_refs, typed_arguments, active),
                           (views, cache, clone_map, clone_queue, constant_adopt))

    def _require_full_module(self):
        if getattr(self, 'execute_only', False):
            raise RuntimeError('Execute-only diagnostics cannot export/decode/compact device results')

    def export_projection(self, call, *, max_nodes=16384, max_edges=32768,
                          max_payload_bytes=131072, active=None):
        """Materialize a public result only; native source refuses Game records."""
        self._require_full_module()
        from types import SimpleNamespace
        from .cuda_vm import DeviceGraph
        from .schema import GraphLayout
        torch, rows = self.torch, call.state.rows
        for value in (max_nodes, max_edges, max_payload_bytes):
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError('Bounded projection capacity required')
        make = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=self.device)
        layout = GraphLayout(rows, max_nodes, max_edges, max_payload_bytes)
        dest = DeviceGraph(layout, make((rows,max_nodes,4),torch.int32),
            make((rows,max_edges,2),torch.int32), make((rows,max_payload_bytes),torch.uint8),
            make((rows,3),torch.int32), make((rows,),torch.int32))
        if active is None:
            active = (call.statuses[:,0] == 0).to(torch.uint8)
        self._tensor(active, torch.uint8, (rows,), 'projection active mask')
        scratch, c = call.scratch, self.constants
        capacity = scratch.layout.max_nodes+len(c['nodes'])
        mapping = make((rows,capacity),torch.int32)
        queue = make((rows,capacity),torch.int32)
        errors = make((rows,),torch.int32)
        args = [scratch.nodes,scratch.edges,scratch.payload,scratch.counts,call.outputs,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            *[call.state.buffers[k] for k in ('i64','f64','i32','u8')], *call.context,
            dest.nodes,dest.edges,dest.payload,dest.counts,dest.roots,mapping,queue,errors,active,
            rows,scratch.layout.max_nodes,scratch.layout.max_edges,scratch.layout.max_payload_bytes,
            max_nodes,max_edges,max_payload_bytes]
        event = self.module.launch('soa_export_projection',args,
            grid=(rows+self.threads-1)//self.threads,block=self.threads,capture_safe=True)
        statuses = call.statuses.clone()
        statuses[:,0].bitwise_or_(errors)
        return SimpleNamespace(arena=dest,executor=self,outputs=dest.roots,statuses=statuses,event=event,
                               mapping=mapping,keepalive=(call,queue,errors,active))

    def metadata(self, actor_call, finished_call, active):
        self._require_full_module()
        torch, rows = self.torch, finished_call.state.rows
        self._tensor(active, torch.uint8, (rows,), 'metadata active mask')
        make = lambda dtype: torch.zeros(rows,dtype=dtype,device=self.device)
        actors, phases, winners = make(torch.int64), make(torch.int32), make(torch.int64)
        finished, versions, errors = make(torch.uint8), make(torch.int64), make(torch.int32)
        input_errors = actor_call.statuses[:,0] | finished_call.statuses[:,0]
        arena, c, state = finished_call.scratch, self.constants, finished_call.state
        args = [arena.nodes,arena.edges,arena.payload,arena.counts,arena.roots,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            actor_call.outputs,finished_call.outputs,input_errors,active,
            actors,phases,winners,finished,versions,errors,rows,
            arena.layout.max_nodes,arena.layout.max_edges,arena.layout.max_payload_bytes,
            *[state.buffers[k] for k in ('i64','f64','i32','u8')],*finished_call.context,state.roots]
        self.module.launch('soa_decode_metadata',args,
            grid=(rows+self.threads-1)//self.threads,block=self.threads,capture_safe=True)
        return CudaSoAMeta(actors,phases,winners,finished.bool(),versions,errors,
                          (actor_call,finished_call,input_errors,finished,active))

    def select_result(self, call, path, *, active=None):
        self._require_full_module()
        from dataclasses import replace
        torch, rows = self.torch, call.state.rows
        path_tensor = self.prepare_result_path(path)
        selected = torch.empty(rows,dtype=torch.int32,device=self.device)
        typed, errors = torch.zeros_like(selected), torch.zeros_like(selected)
        if active is None:
            active = (call.statuses[:,0]==0).to(torch.uint8)
        self._tensor(active,torch.uint8,(rows,),'selection active mask')
        arena, c = call.scratch, self.constants
        args = [arena.nodes,arena.edges,arena.payload,arena.counts,call.outputs,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            *[call.state.buffers[k] for k in ('i64','f64','i32','u8')],*call.context,
            path_tensor,len(path),selected,typed,errors,active,rows,
            arena.layout.max_nodes,arena.layout.max_edges,arena.layout.max_payload_bytes]
        event = self.module.launch('soa_select_result',args,
            grid=(rows+self.threads-1)//self.threads,block=self.threads,capture_safe=True)
        statuses=call.statuses.clone();statuses[:,0].bitwise_or_(errors)
        return replace(call,outputs=selected,roots=typed,statuses=statuses,event=event,
                       keepalive=call.keepalive+(path_tensor,errors,active))

    def read_result(self, call, active, *, pair=False):
        """Decode scalar or packet references, entirely on device."""
        self._require_full_module()
        from types import SimpleNamespace
        torch = self.torch
        arena = getattr(call,'arena',None)
        if arena is None:
            arena = call.scratch
        rows, c = arena.rows, self.constants
        self._tensor(active, torch.uint8, (rows,), 'result active mask')
        errors = torch.zeros(rows,dtype=torch.int32,device=self.device)
        dtype = torch.int32 if pair else torch.float64
        first = torch.zeros(rows,dtype=dtype,device=self.device)
        second = torch.zeros_like(first) if pair else None
        input_errors=call.statuses[:,0].contiguous()
        args = [arena.nodes,arena.edges,arena.payload,arena.counts,
            c['nodes'],c['edges'],c['payload'],len(c['nodes']),len(c['edges']),len(c['payload']),
            call.outputs,input_errors,active,first]
        if pair:
            args.append(second)
        args.extend([errors,rows,arena.layout.max_nodes,arena.layout.max_edges,arena.layout.max_payload_bytes])
        self.module.launch('soa_decode_pair' if pair else 'soa_decode_float',args,
            grid=(rows+self.threads-1)//self.threads,block=self.threads,capture_safe=True)
        return SimpleNamespace(first=first,second=second,values=first,errors=errors,
                               keepalive=(call,input_errors,active,first,second,errors))

    def compact_transient(self, arena, roots, active):
        """Copy selected action/projection subgraphs; state is never passed here."""
        self._require_full_module()
        from types import SimpleNamespace
        from .cuda_vm import DeviceGraph
        torch, rows, c = self.torch, arena.rows, self.constants
        self._tensor(roots,torch.int32,(rows,),'transient roots')
        self._tensor(active,torch.uint8,(rows,),'transient active mask')
        layout=arena.layout
        capacity=layout.max_nodes+len(c['nodes'])
        dest=DeviceGraph(layout,torch.zeros_like(arena.nodes),torch.zeros_like(arena.edges),
            torch.zeros_like(arena.payload),torch.zeros_like(arena.counts),torch.zeros_like(arena.roots))
        mapping=torch.empty((rows,capacity),dtype=torch.int32,device=self.device)
        queue=torch.empty_like(mapping)
        errors=torch.empty(rows,dtype=torch.int32,device=self.device)
        # Inactive lanes use the valid None header at zero; downstream masks
        # discard their compact result, without reading an invalid action ref.
        selected=torch.where(active.bool(),roots,0).contiguous()
        args=[arena.nodes,arena.edges,arena.payload,arena.counts,c['nodes'],c['edges'],c['payload'],
            len(c['nodes']),len(c['edges']),len(c['payload']),selected,dest.nodes,dest.edges,dest.payload,
            dest.counts,dest.roots,mapping,queue,errors,rows,layout.max_nodes,layout.max_edges,
            layout.max_payload_bytes,capacity]
        self.module.launch('vm_compact',args,grid=(rows+self.threads-1)//self.threads,
            block=self.threads,capture_safe=True)
        masked_errors=torch.where(active.bool(),errors,0)
        return SimpleNamespace(arena=dest,mapping=mapping,errors=masked_errors,
            keepalive=(arena,queue,selected,active,errors))

    def close(self):
        self.module.close()
        self.closed = True
