"""Live full-budget H3 CUDA collection for existing seven-team joint learning.

No CPU rule/decision callback runs during replay. Both the main value precheck
and the separate auxiliary scope must qualify before allocating a search cohort.
A bounded replay that fails to finish a natural game exports no learning rows.
"""
from copy import deepcopy
from hashlib import sha256
import math
import time

import numpy as np

from .device_pure_collection import _host_states, _raw_records
from .soa_training_jobs import adapt_training_jobs, policy_source_alias
from .soa_training_results import verified_training_games, _same_graph


PROFILE = dict(simulations=32, gumbel_candidates=16, terminal_horizon=3,
               teacher_mode='covered_value', opponent_mode='rule', max_depth=10,
               world_sampling='per_simulation', q_scale='natural_wdl')


def validate_h3_inputs(jobs, models):
    """Bind real jobs and value evidence; do not infer approval from flags."""
    from ..seven_runtime import value_precheck_ok
    from ..seven_covered_value import load_scope
    if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= 32:
        raise ValueError('One to32 explicit H3 training jobs required')
    metadata = adapt_training_jobs(jobs, models)
    if metadata.search['teacher_mode'] != 'covered_value':
        raise ValueError('H3 collection requires explicit qualified covered_value scope')
    for job in jobs:
        policies = {side: models[policy_source_alias(*job['policies'][side])] for side in ('a', 'b')}
        if not value_precheck_ok(job.get('value_precheck'), policies):
            raise ValueError('Actual current policy value-precheck required before H3 sampling')
    auxiliary, scope = load_scope(metadata.search['covered_value_source'])
    return metadata, auxiliary, scope


class _PostbatchPolicies:
    """Independent CUDA WDL forwards, NumPy checked, outside all device loops."""
    def __init__(self, bank, models, deadline):
        self.bank, self.models, self.deadline = bank, models, deadline
        self.cache, self.calls = {}, 0

    def policy(self, key):
        owner, base = self, self.models[key]
        class Policy:
            def scores(self, x, c): return base.scores(x, c)
            def wdl(self, x, is_actor): return owner.wdl(key, x, is_actor)
            def scores_value(self, x, c):
                probabilities = self.wdl(x, True)
                return self.scores(x, c), float(probabilities[2] - probabilities[0])
        return Policy()

    def wdl(self, key, x, is_actor):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('H3 independent postbatch value proof deadline')
        cache_key = key, bool(is_actor), x.tobytes()
        if cache_key not in self.cache:
            t, bank = self.bank.torch, self.bank
            prediction = bank.predict_device(t.as_tensor(x[None], device=bank.device),
                t.empty((1, 0, bank.candidate_dim), device=bank.device),
                t.empty((1, 0), dtype=t.bool, device=bank.device),
                t.tensor([bank.keys.index(key)], dtype=t.int64, device=bank.device),
                t.tensor([[float(is_actor)]], device=bank.device),
                value_only=t.ones(1, dtype=t.bool, device=bank.device))
            if bool(prediction.invalid_rows.any().cpu()):
                raise ValueError('Independent H3 WDL forward invalid')
            actual = prediction.wdl[0].cpu().numpy().copy()
            np.testing.assert_allclose(actual, self.models[key].wdl(x, is_actor), rtol=2e-5, atol=2e-5)
            self.cache[cache_key] = actual; self.calls += 1
        return self.cache[cache_key]


def _proof_references(initial, raw, policy_rows, metadata, models, inference,
                      auxiliary_models, auxiliary_inference, hashes, diagnostics, deadline,
                      root_proof_seconds=None):
    """Recompute every original full-budget root after the device batch ends."""
    from ...engine.duel_v2 import acting_side, apply_action, observe
    from ..league_rollout import clean
    from ..seven_lineup import encode
    from .. import seven_runtime
    from ..recovery_search import search
    from .soa_training_results import _equal_array
    rows = {(r['owner'], r['step']): r for r in policy_rows}
    states, refs = deepcopy(initial), {}
    policy_oracle = _PostbatchPolicies(inference, models, deadline)
    auxiliary_oracle = _PostbatchPolicies(auxiliary_inference, auxiliary_models, deadline)
    for record in raw:
        if time.monotonic() >= deadline:
            raise TimeoutError('Complete H3 search oracle budget exhausted')
        root_deadline = deadline if root_proof_seconds is None else min(
            deadline, time.monotonic()+root_proof_seconds)
        policy_oracle.deadline = auxiliary_oracle.deadline = root_deadline
        owner, step = record['game'], record['step']
        state, job = states[owner], metadata.jobs[owner]
        actor = acting_side(state)
        view = observe(state, actor, include_previews=False)
        legal = [v['action'] for v in view['legal_actions'] if v['action']['type'] != 'concede']
        x, c = encode(view, legal)
        expected_hash = np.frombuffer(sha256(x.tobytes()+c.tobytes()).digest(), dtype='>u4').astype(np.int64)
        _equal_array(hashes[owner, step], expected_hash)
        policies = {s: policy_oracle.policy(policy_source_alias(*job['policies'][s])) for s in ('a', 'b')}
        values = {s: auxiliary_oracle.policy(job['policies'][s][1]) for s in ('a', 'b')}
        seed = (1 << 256) | int.from_bytes(sha256(f'recovery:{job["seed"]}:{step}:{actor}'.encode()).digest(), 'big')
        reference = search(state, actor, policies, seed=seed, runtime=seven_runtime,
            simulations=32, gumbel_candidates=16, terminal_horizon=3, max_depth=10,
            deadline=time.time()+max(0, root_deadline-time.monotonic()),
            noise=job.get('exploration', job['training']), q_scale='natural_wdl',
            teacher_mode='covered_value', value_policies=values, world_sampling='per_simulation')
        if time.monotonic() >= root_deadline:
            raise TimeoutError('Original full-budget H3 root proof window exhausted')
        row = rows[owner, step]
        if not reference['complete']:
            raise ValueError('Independent H3 oracle did not complete original search budget')
        saved = {name:value[owner,step] for name,value in diagnostics.items()}
        width = len(legal)
        for name,expected in (('q',reference['root_diagnostics']['completed_q']),
                              ('prior',reference['root_prior']), ('mean',reference['mean_values'])):
            np.testing.assert_allclose(saved[name][:width],expected,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(saved['gumbel'][:width],reference['root_gumbel'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(saved['value'],reference['value'],rtol=1e-5,atol=1e-6)
        if saved['errors']:
            raise ValueError('Original H3 statistics snapshot failed')
        if not reference['complete']:
            raise ValueError('Independent H3 oracle did not complete original search budget')
        np.testing.assert_array_equal(row['visits'], reference['visits'])
        np.testing.assert_allclose(row['pi'], reference['pi'], rtol=1e-5, atol=1e-6)
        if row['selected'] != reference['search_choice']:
            raise ValueError('H3 CUDA/independent search selected different actions')
        # The verified adapter uses an exact copy of the GPU-rounded target.
        # Its independent numerical tolerance test above remains explicit.
        refs[owner, step] = dict(actions=deepcopy(legal), pi=row['pi'].copy(),
            visits=np.asarray(reference['visits']).copy(), diagnostics=dict(
                independent_full_budget=True, full_statistics_parity=True, reference_pi=np.asarray(reference['pi']).tolist(),
                reference_visits=np.asarray(reference['visits']).tolist(),
                reference_choice=reference['search_choice'], profile=dict(PROFILE)))
        chosen = int(row['selected'])
        record.update(actor=actor, chosen=chosen, requested=int(row['requested']), completed=int(row['completed']))
        if record['raw_action'] != legal[chosen]:
            raise ValueError('H3 raw action differs from the independently verified choice')
        states[owner] = clean(apply_action(state, actor, deepcopy(record['raw_action'])))
    return refs


class _H3AuditSink:
    """Archive final root statistics at every physical decision, on device."""
    def __init__(self, tape, statistics, search, *, decisions, max_bytes=512 << 20):
        from .device_pure_collection import _ValueAuditSink
        self.audit = _ValueAuditSink(tape,decisions=decisions)
        t, rows, width = tape.t, tape.rows, tape.max_actions
        self.t,self.rows,self.width,self.decisions=t,rows,width,decisions
        count=rows*decisions+rows
        estimated=(count+rows)*(width*8*4+8+1)
        if estimated>max_bytes:
            raise MemoryError('Full H3 per-root statistics exceed the declared archive budget')
        self.latest={n:t.zeros((rows,width),dtype=t.float64,device=tape.device)
                     for n in ('q','prior','mean','gumbel')}
        self.latest['value']=t.zeros(rows,dtype=t.float64,device=tape.device)
        self.latest['errors']=t.zeros(rows,dtype=t.bool,device=tape.device)
        self.saved={n:t.zeros((count,)+v.shape[1:],dtype=v.dtype,device=tape.device)
                    for n,v in self.latest.items()}
        original=statistics.snapshot
        def snapshot(*args,**kwargs):
            result=original(*args,**kwargs)
            q=statistics.completed_q(*args,**kwargs)
            active=kwargs.get('active')
            if active is None:raise ValueError('Exact root finalize mask required for H3 audit')
            values=dict(q=q.completed_q,prior=result.prior,mean=result.total/result.visits.clamp_min(1),
                        gumbel=search._noise,value=result.value,errors=result.error|q.error)
            for n,value in values.items():
                mask=active[:,None] if value.ndim==2 else active
                self.latest[n].copy_(t.where(mask,value,self.latest[n]))
            return result
        statistics.snapshot=snapshot

    @property
    def hashes(self):return self.audit.hashes
    @property
    def valid(self):return self.audit.valid
    @property
    def errors(self):return self.audit.errors
    def begin_device(self,*args):
        for value in (*self.latest.values(),*self.saved.values()):value.zero_()
        return self.audit.begin_device(*args)
    def append_device(self,roots,actors,steps,result,active):
        errors=self.audit.append_device(roots,actors,steps,result,active)
        at=self.audit.ids*self.decisions+steps.clamp(0,self.decisions-1)
        dest=self.t.where(active & (errors==0),at,self.audit.sentinels)
        for n,value in self.latest.items():self.saved[n].index_copy_(0,dest,value)
        return errors
    def finish_device(self,*args):return self.audit.finish_device(*args)
    def after_batch(self):
        return {n:v[:self.rows*self.decisions].cpu().numpy().reshape((self.rows,self.decisions)+v.shape[1:]).copy()
                for n,v in self.saved.items()}


class DeviceH3Collector:
    """Own a real H3 cohort; borrow immutable full native rules executor."""
    formal_training_qualified = False

    def __init__(self, executor, *, seconds=90, proof_seconds=60, max_decisions=800,
                 memory_limit_mib=14336, chunk_search_ticks=1, on_chunk=None):
        from ..batched_duel.native_soa_cuda import CudaSoARules
        if not isinstance(executor, CudaSoARules) or executor.execute_only or executor.device.type != 'cuda':
            raise TypeError('Actual full native CUDA executor required')
        if not {'teacher_continue', 'rule_decision'} <= set(executor.image.entries):
            raise ValueError('Frozen H3 continuation and rule-opponent entries required')
        for name, value, upper in (('seconds', seconds, 120), ('proof_seconds', proof_seconds, 120),
                                  ('max_decisions', max_decisions, 800), ('memory_limit_mib', memory_limit_mib, 14336)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError('Explicit bounded '+name+' required')
        self.executor, self.seconds, self.proof_seconds = executor, seconds, proof_seconds
        self.max_decisions, self.memory_limit_mib = max_decisions, memory_limit_mib
        if chunk_search_ticks is not None and (type(chunk_search_ticks) is not int
                or not 1 <= chunk_search_ticks <= 32):
            raise ValueError('Optional H3 chunk requires one to32 search ticks')
        if chunk_search_ticks is not None and seconds > 90:
            raise ValueError('Persistent H3 chunks require at most90 seconds per replay')
        if on_chunk is not None and not callable(on_chunk):
            raise TypeError('Optional H3 chunk callback must be callable')
        self.chunk_search_ticks, self.on_chunk = chunk_search_ticks, on_chunk
        self.last_diagnostics = None

    def _memory(self):
        free, total = self.executor.torch.cuda.mem_get_info()
        if total-free > self.memory_limit_mib << 20:
            raise MemoryError('H3 training exceeds whole-device memory budget')
        return total-free

    def collect(self, jobs, models):
        metadata, auxiliary_models, scope = validate_h3_inputs(jobs, models)
        self._original_monotonic_stop = time.monotonic()+min(j['deadline'] for j in jobs)-time.time()
        if min(j['deadline'] for j in jobs)-time.time() < self.seconds+self.proof_seconds+10:
            raise TimeoutError('Original deadline cannot contain bounded H3 replay and proof')
        owned = []
        self.last_diagnostics = dict(complete=False, profile=dict(PROFILE), optimizer_steps=0,
            main_model_versions=metadata.model_versions, auxiliary_versions=scope['models'])
        try:
            return self._collect(metadata, models, auxiliary_models, owned)
        finally:
            errors = []
            for obj in reversed(owned):
                close = getattr(obj, 'close', None)
                if callable(close):
                    try: close()
                    except BaseException as error: errors.append(repr(error))
                if hasattr(obj, 'physical_loop'): obj.physical_loop = None
                if hasattr(getattr(obj, 'inner', None), 'physical_loop'): obj.inner.physical_loop = None
            owned.clear(); obj = close = None
            import gc
            gc.collect()
            self.last_diagnostics['cleanup_errors'] = errors
            if errors: raise RuntimeError('H3 cohort cleanup failure: '+repr(errors))

    def _collect(self, metadata, models, auxiliary_models, owned):
        from ..batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
        from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
        from ..batched_duel.gpu_seven_observation import OBS_DIM
        from .device_tiled_soa_rules import DeviceTiledSoARules
        from .device_value_only_inference import DeviceValueOnlyInference
        from .device_raw_action_tape import DeviceRawActionTape
        from .device_info_key import DeviceInformationKey
        from .device_random import DevicePCG64
        from .device_root_random import DeviceRootRandomPlan
        from .device_seeds import DeviceRootSeeds
        from .device_statistics import DeviceStatisticsPool
        from .device_games import DeviceGameLoop
        from .device_physical_action_capture import RawActionPolicySink
        from .soa_covered_search import create_soa_covered_search_bundle
        from .soa_capacity import SoACapacityConfig, archive_capacities
        from .soa_frozen_collection import create_training_tape
        t, device = self.executor.torch, self.executor.device
        rows, decisions, width = len(metadata.jobs), self.max_decisions, 800
        capacity = SoACapacityConfig(roots=rows, rule_lanes=min(rows,32), inference_rows=min(rows,32),
            max_actions=width, max_decisions=decisions, state_reserve_slots=128,
            execution_seconds=self.seconds, proof_seconds=self.proof_seconds)
        def own(value): owned.append(value); self._memory(); return value
        inference = own(DeviceValueOnlyInference(models, device=str(device), max_rows=capacity.inference_tile,
            max_candidate_elements=capacity.inference_tile*width*1576))
        if inference.keys != metadata.inference_keys:
            raise ValueError('Actual H3 main inference source order changed')
        store = own(DeviceSparseSoAStore(self.executor.schema, capacity.state_slots,
            device=str(device), max_bytes=capacity.state_budget_mib << 20))
        actions = own(CompactDeviceGraphStore(slots=rows*6+128, node_pages=8192, edge_pages=8192,
            payload_pages=4096, node_page_bytes=4096, edge_page_bytes=4096, payload_page_bytes=4096,
            max_node_pages_per_slot=64, max_edge_pages_per_slot=64, max_payload_pages_per_slot=32, device=str(device)))
        archive = archive_capacities(capacity, decisions)
        raw = own(DeviceRawActionTape(rows, archive['max_records'], archive['node_capacity'],
            archive['edge_capacity'], archive['payload_capacity'], device=str(device)))
        rules = own(DeviceTiledSoARules(self.executor, store, actions, rows=rows, work_rows=capacity.lanes,
            inference_rows=capacity.inference_tile, max_actions=width, raw_tape=raw, seconds=self.seconds,
            candidate_int32=True, require_teacher=True, staged_teacher=True))
        random = own(DevicePCG64([np.random.PCG64(j['seed']).state for j in metadata.jobs],
            device=str(device), max_rows=rows, max_draws=width, max_choices=width))
        random_plan = DeviceRootRandomPlan(random, [j['seed'] for j in metadata.jobs],
            max_decisions=decisions, max_bytes=64 << 20)
        keys = own(DeviceInformationKey(device=str(device), max_rows=rows))
        seeds = own(DeviceRootSeeds(device=str(device), max_rows=rows))
        statistics = DeviceStatisticsPool(rows*33, width, device=str(device), max_batch=rows, max_path_len=10)
        noise = {j.get('exploration', j['training']) for j in metadata.jobs}
        if len(noise) != 1: raise ValueError('One explicit exploration setting per H3 cohort required')
        bundle = own(create_soa_covered_search_bundle(rules, inference, statistics, keys,
            auxiliary_scope=metadata.search['covered_value_source'], rows=rows, random=random,
            max_actions=width, inference_rows=capacity.inference_tile, noise=noise.pop(), row_work_seconds=self.seconds))
        tape = create_training_tape(rules, capacity, decisions=decisions, max_bytes=capacity.training_tape_budget_mib << 20)
        audit = _H3AuditSink(tape, statistics, bundle.search, decisions=decisions)
        chunked = self.chunk_search_ticks is not None
        if chunked:
            from .device_chunked_games import DeviceChunkedGameLoop
        loop_type = DeviceChunkedGameLoop if chunked else DeviceGameLoop
        options = dict(chunk_search_ticks=self.chunk_search_ticks) if chunked else {}
        loop = own(loop_type(rules, bundle.search, RawActionPolicySink(audit, raw), seeds,
            max_decisions=decisions, random_plan=random_plan, inner_max_ticks=20000,
            outer_max_ticks=decisions+2, seconds=self.seconds, device=str(device), defer_physical_presentation=True,
            **options))
        rules.bind_game_loop(loop)
        roots = rules.initialize_device(list(deepcopy(metadata.initializers)))
        t.cuda.synchronize()
        if bool(roots.errors.any().cpu()): raise ValueError('Actual H3 normal-mulligan initializer failed')
        initial = _host_states(store, roots)
        from ...engine.duel_v2 import new_game
        for actual, parameters in zip(initial, metadata.initializers): _same_graph(actual, new_game(**deepcopy(parameters)))
        policy_ids = t.tensor(metadata.model_indices, dtype=t.int64, device=device)
        aux_ids = t.tensor([[bundle.auxiliary_keys.index(metadata.keys[i][s]) for s in ('a','b')]
                            for i in range(rows)], dtype=t.int64, device=device)
        train = t.tensor([mask if job['training'] else (False,False)
                          for mask,job in zip(metadata.train_masks,metadata.jobs)], dtype=t.bool, device=device)
        job_seeds = t.tensor([j['seed'] for j in metadata.jobs], dtype=t.int64, device=device)
        # Allocate and warm exact inference shapes on all graph streams before capture.
        x = t.zeros((rows, inference.feature_dim), device=device)
        c = t.zeros((rows, width, inference.candidate_dim), device=device)
        valid = t.zeros((rows,width), dtype=t.bool, device=device); valid[:,0] = True
        streams = [t.cuda.current_stream(device), loop._outer.capture_stream,
                   loop._outer.body_stream, loop._inner.body_stream]
        if chunked:
            streams.extend(getattr(loop, name).body_stream
                for name in ('_start_gate', '_commit_gate', '_abort_gate'))
        for stream in streams:
            stream.wait_stream(t.cuda.current_stream(device))
            with t.cuda.stream(stream):
                inference.predict_device(x,c,valid,policy_ids[:,0].contiguous(),t.ones((rows,1),device=device))
                bundle.auxiliary.predict_device(x,c[:,:0],valid[:,:0],aux_ids[:,0].contiguous(),
                    t.ones((rows,1),device=device),value_only=t.ones(rows,dtype=t.bool,device=device))
            stream.synchronize()
        loop.capture(roots, job_seeds, policy_ids, aux_ids, train)
        self._memory()
        if min(j['deadline'] for j in metadata.jobs)-time.time() < self.seconds+self.proof_seconds:
            raise TimeoutError('H3 capture exhausted original deadline margin')
        start = time.monotonic()
        if chunked:
            from .device_h3_segments import replay_h3_chunks
            result = replay_h3_chunks(self, loop, metadata)
        else:
            loop.replay(); t.cuda.synchronize()
            result = loop.result_device()
        last = bundle.search.result_device()
        self.last_diagnostics.update(replay_seconds=time.monotonic()-start, steps=result.steps.cpu().tolist(),
            errors=result.errors.cpu().tolist(), complete_rows=result.complete.cpu().tolist(),
            truncated=result.truncated.cpu().tolist(), completed_simulations=last.completed.cpu().tolist(),
            requested_simulations=last.requested.cpu().tolist(), search_errors=last.errors.cpu().tolist(),
            actual_outer_ticks=int(loop._outer.counter[0].cpu()), device_used_bytes=self._memory())
        if (bool(result.errors.any().cpu()) or bool(last.errors.any().cpu())
                or not bool(result.complete.all().cpu()) or bool(result.truncated.any().cpu())):
            raise RuntimeError('H3 full-budget cohort incomplete; no joint learning labels')
        self.last_diagnostics.update(gpu_natural_complete=True, independently_verified=False,
            phase='postbatch_h3_proof')
        final = _host_states(store, result.states)
        raw_records = _raw_records(raw)
        values = tape.value_rows_after_batch(include_frozen=True)
        policies = tape.numerical_rows_after_batch(include_frozen=True)
        hash_rows = audit.hashes[:rows*decisions].cpu().numpy().reshape(rows,decisions,8)
        expected = {(i,s) for i,n in enumerate(self.last_diagnostics['steps']) for s in range(n)}
        actual = set(map(tuple,np.argwhere(audit.valid[:rows*decisions].cpu().numpy().reshape(rows,decisions))))
        if (actual != expected or len(raw_records) != len(expected)
                or bool(raw.errors_device.any().cpu()) or bool(audit.errors.any().cpu())):
            raise ValueError('H3 raw/hash coordinates incomplete or duplicated')
        deadline = min(self._original_monotonic_stop,
            time.monotonic()+min(j['deadline'] for j in metadata.jobs)-time.time())
        try:
            refs = _proof_references(initial,raw_records,policies,metadata,models,inference,
                                      auxiliary_models,bundle.auxiliary,hash_rows,audit.after_batch(),deadline,
                                      root_proof_seconds=self.proof_seconds)
            dataset = dict(complete=True,error_rows=0,truncated=False,optimizer_steps=0,
                search=metadata.search,model_versions=metadata.model_versions,policy=policies,values=values)
            games = verified_training_games(dataset,initial=initial,final=final,raw_records=raw_records,
                metadata=metadata,models=models,policy_reference=refs,deadline=deadline)
        except BaseException as error:
            self.last_diagnostics.update(proof_error=repr(error), proof_timed_out=isinstance(error,TimeoutError))
            raise
        self.last_diagnostics.update(complete=True,verified=True,raw_records=len(raw_records),
            independently_verified=True, phase='verified', policy_rows=len(policies),
            value_rows=len(values),auxiliary_scope=bundle.scope_receipt)
        return games
