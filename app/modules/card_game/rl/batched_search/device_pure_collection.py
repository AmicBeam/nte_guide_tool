"""Live CUDA argmax collection for the existing value-calibration stage.

No search labels or qualification are manufactured. The complete device cohort
finishes before the CPU rule oracle runs. Failed/truncated batches export no
learning rows. An executor is borrowed; all per-batch arenas are owned here.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import time

import numpy as np

from .soa_pure_jobs import adapt_pure_jobs
from .soa_pure_results import verified_pure_calibration_games
from .soa_training_jobs import policy_source_alias


class _ValueAuditSink:
    def __init__(self, tape, *, decisions):
        self.tape = tape
        t, device, rows = tape.t, tape.device, tape.rows
        self.rows, self.decisions = rows, decisions
        self.ids = t.arange(rows, device=device, dtype=t.int64)
        self.sentinels = rows * decisions + self.ids
        self.valid = t.zeros(rows * decisions + rows, device=device, dtype=t.bool)
        self.hashes = t.zeros((rows * decisions + rows, 8), device=device, dtype=t.int64)
        self.chosen = t.full((rows * decisions + rows,), -1, device=device, dtype=t.int64)
        self.errors = t.zeros(rows, device=device, dtype=t.int32)

    def begin_device(self, *args):
        self.valid.zero_(); self.hashes.zero_(); self.chosen.fill_(-1); self.errors.zero_()
        return self.tape.begin_device(*args)

    def append_device(self, roots, actors, steps, result, active):
        t = self.tape.t
        at = self.ids * self.decisions + steps.clamp(0, self.decisions - 1)
        bad = active & ((steps < 0) | (steps >= self.decisions) | self.valid[at]
                        | (roots.owners != self.ids))
        self.errors.bitwise_or_(bad.to(t.int32))
        take = active & ~bad
        dest = t.where(take, at, self.sentinels)
        self.valid.index_copy_(0, dest, take)
        self.hashes.index_copy_(0, dest, result.information_keys)
        self.chosen.index_copy_(0, dest, result.actions)
        return self.tape.append_device(roots, actors, steps, result, active) | self.errors

    def finish_device(self, roots, winners, steps, complete, errors):
        return self.tape.finish_device(roots, winners, steps, complete, errors | self.errors)


def _host_states(store, leases):
    from ..batched_duel.record_soa import unpack_game_states
    graph = store.gather_device(leases, shared_reads=True).to_host()
    if bool(leases.errors.any().cpu()):
        raise ValueError('Invalid final/initial CUDA state lease')
    return unpack_game_states(graph)


def _raw_records(raw):
    """Decode only after replay; raw action coordinates stay device-authored."""
    from ..batched_duel.device_rule_calls import _empty_graph
    from ..batched_duel.schema import GraphLayout, PackedStateBatch
    from ..batched_duel.codec import unpack_states
    count = int(raw._record_count.cpu())
    actions = []
    for start in range(0, count, 8):
        stop = min(start + 8, count)
        layout = GraphLayout(stop-start, 16384, 32768, 131072)
        graph = _empty_graph(raw.t, layout, raw.device)
        errors = raw.gather_into_device(raw.t.arange(start, stop, device=raw.device, dtype=raw.t.int64),
            graph, raw.t.ones(stop-start, device=raw.device, dtype=raw.t.bool))
        if bool(errors.any().cpu()):
            raise ValueError('Raw action archive could not be decoded after replay')
        arrays = {name: np.ascontiguousarray(getattr(graph, name).cpu().numpy())
                  for name in ('nodes', 'edges', 'payload', 'counts', 'roots')}
        actions.extend(unpack_states(PackedStateBatch(layout, **arrays)))
    fields = {name: getattr(raw, '_rec_' + name)[:count].cpu().tolist()
              for name in ('game', 'step', 'owner', 'seed', 'actor', 'version')}
    return [dict({name: values[i] for name, values in fields.items()}, raw_action=action)
            for i, action in enumerate(actions)]


def _references(initial, records, hashes, choices, metadata, models, inference, deadline):
    """Postbatch CUDA forwards bound to each original GPU information hash."""
    from ...engine.duel_v2 import observe, apply_action, acting_side
    from ..league_rollout import clean
    from ..seven_lineup import encode
    t = inference.torch
    states, result = deepcopy(initial), {}
    for record in records:
        if time.monotonic() >= deadline:
            raise TimeoutError('Pure postbatch proof deadline')
        owner, step = record['game'], record['step']
        state = states[owner]; side = acting_side(state)
        view = observe(state, side, include_previews=False)
        actions = [v['action'] for v in view['legal_actions'] if v['action']['type'] != 'concede']
        x, c = encode(view, actions)
        actual_hash = hashes[owner, step]
        want = np.frombuffer(hashlib.sha256(x.tobytes() + c.tobytes()).digest(), dtype='>u4').astype(np.int64)
        if not np.array_equal(actual_hash, want):
            raise ValueError('Original GPU public-input hash differs from postbatch oracle')
        alias = policy_source_alias(*metadata.jobs[owner]['policies'][side])
        n = len(actions)
        prediction = inference.predict_device(t.as_tensor(x[None], device=inference.device),
            t.as_tensor(c[None], device=inference.device), t.ones((1, n), device=inference.device, dtype=t.bool),
            t.tensor([inference.keys.index(alias)], device=inference.device, dtype=t.int64),
            t.ones((1, 1), device=inference.device))
        logits = prediction.logits[0].cpu().numpy().copy()
        selected = int(choices[owner, step])
        if bool(prediction.invalid_rows.any().cpu()) or selected != int(logits.argmax()):
            raise ValueError('Independent CUDA argmax disagrees with actual device choice')
        result[owner, step] = dict(actions=actions, x=x, c=c, logits=logits, selected=selected,
            info_hash=actual_hash.copy(), source_alias=alias, model_version=models[alias].version,
            owner=owner, step=step, actor=int(side == 'b'), job_seed=metadata.jobs[owner]['seed'],
            version=state['version'], encode_error=0, inference_error=0, requested=0, completed=0,
            inference_backend='cuda')
        states[owner] = clean(apply_action(state, side, deepcopy(record['raw_action'])))
    return result


class DevicePureCollector:
    """Actual GPU warm-value sampler; not a formal Gumbel training driver."""
    formal_training_qualified = False

    def __init__(self, executor, *, seconds=90, proof_seconds=30, max_decisions=800,
                 memory_limit_mib=14336, segment_decisions=None, on_segment=None):
        from ..batched_duel.native_soa_cuda import CudaSoARules
        if not isinstance(executor, CudaSoARules) or executor.execute_only or executor.device.type != 'cuda':
            raise TypeError('Actual full native CUDA rules required')
        for name, value, maximum in (('seconds', seconds, 120), ('proof_seconds', proof_seconds, 60),
                ('max_decisions', max_decisions, 800), ('memory_limit_mib', memory_limit_mib, 14336)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Bounded positive ' + name + ' required')
        self.executor = executor
        self.seconds, self.proof_seconds = seconds, proof_seconds
        self.max_decisions, self.memory_limit_mib = max_decisions, memory_limit_mib
        if segment_decisions is not None and (type(segment_decisions) is not int
                or not 1 <= segment_decisions <= 16):
            raise ValueError('Optional planned pure segment must contain one to16 decisions')
        if on_segment is not None and not callable(on_segment):
            raise TypeError('Optional segment callback must be callable')
        self.segment_decisions, self.on_segment = segment_decisions, on_segment
        self.last_diagnostics = None

    def _memory(self):
        free, total = self.executor.torch.cuda.mem_get_info()
        if total - free > self.memory_limit_mib << 20:
            raise MemoryError('Pure collection exceeds whole-device budget')
        return total - free

    def collect(self, jobs, models):
        if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= 32:
            raise ValueError('One to32 explicit small-batch pure jobs required')
        metadata = adapt_pure_jobs(jobs, models)
        if metadata.reference_inference_keys:
            raise ValueError('Calibration sampler does not implement paired comparison telemetry')
        remaining = min(j['deadline'] for j in jobs) - time.time()
        if remaining < self.seconds + self.proof_seconds + 10:
            raise TimeoutError('Insufficient capture/replay/proof margin before original job deadline')
        self.last_diagnostics = dict(complete=False, optimizer_steps=0, cpu_hot_rule_calls=0,
            algorithm='argmax', counts_as_search=False, simulations=0, steps=[], errors=[], natural_games=0)
        owned = []
        try:
            if self.segment_decisions is not None:
                from .device_pure_segments import collect_segments
                return collect_segments(self, metadata, models)
            return self._collect(metadata, models, owned)
        finally:
            errors = []
            for obj in reversed(owned):
                closer = getattr(obj, 'close', None)
                if callable(closer):
                    try: closer()
                    except BaseException as error: errors.append(repr(error))
            for obj in owned:
                if hasattr(obj, 'physical_loop'):
                    obj.physical_loop = None
                inner = getattr(obj, 'inner', None)
                if inner is not None and hasattr(inner, 'physical_loop'):
                    inner.physical_loop = None
            owned.clear()
            import gc
            gc.collect()
            self.last_diagnostics['cleanup_errors'] = errors
            if errors:
                raise RuntimeError('Pure batch cleanup failed: ' + repr(errors))

    def _collect(self, metadata, models, owned):
        from ..batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
        from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
        from .device_tiled_soa_rules import DeviceTiledSoARules
        from .device_raw_action_tape import DeviceRawActionTape
        from .device_pure_value_tape import DevicePureValueTape
        from .device_pure_policy import DevicePurePolicy
        from .device_value_only_inference import DeviceValueOnlyInference
        from .device_info_key import DeviceInformationKey
        from .device_seeds import DeviceRootSeeds
        from .device_games import DeviceGameLoop
        from .device_physical_action_capture import RawActionPolicySink
        from ..batched_duel.gpu_seven_observation import OBS_DIM
        t, device = self.executor.torch, self.executor.device
        rows, decisions, width = len(metadata.jobs), self.max_decisions, 800
        def own(value):
            owned.append(value); self._memory(); return value
        inference = own(DeviceValueOnlyInference(models, device=str(device), max_rows=rows,
            max_candidate_elements=rows * width * 1576))
        if inference.keys != metadata.inference_keys:
            raise ValueError('Actual inference source order differs from frozen job metadata')
        store = own(DeviceSparseSoAStore(self.executor.schema, rows * 6 + 128,
            device=str(device), max_bytes=256 << 20))
        actions = own(CompactDeviceGraphStore(slots=rows * 6 + 128,
            node_pages=8192, edge_pages=8192, payload_pages=4096,
            node_page_bytes=4096, edge_page_bytes=4096, payload_page_bytes=4096,
            max_node_pages_per_slot=64, max_edge_pages_per_slot=64,
            max_payload_pages_per_slot=32, device=str(device)))
        records = rows * decisions
        raw = own(DeviceRawActionTape(rows, records, records * 1024, records * 1024,
            records * 4096, device=str(device)))
        rules = own(DeviceTiledSoARules(self.executor, store, actions, rows=rows, work_rows=min(rows, 32),
            inference_rows=min(rows, 32), max_actions=width, raw_tape=raw,
            seconds=self.seconds, candidate_int32=True))
        pure = DevicePurePolicy(rules, inference, rows=rows, max_actions=width, device=str(device))
        keys = own(DeviceInformationKey(device=str(device), max_rows=rows))
        seeds = own(DeviceRootSeeds(device=str(device), max_rows=rows))
        original = pure.initialize
        def initialize(*args):
            original(*args)
            result = pure.result_device()
            active = args[0].active & result.complete & (result.errors == 0)
            digest = result.decision.candidates.hash_device(keys, result.decision.observations,
                                                            result.decision.counts, active)
            pure._result = replace(result, information_keys=digest.keys, errors=result.errors | digest.errors)
        pure.initialize = initialize
        tape = DevicePureValueTape(rules, rows=rows, max_records=records, max_bytes=256 << 20,
            observation_dim=OBS_DIM, max_actions=width, device=str(device))
        audit = _ValueAuditSink(tape, decisions=decisions)
        loop = own(DeviceGameLoop(rules, pure, RawActionPolicySink(audit, raw), seeds,
            max_decisions=decisions, inner_max_ticks=2, outer_max_ticks=decisions + 2,
            seconds=self.seconds, device=str(device), defer_physical_presentation=True))
        rules.bind_game_loop(loop)
        roots = rules.initialize_device(list(deepcopy(metadata.initializers)))
        t.cuda.synchronize()
        if bool(roots.errors.any().cpu()):
            raise ValueError('GPU pure initializer failed')
        initial = _host_states(store, roots)
        ids = t.tensor(metadata.model_indices, device=device, dtype=t.int64)
        train = t.tensor(metadata.train_masks, device=device, dtype=t.bool)
        job_seeds = t.tensor([j['seed'] for j in metadata.jobs], device=device, dtype=t.int64)
        x = t.zeros((rows, inference.feature_dim), device=device)
        c = t.zeros((rows, width, inference.candidate_dim), device=device)
        valid = t.zeros((rows, width), device=device, dtype=t.bool); valid[:, 0] = True
        for stream in (t.cuda.current_stream(device), loop._outer.capture_stream,
                       loop._outer.body_stream, loop._inner.body_stream):
            stream.wait_stream(t.cuda.current_stream(device))
            with t.cuda.stream(stream):
                inference.predict_device(x, c, valid, ids[:, 0].contiguous(), t.ones((rows, 1), device=device))
            stream.synchronize()
        loop.capture(roots, job_seeds, ids, ids, train)
        self._memory()
        if min(j['deadline'] for j in metadata.jobs) - time.time() < self.seconds + self.proof_seconds:
            raise TimeoutError('Capture exhausted reserved replay/proof margin')
        before = time.monotonic(); loop.replay(); t.cuda.synchronize()
        result = loop.result_device()
        diag = self.last_diagnostics
        diag.update(replay_seconds=time.monotonic() - before, steps=result.steps.cpu().tolist(),
            errors=result.errors.cpu().tolist(), complete_rows=result.complete.cpu().tolist(),
            truncated=result.truncated.cpu().tolist(), natural_games=int(result.complete.sum().cpu()),
            peak_sampled_device_bytes=self._memory(), actual_outer_ticks=int(loop._outer.counter[0].cpu()))
        if any(diag['errors']) or not all(diag['complete_rows']) or any(diag['truncated']):
            raise RuntimeError('Pure device cohort did not finish naturally; no value labels or updates')
        final = _host_states(store, result.states)
        records = _raw_records(raw)
        hashes = audit.hashes[:rows * decisions].cpu().numpy().reshape(rows, decisions, 8)
        choices = audit.chosen[:rows * decisions].cpu().numpy().reshape(rows, decisions)
        coordinates = {(r['game'], r['step']) for r in records}
        expected = {(i, s) for i, n in enumerate(diag['steps']) for s in range(n)}
        actual = set(map(tuple, np.argwhere(audit.valid[:rows * decisions].cpu().numpy().reshape(rows, decisions))))
        if len(records) != len(expected) or coordinates != expected or actual != expected or bool(raw.errors_device.any().cpu()):
            raise ValueError('Raw/hash/value trajectories omitted or duplicated decisions')
        deadline = min(time.monotonic() + self.proof_seconds,
            time.monotonic() + min(j['deadline'] for j in metadata.jobs) - time.time())
        references = _references(initial, records, hashes, choices, metadata, models, inference, deadline)
        dataset = dict(complete=True, error_rows=0, optimizer_steps=0, truncated=False,
            execution=metadata.execution, model_versions=metadata.model_versions,
            policy=[], values=tape.value_rows_after_batch(include_frozen=True))
        games = verified_pure_calibration_games(dataset, initial=initial, final=final,
            raw_records=records, metadata=metadata, models=models, policy_reference=references, deadline=deadline)
        diag.update(complete=True, verified=True, raw_records=len(records), value_rows=len(dataset['values']))
        return games
