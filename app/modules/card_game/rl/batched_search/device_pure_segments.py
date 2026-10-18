"""Bounded planned pauses for pure CUDA calibration, without partial labels.

Each segment owns fresh arenas. Only the final natural GPU tape is exported,
then the original complete-prefix verifier checks every physical action. Host
work between graphs copies state/archives and checks invariants; it makes no
rule decisions. This module grants no H3 or training qualification.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import gc
import time

import numpy as np

from .device_pure_collection import _ValueAuditSink, _host_states, _raw_records, _references
from .soa_pure_results import verified_pure_calibration_games


@dataclass(frozen=True)
class _History:
    initial: object
    final: object
    steps: tuple
    complete: tuple
    raw: dict
    value: dict
    audit: dict


def _tensor_archive(obj, names):
    return {name: getattr(obj, name).detach().cpu().clone() for name in names}


def _snapshot(initial, final, steps, complete, raw, tape, audit):
    # Entire bounded archive, including used counters and error/eligibility
    # masks, is retained. No raw graph is regenerated from a decoded action.
    names = [name for name, value in vars(raw).items() if isinstance(value, raw.t.Tensor)]
    return _History(deepcopy(initial), deepcopy(final), tuple(steps), tuple(complete),
        _tensor_archive(raw, names), _tensor_archive(tape, tape._tensor_names),
        _tensor_archive(audit, ('valid', 'hashes', 'chosen', 'errors')))


def _check_progress(steps, offsets, complete, limit, segment):
    if not (len(steps) == len(offsets) == len(complete)):
        raise ValueError('Full segment cohort required')
    for now, before, done in zip(steps, offsets, complete):
        if (type(now) is not int or type(before) is not int or type(done) is not bool
                or not 0 <= before <= now <= limit or now - before > segment
                or (not done and (now == before or now >= limit))):
            raise ValueError('Planned segment must advance cleanly within original physical ceiling')


def _validate_history(history, rows, decisions):
    """Reject archive gaps, earlier errors and fabricated partial learning labels."""
    import torch as t
    if not isinstance(history, _History) or len(history.initial) != rows or len(history.final) != rows:
        raise ValueError('Actual full pure segment history required')
    if len(history.steps) != rows or len(history.complete) != rows:
        raise ValueError('Full history outcome masks required')
    raw, value, audit = history.raw, history.value, history.audit
    for field in (raw['_game_errors'], value['_game_errors'], audit['errors']):
        if bool(field.any()):
            raise ValueError('Prior guard/archive errors cannot be resumed')
    count = int(raw['_record_count'])
    if not 0 <= count <= rows * decisions or count != sum(history.steps) or int(value['_count']) != count:
        raise ValueError('Raw and dual-viewer archive counts differ from physical prefix')
    wanted = {(i, step) for i, end in enumerate(history.steps) for step in range(end)}
    for data in (raw, value):
        coords = list(zip(data['_rec_game'][:count].tolist(), data['_rec_step'][:count].tolist()))
        if len(coords) != len(set(coords)) or set(coords) != wanted:
            raise ValueError('Missing or duplicate archive prefix coordinate')
        if data['_expected_step'].tolist() != list(history.steps):
            raise ValueError('Historical expected step counter differs')
        for index in range(count):
            owner, step = coords[index]
            if (int(data['_rec_owner'][index]) != owner
                    or int(data['_rec_seed'][index]) != int(value['_job_seeds'][owner])
                    or int(data['_rec_actor'][index]) not in (0, 1)
                    or int(data['_rec_version'][index]) != int(value['_initial_version'][owner]) + step):
                raise ValueError('Historical archive identity/version differs from original game')
    raw_index = {(int(raw['_rec_game'][i]), int(raw['_rec_step'][i])): i for i in range(count)}
    for index in range(count):
        owner, step = int(value['_rec_game'][index]), int(value['_rec_step'][index])
        actor = int(value['_rec_actor'][index])
        if actor != int(raw['_rec_actor'][raw_index[owner, step]]) or value['_ctx'][index].tolist() != [int(actor == 0), int(actor == 1)]:
            raise ValueError('Historical both-viewer contexts differ from physical actor')
    if bool(value['_rec_errors'][:count].any()) or not bool(value['_x'][:count].isfinite().all()) or not bool(value['_ctx'][:count].isfinite().all()):
        raise ValueError('Historical value records contain errors/nonfinite inputs')
    valid = audit['valid'][:rows * decisions].reshape(rows, decisions)
    actual = set(map(tuple, t.nonzero(valid).tolist()))
    if actual != wanted:
        raise ValueError('Missing or duplicate information hash/choice prefix')
    for owner, (steps, complete, final) in enumerate(zip(history.steps, history.complete, history.final)):
        if (type(steps) is not int or not 0 <= steps <= decisions or type(complete) is not bool
                or type(final.get('version')) is not int
                or final['version'] != int(value['_initial_version'][owner]) + steps
                or complete != (final.get('phase') == 'finished')):
            raise ValueError('Historical state version/terminal differs from prefix')
        chosen = value['_rec_game'][:count] == owner
        if not complete and (bool(value['_z'][:count][chosen].any()) or bool(value['_eligible'][:count][chosen].any())):
            raise ValueError('Partial games cannot carry labels or eligibility')
        if steps:
            selected = (raw['_rec_game'][:count] == owner) & (raw['_rec_step'][:count] == steps - 1)
            if int(raw['_last_version'][owner]) != final['version'] - 1 or int(raw['_rec_version'][:count][selected][0]) != final['version'] - 1:
                raise ValueError('Historical raw version counter differs')
        if bool(value['_complete'][owner]) != complete or bool(value['_finished'][owner]) != complete or bool(value['_truncated'][owner]):
            raise ValueError('Prior partial/terminal value outcome differs')
    for field, used, capacity in (('_nodes', '_node_used', '_node_sink'), ('_edges', '_edge_used', '_edge_sink'), ('_payload', '_payload_used', '_payload_sink')):
        if not 0 <= int(raw[used]) <= int(raw[capacity]) < raw[field].shape[0]:
            raise ValueError('Historical raw archive capacity overflow')
    return True


def _upload_history(history, raw, tape, audit):
    _validate_history(history, tape.rows, audit.decisions)
    uploaded = []
    for obj, data in ((raw, history.raw), (tape, history.value), (audit, history.audit)):
        batch = {}
        for name, source in data.items():
            target = getattr(obj, name, None)
            if (not isinstance(target, raw.t.Tensor) or target.shape != source.shape or target.dtype != source.dtype
                    or source.device.type != 'cpu'):
                raise ValueError('Fresh bounded archive layout differs from saved history')
            batch[name] = source.to(device=target.device)
        uploaded.append(batch)
    return tuple(uploaded)


def _restore_history(raw, tape, audit, uploaded):
    # Called in graph initialization only after all three normal begin/reset
    # protocols; the current tape pool binding remains the fresh state arena.
    for obj, data in zip((raw, tape, audit), uploaded):
        for name, source in data.items():
            getattr(obj, name).copy_(source)


def _cleanup(owned):
    failures = []
    for obj in reversed(owned):
        closer = getattr(obj, 'close', None)
        if callable(closer):
            try: closer()
            except BaseException as error: failures.append(repr(error))
        for name in ('initialize', '_update_predicate'):
            vars(obj).pop(name, None)
        for target in (obj, getattr(obj, 'inner', None)):
            if target is not None and hasattr(target, 'physical_loop'):
                target.physical_loop = None
    owned.clear()
    gc.collect()
    if failures:
        raise RuntimeError('Pure segment cleanup failed: ' + repr(failures))


def collect_segments(collector, metadata, models):
    resume = None
    attempts = []
    for index in range((collector.max_decisions + collector.segment_decisions - 1) // collector.segment_decisions):
        remaining = min(job['deadline'] for job in metadata.jobs) - time.time()
        if remaining < collector.seconds + collector.proof_seconds + 10:
            raise TimeoutError('Original pure job deadline exhausted before next bounded segment')
        owned = []
        try:
            resume, games, diag = _segment(collector, metadata, models, owned, resume)
            attempts.append(diag)
        finally:
            _cleanup(owned)
        collector.last_diagnostics.update(diag, segments=deepcopy(attempts), segment_decisions=collector.segment_decisions,
            complete=games is not None)
        if collector.on_segment is not None:
            collector.on_segment(deepcopy(collector.last_diagnostics))
        if games is not None:
            return games
    raise RuntimeError('Original physical decision limit exhausted without entire natural cohort')


def _segment(self, metadata, models, owned, resume):
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
    pure = own(DevicePurePolicy(rules, inference, rows=rows, max_actions=width, device=str(device)))
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
        max_decisions=decisions, inner_max_ticks=2, outer_max_ticks=self.segment_decisions + 1,
        seconds=self.seconds, device=str(device), defer_physical_presentation=True))
    rules.bind_game_loop(loop)
    if resume is None:
        roots = rules.initialize_device(list(deepcopy(metadata.initializers)))
    else:
        from ..batched_duel.record_soa import pack_game_states
        roots = store.put_device(pack_game_states(deepcopy(resume.final), self.executor.schema).to_torch(device),
            t.arange(rows, device=device, dtype=t.int64), t.ones(rows, device=device, dtype=t.bool))
    t.cuda.synchronize()
    if bool(roots.errors.any().cpu()):
        raise ValueError('GPU pure initializer failed')
    ingress = _host_states(store, roots)
    if resume is not None:
        from .soa_training_results import _same_graph
        for actual, expected in zip(ingress, resume.final):
            _same_graph(actual, expected)
    initial = deepcopy(ingress if resume is None else resume.initial)
    active = t.tensor([True] * rows if resume is None else [not v for v in resume.complete],
                      device=device, dtype=t.bool)
    roots = replace(roots, active=active)
    offsets = t.tensor([0] * rows if resume is None else resume.steps, device=device, dtype=t.int64)
    restored = None if resume is None else _upload_history(resume, raw, tape, audit)
    self._memory()
    original_predicate = loop._update_predicate
    def update_predicate():
        original_predicate()
        loop._unfinished.bitwise_and_(((loop._steps - offsets) < self.segment_decisions).to(t.uint8))
    loop._update_predicate = update_predicate
    original_initialize = loop.initialize
    def initialize_loop(*args):
        original_initialize(*args)
        if restored is not None:
            _restore_history(raw, tape, audit, restored)
        loop._steps.copy_(offsets)
        loop._update_predicate()
    loop.initialize = initialize_loop
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
    diag = dict(
        replay_seconds=time.monotonic() - before, steps=result.steps.cpu().tolist(),
        errors=result.errors.cpu().tolist(), complete_rows=result.complete.cpu().tolist(),
        truncated=result.truncated.cpu().tolist(), natural_games=int(result.complete.sum().cpu()),
        peak_sampled_device_bytes=self._memory(), actual_outer_ticks=int(loop._outer.counter[0].cpu()))
    self.last_diagnostics.update(diag)
    if bool(loop._outer.errors.any().cpu()) or bool(loop._inner.errors.any().cpu()):
        raise RuntimeError('Pure segment conditional graph guard failed; no learning or resume')
    if any(diag['errors']) or any(diag['truncated']) or bool(raw.errors_device.any().cpu()) or bool(tape._game_errors.any().cpu()) or bool(audit.errors.any().cpu()):
        raise RuntimeError('Pure device segment guard/rule/archive failure; no learning or resume')
    previous_complete = [False] * rows if resume is None else resume.complete
    diag['complete_rows'] = [a or b for a, b in zip(previous_complete, diag['complete_rows'])]
    diag['natural_games'] = sum(diag['complete_rows'])
    steps = diag['steps']
    _check_progress(steps, offsets.cpu().tolist(), diag['complete_rows'], decisions, self.segment_decisions)
    # The inactive natural parents still have checked leases; gather them read-only.
    result = replace(result, states=replace(result.states, active=t.ones_like(result.states.active)))
    final = _host_states(store, result.states)
    if not all(diag['complete_rows']):
        frame = _snapshot(initial, final, steps, diag['complete_rows'], raw, tape, audit)
        _validate_history(frame, rows, decisions)
        return frame, None, diag
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
    return None, games, diag
