"""Fit the existing seven-team auxiliary value contract from verified CUDA games.

Collection is injected and never substitutes host rule execution. Archived
postbatch oracle verification belongs to the existing GPU collector. This module
cannot qualify incomplete cohorts or reduce the established calibration gates.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np

from . import seven_covered_value as cv
from .seven_auxiliary_jobs import auxiliary_jobs, adapt_auxiliary_jobs
from .seven_schedule import ROSTER

PHASE_COUNTS = dict(training=168, selection=56, heldout=56)
_PHASE_SEEDS = dict(training='foundation', selection='selection', heldout='confirmation')


def _deadline(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.time():
        raise TimeoutError('Live explicit absolute auxiliary deadline required')


def auxiliary_phase_plans(*, starting_decks, seed_reservation, deadline, phase_counts=None):
    """Return original broad-build plans with exact, disjoint seed partitions."""
    from .build_acceptance import SEED_PHASES
    _deadline(deadline)
    counts = dict(PHASE_COUNTS if phase_counts is None else phase_counts)
    cv.phase_counts(dict(schema='seven_expert_value_v2', phase_counts=counts))
    if (set(seed_reservation) != set(SEED_PHASES)
            or any(type(v) is not int or v < 0 for v in seed_reservation.values())):
        raise ValueError('Complete shared-ledger numeric seed reservation required')
    plans = {phase: auxiliary_jobs(seed_reservation[_PHASE_SEEDS[phase]], count // 56,
             decks=starting_decks, deadline=deadline) for phase, count in counts.items()}
    sets = [set(j['seed'] for j in plans[p]) for p in _PHASE_SEEDS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Auxiliary training/selection/heldout seeds overlap')
    return plans


def _job_key(job):
    return json.dumps(job, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_auxiliary_phases(phase_segments, *, plans, deadline):
    """Reverify actual segments and require every exact planned physical game."""
    from .device_auxiliary_collection import AuxiliarySegment, verified_auxiliary_games
    _deadline(deadline)
    if set(phase_segments) != set(_PHASE_SEEDS) or set(plans) != set(_PHASE_SEEDS):
        raise ValueError('All three exact auxiliary phases required')
    cv.phase_counts(dict(schema='seven_expert_value_v2',
                         phase_counts={p:len(plans[p]) for p in _PHASE_SEEDS}))
    partitions = [set(j['seed'] for j in plans[p]) for p in _PHASE_SEEDS]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Auxiliary training/selection/heldout seeds overlap')
    games = {}; binding = None
    for phase in _PHASE_SEEDS:
        expected = {_job_key(j): j for j in plans[phase]}
        if len(expected) != len(plans[phase]):
            raise ValueError('Duplicate physical planned auxiliary job')
        found = {}
        for segment in phase_segments[phase]:
            _deadline(deadline)
            if not isinstance(segment, AuxiliarySegment):
                raise TypeError('Only actual AuxiliarySegment evidence accepted')
            if binding is None:
                binding = deepcopy(segment.source_binding)
            if segment.source_binding != binding:
                raise ValueError('Auxiliary phases use different frozen CUDA sources')
            actual = verified_auxiliary_games(segment, proof_deadline=time.monotonic() + deadline-time.time())
            for game in actual:
                key = _job_key(game['job'])
                if key not in expected or key in found:
                    raise ValueError('Duplicate or unplanned auxiliary seed/key/build/initiative/deadline')
                if (game.get('complete') is not True or not game.get('value_rows')
                        or game.get('rule_backend') != 'cuda_soa'
                        or game.get('source_binding') != binding):
                    raise ValueError('Only naturally complete verified CUDA value games accepted')
                found[key] = game
        if set(found) != set(expected):
            raise ValueError('Exact complete auxiliary phase count required: '+phase)
        games[phase] = [found[_job_key(job)] for job in plans[phase]]
    return games, binding


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported private auxiliary evidence: '+type(value).__name__)


def _gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        json.dump(value, stream, default=_json, allow_nan=False)


class DeviceAuxiliaryPhaseCollector:
    """Own fresh single-use collectors via ``factory(metadata)`` per segment.

    Factory must create DeviceAuxiliaryCollector with fresh arenas and a shared
    actual executor. No arena/controller is reused across segments. Resume uses
    only the immediately preceding verified archive and preserves guard errors;
    a guarded prefix can never become qualified calibration evidence.
    """
    def __init__(self, factory, *, starting_decks, rows=2):
        if not callable(factory) or type(rows) is not int or not 1 <= rows <= 64:
            raise ValueError('Fresh collector factory and bounded CUDA cohort required')
        self.factory, self.starting_decks, self.rows = factory, deepcopy(starting_decks), rows

    def collect(self, jobs, *, deadline, on_segment=None):
        from .device_auxiliary_collection import DeviceAuxiliaryCollector, verified_auxiliary_games
        _deadline(deadline)
        if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= 672:
            raise ValueError('One to672 actual auxiliary jobs required')
        results = []
        for start in range(0, len(jobs), self.rows):
            metadata = adapt_auxiliary_jobs(jobs[start:start+self.rows], starting_decks=self.starting_decks)
            prior = None
            while True:
                _deadline(deadline)
                collector = self.factory(metadata)
                if not isinstance(collector, DeviceAuxiliaryCollector):
                    raise TypeError('Actual DeviceAuxiliaryCollector factory required')
                with collector:
                    result = collector.run_segment(metadata, deadline=deadline, resume=prior,
                                                   seconds=90, soft_seconds=150)
                if on_segment is not None:
                    on_segment(start, result)
                if (not result.replay_verified or result.export_errors or any(result.errors)
                        or any(result.truncated) or any(any(a['errors']) for a in result.attempts)):
                    raise ValueError('Only clean verified auxiliary segments may continue calibration')
                if all(result.complete):
                    verified_auxiliary_games(result, proof_deadline=time.monotonic()+deadline-time.time())
                    results.append(result); break
                if prior is not None and result.steps == prior.steps:
                    raise RuntimeError('Auxiliary segment made no physical progress')
                prior = result
        return tuple(results)


def fit_device_auxiliary_calibration(source, output, *, starting_decks, seed_reservation,
                                     phase_segments, deadline, phase_counts=None, device='cuda'):
    """Fit value-only AdamW candidates, select temperatures, qualify heldout scope.

    Returns ``(models, report)`` only after current ``load_scope`` succeeds.
    Failed fitting or gates preserve new output evidence; source models and old
    scopes are never modified. This is auxiliary fitting, zero learner updates.
    """
    import torch
    from . import seven_runtime as rt, seven_round as rr
    from .seven_lineup import identity as rule_identity
    from .build_acceptance import canonical_build
    _deadline(deadline)
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Actual CUDA fitting required; no fallback')
    output, source = Path(output), Path(source)
    if output.exists():
        raise ValueError('New isolated auxiliary output required')
    plans = auxiliary_phase_plans(starting_decks=starting_decks, seed_reservation=seed_reservation,
                                  deadline=deadline, phase_counts=phase_counts)
    # Materialize once: validation must not consume caller generators before archiving.
    phase_segments = {p: tuple(v) for p, v in phase_segments.items()}
    games, binding = validate_auxiliary_phases(phase_segments, plans=plans, deadline=deadline)
    signature = rr.versions(source)
    for key in ROSTER:
        if canonical_build(rt.model(source, key).serving_deck) != canonical_build(starting_decks[key]):
            raise ValueError('Actual source model differs from explicit frozen starting build')
    output.mkdir(parents=True)
    rr.write(output/'config.json', dict(source_models=signature, seeds=seed_reservation,
             deadline=deadline, starting_decks=starting_decks, phase_counts={p:len(v) for p,v in plans.items()},
             source_binding=binding, learning_rate=.0003, epochs=[1,2,4,8], temperatures=[1.,2.,4.],
             value_only=True, learner_updates=0))
    files = {}
    for phase in _PHASE_SEEDS:
        for i, segment in enumerate(phase_segments[phase]):
            _gzip(output/f'{phase}-segment-{i}-raw.json.gz', asdict(segment))
        for i, game in enumerate(games[phase]):
            _gzip(output/f'{phase}-{i}-private.json.gz', game)
        rr.write(output/(phase+'.json'), dict(games=len(games[phase]), complete=True,
                 seeds=[j['seed'] for j in plans[phase]], first=[j['first'] for j in plans[phase]],
                 source_binding=binding, rule_backend='cuda_soa', replay_verified=True))
    choices = {}
    for key in ROSTER:
        rows = [r for game in games['training'] for r in game['value_rows'] if r['key'] == key]
        if not rows:
            raise ValueError('Missing complete training value rows for '+key)
        net, build, origin = rt.restore(source, key, device)
        origin = dict(origin, usage='covered_expert_value_auxiliary', quality_approved=False)
        frozen = {n: t.detach().clone() for n,t in net.state_dict().items() if not n.startswith(('value_net.', 'wdl.'))}
        opt = torch.optim.AdamW([p for n,p in net.named_parameters() if n.startswith(('value_net.', 'wdl.'))],
                               lr=.0003, weight_decay=.001)
        rng = np.random.default_rng(seed_reservation['adaptation']); candidates = []
        for epoch in range(1, 9):
            indexes = rng.permutation(len(rows))
            for offset in range(0, len(rows), 128):
                _deadline(deadline)
                rt.update(net, opt, [rows[int(i)] for i in indexes[offset:offset+128]], value_only=True)
            if epoch not in (1,2,4,8):
                continue
            rt.export(net, output/f'epoch-{epoch}'/'models', key, build, origin)
            weight, bias = net.wdl.weight.detach().clone(), net.wdl.bias.detach().clone()
            for temperature in (1.,2.,4.):
                _deadline(deadline)
                with torch.no_grad():
                    net.wdl.weight.copy_(weight/temperature); net.wdl.bias.copy_(bias/temperature)
                calibrated = output/f'epoch-{epoch}-temperature-{temperature}'/'models'
                rt.export(net, calibrated, key, build, dict(origin, value_temperature=temperature))
                metric = rr.value_metrics(games['selection'], {key: rt.model(calibrated, key)},
                                          backend='cuda_batched', device=device)[key]
                if not np.isfinite(metric['log_loss']):
                    raise ValueError('Nonfinite auxiliary selection metric')
                candidates.append((metric['log_loss'], epoch, temperature, metric))
            with torch.no_grad():
                net.wdl.weight.copy_(weight); net.wdl.bias.copy_(bias)
        chosen = min(candidates, key=lambda c:c[0])
        choices[key] = dict(epoch=chosen[1], temperature=chosen[2], selection=chosen[3])
        net, _, _ = rt.restore(output/f'epoch-{chosen[1]}'/'models', key, device)
        with torch.no_grad():
            net.wdl.weight.div_(chosen[2]); net.wdl.bias.div_(chosen[2])
        if any(not torch.equal(value, net.state_dict()[name]) for name,value in frozen.items()):
            raise ValueError('Auxiliary fitting moved policy arrays')
        rt.export(net, output/'models', key, build, dict(origin, value_temperature=chosen[2]))
    _deadline(deadline)
    models = {k:rt.model(output/'models', k) for k in ROSTER}
    metrics = rr.value_metrics(games['heldout'], models, backend='cuda_batched', device=device)
    passed = set(metrics) == set(ROSTER) and all(m.get('normal_games',0)>=8
        and np.isfinite(m.get('brier',np.nan)) and m['brier']<.7
        and np.isfinite(m.get('log_loss',np.nan)) and m['log_loss']<1.2 for m in metrics.values())
    if rr.versions(source) != signature:
        raise ValueError('Primary source models changed')
    for path in output.iterdir():
        if path.is_file() and (path.name in ('training.json','selection.json','heldout.json') or path.name.endswith('.json.gz')):
            files[path.name] = sha256(path.read_bytes()).hexdigest()
    report = dict(schema='seven_expert_value_v2', passed=passed,
        phase_counts={p:len(v) for p,v in plans.items()}, rule_hash=rule_identity(), algorithm_sha256=cv.identity(),
        objective='visible_rule_after_six_exploration_actions', models=rr.versions(output/'models'),
        metrics=metrics, choices=choices, files=files, source_binding=binding,
        primary_policy_arrays_unchanged=True, real_learner_updates=0, quality_approved=False)
    rr.write(output/'scope.json', report)
    _deadline(deadline)
    return cv.load_scope(output)


def create_device_auxiliary_phase_collector(executor, *, starting_decks, rows=8,
                                             segment_decisions=16,
                                             max_device_bytes=14336 << 20):
    """Bind an already loaded full CUDA executor to fresh bounded segment arenas.

    Pools scale the established two-row collection allocation to the actual
    cohort. Their finite page capacity is never treated as evidence of a full
    trajectory: allocation/append overflow fails qualification. The borrowed
    executor is never closed here, including constructor failures. The caller
    still owns the outer process watchdog; the collector cannot preempt a
    stalled CUDA driver.
    """
    from .batched_duel.native_soa_cuda import CudaSoARules
    from .batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
    from .batched_duel.device_graph_storage import CompactDeviceGraphStore
    from .device_auxiliary_collection import DeviceAuxiliaryCollector
    from .batched_search.soa_capacity import rule_workspace_minimum_bytes
    from .batched_duel.vm_soa_views import value_cache_extent
    if (not isinstance(executor, CudaSoARules) or executor.execute_only
            or executor.device.type != 'cuda'):
        raise TypeError('Actual already-loaded full CUDA rule executor required')
    if type(rows) is not int or not 1 <= rows <= 64:
        raise ValueError('One to64 auxiliary rows required')
    if type(segment_decisions) is not int or segment_decisions not in (*range(1,17),32):
        raise ValueError('Explicit1..16 or32 physical decisions per segment required')
    if type(max_device_bytes) is not int or not 1 <= max_device_bytes <= 14336 << 20:
        raise ValueError('Positive whole-device ceiling at most14336MiB required')
    # Validate all seven actual starting builds without creating GPU arenas.
    # A generated one-replica plan is validation metadata only, not collection.
    auxiliary_jobs(0, 1, decks=starting_decks, deadline=1.)

    def factory(metadata):
        n = len(metadata.jobs)
        if not 1 <= n <= rows:
            raise ValueError('Actual auxiliary cohort exceeds configured rows')
        free, total = executor.torch.cuda.mem_get_info(executor.device)
        if total-free >= max_device_bytes:
            raise MemoryError('Whole-device memory ceiling exhausted before arena creation')
        arenas = []
        def memory_guard():
            free, total = executor.torch.cuda.mem_get_info(executor.device)
            if total-free > max_device_bytes:
                raise MemoryError('Whole-device ceiling exceeded while creating auxiliary arenas')
        try:
            arenas.append(DeviceSparseSoAStore(executor.schema, max(32,n*16),
                device=str(executor.device), max_bytes=64*n << 20))
            memory_guard()
            def graph_store(*, audit):
                return CompactDeviceGraphStore(
                    slots=n*512 if audit else max(16,n*8),
                    node_pages=n*(2048 if audit else 256),
                    edge_pages=n*(2048 if audit else 256),
                    payload_pages=n*(1024 if audit else 128),
                    node_page_bytes=4096, edge_page_bytes=4096, payload_page_bytes=4096,
                    max_node_pages_per_slot=64, max_edge_pages_per_slot=64,
                    max_payload_pages_per_slot=32, device=str(executor.device))
            arenas.append(graph_store(audit=False))
            memory_guard()
            arenas.append(graph_store(audit=True))
            memory_guard()
            # Rejection lower bound only: this excludes further graph, model
            # and archive allocations and cannot certify that the batch fits.
            workspace = rule_workspace_minimum_bytes(n,
                schema_lengths=executor.schema.lengths,
                total_objects=executor.schema.total_objects,
                value_cache_entries=value_cache_extent(executor.schema),
                constant_nodes=len(executor.image.nodes),
                max_strings=executor.schema.max_strings)
            free, total = executor.torch.cuda.mem_get_info(executor.device)
            if total-free+workspace > max_device_bytes:
                raise MemoryError('Known auxiliary rule workspace plus resident device memory exceeds ceiling')
            collector = DeviceAuxiliaryCollector(executor, *arenas, rows=n,
                max_decisions=512, segment_decisions=segment_decisions,
                max_device_bytes=max_device_bytes, value_max_bytes=128*n << 20,
                raw_capacities=(32768*n,65536*n,524288*n))
            # Ownership has transferred to the collector. Check the actual
            # device after constructors, while all arenas remain releasable.
            try:
                collector._memory()
            except BaseException:
                try:
                    collector.close()
                finally:
                    arenas.clear()
                raise
            return collector
        except BaseException:
            for arena in reversed(arenas):
                try:
                    arena.close()
                except BaseException:
                    pass  # Preserve the actual allocation/constructor failure.
            arenas.clear()
            raise
    return DeviceAuxiliaryPhaseCollector(factory, starting_decks=starting_decks, rows=rows)
