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


def validate_auxiliary_phases(phase_segments, *, plans, deadline, ledger=None):
    """Reverify every exact planned game; deadline remains exact by default.

    Explicit ``ledger`` permits only that durable plan's latest completed
    checkpoint metadata, retaining the actual per-cohort scheduling history.
    """
    from .device_auxiliary_collection import AuxiliarySegment, verified_auxiliary_games
    _deadline(deadline)
    if set(phase_segments) != set(_PHASE_SEEDS) or set(plans) != set(_PHASE_SEEDS):
        raise ValueError('All three exact auxiliary phases required')
    cv.phase_counts(dict(schema='seven_expert_value_v2',
                         phase_counts={p:len(plans[p]) for p in _PHASE_SEEDS}))
    partitions = [set(j['seed'] for j in plans[p]) for p in _PHASE_SEEDS]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Auxiliary training/selection/heldout seeds overlap')
    if ledger is not None:
        if not isinstance(ledger,DeviceAuxiliaryCalibrationLedger):raise TypeError('Actual durable auxiliary ledger required')
        ledger.validate_phase_segments(phase_segments,plans=plans,deadline=deadline)
    job_key=(lambda job:_job_key(_physical_job(job)))if ledger is not None else _job_key
    games = {}; binding = None
    for phase in _PHASE_SEEDS:
        expected = {job_key(j): j for j in plans[phase]}
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
                key = job_key(game['job'])
                if key not in expected or key in found:
                    raise ValueError('Duplicate or unplanned auxiliary seed/key/build/initiative/deadline')
                if (game.get('complete') is not True or not game.get('value_rows')
                        or game.get('rule_backend') != 'cuda_soa'
                        or game.get('source_binding') != binding):
                    raise ValueError('Only naturally complete verified CUDA value games accepted')
                found[key] = game
        if set(found) != set(expected):
            raise ValueError('Exact complete auxiliary phase count required: '+phase)
        games[phase] = [found[job_key(job)] for job in plans[phase]]
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
                                     phase_segments=None, deadline, phase_counts=None, device='cuda', ledger=None):
    """Fit value-only AdamW candidates, select temperatures, qualify heldout scope.

    Returns ``(models, report)`` only after current ``load_scope`` succeeds.
    Failed fitting or gates preserve new output evidence; source models and old
    scopes are never modified. This is auxiliary fitting, zero learner updates.
    A supplied durable ledger must be fully complete and match actual source
    model versions; its original cohort deadlines and budget chains are kept.
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
    if phase_segments is None:
        if ledger is None:raise ValueError('Actual auxiliary phase segments or durable ledger required')
        phase_segments=ledger.completed_phase_segments(deadline=deadline)
    # Materialize once: validation must not consume caller generators before archiving.
    phase_segments = {p: tuple(v) for p, v in phase_segments.items()}
    games, binding = validate_auxiliary_phases(phase_segments, plans=plans, deadline=deadline,ledger=ledger)
    signature = rr.versions(source)
    if ledger is not None and (signature!=ledger.contract['source_model_versions']
            or binding!=ledger.contract['source_binding']
            or seed_reservation!=ledger.contract['seed_reservation']
            or starting_decks!=ledger.contract['starting_decks']):
        raise ValueError('Fitting source models/builds/seeds differ from durable collection contract')
    for key in ROSTER:
        if canonical_build(rt.model(source, key).serving_deck) != canonical_build(starting_decks[key]):
            raise ValueError('Actual source model differs from explicit frozen starting build')
    output.mkdir(parents=True)
    rr.write(output/'config.json', dict(source_models=signature, seeds=seed_reservation,
             deadline=deadline, starting_decks=starting_decks, phase_counts={p:len(v) for p,v in plans.items()},
             source_binding=binding, learning_rate=.0003, epochs=[1,2,4,8], temperatures=[1.,2.,4.],
             value_only=True, learner_updates=0))
    files = {}
    if ledger is not None:
        rr.write(output/'auxiliary-ledger.json',dict(manifest=deepcopy(ledger.manifest),
            manifest_sha256=ledger.manifest_sha256,contract_sha256=ledger.contract_sha256))
        files['auxiliary-ledger.json']=sha256((output/'auxiliary-ledger.json').read_bytes()).hexdigest()
    for phase in _PHASE_SEEDS:
        for i, segment in enumerate(phase_segments[phase]):
            if ledger is None:
                _gzip(output/f'{phase}-segment-{i}-raw.json.gz', asdict(segment))
            else:
                from .device_auxiliary_continuation import dump_auxiliary_checkpoint
                dump_auxiliary_checkpoint(segment,output/f'{phase}-segment-{i}-raw.json.gz')
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


def _physical_job(job):
    return {key:deepcopy(value) for key,value in job.items() if key!='deadline'}


def _manifest_sha(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class DeviceAuxiliaryCalibrationLedger:
    """Durable exact280-game plan, independently checkpointed physical cohorts.

    A completed cohort is only reloaded/reverified. A planned pause retains its
    latest actual checkpoint. Semantic/guard failures and uncertain interrupted
    RUNNING records are sticky rejections; an earlier clean prefix cannot wash
    them. Each caller must also install its outer180second process watchdog.
    """
    def __init__(self,directory,*,starting_decks,seed_reservation,deadline,
                 source_model_versions,source_binding):
        from .seven_lineup import identity
        _deadline(deadline)
        if (set(source_model_versions)!=set(ROSTER) or any(not self._sha(v) for v in source_model_versions.values())
                or set(source_binding)!={'rule_source','image','schema'}
                or any(not self._sha(v) for v in source_binding.values())):
            raise ValueError('Exact seven model and frozen rule/image/schema SHA bindings required')
        plans=auxiliary_phase_plans(starting_decks=starting_decks,seed_reservation=seed_reservation,deadline=deadline)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.deadline=deadline
        self.contract=dict(schema='device_auxiliary_ledger_v1',phase_counts=deepcopy(PHASE_COUNTS),
            physical_plans={p:[_physical_job(j) for j in jobs] for p,jobs in plans.items()},
            starting_decks=deepcopy(starting_decks),seed_reservation=deepcopy(seed_reservation),
            source_model_versions=deepcopy(source_model_versions),source_binding=deepcopy(source_binding),
            rule_hash=identity(),algorithm_sha256=cv.identity(),decision_limit=512,
            segment_seconds=90,soft_seconds=150,outer_seconds=180)
        self.contract_sha256=_manifest_sha(self.contract)
        self._manifest_path=self.directory/'manifest.json'
        if self._manifest_path.exists():
            self.manifest=json.loads(self._manifest_path.read_text(encoding='utf8'))
            self._manifest_file_sha256=sha256(self._manifest_path.read_bytes()).hexdigest()
            self._check_manifest();self._check_disk_head()
        else:
            self.manifest=dict(contract=self.contract,contract_sha256=self.contract_sha256,
                               revision=0,previous_manifest_sha256=None,cohorts=[],events=[])
            self._manifest_file_sha256=None;self._persist()

    @staticmethod
    def _sha(value):
        return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)

    @property
    def manifest_sha256(self):return self._manifest_file_sha256

    def _check_manifest(self):
        from .seven_lineup import identity
        if (_manifest_sha(self.contract)!=self.contract_sha256
                or self.contract['rule_hash']!=identity() or self.contract['algorithm_sha256']!=cv.identity()
                or self.manifest.get('contract')!=self.contract
                or self.manifest.get('contract_sha256')!=self.contract_sha256):
            raise ValueError('Immutable auxiliary plan/model/source/build/seed contract changed')
        occupied={p:set()for p in _PHASE_SEEDS}
        for entry in self.manifest.get('cohorts',[]):
            phase,offset,count=entry.get('phase'),entry.get('offset'),entry.get('count')
            if (phase not in occupied or type(offset)is not int or type(count)is not int
                    or not 1<=count<=8 or not 0<=offset<=PHASE_COUNTS[phase]-count
                    or entry.get('status')not in('PLANNED','RUNNING','PAUSED','COMPLETE','REJECTED')):
                raise ValueError('Exact bounded cohort coordinates/status required')
            positions=set(range(offset,offset+count))
            if occupied[phase]&positions:raise ValueError('Duplicate physical auxiliary cohort positions')
            occupied[phase].update(positions)
            if not isinstance(entry.get('checkpoints'),list):raise ValueError('Actual checkpoint history required')
            if entry['status'] in('PAUSED','COMPLETE')and not entry['checkpoints']:
                raise ValueError('Actual latest checkpoint required for resumable/complete cohort')
            for checkpoint in entry['checkpoints']:
                self._checkpoint_path(checkpoint['path'])
                if not self._sha(checkpoint.get('sha256')):raise ValueError('Actual checkpoint SHA required')

    def _checkpoint_path(self,name):
        if not isinstance(name,str)or not name:raise ValueError('Ordinary relative checkpoint name required')
        root=self.directory.resolve();path=(root/name).resolve()
        if (Path(name).is_absolute()or not path.is_relative_to(root)
                or path==self._manifest_path.resolve()):raise ValueError('Private relative cohort checkpoint path required')
        return path

    def _check_disk_head(self):
        revisions=sorted(self.directory.glob('manifest-revision-*.json'))
        if self._manifest_file_sha256 is None:
            if revisions:raise ValueError('Orphan immutable ledger revision forbids a fresh restart')
            return
        expected=self.directory/f"manifest-revision-{self.manifest['revision']:08d}.json"
        if (not revisions or revisions[-1]!=expected
                or sha256(expected.read_bytes()).hexdigest()!=self._manifest_file_sha256):
            raise ValueError('Ledger head rollback/tampering differs from immutable latest revision')

    def _persist(self):
        """Exclusive short metadata transaction; immutable revision + atomic head."""
        import os
        from uuid import uuid4
        self._check_manifest()
        lock=self.directory/'.manifest.lock'
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        temporary=None
        try:
            os.close(descriptor)
            actual=sha256(self._manifest_path.read_bytes()).hexdigest()if self._manifest_path.exists()else None
            if actual!=self._manifest_file_sha256:raise RuntimeError('Concurrent/stale auxiliary ledger writer rejected')
            self._check_disk_head()
            self.manifest['previous_manifest_sha256']=actual
            self.manifest['revision']+=1
            data=json.dumps(self.manifest,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
            revision=self.directory/f"manifest-revision-{self.manifest['revision']:08d}.json"
            with revision.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            temporary=self.directory/('.manifest-'+uuid4().hex+'.tmp')
            with temporary.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,self._manifest_path);temporary=None
            self._manifest_file_sha256=sha256(data).hexdigest()
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

    def _assert_current(self):
        self._check_manifest();self._check_disk_head()
        if sha256(self._manifest_path.read_bytes()).hexdigest()!=self._manifest_file_sha256:
            raise RuntimeError('Auxiliary ledger head changed; reopen before continuing')

    def _expected(self,phase,offset,count):
        return self.contract['physical_plans'][phase][offset:offset+count]

    def _check_segment(self,entry,segment,*,deadline,verify=True):
        from .device_auxiliary_collection import AuxiliarySegment,verify_auxiliary_segment
        _deadline(deadline)
        if (not isinstance(segment,AuxiliarySegment)or segment.decision_limit!=512
                or segment.source_binding!=self.contract['source_binding']
                or [_physical_job(j)for j in segment.metadata.jobs]!=self._expected(entry['phase'],entry['offset'],entry['count'])
                or segment.metadata.starting_decks!=self.contract['starting_decks']):
            raise ValueError('Actual cohort physical plan/build/source binding differs')
        if verify:verify_auxiliary_segment(segment,proof_deadline=time.monotonic()+deadline-time.time())
        return segment

    @staticmethod
    def _clean(segment):
        return (segment.replay_verified is True and not segment.export_errors and not any(segment.errors)
                and not any(segment.truncated) and not any(any(a.get(k,()))for a in segment.attempts
                    for k in('errors','raw_errors','value_errors','audit_append_errors'))
                and not any(a.get('memory_limit_exceeded',False)for a in segment.attempts))

    def _save_segment(self,entry,segment):
        from .device_auxiliary_continuation import dump_auxiliary_checkpoint,auxiliary_content_sha256
        ordinal=len(entry['checkpoints'])
        name=f"checkpoints/{entry['phase']}-{entry['offset']:03d}-{entry['count']:02d}-{ordinal:05d}.json.gz"
        path=self._checkpoint_path(name)
        digest=dump_auxiliary_checkpoint(segment,path)
        item=dict(path=name,sha256=digest,content_sha256=auxiliary_content_sha256(segment),
            steps=list(segment.steps),complete=list(segment.complete),errors=list(segment.errors),
            truncated=list(segment.truncated),metadata_sha256=auxiliary_content_sha256(segment.metadata),
            job_deadlines=[j['deadline']for j in segment.metadata.jobs],
            budget_continuation_sha256=[v.receipt_sha256 for v in segment.budget_continuations])
        entry['checkpoints'].append(item)
        return item

    def _load_segment(self,entry,*,deadline,verify=True):
        from .device_auxiliary_continuation import load_auxiliary_checkpoint,auxiliary_content_sha256
        item=entry['checkpoints'][-1]
        segment=load_auxiliary_checkpoint(self._checkpoint_path(item['path']),expected_sha256=item['sha256'])
        if (auxiliary_content_sha256(segment)!=item['content_sha256']
                or auxiliary_content_sha256(segment.metadata)!=item['metadata_sha256']
                or list(segment.steps)!=item['steps']or list(segment.complete)!=item['complete']
                or list(segment.errors)!=item['errors']or list(segment.truncated)!=item['truncated']
                or [j['deadline']for j in segment.metadata.jobs]!=item['job_deadlines']
                or [v.receipt_sha256 for v in segment.budget_continuations]!=item['budget_continuation_sha256']):
            raise ValueError('Latest auxiliary checkpoint/metadata/status/budget ancestry differs')
        return self._check_segment(entry,segment,deadline=deadline,verify=verify)

    def admit_phase_prefix(self,phase,*,offset=0,segment,parent_report_sha256,
                           parent_checkpoint_sha256,authorization):
        """Admit existing actual cohort without changing its old deadline/data."""
        self._assert_current()
        if not self._sha(parent_report_sha256)or not self._sha(parent_checkpoint_sha256):
            raise ValueError('Exact actual parent report/checkpoint SHA required')
        if not isinstance(authorization,str)or not authorization.strip():raise ValueError('Explicit prefix continuation authorization required')
        if phase not in _PHASE_SEEDS:raise ValueError('Actual auxiliary phase required')
        entry=dict(phase=phase,offset=offset,count=len(segment.metadata.jobs),status='PLANNED',checkpoints=[],
            admitted_parent_report_sha256=parent_report_sha256,admitted_parent_checkpoint_sha256=parent_checkpoint_sha256,
            authorization=authorization.strip())
        if type(offset)is not int or not 1<=entry['count']<=8 or not 0<=offset<=PHASE_COUNTS[phase]-entry['count']:
            raise ValueError('Exact bounded admitted cohort positions required')
        if any(e['phase']==phase and set(range(offset,offset+entry['count']))&set(range(e['offset'],e['offset']+e['count']))for e in self.manifest['cohorts']):
            raise ValueError('Physical auxiliary cohort already admitted; never rerun or count twice')
        self._check_segment(entry,segment,deadline=self.deadline)
        if not self._clean(segment):raise ValueError('Only historically clean planned prefixes may be admitted')
        self._save_segment(entry,segment)
        entry['status']='COMPLETE'if all(segment.complete)else'PAUSED'
        self.manifest['cohorts'].append(entry)
        self.manifest['events'].append(dict(event='admit_prefix',phase=phase,offset=offset))
        self._persist()
        return deepcopy(entry)

    def _next_entry(self,rows,deadline):
        if any(e['status']in('RUNNING','REJECTED')for e in self.manifest['cohorts']):
            raise ValueError('Sticky rejected/uncertain cohort cannot rewind')
        for phase in _PHASE_SEEDS:
            entries=sorted([e for e in self.manifest['cohorts']if e['phase']==phase],key=lambda e:e['offset'])
            for entry in entries:
                if entry['status']in('RUNNING','REJECTED'):raise ValueError('Sticky rejected/uncertain cohort cannot rewind to an earlier clean checkpoint')
                if entry['status']!='COMPLETE':return entry
                # Reuse exact persisted proof, never rerun a completed game.
                # Current full oracle reproof is repeated before fitting.
                cached=self._load_segment(entry,deadline=deadline,verify=False)
                if not self._clean(cached)or not all(cached.complete):
                    raise ValueError('Completed cohort cache is not an actual clean natural endpoint')
            covered={i for e in entries for i in range(e['offset'],e['offset']+e['count'])}
            missing=[i for i in range(PHASE_COUNTS[phase])if i not in covered]
            if missing:
                offset=missing[0];count=0
                while count<rows and offset+count<PHASE_COUNTS[phase]and offset+count not in covered:count+=1
                entry=dict(phase=phase,offset=offset,count=count,status='PLANNED',checkpoints=[])
                self.manifest['cohorts'].append(entry);return entry
        return None

    def run_window(self,factory,*,deadline,max_segments=1,rows=8,seconds=90,soft_seconds=150,
                   authorization,on_checkpoint=None):
        """Run bounded whole segments; clean pause is an ordinary durable result."""
        from .device_auxiliary_collection import DeviceAuxiliaryCollector
        from .device_auxiliary_continuation import create_auxiliary_continuation
        _deadline(deadline);self._assert_current()
        if (not callable(factory)or type(max_segments)is not int or not 1<=max_segments<=10000
                or type(rows)is not int or not 1<=rows<=8
                or type(seconds)not in(int,float)or not 0<seconds<=90
                or type(soft_seconds)not in(int,float)or not 0<soft_seconds<=150
                or not isinstance(authorization,str)or not authorization.strip()):
            raise ValueError('Explicit authorized bounded auxiliary window/factory required')
        ran=0
        while ran<max_segments:
            _deadline(deadline)
            try:entry=self._next_entry(rows,deadline)
            except ValueError:
                return dict(status='rejected',segments=ran,manifest_sha256=self.manifest_sha256)
            if entry is None:return dict(status='complete',segments=ran,manifest_sha256=self.manifest_sha256)
            prior=self._load_segment(entry,deadline=deadline)if entry['checkpoints']else None
            receipt=None
            if prior is not None:
                if not self._clean(prior):raise ValueError('Latest cohort prefix is not clean')
                metadata=prior.metadata
                if min(j['deadline']for j in metadata.jobs)<deadline:
                    receipt=create_auxiliary_continuation(prior,deadline=deadline,authorization=authorization,
                        parent_report_sha256=self.manifest_sha256,
                        parent_checkpoint_sha256=entry['checkpoints'][-1]['sha256'])
                    metadata=receipt.new_metadata
            else:
                jobs=[dict(deepcopy(j),deadline=deadline)for j in self._expected(entry['phase'],entry['offset'],entry['count'])]
                metadata=adapt_auxiliary_jobs(jobs,starting_decks=self.contract['starting_decks'])
            entry['status']='RUNNING';entry['authorization']=authorization.strip()
            self._persist()  # Crash after this point is uncertain, never rewindable.
            try:
                collector=factory(metadata)
                if not isinstance(collector,DeviceAuxiliaryCollector):raise TypeError('Actual fresh DeviceAuxiliaryCollector required')
                with collector:
                    actual=collector.run_segment(metadata,deadline=deadline,resume=prior,seconds=seconds,
                                                 soft_seconds=soft_seconds,continuation_budget=receipt)
                checkpoint=self._save_segment(entry,actual)
                self._check_segment(entry,actual,deadline=deadline)
                if not self._clean(actual):raise ValueError('Actual auxiliary guard/rule/archive/memory failure')
                if prior is not None and actual.steps==prior.steps and not all(actual.complete):
                    raise ValueError('Actual auxiliary segment made no physical progress')
                entry['status']='COMPLETE'if all(actual.complete)else'PAUSED'
                entry.pop('failure',None)
                self.manifest['events'].append(dict(event='checkpoint',phase=entry['phase'],offset=entry['offset'],sha256=checkpoint['sha256']))
                self._persist();ran+=1
                if on_checkpoint is not None:on_checkpoint(deepcopy(entry),actual)
            except BaseException as error:
                entry['status']='REJECTED';entry['failure']=type(error).__name__+': '+str(error)[:1000]
                self.manifest['events'].append(dict(event='rejected',phase=entry['phase'],offset=entry['offset']))
                self._persist()
                if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
                return dict(status='rejected',segments=ran,manifest_sha256=self.manifest_sha256,failure=entry['failure'])
        complete=all(sum(e['count']for e in self.manifest['cohorts']if e['phase']==p and e['status']=='COMPLETE')==PHASE_COUNTS[p]for p in _PHASE_SEEDS)
        return dict(status='complete'if complete else'planned_pause',segments=ran,manifest_sha256=self.manifest_sha256)

    def completed_phase_segments(self,*,deadline=None):
        deadline=self.deadline if deadline is None else deadline
        _deadline(deadline);self._assert_current();result={p:[]for p in _PHASE_SEEDS}
        for phase in _PHASE_SEEDS:
            entries=sorted([e for e in self.manifest['cohorts']if e['phase']==phase],key=lambda e:e['offset'])
            if any(e['status']!='COMPLETE'for e in entries)or sum(e['count']for e in entries)!=PHASE_COUNTS[phase]:
                raise ValueError('All280 actual natural games required before auxiliary fitting')
            for entry in entries:
                segment=self._load_segment(entry,deadline=deadline)
                if not self._clean(segment)or not all(segment.complete):raise ValueError('Complete manifest lacks actual natural endpoints')
                result[phase].append(segment)
        return {p:tuple(v)for p,v in result.items()}

    def validate_phase_segments(self,phase_segments,*,plans,deadline):
        from .device_auxiliary_continuation import auxiliary_content_sha256
        if {p:[_physical_job(j)for j in jobs]for p,jobs in plans.items()}!=self.contract['physical_plans']:
            raise ValueError('Fitting plans differ from durable physical auxiliary plan')
        actual=self.completed_phase_segments(deadline=deadline)
        for phase in _PHASE_SEEDS:
            if [auxiliary_content_sha256(v)for v in phase_segments[phase]]!=[auxiliary_content_sha256(v)for v in actual[phase]]:
                raise ValueError('Fitting may only consume latest actual completed ledger checkpoints')
        return True
