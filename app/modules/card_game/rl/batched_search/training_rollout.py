"""Cooperative Gumbel training self-play on the existing batched-search backend.

Evaluation serving search is not reused as a label source. Physical games stream
through CooperativeScheduler with the same rule backend, observation/candidate
encoding, batched inference and optional device statistics. Only completed games
admit policy rows, per-observer terminal WDL rows, episode actions and roots.
Importing this module does not import Torch, start CUDA or train.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import uuid4
import numpy as np

from .cooperative import CooperativeScheduler, RootJob, cooperative_search, _operation
from .evaluator import (
    RULE_BACKEND_CUDA_VM,
    RULE_BACKEND_RESIDENT_CPU,
    RUNTIME_SEVEN,
    make_source_snapshot,
    report_rule_backend,
    select_rule_backend,
    statistics_capacity,
)
from .traversal import PolicyIdentity


class UnsupportedTrainingMode(ValueError):
    """A training contract was requested that this rollout does not implement."""


SUPPORTED_TEACHER_MODES = frozenset({'network', 'covered_terminal', 'covered_expert', 'covered_value'})
SUPPORTED_Q_SCALES = frozenset({'natural_wdl', 'legacy'})
MAX_SEARCH_ROOTS = 1600
DEFAULT_QUERY_BATCH_ROWS = 32
DEFAULT_SIMULATIONS = 32
DEFAULT_GUMBEL_CANDIDATES = 16
DEFAULT_TERMINAL_HORIZON = 3
DEFAULT_MAX_ACTIONS = 800


def root_search_seed(job_seed, step, actor):
    """Match seven_runtime.search_game per-root Gumbel RNG identity."""
    payload = f"recovery:{job_seed}:{step}:{actor}".encode()
    return (1 << 256) | int(hashlib.sha256(payload).hexdigest(), 16)


def policy_key(spec):
    if isinstance(spec, PolicyIdentity):
        return spec.policy_key
    if isinstance(spec, tuple) and len(spec) == 2:
        return spec[1]
    if isinstance(spec, str):
        return spec
    manifest = getattr(spec, 'manifest', None)
    if isinstance(manifest, dict) and manifest.get('deck'):
        return manifest['deck']
    raise ValueError(f'Cannot read training policy key from {spec!r}')


def _observation_vector(payload):
    if isinstance(payload, tuple) and len(payload) == 2:
        return payload[0]
    return payload


def _as_policy_identity(spec, models=None):
    if isinstance(spec, PolicyIdentity):
        return spec
    if models is not None:
        if isinstance(spec, tuple) and len(spec) == 2:
            alias = source_alias(spec)
            model = models[alias]
            return PolicyIdentity(alias, model.version)
        if isinstance(spec, str):
            model = models[spec]
            return PolicyIdentity(spec, getattr(model, 'version', spec))
    if hasattr(spec, 'manifest') and hasattr(spec, 'version'):
        return PolicyIdentity(str(spec.manifest.get('deck', 'model')), str(spec.version))
    if isinstance(spec, tuple) and len(spec) == 2:
        return PolicyIdentity(str(spec[1]), str(spec[0]))
    raise ValueError(f'Explicit policy identity required, got {spec!r}')


def source_alias(spec):
    from pathlib import Path
    folder, key = spec
    return str(key) + '@' + hashlib.sha256(str(Path(folder).resolve()).encode()).hexdigest()


def models_for_jobs(jobs):
    """Bind exact frozen folders, including historical opponents with the same deck key."""
    from app.modules.card_game.rl import seven_runtime
    models = {}
    for job in jobs:
        for spec in job['policies'].values():
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise ValueError('Production training requires exact exported folder/key sources')
            alias = source_alias(spec)
            if alias not in models:
                models[alias] = seven_runtime.model(*spec)
    return models


def validate_training_jobs(jobs):
    """Reject unsupported contracts before any root is admitted."""
    jobs = list(jobs)
    if not jobs:
        raise ValueError('Non-empty training job batch required')
    if len(jobs) > MAX_SEARCH_ROOTS:
        raise ValueError(f'Training admission is at most {MAX_SEARCH_ROOTS} resident games')
    seen = set()
    for job in jobs:
        gid = job.get('id')
        if not isinstance(gid, str) or not gid:
            raise ValueError('Training job id must be a non-empty string')
        if gid in seen:
            raise ValueError(f'Duplicate training game id: {gid}')
        seen.add(gid)
        if job.get('first') not in ('a', 'b'):
            raise ValueError("Training job first must be 'a' or 'b'")
        if type(job.get('seed')) is not int:
            raise TypeError('Training job seed must be an integer')
        if job.get('pure') is True:
            raise UnsupportedTrainingMode(
                'UNSUPPORTED pure argmax collection on cooperative Gumbel training rollout'
            )
        teacher = job.get('teacher_mode', 'network')
        if teacher not in SUPPORTED_TEACHER_MODES:
            raise UnsupportedTrainingMode(
                f'UNSUPPORTED teacher_mode={teacher!r}'
            )
        q_scale = job.get('q_scale', 'natural_wdl')
        if q_scale not in SUPPORTED_Q_SCALES:
            raise UnsupportedTrainingMode(f'UNSUPPORTED q_scale={q_scale!r}')
        if teacher != 'network' and q_scale != 'natural_wdl':
            raise UnsupportedTrainingMode('Covered teachers require natural_wdl')
        if job.get('guided_policy') is not None:
            raise UnsupportedTrainingMode(
                'UNSUPPORTED guided_policy; refusing to drop the guide or emulate it from frozen evaluation'
            )
        algorithm = job.get('search_algorithm', 'gumbel')
        if algorithm not in (None, 'gumbel'):
            raise UnsupportedTrainingMode(f'UNSUPPORTED search_algorithm={algorithm!r}')
        if job.get('zhenhong_passive_reward') not in (None, 0):
            raise UnsupportedTrainingMode('UNSUPPORTED shaped zhenhong_passive_reward')
        if job.get('setup_rewards'):
            raise UnsupportedTrainingMode('UNSUPPORTED setup_rewards')
        if not isinstance(job.get('policies'), Mapping) or set(job['policies']) != {'a', 'b'}:
            raise ValueError('Training jobs require both side policies')
        if not isinstance(job.get('decks'), Mapping) or set(job['decks']) != {'a', 'b'}:
            raise ValueError('Training jobs require both side decks')
        sims = job.get('simulations', DEFAULT_SIMULATIONS)
        if type(sims) is not int or sims < 1:
            raise ValueError('Positive simulation count required')
        for field,default,minimum in (('gumbel_candidates',16,1),('terminal_horizon',3,0),('max_depth',10,1)):
            value=job.get(field,default)
            if type(value) is not int or value<minimum:
                raise ValueError('Invalid bounded search setting: '+field)
        if type(job.get('training',False)) is not bool or type(job.get('exploration',job.get('training',False))) is not bool:
            raise ValueError('Boolean training/exploration controls required')
        if (type(job.get('max_actions', DEFAULT_MAX_ACTIONS)) is not int
                or job.get('max_actions', DEFAULT_MAX_ACTIONS) < 1):
            raise ValueError('Positive per-game action limit required')
        if not isinstance(job.get('train_sides', ('a', 'b')), (tuple, list)) or not set(job.get('train_sides', ('a','b'))) <= {'a','b'}:
            raise ValueError('Known training sides required')
    return jobs


def assert_inference_frozen(inference):
    """Live learner weights must not move while roots are in flight."""
    checker = getattr(inference, 'assert_frozen', None)
    if callable(checker):
        checker()
        return
    networks = getattr(inference, '_networks', None)
    if not networks:
        return
    for net in networks.values():
        params = list(getattr(net, 'parameters', lambda: ())())
        if any(getattr(p, 'requires_grad', False) for p in params):
            raise RuntimeError('Training rollout requires frozen inference weights')
        if getattr(net, 'training', False):
            raise RuntimeError('Training rollout requires eval-mode inference')


def monotonic_deadline(wall_deadline, clock=time.monotonic):
    if wall_deadline is None:
        return None
    if isinstance(wall_deadline, bool) or not isinstance(wall_deadline, (int, float)) or not math.isfinite(wall_deadline):
        raise ValueError('Finite training deadline required')
    remaining = wall_deadline - time.time()
    return clock() + remaining


def bind_side_policies(job, models):
    return {side: _as_policy_identity(spec, models) for side, spec in job['policies'].items()}


def _value_row(x, side, key, step, actor):
    return dict(x=np.array(x, dtype=np.float32, copy=True), side=side, key=key, step=step,
                value_actor=int(side == actor))


def _policy_row(result, *, actor, key, step, phase, chosen):
    pi = np.asarray(result['pi'], dtype=np.float32)
    visit = np.asarray(result.get('visit_pi', pi), dtype=np.float32)
    if pi.ndim != 1 or len(pi) != len(result['c']) or not np.isfinite(pi).all():
        raise ValueError('Invalid Gumbel policy target')
    if not np.isclose(float(pi.sum()), 1.0, atol=1e-5) or np.any(pi < 0):
        raise ValueError('Gumbel pi must be a finite probability vector')
    return dict(
        x=np.array(result['x'], dtype=np.float32, copy=True),
        c=np.array(result['c'], dtype=np.float32, copy=True),
        pi=pi,
        visit_pi=visit,
        selected=int(chosen),
        side=actor,
        key=key,
        step=step,
        phase=phase,
        value_actor=1,
        search_budget=int(result.get('simulations', 0)),
        policy_source=result.get('policy_source', 'gumbel_q_natural_wdl'),
        search_algorithm='gumbel',
        teacher_mode=result.get('teacher_mode', 'network'),
        q_scale=result.get('q_scale', 'natural_wdl'),
    )


def _terminal_z(winner, side):
    if winner not in ('a', 'b', 'draw', None):
        raise ValueError('Missing terminal winner')
    if winner not in ('a', 'b'):
        return 0
    return 1 if winner == side else -1


def _incomplete(job, reason, roots=None):
    return dict(
        complete=False, reason=reason, winner=None, rows=[], value_rows=[],
        episode_actions=[], roots=list(roots or []), decisions=0, searched=0,
        turn=None, learning_eligible=False, model_versions={},
        id=job.get('id'),
    )


@dataclass
class TrainingRollout:
    """One frozen cooperative collector. CPU orchestration, backend-owned rules."""

    inference: Any
    backend: Any
    models: Mapping[str, Any]
    search_roots: int = MAX_SEARCH_ROOTS
    query_batch_rows: int = DEFAULT_QUERY_BATCH_ROWS
    simulations: int = DEFAULT_SIMULATIONS
    statistics: Any = None
    clock: Callable[[], float] = time.monotonic
    rule_backend: str = RULE_BACKEND_RESIDENT_CPU
    runtime_name: str = RUNTIME_SEVEN
    capacity_plan: Any = None
    create_states: Optional[Callable[[Sequence[dict]], list]] = None
    allow_cpu_oracle: bool = False
    own_backend: bool = False
    own_statistics: bool = False
    value_models: Any = None
    max_buffer_bytes: int = 8 * 1024**3
    last_diagnostics: Optional[dict] = None
    async_rule_dispatch: Optional[bool] = None

    def collect(self, jobs, *, deadline=None, max_actions=DEFAULT_MAX_ACTIONS):
        self.last_diagnostics = {}
        return collect_training_games(
            jobs, inference=self.inference, backend=self.backend, models=self.models,
            search_roots=self.search_roots, query_batch_rows=self.query_batch_rows,
            simulations=self.simulations, statistics=self.statistics, clock=self.clock,
            rule_backend=self.rule_backend, runtime_name=self.runtime_name,
            capacity_plan=self.capacity_plan, create_states=self.create_states,
            allow_cpu_oracle=self.allow_cpu_oracle, deadline=deadline,
            max_actions=max_actions, own_backend=self.own_backend,
            own_statistics=self.own_statistics,
            value_models=self.value_models,
            max_buffer_bytes=self.max_buffer_bytes,
            diagnostics=self.last_diagnostics,
            async_rule_dispatch=self.async_rule_dispatch,
        )


def _capture_diagnostics(sink, scheduler, inference, backend, statistics, error):
    """Read aggregate counters before owned resources close; preserve failures."""
    if sink is None:
        return
    sink.update(schema='cooperative_search_diagnostics_v1', method_error=error,
                scheduler_dispatch=deepcopy(getattr(scheduler, 'dispatch_metrics', None)),
                scheduler_progress=deepcopy(getattr(scheduler, 'progress_metrics', None)))
    errors = {}
    for name, resource in (('inference', inference), ('rules', backend),
                           ('device_statistics', statistics)):
        reader = getattr(resource, 'stats', None)
        try:
            sink[name] = deepcopy(reader()) if callable(reader) else None
        except Exception as exc:
            sink[name] = None
            errors[name] = repr(exc)
    sink['diagnostic_errors'] = errors


def _collect_training_games(
    jobs,
    *,
    inference,
    backend,
    models,
    search_roots=MAX_SEARCH_ROOTS,
    query_batch_rows=DEFAULT_QUERY_BATCH_ROWS,
    simulations=DEFAULT_SIMULATIONS,
    statistics=None,
    clock=time.monotonic,
    rule_backend=None,
    runtime_name=RUNTIME_SEVEN,
    capacity_plan=None,
    create_states=None,
    allow_cpu_oracle=False,
    deadline=None,
    max_actions=DEFAULT_MAX_ACTIONS,
    own_backend=False,
    own_statistics=False,
    value_models=None,
    max_buffer_bytes=8 * 1024**3,
    diagnostics=None,
    async_rule_dispatch=None,
):
    """Stream complete official games and admit training rows only after terminal WDL.

    search_roots is the live Gumbel-root window (up to 1600). query_batch_rows is
    the kernel/call workspace (default 32) and is never reported as root concurrency.
    """
    jobs = validate_training_jobs(jobs)
    if async_rule_dispatch is not None and type(async_rule_dispatch) is not bool:
        raise TypeError('Boolean or automatic async rule dispatch required')
    if async_rule_dispatch is True and not getattr(backend, 'supports_async_operations', False):
        raise ValueError('Explicit async rules require a capable resident backend')
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError('Diagnostics sink must be a dictionary')
    if type(search_roots) is not int or not 1 <= search_roots <= MAX_SEARCH_ROOTS:
        raise ValueError('search_roots must be an integer in 1..1600')
    if type(query_batch_rows) is not int or not 1 <= query_batch_rows <= DEFAULT_QUERY_BATCH_ROWS:
        raise ValueError('query_batch_rows must be an integer in 1..32; it is not root concurrency')
    if type(simulations) is not int or simulations < 1:
        raise ValueError('Positive simulation count required')
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError('Positive max_actions required')
    if type(max_buffer_bytes) is not int or max_buffer_bytes < 1:
        raise ValueError('Positive bounded training buffer required')
    resolved_backend = report_rule_backend(backend, rule_backend)
    if runtime_name != RUNTIME_SEVEN:
        raise ValueError('Seven-team training runtime required')
    if resolved_backend == RULE_BACKEND_CUDA_VM:
        if allow_cpu_oracle:
            raise RuntimeError('CUDA training rollout cannot opt into CPU oracle fallback')
        if getattr(backend, 'fake_label', None) and getattr(backend, 'device_verified', False):
            raise RuntimeError('Labeled GPU fakes cannot claim device_verified')
        if statistics is None and not getattr(backend, 'fake_label', None):
            raise RuntimeError(
                'cuda_vm training requires the device statistics pool; host-tree statistics are not a CUDA fallback'
            )
    assert_inference_frozen(inference)
    requested_search_roots = search_roots
    search_roots = min(search_roots, len(jobs))
    wall_deadlines = [job.get('deadline') for job in jobs]
    if deadline is not None:
        wall_deadlines.append(deadline)
    if any(isinstance(d,bool) or not isinstance(d,(int,float)) or not math.isfinite(d) for d in wall_deadlines):
        raise ValueError('Every training game requires a finite original epoch deadline')
    search_deadline = monotonic_deadline(min(wall_deadlines), clock) - 30.0

    identities = {job['id']: bind_side_policies(job, models) for job in jobs}
    qualified = {}
    fixture = create_states is not None and (allow_cpu_oracle or str(getattr(backend,'fake_label','')).startswith('FAKE'))
    from app.modules.card_game.rl.seven_runtime import value_precheck_ok
    value_scopes={}
    for job in jobs:
        policies = {side: models[identity.policy_key] for side,identity in identities[job['id']].items()}
        qualified[job['id']] = fixture or value_precheck_ok(job.get('value_precheck'), policies)
        if not qualified[job['id']] and (job.get('training') or not job.get('uncalibrated_value_diagnostic')):
            raise ValueError('Actual frozen-model WDL precheck required for cooperative training')
        teacher=job.get('teacher_mode','network')
        if teacher!='network' and (not callable(getattr(backend,'rule_decision',None)) or
                hasattr(backend,'program') and 'rule_decision' not in backend.program.entries):
            raise ValueError('Selected backend/program lacks the canonical covered teacher')
        if teacher=='covered_value' and value_models is None:
            raise ValueError('Qualified covered value models required before admission')
        if teacher=='covered_value' and not fixture:
            source=job.get('covered_value_source')
            if not source:
                raise ValueError('Qualified auxiliary scope source required')
            if source not in value_scopes:
                from app.modules.card_game.rl.seven_covered_value import load_scope
                value_scopes[source]=load_scope(source)[0]
            if ({k:model.version for k,model in value_models.items()} !=
                    {k:model.version for k,model in value_scopes[source].items()}):
                raise ValueError('Covered value inference arrays differ from qualified scope')

    entered = False
    stats_entered = False
    if own_backend and hasattr(backend, '__enter__'):
        backend.__enter__()
        entered = True
    if own_statistics and statistics is not None and hasattr(statistics, '__enter__'):
        statistics.__enter__()
        stats_entered = True

    game_ids = [job['id'] for job in jobs]
    job_map = {job['id']: job for job in jobs}
    records = {gid: dict(rows=[], values=[], trace=[], roots=[], step=0, searched=0) for gid in game_ids}
    finished = {}
    uncompleted = {}
    action_counts = {gid: 0 for gid in game_ids}
    active = {}
    searching = {}
    method_error = None
    scheduler = None
    peak = 0
    ready = deque()
    buffer_bytes = 0

    def retain(value):
        nonlocal buffer_bytes
        def size(item):
            if isinstance(item,np.ndarray):return item.nbytes
            if isinstance(item,dict):return 256+sum(size(k)+size(v) for k,v in item.items())
            if isinstance(item,(list,tuple)):return 64+sum(size(v) for v in item)
            return 64+len(item) if isinstance(item,str) else 32
        buffer_bytes += size(value)
        if buffer_bytes > max_buffer_bytes:
            raise MemoryError('Training trajectory buffer budget exceeded; no partial labels admitted')
        return value

    def create_initial_states():
        if create_states is not None:
            return create_states(jobs)
        from app.modules.card_game.engine.duel_v2 import new_game
        return [new_game(seed=job['seed'], decks=job['decks'], first_side=job['first']) for job in jobs]

    def retire(refs, job_rows, reason=None):
        terminal = [ref for ref in refs if ref.phase=='finished' and reason is None]
        exports = backend.export_states(terminal) if terminal else []
        if len(exports)!=len(terminal):
            raise ValueError('Final-state export lost a completed training game')
        final_states = {ref.game_id:state for ref,state in zip(terminal,exports)}
        for ref, job in zip(refs, job_rows):
            gid = job['id']
            active.pop(gid, None)
            rec = records[gid]
            phase = getattr(ref, 'phase', None)
            winner = getattr(ref, 'winner', None)
            complete = phase == 'finished' and reason is None
            if complete and (rec['searched'] != len(rec['rows']) or rec['searched'] != len(rec['trace'])):
                complete = False
                reason = 'incomplete_search'
            if complete:
                z_by_side = {side: _terminal_z(winner, side) for side in ('a', 'b')}
                for row in rec['rows'] + rec['values']:
                    row['z'] = z_by_side[row['side']]
                versions = {side: identity.model_version for side, identity in identities[gid].items()}
                from app.modules.card_game.rl.league_rollout import clean
                final = clean(final_states[ref.game_id])
                digest = hashlib.sha256(__import__('json').dumps(final,sort_keys=True,
                    separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                finished[gid] = dict(
                    complete=True, reason=None, winner=winner,
                    rows=rec['rows'], value_rows=rec['values'],
                    episode_actions=rec['trace'], roots=rec['roots'],
                    decisions=len(rec['trace']), searched=rec['searched'],
                    turn=final['turn'], final_state_sha256=digest,
                    learning_eligible=bool(job.get('training') and qualified[gid] and rec['searched']),
                    model_versions=versions, id=gid,
                )
            else:
                uncompleted[gid] = _incomplete(job, reason or 'evaluation_stopped', rec['roots'])
        if refs:
            backend.release(refs)

    def admit():
        nonlocal peak
        while ready:
            gid, ready_at = ready.popleft()
            job, ref = active[gid]
            if getattr(ref, 'phase', None) == 'finished':
                retire([ref], [job])
                continue
            if action_counts[gid] >= min(max_actions, job.get('max_actions', DEFAULT_MAX_ACTIONS)):
                retire([ref], [job], reason='max_actions')
                continue
            if search_deadline is not None and clock() >= search_deadline:
                retire([ref], [job], reason='deadline')
                continue
            actor = getattr(ref, 'actor', None)
            if actor not in ('a', 'b'):
                raise RuntimeError(f'P0: Abnormal training metadata for game {gid}')
            rec = records[gid]
            step = rec['step']
            policies = identities[gid]
            token = f'{gid}:t{ref.root_generation}:{uuid4().hex[:8]}'
            q_scale = job.get('q_scale', 'natural_wdl')
            noise = bool(job.get('exploration', job.get('training', False)))
            sims = int(job.get('simulations', simulations))
            search = cooperative_search(
                ref, actor, policies, backend=backend,
                seed=root_search_seed(job['seed'], step, actor),
                simulations=sims,
                gumbel_candidates=int(job.get('gumbel_candidates', DEFAULT_GUMBEL_CANDIDATES)),
                terminal_horizon=int(job.get('terminal_horizon', DEFAULT_TERMINAL_HORIZON)),
                max_depth=int(job.get('max_depth', 10)),
                q_scale=q_scale, noise=noise, deadline=search_deadline, clock=clock,
                root_id=token, statistics=statistics, collect_diagnostics=True,
                teacher_mode=job.get('teacher_mode','network'),
                value_policy_identities_by_side={side:PolicyIdentity('aux@'+policy_key(spec),
                    value_models[policy_key(spec)].version) for side,spec in job['policies'].items()}
                    if job.get('teacher_mode')=='covered_value' else None,
            )
            def queued_observations(search=search, ref=ref, job=job, rec=rec, step=step, actor=actor):
                try:
                    for side in ('a','b'):
                        payload = yield from _operation(backend, 'value_observation', ref, side)
                        rec['values'].append(retain(_value_row(_observation_vector(payload), side,
                                                       policy_key(job['policies'][side]), step, actor)))
                    return (yield from search)
                finally:
                    search.close()
            gen = queued_observations()
            searching[token] = (gid, job, ref, actor, rec)
            peak = max(peak, len(searching))
            return RootJob(token, ref.game_id, ref.root_generation, gen, search_deadline)
        return None

    def completed(rows):
        commits = []
        old_refs = []
        rejected_refs = []
        rejected_jobs = []
        rejected_reasons = []
        commit_context = []
        for root, search_res in rows:
            gid, job, ref, actor, rec = searching.pop(root.root_id)
            deadline_hit = search_deadline is not None and clock() >= search_deadline
            if not search_res.get('complete') or search_res.get('search_choice') is None:
                rec['rows'] = []
                rec['values'] = []
                rec['trace'] = []
                rec['roots'].append(dict(
                    step=rec['step'], side=actor, complete=False,
                    stop_reason=search_res.get('stop_reason'),
                ))
                rejected_refs.append(ref)
                rejected_jobs.append(job)
                rejected_reasons.append('deadline' if deadline_hit else 'incomplete_search')
                continue
            actions = search_res.get('actions') or []
            chosen = search_res['search_choice']
            selected = actions[chosen]
            rec['rows'].append(retain(_policy_row(
                search_res, actor=actor, key=policy_key(job['policies'][actor]),
                step=rec['step'], phase=getattr(ref, 'phase', None), chosen=chosen,
            )))
            rec['trace'].append(dict(side=actor, action=selected))
            rec['roots'].append(retain(dict(
                step=rec['step'], side=actor, chosen=selected, legal=len(actions),
                diagnostics=search_res.get('root_diagnostics'),
            )))
            rec['searched'] += 1
            rec['step'] += 1
            if deadline_hit:
                rec['rows'] = []
                rec['values'] = []
                rec['trace'] = []
                rejected_refs.append(ref)
                rejected_jobs.append(job)
                rejected_reasons.append('deadline')
                continue
            commits.append((ref, actor, selected, ref.root_generation + 1, dict(
                simulations=search_res.get('simulations'),
                policy_source=search_res.get('policy_source'),
                q_scale=search_res.get('q_scale'),
            )))
            old_refs.append(ref)
            commit_context.append(gid)
        if commits:
            new_refs = backend.commit_actions(commits)
            if len(new_refs) != len(commits):
                raise ValueError('Physical commit lost a training game')
            backend.release(old_refs)
            for ref, gid in zip(new_refs, commit_context):
                active[gid] = (job_map[gid], ref)
                action_counts[gid] += 1
                ready.append((gid, clock()))
        if rejected_refs:
            grouped = {}
            for ref, job, reason in zip(rejected_refs, rejected_jobs, rejected_reasons):
                grouped.setdefault(reason, ([], []))
                grouped[reason][0].append(ref)
                grouped[reason][1].append(job)
            for reason, (refs, job_rows) in grouped.items():
                retire(refs, job_rows, reason=reason)

    try:
        states = create_initial_states()
        if len(states) != len(jobs):
            raise ValueError('State factory must return one state per job')
        refs = backend.import_states(states, game_ids, [1] * len(jobs))
        del states
        if len(refs) != len(jobs):
            raise ValueError('Backend import lost a training game')
        for job, ref in zip(jobs, refs):
            active[job['id']] = (job, ref)
            ready.append((job['id'], clock()))
        scheduler = CooperativeScheduler(
            inference, max_roots=search_roots, allow_cpu_oracle=allow_cpu_oracle,
            async_rule_dispatch=async_rule_dispatch,
        )
        cancelled = (lambda: clock() >= search_deadline) if search_deadline is not None else None
        scheduler_stats = scheduler.run_stream(admit, completed, cancelled=cancelled)
        peak = max(peak, scheduler_stats.get('peak_outstanding_roots', 0))
    except Exception as exc:
        method_error = repr(exc)
        raise
    finally:
        leftover = [ref for _, ref in active.values()]
        leftover_jobs = [job for job, _ in active.values()]
        if leftover:
            reason = 'method_error' if method_error else (
                'deadline' if search_deadline is not None and clock() >= search_deadline else 'evaluation_stopped'
            )
            try:
                retire(leftover, leftover_jobs, reason=reason)
            except Exception:
                try:
                    backend.release(leftover)
                except Exception:
                    pass
                for job in leftover_jobs:
                    uncompleted.setdefault(job['id'], _incomplete(job, 'backend_unrecoverable'))
            active.clear()
        _capture_diagnostics(diagnostics, scheduler, inference, backend, statistics, method_error)
        if stats_entered:
            statistics.__exit__(None, None, None)
        if entered:
            backend.__exit__(None, None, None)

    results = []
    for job in jobs:
        gid = job['id']
        if gid in finished:
            results.append(finished[gid])
        elif gid in uncompleted:
            results.append(uncompleted[gid])
        else:
            results.append(_incomplete(job, 'missing_result'))
    for game in results:
        game['search_root_limit'] = search_roots
        game['requested_search_roots'] = requested_search_roots
        game['query_batch_rows'] = query_batch_rows
        game['observed_search_root_peak'] = peak
        game['rule_backend'] = resolved_backend
        game['runtime'] = runtime_name
        game['capacity_plan'] = capacity_plan if capacity_plan is not None else getattr(backend, 'capacity_plan', None)
        game['training_buffer_estimated_bytes'] = buffer_bytes
        game['training_buffer_budget_bytes'] = max_buffer_bytes
    return results


def collect_training_games(jobs, **kwargs):
    """Close owned resources even when pre-admission validation fails."""
    try:
        return _collect_training_games(jobs, **kwargs)
    except Exception as exc:
        sink = kwargs.get('diagnostics')
        if isinstance(sink, dict) and 'schema' not in sink:
            _capture_diagnostics(sink, None, kwargs.get('inference'), kwargs.get('backend'),
                                 kwargs.get('statistics'), repr(exc))
        raise
    finally:
        for flag, name in (('own_statistics', 'statistics'), ('own_backend', 'backend')):
            resource = kwargs.get(name)
            if kwargs.get(flag) and resource is not None:
                close = getattr(resource, 'close', None)
                if callable(close):
                    close()


def gpu_training_requested(config):
    backend = config.get('rule_backend') or config.get('training_rule_backend')
    return backend in (RULE_BACKEND_CUDA_VM, RULE_BACKEND_RESIDENT_CPU, 'cuda_batched')


def verify_training_replay(job, game, *, initial_state=None):
    """Explicit CPU replay validation of completed GPU actions, never a rule fallback."""
    import json
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side, apply_action
    from app.modules.card_game.rl.league_rollout import clean
    if not game.get('complete') or not game.get('final_state_sha256'):
        raise ValueError('Completed physical GPU state digest required')
    state = initial_state if initial_state is not None else new_game(seed=job['seed'],decks=job['decks'],first_side=job['first'])
    for entry in game['episode_actions']:
        if acting_side(state)!=entry['side']:
            raise ValueError('Replay decision owner differs from GPU trajectory')
        state=clean(apply_action(state,entry['side'],entry['action']))
    digest=hashlib.sha256(json.dumps(clean(state),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if state['phase']!='finished' or state.get('winner')!=game['winner'] or digest!=game['final_state_sha256']:
        raise ValueError('GPU training trajectory disagrees with formal CPU replay')
    return dict(id=job['id'],passed=True,actions=len(game['episode_actions']),final_state_sha256=digest)


def bind_training_backend(config, *, gpu_backend_cls=None, resident_backend_cls=None,
                          program_api=None, capacity_planner=None, fake_label=None):
    """Construct the explicit rule backend. cuda_vm never falls back to resident CPU."""
    rule_backend = config.get('rule_backend') or config.get('training_rule_backend')
    if rule_backend == 'cuda_batched':
        rule_backend = RULE_BACKEND_CUDA_VM
    if rule_backend not in (RULE_BACKEND_RESIDENT_CPU, RULE_BACKEND_CUDA_VM):
        raise ValueError(f'Unsupported training rule backend {rule_backend!r}')
    runtime = config.get('runtime', RUNTIME_SEVEN)
    if runtime != RUNTIME_SEVEN:
        raise ValueError('Cooperative GPU training rollout is wired for the seven-team runtime')
    if rule_backend == RULE_BACKEND_CUDA_VM:
        if config.get('device') != 'cuda':
            raise ValueError('CUDA training rollout cannot silently use a CPU device')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA device requested but torch.cuda.is_available() is False. No CPU rule fallback.'
            )
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Explicit FP32 contract requires TF32 disabled by the calling process')
    search_roots = int(config.get('search_roots', MAX_SEARCH_ROOTS))
    max_games = int(config.get('max_games', search_roots))
    return select_rule_backend(
        rule_backend=rule_backend,
        runtime=runtime,
        workers=int(config.get('workers', 1)),
        max_games=max_games,
        search_roots=search_roots,
        gpu_rule_program=config.get('gpu_rule_program'),
        guard_standalone_entities=bool(config.get('guard_standalone_entities', False)),
        gpu_memory_budget_mib=config.get('gpu_memory_budget_mib'),
        max_nodes=config.get('gpu_max_nodes'),
        max_edges=config.get('gpu_max_edges'),
        max_payload_bytes=config.get('gpu_max_payload_bytes'),
        max_candidates=config.get('gpu_max_candidates'),
        resident_backend_cls=resident_backend_cls,
        gpu_backend_cls=gpu_backend_cls,
        capacity_planner=capacity_planner,
        program_api=program_api,
        fake_label=fake_label,
        query_batch_rows=config.get('query_batch_rows',DEFAULT_QUERY_BATCH_ROWS),
    )


def bind_training_inference(models, *, device='cpu', query_batch_rows=DEFAULT_QUERY_BATCH_ROWS):
    from .inference import BatchedInference
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA inference requested but torch.cuda.is_available() is False. No CPU fallback.'
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from app.modules.card_game.rl.seven_lineup import CAND_DIM
    inference = BatchedInference(models, device=device, max_batch_rows=query_batch_rows,
                                 max_candidate_elements=query_batch_rows * 128 * CAND_DIM)
    inference.assert_frozen()
    return inference


def training_source_identity(models, config, backend=None, models_dir=None):
    plan = config.get('capacity_plan')
    if plan is None:
        plan = getattr(backend, 'capacity_plan', None)
    snapshot = make_source_snapshot(
        models,
        budget_mode='simulations',
        models_dir=models_dir,
        rule_backend=config.get('rule_backend', RULE_BACKEND_RESIDENT_CPU),
        runtime=config.get('runtime', RUNTIME_SEVEN),
        search_roots=config.get('search_roots', MAX_SEARCH_ROOTS),
        games=config.get('max_games'),
        capacity_plan=plan,
        gpu_program_identity=getattr(getattr(backend, 'program', None), 'identity', None),
        gpu_memory_budget_mib=config.get('gpu_memory_budget_mib'),
        guard_standalone_entities=bool(config.get('guard_standalone_entities', False))
        if config.get('rule_backend') == RULE_BACKEND_CUDA_VM else None,
        device_verified=bool(getattr(backend, 'device_verified', False)) if backend is not None else False,
    )
    snapshot['training_rollout'] = True
    snapshot['query_batch_rows'] = int(config.get('query_batch_rows', DEFAULT_QUERY_BATCH_ROWS))
    snapshot['simulations'] = int(config.get('simulations', DEFAULT_SIMULATIONS))
    snapshot['training_buffer_mib'] = int(config.get('training_buffer_mib',8192))
    snapshot['search_statistics_device'] = config.get('search_statistics_device')
    snapshot['async_rule_dispatch'] = config.get('async_rule_dispatch')
    snapshot['collection_replicas'] = config.get('collection_replicas', 1)
    snapshot['search_code_includes_training_rollout'] = True
    snapshot['query_batch_is_not_root_concurrency'] = snapshot['query_batch_rows'] != snapshot.get('requested_search_roots')
    return snapshot


def rollout_from_config(config, models, *, inference=None, backend=None, statistics=None,
                        create_states=None, gpu_backend_cls=None, fake_label=None):
    """Build a frozen collector. Caller owns live learner nets separately."""
    own_backend = backend is None
    own_statistics = False
    requested_statistics = config.get('search_statistics_device')
    if requested_statistics not in (None, 'cuda'):
        raise ValueError('Known explicit search statistics device required')
    if requested_statistics == 'cuda' and config.get('device') != 'cuda':
        raise ValueError('CUDA statistics require the explicit CUDA training device')
    value_models=None
    if config.get('teacher_mode')=='covered_value':
        from app.modules.card_game.rl.seven_covered_value import load_scope
        if not config.get('covered_value_source'):
            raise ValueError('Qualified seven-team covered value source required')
        value_models,_=load_scope(config['covered_value_source'])
        if inference is not None:
            raise ValueError('Explicit inference cannot bypass qualified auxiliary model binding')
        inference_models=dict(models,**{'aux@'+key:model for key,model in value_models.items()})
    else:
        inference_models=models
    if backend is None:
        backend = bind_training_backend(
            config, gpu_backend_cls=gpu_backend_cls, fake_label=fake_label,
        )
    try:
        if inference is None:
            inference = bind_training_inference(inference_models, device=config.get('device', 'cpu'),
                                               query_batch_rows=config.get('query_batch_rows',32))
        else:
            assert_inference_frozen(inference)
        search_roots = int(config.get('search_roots', MAX_SEARCH_ROOTS))
        query_batch_rows = int(config.get('query_batch_rows', DEFAULT_QUERY_BATCH_ROWS))
        resolved_backend = report_rule_backend(backend, config.get('rule_backend'))
        if statistics is None and (resolved_backend == RULE_BACKEND_CUDA_VM or requested_statistics == 'cuda'):
            from .statistics import BatchedStatistics
            node_capacity, candidate_capacity = statistics_capacity(
                resolved_backend, search_roots, getattr(backend, 'capacity_plan', None),
            )
            statistics = BatchedStatistics(
                node_capacity, candidate_capacity, device=config.get('device', 'cuda'),
                q_scale='natural_wdl',
            )
            own_statistics = True
        if requested_statistics == 'cuda' and not str(getattr(getattr(statistics, 'pool', None), 'device', '')).startswith('cuda'):
            raise ValueError('Explicit CUDA statistics cannot use a host statistics pool')
    except BaseException:
        if own_statistics and statistics is not None:
            statistics.close()
        if own_backend:
            backend.close()
        raise
    return TrainingRollout(
        inference=inference, backend=backend, models=models,
        search_roots=search_roots, query_batch_rows=query_batch_rows,
        simulations=int(config.get('simulations', DEFAULT_SIMULATIONS)),
        statistics=statistics, rule_backend=config.get('rule_backend', RULE_BACKEND_RESIDENT_CPU),
        runtime_name=config.get('runtime', RUNTIME_SEVEN),
        capacity_plan=getattr(backend, 'capacity_plan', None),
        create_states=create_states, own_backend=own_backend, own_statistics=own_statistics,
        value_models=value_models,
        max_buffer_bytes=int(config.get('training_buffer_mib',8192))*1024**2,
        async_rule_dispatch=config.get('async_rule_dispatch'),
    )
