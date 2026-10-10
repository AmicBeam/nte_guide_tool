"""Complete CUDA collection timing and like-for-like receipt comparison.

No optimizer or learner is created here. CPU tests validate the receipt contract,
not CUDA performance. Production numerical models retain their normal checks.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np


SCHEMA = 'duel_v2_complete_cuda_collection_v1'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode()).hexdigest()


def prepare_case(case):
    """Validate an independently bounded benchmark, without starting any work."""
    from .batched_search.training_rollout import validate_training_jobs
    config = deepcopy(case['config'])
    for field, low, high in (('search_roots', 1, 1600), ('query_batch_rows', 1, 32),
                             ('workers', 1, 64), ('training_buffer_mib', 1, 1048576)):
        value = config.get(field)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Explicit bounded benchmark capacity required: ' + field)
    if (config.get('runtime') != 'seven' or config.get('device') != 'cuda'
            or config.get('rule_backend') != 'resident_cpu'
            or config.get('search_statistics_device') != 'cuda'
            or config.get('simulations') != 32
            or config.get('teacher_mode') != 'covered_value'):
        raise ValueError('Explicit seven-team hybrid CUDA Gumbel32 covered_value required')
    if not isinstance(config.get('covered_value_source'), str) or not config['covered_value_source']:
        raise ValueError('Qualified auxiliary scope source required')
    jobs = deepcopy(case['jobs'])
    for job in jobs:
        if (not isinstance(job.get('id'), str) or not job['id']
                or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                       for ch in job['id'])):
            raise ValueError('Safe unique benchmark game id required')
        if (job.get('simulations') != 32 or job.get('gumbel_candidates') != 16
                or job.get('terminal_horizon') != 3
                or job.get('teacher_mode') != 'covered_value'
                or job.get('q_scale') != 'natural_wdl'
                or not job.get('covered_value_source') or not isinstance(job.get('value_precheck'), dict)
                or job.get('uncalibrated_value_diagnostic')):
            raise ValueError('Complete qualified Gumbel32/16/3 jobs required')
        job['policies'] = {side: tuple(spec) for side, spec in job['policies'].items()}
        job.setdefault('train_sides', ['a', 'b'])
        job.setdefault('max_actions', 800)
        job.setdefault('max_depth', 10)
        job.setdefault('training', False)
        job.setdefault('exploration', job['training'])
        for spec in job['policies'].values():
            if len(spec) != 2 or not all(isinstance(v, str) and v for v in spec):
                raise ValueError('Exact numerical model folder/key required')
        # These remain collector trajectories; this module never calls an update.
    jobs = validate_training_jobs(jobs)
    chunk_games = case.get('chunk_games', len(jobs))
    if type(chunk_games) is not int or not 1 <= chunk_games <= len(jobs):
        raise ValueError('Explicit bounded chunk size required')
    deadlines = [job.get('deadline') for job in jobs]
    if any(type(d) not in (int, float) or not math.isfinite(d) for d in deadlines):
        raise ValueError('Original finite absolute benchmark deadlines required')
    config['max_games'] = len(jobs)
    return config, jobs


def _numerical_hash(model):
    h = sha256()
    manifest = model.manifest
    h.update(json.dumps({k: manifest.get(k) for k in
                         ('schema', 'features', 'candidates', 'build', 'hidden')},
                        sort_keys=True, ensure_ascii=False).encode())
    for name, array in sorted(model.weights.items()):
        array = np.asarray(array)
        h.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        h.update(array.tobytes(order='C'))
    return h.hexdigest()


def task_identity(jobs, models, value_models):
    """Exclude relocatable source paths; include actual numerical tensors."""
    from .batched_search.training_rollout import source_alias
    rows = []
    model_hashes = {}
    def numeric(model):
        key = id(model)
        if key not in model_hashes:
            model_hashes[key] = _numerical_hash(model)
        return model_hashes[key]
    fields = ('id', 'seed', 'first', 'decks', 'training', 'exploration',
              'train_sides', 'max_actions', 'max_depth', 'simulations',
              'gumbel_candidates', 'terminal_horizon', 'teacher_mode', 'q_scale')
    for job in sorted(jobs, key=lambda j: j['id']):
        row = {k: job.get(k) for k in fields}
        row['policies'] = {side: {'key': spec[1],
                                 'numerical_sha256': numeric(models[source_alias(spec)])}
                           for side, spec in job['policies'].items()}
        rows.append(row)
    return digest({'jobs': rows, 'auxiliary': {k: numeric(v)
                                             for k, v in sorted(value_models.items())}})


def check_games(jobs, games):
    """Partial games or reduced search budgets cannot grant a timing result."""
    if len(jobs) != len(games):
        raise ValueError('Complete benchmark game count mismatch')
    summaries = {}
    for job, game in zip(jobs, games):
        n = game.get('decisions')
        if (not game.get('complete') or game.get('id') != job['id']
                or type(n) is not int or n < 1 or n != game.get('searched')
                or n != len(game.get('rows', ()))
                or n != len(game.get('episode_actions', ()))
                or not game.get('final_state_sha256')):
            raise ValueError('Actual complete searched episodes required')
        for row in game['rows']:
            candidates = np.asarray(row['c'])
            pi = np.asarray(row['pi'])
            budget = 1 if len(candidates) == 1 else 32
            if (row.get('search_budget') != budget or row.get('teacher_mode') != 'covered_value'
                    or row.get('search_algorithm') != 'gumbel' or row.get('z') not in (-1, 0, 1)
                    or len(pi) != len(candidates) or not len(pi)
                    or not np.isfinite(pi).all() or (pi < 0).any()
                    or not np.isclose(pi.sum(), 1, atol=1e-5)):
                raise ValueError('Complete original search budget and terminal labels required')
        summaries[job['id']] = {'final_state_sha256': game['final_state_sha256'],
                                'actions_sha256': digest(game['episode_actions']),
                                'decisions': n, 'winner': game.get('winner')}
    return summaries


def run_case(case, output):
    """Execute once on explicit CUDA; save genuine episodes and replay checks."""
    config, jobs = prepare_case(case)
    output = Path(output)
    if output.exists():
        raise ValueError('Benchmark output must be a new isolated directory')
    if min(job['deadline'] for job in jobs) <= time.time() + 30:
        raise ValueError('Original benchmark deadline has expired')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU performance fallback')
    from .batched_search.training_rollout import (
        models_for_jobs, rollout_from_config, verify_training_replay, training_source_identity,
    )
    from .episode_replay import dump_episode
    from .seven_round import write
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    hardware = {'host': platform.node(), 'gpu': props.name, 'uuid': str(getattr(props, 'uuid', 'unavailable')),
                'total_memory': props.total_memory, 'device': device,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'torch_threads': 1}
    setup_start = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = models_for_jobs(jobs)
    chunk_games = case.get('chunk_games', len(jobs))
    chunks = [jobs[start:start + chunk_games] for start in range(0, len(jobs), chunk_games)]
    first_config = dict(config, max_games=len(chunks[0]))
    collector = rollout_from_config(first_config, models)
    try:
        source = training_source_identity(models, config, collector.backend)
        task = task_identity(jobs, models, collector.value_models)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        output.mkdir(parents=True)
    except BaseException:
        if collector.own_statistics:
            collector.statistics.close()
        if collector.own_backend:
            collector.backend.close()
        raise
    setup_seconds = time.perf_counter() - setup_start
    games = []
    chunk_diagnostics = []
    error = None
    # Includes initial states, real complete searches, terminal encoding and
    # collector cleanup. Episode serialization/replay checks are timed separately.
    start = time.perf_counter()
    try:
        for index, chunk in enumerate(chunks):
            if index:
                collector = rollout_from_config(dict(config, max_games=len(chunk)), models)
            try:
                games.extend(collector.collect(chunk))
            finally:
                chunk_diagnostics.append(collector.last_diagnostics)
            if index < len(chunks) - 1:
                # Drop closed device pools before constructing the next chunk's
                # pools; do not temporarily allocate two full capacities.
                collector = None
        torch.cuda.synchronize(device)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        collection_seconds = time.perf_counter() - start
        write(output / 'timing.json', {'schema': SCHEMA, 'source': source, 'task_sha256': task,
              'hardware': hardware, 'setup_seconds': setup_seconds,
              'collection_seconds': collection_seconds, 'attempted_games': len(jobs),
              'complete_games': sum(bool(g.get('complete')) for g in (games or ())),
              'observed_search_root_peak': max((g.get('observed_search_root_peak', 0)
                                                for g in games), default=0),
              'training_buffer_estimated_peak_bytes': max((g.get('training_buffer_estimated_bytes', 0)
                                                          for g in games), default=0),
              'chunk_games': chunk_games, 'chunk_diagnostics': chunk_diagnostics, 'error': error,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'timing_qualified': False, 'optimization_steps': 0})
    summaries = check_games(jobs, games)
    verification_start = time.perf_counter()
    checks = []
    for job, game in zip(jobs, games):
        dump_episode(output / 'episodes' / (job['id'] + '.json.gz'), job, game)
        checks.append(verify_training_replay(job, game))
    receipt = json.loads((output / 'timing.json').read_text())
    receipt.update(episodes=summaries, replay_checks=checks,
                   verification_seconds=time.perf_counter() - verification_start,
                   gpu_runtime_verified=True,
                   files={p.relative_to(output).as_posix(): sha256(p.read_bytes()).hexdigest()
                          for p in (output / 'episodes').glob('*.json.gz')},
                   timing_qualified=all(c.get('passed') is True for c in checks))
    write(output / 'timing.json', receipt)
    return receipt


def read_receipt(path):
    """Comparison requires the real returned episode files, not a JSON manifest alone."""
    path = Path(path)
    row = json.loads(path.read_text())
    files = row.get('files') or {}
    if len(files) != row.get('complete_games') or not files:
        raise ValueError('Actual complete episode files required')
    root = path.parent.resolve()
    from .episode_replay import load_episode
    seen = set()
    for name, expected in files.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Receipt episode is missing or outside its archive')
        if sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Returned episode SHA mismatch')
        data = load_episode(file)
        gid = data['job']['id']
        summary = row.get('episodes', {}).get(gid)
        if (gid in seen or not summary or summary['actions_sha256'] != digest(data['actions'])
                or summary['decisions'] != len(data['actions'])
                or summary['winner'] != data['winner']):
            raise ValueError('Returned episode does not match its timing receipt')
        seen.add(gid)
    if seen != set(row.get('episodes', {})):
        raise ValueError('Returned episode coverage mismatch')
    return row


def compare_receipts(baseline, optimized):
    """At least three complete matched trials; never extrapolate from CPU roots."""
    if len(baseline) < 3 or len(optimized) < 3 or len(baseline) != len(optimized):
        raise ValueError('At least three trials of each matched variant required')
    all_rows = baseline + optimized
    reference = baseline[0]
    for row in all_rows:
        seconds = row.get('collection_seconds')
        if (row.get('schema') != SCHEMA or row.get('timing_qualified') is not True
                or row.get('gpu_runtime_verified') is not True
                or row.get('error') or row.get('complete_games') != row.get('attempted_games')
                or not row.get('complete_games') or row.get('optimization_steps') != 0
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                or row.get('task_sha256') != reference.get('task_sha256')
                or row.get('hardware') != reference.get('hardware')
                or row.get('episodes') != reference.get('episodes')):
            raise ValueError('Only same-device, same-numerics, same-task complete CUDA trials compare')
    old = statistics.median(r['collection_seconds'] for r in baseline)
    new = statistics.median(r['collection_seconds'] for r in optimized)
    ratio = old / new
    return {'schema': SCHEMA, 'trials_each': len(baseline), 'baseline_median_seconds': old,
            'optimized_median_seconds': new, 'ratio': ratio, 'fivefold_measured': ratio >= 5,
            'task_sha256': reference['task_sha256'], 'hardware': reference['hardware'],
            'model_quality_approved': False}
