"""Lossless, disposable isolation fitting between bounded GPU processes.

READY archives contain complete Adam, permutation and RNG state. An uncertain
RUNNING unit is never replayed. This is a capacity gate, never strength evidence.
"""
from pathlib import Path
import json
import math
import os
import time

import numpy as np

from . import recovery_capacity as capacity, seven_runtime as rt
from .batched_search.training_checkpoint import digest
from .batched_search.pure_checkpoint import read_payload, write_payload

SCHEMA = 'hybrid_capacity_segment_v1'


def _partitions(jobs, games, held_jobs, held_games, key):
    if not jobs or len(jobs) != len(games) or not held_jobs or len(held_jobs) != len(held_games):
        raise ValueError('Complete balanced training and heldout batches required')
    def group(js, gs, training):
        groups = {}
        for job, game in zip(js, gs):
            if not game['complete'] or (training and not job['training']):
                raise ValueError('Complete training/heldout games required')
            rows = [r for r in game['rows'] if r['key'] == key
                    and (not training or r['side'] in job['train_sides'])]
            if rows:
                groups.setdefault(job['seed'], []).extend(rows)
        return groups
    train, held = group(jobs, games, True), group(held_jobs, held_games, False)
    seeds, held_seeds = sorted(train), sorted(held)
    if len(seeds) < 3 or set(seeds) & set(held_seeds) or len(seeds) != len(held_seeds):
        raise ValueError('At least three balanced independent paired seed groups required')
    def opponents(js):
        return sorted((min(j['left'], j['right']), max(j['left'], j['right']), j['first'])
                      for j in js if key in (j['left'], j['right']))
    if opponents(jobs) != opponents(held_jobs):
        raise ValueError('Heldout opponent/initiative coverage differs')
    return ([r for s in seeds for r in train[s]],
            [r for s in held_seeds for r in held[s]], seeds, held_seeds)


def _admit(source, jobs, games, *, training):
    from .hybrid_joint_update import _validate_policy_row
    from .seven_covered_value import load_scope
    from .seven_round import versions
    models = versions(source)
    scopes = {}
    for job, game in zip(jobs, games):
        if (job.get('pure') is True or job.get('training') is not training
                or job.get('simulations', 32) != 32 or job.get('gumbel_candidates', 16) != 16
                or job.get('terminal_horizon', 3) != 3 or job.get('max_depth', 10) != 10
                or job.get('teacher_mode') != 'covered_value' or job.get('q_scale') != 'natural_wdl'
                or job.get('zhenhong_passive_reward') not in (None, 0) or job.get('setup_rewards')):
            raise ValueError('Qualified Gumbel32/cap16/H3 training/heldout profile required')
        policies = {s: rt.model(*spec) for s, spec in job['policies'].items()}
        if not rt.value_precheck_ok(job.get('value_precheck'), policies):
            raise ValueError('Frozen main-model WDL precheck required')
        scope = job.get('covered_value_source')
        if not scope:
            raise ValueError('Qualified auxiliary scope required')
        if scope not in scopes:
            scopes[scope] = load_scope(scope)[1]
        cert = game.get('replay_verification') or {}
        n = game.get('decisions')
        if (game.get('complete') is not True or game.get('id') != job['id']
                or game.get('rule_backend') != 'resident_cpu'
                or (training and game.get('learning_eligible') is not True)
                or type(n) is not int or n <= 0 or game.get('searched') != n
                or len(game.get('rows', ())) != n or len(game.get('episode_actions', ())) != n
                or cert.get('passed') is not True or cert.get('id') != job['id']
                or cert.get('actions') != n or not game.get('final_state_sha256')
                or cert.get('final_state_sha256') != game['final_state_sha256']
                or game.get('model_versions') != {s:p.version for s,p in policies.items()}):
            raise ValueError('Naturally complete replay-verified frozen games required')
        for side, policy in policies.items():
            if models[job['policies'][side][1]] != policy.version:
                raise ValueError('Capacity labels differ from frozen source bank')
        for step, row in enumerate(game['rows']):
            _validate_policy_row(row, step, game['episode_actions'][step], job)
            winner = game.get('winner')
            target = 0 if winner == 'draw' else (1 if winner == row['side'] else -1)
            if (winner not in ('a', 'b', 'draw') or row['z'] != target
                    or row.get('value_actor') != 1 or row.get('reward_to_go') is not None
                    or row.get('setup_reward', 0) or not np.isfinite(np.asarray(row['x'])).all()):
                raise ValueError('Exact unshaped terminal WDL policy rows required')
    return scopes


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def fit_capacity_segment(source, jobs, games, config, deadline, held_jobs, held_games,
                         key, checkpoint_dir, process_seconds=120, *, steps=200):
    """Fit one learner, resuming immutable READY state; never modify the bank.

    ``metric`` is the original recovery_capacity per-key result only after all
    epochs finish. READY means safely paused, not passed. Deadlines and the
    process budget may change without changing any training/data binding.
    """
    from .seven_checkpoint import algorithm_identity
    from .seven_round import versions, learning_keys
    if key not in learning_keys(config):
        raise ValueError('Configured learning key required')
    if type(steps) is not int or not 2 <= steps <= 400:
        raise ValueError('Bounded isolation steps required')
    if (type(process_seconds) not in (int, float) or not math.isfinite(process_seconds)
            or not 0 < process_seconds <= 120 or type(deadline) not in (int, float)
            or not math.isfinite(deadline)):
        raise ValueError('Finite deadline and at most 120 second segment required')
    train, held, seeds, held_seeds = _partitions(jobs, games, held_jobs, held_games, key)
    scopes = [_admit(source, jobs, games, training=True),
              _admit(source, held_jobs, held_games, training=False)]
    signature = versions(source)
    binding = dict(schema=SCHEMA, contract=digest(dict(source=signature,
        source_path=str(Path(source).resolve()), jobs=jobs, games=games, held_jobs=held_jobs,
        held_games=held_games, config=config, key=key, steps=steps, scopes=scopes,
        algorithm=algorithm_identity(), implementation=Path(__file__).read_text())))
    result = _fit_segment(source, train, held, seeds, held_seeds, config, deadline, key,
                          checkpoint_dir, process_seconds, steps, binding)
    if versions(source) != signature:
        raise ValueError('Isolation fit changed frozen source')
    return result


def _fit_segment(source, train, held, seeds, held_seeds, config, deadline, key,
                 checkpoint_dir, process_seconds, steps, binding):
    import torch
    device = config['device']
    stop = min(deadline, time.time() + process_seconds)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    units = sorted(root.glob('unit-*'))
    payload = None
    for index, unit in enumerate(units):
        if unit.name != f'unit-{index:06d}':
            raise ValueError('Capacity checkpoint sequence has a gap')
        receipt = json.loads((unit/'status.json').read_text())
        if (receipt.get('schema') != SCHEMA or receipt.get('binding') != binding
                or receipt.get('status') not in ('READY', 'COMPLETE')
                or receipt.get('quality_approved') is not False
                or receipt.get('strength_approved') is not False):
            raise ValueError('Uncertain RUNNING/FAILED or changed capacity stage refuses replay')
        payload = read_payload(unit/'state', binding, kind=SCHEMA)
        if payload['epoch'] != receipt['steps'] or payload['updates'] != receipt['updates']:
            raise ValueError('Capacity receipt counters differ from numeric archive')
        if (receipt.get('metric') != payload['metric'] or receipt.get('row_uses') != payload['progress']['row_uses']
                or receipt.get('complete') != (payload['metric'] is not None)
                or receipt.get('status') != ('COMPLETE' if payload['metric'] is not None else 'READY')
                or receipt.get('passed') != bool(payload['metric'] and payload['metric']['passed'])):
            raise ValueError('Capacity receipt differs from immutable numeric state')
        if receipt['status'] == 'COMPLETE':
            if unit != units[-1]:
                raise ValueError('Capacity update exists after completion')
            return receipt
    if time.time() >= stop:
        raise TimeoutError('Capacity budget expired before optimizer admission')
    unit = root/f'unit-{len(units):06d}'
    unit.mkdir(exist_ok=False)
    running = dict(schema=SCHEMA, status='RUNNING', binding=binding,
                   quality_approved=False, strength_approved=False, disposable_networks=True)
    with (unit/'status.json').open('x') as output:
        json.dump(running, output); output.flush(); os.fsync(output.fileno())
    try:
        net, _, _ = rt.restore(source, key, device)
        opt = torch.optim.Adam(net.parameters(), lr=.003)
        rng = np.random.default_rng(config['seeds']['probes'])
        if payload is None:
            state = dict(before=capacity.label_metrics(net, train, device), epoch=0,
                updates=0, row_uses=0, cursor=0, permutation=None,
                gradients={k:0. for k in ('state_net', 'cand_net', 'score')})
        else:
            state = payload['progress']
            net.load_state_dict(payload['network'])
            opt.load_state_dict(payload['optimizer'])
            rng.bit_generator.state = payload['numpy_rng']
            torch.set_rng_state(payload['torch_rng'])
            if payload['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
        while state['epoch'] < steps and time.time() < stop:
            if state['permutation'] is None:
                state['permutation'] = rng.permutation(len(train))
                state['cursor'] = 0
            indexes = state['permutation'][state['cursor']:state['cursor']+128]
            batch = [train[int(i)] for i in indexes]
            stats = rt.update(net, opt, batch)
            state['updates'] += 1
            state['row_uses'] += len(batch)
            state['cursor'] += len(batch)
            for prefix in state['gradients']:
                state['gradients'][prefix] += stats['policy_gradient_norms'].get(prefix, 0.)
            if state['cursor'] == len(train):
                state['epoch'] += 1
                state['cursor'] = 0
                state['permutation'] = None
        metric = None
        if state['epoch'] == steps:
            after = capacity.label_metrics(net, train, device)
            validation = capacity.label_metrics(net, held, device)
            train_passed = after['kl'] < .025 and after['confident_rows'] >= 8 and after['confident_accuracy'] >= .95
            held_passed = validation['kl'] < .15 and validation['confident_rows'] >= 8 and validation['confident_accuracy'] >= .75
            metric = dict(train_before=state['before'], train_after=after, heldout=validation,
                train_seed_groups=seeds, heldout_seed_groups=held_seeds, steps=state['epoch'],
                updates=state['updates'], row_uses=state['row_uses'], train_rows=len(train),
                heldout_rows=len(held), gradient_sums=state['gradients'], train_passed=train_passed,
                heldout_passed=held_passed, passed=train_passed and held_passed
                and all(np.isfinite(v) and v > 0 for v in state['gradients'].values()))
        payload = dict(progress=state, epoch=state['epoch'], updates=state['updates'], metric=metric,
            network=_cpu(dict(net.state_dict())), optimizer=_cpu(opt.state_dict()),
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=_cpu(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None)
        checkpoint = write_payload(unit/'state', payload, binding, kind=SCHEMA)
        receipt = dict(running, status='COMPLETE' if metric is not None else 'READY',
            steps=state['epoch'], updates=state['updates'], row_uses=state['row_uses'],
            checkpoint=checkpoint, metric=metric, complete=metric is not None,
            passed=bool(metric and metric['passed']))
        _write(unit/'status.json', receipt)
        return receipt
    except BaseException as exc:
        _write(unit/'status.json', dict(running, status='FAILED', error=repr(exc), replay_forbidden=True))
        raise
