#!/usr/bin/env python3
"""Five-preset residual recovery with independent WDL and paired initiative.

Default hard stop: next 09:00 Asia/Shanghai. Fixed builds; no deck search or
website approval. --smoke validates the chain, never model quality.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.modules.card_game.rl.cross_schedule import ROSTER, learning_schedule, matrix_schedule, coverage
from app.modules.card_game.rl.five_cross_gates import first_gate, late_gate, mid_gate, schedule_gates


def shanghai_zone():
    return timezone(timedelta(hours=8))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def append(path, data):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def policy_update_record(key, cycle, refs, sampling, metrics):
    return {**metrics, 'key': key, 'cycle': cycle, 'refs': refs, 'sampling': sampling}


def write_training_coverage(out, training_games):
    planned = [dict(left=row['left'], right=row['right'], first=row['first'])
               for row in training_games]
    write(Path(out) / 'training-coverage.json', coverage(training_games, planned))


def cross_pairs():
    return [(a, b) for a in ROSTER for b in ROSTER if a != b]


def validate_round_launch(config, *, resume=False):
    """Check recovery evidence at every entry; keep historical formal cross closed."""
    if isinstance(config,dict) and config.get('runtime')=='recovery':
        from app.modules.card_game.rl.recovery_round import validate_launch
        validate_launch(config,resume=resume)
        return
    if not isinstance(config, dict) or config.get('smoke') is not True:
        raise ValueError('Formal five-preset training is disabled: recovery stack integration, '
                         'all-five readiness and CUDA/throughput validation are pending; '
                         'worker entry cannot bypass this gate')


def preset_decks():
    from copy import deepcopy
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
    found = {d['id']: d for d in STARTER_DECKS}
    return {k: validate_deck(deepcopy(found[k])) for k in ROSTER}


def play_job(job):
    from app.modules.card_game.rl.information_search import search_game
    return search_game(job)


def evaluate_job(job):
    from app.modules.card_game.rl.full_cycle_experiment import evaluate_job as run
    return run(job)


def make_job(item, *, seed, models, decks, config, deadline, identity, training):
    left, right = item['left'], item['right']
    return dict(item, id=identity, runtime='cross', seed=seed,
                decks={'a': decks[left], 'b': decks[right]},
                policies={'a': (str(models), left), 'b': (str(models), right)},
                simulations=config['simulations'], search_algorithm='gumbel',
                gumbel_candidates=config['gumbel_candidates'], terminal_horizon=config['terminal_horizon'],
                fast_simulation=True, training=training, collect=training,
                record_episode=training, train_sides=['a', 'b'],
                zhenhong_passive_reward=0, setup_rewards={},
                deadline=deadline, max_actions=800)


def _play_fraction(net, rows, device):
    import torch
    from app.modules.card_game.rl.league_learning import tensors
    from app.modules.card_game.rl.league_schema import ACT_PLAY
    if not rows:
        return 0, 0
    played = 0
    xx, c, mask = tensors([(row['x'], row['c']) for row in rows], device)
    scores, _ = net(xx, c, mask)
    chosen = scores.argmax(-1).tolist()
    for index, row in enumerate(rows):
        played += int(int(row['c'][chosen[index]][0]) == ACT_PLAY)
    return played / len(rows), len(rows)


def _reference_fraction(directory, key, rows):
    from app.modules.card_game.rl.cross_runtime import model
    from app.modules.card_game.rl.league_schema import ACT_PLAY
    if not rows:
        return 0
    policy = model(directory, key)
    played = 0
    for row in rows:
        scores = policy.scores(row['x'], row['c'])
        played += int(int(row['c'][int(scores.argmax())][0]) == ACT_PLAY)
    return played / len(rows)


def _legal_rows(pools, key):
    from app.modules.card_game.rl.league_schema import ACT_PLAY
    rows = []
    for episode in pools[key].episodes:
        for row in episode['rows']:
            if 'c' in row and any(int(candidate[0]) == ACT_PLAY for candidate in row['c']):
                rows.append(row)
    return rows[:64]


def _pure_report(nets, pools, initial, last_policy, device):
    import math
    report = {}
    for key in ROSTER:
        rows = _legal_rows(pools, key)
        top, count = _play_fraction(nets[key], rows, device)
        updates = last_policy.get(key) or []
        finite = bool(updates) and all(math.isfinite(item['policy_loss']) and math.isfinite(item['value_loss']) for item in updates)
        coverage = pools[key].coverage_report()
        prefixes = ('cand_net', 'score', 'state_net')
        policy_gradient = all(any(
            item.get('policy_samples', 0) > 0
            and math.isfinite((item.get('policy_gradient_norms') or {}).get(prefix, 0))
            and (item.get('policy_gradient_norms') or {}).get(prefix, 0) > 0
            for item in updates) for prefix in prefixes)
        report[key] = dict(finite=finite,
                           policy_gradient=policy_gradient,
                           play_legal=count, play_top1=top,
                           reference_play_top1=_reference_fraction(initial, key, rows),
                           first_use_coverage=coverage['first_use_coverage'],
                           unique_covered=coverage['unique_covered'],
                           fresh_policy_rows=coverage['fresh_policy_rows'],
                           first_use=coverage['first_use'],
                           repeated_use=coverage['repeated_use'],
                           diagnostic_replay_rows=True,
                           independent_selection=False)
    return report


def _search_tally(directory, foe, seed, decks, config):
    wins = 0
    complete = True
    for first in ('a', 'b'):
        job = make_job(dict(left='zhenhong', right=foe, first=first), seed=seed, models=directory,
                       decks=decks, config=config, deadline=config['train_until'] - 30,
                       identity=f'gate-{seed}-{first}', training=False)
        job['search_sides'] = ['a', 'b']
        result = play_job(job)
        complete = complete and bool(result.get('complete'))
        wins += int(result.get('winner') == 'a')
    return dict(complete=complete, wins=wins)


def _run_gate(name, out, config, cycle, nets, pools, initial, best_dir, prior_no_gain, last_policy, decks):
    metrics = _pure_report(nets, pools, initial, last_policy, config['device'])
    for key, item in metrics.items():
        if item['fresh_policy_rows'] and item['unique_covered'] < item['fresh_policy_rows']:
            item['coverage_gap'] = True
    extra = [dict(key=key, reason='no_coverage') for key, item in metrics.items() if item.get('coverage_gap')]
    pure = first_gate(metrics)
    reasons = list(pure['reasons'])
    for row in extra:
        if row not in reasons:
            reasons.append(row)
    if extra or pure['decision'] == 'p0':
        pure = dict(decision='p0', reasons=reasons)
    else:
        pure = dict(decision=pure['decision'], reasons=reasons)
    if name == 'first' or pure['decision'] == 'p0':
        return dict(gate=name, decision=pure['decision'], reasons=pure['reasons'],
                    diagnostic_replay_rows=True, independent_selection=False)
    foe = config['screen_foes'][0]
    seed = config['seeds']['selection'] + cycle * 1000
    def tally(directory, probe):
        screen = _search_tally(directory, foe, probe, decks, config)
        return screen
    candidate = tally(out / 'generations' / f'cycle-{cycle:04}', seed)
    retained = tally(best_dir, seed)
    confirmed = tally(out / 'generations' / f'cycle-{cycle:04}', seed + 50000)
    retained_confirm = tally(best_dir, seed + 50000)
    screen = dict(complete=candidate['complete'] and retained['complete'], candidate=candidate['wins'], best=retained['wins'])
    confirm = dict(complete=confirmed['complete'] and retained_confirm['complete'], candidate=confirmed['wins'], best=retained_confirm['wins'])
    if name == 'mid':
        return dict(gate='mid', decision=mid_gate(pure['decision'], screen, confirm), reasons=pure['reasons'], diagnostic_replay_rows=True, independent_selection=False)
    improved = pure['decision'] == 'pass' and screen['candidate'] > screen['best']
    search_confirms = confirm['complete'] and confirm['candidate'] > confirm['best']
    rollback = pure['decision'] == 'pass' and screen['candidate'] + 2 <= screen['best'] and confirm['candidate'] <= confirm['best']
    decision = late_gate('improved' if improved else pure['decision'], search_confirms, prior_no_gain, rollback)
    return dict(gate='late', decision=decision, reasons=pure['reasons'], diagnostic_replay_rows=True, independent_selection=False)


def _rollback(out, config, nets, best_state, best_dir, decks, origins, learning_rate, cycle):
    import torch
    from app.modules.card_game.rl import cross_runtime as rt
    learning_rate = max(float(config['min_learning_rate']), learning_rate * 0.1)
    for key, net in nets.items():
        net.load_state_dict(best_state[key])
    folder = out / 'generations' / f'cycle-{cycle:04}-rollback'
    for key in ROSTER:
        rt.export(nets[key], folder, key, decks[key], dict(origins[key], rollback_of=str(best_dir), learning_rate=learning_rate))
        import numpy as np
        exported = rt.model(folder, key)
        retained = rt.model(best_dir, key)
        for name, value in retained.weights.items():
            if not np.array_equal(value, exported.weights[name]):
                raise ValueError('Rollback export did not preserve the best weights')
    return learning_rate, folder


def learn(out, config, *, resume=False):
    validate_round_launch(config,resume=resume)
    if config.get('runtime')=='recovery':
        from app.modules.card_game.rl.recovery_round import run
        return run(out,config,resume=resume)
    import numpy as np
    import torch
    from app.modules.card_game.rl import cross_runtime as rt
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.rl.episode_replay import EpisodePool, dump_episode
    if config['rule_hash'] != identity():
        raise ValueError('Frozen identity changed')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(config['seeds']['foundation'])
    rng = np.random.default_rng(config['seeds']['foundation'])
    out = Path(out)
    decks = preset_decks()
    nets, origins = {}, {}
    current = out / 'initial'
    for key in ROSTER:
        source = config['murk_source'] if key == 'murk' else config['roster_source']
        if source:
            nets[key], decks[key], origins[key] = rt.migrate(source, key, config['device'],
                source_rule_hash=config.get('source_rule_hashes', {}).get(key))
        else:
            nets[key] = rt.create_network(config['hidden'], config['device'])
            origins[key] = dict(initialization='fresh_grounded_wdl', optimizer_reset=True)
        rt.export(nets[key], current, key, decks[key], origins[key])
    write(out / 'builds.json', decks)
    initial = current
    counts = {key: dict(games=0, updates=0, samples=0) for key in ROSTER}
    learning_rate = float(config.get('learning_rate', 3e-5))
    opts = {k: torch.optim.Adam([p for p in n.parameters() if p.requires_grad], lr=learning_rate)
            for k, n in nets.items()}
    max_row_reuse = int(config.get('max_row_reuse', 8))
    policy_passes = rt.validate_policy_passes(config.get('policy_passes', 4))
    pools = {k: EpisodePool(limit=64, max_row_reuse=max_row_reuse) for k in ROSTER}
    best_dir = initial
    best_state = {k: {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()}
                  for k, net in nets.items()}
    prior_no_gain = False
    stop_learning = None
    cycle = 0
    previous = initial
    training_games = []
    plan = learning_schedule()
    write(out / 'training-schedule.json', plan)

    def status(phase, **extra):
        write(out / 'status.json', dict(phase=phase, roster=list(ROSTER), counts=counts,
            cycle=cycle, latest=str(current), initial=str(initial), hard_deadline=config['hard_deadline'],
            full_matrix=False, website_approval=False, **extra))

    status('preflight')
    # Every model: Torch/NumPy scores and value agree on a cross-contract root.
    from app.modules.card_game.engine.duel_v2 import new_game
    for key in ROSTER:
        foe = 'starter' if key == 'murk' else 'murk'
        state = new_game(seed=config['seeds']['probes'], decks={'a': decks[key], 'b': decks[foe]})
        actions, (x, c) = rt.decision(state, 'a')
        numeric = rt.model(initial, key)
        with torch.no_grad():
            xx = torch.as_tensor(x[None], device=config['device'])
            cc = torch.as_tensor(c[None], device=config['device'])
            logits, _ = nets[key](xx, cc, torch.ones((1, len(actions)), dtype=torch.bool, device=config['device']))
            hidden = nets[key].state_net(xx)
            wdl = nets[key].wdl(torch.cat((hidden, hidden.new_ones((1, 1))), -1)).softmax(-1)
        np.testing.assert_allclose(logits.cpu().numpy()[0], numeric.scores(x, c), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(wdl.cpu().numpy()[0], numeric.wdl(x, True), atol=2e-5, rtol=2e-5)
    write(out / 'numeric-preflight.json', dict(passed=True, keys=list(ROSTER)))

    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool:
        while time.time() < config['train_until'] - 90:
            status('searching')
            # Both initiatives of each pair use the same frozen snapshots and seed.
            historical = cycle % 4 == 3
            jobs = []
            for index, item in enumerate(plan):
                job = make_job(item, seed=config['seeds']['foundation'] + cycle * 1000 + index // 2,
                    models=current, decks=decks, config=config, deadline=config['train_until'] - 60,
                    identity=f'train-{cycle:04}-{index:02}', training=True)
                job['failure_dir'] = str(out / 'failure-roots')
                if historical:
                    # Rotate current learner seat; both seats get all opponents.
                    for learner in ('a', 'b'):
                        j = dict(job, policies=dict(job['policies']), train_sides=[learner], id=job['id'] + learner)
                        enemy = 'b' if learner == 'a' else 'a'
                        j['policies'][enemy] = (str(previous), j['policies'][enemy][1])
                        jobs.append(j)
                else:
                    jobs.append(job)
            if not config['smoke']:
                for index, key in enumerate(ROSTER):
                    for first in ('a', 'b'):
                        jobs.append(make_job(dict(left=key, right=key, first=first),
                            seed=config['seeds']['foundation'] + cycle * 1000 + 100 + index,
                            models=current, decks=decks, config=config, deadline=config['train_until'] - 60,
                            identity=f'mirror-{cycle:04}-{key}-{first}', training=True))
            started = time.time()
            results = list(pool.map(play_job, jobs))
            fresh = {k: 0 for k in ROSTER}
            fresh_value_rows = {k: [] for k in ROSTER}
            for key in ROSTER:
                pools[key].begin_fresh_batch()
            for job, result in zip(jobs, results):
                row = {k: job[k] for k in ('id', 'seed', 'first', 'left', 'right', 'policies', 'train_sides')}
                row.update({k: result.get(k) for k in ('complete', 'winner', 'reason', 'decisions', 'searched', 'turn', 'passive_triggers', 'setup_achieved')})
                append(out / 'training-games.jsonl', row)
                training_games.append(row)
                if not result['complete']:
                    continue
                if result['decisions'] != result['searched'] or result['searched'] != len(result['rows']):
                    raise ValueError('Incomplete search labels')
                data = dump_episode(out / 'episodes' / (job['id'] + '.json.gz'), job, result)
                for key in set(job['policies'][s][1] for s in job['train_sides']):
                    rows = [r for r in data['rows'] if r['key'] == key]
                    pools[key].add(job['id'], dict(rows=rows))
                    fresh_value_rows[key].extend(r for r in rows if 'pi' not in r)
                    counts[key]['games'] += 1
                    counts[key]['samples'] += len(rows)
                    fresh[key] += len(rows)
            write(out / 'last-cycle-capacity.json', dict(seconds=time.time()-started, games=len(jobs),
                completed=sum(r['complete'] for r in results), simulations=config['simulations'],
                gumbel_candidates=config['gumbel_candidates'], terminal_horizon=config['terminal_horizon'],
                workers=config['workers'], rule_hash=identity()))
            status('updating')
            last_policy = {key: [] for key in ROSTER}
            for key in ROSTER:
                if not fresh[key]:
                    continue
                for rows, refs, sampling in pools[key].policy_pass_batches(
                        rng, passes=policy_passes, batch_size=128, deadline=config['train_until']):
                    metrics = rt.update(nets[key], opts[key], rows)
                    counts[key]['updates'] += 1
                    counts[key]['policy_first_use'] = pools[key].batch_first_use
                    counts[key]['policy_repeated_use'] = pools[key].batch_repeated_use
                    last_policy[key].append(dict(metrics, sampling=sampling))
                    append(out / 'updates.jsonl', policy_update_record(key, cycle, refs, sampling, metrics))
                # Preserve the terminal-WDL observations that have no policy
                # target. They update the value trunk and WDL head only.
                value_rows = fresh_value_rows[key]
                for start in range(0, len(value_rows), 128):
                    if time.time() >= config['train_until']:
                        break
                    rows = value_rows[start:start + 128]
                    metrics = rt.update(nets[key], opts[key], rows, value_only=True)
                    counts[key]['updates'] += 1
                    counts[key]['value_updates'] = counts[key].get('value_updates', 0) + 1
                    append(out / 'updates.jsonl', dict(key=key, cycle=cycle,
                        sampling=dict(source='fresh_value_only', rows=len(rows)), **metrics))
                policy_coverage = pools[key].coverage_report()
                counts[key].update(fresh_policy_rows=policy_coverage['fresh_policy_rows'],
                                   unique_covered=policy_coverage['unique_covered'],
                                   first_use_coverage=policy_coverage['first_use_coverage'])
                write(out / 'policy-coverage' / f'{key}-cycle-{cycle:04}.json', dict(key=key, cycle=cycle, **policy_coverage))
            previous = current
            cycle += 1
            current = out / 'generations' / f'cycle-{cycle:04}'
            for key in ROSTER:
                rt.export(nets[key], current, key, decks[key], dict(origins[key], **counts[key]))
            checkpoint = out / 'recovery' / f'cycle-{cycle % 2}.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            tmp = checkpoint.with_suffix('.tmp')
            torch.save(dict(cycle=cycle, networks={k:n.state_dict() for k,n in nets.items()},
                optimizers={k:o.state_dict() for k,o in opts.items()},
                pools={k:p.export_state() for k,p in pools.items()}, counts=counts,
                policy_passes=policy_passes, max_row_reuse=max_row_reuse,
                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), config=config), tmp)
            tmp.replace(checkpoint)
            write(out / 'recovery' / 'pointer.json', dict(path=str(checkpoint), cycle=cycle,
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            write_training_coverage(out, training_games)
            gate_name = {config['gates'][name]: name for name in ('first', 'mid', 'late') if config['gates'].get(name) == cycle}
            if gate_name:
                decision = _run_gate(gate_name[cycle], out, config, cycle, nets, pools, initial, best_dir,
                                      prior_no_gain, last_policy, decks)
                append(out / 'gates.jsonl', dict(cycle=cycle, **decision))
                if decision['decision'] == 'promote':
                    best_dir = current
                    best_state = {k: {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()}
                                  for k, net in nets.items()}
                    prior_no_gain = False
                elif decision['decision'] == 'rollback':
                    learning_rate, current = _rollback(out, config, nets, best_state, best_dir, decks, origins, learning_rate, cycle)
                    opts = {k: torch.optim.Adam([p for p in n.parameters() if p.requires_grad], lr=learning_rate)
                            for k, n in nets.items()}
                    pools = {k: EpisodePool(limit=64, max_row_reuse=max_row_reuse) for k in ROSTER}
                    prior_no_gain = False
                elif decision['decision'] == 'plateau':
                    stop_learning = 'plateau'
                    current = best_dir
                elif decision['decision'] == 'p0' or (decision['decision'] == 'insufficient' and not config['smoke']):
                    stop_learning = decision['decision']
                    current = best_dir
                else:
                    prior_no_gain = decision['decision'] == 'keep' and gate_name[cycle] != 'first'
            status('searching', best=str(best_dir), learning_rate=learning_rate, stop_learning=stop_learning)
            if config['smoke'] or stop_learning:
                break
        if not all(c['updates'] > 0 for c in counts.values()):
            raise RuntimeError('Not all five policies updated')
        status('evaluating')
        # Freeze exact arrays. latest is reported as latest, never claimed best.
        signatures = {k: rt.model(current, k).version for k in ROSTER}
        changed = {}
        for key in ROSTER:
            before, after = rt.model(initial, key), rt.model(current, key)
            changed[key] = {n: not np.array_equal(v, after.weights[n]) for n,v in before.weights.items()}
        write(out / 'model-changes.json', changed)
        results_by_mode = {}
        matrix = matrix_schedule(config['matrix_games'])
        for mode in ('argmax', 'search'):
            folder = out / mode
            for name in ('evaluation/raw', 'evaluation/replays'):
                (folder / name).mkdir(parents=True, exist_ok=True)
            jobs = []
            for i, item in enumerate(matrix):
                job = make_job(item, seed=config['seeds']['confirmation'] + i,
                    models=current, decks=decks, config=config, deadline=config['evaluate_until'],
                    identity=f'{mode}-{i:05}', training=False)
                job.update(cell=('mirror/' if item['left'] == item['right'] else 'cross/') + item['row'] + '/' + item['column'],
                    output=str(folder / 'evaluation'), replay=item['replica'] == 0,
                    search_sides=['a', 'b'] if mode == 'search' else [])
                jobs.append(job)
            write(folder / 'planned-jobs.json', jobs)
            results = []
            for offset in range(0, len(jobs), config['workers']):
                if time.time() >= config['evaluate_until']:
                    break
                batch = jobs[offset:offset+config['workers']]
                for result in pool.map(evaluate_job, batch):
                    results.append(result)
                    append(folder / 'evaluation/index.jsonl', result)
                status('evaluating', inference=mode, evaluated=len(results), planned=len(jobs))
            results_by_mode[mode] = results
            from app.modules.card_game.rl.new_model_report import write_matrix
            write_matrix(folder, results, ROSTER, current, dict(config, initial_models=str(initial), inference=mode, smoke=False))
            cells = coverage(results, matrix)
            write(folder / 'coverage.json', cells)
        if signatures != {k: rt.model(current,k).version for k in ROSTER}:
            raise ValueError('Weights changed during evaluation')
    complete = all(all(c['complete'] for c in coverage(rs, matrix)) for rs in results_by_mode.values())
    write(out / 'status.json', dict(phase='complete' if complete else 'evaluation_incomplete',
        counts=counts, latest=str(current), initial=str(initial), full_matrix=complete,
        complete_report=False, website_approval=False, hard_deadline=config['hard_deadline']))
    write(out / 'delivery-manifest.json', {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob('*') if p.is_file() and p.name != 'delivery-manifest.json'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--roster-source', type=Path)
    p.add_argument('--murk-source', type=Path)
    p.add_argument('--seed-reservation', type=Path)
    p.add_argument('--source-rule-hashes', type=Path, help='Reviewed historical source rule identities by preset')
    p.add_argument('--hard-deadline', help='ISO time with explicit timezone; default next Shanghai 09:00')
    p.add_argument('--seconds', type=int, help='Optional shorter total window')
    p.add_argument('--matrix-games', type=int, default=16)
    p.add_argument('--device',choices=('cpu','cuda'), default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--policy-passes', type=int, choices=(1, 4, 8), default=4,
                   help='Fresh policy-row passes after each frozen batch; 4 is the bounded research default')
    p.add_argument('--max-row-reuse', type=int, default=8)
    p.add_argument('--run', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume existing recovery config within its original deadline')
    p.add_argument('--smoke-batches',type=int,choices=(1,2,3),default=1)
    p.add_argument('--runtime',choices=('recovery','cross'),default='recovery')
    p.add_argument('--teacher-mode',choices=('network','covered_terminal','covered_expert','covered_value'),default='network',
                   help='Explicit recovery teacher; covered modes use visible rules and real-terminal leaves')
    p.add_argument('--recovery-source',type=Path)
    p.add_argument('--covered-value-source',type=Path)
    p.add_argument('--readiness-report',type=Path)
    p.add_argument('--preflight',action='store_true',help='One all-five Gumbel32 batch, no strength approval')
    p.add_argument('--preflight-replicas',type=int,choices=(1,2,3),default=1,
                   help='New paired training and heldout games per matchup; never reuse old-rule data')
    p.add_argument('--planned-batches',type=int,choices=(4,5,6),default=6)
    p.add_argument('--gate-roots',type=int,default=32)
    p.add_argument('--search-gate-roots',type=int,default=2)
    p.add_argument('--train-fraction',type=float,default=.55)
    p.add_argument('--value-full-tower',action='store_true',
                   help='Fit the independent value tower; choose epoch and temperature only on selection games')
    p.add_argument('--value-replicas',type=int,choices=(2,6,12),default=2)
    p.add_argument('--value-check-replicas',type=int,choices=(1,2,3),default=1)
    a = p.parse_args()
    out = a.output.resolve()
    if a.worker:
        config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        validate_round_launch(config,resume=a.resume)
        try:
            learn(out, config,resume=a.resume)
        except Exception as exc:
            write(out / 'failure.json', dict(error=repr(exc), time=time.time()))
            raise
        return
    if a.resume:
        if not a.run: p.error('Resume requires --run and an existing recovery output')
        config=json.loads((out/'config.json').read_text())
        if config.get('runtime')!='recovery': p.error('Only recovery runtime supports this resume')
        guard=json.loads((out/'guard.json').read_text())
        pid=guard.get('worker_pid')
        if type(pid) is int and pid>0:
            try: os.kill(pid,0)
            except ProcessLookupError: pass
            else: p.error('Previous worker is still live; refuse concurrent resume')
        validate_round_launch(config,resume=True)
        if time.time()>=config['train_until']: p.error('Original learning deadline expired')
        supervise(out,config,resume=True)
        return
    if a.workers < 1 or a.workers > 16:
        p.error('workers must be 1..16')
    if not 5<=a.gate_roots<=64 or not 1<=a.search_gate_roots<=8 or not .5<=a.train_fraction<=.8:
        p.error('Bounded formal gate roots and training fraction required')
    matrix_schedule(a.matrix_games)
    now = datetime.now(shanghai_zone())
    hard = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if hard <= now:
        hard += timedelta(days=1)
    if a.hard_deadline:
        hard = datetime.fromisoformat(a.hard_deadline)
        if hard.tzinfo is None:
            p.error('hard deadline needs timezone')
    if a.seconds:
        hard = min(hard, now + timedelta(seconds=a.seconds))
    if a.smoke or a.preflight:
        hard = min(hard, now + timedelta(minutes=120 if a.preflight and a.preflight_replicas>1 else 30))
    seconds = (hard-now).total_seconds()
    if seconds < 300:
        p.error('Insufficient time before hard stop')
    from app.modules.card_game.rl.cross_lineup import identity
    config = dict(kind='five_cross_grounded_v2', runtime='cross', roster=list(ROSTER),
        rule_hash=identity(), hidden=a.hidden, device=a.device, workers=a.workers,
        train_until=now.timestamp()+seconds*a.train_fraction, evaluate_until=hard.timestamp()-60,
        hard_deadline=hard.timestamp(), hard_deadline_iso=hard.isoformat(),
        simulations=2 if a.smoke else 32, gumbel_candidates=4 if a.smoke else 16,
        terminal_horizon=3, rewards=0, deck_search=False, self_imitation=0,
        matrix_games=2 if a.smoke else a.matrix_games, smoke=a.smoke,
        gates=dict(first=1, mid=None, late=None) if a.smoke else schedule_gates(a.planned_batches),
        learning_rate=3e-5, min_learning_rate=1e-5, screen_foes=['starter'],
        planned_batches=1 if a.smoke else a.planned_batches,
        policy_passes=a.policy_passes, max_row_reuse=a.max_row_reuse,
        website_approval=False, roster_source=str(a.roster_source) if a.roster_source else '',
        murk_source=str(a.murk_source) if a.murk_source else '')
    if a.runtime=='recovery':
        if a.roster_source or a.murk_source:
            p.error('Old sources cannot initialize recovery; use explicit recovery-source')
        config.update(started_at=now.timestamp(),kind='five_recovery_v1',runtime='recovery',smoke=a.smoke or a.preflight,
            preflight=a.preflight,teacher_mode=a.teacher_mode,
            preflight_replicas=a.preflight_replicas if a.preflight else 1,
            covered_value_source=str(a.covered_value_source.resolve()) if a.covered_value_source else None,
            terminal_horizon=0 if a.teacher_mode in ('covered_terminal','covered_expert') else 3,
            simulations=32 if a.preflight or not a.smoke else 2,
            learning_rate=.0003,value_replicas=a.value_replicas,value_passes=16,
            value_full_tower=a.value_full_tower,value_check_replicas=a.value_check_replicas,
            gate_roots=5 if a.preflight else 2 if a.smoke else a.gate_roots,
            search_gate_roots=1 if a.smoke or a.preflight else a.search_gate_roots,
            gate_seconds=600 if a.smoke or a.preflight else 1800,
            planned_batches=1 if a.preflight else a.smoke_batches if a.smoke else a.planned_batches,
            readiness_report=str(a.readiness_report.resolve()) if a.readiness_report else None,
            recovery_source=str(a.recovery_source.resolve()) if a.recovery_source else None)
        if a.preflight:
            config['train_until']=hard.timestamp()-60
        elif not a.smoke:
            config['evaluate_until']=hard.timestamp()-900
    config['source_rule_hashes'] = json.loads(a.source_rule_hashes.read_text()) if a.source_rule_hashes else {}
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if not a.run:
        return
    try:
        validate_round_launch(config)
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))
    if a.runtime=='recovery':
        import torch
        if a.device=='cuda' and not torch.cuda.is_available():
            p.error('CUDA unavailable; no fallback, output or seeds created')
    if not (a.smoke or a.preflight) and a.device != 'cuda':
        p.error('Formal learning requires CUDA')
    if out.exists() and any(out.iterdir()):
        p.error('Output must be new and empty')
    out.mkdir(parents=True, exist_ok=True)
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    config['seeds'] = (json.loads(a.seed_reservation.read_text()) if a.seed_reservation else
                       claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(out)))
    write(out / 'config.json', config)
    supervise(out,config)

def supervise(out,config,*,resume=False):
    hard=datetime.fromisoformat(config['hard_deadline_iso'])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    worker = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--output', str(out), '--worker']+(['--resume'] if resume else []),
        cwd=ROOT, env=env, start_new_session=os.name != 'nt')
    write(out / 'guard.json', dict(phase='running', worker_pid=worker.pid, hard_deadline=hard.isoformat()))
    monotonic_end = time.monotonic() + max(0, hard.timestamp()-time.time())
    while worker.poll() is None:
        if time.time() >= hard.timestamp() or time.monotonic() >= monotonic_end:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(worker.pid), '/T', '/F'], check=False)
            else:
                os.killpg(worker.pid, signal.SIGKILL)
            write(out / 'guard.json', dict(phase='hard_stop', worker_pid=worker.pid, hard_deadline=hard.isoformat()))
            return
        time.sleep(2)
    write(out / 'guard.json', dict(phase='exited', exit_code=worker.returncode, hard_deadline=hard.isoformat()))
    if worker.returncode:
        raise SystemExit(worker.returncode)


if __name__ == '__main__':
    main()
