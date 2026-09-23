#!/usr/bin/env python3
"""Five-preset grounded-WDL cross play with explicit paired initiative.

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


def cross_pairs():
    return [(a, b) for a in ROSTER for b in ROSTER if a != b]


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


def learn(out, config):
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
    opts = {k: torch.optim.Adam([p for p in n.parameters() if p.requires_grad], lr=3e-5)
            for k, n in nets.items()}
    pools = {k: EpisodePool(limit=64) for k in ROSTER}
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
                    counts[key]['games'] += 1
                    counts[key]['samples'] += len(rows)
                    fresh[key] += len(rows)
            write(out / 'last-cycle-capacity.json', dict(seconds=time.time()-started, games=len(jobs),
                completed=sum(r['complete'] for r in results), simulations=config['simulations'],
                gumbel_candidates=config['gumbel_candidates'], terminal_horizon=config['terminal_horizon'],
                workers=config['workers'], rule_hash=identity()))
            status('updating')
            for key in ROSTER:
                for step in range(min(8, max(1, fresh[key] // 128)) if fresh[key] else 0):
                    if time.time() >= config['train_until']:
                        break
                    rows, refs, sampling = pools[key].sample(rng, 128, prioritized=False)
                    if not rows:
                        break
                    metrics = rt.update(nets[key], opts[key], rows)
                    counts[key]['updates'] += 1
                    append(out / 'updates.jsonl', dict(key=key, cycle=cycle, refs=refs, sampling=sampling, **metrics))
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
                pools={k:p.episodes for k,p in pools.items()}, counts=counts,
                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), config=config), tmp)
            tmp.replace(checkpoint)
            write(out / 'recovery' / 'pointer.json', dict(path=str(checkpoint), cycle=cycle,
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            write(out / 'training-coverage.json', coverage(training_games,
                [dict(left=r['left'], right=r['right'], first=r['first']) for r in training_games]))
            status('searching')
            if config['smoke']:
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
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if a.worker:
        config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        try:
            learn(out, config)
        except Exception as exc:
            write(out / 'failure.json', dict(error=repr(exc), time=time.time()))
            raise
        return
    if a.workers < 1 or a.workers > 16:
        p.error('workers must be 1..16')
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
    if a.smoke:
        hard = min(hard, now + timedelta(minutes=20))
    seconds = (hard-now).total_seconds()
    if seconds < 300:
        p.error('Insufficient time before hard stop')
    from app.modules.card_game.rl.cross_lineup import identity
    config = dict(kind='five_cross_grounded_v2', runtime='cross', roster=list(ROSTER),
        rule_hash=identity(), hidden=a.hidden, device=a.device, workers=a.workers,
        train_until=now.timestamp()+seconds*.55, evaluate_until=hard.timestamp()-60,
        hard_deadline=hard.timestamp(), hard_deadline_iso=hard.isoformat(),
        simulations=2 if a.smoke else 32, gumbel_candidates=4 if a.smoke else 16,
        terminal_horizon=3, rewards=0, deck_search=False, self_imitation=0,
        matrix_games=2 if a.smoke else a.matrix_games, smoke=a.smoke,
        website_approval=False, roster_source=str(a.roster_source) if a.roster_source else '',
        murk_source=str(a.murk_source) if a.murk_source else '')
    config['source_rule_hashes'] = json.loads(a.source_rule_hashes.read_text()) if a.source_rule_hashes else {}
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if not a.run:
        return
    if not a.smoke and a.device != 'cuda':
        p.error('Formal learning requires CUDA')
    if out.exists() and any(out.iterdir()):
        p.error('Output must be new and empty')
    out.mkdir(parents=True, exist_ok=True)
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    config['seeds'] = (json.loads(a.seed_reservation.read_text()) if a.seed_reservation else
                       claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(out)))
    write(out / 'config.json', config)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    worker = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--output', str(out), '--worker'],
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
