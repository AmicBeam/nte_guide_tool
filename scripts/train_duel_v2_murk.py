#!/usr/bin/env python3
"""Self-play 浊燃预组 with the categorical WDL and Gumbel method.

This contract is separate from the ten-seat weights. It does not approve a
website model and it does not call the retired five-team scalar entry.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEY = 'murk'


def shanghai_zone():
    try:
        return ZoneInfo('Asia/Shanghai')
    except Exception:
        return timezone(timedelta(hours=8))


def shanghai_nine(now=None):
    zone = shanghai_zone()
    current = now or datetime.now(zone)
    if current.tzinfo is None:
        raise ValueError('Deadline clock must carry a timezone')
    current = current.astimezone(zone)
    nine = current.replace(hour=9, minute=0, second=0, microsecond=0)
    if current >= nine:
        nine += timedelta(days=1)
    return nine


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def play_job(job):
    from app.modules.card_game.rl.information_search import search_game as run_game
    return run_game(job)


def learn(out, config):
    import numpy as np
    import torch
    from app.modules.card_game.rl.information_search import search_game
    from app.modules.card_game.rl.murk_lineup import deck as murk_deck, encode, identity
    from app.modules.card_game.rl.murk_runtime import MurkModel, create_network, export, model, update
    from app.modules.card_game.engine.duel_v2 import new_game, observe

    if identity() != config['rule_hash']:
        raise ValueError('Rule identity changed')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(config['seeds']['foundation'])
    build = murk_deck()
    out = Path(out)
    net = create_network(config['hidden'], config['device'])
    origin = dict(initialization='fresh_murk_wdl', auxiliary_rewards=0, optimizer_reset=True,
                  automatic_serving_approval=False)
    initial = out / 'initial'
    export(net, initial, KEY, build, origin)
    numeric = MurkModel(initial, KEY)
    probe = new_game(seed=config['seeds']['foundation'], first_side='a', skip_mulligan=True,
                     decks={'a': build, 'b': build})
    encoded, _ = encode(observe(probe, 'a', include_previews=False), [])
    with torch.no_grad():
        hidden = net.state_net(torch.as_tensor(encoded, device=config['device']))
        prob = net.wdl(torch.cat((hidden, hidden.new_tensor([1.0])))).softmax(-1).cpu().numpy()
    if not np.allclose(prob, numeric.wdl(encoded, True), atol=2e-5):
        raise ValueError('Murk numeric export mismatch')
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-5)
    current = initial
    games = 0
    updates = 0
    wins = 0
    scheduled = 0
    write(out / 'status.json', dict(phase='searching', games=0, updates=0, wins=0))

    def one_job():
        nonlocal scheduled
        mirror = scheduled % 2 == 1
        foe = current if mirror else initial
        job = dict(
            id=f'game-{scheduled}', runtime='murk', seed=config['seeds']['foundation'] + scheduled,
            first=('a', 'b')[scheduled % 2], decks={'a': build, 'b': build},
            policies={'a': (str(current), KEY), 'b': (str(foe), KEY)},
            simulations=config['simulations'], search_algorithm='gumbel',
            gumbel_candidates=config['gumbel_candidates'], terminal_horizon=config['terminal_horizon'],
            fast_simulation=True, training=True, collect=True, record_episode=True,
            train_sides=['a', 'b'] if mirror else ['a'],
            deadline=min(config['train_until'] - 30, time.time() + config['game_seconds']),
            max_actions=config['max_actions'], failure_dir=str(out / 'failure-roots'), mirror=mirror)
        scheduled += 1
        return job

    def absorb(job, result):
        nonlocal games, updates, wins, current
        record = dict(id=job['id'], complete=result['complete'], winner=result.get('winner'),
                      decisions=result['decisions'], searched=result['searched'], mirror=job['mirror'])
        with (out / 'training-games.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if not result['complete']:
            if config['smoke']:
                raise RuntimeError('Smoke game did not finish: ' + str(result.get('reason')))
            write(out / 'status.json', dict(phase='searching', games=games, updates=updates, wins=wins,
                                            last_incomplete=result.get('reason')))
            return
        rows = [row for row in result['rows'] + result['value_rows'] if row.get('z') in (-1, 0, 1)]
        if not rows:
            raise ValueError('Completed game produced no labels')
        stats = update(net, opt, rows)
        for value in stats.values():
            if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
                raise ValueError('Non-finite murk update')
        updates += 1
        games += 1
        wins += int(result.get('winner') == 'a')
        current = out / 'generations' / f'update-{updates:04d}'
        export(net, current, KEY, build, {**origin, 'updates': updates, 'games': games})
        write(out / 'status.json', dict(phase='searching', games=games, updates=updates, wins=wins,
                                        latest=str(current)))
        write(out / 'launch.json', dict(started=True, games=games, updates=updates, latest=str(current)))

    if config['workers'] == 1:
        while time.time() < config['train_until'] and not (config['smoke'] and games):
            job = one_job()
            absorb(job, search_game(job))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool:
            while time.time() < config['train_until']:
                jobs = [one_job() for _ in range(config['workers'])]
                for job, result in zip(jobs, pool.map(play_job, jobs)):
                    absorb(job, result)
                    if time.time() >= config['train_until']:
                        break
    if games == 0:
        raise RuntimeError('No completed training game')
    frozen = model(current, KEY)
    if not np.array_equal(frozen.weights['state_net.0.weight'], net.state_dict()['state_net.0.weight'].detach().cpu().numpy()):
        raise ValueError('Exported weights drifted from the optimizer')
    eval_games = []
    played = 0
    while time.time() < config['evaluate_until'] and played < config['eval_games']:
        job = dict(
            id=f'eval-{played}', runtime='murk', seed=config['seeds']['confirmation'] + played,
            first=('a', 'b')[played % 2], decks={'a': build, 'b': build},
            policies={'a': (str(current), KEY), 'b': (str(initial), KEY)},
            simulations=config['simulations'], search_algorithm='gumbel',
            gumbel_candidates=config['gumbel_candidates'], terminal_horizon=config['terminal_horizon'],
            fast_simulation=True, training=False, collect=False, record_episode=False,
            deadline=min(config['evaluate_until'] - 10, time.time() + config['game_seconds']),
            max_actions=config['max_actions'])
        result = search_game(job)
        eval_games.append(dict(complete=result['complete'], winner=result.get('winner'), first=job['first'],
                               decisions=result['decisions']))
        played += 1
        if config['smoke']:
            break
    write(out / 'eval.json', dict(games=eval_games, trained=str(current), initial=str(initial),
                                  approval=False, matrix='murk-self-play-only'))
    write(out / 'status.json', dict(phase='complete', games=games, updates=updates, wins=wins,
                                    evaluated=len(eval_games)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--hard-deadline', default='')
    parser.add_argument('--seed-reservation', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = '1'
        try:
            learn(out, config)
        except Exception as exc:
            write(out / 'status.json', dict(phase='failed', error=repr(exc)))
            raise
        return
    from app.modules.card_game.rl.murk_lineup import identity
    zone = shanghai_zone()
    if args.hard_deadline:
        hard = datetime.fromisoformat(args.hard_deadline)
        if hard.tzinfo is None:
            raise SystemExit('hard deadline needs a timezone offset')
        hard = hard.astimezone(zone)
    else:
        hard = shanghai_nine()
    now = datetime.now(zone)
    if args.smoke:
        train_until = now + timedelta(seconds=240)
        evaluate_until = now + timedelta(seconds=300)
        hard = now + timedelta(seconds=360)
        simulations = 2
        candidates = 4
        horizon = 1
        game_seconds = 180
        max_actions = 400
        eval_games = 1
        device = args.device
    else:
        if args.device != 'cuda':
            raise SystemExit('Overnight murk training requires CUDA. Smoke may use CPU.')
        train_until = hard - timedelta(minutes=70)
        evaluate_until = hard - timedelta(minutes=20)
        if not (now < train_until < evaluate_until < hard):
            raise SystemExit('Deadlines must satisfy now < stop < evaluation < hard stop')
        simulations = 32
        candidates = 16
        horizon = 3
        game_seconds = 900
        max_actions = 400
        eval_games = 32
        device = 'cuda'
    plan = dict(
        kind='murk_grounded_wdl_v1', roster=['anhunqu', 'canhong', 'zaowu', 'adler'],
        deck='murk', opponents='mirror-and-frozen-initial', rewards=0, search='gumbel',
        simulations=simulations, gumbel_candidates=candidates, terminal_horizon=horizon,
        hidden=args.hidden, device=device, workers=1 if args.smoke else args.workers,
        now=now.isoformat(), train_until=train_until.isoformat(),
        evaluate_until=evaluate_until.isoformat(), hard_deadline=hard.isoformat(),
        rule_hash=identity(), website_approval=False, smoke=args.smoke)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.run:
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory must be a new empty directory')
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        seeds = dict(foundation=910_000_001, confirmation=910_100_001)
    elif args.seed_reservation:
        seeds = json.loads(args.seed_reservation.read_text(encoding='utf-8'))
    else:
        from app.modules.card_game.rl.build_acceptance import claim_seeds
        seeds = claim_seeds(ROOT / 'artifacts' / 'rl-seed-ledger.json', str(out))
    config = dict(plan)
    config.update(seeds=seeds, game_seconds=game_seconds, max_actions=max_actions, eval_games=eval_games,
                  train_until=train_until.timestamp(), evaluate_until=evaluate_until.timestamp(),
                  hard_deadline=hard.timestamp())
    write(out / 'config.json', config)
    write(out / 'guard.json', dict(phase='launching', hard_deadline=hard.isoformat()))
    if args.smoke:
        learn(out, config)
        return
    command = [sys.executable, '-X', 'utf8', __file__, '--output', str(out), '--worker']
    worker = subprocess.Popen(command, cwd=ROOT, start_new_session=os.name != 'nt')
    write(out / 'guard.json', dict(phase='running', worker_pid=worker.pid, hard_deadline=hard.isoformat()))
    try:
        while worker.poll() is None:
            if time.time() >= hard.timestamp():
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(worker.pid), '/T', '/F'], check=False)
                else:
                    os.killpg(worker.pid, signal.SIGKILL)
                write(out / 'status.json', dict(phase='hard_stop'))
                break
            time.sleep(5)
    except KeyboardInterrupt:
        if os.name != 'nt':
            os.killpg(worker.pid, signal.SIGTERM)
        raise


if __name__ == '__main__':
    main()
