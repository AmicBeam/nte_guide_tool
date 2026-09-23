#!/usr/bin/env python3
"""One-hour trial of the current method for five presets.

创生、覆纹、快攻 and 真红 share the ten-seat contract and play each other.
浊燃 is the fifth preset and learns in the same hour, but its games stay in
the murk contract. This is not a full N×N delivery, does not search decks,
and does not approve a website model.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROSTER = ('starter', 'weave-rush', 'quick-rush', 'zhenhong')


def shanghai_zone():
    try:
        return ZoneInfo('Asia/Shanghai')
    except Exception:
        return timezone(timedelta(hours=8))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def cross_pairs():
    return [(left, right) for left in ROSTER for right in ROSTER if left != right]


def learning_schedule():
    """Equal appearances: each of the five presets is in six games per cycle."""
    pairs = cross_pairs()
    plan = []
    for index in range(6):
        plan.append(pairs[index])
        plan.append(pairs[index + 6])
        plan.append(('murk', 'murk'))
    return plan


def preset_decks():
    from copy import deepcopy
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
    found = {item['id']: item for item in STARTER_DECKS}
    return {key: validate_deck(deepcopy(found[key])) for key in ROSTER}


def play_job(job):
    from app.modules.card_game.rl.information_search import search_game
    return search_game(job)


def fresh_net(hidden, device):
    import torch
    from app.modules.card_game.rl.grounded_candidates import network
    net = network(hidden)
    with torch.no_grad():
        net.wdl.weight.zero_()
        net.wdl.bias.zero_()
        net.cand_net[2].weight.mul_(0.01)
        net.cand_net[2].bias.mul_(0.01)
        net.score.weight.zero_()
        net.score.bias.zero_()
    for parameter in net.value.parameters():
        parameter.requires_grad_(False)
    return net.to(device)


def load_roster(directory, device):
    import torch
    from app.modules.card_game.rl.outcome_runtime import model
    nets, decks = {}, {}
    for key in ROSTER:
        policy = model(directory, key)
        net = fresh_net(policy.hidden, device)
        state = net.state_dict()
        for name, value in policy.weights.items():
            state[name] = torch.from_numpy(value.copy())
        net.load_state_dict(state)
        nets[key] = net
        decks[key] = policy.serving_deck
    return nets, decks


def load_murk(source, device):
    import torch
    from app.modules.card_game.rl.murk_runtime import create_network, model
    policy = model(source, 'murk')
    net = create_network(policy.hidden, device)
    state = net.state_dict()
    for name, value in policy.weights.items():
        state[name] = torch.from_numpy(value.copy())
    net.load_state_dict(state)
    return net, policy.serving_deck


def learn(out, config):
    import numpy as np
    import torch
    from app.modules.card_game.engine.duel_v2 import new_game, observe
    from app.modules.card_game.rl.fixed_lineup import encode, identity
    from app.modules.card_game.rl.murk_lineup import deck as murk_deck
    from app.modules.card_game.rl.murk_runtime import export as export_murk
    from app.modules.card_game.rl.murk_runtime import update as update_murk
    from app.modules.card_game.rl.outcome_runtime import export, model, update

    if identity() != config['rule_hash']:
        raise ValueError('Rule identity changed')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(config['seeds']['foundation'])
    out = Path(out)
    origin = dict(initialization='fresh_grounded_wdl', auxiliary_rewards=0, optimizer_reset=True,
                  automatic_serving_approval=False, deck_search=False)
    if config.get('roster_source'):
        nets, decks = load_roster(config['roster_source'], config['device'])
        origin['initialization'] = 'continued_grounded_wdl'
    else:
        decks = preset_decks()
        nets = {key: fresh_net(config['hidden'], config['device']) for key in ROSTER}
    current = out / 'initial'
    for key in ROSTER:
        export(nets[key], current, key, decks[key], origin)
    numeric = model(current, 'starter')
    probe = new_game(seed=config['seeds']['foundation'], first_side='a', skip_mulligan=True,
                     decks={'a': decks['starter'], 'b': decks['weave-rush']})
    encoded, _ = encode(observe(probe, 'a', include_previews=False), [])
    with torch.no_grad():
        hidden = nets['starter'].state_net(torch.as_tensor(encoded, device=config['device']))
        prob = nets['starter'].wdl(torch.cat((hidden, hidden.new_tensor([1.0])))).softmax(-1).cpu().numpy()
    if not np.allclose(prob, numeric.wdl(encoded, True), atol=2e-5):
        raise ValueError('Grounded numeric export mismatch')
    opts = {key: torch.optim.Adam([p for p in nets[key].parameters() if p.requires_grad], lr=3e-5)
            for key in ROSTER}
    if config.get('murk_source'):
        murk_net, murk_build = load_murk(config['murk_source'], config['device'])
        frozen_murk = config['murk_source']
        murk_origin = 'continued_murk_wdl'
    else:
        from app.modules.card_game.rl.murk_runtime import create_network as fresh_murk
        murk_net = fresh_murk(config['hidden'], config['device'])
        murk_build = murk_deck()
        frozen_murk = out / 'murk-initial'
        export_murk(murk_net, frozen_murk, 'murk', murk_build,
                    dict(initialization='fresh_murk_wdl', automatic_serving_approval=False))
        murk_origin = 'fresh_murk_wdl'
    current_murk = frozen_murk
    murk_opt = torch.optim.Adam([p for p in murk_net.parameters() if p.requires_grad], lr=3e-5)
    counts = {key: dict(games=0, updates=0, wins=0) for key in (*ROSTER, 'murk')}
    scheduled = 0
    plan = learning_schedule()
    write(out / 'status.json', dict(phase='searching', roster=[*ROSTER, 'murk'], counts=counts, scheduled=0))

    def one_job(kind, left, right, seed, mirror=False):
        folder = str(current)
        return dict(
            id=f'{kind}-{seed}', runtime='outcome', seed=seed, first=('a', 'b')[seed % 2],
            decks={'a': decks[left], 'b': decks[right]},
            policies={'a': (folder, left), 'b': (folder, right)},
            simulations=config['simulations'], search_algorithm='gumbel',
            gumbel_candidates=config['gumbel_candidates'], fast_simulation=True,
            training=kind == 'train', collect=kind == 'train', record_episode=False,
            train_sides=['a', 'b'], deadline=min(config['hard_deadline'] - 20, time.time() + config['game_seconds']),
            max_actions=config['max_actions'], failure_dir=str(out / 'failure-roots'),
            mirror=mirror, left=left, right=right)

    def absorb_train(job, result):
        nonlocal current
        record = dict(id=job['id'], complete=result['complete'], winner=result.get('winner'),
                      left=job['left'], right=job['right'], decisions=result['decisions'],
                      reason=result.get('reason'))
        with (out / 'training-games.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if not result['complete']:
            if config['smoke']:
                raise RuntimeError('Smoke game did not finish: ' + str(result.get('reason')))
            return False
        grouped = {}
        for row in result['rows'] + result['value_rows']:
            if row.get('z') in (-1, 0, 1):
                grouped.setdefault(row['key'], []).append(row)
        if not grouped:
            raise ValueError('Completed game produced no labels')
        for key, rows in grouped.items():
            stats = update(nets[key], opts[key], rows)
            for value in stats.values():
                if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
                    raise ValueError('Non-finite update')
            counts[key]['updates'] += 1
            counts[key]['games'] += 1
            counts[key]['wins'] += int(result.get('winner') == ('a' if job['left'] == key else 'b'))
        step = sum(item['updates'] for item in counts.values())
        current = out / 'generations' / f'update-{step:04d}'
        for key in ROSTER:
            export(nets[key], current, key, decks[key], {**origin, 'updates': counts[key]['updates']})
        write(out / 'status.json', dict(phase='searching', counts=counts, latest=str(current), scheduled=scheduled))
        return True

    def run_batch(jobs):
        if config['workers'] == 1:
            return [(job, play_job(job)) for job in jobs]
        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool:
            return list(zip(jobs, pool.map(play_job, jobs)))

    def murk_job(seed):
        mirror = seed % 2 == 1
        foe = current_murk if mirror else frozen_murk
        return dict(
            id=f'train-{seed}', runtime='murk', seed=seed, first=('a', 'b')[seed % 2],
            decks={'a': murk_build, 'b': murk_build},
            policies={'a': (str(current_murk), 'murk'), 'b': (str(foe), 'murk')},
            simulations=config['simulations'], search_algorithm='gumbel',
            gumbel_candidates=config['gumbel_candidates'], fast_simulation=True,
            training=True, collect=True, record_episode=False,
            train_sides=['a', 'b'] if mirror else ['a'],
            deadline=min(config['hard_deadline'] - 20, time.time() + config['game_seconds']),
            max_actions=config['max_actions'], failure_dir=str(out / 'failure-roots'),
            left='murk', right='murk')

    def absorb_murk(job, result):
        nonlocal current_murk
        record = dict(id=job['id'], complete=result['complete'], winner=result.get('winner'),
                      left='murk', right='murk', decisions=result['decisions'], reason=result.get('reason'))
        with (out / 'training-games.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if not result['complete']:
            if config['smoke']:
                raise RuntimeError('Smoke game did not finish: ' + str(result.get('reason')))
            return False
        rows = [row for row in result['rows'] + result['value_rows'] if row.get('z') in (-1, 0, 1)]
        if not rows:
            raise ValueError('Completed murk game produced no labels')
        stats = update_murk(murk_net, murk_opt, rows)
        for value in stats.values():
            if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
                raise ValueError('Non-finite murk update')
        counts['murk']['updates'] += 1
        counts['murk']['games'] += 1
        counts['murk']['wins'] += int(result.get('winner') == 'a')
        current_murk = out / 'murk' / f"update-{counts['murk']['updates']:04d}"
        export_murk(murk_net, current_murk, 'murk', murk_build,
                    dict(initialization=murk_origin, games=counts['murk']['games'],
                         automatic_serving_approval=False))
        write(out / 'status.json', dict(phase='searching', roster=[*ROSTER, 'murk'], counts=counts,
                                        latest=str(current), murk=str(current_murk), scheduled=scheduled))
        return True

    def trained_enough():
        return counts['murk']['updates'] and any(counts[key]['updates'] for key in ROSTER)

    while time.time() < config['train_until'] and not (config['smoke'] and trained_enough()):
        jobs = []
        for _ in range(config['workers']):
            if time.time() >= config['train_until']:
                break
            left, right = plan[scheduled % len(plan)]
            seed = config['seeds']['foundation'] + scheduled
            jobs.append(murk_job(seed) if left == 'murk' else one_job('train', left, right, seed))
            scheduled += 1
        if not jobs:
            break
        for job, result in run_batch(jobs):
            if job['runtime'] == 'murk':
                absorb_murk(job, result)
            else:
                absorb_train(job, result)
            if config['smoke'] and trained_enough():
                break
    if not trained_enough():
        raise RuntimeError('Five-team round finished no roster game and no murk game')
    write(out / 'murk.json', dict(ran=True, initialization=murk_origin, games=counts['murk']['games'],
                                  latest=str(current_murk)))

    eval_games = []
    matrix = [(left, right) for left in ROSTER for right in ROSTER]
    played = 0
    while time.time() < config['evaluate_until'] and played < len(matrix):
        jobs = []
        while len(jobs) < config['workers'] and played + len(jobs) < len(matrix) and time.time() < config['evaluate_until']:
            left, right = matrix[played + len(jobs)]
            job = one_job('eval', left, right, config['seeds']['confirmation'] + played + len(jobs), mirror=left == right)
            job['training'] = False
            job['collect'] = False
            jobs.append(job)
        if not jobs:
            break
        for job, result in run_batch(jobs):
            eval_games.append(dict(left=job['left'], right=job['right'], complete=result['complete'],
                                   winner=result.get('winner'), first=job['first'], decisions=result['decisions']))
        played += len(jobs)
        if config['smoke']:
            break
    write(out / 'eval.json', dict(planned=len(matrix), games=eval_games, weights=str(current),
                                  approval=False, full_matrix=False, deck_search=False))
    write(out / 'status.json', dict(phase='complete', counts=counts, latest=str(current),
                                    murk=dict(games=counts['murk']['games'], latest=str(current_murk),
                                              initialization=murk_origin),
                                    evaluated=len(eval_games), full_matrix=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--roster-source', type=Path)
    parser.add_argument('--murk-source', type=Path)
    parser.add_argument('--seed-reservation', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
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
    from app.modules.card_game.rl.fixed_lineup import identity
    zone = shanghai_zone()
    now = datetime.now(zone)
    if args.smoke:
        train_until = now + timedelta(seconds=420)
        evaluate_until = now + timedelta(seconds=480)
        hard = now + timedelta(seconds=540)
        simulations, candidates, game_seconds, max_actions = 2, 4, 180, 400
        workers, device = 1, args.device
    else:
        if args.device != 'cuda':
            raise SystemExit('The current-method round requires CUDA')
        if not 3600 <= args.seconds <= 3600:
            raise SystemExit('This trial entry is one hour')
        hard = now + timedelta(seconds=args.seconds)
        train_until = hard - timedelta(minutes=10)
        evaluate_until = hard - timedelta(minutes=2)
        if not (now < train_until < evaluate_until < hard):
            raise SystemExit('Deadlines must satisfy now < training < evaluation < hard stop')
        simulations, candidates, game_seconds, max_actions = 32, 16, 900, 400
        workers, device = args.workers, 'cuda'
    plan = dict(
        kind='current_grounded_round_v1', roster=[*ROSTER, 'murk'],
        murk_games='same-contract-only', rewards=0, deck_search=False, midrange=False, search='gumbel',
        simulations=simulations, gumbel_candidates=candidates, terminal_horizon=3,
        hidden=args.hidden, device=device, workers=workers, now=now.isoformat(),
        train_until=train_until.isoformat(), evaluate_until=evaluate_until.isoformat(),
        hard_deadline=hard.isoformat(), rule_hash=identity(), website_approval=False,
        full_matrix=False, smoke=args.smoke,
        roster_source=str(args.roster_source) if args.roster_source else '',
        murk_source=str(args.murk_source) if args.murk_source else '')
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.run:
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory must be a new empty directory')
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        seeds = dict(foundation=910_200_001, search=910_300_001, confirmation=910_400_001)
    elif args.seed_reservation:
        seeds = json.loads(args.seed_reservation.read_text(encoding='utf-8'))
    else:
        from app.modules.card_game.rl.build_acceptance import claim_seeds
        seeds = claim_seeds(ROOT / 'artifacts' / 'rl-seed-ledger.json', str(out))
    config = dict(plan)
    config.update(seeds=seeds, game_seconds=game_seconds, max_actions=max_actions,
                  train_until=train_until.timestamp(), evaluate_until=evaluate_until.timestamp(),
                  hard_deadline=hard.timestamp())
    write(out / 'config.json', config)
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
                write(out / 'status.json', dict(phase='hard_stop', reason='one_hour'))
                break
            time.sleep(5)
    except KeyboardInterrupt:
        if os.name != 'nt':
            os.killpg(worker.pid, signal.SIGTERM)
        raise


if __name__ == '__main__':
    main()
