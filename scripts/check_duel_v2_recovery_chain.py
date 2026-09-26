#!/usr/bin/env python3
"""Bounded CPU three-team search/update/export/pure-game recovery chain.

Requires a current-rule, matching-array WDL precheck. It does not approve
formal five-team training, deck search, deployment, or a stronger model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEYS = ('starter', 'weave-rush', 'quick-rush')


def play(job):
    from app.modules.card_game.rl.recovery_runtime import search_game
    return search_game(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--value-precheck', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--episodes', type=Path, help='Reuse only matching completed TRAINING episodes; reserve fresh evaluation seeds')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 300 <= args.seconds <= 1800: parser.error('Bounded chain budget must be 300..1800 seconds')
    config = dict(keys=list(KEYS), search_games=6, pure_comparison_games=24, simulations=32,
                  candidates=16, q_scale='natural_wdl', learning_rate=.0003, passes=4,
                  workers=2, seconds=args.seconds, complete_report=False, website_approval=False)
    config['device'] = args.device
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists(): raise ValueError('New output required')
    import torch
    from app.modules.card_game.rl import recovery_runtime as rt, preserved_policy
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.episode_replay import dump_episode
    from scripts.fit_duel_v2_policy_recovery import write
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; no fallback or output created')
    report = json.loads(args.value_precheck.read_text())
    policies = {key: rt.model(args.models, key) for key in KEYS}
    if not rt.value_precheck_ok(report, policies):
        raise ValueError('Current matching-array WDL precheck has not passed')
    seeds = claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(args.output.resolve()))
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=8)))
    hard = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if hard <= now: hard += timedelta(days=1)
    deadline = min(time.time() + args.seconds, hard.timestamp())
    sampling_end = time.time() + .7 * (deadline - time.time())
    config.update(seeds=seeds, deadline=deadline, hard_deadline=hard.isoformat(), sampling_end=sampling_end,
                  initial_sha={key: policies[key].version for key in KEYS})
    config.update(training_episode_source=str(args.episodes) if args.episodes else None,
                  nonactor_value_updates=True)
    write(args.output / 'config.json', config)
    from app.modules.card_game.rl.offline_sources import snapshot
    from scripts.prepare_duel_v2_offline_readiness import EXTRA
    snapshot(ROOT, args.output / 'source', EXTRA)
    decks = {key: policies[key].serving_deck for key in KEYS}
    jobs = []
    pair = 0
    for i, left in enumerate(KEYS):
        for right in KEYS[i + 1:]:
            for first in ('a', 'b'):
                jobs.append(dict(seed=seeds['foundation'] + pair, first=first, training=True,
                                 q_scale='natural_wdl', value_precheck=report, deadline=sampling_end,
                                 decks={'a': decks[left], 'b': decks[right]}, train_sides=['a', 'b'],
                                 policies={'a': (str(args.models), left), 'b': (str(args.models), right)},
                                 id=f'foundation-{len(jobs)}'))
            pair += 1
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    games = []
    if args.episodes:
        from app.modules.card_game.rl.episode_replay import load_episode
        for path in sorted(args.episodes.glob('*.json.gz')):
            data = load_episode(path); job = data['job']
            if job.get('training') is not True or job.get('q_scale') != 'natural_wdl':
                raise ValueError('Only natural-WDL training episodes may be reused')
            expected = {side: policies[spec[1]].version for side, spec in job['policies'].items()}
            if data.get('model_versions') != expected or job.get('value_precheck', {}).get('rule_hash') != report['rule_hash']:
                raise ValueError('Training episode source arrays/rules mismatch')
            if not data.get('winner') or not data.get('actions'):
                raise ValueError('Incomplete training episode')
            games.append(dict(complete=True, rows=[r for r in data['rows'] if 'pi' in r],
                              value_rows=[r for r in data['rows'] if 'pi' not in r],
                              episode_actions=data['actions']))
        if len(games) != 6: raise ValueError('The bounded chain requires six completed cross-training games')
    else:
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
            for job, result in zip(jobs, pool.map(play, jobs)):
                if not result['complete']:
                    write(args.output / 'result.json', dict(complete=False, stage='sampling', reason=result['reason'],
                                                           completed=len(games), quality_approved=False)); return
                dump_episode(args.output / 'episodes' / f"{job['id']}.json.gz", job, result)
                games.append(result)
                print(job['id'], result['decisions'], result['turn'], flush=True)
    updates = {}; candidates = args.output / 'candidates'
    restore_checks = {}
    for key in KEYS:
        net, build, origin = rt.restore(args.models, key, device=args.device)
        rows = [r for game in games for r in game['rows'] if r['key'] == key]
        # Fit diagnostics use training roots, never replace independent games.
        baseline_policy = rt.model(args.models, key)
        def label_kl(policy):
            values = []
            for row in rows:
                logits = np.asarray(policy.scores(row['x'], row['c']), dtype=np.float64)
                logp = logits - np.logaddexp.reduce(logits)
                target = np.asarray(row['pi'], dtype=np.float64); positive = target > 0
                values.append(float(np.sum(target[positive] * (np.log(target[positive]) - logp[positive]))))
            return float(np.mean(values))
        initial_kl = label_kl(baseline_policy)
        opt = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
        rng = np.random.default_rng(seeds['foundation']); seen = np.zeros(len(rows), dtype=np.int32)
        gradients = dict(cand_net=0., state_net=0., score=0.); count = 0
        for _ in range(config['passes']):
            order = rng.permutation(len(rows))
            for start in range(0, len(rows), 128):
                if time.time() >= deadline:
                    write(args.output / 'result.json', dict(complete=False, stage='update', quality_approved=False)); return
                indexes = order[start:start + 128]
                stats = rt.update(net, opt, [rows[int(i)] for i in indexes]); seen[indexes] += 1; count += 1
                for name in gradients: gradients[name] += stats['policy_gradient_norms'].get(name, 0.)
        if not len(rows) or not all(np.isfinite(g) and g > 0 for g in gradients.values()):
            raise ValueError('Fresh batch did not reach every policy branch')
        values = [r for game in games for r in game['value_rows']
                  if r['key'] == key and r.get('value_actor') == 0]
        value_steps = 0
        for start in range(0, len(values), 128):
            if time.time() >= deadline:
                write(args.output / 'result.json', dict(complete=False, stage='value_update', quality_approved=False)); return
            rt.update(net, opt, values[start:start + 128], value_only=True)
            value_steps += 1
        preserved_policy.export(net, candidates, key, build, dict(origin, policy_updates=count,
                                recovery_chain=True, q_scale='natural_wdl', quality_approved=False))
        numeric = rt.model(candidates, key)
        from app.modules.card_game.rl.league_learning import tensors
        sample_rows = rows[:min(32, len(rows))]
        with torch.no_grad():
            xx, cc, mask = tensors([(r['x'], r['c']) for r in sample_rows], args.device)
            scores, _ = net(xx, cc, mask)
            for i, row in enumerate(sample_rows):
                predicted = scores[i, :len(row['c'])].cpu().numpy()
                exported = numeric.scores(row['x'], row['c'])
                np.testing.assert_allclose(predicted, exported, atol=2e-4, rtol=2e-4)
                if int(predicted.argmax()) != int(exported.argmax()):
                    raise ValueError('Export changed actual selected action')
        updates[key] = dict(policy_rows=len(rows), optimizer_steps=count, min_reuse=int(seen.min()),
                            max_reuse=int(seen.max()), policy_gradient_sums=gradients,
                            nonactor_value_rows=len(values), value_optimizer_steps=value_steps,
                            initial_training_kl=initial_kl, candidate_training_kl=label_kl(numeric),
                            torch_numpy_checked=len(sample_rows), torch_numpy_actions_equal=True,
                            candidate_sha=rt.model(candidates, key).version)
        from app.modules.card_game.rl.recovery_checkpoint import save as save_checkpoint, restore as restore_checkpoint
        checkpoint = args.output / 'restore-checkpoints' / f'{key}.pt'
        save_checkpoint(checkpoint, net, opt, directory=candidates, key=key,
                        rng_state=rng.bit_generator.state, metadata=dict(config=config, updates=updates[key]))
        restored, restored_opt, data = restore_checkpoint(checkpoint, directory=candidates, key=key, device=args.device)
        sample = rows[:min(32, len(rows))]
        # A diagnostic next step on copies proves Adam moments/step counts
        # were restored, without changing the exported candidate or its report.
        from copy import deepcopy
        control = deepcopy(net)
        control_opt = torch.optim.Adam(control.parameters()); control_opt.load_state_dict(deepcopy(opt.state_dict()))
        rt.update(control, control_opt, sample); rt.update(restored, restored_opt, sample)
        if any(not torch.equal(value, restored.state_dict()[name]) for name, value in control.state_dict().items()):
            raise ValueError('Restored optimizer next step mismatch')
        if data['rng_state'] != rng.bit_generator.state:
            raise ValueError('Restored sampling RNG mismatch')
        restore_checks[key] = dict(numeric_equal=True, adam_next_step_equal=True, rng_equal=True,
                                   diagnostic_only=True, exported_candidate_unchanged=True)
    evaluations = []
    for key_index, key in enumerate(KEYS):
        for replica in range(2):
            for learner in ('a', 'b'):
                for first in ('a', 'b'):
                    job = dict(seed=seeds['confirmation'] + key_index * 100 + replica, first=first,
                               pure=True, training=False, deadline=deadline,
                               decks={'a': decks[key], 'b': decks[key]},
                               policies={s: (str(candidates if s == learner else args.models), key) for s in ('a', 'b')})
                    result = play(job)
                    counts = Counter((r['side'], r['action']['type']) for r in result['episode_actions'])
                    row = dict(key=key, replica=replica, candidate_side=learner, first=first,
                               complete=result['complete'], winner=result.get('winner'),
                               candidate_win=result.get('winner') == learner,
                               candidate_plays=counts[learner, 'play_card'], baseline_plays=counts['b' if learner == 'a' else 'a', 'play_card'])
                    evaluations.append(row)
                    with (args.output / 'games.jsonl').open('a') as handle: handle.write(json.dumps(row) + '\n')
                    if not result['complete']:
                        write(args.output / 'result.json', dict(complete=False, stage='evaluation', games=evaluations,
                                                              quality_approved=False)); return
    summary = dict(complete=True, search_games=len(games), pure_games=len(evaluations), updates=updates,
                   restore_checks=restore_checks,
                   pure={key: dict(wins=sum(r['candidate_win'] for r in evaluations if r['key'] == key), games=8,
                                   candidate_plays=sum(r['candidate_plays'] for r in evaluations if r['key'] == key),
                                   baseline_plays=sum(r['baseline_plays'] for r in evaluations if r['key'] == key)) for key in KEYS},
                   quality_approved=False, complete_report=False, website_approval=False,
                   reason='bounded_chain_and_two_paired_roots_per_team_not_strength_acceptance')
    write(args.output / 'result.json', summary)
    lines = ['# CPU三套搜索学习闭环短测', '',
             '先通过现行规则WDL预检，再用自然WDL尺度逐决策Gumbel32得到6个真实完整跨队训练局。策略行完整遍历，另更新非行动方的终局价值行。复用时严格核对训练归属、规则与源权重，不复用确认局；独立纯网络比较使用新种子，交换引擎座位和先手。', '',
             '| 队伍 | 新策略行／每行使用次数 | 候选胜局／正常局 | 候选主动出牌 | 起点主动出牌 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for key in KEYS:
        u, p = updates[key], summary['pure'][key]
        lines.append(f"| {key} | {u['policy_rows']}/{u['min_reuse']} | {p['wins']}/{p['games']} | {p['candidate_plays']} | {p['baseline_plays']} |")
    lines += ['', '有限短测不批准强度。没有镜像训练、历史联赛、配牌搜索、完整五套主表或服务替换；真红和浊燃不在此三套教师恢复闭环中。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__': main()
