#!/usr/bin/env python3
"""Current-rule CPU WDL warmup on frozen old policies, then untouched seed audit.

Only the new WDL head receives gradient. The old policy and residual stay
unchanged, so value-learning cannot silently alter the evaluated trajectory
policy. This is a three-team value precheck, not complete five-team training.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_duel_v2_policy_recovery import write
from scripts.check_duel_v2_value_calibration import probability_metrics

KEYS = ('starter', 'weave-rush', 'quick-rush')


def jobs(seeds, phase, models, decks, deadline, replicas=2):
    pair = 0
    for replica in range(replicas):
        for i, left in enumerate(KEYS):
            for right in KEYS[i:]:
                for first in ('a', 'b'):
                    yield dict(seed=seeds[phase] + replica * 100 + pair, first=first, pure=True,
                               training=phase == 'foundation', phase=phase, value_warmup=True,
                               decks={'a': decks[left], 'b': decks[right]}, deadline=deadline,
                               policies={'a': (str(models), left), 'b': (str(models), right)})
                pair += 1


def rows_for(games, key):
    return [r for game in games for r in game['value_rows'] if r['key'] == key]


def metrics(net, games, key, temperature=1.):
    import torch
    probs = []; labels = []; weights = []; count = 0
    with torch.no_grad():
        for game in games:
            rows = [r for r in game['value_rows'] if r['key'] == key]
            if not rows: continue
            count += 1
            x = torch.as_tensor(np.stack([r['x'] for r in rows]))
            h = net.state_net(x)
            context = torch.tensor([[r['value_actor']] for r in rows], dtype=h.dtype)
            probabilities = (net.wdl(torch.cat((h, context), -1)) / temperature).softmax(-1).numpy()
            probs.extend(probabilities); labels.extend(int(r['z']) + 1 for r in rows)
            weights.extend([1 / len(rows)] * len(rows))
    return dict(**probability_metrics(probs, labels, weights), normal_games=count, observations=len(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, default=Path('artifacts/rl-evals/residual-recovery-20260927/preservation/models'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--train-replicas', type=int, default=2)
    parser.add_argument('--calibrate', action='store_true', help='Separate selection seeds choose temperature from (1,2,4)')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 60 <= args.seconds <= 1200: parser.error('CPU budget must be 60..1200 seconds')
    if not 2 <= args.train_replicas <= 8: parser.error('Train replicas must be 2..8')
    config = dict(keys=list(KEYS), episodes_per_phase=24, phases=['foundation', 'confirmation'],
                  learning_rate=.003, passes=16, batch=128, head_only=True, optimizer='Adam',
                  policy_updates=0, environment_reward='terminal_wdl_only',
                  seconds=args.seconds, strength_approval=False, formal_training_allowed=False)
    config.update(train_episodes=12 * args.train_replicas, train_replicas=args.train_replicas,
                  calibration_episodes=24 if args.calibrate else 0,
                  temperatures=[1., 2., 4.] if args.calibrate else [1.],
                  calibration_partition='selection' if args.calibrate else None)
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists(): raise ValueError('New output required')
    import torch
    from app.modules.card_game.rl import recovery_runtime as rt, preserved_policy
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.rl.episode_replay import dump_episode
    torch.set_num_threads(2); torch.manual_seed(20260927)
    seeds = claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(args.output.resolve()))
    now = datetime.now(timezone(timedelta(hours=8)))
    hard = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if hard <= now: hard += timedelta(days=1)
    deadline = min(time.time() + args.seconds, hard.timestamp())
    config.update(seeds=seeds, hard_deadline=hard.isoformat(), deadline=deadline, rule_hash=identity())
    decks = {key: rt.model(args.initial, key).serving_deck for key in KEYS}
    config['initial_sha'] = {key: rt.model(args.initial, key).version for key in KEYS}
    write(args.output / 'config.json', config)
    train = []
    for index, job in enumerate(jobs(seeds, 'foundation', args.initial, decks, deadline, args.train_replicas)):
        result = rt.search_game(job)
        if not result['complete']:
            write(args.output / 'result.json', dict(complete=False, reason=result['reason'], stage='collection')); return
        job.update(train_sides=['a', 'b'], id=f'foundation-{index}')
        dump_episode(args.output / 'episodes' / f'foundation-{index}.json.gz', job, result)
        train.append(result)
    nets = {}; train_stats = {}
    for key in KEYS:
        net, build, origin = rt.restore(args.initial, key)
        before = {name: tensor.clone() for name, tensor in net.state_dict().items()}
        rows = rows_for(train, key); rng = np.random.default_rng(20260927)
        opt = torch.optim.Adam(net.wdl.parameters(), lr=config['learning_rate'])
        updates = 0
        for epoch in range(config['passes']):
            order = rng.permutation(len(rows))
            for start in range(0, len(rows), config['batch']):
                if time.time() >= deadline:
                    write(args.output / 'result.json', dict(complete=False, reason='deadline', stage='warmup')); return
                indexes = order[start:start + config['batch']]
                rt.update(net, opt, [rows[int(i)] for i in indexes], warmup=True)
                updates += 1
        for name, tensor in net.state_dict().items():
            if not name.startswith('wdl.') and not torch.equal(before[name], tensor):
                raise ValueError('Head-only warmup changed policy or trunk')
        origin = dict(origin, value_warmup_updates=updates, policy_updates=0,
                      value_training_source='current_rule_complete_frozen_policy_games',
                      value_training_seed_partition='foundation')
        preserved_policy.export(net, args.output / 'models', key, build, origin)
        nets[key] = net; train_stats[key] = metrics(net, train, key)
        print(key, 'warmup', updates, train_stats[key]['brier'], flush=True)
    calibration = {}; chosen_temperatures = {}
    if args.calibrate:
        selected_games = []
        for index, job in enumerate(jobs(seeds, 'selection', args.initial, decks, deadline)):
            result = rt.search_game(job)
            if not result['complete']:
                write(args.output / 'result.json', dict(complete=False, reason=result['reason'], stage='calibration')); return
            job.update(train_sides=['a', 'b'], id=f'selection-{index}')
            dump_episode(args.output / 'calibration' / f'selection-{index}.json.gz', job, result)
            selected_games.append(result)
        for key in KEYS:
            calibration[key] = {str(t): metrics(nets[key], selected_games, key, t) for t in config['temperatures']}
            chosen = min(config['temperatures'], key=lambda t: (calibration[key][str(t)]['log_loss'], t))
            chosen_temperatures[key] = chosen
            with torch.no_grad():
                nets[key].wdl.weight.div_(chosen); nets[key].wdl.bias.div_(chosen)
            old_origin = rt.model(args.output / 'models', key).manifest['origin']
            preserved_policy.export(nets[key], args.output / 'calibrated-models', key, decks[key],
                                    dict(old_origin, value_temperature=chosen, calibration_partition='selection'))
        write(args.output / 'temperature-selection.json', dict(grid=calibration, chosen=chosen_temperatures,
                                                              selection_only=True))
    # Only now score the reserved heldout games. No optimization or selection
    # follows this check. They are separate from foundation even on failures.
    held = []
    for index, job in enumerate(jobs(seeds, 'confirmation', args.initial, decks, deadline)):
        result = rt.search_game(job)
        if not result['complete']:
            write(args.output / 'result.json', dict(complete=False, reason=result['reason'], stage='heldout')); return
        job.update(train_sides=['a', 'b'], id=f'confirmation-{index}')
        dump_episode(args.output / 'heldout' / f'confirmation-{index}.json.gz', job, result)
        held.append(result)
    model_folder = args.output / ('calibrated-models' if args.calibrate else 'models')
    report = dict(schema='recovery_value_precheck_v1', complete=True, heldout=True,
                  rule_hash=identity(), models={key: rt.model(model_folder, key).version for key in KEYS},
                  metrics={key: metrics(nets[key], held, key) for key in KEYS},
                  train_metrics=train_stats, independent_current_rule_seeds=True,
                  model_folder=str(model_folder), temperature_selection=chosen_temperatures,
                  seed_partitions=dict(train='foundation', heldout='confirmation'),
                  policy_preserved=True, website_approval=False, formal_training_allowed=False)
    report['passed'] = all(m['brier'] < 2 / 3 and m['log_loss'] < np.log(3) and m['normal_games'] >= 8
                           for m in report['metrics'].values())
    write(args.output / 'value-precheck.json', report)
    write(args.output / 'result.json', report)
    lines = ['# 现行规则的三套WDL预热与留出核验', '',
             f"{config['train_episodes']}个完整冻结策略训练局、{config['calibration_episodes']}个温度选择局、另24个未参与优化或温度选择的确认局。只更新新WDL头，旧策略与状态主干逐张量保持不变。", '',
             '| 队伍 | 训练Brier | 确认Brier | 确认对数损失 | 确认完整局数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for key, m in report['metrics'].items():
        lines.append(f"| {key} | {train_stats[key]['brier']:.3f} | {m['brier']:.3f} | {m['log_loss']:.3f} | {m['normal_games']} |")
    lines += ['', f"预检通过：{report['passed']}。只允许下一项有界研究检查，不批准五套正式长训、策略改进或服务。", '',
              '确认样本此后已消费，不再次用于选择超参数或声明独立确认。真红、浊燃没有旧教师，不在本次三套范围。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print('value precheck', report['passed'], flush=True)


if __name__ == '__main__': main()
