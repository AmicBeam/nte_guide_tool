#!/usr/bin/env python3
"""Representation aliases, target inconsistency and value metrics on training archives."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_duel_v2_policy_recovery import load_games, write
from app.modules.card_game.rl.policy_equivalence import representation_groups, irreducible_kl, canonical_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/episodes'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New output required')
    games = load_games(args.episodes)
    counts = defaultdict(lambda: dict(policy_rows=0, alias_roots=0, duplicate_actions=0,
                                     target_top_in_alias=0, irreducible_kl_sum=0., impossible_top_margin=0))
    root_targets = defaultdict(list); examples = defaultdict(list)
    for game in games:
        for row in game['rows']:
            if 'pi' not in row:
                continue
            key = row['key']; stat = counts[key]; stat['policy_rows'] += 1
            groups = representation_groups(row['c']); duplicate = [g for g in groups if len(g) > 1]
            floor = irreducible_kl(row['c'], row['pi'])
            top = int(np.argmax(row['pi']))
            stat['alias_roots'] += bool(duplicate)
            stat['duplicate_actions'] += sum(len(g) - 1 for g in duplicate)
            stat['target_top_in_alias'] += any(top in group for group in duplicate)
            stat['irreducible_kl_sum'] += floor
            ordered = np.sort(row['pi'])
            stat['impossible_top_margin'] += int(len(ordered) > 1 and ordered[-1] - ordered[-2] >= .05
                                                 and any(top in group for group in duplicate))
            identity, distribution = canonical_root(row['x'], row['c'], row['pi'])
            root_targets[key, identity].append(distribution)
            if floor > .05 and len(examples[key]) < 8:
                examples[key].append(dict(episode=game['episode'], step=row['step'], groups=duplicate,
                                          pi=row['pi'].tolist(), irreducible_kl=floor))
    repeated = defaultdict(lambda: dict(repeated_visible_roots=0, repeated_rows=0, conflicting_top=0,
                                       mean_target_l1_sum=0.))
    for (key, _), targets in root_targets.items():
        if len(targets) < 2:
            continue
        stat = repeated[key]; stat['repeated_visible_roots'] += 1; stat['repeated_rows'] += len(targets)
        stat['conflicting_top'] += len({int(np.argmax(pi)) for pi in targets}) > 1
        center = np.mean(targets, 0)
        stat['mean_target_l1_sum'] += float(np.mean([np.abs(pi - center).sum() for pi in targets]))
    result = dict(complete=True, training_episodes=len(games), representation_is_not_rule_equivalence=True,
                  keys={key: dict(**stat, irreducible_kl_mean=stat['irreducible_kl_sum'] / stat['policy_rows'],
                                  repeated=repeated[key], examples=examples[key]) for key, stat in counts.items()})
    write(args.output / 'result.json', result)
    lines = ['# 训练数据表示审计', '',
             '相同候选特征必须得到相同logits；不据此认定两个真实动作在规则上等价，也不改变动作选择。', '',
             '| 队伍 | 策略行 | 有不可区分候选的根 | 明确目标落在不可区分候选组 | 不可消除KL均值 | 重复可见根／首选冲突根 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, s in result['keys'].items():
        r = s['repeated']
        lines.append(f"| {key} | {s['policy_rows']} | {s['alias_roots']} | {s['impossible_top_margin']} | {s['irreducible_kl_mean']:.4f} | {r['repeated_visible_roots']}/{r['conflicting_top']} |")
    lines += ['', '这些数字只定位表示与标签可拟合性。不可消除KL不是全部失败的解释；其他误差仍须检查优化、泛化和老师质量。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({k: {n: s[n] for n in ('policy_rows', 'alias_roots', 'irreducible_kl_mean')} for k, s in result['keys'].items()}))


if __name__ == '__main__':
    main()
