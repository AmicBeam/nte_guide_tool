#!/usr/bin/env python3
"""Private same-root hidden/full-hand search comparison; no training or service change."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_duel_v2_policy_recovery import write
from scripts.check_duel_v2_search_stability import json_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/initial'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--visibility', choices=('root_only', 'both'), default='root_only')
    args = parser.parse_args()
    if not 60 <= args.seconds <= 900: parser.error('Bounded 60..900 seconds required')
    config = dict(learning=False, website_change=False, source=str(args.source), roots_per_team=1,
                  repeats=2, simulations=32, q_scale='legacy', declared_visibility='opponent_current_hand_only',
                  deck_order='sampled_not_revealed', trained_for_full_hand=False, strength_claim=False)
    config['visibility'] = args.visibility
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists(): raise ValueError('New output required')
    from app.modules.card_game.rl.archive_diagnostics import ArchivedDiagnosticModel
    from app.modules.card_game.rl import cross_runtime as hidden, full_hand_diagnostic as opened
    from app.modules.card_game.rl.information_search import search
    from app.modules.card_game.rl.search_stability import summarize
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side, apply_action
    from app.modules.card_game.rl.league_rollout import clean
    source = json.loads((args.source / 'config.json').read_text())
    if source['current_rule_hash'] != identity(): raise ValueError('Rule identity changed')
    roots = [json.loads(line) for line in (args.source / 'roots.jsonl').read_text().splitlines()]
    from datetime import datetime
    deadline = min(time.time() + args.seconds, datetime.fromisoformat(source['hard_deadline']).timestamp())
    config['deadline'] = deadline; write(args.output / 'config.json', config)
    results = []; incomplete = []
    for key_index, key in enumerate(source['keys']):
        policy = ArchivedDiagnosticModel(args.models, key); policies = {'a': policy, 'b': policy}
        root = next(r for r in roots if r['key'] == key and r['root'] == 0)
        if root['model_sha'] != policy.version: raise ValueError('Numeric source changed')
        state = new_game(seed=source['seeds']['probes'] + key_index, first_side='a',
                         decks={'a': policy.serving_deck, 'b': policy.serving_deck})
        for _ in range(root['step']):
            side = acting_side(state); actions, (x, c) = hidden.decision(state, side)
            state = clean(apply_action(state, side, actions[int(policy.scores(x, c).argmax())]))
        modes = {}
        oracle_runtime = opened.for_viewer(acting_side(state)) if args.visibility == 'root_only' else opened
        for name, runtime in (('hidden', hidden), ('full_hand', oracle_runtime)):
            searches = []
            for repeat in range(2):
                result = search(state, acting_side(state), policies,
                                seed=(1 << 128) + source['seeds']['probes'] + key_index * 10000 + repeat,
                                runtime=runtime, algorithm='gumbel', simulations=32, noise=False,
                                deadline=deadline, diagnostics=True)
                if not result['complete']:
                    incomplete.append(dict(key=key, mode=name)); break
                searches.append(result)
            if len(searches) == 2:
                modes[name] = summarize(searches)
                with (args.output / 'roots.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(key=key, mode=name,
                        searches=[dict(pi=r['pi'], selected=r['search_choice'], diagnostics=r['root_diagnostics'])
                                  for r in searches]), default=json_value) + '\n')
        results.append(dict(key=key, modes=modes))
        print(key, json.dumps({mode: m['represented_group_js'] for mode, m in modes.items()}), flush=True)
        if incomplete: break
    write(args.output / 'result.json', dict(complete=not incomplete and len(results) == 5, results=results,
                                          incomplete=incomplete, learning=False, trained_for_full_hand=False,
                                          website_approval=False, strength_verified=False))
    lines = ['# 暗牌／当前手牌明牌的同根诊断', '',
             f"相同冻结数值、相同根、Gumbel32预算与搜索种子；牌库顺序仍采样。手牌权限为{args.visibility}，root_only仅授予搜索方、对手模拟策略仍使用暗牌观察。仅五个已消费诊断根，各重复两次，不训练或修改网站观察。", '',
             '| 队伍 | 暗牌目标JS | 明牌目标JS |', '| --- | ---: | ---: |']
    for r in results:
        if len(r['modes']) == 2:
            lines.append(f"| {r['key']} | {r['modes']['hidden']['represented_group_js']:.4f} | {r['modes']['full_hand']['represented_group_js']:.4f} |")
    lines += ['', '模型没有按明牌分布训练；这只能检查搜索标签的敏感性，不能据此给出训练加速倍数或明牌策略胜率。明牌也不会让未校准价值或微小Q差的极差放大自动正确。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__': main()
