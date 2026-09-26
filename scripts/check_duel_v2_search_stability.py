#!/usr/bin/env python3
"""Fresh same-root Gumbel32 repetitions, diagnostic-only and bounded on CPU."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/initial'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 60 <= args.seconds <= 1200:
        parser.error('Bounded CPU budget must be 60..1200 seconds')
    config = dict(keys=['starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk'], roots_per_team=2,
                  repetitions=4, simulations=32, candidates=16, terminal_horizon=3,
                  noise=False, training=False, label_reuse=False, seconds=args.seconds,
                  same_tree_unscaled='diagnostic target only, no alternate-search or strength claim')
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists():
        raise ValueError('New output required')
    from app.modules.card_game.rl.archive_diagnostics import ArchivedDiagnosticModel
    from app.modules.card_game.rl import cross_runtime as rt
    from app.modules.card_game.rl.information_search import search
    from app.modules.card_game.rl.search_stability import summarize
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side, apply_action
    from app.modules.card_game.rl.league_rollout import clean
    from scripts.fit_duel_v2_policy_recovery import write
    seeds = claim_seeds(ROOT / 'artifacts/rl-seed-ledger.json', str(args.output.resolve()))
    zone = timezone(timedelta(hours=8)); now = datetime.now(zone)
    hard = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if hard <= now: hard += timedelta(days=1)
    deadline = min(time.time() + args.seconds, hard.timestamp())
    config.update(seeds=seeds, deadline=deadline, hard_deadline=hard.isoformat(), current_rule_hash=identity())
    write(args.output / 'config.json', config)
    all_roots = []; errors = []
    for key_index, key in enumerate(config['keys']):
        policy = ArchivedDiagnosticModel(args.models, key)
        s = new_game(seed=seeds['probes'] + key_index, first_side='a',
                     decks={'a': policy.serving_deck, 'b': policy.serving_deck})
        policies = {'a': policy, 'b': policy}; count = 0
        for step in range(80):
            if s['phase'] == 'finished' or count == config['roots_per_team'] or time.time() >= deadline:
                break
            actor = acting_side(s); actions, (x, c) = rt.decision(s, actor)
            eligible = (s['phase'] == 'playing' and s['sides'][actor]['ap'] == 2
                        and any(a['type'] == 'play_card' for a in actions))
            if eligible:
                repetitions = []
                for repeat in range(config['repetitions']):
                    result = search(s, actor, policies,
                                    seed=(1 << 128) + seeds['probes'] + key_index * 10000 + count * 100 + repeat,
                                    simulations=32, algorithm='gumbel', gumbel_candidates=16,
                                    terminal_horizon=3, runtime=rt, noise=False, diagnostics=True, deadline=deadline)
                    if not result['complete']:
                        errors.append(dict(key=key, root=count, repeat=repeat, reason='incomplete_search')); break
                    repetitions.append(result)
                if len(repetitions) != config['repetitions']:
                    break
                metrics = summarize(repetitions)
                record = dict(key=key, root=count, step=step, actor=actor, metrics=metrics,
                              model_sha=policy.version, historical_rule_hash=policy.historical_rule_hash,
                              x=x, c=c, actions=actions,
                              searches=[dict(pi=r['pi'], selected=r['search_choice'], value=r['value'],
                                             diagnostics=r['root_diagnostics']) for r in repetitions])
                with (args.output / 'roots.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, default=json_value) + '\n')
                all_roots.append(dict(key=key, root=count, **metrics)); count += 1
                print(key, count, json.dumps(metrics), flush=True)
            s = clean(apply_action(s, actor, actions[int(policy.scores(x, c).argmax())]))
        if count < config['roots_per_team'] and not any(e['key'] == key for e in errors):
            errors.append(dict(key=key, reason='insufficient_roots', roots=count))
    result = dict(complete=not errors and len(all_roots) == 10, roots=all_roots, errors=errors,
                  historical_numeric_adaptation=True, learning=False, website_approval=False)
    write(args.output / 'result.json', result)
    lines = ['# 同可见根搜索标签稳定性', '',
             '现行规则、新种子根，失败归档数值只读适配；不把它作为续训起点，不生成可学习新局。每根四次Gumbel32，无探索噪声。', '',
             '| 队伍／根 | 目标首选组 | 实际选择组 | 表示组JS | 同树未缩放目标JS | 原始Q极差范围 |',
             '| --- | --- | --- | ---: | ---: | --- |']
    for r in all_roots:
        spans = r['raw_q_spans']
        lines.append(f"| {r['key']}/{r['root']} | {r['target_top_groups']} | {r['actual_selected_groups']} | {r['represented_group_js']:.4f} | {r['same_tree_unscaled_target_js']:.4f} | {min(spans):.5f}..{max(spans):.5f} |")
    lines += ['', '未缩放目标只重算同一棵树，不能证明另一种搜索更强。首选组按模型表示分组，不据此断定规则等价。单批重复搜索不能替代完整对局胜率或独立强度验收。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
