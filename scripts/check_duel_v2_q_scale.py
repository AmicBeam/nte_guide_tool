#!/usr/bin/env python3
"""Equal-budget alternate Gumbel traversal on consumed diagnostic roots only."""
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
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = dict(modes=['legacy', 'natural_wdl'], simulations=32, candidates=16,
                  repeats=4, roots=10, noise=False, learning=False, source=str(args.source),
                  independent=False, seconds=args.seconds, serving_approval=False)
    if not 60 <= args.seconds <= 1200:
        parser.error('Bounded CPU seconds must be 60..1200')
    if not args.run:
        print(json.dumps(dict(dry_run=True, config=config))); return
    if args.output.exists(): raise ValueError('New output required')
    source = json.loads((args.source / 'config.json').read_text())
    roots = [json.loads(line) for line in (args.source / 'roots.jsonl').read_text().splitlines()]
    from app.modules.card_game.rl.cross_lineup import identity
    if source['current_rule_hash'] != identity() or len(roots) != 10:
        raise ValueError('Source diagnostic root identity/coverage mismatch')
    from app.modules.card_game.rl.archive_diagnostics import ArchivedDiagnosticModel
    from app.modules.card_game.rl import cross_runtime as rt
    from app.modules.card_game.rl.recovery_search import search
    from app.modules.card_game.rl.search_stability import summarize
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side, apply_action
    from app.modules.card_game.rl.league_rollout import clean
    # The new bounded run shares the original absolute hard deadline, not its
    # short diagnostic time slice.
    from datetime import datetime
    deadline = min(time.time() + args.seconds, datetime.fromisoformat(source['hard_deadline']).timestamp())
    config['deadline'] = deadline; config['hard_deadline'] = source['hard_deadline']
    write(args.output / 'config.json', config)
    records = []; errors = []
    for key_index, key in enumerate(source['keys']):
        policy = ArchivedDiagnosticModel(args.models, key); policies = {'a': policy, 'b': policy}
        s = new_game(seed=source['seeds']['probes'] + key_index, first_side='a',
                     decks={'a': policy.serving_deck, 'b': policy.serving_deck})
        expected = {r['step']: r for r in roots if r['key'] == key}
        for step in range(max(expected) + 1):
            actor = acting_side(s); actions, (x, c) = rt.decision(s, actor)
            if step in expected:
                saved = expected[step]
                if saved['model_sha'] != policy.version:
                    raise ValueError('Diagnostic source arrays changed')
                np.testing.assert_array_equal(x, np.asarray(saved['x'], dtype=np.float32))
                np.testing.assert_array_equal(c, np.asarray(saved['c'], dtype=np.float32))
                modes = {}
                for mode in config['modes']:
                    searches = []
                    for repeat in range(config['repeats']):
                        r = search(s, actor, policies,
                                   seed=(1 << 128) + source['seeds']['probes'] + key_index * 10000 + saved['root'] * 100 + repeat,
                                   runtime=rt, simulations=32, q_scale=mode, noise=False, deadline=deadline)
                        if not r['complete']:
                            errors.append(dict(key=key, root=saved['root'], mode=mode, repeat=repeat)); break
                        searches.append(r)
                        if mode == 'legacy':
                            np.testing.assert_array_equal(r['pi'], np.asarray(saved['searches'][repeat]['pi'], dtype=np.float32))
                            if r['search_choice'] != saved['searches'][repeat]['selected']:
                                raise ValueError('Legacy traversal drifted from shared source search')
                    if len(searches) != config['repeats']: break
                    metrics = summarize(searches)
                    entropy = [float(-np.sum(r['pi'] * np.log(np.maximum(r['pi'], 1e-12)))) for r in searches]
                    modes[mode] = dict(**metrics, mean_entropy=float(np.mean(entropy)),
                                       full_traversal=True)
                    with (args.output / 'searches.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(key=key, root=saved['root'], mode=mode,
                            searches=[dict(pi=r['pi'], selected=r['search_choice'], diagnostics=r['root_diagnostics'])
                                      for r in searches]), default=json_value) + '\n')
                records.append(dict(key=key, root=saved['root'], modes=modes))
                print(key, saved['root'], json.dumps({mode: (m['represented_group_js'], m['mean_entropy'])
                                                      for mode, m in modes.items()}), flush=True)
                if errors: break
            s = clean(apply_action(s, actor, actions[int(policy.scores(x, c).argmax())]))
        if errors: break
    write(args.output / 'result.json', dict(complete=not errors and len(records) == 10,
                                          roots=records, errors=errors, independent=False,
                                          strength_verified=False, learning=False))
    lines = ['# 同根同预算Q尺度对照', '',
             '两分支都重新执行完整Gumbel32，不再只重算原树目标。根来自已消费诊断，不进入梯度；不是胜率实验。', '',
             '| 队伍／根 | 极差缩放JS | 自然WDL尺度JS | 极差缩放熵 | 自然WDL尺度熵 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        if len(r['modes']) != 2: continue
        a, b = r['modes']['legacy'], r['modes']['natural_wdl']
        lines.append(f"| {r['key']}/{r['root']} | {a['represented_group_js']:.5f} | {b['represented_group_js']:.5f} | {a['mean_entropy']:.3f} | {b['mean_entropy']:.3f} |")
    lines += ['', '自然尺度抑制微小Q差的放大，不能让未校准价值自动变准确。接近原先验／均匀的稳定目标也可能没有新的教学信息，因此仍要求价值预检、旧能力保持与完整独立对局。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__': main()
