#!/usr/bin/env python3
"""Bounded CPU learning-recovery experiment on completed training episodes.

Keeps previously viewed confirmation and selection seeds out of fitting.
Archive holdouts are exploratory, never independent strength evidence. No
self-play, deployment or formal training. A new schema prevents old exports
from silently acquiring the new head.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_duel_v2_policy_recovery import load_games, split_games, write
from scripts.fit_duel_v2_ranking_auxiliary import policy_rows, rows_for, load_initial

KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')


def partitions(games, confirmation_seeds):
    fit, original_selection, original_seeds = split_games(games)
    confirmation = set(map(int, confirmation_seeds))
    if confirmation & set(original_seeds):
        raise ValueError('Previously consumed partitions overlap')
    available = [g for g in fit if g['seed'] not in confirmation]
    # Reserve seeds for every team, including quick/murk absent in old selection.
    held = set(original_seeds)
    for key in KEYS:
        seeds = sorted({g['seed'] for g in available if any(r.get('key') == key and 'pi' in r for r in g['rows'])})
        held.update(seeds[::5])
    train = [g for g in available if g['seed'] not in held]
    selection = original_selection + [g for g in available if g['seed'] in held]
    if {g['seed'] for g in train} & ({g['seed'] for g in selection} | confirmation):
        raise ValueError('Paired-root leakage')
    for key in KEYS:
        if not rows_for(train, key) or not rows_for(selection, key):
            raise ValueError(f'Missing per-team train/holdout coverage: {key}')
    return train, selection


def evaluate(net, rows):
    import torch
    from app.modules.card_game.rl.league_learning import tensors
    from app.modules.card_game.rl.league_schema import ACT_PLAY, ACT_ATTACK, ACT_END
    counts = dict(n=0, exact=0, confident_n=0, confident_exact=0, play_legal=0,
                  play_top=0, target_play=0, target_play_hit=0, attack_n=0,
                  attack_to_play=0, end_n=0, end_to_play=0, end_top=0)
    kl = entropy = ce = 0.
    net.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            x, c, mask = tensors([(r['x'], r['c']) for r in batch], 'cpu')
            logits, _ = net(x, c, mask)
            log_probs = logits.log_softmax(-1).cpu().numpy()
            for i, row in enumerate(batch):
                pi = np.asarray(row['pi'], dtype=np.float64)
                top = int(np.argmax(log_probs[i, :len(pi)])); target = int(np.argmax(pi))
                kind = np.asarray(row['c'])[:, 0].astype(int)
                positive = pi > 0
                h = -float(np.sum(pi[positive] * np.log(pi[positive])))
                loss = -float(np.sum(pi * log_probs[i, :len(pi)]))
                counts['n'] += 1; counts['exact'] += top == target
                sorted_pi = np.sort(pi)
                confident = len(pi) > 1 and sorted_pi[-1] - sorted_pi[-2] >= .05
                counts['confident_n'] += confident
                counts['confident_exact'] += confident and top == target
                if np.any(kind == ACT_PLAY):
                    counts['play_legal'] += 1
                    counts['play_top'] += kind[top] == ACT_PLAY
                    counts['target_play'] += kind[target] == ACT_PLAY
                    counts['target_play_hit'] += kind[target] == ACT_PLAY and top == target
                counts['end_top'] += kind[top] == ACT_END
                for name, label in (('attack', ACT_ATTACK), ('end', ACT_END)):
                    counts[name + '_n'] += kind[target] == label
                    counts[name + '_to_play'] += kind[target] == label and kind[top] == ACT_PLAY
                entropy += h; ce += loss; kl += loss - h
    net.train()
    counts = {k: int(v) for k, v in counts.items()}
    n = counts['n']
    return dict(**counts, cross_entropy=ce / n, target_entropy=entropy / n, kl=kl / n,
                confident_accuracy=counts['confident_exact'] / max(1, counts['confident_n']))


def fitted_gate(baseline, candidate):
    reasons = []
    if candidate['kl'] > .20:
        reasons.append('target_distribution_not_fit')
    if candidate['confident_n'] < 32 or candidate['confident_accuracy'] < .75:
        reasons.append('confident_action_not_fit')
    if candidate['exact'] <= baseline['exact']:
        reasons.append('exact_action_not_improved')
    if candidate['target_play'] < 8 or candidate['target_play_hit'] <= baseline['target_play_hit']:
        reasons.append('specific_play_not_improved_or_insufficient')
    for kind in ('attack', 'end'):
        if candidate[kind + '_n'] < 8:
            reasons.append(kind + '_preservation_insufficient')
        elif candidate[kind + '_to_play'] > baseline[kind + '_to_play']:
            reasons.append(kind + '_mislabel_regressed')
    if candidate['play_top'] == 0 or candidate['end_top'] >= .95 * candidate['n']:
        reasons.append('policy_collapsed')
    return dict(passed=not reasons, reasons=reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/episodes'))
    parser.add_argument('--initial', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/initial'))
    parser.add_argument('--confirmation-plan', type=Path, default=Path('artifacts/rl-evals/policy-recovery-20260924/calibration-plan.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--memorize', action='store_true', help='128 real training roots per team, 400 updates; capacity check only')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--passes', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1200 or not 1 <= args.passes <= 32:
        parser.error('Bounded CPU experiment: 1..1200 seconds, 1..32 passes')
    config = dict(schema='cross_five_residual_wdl_v2', keys=list(KEYS), learning_rate=.001,
                  passes=args.passes, batch=128, seed=20260927, seconds=args.seconds,
                  policy_target='full_saved_gumbel_pi', value_target='actual_terminal_wdl',
                  gates=dict(max_kl=.20, min_confident_accuracy=.75, confident_target_gap=.05,
                             exact_and_specific_play='strictly improve', nonplay_mislabels='must not increase'),
                  independent_strength=False, archived_rules=True, serving_approval=False)
    config['memorization_only'] = args.memorize
    if args.memorize:
        config.update(rows_per_team=128, updates_per_team=400)
    print(json.dumps(config), flush=True)
    if not args.run:
        return
    if args.output.exists():
        raise ValueError('Output directory must be new')
    import torch
    from app.modules.card_game.rl import residual_runtime as rt
    torch.set_num_threads(2)
    deadline = time.monotonic() + args.seconds
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    write(args.output / 'config.json', config)
    games = load_games(args.episodes)
    confirmation = json.loads(args.confirmation_plan.read_text())['confirmation_seeds']
    train, held = partitions(games, confirmation)
    write(args.output / 'partitions.json', dict(
        train_seeds=sorted({g['seed'] for g in train}),
        held_seeds=sorted({g['seed'] for g in held}), excluded_confirmation=sorted(confirmation),
        train_rows={key: len(rows_for(train, key)) for key in KEYS},
        held_rows={key: len(rows_for(held, key)) for key in KEYS}, independent=False))
    results = {}
    for key in KEYS:
        rows = rows_for(train, key); held_rows = rows_for(held, key)
        manifest = json.loads((args.initial / f'{key}.json').read_text())
        source_sha = sha256((args.initial / f'{key}.npz').read_bytes()).hexdigest()
        if source_sha != manifest['sha256']:
            raise ValueError('Archived source SHA mismatch')
        from app.modules.card_game.rl import cross_grounded
        from app.modules.card_game.rl.cross_lineup import all_features, candidate_names
        if (manifest.get('schema') != cross_grounded.SCHEMA
                or manifest.get('features') != all_features()
                or manifest.get('candidates') != candidate_names()
                or manifest.get('candidate_transform_sha256') != cross_grounded.fingerprint()):
            raise ValueError('Archived feature identity mismatch')
        if args.memorize:
            picked = np.random.default_rng(config['seed']).choice(len(rows), 128, replace=False)
            rows = [rows[int(i)] for i in picked]
            held_rows = rows
        baseline_net = load_initial(key, args.initial)
        expected = baseline_net.state_dict()
        with np.load(args.initial / f'{key}.npz', allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Archived tensor set mismatch')
            if any(arrays[name].shape != tuple(expected[name].shape)
                   or not np.isfinite(arrays[name]).all() for name in expected):
                raise ValueError('Archived tensor shape/nonfinite mismatch')
        baseline = evaluate(baseline_net, held_rows)
        torch.manual_seed(config['seed'])
        net = rt.create_network(64, 'cpu')
        opt = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
        rng = np.random.default_rng(config['seed'])
        seen = np.zeros(len(rows), dtype=np.int32)
        updates = 0; epochs = []; gradient_totals = dict(cand_net=0., state_net=0., score=0.)
        for epoch in range(400 if args.memorize else args.passes):
            for start in range(0, len(rows), config['batch']):
                if start == 0:
                    order = rng.permutation(len(rows))
                if time.monotonic() >= deadline:
                    write(args.output / 'result.json', dict(complete=False, reason='bounded_deadline', cells=results))
                    raise SystemExit('Bounded CPU deadline; incomplete, no approval')
                indexes = order[start:start + config['batch']]
                stats = rt.update(net, opt, [rows[int(i)] for i in indexes])
                seen[indexes] += 1; updates += 1
                for name in gradient_totals:
                    gradient_totals[name] += stats['policy_gradient_norms'].get(name, 0.)
            if not args.memorize or (epoch + 1) % 50 == 0:
                epoch_metrics = evaluate(net, held_rows)
                epochs.append(dict(epoch=epoch + 1, **epoch_metrics))
                print(key, epoch + 1, round(epoch_metrics['kl'], 4), round(epoch_metrics['confident_accuracy'], 3), flush=True)
        after = evaluate(net, held_rows)
        gate = fitted_gate(baseline, after)
        if args.memorize:
            passed = after['kl'] <= .05 and after['confident_accuracy'] >= .95
            gate = dict(passed=passed, reasons=[] if passed else ['real_batch_not_fit'], capacity_only=True)
        if not all(np.isfinite(v) and v > 0 for v in gradient_totals.values()):
            gate['passed'] = False; gate['reasons'].append('policy_gradient_missing')
        origin = dict(initialization='fresh_normal_scale_residual', archived_source_sha=source_sha,
                      archived_rule_hash=manifest['rule_hash'], archived_schema=manifest['schema'],
                      fitting_only=True, strength_verified=False, updates=updates)
        # Research numeric files may be inspected by inference but are not a
        # current-rule trained candidate: archive identity remains in origin.
        rt.export(net, args.output / 'models', key, manifest['build'], origin)
        numeric = rt.model(args.output / 'models', key)
        parity = 0.
        with torch.no_grad():
            for row in held_rows[:8]:
                x = torch.as_tensor(row['x'][None]); c = torch.as_tensor(row['c'][None])
                scores, _ = net(x, c, torch.ones(c.shape[:2], dtype=torch.bool))
                other = numeric.scores(row['x'], row['c'])
                np.testing.assert_allclose(scores[0].numpy(), other, atol=2e-5, rtol=2e-5)
                parity = max(parity, float(np.max(np.abs(scores[0].numpy() - other))))
        results[key] = dict(baseline=baseline, residual=after, gate=gate, epochs=epochs,
                            updates=updates, gradient_totals=gradient_totals, min_reuse=int(seen.min()),
                            max_reuse=int(seen.max()), numpy_max_error=parity)
        write(args.output / 'cells.json', results)
    result = dict(complete=True, offline_fit_passed=all(r['gate']['passed'] for r in results.values()),
                  cells=results, independent_strength=False, formal_training_allowed=False,
                  serving_approval=False, elapsed_seconds=args.seconds - (deadline - time.monotonic()))
    if args.memorize:
        result['memorization_passed'] = result.pop('offline_fit_passed')
    write(args.output / 'result.json', result)
    lines = ['# 本机残差策略学习核验', '',
             '只拟合已归档的完整训练对局，保留原规则身份。留出按成对根种子隔离，但已经用于研究，不能称为独立强度验收。', '',
             '| 队伍 | 原模型 KL | 新模型 KL | 明确目标动作命中 | 出牌目标的具体动作命中 | 拟合闸门 |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for key, r in results.items():
        m = r['residual']
        lines.append(f"| {key} | {r['baseline']['kl']:.3f} | {m['kl']:.3f} | {m['confident_exact']}/{m['confident_n']} | {m['target_play_hit']}/{m['target_play']} | {r['gate']} |")
    lines += ['', '正式长训和服务批准仍为 false；需要现行规则的新根、完整对局与同预算对照。', '']
    if args.memorize:
        lines.insert(2, '**本次仅记忆128个训练局面，表内命中来自训练样本；不包含留出或泛化结论。**')
    (args.output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
