#!/usr/bin/env python3
"""Episode-weighted WDL audit; old consumed holdouts never grant readiness."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_duel_v2_policy_recovery import load_games, write
from scripts.check_duel_v2_residual_learning import partitions, KEYS
from app.modules.card_game.rl.archive_diagnostics import ArchivedDiagnosticModel
from app.modules.card_game.rl.residual_runtime import model as residual_model


def probability_metrics(probabilities, labels, weights=None):
    p = np.asarray(probabilities, dtype=np.float64); z = np.asarray(labels, dtype=np.int64)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) != len(z) or not len(z)
            or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1.)
            or np.any((z < 0) | (z > 2))):
        raise ValueError('Invalid WDL probability/terminal labels')
    w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != z.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Invalid episode weights')
    w /= w.sum(); actual = np.eye(3)[z]
    logloss = float(np.sum(w * -np.log(np.maximum(p[np.arange(len(z)), z], 1e-12))))
    brier = float(np.sum(w * np.sum((p - actual) ** 2, -1)))
    bins = []
    for low in np.arange(0, 1, .1):
        high = low + .1; take = (p[:, 2] >= low) & (p[:, 2] < high if high < .999 else p[:, 2] <= 1)
        if take.any():
            local = w[take] / w[take].sum()
            bins.append(dict(low=float(low), n=int(take.sum()), weight=float(w[take].sum()),
                             predicted_win=float(np.sum(local * p[take, 2])),
                             actual_win=float(np.sum(local * (z[take] == 2)))))
    return dict(log_loss=logloss, brier=brier,
                win_ece=float(sum(b['weight'] * abs(b['predicted_win'] - b['actual_win']) for b in bins)), bins=bins)


def score_games(policy, games, key):
    probs = []; labels = []; weights = []; rows_by_actor = {0: [], 1: []}; used_games = set(); seeds = set()
    for game in games:
        rows = [r for r in game['rows'] if r['key'] == key]
        if not rows:
            continue
        used_games.add(game['episode']); seeds.add(game['seed'])
        for row in rows:
            index = len(labels)
            probs.append(policy.wdl(row['x'], row.get('value_actor', 1)))
            labels.append(int(row['z']) + 1); weights.append(1 / len(rows))
            rows_by_actor[int(row.get('value_actor', 1))].append(index)
    result = probability_metrics(probs, labels, weights)
    result.update(normal_games=len(used_games), paired_seeds=len(seeds), observations=len(labels))
    result['by_actor'] = {str(actor): probability_metrics(np.asarray(probs)[index], np.asarray(labels)[index],
                                                        np.asarray(weights)[index])
                          for actor, index in rows_by_actor.items() if index}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/episodes'))
    parser.add_argument('--initial', type=Path, default=Path('artifacts/rl-evals/five-cross-20260924/formal/initial'))
    parser.add_argument('--residual', type=Path, default=Path('artifacts/rl-evals/residual-recovery-20260927/offline-v2/models'))
    parser.add_argument('--confirmation-plan', type=Path, default=Path('artifacts/rl-evals/policy-recovery-20260924/calibration-plan.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New output required')
    games = load_games(args.episodes)
    confirmation = json.loads(args.confirmation_plan.read_text())['confirmation_seeds']
    train, held = partitions(games, confirmation)
    cells = {}
    for key in KEYS:
        old = ArchivedDiagnosticModel(args.initial, key); new = residual_model(args.residual, key)
        cells[key] = dict(initial=score_games(old, held, key), residual=score_games(new, held, key),
                          model_sha256=new.version, historical_rule_hash=old.historical_rule_hash,
                          source_actual_games=True, deployment_value_claim=False)
        write(args.output / 'cells.json', cells)
    result = dict(complete=True, passed=False, heldout=True, untouched=False,
                  reason='consumed_historical_rule_holdout_cannot_approve_current_search_teacher',
                  policy_afterwards='archive_recorded_policy_not_new_policy_unbiased_value', cells=cells)
    write(args.output / 'result.json', result)
    lines = ['# WDL 校准审计', '',
             '按完整对局等权，回合长的局不会因观察更多而获得更高权重。根种子成对隔离；已消费归档留出不批准现行搜索。', '',
             '| 队伍 | 对局／根种子 | 初始Brier | 残差Brier | 初始对数损失 | 残差对数损失 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, cell in cells.items():
        old, new = cell['initial'], cell['residual']
        lines.append(f"| {key} | {new['normal_games']}/{new['paired_seeds']} | {old['brier']:.3f} | {new['brier']:.3f} | {old['log_loss']:.3f} | {new['log_loss']:.3f} |")
    lines += ['', '三类均匀预测的Brier为2/3，对数损失为ln(3)。这些对照只判断档案分布中的预测；旧策略轨迹终局不是新策略后续的无偏胜率。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({key: {mode: {name: cell[mode][name] for name in ('brier', 'log_loss')}
                              for mode in ('initial', 'residual')} for key, cell in cells.items()}))


if __name__ == '__main__':
    main()
