#!/usr/bin/env python3
"""Calibrate the offline Zhenhong fit, then confirm on seeds withheld from selection.

The sweep is scored only on the original holdout. Confirmation seeds are removed
from the fit rows before that sweep and are scored once, after the choice is frozen.
The same frozen setting is then applied to the other four presets. No self-play.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_duel_v2_policy_recovery import (  # noqa: E402
    GATES, holdout_seeds, load_games, rows_for, split_games, write)


def choose_variant(measured, target):
    """Pick a predeclared fit. None when nothing lands inside the gate."""
    eligible = [row for row in measured
                if row['cross_entropy_drop'] >= GATES['cross_entropy_drop_nats']
                and abs(row['model_play_top1'] - target) <= GATES['play_top1_tolerance']
                and row['model_play_top1'] < GATES['collapse_fraction']
                and row['model_end_top1'] < GATES['collapse_fraction']]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row['cross_entropy_drop'], -abs(row['model_play_top1'] - target)))


def main():
    import torch
    from app.modules.card_game.rl.cross_runtime import create_network
    from app.modules.card_game.rl.league_learning import tensors
    from app.modules.card_game.rl.league_schema import ACT_END, ACT_PLAY
    from app.modules.card_game.rl.outcome_runtime import update
    from scripts.fit_duel_v2_policy_recovery import _kind

    episodes = Path('artifacts/rl-evals/five-cross-20260924/formal/episodes')
    initial = Path('artifacts/rl-evals/five-cross-20260924/formal/initial')
    out = Path('artifacts/rl-evals/policy-recovery-20260924')
    games = load_games(episodes)
    fit_games, selection_games, _ = split_games(games)
    confirmation_seeds = holdout_seeds((game['seed'] for game in fit_games), every=5)
    trainable = [game for game in fit_games if game['seed'] not in confirmation_seeds]
    confirmation_games = [game for game in fit_games if game['seed'] in confirmation_seeds]
    if set(confirmation_seeds) & {game['seed'] for game in selection_games}:
        raise ValueError('Confirmation seeds overlap the selection holdout')
    plan = dict(selection='original_holdout', confirmation_seeds=sorted(confirmation_seeds),
                grid=[dict(learning_rate=lr, passes=4) for lr in (1.2e-4, 1.5e-4, 1.8e-4, 2e-4)],
                rule='highest cross-entropy drop inside the play top-1 tolerance; otherwise no pass',
                gates=GATES, serving_export=False)
    write(out / 'calibration-plan.json', plan)

    def load_net(key):
        manifest = json.loads((initial / f'{key}.json').read_text(encoding='utf-8'))
        net = create_network(int(manifest['hidden']), 'cpu')
        own = net.state_dict()
        with np.load(initial / f'{key}.npz', allow_pickle=False) as arrays:
            for name in own:
                if name in arrays.files and own[name].shape == torch.from_numpy(arrays[name]).shape:
                    own[name] = torch.from_numpy(arrays[name].copy())
        net.load_state_dict(own)
        return net

    @torch.no_grad()
    def evaluate(net, rows):
        legal = [row for row in rows if 'pi' in row and np.any(_kind(row) == ACT_PLAY)]
        total = play = target = end = 0
        cross = uniform = 0.0
        for start in range(0, len(legal), 64):
            batch = legal[start:start + 64]
            xx, c, mask = tensors([(row['x'], row['c']) for row in batch], 'cpu')
            scores, _ = net(xx, c, mask)
            probs = scores.softmax(-1)
            for index, row in enumerate(batch):
                width = len(row['pi'])
                pred = probs[index, :width].cpu().numpy()
                kind = _kind(row)
                top = int(np.argmax(pred))
                total += 1
                play += int(kind[top] == ACT_PLAY)
                end += int(kind[top] == ACT_END)
                target += int(kind[int(np.argmax(row['pi']))] == ACT_PLAY)
                cross += float(-(row['pi'] * np.log(np.clip(pred, 1e-12, 1))).sum())
                uniform += float(np.log(width))
        return dict(play_legal=total, model_play_top1=play / total, target_play_top1=target / total,
                    model_end_top1=end / total, cross_entropy_drop=uniform / total - cross / total)

    def fit(key, lr, passes, train_rows):
        net = load_net(key)
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
        policy = [row for row in train_rows if 'pi' in row]
        rng = np.random.default_rng(20260924)
        for _ in range(passes):
            order = rng.permutation(len(policy))
            for start in range(0, len(policy), 128):
                batch = [policy[int(i)] for i in order[start:start + 128]]
                if len(batch) < 32:
                    continue
                update(net, opt, batch, self_imitation=0)
        return net

    zhenhong_train = rows_for(trainable, 'zhenhong')
    zhenhong_selection = rows_for(selection_games, 'zhenhong')
    measured = []
    for item in plan['grid']:
        net = fit('zhenhong', item['learning_rate'], item['passes'], zhenhong_train)
        metrics = evaluate(net, zhenhong_selection)
        metrics.update(item, key='zhenhong', split='selection')
        measured.append(metrics)
        write(out / 'calibration-selection.json', measured)
        print('selection', json.dumps(metrics, ensure_ascii=False), flush=True)
    target = measured[0]['target_play_top1']
    chosen = choose_variant(measured, target)
    write(out / 'calibration-choice.json', dict(chosen=chosen, target_play_top1=target, passed=chosen is not None))
    if chosen is None:
        print('no variant passed the selection gate')
        return
    confirmation = {}
    for key in ('zhenhong', 'starter', 'weave-rush', 'quick-rush', 'murk'):
        net = fit(key, chosen['learning_rate'], chosen['passes'], rows_for(trainable, key))
        confirmation[key] = evaluate(net, rows_for(confirmation_games, key))
        confirmation[key]['passed'] = choose_variant([confirmation[key]], confirmation[key]['target_play_top1']) is not None
        write(out / 'calibration-confirmation.json', confirmation)
        print(key, json.dumps(confirmation[key], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
