#!/usr/bin/env python3
"""Offline diagnosis of the play-rate jump. Does not start self-play or export models.

Selection uses the original holdout. The confirmation seeds already scored on
2026-09-24 are reported only as a consumed set; they are not used to choose a
learning rate, pass count, or structure.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.cross_lineup import CARD_IDS, all_features
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY, ACT_ULTIMATE
from app.modules.card_game.rl.policy_margin import margin_bins, play_margin
from scripts.fit_duel_v2_policy_recovery import holdout_seeds, load_games, split_games, write

KIND_NAME = {ACT_END: 'end_turn', ACT_ATTACK: 'attack', ACT_PLAY: 'play_card', ACT_ULTIMATE: 'ultimate'}
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
LEARNING_RATES = (1e-4, 1.5e-4, 1.8e-4, 2e-4, 3e-4)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def kind_vector(row):
    return np.asarray(row['c'], dtype=np.int64)[:, 0]


def card_name(row, index):
    raw = int(np.asarray(row['c'])[index, 2])
    if raw < 0 or raw >= len(CARD_IDS):
        return ''
    return CARD_IDS[raw]


def audit_rows(games):
    problems = []
    checked = 0
    for game in games:
        for row in game['rows']:
            if 'pi' not in row:
                if int(row['z']) not in (-1, 0, 1) or not np.isfinite(row['x']).all():
                    problems.append(dict(episode=game['episode'], error='value row'))
                continue
            checked += 1
            kind = kind_vector(row)
            pi = np.asarray(row['pi'])
            chosen = int(row['selected'])
            if (len(pi) != len(kind) or not np.isfinite(pi).all() or np.any(pi < 0)
                    or not np.isclose(pi.sum(), 1) or not 0 <= chosen < len(kind)
                    or int(row['z']) not in (-1, 0, 1) or not np.isfinite(row['c']).all()):
                if len(problems) < 5:
                    problems.append(dict(episode=game['episode'], step=row.get('step'), error='policy row'))
    return dict(policy_rows_checked=checked, problems=problems)


def group_summary(games):
    seeds = sorted({game['seed'] for game in games})
    seats = defaultdict(set)
    rows = defaultdict(int)
    for game in games:
        seats[game['seed']].add(game['first'])
        for row in game['rows']:
            if 'pi' in row:
                rows[row['key']] += 1
    return dict(episodes=len(games), seeds=seeds, seed_count=len(seeds),
                both_seats=sum(len(value) == 2 for value in seats.values()),
                one_seat=sum(len(value) == 1 for value in seats.values()),
                policy_rows=dict(rows))


def attach(games):
    tagged = []
    for game in games:
        for row in game['rows']:
            if 'pi' not in row:
                continue
            row['episode_seed'] = game['seed']
            row['episode_first'] = game['first']
            row['seat_is_first'] = row['side'] == game['first']
            tagged.append(row)
    return tagged


def score_rows(net, rows):
    import torch
    from app.modules.card_game.rl.league_learning import tensors
    records = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            legal = [row for row in batch if np.any(kind_vector(row) == ACT_PLAY)]
            if not legal:
                continue
            xx, c, mask = tensors([(row['x'], row['c']) for row in legal], 'cpu')
            scores, _ = net(xx, c, mask)
            probabilities = scores.softmax(-1).cpu().numpy()
            logits = scores.cpu().numpy()
            for index, row in enumerate(legal):
                width = len(row['pi'])
                logit = logits[index, :width]
                prob = probabilities[index, :width]
                kind = kind_vector(row)
                top = int(np.argmax(logit))
                second = int(np.argsort(logit)[-2])
                target = int(np.argmax(row['pi']))
                margin = play_margin(logit, kind, ACT_PLAY)
                records.append(dict(
                    margin=margin,
                    play_top=bool(kind[top] == ACT_PLAY),
                    target_play=bool(kind[target] == ACT_PLAY),
                    same_kind=bool(kind[top] == kind[target]),
                    same_card=bool(kind[top] == ACT_PLAY and kind[target] == ACT_PLAY and card_name(row, top) == card_name(row, target) and card_name(row, top) != ''),
                    top_gap=float(logit[top] - logit[second]),
                    play_mass=float(prob[kind == ACT_PLAY].sum()),
                    target_mass=float(np.asarray(row['pi'])[kind == ACT_PLAY].sum()),
                    cross_entropy=float(-(row['pi'] * np.log(np.clip(prob, 1e-12, 1))).sum()),
                    model_entropy=float(-(prob * np.log(np.clip(prob, 1e-12, 1))).sum()),
                    target_entropy=float(-(row['pi'] * np.log(np.clip(row['pi'], 1e-12, 1))).sum()),
                    uniform_entropy=float(np.log(width)),
                    legal_plays=int(np.sum(kind == ACT_PLAY)),
                    phase=str(row.get('phase')),
                    first=bool(row['seat_is_first']),
                    target_kind=KIND_NAME.get(int(kind[target]), str(int(kind[target]))),
                    model_kind=KIND_NAME.get(int(kind[top]), str(int(kind[top]))),
                    target_card=card_name(row, target),
                ))
    return records


def aggregate(records):
    if not records:
        return dict(n=0)
    margins = [row['margin'] for row in records if row['margin'] is not None]
    def rate(key):
        return sum(bool(row[key]) for row in records) / len(records)
    near = [row for row in records if row['margin'] is not None and abs(row['margin']) < 0.01]
    flipped_by_near = sum(1 for row in near if row['margin'] < 0)
    summary = dict(
        n=len(records),
        play_top1=rate('play_top'),
        target_play_top1=rate('target_play'),
        same_kind=rate('same_kind'),
        same_card_among_both_play=(
            sum(row['same_card'] for row in records) / max(1, sum(row['target_play'] and row['play_top'] for row in records))),
        both_play=sum(row['target_play'] and row['play_top'] for row in records),
        mean_cross_entropy=float(np.mean([row['cross_entropy'] for row in records])),
        mean_model_entropy=float(np.mean([row['model_entropy'] for row in records])),
        mean_target_entropy=float(np.mean([row['target_entropy'] for row in records])),
        mean_uniform_entropy=float(np.mean([row['uniform_entropy'] for row in records])),
        mean_play_mass=float(np.mean([row['play_mass'] for row in records])),
        mean_target_play_mass=float(np.mean([row['target_mass'] for row in records])),
        mean_top_gap=float(np.mean([row['top_gap'] for row in records])),
        median_margin=float(np.median(margins)) if margins else None,
        margins=margin_bins(margins),
        near_zero_abs_lt_0_01=len(near),
        negative_near_zero=flipped_by_near,
    )
    summary['strata'] = stratify(records)
    return summary


def stratify(records):
    groups = defaultdict(list)
    for row in records:
        play_bucket = '1' if row['legal_plays'] == 1 else '2-3' if row['legal_plays'] <= 3 else '4+'
        groups[('first' if row['first'] else 'second', row['phase'], play_bucket, row['target_kind'])].append(row)
    rows = []
    for key, items in sorted(groups.items()):
        rows.append(dict(
            seat=key[0], phase=key[1], legal_plays=key[2], target_kind=key[3], n=len(items),
            play_top1=sum(item['play_top'] for item in items),
            mean_margin=float(np.mean([item['margin'] for item in items if item['margin'] is not None])),
            mean_top_gap=float(np.mean([item['top_gap'] for item in items])),
        ))
    return rows


def fit_policy(key, learning_rate, passes, train_rows, initial):
    import torch
    from app.modules.card_game.rl.cross_runtime import create_network
    from app.modules.card_game.rl.outcome_runtime import update
    manifest = json.loads((initial / f'{key}.json').read_text(encoding='utf-8'))
    net = create_network(int(manifest['hidden']), 'cpu')
    own = net.state_dict()
    with np.load(initial / f'{key}.npz', allow_pickle=False) as arrays:
        for name in own:
            if name in arrays.files:
                value = torch.from_numpy(arrays[name].copy())
                if own[name].shape != value.shape:
                    raise ValueError(f'{key} {name} shape mismatch')
                own[name] = value
    net.load_state_dict(own)
    opt = torch.optim.Adam([parameter for parameter in net.parameters() if parameter.requires_grad], lr=learning_rate)
    policy = [row for row in train_rows if 'pi' in row]
    seen = np.zeros(len(policy), dtype=np.int32)
    losses = []
    rng = np.random.default_rng(20260924)
    for _ in range(passes):
        order = rng.permutation(len(policy))
        for start in range(0, len(policy), 128):
            batch_index = order[start:start + 128]
            if len(batch_index) < 32:
                continue
            seen[batch_index] += 1
            stats = update(net, opt, [policy[int(i)] for i in batch_index], self_imitation=0)
            losses.append(dict(policy_loss=stats['policy_loss'], value_loss=stats['value_loss'],
                               policy_samples=stats['policy_samples'], samples=stats['samples']))
    reuse = dict(rows=int(seen.size), min=int(seen.min()) if seen.size else 0, max=int(seen.max()) if seen.size else 0,
                 mean=float(seen.mean()) if seen.size else 0, never=int(np.sum(seen == 0)))
    return net, losses, reuse


def gradient_probe(net, mixed_rows):
    import torch
    from app.modules.card_game.rl.league_learning import tensors
    selected = [row for row in mixed_rows if 'pi' in row]
    device = 'cpu'
    x = torch.as_tensor(np.stack([row['x'] for row in mixed_rows]), device=device)
    hidden = net.state_net(x)
    context = torch.tensor([[row.get('value_actor', 1)] for row in mixed_rows], dtype=hidden.dtype)
    value_logits = net.wdl(torch.cat((hidden, context), -1))
    target_z = torch.tensor([int(row['z']) + 1 for row in mixed_rows])
    value_loss = torch.nn.functional.cross_entropy(value_logits, target_z)
    xx, c, mask = tensors([(row['x'], row['c']) for row in selected], device)
    policy, _ = net(xx, c, mask)
    target = torch.zeros_like(policy)
    for index, row in enumerate(selected):
        target[index, :len(row['pi'])] = torch.as_tensor(row['pi'])
    policy_loss = -(target * policy.log_softmax(-1).masked_fill(~mask, 0)).sum(-1).mean()
    names = [name for name, parameter in net.named_parameters() if name.startswith('state_net') and parameter.requires_grad]

    def capture(loss):
        net.zero_grad()
        loss.backward(retain_graph=True)
        vectors = {}
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            vectors[name] = parameter.grad.detach().flatten().clone()
        return vectors

    policy_grad = capture(policy_loss)
    value_grad = capture(value_loss)
    shared_policy = torch.cat([policy_grad[name] for name in names if name in policy_grad])
    shared_value = torch.cat([value_grad[name] for name in names if name in value_grad])
    cosine = float(torch.nn.functional.cosine_similarity(shared_policy, shared_value, dim=0))
    norms = {name: float(vector.norm()) for name, vector in policy_grad.items()
             if name.startswith(('state_net', 'cand_net', 'score'))}
    return dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                policy_samples=len(selected), mixed_samples=len(mixed_rows),
                state_gradient_cosine=cosine, policy_gradient_norms=norms,
                policy_gradient_finite=all(np.isfinite(value) and value > 0 for value in norms.values()))


def numpy_match(net, rows):
    import torch
    mismatches = 0
    checked = 0
    example = None
    weights = {name: tensor.detach().cpu().numpy() for name, tensor in net.named_parameters()}
    # Buffers such as actor indices are not parameters. Score through the module and a NumPy twin.
    from app.modules.card_game.rl.cross_runtime import CrossModel
    # Identity may reject the archived manifest. Compare the module to its own exported arithmetic.
    for row in rows[:8]:
        with torch.no_grad():
            xx = torch.as_tensor(row['x'][None])
            cc = torch.as_tensor(row['c'][None])
            mask = torch.ones((1, len(row['c'])), dtype=torch.bool)
            torch_scores, _ = net(xx, cc, mask)
        torch_scores = torch_scores[0, :len(row['c'])].numpy()
        hidden = np.maximum(weights['state_net.0.weight'] @ row['x'] + weights['state_net.0.bias'], 0)
        hidden = np.maximum(weights['state_net.2.weight'] @ hidden + weights['state_net.2.bias'], 0)
        from app.modules.card_game.rl.cross_grounded import transform
        cand = transform(row['x'], row['c'])
        cand_h = np.maximum(cand @ weights['cand_net.0.weight'].T + weights['cand_net.0.bias'], 0)
        cand_h = np.maximum(cand_h @ weights['cand_net.2.weight'].T + weights['cand_net.2.bias'], 0)
        numpy_scores = cand_h @ hidden / np.sqrt(np.float32(hidden.shape[0])) + cand_h @ weights['score.weight'].reshape(-1) + weights['score.bias'].reshape(-1)
        checked += 1
        if not np.allclose(torch_scores, numpy_scores, atol=2e-4, rtol=2e-4):
            mismatches += 1
            if example is None:
                example = dict(max_abs=float(np.max(np.abs(torch_scores - numpy_scores))))
    try:
        CrossModel(str(Path('artifacts/rl-evals/five-cross-20260924/formal/initial')), 'zhenhong')
        identity_ok = True
    except ValueError as exc:
        identity_ok = False
        example = example or {}
        example['identity'] = str(exc)
    return dict(checked=checked, mismatches=mismatches, example=example, archived_identity_loads=identity_ok)


def main():
    episodes = Path('artifacts/rl-evals/five-cross-20260924/formal/episodes')
    initial = Path('artifacts/rl-evals/five-cross-20260924/formal/initial')
    out = Path('artifacts/rl-evals/policy-recovery-20260924/offline-diagnosis')
    out.mkdir(parents=True, exist_ok=True)
    games = load_games(episodes)
    fit_games, selection_games, selection_seeds = split_games(games)
    confirmation_seeds = holdout_seeds((game['seed'] for game in fit_games), every=5)
    saved = json.loads(Path('artifacts/rl-evals/policy-recovery-20260924/calibration-plan.json').read_text())
    if sorted(confirmation_seeds) != sorted(saved['confirmation_seeds']):
        raise SystemExit('Confirmation seed list does not match the already used calibration plan')
    trainable = [game for game in fit_games if game['seed'] not in confirmation_seeds]
    confirmation = [game for game in fit_games if game['seed'] in confirmation_seeds]
    overlap = set(selection_seeds) & confirmation_seeds
    write(out / 'config.json', dict(
        question='why cross-entropy improves while argmax play rate jumps',
        selection_split='original holdout from fit_duel_v2_policy_recovery.split_games every=5',
        confirmation='already used on 2026-09-24; not used to choose anything in this diagnosis',
        learning_rates=list(LEARNING_RATES), passes=4, seed=20260924, serving_export=False))
    write(out / 'source-sha.json', dict(files={
        str(path): sha256_file(ROOT / path) for path in (
            'scripts/calibrate_duel_v2_policy_fit.py', 'scripts/fit_duel_v2_policy_recovery.py',
            'app/modules/card_game/rl/outcome_runtime.py', 'app/modules/card_game/rl/cross_runtime.py',
            'app/modules/card_game/rl/cross_grounded.py')},
        initial_zhenhong=sha256_file(initial / 'zhenhong.npz')))
    write(out / 'split-audit.json', dict(
        fit=group_summary(trainable), selection=group_summary(selection_games),
        used_confirmation=group_summary(confirmation),
        seed_overlap=dict(fit_selection=len(set(game['seed'] for game in trainable) & set(selection_seeds)),
                          fit_confirmation=len(set(game['seed'] for game in trainable) & confirmation_seeds),
                          selection_confirmation=len(overlap)),
        rows=audit_rows(games)))
    parity_net, _, _ = fit_policy('zhenhong', 1e-4, 0, [], initial)
    write(out / 'parity.json', numpy_match(parity_net, attach(selection_games)))

    summaries = {}
    probes = {}
    zhenhong_train = [row for row in attach(trainable) if row['key'] == 'zhenhong']
    for rate in LEARNING_RATES:
        net, losses, reuse = fit_policy('zhenhong', rate, 4, zhenhong_train, initial)
        records = score_rows(net, [row for row in attach(selection_games) if row['key'] == 'zhenhong'])
        summaries[f'zhenhong-{rate}'] = dict(aggregate=aggregate(records), reuse=reuse,
                                             mean_policy_loss=float(np.mean([item['policy_loss'] for item in losses])),
                                             mean_value_loss=float(np.mean([item['value_loss'] for item in losses])) if losses else None)
        if rate == 1.8e-4:
            mixed = []
            for game in trainable:
                for row in game['rows']:
                    if row['key'] == 'zhenhong':
                        mixed.append(row)
                    if len(mixed) == 128:
                        break
                if len(mixed) == 128:
                    break
            probes['zhenhong_after_1.8e-4'] = gradient_probe(net, mixed)
        print(rate, summaries[f'zhenhong-{rate}']['aggregate']['play_top1'],
              summaries[f'zhenhong-{rate}']['aggregate']['mean_cross_entropy'],
              summaries[f'zhenhong-{rate}']['aggregate']['median_margin'], flush=True)
        write(out / 'margin-by-lr.json', summaries)
    initial_net, _, _ = fit_policy('zhenhong', 1e-4, 0, [], initial)
    initial_rows = []
    for game in trainable:
        for row in game['rows']:
            if row['key'] == 'zhenhong':
                initial_rows.append(row)
            if len(initial_rows) == 128:
                break
        if len(initial_rows) == 128:
            break
    probes['zhenhong_initial'] = gradient_probe(initial_net, initial_rows)
    for key in KEYS:
        if key == 'zhenhong':
            continue
        train_rows = [row for row in attach(trainable) if row['key'] == key]
        net, losses, reuse = fit_policy(key, 1.8e-4, 4, train_rows, initial)
        selection_rows = [row for row in attach(selection_games) if row['key'] == key]
        scored = selection_rows or train_rows
        records = score_rows(net, scored)
        summaries[f'{key}-0.00018'] = dict(
            aggregate=aggregate(records), reuse=reuse, scored_split='selection' if selection_rows else 'fit_in_sample',
            mean_policy_loss=float(np.mean([item['policy_loss'] for item in losses])) if losses else None)
        print(key, summaries[f'{key}-0.00018']['aggregate'].get('play_top1'), flush=True)
        write(out / 'margin-by-lr.json', summaries)
    write(out / 'gradient-signal.json', probes)
    print('wrote', out)


if __name__ == '__main__':
    main()
