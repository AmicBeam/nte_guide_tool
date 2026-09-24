#!/usr/bin/env python3
"""Same-data ranking-loss ablation on archived five-cross episodes.

Coefficient 0 is the old optimizer. Confirmation seeds already viewed on
2026-09-24 stay out of model selection. No new games and no serving export.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.cross_lineup import CARD_IDS
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY, ACT_ULTIMATE
from app.modules.card_game.rl.outcome_runtime import DEFAULT_RANKING_CONFIDENCE
from scripts.fit_duel_v2_policy_recovery import holdout_seeds, load_games, split_games, write

KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
SELECTION_KEYS = ('starter', 'weave-rush', 'zhenhong')
KIND_NAME = {ACT_END: 'end_turn', ACT_ATTACK: 'attack', ACT_PLAY: 'play_card', ACT_ULTIMATE: 'ultimate'}
COEFFICIENTS = (0.0, 0.05, 0.1)
LEARNING_RATE = 1.5e-4
PASSES = 4
BATCH = 128
MINIMUM_BATCH = 32
SEED = 20260924
COLLAPSE = 0.95


def kind_vector(row):
    return np.asarray(row['c'], dtype=np.int64)[:, 0]


def card_name(row, index):
    raw = int(np.asarray(row['c'])[index, 2])
    if raw < 0 or raw >= len(CARD_IDS):
        return ''
    return CARD_IDS[raw]


def policy_rows(rows):
    kept = []
    for row in rows:
        if 'pi' not in row or row.get('truncated') or row.get('complete') is False:
            continue
        kept.append(row)
    return kept


def partitions(games, confirmation_seeds=None):
    fit, selection, selection_seeds = split_games(games)
    if confirmation_seeds is None:
        confirmation_seeds = holdout_seeds((game['seed'] for game in fit), every=5)
    confirmation_seeds = frozenset(int(seed) for seed in confirmation_seeds)
    overlap = set(selection_seeds) & confirmation_seeds
    if overlap:
        raise ValueError('Confirmation seeds overlap the selection holdout')
    train = [game for game in fit if game['seed'] not in confirmation_seeds]
    confirmation = [game for game in fit if game['seed'] in confirmation_seeds]
    if set(game['seed'] for game in train) & set(game['seed'] for game in selection):
        raise ValueError('A paired seed crossed the fit boundary')
    if set(game['seed'] for game in train) & confirmation_seeds:
        raise ValueError('Confirmation seeds entered the fit set')
    return dict(train=train, selection=selection, confirmation=confirmation,
                selection_seeds=tuple(sorted(selection_seeds)),
                confirmation_seeds=tuple(sorted(confirmation_seeds)))


def rows_for(games, key):
    return policy_rows(row for game in games for row in game['rows'] if row.get('key') == key)


def score_policy_rows(net, rows):
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
                target = int(np.argmax(row['pi']))
                records.append(dict(
                    exact_top=bool(top == target),
                    same_kind=bool(kind[top] == kind[target]),
                    same_card=bool(kind[top] == ACT_PLAY and kind[target] == ACT_PLAY
                                   and card_name(row, top) == card_name(row, target)
                                   and card_name(row, top) != ''),
                    category_agreement=bool((kind[top] == ACT_PLAY) == (kind[target] == ACT_PLAY)),
                    play_top=bool(kind[top] == ACT_PLAY),
                    target_play=bool(kind[target] == ACT_PLAY),
                    target_kind=KIND_NAME.get(int(kind[target]), str(int(kind[target]))),
                    model_kind=KIND_NAME.get(int(kind[top]), str(int(kind[top]))),
                    cross_entropy=float(-(row['pi'] * np.log(np.clip(prob, 1e-12, 1))).sum()),
                    uniform_entropy=float(np.log(width)),
                ))
    return records


def summarize(records):
    n = len(records)
    if not n:
        return dict(n=0, attack_n=0, end_n=0, target_play_n=0, exact_top=0, same_card=0,
                    category_agreement=0, attack_called_play=0, end_called_play=0,
                    play_top=0, end_top=0, cross_entropy=None, collapse=None)
    attack = [row for row in records if row['target_kind'] == 'attack']
    end = [row for row in records if row['target_kind'] == 'end_turn']
    target_play = [row for row in records if row['target_play']]
    play_top = sum(row['play_top'] for row in records)
    end_top = sum(row['model_kind'] == 'end_turn' for row in records)
    return dict(
        n=n,
        attack_n=len(attack),
        end_n=len(end),
        target_play_n=len(target_play),
        exact_top=sum(row['exact_top'] for row in records),
        same_kind=sum(row['same_kind'] for row in records),
        same_card=sum(row['same_card'] for row in records),
        category_agreement=sum(row['category_agreement'] for row in records),
        attack_called_play=sum(row['model_kind'] == 'play_card' for row in attack),
        end_called_play=sum(row['model_kind'] == 'play_card' for row in end),
        play_top=play_top,
        end_top=end_top,
        never_play=play_top == 0,
        always_play=play_top >= COLLAPSE * n,
        always_end=end_top >= COLLAPSE * n,
        cross_entropy=float(np.mean([row['cross_entropy'] for row in records])),
        uniform_entropy=float(np.mean([row['uniform_entropy'] for row in records])),
    )


def ranking_beats(baseline, ranked):
    """True only when identity and non-play preservation both move the right way."""
    if not baseline['n'] or baseline['n'] != ranked['n']:
        return False
    if min(baseline['attack_n'], baseline['end_n'], baseline['target_play_n']) == 0:
        return False
    if ranked['always_play'] or ranked['always_end'] or ranked['never_play']:
        return False
    return (ranked['exact_top'] > baseline['exact_top']
            and ranked['same_card'] > baseline['same_card']
            and ranked['category_agreement'] > baseline['category_agreement']
            and ranked['attack_called_play'] < baseline['attack_called_play']
            and ranked['end_called_play'] < baseline['end_called_play'])


def scheme_passes(by_coefficient):
    for coefficient, flags in by_coefficient.items():
        if coefficient == 0:
            continue
        if all(bool(flags.get(key)) for key in SELECTION_KEYS):
            return float(coefficient)
    return None


def config_payload():
    return dict(
        question='Does a confident Gumbel-top ranking term raise exact action/card hits without turning attack/end into play?',
        pass_rule=dict(
            exact_top='strictly more than coefficient 0 on the same legal-play selection rows',
            same_card='strictly more target-play rows whose argmax card name matches',
            category_agreement='strictly more play-versus-nonplay agreement',
            attack_called_play='strictly fewer than coefficient 0',
            end_called_play='strictly fewer than coefficient 0',
            collapse='never-play, always-play, or always-end fails',
            eligible_teams=list(SELECTION_KEYS),
            in_sample_teams=['quick-rush', 'murk'],
            in_sample_cannot_pass=True,
            confirmation_scored=False,
            independent_confirmation='no unused independent confirmation roots in the archive',
        ),
        ranking=dict(
            coefficients=list(COEFFICIENTS),
            confidence=DEFAULT_RANKING_CONFIDENCE,
            competitor='strongest remaining legal logit, including other cards and non-play kinds',
            play_bonus=False,
            truncated_rows=False,
            value_labels='completed terminal WDL only',
        ),
        learning_rate=LEARNING_RATE, passes=PASSES, batch=BATCH, minimum_batch=MINIMUM_BATCH,
        seed=SEED, optimizer='Adam', self_imitation=0,
        selection_split='original holdout from fit_duel_v2_policy_recovery.split_games every=5',
        confirmation='already used on 2026-09-24; excluded from fit and selection',
        serving_export=False, formal_run=False, cuda=False, exploratory=True,
    )


def load_initial(key, initial):
    import torch
    from app.modules.card_game.rl.cross_runtime import create_network
    manifest = json.loads((initial / f'{key}.json').read_text(encoding='utf-8'))
    net = create_network(int(manifest['hidden']), 'cpu')
    own = net.state_dict()
    with np.load(initial / f'{key}.npz', allow_pickle=False) as arrays:
        for name in own:
            if name not in arrays.files:
                continue
            value = torch.from_numpy(arrays[name].copy())
            if own[name].shape != value.shape:
                raise ValueError(f'{key} {name} shape mismatch')
            own[name] = value
    net.load_state_dict(own)
    return net


def fit_key(key, coefficient, train_rows, initial):
    import torch
    from app.modules.card_game.rl.outcome_runtime import update
    net = load_initial(key, initial)
    opt = torch.optim.Adam([parameter for parameter in net.parameters() if parameter.requires_grad], lr=LEARNING_RATE)
    policy = policy_rows(train_rows)
    seen = np.zeros(len(policy), dtype=np.int32)
    losses = []
    rng = np.random.default_rng(SEED)
    for _ in range(PASSES):
        order = rng.permutation(len(policy))
        for start in range(0, len(policy), BATCH):
            batch_index = order[start:start + BATCH]
            if len(batch_index) < MINIMUM_BATCH:
                continue
            seen[batch_index] += 1
            stats = update(net, opt, [policy[int(index)] for index in batch_index],
                           self_imitation=0, ranking_coefficient=coefficient,
                           ranking_confidence=DEFAULT_RANKING_CONFIDENCE)
            losses.append(dict(policy_loss=stats['policy_loss'], value_loss=stats['value_loss'],
                               ranking_loss=stats['ranking_loss'], ranking_samples=stats['ranking_samples'],
                               ranking_excluded=stats['ranking_excluded'],
                               policy_samples=stats['policy_samples'],
                               policy_gradient_norms={
                                   name: stats['policy_gradient_norms'].get(name)
                                   for name in ('cand_net', 'score', 'state_net')
                               }))
    reuse = dict(rows=int(seen.size), min=int(seen.min()) if seen.size else 0,
                 max=int(seen.max()) if seen.size else 0,
                 mean=float(seen.mean()) if seen.size else 0, never=int(np.sum(seen == 0)))
    grad = {}
    for name in ('cand_net', 'score', 'state_net'):
        values = [item['policy_gradient_norms'][name] for item in losses
                  if item['policy_gradient_norms'][name] is not None]
        grad[name] = float(np.mean(values)) if values else None
    return net, losses, reuse, grad


def pack(records, losses, reuse, grad):
    counts = summarize(records)
    return dict(
        counts=counts, reuse=reuse, gradient_norms=grad,
        mean_policy_loss=float(np.mean([item['policy_loss'] for item in losses])) if losses else None,
        mean_value_loss=float(np.mean([item['value_loss'] for item in losses])) if losses else None,
        mean_ranking_loss=float(np.mean([item['ranking_loss'] for item in losses])) if losses else None,
        mean_ranking_samples=float(np.mean([item['ranking_samples'] for item in losses])) if losses else None,
    )


def render_report(payload):
    lines = [
        '# Offline ranking auxiliary ablation',
        '',
        'Same archived five-cross training episodes. No new games, no serving export, no CUDA, no formal training. Confirmation seeds already viewed on 2026-09-24 are excluded from fit and selection. The archive has no unused independent confirmation roots, so this result is exploratory.',
        '',
        '## Pass rule',
        '',
        'Frozen in config.json before the first optimizer step. Learning rate 1.5e-4, four policy passes, batch 128, drop tails shorter than 32, Adam, self-imitation 0. The ranking term compares a confident Gumbel pi top action to the strongest remaining legal logit, including other card identities and non-play kinds. Coefficient 0 is the old optimizer.',
        '',
        'A nonzero coefficient may be recommended only if starter, weave-rush, and zhenhong all improve exact top-action hits, same-card hits, and play/non-play category agreement, while also reducing attack-called-play and end-called-play. Never-play, always-play, or always-end fails. quick-rush and murk have no original selection rows and cannot pass.',
        '',
        '## Result',
        '',
        'Recommend for CUDA / formal training: %s' % ('no' if payload['chosen'] is None else payload['chosen']),
        'Verdict: %s' % payload['verdict'],
        '',
    ]
    for key in KEYS:
        eligible = key in SELECTION_KEYS
        lines.append('### %s%s' % (key, ' (selection)' if eligible else ' (fit in-sample, cannot pass)'))
        lines.append('')
        for coefficient in COEFFICIENTS:
            counts = payload['cells'][key][str(coefficient)]['counts']
            lines.append(
                '- coefficient %s: n=%s exact_top=%s/%s same_card=%s/%s category=%s/%s attack->play=%s/%s end->play=%s/%s play_top=%s/%s CE=%s'
                % (coefficient, counts['n'], counts['exact_top'], counts['n'], counts['same_card'],
                   counts['target_play_n'], counts['category_agreement'], counts['n'],
                   counts['attack_called_play'], counts['attack_n'], counts['end_called_play'],
                   counts['end_n'], counts['play_top'], counts['n'], counts['cross_entropy'])
            )
        lines.append('')
    lines.extend([
        '## Evidence limits',
        '',
        '- Selection is the original paired-root holdout, not a new independent confirmation set.',
        '- Confirmation seeds were already viewed on 2026-09-24 and were not scored here.',
        '- No new games, no serving export, and no formal training-entry change.',
        '- Value labels remain completed terminal WDL; truncated rows are excluded from ranking.',
        '',
    ])
    return chr(10).join(lines) + chr(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path,
                        default=Path('artifacts/rl-evals/five-cross-20260924/formal/episodes'))
    parser.add_argument('--initial', type=Path,
                        default=Path('artifacts/rl-evals/five-cross-20260924/formal/initial'))
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/rl-evals/policy-recovery-20260924/offline-diagnosis/ranking-auxiliary'))
    parser.add_argument('--confirmation-plan', type=Path,
                        default=Path('artifacts/rl-evals/policy-recovery-20260924/calibration-plan.json'))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = config_payload()
    write(out / 'config.json', config)
    games = load_games(args.episodes)
    saved = json.loads(args.confirmation_plan.read_text(encoding='utf-8'))
    parts = partitions(games, saved['confirmation_seeds'])
    if tuple(parts['confirmation_seeds']) != tuple(sorted(saved['confirmation_seeds'])):
        raise SystemExit('Confirmation seed list does not match the already used calibration plan')
    write(out / 'split-audit.json', dict(
        train_episodes=len(parts['train']), selection_episodes=len(parts['selection']),
        confirmation_episodes=len(parts['confirmation']),
        selection_seeds=list(parts['selection_seeds']),
        confirmation_seeds=list(parts['confirmation_seeds']),
        confirmation_scored=False,
        selection_policy_rows={key: len(rows_for(parts['selection'], key)) for key in KEYS},
        train_policy_rows={key: len(rows_for(parts['train'], key)) for key in KEYS},
    ))
    if args.audit_only:
        return
    cells = {key: {} for key in KEYS}
    flags = {coefficient: {} for coefficient in COEFFICIENTS if coefficient}
    for key in KEYS:
        train_rows = rows_for(parts['train'], key)
        score_games = parts['selection'] if key in SELECTION_KEYS else parts['train']
        scored_rows = rows_for(score_games, key)
        baseline = None
        for coefficient in COEFFICIENTS:
            net, losses, reuse, grad = fit_key(key, coefficient, train_rows, args.initial)
            packed = pack(score_policy_rows(net, scored_rows), losses, reuse, grad)
            packed.update(key=key, coefficient=coefficient, eligible=key in SELECTION_KEYS,
                          split='selection' if key in SELECTION_KEYS else 'fit_in_sample')
            cells[key][str(coefficient)] = packed
            write(out / 'cells.json', cells)
            print(key, coefficient, json.dumps(packed['counts'], ensure_ascii=False), flush=True)
            if coefficient == 0:
                baseline = packed['counts']
            elif key in SELECTION_KEYS:
                flags[coefficient][key] = ranking_beats(baseline, packed['counts'])
    chosen = scheme_passes(flags)
    verdict = ('do_not_recommend_for_cuda_or_formal_training' if chosen is None
               else 'exploratory_only_no_untouched_confirmation_set')
    payload = dict(config=config, cells=cells, flags=flags, chosen=chosen, verdict=verdict,
                   serving_export=False, exploratory=True)
    write(out / 'result.json', payload)
    (out / 'report.md').write_text(render_report(payload), encoding='utf-8')
    print(verdict, 'chosen', chosen)


if __name__ == '__main__':
    main()
