#!/usr/bin/env python3
"""Bounded research fit: Gumbel/WDL plus verified old-policy behavior anchor.

Selection uses archived paired roots only. It cannot approve formal training or
serving; new independent games are required after this screening.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl import cross_runtime as runtime
from app.modules.card_game.rl.cross_teacher_distillation import (
    ARCHIVED_EPISODES, REPO_TEACHERS, TEAMS, agreement_record,
    overlapping_root, split_paired_roots, summarize_agreement,
    update_student, validate_kl_grid, verify_repository_teachers,
)
from scripts.fit_duel_v2_policy_recovery import load_games


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def selection_pass(baseline, candidate):
    if (not baseline['n'] or not candidate['legal_play']
            or candidate['n'] != baseline['n'] or candidate['collapse']):
        return False
    return bool(
        candidate['exact_action'] > baseline['exact_action']
        and candidate['category_agreement'] > baseline['category_agreement']
        and candidate['card_agreement'] >= baseline['card_agreement']
        and candidate['attack_called_play'] <= baseline['attack_called_play']
        and candidate['end_called_play'] <= baseline['end_called_play']
        and abs(candidate['legal_play_top1'] - candidate['teacher_play'] / candidate['legal_play']) <= 0.10
    )


def rows_for(games, key):
    return [row for game in games for row in game['rows']
            if row.get('key') == key and 'pi' in row]


def evaluate(policy, rows, teacher):
    records = []
    for row in rows:
        mapped = overlapping_root(row, teacher)
        if not mapped['overlapping']:
            continue
        scores = policy.scores(row['x'], row['c'])
        teacher_prob = np.exp(mapped['scores'] - mapped['scores'].max())
        teacher_prob /= teacher_prob.sum()
        records.append(agreement_record(row, mapped, scores, teacher_prob))
    return summarize_agreement(records)


def initial_network(directory, key, device):
    import torch
    source = runtime.model(directory, key)
    net = runtime.create_network(source.hidden, device)
    net.load_state_dict({name: torch.as_tensor(value.copy(), device=device)
                         for name, value in source.weights.items()})
    return net, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=ARCHIVED_EPISODES)
    parser.add_argument('--teachers', type=Path, default=REPO_TEACHERS)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--confirmation-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coefficients', type=float, nargs='+', default=[0.0, 0.2, 1.0])
    parser.add_argument('--anchor-mode', choices=('kl', 'top1'), default='kl')
    parser.add_argument('--learning-rate', type=float, default=1.5e-4)
    parser.add_argument('--passes', type=int, default=4)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--hard-deadline', default='2026-09-25T09:00:00+08:00')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be new and empty')
    if args.anchor_mode == 'kl':
        coefficients = validate_kl_grid(args.coefficients)
    else:
        coefficients = tuple(args.coefficients)
        if (not coefficients or 0.0 not in coefficients or len(set(coefficients)) != len(coefficients)
                or any(not np.isfinite(value) or not 0 <= value <= 5 for value in coefficients)):
            parser.error('Invalid predeclared teacher top-action grid')
    if (args.passes not in (1, 4, 8) or args.batch < 32 or not 1 <= args.seconds <= 3600
            or not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3):
        parser.error('Invalid bounded optimizer budget')
    deadline_iso = datetime.fromisoformat(args.hard_deadline)
    if deadline_iso.tzinfo is None:
        parser.error('Hard deadline needs an explicit timezone')
    deadline = min(time.time() + args.seconds, deadline_iso.timestamp() - 900)
    if deadline <= time.time() + 60:
        parser.error('Insufficient time before reserved independent evaluation window')
    teachers = verify_repository_teachers(args.teachers)
    games = load_games(args.episodes)
    confirmation = json.loads(args.confirmation_plan.read_text(encoding='utf-8'))['confirmation_seeds']
    split = split_paired_roots(games, confirmation)
    config = dict(kind='cross_teacher_offline_research', coefficients=coefficients,
                  anchor_mode=args.anchor_mode,
                  learning_rate=args.learning_rate, passes=args.passes, batch=args.batch,
                  device=args.device, hard_deadline=args.hard_deadline,
                  stop_at=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                  teacher_sha256={key: teachers[key]['sha256'] for key in TEAMS},
                  initial_sha256={key: hashlib.sha256((args.initial / f'{key}.npz').read_bytes()).hexdigest()
                                  for key in TEAMS},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fit_seeds=sorted({game['seed'] for game in split['train']}),
                  selection_seeds=list(split['selection_seeds']),
                  excluded_confirmation_seeds=list(split['confirmation_seeds']),
                  confirmation_scored=False, full_game_win_rate=False,
                  website_approval=False, formal_training=False)
    if not args.run:
        print(json.dumps(dict(config=config, dry_run=True, output_created=False), ensure_ascii=False))
        return
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True)
    write(args.output / 'config.json', config)
    torch.set_num_threads(1)
    measured = {}
    for key in TEAMS:
        fit = rows_for(split['train'], key)
        selection = rows_for(split['selection'], key)
        if not selection:
            measured[key] = dict(status='no_original_selection_rows', variants={})
            continue
        for row in fit:
            mapped = overlapping_root(row, teachers[key])
            row['_teacher_mapped'] = mapped
            row['_teacher_sha256'] = teachers[key]['sha256']
        orders = []
        rng = np.random.default_rng(20260925)
        for _ in range(args.passes):
            orders.append(rng.permutation(len(fit)).tolist())
        variants = {}
        for coefficient in coefficients:
            if time.time() >= deadline:
                variants[str(coefficient)] = dict(status='not_started_deadline')
                continue
            torch.manual_seed(20260925)
            net, source = initial_network(args.initial, key, args.device)
            optimizer = torch.optim.Adam([parameter for parameter in net.parameters()
                                          if parameter.requires_grad], lr=args.learning_rate)
            steps = 0
            complete = True
            for order in orders:
                for start in range(0, len(order), args.batch):
                    indexes = order[start:start + args.batch]
                    if len(indexes) < 32:
                        continue
                    if time.time() >= deadline:
                        complete = False
                        break
                    update_student(net, optimizer, [fit[index] for index in indexes], teachers,
                                   kl_coefficient=coefficient if args.anchor_mode == 'kl' else 0.0,
                                   teacher_top_weight=coefficient if args.anchor_mode == 'top1' else 0.0)
                    steps += 1
                if not complete:
                    break
            if not complete:
                variants[str(coefficient)] = dict(status='incomplete_deadline', steps=steps)
                continue
            folder = args.output / 'models' / str(coefficient)
            runtime.export(net, folder, key, source.serving_deck,
                           dict(initialization='research_teacher_anchor', source_sha256=source.version,
                                teacher_sha256=teachers[key]['sha256'],
                                anchor_mode=args.anchor_mode, anchor_weight=coefficient,
                                automatic_serving_approval=False))
            policy = runtime.model(folder, key)
            variants[str(coefficient)] = dict(status='complete', steps=steps,
                                              model_sha256=policy.version,
                                              selection=evaluate(policy, selection, teachers[key]))
            write(args.output / 'progress.json', dict(key=key, variants=variants))
        baseline = variants.get('0.0', {}).get('selection')
        for coefficient, item in variants.items():
            item['passed_selection'] = (bool(baseline and item.get('selection')
                                             and selection_pass(baseline, item['selection']))
                                        if coefficient != '0.0' else False)
        measured[key] = dict(status='complete', fit_rows=len(fit), selection_rows=len(selection),
                             variants=variants)
        write(args.output / 'result.json', dict(config=config, measured=measured,
                                                full_game_win_rate=False, formal_training=False))
    write(args.output / 'result.json', dict(config=config, measured=measured,
                                            full_game_win_rate=False, formal_training=False))
    print(json.dumps({'output':str(args.output),'measured':{key:value['status'] for key,value in measured.items()}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
