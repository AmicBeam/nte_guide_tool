#!/usr/bin/env python3
"""Collect frozen Gumbel32 roots for teacher selection or confirmation only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = (('starter', 'weave-rush'), ('starter', 'quick-rush'),
         ('weave-rush', 'quick-rush'))


def schedule(base, replicas):
    return [dict(left=left, right=right, first=first,
                 seed=base + replica * 100 + index, replica=replica)
            for replica in range(replicas)
            for index, (left, right) in enumerate(PAIRS)
            for first in ('a', 'b')]


def run_job(job):
    from app.modules.card_game.rl.information_search import search_game
    from app.modules.card_game.rl.episode_replay import dump_episode
    output = Path(job.pop('episode_output'))
    result = search_game(job)
    if result.get('complete'):
        if result['decisions'] != result['searched'] or result['searched'] != len(result['rows']):
            raise ValueError('Incomplete search labels in completed game')
        dump_episode(output, job, result)
    return dict(id=job['id'], seed=job['seed'], first=job['first'],
                left=job['left'], right=job['right'], complete=bool(result.get('complete')),
                reason=result.get('reason'), winner=result.get('winner'),
                decisions=result.get('decisions'), searched=result.get('searched'),
                policy_rows=len(result.get('rows') or []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--seed-reservation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('selection', 'confirmation'), required=True)
    parser.add_argument('--replicas', type=int, default=4)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--hard-deadline', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.replicas <= 16 or not 1 <= args.workers <= 8:
        parser.error('New output and bounded replicas/workers required')
    if not 60 <= args.seconds <= 3600:
        parser.error('Bounded seconds must be 60..3600')
    hard = datetime.fromisoformat(args.hard_deadline)
    if hard.tzinfo is None:
        parser.error('Hard deadline needs timezone')
    deadline = min(time.time() + args.seconds, hard.timestamp() - 900)
    if deadline <= time.time() + 60:
        parser.error('Insufficient time before independent evaluation reserve')
    from app.modules.card_game.rl.cross_runtime import model
    from app.modules.card_game.rl.cross_lineup import identity
    from scripts.train_duel_v2_current_round import make_job
    seeds = json.loads(args.seed_reservation.read_text(encoding='utf-8'))
    if set(('selection', 'confirmation')) - set(seeds):
        parser.error('Shared seed reservation must include both holdout phases')
    decks = {key: model(args.models, key).serving_deck
             for key in ('starter', 'weave-rush', 'quick-rush')}
    plan = schedule(int(seeds[args.phase]), args.replicas)
    config = dict(kind='frozen_teacher_holdout', phase=args.phase,
                  rule_hash=identity(), model_sha256={key:model(args.models,key).version for key in decks},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seeds=seeds, plan=plan, search_algorithm='gumbel', simulations=32,
                  gumbel_candidates=16, terminal_horizon=3, workers=args.workers,
                  deadline=deadline, training=False, selection_only=args.phase=='selection',
                  confirmation_only=args.phase=='confirmation', website_approval=False)
    if not args.run:
        print(json.dumps(dict(dry_run=True, output_created=False, config=config), ensure_ascii=False))
        return
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    jobs=[]
    for index, item in enumerate(plan):
        job=make_job(item,seed=item['seed'],models=args.models,decks=decks,
                     config=dict(simulations=32,gumbel_candidates=16,terminal_horizon=3),
                     deadline=deadline,identity=f'{args.phase}-{index:03}',training=False)
        job.update(collect=True,record_episode=True,episode_output=str(args.output/'episodes'/f'{job["id"]}.json.gz'))
        jobs.append(job)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for result in pool.map(run_job,jobs):
            results.append(result)
            with (args.output/'games.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(result,ensure_ascii=False)+'\n')
    summary=dict(config=config, attempted=len(results), completed=sum(r['complete'] for r in results),
                 incomplete=len(results)-sum(r['complete'] for r in results),
                 policy_rows=sum(r['policy_rows'] for r in results),
                 learning=False, optimizer_steps=0, serving_export=False)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('attempted','completed','incomplete','policy_rows')},ensure_ascii=False))


if __name__ == '__main__':
    main()
