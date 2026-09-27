#!/usr/bin/env python3
"""Balanced new-seed capacity diagnosis on completed Gumbel32 training games.

Does not reopen a stopped trainer, change its receipt, optimize real models,
reuse strength confirmation, or grant CUDA/serving approval.
"""
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def diagnostic_deadline(start, seconds):
    """A new diagnosis has its own bounded budget; it never resumes training."""
    local=datetime.fromtimestamp(start,timezone(timedelta(hours=8)))
    hard=local.replace(hour=9,minute=0,second=0,microsecond=0)
    if hard<=local:hard+=timedelta(days=1)
    return min(start+seconds,hard.timestamp())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if not 300<=a.seconds<=1800 or not 1<=a.workers<=16:p.error('Bounded CPU seconds/workers required')
    if not a.run:print(json.dumps(dict(dry_run=True,seconds=a.seconds,workers=a.workers)));return
    from app.modules.card_game.rl import recovery_runtime as rt,recovery_round as rr
    from app.modules.card_game.rl.recovery_capacity import fit
    from app.modules.card_game.rl.episode_replay import load_episode
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.cross_schedule import ROSTER
    from app.modules.card_game.rl.offline_sources import snapshot
    from scripts.prepare_duel_v2_offline_readiness import EXTRA
    if a.output.exists():raise ValueError('New output required')
    source=a.source_run.resolve();models=source/'initial-value/models'
    report=json.loads((source/'initial-value/value-precheck.json').read_text())
    if not rt.value_precheck_ok(report,{k:rt.model(models,k) for k in ROSTER}):raise ValueError('Matching source value report required')
    jobs=[];games=[]
    for path in sorted((source/'cycles/1/sampling/episodes').glob('*.json.gz')):
        data=load_episode(path);job=data['job']
        if (job.get('training') is not True or job.get('simulations')!=32
                or job.get('q_scale')!='natural_wdl'
                or data['model_versions']!={s:rt.model(models,k).version for s,(_,k) in job['policies'].items()}):
            raise ValueError('Only matching completed Gumbel32 TRAINING data may be reused')
        jobs.append(job);games.append(dict(complete=True,rows=[r for r in data['rows'] if 'pi' in r],
            value_rows=[r for r in data['rows'] if 'pi' not in r]))
    if len(games)!=30:raise ValueError('Full 30-game fixed-build foundation required')
    started=time.time();deadline=diagnostic_deadline(started,a.seconds)
    if deadline-started<300:raise ValueError('Insufficient time before this diagnosis hard stop')
    seeds=claim_seeds(ROOT/'artifacts/rl-seed-ledger.json',str(a.output.resolve()))
    config=dict(device='cpu',seeds=seeds,source_run=str(source),deadline=deadline,workers=a.workers,
                reused_training_games=30,heldout_games=30,real_policy_updates=0,quality_approved=False)
    rr.write(a.output/'config.json',config);snapshot(ROOT,a.output/'source',EXTRA)
    decks={k:rt.model(models,k).serving_deck for k in ROSTER}
    held_jobs=rr.jobs_for(models,decks,seeds['probes'],deadline-90,pure=False,cycle=100,
                          report=report,simulations=32)
    for job in held_jobs:job.update(training=False,exploration=True,data_partition='capacity_heldout')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        held=rr.collect(pool,held_jobs,a.output/'heldout',learning=False,require_search=True)
    import torch
    torch.set_num_threads(1)
    result=fit(models,jobs,games,config,deadline-10,held_jobs=held_jobs,held_games=held)
    rr.write(a.output/'result.json',result)
    print(json.dumps(dict(passed=result['passed'],metrics={k:dict(train=r['train_after']['kl'],
                    held=r['heldout']['kl'],accuracy=r['heldout']['confident_accuracy'])
                    for k,r in result['metrics'].items()})),flush=True)


if __name__=='__main__':main()
