#!/usr/bin/env python3
"""Bounded six-worker throughput probe; no optimization or model publication."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def play(job):
    from app.modules.card_game.rl.information_search import search_game
    start=time.time()
    result=search_game(job)
    return {k:result.get(k) for k in ('complete','winner','decisions','searched','reason','model_versions')} | {'seconds':time.time()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-reservation',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args()
    if a.output.exists():p.error('New output required')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    from app.modules.card_game.rl.cross_runtime import model
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.rl.cross_schedule import ROSTER
    from scripts.train_duel_v2_current_round import make_job,write
    models=a.models.resolve()
    decks={k:model(models,k).serving_deck for k in ROSTER}
    seeds=json.loads(a.seed_reservation.read_text())
    total_start=time.time();deadline=total_start+a.seconds
    config=dict(simulations=32,gumbel_candidates=16,terminal_horizon=3)
    results={};durations={}
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for mode in ('argmax','search'):
            jobs=[]
            for i in range(a.workers):
                left=ROSTER[i%len(ROSTER)];right='murk' if left!='murk' else 'zhenhong'
                job=make_job(dict(left=left,right=right,first='a' if i%2==0 else 'b'),
                    seed=seeds['foundation']+i,models=models,decks=decks,config=config,
                    deadline=deadline,identity=f'capacity-{mode}-{i}',training=mode=='search')
                if mode=='argmax':job['raw_sides']=['a','b']
                jobs.append(job)
            started=time.time()
            results[mode]=list(pool.map(play,jobs));durations[mode]=time.time()-started
    write(a.output,dict(complete=all(r['complete'] for rs in results.values() for r in rs),
        rule_hash=identity(),search_algorithm='gumbel',**config,workers=a.workers,
        measured_at=time.time(),wall_seconds=time.time()-total_start,durations=durations,
        games_per_mode=a.workers,results=results,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(a.output,flush=True)


if __name__=='__main__':main()
