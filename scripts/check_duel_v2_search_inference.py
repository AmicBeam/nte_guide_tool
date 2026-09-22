#!/usr/bin/env python3
"""Paired raw-policy versus quiet low-budget search evaluation; no training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import argparse,hashlib,json,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.fixed_lineup import write,identity
from app.modules.card_game.rl.full_cycle_experiment import evaluate_job


def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    folder=out/'evaluation';(folder/'raw').mkdir(parents=True);(folder/'replays').mkdir()
    seeds=json.loads(a.seed_reservation.read_text());deadline=time.time()+a.seconds;models={};jobs=[]
    for label in (('initial','grounded-initial','trained') if a.extra_initial else ('initial','trained')):
        models[label]=rt.model(a.models/label,'zhenhong')
        for mode in ('raw','search16'):
            for fi,foe in enumerate(('starter','weave-rush','quick-rush')):
                enemy=rt.model(a.foes,foe)
                for i in range(a.pairs):
                    for first in ('a','b'):
                        variant=label+'-'+mode
                        jobs.append(dict(id=f'{variant}-{foe}-{i}-{first}',cell=f'{variant}/{foe}',variant=variant,kind='reference',foe=foe,
                            runtime='outcome',seed=seeds['audit']+fi*10000+i,first=first,deadline=deadline,
                            policies={'a':(str(a.models/label),'zhenhong'),'b':(str(a.foes),foe)},
                            decks={'a':models[label].serving_deck,'b':enemy.serving_deck},
                            search_sides=['a'] if mode=='search16' else [],simulations=16,search_algorithm='gumbel',gumbel_candidates=8,
                            output=str(folder),replay=i==0))
    write(out/'config.json',dict(rule_hash=identity(),hard_deadline=deadline,workers=a.workers,training=False,
        search_algorithm='gumbel',simulations=16,gumbel_candidates=8,noise=False,
        model_versions={k:p.version for k,p in models.items()},seeds=seeds,policy_and_planner_results_separate=True))
    write(folder/'jobs.json',jobs);results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for r in pool.map(evaluate_job,jobs):
            results.append(r)
            with (folder/'index.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
            write(out/'status.json',dict(phase='evaluation',completed=len(results),planned=len(jobs),hard_deadline=deadline))
    metrics={}
    for r in results:
        m=metrics.setdefault(r['variant'],dict(planned=0,complete=0,wins=0,passive=0,search_simulations=0))
        m['planned']+=1;m['complete']+=r['complete'];m['wins']+=r['complete'] and r['winner']=='a'
        m['passive']+=r['passive_triggers']['a'];m['search_simulations']+=r['search_simulations']
    write(out/'summary.json',dict(complete=all(r['complete'] for r in results),metrics=metrics,training=False,automatic_serving_approval=False))
    write(out/'status.json',dict(phase='finished',completed=len(results),planned=len(jobs),hard_deadline=deadline))
    members=[p for p in out.rglob('*') if p.is_file() and p.name not in ('delivery.zip','delivery.json')]
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
        for p in members:z.write(p,p.relative_to(out))
        z.writestr('manifest.json',json.dumps(manifest))
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),members=len(members)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--foes',type=Path,required=True)
    p.add_argument('--seed-reservation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=1800);p.add_argument('--pairs',type=int,default=8)
    p.add_argument('--workers',type=int,choices=(2,4),default=2);p.add_argument('--extra-initial',action='store_true',help='Also test the untrained categorical head, with identical budgets')
    a=p.parse_args()
    if not 60<=a.seconds<=3600 or not 1<=a.pairs<=32:p.error('Invalid bounded evaluation')
    run(a)
