#!/usr/bin/env python3
"""Disposable full-corpus fit, contrasting the small-batch capacity probe.

Fits only declared completed TRAINING episodes. It does not promote models,
consume heldout labels for optimization, or alter any source arrays.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if not 2<=a.steps<=400 or not 60<=a.seconds<=600:p.error('Bounded steps/seconds required')
    if not a.run:print(json.dumps(dict(dry_run=True,steps=a.steps,seconds=a.seconds)));return
    if a.output.exists():raise ValueError('New output required')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    import numpy as np
    from app.modules.card_game.rl import recovery_runtime as rt,recovery_round as rr
    from app.modules.card_game.rl.recovery_capacity import label_metrics
    from app.modules.card_game.rl.episode_replay import load_episode
    from app.modules.card_game.rl.cross_schedule import ROSTER
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    torch.set_num_threads(1)
    models=a.source_run/'initial-value/models';signature=rr.versions(models)
    games=[]
    for path in sorted((a.source_run/'cycles/1/sampling/episodes').glob('*.json.gz')):
        game=load_episode(path);job=game['job']
        if (job.get('training') is not True or job.get('simulations')!=32 or
                game['model_versions']!={s:rt.model(models,k).version for s,(_,k) in job['policies'].items()}):
            raise ValueError('Matching completed Gumbel32 TRAINING source required')
        games.append(game)
    if len(games)!=30:raise ValueError('Full 30-game source required')
    config=dict(source_run=str(a.source_run),source_models=signature,steps=a.steps,learning_rate=.003,
                batch=128,seconds=a.seconds,disposable_only=True,real_policy_updates=0,
                heldout_protocol=dict(kl_max=.15,confident_accuracy_min=.75),quality_approved=False)
    rr.write(a.output/'config.json',config)
    end=diagnostic_deadline(time.time(),a.seconds);results={}
    for key in ROSTER:
        rows=[r for game in games for r in game['rows'] if r['key']==key and 'pi' in r]
        net,build,origin=rt.restore(models,key);before=label_metrics(net,rows,'cpu')
        opt=torch.optim.Adam(net.parameters(),lr=.003);rng=np.random.default_rng(20260927)
        gradients={k:0. for k in ('state_net','cand_net','score')};count=0;uses=np.zeros(len(rows),dtype=int)
        while count<a.steps and time.time()<end:
            order=rng.permutation(len(rows))
            for start in range(0,len(rows),128):
                if count>=a.steps or time.time()>=end:break
                ids=order[start:start+128]
                stats=rt.update(net,opt,[rows[int(i)] for i in ids]);count+=1;uses[ids]+=1
                for k in gradients:gradients[k]+=stats['policy_gradient_norms'].get(k,0.)
        rt.export(net,a.output/'copies',key,build,dict(origin,full_corpus_isolation=True,quality_approved=False))
        after=label_metrics(net,rows,'cpu')
        results[key]=dict(before=before,after=after,steps=count,policy_rows=len(rows),
                          min_reuse=int(uses.min()),max_reuse=int(uses.max()),gradient_sums=gradients)
        print(key,results[key],flush=True)
    if rr.versions(models)!=signature:raise ValueError('Source changed')
    rr.write(a.output/'result.json',dict(complete=all(r['steps']==a.steps for r in results.values()),
                                       metrics=results,source_unchanged=True,quality_approved=False))


if __name__=='__main__':main()
