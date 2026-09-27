#!/usr/bin/env python3
"""Fit an isolated broad rule-value proxy; never update learner/service models."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=900);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    if not 300<=a.seconds<=1800 or not 1<=a.workers<=16:p.error('Bounded CPU preparation required')
    if not a.run:print(json.dumps(dict(dry_run=True,training_games=90,selection_games=30,heldout_games=30)));return
    if a.output.exists():raise ValueError('New output required')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    import numpy as np
    from app.modules.card_game.rl import recovery_runtime as rt,recovery_round as rr,covered_value as cv
    from app.modules.card_game.rl.cross_schedule import ROSTER,learning_schedule
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
    from app.modules.card_game.rl.cross_lineup import identity
    from app.modules.card_game.rl.offline_sources import snapshot
    from scripts.prepare_duel_v2_offline_readiness import EXTRA
    from scripts.train_duel_v2_current_round import preset_decks
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    torch.set_num_threads(1);signature=rr.versions(a.source)
    start=time.time();end=diagnostic_deadline(start,a.seconds)
    if end-start<300:raise ValueError('Insufficient preparation window')
    seeds=claim_seeds(ROOT/'artifacts/rl-seed-ledger.json',str(a.output.resolve()))
    rr.write(a.output/'config.json',dict(seeds=seeds,deadline=end,source_models=signature,
        learning_rate=.0003,epochs=[1,2,4,8],value_only=True,learner_updates=0,objective='visible_rule_after_six_exploration_actions'))
    snapshot(ROOT,a.output/'source',EXTRA+('app/modules/card_game/rl/covered_value.py','app/modules/card_game/rl/recovery_sampler.py','scripts/prepare_duel_v2_covered_value.py'))
    decks=preset_decks();plan=learning_schedule()+[dict(left=k,right=k,first=f) for k in ROSTER for f in ('a','b')]
    def jobs(base,replicas):
        result=[]
        for replica in range(replicas):
            for i,item in enumerate(plan):
                seed=base+replica*100+i//2;rng=np.random.default_rng(seed);builds={}
                for side,key in (('a',item['left']),('b',item['right'])):
                    cards=[]
                    for hero in decks[key]['character_ids']:
                        kit=[c for c,v in CARDS.items() if v['character_id']==hero and not v.get('derived')]
                        if (replica+i//2)%3==0:cards+=kit
                        else:
                            bag=[c for c in kit for _ in range(2)]
                            cards+=[bag[int(j)] for j in rng.choice(len(bag),8,replace=False)]
                    builds[side]=validate_deck(dict(decks[key],card_ids=cards))
                result.append(dict(seed=seed,first=item['first'],keys={'a':item['left'],'b':item['right']},
                    decks=builds,deadline=end-30))
        return result
    def collect(pool,phase,planned):
        games=[]
        for i,g in enumerate(pool.map(cv.collect_game,planned)):
            if not g['complete'] or not g['value_rows']:raise ValueError('Incomplete broad value trajectory')
            with gzip.open(a.output/f'{phase}-{i}-private.json.gz','wt') as f:
                json.dump(g,f,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)
            games.append(g)
        rr.write(a.output/(phase+'.json'),dict(games=len(games),seeds=[j['seed'] for j in planned],
            complete=True,first=[j['first'] for j in planned]))
        return games
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        train=collect(pool,'training',jobs(seeds['foundation'],3))
        selection=collect(pool,'selection',jobs(seeds['selection'],1))
        metrics={};choices={};nets={};builds={};origins={}
        for key in ROSTER:
            rows=[r for g in train for r in g['value_rows'] if r['key']==key]
            net,build,origin=rt.restore(a.source,key);nets[key]=net;builds[key]=build
            origins[key]=dict(origin,usage='covered_expert_value_auxiliary',quality_approved=False)
            frozen={n:t.detach().clone() for n,t in net.state_dict().items() if not n.startswith(('value_net.','wdl.'))}
            opt=torch.optim.AdamW([p for n,p in net.named_parameters() if n.startswith(('value_net.','wdl.'))],lr=.0003,weight_decay=.001)
            rng=np.random.default_rng(seeds['adaptation']);candidates=[]
            for epoch in range(1,9):
                indexes=rng.permutation(len(rows))
                for offset in range(0,len(rows),128):
                    if time.time()>=end-30:raise TimeoutError('Covered value fitting deadline')
                    rt.update(net,opt,[rows[int(i)] for i in indexes[offset:offset+128]],value_only=True)
                if epoch in (1,2,4,8):
                    folder=a.output/f'epoch-{epoch}'/'models';rt.export(net,folder,key,build,origins[key])
                    metric=rr.value_metrics(selection,{key:rt.model(folder,key)})[key]
                    candidates.append((metric['log_loss'],epoch,metric))
            chosen=min(candidates,key=lambda x:x[0]);choices[key]=dict(epoch=chosen[1],selection=chosen[2])
            net,_,_=rt.restore(a.output/f'epoch-{chosen[1]}'/'models',key)
            if any(not torch.equal(t,net.state_dict()[n]) for n,t in frozen.items()):raise ValueError('Auxiliary training moved policy arrays')
            rt.export(net,a.output/'models',key,build,origins[key]);print(key,choices[key],flush=True)
        held=collect(pool,'heldout',jobs(seeds['confirmation'],1))
    models={k:rt.model(a.output/'models',k) for k in ROSTER};metrics=rr.value_metrics(held,models)
    passed=all(m['normal_games']>=8 and m['brier']<.7 and m['log_loss']<1.2 for m in metrics.values())
    if rr.versions(a.source)!=signature:raise ValueError('Primary source changed')
    files={}
    for phase in ('training','selection','heldout'):
        for path in [a.output/(phase+'.json'),*a.output.glob(phase+'-*-private.json.gz')]:
            files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    rr.write(a.output/'scope.json',dict(schema='covered_expert_value_v1',passed=passed,
        rule_hash=identity(),algorithm_sha256=cv.identity(),objective='visible_rule_after_six_exploration_actions',
        models=rr.versions(a.output/'models'),metrics=metrics,choices=choices,files=files,
        primary_policy_arrays_unchanged=True,real_learner_updates=0,quality_approved=False))
    print(json.dumps(dict(passed=passed,metrics=metrics)),flush=True)


if __name__=='__main__':main()
