#!/usr/bin/env python3
"""Bounded low-budget reanalysis/distillation using TRAINING episodes only.

The previously diagnosed evaluation case is a separate read-only holdout.
No website model is changed. No process reward or terminal-action override.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import argparse,copy,gzip,hashlib,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.information_search import search,terminal_value
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.fixed_lineup import write
from app.modules.card_game.rl.full_cycle_experiment import evaluate_job
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
from scripts.check_duel_v2_simulation import research_copy
from scripts.check_duel_v2_learning_pipeline import metrics


def save_data(path,data):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(data,f,default=convert)


def roots_from_episode(source,index,count):
    path=source/'private-episodes'/f'selfmixed-{index}.json.gz'
    d=json.loads(gzip.open(path,'rt',encoding='utf-8').read());j=d['job']
    if not j.get('training') or j.get('held') or j.get('matchup')!='mirror':raise ValueError('Training mirror provenance required')
    steps=set(np.linspace(0,len(d['actions'])-1,min(count,len(d['actions'])),dtype=int).tolist())
    s=new_game(seed=j['seed'],first_side=j['first'],decks=j['decks']);roots=[]
    for step,a in enumerate(d['actions']):
        if step in steps:roots.append(dict(state=s,side=a['side'],z=0 if d['winner'] not in ('a','b') else 1 if d['winner']==a['side'] else -1,episode=index,step=step))
        s=clean(apply_action(s,a['side'],a['action']))
    return roots,hashlib.sha256(path.read_bytes()).hexdigest()


def held_case(source):
    d=json.loads(gzip.open(source/'evaluation/raw/matrix-selfmixed-selfmixed-6.json.gz','rt',encoding='utf-8').read())
    s=new_game(seed=d['seed'],first_side=d['first'],decks=d['decks'])
    for a in d['actions'][:138]:s=clean(apply_action(s,a['side'],a['action']))
    return s,'b'


def check_raw(policy,state,side):
    actions,(x,c)=rt.decision(state,side);i=int(policy.scores(x,c).argmax());after=apply_action(state,side,actions[i])
    return dict(action=actions[i],immediate_win=after.get('winner')==side,phase=after['phase'])


def run(a):
    import torch
    out=a.output;source=a.source_run;seeds=json.loads(a.seed_reservation.read_text());deadline=time.time()+a.seconds
    torch.set_num_threads(1);torch.manual_seed(seeds['foundation'])
    research_copy(source/'selfmixed/final',out/'initial','zhenhong',a.source_rule_hash)
    for k in ('starter','weave-rush','quick-rush'):research_copy(source/'sources',out/'foes',k,a.source_rule_hash)
    policy=rt.model(out/'initial','zhenhong');policies={'a':policy,'b':policy}
    groups={'train':[],'holdout':[]};provenance={}
    for i in (0,1,4,5,8,9):
        roots,sha=roots_from_episode(source,i,32 if i in (0,1,4) else 16)
        groups['train' if i in (0,1,4) else 'holdout'].extend(roots);provenance[str(i)]=sha
    datasets={budget:{'train':[],'holdout':[]} for budget in (16,32)};evidence=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        tasks=[]
        for split,roots in groups.items():
            for root in roots:
                for budget in (16,32):
                    kwargs=dict(seed=seeds['probes']+root['episode']*1000+root['step'],simulations=budget,algorithm='gumbel',
                                noise=True,runtime=rt,deadline=deadline-180)
                    # Modules cannot cross the process boundary; use the top-level wrapper below.
                    tasks.append((root,split,budget,pool.submit(teacher_job,root,str(out/'initial'),kwargs['seed'],budget,deadline-180)))
        for root,split,budget,future in tasks:
            result=future.result();assert result['complete'] and result['simulations']<=budget
            action=result['actions'][result['search_choice']];after=apply_action(root['state'],root['side'],action)
            z=terminal_value(after,root['side']) if after['phase']=='finished' else root['z']
            row=dict(x=result['x'],c=result['c'],pi=result['pi'],z=z,value_actor=1,episode=root['episode'],step=root['step'],
                     value_source='actual_terminal_branch' if after['phase']=='finished' else 'historical_continuation')
            datasets[budget][split].append(row)
            evidence.append(dict(split=split,budget=budget,episode=root['episode'],step=root['step'],simulations=result['simulations'],
                visited=result['roots_covered'],legal=len(result['actions']),chosen=action,immediate_win=after.get('winner')==root['side']))
        save_data(out/'teacher-samples.json.gz',datasets);write(out/'teacher-evidence.json',evidence);write(out/'provenance.json',provenance)
        state,side=held_case(source);held_search=[]
        for budget in (16,32,64):
            for noise in (False,True):
                for salt in (181,383,587):
                    result=search(state,side,policies,seed=seeds['selection']+salt,simulations=budget,algorithm='gumbel',noise=noise,runtime=rt,deadline=deadline-90)
                    assert result['complete'];chosen=result['actions'][result['search_choice']]
                    after=apply_action(state,side,chosen)
                    held_search.append(dict(budget=budget,noise=noise,seed=seeds['selection']+salt,action=chosen,win=after.get('winner')==side,
                                           simulations=result['simulations'],policy_source=result['policy_source']))
        write(out/'held-search.json',held_search)
        fit={}
        for budget in (16,32):
            net=rt.create_network(out/'initial','zhenhong','cuda');opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=3e-4)
            train=datasets[budget]['train'];held=datasets[budget]['holdout'];rng=np.random.default_rng(seeds['foundation'])
            fit[budget]=dict(before_train=metrics(net,train),before_holdout=metrics(net,held),raw_case_before=check_raw(policy,state,side))
            for step in range(128):
                if time.time()>=deadline-75:raise RuntimeError('Insufficient fit/evaluation budget')
                sample=[train[i] for i in rng.choice(len(train),min(64,len(train)),replace=False)]
                rt.update(net,opt,sample)
                if (step+1)%32==0:write(out/f'fit-{budget}-{step+1}.json',dict(train=metrics(net,train),holdout=metrics(net,held)))
            target=out/f'learned-{budget}';rt.export(net,target,'zhenhong',policy.serving_deck,dict(kind='bounded_gumbel_distillation',steps=128,source_sha=policy.version,learning_rate=3e-4,auxiliary_rewards=0))
            fit[budget].update(after_train=metrics(net,train),after_holdout=metrics(net,held),steps=128,raw_case_after=check_raw(rt.model(target,'zhenhong'),state,side))
        write(out/'fit.json',fit)
        folder=out/'evaluation';(folder/'raw').mkdir(parents=True);(folder/'replays').mkdir();jobs=[]
        for label,path in [('initial',out/'initial'),('gumbel16',out/'learned-16'),('gumbel32',out/'learned-32')]:
            for fi,foe in enumerate(('starter','weave-rush','quick-rush')):
                opponent=rt.model(out/'foes',foe)
                for i in range(8):
                    for first in ('a','b'):
                        jobs.append(dict(id=f'{label}-{foe}-{i}-{first}',cell=f'{label}/{foe}',variant=label,foe=foe,runtime='outcome',seed=seeds['audit']+fi*1000+i,first=first,
                            deadline=deadline,policies={'a':(str(path),'zhenhong'),'b':(str(out/'foes'),foe)},decks={'a':policy.serving_deck,'b':opponent.serving_deck},output=str(folder),replay=i==0))
        write(folder/'jobs.json',jobs);results=list(pool.map(evaluate_job,jobs));write(folder/'index.json',results)
    write(out/'summary.json',dict(complete=all(r['complete'] for r in results),evaluation_games=len(results),held_search_wins=sum(r['win'] for r in held_search),held_search_tests=len(held_search),
        fit=fit,metrics={v:dict(games=sum(r['complete'] for r in results if r['variant']==v),wins=sum(r['complete'] and r['winner']=='a' for r in results if r['variant']==v)) for v in ('initial','gumbel16','gumbel32')},
        no_serving_approval=True,scope='bounded reanalysis and distillation, not full RL quality approval'))


def teacher_job(root,directory,seed,budget,deadline):
    p=rt.model(directory,'zhenhong')
    return search(root['state'],root['side'],{'a':p,'b':p},seed=seed,simulations=budget,algorithm='gumbel',noise=True,runtime=rt,deadline=deadline)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--source-rule-hash',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-reservation',type=Path,required=True);p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args()
    if not 300<=a.seconds<=1200:raise ValueError('Bounded check requires 300..1200 seconds')
    a.output.mkdir(parents=True,exist_ok=False);run(a)
