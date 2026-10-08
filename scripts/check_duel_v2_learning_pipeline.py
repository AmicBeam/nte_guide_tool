#!/usr/bin/env python3
"""Bounded offline diagnosis: exact search, resumed games, label fit, root interventions.

Never continues an expired run or publishes models. Fitting is an isolated test.
"""
from pathlib import Path
import argparse,cProfile,gzip,hashlib,json,pstats,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.information_search import search,search_game,sample_world
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.episode_replay import load_episode
from app.modules.card_game.rl.fixed_lineup import write,identity
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side


def metrics(net,rows):
    import torch
    from app.modules.card_game.rl.league_learning import tensors
    with torch.no_grad():
        x,c,m=tensors([(r['x'],r['c']) for r in rows],next(net.parameters()).device)
        logits,_=net(x,c,m);logp=logits.log_softmax(-1).cpu().numpy();values=[];agree=[]
        for i,r in enumerate(rows):
            p=np.asarray(r['pi']);entropy=-float(np.sum(p[p>0]*np.log(p[p>0])))
            values.append(float(-np.dot(p,logp[i,:len(p)]))-entropy)
            agree.append(int(np.argmax(logp[i,:len(p)]))==int(np.argmax(p)))
        return dict(policy_kl=float(np.mean(values)),top_action_agreement=float(np.mean(agree)),samples=len(rows))


def fit(source,out,deadline):
    import torch
    torch.manual_seed(7881);torch.set_num_threads(1)
    train=[];held=[];provenance={}
    for i in (0,1,4,5,8,9):
        p=source/'private-episodes'/f'selfmixed-{i}.json.gz';episode=load_episode(p)
        if episode['job'].get('held') or episode['job'].get('matchup')!='mirror':raise ValueError('Only original training mirror episodes')
        selected=[r for r in episode['rows'] if 'pi' in r]
        (train if i in (0,1,4) else held).extend(selected)
        provenance[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    net=rt.create_network(source/'common-initial','zhenhong','cuda')
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=3e-5)
    result=dict(before_train=metrics(net,train),before_holdout=metrics(net,held),provenance=provenance,
                scope='fit diagnostic on previously generated TRAINING episodes; no final evaluation data')
    updates=0
    for i in range(64):
        if time.time()>=deadline:break
        rt.update(net,opt,train);updates+=1
        if (i+1)%8==0:write(out/f'fit-step-{i+1}.json',dict(train=metrics(net,train),holdout=metrics(net,held)))
    result.update(steps=updates,after_train=metrics(net,train),after_holdout=metrics(net,held))
    if not updates:raise RuntimeError('Fit check had no optimization budget')
    write(out/'fit.json',result)


def continuation(state,policies,deadline):
    steps=0
    while state['phase']!='finished' and steps<400 and time.time()<deadline:
        side=acting_side(state);actions,(x,c)=rt.decision(state,side)
        state=clean(apply_action(state,side,actions[int(policies[side].scores(x,c).argmax())]));steps+=1
    return dict(complete=state['phase']=='finished',winner=state.get('winner'),steps=steps)


def run(source,out,seeds,seconds):
    start=time.time();deadline=start+seconds
    policy=rt.model(source/'common-initial','zhenhong');foe=rt.model(source/'sources','starter')
    policies={'a':policy,'b':foe};decks={s:p.serving_deck for s,p in policies.items()}
    # New diagnostic seeds; these natural states are never used for gradients.
    roots=[]
    for i in range(2):
        state=new_game(seed=seeds['probes']+i,first_side=('a','b')[i],decks=decks)
        for step in range(60):
            if state['phase']=='finished':break
            side=acting_side(state)
            if side=='a' and ((i,len(roots))==(0,0) or step>=12 and len(roots)==i*2+1 or i==1 and len(roots)==2):
                roots.append(state)
                if len(roots)==(i+1)*2:break
            actions,(x,c)=rt.decision(state,side);state=clean(apply_action(state,side,actions[int(policies[side].scores(x,c).argmax())]))
    comparisons=[];quality=[]
    for i,state in enumerate(roots):
        if time.time()>=deadline:break
        answers={};times={}
        for reuse in ((False,True) if i%2==0 else (True,False)):
            t=time.time()
            answers[reuse]=search(state,'a',policies,seed=seeds['probes']+1000+i,simulations=256,deadline=deadline,runtime=rt,reuse_inference=reuse)
            times[str(reuse)]=time.time()-t
            if i==0:
                profiler=cProfile.Profile()
                profiler.runcall(search,state,'a',policies,seed=seeds['probes']+1000+i,simulations=64,deadline=deadline,runtime=rt,reuse_inference=reuse)
                profiler.dump_stats(str(out/f'profile-{i}-{reuse}.pstats'))
                with (out/f'profile-{i}-{reuse}.txt').open('w') as stream:pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(25)
        old,new=answers[False],answers[True]
        assert old['complete'] and new['complete']
        np.testing.assert_array_equal(old['visits'],new['visits']);np.testing.assert_array_equal(old['mean_values'],new['mean_values'])
        raw,new_action=old['raw_choice'],new['search_choice']
        comparisons.append(dict(root=i,seconds=times,visits_identical=True,values_identical=True,raw_choice=raw,search_choice=new_action,
                                simulations=new['simulations'],nodes=new['nodes']))
        for world in range(4):
            sampled=sample_world(state,'a',seeds['probes']+10000+i*100+world,runtime=rt)
            results={}
            for label,choice in (('raw',raw),('search',new_action)):
                after=clean(apply_action(sampled,'a',new['actions'][choice]))
                results[label]=continuation(after,policies,deadline)
            quality.append(dict(root=i,world=world,same_action=raw==new_action,**results))
    write(out/'search-comparison.json',comparisons);write(out/'root-quality.json',quality)
    # Actual shared search, serialized between every three complete decisions.
    job=dict(runtime='outcome',seed=seeds['foundation'],first='a',decks=decks,
        policies={'a':(str(source/'common-initial'),'zhenhong'),'b':(str(source/'sources'),'starter')},
        deadline=deadline,simulations=4,search_mix=dict(low=2,high=4,deep_probability=.25),training=True,
        collect=True,record_episode=True,train_sides=['a'])
    full=search_game(job);assert full['complete'],'Uninterrupted resume check did not finish in diagnostic budget'
    sliced={**job,'resume_path':str(out/'private-progress.json.gz'),'slice_actions':3};slices=0
    while time.time()<deadline:
        resumed=search_game(sliced);slices+=1
        if resumed['complete']:break
        assert resumed['resumable']
    assert resumed['complete'] and resumed['episode_actions']==full['episode_actions'] and resumed['winner']==full['winner']
    assert len(full['rows'])==len(resumed['rows']) and len(full['value_rows'])==len(resumed['value_rows'])
    for a,b in zip(full['rows']+full['value_rows'],resumed['rows']+resumed['value_rows']):
        assert a.keys()==b.keys()
        for k in a:
            if isinstance(a[k],np.ndarray):np.testing.assert_array_equal(a[k],b[k])
            else:assert a[k]==b[k]
    write(out/'resume.json',dict(complete=True,slices=slices,decisions=full['decisions'],winner=full['winner'],rows=len(full['rows']),identical_actions_and_labels=True))
    (out/'private-progress.json.gz').unlink()
    if time.time()>=deadline:raise RuntimeError('No budget for learning fit check')
    fit(source,out,deadline)
    write(out/'summary.json',dict(complete=True,elapsed=time.time()-start,roots=len(comparisons),counterfactual_pairs=len(quality),rule_hash=identity(),
                                no_serving_changes=True,no_long_training=True,diagnostic_deadline=deadline))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-reservation',type=Path,required=True);p.add_argument('--seconds',type=int,default=600)
    args=p.parse_args()
    if not 60<=args.seconds<=900:raise ValueError('Bounded diagnostic requires 60..900 seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    seeds=json.loads(args.seed_reservation.read_text());run(args.source,args.output,seeds,args.seconds)
