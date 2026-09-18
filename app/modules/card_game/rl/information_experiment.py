"""Bounded full-game search-policy/value self-play pilot, separate from serving."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from itertools import combinations
from pathlib import Path
import json,time
import numpy as np
from .league_schema import PRESETS,deck,rule_hash
from .league_policy import network_from_old,export_model,LeagueModel
from .information_search import search_game,policy_value_update


def write(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def compact(game):
    return {k:v for k,v in game.items() if k!='rows'}


def experiment(out,sources,deadline,*,cycles=2,simulations=24,workers=4):
    import torch
    torch.set_num_threads(1);torch.manual_seed(984100000)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.time()
    config=dict(kind='observation_policy_ismcts_v1',rule_hash=rule_hash(),cycles=cycles,
                simulations=simulations,max_depth=10,workers=workers,deadline=deadline,
                opponent_model='frozen visible-information policy within each root search',
                belief='public-count-constrained uniform deck sampling; no equilibrium guarantee',
                value='tanh(raw/10), terminal +/-1/draw 0',policy_target='root visit distribution',
                shaping_rewards=False,synthetic_tactical_curriculum=False,automatic_serving_approval=False)
    write(out/'config.json',config)
    nets={};opts={};origins={};replay={k:[] for k in PRESETS};training=[];losses=[]
    for k in PRESETS:
        nets[k],origins[k]=network_from_old(sources['weave-rush' if k=='quick-rush' else k],k,'cpu')
        opts[k]=torch.optim.Adam(nets[k].parameters(),lr=3e-5)
        export_model(nets[k],out/'baseline',k,deck(k),origin=origins[k])
    current=out/'baseline';rng=np.random.default_rng(984200000)
    eval_reserve=min(240,max(90,(deadline-started)*.45))
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        for cycle in range(cycles):
            write(out/'status.json',dict(phase='selfplay',cycle=cycle+1,deadline=deadline))
            jobs=[]
            for pair,(a,b) in enumerate(combinations(PRESETS,2)):
                for first in ('a','b'):
                    jobs.append(dict(seed=984300000+cycle*10000+pair*100,first=first,
                                     deadline=deadline-eval_reserve,decks={'a':deck(a),'b':deck(b)},
                                     policies={'a':(str(current),a),'b':(str(current),b)},
                                     simulations=simulations,training=True,collect=True))
            games=list(pool.map(search_game,jobs))
            for job,g in zip(jobs,games):
                training.append(dict(cycle=cycle+1,seed=job['seed'],first=job['first'],
                                     decks={s:d['id'] for s,d in job['decks'].items()},**compact(g)))
                if g['complete']:
                    if g['decisions']!=g['searched'] or len(g['rows'])!=g['decisions']:
                        raise ValueError('Self-play skipped decision search/target')
                    for row in g['rows']:replay[row['key']].append(row)
            write(out/'training-games.json',training)
            if not all(g['complete'] for g in games):
                write(out/'status.json',dict(phase='incomplete_selfplay',cycle=cycle+1,deadline=deadline))
                break
            dest=out/f'cycle-{cycle+1}'
            for k in PRESETS:
                replay[k]=replay[k][-4096:]
                if not replay[k]:raise ValueError('No complete games for '+k)
                for step in range(48):
                    if time.time()>=deadline-eval_reserve:break
                    indices=rng.choice(len(replay[k]),min(64,len(replay[k])),replace=False)
                    result=policy_value_update(nets[k],opts[k],[replay[k][i] for i in indices])
                    losses.append(dict(cycle=cycle+1,key=k,step=step+1,**result))
                export_model(nets[k],dest,k,deck(k),origin=origins[k])
            current=dest
            write(out/'losses.json',losses)
        for k in PRESETS:
            export_model(nets[k],out/'final',k,deck(k),origin=origins[k])
            torch.save(dict(schema='observation_policy_ismcts_v1',rule_hash=rule_hash(),
                            model=nets[k].state_dict()),out/(k+'.pt'))
            # Numeric-only full-game samples for reproducibility; no pickled objects.
            arrays={}
            for i,row in enumerate(replay[k]):
                for f in ('x','c','pi'):arrays[f'{i}_{f}']=row[f]
                arrays[f'{i}_z']=np.float32(row['z'])
            np.savez_compressed(out/(k+'-samples.npz'),**arrays)
        # Same frozen opponent and paired opening seeds across all four variants.
        evaluations=[]
        for variant,raw in (('baseline',True),('baseline',False),('final',True),('final',False)):
            label=variant+('-raw' if raw else '-search')
            write(out/'status.json',dict(phase='evaluate:'+label,deadline=deadline))
            jobs=[];meta=[]
            for ki,k in enumerate(PRESETS):
                foe=PRESETS[(ki+1)%len(PRESETS)]
                for pair in range(2):
                    for first in ('a','b'):
                        jobs.append(dict(seed=985100000+ki*1000+pair,first=first,deadline=deadline-15,
                                         decks={'a':deck(k),'b':deck(foe)},
                                         policies={'a':(str(out/variant),k),'b':(str(out/'baseline'),foe)},
                                         raw_sides=['a','b'] if raw else ['b'],simulations=simulations))
                        meta.append(dict(key=k,foe=foe,first=first,pair=pair))
            games=list(pool.map(search_game,jobs))
            evaluations.extend(dict(label=label,**m,**compact(g)) for m,g in zip(meta,games))
            write(out/'evaluations.json',evaluations)
    # Old tactical fixtures stay outside training and checkpoint decisions.
    from .league_gate import evaluate
    gates={k:{v:evaluate(LeagueModel(out/v,k)) for v in ('baseline','final')} for k in PRESETS}
    write(out/'gates.json',gates)
    summary=dict(training_games=len(training),complete_training=sum(g['complete'] for g in training),
                 decisions=sum(g['decisions'] for g in training if g['complete']),
                 searched=sum(g['searched'] for g in training if g['complete']),
                 samples={k:len(v) for k,v in replay.items()},optimizer_steps=len(losses),
                 evaluation_games=len(evaluations),complete_evaluation=sum(g['complete'] for g in evaluations),
                 started=started,finished=time.time(),deadline=deadline,automatic_serving_approval=False)
    summary['results']={}
    for k in PRESETS:
        summary['results'][k]={}
        for label in ('baseline-raw','baseline-search','final-raw','final-search'):
            group=[g for g in evaluations if g['key']==k and g['label']==label]
            summary['results'][k][label]=dict(wins=sum(g.get('winner')=='a' for g in group),
                                             complete=sum(g['complete'] for g in group),n=len(group))
    write(out/'summary.json',summary)
    write(out/'status.json',dict(phase='complete' if all(g['complete'] for g in training+evaluations) else 'incomplete',
                                finished=time.time(),deadline=deadline))
