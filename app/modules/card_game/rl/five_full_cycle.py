"""Five-team PPO, bounded card search, paired adaptation and fail-closed audit."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from copy import deepcopy
from random import Random
import json
import time
import shutil
import numpy as np
from .fixed_lineup import KEYS, FOES, CARD_IDS, FixedModel, all_features, CAND_DIM, save_weights, write, identity
from .fixed_lineup_training import play, append
from .build_acceptance import canonical_build, canonical_candidates, build_identity, paired_result, delivery_decision


def bind(source, target, key, build):
    policy = FixedModel(source, key)
    save_weights(policy.weights, target, key, canonical_build(build), policy.manifest.get('origin'))


def shortlist(parent, baseline, seed, limit=16):
    from .full_cycle_experiment import candidates
    all_rows = candidates(parent, seed, card_ids=CARD_IDS)
    fixed = canonical_candidates([baseline, parent]); seen = {build_identity(b) for b in fixed}
    rest = [b for b in all_rows if build_identity(b) not in seen]
    Random(seed).shuffle(rest)
    return fixed + rest[:max(0, limit-len(fixed))]


def jobs_for(root, key, variants, builds, seeds, deadline, train=False, coefficient=.2):
    reference = {foe: FixedModel(root/'reference', foe).serving_deck for foe in FOES}
    return [dict(key=key, variant=label, foe=foe, seed=seed, first=first, deadline=deadline,
                 decks={'a':canonical_build(builds[label]), 'b':canonical_build(reference[foe])},
                 policies={'a':(str(directory),key),'b':(str(root/'reference'),foe)},
                 train=train,zhenhong_passive_reward=coefficient)
            for seed in seeds for foe in FOES for first in ('a','b') for label,directory in variants.items()]


def compare(pool, root, folder, key, variants, builds, seeds, deadline):
    jobs = jobs_for(root,key,variants,builds,seeds,deadline)
    rows = []
    for job,result in zip(jobs,pool.map(play,jobs)):
        row = {k:job[k] for k in ('key','variant','foe','seed','first')}
        row.update({k:v for k,v in result.items() if k not in ('trajectory','trajectory_returns')})
        append(folder,f'{key}-games.jsonl',row); rows.append(row)
    return rows


def scores(rows, labels):
    result = {}
    for label in labels:
        group=[r for r in rows if r['variant']==label]
        result[label]=dict(complete=bool(group) and all(r['complete'] for r in group),
                           wins=sum(r.get('winner')=='a' for r in group),n=len(group))
    return result


def train_stage(pool, root, source, builds, config, stage, deadline, target=None):
    import torch
    from .gpu_duel.ppo import CompactScorer
    from .league_learning import ordinary_update
    folder=root/stage; folder.mkdir(); current=folder/'initial'; nets={}; opts={}; origins={}
    for key in KEYS:
        p=FixedModel(source,key);net=CompactScorer(len(all_features()),CAND_DIM,p.hidden).to(config['device'])
        net.load_state_dict({n:torch.from_numpy(v.copy()) for n,v in p.weights.items()})
        nets[key]=net;opts[key]=torch.optim.Adam(net.parameters(),lr=3e-5)
        origins[key]=dict(p.manifest.get('origin') or {}, training_reward=dict(
            win=10,loss=-10,passive_per_trigger=config['zhenhong_passive_reward'] if key=='zhenhong' else 0))
        save_weights(p.weights,current,key,builds[key],origins[key])
    best={key:current for key in KEYS};cycle=0;total={k:dict(steps=0,games=0,samples=0) for k in KEYS}
    next_check=time.time()+600
    start_seed=config['seeds']['foundation'] if stage=='foundation' else config['seeds']['adaptation']
    # A cycle occupies three seeds. Both adaptation branches intentionally share them.
    while time.time()<deadline and (target is None or cycle<target) and not (config['smoke'] and cycle>=1):
        if cycle>=100000: raise ValueError('Training seed reservation exhausted')
        write(root/'status.json',dict(phase=stage,cycle=cycle,totals=total,stage_deadline=deadline,hard_deadline=config['deadlines']['shutdown']))
        jobs=[]
        for key in KEYS:
            jobs += jobs_for(root,key,{'learner':current},{'learner':builds[key]},[start_seed+cycle],deadline,
                             train=True,coefficient=config['zhenhong_passive_reward'])
        batches={k:[] for k in KEYS};results=list(pool.map(play,jobs))
        for job,result in zip(jobs,results):
            append(folder,'games.jsonl',dict(cycle=cycle,**{k:job[k] for k in ('key','foe','seed','first')},
                   **{k:v for k,v in result.items() if k not in ('trajectory','trajectory_returns')}))
            batches[job['key']].append(result)
        if not all(g['complete'] for g in results) or time.time()>=deadline: break
        for key in KEYS:
            stats=ordinary_update(nets[key],opts[key],batches[key],start_seed+cycle)
            if stats.get('updates')!=1: raise ValueError('Missing complete learning batch')
            total[key]['steps']+=1;total[key]['games']+=len(batches[key]);total[key]['samples']+=stats['samples']
            append(folder,'losses.jsonl',dict(key=key,cycle=cycle,**stats))
        cycle+=1; current=folder/'generations'/f'cycle-{cycle:06}'
        for key in KEYS:save_weights({n:v.detach().cpu().numpy() for n,v in nets[key].state_dict().items()},current,key,builds[key],origins[key])
        write(folder/'checkpoint.json',dict(current=str(current),cycle=cycle,totals=total))
        if target is None and (time.time()>=next_check or config['smoke']):
            for ki,key in enumerate(KEYS):
                if time.time()>=deadline:break
                variants={'candidate':current,'best':best[key]}
                seeds=range(config['seeds']['selection']+cycle*100+ki*10,config['seeds']['selection']+cycle*100+ki*10+(1 if config['smoke'] else 4))
                rs=compare(pool,root,folder,key,variants,{v:builds[key] for v in variants},seeds,min(deadline,time.time()+60))
                result=scores(rs,variants);promote=all(v['complete'] for v in result.values()) and result['candidate']['wins']>result['best']['wins']
                if promote:best[key]=current
                append(folder,'selection.jsonl',dict(key=key,cycle=cycle,result=result,promote=promote))
            next_check=time.time()+600
    selected=folder/'selected'
    for key in KEYS:
        bind(current if target is not None else best[key], selected,key,builds[key])
    write(folder/'summary.json',dict(cycles=cycle,totals=total,source=str(source),selected=str(selected)))
    return selected,total


def tune(pool,root,models,builds,config,deadline):
    folder=root/'cards';folder.mkdir();selected=deepcopy(builds);records=[]
    rounds=1 if config['smoke'] else 4
    for rnd in range(rounds):
        for ki,key in enumerate(KEYS):
            remaining=(rounds-rnd)*len(KEYS)-ki
            end=min(deadline,time.time()+max(0,deadline-time.time())/remaining)
            candidates=shortlist(selected[key],builds[key],config['seeds']['search']+rnd*100+ki,
                                 2 if config['smoke'] else config['search_candidates'])
            original=build_identity(builds[key]);incumbent=build_identity(selected[key])
            baseline_index=next(i for i,b in enumerate(candidates) if build_identity(b)==original)
            parent_index=next(i for i,b in enumerate(candidates) if build_identity(b)==incumbent)
            active=list(range(len(candidates)));complete=False
            write(folder/f'{rnd}-{key}-candidates.json',candidates)
            write(root/'status.json',dict(phase='cards',round=rnd,key=key,candidates=len(candidates),stage_deadline=end,hard_deadline=config['deadlines']['shutdown']))
            levels=(1,) if config['smoke'] else (8,32,128)
            for level,pairs in enumerate(levels):
                if time.time()>=end:break
                variants={str(i):models for i in active};bs={str(i):candidates[i] for i in active}
                seed=config['seeds']['search']+1_000_000+rnd*100000+ki*10000+level*1000
                result=scores(compare(pool,root,folder,key,variants,bs,range(seed,seed+pairs),end),variants)
                if not all(r['complete'] for r in result.values()):break
                ranked=sorted(active,key=lambda i:(-result[str(i)]['wins'],i!=baseline_index,i!=parent_index,i))
                if level==len(levels)-1:
                    winner=ranked[0];selected[key]=candidates[winner];complete=True
                active=list(dict.fromkeys([baseline_index,parent_index,*ranked[:4 if level==0 else 2]]))
                append(folder,'screens.jsonl',dict(round=rnd,key=key,level=level,results=result,active=active))
            records.append(dict(round=rnd,key=key,complete=complete,selected=selected[key]))
    write(folder/'selected-builds.json',selected);write(folder/'status.json',dict(complete=all(r['complete'] for r in records),rounds=records))
    return selected,all(r['complete'] for r in records)


def tactical_gate(directory,key,build):
    """Small explicit public fixtures; never call these full human-play acceptance."""
    from ..engine.duel_v2 import new_game,observe,apply_action
    policy=FixedModel(directory,key);results=[]
    labels=['visible_lethal']+(['immune_no_free_sacrifice'] if key=='zhenhong' else [])
    for label in labels:
        state=new_game(seed=73,first_side='a',skip_mulligan=True,decks={'a':build,'b':build},escalation=False)
        for team in state['sides'].values():
            team['deck']+=team['hand'];team['hand']=[];team['ap']=1;team['shield']=0
            for h in team['characters'].values():h['energy']=0;h['shield']=0
        if label=='visible_lethal':state['sides']['b']['hp']=1
        else:
            own=state['sides']['a'];foe=state['sides']['b'];enemy=build['character_ids'][0]
            own['characters']['zhenhong']['flags']['damage_immunity_until']=own['turn_count']+1
            own['front']='zhenhong'
            for cid,h in own['characters'].items():
                if cid!='zhenhong':h['hp']=1
            foe['front']=enemy;foe['characters'][enemy].update(hp=50,max_hp=50,base_attack=10)
        view=observe(state,'a',include_previews=False);actions=[e['action'] for e in view['legal_actions'] if e['action']['type']!='concede']
        chosen=actions[policy.select_public_action(view,actions)];after=apply_action(state,'a',chosen)
        passed=after.get('winner')=='a' if label=='visible_lethal' else (after.get('winner')=='a' or all(
            after['sides']['a']['characters'][cid]['hp']>0 for cid in build['character_ids']))
        results.append(dict(case=label,passed=passed,action=chosen))
    return dict(complete=True,passed=all(r['passed'] for r in results),cases=results,
                coverage='visible immediate lethal and Zhenhong immunity-window avoidable sacrifice; limited synthetic fixtures')


def gate(pool,root,key,baseline,candidate,builds,config,phase,deadline):
    folder=root/phase;folder.mkdir(exist_ok=True)
    start=config['seeds']['confirmation' if phase=='confirmation' else 'audit']
    seeds=range(start,start+config['confirmation_seeds'])
    rows=compare(pool,root,folder,key,{'baseline':baseline,'candidate':candidate},builds,seeds,deadline)
    result=paired_result(rows,expected_seeds=seeds,opponents=FOES,alpha=.025,family_size=len(KEYS),minimum_gain=.02)
    write(folder/f'{key}-result.json',result)
    return result


def run(root,config):
    import torch
    torch.set_num_threads(1);torch.manual_seed(config['seeds']['foundation'])
    root=Path(root);deadlines=config['deadlines'];initial=root/'initial'
    builds={k:canonical_build(config['builds'][k]) for k in KEYS}
    for key in KEYS:bind(initial,root/'canonical-initial',key,builds[key])
    initial=root/'canonical-initial'
    reference_hashes={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (root/'reference').iterdir()}
    write(root/'protocol.json',dict(config=config,scoring='three frozen opponents × both seats equally weighted',family_size=5,quality_approved=False))
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        foundation,_=train_stage(pool,root,initial,builds,config,'foundation',deadlines['foundation'])
        proposals,search_complete=tune(pool,root,foundation,builds,config,deadlines['search'])
        write(root/'locked-proposals.json',dict(builds=proposals,locked_at=time.time(),confirmation_not_read=True))
        target=1 if config['smoke'] else config['adaptation_steps']
        base,bstat=train_stage(pool,root,foundation,builds,config,'baseline-adaptation',deadlines['baseline_adaptation'],target)
        cand,cstat=train_stage(pool,root,foundation,proposals,config,'candidate-adaptation',deadlines['candidate_adaptation'],target)
        fair={k:bstat[k]['steps']==cstat[k]['steps']==target and FixedModel(root/'baseline-adaptation/initial',k).version==FixedModel(root/'candidate-adaptation/initial',k).version for k in KEYS}
        write(root/'adaptation-budget.json',dict(target=target,baseline=bstat,candidate=cstat,fair=fair))
        decisions={};confirm={}
        for key in KEYS:
            write(root/'status.json',dict(phase='confirmation',key=key,hard_deadline=deadlines['shutdown']))
            confirm[key]=gate(pool,root,key,base,cand,{'baseline':builds[key],'candidate':proposals[key]},config,'confirmation',deadlines['confirmation'])
        for key in KEYS:
            write(root/'status.json',dict(phase='audit',key=key,hard_deadline=deadlines['shutdown']))
            audit=gate(pool,root,key,initial,cand,{'baseline':builds[key],'candidate':proposals[key]},config,'audit',deadlines['audit'])
            tactics=tactical_gate(cand,key,proposals[key]) if time.time()<deadlines['audit'] else dict(complete=False,passed=False)
            d=delivery_decision(builds[key],proposals[key],confirm[key],audit,fair_adaptation=fair[key],tactical_passed=tactics['passed'],smoke=config['smoke'])
            d['tactical']=tactics;decisions[key]=d
        complete=search_complete and all(fair.values()) and all(d['confirmation'].get('complete') and d['audit'].get('complete') and d['tactical'].get('complete') for d in decisions.values())
        for key,d in decisions.items():
            if not complete:d.update(build_quality_approved=False,recommendation_status='retained_baseline',recommended_build=builds[key],reason='workflow_incomplete')
            bind(cand if d['build_quality_approved'] else initial,root/'recommended',key,d['recommended_build'])
            d['model_sha256']=FixedModel(root/'recommended',key).version
        if reference_hashes!={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (root/'reference').iterdir()}:
            raise ValueError('Frozen opponents changed')
    write(root/'delivery-decision.json',decisions)
    write(root/'status.json',dict(phase='finished',workflow_complete=complete,quality_approved=complete and all(d['build_quality_approved'] for d in decisions.values()),
          automatic_serving_approval=False,finished=time.time(),recommendations={k:d['recommendation_status'] for k,d in decisions.items()}))
