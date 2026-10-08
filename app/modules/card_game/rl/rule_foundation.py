"""Offline neural initialization from complete current-rule public-AI games.

Only original TRAINING trajectories are used. This is an explicitly supervised
initialization, not Gumbel learning, a process reward, or website approval.
"""
from copy import deepcopy
from pathlib import Path
import gzip
import json
import numpy as np


def imitation_target(observation, actions):
    from .policies import VisibleEngineRulePolicy
    scores=np.asarray(VisibleEngineRulePolicy().ranking_scores(observation,tuple(actions)),dtype=np.float64)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite public teacher ranking')
    best=np.isclose(scores,scores.max(),rtol=0,atol=1e-9)
    return (best/best.sum()).astype(np.float32)


def replay_training_job(job):
    from .covered_rollout import rule_decision
    from .league_rollout import clean
    from .recovery_runtime import decision
    from .information_search import terminal_value
    from ..engine.duel_v2 import new_game,observe,acting_side
    from ..engine.duel_v2.simulation import simulate_action
    from .cross_lineup import encode
    import time
    with gzip.open(job['path'],'rt',encoding='utf-8') as f:game=json.load(f)
    if not game['complete'] or game['seed']!=job['seed'] or game['first']!=job['first']:
        raise ValueError('Complete original TRAINING trajectory required')
    original={(r['step'],r['side']):r for r in game['value_rows']}
    state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks']);rows=[]
    for step,entry in enumerate(game['actions']):
        if time.time()>=job['deadline']:raise TimeoutError('Foundation replay deadline')
        side=acting_side(state)
        if side!=entry['side']:raise ValueError('Recorded actor differs')
        if step>=6:
            for viewer in ('a','b'):
                x,_=encode(observe(state,viewer,include_previews=False),[])
                np.testing.assert_array_equal(x,np.asarray(original[step,viewer]['x'],dtype=np.float32))
            view=observe(state,side,include_previews=True)
            actions,(x,c)=decision(state,side)
            expected=int(next(i for i,a in enumerate(actions) if a==entry['action']))
            legal,index=rule_decision(state,side)
            if legal!=actions or index!=expected:raise ValueError('Current public rule teacher differs')
            pi=imitation_target(view,actions)
            if pi[expected]<=0:raise ValueError('Recorded teacher action outside tied best ranking')
            if job['keys'][side] in ('zhenhong','murk'):
                rows.append(dict(x=x,c=c,pi=pi,selected=expected,step=step,side=side,
                    key=job['keys'][side],value_actor=1,policy_source='public_rule_initialization',seed=job['seed']))
        state=clean(simulate_action(state,side,entry['action']))
    if state['phase']!='finished' or state.get('winner')!=game['winner']:
        raise ValueError('Recorded true terminal differs')
    for row in rows:row['z']=terminal_value(state,row['side'])
    return dict(complete=True,seed=job['seed'],rows=rows,decisions=len(game['actions']),winner=game['winner'])


def training_jobs(folder,deadline):
    """Reconstruct only authenticated TRAINING builds; never read heldout labels."""
    from .cross_schedule import ROSTER,learning_schedule
    from .cross_lineup import all_features,candidate_names
    from .covered_value import phase_counts
    from ..content.duel_v2 import CARDS,validate_deck
    from scripts.train_duel_v2_current_round import preset_decks
    from hashlib import sha256
    folder=Path(folder);scope=json.loads((folder/'scope.json').read_text());config=json.loads((folder/'config.json').read_text())
    if scope.get('passed') is not True or scope.get('primary_policy_arrays_unchanged') is not True:
        raise ValueError('Qualified original expert scope required')
    counts=phase_counts(scope);phase=json.loads((folder/'training.json').read_text())
    if phase.get('complete') is not True or phase['games']!=counts['training']:
        raise ValueError('Complete declared TRAINING endpoint required')
    if sha256((folder/'training.json').read_bytes()).hexdigest()!=scope['files']['training.json']:
        raise ValueError('Training endpoint SHA mismatch')
    for key in ROSTER:
        manifest=json.loads((folder/'models'/f'{key}.json').read_text())
        if manifest['features']!=all_features() or manifest['candidates']!=candidate_names():
            raise ValueError('Original raw observations must have identical named columns')
    decks=preset_decks();plan=learning_schedule()+[dict(left=k,right=k,first=f) for k in ROSTER for f in ('a','b')];jobs=[]
    for ordinal in range(counts['training']):
        replica=ordinal//30;i=ordinal%30;item=plan[i]
        seed=config['seeds']['foundation']+replica*100+i//2
        if phase['seeds'][ordinal]!=seed or phase['first'][ordinal]!=item['first']:
            raise ValueError('Original TRAINING seed/initiative differs')
        path=folder/f'training-{ordinal}-private.json.gz'
        if sha256(path.read_bytes()).hexdigest()!=scope['files'][path.name]:raise ValueError('Original TRAINING SHA differs')
        rng=np.random.default_rng(seed);builds={}
        for side,key in (('a',item['left']),('b',item['right'])):
            cards=[]
            for hero in decks[key]['character_ids']:
                kit=[cid for cid,v in CARDS.items() if v['character_id']==hero and not v.get('derived')]
                if (replica+i//2)%3==0:cards+=kit
                else:
                    bag=[cid for cid in kit for _ in range(2)]
                    cards+=[bag[int(j)] for j in rng.choice(len(bag),8,replace=False)]
            builds[side]=validate_deck(dict(decks[key],card_ids=cards))
        jobs.append(dict(path=str(path),seed=seed,first=item['first'],decks=builds,
                         keys={'a':item['left'],'b':item['right']},deadline=deadline))
    return jobs


def split_training_seeds(jobs,seed):
    """Balanced TRAINING-internal validation; paired initiatives stay together."""
    from collections import defaultdict
    import math
    groups=defaultdict(set)
    for job in jobs:groups[tuple(sorted(job['keys'].values()))].add(job['seed'])
    rng=np.random.default_rng(seed);held=set()
    for matchup,values in sorted(groups.items()):
        values=sorted(values)
        if len(values)<5:raise ValueError('At least five paired TRAINING roots per matchup')
        held.update(int(v) for v in rng.choice(values,math.ceil(len(values)*.2),replace=False))
    return held


def imitation_metrics(net,rows,device):
    import torch
    from .league_learning import tensors
    if not rows:raise ValueError('Nonempty imitation partition required')
    correct=0;kl=0.;plays=0;legal_play=0
    from .league_schema import ACT_PLAY
    with torch.no_grad():
        for start in range(0,len(rows),128):
            batch=rows[start:start+128];x,c,m=tensors([(r['x'],r['c']) for r in batch],device)
            logits,_=net(x,c,m);logp=logits.log_softmax(-1).cpu().numpy()
            for i,row in enumerate(batch):
                p=np.asarray(row['pi']);positive=p>0;chosen=int(logp[i,:len(p)].argmax())
                correct+=int(p[chosen]>0)
                kl+=float(np.sum(p[positive]*(np.log(p[positive])-logp[i,:len(p)][positive])))
                if (row['c'][:,0]==ACT_PLAY).any():
                    legal_play+=1;plays+=int(row['c'][chosen,0]==ACT_PLAY)
    return dict(rows=len(rows),public_rank_match=correct/len(rows),kl=max(0.,kl/len(rows)),
                legal_play_rows=legal_play,played=plays,rank_ties_not_rule_equivalence=True)
