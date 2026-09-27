"""Frozen root counterfactuals; no optimization or model promotion."""
from copy import deepcopy
import math
import time
import numpy as np
from . import information_search as core
from .league_rollout import clean
from ..engine.duel_v2 import acting_side, apply_action


def continuation(world, viewer, action, policies, runtime, *, seed, deadline, max_actions=512):
    """Apply one root action, then run visible-policy play to a real terminal."""
    if time.time()>=deadline:return dict(complete=False,reason='deadline',actions=[])
    state=clean(apply_action(deepcopy(world),viewer,action))
    trace=[dict(side=viewer,action=action)]
    rng=np.random.default_rng(seed)
    for _ in range(max_actions):
        if state['phase']=='finished':
            return dict(complete=True,value=core.terminal_value(state,viewer),
                        winner=state.get('winner'),actions=trace)
        if time.time()>=deadline:
            return dict(complete=False,reason='deadline',actions=trace)
        actor=acting_side(state);legal,(x,c)=runtime.decision(state,actor)
        scores=runtime.action_scores(policies[actor],x,c)
        index=int(rng.choice(len(legal),p=core.softmax(scores)))
        state=clean(apply_action(state,actor,legal[index]))
        trace.append(dict(side=actor,action=legal[index]))
    if state['phase']=='finished':
        return dict(complete=True,value=core.terminal_value(state,viewer),
                    winner=state.get('winner'),actions=trace)
    return dict(complete=False,reason='action_limit',actions=trace)


def paired_summary(records, actions, reference, *, expected_worlds, alpha=.05):
    """One hidden-world/continuation seed is a cluster; never count branches as n."""
    actions=tuple(actions)
    if not actions or len(set(actions))!=len(actions) or reference not in actions:
        raise ValueError('Distinct branches including reference required')
    if type(expected_worlds) is not int or expected_worlds<1 or not 0<alpha<1:
        raise ValueError('Invalid fixed counterfactual endpoint')
    if len(records)>expected_worlds:raise ValueError('Too many counterfactual worlds')
    seen=set();complete=[]
    for row in records:
        if row['world_seed'] in seen:raise ValueError('Duplicate world cluster')
        seen.add(row['world_seed'])
        if set(row['branches'])!=set(actions):raise ValueError('Missing or extra branch')
        if all(b['complete'] for b in row['branches'].values()):
            if any(b.get('value') not in (-1.,0.,1.) for b in row['branches'].values()):
                raise ValueError('Only real terminal WDL permitted')
            complete.append(row)
    n=len(complete);comparisons=max(1,len(actions)-1)
    # Hoeffding for paired WDL differences in [-2,2], union bound across branches.
    radius=4*math.sqrt(math.log(2*comparisons/alpha)/(2*n)) if n else None
    metrics={}
    for action in actions:
        outcomes=[r['branches'][action]['value'] for r in complete]
        differences=[r['branches'][action]['value']-r['branches'][reference]['value'] for r in complete]
        mean=float(np.mean(differences)) if n else None
        metrics[action]=dict(mean_wdl=float(np.mean(outcomes)) if n else None,
            paired_gain=mean,paired_interval=([max(-2.,mean-radius),min(2.,mean+radius)]
                if n and action!=reference else ([0.,0.] if n else None)),
            wins=sum(v==1 for v in outcomes),draws=sum(v==0 for v in outcomes),losses=sum(v==-1 for v in outcomes))
    return dict(complete=len(records)==expected_worlds and n==expected_worlds,
        planned_worlds=expected_worlds,complete_paired_worlds=n,incomplete_worlds=len(records)-n,
        branches=metrics,alpha=alpha,interval='paired_WDL_Hoeffding_family_corrected',
        independent_strength_confirmation=False,quality_approved=False)


def counterfactuals(state, viewer, actions, policies, runtime, *, world_seed,
                    continuation_seed, worlds, deadline, max_actions=512):
    if type(worlds) is not int or worlds<1 or type(max_actions) is not int or max_actions<1:
        raise ValueError('Positive bounded world/action counts required')
    legal,(x,c)=runtime.decision(state,viewer)
    indexes=tuple(actions)
    if len(set(indexes))!=len(indexes) or any(type(i) is not int or not 0<=i<len(legal) for i in indexes):
        raise ValueError('Invalid root branch indexes')
    records=[]
    for i in range(worlds):
        if time.time()>=deadline:break
        world=core.sample_world(state,viewer,world_seed+i,runtime=runtime)
        sampled,(xx,cc)=runtime.decision(world,viewer)
        if sampled!=legal or not np.array_equal(x,xx) or not np.array_equal(c,cc):
            raise ValueError('Counterfactual world changed visible root/actions')
        branches={}
        for index in indexes:
            branches[index]=continuation(world,viewer,legal[index],policies,runtime,
                seed=continuation_seed+i,deadline=deadline,max_actions=max_actions)
        records.append(dict(world_seed=world_seed+i,continuation_seed=continuation_seed+i,branches=branches))
        if any(not b['complete'] for b in branches.values()):break
    return records
