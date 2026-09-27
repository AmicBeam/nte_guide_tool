"""Frozen root counterfactuals; no optimization or model promotion."""
from copy import deepcopy
from collections import Counter
import math
import time
import numpy as np
from . import information_search as core
from .league_rollout import clean
from ..engine.duel_v2 import acting_side, apply_action


def sampled_build_support(world,policies):
    """Audit sampled card pools against frozen build provenance, not policy skill.

    Source builds never constrain the sampler or enter observations here.
    Counts omit explicitly derived cards; unflagged generated copies can also
    cause mismatches, so this must not automatically choose a rollout policy.
    """
    from ..content.duel_v2 import CARDS
    result={}
    for side,policy in policies.items():
        source=Counter(policy.serving_deck['card_ids'])
        team=world['sides'][side]
        cards=[c for zone in ('hand','deck','discard') for c in team[zone]
               if not c.get('derived') and not CARDS[c['card_id']].get('derived')]
        sampled=Counter(c['card_id'] for c in cards)
        excess={cid:n-source[cid] for cid,n in sampled.items() if n>source[cid]}
        result[side]=dict(pool_cards=sum(sampled.values()),source_cards=sum(source.values()),
            compatible_remaining_pool=not excess,excess_counts=excess,
            policy_capability_proven=False,unflagged_generation_may_confound=True)
    return result


def continuation(world, viewer, action, policies, runtime, *, seed, deadline, max_actions=512,
                 capture_value=False):
    """Apply one root action, then run visible-policy play to a real terminal."""
    if time.time()>=deadline:return dict(complete=False,reason='deadline',actions=[])
    state=clean(apply_action(deepcopy(world),viewer,action))
    prediction={}
    if capture_value:
        terminal=state['phase']=='finished'
        estimate=(core.terminal_value(state,viewer) if terminal else runtime.value_for(state,viewer,policies))
        if estimate is None or not np.isfinite(estimate) or abs(estimate)>1+1e-6:
            raise ValueError('Invalid viewer-perspective WDL estimate')
        prediction=dict(predicted_value=float(estimate),prediction_source='terminal' if terminal else 'network')
    trace=[dict(side=viewer,action=action)]
    rng=np.random.default_rng(seed)
    for _ in range(max_actions):
        if state['phase']=='finished':
            return dict(complete=True,value=core.terminal_value(state,viewer),
                        winner=state.get('winner'),actions=trace,**prediction)
        if time.time()>=deadline:
            return dict(complete=False,reason='deadline',actions=trace,**prediction)
        actor=acting_side(state);legal,(x,c)=runtime.decision(state,actor)
        scores=runtime.action_scores(policies[actor],x,c)
        index=int(rng.choice(len(legal),p=core.softmax(scores)))
        state=clean(apply_action(state,actor,legal[index]))
        trace.append(dict(side=actor,action=legal[index]))
    if state['phase']=='finished':
        return dict(complete=True,value=core.terminal_value(state,viewer),
                    winner=state.get('winner'),actions=trace,**prediction)
    return dict(complete=False,reason='action_limit',actions=trace,**prediction)


def value_diagnostics(records, actions):
    """One-ply value vs conditional frozen-policy outcomes, not calibration approval."""
    complete=[r for r in records if all(b['complete'] for b in r['branches'].values())]
    metrics={}
    for action in actions:
        branches=[r['branches'][action] for r in complete]
        if any(b.get('prediction_source') not in ('network','terminal')
               or not np.isfinite(b.get('predicted_value',np.nan))
               or abs(b['predicted_value'])>1+1e-6
               or (b['prediction_source']=='terminal' and b['predicted_value']!=b['value'])
               for b in branches):
            raise ValueError('Missing or invalid value prediction provenance')
        learned=[b for b in branches if b.get('prediction_source')=='network']
        if any(not np.isfinite(b['predicted_value']) or abs(b['predicted_value'])>1+1e-6
               or b['value'] not in (-1.,0.,1.) for b in learned):
            raise ValueError('Invalid value diagnostic')
        errors=[b['predicted_value']-b['value'] for b in learned]
        metrics[action]=dict(network_rows=len(learned),terminal_rows=len(branches)-len(learned),
            predicted_mean=float(np.mean([b['predicted_value'] for b in learned])) if learned else None,
            actual_mean=float(np.mean([b['value'] for b in learned])) if learned else None,
            mean_error=float(np.mean(errors)) if learned else None,
            mean_squared_error=float(np.mean(np.square(errors))) if learned else None)
    comparable=[a for a in actions if metrics[a]['network_rows']==len(complete) and complete]
    pairs=[]
    for i,left in enumerate(comparable):
        for right in comparable[i+1:]:
            predicted=metrics[left]['predicted_mean']-metrics[right]['predicted_mean']
            actual=metrics[left]['actual_mean']-metrics[right]['actual_mean']
            pairs.append(dict(left=left,right=right,predicted_difference=predicted,
                actual_difference=actual,opposite_sign=bool(predicted*actual<0)))
    return dict(branches=metrics,pairwise_order=pairs,complete_paired_worlds=len(complete),
        continuation_distribution='frozen_visible_policy_softmax',
        calibration_approved=False,search_policy_distribution_mismatch_possible=True)


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
                    continuation_seed, worlds, deadline, max_actions=512,capture_value=False):
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
                seed=continuation_seed+i,deadline=deadline,max_actions=max_actions,capture_value=capture_value)
        record=dict(world_seed=world_seed+i,continuation_seed=continuation_seed+i,branches=branches)
        if capture_value:record['sampled_build_support']=sampled_build_support(world,policies)
        records.append(record)
        if any(not b['complete'] for b in branches.values()):break
    return records


def verify_counterfactuals(state,viewer,records,runtime,*,deadline,policies=None):
    """Replay private terminal branches with fresh clones of each sampled world."""
    verified=0
    for record in records:
        if time.time()>=deadline:return dict(complete=False,verified_branches=verified,reason='deadline')
        world=core.sample_world(state,viewer,record['world_seed'],runtime=runtime)
        legal,_=runtime.decision(world,viewer)
        for index,branch in record['branches'].items():
            if not branch['complete']:return dict(complete=False,verified_branches=verified,reason='unfinished_branch')
            trace=branch['actions']
            if not trace or trace[0]!=dict(side=viewer,action=legal[int(index)]):
                raise ValueError('Counterfactual root action mismatch')
            replay=deepcopy(world)
            for step,entry in enumerate(trace):
                if time.time()>=deadline:return dict(complete=False,verified_branches=verified,reason='deadline')
                if replay['phase']=='finished' or acting_side(replay)!=entry['side']:
                    raise ValueError('Counterfactual replay actor mismatch')
                available,_=runtime.decision(replay,entry['side'])
                if entry['action'] not in available:raise ValueError('Counterfactual replay illegal action')
                replay=clean(apply_action(replay,entry['side'],entry['action']))
                if step==0 and policies is not None:
                    terminal=replay['phase']=='finished'
                    value=core.terminal_value(replay,viewer) if terminal else runtime.value_for(replay,viewer,policies)
                    if (branch.get('prediction_source')!=('terminal' if terminal else 'network')
                            or not np.isclose(value,branch.get('predicted_value',np.nan),atol=1e-7,rtol=0)):
                        raise ValueError('Counterfactual value prediction replay mismatch')
            if (replay['phase']!='finished' or replay.get('winner')!=branch['winner']
                    or core.terminal_value(replay,viewer)!=branch['value']):
                raise ValueError('Counterfactual terminal replay mismatch')
            verified+=1
    return dict(complete=True,verified_branches=verified)
