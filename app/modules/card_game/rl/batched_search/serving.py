"""Cooperative serving planner: public proof and Gumbel share a 32-work budget.

Matches the current recovery serving surrogate: inside each search both sides
use the root bot's frozen model. Physical opponents search with their own model.
"""
from copy import deepcopy
from hashlib import sha256
import math
import time
from uuid import uuid4
import numpy as np
from .cooperative import _operation,cooperative_search
from .traversal import InferenceQuery,_unpack_response,_get_policy_identity,TraversalCancelled,_DeadlineReached,RootEvaluation


def serving_search(state,side,model,*,backend,budget_mode='simulations',seconds=3.,
                   simulations=32,proof_cap=8,deadline=None,root_id=None,queued_at=None,
                   statistics=None,clock=time.monotonic):
    """Yield exact backend/neural queries; no CPU or hidden-state shortcut."""
    if budget_mode not in ('simulations','wall_clock'):raise ValueError('Explicit serving budget mode required')
    if budget_mode=='wall_clock' and queued_at is None:
        raise ValueError('Wall-clock roots require an explicit queue timestamp')
    if type(simulations) is not int or simulations<1:raise ValueError('Positive simulation count required')
    if type(proof_cap) is not int or not 0<=proof_cap<simulations:raise ValueError('Proof allowance must leave tree work')
    if isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Positive finite decision seconds required')
    now=clock();start=now if queued_at is None else queued_at;end=deadline
    if (isinstance(start,bool) or not isinstance(start,(int,float)) or
            not math.isfinite(start) or start>now):
        raise ValueError('Queue timestamp must be finite and no later than root execution')
    if deadline is not None and (isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not math.isfinite(deadline)):
        raise ValueError('Finite absolute monotonic deadline required')
    if budget_mode=='wall_clock':end=min(start+seconds,end) if end is not None else start+seconds
    key,version=_get_policy_identity({side:model},side)
    token=str(root_id or uuid4().hex)+':serving:'+uuid4().hex
    actions=[];fallback=None;checks=0;proof_attempts=0;remaining=simulations;tree=None
    def expired():return end is not None and clock()>=end
    def result(reason,selection=None):
        selected=fallback if selection is None else selection
        finished=clock()
        out=dict(tree or {})
        out.update(actions=actions,selected_action=deepcopy(actions[selected]) if selected is not None else None,
            search_choice=selected,proof_checks=checks,tree_simulations=int((tree or {}).get('simulations',0)),
            proof_attempts=proof_attempts,
            total_simulations=checks+int((tree or {}).get('simulations',0)),budget_simulations=simulations,
            budget_mode=budget_mode,stop_reason=reason,wall_seconds=finished-start,finished_monotonic=finished,
            tree_stop_reason=(tree or {}).get('stop_reason'),tree_complete=bool((tree or {}).get('complete',False)),
            deadline_reached=end is not None and finished>=end,
            used_fallback=selection is None and reason in ('unsupported_root','no_complete_tree','deadline','cancelled'),
            surrogate_model_key=key,surrogate_model_version=version)
        return out
    try:
        if expired():return result('deadline')
        actor=yield from _operation(backend,'acting_side',state)
        if actor!=side:raise ValueError('Serving root is not owned by the observer')
        actions,(x,c)=yield from _operation(backend,'decision',state,side)
        if not actions:raise ValueError('No legal serving actions')
        if len(actions)==1:return result('forced_action',0)
        query=InferenceQuery(token,key,version,x,c,True,side)
        response=yield query
        scores,_=_unpack_response(response,query,'natural_wdl');fallback=int(np.argmax(scores))
        if expired():return result('deadline')
        metadata=yield from _operation(backend,'serving_root_metadata',state,side)
        if not metadata['search_supported']:return result('unsupported_root')
        seed=int.from_bytes(sha256(f"serving-search-v1:{metadata['version']}:{side}:{version}".encode()).digest(),'big')
        if metadata['public_terminal_allowed'] and not expired():
            world=yield from _operation(backend,'sample_world',state,side,seed)
            world=yield from _operation(backend,'clean',world)
            for index in sorted(range(len(actions)),key=lambda i:actions[i]['type']!='attack'):
                if checks>=proof_cap or remaining<=1 or expired():break
                action=actions[index]
                if action['type'] not in ('attack','play_card','ultimate'):continue
                remaining-=1
                proof_attempts+=1
                won=yield from _operation(backend,'public_terminal_probe',world,side,action)
                if expired():return result('deadline')
                checks+=1
                if won:return result('public_terminal_win',index)
        if expired():return result('deadline')
        tree=yield from cooperative_search(state,side,{'a':model,'b':model},backend=backend,
            seed=seed,simulations=remaining,gumbel_candidates=16,terminal_horizon=3,max_depth=10,
            q_scale='natural_wdl',noise=False,deadline=end,clock=clock,root_id=token,statistics=statistics,collect_diagnostics=False,
            root_evaluation=RootEvaluation(state,side,actions,query,response))
        if tree.get('simulations',0)>0:
            action=tree['actions'][tree['search_choice']]
            if action not in actions:raise ValueError('Search selected an action outside its physical root')
            return result('cancelled' if tree.get('stop_reason')=='cancelled' else 'tree_search',actions.index(action))
        if tree.get('stop_reason')=='cancelled':return result('cancelled')
        return result('no_complete_tree')
    except TraversalCancelled as exc:
        return result('deadline' if isinstance(exc,_DeadlineReached) else 'cancelled')
