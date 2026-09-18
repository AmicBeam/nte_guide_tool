"""Single-observer information-set PUCT with observation-only opponent policy.

Actual players both search at every decision. Inside a search the opponent uses
its frozen observation-only policy, not Q values conditioned on our private hand.
This is an approximate belief sampler, not a ReBeL equilibrium solver.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .league_rollout import decision, determinize, clean, model
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side


def softmax(scores):
    p=np.exp(np.asarray(scores,dtype=np.float64)-np.max(scores))
    return p/p.sum()


def sample_world(state, viewer, seed):
    if state['phase']=='playing':
        return determinize(state,viewer,seed)
    if state['phase']!='mulligan':
        raise ValueError('Root pending-choice belief sampling is not implemented for this pilot')
    # Mulligan uses the same known-card accounting, but retain its original phase
    # and completion flags after sampling; no actual hidden shuffle is consulted.
    proxy=deepcopy(state);proxy['phase']='playing';proxy['active_side']=viewer
    out=determinize(proxy,viewer,seed);out['phase']='mulligan';out['active_side']=state['active_side']
    a,(x,c)=decision(state,viewer);_,(xx,cc)=decision(out,viewer)
    if not np.array_equal(x,xx) or not np.array_equal(c,cc):
        raise ValueError('Mulligan sample changed visible information')
    return out


def info_key(x,c):
    return hashlib.sha256(x.tobytes()+c.tobytes()).digest()


@dataclass
class Node:
    prior: np.ndarray
    visits: np.ndarray
    total: np.ndarray

    @classmethod
    def create(cls,prior):
        return cls(prior,np.zeros(len(prior),dtype=np.int32),np.zeros(len(prior)))

    def choose(self):
        unseen=np.flatnonzero(self.visits==0)
        if len(unseen):
            return int(unseen[np.argmax(self.prior[unseen])])
        q=self.total/self.visits
        u=1.5*self.prior*np.sqrt(self.visits.sum())/(1+self.visits)
        return int(np.argmax(q+u))


def terminal_value(state,viewer):
    winner=state.get('winner')
    return 0. if winner not in ('a','b') else 1. if winner==viewer else -1.


def leaf_value(state, viewer, policies):
    if state['phase']=='finished':return terminal_value(state,viewer)
    actor=acting_side(state);_,(x,c)=decision(state,actor)
    _,raw=policies[actor].scores_value(x,c)
    value=float(np.tanh(raw/10))
    # Card actions can leave the SAME player acting: never negate by ply count.
    return value if actor==viewer else -value


def search(state,viewer,policies,*,seed,simulations=32,max_depth=10,deadline=None,noise=False):
    if simulations<1 or max_depth<1:raise ValueError('Positive search limits required')
    if acting_side(state)!=viewer:raise ValueError('Search must use the decision owner')
    actions,(x,c)=decision(state,viewer)
    logits,raw=policies[viewer].scores_value(x,c);prior=softmax(logits)
    rng=np.random.default_rng(seed)
    if noise and len(actions)>1:
        prior=.75*prior+.25*rng.dirichlet(np.full(len(actions),.3))
    root=Node.create(prior);tree={info_key(x,c):root}
    requested=1 if len(actions)==1 else max(simulations,len(actions)+8)
    depths=[];worlds=0
    for simulation in range(requested):
        if deadline is not None and time.time()>=deadline:break
        world=sample_world(state,viewer,seed+simulation*104729)
        worlds+=1;path=[];value=0.;finished_sim=True
        for depth in range(max_depth):
            if deadline is not None and time.time()>=deadline:
                finished_sim=False;break
            if world['phase']=='finished':
                value=terminal_value(world,viewer);break
            actor=acting_side(world);legal,(xx,cc)=decision(world,actor)
            if actor!=viewer:
                # Never use a search Q conditioned on the root player's hand to
                # choose for the opponent. Its visible-policy is the response model.
                scores,_=policies[actor].scores_value(xx,cc)
                chosen=int(rng.choice(len(legal),p=softmax(scores)))
            else:
                key=info_key(xx,cc)
                if key not in tree:
                    scores,v=policies[actor].scores_value(xx,cc)
                    tree[key]=Node.create(softmax(scores));value=float(np.tanh(v/10));break
                node=tree[key];chosen=node.choose();path.append((node,chosen))
            world=clean(apply_action(world,actor,legal[chosen]))
        else:
            value=leaf_value(world,viewer,policies)
        if not finished_sim:break
        for node,chosen in path:
            node.visits[chosen]+=1;node.total[chosen]+=value
        depths.append(depth+1)
    visits=root.visits.copy();complete=len(depths)==requested
    pi=visits/visits.sum() if visits.sum() else prior.copy()
    return dict(actions=actions,x=x,c=c,pi=pi.astype(np.float32),visits=visits,
                mean_values=np.divide(root.total,visits,out=np.zeros(len(visits)),where=visits>0),
                complete=complete,simulations=len(depths),requested=requested,
                roots_covered=int((visits>0).sum()),nodes=len(tree),worlds=worlds,
                max_depth=max(depths,default=0),raw_choice=int(np.argmax(logits)),
                search_choice=int(np.argmax(visits)),value=float(np.tanh(raw/10)))


def search_game(job):
    """Full real trajectory; BOTH sides search, including opening mulligans.

    Only completed games emit training rows. Every row has its own acting-side
    outcome and visit target. No heuristic or timing rewards are used.
    """
    state=new_game(seed=job['seed'],first_side=job.get('first','a'),decks=job['decks'])
    policies={side:model(*spec) for side,spec in job['policies'].items()}
    rng=np.random.default_rng(job['seed']);rows=[];stats=[];decisions=0
    for step in range(job.get('max_actions',400)):
        if state['phase']=='finished':break
        if time.time()>=job['deadline']:
            return dict(complete=False,reason='deadline',rows=[],decisions=decisions)
        actor=acting_side(state)
        use_search=actor not in job.get('raw_sides',[])
        if use_search:
            result=search(state,actor,policies,seed=job['seed']+1000003*(step+1),
                          simulations=job.get('simulations',32),deadline=job['deadline'],
                          noise=job.get('training',False))
            if not result['complete']:
                return dict(complete=False,reason='incomplete_search',rows=[],decisions=decisions)
            distribution=result['pi'].astype(float);distribution/=distribution.sum()
            chosen=int(rng.choice(len(result['actions']),p=distribution)) if job.get('training') else result['search_choice']
            actions=result['actions']
            if job.get('collect'):
                rows.append(dict(x=result['x'],c=result['c'],pi=result['pi'],side=actor,
                                 key=job['policies'][actor][1],phase=state['phase']))
            stats.append(dict(side=actor,phase=state['phase'],simulations=result['simulations'],
                              legal=len(actions),covered=result['roots_covered'],nodes=result['nodes'],
                              changed=result['raw_choice']!=result['search_choice'],
                              value_estimate=result['value']))
        else:
            actions,(x,c)=decision(state,actor);chosen=int(policies[actor].scores(x,c).argmax())
        state=clean(apply_action(state,actor,actions[chosen]));decisions+=1
    complete=state['phase']=='finished'
    if complete:
        for row in rows:row['z']=terminal_value(state,row['side'])
    return dict(complete=complete,winner=state.get('winner'),rows=rows if complete else [],
                decisions=decisions,searched=len(stats),search_stats=stats,turn=state['turn'],
                reason=None if complete else 'action_limit')


def policy_value_update(net,opt,rows):
    import torch
    from .league_learning import tensors
    if not rows:raise ValueError('No completed trajectory samples')
    x,c,m=tensors([(r['x'],r['c']) for r in rows],next(net.parameters()).device)
    pi=torch.zeros(m.shape,device=x.device)
    for i,r in enumerate(rows):
        p=np.asarray(r['pi'])
        if len(p)!=len(r['c']) or np.any(p<0) or not np.isfinite(p).all() or not np.isclose(p.sum(),1):
            raise ValueError('Invalid search policy target')
        pi[i,:len(p)]=torch.as_tensor(p,device=x.device)
    z=torch.tensor([r['z'] for r in rows],dtype=torch.float32,device=x.device)
    if not torch.all((z==-1)|(z==0)|(z==1)):raise ValueError('Non-terminal value target')
    logits,value=net(x,c,m);logp=logits.log_softmax(-1)
    policy_loss=-(pi*logp.masked_fill(~m,0)).sum(-1).mean()
    value_loss=(torch.tanh(value/10)-z).square().mean()
    loss=policy_loss+value_loss
    if not torch.isfinite(loss):raise ValueError('Non-finite policy/value loss')
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                samples=len(rows),optimizer_steps=1)
