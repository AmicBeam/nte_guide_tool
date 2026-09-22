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


def sample_world(state, viewer, seed, *, runtime=None):
    decide = runtime.decision if runtime else decision
    sample = runtime.determinize if runtime else determinize
    if state['phase'] in ('playing', 'choice'):
        return sample(state,viewer,seed)
    if state['phase']!='mulligan':
        raise ValueError('Unsupported information-search root phase')
    # Mulligan uses the same known-card accounting, but retain its original phase
    # and completion flags after sampling; no actual hidden shuffle is consulted.
    proxy=deepcopy(state);proxy['phase']='playing';proxy['active_side']=viewer
    out=sample(proxy,viewer,seed);out['phase']='mulligan';out['active_side']=state['active_side']
    a,(x,c)=decide(state,viewer);_,(xx,cc)=decide(out,viewer)
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

    def choose(self, minimum_visits=1):
        unseen=np.flatnonzero(self.visits<minimum_visits)
        if len(unseen):
            least=unseen[self.visits[unseen]==self.visits[unseen].min()]
            return int(least[np.argmax(self.prior[least])])
        q=self.total/self.visits
        u=1.5*self.prior*np.sqrt(self.visits.sum())/(1+self.visits)
        return int(np.argmax(q+u))



def search_prior(logits, mode='model', temperature=1.0, uniform_mix=0.0):
    if mode not in ('model','uniform','tempered') or not np.isfinite(temperature) or temperature<=0 or not 0<=uniform_mix<=1:
        raise ValueError('Invalid search prior settings')
    if mode=='uniform':return np.full(len(logits),1/len(logits),dtype=np.float64)
    prior=softmax(np.asarray(logits)/temperature if mode=='tempered' else logits)
    return (1-uniform_mix)*prior+uniform_mix/len(prior)

def terminal_value(state,viewer):
    winner=state.get('winner')
    return 0. if winner not in ('a','b') else 1. if winner==viewer else -1.


def leaf_value(state, viewer, policies, *, runtime=None):
    decide = runtime.decision if runtime else decision
    if state['phase']=='finished':return terminal_value(state,viewer)
    if runtime and hasattr(runtime, 'value_for'):
        value=runtime.value_for(state,viewer,policies)
        if value is not None:return value
    actor=acting_side(state);_,(x,c)=decide(state,actor)
    _,raw=policies[actor].scores_value(x,c)
    value=float(np.tanh(raw/10))
    # Card actions can leave the SAME player acting: never negate by ply count.
    return value if actor==viewer else -value


def search(state,viewer,policies,*,seed,simulations=32,max_depth=10,deadline=None,noise=False,runtime=None,passive_reward=0.0,prior_mode="model",prior_temperature=1.0,uniform_mix=0.0,root_min_visits=1,node_min_visits=1,setup_reward=0.0,setup_already_achieved=False):
    decide = runtime.decision if runtime else decision
    if simulations<1 or max_depth<1:raise ValueError('Positive search limits required')
    if acting_side(state)!=viewer:raise ValueError('Search must use the decision owner')
    if not np.isfinite(passive_reward) or passive_reward < 0:
        raise ValueError('Invalid passive reward')
    if type(root_min_visits)!=int or type(node_min_visits)!=int or min(root_min_visits,node_min_visits)<1:
        raise ValueError('Positive integer minimum visits required')
    from .setup_reward import ready as setup_ready
    if not np.isfinite(setup_reward) or setup_reward<0:raise ValueError('Invalid setup reward')
    setup_initial=bool(setup_already_achieved or (setup_reward and setup_ready(state,viewer)))
    initial_passive = runtime.passive_count(state,viewer) if runtime and passive_reward else 0
    actions,(x,c)=decide(state,viewer)
    logits,raw=policies[viewer].scores_value(x,c);prior=search_prior(logits,prior_mode,prior_temperature,uniform_mix)
    rng=np.random.default_rng(seed)
    if noise and len(actions)>1:
        prior=.75*prior+.25*rng.dirichlet(np.full(len(actions),.3))
    root=Node.create(prior);root_key=info_key(x,c)+(bytes([setup_initial]) if setup_reward else b'');tree={root_key:root}
    requested=1 if len(actions)==1 else max(simulations,len(actions)+8)
    if len(actions)>1 and root_min_visits*len(actions)>requested:
        raise ValueError("Search budget cannot cover requested root exploration floor")
    depths=[];worlds=0
    for simulation in range(requested):
        if deadline is not None and time.time()>=deadline:break
        world=sample_world(state,viewer,seed+simulation*104729,runtime=runtime)
        worlds+=1;path=[];value=0.;finished_sim=True;setup_seen=setup_initial
        for depth in range(max_depth):
            if deadline is not None and time.time()>=deadline:
                finished_sim=False;break
            if world['phase']=='finished':
                value=terminal_value(world,viewer);break
            actor=acting_side(world);legal,(xx,cc)=decide(world,actor)
            if actor!=viewer:
                # Never use a search Q conditioned on the root player's hand to
                # choose for the opponent. Its visible-policy is the response model.
                scores,_=policies[actor].scores_value(xx,cc)
                chosen=int(rng.choice(len(legal),p=softmax(scores)))
            else:
                key=info_key(xx,cc)+(bytes([setup_seen]) if setup_reward else b'')
                if key not in tree:
                    scores,v=policies[actor].scores_value(xx,cc)
                    tree[key]=Node.create(search_prior(scores,prior_mode,prior_temperature,uniform_mix));value=leaf_value(world,viewer,policies,runtime=runtime) if runtime and hasattr(runtime,'value_for') else float(np.tanh(v/10));break
                node=tree[key];chosen=node.choose(root_min_visits if node is root else node_min_visits);path.append((node,chosen))
            world=clean(apply_action(world,actor,legal[chosen]))
            if setup_reward:setup_seen=setup_seen or setup_ready(world,viewer)
        else:
            value=leaf_value(world,viewer,policies,runtime=runtime)
        if not finished_sim:break
        if runtime and passive_reward:
            value += passive_reward/10 * (runtime.passive_count(world,viewer)-initial_passive)
        value+=setup_reward/10*int(setup_seen and not setup_initial)
        for node,chosen in path:
            node.visits[chosen]+=1;node.total[chosen]+=value
        depths.append(depth+1)
    visits=root.visits.copy();complete=len(depths)==requested
    pi=visits/visits.sum() if visits.sum() else prior.copy()
    return dict(actions=actions,x=x,c=c,pi=pi.astype(np.float32),visits=visits,
                mean_values=np.divide(root.total,visits,out=np.zeros(len(visits)),where=visits>0),
                complete=complete,simulations=len(depths),requested=requested,
                roots_covered=int((visits>0).sum()),root_min_visits=int(visits.min()),prior_mode=prior_mode,nodes=len(tree),worlds=worlds,
                max_depth=max(depths,default=0),raw_choice=int(np.argmax(logits)),
                search_choice=int(np.argmax(visits)),value=leaf_value(state,viewer,policies,runtime=runtime) if runtime and hasattr(runtime,'value_for') else float(np.tanh(raw/10)))


def decision_budget(job, rng):
    """Independent budget RNG; shallow positions never become training labels."""
    mix=job.get('search_mix')
    if mix is None:return job.get('simulations',32),True
    low,high,probability=mix['low'],mix['high'],mix['deep_probability']
    if type(low)!=int or type(high)!=int or not 1<=low<=high or not 0<probability<=1:
        raise ValueError('Invalid search budget mixture')
    deep=bool(rng.random()<probability)
    return high if deep else low,deep


def search_game(job):
    """Full real trajectory; BOTH sides search, including opening mulligans.

    Only completed games emit training rows. Every row has its own acting-side
    outcome and visit target. Optional training rewards remain separate from wins.
    """
    runtime = None
    if job.get('runtime') == 'ten':
        from . import ten_search_runtime as runtime
    elif job.get('runtime') == 'outcome':
        from . import outcome_runtime as runtime
    elif job.get('runtime') is not None:
        raise ValueError('Unknown search runtime')
    loader = runtime.model if runtime else model
    decide = runtime.decision if runtime else decision
    state=new_game(seed=job['seed'],first_side=job.get('first','a'),decks=job['decks'])
    policies={side:loader(*spec) for side,spec in job['policies'].items()}
    rng=np.random.default_rng(job['seed']);rows=[];stats=[];decisions=0;trace=[];value_rows=[]
    budget_rng=np.random.default_rng(job['seed'] ^ 0x5A17)
    coefficient=float(job.get('zhenhong_passive_reward',0.0)) if job.get('training') else 0.0
    if not np.isfinite(coefficient) or coefficient < 0: raise ValueError('Invalid reward coefficient')
    from .setup_reward import ready as setup_ready
    setup_seen={side:setup_ready(state,side) for side in ('a','b')} if runtime else {'a':False,'b':False}
    setup_initial_seen=dict(setup_seen)
    setup_coefficients=job.get('setup_rewards',{}) if job.get('training') else {}
    if any(not np.isfinite(v) or v<0 for v in setup_coefficients.values()):raise ValueError('Invalid setup reward')
    for step in range(job.get('max_actions',400)):
        if state['phase']=='finished':break
        if time.time()>=job['deadline']:
            return dict(complete=False,reason='deadline',rows=[],decisions=decisions)
        actor=acting_side(state)
        budget,save_labels=decision_budget(job,budget_rng)
        if job.get('record_episode') and save_labels:
            from ..engine.duel_v2 import observe
            for owner in job.get('train_sides',['a']):
                if owner!=actor or job.get('collect_value_only'):
                    from .fixed_lineup import encode
                    vx,_=encode(observe(state,owner,include_previews=False),[])
                    value_rows.append(dict(x=vx,side=owner,key=job['policies'][owner][1],step=step,value_actor=int(owner==actor)))
        use_search=actor not in job.get('raw_sides',[])
        if use_search:
            try:
                result=search(state,actor,policies,seed=job['seed']+1000003*(step+1),
                              simulations=budget,deadline=job['deadline'],
                              noise=job.get('training',False) and save_labels,**({'runtime':runtime,'passive_reward':coefficient if job['policies'][actor][1]=='zhenhong' and actor in job.get('train_sides',['a','b']) else 0.0,'setup_reward':setup_coefficients.get(actor,0.) if job['policies'][actor][1]=='zhenhong' and actor in job.get('train_sides',['a','b']) else 0.,'setup_already_achieved':setup_seen[actor]} if runtime else {}))
            except Exception as exc:
                if job.get('failure_dir'):
                    import json
                    from pathlib import Path
                    directory=Path(job['failure_dir']);directory.mkdir(parents=True,exist_ok=True)
                    label=hashlib.sha256(repr((job['seed'],job.get('first'),job['policies'])).encode()).hexdigest()[:16]
                    path=directory/(label+'.json');temp=path.with_suffix('.tmp')
                    temp.write_text(json.dumps(dict(job=job,step=step,actor=actor,error=repr(exc),state=state),ensure_ascii=False),encoding='utf-8')
                    temp.replace(path)
                raise
            if not result['complete']:
                return dict(complete=False,reason='incomplete_search',rows=[],decisions=decisions)
            distribution=result['pi'].astype(float);distribution/=distribution.sum()
            chosen=int(rng.choice(len(result['actions']),p=distribution)) if job.get('training') and save_labels else result['search_choice']
            actions=result['actions']
            if job.get('collect') and save_labels:
                rows.append(dict(x=result['x'],c=result['c'],pi=result['pi'],side=actor,
                                 key=job['policies'][actor][1],phase=state['phase'],step=step,value_actor=1,search_budget=budget))
                if runtime and coefficient and job['policies'][actor][1]=='zhenhong' and actor in job.get('train_sides',['a','b']):
                    rows[-1]['passive_before']=runtime.passive_count(state,actor)
                if runtime and job['policies'][actor][1]=='zhenhong' and actor in job.get('train_sides',['a','b']) and setup_coefficients.get(actor,0)>0:
                    rows[-1].update(setup_before=int(setup_seen[actor]),setup_reward=setup_coefficients[actor])
            stats.append(dict(side=actor,phase=state['phase'],simulations=result['simulations'],
                              requested_budget=budget,policy_label=save_labels,legal=len(actions),covered=result['roots_covered'],nodes=result['nodes'],
                              changed=result['raw_choice']!=result['search_choice'],
                              value_estimate=result['value']))
        else:
            actions,(x,c)=decide(state,actor);chosen=int(policies[actor].scores(x,c).argmax())
        if job.get('record_episode'):trace.append(dict(side=actor,action=deepcopy(actions[chosen])))
        state=clean(apply_action(state,actor,actions[chosen]));decisions+=1
        if runtime:
            for side in ('a','b'):setup_seen[side]=setup_seen[side] or setup_ready(state,side)
    complete=state['phase']=='finished'
    if complete:
        for row in rows+value_rows:
            row['z']=terminal_value(state,row['side'])
            if 'passive_before' in row:
                row['reward_to_go']=10*row['z']+coefficient*(runtime.passive_count(state,row['side'])-row.pop('passive_before'))
            if 'setup_before' in row:
                row['setup_future']=int(setup_seen[row['side']])-row.pop('setup_before')
    return dict(complete=complete,winner=state.get('winner'),rows=rows if complete else [],
                value_rows=value_rows if complete else [],episode_actions=trace if complete else [],
                model_versions={s:getattr(p,'version',None) for s,p in policies.items()},
                decisions=decisions,searched=len(stats),search_stats=stats,turn=state['turn'],
                reason=None if complete else 'action_limit',
                passive_triggers={s:runtime.passive_count(state,s) for s in ('a','b')} if runtime else {},
                setup_achieved=setup_seen,setup_rewards={s:setup_coefficients.get(s,0)*int(setup_seen[s] and not setup_initial_seen[s]) if job['policies'][s][1]=='zhenhong' and s in job.get('train_sides',['a','b']) else 0. for s in ('a','b')})


def policy_value_update(net,opt,rows,*,setup_coefficient=None):
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
    from .setup_reward import learning_return
    target=torch.tensor([learning_return(r,setup_coefficient)/10 for r in rows],dtype=torch.float32,device=x.device)
    shaped=torch.tensor([r.get('reward_to_go') is not None or (r.get('setup_reward',0) if setup_coefficient is None else setup_coefficient)>0 for r in rows],dtype=torch.bool,device=x.device)
    if not torch.isfinite(target).all(): raise ValueError('Invalid reward-to-go')
    prediction=torch.where(shaped,value/10,torch.tanh(value/10))
    value_loss=(prediction-target).square().mean()
    loss=policy_loss+value_loss
    if not torch.isfinite(loss):raise ValueError('Non-finite policy/value loss')
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                samples=len(rows),optimizer_steps=1)
