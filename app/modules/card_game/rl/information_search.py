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
from app.modules.card_game.engine.duel_v2.simulation import simulate_action


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
    raw_value: float = 0.

    @classmethod
    def create(cls,prior,raw_value=0.):
        return cls(prior,np.zeros(len(prior),dtype=np.int32),np.zeros(len(prior)),raw_value)

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


def _follow_terminal(world, viewer, policies, rng, steps, runtime, optimized, fast_simulation, deadline):
    """Play `steps` more decisions. Return the terminal value, or None if none occurs."""
    decide = runtime.decision if runtime else decision
    step = simulate_action if fast_simulation else apply_action
    for _ in range(steps + 1):
        if deadline is not None and time.time() >= deadline:
            return None
        if world['phase'] == 'finished':
            return terminal_value(world, viewer)
        if _ == steps:
            return None
        actor = acting_side(world)
        legal, (encoded, encoded_actions) = decide(world, actor)
        if actor != viewer:
            scores = runtime.action_scores(policies[actor], encoded, encoded_actions) if optimized else policies[actor].scores_value(encoded, encoded_actions)[0]
            chosen = int(rng.choice(len(legal), p=softmax(scores)))
        else:
            scores, _ = (runtime.predict_encoded(policies[actor], encoded, encoded_actions) if optimized else policies[actor].scores_value(encoded, encoded_actions))
            chosen = int(np.argmax(scores))
        world = clean(step(world, actor, legal[chosen]))
    return None


def short_terminal_value(world, viewer, policies, rng, *, scores, legal, horizon, runtime, optimized, fast_simulation, deadline):
    """Replace a leaf value only when a short real line proves the result.

    Attacks are always checked, plus the highest-scoring actions, at most eight
    unless there are more attacks. A win on any checked line is decisive for the
    side to move. A loss overrides the value head only after every legal action
    has been played to a terminal. Anything unfinished keeps the value head.
    """
    if horizon < 1 or world['phase'] == 'finished':
        return terminal_value(world, viewer) if world['phase'] == 'finished' else None
    step = simulate_action if fast_simulation else apply_action
    attacks = [i for i, action in enumerate(legal) if action.get('type') == 'attack']
    ranked = [int(i) for i in np.argsort(-np.asarray(scores, dtype=np.float64))]
    chosen = []
    for index in attacks + ranked:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= max(8, len(attacks)) and index not in attacks:
            break
    outcomes = []
    complete = True
    for index in chosen:
        if deadline is not None and time.time() >= deadline:
            return None
        nxt = clean(step(world, viewer, legal[index]))
        outcome = _follow_terminal(nxt, viewer, policies, rng, horizon - 1, runtime, optimized, fast_simulation, deadline)
        if outcome is None:
            complete = False
            continue
        outcomes.append(outcome)
        if outcome > 0:
            return 1.
    if complete and len(chosen) == len(legal) and outcomes and all(item < 0 for item in outcomes):
        return -1.
    return None


def search(state,viewer,policies,*,seed,simulations=32,max_depth=10,deadline=None,noise=False,runtime=None,passive_reward=0.0,prior_mode="model",prior_temperature=1.0,uniform_mix=0.0,root_min_visits=1,node_min_visits=1,setup_reward=0.0,setup_already_achieved=False,reuse_inference=True,fast_simulation=True,algorithm="puct",gumbel_candidates=16,terminal_horizon=3,diagnostics=False):
    decide = runtime.decision if runtime else decision
    if type(gumbel_candidates)!=int or gumbel_candidates<1:raise ValueError('Invalid Gumbel candidate limit')
    if type(terminal_horizon)!=int or terminal_horizon<0:raise ValueError('Invalid terminal horizon')
    if algorithm not in ('puct','gumbel'):raise ValueError('Unknown search algorithm')
    if algorithm=='gumbel' and (root_min_visits!=1 or node_min_visits!=1):raise ValueError('PUCT visit floors do not apply to Gumbel')
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
    optimized=reuse_inference and runtime and hasattr(runtime,'predict_encoded')
    if optimized:logits,root_value=runtime.predict_encoded(policies[viewer],x,c)
    else:
        logits,raw=policies[viewer].scores_value(x,c)
        root_value=leaf_value(state,viewer,policies,runtime=runtime) if runtime and hasattr(runtime,'value_for') else float(np.tanh(raw/10))
    prior=search_prior(logits,prior_mode,prior_temperature,uniform_mix)
    rng=np.random.default_rng(seed)
    if algorithm=='puct' and noise and len(actions)>1:
        prior=.75*prior+.25*rng.dirichlet(np.full(len(actions),.3))
    root=Node.create(prior,root_value);root_key=info_key(x,c)+(bytes([setup_initial]) if setup_reward else b'');tree={root_key:root}
    requested=1 if len(actions)==1 else simulations if algorithm=='gumbel' else max(simulations,len(actions)+8)
    if algorithm=='puct' and len(actions)>1 and root_min_visits*len(actions)>requested:
        raise ValueError("Search budget cannot cover requested root exploration floor")
    from . import search_policy as planning
    schedule=planning.visit_schedule(min(len(actions),gumbel_candidates,requested),requested) if algorithm=='gumbel' else None
    gumbel=rng.gumbel(size=len(actions)) if algorithm=='gumbel' and noise else np.zeros(len(actions))
    depths=[];worlds=0
    terminal_checks=np.zeros(len(actions),dtype=np.int32);terminal_wins=np.zeros(len(actions),dtype=np.int32)
    for simulation in range(requested):
        if deadline is not None and time.time()>=deadline:break
        world=sample_world(state,viewer,seed+simulation*104729,runtime=runtime)
        worlds+=1;path=[];value=0.;finished_sim=True;setup_seen=setup_initial;root_action=None;root_terminal_win=False
        for depth in range(max_depth):
            if deadline is not None and time.time()>=deadline:
                finished_sim=False;break
            if world['phase']=='finished':
                value=terminal_value(world,viewer);break
            actor=acting_side(world);legal,(xx,cc)=decide(world,actor)
            if actor!=viewer:
                # Never use a search Q conditioned on the root player's hand to
                # choose for the opponent. Its visible-policy is the response model.
                scores=runtime.action_scores(policies[actor],xx,cc) if optimized else policies[actor].scores_value(xx,cc)[0]
                chosen=int(rng.choice(len(legal),p=softmax(scores)))
            else:
                key=info_key(xx,cc)+(bytes([setup_seen]) if setup_reward else b'')
                if key not in tree:
                    if optimized:scores,value=runtime.predict_encoded(policies[actor],xx,cc)
                    else:
                        scores,v=policies[actor].scores_value(xx,cc)
                        value=leaf_value(world,viewer,policies,runtime=runtime) if runtime and hasattr(runtime,'value_for') else float(np.tanh(v/10))
                    tree[key]=Node.create(search_prior(scores,prior_mode,prior_temperature,uniform_mix),value)
                    remaining=max_depth-depth
                    if terminal_horizon and remaining>0 and actor==viewer:
                        proved=short_terminal_value(world,viewer,policies,rng,scores=scores,legal=legal,horizon=min(terminal_horizon,remaining),runtime=runtime,optimized=optimized,fast_simulation=fast_simulation,deadline=deadline)
                        if proved is not None:
                            value=proved
                            tree[key].raw_value=proved
                    break
                node=tree[key]
                if algorithm=='gumbel':
                    chosen=planning.choose_root(node,gumbel,schedule[simulation]) if depth==0 else planning.choose_interior(node)
                else:chosen=node.choose(root_min_visits if node is root else node_min_visits)
                path.append((node,chosen))
            world=clean((simulate_action if fast_simulation else apply_action)(world,actor,legal[chosen]))
            if depth==0:
                root_action=chosen;root_terminal_win=world['phase']=='finished' and world.get('winner')==viewer
            if setup_reward:setup_seen=setup_seen or setup_ready(world,viewer)
        else:
            value=leaf_value(world,viewer,policies,runtime=runtime)
        if not finished_sim:break
        if root_action is not None:
            terminal_checks[root_action]+=1;terminal_wins[root_action]+=int(root_terminal_win)
        if runtime and passive_reward:
            value += passive_reward/10 * (runtime.passive_count(world,viewer)-initial_passive)
        value+=setup_reward/10*int(setup_seen and not setup_initial)
        for node,chosen in path:
            node.visits[chosen]+=1;node.total[chosen]+=value
        depths.append(depth+1)
    visits=root.visits.copy();complete=len(depths)==requested
    visit_pi=visits/visits.sum() if visits.sum() else prior.copy()
    pi=planning.improved_policy(root) if algorithm=='gumbel' else visit_pi.copy()
    policy_source='gumbel_q' if algorithm=='gumbel' else 'visits'
    selected=planning.choose_root(root,gumbel,int(visits.max())) if algorithm=='gumbel' else int(np.argmax(visits))
    return dict(actions=actions,x=x,c=c,pi=pi.astype(np.float32),visits=visits,
                mean_values=np.divide(root.total,visits,out=np.zeros(len(visits)),where=visits>0),
                visit_pi=visit_pi,root_prior=prior,root_gumbel=gumbel,terminal_checks=terminal_checks,terminal_wins=terminal_wins,policy_source=policy_source,
                complete=complete,simulations=len(depths),requested=requested,
                roots_covered=int((visits>0).sum()),root_min_visits=int(visits.min()),prior_mode=prior_mode,nodes=len(tree),worlds=worlds,
                max_depth=max(depths,default=0),raw_choice=int(np.argmax(logits)),
                search_choice=selected,value=root_value,algorithm=algorithm,considered_actions=min(len(actions),gumbel_candidates,requested) if algorithm=='gumbel' else len(actions),
                **(_root_diagnostics(actions,prior,visits,root,selected,requested,len(depths),gumbel_candidates,algorithm) if diagnostics else {}))


def action_identity(action):
    """Stable legal-action identity. Search still indexes by legal-list order."""
    import json
    return json.dumps(action, sort_keys=True, ensure_ascii=False, default=str)


def _root_diagnostics(actions, prior, visits, root, selected, requested, completed, gumbel_candidates, algorithm):
    from . import search_policy as planning
    q = planning.completed_q(root.prior, root.visits, root.total, root.raw_value) if algorithm == 'gumbel' else np.divide(root.total, visits, out=np.full(len(visits), np.nan), where=visits > 0)
    cap = min(len(actions), gumbel_candidates, requested) if algorithm == 'gumbel' else len(actions)
    admitted = np.flatnonzero(visits > 0).tolist() if algorithm == 'gumbel' else list(range(len(actions)))
    admitted_set = set(admitted)
    return dict(
        root_diagnostics=dict(
            action_identities=[action_identity(action) for action in actions],
            prior=np.asarray(prior, dtype=np.float64),
            admitted_to_max16=[index in admitted_set for index in range(len(actions))] if algorithm == 'gumbel' else [True] * len(actions),
            admitted_indexes=admitted,
            visits=np.asarray(visits, dtype=np.int32),
            completed_q=np.asarray(q, dtype=np.float64),
            completed_q_definition='search_policy.completed_q: mixed and scaled search score',
            visited_mean_value=[float(root.total[index] / count) if count else None for index, count in enumerate(visits)],
            selected_index=int(selected),
            selected_identity=action_identity(actions[selected]) if actions else None,
            requested_simulations=int(requested),
            completed_simulations=int(completed),
            candidate_cap=int(cap),
        )
    )


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
    outcome and algorithm-specific policy target. Optional training rewards remain separate from wins.
    """
    runtime = None
    if job.get('runtime') == 'ten':
        from . import ten_search_runtime as runtime
    elif job.get('runtime') == 'outcome':
        from . import outcome_runtime as runtime
    elif job.get('runtime') == 'murk':
        from . import murk_runtime as runtime
    elif job.get('runtime') == 'cross':
        from . import cross_runtime as runtime
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
    from pathlib import Path
    from . import trajectory_checkpoint as checkpoint
    from .league_schema import rule_hash
    versions={side:getattr(policy,'version',None) for side,policy in policies.items()}
    resume_path=job.get('resume_path')
    if (job.get('slice_actions') or job.get('slice_seconds')) and not resume_path:raise ValueError('Slices require a checkpoint path')
    if resume_path and not job.get('record_episode'):raise ValueError('Resume requires full private action recording')
    if job.get('slice_actions') is not None and (type(job['slice_actions'])!=int or job['slice_actions']<1):raise ValueError('Invalid action slice')
    if job.get('slice_seconds') is not None and (not np.isfinite(job['slice_seconds']) or job['slice_seconds']<=0):raise ValueError('Invalid time slice')
    if resume_path and Path(resume_path).exists():
        progress=checkpoint.load(resume_path,job,versions,rule_hash())
        state=progress['state'];rows=progress['rows'];stats=progress['stats'];trace=progress['trace'];value_rows=progress['value_rows']
        decisions=progress['decisions'];setup_seen=progress['setup_seen'];setup_initial_seen=progress['setup_initial_seen']
        rng.bit_generator.state=progress['rng'];budget_rng.bit_generator.state=progress['budget_rng']
    slice_started=time.time();slice_start_decision=decisions
    def suspend(reason):
        if resume_path:
            checkpoint.save(resume_path,job,versions,rule_hash(),dict(state=state,rows=rows,stats=stats,trace=trace,value_rows=value_rows,
                decisions=decisions,setup_seen=setup_seen,setup_initial_seen=setup_initial_seen,
                rng=rng.bit_generator.state,budget_rng=budget_rng.bit_generator.state))
        return dict(complete=False,reason=reason,rows=[],value_rows=[],decisions=decisions,searched=len(stats),
                    resumable=bool(resume_path and time.time()<job['deadline'] and decisions<job.get('max_actions',400)),
                    resume_path=resume_path)
    for step in range(decisions,job.get('max_actions',400)):
        if state['phase']=='finished':break
        if time.time()>=job['deadline']:
            return suspend('deadline')
        if decisions>slice_start_decision and (
            (job.get('slice_actions') and decisions-slice_start_decision>=job['slice_actions']) or
            (job.get('slice_seconds') and time.time()-slice_started>=job['slice_seconds'])):
            return suspend('slice_limit')
        actor=acting_side(state)
        budget_before=deepcopy(budget_rng.bit_generator.state);value_count_before=len(value_rows)
        budget,save_labels=decision_budget(job,budget_rng)
        if job.get('record_episode') and save_labels:
            from ..engine.duel_v2 import observe
            for owner in job.get('train_sides',['a']):
                if owner!=actor or job.get('collect_value_only'):
                    encode = getattr(runtime, 'encode', None) if runtime else None
                    if encode is None:
                        from .fixed_lineup import encode
                    vx,_=encode(observe(state,owner,include_previews=False),[])
                    value_rows.append(dict(x=vx,side=owner,key=job['policies'][owner][1],step=step,value_actor=int(owner==actor)))
        use_search=actor not in job.get('raw_sides',[])
        if use_search:
            try:
                result=search(state,actor,policies,seed=job['seed']+1000003*(step+1),
                              simulations=budget,deadline=job['deadline'],reuse_inference=job.get('reuse_inference',True),fast_simulation=job.get('fast_simulation',True),algorithm=job.get('search_algorithm','puct'),gumbel_candidates=job.get('gumbel_candidates',16),terminal_horizon=job.get('terminal_horizon',3),
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
                budget_rng.bit_generator.state=budget_before;del value_rows[value_count_before:]
                return suspend('incomplete_search')
            distribution=result['pi'].astype(float);distribution/=distribution.sum()
            chosen=int(rng.choice(len(result['actions']),p=distribution)) if job.get('training') and save_labels and job.get('search_algorithm','puct')=='puct' else result['search_choice']
            actions=result['actions']
            if job.get('collect') and save_labels:
                rows.append(dict(x=result['x'],c=result['c'],pi=result['pi'],side=actor,
                                 key=job['policies'][actor][1],phase=state['phase'],step=step,value_actor=1,search_budget=budget,policy_source=result.get('policy_source','visits'),
                                 visit_pi=result.get('visit_pi',result['pi']),selected=chosen,search_algorithm=job.get('search_algorithm','puct')))
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
    if not complete and resume_path:return suspend('action_limit')
    if complete:
        if resume_path:suspend('complete_pending_delivery')
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
