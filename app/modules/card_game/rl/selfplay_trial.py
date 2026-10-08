"""Equal actor slots: fixed opponents vs self-play vs mixed-budget self-play."""
from concurrent.futures import wait, FIRST_COMPLETED
import time
import numpy as np
from . import outcome_runtime as rt
from .information_search import search_game
from .episode_replay import dump_episode
from .information_longrun import append
from .fixed_lineup import write

ARMS=('fixed256','self256','selfmixed')
FOES=('starter','weave-rush','quick-rush')


def make_job(arm, number, out, config, base, current, source, initial):
    key='zhenhong';mirror=arm!='fixed256' and number%4<2
    history=arm!='fixed256' and number%4==2
    kind='mirror' if mirror else 'history' if history else 'cross'
    foe=key if mirror or history else FOES[(number//4 if arm!='fixed256' else number//2)%3]
    other=current if mirror else initial if history else source
    j=dict(id=f'{arm}-{number}',arm=arm,group=number,matchup=kind,
        seed=config['seeds']['foundation']+number,first=('a','b')[(number//4 if arm!='fixed256' else number)%2],
        runtime='outcome',simulations=config.get('search_budgets',{}).get(arm,config['simulations']),
        search_algorithm=config.get('search_algorithm','puct'),gumbel_candidates=config.get('gumbel_candidates',16),fast_simulation=config.get('fast_simulation',True),training=True,collect=True,record_episode=True,
        train_sides=['a','b'] if mirror else ['a'],deadline=config['train_until']-45,slice_seconds=config.get('slice_seconds',60),
        resume_path=str(out/'private-progress'/f'{arm}-{number}.json.gz'),
        decks={'a':base[key].serving_deck,'b':base[foe].serving_deck},
        policies={'a':(str(current),key),'b':(str(other),foe)},failure_dir=str(out/'failure-roots'))
    if arm=='selfmixed':
        j['search_mix']=dict(low=2 if config['smoke'] else 32,high=config['simulations'],deep_probability=.25)
    return j


def learn(executor,out,config,base,source,initial,nets,opts,pools,steps,totals,status,origin):
    import torch
    from .outcome_trial import game_summary
    ARMS=tuple(config.get('arms',('fixed256','self256','selfmixed')))
    current={a:config.get('arm_initial_models',{}).get(a,initial) for a in ARMS};numbers={a:0 for a in ARMS};slots={a:0 for a in ARMS}
    rngs={a:np.random.default_rng(config['seeds']['foundation']+i*1000000) for i,a in enumerate(ARMS)}
    pending={};costs={};started=time.time();slots_per_arm=config.get('actor_slots_per_arm',1 if config['smoke'] else 2)
    if type(slots_per_arm)!=int or not 1<=slots_per_arm<=4:raise ValueError('Invalid actor slots')
    def submit(arm):
        n=numbers[arm]
        if time.time()>=config['train_until']-(90 if config['smoke'] else config.get('finish_reserve_seconds',300)) or (config['smoke'] and n>=1):return
        j=make_job(arm,n,out,config,base,current[arm],source,initial)
        pending[executor.submit(search_game,j)]=(j,time.time());numbers[arm]+=1;slots[arm]+=1
        write(out/'pending-trajectories.json',[item[0] for item in pending.values()])
    def checkpoint():
        torch.save(dict(steps=steps,nets={a:n.state_dict() for a,n in nets.items()},opts={a:o.state_dict() for a,o in opts.items()},
            replay={a:p.episodes for a,p in pools.items()},rng={a:r.bit_generator.state for a,r in rngs.items()},
            groups=numbers),out/'checkpoint.tmp')
        (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    for a in ARMS:
        totals[a].update(resumed_slices=0,policy_samples=0,positive_policy_samples=0,negative_policy_samples=0,simulations=0,actor_seconds=0.)
        for _ in range(slots_per_arm):submit(a)
    while pending:
        done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
        if not done:
            status('searching',scheduled_groups=numbers,slots=slots);continue
        for future in done:
            j,start=pending.pop(future);a=j['arm'];slots[a]-=1;g=future.result();elapsed=time.time()-start
            totals[a]['actor_seconds']+=elapsed;costs[j['id']]=costs.get(j['id'],0)+elapsed
            if g.get('resumable') and time.time()<j['deadline']:
                append(out,'trajectory-slices.jsonl',dict(id=j['id'],arm=a,decisions=g['decisions'],reason=g['reason'],wall_seconds=elapsed))
                totals[a]['resumed_slices']+=1
                pending[executor.submit(search_game,j)]=(j,time.time());slots[a]+=1
                write(out/'pending-trajectories.json',[item[0] for item in pending.values()])
                status('searching',scheduled_groups=numbers,slots=slots);continue
            write(out/'pending-trajectories.json',[item[0] for item in pending.values()])
            append(out,'training-games.jsonl',dict(job=j,wall_seconds=costs.pop(j['id']),**game_summary(g),search_stats=g.get('search_stats',[])))
            if g['complete']:
                if g['searched']!=g['decisions']:raise ValueError('Unsearched actual decision')
                if len(g['rows'])!=sum(s['policy_label'] for s in g['search_stats']):raise ValueError('Deep-label count mismatch')
                data=dump_episode(out/'private-episodes'/f"{j['id']}.json.gz",j,g)
                from pathlib import Path
                Path(j['resume_path']).unlink(missing_ok=True)
                policy_rows=[r for r in data['rows'] if 'pi' in r]
                if any(r['search_budget']!=j['simulations'] for r in policy_rows):raise ValueError('Shallow policy label')
                if j['matchup']=='mirror' and g['winner'] in ('a','b') and policy_rows:
                    # Both sides are learning; count sides, never invent an extra game.
                    if len({r['side'] for r in policy_rows})==2 and {r['z'] for r in policy_rows}!={-1.,1.}:
                        raise ValueError('Mirror terminal perspective mismatch')
                totals[a]['games']+=1;totals[a]['wins']+=g['winner']=='a';totals[a]['samples']+=len(data['rows'])
                totals[a]['policy_samples']+=len(policy_rows)
                totals[a]['positive_policy_samples']+=sum(r['z']==1 for r in policy_rows)
                totals[a]['negative_policy_samples']+=sum(r['z']==-1 for r in policy_rows)
                totals[a]['simulations']+=sum(s['simulations'] for s in g['search_stats'])
                if data['rows'] and time.time()<config['train_until']-5:
                    pools[a].add(j['id'],data)
                    # Same update rule per new policy evidence, not equal updates despite unequal throughput.
                    count=1 if config['smoke'] else min(4,max(1,(len(policy_rows)+31)//32))
                    for _ in range(count):
                        if time.time()>=config['train_until']-5:break
                        batch,refs,diag=pools[a].sample(rngs[a],64,False)
                        if not batch:break
                        result=rt.update(nets[a],opts[a],batch,self_imitation=config.get('self_imitation',{}).get(a,0.));pools[a].priorities(refs,result.pop('priority'));steps[a]+=1
                        append(out,'losses.jsonl',dict(arm=a,step=steps[a],episode=j['id'],sample_refs=refs,
                            positive_policy_samples=sum('pi' in r and r['z']==1 for r in batch),
                            negative_policy_samples=sum('pi' in r and r['z']==-1 for r in batch),**diag,**result))
                    current[a]=out/a/'generations'/f"update-{steps[a]:04}"
                    rt.export(nets[a],current[a],'zhenhong',base['zhenhong'].serving_deck,{**origin,'steps':steps[a],'arm':a})
                    checkpoint()
            status('searching',scheduled_groups=numbers,slots=slots)
            submit(a)
    checkpoint()
    write(out/'selfplay-learning.json',dict(steps=steps,training=totals,scheduled=numbers,elapsed=time.time()-started,
        actor_slots_per_arm=slots_per_arm,equal_wallclock_deadline=True,equal_update_counts_required=False,
        no_reanalysis=True,auxiliary_rewards=0))
    return current,sum(numbers.values())
