"""One-preset search adaptation, frozen opponents and event-based contribution audit."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from copy import deepcopy
from collections import Counter
import hashlib,json,math,time,zipfile,gzip
import numpy as np
from .league_schema import PRESETS,SEATS,CARD_IDS,CAND_DIM,SCHEMA,feature_names,rule_hash,deck
from .league_policy import LeagueModel,shapes,export_model
from .information_longrun import load_numeric,checkpoint,restore,append,build_probes,probe_results
from .information_search import search_game,policy_value_update
from .information_experiment import write,compact

KEY='quick-rush'
FOES=('starter','weave-rush')

def migrate_numeric(source,target):
    """Explicit same-schema warm start under new rules, never inherited approval."""
    import torch
    from .gpu_duel.ppo import CompactScorer
    source=Path(source);target=Path(target)
    for key in PRESETS:
        m=json.loads((source/(key+'.json')).read_text(encoding='utf-8'));p=source/(key+'.npz')
        if m['schema']!=SCHEMA or m['deck']!=key or m['features']!=feature_names() or m['seat_ids']!=list(SEATS) or m['card_ids']!=list(CARD_IDS) or m['cand_dim']!=CAND_DIM:
            raise ValueError('Warm start feature identity mismatch')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Source weight hash mismatch')
        hidden=m['hidden']
        if type(hidden)!=int or not 1<=hidden<=256:raise ValueError('Invalid hidden width')
        with np.load(p,allow_pickle=False) as z:
            weights={k:z[k].copy() for k in shapes(hidden)}
        for k,shape in shapes(hidden).items():
            if weights[k].shape!=shape or not np.isfinite(weights[k]).all():raise ValueError('Invalid source tensor')
        net=CompactScorer(len(feature_names()),CAND_DIM,hidden)
        net.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
        export_model(net,target,key,m['build'],origin=dict(kind='explicit_rule_adaptation',source_sha256=m['sha256'],source_rule_hash=m['rule_hash'],target_rule_hash=rule_hash(),source_deck=key,approval_inherited=False))
        loaded=LeagueModel(target,key)
        assert all(np.array_equal(weights[k],loaded.weights[k]) for k in weights)


def contribution_events(events,heroes):
    metrics={cid:Counter() for cid in heroes}
    unattributed=Counter()
    for e in events:
        actor=e.get('actor');target=e.get('target','');before=e.get('before') or {};after=e.get('after') or {};kind=e.get('type')
        if not isinstance(actor,str) or not actor.startswith('a:') or actor[2:] not in metrics:continue
        bucket=metrics[actor[2:]]
        if 'hp' in before and 'hp' in after:
            delta=after['hp']-before['hp']
            if target.startswith('b:') and delta<0 and kind not in ('down','recover','shape'):
                bucket['player_hp_damage' if target=='b:player' else 'esper_hp_damage']-=delta
            if target.startswith('a:') and delta>0 and kind=='heal':bucket['healing']+=delta
        if target.startswith('b:') and 'shield' in before and 'shield' in after and e.get('amount') is not None:
            bucket['enemy_shield_absorbed']+=max(0,before['shield']-after['shield'])
    return {k:dict(v) for k,v in metrics.items()}


def evaluate_game(job):
    from .league_rollout import model,decision,clean
    from .report_telemetry import GameTelemetry
    from ..engine.duel_v2 import new_game,apply_action,acting_side
    from ..content.duel_v2 import CARDS
    s=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks']);telemetry=GameTelemetry(s)
    models={side:model(*spec) for side,spec in job['policies'].items()};actions_log=[];events=[];usage={cid:Counter() for cid in job['decks']['a']['character_ids']}
    error=None
    try:
        for _ in range(800):
            if s['phase']=='finished':break
            if time.time()>=job['deadline']:break
            side=acting_side(s);actions,(x,c)=decision(s,side);telemetry.before(s,side,actions)
            action=actions[int(models[side].scores(x,c).argmax())]
            if side=='a':
                cid=action.get('character_id')
                if action['type']=='play_card':
                    card=next(h for h in s['sides'][side]['hand'] if h['instance_id']==action['card_id']);cid=card['character_id']
                    if cid in usage:
                        usage[cid]['manual_card_plays']+=1
                        usage[cid]['battle_card_plays']+=card['type']=='battle'
                elif cid in usage:usage[cid][action['type']]+=1
            after=apply_action(s,side,action);telemetry.after(s,after,action,side)
            actions_log.append(dict(side=side,action=action,turn=s['turn']));events.extend(deepcopy(after.get('events',[])))
            s=clean(after)
    except Exception as exc:error=repr(exc)
    result=dict(id=job['id'],variant=job['variant'],foe=job['foe'],seed=job['seed'],first=job['first'],complete=s['phase']=='finished' and error is None,winner=s.get('winner'),turn=s['turn'],decisions=len(actions_log),error=error,usage={k:dict(v) for k,v in usage.items()},contribution=contribution_events(events,usage),telemetry=telemetry.finish(s),actions=actions_log,events=events)
    path=Path(job['output'])/'evaluation-raw'/f"{job['id']}.json.gz";path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False)
    return {k:v for k,v in result.items() if k not in ('actions','events','telemetry')}


def adaptation_probes(out,deadline):
    # New rules deliberately invalidate the old AP-zero hidden-hand proof.
    # Use explicit empty opposing hands in synthetic diagnostics instead;
    # conserve every original card by moving it into the opposing deck.
    from .search_pilot import position
    from .league_search import prove_turn
    from .league_rollout import decision
    rows=[]
    for i in range(1500):
        if time.time()>=deadline:break
        state,side=position(KEY,1140000000+i);foe='b' if side=='a' else 'a'
        state['sides'][foe]['deck']+=state['sides'][foe]['hand'];state['sides'][foe]['hand']=[]
        proof=prove_turn(state,side,depth=1,node_budget=128,deadline=deadline)
        if proof['paths'] and proof['roots_visited']==len(decision(state,side)[0]):
            rows.append(dict(state=state,side=side,targets=list(proof['paths'])))
        if len(rows)==12:break
    if len(rows)!=12:raise ValueError('Insufficient fully visible adaptation probes')
    probes={KEY:rows};write(Path(out)/'selection-probes.json',probes);return probes


def select_candidate(raw,retained):
    return bool(raw['complete'] and retained and raw['candidate']>=raw['best']+2)


def run(out,config,resume=False):
    import torch
    torch.set_num_threads(1);torch.manual_seed(1110000000)
    out=Path(out);initial=out/'initial';nets={};opts={};origins={};replay={k:[] for k in PRESETS}
    builds={k:LeagueModel(initial,k).serving_deck for k in PRESETS}
    for k in PRESETS:
        nets[k],origins[k]=load_numeric(initial,k,config['device']);opts[k]=torch.optim.Adam(nets[k].parameters(),lr=3e-5)
    state=dict(cycle=0,rule_hash=rule_hash(),current=str(initial),best=str(initial),games=0,searched_decisions=0,totals={k:dict(samples=0,optimizer_steps=0) for k in PRESETS},next_selection=time.time()+900)
    if resume:replay,state=restore(out,nets,opts)
    elif not (out/'recovery/pointer.json').exists():checkpoint(out,nets,opts,replay,state)
    probes=json.loads((out/'selection-probes.json').read_text()) if (out/'selection-probes.json').exists() else adaptation_probes(out,min(config['train_until'],time.time()+180))
    original=probe_results(initial,KEY,probes)
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        while time.time()<config['train_until'] and not (config.get('smoke') and state['cycle']>=1):
            cycle=state['cycle']+1;current=Path(state['current']);jobs=[]
            write(out/'status.json',dict(phase='training',**state,train_until=config['train_until']))
            for fi,foe in enumerate(FOES):
                for first in ('a','b'):
                    jobs.append(dict(seed=1110000000+cycle*1000+fi*100,first=first,deadline=config['train_until'],decks={'a':builds[KEY],'b':builds[foe]},policies={'a':(str(current),KEY),'b':(str(initial),foe)},simulations=config['simulations'],training=True,collect=True))
            games=list(pool.map(search_game,jobs));fresh=[]
            for job,g in zip(jobs,games):
                append(out,'training-games.jsonl',dict(cycle=cycle,seed=job['seed'],first=job['first'],foe=job['policies']['b'][1],**compact(g)))
                if g['complete']:
                    state['games']+=1;state['searched_decisions']+=g['searched'];fresh += [r for r in g['rows'] if r['side']=='a']
            replay[KEY]=(replay[KEY]+fresh)[-16384:];state['totals'][KEY]['samples']+=len(fresh)
            if not all(g['complete'] for g in games) or time.time()>=config['train_until']:
                checkpoint(out,nets,opts,replay,state);break
            rng=np.random.default_rng(1111000000+cycle)
            for step in range(min(48,2*math.ceil(len(fresh)/128))):
                if time.time()>=config['train_until']:break
                ids=rng.choice(len(fresh),min(64,len(fresh)),replace=False);old=rng.choice(len(replay[KEY]),min(64,len(replay[KEY])),replace=False)
                result=policy_value_update(nets[KEY],opts[KEY],[fresh[i] for i in ids]+[replay[KEY][i] for i in old])
                state['totals'][KEY]['optimizer_steps']+=1;append(out,'losses.jsonl',dict(cycle=cycle,key=KEY,**result))
            dest=out/'generations'/f'cycle-{cycle:04}'
            for k in PRESETS:export_model(nets[k],dest,k,builds[k],origin=origins[k])
            state['cycle']=cycle;state['current']=str(dest);checkpoint(out,nets,opts,replay,state)
            if time.time()>=state['next_selection'] and time.time()<config['train_until']-90:
                jobs=[];labels=[]
                for label,d in [('candidate',dest),('best',Path(state['best']))]:
                    for fi,foe in enumerate(FOES):
                        for i in range(4):
                            for first in ('a','b'):
                                jobs.append(dict(seed=1120000000+cycle*1000+fi*100+i,first=first,deadline=min(config['train_until'],time.time()+90),decks={'a':builds[KEY],'b':builds[foe]},policies={'a':(str(d),KEY),'b':(str(initial),foe)},raw_sides=['a','b']));labels.append(label)
                rs=list(pool.map(search_game,jobs));raw=dict(complete=all(r['complete'] for r in rs),candidate=sum(r.get('winner')=='a' for r,l in zip(rs,labels) if l=='candidate'),best=sum(r.get('winner')=='a' for r,l in zip(rs,labels) if l=='best'))
                now=probe_results(dest,KEY,probes);retained=all(not x or y for x,y in zip(original,now));promote=select_candidate(raw,retained)
                if promote:state['best']=str(dest)
                append(out,'selections.jsonl',dict(cycle=cycle,raw=raw,retained=retained,probe=now,decision='promote' if promote else 'keep'))
                state['next_selection']=time.time()+900;checkpoint(out,nets,opts,replay,state)
        for label,src in [('latest',Path(state['current'])),('best',Path(state['best']))]:
            import shutil
            (out/label).mkdir(exist_ok=True)
            for k in PRESETS:
                for ext in ('json','npz'):shutil.copy2(src/f'{k}.{ext}',out/label/f'{k}.{ext}')
        # Frozen opponents must remain bitwise identical throughout learning.
        for k in FOES:
            old=LeagueModel(initial,k);new=LeagueModel(out/'latest',k)
            assert all(np.array_equal(old.weights[n],new.weights[n]) for n in old.weights)
            assert state['totals'][k]['optimizer_steps']==0
        write(out/'training-summary.json',state)
        jobs=[]
        for variant in ('initial','best','latest'):
            for fi,foe in enumerate(FOES):
                for i in range(1 if config.get('smoke') else 32):
                    for first in ('a','b'):
                        jobs.append(dict(id=f'{variant}-{foe}-{i:03}-{first}',variant=variant,foe=foe,seed=1130000000+fi*10000+i,first=first,deadline=config['evaluate_until'],decks={'a':builds[KEY],'b':builds[foe]},policies={'a':(str(out/variant),KEY),'b':(str(initial),foe)},output=str(out)))
        write(out/'status.json',dict(phase='evaluating',jobs=len(jobs),training=state))
        results=[]
        for result in pool.map(evaluate_game,jobs):
            results.append(result);append(out,'evaluation-index.jsonl',result)
        write(out/'summary.json',dict(complete=all(r['complete'] for r in results),training=state,games=results,automatic_serving_approval=False,opponents_frozen=True,rule_hash=rule_hash()))
        write(out/'status.json',dict(phase='finished',complete=all(r['complete'] for r in results),finished=time.time()))
    members=[p for p in out.rglob('*') if p.is_file() and not any(k in p.parts for k in ('recovery','generations')) and p.name not in ('delivery.zip','delivery.json','console.log','guard.json')]
    manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
    write(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in members:z.write(p,p.relative_to(out))
        z.write(out/'manifest.json','manifest.json')
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),members=len(members)+1))
