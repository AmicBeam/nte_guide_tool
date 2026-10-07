"""Task-scoped three-strategy full cycle; never imports from the serving path."""
import time, json, math, shutil, hashlib, zipfile, gzip
from pathlib import Path
from copy import deepcopy
from collections import Counter
from itertools import combinations
from random import Random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from app.modules.card_game.rl.league_schema import PRESETS,SEATS,CARD_IDS,deck,rule_hash,build_hash
from app.modules.card_game.rl.league_policy import LeagueModel,export_model
from app.modules.card_game.rl.information_longrun import load_numeric,append,probe_results
from app.modules.card_game.rl.information_search import search_game,policy_value_update
from app.modules.card_game.rl.information_experiment import write,compact
from app.modules.card_game.rl.quick_adaptation import migrate_numeric
from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
from app.modules.card_game.rl.build_acceptance import (
    canonical_build, canonical_candidates, build_identity, paired_result, delivery_decision)



def cycle_ops(config):
    if not config.get('runtime'):
        return PRESETS, SEATS, CARD_IDS, LeagueModel, load_numeric, export_model, probe_results, deck
    if config['runtime'] != 'ten':raise ValueError('Unknown cycle runtime')
    from .fixed_lineup import KEYS, SEATS as seats, CARD_IDS as cards, FixedModel
    from .ten_search_runtime import load_numeric as load, export_model as export, probe_results as results
    return KEYS, seats, cards, FixedModel, load, export, results, lambda key: config['baseline_builds'][key]


def random_build(seed,team,card_ids=CARD_IDS):
    rng=Random(seed);cards=[]
    for cid in team:cards+=rng.sample([c for c in card_ids if CARDS[c]['character_id']==cid and not CARDS[c].get('derived') for _ in range(2)],8)
    return canonical_build(dict(id='variation',name='变化配牌',character_ids=list(team),card_ids=cards))


def candidates(parent,seed,card_ids=CARD_IDS):
    """All legal one-card swaps plus independently sampled two-card combinations."""
    parent=canonical_build(parent)
    rows=[deepcopy(parent)];seen={build_identity(parent)};rng=Random(seed)
    def add(cards):
        b=canonical_build({**parent,'card_ids':cards});h=build_identity(b)
        if h not in seen:seen.add(h);rows.append(b)
    for hero in parent['character_ids']:
        pool=[c for c in card_ids if CARDS[c]['character_id']==hero and not CARDS[c].get('derived')]
        count=Counter(parent['card_ids'])
        for old in pool:
            if not count[old]:continue
            for new in pool:
                if old==new or count[new]>=2:continue
                cards=list(parent['card_ids']);cards.remove(old);cards.append(new);add(cards)
    for _ in range(48):
        cards=list(parent['card_ids'])
        for step in range(2):
            old=rng.choice(cards);hero=CARDS[old]['character_id'];count=Counter(cards)
            pool=[c for c in card_ids if CARDS[c]['character_id']==hero and not CARDS[c].get('derived') and c!=old and count[c]<2]
            new=rng.choice(pool);cards.remove(old);cards.append(new)
        add(cards)
    for i in range(8):add(random_build(seed+i,parent['character_ids'],card_ids)['card_ids'])
    return rows


def copy_models(source,target,keys=PRESETS):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for key in keys:
        for ext in ('npz','json'):shutil.copy2(Path(source)/f'{key}.{ext}',target/f'{key}.{ext}')


def probes(out,deadline,seed_base,config=None):
    if config and config.get("runtime")=="ten":
        from .ten_search_runtime import probes as ten_probes
        result=ten_probes(config["baseline_builds"],seed_base)
        write(out/"selection-probes.json",result);return result
    from app.modules.card_game.rl.search_pilot import position
    from app.modules.card_game.rl.league_search import prove_turn
    from app.modules.card_game.rl.league_rollout import decision
    result={}
    for ki,key in enumerate(PRESETS):
        result[key]=[]
        for i in range(1500):
            if time.time()>=deadline:raise TimeoutError('selection probe generation')
            s,side=position(key,seed_base+ki*10000+i);foe='b' if side=='a' else 'a'
            s['sides'][foe]['deck']+=s['sides'][foe]['hand'];s['sides'][foe]['hand']=[]
            proof=prove_turn(s,side,depth=1,node_budget=128,deadline=deadline)
            if proof['paths'] and proof['roots_visited']==len(decision(s,side)[0]):result[key].append(dict(state=s,side=side,targets=list(proof['paths'])))
            if len(result[key])==12:break
        if len(result[key])!=12:raise ValueError('Insufficient selection probes')
    write(out/'selection-probes.json',result);return result


def compare(pool,out,key,dirs,builds,reference,reference_builds,seed,deadline,pairs,config=None,searching=False):
    config=config or {}
    PRESETS,_,_,_,_,_,_,_=cycle_ops(config)
    jobs=[];labels=[]
    for label,folder in dirs.items():
        for fi,foe in enumerate(k for k in PRESETS if k!=key):
            for i in range(pairs):
                for first in ('a','b'):
                    jobs.append(dict(seed=seed+fi*100+i,first=first,deadline=deadline,decks={'a':canonical_build(builds[label]),'b':canonical_build(reference_builds[foe])},policies={'a':(str(folder),key),'b':(str(reference),foe)},raw_sides=['b'] if searching else ['a','b'],runtime=config.get('runtime'),search_algorithm=config.get('search_algorithm','puct'),gumbel_candidates=config.get('gumbel_candidates',16),simulations=config.get('simulations',256)))
                    labels.append(label)
    rs=list(pool.map(search_game,jobs));summary={}
    for label in dirs:
        own=[g for g,l in zip(rs,labels) if l==label]
        summary[label]=dict(complete=all(g['complete'] for g in own),wins=sum(g.get('winner')=='a' for g in own),n=len(own))
    append(out,'comparisons.jsonl',dict(key=key,seed=seed,summary=summary,builds=builds,models={k:str(v) for k,v in dirs.items()},games=[compact(g) for g in rs]))
    return summary


def _train_iter(pool,out,source,reference,builds,reference_builds,deadline,config,stage,probe,update_target=None):
    PRESETS,SEATS,CARD_IDS,LeagueModel,load_numeric,export_model,probe_results,deck=cycle_ops(config)
    import torch,numpy as np
    folder=out/stage;folder.mkdir();nets={};opts={};origins={};replay={k:[] for k in PRESETS}
    for key in PRESETS:
        nets[key],origins[key]=load_numeric(source,key,config['device']);opts[key]=torch.optim.Adam(nets[key].parameters(),lr=3e-5)
    from .setup_reward import coefficient as setup_coefficient
    def bonus(key):
        return setup_coefficient(config.get('zhenhong_setup_reward',0.),origins[key].get('setup_reward_updates',0)+totals[key]['steps'],config.get('zhenhong_setup_decay_steps',64)) if key=='zhenhong' else 0.
    def export_origin(key):
        return dict(origins[key],setup_reward_updates=origins[key].get('setup_reward_updates',0)+totals[key]['steps'] if key=='zhenhong' else 0)
    current=folder/'initial'
    for key in PRESETS:export_model(nets[key],current,key,builds[key],origin=origins[key])
    best={k:current for k in PRESETS};baseline={k:probe_results(current,k,probe) for k in PRESETS}
    cycle=0;totals={k:dict(samples=0,steps=0) for k in PRESETS};next_check=time.time()+1800;games_n=0
    stage_seed=config['seeds']['foundation'] if stage=='foundation' else config['seeds']['adaptation']
    selection_seed=config['seeds']['selection']+(0 if stage=='foundation' else 2_000_000)
    while time.time()<deadline and not (config['smoke'] and cycle>=1) and not (update_target is not None and all(t['steps']>=update_target for t in totals.values())):
        if cycle>=800:break  # keep training seeds inside the reserved 10M block
        cycle+=1;jobs=[]
        rollout_deadline=deadline-min(float(config.get('optimizer_reserve_seconds',60)),max(0,(deadline-time.time())/4))
        write(out/'status.json',dict(phase=stage,cycle=cycle,totals=totals,games=games_n,stage_deadline=deadline,hard_deadline=config['deadlines']['shutdown']))
        for ki,key in enumerate(PRESETS):
            opponents=[k for k in PRESETS if k!=key]
            width=config.get('opponents_per_cycle',1 if config.get('runtime')=='ten' else len(opponents))
            if config['smoke']:width=1
            if not 1<=width<=len(opponents):raise ValueError('Invalid opponent batch width')
            start=(cycle-1)*width+ki
            opponents=[opponents[(start+i)%len(opponents)] for i in range(width)]
            for fi,foe in enumerate(opponents):
                for first in ('a','b'):
                    historical=cycle%4==0 or update_target is not None
                    enemy=reference if historical else current
                    enemy_build=reference_builds[foe] if historical else builds[foe]
                    if update_target is None and cycle%8==0:
                        enemy=best[foe];enemy_build=builds[foe]
                    if cycle%5==0 or config['smoke']:enemy_build=random_build(stage_seed+cycle*100+fi,enemy_build['character_ids'],CARD_IDS)
                    jobs.append(dict(seed=stage_seed+cycle*10000+ki*1000+fi*100,first=first,deadline=rollout_deadline,decks={'a':builds[key],'b':enemy_build},policies={'a':(str(current),key),'b':(str(enemy),foe)},setup_rewards={'a':bonus(key),'b':0. if historical else bonus(foe)},runtime=config.get('runtime'),zhenhong_passive_reward=config.get('zhenhong_passive_reward',0),simulations=config['simulations'],search_algorithm=config.get('search_algorithm','puct'),gumbel_candidates=config.get('gumbel_candidates',16),training=True,collect=True,failure_dir=str(out/'failure-roots'),train_sides=['a'] if historical else ['a','b']))
        results=[];fresh={k:[] for k in PRESETS}
        for job,g in zip(jobs,pool.map(search_game,jobs)):
            results.append(g)
            append(folder,'games.jsonl',dict(cycle=cycle,seed=job['seed'],first=job['first'],builds=job['decks'],policies=job['policies'],train_sides=job['train_sides'],**compact(g)))
            if g['complete']:
                if g['decisions']!=g['searched'] or len(g['rows'])!=g['decisions']:raise ValueError('Unsearched training decision')
                games_n+=1
                for row in g['rows']:
                    if row['side'] in job['train_sides']:fresh[row['key']].append(row)
            write(out/'status.json',dict(phase=stage,cycle=cycle,completed_batch_jobs=len(results),scheduled_batch_jobs=len(jobs),totals=totals,games=games_n,stage_deadline=deadline,hard_deadline=config['deadlines']['shutdown']))
        for key in PRESETS:
            replay[key]=(replay[key]+fresh[key])[-16384:];totals[key]['samples']+=len(fresh[key])
            if bonus(key)==0:replay[key]=[r for r in replay[key] if not r.get('setup_reward',0)]
        if not can_update_completed(fresh,time.time(),deadline):break
        rng=np.random.default_rng(stage_seed+cycle)
        for key in PRESETS:
            if not fresh[key]:continue
            steps=min(4,update_target-totals[key]['steps']) if update_target is not None else min(48,2*math.ceil(len(fresh[key])/128))
            for step in range(steps):
                if time.time()>=deadline:break
                current_bonus=bonus(key)
                if current_bonus==0 and any(r.get('setup_reward',0)>0 for r in fresh[key]):
                    replay[key]=[];break  # Collect new zero-bonus search labels before further updates.
                ids=rng.choice(len(fresh[key]),min(64,len(fresh[key])),replace=False);old=rng.choice(len(replay[key]),min(64,len(replay[key])),replace=False)
                stats=policy_value_update(nets[key],opts[key],[fresh[key][i] for i in ids]+[replay[key][i] for i in old],setup_coefficient=current_bonus);totals[key]['steps']+=1
                append(folder,'losses.jsonl',dict(cycle=cycle,key=key,setup_coefficient=current_bonus,**stats))
        current=folder/'generations'/f'cycle-{cycle:04}'
        for key in PRESETS:export_model(nets[key],current,key,builds[key],origin=export_origin(key))
        # Numeric generations survive a failed later stage; optimizer checkpoint is atomic.
        temp=folder/'checkpoint.tmp';torch.save(dict(cycle=cycle,totals=totals,origins=origins,current=str(current),best={k:str(v) for k,v in best.items()},next_check=next_check,nets={k:n.state_dict() for k,n in nets.items()},opts={k:o.state_dict() for k,o in opts.items()},rule_hash=config['rule_hash'],replay={k:[dict(x=torch.from_numpy(r['x']),c=torch.from_numpy(r['c']),pi=torch.from_numpy(r['pi']),z=r['z'],**{field:r[field] for field in ('reward_to_go','setup_reward','setup_future') if field in r}) for r in rows] for k,rows in replay.items()}),temp)
        if (folder/'checkpoint.pt').exists():shutil.copy2(folder/'checkpoint.pt',folder/'checkpoint-previous.pt')
        temp.replace(folder/'checkpoint.pt')
        write(folder/'checkpoint-manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('checkpoint*.pt')})
        if update_target is None and (time.time()>=next_check or config['smoke']):
            for ki,key in enumerate(PRESETS):
                if time.time()>=deadline-10:break
                end=min(deadline,time.time()+(90 if config['smoke'] else 600));dirs={'candidate':current,'best':best[key]};bs={l:builds[key] for l in dirs}
                rs=compare(pool,folder,key,dirs,bs,reference,reference_builds,selection_seed+cycle*1000+ki*100,end,1 if config['smoke'] else 4,config)
                retained=all(not a or b for a,b in zip(baseline[key],probe_results(current,key,probe)))
                promote=retained and all(r['complete'] for r in rs.values()) and rs['candidate']['wins']>=rs['best']['wins']+2
                if promote:
                    confirm=compare(pool,folder,key,dirs,bs,reference,reference_builds,selection_seed+1000000+cycle*1000+ki*100,end,1 if config['smoke'] else 4,config)
                    promote=all(r['complete'] for r in confirm.values()) and confirm['candidate']['wins']>confirm['best']['wins']
                if promote:
                    screen=compare(pool,folder,key,dirs,bs,reference,reference_builds,selection_seed+3000000+cycle*1000+ki*100,end,1 if config['smoke'] else 2,config,searching=True)
                    confirm=compare(pool,folder,key,dirs,bs,reference,reference_builds,selection_seed+4000000+cycle*1000+ki*100,end,1 if config['smoke'] else 2,config,searching=True)
                    promote=all(x['complete'] for group in (screen,confirm) for x in group.values()) and all(group['candidate']['wins']>=group['best']['wins'] for group in (screen,confirm))
                rollback=False
                if all(r['complete'] for r in rs.values()) and rs['candidate']['wins']<=rs['best']['wins']-4:
                    confirm_bad=compare(pool,folder,key,dirs,bs,reference,reference_builds,selection_seed+5000000+cycle*1000+ki*100,end,1 if config['smoke'] else 4,config)
                    rollback=all(r['complete'] for r in confirm_bad.values()) and confirm_bad['candidate']['wins']<=confirm_bad['best']['wins']-4
                if promote:best[key]=current
                if rollback:
                    restored,_=load_numeric(best[key],key,config['device']);nets[key].load_state_dict(restored.state_dict())
                    opts[key]=torch.optim.Adam(nets[key].parameters(),lr=max(1e-5,opts[key].param_groups[0]['lr']/2));replay[key]=[]
                append(folder,'selection.jsonl',dict(cycle=cycle,key=key,retained=retained,promote=promote,rollback=rollback))
            current=folder/'generations'/f'selected-{cycle:04}'
            for key in PRESETS:export_model(nets[key],current,key,builds[key],origin=export_origin(key))
            next_check=time.time()+1800
        yield dict(cycle=cycle,totals=deepcopy(totals))
    selected=folder/'selected';selected.mkdir()
    for key in PRESETS:
        for ext in ('json','npz'):shutil.copy2(best[key]/f'{key}.{ext}',selected/f'{key}.{ext}')
    if 'zhenhong' in PRESETS:
        manifest=json.loads((selected/'zhenhong.json').read_text())
        origin=manifest.get('origin') or {}
        manifest['origin']=dict(origin,setup_reward_updates=max(origin.get('setup_reward_updates',0),export_origin('zhenhong')['setup_reward_updates']))
        write(selected/'zhenhong.json',manifest)
    copy_models(current,folder/'latest',PRESETS)
    write(folder/'summary.json',dict(cycles=cycle,games=games_n,totals=totals,best={k:str(v) for k,v in best.items()},latest=str(current),finished=time.time()))
    return folder/'latest' if update_target is not None else selected



def can_update_completed(fresh, now, deadline):
    """A truncated sibling game does not invalidate finished trajectories."""
    return now < deadline and any(fresh.values())


def train(*args, **kwargs):
    iterator=_train_iter(*args,**kwargs)
    while True:
        try:next(iterator)
        except StopIteration as done:return done.value


def interleave_training(iterators):
    """Advance equally budgeted branches one cycle at a time; never resume a finished branch."""
    active=dict(iterators);results={}
    while active:
        for key,iterator in list(active.items()):
            try:next(iterator)
            except StopIteration as done:
                results[key]=done.value;del active[key]
    return results


def tune(pool,out,models,builds,deadline,config):
    PRESETS,SEATS,CARD_IDS,LeagueModel,load_numeric,export_model,probe_results,deck=cycle_ops(config)
    selected=deepcopy(builds);reference=deepcopy(builds);folder=out/'cards';folder.mkdir()
    search_keys=list(config.get('build_search_keys') or PRESETS)
    if any(key not in PRESETS for key in search_keys) or len(set(search_keys))!=len(search_keys) or not search_keys:
        raise ValueError('Invalid build search keys')
    rounds=1 if config['smoke'] else int(config.get('build_search_rounds') or 4)
    if rounds<1:raise ValueError('Invalid build search rounds')
    screen=tuple(config.get('build_search_pairs') or ((1,) if config['smoke'] else (8,32,128)))
    if config['smoke']:screen=(1,)
    limit=config.get('build_search_limit')
    for round_id in range(rounds):
        for ki,key in enumerate(search_keys):
            left=(rounds-round_id)*len(search_keys)-ki
            end=min(deadline,time.time()+max(0,(deadline-time.time())/left));parent=deepcopy(selected[key]);rows=candidates(parent,config['seeds']['search']+round_id*10000+ki*100,CARD_IDS)
            if config['smoke']:rows=rows[:2]
            elif limit:rows=rows[:int(limit)]
            original=deepcopy(builds[key]);official=deck(key)
            for b in (original,official):
                if build_identity(b) not in {build_identity(r) for r in rows}:rows.append(canonical_build(b))
            write(out/'status.json',dict(phase='cards',round=round_id,key=key,candidates=len(rows),stage_deadline=end,hard_deadline=config['deadlines']['shutdown']))
            write(folder/f'{round_id}-{key}-candidates.json',dict(parent=parent,candidates=rows))
            baseline_index=next(i for i,b in enumerate(rows) if build_identity(b)==build_identity(original))
            active=list(range(len(rows)));winner=0
            for level,pairs in enumerate(screen):
                if time.time()>=end:break
                dirs={str(i):models for i in active};bs={str(i):rows[i] for i in active}
                rs=compare(pool,folder,key,dirs,bs,models,reference,config['seeds']['search']+1_000_000+round_id*100000+ki*10000+level*1000,end,pairs,**({'config':config} if config.get('runtime') else {}))
                if not all(r['complete'] for r in rs.values()):
                    append(folder,'decisions.jsonl',dict(round=round_id,key=key,level=level,decision='incomplete_keep_parent'));break
                ranked=sorted(active,key=lambda i:(-rs[str(i)]['wins'],i!=baseline_index,i));winner=ranked[0]
                # Original baseline and incumbent survive every screen; ties prefer original.
                active=list(dict.fromkeys([0,baseline_index,*ranked[:4 if level==0 else 2]]))
                if level==len(screen)-1:
                    if rs[str(winner)]['wins']>rs['0']['wins'] or winner==baseline_index:
                        selected[key]=rows[winner]
                    append(folder,'decisions.jsonl',dict(round=round_id,key=key,level=level,parent=parent,selected=selected[key],winner=winner,results=rs))
    records=[json.loads(line) for line in (folder/'decisions.jsonl').read_text().splitlines()] if (folder/'decisions.jsonl').exists() else []
    complete=sum('results' in r and all(x['complete'] for x in r['results'].values()) for r in records)==rounds*len(search_keys)
    write(folder/'status.json',dict(workflow_complete=complete,candidate_quality_approved=False))
    write(folder/'selected-builds.json',selected);return selected


def evaluate_job(j):
    from app.modules.card_game.rl.league_rollout import model,decision,clean
    from app.modules.card_game.rl.report_telemetry import GameTelemetry
    from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side,observe
    runtime=None
    if j.get('runtime')=='ten':
        from . import ten_search_runtime as runtime
        from .ten_search_runtime import model,decision
    if j.get('runtime')=='outcome':
        from . import outcome_runtime as runtime
        from .outcome_runtime import model,decision
    if j.get('runtime')=='cross':
        from . import cross_runtime as runtime
        from .cross_runtime import model,decision
    if j.get('runtime')=='recovery':
        from . import recovery_runtime as runtime
        from .recovery_runtime import model,decision
    j={**j,'decks':{k:canonical_build(v) for k,v in j['decks'].items()}}
    search_sides=j.get('search_sides',[])
    if not set(search_sides)<= {'a','b'}:raise ValueError('Invalid search evaluation sides')
    if search_sides and (j.get('details') is False or any(p[0]=='rule' for p in j['policies'].values())):
        raise ValueError('Search evaluation requires numeric opponents and full recording')
    search_policies={s:model(*p) for s,p in j['policies'].items()} if search_sides else {}
    value_policies=None
    if search_sides and j.get('teacher_mode')=='covered_value':
        from .covered_value import load_scope
        values,_=load_scope(j['covered_value_source'])
        value_policies={side:values[spec[1]] for side,spec in j['policies'].items()}
    if runtime is not None and j.get('runtime')=='recovery' and search_sides:
        if not runtime.value_precheck_ok(j.get('value_precheck'),search_policies):
            raise ValueError('Matching WDL checks required for recovery evaluation search')
    if j.get('details') is False and j.get('runtime')=='recovery':
        raise ValueError('Recovery evaluation requires complete telemetry')
    if j.get('details') is False:
        g=search_game(dict(j,raw_sides=['a','b']))
        return dict(j,**{k:v for k,v in g.items() if k not in ('rows','search_stats')})
    s=new_game(seed=j['seed'],first_side=j['first'],decks=j['decks']);opening=deepcopy(s);tele=GameTelemetry(s);actions_log=[];events=[];error=None;search_stats=[]
    immune_metrics={side:Counter() for side in ('a','b')};immune_risks={side:[] for side in ('a','b')}
    from .setup_reward import ready as setup_ready
    setup_seen={side:setup_ready(s,side) for side in ('a','b')}
    try:
        for _ in range(800):
            if s['phase']=='finished' or time.time()>=j['deadline']:break
            side=acting_side(s);spec=j['policies'][side]
            if spec[0]=='rule':
                v=observe(s,side);actions=[e['action'] for e in v['legal_actions'] if e['action']['type']!='concede'];idx=VisibleEngineRulePolicy()(v,tuple(actions))
            elif side in search_sides:
                from .information_search import search
                # A separate, explicit high-bit namespace keeps evaluation
                # search worlds out of the legacy training/game seed ranges.
                digest=hashlib.sha256(f'evaluation-search-v1:{j["seed"]}:{len(actions_log)}:{side}'.encode()).hexdigest()
                root_seed=(1<<256)|int(digest,16)
                if j.get('runtime')=='recovery':
                    from .recovery_search import search as recovery_search
                    result=recovery_search(s,side,search_policies,seed=root_seed,
                        simulations=j.get('simulations',32),gumbel_candidates=16,
                        noise=False,runtime=runtime,deadline=j['deadline'],terminal_horizon=j.get('terminal_horizon',3),
                        teacher_mode=j.get('teacher_mode','network'),value_policies=value_policies)
                else:
                    result=search(s,side,search_policies,seed=root_seed,simulations=j.get('simulations',16),
                        algorithm=j.get('search_algorithm','gumbel'),gumbel_candidates=j.get('gumbel_candidates',8),
                        noise=False,runtime=runtime,deadline=j['deadline'],terminal_horizon=j.get('terminal_horizon',3))
                if not result['complete']:
                    error='Incomplete evaluation search';break
                actions=result['actions'];idx=result['search_choice']
                search_stats.append(dict(step=len(actions_log),side=side,simulations=result['simulations'],
                    requested=result['requested'],covered=result['roots_covered'],legal=len(actions),
                    root_seed_sha256=digest,policy_source=result['policy_source'],noise=False))
            else:
                actions,(x,c)=decision(s,side);idx=int(model(*spec).scores(x,c).argmax())
            guide_audit=None
            if j.get('guided_policy'):
                if j.get('runtime')!='recovery':raise ValueError('Guide evaluation requires recovery runtime')
                from .guided_recovery import apply_guide
                idx,_,guide_audit=apply_guide(s,side,actions,idx,j,runtime)
            if side=='a' and j.get('keep_all') and s['phase']=='mulligan':idx=next(i for i,a in enumerate(actions) if a['type']=='mulligan' and not a.get('card_ids'))
            tele.before(s,side,actions);action=actions[idx];after=apply_action(s,side,action);tele.after(s,after,action,side)
            from .fixed_lineup_training import track_immune_window
            track_immune_window(s,after,action,side,immune_metrics['a'],immune_risks['a'])
            swapped_before={**s,'sides':{'a':s['sides']['b'],'b':s['sides']['a']}}
            swapped_after={**after,'sides':{'a':after['sides']['b'],'b':after['sides']['a']}}
            track_immune_window(swapped_before,swapped_after,action,'b' if side=='a' else 'a',immune_metrics['b'],immune_risks['b'])
            actions_log.append(dict(side=side,action=action,turn=s['turn']));events.extend(deepcopy(after.get('events',[])));s=clean(after)
            if guide_audit is not None:actions_log[-1]['guide']=guide_audit
            for owner in ('a','b'):setup_seen[owner]=setup_seen[owner] or setup_ready(s,owner)
            if j.get('replay'):
                actions_log[-1]['state_sha256']=hashlib.sha256(json.dumps(s,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    except Exception as exc:error=repr(exc)
    replay_verified=False
    if j.get('replay') and s['phase']=='finished' and error is None:
        try:
            replayed=deepcopy(opening)
            for record in actions_log:
                replayed=clean(apply_action(replayed,record['side'],record['action']))
                if hashlib.sha256(json.dumps(replayed,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=record['state_sha256']:
                    raise ValueError('Replay action/state mismatch')
            replay_verified=True
        except Exception as exc:error=repr(exc)
    result=dict(**j,setup_achieved=setup_seen,passive_triggers={side:int(s['sides'][side]['characters'].get('zhenhong',{}).get('surplus_passive_triggers',0)) for side in ('a','b')},replay_verified=replay_verified,complete=s['phase']=='finished' and error is None,winner=s.get('winner'),turn=s['turn'],error=error,telemetry=tele.finish(s),actions=actions_log,events=events,immune_window={k:dict(v) for k,v in immune_metrics.items()},immune_risks=immune_risks,opening_state=opening,
        search_stats=search_stats,search_decisions=len(search_stats),search_simulations=sum(r['simulations'] for r in search_stats))
    out=Path(j['output'])
    with gzip.open(out/'raw'/f"{j['id']}.json.gz",'wt',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False)
    if j.get('replay') and result['complete']:
        from app.modules.card_game.rl.offline_analysis import _public_payload
        s['events']=events;payload=_public_payload(opening,s,game_id=j['id'])
        with gzip.open(out/'replays'/f"{j['id']}.json.gz",'wt',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False)
    return {k:v for k,v in result.items() if k not in ('telemetry','actions','events','output','opening_state','search_stats')}


def evaluate(pool,out,initial,final,old_builds,builds,deadline,config):
    PRESETS,SEATS,CARD_IDS,LeagueModel,load_numeric,export_model,probe_results,deck=cycle_ops(config)
    folder=out/'evaluation';folder.mkdir();(folder/'raw').mkdir();(folder/'replays').mkdir();jobs=[];pairs=1 if config['smoke'] else 16
    def add(cell,a,b,source,bs,seed,n=pairs,enemy=None,enemy_build=None,keep=False,replay=False):
        for i in range(n):
            for first in ('a','b'):
                jobs.append(dict(id=f'g{len(jobs):05}',cell=cell,seed=seed+i,first=first,deadline=deadline,decks={'a':bs,'b':enemy_build or old_builds[b]},policies={'a':(str(source),a),'b':(str(enemy or initial),b)},keep_all=keep,replay=replay and i<4,output=str(folder),runtime=config.get('runtime')))
    for ki,key in enumerate(PRESETS):
        for fi,foe in enumerate(k for k in PRESETS if k!=key):
            seed=(config['seeds']['audit']+0)+ki*10000+fi*1000
            for label,src,bs in [('initial',initial,old_builds[key]),('recommended',final,builds[key])]:add(f'{label}/{key}/{foe}',key,foe,src,bs,seed)
        add(f'mirror/{key}',key,key,final,builds[key],(config['seeds']['audit']+1000000)+ki*10000,enemy=final,enemy_build=builds[key],replay=True,n=1 if config['smoke'] else config.get('matrix_games',128)//2)
    for ki,(key,foe) in enumerate(combinations(PRESETS,2)):add(f'cross/{key}/{foe}',key,foe,final,builds[key],(config['seeds']['audit']+2000000)+ki*10000,enemy=final,enemy_build=builds[foe],replay=True,n=1 if config['smoke'] else config.get('matrix_games',128))
    for ki,key in enumerate(PRESETS):
        for fi,foe in enumerate(PRESETS):
            for keep in (False,True):add(f'opening/{key}/{foe}/{keep}',key,foe,final,builds[key],(config['seeds']['audit']+3000000)+ki*10000+fi*1000,n=1 if config['smoke'] else 8,enemy=final,enemy_build=builds[foe],keep=keep)
            add(f'rule/{key}/{foe}',key,foe,final,builds[key],(config['seeds']['audit']+4000000)+ki*10000+fi*1000,n=1 if config['smoke'] else 8,enemy='rule',enemy_build=builds[foe])
        teams=list(combinations(SEATS,4))[:1] if config['smoke'] else list(combinations(SEATS,4))
        for ti,team in enumerate(teams):
            var=random_build((config['seeds']['audit']+5000000)+ti,team,CARD_IDS)
            for enemy in (initial,'rule'):add(f'variation/{key}/{"model" if enemy==initial else "rule"}',key,PRESETS[ti%len(PRESETS)],final,builds[key],(config['seeds']['audit']+6000000)+ki*10000+ti,n=1,enemy=enemy,enemy_build=var)
    write(folder/'jobs.json',jobs);results=[]
    for r in pool.map(evaluate_job,jobs):
        results.append(r);append(folder,'index.jsonl',r)
        write(out/'status.json',dict(phase='evaluation',completed=len(results),scheduled=len(jobs),hard_deadline=config['deadlines']['shutdown']))
    from app.modules.card_game.rl.league_gate import cases
    from app.modules.card_game.rl.search_budget_eval import consistent_tactical_fixture,expected_actions,case_job
    tactical=[]
    for key,name,s,side,goal in ([] if config.get('runtime')=='ten' else cases()):
        if time.time()>=deadline:break
        s,_=consistent_tactical_fixture(s);foe='b' if side=='a' else 'a'
        row=case_job(dict(state=s,side=side,budget=0,seed=(config['seeds']['audit']+7000000),deadline=deadline,expected=expected_actions(s,side,goal),policies={side:(str(final),key),foe:(str(initial),'weave-rush')}))
        tactical.append(dict(key=key,name=name,**row))
    if config.get('runtime')=='ten':
        from .five_full_cycle import tactical_gate
        tactical=[dict(key=key,**tactical_gate(final,key,builds[key])) for key in PRESETS]
    write(folder/'tactical.json',tactical)
    cells={}
    for r in results:
        c=cells.setdefault(r['cell'],dict(n=0,wins=0,losses=0,draws=0,incomplete=0,first_wins=0,first_n=0,second_wins=0,second_n=0,turns=0))
        if not r['complete']:c['incomplete']+=1;continue
        c['n']+=1;c['wins']+=r['winner']=='a';c['losses']+=r['winner']=='b';c['draws']+=r['winner'] not in ('a','b');c['turns']+=r['turn']
        pos='first' if r['first']=='a' else 'second';c[pos+'_n']+=1;c[pos+'_wins']+=r['winner']=='a'
    write(folder/'summary.json',dict(cells=cells,complete=all(r['complete'] for r in results),scheduled=len(jobs),automatic_serving_approval=False))
    lines=['# 三阵容完整训练冻结验收','', '训练、配牌搜索、适应与冻结评估分别归档。未自动发布模型；下表胜率只对应本轮冻结条件。','','| 测试单元 | 完成 | 胜 | 负 | 平 | 未完成 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,c in cells.items():lines.append(f"| {name} | {c['n']} | {c['wins']} | {c['losses']} | {c['draws']} | {c['incomplete']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from .new_model_report import write_matrix
    matrix_complete=write_matrix(out,results,PRESETS,final,config)
    return matrix_complete and all(r['complete'] for r in results) and (len(tactical)==len(cases()) if not config.get('runtime') else len(tactical)==len(PRESETS) and all(r['complete'] for r in tactical))


def gate_comparison(pool,out,key,baseline,candidate,base_build,candidate_build,reference,reference_builds,config,phase,deadline):
    PRESETS,SEATS,CARD_IDS,LeagueModel,load_numeric,export_model,probe_results,deck=cycle_ops(config)
    """Freeze one proposal before touching confirmation/audit; no repeated peeking."""
    folder=out/phase;folder.mkdir(exist_ok=True);(folder/'raw').mkdir(exist_ok=True);(folder/'replays').mkdir(exist_ok=True)
    offset=8_000_000 if phase=='audit-gate' else 0
    namespace='audit' if phase=='audit-gate' else 'confirmation'
    seeds=list(range(config['seeds'][namespace]+offset+PRESETS.index(key)*10000,
                     config['seeds'][namespace]+offset+PRESETS.index(key)*10000+config['confirmation_seeds']))
    foes=[k for k in PRESETS if k!=key];jobs=[]
    identities={label:LeagueModel(src,key).version for label,src in [('baseline',baseline),('candidate',candidate)]}
    for label,src,build in [('baseline',baseline,base_build),('candidate',candidate,candidate_build)]:
        for foe in foes:
            for seed in seeds:
                for first in ('a','b'):
                    jobs.append(dict(id=f'{key}-{label}-{foe}-{seed}-{first}',cell=f'{phase}/{key}',
                        variant=label,foe=foe,seed=seed,first=first,deadline=deadline,
                        decks={'a':canonical_build(build),'b':canonical_build(reference_builds[foe])},
                        policies={'a':(str(src),key),'b':(str(reference),foe)},output=str(folder),runtime=config.get('runtime'),details=False))
    write(folder/f'{key}-plan.json',dict(seeds=seeds,opponents=foes,identities=identities,
                                       builds={'baseline':base_build,'candidate':candidate_build}))
    results=[]
    for row in pool.map(evaluate_job,jobs):
        append(folder,f'{key}-games.jsonl',row);results.append(row)
    assert all(LeagueModel(src,key).version==identities[label] for label,src in [('baseline',baseline),('candidate',candidate)])
    result=paired_result(results,expected_seeds=seeds,opponents=foes,
                         alpha=config['alpha']/2,family_size=len(PRESETS),minimum_gain=config['minimum_gain'])
    write(folder/f'{key}-result.json',result)
    return result


def proposal_tactics(directory,reference,key,deadline,config=None):
    if config and config.get("runtime")=="ten":
        from .five_full_cycle import tactical_gate
        from .fixed_lineup import FixedModel
        return tactical_gate(directory,key,FixedModel(directory,key).serving_deck)
    from .league_gate import cases
    from .search_budget_eval import consistent_tactical_fixture,expected_actions,case_job
    results=[]
    for owner,name,s,side,goal in cases():
        if owner!=key:continue
        if time.time()>=deadline:return dict(complete=False,passed=False,cases=results)
        s,_=consistent_tactical_fixture(s);foe='b' if side=='a' else 'a'
        row=case_job(dict(state=s,side=side,budget=0,seed=0,deadline=deadline,
            expected=expected_actions(s,side,goal),policies={side:(str(directory),key),foe:(str(reference),'weave-rush')}))
        results.append(dict(name=name,**row))
    return dict(complete=bool(results) and all(r['complete'] for r in results),
                passed=bool(results) and all(r['complete'] and r['passed'] for r in results),cases=results)


def run(out,config):
    PRESETS,SEATS,CARD_IDS,LeagueModel,load_numeric,export_model,probe_results,deck=cycle_ops(config)
    if config.get("runtime")=="ten":
        from .fixed_lineup import identity as active_rule_hash
    else:
        active_rule_hash=rule_hash
    import torch
    torch.set_num_threads(1);torch.manual_seed(20260920)
    out=Path(out);initial=out/'initial';deadlines=config['deadlines']
    if active_rule_hash()!=config['rule_hash']:raise ValueError('Rules changed after planning')
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    from .build_acceptance import check_source
    check_source(config['initial'],research_warm_start=config['research_warm_start'],baseline_builds=config.get('baseline_builds'))
    if config.get('runtime')=='ten':
        copy_models(config['initial'],initial,PRESETS)
    else:
        migrate_numeric(config['initial'],initial)
    builds={k:canonical_build((config.get('baseline_builds') or {}).get(k) or LeagueModel(initial,k).serving_deck) for k in PRESETS}
    # Bind a canonical deck copy to the unchanged initial numerical weights.
    for key in PRESETS:
        net,origin=load_numeric(initial,key,'cpu')
        origin={**(getattr(LeagueModel(initial,key),'manifest',{}).get('origin') or origin),
                'canonical_build_identity':build_identity(builds[key])}
        export_model(net,initial,key,builds[key],origin=origin)
    initial_identity={k:LeagueModel(initial,k).version for k in PRESETS}
    write(out/'protocol.json',dict(schema='paired_full_cycle_v1',scoring='equal_opponent_and_seat',
        plan=config,initial_models=initial_identity,baseline_builds=builds,
        fixed_endpoint=True,confirmation_family_size=len(PRESETS),automatic_serving_approval=False))
    probe=probes(out,min(deadlines['foundation'],time.time()+180),config['seeds']['probes'],config)
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        foundation=train(pool,out,initial,initial,builds,builds,deadlines['foundation'],config,'foundation',probe)
        proposals=tune(pool,out,foundation,builds,deadlines['search'],config)
        write(out/'locked-proposals.json',dict(proposals=proposals,locked_at=time.time(),confirmation_not_read=True))
        # Both variants start from identical policy weights, with freshly reset Adam.
        target=config['adaptation_steps']
        if config.get('runtime')=='ten':
            branches=interleave_training({
                'baseline':_train_iter(pool,out,foundation,foundation,builds,builds,deadlines['candidate_adaptation'],config,'baseline-adaptation',probe,update_target=target),
                'candidate':_train_iter(pool,out,foundation,foundation,proposals,builds,deadlines['candidate_adaptation'],config,'candidate-adaptation',probe,update_target=target)})
            baseline_adapt=branches['baseline'];candidate_adapt=branches['candidate']
        else:
            baseline_adapt=train(pool,out,foundation,foundation,builds,builds,deadlines['baseline_adaptation'],config,'baseline-adaptation',probe,update_target=target)
            candidate_adapt=train(pool,out,foundation,foundation,proposals,builds,deadlines['candidate_adaptation'],config,'candidate-adaptation',probe,update_target=target)
        bstats=json.loads((out/'baseline-adaptation/summary.json').read_text())['totals']
        cstats=json.loads((out/'candidate-adaptation/summary.json').read_text())['totals']
        fairness={k:bstats[k]['steps']==cstats[k]['steps']==target and
            LeagueModel(out/'baseline-adaptation/initial',k).version==LeagueModel(out/'candidate-adaptation/initial',k).version for k in PRESETS}
        write(out/'adaptation-budget.json',dict(planned_steps_per_variant=target,baseline=bstats,candidate=cstats,fair=fairness,
                                              common_source=str(foundation),common_opponents=str(foundation)))
        decisions={};confirmations={}
        for key in PRESETS:
            if not fairness[key] or build_identity(builds[key])==build_identity(proposals[key]):
                confirmations[key]=dict(complete=False,decision='inconclusive',reason='no_change_or_incomplete_adaptation')
            else:
                confirmations[key]=gate_comparison(pool,out,key,baseline_adapt,candidate_adapt,builds[key],proposals[key],
                    foundation,builds,config,'confirmation',deadlines['confirmation'])
        write(out/'confirmation-decisions.json',confirmations)
        # Final audit compares the locked candidate to the ORIGINAL policy/build,
        # so degradation in the baseline adaptation cannot make a weak candidate pass.
        for key in PRESETS:
            audit=dict(complete=False,reason='not_accepted_for_audit');tactical=dict(passed=False,complete=False)
            if confirmations[key].get('decision')=='accepted':
                audit=gate_comparison(pool,out,key,initial,candidate_adapt,builds[key],proposals[key],
                    initial,builds,config,'audit-gate',deadlines['audit'])
                tactical=proposal_tactics(candidate_adapt,initial,key,deadlines['audit'],config)
            decisions[key]=delivery_decision(builds[key],proposals[key],confirmations[key],audit,
                fair_adaptation=fairness[key],tactical_passed=tactical['passed'],smoke=config['smoke'],audit_margin=config['audit_margin'])
            decisions[key]['tactical']=tactical
        recommended=out/'audited-models';recommended.mkdir()
        delivered_builds={}
        for key,d in decisions.items():
            source=candidate_adapt if d['build_quality_approved'] else initial
            for ext in ('json','npz'):shutil.copy2(source/f'{key}.{ext}',recommended/f'{key}.{ext}')
            delivered_builds[key]=d['recommended_build']
            d['model_sha256']=LeagueModel(recommended,key).version
            assert build_identity(LeagueModel(recommended,key).serving_deck)==build_identity(delivered_builds[key])
        write(out/'audit-decisions.json',decisions)
        report_models=recommended;report_builds=delivered_builds
        if config.get('runtime'):
            report_models=out/'report-models';report_models.mkdir();report_builds={}
            for key,d in decisions.items():
                src=candidate_adapt if d['build_quality_approved'] else baseline_adapt
                for ext in ('json','npz'):shutil.copy2(src/f'{key}.{ext}',report_models/f'{key}.{ext}')
                report_builds[key]=LeagueModel(report_models,key).serving_deck
        report_complete=evaluate(pool,out,initial,report_models,builds,report_builds,deadlines.get('evaluation',deadlines['audit']),config)
        search_complete=json.loads((out/'cards/status.json').read_text())['workflow_complete']
        evidence_complete=all(build_identity(builds[k])==build_identity(proposals[k]) or
            (fairness[k] and confirmations[k].get('complete') and
             (confirmations[k].get('decision')!='accepted' or
              (decisions[k]['audit'].get('complete') and decisions[k]['tactical'].get('complete')))) for k in PRESETS)
        complete=report_complete and search_complete and evidence_complete and all(fairness.values())
    delivery=out/'recommended';delivery.mkdir()
    for key,d in decisions.items():
        d['audited_model_sha256']=d['model_sha256']
        if not complete:
            d.update(recommendation_status='retained_baseline',build_quality_approved=False,
                     recommended_build=builds[key],reason='report_incomplete')
        source=candidate_adapt if d['build_quality_approved'] else initial
        for ext in ('npz','json'):shutil.copy2(source/f'{key}.{ext}',delivery/f'{key}.{ext}')
        if key=='zhenhong':
            manifest=json.loads((delivery/(key+'.json')).read_text());origin=manifest.get('origin') or {}
            progress=max((LeagueModel(src,key).manifest.get('origin') or {}).get('setup_reward_updates',0) for src in (foundation,baseline_adapt,candidate_adapt))
            manifest['origin']=dict(origin,setup_reward_updates=max(progress,origin.get('setup_reward_updates',0)))
            write(delivery/(key+'.json'),manifest)
        d['model_sha256']=LeagueModel(delivery,key).version
    write(out/'delivery-decision.json',decisions)
    write(out/'status.json',dict(workflow_complete=complete,quality_approved=complete and all(d['build_quality_approved'] for d in decisions.values()),
        phase='complete' if complete else 'incomplete',recommendations={k:d['recommendation_status'] for k,d in decisions.items()},
        automatic_serving_approval=False,finished=time.time(),hard_deadline=deadlines['shutdown']))
    with (out/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## 交付决定\n\n')
        for key,d in decisions.items():f.write(f"- {key}: {d['recommendation_status']}；{d['reason']}。\n")
        f.write('\n候选、确认通过、审计通过和保留原配牌分别记录；流程完成不代表质量通过，不自动发布。\n')
    members=[p for p in out.rglob('*') if p.is_file() and
             not any(part in ('frozen','generations','sources') for part in p.relative_to(out).parts) and
             p.suffix in ('.json','.jsonl','.npz','.pt','.gz','.md','.csv') and p.name!='delivery.json']
    manifest={}
    with zipfile.ZipFile(out/'delivery.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(members):
            name=p.relative_to(out).as_posix();data=p.read_bytes()
            manifest[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2))
    write(out/'delivery.json',dict(sha256=hashlib.sha256((out/'delivery.zip').read_bytes()).hexdigest(),members=len(manifest)))
