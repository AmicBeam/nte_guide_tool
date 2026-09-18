"""Frozen-model budget comparison. NumPy only; no training/update entry points."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from copy import deepcopy
from random import Random
import hashlib,json,shutil,time
import numpy as np
from .league_schema import PRESETS,deck,rule_hash
from .league_policy import LeagueModel
from .league_rollout import decision,model,clean
from .information_search import search
from .league_search import prove_turn
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side

BUDGETS=(0,24,128,256)


def write(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def hashes(directory):
    directory=Path(directory)
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def freeze_models(learner,opponent,out):
    out=Path(out)
    for label,source in (('learner',Path(learner)),('opponent',Path(opponent))):
        dest=out/label;dest.mkdir(parents=True)
        for key in PRESETS:
            LeagueModel(source,key)  # Identity/schema/weights validated before copy.
            for suffix in ('.json','.npz'):
                path=source/(key+suffix)
                if path.is_symlink():raise ValueError('Model symlinks not accepted')
                shutil.copyfile(path,dest/path.name)
            LeagueModel(dest,key)
    return hashes(out)


def named_action(state,side,action):
    cid=next((c['card_id'] for c in state['sides'][side]['hand'] if c['instance_id']==action.get('card_id')),None)
    return dict(action=action,card=cid)


def choose(state,side,policies,budget,seed,deadline):
    """Time encoding+inference/search; model loading and rule commit are excluded."""
    started=time.perf_counter()
    if budget:
        r=search(state,side,policies,seed=seed,simulations=budget,deadline=deadline)
        ix=r['search_choice'];actions=r['actions']
        result=dict(complete=r['complete'],simulations=r['simulations'],requested=r['requested'],
                    legal=len(actions),covered=r['roots_covered'],nodes=r['nodes'],
                    changed=ix!=r['raw_choice'],visits=r['visits'].tolist(),
                    values=r['mean_values'].tolist())
    else:
        actions,(x,c)=decision(state,side);ix=int(policies[side].scores(x,c).argmax())
        result=dict(complete=True,simulations=0,requested=0,legal=len(actions),
                    covered=None,nodes=0,changed=False,visits=None,values=None)
    result['milliseconds']=(time.perf_counter()-started)*1000
    result['index']=ix;result['selected']=named_action(state,side,actions[ix])
    return actions[ix],result


def case_job(job):
    try:
        policies={s:model(*spec) for s,spec in job['policies'].items()}
        _,row=choose(job['state'],job['side'],policies,job['budget'],job['seed'],job['deadline'])
        expected=job['expected']
        row['passed']=row['index'] in expected if expected is not None and row['complete'] else None
        return row
    except Exception as exc:
        return dict(complete=False,error=repr(exc),passed=None)


def game_job(job):
    state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks'])
    policies={s:model(*spec) for s,spec in job['policies'].items()}
    rows=[]
    try:
        for step in range(400):
            if state['phase']=='finished':break
            if time.time()>=job['deadline']:
                return dict(complete=False,reason='deadline',decisions=rows)
            side=acting_side(state)
            action,result=choose(state,side,policies,job['budget'] if side=='a' else 0,
                                 job['seed']+1000003*(step+1),job['deadline'])
            if not result['complete']:
                return dict(complete=False,reason='incomplete_search',decisions=rows)
            if side=='a':
                # Audit AFTER selecting, outside decision timing; no action override.
                proof=prove_turn(state,side,depth=1,node_budget=result['legal']+1,deadline=job['deadline']) if state['phase']=='playing' else None
                audited=bool(proof and proof['eligible'] and proof['roots_visited']==result['legal'])
                result['certified_lethal_available']=bool(proof['paths']) if audited else None
                result['missed_certified_lethal']=bool(proof['paths'] and result['index'] not in proof['paths']) if audited else None
                result['phase']=state['phase'];result['turn']=state['turn']
                rows.append(result)
            state=clean(apply_action(state,side,action))
        return dict(complete=state['phase']=='finished',winner=state.get('winner'),
                    turn=state['turn'],decisions=rows)
    except Exception as exc:
        return dict(complete=False,error=repr(exc),decisions=rows)


def consistent_tactical_fixture(state):
    """Repair ONLY authored fixtures whose hand overrides broke deck accounting.

    Preserve hand and tactical board; rebuild missing deck stock within the legal
    4x8, max-two rule. Missing generated family cards go to discard, not hand.
    Never apply to natural games or user recordings. Report the changed counts.
    """
    from collections import Counter
    from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
    from app.modules.card_game.engine.duel_v2.state import card_instance
    out=deepcopy(state);changes={}
    for side,t in out['sides'].items():
        before={z:len(t[z]) for z in ('hand','deck','discard')}
        family=sum(c['card_id']=='NF01' for c in t['hand']+t['discard'])
        missing=int(t.get('nanali_family',0))-family
        if missing<0:raise ValueError('Unsupported generated fixture accounting')
        t['discard'] += [card_instance(out,'NF01',side) for _ in range(missing)]
        known=Counter(c['card_id'] for c in t['hand']+t['discard'] if not c.get('derived'))
        pool=[]
        for hero in t['characters']:
            ids=[cid for cid,c in CARDS.items() if c['character_id']==hero and not c.get('derived')]
            n=8-sum(known[c] for c in ids)
            if n<0 or any(known[c]>2 for c in ids):raise ValueError('Invalid fixture hand/discard')
            ordered=[c['card_id'] for c in t['deck'] if c['character_id']==hero and not c.get('derived')]
            ordered += [cid for cid in ids for _ in range(2)]
            for cid in ordered:
                if n and known[cid]<2:
                    pool.append(cid);known[cid]+=1;n-=1
            if n:raise ValueError('Cannot complete fixture deck')
        t['deck']=[card_instance(out,cid,side) for cid in pool]
        t['revealed_ids']=[cid for cid in t.get('revealed_ids',[]) if any(c['instance_id']==cid for c in t['hand'])]
        validate_deck(dict(id='budget-fixture-v2',name='预算对照夹具',character_ids=list(t['characters']),
                           card_ids=[c['card_id'] for c in t['hand']+t['deck']+t['discard'] if not c.get('derived')]))
        changes[side]=dict(before=before,after={z:len(t[z]) for z in before})
    return out,changes


def expected_actions(state,side,goal):
    from app.modules.card_game.engine.duel_v2.state import hero
    actions,_=decision(state,side);expected=[]
    for i,a in enumerate(actions):
        cid=named_action(state,side,a)['card']
        if goal in ('M02','N06'):ok=cid==goal
        elif goal=='user_nanali':ok=a=={'type':'attack','character_id':'nanali'}
        else:
            t=apply_action(state,side,a)
            ok=t.get('winner')==side if goal=='lethal' else (hero(t,side,'zero')['hp']>0 and hero(t,'b' if side=='a' else 'a','baicang')['hp']==0 and any(c['card_id']=='NF01' for c in t['sides'][side]['hand']))
        if ok:expected.append(i)
    return expected


def natural_cases(learner,opponent,deadline):
    """Deterministic stratified reservoir from new complete RAW policy games."""
    cases=[];seen=set()
    for ki,key in enumerate(PRESETS):
        foe=PRESETS[(ki+1)%3];candidates=[]
        policies={'a':LeagueModel(learner,key),'b':LeagueModel(opponent,foe)}
        for game in range(2):
            seed=990100000+ki*100+game
            s=new_game(seed=seed,first_side='ab'[game],decks={'a':deck(key),'b':deck(foe)})
            pending=[]
            for step in range(400):
                if s['phase']=='finished':break
                if time.time()>=deadline:raise TimeoutError('Natural case generation deadline')
                side=acting_side(s);actions,(x,c)=decision(s,side)
                if side=='a' and s['phase']=='playing' and len(actions)>1:
                    fp=hashlib.sha256(x.tobytes()+c.tobytes()).hexdigest()
                    pending.append(dict(state=deepcopy(s),side=side,key=key,foe=foe,source_seed=seed,
                                        source_first='ab'[game],step=step,fingerprint=fp))
                s=clean(apply_action(s,side,actions[int(policies[side].scores(x,c).argmax())]))
            if s['phase']=='finished':candidates+=pending
        rng=Random(990200000+ki);rng.shuffle(candidates);selected=[]
        for lo,hi in ((0,5),(6,12),(13,1000)):
            bucket=[c for c in candidates if lo<=c['state']['turn']<=hi and c['fingerprint'] not in seen]
            taken=0
            for c in bucket:
                if c['fingerprint'] in seen:continue
                selected.append(c);seen.add(c['fingerprint']);taken+=1
                if taken==2:break
        for c in candidates:
            if len(selected)>=6:break
            if c['fingerprint'] not in seen:selected.append(c);seen.add(c['fingerprint'])
        if len(selected)<6:raise ValueError('Insufficient distinct natural positions')
        for i,c in enumerate(selected):
            proof=prove_turn(c['state'],'a',depth=1,node_budget=256,deadline=deadline)
            c.update(name=f'natural-{key}-{i}',kind='natural',
                     expected=list(proof['paths']) if proof['paths'] else None)
            cases.append(c)
    return cases


def game_jobs(models,deadline,seeds_per_pair):
    jobs=[];index=0
    for ki,key in enumerate(PRESETS):
        for fi,foe in enumerate(k for k in PRESETS if k!=key):
            for seed_index in range(seeds_per_pair):
                for first in ('a','b'):
                    order=BUDGETS[index%4:]+BUDGETS[:index%4];index+=1
                    for budget in order:
                        jobs.append(dict(key=key,foe=foe,budget=budget,first=first,
                                         seed=990300000+ki*10000+fi*1000+seed_index,
                                         deadline=deadline,decks={'a':deck(key),'b':deck(foe)},
                                         policies={'a':(str(models/'learner'),key),'b':(str(models/'opponent'),foe)}))
    return jobs


def latency(rows):
    values=[r['milliseconds'] for r in rows if r.get('complete') and 'milliseconds' in r]
    return dict(n=len(values),median_ms=float(np.median(values)) if values else None,
                p95_ms=float(np.percentile(values,95)) if values else None,
                max_ms=max(values) if values else None)


def summarize(games,positions,expected_games):
    summary=dict(complete=len(games)==expected_games and all(g['complete'] for g in games) and all(p['complete'] for p in positions),
                 expected_games=expected_games,completed_games=sum(g['complete'] for g in games),budgets={})
    def score(g):
        if g.get('winner') not in ('a','b','draw'):raise ValueError('Invalid completed game winner')
        return {'a':1.,'b':0.,'draw':.5}[g['winner']]
    baseline={(g['key'],g['foe'],g['seed'],g['first']):g for g in games if g['budget']==0}
    for budget in BUDGETS:
        group=[g for g in games if g['budget']==budget];fixed=[p for p in positions if p['budget']==budget]
        gains=losses=ties=missing=0
        for g in group:
            b=baseline.get((g['key'],g['foe'],g['seed'],g['first']))
            if not g['complete'] or b is None or not b['complete']:missing+=1;continue
            d=score(g)-score(b);gains+=d>0;losses+=d<0;ties+=d==0
        all_cases={}
        for row in fixed:all_cases.setdefault(row['case'],[]).append(row)
        by_case={name:[json.dumps(r['selected']['action'],sort_keys=True) for r in rows]
                 for name,rows in all_cases.items() if all(r['complete'] for r in rows)}
        tactical=[p for p in fixed if p['kind']=='tactical' and p.get('passed') is not None]
        decisions=[r for g in group if g['complete'] for r in g['decisions']]
        summary['budgets'][budget]=dict(wins=sum(g['complete'] and g.get('winner')=='a' for g in group),
            complete=sum(g['complete'] for g in group),scheduled=len(group),paired_gains=int(gains),paired_losses=int(losses),
            paired_ties=int(ties),paired_missing=missing,fixed_latency=latency(fixed),
            natural_fixed_latency=latency([r for r in fixed if r['kind']=='natural']),game_latency=latency(decisions),
            tactical_passes=sum(p['passed'] for p in tactical),tactical_trials=len(tactical),
            unstable_cases=sum(len(set(v))>1 for v in by_case.values()),assessed_cases=len(by_case),partial_cases=len(all_cases)-len(by_case),
            known_lethal_opportunities=sum(r.get('certified_lethal_available') is True for r in decisions),
            missed_known_lethal=sum(r.get('missed_certified_lethal') is True for r in decisions),
            by_strategy={k:dict(wins=sum(g['complete'] and g.get('winner')=='a' for g in group if g['key']==k),
                                complete=sum(g['complete'] for g in group if g['key']==k)) for k in PRESETS})
    paired={}
    for g in games:paired.setdefault((g['key'],g['foe'],g['seed'],g['first']),{})[g['budget']]=g
    common=[v for v in paired.values() if set(v)==set(BUDGETS) and all(g['complete'] for g in v.values())]
    summary['all_budget_complete_pairs']=len(common)
    summary['all_budget_matched_wins']={b:sum(v[b].get('winner')=='a' for v in common) for b in BUDGETS}
    summary['automatic_recommendation']=False
    summary['uncertainty']='Small paired sample; no claim of stable overall gain from aggregate wins alone.'
    return summary


def experiment(out,learner,opponent,deadline,*,seeds_per_pair=2,repeats=3,workers=4,user_snapshot=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.time();models=out/'models'
    before=freeze_models(learner,opponent,models)
    write(out/'frozen-models.json',dict(learner_source=str(learner),opponent_source=str(opponent),files=before))
    config=dict(kind='frozen_search_budget_comparison_v1',budgets=BUDGETS,seeds_per_pair=seeds_per_pair,
                repeats=repeats,workers=workers,rule_hash=rule_hash(),started=started,deadline=deadline,
                weights_updated=False,opponents='same frozen raw-policy models for every budget',
                timing='wall time for encoding+inference/search, excludes model loading, oracle audit and rule commit; concurrent workers')
    write(out/'config.json',config);write(out/'status.json',dict(phase='collect_cases',deadline=deadline))
    cases=natural_cases(models/'learner',models/'opponent',deadline-60)
    from .league_gate import cases as tactical_cases
    for key,name,s,side,goal in tactical_cases():
        s,adjustments=consistent_tactical_fixture(s)
        foe_side='b' if side=='a' else 'a'
        foe=next((k for k in PRESETS if set(deck(k)['character_ids'])==set(s['sides'][foe_side]['characters'])),'starter')
        cases.append(dict(key=key,foe=foe,name=name,state=s,side=side,kind='tactical',fixture_version=2,adjustments=adjustments,expected=expected_actions(s,side,goal)))
    if user_snapshot:
        s=json.loads(Path(user_snapshot).read_text())
        cases.append(dict(key='starter',foe='weave-rush',name='user-turn3',state=s,side='a',kind='user_review',expected=expected_actions(s,'a','user_nanali')))
    from .information_search import sample_world
    for case in cases:sample_world(case['state'],case['side'],990499999)
    write(out/'cases.json',cases)
    positions=[];games=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        jobs=[]
        for ci,case in enumerate(cases):
            side=case['side'];foe_side='b' if side=='a' else 'a'
            for rep in range(repeats):
                for budget in BUDGETS:
                    jobs.append(dict(state=case['state'],side=side,budget=budget,case=case['name'],kind=case['kind'],
                                     expected=case['expected'],rep=rep,seed=990500000+ci*100+rep,
                                     deadline=deadline-60,policies={side:(str(models/'learner'),case['key']),foe_side:(str(models/'opponent'),case['foe'])}))
        futures={pool.submit(case_job,j):j for j in jobs}
        for f in as_completed(futures):
            j=futures[f];positions.append(dict(case=j['case'],kind=j['kind'],budget=j['budget'],rep=j['rep'],seed=j['seed'],**f.result()))
            if len(positions)%12==0:
                write(out/'positions.json',positions);write(out/'status.json',dict(phase='fixed_positions',done=len(positions),total=len(jobs),deadline=deadline))
        write(out/'positions.json',positions)
        jobs=game_jobs(models,deadline-30,seeds_per_pair)
        futures={pool.submit(game_job,j):j for j in jobs}
        for f in as_completed(futures):
            j=futures[f];games.append(dict(key=j['key'],foe=j['foe'],seed=j['seed'],first=j['first'],budget=j['budget'],**f.result()))
            write(out/'games.json',games);write(out/'status.json',dict(phase='paired_games',done=len(games),total=len(jobs),deadline=deadline))
    summary=summarize(games,positions,len(jobs));summary.update(started=started,finished=time.time(),deadline=deadline)
    after=hashes(models);summary['weights_unchanged']=before==after
    if before!=after:raise ValueError('Frozen model files changed during evaluation')
    write(out/'summary.json',summary)
    write(out/'status.json',dict(phase='complete' if summary['complete'] else 'incomplete',finished=time.time(),deadline=deadline,weights_unchanged=True))
