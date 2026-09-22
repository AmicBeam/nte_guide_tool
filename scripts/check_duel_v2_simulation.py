#!/usr/bin/env python3
"""Bounded simulation parity and timing; no optimization, training or serving writes."""
from pathlib import Path
from copy import deepcopy
import argparse,gzip,hashlib,json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.fixed_lineup import identity,all_features,candidate_names,write
from app.modules.card_game.rl.information_search import search
from app.modules.card_game.rl.build_acceptance import check_source


def research_copy(source,target,key,expected_rule):
    """Explicit research-only rebind for a validated schema, never edits sources."""
    check_source(source)
    mpath=source/f'{key}.json';path=source/f'{key}.npz';m=json.loads(mpath.read_text(encoding='utf-8'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if (m['schema'] not in ('fixed_ten_v1','fixed_ten_wdl_v1') or m['rule_hash']!=expected_rule
        or m['sha256']!=digest or m['features']!=all_features() or m['candidates']!=candidate_names() or m['deck']!=key):
        raise ValueError('Unverified source for simulation research rebind')
    target.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target/path.name)
    m['origin']=dict(kind='presentation_simulation_research_only',source_manifest_sha=hashlib.sha256(mpath.read_bytes()).hexdigest(),
        source_rule_hash=expected_rule,source_weights_sha=digest,source_origin=m.get('origin'))
    m['rule_hash']=identity();m['validation']={'approved':False};m.pop('serving_authorization',None);m.pop('runtime_compatibility',None)
    m['automatic_serving_approval']=False;write(target/mpath.name,m)
    rt.model(target,key) # Validates shapes, finite weights, build hash and current contract.
    return dict(source=str(path),sha256=digest,source_rule_hash=expected_rule,target_rule_hash=identity())


def materialize(d):
    return new_game(seed=d['seed'],first_side=d['first'],decks=d['decks'])


def run(source,out,expected_rule,seeds,seconds):
    started=time.time();deadline=started+seconds
    migrated=[research_copy(source/'common-initial',out/'models/wdl','zhenhong',expected_rule)]
    for key in ('starter','weave-rush','quick-rush','zhenhong'):
        migrated.append(research_copy(source/'sources',out/'models/fixed',key,expected_rule))
    write(out/'migration.json',migrated)
    # Replay originally recorded real games. The normal path must also match
    # the original action SHA, independently of the new hypothetical path.
    index=[json.loads(x) for x in (source/'evaluation/index.jsonl').read_text().splitlines()]
    selected=[r for r in index if r.get('replay')]
    parity=[];saved=[]
    for r in selected:
        if time.time()>=deadline:raise RuntimeError('Parity budget exhausted')
        d=json.loads(gzip.open(source/'evaluation/raw'/(r['id']+'.json.gz'),'rt',encoding='utf-8').read())
        slow=materialize(d);fast=deepcopy(slow)
        for record in d['actions']:
            side,action=record['side'],record['action']
            slow=clean(apply_action(slow,side,action));fast=clean(simulate_action(fast,side,action))
            assert slow==fast,(r['id'],record)
            digest=hashlib.sha256(json.dumps(slow,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            assert digest==record['state_sha256'],(r['id'],'historical normal-path mismatch')
        parity.append(dict(id=r['id'],steps=len(d['actions']),winner=slow['winner']))
        if len(saved)<6:saved.append(d)
    write(out/'parity.json',parity)
    timing=[]
    for i,d in enumerate(saved):
        for repeat in range(2):
            row=dict(id=d['id'],repeat=repeat,steps=len(d['actions']))
            for fast in ((False,True) if (i+repeat)%2==0 else (True,False)):
                state=materialize(d);t=time.perf_counter()
                for record in d['actions']:
                    if time.time()>=deadline:raise RuntimeError('Timing budget exhausted')
                    state=clean((simulate_action if fast else apply_action)(state,record['side'],record['action']))
                row['fast' if fast else 'full']=time.perf_counter()-t
            timing.append(row)
    write(out/'transition-timing.json',timing)
    policies={'a':rt.model(out/'models/wdl','zhenhong'),'b':rt.model(out/'models/fixed','starter')}
    roots=[]
    for i in range(2):
        state=new_game(seed=seeds['probes']+i,first_side=('a','b')[i],decks={s:p.serving_deck for s,p in policies.items()})
        captured=0
        for step in range(70):
            if state['phase']=='finished':break
            side=acting_side(state)
            if side=='a' and (captured==0 or step>=14):
                roots.append(state);captured+=1
                if captured==2:break
            actions,(x,c)=rt.decision(state,side);state=clean(apply_action(state,side,actions[int(policies[side].scores(x,c).argmax())]))
    searches=[]
    for i,state in enumerate(roots):
        pair={};row=dict(root=i)
        for fast in ((False,True) if i%2==0 else (True,False)):
            t=time.perf_counter();pair[fast]=search(state,'a',policies,seed=seeds['probes']+1000+i,simulations=256,
                deadline=deadline,runtime=rt,fast_simulation=fast)
            row['fast' if fast else 'full']=time.perf_counter()-t
        assert pair[False]['complete'] and pair[True]['complete']
        np.testing.assert_array_equal(pair[False]['visits'],pair[True]['visits'])
        np.testing.assert_array_equal(pair[False]['mean_values'],pair[True]['mean_values'])
        row.update(simulations=pair[True]['simulations'],visits_identical=True,values_identical=True)
        searches.append(row)
    write(out/'search-timing.json',searches)
    for record in migrated:assert hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()==record['sha256']
    write(out/'summary.json',dict(complete=True,elapsed=time.time()-started,historical_games=len(parity),
        historical_actions=sum(r['steps'] for r in parity),timing_games=len(timing),roots=len(searches),
        transition_speedup=sum(r['full'] for r in timing)/sum(r['fast'] for r in timing),
        search_speedup=sum(r['full'] for r in searches)/sum(r['fast'] for r in searches),
        source_unchanged=True,no_training=True,no_serving_approval=True,current_rule_hash=identity()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-rule-hash',required=True);p.add_argument('--seed-reservation',type=Path,required=True);p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args()
    if not 60<=a.seconds<=900:raise ValueError('Diagnostic bound is 60..900 seconds')
    a.output.mkdir(parents=True,exist_ok=False);run(a.source_run,a.output,a.source_rule_hash,json.loads(a.seed_reservation.read_text()),a.seconds)
