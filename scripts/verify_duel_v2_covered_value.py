#!/usr/bin/env python3
"""Replay unchanged auxiliary value data and attest equivalent rollout changes.

Does not optimize, choose epochs, change weights, or claim a fresh independent
confirmation. Checks every recorded action and value observation from the
already-consumed train/selection/heldout endpoints.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=600);a=p.parse_args()
    if not 60<=a.seconds<=1200:p.error('Bounded verification required')
    if a.output.exists():raise ValueError('New attestation output required')
    import numpy as np
    from app.modules.card_game.rl import covered_value as cv,recovery_runtime as rt,recovery_round as rr
    from app.modules.card_game.rl.covered_rollout import rule_decision
    from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
    from app.modules.card_game.rl.cross_schedule import ROSTER,learning_schedule
    from app.modules.card_game.rl.cross_lineup import encode,identity
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,observe,acting_side
    from scripts.train_duel_v2_current_round import preset_decks
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    end=diagnostic_deadline(time.time(),a.seconds)
    old=json.loads((a.source/'scope.json').read_text());config=json.loads((a.source/'config.json').read_text())
    if old['rule_hash']!=identity():raise ValueError('Rules changed; cannot reuse calibration')
    for name,digest in old['files'].items():
        path=(a.source/name).resolve()
        if not path.is_relative_to(a.source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Original evidence changed')
    decks=preset_decks();plan=learning_schedule()+[dict(left=k,right=k,first=f) for k in ROSTER for f in ('a','b')]
    counts={};held=[];total_actions=0
    endpoints=cv.phase_counts(old)
    for phase,n,base in (('training',endpoints['training'],config['seeds']['foundation']),
                         ('selection',endpoints['selection'],config['seeds']['selection']),
                         ('heldout',endpoints['heldout'],config['seeds']['confirmation'])):
        for ordinal in range(n):
            if time.time()>=end:raise TimeoutError('Auxiliary verification deadline')
            replica=ordinal//30;i=ordinal%30;item=plan[i];seed=base+replica*100+i//2
            rng=np.random.default_rng(seed);builds={}
            for side,key in (('a',item['left']),('b',item['right'])):
                cards=[]
                for hero in decks[key]['character_ids']:
                    kit=[c for c,v in CARDS.items() if v['character_id']==hero and not v.get('derived')]
                    if (replica+i//2)%3==0:cards+=kit
                    else:
                        bag=[c for c in kit for _ in range(2)];cards+=[bag[int(j)] for j in rng.choice(len(bag),8,replace=False)]
                builds[side]=validate_deck(dict(decks[key],card_ids=cards))
            with gzip.open(a.source/f'{phase}-{ordinal}-private.json.gz','rt') as f:game=json.load(f)
            for row in game['value_rows']:row['x']=np.asarray(row['x'],dtype=np.float32)
            if not game['complete'] or game['seed']!=seed or game['first']!=item['first']:raise ValueError('Trajectory identity mismatch')
            rows={(r['step'],r['side']):r for r in game['value_rows']}
            state=new_game(seed=seed,first_side=item['first'],decks=builds);exploration=np.random.default_rng(seed)
            for step,entry in enumerate(game['actions']):
                if time.time()>=end:raise TimeoutError('Auxiliary action verification deadline')
                actor=acting_side(state);legal,index=rule_decision(state,actor)
                reference=observe(state,actor,include_previews=True)
                ref_legal=[e['action'] for e in reference['legal_actions'] if e['action']['type']!='concede']
                if legal!=ref_legal or index!=VisibleEngineRulePolicy()(reference,tuple(ref_legal)):
                    raise ValueError('Selective previews changed rule action')
                if step<6 and state['phase']=='playing':index=int(exploration.integers(len(legal)))
                if entry!=dict(side=actor,action=legal[index]):raise ValueError('Recorded expert action differs')
                if step>=6:
                    for side in ('a','b'):
                        row=rows[step,side];x,_=encode(observe(state,side,include_previews=False),[])
                        if not np.array_equal(x,np.asarray(row['x'],dtype=np.float32)) or row['value_actor']!=int(side==actor):
                            raise ValueError('Value observation drift')
                state=clean(apply_action(state,actor,entry['action']));total_actions+=1
            if state['phase']!='finished' or state.get('winner')!=game['winner']:raise ValueError('True terminal differs')
            if any(r['z']!=(0 if state['winner']=='draw' else 1 if r['side']==state['winner'] else -1) for r in game['value_rows']):
                raise ValueError('Terminal value label differs')
            if phase=='heldout':held.append(game)
        counts[phase]=n;print(phase,n,'verified',flush=True)
    models={k:rt.model(a.source/'models',k) for k in ROSTER};metrics=rr.value_metrics(held,models)
    for key,m in metrics.items():
        for field in ('brier','log_loss'):
            if not np.isclose(m[field],old['metrics'][key][field],atol=1e-7,rtol=0):raise ValueError('Calibration changed')
    a.output.mkdir(parents=True)
    shutil.copyfile(a.source/'config.json',a.output/'config.json')
    for name in old['files']:shutil.copyfile(a.source/name,a.output/name)
    shutil.copytree(a.source/'models',a.output/'models')
    report=dict(old,algorithm_sha256=cv.identity(),metrics=metrics,
        reattestation=dict(counts=counts,actions=total_actions,source_scope_sha256=hashlib.sha256((a.source/'scope.json').read_bytes()).hexdigest(),
                          weight_updates=0,epoch_selection=False,fresh_confirmation=False,exact_rule_decisions=True))
    rr.write(a.output/'scope.json',report);cv.load_scope(a.output)
    print(json.dumps(report['reattestation']))


if __name__=='__main__':main()
