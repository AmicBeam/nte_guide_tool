#!/usr/bin/env python3
"""Complete visible-rule trajectories on alternative kits; no model learning."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--sampler',choices=('legacy','recovery-v2'),default='recovery-v2')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if not 60<=a.seconds<=600:p.error('Bounded CPU diagnostic required')
    if not a.run:print(json.dumps(dict(dry_run=True,games=20)));return
    if a.output.exists():raise ValueError('New output required')
    from app.modules.card_game.rl.covered_rollout import rule_decision
    from app.modules.card_game.rl import recovery_runtime as rt,information_search as core
    from types import SimpleNamespace
    from app.modules.card_game.rl.recovery_sampler import determinize as corrected_sample
    sampler=SimpleNamespace(decision=rt.decision,determinize=corrected_sample if a.sampler=='recovery-v2' else rt.determinize)
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.cross_schedule import ROSTER
    from app.modules.card_game.content.duel_v2 import CARDS,validate_deck
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
    from app.modules.card_game.engine.duel_v2.simulation import simulate_action
    from scripts.train_duel_v2_current_round import preset_decks
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    start=time.time();deadline=diagnostic_deadline(start,a.seconds)
    if deadline-start<60:raise ValueError('Insufficient diagnostic window')
    decks=preset_decks();variants={}
    for key,build in decks.items():
        all_ids=[c for hero in build['character_ids'] for c,v in CARDS.items()
                 if v['character_id']==hero and not v.get('derived')]
        alternate=[]
        for hero in build['character_ids']:
            kit=[c for c in all_ids if CARDS[c]['character_id']==hero]
            unused=[c for c in kit if c not in build['card_ids']]
            chosen=(unused+[c for c in kit if c not in unused])[:4]
            alternate.extend(c for c in chosen for _ in range(2))
        variants[key]=[validate_deck(dict(build,card_ids=all_ids)),
                       validate_deck(dict(build,card_ids=alternate))]
    seeds=claim_seeds(ROOT/'artifacts/rl-seed-ledger.json',str(a.output.resolve()))
    a.output.mkdir(parents=True)
    (a.output/'config.json').write_text(json.dumps(dict(seeds=seeds,deadline=deadline,variants=variants,
        games=20,policy='VisibleEngineRulePolicy',sampler=a.sampler,policy_updates=0,strength_approved=False),indent=2))
    results=[]
    for k,key in enumerate(ROSTER):
        for variant,build in enumerate(variants[key]):
            for first in ('a','b'):
                if time.time()>=deadline:break
                seed=seeds['audit']+k*10+variant
                state=new_game(seed=seed,first_side=first,decks={'a':build,'b':decks[ROSTER[(k+1)%5]]})
                fast=deepcopy(state);trace=[];seen=set();particle_checks=0
                for step in range(512):
                    if state['phase']=='finished' or time.time()>=deadline:break
                    actor=acting_side(state);legal,index=rule_decision(state,actor)
                    numeric,_=rt.decision(state,actor)
                    if legal!=numeric:raise ValueError('Rule/numeric legal masks differ')
                    seen.update(c['card_id'] for t in state['sides'].values() for c in t['hand'])
                    if step%8==0 and state['phase'] in ('playing','mulligan'):
                        core.sample_world(state,actor,seeds['probes']+k*1000+variant*100+step,runtime=sampler)
                        particle_checks+=1
                    action=legal[index]
                    state=clean(apply_action(state,actor,action));fast=clean(simulate_action(fast,actor,action))
                    if state!=fast:raise ValueError('Full trajectory simulation differs from official rules')
                    trace.append(dict(side=actor,action=action))
                results.append(dict(key=key,variant=variant,first=first,seed=seed,
                    complete=state['phase']=='finished',winner=state.get('winner'),actions=len(trace),
                    hand_templates_seen=sorted(seen),particle_checks=particle_checks))
                (a.output/f'{key}-{variant}-{first}-private-actions.json').write_text(json.dumps(trace))
                print(json.dumps(results[-1]),flush=True)
    (a.output/'result.json').write_text(json.dumps(dict(complete=len(results)==20 and all(r['complete'] for r in results),
        games=results,rule_mask_and_simulation_parity=True,quality_approved=False),indent=2))


if __name__=='__main__':main()
