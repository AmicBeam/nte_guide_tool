#!/usr/bin/env python3
"""Reverify an existing private teacher diagnosis, without new seeds or updates."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=180)
    a=p.parse_args()
    if not 30<=a.seconds<=600:p.error('Bounded verification seconds required')
    if a.output.exists():raise ValueError('New verification output required')
    from app.modules.card_game.rl import recovery_runtime as rt,information_search as core
    from app.modules.card_game.rl.teacher_validation import verify_counterfactuals,sampled_build_support
    from app.modules.card_game.rl.episode_replay import load_episode
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    config=json.loads((a.run_dir/'config.json').read_text());result=json.loads((a.run_dir/'result.json').read_text())
    paths={hashlib.sha256(p.read_bytes()).hexdigest():p for p in
           Path(config['source_run']).joinpath('cycles/1/sampling/episodes').glob('*.json.gz')}
    deadline=diagnostic_deadline(time.time(),a.seconds);reports=[]
    for index,record in enumerate(result['roots']):
        if time.time()>=deadline:break
        data=load_episode(paths[record['episode_sha256']]);job=data['job']
        if job.get('training') is not True:raise ValueError('Declared TRAINING roots required')
        policies={s:rt.model(*spec) for s,spec in job['policies'].items()}
        if {s:p.version for s,p in policies.items()}!=record['models']:raise ValueError('Model arrays changed')
        state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks']);root=None
        for step,entry in enumerate(data['actions']):
            if time.time()>=deadline:break
            if step==record['step']:root=deepcopy(state)
            state=clean(apply_action(state,entry['side'],entry['action']))
        if time.time()>=deadline:break
        if root is None or state['phase']!='finished' or state.get('winner')!=data['winner']:
            raise ValueError('Source trajectory terminal mismatch')
        viewer=acting_side(root)
        with gzip.open(a.run_dir/f'root-{index}-private-continuations.json.gz','rt') as f:records=json.load(f)
        verification=verify_counterfactuals(root,viewer,records,rt,deadline=deadline,policies=policies)
        support=[]
        if verification['complete']:
            for row in records:
                if time.time()>=deadline:break
                world=core.sample_world(root,viewer,row['world_seed'],runtime=rt)
                support.append(dict(world_seed=row['world_seed'],sides=sampled_build_support(world,policies)))
        reports.append(dict(key=record['key'],game=record['game'],step=record['step'],viewer=viewer,
            verification=verification,sampled_build_support=support,
            sample_support_complete=len(support)==len(records),
            input_trace_sha256=hashlib.sha256((a.run_dir/f'root-{index}-private-continuations.json.gz').read_bytes()).hexdigest()))
    complete=len(reports)==len(result['roots']) and result['complete'] and all(
        r['verification']['complete'] and r['sample_support_complete'] for r in reports)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(complete=complete,roots=reports,
        input_result_sha256=hashlib.sha256((a.run_dir/'result.json').read_bytes()).hexdigest(),
        policy_updates=0,new_seeds=0,quality_approved=False),indent=2))
    print(json.dumps(dict(complete=complete,verified_branches=sum(r['verification']['verified_branches'] for r in reports))))


if __name__=='__main__':main()
