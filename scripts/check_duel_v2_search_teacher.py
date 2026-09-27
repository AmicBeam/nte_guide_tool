#!/usr/bin/env python3
"""Budget/repeat search diagnosis with fresh paired terminal continuations.

Uses consumed TRAINING roots only. New hidden worlds are reserved separately
from search seeds. Frozen visible-policy continuations are a conditional probe,
not whole-game strength confirmation or a training/serving approval.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keys',nargs='+',default=['zhenhong','murk'])
    p.add_argument('--roots-per-key',type=int,default=1)
    p.add_argument('--budgets',nargs='+',type=int,default=[16,32,64])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--teacher-modes',nargs='+',default=['network'])
    p.add_argument('--covered-value-source',type=Path)
    p.add_argument('--worlds',type=int,default=8)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if (not 1<=a.roots_per_key<=4 or not 2<=a.repeats<=8 or not 4<=a.worlds<=64
            or not 60<=a.seconds<=1800 or len(a.budgets)<2 or len(set(a.budgets))!=len(a.budgets)
            or any(b not in (8,16,32,64,128) for b in a.budgets) or len(set(a.keys))!=len(a.keys)
            or len(set(a.teacher_modes))!=len(a.teacher_modes)
            or any(m not in ('network','covered_terminal','covered_expert','covered_value') for m in a.teacher_modes)):
        p.error('Bounded distinct budgets, keys and fixed repeat/world endpoints required')
    if not a.run:print(json.dumps(dict(dry_run=True,seconds=a.seconds)));return
    if a.output.exists():raise ValueError('New output required; consumed diagnostic seeds cannot be reused')
    from app.modules.card_game.rl import recovery_runtime as rt
    from app.modules.card_game.rl.recovery_search import search
    from app.modules.card_game.rl.search_stability import jensen_shannon
    from app.modules.card_game.rl.teacher_validation import counterfactuals,paired_summary,value_diagnostics,verify_counterfactuals
    from app.modules.card_game.rl.episode_replay import load_episode
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    from app.modules.card_game.rl.offline_sources import snapshot
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
    from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
    values=None
    if 'covered_value' in a.teacher_modes:
        from app.modules.card_game.rl.covered_value import load_scope
        if not a.covered_value_source:raise ValueError('Qualified covered-value-source required')
        values,_=load_scope(a.covered_value_source)
    source=a.source_run.resolve();by_key=[]
    for key in a.keys:
        found=[]
        for path in sorted((source/'cycles/1/sampling/episodes').glob('*.json.gz')):
            data=load_episode(path);job=data['job']
            if job.get('training') is not True or len(data['actions'])==0:
                raise ValueError('Only completed declared TRAINING episodes allowed')
            rows=[r for r in data['rows'] if r['key']==key and 'pi' in r
                  and r['phase']=='playing' and len(r['pi'])>1
                  and np.sort(r['pi'])[-1]-np.sort(r['pi'])[-2]>=.05]
            if rows:found.append((path,data,rows[0]))
            if len(found)==a.roots_per_key:break
        if len(found)!=a.roots_per_key:raise ValueError('Insufficient declared confident roots for '+key)
        by_key.append(found)
    # Interleave teams so an absolute deadline cannot silently favor the first team.
    selected=[group[index] for index in range(a.roots_per_key) for group in by_key]
    start=time.time();end=diagnostic_deadline(start,a.seconds)
    if end-start<60:raise ValueError('Insufficient time before dated 09:00 hard stop')
    seeds=claim_seeds(ROOT/'artifacts/rl-seed-ledger.json',str(a.output.resolve()))
    a.output.mkdir(parents=True)
    def write(name,data):
        (a.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    write('config.json',dict(source_run=str(source),deadline=end,seeds=seeds,
          keys=a.keys,roots_per_key=a.roots_per_key,budgets=a.budgets,repeats=a.repeats,
          worlds=a.worlds,continuation='frozen_visible_policy_softmax',max_actions=512,
          teacher_modes=a.teacher_modes,
          covered_value_source=str(a.covered_value_source) if a.covered_value_source else None,
          value_probe='viewer_one_ply_network_vs_conditional_terminal',
          real_policy_updates=0,training_roots_only=True,quality_approved=False))
    snapshot(ROOT,a.output/'source',('scripts/check_duel_v2_search_teacher.py',
        'app/modules/card_game/rl/teacher_validation.py','app/modules/card_game/rl/covered_rollout.py'))
    reports=[]
    for ordinal,(path,data,row) in enumerate(selected):
        if time.time()>=end:break
        job=data['job'];policies={s:rt.model(*spec) for s,spec in job['policies'].items()}
        versions={s:v.version for s,v in policies.items()}
        if versions!=data['model_versions']:raise ValueError('Frozen source versions changed')
        state=new_game(seed=job['seed'],first_side=job['first'],decks=job['decks'])
        from copy import deepcopy
        root_state=None
        for step,entry in enumerate(data['actions']):
            if step==row['step']:root_state=deepcopy(state)
            state=clean(apply_action(state,entry['side'],entry['action']))
        if state['phase']!='finished' or state.get('winner')!=data['winner'] or root_state is None:
            raise ValueError('Source episode is not a verified complete terminal trajectory')
        state=root_state
        actor=acting_side(state);legal,(x,c)=rt.decision(state,actor)
        if actor!=row['side'] or not np.array_equal(x,row['x']) or not np.array_equal(c,row['c']):
            raise ValueError('Training root reconstruction mismatch')
        scores=rt.action_scores(policies[actor],x,c);prior=int(np.argmax(scores))
        record=dict(key=row['key'],game=job['id'],step=row['step'],models=versions,
            first=job['first'],left=job['left'],right=job['right'],viewer=actor,
            episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prior=prior,searches=[])
        branches={prior};full=True
        for mode,noise,budget in [(m,n,b) for m in a.teacher_modes for n in (False,True) for b in sorted(a.budgets)]:
            runs=[]
            for repeat in range(a.repeats):
                result=search(state,actor,policies,runtime=rt,simulations=budget,noise=noise,
                    seed=(1<<256)+seeds['search']+ordinal*10000+repeat,deadline=end,teacher_mode=mode,
                    value_policies={s:values[spec[1]] for s,spec in job['policies'].items()} if values else None)
                if not result['complete']:full=False;break
                runs.append(result)
            if not full:break
            mean=np.mean([r['pi'] for r in runs],axis=0);top=int(mean.argmax())
            visits=np.sum([r['visits'] for r in runs],axis=0)
            total=np.sum([r['mean_values']*r['visits'] for r in runs],axis=0)
            branches.add(top)
            branches.update(int(r['search_choice']) for r in runs if not noise)
            record['searches'].append(dict(noise=noise,budget=budget,teacher_mode=mode,
                target_js=jensen_shannon([r['pi'] for r in runs]),mean_target=mean.tolist(),
                mean_top=top,top=[int(r['pi'].argmax()) for r in runs],
                root_value=float(runs[0]['value']),visits=visits.tolist(),
                visited_mean_q=[float(total[i]/visits[i]) if visits[i] else None for i in range(len(visits))],
                chosen=[int(r['search_choice']) for r in runs]))
        record['search_complete']=full;record['root_actions']=legal
        if full:
            records=counterfactuals(state,actor,sorted(branches),policies,rt,
                world_seed=seeds['audit']+ordinal*10000,continuation_seed=seeds['confirmation']+ordinal*10000,
                worlds=a.worlds,deadline=end,capture_value=True)
            with gzip.open(a.output/f'root-{ordinal}-private-continuations.json.gz','wt') as f:json.dump(records,f)
            record['counterfactual']=paired_summary(records,sorted(branches),prior,expected_worlds=a.worlds)
            record['value_diagnostics']=value_diagnostics(records,sorted(branches))
            record['replay_verification']=verify_counterfactuals(state,actor,records,rt,deadline=end)
        reports.append(record)
        write('result.json',dict(complete=len(reports)==len(selected) and all(
            r['search_complete'] and r.get('counterfactual',{}).get('complete')
            and r.get('replay_verification',{}).get('complete') for r in reports),
            roots=reports,quality_approved=False,independent_strength_confirmation=False))
        print(json.dumps(dict(key=row['key'],game=job['id'],step=row['step'],
            search_complete=full,counterfactual=record.get('counterfactual')),ensure_ascii=False),flush=True)
        if not full:break
    if not reports:write('result.json',dict(complete=False,roots=[],reason='deadline',quality_approved=False))


if __name__=='__main__':main()
