#!/usr/bin/env python3
"""Bounded learned-mulligan vs keep-all control against actual fixed preset builds."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--starter-model',type=Path)
    parser.add_argument('--weave-model',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pairs',type=int,default=4)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--seed',type=int,default=202609300)
    args=parser.parse_args(argv)
    if not 1<=args.pairs<=32 or not 1<=args.seconds<=7200:parser.error('pairs=1..32; seconds=1..7200')
    if args.check:
        print(json.dumps({'training_started':False,'cells':8,'games':16*args.pairs,
              'matchups':'each actual preset vs both actual presets','modes':['learned','keep_all'],
              'models_required':'both resident_public_v3 exports','budget_seconds':args.seconds}));return 0
    if not args.output or not args.starter_model or not args.weave_model:parser.error('Both numeric models and new output directory required')
    if args.output.exists():parser.error('Choose a new output directory')
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    from app.modules.card_game.rl.offline_analysis import resolve_numeric_model_dir,frozen_model_policy,run_offline_analysis
    from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA
    from app.modules.card_game.rl.opening_report import summarize_opening,keep_all_policy,write_opening_report
    from app.modules.card_game.rl.capability import fixed_team_serving_deck
    models={};builds={};policies={}
    for key,path in [('starter',args.starter_model),('weave-rush',args.weave_model)]:
        directory,resolved=resolve_numeric_model_dir(path,deck_id=key)
        model=FrozenModel(directory,resolved)
        if model.schema!=OPENING_SCHEMA:parser.error(f'{key} has no learned mulligan policy')
        models[key]=model;policies[key]=frozen_model_policy(model)
        builds[key]=fixed_team_serving_deck(model.manifest.get('serving_build') or model.manifest.get('learner_deck') or model.serving_deck,key)
    args.output.mkdir(parents=True)
    deadline=time.monotonic()+args.seconds;rows=[]
    try:
        for key in models:
            for foe in models:
                for mode in ('learned','keep_all'):
                    left=deadline-time.monotonic()
                    if left<=0:raise TimeoutError('Opening analysis budget exhausted')
                    out=args.output/f'{key}-vs-{foe}-{mode}'
                    learner=policies[key] if mode=='learned' else keep_all_policy(policies[key])
                    report=run_offline_analysis(out,learner,deck=key,learner_deck=builds[key],
                        opponent_policy=policies[foe],opponent_deck=builds[foe],pairs=args.pairs,
                        max_actions=400,max_seconds=min(600,left),seed_base=args.seed)
                    rows.append({'learner':key,'opponent':foe,'mode':mode,'seed_base':args.seed,'summary':summarize_opening(out)})
                    if not report['complete']:raise RuntimeError('Incomplete opening evaluation')
    finally:
        complete=write_opening_report(args.output,rows)
    print(json.dumps({'complete':complete,'report':str(args.output/'opening-report.md')}))
    return 0 if complete else 2


if __name__=='__main__':raise SystemExit(main())
