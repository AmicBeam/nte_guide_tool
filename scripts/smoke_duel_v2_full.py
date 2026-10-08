#!/usr/bin/env python3
"""Bounded dual-preset CUDA pipeline smoke; results are feasibility evidence, not strength certification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PRESETS=('starter','weave-rush')


def validate_training_output(directory):
    directory=Path(directory)
    metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    if not any(row.get('updated',True) for row in metrics):
        raise RuntimeError('Training completed without an optimizer update')
    return len(metrics)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Print opponent/stage budgets without importing ML or training')
    parser.add_argument('--seconds',type=int,default=1800)
    parser.add_argument('--hidden',type=int,help='Policy hidden width; match warm-start checkpoints')
    parser.add_argument('--profile',choices=('smoke','comprehensive'),default='smoke')
    parser.add_argument('--phase',choices=('all','train','evaluate'),default='all')
    parser.add_argument('--candidate-seeds',type=Path,help='Preset-to-list mapping of prior legal candidate decks')
    parser.add_argument('--starter-checkpoint',type=Path)
    parser.add_argument('--weave-checkpoint',type=Path)
    args=parser.parse_args(argv)
    if args.hidden is not None and not 1<=args.hidden<=2048:parser.error('hidden must be 1..2048')
    if not 60<=args.seconds<=21600:parser.error('seconds must be 60..21600')
    from app.modules.card_game.rl.candidate_selection import PROFILES
    from app.modules.card_game.rl.episode_return import REWARD_MODE
    full=args.profile=='comprehensive'
    budget={stage:args.seconds*fraction for stage,fraction in
            (('base',.15),('cards',.15),('adapt',.075),('selection',.10))}
    if args.check:
        print(json.dumps({'training_started':False,'reward_mode':REWARD_MODE,'presets':PRESETS,'profile':args.profile,
              'training_order':'both bases, then each cards/selection/adaptation','learn_mulligan':True,'adaptation_pinned_peer':True,
              'seconds_total':args.seconds,'seconds_per_preset':budget,'reserve_seconds':args.seconds*.05,
              'candidate_plan':PROFILES[args.profile],'local_search_rounds':4 if full else 1,'local_two_card_samples':128 if full else 4,'random_team_fraction':.2,'rule_fraction':.2,
              'selection_opponents':'both frozen models with fixed preset roles and variable cards',
              'selection_seed_ranges':[[400000000,402999999],[410000000,412999999]],
              'holdout_seed_base':202609160},ensure_ascii=False));return 0
    if args.output is None:parser.error('--output is required unless --check')
    hidden=args.hidden or (64 if full else 32);parallel=1024 if full else 128
    fixed_pairs=16 if full else 1;varied_pairs=32 if full else 15;screen_pairs=16 if full else 2
    out=args.output.resolve()
    if out.exists() and args.phase!='evaluate':parser.error('Choose a new output directory')
    if args.phase=='evaluate' and json.loads((out/'status.json').read_text()).get('phase') not in ('training_complete','failed','complete'):parser.error('Training is not complete')
    out.mkdir(parents=True,exist_ok=args.phase=='evaluate')
    started=time.monotonic();deadline=started+args.seconds
    from app.modules.card_game.rl.public_schema import rule_identity as public_rule_identity
    from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA
    rule_identity=lambda:public_rule_identity(OPENING_SCHEMA)
    from app.modules.card_game.rl.capability import preset_opponent_deck
    from app.modules.card_game.rl.public_export import export_checkpoint
    from app.modules.card_game.rl.offline_analysis import run_offline_analysis,frozen_model_policy,export_game_replay
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
    stages=[];reports=[]
    if args.phase=='evaluate':
        stages=json.loads((out/'status.json').read_text()).get('stages',[])
        required={f'{key}-{stage}' for key in PRESETS for stage in ('base','cards','adapt')}
        if not required<=set(stages):parser.error('Training stages incomplete; cannot run final evaluation')
        reports=json.loads((out/'evaluation-index.json').read_text())
    def write(name,data):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('Overall smoke budget exhausted')
        return value
    def run(label,script,options):
        if full:
            options=list(options)
            wanted=float(options[options.index('--seconds')+1])
            options[options.index('--seconds')+1]=max(1,min(wanted,remaining()-60))
            options.append('--stop-if-stalled')
        print(label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/script),*map(str,options)],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining(),float(options[options.index('--seconds')+1])+120 if full else 180))
        if result.returncode:raise RuntimeError(f'{label} exited {result.returncode}; see {label}.log')
        stage_out=Path(options[options.index('--output')+1])
        validate_training_output(stage_out)
        stages.append(label);write('progress.json',{'stages':stages,'evaluations':len(reports),'seconds':time.monotonic()-started})
    if args.phase!='evaluate':write('config.json',{'kind':'comprehensive' if full else 'full_chain_smoke','presets':PRESETS,'rule_hash':rule_identity(),
        'reward_mode':REWARD_MODE,'budget_seconds':args.seconds,'training_updates_base':0 if full else 4,'training_updates_adapt':0 if full else 2,'card_updates':0 if full else 1,
        'phase':args.phase,'hidden':hidden,'parallel':parallel,'base_seconds':budget['base'],'card_seconds':budget['cards'],'adapt_seconds':budget['adapt'],
        'selection_seconds':budget['selection'],'candidate_plan':PROFILES[args.profile],
        'local_search':{'rounds':4 if full else 1,'two_limit':128 if full else 4,'screen_pairs':8 if full else 1,'final_pairs':128 if full else 2},
        'no_progress':{'window_seconds':300,'patience':6,'meaning':'six stable window means; not convergence'},'final_evaluation_separate':args.phase=='train',
        'fixed_pairs':fixed_pairs,'varied_pairs':varied_pairs,'screen_pairs':screen_pairs,'max_actions':400,
        'seed_base':202609160,'selection':'three-stage preset-lineup tournament; rule screen rows diagnostic only; holdout never selects',
        'prediction_probability':'unavailable; no calibrated probability model',
        'strength_claim':False})
    if args.phase=='evaluate':write('evaluation-config.json',{'fixed_pairs':fixed_pairs,'varied_pairs':varied_pairs,'budget_seconds':args.seconds,'phase':'frozen_evaluation'})
    seed_candidates=json.loads(args.candidate_seeds.read_text(encoding='utf-8')) if args.candidate_seeds else {}
    from app.modules.card_game.rl.capability import fixed_team_serving_deck
    if not isinstance(seed_candidates,dict) or set(seed_candidates)-set(PRESETS):raise ValueError('Invalid candidate seed presets')
    for key,items in seed_candidates.items():
        if not isinstance(items,list) or len(items)>2:raise ValueError('At most two seed builds per preset')
        seed_candidates[key]=[fixed_team_serving_deck(b,key) for b in items]
    policies={};builds={};choices={}
    def evaluate(label,key,variant,opponent=None,opponent_deck=None,pairs=1,seed=202609160):
        if pairs==1:pairs=fixed_pairs
        existing=out/'evaluations'/label/'report.json'
        if existing.is_file():
            cached=json.loads(existing.read_text(encoding='utf-8'))
            if cached['complete']:
                expected={'rules':rule_identity(),'seed':seed,'pairs':pairs,
                          'learner':policies[key,variant].model_metadata,'learner_build':builds[key,variant],
                          'opponent':getattr(opponent or VisibleEngineRulePolicy(),'model_metadata',{'type':'VisibleEngineRulePolicy'}),
                          'opponent_build':opponent_deck,'max_actions':400}
                if json.loads(existing.with_name('cache-identity.json').read_text(encoding='utf-8'))!=expected:
                    raise RuntimeError(f'Cached opponent/seed/rule identity changed: {label}')
                if cached['pairs']!=pairs or cached['model'].get('sha256')!=policies[key,variant].model_metadata.get('sha256') or cached['learner_build']!=builds[key,variant]:
                    raise RuntimeError(f'Cached evaluation identity changed: {label}')
                if not any(entry['label']==label for entry in reports):
                    reports.append({'label':label,'preset':key,'variant':variant,'report':f'evaluations/{label}/report.json','complete':True})
                    write('evaluation-index.json',reports)
                return cached
            raise RuntimeError(f'Incomplete prior evaluation: {label}')
        print('evaluate '+label,flush=True)
        report=run_offline_analysis(out/'evaluations'/label,policies[key,variant],deck=key,
            learner_deck=builds[key,variant],opponent_policy=opponent or VisibleEngineRulePolicy(),
            opponent_deck=opponent_deck,pairs=pairs,max_actions=400,max_seconds=min(600 if full else 300,remaining()),seed_base=seed)
        write(f'evaluations/{label}/cache-identity.json',{'rules':rule_identity(),'seed':seed,'pairs':pairs,
              'learner':policies[key,variant].model_metadata,'learner_build':builds[key,variant],
              'opponent':getattr(opponent or VisibleEngineRulePolicy(),'model_metadata',{'type':'VisibleEngineRulePolicy'}),
              'opponent_build':opponent_deck,'max_actions':400})
        reports.append({'label':label,'preset':key,'variant':variant,'report':f'evaluations/{label}/report.json',
                        'complete':report['complete']})
        write('evaluation-index.json',reports)
        if not report['complete']:raise RuntimeError(f'Incomplete evaluation: {label}')
        return report
    try:
        if args.phase!='evaluate':
            for key in PRESETS:
                builds[key,'original']=preset_opponent_deck(key)
                write(f'{key}/original-deck.json',builds[key,'original'])
                write(f'{key}/seed-candidates.json',seed_candidates.get(key,[]))
                base=out/key/'base'
                initial=args.starter_checkpoint if key=='starter' else args.weave_checkpoint
                resume_options=[]
                if initial:
                    # A new pipeline may change its pinned opponents; transfer weights only.
                    resume_options=['--warm-start',initial]
                peer_initial=args.weave_checkpoint if key=='starter' else args.starter_checkpoint
                if peer_initial:resume_options+=['--opponent-checkpoint',peer_initial]
                run(key+'-base','train_duel_v2_public.py',['--train','--deck',key,'--hidden',hidden,'--n',parallel,'--horizon',16,
                     '--seconds',budget['base'],'--max-updates',0 if full else 4,'--snapshot-updates',20 if full else 1,'--preflight-cases',16 if full else 4,'--output',base]+resume_options)
                exported=export_checkpoint(base/'latest.pt',out/key/'base-export')
                policies[key,'original']=frozen_model_policy(FrozenModel(out/key/'base-export',key))
        for key in PRESETS:
            if args.phase=='evaluate':
                builds[key,'original']=json.loads((out/key/'original-deck.json').read_text(encoding='utf-8'))
                selected=out/key/'selected-deck.json'
                builds[key,'candidate']=json.loads((selected if selected.exists() else out/key/'cards/candidate.json').read_text(encoding='utf-8'))
                policies[key,'original']=frozen_model_policy(FrozenModel(out/key/'base-export',key))
                policies[key,'candidate']=frozen_model_policy(FrozenModel(out/key/'adapted-export',key))
                choices=json.loads((out/'construction-decisions.json').read_text(encoding='utf-8'))
                continue
            base=out/key/'base'
            cards=out/key/'cards'
            run(key+'-cards','train_duel_v2_cards.py',['--train','--team',key,'--policy-checkpoint',base/'latest.pt',
                 '--opponent-checkpoint',out/('weave-rush' if key=='starter' else 'starter')/'base/latest.pt',
                 '--hidden',64 if full else 16,'--n',16 if full else 4,'--pairs',8 if full else 1,'--max-actions',400,'--seconds',budget['cards'],'--max-updates',0 if full else 1,'--output',cards])
            from app.modules.card_game.rl.card_tuning import load_battle_policy,CardBattleEvaluator
            from app.modules.card_game.rl.candidate_selection import candidate_pool,run_tournament,evaluate_tensor_candidates
            own,own_id=load_battle_policy(base/'latest.pt','cuda')
            peer_key='weave-rush' if key=='starter' else 'starter'
            peer,peer_id=load_battle_policy(out/peer_key/'base/latest.pt','cuda')
            selection_deadline=min(deadline,time.monotonic()+budget['selection'])
            def selection_stop():
                if time.monotonic()>=selection_deadline:raise TimeoutError('Candidate selection budget exhausted')
            evaluator=CardBattleEvaluator(own,opponents=[peer],device='cuda')
            try:
                selected=run_tournament(candidate_pool(out/key,key),
                    lambda decks,schedule:evaluate_tensor_candidates(evaluator,decks,schedule,stop_check=selection_stop),
                    profile=args.profile,seed_base=400000000+PRESETS.index(key)*10000000,stop_check=selection_stop)
                if selected['complete']:
                    from app.modules.card_game.rl.local_card_search import refine_locally
                    local=refine_locally(builds[key,'original'],
                        lambda decks,schedule:evaluate_tensor_candidates(evaluator,decks,schedule,stop_check=selection_stop),
                        initial=selected['build'],rounds=4 if full else 1,two_limit=128 if full else 4,
                        screen_pairs=8 if full else 1,final_pairs=128 if full else 2,
                        seed_base=420000000+PRESETS.index(key)*20000000,stop_check=selection_stop)
                    write(f'{key}/local-search.json',local)
                    local['stages']=selected['stages']+local['stages']
                    selected=local
            finally:evaluator.close()
            selected['model_identities']={'learner':own_id,'peer-0':peer_id}
            selected['rule_hash']=rule_identity()
            write(f'{key}/candidate-selection.json',selected)
            write(f'{key}/selected-deck.json',selected['build'])
            if not selected['complete']:raise RuntimeError('Candidate selection incomplete; baseline retained')
            builds[key,'candidate']=selected['build']
            policies[key,'candidate']=policies[key,'original']
            # Legacy screen rows are diagnostic only; tournament above owns selection.
            before=evaluate(key+'-screen-original',key,'original',pairs=screen_pairs,seed=202609000)
            after=evaluate(key+'-screen-candidate',key,'candidate',pairs=screen_pairs,seed=202609000)
            final=selected['stages'][-1]['results']
            original=next(r for r in final if r['id']=='original')
            winner=next(r for r in final if r['id']==selected['selection'])
            choices[key]={'selection':'candidate' if selected['selection']!='original' else 'original',
                'reason':selected['reason'],'original_win_rate':original['metrics']['macro'],
                'candidate_win_rate':winner['metrics']['macro'],
                'sample_games_each':sum(g['games'] for group in winner['groups'].values() for g in group.values()),
                'evidence':f'{key}/candidate-selection.json','candidate_id':selected['selection'],
                'selection_complete':True,'worst_lineup':winner['metrics']['worst_lineup'],
                'worst_seat':winner['metrics']['worst_seat']}
            write('construction-decisions.json',choices)
            adapted=out/key/'adapted'
            run(key+'-adapt','train_duel_v2_public.py',['--train','--deck',key,'--warm-start',base/'latest.pt',
                '--opponent-checkpoint',out/peer_key/'base/latest.pt',
                '--learner-deck',out/key/'selected-deck.json','--hidden',hidden,'--n',parallel,'--horizon',16,
                '--seconds',budget['adapt'],'--max-updates',0 if full else 2,'--snapshot-updates',20 if full else 1,'--preflight-cases',4,'--output',adapted])
            export_checkpoint(adapted/'latest.pt',out/key/'adapted-export')
            policies[key,'candidate']=frozen_model_policy(FrozenModel(out/key/'adapted-export',key))
        if args.phase=='train':
            write('status.json',{'phase':'training_complete','seconds':time.monotonic()-started,'stages':stages,'evaluations':len(reports),'final_evaluation_pending':True})
            return 0
        for variant in ('original','candidate'):
            for key in PRESETS:
                for opponent_key in PRESETS:
                    evaluate(f'{key}-{variant}-rule-{opponent_key}',key,variant,
                             opponent_deck=builds[opponent_key,'original'])
                evaluate(f'{key}-{variant}-rule-varied',key,variant,pairs=varied_pairs)
                # Two distinct frozen opponents, with all 15 teams covered in each run.
                for opponent_key in PRESETS:
                    evaluate(f'{key}-{variant}-model-varied-{opponent_key}',key,variant,
                             opponent=policies[opponent_key,'original'],pairs=varied_pairs)
                evaluate(f'{key}-{variant}-mirror',key,variant,opponent=policies[key,variant],
                         opponent_deck=builds[key,variant])
            evaluate(f'cross-{variant}','starter',variant,opponent=policies['weave-rush',variant],
                     opponent_deck=builds['weave-rush',variant])
        # Additional paired new/old model checks under the exact same candidate build.
        for key in PRESETS:
            current=policies[key,'candidate']
            policies[key,'candidate']=policies[key,'original']
            evaluate(key+'-candidate-old-policy',key,'candidate',pairs=1,seed=202609260)
            policies[key,'candidate']=current
            evaluate(key+'-candidate-new-policy',key,'candidate',pairs=1,seed=202609260)
        if len(reports)!=34:raise RuntimeError('Incomplete full evaluation matrix')
        if all(policies[key,'candidate'].model_metadata.get('schema')==OPENING_SCHEMA for key in PRESETS):
            from scripts.analyze_duel_v2_opening import main as opening_main
            opening_pairs=16 if full else 1
            opening_models={key:out/key/('adapted-export' if choices[key]['selection']=='candidate' else 'base-export')/f'{key}.npz' for key in PRESETS}
            opening_identity={'rules':rule_identity(),'pairs':opening_pairs,'seed':202609300,'max_actions':400,
                'models':{key:policies[key,choices[key]['selection']].model_metadata for key in PRESETS},
                'builds':{key:builds[key,choices[key]['selection']] for key in PRESETS}}
            opening_root=out/'opening-analysis'
            if opening_root.exists():
                if (json.loads((opening_root/'cache-identity.json').read_text())!=opening_identity or
                    not json.loads((opening_root/'opening-summary.json').read_text())['complete']):
                    raise RuntimeError('Opening evaluation cache incomplete or identity changed')
            else:
                if opening_main(['--starter-model',str(opening_models['starter']),'--weave-model',str(opening_models['weave-rush']),
                    '--output',str(opening_root),'--pairs',str(opening_pairs),'--seconds',str(max(1,min(7200,int(remaining())))),
                    '--seed','202609300']):raise RuntimeError('Opening evaluation incomplete')
                write('opening-analysis/cache-identity.json',opening_identity)
        sample=reports[0];r=json.loads((out/sample['report']).read_text(encoding='utf-8'))
        export_game_replay(out/'evaluations'/sample['label'],r['games_index'][0]['game_id'],out/'sample-replay.json')
        from app.modules.card_game.rl.full_report import write_full_report
        if not write_full_report(out)['report_complete']:raise RuntimeError('Final report incomplete')
        write('status.json',{'phase':'complete','seconds':time.monotonic()-started,'stages':stages,
                            'evaluations':len(reports),'prediction_probability':'unavailable'})
    except Exception as exc:
        write('status.json',{'phase':'failed','error':repr(exc),'seconds':time.monotonic()-started,
                            'stages':stages,'evaluations':len(reports)})
        raise
    print(str(out/'report.md'),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
