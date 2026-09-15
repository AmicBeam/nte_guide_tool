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
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=1800)
    args=parser.parse_args(argv)
    if not 60<=args.seconds<=3600:parser.error('seconds must be 60..3600')
    out=args.output.resolve()
    if out.exists():parser.error('Choose a new output directory')
    out.mkdir(parents=True)
    started=time.monotonic();deadline=started+args.seconds
    from app.modules.card_game.rl.public_schema import rule_identity
    from app.modules.card_game.rl.capability import preset_opponent_deck
    from app.modules.card_game.rl.public_export import export_checkpoint
    from app.modules.card_game.rl.offline_analysis import run_offline_analysis,frozen_model_policy,export_game_replay
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
    stages=[];reports=[]
    def write(name,data):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('Overall smoke budget exhausted')
        return value
    def run(label,script,options):
        print(label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/script),*map(str,options)],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining(),180))
        if result.returncode:raise RuntimeError(f'{label} exited {result.returncode}; see {label}.log')
        stage_out=Path(options[options.index('--output')+1])
        validate_training_output(stage_out)
        stages.append(label);write('progress.json',{'stages':stages,'evaluations':len(reports),'seconds':time.monotonic()-started})
    write('config.json',{'kind':'full_chain_smoke','presets':PRESETS,'rule_hash':rule_identity(),
        'budget_seconds':args.seconds,'training_updates_base':4,'training_updates_adapt':2,'card_updates':1,
        'fixed_pairs':1,'varied_pairs':15,'screen_pairs':2,'max_actions':400,
        'seed_base':202609160,'selection':'screen candidates with fixed base policy; do not select using final test set',
        'prediction_probability':'unavailable; no calibrated probability model',
        'strength_claim':False})
    policies={};builds={};choices={}
    def evaluate(label,key,variant,opponent=None,opponent_deck=None,pairs=1,seed=202609160):
        print('evaluate '+label,flush=True)
        report=run_offline_analysis(out/'evaluations'/label,policies[key,variant],deck=key,
            learner_deck=builds[key,variant],opponent_policy=opponent or VisibleEngineRulePolicy(),
            opponent_deck=opponent_deck,pairs=pairs,max_actions=400,max_seconds=min(300,remaining()),seed_base=seed)
        reports.append({'label':label,'preset':key,'variant':variant,'report':f'evaluations/{label}/report.json',
                        'complete':report['complete']})
        write('evaluation-index.json',reports)
        if not report['complete']:raise RuntimeError(f'Incomplete evaluation: {label}')
        return report
    try:
        for key in PRESETS:
            builds[key,'original']=preset_opponent_deck(key)
            write(f'{key}/original-deck.json',builds[key,'original'])
            base=out/key/'base'
            run(key+'-base','train_duel_v2_public.py',['--train','--deck',key,'--hidden',32,'--n',128,'--horizon',16,
                 '--seconds',90,'--max-updates',4,'--snapshot-updates',1,'--preflight-cases',4,'--output',base])
            exported=export_checkpoint(base/'latest.pt',out/key/'base-export')
            policies[key,'original']=frozen_model_policy(FrozenModel(out/key/'base-export',key))
            cards=out/key/'cards'
            run(key+'-cards','train_duel_v2_cards.py',['--train','--team',key,'--policy-checkpoint',base/'latest.pt',
                 '--hidden',16,'--n',4,'--pairs',1,'--max-actions',400,'--seconds',90,'--max-updates',1,'--output',cards])
            builds[key,'candidate']=json.loads((cards/'candidate.json').read_text(encoding='utf-8'))
            policies[key,'candidate']=policies[key,'original']
            # Paired construction comparison keeps the exact battle weights fixed.
            before=evaluate(key+'-screen-original',key,'original',pairs=2,seed=202609000)
            after=evaluate(key+'-screen-candidate',key,'candidate',pairs=2,seed=202609000)
            score=lambda r:sum(v['win'] for v in r['by_position'].values())/r['games']
            choices[key]={'selection':'candidate' if score(after)>score(before) else 'original',
                'reason':'Higher screening win rate' if score(after)>score(before) else 'Candidate did not beat baseline; retain baseline',
                'original_win_rate':score(before),'candidate_win_rate':score(after),
                'sample_games_each':before['games'],'evidence':'small paired screening sample, not proof of optimality'}
            write('construction-decisions.json',choices)
            adapted=out/key/'adapted'
            run(key+'-adapt','train_duel_v2_public.py',['--train','--deck',key,'--resume',base/'latest.pt',
                '--learner-deck',cards/'candidate.json','--hidden',32,'--n',128,'--horizon',16,
                '--seconds',90,'--max-updates',2,'--snapshot-updates',1,'--preflight-cases',4,'--output',adapted])
            export_checkpoint(adapted/'latest.pt',out/key/'adapted-export')
            policies[key,'candidate']=frozen_model_policy(FrozenModel(out/key/'adapted-export',key))
        for variant in ('original','candidate'):
            for key in PRESETS:
                for opponent_key in PRESETS:
                    evaluate(f'{key}-{variant}-rule-{opponent_key}',key,variant,
                             opponent_deck=builds[opponent_key,'original'])
                evaluate(f'{key}-{variant}-rule-varied',key,variant,pairs=15)
                # Two distinct frozen opponents, with all 15 teams covered in each run.
                for opponent_key in PRESETS:
                    evaluate(f'{key}-{variant}-model-varied-{opponent_key}',key,variant,
                             opponent=policies[opponent_key,'original'],pairs=15)
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
        sample=reports[0];r=json.loads((out/sample['report']).read_text(encoding='utf-8'))
        export_game_replay(out/'evaluations'/sample['label'],r['games_index'][0]['game_id'],out/'sample-replay.json')
        from app.modules.card_game.rl.full_report import write_full_report
        write_full_report(out)
        write('status.json',{'phase':'complete','seconds':time.monotonic()-started,'stages':stages,
                            'evaluations':len(reports),'prediction_probability':'unavailable'})
    except Exception as exc:
        write('status.json',{'phase':'failed','error':repr(exc),'seconds':time.monotonic()-started,
                            'stages':stages,'evaluations':len(reports)})
        raise
    print(str(out/'report.md'),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
