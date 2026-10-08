#!/usr/bin/env python3
"""Enumerate N06 replacements and adapt each policy against one frozen weave build."""
import argparse
from collections import Counter
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def enumerate_builds(baseline):
    from app.modules.card_game.rl.capability import fixed_team_serving_deck
    baseline=fixed_team_serving_deck(baseline,'starter')
    counts=Counter(baseline['card_ids'])
    if counts['N06']:raise ValueError('Enumeration baseline must carry zero N06')
    removable=sorted(c for c in counts if c.startswith('N') and c!='NF01')
    result=[]
    for copies in (0,1,2):
        for removed in combinations_with_replacement(removable,copies):
            if any(n>counts[c] for c,n in Counter(removed).items()):continue
            cards=list(baseline['card_ids'])
            for c in removed:cards.remove(c)
            cards+=['N06']*copies
            label=f'n06-{copies}-'+('-'.join(removed) or 'baseline')
            build=fixed_team_serving_deck({**baseline,'id':label,'name':label,'card_ids':cards},'starter')
            result.append({'id':label,'copies':copies,'removed':list(removed),'build':build})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learner',type=Path,required=True);p.add_argument('--opponent',type=Path,required=True)
    p.add_argument('--learner-deck',type=Path,required=True);p.add_argument('--opponent-deck',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=7080)
    p.add_argument('--smoke-device',choices=('cpu','cuda'),default='cuda',help='CPU is allowed only for explicit full-chain smoke')
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--candidate-updates',type=int,default=24)
    p.add_argument('--foundation-updates',type=int,default=96)
    p.add_argument('--refine-updates',type=int,default=96)
    p.add_argument('--consolidate-updates',type=int,default=48)
    p.add_argument('--validation-every',type=int,default=24)
    p.add_argument('--curriculum-fraction',type=float,default=.2)
    p.add_argument('--check',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if (args.smoke_device=='cpu' and not args.smoke) or not 0<=args.entropy_coef<=1:
        p.error('CPU requires --smoke; entropy coefficient must be in [0,1]')
    if min(args.candidate_updates,args.foundation_updates,args.refine_updates,args.consolidate_updates,args.validation_every)<1 or not 0<=args.curriculum_fraction<=.5:
        p.error('Positive update budgets and curriculum fraction in [0,.5] required')
    baseline=json.loads(args.learner_deck.read_text(encoding='utf-8'));foe=json.loads(args.opponent_deck.read_text(encoding='utf-8'))
    from app.modules.card_game.rl.capability import fixed_team_serving_deck
    foe=fixed_team_serving_deck(foe,'weave-rush');candidates=enumerate_builds(baseline)
    if args.check:
        print(json.dumps({'candidates':candidates,'fixed_opponent':foe,'training':False,'stages':['diagnostics','foundation','equal_update_candidates','refine','mixed_consolidation','official_selection','independent_holdout'],'candidate_updates':args.candidate_updates,'reward_mode':'outcome_only_v1'},ensure_ascii=False));return
    if args.seconds<120:raise ValueError('Budget too short')
    if args.output.exists():raise ValueError('Output must be new')
    import torch
    from app.modules.card_game.rl.public_training import PublicTrainingEnv,FrozenLeague,warm_start_weights
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout,rollout_to_batch,ppo_update
    from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM,opening_observe
    from app.modules.card_game.rl.public_schema import state_dim,rule_identity
    from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,encode_public_deck_row
    from app.modules.card_game.rl.card_tuning import load_battle_policy,CardBattleEvaluator
    from app.modules.card_game.rl.public_export import export_checkpoint
    from scripts.verify_duel_v2_public import main as verify
    device=args.smoke_device if args.smoke else 'cuda'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if device=='cpu':torch.set_num_threads(1)
    out=args.output;out.mkdir(parents=True);start=time.monotonic();deadline=start+args.seconds
    def write(name,data):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(target)
    def status(stage,**extra):write('status.json',{'phase':stage,'elapsed':time.monotonic()-start,**extra})
    def remaining():return deadline-time.monotonic()
    verify(['--backend','cuda' if device=='cuda' else 'native','--cases','16'])
    from app.modules.card_game.rl.opening_preflight import verify_opening
    write('opening-preflight.json',verify_opening(device=device))
    peer,peer_meta=load_battle_policy(args.opponent,device);peer._trained_deck='weave-rush';peer._learn_mulligan=True
    league=FrozenLeague(learn_mulligan=True,greedy=True);league.add_pinned(peer)
    original=torch.load(args.learner,map_location='cpu',weights_only=True)
    from app.modules.card_game.rl.strategy_quality import diagnose_network,ValidationTracker,reachable_curriculum
    warm_seconds=60 if args.smoke else args.seconds*.26/len(candidates)
    refine_seconds=60 if args.smoke else args.seconds*.24/3
    write('config.json',{'candidates':candidates,'opponent':peer_meta,'opponent_deck':foe,
        'rule_hash':rule_identity(OPENING_SCHEMA),'smoke_only':args.smoke,'device':device,'seconds':args.seconds,'warm_seconds_each':warm_seconds,
        'refine_seconds_each':refine_seconds,'opponent_action':'argmax','random_opponents':False,
        'reward_mode':'outcome_only_v1','detailed_training_logs':False,'candidate_updates':args.candidate_updates,'validation_every':args.validation_every,'curriculum_fraction':args.curriculum_fraction,'holdout_used_for_training':False})
    def train(candidate,source,seconds,stage,update_limit,*,mixed=False):
        label=candidate['id'];status(stage,candidate=label)
        torch.manual_seed(202609179);torch.cuda.manual_seed_all(202609179)
        net=CompactScorer(state_dim(),OPENING_CAND_DIM,source['hidden']).to(device)
        warm_start_weights(source,net,'starter');opt=torch.optim.Adam(net.parameters(),lr=3e-4)
        net._trained_deck='starter';net._learn_mulligan=True;net._public_schema=OPENING_SCHEMA
        active_league=league
        if mixed:
            active_league=FrozenLeague(learn_mulligan=True,greedy=True);active_league.add_pinned(peer);active_league.add(net)
        curriculum=reachable_curriculum(candidate['build'],foe,count=4 if args.smoke else 12) if args.curriculum_fraction and not mixed else []
        env=PublicTrainingEnv(8 if device=='cpu' else (128 if args.smoke else 256),deck_id='starter',league=active_league,device=device,compiled_dir=out/'native-env' if device=='cpu' else None,
            learner_deck=candidate['build'],learn_mulligan=True,random_team_fraction=0,rule_fraction=0,fixed_opponent_deck=None if mixed else foe,
            curriculum_rows=curriculum,curriculum_fraction=args.curriculum_fraction if curriculum else 0)
        end=min(deadline-60,time.monotonic()+seconds);updates=0;folder=out/stage/label;folder.mkdir(parents=True)
        tracker=ValidationTracker();best_weights=None
        def validate():
            nonlocal best_weights
            result=evaluate(net,candidate['build'],2 if args.smoke else 16,660000000,mixed=mixed)
            if tracker.observe(updates,result['score']):best_weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            write(f'{stage}/{label}/validation.json',{'history':tracker.rows,'best':tracker.best,'plateau':tracker.plateau,'fixed_result':result,'holdout':False})
            write(f'{stage}/{label}/diagnostics-{updates}.json',diagnose_network(net))
        if stage=='foundation':
            from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
            native=CompiledStarterBackend(out/'diagnostic-native',backend='native')
            try:write('tactical-preflight.json',diagnose_network(net,backend=native))
            finally:native.close()
        validate()
        class Stop(Exception):pass
        def stop():
            if time.monotonic()>=end:raise Stop()
        try:
            while updates<(1 if args.smoke else update_limit):
                stop();roll=collect_rollout(env,net,16,observe_fn=opening_observe,stop_check=stop)
                stats=ppo_update(net,opt,rollout_to_batch(roll),stop_check=stop,ent_coef=args.entropy_coef);updates+=1
                with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps({'update':updates,'games':int(roll['done'].sum()),'low_entropy_warning':stats.get('entropy',1)<.01,**stats})+'\n')
                status(stage,candidate=label,update=updates)
                if updates%args.validation_every==0 and not args.smoke:validate()
                if mixed and updates%20==0:active_league.add(net)
        except Stop:pass
        finally:env.close()
        if updates<(1 if args.smoke else update_limit):
            write(f'{stage}/{label}/incomplete.json',{'updates':updates,'required':update_limit,'reason':'insufficient_budget'})
            raise RuntimeError('Insufficient equal-update adaptation budget; do not rank partially trained candidates')
        validate();net.load_state_dict(best_weights)
        write(f'{stage}/{label}/training-quality.json',{'updates':updates,'best_update':tracker.best['update'],'curriculum_resets':env.curriculum_resets,'optimizer_updates_equalized':True})
        saved={'schema':OPENING_SCHEMA,'reward_mode':'outcome_only_v1','deck':'starter','hidden':source['hidden'],
            'state_dim':state_dim(),'cand_dim':OPENING_CAND_DIM,'gpu_lock':GPU_LOCK,'rule_identity':{'rule_hash':rule_identity(OPENING_SCHEMA)},
            'model':net.state_dict(),'update':int(source.get('update',0))+tracker.best['update'],'updates_this_stage':updates,'selected_update':tracker.best['update'],'learner_deck':candidate['build'],'learn_mulligan':True,'opening_policy':'learned_joint_v1',
            'source_rule_identity':source.get('rule_identity'),'source_update':source.get('update',0),'smoke_only':args.smoke,
            'pinned_opponents':[peer_meta],'validated':False,'training_distribution':'fixed_final_weave_v1'}
        torch.save(saved,folder/'latest.pt');net.eval();net._public_schema=OPENING_SCHEMA
        return net,folder/'latest.pt'
    def evaluate(net,build,pairs,seed,*,mixed=False):
        evaluator=CardBattleEvaluator(net,opponents=[peer],device=device,compiled_dir=out/'native-eval' if device=='cpu' else None)
        plan=[{'lineup':'weave-rush','opponent_model':'peer-0','seed':seed+2*i,'deck':foe} for i in range(pairs)]
        if mixed:
            from app.modules.card_game.rl.candidate_opponents import build_opponent_schedule
            plan=build_opponent_schedule(pairs,seed=seed+1000000,random_fraction=0,rule_fraction=0,model_ids=('learner','peer-0'))
            for row in plan:row['opponent_model']='learner' if row['lineup']=='starter' else 'peer-0'

        def stop():
            if remaining()<45:raise TimeoutError('Evaluation budget exhausted')
        try:
            result=evaluator.evaluate(torch.tensor([encode_public_deck_row(build)],device=device,dtype=torch.int32),
                schedule=plan,stop_check=stop,max_actions=400,required_lineups=('starter','weave-rush') if mixed else ('weave-rush',))
            groups={pos:{k:int(v[0]) for k,v in fields.items()} for pos,fields in result['by_lineup']['weave-rush'].items()}
            if not bool(result['valid'][0]):raise RuntimeError('Incomplete screening games')
            report={'groups':groups,'score':float((result['rewards'][0]+1)/2),'mixed':mixed,'by_lineup':{k:{pos:{f:int(v[0]) for f,v in fields.items()} for pos,fields in group.items()} for k,group in result['by_lineup'].items()}}
            if mixed:report['fixed_weave']=evaluate(net,build,pairs,seed+2000000)
            return report
        finally:evaluator.close()
    foundation={'id':'shared-foundation','copies':0,'build':baseline}
    common,path=train(foundation,original,60 if args.smoke else args.seconds*.18,'foundation',args.foundation_updates)
    original=torch.load(path,map_location='cpu',weights_only=True);del common;torch.cuda.empty_cache()
    baseline_export=out/'foundation-export';export_checkpoint(path,baseline_export);write('foundation-export/deck.json',baseline)
    foundation_eval={**foundation,'export':str(baseline_export),'checkpoint':str(path)}
    rows=[]
    for c in candidates:
        net,path=train(c,original,warm_seconds,'warm',args.candidate_updates)
        result=evaluate(net,c['build'],2 if args.smoke else 64,620000000)
        rows.append({**c,**result,'checkpoint':str(path)});write('screening.json',rows)
        del net;torch.cuda.empty_cache()
    finalists=[max((r for r in rows if r['copies']==n),key=lambda r:(r['score'],r['id'])) for n in (0,1,2)]
    refined=[]
    for c in finalists:
        saved=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)
        net,path=train(c,saved,refine_seconds,'refine',args.refine_updates)
        result=evaluate(net,c['build'],2 if args.smoke else 256,630000000)
        write(f'refine/{c["id"]}/fixed-validation.json',result)
        saved=torch.load(path,map_location='cpu',weights_only=True);del net;torch.cuda.empty_cache()
        net,path=train(c,saved,60 if args.smoke else args.seconds*.12/3,'consolidate',args.consolidate_updates,mixed=True)
        result=evaluate(net,c['build'],2 if args.smoke else 256,635000000)
        export=out/'exports'/c['id'];export_checkpoint(path,export)
        write(f'exports/{c["id"]}/deck.json',c['build'])
        refined.append({**c,**result,'checkpoint':str(path),'export':str(export)})
        write('finalists.json',refined);del net;torch.cuda.empty_cache()
    # Official-engine selection and disjoint holdout: all three N06 counts remain visible.
    import subprocess,concurrent.futures,os
    write('opponent-deck.json',foe)
    foe_export=out/'opponent-export';export_checkpoint(args.opponent,foe_export)
    def official(c,part,seed,pairs):
        label=c['id'];dest=out/part/label
        cmd=[sys.executable,'-X','utf8','scripts/analyze_duel_v2_policy.py','--model',str(Path(c['export'])/'starter.npz'),
             '--opponent-model',str(foe_export/'weave-rush.npz'),'--learner-deck',str(Path(c['export'])/'deck.json'),
             '--opponent-deck',str(out/'opponent-deck.json'),'--pairs',str(pairs),'--seed',str(seed),'--max-actions','400',
             '--max-seconds',str(min(600,max(1,int(remaining()-45)))),'--output',str(dest)]
        dest.parent.mkdir(parents=True,exist_ok=True)
        with (dest.parent/(label+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(620,max(1,remaining())))
        report=json.loads((dest/'report.json').read_text(encoding='utf-8'))
        if not report['complete']:raise RuntimeError('Official evaluation incomplete: '+label)
        groups=report['by_position'];return {'id':label,'copies':c['copies'],'groups':groups,'score':sum(g['win']/g['games'] for g in groups.values())/2}
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    status('official-selection')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        formal=list(pool.map(lambda c:official(c,'official-selection',640000000,1 if args.smoke else 32),refined))
    write('official-selection.json',formal)
    winner=max(formal,key=lambda r:(r['score'],-r['copies']))
    chosen=next(c for c in refined if c['id']==winner['id']);write('selected.json',{'selection':winner,'candidate':chosen,'automatic_serving_replacement':False})
    status('holdout')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        holdout=list(pool.map(lambda c:official(c,'holdout',650000000,1 if args.smoke else 32),[foundation_eval,*refined]))
    write('holdout.json',holdout)
    generalization=[]
    for candidate in [foundation_eval,*refined]:
        net,_=load_battle_policy(Path(candidate['checkpoint']),device)
        generalization.append({'id':candidate['id'],**evaluate(net,candidate['build'],2 if args.smoke else 64,680000000,mixed=True)})
        del net;torch.cuda.empty_cache()
    write('generalization-holdout.json',generalization)
    selected_holdout=next(r for r in holdout if r['id']==winner['id']);baseline_holdout=holdout[0]
    write('acceptance.json',{'selected':winner['id'],'fixed_matchup_delta':selected_holdout['score']-baseline_holdout['score'],
        'generalization_delta':next(r['score'] for r in generalization if r['id']==winner['id'])-generalization[0]['score'],
        'baseline':'shared foundation before card search','requires_multi_seed_confirmation':True,'automatic_serving_approval':False,
        'diagnostics_required':True,'holdout_used_to_reselect':False,'note':'Completion is not evidence of convergence or statistical significance'})
    status('complete',selected=winner['id'])

if __name__=='__main__':main()
