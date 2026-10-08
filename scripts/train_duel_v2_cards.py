#!/usr/bin/env python3
"""Explicit card-allocation learning for a fixed team. Never selects characters."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--train',action='store_true')
    modes.add_argument('--propose',action='store_true')
    p.add_argument('--team',choices=('starter','weave-rush'))
    p.add_argument('--team-json',type=Path,help='User-specified four character_ids; never optimized')
    p.add_argument('--policy-checkpoint',type=Path)
    p.add_argument('--opponent-checkpoint',type=Path,action='append',default=[])
    p.add_argument('--opponent-builds',type=Path,help='JSON mapping preset IDs to historical legal deck lists')
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--n',type=int,default=32)
    p.add_argument('--pairs',type=int,default=8,help='Pairs per public preset lineup')
    p.add_argument('--random-team-fraction',type=float,default=.2)
    p.add_argument('--rule-fraction',type=float,default=.2)
    p.add_argument('--max-actions',type=int,default=400)
    p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--stop-if-stalled',action='store_true')
    p.add_argument('--detach',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.team and args.team_json:p.error('Specify --team or --team-json, not both')
    if not 1<=args.n<=128 or not 1<=args.pairs<=16 or not 1<=args.max_actions<=2000:
        p.error('n=1..128, pairs=1..16, max-actions=1..2000')
    if (not _finite(args.seconds) or not 0<args.seconds<=7200 or args.max_updates<0
            or not _finite(args.learning_rate) or args.learning_rate<=0
            or not _finite(args.entropy_coef) or args.entropy_coef<0):
        p.error('Invalid training budget or optimizer options')
    if not 1<=args.hidden<=1024:
        p.error('hidden must be 1..1024')
    if len(args.opponent_checkpoint)>4:p.error('At most four frozen opponent checkpoints')
    from app.modules.card_game.rl.capability import preset_opponent_deck,PUBLIC_CHARACTERS
    team=preset_opponent_deck(args.team or 'starter')['character_ids']
    if args.team_json:
        try:
            value=json.loads(args.team_json.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            p.error(f'Cannot read team JSON: {exc}')
        team=value.get('character_ids') if isinstance(value,dict) else value
    if (not isinstance(team,list) or len(team)!=4 or not all(isinstance(c,str) for c in team)
            or len(set(team))!=4 or not set(team)<=set(PUBLIC_CHARACTERS)):
        p.error('Fixed team must contain four distinct public characters')
    if not (0<=args.random_team_fraction<1 and 0<=args.rule_fraction<1):p.error('Opponent fractions must be finite and in [0,1)')
    historical_builds=None
    if args.opponent_builds:
        try:historical_builds=json.loads(args.opponent_builds.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:p.error(f'Cannot read opponent builds: {exc}')
    distribution={'scoring':'equal_four_scenario_win_rate_v1','seed':args.seed,'historical_builds':historical_builds,'random_team_fraction':args.random_team_fraction,'rule_fraction':args.rule_fraction,
                  'pairs_per_preset':args.pairs,'presets':['starter','weave-rush'],
                  'frozen_model_ids':['learner',*[f'peer-{i}' for i in range(len(args.opponent_checkpoint))]],
                  'peer_available':bool(args.opponent_checkpoint)}
    if not args.propose:
        from app.modules.card_game.rl.candidate_opponents import build_opponent_schedule
        try:
            probe=build_opponent_schedule(3,seed=args.seed,random_fraction=args.random_team_fraction,
                  rule_fraction=args.rule_fraction,model_ids=distribution['frozen_model_ids'],historical_builds=historical_builds)
        except (ValueError,TypeError) as exc:p.error(str(exc))
        distribution['serving_build_sha256']={row['lineup']:row['build_sha256'] for row in probe if row['allocation']=='serving'}
        distribution['sampling_schema']='preset_lineups_v1'
    if args.check:
        print(json.dumps({'character_ids':team,'learns_characters':False,'cards_per_character':8,
                          'max_copies':2,'training_started':False,'opponent_distribution':distribution,
                          'peer_note':'Peer checkpoints provided' if args.opponent_checkpoint else 'Only frozen learner available; no peer model coverage'},ensure_ascii=False));return 0
    if not args.output:p.error('--output NEWDIR is required')
    if args.train and not args.policy_checkpoint:p.error('--policy-checkpoint is required for battle scoring')
    if args.propose and (not args.resume or args.detach):p.error('--propose requires --resume and cannot detach')
    if args.worker and (not args.train or args.detach):p.error('--worker is only for a detached training child')
    if args.output.is_symlink():p.error('Use a new output directory')
    if args.detach:
        if args.output.exists():p.error('Use a new output directory')
        args.output.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).resolve()),*[x for x in (sys.argv[1:] if argv is None else argv) if x!='--detach'],'--worker']
        kwargs={'cwd':ROOT,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':kwargs['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|getattr(subprocess,'CREATE_BREAKAWAY_FROM_JOB',0x01000000)
        else:kwargs['start_new_session']=True
        with (args.output/'console.log').open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,**kwargs)
        (args.output/'launch.json').write_text(json.dumps({'pid':process.pid,'command':command})+chr(10))
        print(json.dumps({'pid':process.pid,'status':'starting'}));return 0
    if args.output.exists():
        allowed_child = (args.worker and args.output.is_dir()
                         and {f.name for f in args.output.iterdir()} <= {'console.log','launch.json'})
        if not allowed_child:p.error('Use a new output directory')
    import torch
    from app.modules.card_game.rl.card_tuning import CardTuner,CardBattleEvaluator,load_battle_policy,update_tuner,SCHEMA,GPU_LOCK
    from app.modules.card_game.rl.public_schema import rule_identity
    identity=rule_identity()
    saved=None
    if args.resume:
        if not args.resume.is_file():
            p.error('Resume checkpoint not found')
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if (not isinstance(saved,dict) or saved.get('schema')!=SCHEMA
                or saved.get('gpu_lock')!=GPU_LOCK or saved.get('rule_hash')!=identity):
            p.error('Card-tuner checkpoint schema/rules mismatch')
        saved_team=list(saved.get('character_ids') or ())
        if not args.team and not args.team_json:
            team=saved_team
        if list(team)!=saved_team:
            p.error('Resume cannot change the fixed team')
        if args.train:
            expected=saved.get('policy') or {}
            if expected.get('sha256')!=_sha256(args.policy_checkpoint):
                p.error('Resume requires the same frozen battle policies')
            saved_opponents=saved.get('opponents') or []
            actual=[_sha256(path) for path in args.opponent_checkpoint]
            recorded=[item.get('sha256') if isinstance(item,dict) else None for item in saved_opponents]
            if actual!=recorded:
                p.error('Resume requires the same frozen battle policies')
            if saved.get('opponent_distribution')!=distribution:p.error('Resume requires the same opponent distribution')
    device='cuda' if args.train else 'cpu'
    if args.train and not torch.cuda.is_available():p.error('CUDA unavailable; no CPU training fallback')
    hidden=saved['hidden'] if saved else args.hidden
    torch.manual_seed(args.seed)
    tuner=CardTuner(team,hidden).to(device)
    if saved:tuner.load_state_dict(saved['model'])
    args.output.mkdir(parents=True,exist_ok=args.worker)
    def write(name,data):
        target=args.output/name;temp=target.with_suffix(target.suffix+'.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8');temp.replace(target)
    if args.propose:
        with torch.no_grad():sample=tuner.sample(args.n)
        for i,row in enumerate(sample['rows']):write(f'candidate-{i+1:03d}.json',tuner.deck(row))
        write('proposal.json',{**tuner.metadata(),'rule_hash':identity,'resume':str(args.resume),
                               'checkpoint_sha256':_sha256(args.resume),'seed':args.seed,'n':args.n,
                               'update':saved.get('update',0),'policy':saved.get('policy'),
                               'candidates':args.n,'independently_evaluated':False,
                               'learns_characters':False,'training_started':False})
        return 0
    from scripts.verify_duel_v2_public import main as verify
    verify(['--backend','cuda','--cases','16'])
    policy,policy_id=load_battle_policy(args.policy_checkpoint,device)
    opponents=[];opponent_ids=[]
    for path in args.opponent_checkpoint:
        net,meta=load_battle_policy(path,device);opponents.append(net);opponent_ids.append(meta)
    if saved and (saved.get('policy')!=policy_id or list(saved.get('opponents') or [])!=opponent_ids):
        p.error('Resume requires the same frozen battle policies')
    optimizer=torch.optim.Adam(tuner.parameters(),lr=args.learning_rate)
    update=int(saved.get('update',0)) if saved else 0
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:group['lr']=args.learning_rate
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    from app.modules.card_game.rl.candidate_opponents import build_opponent_schedule
    evaluator=CardBattleEvaluator(policy,opponents=opponents,device=device)
    write('config.json',{**tuner.metadata(),'rule_hash':identity,'policy':policy_id,'opponents':opponent_ids,
                         'resume':str(args.resume) if args.resume else None,
                         'checkpoint_sha256':_sha256(args.resume) if args.resume else None,
                         'pairs':args.pairs,'n':args.n,'max_actions':args.max_actions,'seconds':args.seconds,
                         'detailed_training_logs':False,'learns_characters':False,'opponent_distribution':distribution})
    from app.modules.card_game.rl.progress_stop import ProgressStop
    monitor=ProgressStop();stop_reason='budget_or_update_limit'
    started=time.monotonic();count=0;attempt=0;failed=False
    class Deadline(Exception):pass
    def stop():
        if time.monotonic()-started>=args.seconds:raise Deadline()
    def save():
        temp=args.output/'latest.tmp'
        torch.save({**tuner.metadata(),'rule_hash':identity,'model':tuner.state_dict(),
                    'optimizer':optimizer.state_dict(),'opponent_distribution':distribution,'policy':policy_id,'opponents':opponent_ids,'update':update,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
        temp.replace(args.output/'latest.pt')
    try:
        while not args.max_updates or count<args.max_updates:
            stop();attempt+=1
            sample=tuner.sample(args.n)
            schedule=build_opponent_schedule(args.pairs,seed=args.seed+update*1009+attempt,
                random_fraction=args.random_team_fraction,rule_fraction=args.rule_fraction,model_ids=tuple(evaluator.model_ids),historical_builds=historical_builds)
            (args.output/'opponent-schedules').mkdir(exist_ok=True)
            schedule_file=f'opponent-schedules/attempt-{attempt:05d}.json'
            write(schedule_file,{'pairs':schedule,'model_identities':dict(zip(evaluator.model_ids,[policy_id,*opponent_ids])),
                                 'pairing':'compiled even/odd seeds swap seats, not identical shuffle'})
            result=evaluator.evaluate(sample['rows'],schedule=schedule,max_actions=args.max_actions,stop_check=stop)
            source_update = update
            (args.output/'candidates').mkdir(exist_ok=True)
            for i, card_row in enumerate(sample['rows']):
                candidate = tuner.deck(card_row)
                filename = f'candidates/attempt-{attempt:05d}-{i:03d}.json'
                write(filename,candidate)
                record = {'attempt':attempt,'sampling_update':source_update,'candidate_id':candidate['id'],
                          'candidate_file':filename,'character_ids':team,'policy':policy_id,'opponents':opponent_ids,
                          'rule_hash':identity,'valid':bool(result['valid'][i]),'trials':result['trials'],
                          'score':float(result['rewards'][i]),'mean_turn':float(result['mean_turn'][i]),
                          'wins':int(result['wins'][i]),'losses':int(result['losses'][i]),
                          'truncated':int(result['truncated'][i]),'independently_evaluated':False,
                          'generation':'sampled from allocation policy; not a mutation of a parent deck',
                          'opponent_schedule':schedule_file,
                          'by_opponent':{k:{pos:{f:int(v[i]) for f,v in counts.items()} for pos,counts in group.items()} for k,group in result['by_opponent'].items()},'by_lineup':{k:{pos:{f:int(v[i]) for f,v in counts.items()} for pos,counts in group.items()} for k,group in result['by_lineup'].items()},
                          'by_position':{position:{key:int(result[f'{position}_{key}'][i])
                              for key in ('wins','losses','games')} for position in ('first','second')}}
                with (args.output/'candidate-history.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stats=update_tuner(tuner,optimizer,sample,result['rewards'],result['valid'],entropy_coef=args.entropy_coef)
            if stats['updated']:
                update+=1;count+=1
                eligible=result['valid'].nonzero().flatten()
                best=int(eligible[result['rewards'][eligible].argmax()])
                write('candidate.json',tuner.deck(sample['rows'][best]))
                write('candidate-score.json',{'score':float(result['rewards'][best]),'trials':result['trials'],
                      'update':update,'independently_evaluated':False,'label':'Current-batch candidate, not a global optimum',
                      'schema':SCHEMA,'gpu_lock':GPU_LOCK,'rule_hash':identity,
                      'resume':str(args.resume) if args.resume else None,
                      'sampling_update':update-1,'candidate_sha256':_sha256(args.output/'candidate.json'),
                      'policy':policy_id,'opponents':opponent_ids})
            row={'attempt':attempt,'update':update,'seconds':time.monotonic()-started,
                 'truncated_games':int(result['truncated'].sum()),**stats}
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+chr(10))
            write('status.json',{'phase':'tuning',**row})
            if count and count%10==0:save()
            del sample,result
            stalled=args.stop_if_stalled and stats['updated'] and monitor.observe(row['seconds'],row)
            if args.stop_if_stalled and monitor.last_summary:write('progress-windows.json',monitor.last_summary)
            if stalled:stop_reason='no_progress';break
    except (Deadline,KeyboardInterrupt):pass
    except Exception as exc:
        failed=True;write('error.json',{'error':repr(exc)});raise
    finally:
        try:save()
        finally:evaluator.close()
        write('status.json',{'phase':'failed' if failed else 'stopped','update':update,
                             'updates_this_run':count,'attempts':attempt,'formal_analysis_run':False,'stop_reason':stop_reason})
    return 0 if count else 2


if __name__=='__main__':raise SystemExit(main())
