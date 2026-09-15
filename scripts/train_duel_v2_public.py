#!/usr/bin/env python3
"""Explicit bounded public CUDA league training; analysis is a separate command."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--deck', choices=('starter','weave-rush'),default='starter')
    p.add_argument('--n', type=int,default=4096)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--seconds',type=float,default=7200)
    p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--history',type=int,default=4)
    p.add_argument('--snapshot-updates',type=int,default=20)
    p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--preflight-cases',type=int,default=32)
    p.add_argument('--resume',type=Path)
    p.add_argument('--learner-deck',type=Path,help='Legal card allocation for the same preset characters')
    p.add_argument('--warm-start',type=Path,help='Transfer legacy input weights; start a fresh optimizer')
    p.add_argument('--stop-if-stalled',action='store_true')
    p.add_argument('--detach',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.resume and args.warm_start:p.error('--resume and --warm-start are mutually exclusive')
    if min(args.n,args.horizon,args.seconds,args.hidden,args.history,args.snapshot_updates)<=0 or args.max_updates<0:
        p.error('Budgets and dimensions must be positive')
    if not 1 <= args.preflight_cases <= 1024:p.error('preflight-cases must be 1..1024')
    from app.modules.card_game.rl.episode_return import EPISODE_RETURN_CAP
    from app.modules.card_game.rl.public_schema import SCHEMA,state_dim,rule_identity,CAND_DIM
    from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
    from app.modules.card_game.rl.capability import encoder_public_support
    from app.modules.card_game.rl.public_training import training_deck
    learner_deck = training_deck(args.deck, json.loads(args.learner_deck.read_text(encoding='utf-8')) if args.learner_deck else None)
    if args.check:
        print(json.dumps({'schema':SCHEMA,'state_dim':state_dim(),'rule_hash':rule_identity(),
                          'public_catalog':encoder_public_support(),'learner_deck':learner_deck,'training_started':False},ensure_ascii=False))
        return 0
    if not args.train or not args.output:
        p.error('Use --check, or --train --output NEWDIR')
    if args.detach:
        if args.output.exists():p.error('Use a new output directory')
        args.output.mkdir(parents=True)
        command=[sys.executable,str(Path(__file__).resolve()),
                 *[a for a in (sys.argv[1:] if argv is None else argv) if a!='--detach'],'--worker']
        options={'cwd':ROOT,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':
            options['creationflags']=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
                                     |getattr(subprocess,'CREATE_BREAKAWAY_FROM_JOB',0x01000000))
        else:options['start_new_session']=True
        with (args.output/'console.log').open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,**options)
        (args.output/'launch.json').write_text(json.dumps({'pid':process.pid,'command':command})+'\n')
        print(json.dumps({'pid':process.pid,'output':str(args.output),'status':'starting'}))
        return 0
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA unavailable; no CPU training fallback')
    if args.output.exists() and (not args.worker or (args.output/'config.json').exists()):
        p.error('Use a new output directory')
    from scripts.verify_duel_v2_public import main as verify
    verify(['--backend','cuda','--cases',str(args.preflight_cases)])
    from app.modules.card_game.rl.public_training import FrozenLeague,PublicTrainingEnv,warm_start_weights
    from app.modules.card_game.rl.public_observation import public_observe
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout,ppo_update,rollout_to_batch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    league=FrozenLeague(args.history)
    net=CompactScorer(state_dim(),CAND_DIM,args.hidden).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    update=0
    identity=rule_identity()
    if args.warm_start:
        warm_start_weights(torch.load(args.warm_start,map_location='cuda',weights_only=True),net,args.deck)
    if args.resume:
        saved=torch.load(args.resume,map_location='cuda',weights_only=True)
        if (saved.get('schema'),saved.get('deck'),saved.get('rule_identity',{}).get('rule_hash')) != (SCHEMA,args.deck,identity):
            raise ValueError('Resume observation/deck/rule identity mismatch')
        if not args.learner_deck:
            learner_deck = training_deck(args.deck, saved.get('learner_deck'))
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        update=int(saved['update'])
        for weights in saved.get('league',[]):
            previous=CompactScorer(state_dim(),CAND_DIM,args.hidden).cuda()
            previous.load_state_dict(weights);league.add(previous)
        if saved.get('torch_rng') is not None:
            torch.set_rng_state(saved['torch_rng'].cpu())
        if saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    if not league.history:
        league.add(net)
    env=PublicTrainingEnv(args.n,deck_id=args.deck,league=league,device='cuda',learner_deck=learner_deck)
    args.output.mkdir(parents=True,exist_ok=args.worker)
    started=time.monotonic()
    def write(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def save():
        temporary=args.output/'latest.tmp'
        torch.save({'schema':SCHEMA,'deck':args.deck,'hidden':args.hidden,'state_dim':state_dim(),
                    'cand_dim':CAND_DIM,'gpu_lock':GPU_LOCK,'rule_identity':{'rule_hash':identity},
                    'compiled_identity':env.compiled.identity(),'model':net.state_dict(),
                    'optimizer':opt.state_dict(),'update':update,'league':[n.state_dict() for n in league.history],
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'learner_deck':learner_deck,'training_distribution':'public_asymmetric_v1','validated':False},temporary)
        temporary.replace(args.output/'latest.pt')
    class Deadline(Exception): pass
    def stop():
        if time.monotonic()-started>=args.seconds: raise Deadline()
    write('config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                         'schema':SCHEMA,'learner_build':learner_deck,'detailed_training_logs':False,'episode_return_cap':EPISODE_RETURN_CAP})
    write('preflight.json',{'ok':True,'backend':'cuda','cases':args.preflight_cases,'rule_hash':identity})
    from app.modules.card_game.rl.progress_stop import ProgressStop
    monitor=ProgressStop();stop_reason='budget_or_update_limit'
    code=0
    try:
        count=0
        while not args.max_updates or count<args.max_updates:
            stop()
            rollout=collect_rollout(env,net,args.horizon,observe_fn=public_observe,stop_check=stop)
            stats=ppo_update(net,opt,rollout_to_batch(rollout),stop_check=stop)
            update+=1;count+=1
            row={'update':update,'seconds':time.monotonic()-started,
                 'finished_games':int(rollout['done'].sum()),'by_opponent_position':env.terminal_counters.summary(),**stats}
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
            del rollout
            if update%args.snapshot_updates==0: league.add(net)
            if update%10==0: save()
            write('status.json',{'phase':'training',**row})
            stalled=args.stop_if_stalled and monitor.observe(row['seconds'],row)
            if args.stop_if_stalled and monitor.last_summary:write('progress-windows.json',monitor.last_summary)
            if stalled:stop_reason='no_progress';break
    except (Deadline,KeyboardInterrupt):
        pass
    except Exception as exc:
        code=1
        write('error.json',{'error':repr(exc)})
        raise
    finally:
        save();env.close()
        write('position-results.json',{'opponents':env.terminal_counters.summary(),'unfinished_at_stop':int((env.state.phase!=3).sum())})
        write('status.json',{'phase':'failed' if code else 'stopped','update':update,
                             'seconds':time.monotonic()-started,'analysis_run':False,'stop_reason':stop_reason})
    return code


if __name__=='__main__':
    raise SystemExit(main())
