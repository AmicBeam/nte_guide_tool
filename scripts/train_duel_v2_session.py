#!/usr/bin/env python3
"""Bounded two-preset resident PPO session with paired eval and final play logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


class TrainingStop(Exception):
    pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10800)
    p.add_argument('--deadline',default='2026-09-15T09:00:00+08:00')
    p.add_argument('--n',type=int,default=1024)
    p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--minibatch',type=int,default=1024)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--max-updates',type=int,default=0,help='Optional per-preset short-test limit')
    p.add_argument('--eval-seconds',type=float,default=300)
    p.add_argument('--eval-pairs',type=int,default=64)
    p.add_argument('--eval-timeout',type=float,default=240)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--plateau-seconds',type=float,default=1800)
    p.add_argument('--min-delta',type=float,default=.01)
    p.add_argument('--log-pairs',type=int,default=128)
    p.add_argument('--log-timeout',type=float,default=600)
    p.add_argument('--resume-root', type=Path)
    p.add_argument('--detach',action='store_true')
    args=p.parse_args()
    if min(args.seconds,args.n,args.horizon,args.epochs,args.minibatch,args.eval_pairs,args.log_pairs)<=0:
        p.error('Budgets, concurrency and sample counts must be positive')
    absolute=datetime.fromisoformat(args.deadline).timestamp()
    if time.time()>=absolute:
        p.error('Absolute training deadline already passed; no training started')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_status.json').exists():
        p.error('Output already contains a session; choose a new run directory')
    if args.detach:
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--detach']]
        kwargs=dict(cwd=ROOT,stdin=subprocess.DEVNULL,close_fds=True)
        if os.name=='nt':
            kwargs['creationflags']=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                     | getattr(subprocess,'CREATE_BREAKAWAY_FROM_JOB',0x01000000))
        else:
            kwargs['start_new_session']=True
        with (out/'console.log').open('ab') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,**kwargs)
        write_json(out/'launch.json',dict(pid=proc.pid,command=command,launched_at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(pid=proc.pid,output=str(out)),ensure_ascii=False))
        return 0

    import torch
    from app.modules.card_game.rl.episode_return import EPISODE_RETURN_CAP
    from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout,rollout_to_batch,ppo_update
    from app.modules.card_game.rl.gpu_duel.resident_obs import resident_observe,SCHEMA
    from app.modules.card_game.rl.gpu_duel.resident_eval import evaluate_resident
    from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
    from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU training fallback')
    torch.set_num_threads(1)
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    started=time.monotonic()
    end=started+min(args.seconds,absolute-time.time())
    status=dict(pid=os.getpid(),phase='initializing',started_at=datetime.now(timezone.utc).isoformat(),
                train_seconds_limit=args.seconds,absolute_deadline=args.deadline,device=torch.cuda.get_device_name(),
                decks={},training_play_logging=False)
    write_json(out/'run_status.json',status)
    config={**{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},'output':str(out),'resume_root':str(args.resume_root) if args.resume_root else None,'schema':SCHEMA,'gpu_lock':GPU_LOCK,
            'eval_seed_base':900_000_000,'log_seed_base':1_100_000_000,
            'episode_return_cap':EPISODE_RETURN_CAP,
            'threshold':.5,'threshold_scope':'each deck: equal-weight first/second aggregate; draws not wins'}
    write_json(out/'config.json',config)
    cases=[]
    def checkpoint(case,name):
        payload=dict(model=case['net'].state_dict(),optimizer=case['opt'].state_dict(),
                     update=case['updates'],schema=SCHEMA,deck=case['deck'],gpu_lock=GPU_LOCK,
                     hidden=args.hidden,state_dim=case['state_dim'],cand_dim=case['cand_dim'],
                     rule_identity=case['env'].compiled.identity(),best_score=case['best_score'])
        path=case['dir']/name
        tmp=path.with_suffix('.tmp')
        torch.save(payload,tmp)
        tmp.replace(path)
    def stop_check():
        if time.monotonic()>=end or time.time()>=absolute:
            raise TrainingStop('time_limit')
    def evaluate(case):
        remaining=min(end-time.monotonic(),absolute-time.time())
        if remaining<=0:
            raise TrainingStop('time_limit')
        report=evaluate_resident(case['net'],case['oracle'],deck_id=case['deck'],pairs=args.eval_pairs,
                                 max_seconds=min(args.eval_timeout,remaining),
                                 checkpoint=f'update_{case["updates"]}')
        report['update']=case['updates']
        report['elapsed_seconds']=time.monotonic()-started
        case['last_eval']=time.monotonic()
        case['evals']+=1
        write_json(case['dir']/f'eval-{case["evals"]:04d}.json',report)
        if report['complete']:
            score=report['equal_weight_win_rate']
            if score>case['best_score']:
                case['best_score']=score
                checkpoint(case,'best.pt')
            if score>case['significant_best']+args.min_delta:
                case['significant_best']=score
                case['no_improve']=0
                case['improved_at']=time.monotonic()
            else:
                case['no_improve']+=1
            case['plateau']=case['best_score']>=.5 and case['no_improve']>=args.patience and time.monotonic()-case['improved_at']>=args.plateau_seconds
        status['decks'][case['deck']].update(best_score=case['best_score'], plateau=case['plateau'], evaluations=case['evals'])
        write_json(out/'run_status.json',status)
        print(json.dumps(dict(type='evaluation',deck=case['deck'],update=case['updates'],
                              complete=report['complete'],win_rate=report['equal_weight_win_rate'],
                              best=case['best_score'],plateau=case['plateau'])),flush=True)
    exit_code=0
    stop_reason='time_limit'
    try:
        for deck in ('starter','weave-rush'):
            stop_check()
            directory=out/deck;directory.mkdir()
            env=GpuDuelEnv(args.n,device='cuda',deck=deck,compiled_backend='cuda')
            env.reset(seeds=list(range(args.n)))
            state,cand,mask=resident_observe(env.state)
            net=CompactScorer(state.shape[-1],cand.shape[-1],hidden=args.hidden).cuda()
            oracle=CompiledStarterBackend(directory/'native-cache',backend='native',deck_id=deck)
            case=dict(deck=deck,dir=directory,env=env,net=net,oracle=oracle,
                      opt=torch.optim.Adam(net.parameters(),lr=3e-4),updates=0,
                      state_dim=state.shape[-1],cand_dim=cand.shape[-1],last_eval=time.monotonic(),
                      best_score=-1.,significant_best=-1.,no_improve=0,improved_at=time.monotonic(),
                      evals=0,plateau=False,train_seconds=0.)
            if args.resume_root:
                source=args.resume_root/deck
                selected=source/('best.pt' if (source/'best.pt').exists() else 'latest.pt')
                saved=torch.load(selected,map_location='cuda',weights_only=False)
                if saved.get('schema')!=SCHEMA or saved.get('gpu_lock')!=GPU_LOCK or saved.get('deck')!=deck:
                    raise ValueError('Checkpoint schema/catalog/deck mismatch')
                if saved.get('rule_identity',{}).get('rule_hash')!=env.compiled.identity()['rule_hash']:
                    raise ValueError('Checkpoint rule source mismatch')
                net.load_state_dict(saved['model']);case['opt'].load_state_dict(saved['optimizer'])
                case['resume_checkpoint']=str(selected)
            cases.append(case)
            status['decks'][deck]=dict(updates=0,best_score=-1.)
        status['phase']='training'
        write_json(out/'run_status.json',status)
        while True:
            stop_check()
            if all(c['plateau'] for c in cases):
                stop_reason='both_above_threshold_and_plateau';break
            if args.max_updates and all(c['updates']>=args.max_updates for c in cases):
                stop_reason='short_update_limit';break
            for case in cases:
                if case['plateau'] or (args.max_updates and case['updates']>=args.max_updates):continue
                t=time.monotonic()
                rollout=collect_rollout(case['env'],case['net'],args.horizon,observe_fn=resident_observe,stop_check=stop_check)
                collected=time.monotonic()
                batch=rollout_to_batch(rollout)
                stats=ppo_update(case['net'],case['opt'],batch,epochs=args.epochs,minibatch=args.minibatch,stop_check=stop_check)
                torch.cuda.synchronize()
                elapsed=time.monotonic()-t
                case['updates']+=1;case['train_seconds']+=elapsed
                row=dict(update=case['updates'],deck=case['deck'],seconds=elapsed,
                         collect_seconds=collected-t,steps=args.n*args.horizon,
                         steps_per_second=args.n*args.horizon/elapsed,
                         episodes_finished=int(rollout['done'].sum()),reward_mean=float(rollout['reward'].mean()),
                         peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,**stats)
                with (case['dir']/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                del rollout,batch
                if case['updates']%10==0 or case['updates']==1:
                    checkpoint(case,'latest.pt')
                    print(json.dumps(row),flush=True)
                if time.monotonic()-case['last_eval']>=args.eval_seconds:
                    evaluate(case)
                status['decks'][case['deck']]=dict(updates=case['updates'],best_score=case['best_score'],
                                                   plateau=case['plateau'],train_seconds=case['train_seconds'])
                status['elapsed_seconds']=time.monotonic()-started
                write_json(out/'run_status.json',status)
        # Short tests exercise the same real paired evaluator before final logging.
        if args.max_updates:
            for case in cases:evaluate(case)
    except TrainingStop as exc:
        stop_reason=str(exc)
    except Exception:
        exit_code=1
        stop_reason='error'
        status['error']=traceback.format_exc()
        print(status['error'],flush=True)
    finally:
        status['stop_reason']=stop_reason
        status['optimization_finished_at']=datetime.now(timezone.utc).isoformat()
        for case in cases:
            checkpoint(case,'latest.pt')
            case['env'].compiled.close()
        status['phase']='logging_eval' if not exit_code else 'failed'
        write_json(out/'run_status.json',status)
    if not exit_code:
        try:
            for case in cases:
                selected=case['dir']/('best.pt' if (case['dir']/'best.pt').exists() else 'latest.pt')
                saved=torch.load(selected,map_location='cuda',weights_only=False)
                case['net'].load_state_dict(saved['model'])
                report=evaluate_resident(case['net'],case['oracle'],deck_id=case['deck'],pairs=args.log_pairs,
                                         seed_base=1_100_000_000,max_seconds=args.log_timeout,
                                         log_path=case['dir']/'final-play-log.jsonl',checkpoint=str(selected),run_id=out.name)
                write_json(case['dir']/'final-eval.json',report)
                status['decks'][case['deck']]['selected_checkpoint']=str(selected)
                status['decks'][case['deck']]['final_eval_complete']=report['complete']
                status['decks'][case['deck']]['final_win_rate']=report['equal_weight_win_rate']
                status['decks'][case['deck']]['threshold_met']=report['complete'] and report['equal_weight_win_rate']>=.5
                if not report['complete']:
                    exit_code=1
        except Exception:
            exit_code=1
            status['error']=traceback.format_exc()
    for case in cases:case['oracle'].close()
    status['phase']='complete' if exit_code==0 else 'failed'
    status['exit_code']=exit_code
    status['finished_at']=datetime.now(timezone.utc).isoformat()
    write_json(out/'run_status.json',status)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
