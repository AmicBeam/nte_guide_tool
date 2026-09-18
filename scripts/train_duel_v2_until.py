#!/usr/bin/env python3
"""Detached cross-platform supervisor for one explicitly dated training window."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def epoch(value):
    d=datetime.fromisoformat(value)
    if d.tzinfo is None:raise argparse.ArgumentTypeError('Explicit timezone offset required')
    return d.timestamp()


def write(p,data):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def stop_tree(proc):
    if proc.poll() is not None:return
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=False,stdout=subprocess.DEVNULL)
    else:os.killpg(proc.pid,signal.SIGKILL)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--train-until',type=epoch);p.add_argument('--evaluate-until',type=epoch);p.add_argument('--hard-deadline',type=epoch)
    p.add_argument('--user-snapshot',type=Path);p.add_argument('--workers',type=int,default=6);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--simulations',type=int,default=256);p.add_argument('--selection-interval',type=int,default=3600)
    p.add_argument('--smoke',action='store_true');p.add_argument('--supervise',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if not a.supervise and not a.worker:
        from app.modules.card_game.rl.information_longrun import validate_deadlines
        from app.modules.card_game.rl.search_budget_eval import freeze_models
        from app.modules.card_game.rl.league_schema import rule_hash
        if a.initial is None or a.reference is None or None in (a.train_until,a.evaluate_until,a.hard_deadline):p.error('Initial/reference and all three dated deadlines required')
        validate_deadlines(a.train_until,a.evaluate_until,a.hard_deadline)
        if not 1<=a.workers<=8 or not 8<=a.simulations<=512 or a.selection_interval<300:p.error('Invalid workload bounds')
        if out.exists():p.error('Output must be new; no duplicate launch')
        out.mkdir(parents=True)
        frozen=out/'sources';digest=freeze_models(a.initial,a.reference,frozen)
        (frozen/'learner').rename(out/'initial');(frozen/'opponent').rename(out/'reference');frozen.rmdir()
        config=dict(kind='information_noon_longrun_v1',rule_hash=rule_hash(),created=time.time(),
                    timezone='Asia/Shanghai',train_until=a.train_until,evaluate_until=a.evaluate_until,hard_deadline=a.hard_deadline,
                    workers=a.workers,device=a.device,simulations=a.simulations,selection_interval=a.selection_interval,
                    smoke=a.smoke,max_cycles=1 if a.smoke else 0,user_snapshot=str(a.user_snapshot.resolve()) if a.user_snapshot else None,source_files=digest,automatic_serving_approval=False)
        write(out/'config.json',config)
        opts=dict(cwd=ROOT,stdin=subprocess.DEVNULL,close_fds=True)
        if os.name=='nt':opts['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|0x01000000
        else:opts['start_new_session']=True
        with (out/'console.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-X','utf8',__file__,'--output',str(out),'--supervise'],stdout=log,stderr=subprocess.STDOUT,**opts)
        for _ in range(100):
            if (out/'guard.json').exists():break
            if proc.poll() is not None:raise RuntimeError('Supervisor exited before acknowledgement')
            time.sleep(.1)
        if not (out/'guard.json').exists():raise RuntimeError('Supervisor did not acknowledge startup')
        receipt=dict(supervisor_pid=proc.pid,started=time.time(),train_until=a.train_until,evaluate_until=a.evaluate_until,hard_deadline=a.hard_deadline)
        write(out/'launch.json',receipt);print(json.dumps(receipt));return
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if a.supervise:
        # Prevent idle system sleep only while this owned supervisor is alive.
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        end_monotonic=time.monotonic()+max(0,config['hard_deadline']-time.time());attempt=0
        try:
            while time.time()<config['hard_deadline'] and time.monotonic()<end_monotonic:
                args=[sys.executable,'-X','utf8',__file__,'--output',str(out),'--worker']+(['--resume'] if attempt else [])
                proc=subprocess.Popen(args,cwd=ROOT,start_new_session=os.name!='nt')
                guard=dict(phase='running',worker_pid=proc.pid,attempt=attempt,hard_deadline=config['hard_deadline'])
                write(out/'guard.json',guard)
                while proc.poll() is None:
                    remaining=min(config['hard_deadline']-time.time(),end_monotonic-time.monotonic())
                    if remaining<=0:stop_tree(proc);break
                    try:proc.wait(timeout=min(30,remaining))
                    except subprocess.TimeoutExpired:pass
                guard.update(exit_code=proc.poll(),finished=time.time())
                if time.time()>=config['hard_deadline'] or time.monotonic()>=end_monotonic:guard['phase']='hard_deadline'
                elif proc.returncode==0:guard['phase']='finished'
                else:guard['phase']='failed'
                write(out/'guard.json',guard)
                status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                if guard['phase']=='failed' and status.get('recoverable') and attempt<1 and (out/'recovery/pointer.json').exists() and time.time()<config['train_until']-600:
                    attempt+=1;continue
                break
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from concurrent.futures.process import BrokenProcessPool
    from app.modules.card_game.rl.information_longrun import run
    try:
        if time.time()>=config['hard_deadline']:raise TimeoutError('Original deadline has expired')
        run(out,config,a.resume)
    except Exception as exc:
        write(out/'status.json',dict(phase='failed',error=repr(exc),recoverable=isinstance(exc,(OSError,BrokenProcessPool)),finished=time.time(),deadline=config['hard_deadline']))
        raise

if __name__=='__main__':main()
