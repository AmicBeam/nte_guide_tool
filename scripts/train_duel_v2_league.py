#!/usr/bin/env python3
"""Bounded official-rule three-strategy league, with a process-tree supervisor."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def write(p,data):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2),encoding='utf-8');q.replace(p)

def phase_budget(seconds, smoke=False):
    """Reserve evaluation and setup time before splitting all six stages."""
    if smoke:
        return min(100, max(1, (seconds-45)//3)), 30
    if seconds < 600:
        raise ValueError('Full three-policy runs require at least 600 seconds')
    reserve=min(900,seconds//3)
    return min(720,max(1,(seconds-reserve-60)//6)),reserve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--starter',type=Path,required=True);p.add_argument('--weave',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=5400);p.add_argument('--deadline',type=float);p.add_argument('--worker',action='store_true');p.add_argument('--supervise',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--workers',type=int,default=6);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not 60<=a.seconds<=5400:p.error('Budget must be 60..5400 seconds total')
    try:phase_seconds,evaluation_reserve=phase_budget(a.seconds,a.smoke)
    except ValueError as exc:p.error(str(exc))
    out=a.output.resolve();sources={'starter':a.starter.resolve(),'weave-rush':a.weave.resolve()}
    args=['--output',str(out),'--starter',str(sources['starter']),'--weave',str(sources['weave-rush']),'--seconds',str(a.seconds),'--device',a.device,'--workers',str(a.workers)]+(['--smoke'] if a.smoke else [])
    if not a.worker and not a.supervise:
        if out.exists():p.error('Output must be new')
        for path in sources.values():
            if not path.is_file():p.error('Missing source checkpoint')
        out.mkdir(parents=True);start=time.time();deadline=start+a.seconds
        opts={'cwd':ROOT,'stdin':subprocess.DEVNULL,'close_fds':True}
        if os.name=='nt':opts['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|0x01000000
        else:opts['start_new_session']=True
        with (out/'console.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-X','utf8',__file__,*args,'--deadline',str(deadline),'--supervise'],stdout=log,stderr=subprocess.STDOUT,**opts)
        receipt={'started_epoch':start,'deadline_epoch':deadline,'supervisor_pid':proc.pid,'seconds_limit':a.seconds,'presets':['starter','weave-rush','quick-rush']}
        write(out/'launch.json',receipt);print(json.dumps(receipt));return
    if not a.deadline:p.error('Internal absolute deadline required')
    if a.supervise:
        proc=subprocess.Popen([sys.executable,'-X','utf8',__file__,*args,'--deadline',str(a.deadline),'--worker'],cwd=ROOT,start_new_session=os.name!='nt')
        guard={'phase':'running','worker_pid':proc.pid,'deadline_epoch':a.deadline};write(out/'guard.json',guard)
        try:guard['exit_code']=proc.wait(timeout=max(1,a.deadline-time.time()));guard['phase']='finished' if guard['exit_code']==0 else 'failed'
        except subprocess.TimeoutExpired:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=False)
            else:
                import signal
                os.killpg(proc.pid,signal.SIGKILL)
            guard.update(phase='hard_deadline',exit_code=124)
        guard['finished_epoch']=time.time();write(out/'guard.json',guard);return
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    try:
        from app.modules.card_game.rl.league_experiment import experiment
        experiment(out,sources,a.deadline,device=a.device,workers=a.workers,phase_seconds=phase_seconds,evaluation_reserve=evaluation_reserve,total_seconds=a.seconds,max_cycles=1 if a.smoke else 0,smoke=a.smoke)
    except Exception as exc:
        write(out/'status.json',{'phase':'failed','error':repr(exc),'finished_epoch':time.time()});raise
if __name__=='__main__':main()
