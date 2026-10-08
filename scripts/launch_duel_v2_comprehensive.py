#!/usr/bin/env python3
"""Detached comprehensive training with a process-tree wall-clock watchdog."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=21600)
    p.add_argument('--starter-checkpoint',type=Path)
    p.add_argument('--weave-checkpoint',type=Path)
    p.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not 120<=args.seconds<=21600:p.error('seconds must be 120..21600')
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    receipt=out.with_name(out.name+'-launch.json');guard=out.with_name(out.name+'-guard.json')
    if not args.supervise:
        if out.exists() or receipt.exists():p.error('Choose a new run path')
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--supervise']
        options=dict(cwd=ROOT,stdin=subprocess.DEVNULL,close_fds=True)
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|getattr(subprocess,'CREATE_BREAKAWAY_FROM_JOB',0x01000000)
        else:options['start_new_session']=True
        with out.with_name(out.name+'-console.log').open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,**options)
        record={'supervisor_pid':process.pid,'seconds_limit':args.seconds,'launched_epoch':time.time(),'output':str(out)}
        receipt.write_text(json.dumps(record)+'\n');print(json.dumps(record));return
    started=time.time()
    command=[sys.executable,'-X','utf8',str(ROOT/'scripts/smoke_duel_v2_full.py'),'--profile','comprehensive',
             '--phase','train','--seconds',str(args.seconds-60),'--output',str(out)]
    for flag,path in (('--starter-checkpoint',args.starter_checkpoint),('--weave-checkpoint',args.weave_checkpoint)):
        if path:command += [flag,str(path.resolve())]
    process=subprocess.Popen(command,cwd=ROOT,start_new_session=os.name!='nt')
    state={'worker_pid':process.pid,'started_epoch':started,'deadline_epoch':started+args.seconds,'phase':'running'}
    guard.write_text(json.dumps(state)+'\n')
    try:
        code=process.wait(timeout=max(1,args.seconds-(time.time()-started)))
        state.update(phase='finished' if code==0 else 'failed',exit_code=code)
    except subprocess.TimeoutExpired:
        if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=False)
        else:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
        state.update(phase='watchdog_timeout',exit_code=124)
    state['elapsed_seconds']=time.time()-started
    guard.write_text(json.dumps(state)+'\n')


if __name__=='__main__':main()
