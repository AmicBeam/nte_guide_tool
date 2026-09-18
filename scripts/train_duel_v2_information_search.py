#!/usr/bin/env python3
"""Local CPU full-game search/value pilot with an absolute process-tree deadline."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starter',type=Path,required=True);p.add_argument('--weave',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--simulations',type=int,default=24)
    p.add_argument('--cycles',type=int,default=2);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if not 180<=a.seconds<=600:p.error('Short pilot budget must be 180..600 seconds')
    if not 1<=a.cycles<=3 or not 8<=a.simulations<=128 or not 1<=a.workers<=8:p.error('Invalid pilot limits')
    if not a.worker:
        if a.output.exists():p.error('Output must be new')
        deadline=time.time()+a.seconds
        child=subprocess.Popen([sys.executable,__file__,*sys.argv[1:],'--worker','--deadline',str(deadline)],cwd=ROOT,start_new_session=True)
        try:code=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=124
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'guard.json').write_text(json.dumps(dict(exit_code=code,finished=time.time(),deadline=deadline)))
        raise SystemExit(code)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from app.modules.card_game.rl.information_experiment import experiment
    try:
        experiment(a.output,{'starter':a.starter,'weave-rush':a.weave},a.deadline,cycles=a.cycles,
                   simulations=a.simulations,workers=a.workers)
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(exc),finished=time.time(),deadline=a.deadline)))
        raise

if __name__=='__main__':main()
