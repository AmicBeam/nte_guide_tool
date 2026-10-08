#!/usr/bin/env python3
"""Run a bounded search-distillation pilot; never publish its weights."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starter',type=Path,required=True);p.add_argument('--weave',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--worker',action='store_true')
    p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if not 180<=a.seconds<=600:p.error('Pilot budget must be 180..600 seconds')
    if not a.worker:
        if a.output.exists():p.error('Output must be new')
        deadline=time.time()+a.seconds
        cmd=[sys.executable,__file__,*sys.argv[1:],'--worker','--deadline',str(deadline)]
        child=subprocess.Popen(cmd,start_new_session=True,cwd=ROOT)
        try:code=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=124
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'guard.json').write_text(json.dumps(dict(exit_code=code,finished=time.time(),deadline=deadline)))
        raise SystemExit(code)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    from app.modules.card_game.rl.search_pilot import run
    run(a.output,{'starter':a.starter,'weave-rush':a.weave},a.deadline)

if __name__=='__main__':main()
