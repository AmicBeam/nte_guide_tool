#!/usr/bin/env python3
"""Compare raw/24/128/256 search with frozen NumPy policies and a hard deadline."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--learner',type=Path,required=True)
    p.add_argument('--opponent',type=Path,required=True);p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--seeds-per-pair',type=int,default=2);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--user-snapshot',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if not 120<=a.seconds<=1800:p.error('Evaluation budget must be 120..1800 seconds')
    if not 1<=a.seeds_per_pair<=8 or not 1<=a.repeats<=5 or not 1<=a.workers<=8:p.error('Invalid comparison limits')
    if not a.worker:
        if a.output.exists():p.error('Output must be new')
        deadline=time.time()+a.seconds
        child=subprocess.Popen([sys.executable,__file__,*sys.argv[1:],'--worker','--deadline',str(deadline)],cwd=ROOT,start_new_session=True)
        try:code=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=124
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'guard.json').write_text(json.dumps(dict(exit_code=code,deadline=deadline,finished=time.time())))
        raise SystemExit(code)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from app.modules.card_game.rl.search_budget_eval import experiment
    try:
        experiment(a.output,a.learner,a.opponent,a.deadline,seeds_per_pair=a.seeds_per_pair,
                   repeats=a.repeats,workers=a.workers,user_snapshot=a.user_snapshot)
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(exc),finished=time.time())))
        raise

if __name__=='__main__':main()
