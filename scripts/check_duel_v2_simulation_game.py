#!/usr/bin/env python3
"""Two bounded, identical-seed full rollouts; never creates an optimizer."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import argparse,hashlib,json,os,sys,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.rl.information_search import search_game
from app.modules.card_game.rl.outcome_runtime import model
from app.modules.card_game.rl.fixed_lineup import write
from app.modules.card_game.rl.episode_replay import dump_episode


def job_run(job):
    started=time.perf_counter();result=search_game(job)
    return result,time.perf_counter()-started


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-reservation',type=Path,required=True);p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args()
    if not 60<=a.seconds<=900:raise ValueError('Bound is 60..900 seconds')
    a.output.mkdir(parents=True,exist_ok=False);seeds=json.loads(a.seed_reservation.read_text());policy=model(a.models,'zhenhong')
    source_hash=policy.version;deadline=time.time()+a.seconds
    common=dict(runtime='outcome',seed=seeds['foundation'],first='a',deadline=deadline,simulations=32,training=True,collect=True,
        record_episode=True,train_sides=['a','b'],decks={'a':policy.serving_deck,'b':policy.serving_deck},
        policies={'a':(str(a.models),'zhenhong'),'b':(str(a.models),'zhenhong')})
    jobs=[{**common,'id':label,'fast_simulation':fast} for label,fast in [('full',False),('fast',True)]]
    write(a.output/'jobs.json',jobs)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:results=list(pool.map(job_run,jobs))
    summary=dict(simulations_requested=32,optimizer_steps=0,source_model_sha=source_hash,results={})
    for j,(g,seconds) in zip(jobs,results):
        summary['results'][j['id']]=dict(complete=g['complete'],seconds=seconds,decisions=g['decisions'],winner=g.get('winner'),reason=g.get('reason'))
        if g['complete']:dump_episode(a.output/(j['id']+'.json.gz'),j,g)
    summary['complete']=all(g['complete'] for g,t in results)
    if summary['complete']:
        import numpy as np
        old,new=results[0][0],results[1][0]
        assert old['episode_actions']==new['episode_actions'] and old['winner']==new['winner']
        assert len(old['rows'])==len(new['rows']) and len(old['value_rows'])==len(new['value_rows'])
        for x,y in zip(old['rows']+old['value_rows'],new['rows']+new['value_rows']):
            assert x.keys()==y.keys()
            for k in x:
                if isinstance(x[k],np.ndarray):np.testing.assert_array_equal(x[k],y[k])
                else:assert x[k]==y[k]
        summary.update(identical_actions_and_labels=True,speedup=results[0][1]/results[1][1])
    assert hashlib.sha256((a.models/'zhenhong.npz').read_bytes()).hexdigest()==source_hash
    write(a.output/'summary.json',summary);print(json.dumps(summary))
    if not summary['complete']:raise SystemExit(2)
