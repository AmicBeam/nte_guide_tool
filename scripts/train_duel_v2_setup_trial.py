#!/usr/bin/env python3
"""Launch a one-hour isolated Zhenhong setup-reward A/B trial; no serving writes."""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.rl.fixed_lineup import FixedModel,identity,write
from app.modules.card_game.rl.build_acceptance import check_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--seed-reservation',type=Path);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true')
    p.add_argument('--supervise',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.worker or a.supervise:
        c=json.loads((out/'config.json').read_text())
        if a.worker:
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
            for name,digest in json.loads((out/'source-manifest.json').read_text()).items():
                f=(ROOT/name).resolve()
                if not f.is_relative_to(out) or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen identity mismatch')
            from app.modules.card_game.rl.zhenhong_setup_trial import run
            try:run(out,c)
            except Exception as e:write(out/'status.json',dict(phase='failed',error=repr(e)));raise
            return
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        command=[sys.executable,'-X','utf8',__file__,'--output',str(out),'--worker']
        worker=subprocess.Popen(command,cwd=ROOT,start_new_session=os.name!='nt');end=time.monotonic()+max(0,c['hard_deadline']-time.time())
        guard=dict(phase='running',worker_pid=worker.pid,hard_deadline=c['hard_deadline']);write(out/'guard.json',guard)
        try:
            while worker.poll() is None:
                remaining=min(end-time.monotonic(),c['hard_deadline']-time.time())
                if remaining<=0:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],check=False)
                    else:os.killpg(worker.pid,signal.SIGKILL)
                    worker.wait(timeout=15);guard['phase']='hard_deadline';break
                try:worker.wait(timeout=min(30,remaining))
                except subprocess.TimeoutExpired:pass
            if guard['phase']=='running':guard['phase']='finished' if worker.returncode==0 else 'failed'
            guard.update(exit_code=worker.returncode,finished=time.time());write(out/'guard.json',guard)
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    if a.source is None or a.seed_reservation is None:p.error('Source and independent seed reservation required')
    if not 1<=a.workers<=8 or not (120 if a.smoke else 600)<=a.seconds<=3600:p.error('Invalid bounded workload')
    keys=('starter','weave-rush','quick-rush','zhenhong');source=a.source.resolve();check_source(source)
    for k in keys:FixedModel(source,k)
    now=time.time();c=dict(kind='zhenhong_setup_ab_v1',created=now,rule_hash=identity(),device='cuda',workers=a.workers,simulations=8 if a.smoke else 256,smoke=a.smoke,
        train_until=now+a.seconds*.75,evaluate_until=now+a.seconds*.90,hard_deadline=now+a.seconds,
        arms={'control':0.,'setup':.1},passive_reward=.2,decay_steps=64,source=str(source),automatic_serving_approval=False)
    if not a.run:print(json.dumps(c));return
    if out.exists():raise ValueError('Output exists')
    out.mkdir(parents=True)
    marker=a.seed_reservation.with_suffix('.claimed');fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(str(out))
    c['seeds']=json.loads(a.seed_reservation.read_text())
    from scripts.train_duel_v2_full_cycle import freeze_source
    frozen,_=freeze_source(out,source,keys)
    target=frozen/'scripts'/Path(__file__).name;shutil.copy2(__file__,target)
    manifest=json.loads((out/'source-manifest.json').read_text());manifest['scripts/'+target.name]=hashlib.sha256(target.read_bytes()).hexdigest();write(out/'source-manifest.json',manifest)
    write(out/'config.json',c)
    opts=dict(cwd=frozen,stdin=subprocess.DEVNULL,close_fds=True)
    if os.name=='nt':opts['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|0x01000000
    else:opts['start_new_session']=True
    with (out/'console.log').open('xb') as log:
        sup=subprocess.Popen([sys.executable,'-X','utf8',str(target),'--output',str(out),'--supervise'],stdout=log,stderr=subprocess.STDOUT,**opts)
    for _ in range(100):
        if (out/'guard.json').exists():break
        if sup.poll() is not None:raise RuntimeError('Supervisor failed')
        time.sleep(.1)
    if not (out/'guard.json').exists():raise RuntimeError('No guard acknowledgement')
    receipt=dict(supervisor_pid=sup.pid,output=str(out),hard_deadline=c['hard_deadline']);write(out/'launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
