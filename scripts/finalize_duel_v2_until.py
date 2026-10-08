#!/usr/bin/env python3
"""Evaluation-only supervised continuation from an owned completed checkpoint.

Training must already be stopped. The original config/source are not modified.
A positive cycle cap equal to the restored cycle disables the training loop even
if the wall clock moves backwards. Original evaluation and hard deadlines remain.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def finalization_config(original,cycle,now):
    if cycle<1:raise ValueError('A completed training checkpoint is required')
    if now>=original['hard_deadline']:raise ValueError('Original hard deadline expired')
    result=dict(original)
    result.update(train_until=min(original['train_until'],now),max_cycles=cycle,
                  finalize_only=True,original_train_until=original['train_until'])
    return result


def write(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--supervise',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=a.output.resolve();source=a.source_root.resolve();sys.path.insert(0,str(source))
    original=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if time.time()>=original['hard_deadline']:raise ValueError('Original hard deadline expired')
    args=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--output',str(out),'--source-root',str(source)]
    if not a.supervise and not a.worker:
        guard=json.loads((out/'guard.json').read_text(encoding='utf-8'))
        if guard.get('phase')=='running':raise ValueError('Original supervisor must have stopped first')
        if (out/'finalizer-launch.json').exists():raise ValueError('Finalizer already launched')
        opts=dict(cwd=source,stdin=subprocess.DEVNULL,close_fds=True)
        if os.name=='nt':opts['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|0x01000000
        else:opts['start_new_session']=True
        with (out/'finalizer-console.log').open('xb') as log:
            proc=subprocess.Popen(args+['--supervise'],stdout=log,stderr=subprocess.STDOUT,**opts)
        for _ in range(100):
            g=json.loads((out/'guard.json').read_text(encoding='utf-8'))
            if g.get('kind')=='evaluation_only':break
            if proc.poll() is not None:raise RuntimeError('Finalizer supervisor exited before acknowledgement')
            time.sleep(.1)
        else:raise RuntimeError('Finalizer did not acknowledge')
        receipt=dict(supervisor_pid=proc.pid,started=time.time(),kind='evaluation_only',hard_deadline=original['hard_deadline'])
        write(out/'finalizer-launch.json',receipt);print(json.dumps(receipt));return
    if a.supervise:
        from scripts.train_duel_v2_until import stop_tree
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            end=time.monotonic()+max(0,original['hard_deadline']-time.time())
            proc=subprocess.Popen(args+['--worker'],cwd=source,start_new_session=os.name!='nt')
            g=dict(phase='running',kind='evaluation_only',worker_pid=proc.pid,hard_deadline=original['hard_deadline'])
            write(out/'guard.json',g)
            while proc.poll() is None:
                remaining=min(original['hard_deadline']-time.time(),end-time.monotonic())
                if remaining<=0:stop_tree(proc);break
                try:proc.wait(timeout=min(30,remaining))
                except subprocess.TimeoutExpired:pass
            g.update(phase='finished' if proc.returncode==0 else 'hard_deadline' if time.time()>=original['hard_deadline'] else 'failed',exit_code=proc.poll(),finished=time.time())
            write(out/'guard.json',g)
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    try:
        import torch
        pointer=json.loads((out/'recovery/pointer.json').read_text(encoding='utf-8'))
        name=pointer['path']
        if Path(name).name!=name:raise ValueError('Invalid recovery pointer')
        path=out/'recovery'/name;raw=(path/'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pointer['manifest_sha256']:raise ValueError('Recovery manifest mismatch')
        for name,h in json.loads(raw).items():
            if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=h:raise ValueError('Checkpoint hash mismatch')
        state=torch.load(path/'training.pt',map_location='cpu',weights_only=True)['state']
        effective=finalization_config(original,state['cycle'],time.time())
        write(out/'finalization-config.json',effective)
        write(out/'finalization-baseline.json',dict(cycle=state['cycle'],games=state['games'],totals=state['totals'],checkpoint=pointer['path']))
        from app.modules.card_game.rl.information_longrun import run
        run(out,effective,resume=True)
        result=json.loads((out/'training-summary.json').read_text(encoding='utf-8'))
        if result['cycle']!=state['cycle'] or result['totals']!=state['totals'] or result['games']!=state['games']:
            raise ValueError('Evaluation-only continuation changed training counters')
        write(out/'finalization-verification.json',dict(training_counters_unchanged=True,finished=time.time()))
    except Exception as exc:
        write(out/'status.json',dict(phase='failed',error=repr(exc),kind='evaluation_only',finished=time.time()))
        raise

if __name__=='__main__':main()
