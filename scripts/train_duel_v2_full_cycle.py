#!/usr/bin/env python3
"""Three-strategy protocol with canonical builds and fail-closed delivery gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def freeze_source(out,initial,keys=('starter','weave-rush','quick-rush')):
    """Explicit source/asset/model whitelist; never copy DB, env, or credentials."""
    frozen=out/'frozen';files=list((ROOT/'app').rglob('*.py'))
    files.append(ROOT/'app/modules/card_game/content/duel_v2/catalog.json')
    catalog=json.loads(files[-1].read_text())
    registry=ROOT/'app/modules/card_game/rl/experiment_dispositions.json'
    if registry.exists():files.append(registry)
    for character in catalog['characters']:
        for field in ('avatar','portrait'):
            value=character.get(field,'')
            if value.startswith('/static/images/'):
                p=ROOT/'app'/value.lstrip('/')
                if p.is_file():files.append(p)
    files.append(Path(__file__).resolve());manifest={}
    for p in sorted(set(files)):
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Out-of-scope source path')
        name=p.relative_to(ROOT);target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);manifest[name.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    sources=out/'sources';sources.mkdir()
    for key in keys:
        for ext in ('npz','json'):
            p=initial/f'{key}.{ext}'
            if p.is_symlink():raise ValueError('Symlink model source')
            shutil.copy2(p,sources/p.name)
            manifest['../sources/'+p.name]=hashlib.sha256((sources/p.name).read_bytes()).hexdigest()
    write(out/'source-manifest.json',manifest)
    return frozen,sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial',type=Path,default=ROOT/'app/modules/card_game/engine/ai/models')
    p.add_argument('--seconds',type=float)
    p.add_argument('--throughput-report',type=Path,help='Measured formal-budget stage estimates; required for a ten-character long run')
    p.add_argument('--ten',action='store_true',help='Five-team ten-character information search cycle')
    p.add_argument('--hard-deadline',type=float)
    p.add_argument('--seed-reservation',type=Path,help='Existing run-specific reservation exported from the shared ledger')
    p.add_argument('--seed-ledger',type=Path,default=ROOT/'artifacts/rl-seed-ledger.json')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--confirmation-seeds',type=int,default=512)
    p.add_argument('--adaptation-steps',type=int)
    p.add_argument('--research-warm-start',action='store_true')
    p.add_argument('--baseline-builds',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run',action='store_true',help='Explicitly authorize this bounded execution')
    p.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve()
    from app.modules.card_game.rl.build_acceptance import phase_deadlines,claim_seeds,check_source,canonical_build
    from app.modules.card_game.rl.league_schema import rule_hash,PRESETS,deck
    if a.worker or a.supervise:
        config=json.loads((out/'config.json').read_text())
        if time.time()>=config['deadlines']['shutdown']:raise TimeoutError('Original deadline expired')
        if a.worker:
            for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
            manifest=json.loads((out/'source-manifest.json').read_text())
            for name,digest in manifest.items():
                source=(ROOT/name).resolve()
                if not source.is_relative_to(out) or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                    raise ValueError('Frozen source/model identity mismatch')
            from app.modules.card_game.rl.full_cycle_experiment import run
            try:run(out,config)
            except Exception as e:
                write(out/'status.json',dict(phase='failed',workflow_complete=False,quality_approved=False,error=repr(e)));raise
            return
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        end=time.monotonic()+max(0,config['deadlines']['shutdown']-time.time())
        proc=subprocess.Popen([sys.executable,'-X','utf8',__file__,'--output',str(out),'--worker'],cwd=ROOT,start_new_session=os.name!='nt')
        guard=dict(phase='running',worker_pid=proc.pid,hard_deadline=config['deadlines']['shutdown']);write(out/'guard.json',guard)
        try:
            while proc.poll() is None:
                left=min(end-time.monotonic(),config['deadlines']['shutdown']-time.time())
                if left<=0:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=False)
                    else:os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=15);guard['phase']='hard_deadline';break
                try:proc.wait(timeout=min(30,left))
                except subprocess.TimeoutExpired:pass
            if guard['phase']=='running':guard['phase']='finished' if proc.returncode==0 else 'failed'
            guard.update(exit_code=proc.returncode,finished=time.time());write(out/'guard.json',guard)
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    if a.adaptation_steps is None:a.adaptation_steps=8 if a.ten else 32
    if not 1<=a.workers<=8 or not 32<=a.confirmation_seeds<=4096 or not 4<=a.adaptation_steps<=128 or a.adaptation_steps%4:
        p.error('Invalid bounded workload')
    if a.ten:
        from app.modules.card_game.rl.fixed_lineup import identity as rule_hash, KEYS as PRESETS
    now=time.time()
    if a.seconds is None and a.hard_deadline is None:
        from datetime import datetime,timedelta,timezone
        local=datetime.now(timezone(timedelta(hours=8)))
        stop=local.replace(hour=9,minute=0,second=0,microsecond=0)
        if stop<=local:stop+=timedelta(days=1)
        a.hard_deadline=stop.timestamp()
    seconds=min(a.seconds,a.hard_deadline-now) if a.seconds is not None and a.hard_deadline else a.seconds if a.seconds is not None else a.hard_deadline-now
    if a.ten:
        if not 600<=seconds<=86400:raise ValueError('Budget must be 10 minutes to 24 hours')
        fractions={'foundation':.25,'search':.50,'baseline_adaptation':.62,'candidate_adaptation':.74,'confirmation':.81,'audit':.85,'evaluation':.985,'shutdown':1.0}
        deadlines={k:now+seconds*v for k,v in fractions.items()}
    else:
        deadlines=phase_deadlines(now,seconds)
    baselines=json.loads(a.baseline_builds.read_text()) if a.baseline_builds else None
    if baselines is not None:
        if set(baselines)!=set(PRESETS):raise ValueError('All three baseline builds required')
        baselines={k:canonical_build(v) for k,v in baselines.items()}
        if not a.ten and any(baselines[k]['character_ids']!=deck(k)['character_ids'] for k in PRESETS):raise ValueError('Fixed lineup order changed')
    initial=a.initial.resolve();disposition=check_source(initial,research_warm_start=a.research_warm_start,baseline_builds=baselines)
    if a.ten and baselines is None:raise ValueError('Five explicit baseline builds required')
    if not a.research_warm_start and a.run:
        if a.ten:
            from app.modules.card_game.rl.fixed_lineup import FixedModel
            for key in PRESETS:FixedModel(initial,key)
        else:
            from app.modules.card_game.rl.league_policy import LeagueModel
            for key in PRESETS:LeagueModel(initial,key,require_approved=True)
    config=dict(runtime='ten' if a.ten else None,matrix_games=128,zhenhong_passive_reward=.2 if a.ten else 0,schema='paired_full_cycle_v1',rule_hash=rule_hash(),created=now,deadlines=deadlines,
        initial=str(initial),device=a.device,workers=a.workers,smoke=a.smoke,simulations=8 if a.smoke else 256,
        confirmation_seeds=32 if a.smoke else a.confirmation_seeds,adaptation_steps=4 if a.smoke else a.adaptation_steps,
        alpha=.05,minimum_gain=.02,audit_margin=.05,research_warm_start=a.research_warm_start,
        baseline_builds=baselines,automatic_serving_approval=False)
    if not a.run:
        print(json.dumps(dict(config=config,training_started=False,seed_reservation_pending=True,source_validation_pending=True),ensure_ascii=False));return
    if a.ten and not a.smoke:
        if a.throughput_report is None:raise ValueError('Measure the formal 256-search workload and provide --throughput-report before a long run')
        from app.modules.card_game.rl.training_capacity import validate_capacity
        capacity=json.loads(a.throughput_report.read_text())
        validate_capacity(capacity,config);config['capacity_measurement']=capacity
    if out.exists():raise ValueError('New output required; audit evidence cannot be reused')
    out.mkdir(parents=True)
    if a.seed_reservation:
        marker=a.seed_reservation.with_suffix('.claimed')
        fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(str(out))
    config['seeds']=json.loads(a.seed_reservation.read_text()) if a.seed_reservation else claim_seeds(a.seed_ledger,str(out))
    from app.modules.card_game.rl.build_acceptance import SEED_PHASES,SEED_WIDTH
    if set(config['seeds'])!=set(SEED_PHASES) or any(type(v)!=int or v<0 for v in config['seeds'].values()):raise ValueError('Invalid seed reservation')
    ordered=sorted(config['seeds'].values())
    if any(b-a<SEED_WIDTH for a,b in zip(ordered,ordered[1:])):raise ValueError('Overlapping seed namespaces')
    frozen,sources=freeze_source(out,initial,PRESETS);config['source_initial']=str(initial);config['initial']=str(sources)
    if disposition:write(sources/'disposition.json',disposition)
    write(out/'config.json',config)
    opts=dict(cwd=frozen,stdin=subprocess.DEVNULL,close_fds=True)
    if os.name=='nt':opts['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|0x01000000
    else:opts['start_new_session']=True
    with (out/'console.log').open('xb') as log:
        proc=subprocess.Popen([sys.executable,'-X','utf8',str(frozen/'scripts'/Path(__file__).name),'--output',str(out),'--supervise'],stdout=log,stderr=subprocess.STDOUT,**opts)
    for _ in range(100):
        if (out/'guard.json').exists():break
        if proc.poll() is not None:raise RuntimeError('Supervisor exited before acknowledgement')
        time.sleep(.1)
    if not (out/'guard.json').exists():raise RuntimeError('No supervisor acknowledgement')
    write(out/'launch.json',dict(supervisor_pid=proc.pid,**config));print(json.dumps(dict(pid=proc.pid,output=str(out),hard_deadline=deadlines['shutdown'])))

if __name__=='__main__':main()
