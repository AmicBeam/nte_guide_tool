#!/usr/bin/env python3
"""Bounded neural policy initialization for cold-start R/M, before Gumbel readiness."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--expert-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-reservation',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--workers',type=int,default=12)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--hard-deadline',required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    if not 300<=a.seconds<=1800 or not 1<=a.workers<=16:p.error('Bounded initialization required')
    if not a.run:print(json.dumps(dict(dry_run=True,supervised_initialization=True,epochs=[4,8,16],formal_learning=False)));return
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch,numpy as np
    from datetime import datetime
    from app.modules.card_game.rl import rule_foundation as rf,recovery_runtime as rt,recovery_round as rr
    from app.modules.card_game.rl.cross_schedule import ROSTER
    from app.modules.card_game.rl.offline_sources import snapshot
    if a.output.exists():raise ValueError('New output required')
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback')
    torch.set_num_threads(1);hard=datetime.fromisoformat(a.hard_deadline)
    if hard.tzinfo is None:raise ValueError('Timezone-aware hard cutoff required')
    deadline=min(time.time()+a.seconds,hard.timestamp()-900)
    if deadline-time.time()<300:raise ValueError('Insufficient in-budget initialization window')
    seeds=json.loads(a.seed_reservation.read_text());signature=rr.versions(a.source)
    jobs=rf.training_jobs(a.expert_source,deadline);held_seeds=rf.split_training_seeds(jobs,seeds['selection'])
    from hashlib import sha256
    initialization_code=sha256(Path(__file__).read_bytes()+Path(rf.__file__).read_bytes()).hexdigest()
    rr.write(a.output/'config.json',dict(source_models=signature,expert_source=str(a.expert_source),seeds=seeds,
        initialization_code_sha256=initialization_code,expert_scope_sha256=sha256((a.expert_source/'scope.json').read_bytes()).hexdigest(),
        epochs=[4,8,16],learning_rate=.0003,weight_decay=.001,deadline=deadline,
        training_internal_validation_seeds=sorted(held_seeds),supervised_initialization=True,
        formal_gumbel_updates=0,rewards=0,rule_rank_ties_uniform=True,original_heldout_labels_used=False))
    snapshot(ROOT,a.output/'source')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        games=list(pool.map(rf.replay_training_job,jobs))
    import gzip
    for i,game in enumerate(games):
        with gzip.open(a.output/f'grounded-training-{i}.json.gz','wt',encoding='utf-8') as f:
            json.dump(game,f,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)
    metrics={};counts={}
    for key in ROSTER:
        net,build,origin=rt.restore(a.source,key,a.device)
        if key not in ('zhenhong','murk'):
            rt.export(net,a.output/'models',key,build,origin);continue
        train=[r for g in games if g['seed'] not in held_seeds for r in g['rows'] if r['key']==key]
        valid=[r for g in games if g['seed'] in held_seeds for r in g['rows'] if r['key']==key]
        before=rf.imitation_metrics(net,valid,a.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001)
        rng=np.random.default_rng(seeds['adaptation']);candidates=[];updates=0
        for epoch in range(1,17):
            for start in range(0,len(train),128):
                if start==0:order=rng.permutation(len(train))
                if time.time()>=deadline:raise TimeoutError('Initialization optimization deadline')
                stats=rt.update(net,opt,[train[int(i)] for i in order[start:start+128]]);updates+=1
            if epoch in (4,8,16):
                folder=a.output/f'epoch-{epoch}'/'models'
                rt.export(net,folder,key,build,dict(origin,supervised_initialization=True))
                metric=rf.imitation_metrics(net,valid,a.device);candidates.append((metric['kl'],epoch,metric))
        selected=min(candidates,key=lambda x:x[0])
        selected_updates=((len(train)+127)//128)*selected[1]
        net,_,_=rt.restore(a.output/f'epoch-{selected[1]}'/'models',key,a.device)
        rt.export(net,a.output/'models',key,build,dict(origin,
            initialization='learned_public_rule_policy_foundation',source=str(a.source),
            teacher='existing_public_rule_ranking',epochs=selected[1],supervised_updates=selected_updates,
            total_optimizer_steps=updates,selected_optimizer_steps=selected_updates,
            initialization_code_sha256=initialization_code,
            internal_validation_only=True,formal_gumbel_updates=0,optimizer_reset=True))
        metrics[key]=dict(before=before,after=selected[2],selected_epoch=selected[1],
                         train=rf.imitation_metrics(net,train,a.device))
        counts[key]=dict(training_rows=len(train),validation_rows=len(valid),updates=updates,
                        total_optimizer_steps=updates,selected_optimizer_steps=selected_updates)
        print(key,json.dumps(metrics[key]),flush=True)
    if rr.versions(a.source)!=signature:raise ValueError('Initialization altered source arrays')
    for key in ROSTER[:3]:
        for name,array in rt.model(a.source,key).weights.items():
            np.testing.assert_array_equal(array,rt.model(a.output/'models',key).weights[name])
    rr.write(a.output/'result.json',dict(complete=True,models=rr.versions(a.output/'models'),metrics=metrics,
        counts=counts,source_models_unchanged=True,three_legacy_policies_unchanged=True,
        original_heldout_labels_used=False,supervised_initialization=True,quality_approved=False,
        formal_readiness=False,website_approval=False))


if __name__=='__main__':main()
