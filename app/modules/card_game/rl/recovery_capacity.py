"""Isolation fit for complete TRAINING roots; never a strength confirmation.

Paired first/second games stay in one seed group. The source network is
frozen. Only a disposable copy fits the selected training groups.
"""
from pathlib import Path
import time
import numpy as np
from .cross_schedule import ROSTER


def label_metrics(net, rows, device):
    import torch
    from .league_learning import tensors
    if not rows: raise ValueError('Empty isolation partition')
    values=[]; correct=0; confident=0
    with torch.no_grad():
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            x,c,mask=tensors([(r['x'],r['c']) for r in batch],device)
            logits,_=net(x,c,mask)
            logp=logits.log_softmax(-1).cpu().numpy()
            for index,row in enumerate(batch):
                target=np.asarray(row['pi'],dtype=np.float64);positive=target>0
                values.append(float(np.sum(target[positive]*(np.log(target[positive])-logp[index,:len(target)][positive]))))
                ordered=np.sort(target)
                if len(ordered)>1 and ordered[-1]-ordered[-2]>=.05:
                    confident+=1
                    correct+=int(logp[index,:len(target)].argmax()==target.argmax())
    return dict(kl=max(0.,float(np.mean(values))),confident_rows=confident,
                confident_accuracy=correct/confident if confident else None,rows=len(rows))


def fit(models,jobs,games,config,deadline,*,steps=200,held_jobs=None,held_games=None):
    import torch
    from . import recovery_runtime as rt
    from .recovery_round import versions
    if type(steps) is not int or not 2<=steps<=400: raise ValueError('Bounded isolation steps required')
    signature=versions(models);result={}
    for key in ROSTER:
        groups={}
        for job,game in zip(jobs,games):
            if not game['complete'] or not job['training']:raise ValueError('Training complete games required')
            rows=[r for r in game['rows'] if r['key']==key and r['side'] in job['train_sides']]
            if rows:groups.setdefault(job['seed'],[]).extend(rows)
        seeds=sorted(groups)
        if len(seeds)<3:raise ValueError('At least three independent paired seed groups required')
        held_seeds=seeds[-2:];train_seeds=seeds[:-2]
        if held_jobs is not None:
            held_groups={}
            for job,game in zip(held_jobs,held_games):
                if not game['complete']:raise ValueError('Complete heldout games required')
                rows=[r for r in game['rows'] if r['key']==key]
                if rows:held_groups.setdefault(job['seed'],[]).extend(rows)
            if set(groups)&set(held_groups):raise ValueError('Heldout paired seeds overlap')
            train_seeds=seeds;held_seeds=sorted(held_groups)
            if len(held_seeds)!=len(train_seeds):raise ValueError('Balanced heldout coverage required')
            # Both partitions must include each identical unordered matchup,
            # not systematically leave mirror/new opponents out of fitting.
            def opponents(jobs):
                return sorted((min(j['left'],j['right']),max(j['left'],j['right']),j['first'])
                              for j in jobs if key in (j['left'],j['right']))
            if opponents(jobs)!=opponents(held_jobs):raise ValueError('Heldout opponent/initiative coverage differs')
        rng=np.random.default_rng(config['seeds']['probes'])
        train=[r for seed in train_seeds for r in groups[seed]]
        held=[r for seed in held_seeds for r in (held_groups if held_jobs is not None else groups)[seed]]
        if len(train)>128:train=[train[int(i)] for i in rng.choice(len(train),128,replace=False)]
        if len(held)>256:held=[held[int(i)] for i in rng.choice(len(held),256,replace=False)]
        net,_,_=rt.restore(models,key,config['device']);before=label_metrics(net,train,config['device'])
        opt=torch.optim.Adam(net.parameters(),lr=.003);gradients={k:0. for k in ('state_net','cand_net','score')}
        completed=0
        for _ in range(steps):
            if time.time()>=deadline:break
            stats=rt.update(net,opt,train)
            completed+=1
            for prefix in gradients:gradients[prefix]+=stats['policy_gradient_norms'].get(prefix,0.)
        after=label_metrics(net,train,config['device']);validation=label_metrics(net,held,config['device'])
        train_passed=after['kl']<.025 and (after['confident_rows']>=8 and after['confident_accuracy']>=.95)
        held_passed=validation['kl']<.15 and (validation['confident_rows']>=8 and validation['confident_accuracy']>=.75)
        result[key]=dict(train_before=before,train_after=after,heldout=validation,
            train_seed_groups=train_seeds,heldout_seed_groups=held_seeds,steps=completed,
            gradient_sums=gradients,train_passed=train_passed,heldout_passed=held_passed,
            passed=train_passed and held_passed and completed==steps and all(np.isfinite(v) and v>0 for v in gradients.values()))
    if versions(models)!=signature:raise ValueError('Isolation fit changed frozen source')
    return dict(schema='five_recovery_isolation_fit_v1',source_models=signature,device=config['device'],
                complete=all(r['steps']==steps for r in result.values()),
                passed=all(r['passed'] for r in result.values()),metrics=result,
                strength_approved=False,disposable_networks=True,training_heldout_not_independent_strength=True,
                learning_rate=.003,planned_steps=steps,
                partition='balanced_new_paired_seeds' if held_jobs is not None else 'unseen_opponent_seed_groups')
