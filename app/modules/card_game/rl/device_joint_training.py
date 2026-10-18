"""Bounded real Gumbel32/H3 joint update using the existing seven-team learner.

This never promotes an auxiliary/warm-value update into policy learning.
Complete full-budget CUDA search games and independent postbatch proof must
exist before update_batch sees a row. Formal long-run readiness stays separate.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import math
import time


def run_joint_training_smoke(output, *, collector, pure_collector, source, train_jobs,
                             evaluation_jobs, deadline, policy_passes=4):
    from .batched_search.device_h3_collection import DeviceH3Collector, validate_h3_inputs
    from .batched_search.device_pure_collection import DevicePureCollector
    from .batched_search.soa_pure_jobs import adapt_pure_jobs
    from .batched_search.soa_training_jobs import policy_source_alias
    from .device_value_training import _check_jobs
    from .seven_round import write, update_batch
    from .seven_schedule import ROSTER
    from .episode_replay import EpisodePool, dump_episode
    from . import seven_runtime as rt
    if not isinstance(collector, DeviceH3Collector) or not isinstance(pure_collector, DevicePureCollector):
        raise TypeError('Live actual H3 and pure CUDA collectors required')
    if collector.executor is not pure_collector.executor:
        raise ValueError('Short training/evaluation must use the same frozen rules executor')
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=time.time():
        raise ValueError('Future absolute joint-smoke deadline required')
    if type(policy_passes) is not int or policy_passes not in (1,4,8):
        raise ValueError('Existing policy pass protocol required')
    if (not train_jobs or any(j.get('training') is not True or j.get('pure') is not False
                             or j.get('deadline')!=deadline for j in train_jobs)):
        raise ValueError('Explicit real search-learning jobs with original deadline required')
    _check_jobs(evaluation_jobs,deadline=deadline)
    if {j['seed'] for j in train_jobs} & {j['seed'] for j in evaluation_jobs}:
        raise ValueError('Joint training/frozen evaluation seed partitions overlap')
    train_jobs, evaluation_jobs = list(train_jobs), list(evaluation_jobs)
    source, output = Path(source).resolve(), Path(output)
    if any(Path(spec[0]).resolve()!=source for job in train_jobs+evaluation_jobs
           for spec in job['policies'].values()):
        raise ValueError('One exact original seven-team source bank required for this short run')
    models = {policy_source_alias(str(source),key):rt.model(source,key) for key in ROSTER}
    validate_h3_inputs(train_jobs,models); adapt_pure_jobs(evaluation_jobs,models)
    # IDs become episode filenames in the existing learner.
    import re
    if any(not isinstance(j.get('id'),str) or re.fullmatch('[A-Za-z0-9_-]+',j['id']) is None for j in train_jobs):
        raise ValueError('Safe exact training job filenames required')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,stage='gumbel_h3_sampling',formal_training_complete=False,
        quality_approved=False,formal_policy_updates=0,optimizer_updates={},complete_gumbel_games=0,
        deadline=deadline,policy_passes=policy_passes,algorithm='gumbel',simulations=32,gumbel_candidates=16,
        terminal_horizon=3,opponent_mode='rule',world_sampling='per_simulation',cpu_hot_rule_calls=0)
    def save():write(output/'report.json',report)
    save()
    try:
        import numpy as np
        import torch
        signature={key+ext:sha256((source/(key+ext)).read_bytes()).hexdigest()for key in ROSTER for ext in('.json','.npz')}
        games=collector.collect(train_jobs,models)
        report.update(sampling=deepcopy(collector.last_diagnostics),complete_gumbel_games=len(games))
        if not games or not all(g.get('complete') is True and g.get('learning_eligible') is True
                and g['searched']==g['decisions']==len(g['rows']) and g['rows'] for g in games):
            raise ValueError('Actual complete eligible Gumbel episodes required before policy update')
        save()
        selected=tuple(k for k in ROSTER if any(j['policies'][s][1]==k for j in train_jobs for s in j['train_sides']))
        nets,opts,pools,builds,origins,before={},{},{},{},{},{}
        for key in ROSTER:
            net,build,origin=rt.restore(source,key,str(collector.executor.device))
            nets[key],builds[key],origins[key]=net,build,origin
            before[key]={n:t.detach().clone() for n,t in net.state_dict().items()}
            opts[key]=torch.optim.Adam(net.parameters(),lr=.0003)
            pools[key]=EpisodePool(limit=128,max_row_reuse=8)
        report['stage']='joint_policy_wdl_update';report['update_stage_started']=True;save()
        seed=train_jobs[0]['seed']
        health,_=update_batch(nets,opts,pools,train_jobs,games,np.random.default_rng(seed),
            dict(learning_keys=selected,policy_passes=policy_passes),deadline,output/'updates')
        report['optimizer_updates']=health
        report['attempted_policy_updates']=sum(h['updates'] for h in health.values())
        save()
        if not all(h['finite'] and h['gradient'] and h['coverage']==1 and h['updates']>0 for h in health.values()):
            raise ValueError('Joint update did not prove finite policy gradients and full fresh-row coverage')
        differences={}
        for key,net in nets.items():
            changed=[n for n,tensor in net.state_dict().items()if not torch.equal(tensor,before[key][n])]
            differences[key]=changed
            if key in selected:
                if not any(n.startswith(('state_net.','cand_net.','score.'))for n in changed):
                    raise ValueError('Gumbel policy parameters did not actually change')
                if not any(n.startswith(('value_net.','wdl.'))for n in changed):
                    raise ValueError('Joint terminal-WDL parameters did not actually change')
            elif changed:raise ValueError('Frozen team parameters changed')
            rt.export(net,output/'models',key,builds[key],dict(origins[key],joint_h3_smoke=True))
            numeric=rt.model(output/'models',key)
            for n,value in net.state_dict().items():np.testing.assert_array_equal(value.detach().cpu().numpy(),numeric.weights[n])
            if key in selected:torch.save(dict(model=net.state_dict(),optimizer=opts[key].state_dict()),output/(key+'-checkpoint.pt'))
        report.update(optimizer_updates=health,parameter_changes=differences,
            formal_policy_updates=sum(h['updates']for h in health.values()),stage='frozen_evaluation')
        save()
        folder=output/'models'
        from .batched_search.soa_training_jobs import policy_source_alias
        bank={policy_source_alias(str(folder),key):rt.model(folder,key)for key in ROSTER}
        jobs=deepcopy(evaluation_jobs)
        for job in jobs:
            for side in ('a','b'):job['policies'][side]=(str(folder),job['policies'][side][1])
        frozen={p.name:sha256(p.read_bytes()).hexdigest()for p in folder.iterdir()}
        evaluated=pure_collector.collect(jobs,bank)
        for job,game in zip(jobs,evaluated):dump_episode(output/'evaluation'/(job['id']+'.json.gz'),job,game)
        if frozen!={p.name:sha256(p.read_bytes()).hexdigest()for p in folder.iterdir()}:
            raise ValueError('Frozen evaluation changed new weights')
        if signature!={key+ext:sha256((source/(key+ext)).read_bytes()).hexdigest()for key in ROSTER for ext in('.json','.npz')}:
            raise ValueError('Original source models changed')
        report.update(passed=True,stage='complete',evaluation=deepcopy(pure_collector.last_diagnostics),
            natural_evaluation_games=len(evaluated),initial_weights_unchanged=True,frozen_weights_unchanged=True)
        save();return report
    except BaseException as error:
        report.update(error=repr(error),last_sampling=deepcopy(collector.last_diagnostics),
            last_evaluation=deepcopy(pure_collector.last_diagnostics))
        save();raise


def prepare_joint_training_inputs(source, output, *, executor, starting_decks,
                                  auxiliary_seeds, main_seeds, auxiliary_collect_until,
                                  auxiliary_fit_until, value_until, hard_deadline,
                                  checkpoint_segment, rows=8):
    """Actual auxiliary calibration and current-policy WDL precheck for H3.

    Each rule segment is <=90 seconds. The caller supplies an outer watchdog
    and a lossless segment-checkpoint callback; interrupted data cannot become
    qualified scope. No phase is replaced by a smaller smoke calibration.
    """
    from .device_auxiliary_calibration import (auxiliary_phase_plans,
        create_device_auxiliary_phase_collector, fit_device_auxiliary_calibration)
    from .batched_search.device_pure_collection import DevicePureCollector
    from .seven_round import jobs_for, run_device_value_calibration, write
    from .seven_schedule import ROSTER
    from .seven_lineup import identity
    from . import seven_runtime as rt
    from .build_acceptance import canonical_build, SEED_PHASES
    from ..content.duel_v2 import validate_deck
    if not callable(checkpoint_segment):
        raise TypeError('Actual lossless auxiliary segment checkpoint callback required')
    times=(auxiliary_collect_until,auxiliary_fit_until,value_until,hard_deadline)
    if (any(type(t) not in (int,float) or not math.isfinite(t) for t in times)
            or not time.time()<auxiliary_collect_until<auxiliary_fit_until<value_until<hard_deadline):
        raise ValueError('Ordered explicit collection/fit/value/hard deadlines required')
    if (set(main_seeds)!=set(SEED_PHASES) or any(type(v) is not int or v<0 for v in main_seeds.values())):
        raise ValueError('Separate complete main-learning seed reservation required')
    source,output=Path(source).resolve(),Path(output)
    decks={k:validate_deck(deepcopy(starting_decks[k])) for k in ROSTER}
    policies={k:rt.model(source,k)for k in ROSTER}
    if any(canonical_build(policies[k].serving_deck)!=canonical_build(decks[k])for k in ROSTER):
        raise ValueError('Explicit actual seven-team starting builds differ from source models')
    plans=auxiliary_phase_plans(starting_decks=decks,seed_reservation=auxiliary_seeds,
                                deadline=auxiliary_fit_until)
    auxiliary_game_seeds={j['seed']for phase in plans.values()for j in phase}
    # Preserve the existing full-tower2/1/1 policy-value protocol, including the
    # reserved heldout offset, and audit its seeds before creating outputs.
    calibration_seed_sets=[]
    for base in (main_seeds['adaptation'],main_seeds['adaptation']+100,
                 main_seeds['selection'],main_seeds['selection']+1000):
        calibration_seed_sets.append({j['seed']for j in jobs_for(source,decks,base,value_until)})
    if any(a & b for i,a in enumerate(calibration_seed_sets) for b in calibration_seed_sets[i+1:]):
        raise ValueError('Current-policy training/selection/heldout seed partitions overlap')
    calibration_game_seeds=set().union(*calibration_seed_sets)
    if auxiliary_game_seeds & calibration_game_seeds:
        raise ValueError('Auxiliary and current-policy calibration seed partitions overlap')
    output.mkdir(parents=True,exist_ok=False)
    status=dict(ready=False,stage='auxiliary_collection',formal_policy_updates=0,
        phase_counts={p:len(j)for p,j in plans.items()},main_value_games=224,
        auxiliary_collect_until=auxiliary_collect_until,auxiliary_fit_until=auxiliary_fit_until,
        value_until=value_until,hard_deadline=hard_deadline)
    def save():write(output/'preparation.json',status)
    save()
    try:
        phase_collector=create_device_auxiliary_phase_collector(executor,starting_decks=decks,rows=rows,
                                                               segment_decisions=4)
        segments={}
        for phase,jobs in plans.items():
            def checkpoint(offset,segment,phase=phase):
                checkpoint_segment(phase,offset,segment)
                status['last_segment']=dict(phase=phase,offset=offset,steps=list(segment.steps),
                    complete=list(segment.complete),errors=list(segment.errors),replay_verified=segment.replay_verified)
                save()
            segments[phase]=phase_collector.collect(jobs,deadline=auxiliary_collect_until,on_segment=checkpoint)
        status['stage']='auxiliary_fit';save()
        _,scope=fit_device_auxiliary_calibration(source,output/'auxiliary',starting_decks=decks,
            seed_reservation=auxiliary_seeds,phase_segments=segments,deadline=auxiliary_fit_until)
        status['auxiliary_scope']=dict(models=scope['models'],passed=scope['passed'])
        status['stage']='current_policy_value_precheck';save()
        pure=DevicePureCollector(executor,seconds=90,proof_seconds=60,max_decisions=800,segment_decisions=16)
        config=dict(runtime='seven',device='cuda',hidden=policies[ROSTER[0]].hidden,
            recovery_source=str(source),starting_decks=decks,seeds=dict(main_seeds),rule_hash=identity(),train_until=value_until,
            value_full_tower=True,value_replicas=2,value_check_replicas=1,value_passes=16,
            value_metric_backend='cuda_batched')
        value=run_device_value_calibration(output/'policy-value',config,collector=pure)
        if value['value_precheck'].get('passed') is not True:
            raise ValueError('Current-policy heldout WDL precheck failed; H3 joint learning blocked')
        status.update(ready=True,stage='ready_for_h3_search',models=value['models'],
            value_precheck=value['value_precheck'],auxiliary_scope_directory=str(output/'auxiliary'))
        save();return status
    except BaseException as error:
        status.update(error=repr(error),ready=False)
        save();raise
