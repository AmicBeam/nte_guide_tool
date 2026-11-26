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
                             evaluation_jobs, deadline, policy_passes=4, resume=False,
                             checkpoint_dir=None, max_chunks=None, continuation=None,
                             evaluation_checkpoint_dir=None, evaluation_max_segments=None,
                             resume_authorization=None, pause_at_stage_boundary=None):
    """Persistent sampling -> once-only joint update -> independent frozen eval.

    Clean process pauses have no labels or optimizer updates. A crash after
    ``updating`` starts is deliberately not replayed: only a complete numerical
    export/optimizer/health stage marker permits evaluation in another process.
    Original jobs remain immutable; new budget windows require explicit receipts.
    """
    import json
    import re
    import numpy as np
    import torch
    from .batched_search.device_h3_collection import DeviceH3Collector, validate_h3_inputs, h3_profile
    from .batched_search.device_h3_segments import H3SegmentPaused
    from .batched_search.device_pure_collection import DevicePureCollector
    from .batched_search.device_pure_segments import PureSegmentPaused
    from .batched_search.pure_checkpoint import write_payload,read_payload
    from .batched_search.soa_pure_jobs import adapt_pure_jobs
    from .batched_search.soa_training_jobs import policy_source_alias
    from .device_value_training import _check_jobs
    from .seven_round import write, update_batch
    from .seven_schedule import ROSTER
    from .episode_replay import EpisodePool, dump_episode
    from . import seven_runtime as rt
    if not isinstance(collector,DeviceH3Collector)or not isinstance(pure_collector,DevicePureCollector):
        raise TypeError('Live actual H3 and pure CUDA collectors required')
    if collector.executor is not pure_collector.executor:
        raise ValueError('Short training/evaluation must use the same frozen rules executor')
    if type(deadline)not in(int,float)or not math.isfinite(deadline)or deadline<=0:
        raise ValueError('Finite positive joint-smoke deadline required')
    if type(resume)is not bool:raise TypeError('Explicit resume boolean required')
    if type(policy_passes)is not int or policy_passes not in(1,4,8):raise ValueError('Existing policy pass protocol required')
    if max_chunks is not None and(type(max_chunks)is not int or not 1<=max_chunks<=10000):
        raise ValueError('Positive bounded H3 chunk count required')
    if evaluation_max_segments is not None and(type(evaluation_max_segments)is not int or not 1<=evaluation_max_segments<=800):
        raise ValueError('Positive bounded evaluation segment count required')
    if (not train_jobs or any(j.get('training')is not True or j.get('pure')is not False
            or j.get('deadline')!=deadline for j in train_jobs)):
        raise ValueError('Explicit real search-learning jobs with original deadline required')
    _check_jobs(evaluation_jobs,deadline=deadline)
    if {j['seed']for j in train_jobs}&{j['seed']for j in evaluation_jobs}:
        raise ValueError('Joint training/frozen evaluation seed partitions overlap')
    train_jobs,evaluation_jobs=deepcopy(list(train_jobs)),deepcopy(list(evaluation_jobs))
    source,output=Path(source).resolve(),Path(output).resolve()
    if any(Path(spec[0]).resolve()!=source for job in train_jobs+evaluation_jobs for spec in job['policies'].values()):
        raise ValueError('One exact original seven-team source bank required for this short run')
    models={policy_source_alias(str(source),key):rt.model(source,key)for key in ROSTER}
    metadata, _, _ = validate_h3_inputs(train_jobs,models)
    profile = h3_profile(metadata)
    adapt_pure_jobs(evaluation_jobs,models)
    if any(not isinstance(j.get('id'),str)or re.fullmatch('[A-Za-z0-9_-]+',j['id'])is None for j in train_jobs):
        raise ValueError('Safe exact training job filenames required')
    if len({j['id']for j in train_jobs})!=len(train_jobs):raise ValueError('Unique physical training job IDs required')

    def normal(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,Path):return str(value)
        if isinstance(value,dict):return{k:normal(v)for k,v in value.items()}
        if isinstance(value,(list,tuple)):return[normal(v)for v in value]
        if type(value)in(type(None),bool,int,float,str):return value
        raise TypeError('Unsupported joint input binding type: '+type(value).__name__)
    def encoded(value):return json.dumps(normal(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    def identity(value):return sha256(encoded(value)).hexdigest()
    def bank_hash(folder):
        return {key+ext:sha256((Path(folder)/(key+ext)).read_bytes()).hexdigest()for key in ROSTER for ext in('.json','.npz')}
    def physical(jobs):return[{k:deepcopy(v)for k,v in j.items()if k!='deadline'}for j in jobs]
    def live():
        if time.time()>=deadline:raise TimeoutError('Original/new authorized joint-smoke budget exhausted')
    def phase_continuation(phase):
        if isinstance(continuation,dict)and('training'in continuation or'evaluation'in continuation):
            return continuation.get(phase)
        return continuation
    source_signature=bank_hash(source)
    executor=collector.executor
    binding=dict(device=str(executor.device),rule=getattr(executor,'rule_source_sha256',None),
        image=getattr(getattr(executor,'image',None),'identity',None),
        schema=getattr(getattr(executor,'schema',None),'identity',None))
    hydration=getattr(executor,'hydration_probe_binding',None)
    if hydration is not None:binding['hydration']=hydration.to_dict()
    module_profile=getattr(getattr(executor,'module',None),'compilation_profile',None)
    if isinstance(module_profile,dict):
        binding['native_units']={u['name']:u['source_sha256']for u in module_profile.get('compilation_units',[])}
    auxiliary={}
    for job in train_jobs:
        value=job.get('covered_value_source')
        if value is None:continue  # Real H3 validator already forbids a missing scope.
        folder=Path(value).resolve();scope_file=folder/'scope.json'
        scope=json.loads(scope_file.read_text())
        files={'scope.json':sha256(scope_file.read_bytes()).hexdigest()}
        for name,expected in scope['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder)or sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Qualified auxiliary raw evidence changed')
            files[name]=expected
        files.update({'models/'+name:digest for name,digest in bank_hash(folder/'models').items()})
        auxiliary[str(folder)]=files
    stored_input=json.loads((output/'input-binding.json').read_text())if resume and(output/'input-binding.json').is_file()else None
    persistent=any(v is not None for v in(checkpoint_dir,max_chunks,continuation,evaluation_checkpoint_dir,evaluation_max_segments))
    if stored_input is not None and stored_input['contract']['training_checkpoint_dir']is not None:persistent=True
    if pause_at_stage_boundary is not None and type(pause_at_stage_boundary)is not bool:
        raise TypeError('Optional stage-boundary pause must be boolean')
    pause_at_stage_boundary=persistent if pause_at_stage_boundary is None else pause_at_stage_boundary
    training_directory=Path(checkpoint_dir).resolve()if checkpoint_dir is not None else(output/'training-checkpoints'if persistent else None)
    evaluation_directory=Path(evaluation_checkpoint_dir).resolve()if evaluation_checkpoint_dir is not None else(output/'evaluation-checkpoints'if persistent else None)
    if stored_input is not None:
        if checkpoint_dir is None:
            value=stored_input['contract']['training_checkpoint_dir'];training_directory=None if value is None else Path(value)
        if evaluation_checkpoint_dir is None:
            value=stored_input['contract']['evaluation_checkpoint_dir'];evaluation_directory=None if value is None else Path(value)
    import importlib
    implementation_names=('seven_round','seven_runtime','outcome_runtime','episode_replay',
        'batched_search.device_h3_collection','batched_search.device_h3_segments',
        'batched_search.device_chunked_games','batched_search.device_search',
        'batched_search.device_statistics','batched_search.pure_checkpoint',
        'batched_search.device_pure_collection','batched_search.device_pure_segments',
        'batched_search.soa_training_results','batched_search.soa_pure_results')
    implementations={name:sha256(Path(importlib.import_module(__package__+'.'+name).__file__).read_bytes()).hexdigest()
                     for name in implementation_names}
    fixed=normal(dict(schema='device_joint_smoke_input_v1',source=str(source),source_files=source_signature,
        physical_train_jobs=physical(train_jobs),physical_evaluation_jobs=physical(evaluation_jobs),
        profile=profile,policy_passes=policy_passes,rules=binding,auxiliary_scopes=auxiliary,
        joint_driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_files=implementations,
        training_checkpoint_dir=None if training_directory is None else str(training_directory),
        evaluation_checkpoint_dir=None if evaluation_directory is None else str(evaluation_directory)))
    input_path=output/'input-binding.json';head=output/'stage.json'
    if resume:
        if not output.is_dir()or not input_path.is_file():raise ValueError('Actual previous joint stage input binding required')
        original=stored_input
        if original['contract']!=fixed:raise ValueError('Joint source/model/profile/physical jobs/auxiliary scope changed')
        marker=json.loads(head.read_text())
        revisions=sorted(output.glob('stage-revision-*.json'))
        if not revisions or revisions[-1].name!=f"stage-revision-{marker['revision']:08d}.json"or revisions[-1].read_bytes()!=head.read_bytes():
            raise ValueError('Joint stage rollback/tampering differs from immutable latest revision')
        if marker['input_sha256']!=sha256(input_path.read_bytes()).hexdigest():raise ValueError('Joint original input binding changed')
        report=json.loads((output/'report.json').read_text())
        old_deadline=marker['budget_windows'][-1]['deadline']
        if deadline!=old_deadline:
            if (deadline<=old_deadline or not isinstance(continuation,dict)or not continuation or not isinstance(resume_authorization,str)
                    or not resume_authorization.strip()):raise ValueError('Explicit deadline-only joint continuation receipt and authorization required')
            marker['budget_windows'].append(dict(deadline=deadline,authorization=resume_authorization.strip(),
                                                 continuation_sha256=identity(continuation)))
        if marker['stage']=='updating':raise RuntimeError('Uncommitted optimizer stage cannot be repeated or inferred successful')
        if marker['stage']=='failed':raise RuntimeError('Prior joint failure remains rejected')
        if sha256((output/'report.json').read_bytes()).hexdigest()!=marker['report_sha256']:
            raise ValueError('Committed joint stage report changed')
    else:
        live();output.mkdir(parents=True,exist_ok=False)
        write(input_path,dict(contract=fixed,original_train_jobs=normal(train_jobs),
                             original_evaluation_jobs=normal(evaluation_jobs),original_deadline=deadline))
        marker=dict(schema='device_joint_smoke_stage_v1',input_sha256=sha256(input_path.read_bytes()).hexdigest(),
            stage='sampling',revision=0,budget_windows=[dict(deadline=deadline,authorization='initial explicit joint smoke')])
        report=dict(passed=False,stage='gumbel_h3_sampling',formal_training_complete=False,
            quality_approved=False,formal_policy_updates=0,optimizer_updates={},complete_gumbel_games=0,
            deadline=deadline,policy_passes=policy_passes,algorithm='gumbel',simulations=32,gumbel_candidates=16,
            terminal_horizon=3,opponent_mode='rule',world_sampling='per_simulation',cpu_hot_rule_calls=0)
    cache_binding=dict(input_sha256=marker['input_sha256'],contract_sha256=identity(fixed))
    def save():write(output/'report.json',report)
    def commit(stage,**values):
        marker.update(values,stage=stage,revision=marker['revision']+1,
            report_sha256=sha256((output/'report.json').read_bytes()).hexdigest())
        path=output/f"stage-revision-{marker['revision']:08d}.json"
        if path.exists():raise ValueError('New immutable joint stage revision required')
        # Revision is complete before its atomic live pointer is published.
        import os
        with path.open('x',encoding='utf8')as stream:
            json.dump(marker,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        write(head,marker)
    def frozen_source():
        if bank_hash(source)!=source_signature:raise ValueError('Original source models changed')
    def check_updated():
        if bank_hash(output/'models')!=marker['updated_models']:raise ValueError('Committed joint numerical models changed')
        for name,digest in marker['optimizer_files'].items():
            if sha256((output/name).read_bytes()).hexdigest()!=digest:raise ValueError('Committed joint optimizer checkpoint changed')
    def cache(path,games,proof,kind,jobs):
        actual=write_payload(path,dict(games=games,proof=proof,jobs=deepcopy(jobs)),cache_binding,kind=kind)
        return dict(path=str(Path(actual).relative_to(output)),
            manifest_sha256=sha256((Path(actual)/'manifest.json').read_bytes()).hexdigest(),
            arrays_sha256=sha256((Path(actual)/'arrays.npz').read_bytes()).hexdigest())
    def cached(reference,kind):
        path=(output/reference['path']).resolve()
        if not path.is_relative_to(output):raise ValueError('Joint episode cache path escapes output')
        if (sha256((path/'manifest.json').read_bytes()).hexdigest()!=reference['manifest_sha256']
                or sha256((path/'arrays.npz').read_bytes()).hexdigest()!=reference['arrays_sha256']):
            raise ValueError('Committed joint episode/proof cache changed')
        return read_payload(path,cache_binding,kind=kind)
    def validate_games(games,jobs,learning):
        if not isinstance(games,list)or len(games)!=len(jobs):raise ValueError('Exact complete natural game cohort required')
        for job,game in zip(jobs,games):
            if (game.get('complete')is not True or game.get('id')!=job['id']or not game.get('episode_actions')
                    or game.get('decisions')!=len(game['episode_actions'])or game.get('winner')not in('a','b','draw')):
                raise ValueError('Actual complete natural episode identity required')
            if learning:
                if (game.get('learning_eligible')is not True or not game.get('rows')
                        or game.get('searched')!=game['decisions']or len(game['rows'])!=game['decisions']):
                    raise ValueError('Actual complete eligible Gumbel episodes required before policy update')
                for row in game['rows']:
                    expected=1 if len(row['c'])==1 else 32
                    if row.get('search_algorithm')!='gumbel'or row.get('search_budget')!=expected:
                        raise ValueError('Full original Gumbel root budget required')
            elif game.get('searched')!=0:raise ValueError('Independent frozen evaluation must use pure full games')
    def proof_valid(proof,learning):
        if not isinstance(proof,dict)or proof.get('complete')is not True:return False
        # Real collector constructors require CUDA. Explicit CPU mock fixtures
        # exercise orchestration only and can never construct that backend.
        fixture=str(executor.device)=='cpu'and proof.get('cpu_fixture')is True
        if proof.get('verified')is not True and not fixture:return False
        if learning and proof.get('independently_verified')is not True and not fixture:return False
        return True
    frozen_source()
    if marker['stage']=='complete':
        check_updated();cached(marker['training_cache'],'joint_training_episodes');cached(marker['evaluation_cache'],'joint_evaluation_episodes')
        if sha256((output/'report.json').read_bytes()).hexdigest()!=marker['report_sha256']:
            raise ValueError('Completed joint report changed')
        return report
    report['active_deadline']=deadline
    if not resume:save();commit('sampling')
    elif deadline!=old_deadline:save();commit(marker['stage'])
    try:
        if marker['stage']=='sampling':
            live();report.update(stage='gumbel_h3_sampling',planned_pause=False);save();commit('sampling')
            options={}
            if training_directory is not None:options['checkpoint_dir']=str(training_directory)
            if max_chunks is not None:options['max_chunks']=max_chunks
            if phase_continuation('training')is not None:options['continuation']=phase_continuation('training')
            try:games=collector.collect(train_jobs,models,**options)
            except H3SegmentPaused as paused:
                report.update(stage='gumbel_h3_sampling_paused',sampling=deepcopy(collector.last_diagnostics),
                              checkpoint=str(paused.checkpoint),passed=False)
                save();commit('sampling');return report
            validate_games(games,train_jobs,True)
            proof=deepcopy(collector.last_diagnostics)
            if not proof_valid(proof,True):raise ValueError('Independent complete H3 proof diagnostics required')
            frozen_source()
            reference=cache(output/'training-episodes',games,proof,'joint_training_episodes',train_jobs)
            write(output/'sampling-proof.json',normal(proof))
            report.update(sampling=proof,complete_gumbel_games=len(games),stage='joint_policy_wdl_update')
            save();commit('train_cached',training_cache=reference,
                sampling_proof_sha256=sha256((output/'sampling-proof.json').read_bytes()).hexdigest())
            if pause_at_stage_boundary:
                report.update(stage='joint_training_samples_cached',planned_pause=True);save();commit('train_cached');return report
        if marker['stage']=='train_cached':
            live();payload=cached(marker['training_cache'],'joint_training_episodes');games=payload['games']
            learning_jobs=payload['jobs']
            if normal(physical(learning_jobs))!=fixed['physical_train_jobs']:
                raise ValueError('Cached original training job metadata changed')
            validate_games(games,learning_jobs,True)
            if not proof_valid(payload['proof'],True):raise ValueError('Cached H3 proof is not independently verified')
            if (sha256((output/'sampling-proof.json').read_bytes()).hexdigest()!=marker['sampling_proof_sha256']
                    or normal(payload['proof'])!=json.loads((output/'sampling-proof.json').read_text())):
                raise ValueError('Independent H3 proof diagnostics changed')
            selected=tuple(k for k in ROSTER if any(j['policies'][s][1]==k for j in train_jobs for s in j['train_sides']))
            if (output/'models').exists()or any(output.glob('*-checkpoint.pt')):
                raise ValueError('Uncommitted numerical/optimizer exports cannot be overwritten or counted again')
            nets,opts,pools,builds,origins,before={},{},{},{},{},{}
            for key in ROSTER:
                net,build,origin=rt.restore(source,key,str(executor.device))
                nets[key],builds[key],origins[key]=net,build,origin
                before[key]={n:t.detach().clone()for n,t in net.state_dict().items()}
                opts[key]=torch.optim.Adam(net.parameters(),lr=.0003)
                pools[key]=EpisodePool(limit=128,max_row_reuse=8)
            frozen_source();report.update(stage='joint_policy_wdl_update',update_stage_started=True,planned_pause=False);save();commit('updating')
            health,_=update_batch(nets,opts,pools,learning_jobs,games,np.random.default_rng(learning_jobs[0]['seed']),
                dict(learning_keys=selected,policy_passes=policy_passes),deadline,output/'updates')
            report.update(optimizer_updates=normal(health),attempted_policy_updates=sum(h['updates']for h in health.values()));save()
            if set(health)!=set(selected)or not all(h['finite']and h['gradient']and h['coverage']==1 and h['updates']>0 for h in health.values()):
                raise ValueError('Joint update did not prove finite policy gradients and full fresh-row coverage')
            differences={};optimizer_files={}
            for key,net in nets.items():
                changed=[n for n,tensor in net.state_dict().items()if not torch.equal(tensor,before[key][n])]
                differences[key]=changed
                if key in selected:
                    if not any(n.startswith(('state_net.','cand_net.','score.'))for n in changed):raise ValueError('Gumbel policy parameters did not actually change')
                    if not any(n.startswith(('value_net.','wdl.'))for n in changed):raise ValueError('Joint terminal-WDL parameters did not actually change')
                elif changed:raise ValueError('Frozen team parameters changed')
                rt.export(net,output/'models',key,builds[key],dict(origins[key],joint_h3_smoke=True))
                numeric=rt.model(output/'models',key)
                for name,value in net.state_dict().items():np.testing.assert_array_equal(value.detach().cpu().numpy(),numeric.weights[name])
                if key in selected:
                    path=output/(key+'-checkpoint.pt');torch.save(dict(model=net.state_dict(),optimizer=opts[key].state_dict()),path)
                    optimizer_files[path.name]=sha256(path.read_bytes()).hexdigest()
            frozen_source()
            report.update(optimizer_updates=health,parameter_changes=differences,
                formal_policy_updates=sum(h['updates']for h in health.values()),stage='frozen_evaluation')
            save();commit('evaluation',updated_models=bank_hash(output/'models'),optimizer_files=optimizer_files,
                optimizer_updates=normal(health),parameter_changes=differences,formal_policy_updates=report['formal_policy_updates'])
            if pause_at_stage_boundary:
                report.update(stage='joint_update_committed',planned_pause=True);save();commit('evaluation');return report
        if marker['stage']=='evaluation':
            live();frozen_source();check_updated();folder=output/'models'
            report.update(stage='frozen_evaluation',planned_pause=False);save();commit('evaluation')
            bank={policy_source_alias(str(folder),key):rt.model(folder,key)for key in ROSTER}
            jobs=deepcopy(evaluation_jobs)
            for job in jobs:
                for side in('a','b'):job['policies'][side]=(str(folder),job['policies'][side][1])
            options={}
            if evaluation_directory is not None:options['checkpoint_dir']=str(evaluation_directory)
            if evaluation_max_segments is not None:options['max_segments']=evaluation_max_segments
            if phase_continuation('evaluation')is not None:options['continuation']=phase_continuation('evaluation')
            try:evaluated=pure_collector.collect(jobs,bank,**options)
            except PureSegmentPaused as paused:
                check_updated();frozen_source()
                report.update(stage='frozen_evaluation_paused',evaluation_checkpoint=str(paused.checkpoint),
                              evaluation=deepcopy(pure_collector.last_diagnostics),passed=False)
                save();commit('evaluation');return report
            validate_games(evaluated,jobs,False);check_updated();frozen_source()
            proof=deepcopy(pure_collector.last_diagnostics)
            if not proof_valid(proof,False):raise ValueError('Actual complete frozen evaluation diagnostics required')
            reference=cache(output/'evaluation-episodes',evaluated,proof,'joint_evaluation_episodes',jobs)
            report.update(evaluation=proof,natural_evaluation_games=len(evaluated));save()
            commit('eval_cached',evaluation_cache=reference)
            if pause_at_stage_boundary:
                report.update(stage='frozen_evaluation_cached',planned_pause=True);save();commit('eval_cached');return report
        if marker['stage']=='eval_cached':
            check_updated();frozen_source();payload=cached(marker['evaluation_cache'],'joint_evaluation_episodes')
            evaluated=payload['games'];evaluated_jobs=payload['jobs']
            expected_jobs=deepcopy(evaluation_jobs)
            for job in expected_jobs:
                for side in('a','b'):job['policies'][side]=(str(output/'models'),job['policies'][side][1])
            if normal(physical(evaluated_jobs))!=normal(physical(expected_jobs)):
                raise ValueError('Cached actual frozen-evaluation job metadata changed')
            validate_games(evaluated,evaluated_jobs,False)
            for job,game in zip(evaluated_jobs,evaluated):dump_episode(output/'evaluation'/(job['id']+'.json.gz'),job,game)
            report.update(passed=True,stage='complete',planned_pause=False,evaluation=payload['proof'],natural_evaluation_games=len(evaluated),
                          initial_weights_unchanged=True,frozen_weights_unchanged=True)
            save();commit('complete')
            return report
        raise RuntimeError('Unknown persisted joint stage')
    except BaseException as error:
        report.update(passed=False,error=repr(error),last_sampling=deepcopy(collector.last_diagnostics),
                      last_evaluation=deepcopy(pure_collector.last_diagnostics))
        save()
        # An interrupted update never receives an evaluation marker and cannot
        # be replayed. Other real failures are equally sticky, unlike clean pauses.
        if marker['stage']!='updating':commit('failed',failure=repr(error))
        raise


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
