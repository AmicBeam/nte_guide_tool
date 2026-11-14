"""Once-only joint update of already verified complete hybrid Gumbel episodes.

Collection, independent value calibration and strength acceptance are separate.
A RUNNING/FAILED update is never replayed. COMPLETE saves actual seven-team
weights, Adam, replay usage and RNG and grants no serving or quality approval.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import time

import numpy as np

from . import seven_round as rr, seven_runtime as rt
from .seven_schedule import ROSTER
from .batched_search.training_checkpoint import digest

SCHEMA = 'hybrid_joint_update_v1'


def _read(path):
    return json.loads(Path(path).read_text())


def _files(output):
    return {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob('*') if p.is_file() and p.name != 'joint-status.json'}


def _completed(output):
    output = Path(output)
    receipt = _read(output/'joint-status.json')
    if receipt.get('schema') != SCHEMA or receipt.get('status') != 'COMPLETE':
        raise ValueError('Uncertain RUNNING/FAILED joint stage cannot be replayed or restored')
    if receipt.get('quality_approved') is not False or receipt.get('automatic_serving_approval') is not False:
        raise ValueError('Joint update cannot grant quality or serving approval')
    mandatory = {'recovery/pointer.json', 'recovery/1/pools.json'}
    mandatory |= {f'models/{key}{ext}' for key in ROSTER for ext in ('.json','.npz')}
    mandatory |= {f'recovery/1/{key}.pt' for key in ROSTER}
    if not mandatory <= set(receipt.get('files',{})):
        raise ValueError('Complete numeric and optimizer state required')
    for name, sha in receipt['files'].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Completed joint update file integrity mismatch')
    return receipt


def load_joint_state(output, *, device='cpu'):
    """Restore a completed stage without performing another optimizer update."""
    import torch
    from .seven_checkpoint import restore
    from .episode_replay import EpisodePool
    output = Path(output)
    receipt = _completed(output)
    if receipt['config'].get('policy_layout') == 'phase_isolated_v1':
        from .phase_hybrid import load_joint_state as load
        return load(output, device=device)
    pointer = _read(output/'recovery/pointer.json')
    if (not Path(pointer['folder']).resolve().is_relative_to(output.resolve())
            or Path(pointer['models']).resolve() != Path(receipt['models']).resolve()
            or not Path(receipt['models']).resolve().is_relative_to(output.resolve())):
        raise ValueError('Joint recovery files must belong to their stage')
    nets, opts = {}, {}
    saved = None
    for key in ROSTER:
        nets[key], opts[key], data = restore(Path(pointer['folder'])/(key+'.pt'),
            directory=receipt['models'], key=key, device=device)
        if data['metadata']['config'] != receipt['config']:
            raise ValueError('Joint checkpoint configuration changed')
        if saved is not None and data['rng_state'] != saved['rng_state']:
            raise ValueError('Joint checkpoint RNG differs across teams')
        saved = data
    pools = {key: EpisodePool.from_state(state) for key, state in
             _read(Path(pointer['folder'])/'pools.json').items()}
    if set(pools) != set(ROSTER):
        raise ValueError('All seven replay pools required')
    for pool in pools.values():
        for episode in pool.episodes:
            for row in episode['rows']:
                for field in ('x','c','pi','visit_pi'):
                    if field in row:
                        row[field] = np.asarray(row[field], dtype=np.float32)
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['rng_state']
    torch.set_rng_state(saved['torch_rng'])
    if str(device).startswith('cuda') and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return nets, opts, pools, rng, receipt


def _validate_policy_row(row, step, trace, job):
    # A forced action has no competing root to allocate sequential halving to.
    # Traversal's existing contract records its one actual simulation.
    budget = 1 if len(row.get('c',())) == 1 else 32
    q_scale = job.get('q_scale', 'natural_wdl')
    if (row.get('step') != step or row.get('side') != trace['side']
            or row.get('search_budget') != budget or row.get('search_algorithm') != 'gumbel'
            or row.get('teacher_mode') != 'covered_value'
            or q_scale not in ('natural_wdl', 'mctx_default')
            or row.get('q_scale') != q_scale or row.get('z') not in (-1,0,1)
            or row.get('key') != job['policies'][row['side']][1]):
        raise ValueError('Complete per-decision Gumbel label contract required')


def _validate(source, jobs, games, config, *, model_loader=None):
    learners = set(rr.learning_keys(config))
    if config.get('policy_passes') not in (1,4,8):
        raise ValueError('policy_passes must be 1, 4 or 8')
    if type(config.get('max_row_reuse',8)) is not int or config.get('max_row_reuse',8) < 1:
        raise ValueError('Positive bounded max_row_reuse required')
    q_scale = config.get('q_scale')
    if (config.get('teacher_mode') != 'covered_value' or type(q_scale) is not str
            or q_scale not in ('natural_wdl', 'mctx_default')):
        raise ValueError('Qualified H3 covered value Gumbel contract required')
    if not jobs or len(jobs) != len(games) or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Distinct complete job/game batch required')
    from .seven_covered_value import load_scope
    scopes = {}
    counts = {k: 0 for k in learners}
    for job, game in zip(jobs, games):
        if (job.get('training') is not True or job.get('pure') is True
                or job.get('simulations',32) != 32 or job.get('gumbel_candidates',16) != 16
                or job.get('terminal_horizon',3) != 3 or job.get('max_depth',10) != 10
                or job.get('teacher_mode') != 'covered_value' or job.get('q_scale') != q_scale
                or job.get('zhenhong_passive_reward') not in (None,0) or job.get('setup_rewards')):
            raise ValueError('Complete Gumbel32/cap16/H3/depth10 terminal-WDL jobs required')
        policies = {s: (model_loader or rt.model)(*spec) for s,spec in job['policies'].items()}
        if not rt.value_precheck_ok(job.get('value_precheck'), policies):
            raise ValueError('Frozen main-model WDL precheck required')
        scope = job.get('covered_value_source')
        if not scope:
            raise ValueError('Qualified auxiliary load_scope required')
        if scope not in scopes:
            scopes[scope] = load_scope(scope)[1]
        cert = game.get('replay_verification') or {}
        if (game.get('id') != job['id'] or game.get('complete') is not True
                or game.get('learning_eligible') is not True
                or game.get('rule_backend') != 'resident_cpu'
                or game.get('searched') != game.get('decisions')
                or game.get('decisions') != len(game.get('episode_actions',()))
                or len(game.get('rows',())) != game.get('decisions')
                or cert.get('passed') is not True or cert.get('id') != job['id']
                or cert.get('actions') != game['decisions']
                or not game.get('final_state_sha256') or cert.get('final_state_sha256') != game['final_state_sha256']):
            raise ValueError('Naturally complete replay-verified eligible hybrid game required')
        if game.get('model_versions') != {s:p.version for s,p in policies.items()}:
            raise ValueError('Collected frozen model versions changed')
        if len(game.get('value_rows',())) != 2*game['decisions']:
            raise ValueError('Both observer value prefixes required')
        for side in job['train_sides']:
            key = job['policies'][side][1]
            if key not in learners:
                raise ValueError('Frozen opponent cannot contribute learning rows')
            if key not in source or source[key].version != policies[side].version:
                raise ValueError('Learner source differs from collected source')
        for step, row in enumerate(game['rows']):
            trace = game['episode_actions'][step]
            _validate_policy_row(row,step,trace,job)
            if row['side'] in job['train_sides']:
                counts[row['key']] += 1
        expected_values = {(step,side) for step in range(game['decisions']) for side in ('a','b')}
        if {(r.get('step'),r.get('side')) for r in game['value_rows']} != expected_values:
            raise ValueError('Complete unique observer decisions required')
        for row in game['rows'] + game['value_rows']:
            side = row['side']
            target = 0 if game.get('winner') == 'draw' else (1 if game.get('winner') == side else -1)
            if (game.get('winner') not in ('a','b','draw') or row.get('z') != target
                    or row.get('reward_to_go') is not None or row.get('setup_reward',0)
                    or row.get('key') != job['policies'][side][1]
                    or row.get('value_actor') != int(game['episode_actions'][row['step']]['side'] == side)
                    or not np.isfinite(np.asarray(row['x'])).all()):
                raise ValueError('Unshaped terminal WDL and exact observer identity required')
    if not all(counts.values()):
        raise ValueError('Every learner requires fresh complete policy rows')
    return scopes


def _rebind_value_source(nets, opts, previous, source_bank, report):
    """Carry policy Adam across the standard, qualified full-value recalibration."""
    import torch
    prior = {k:rt.model(previous['models'],k) for k in ROSTER}
    current = {k:rt.model(source_bank,k) for k in ROSTER}
    changed_bank = (Path(previous['models']).resolve() != Path(source_bank).resolve()
                    or any(prior[k].version != current[k].version for k in ROSTER))
    result = dict(previous_joint_bank=str(Path(previous['models']).resolve()),
        current_source_bank=str(Path(source_bank).resolve()),
        previous_versions={k:p.version for k,p in prior.items()},
        current_versions={k:p.version for k,p in current.items()},
        value_recalibrated=changed_bank,policy_optimizer_retained=True,
        selective_value_adam_reset=False,reset_parameters={})
    # Check every array before writing any parameter. Norm masks and all legacy
    # tensors remain bit-exact, including buffers within the value tower.
    allowed = {}
    for key in ROSTER:
        allowed[key] = {name for name,_ in nets[key].named_parameters()
                        if name.startswith(('wdl.','value_net.'))}
        old, new = prior[key].weights, current[key].weights
        if set(old) != set(new) or prior[key].serving_deck != current[key].serving_deck:
            raise ValueError('Recalibration changed model schema or fixed build')
        if any(not np.array_equal(old[n],new[n]) for n in old if n not in allowed[key]):
            raise ValueError('Value recalibration changed policy, legacy tensors or normalization masks')
    if changed_bank:
        if not rt.value_precheck_ok(report,current):
            raise ValueError('Qualified main value report bound to recalibrated source required')
        with torch.no_grad():
            for key in ROSTER:
                state = nets[key].state_dict()
                for name in allowed[key]:
                    state[name].copy_(torch.as_tensor(current[key].weights[name].copy(),device=state[name].device))
                # Match _run_learning_core after full-tower warm_value: preserve
                # policy Adam, but calibrated heads must not reuse old moments.
                for name,parameter in nets[key].named_parameters():
                    if name in allowed[key]:
                        opts[key].state.pop(parameter,None)
                result['reset_parameters'][key] = sorted(allowed[key])
        result['selective_value_adam_reset'] = True
        result['main_value_report_sha256'] = digest(report)
    return result


def _execute_update(source_bank, jobs, games, output, config, *, deadline, prior_stage=None,
                    main_value_report=None):
    """Optimizer/export implementation; caller must first validate collection gates."""
    import torch
    from .episode_replay import EpisodePool
    output = Path(output)
    device = config.get('device','cuda')
    if prior_stage is None:
        torch.manual_seed(config.get('seed',0))
        nets = {k:rt.restore(source_bank,k,device)[0] for k in ROSTER}
        opts = {k:torch.optim.Adam(nets[k].parameters(),lr=config.get('learning_rate',.0003)) for k in ROSTER}
        pools = {k:EpisodePool(limit=config.get('episode_pool_limit',128),
                  max_row_reuse=config.get('max_row_reuse',8)) for k in ROSTER}
        rng = np.random.default_rng(config.get('seed',0))
        source_transition = dict(value_recalibrated=False,previous_joint_bank=None,
                                 current_source_bank=str(Path(source_bank).resolve()))
    else:
        nets,opts,pools,rng,previous = load_joint_state(prior_stage,device=device)
        if (previous['config']['hard_deadline'] != config['hard_deadline']
                or previous['config']['train_until'] != config['train_until']
                or previous['config']['policy_passes'] != config['policy_passes']
                or rr.learning_keys(previous['config']) != rr.learning_keys(config)
                or previous['config'].get('learning_rate',.0003) != config.get('learning_rate',.0003)
                or previous['config'].get('seed',0) != config.get('seed',0)
                or previous['config'].get('policy_batch_size',128) != config.get('policy_batch_size',128)
                or previous['config'].get('value_batch_size',128) != config.get('value_batch_size',128)
                or previous['config'].get('episode_pool_limit',128) != config.get('episode_pool_limit',128)
                or any(p.max_row_reuse != config.get('max_row_reuse',8) for p in pools.values())):
            raise ValueError('Prior joint state source, deadline or reuse contract changed')
        source_transition = _rebind_value_source(nets,opts,previous,source_bank,main_value_report)
    before = {k:rt.model(source_bank,k).version for k in ROSTER}
    health, rows = rr.update_batch(nets,opts,pools,jobs,games,rng,config,deadline,output)
    learners = rr.learning_keys(config)
    for key in learners:
        value_gradients = {prefix: math.sqrt(sum(float(p.grad.detach().norm())**2
            for name,p in nets[key].named_parameters() if name.startswith(prefix) and p.grad is not None))
            for prefix in ('value_net','wdl')}
        health[key]['value_gradient_norms'] = value_gradients
        if any(not math.isfinite(n) or n <= 0 for n in value_gradients.values()):
            raise ValueError('Positive finite terminal-value gradients required for every learner')
        if health[key]['policy_rows'] != len(rows[key]):
            raise ValueError('Fresh policy rows were dropped from the replay pool')
    if set(health) != set(learners) or any(not h['finite'] or not h['gradient'] or
            h['coverage'] != 1 or h['policy_rows'] < 1 or h['value_rows'] < 1 for h in health.values()):
        raise ValueError('Every selected learner requires finite positive-gradient full fresh coverage')
    if time.time() >= deadline:
        raise TimeoutError('Joint update export deadline')
    models = output/'models'
    for key in ROSTER:
        prior = rt.model(source_bank,key)
        if any(not torch.isfinite(t).all() for t in nets[key].state_dict().values()):
            raise ValueError('Nonfinite joint model tensor')
        changed = [name for name,tensor in nets[key].state_dict().items()
                   if not torch.equal(tensor.detach().cpu(),torch.as_tensor(prior.weights[name].copy()))]
        if key in learners:
            health[key]['changed_policy_arrays'] = [n for n in changed if not n.startswith(('value_net.','wdl.'))]
            health[key]['changed_value_arrays'] = [n for n in changed if n.startswith(('value_net.','wdl.'))]
            if not health[key]['changed_policy_arrays'] or not health[key]['changed_value_arrays']:
                raise ValueError('Each learner must change both policy and terminal-value arrays')
        elif changed:
            raise ValueError('Frozen model arrays changed during scoped joint update')
        rt.export(nets[key],models,key,prior.serving_deck,dict(prior.manifest['origin'],
            hybrid_joint_update=key in learners,frozen_during_scoped_update=key not in learners,
            quality_approved=False,automatic_serving_approval=False))
    numeric = rr.numeric_check({k:nets[k] for k in learners},models,rows,device)
    after = {k:rt.model(models,k).version for k in ROSTER}
    if any(before[k] == after[k] for k in learners):
        raise ValueError('Every learner must change actual model arrays')
    rr.save_round(output,nets,opts,pools,models,rng,config,1,source_bank,False)
    # Verify on disposable copies while the original live optimizer is still
    # available. Two restores alone could agree while sharing the same bug.
    from copy import deepcopy
    from .seven_checkpoint import restore
    pointer = _read(output/'recovery/pointer.json')
    restore_checks = {}
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for key in learners:
            if time.time() >= deadline:
                raise TimeoutError('Joint next-Adam verification deadline')
            copied,copied_opt,data = restore(Path(pointer['folder'])/(key+'.pt'),
                directory=models,key=key,device=device)
            control = deepcopy(nets[key])
            control_opt = torch.optim.Adam(control.parameters())
            control_opt.load_state_dict(deepcopy(opts[key].state_dict()))
            sample = rows[key][:32]
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
            rt.update(control,control_opt,sample)
            expected_rng = torch.get_rng_state()
            expected_cuda = torch.cuda.get_rng_state_all() if cuda_rng is not None else None
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
            rt.update(copied,copied_opt,sample)
            equal = all(torch.equal(t,copied.state_dict()[name]) for name,t in control.state_dict().items())
            random_equal = torch.equal(expected_rng,torch.get_rng_state()) and (
                expected_cuda is None or all(torch.equal(x,y) for x,y in
                    zip(expected_cuda,torch.cuda.get_rng_state_all())))
            if not equal or not random_equal or data['rng_state'] != rng.bit_generator.state:
                raise ValueError('Restored next Adam step or RNG differs from live optimizer')
            restore_checks[key] = dict(adam_next_step_equal=True,rng_equal=True)
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    rr.write(output/'restore-check.json',restore_checks)
    if time.time() >= deadline:
        raise TimeoutError('Joint checkpoint deadline')
    return dict(models=str(models.resolve()),health=health,numeric=numeric,before=before,after=after,
                source_transition=source_transition,restore_checks=restore_checks)


def run_joint_update(source_bank, jobs, games, output, config, *, deadline, prior_stage=None,
                     main_value_report=None):
    """Apply a verified batch once, or validate and return its COMPLETE receipt."""
    layout = config.get('policy_layout', 'shared')
    if layout == 'phase_isolated_v1':
        from .phase_hybrid import run_joint_update as run
        return run(source_bank, jobs, games, output, config, deadline=deadline,
                   prior_stage=prior_stage, main_value_report=main_value_report)
    if layout != 'shared':
        raise ValueError('Unknown policy layout')
    from .seven_checkpoint import algorithm_identity
    source_bank, output = Path(source_bank), Path(output)
    source = {k:rt.model(source_bank,k) for k in ROSTER}
    scopes = _validate(source,jobs,games,config)
    contract = digest(dict(source={k:p.version for k,p in source.items()},jobs=jobs,games=games,
        config=config,algorithm=algorithm_identity(),implementation=Path(__file__).read_text(),
        prior_stage=str(Path(prior_stage).resolve()) if prior_stage else None,scopes=scopes,
        main_value_report=main_value_report))
    return _run_once(source_bank,jobs,games,output,config,deadline=deadline,
                     contract=contract,prior_stage=prior_stage,main_value_report=main_value_report)


def _run_once(source_bank,jobs,games,output,config,*,deadline,contract,prior_stage=None,
              main_value_report=None, executor=None):
    """Once-only transaction boundary after independent admission validation."""
    output = Path(output)
    marker = output/'joint-status.json'
    if marker.exists():
        receipt = _completed(output)
        if receipt['contract'] != contract:
            raise ValueError('Completed joint stage input contract changed')
        return receipt
    for name,value in (('deadline',deadline),('train_until',config.get('train_until')),
                       ('hard_deadline',config.get('hard_deadline'))):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Finite live joint '+name+' required')
    deadline = min(deadline,config['train_until'],config['hard_deadline'])
    if time.time() >= deadline:
        raise TimeoutError('Joint update budget expired before admission')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('New empty joint stage output required')
    running = dict(schema=SCHEMA,status='RUNNING',contract=contract,config=config,
        quality_approved=False,automatic_serving_approval=False)
    with marker.open('x') as stream:
        json.dump(running,stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        result = (executor or _execute_update)(source_bank,jobs,games,output,config,deadline=deadline,
                                prior_stage=prior_stage,main_value_report=main_value_report)
        receipt = dict(running,**result,status='COMPLETE',files=_files(output))
        rr.write(marker,receipt)
        return receipt
    except BaseException as exc:
        rr.write(marker,dict(running,status='FAILED',error=repr(exc),replay_forbidden=True))
        raise
