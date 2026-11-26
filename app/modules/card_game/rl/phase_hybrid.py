"""Opt-in phase-isolated optimizer stage for verified resident-rule hybrid episodes."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from . import hybrid_joint_update as transaction, policy_runtime as rt, seven_round as rr
from .seven_schedule import ROSTER
from .phase_replay import PhaseEpisodePool, balanced_slices, row_phase
from .phase_update import update


def _require_layout(config):
    if config.get('policy_layout') != rt.LAYOUT:
        raise ValueError('Explicit phase-isolated hybrid policy layout required')
    passes = config.get('policy_passes')
    reuse = config.get('max_row_reuse', 8)
    if passes not in (1, 4, 8) or type(reuse) is not int or reuse < passes:
        raise ValueError('Declared phase passes require a sufficient bounded row-reuse limit')


def run_joint_update(source_bank, jobs, games, output, config, *, deadline, prior_stage=None,
                     main_value_report=None):
    from .phase_checkpoint import algorithm_identity
    from .phase_policy import SCHEMA
    _require_layout(config)
    source = {k: rt.model(source_bank, k) for k in ROSTER}
    if any(p.manifest['schema'] != SCHEMA for p in source.values()):
        raise ValueError('Explicitly migrated all-seven phase model bank required')
    scopes = transaction._validate(source, jobs, games, config, model_loader=rt.model)
    contract = transaction.digest(dict(source={k: p.version for k, p in source.items()}, jobs=jobs,
        games=games, config=config, algorithm=algorithm_identity(),
        implementation=Path(__file__).read_text(), prior_stage=str(Path(prior_stage).resolve()) if prior_stage else None,
        scopes=scopes, main_value_report=main_value_report))
    return transaction._run_once(source_bank, jobs, games, output, config, deadline=deadline,
        contract=contract, prior_stage=prior_stage, main_value_report=main_value_report, executor=_execute_update)


def load_joint_state(output, *, device='cpu'):
    import torch
    from .phase_checkpoint import restore
    output = Path(output)
    receipt = transaction._completed(output)
    _require_layout(receipt['config'])
    pointer = transaction._read(output / 'recovery/pointer.json')
    folder, models = Path(pointer['folder']).resolve(), Path(pointer['models']).resolve()
    if (not folder.is_relative_to(output.resolve()) or not models.is_relative_to(output.resolve())
            or models != Path(receipt['models']).resolve()):
        raise ValueError('Phase recovery files must belong to their stage')
    nets, opts, saved = {}, {}, None
    for key in ROSTER:
        nets[key], opts[key], data = restore(folder / (key + '.pt'), directory=models, key=key, device=device)
        if data['metadata']['config'] != receipt['config']:
            raise ValueError('Phase checkpoint config changed')
        if saved is not None and data['rng_state'] != saved['rng_state']:
            raise ValueError('Phase checkpoint RNG differs across teams')
        saved = data
    pools = {k: PhaseEpisodePool.from_state(v) for k, v in transaction._read(folder / 'pools.json').items()}
    if set(pools) != set(ROSTER):
        raise ValueError('Complete phase replay pools required')
    for pool in pools.values():
        for episode in pool.episodes:
            for row in episode['rows']:
                for field in ('x', 'c', 'pi', 'visit_pi'):
                    if field in row:
                        row[field] = np.asarray(row[field], dtype=np.float32)
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['rng_state']
    torch.set_rng_state(saved['torch_rng'])
    if str(device).startswith('cuda') and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return nets, opts, pools, rng, receipt


def _rebind_value(nets, opts, previous, source_bank, report):
    import torch
    prior = {k: rt.model(previous['models'], k) for k in ROSTER}
    current = {k: rt.model(source_bank, k) for k in ROSTER}
    changed = any(prior[k].version != current[k].version for k in ROSTER)
    allowed = {k: {n for n, _ in net.named_parameters() if n.startswith(('value_net.', 'wdl.'))}
               for k, net in nets.items()}
    for k in ROSTER:
        old, new = prior[k].weights, current[k].weights
        if (set(old) != set(new) or prior[k].serving_deck != current[k].serving_deck
                or any(not np.array_equal(old[n], new[n]) for n in old if n not in allowed[k])):
            raise ValueError('Phase recalibration changed policy, masks, teacher or build')
    if changed:
        if not rt.value_precheck_ok(report, current):
            raise ValueError('Qualified recalibrated main-value report required')
        with torch.no_grad():
            for k, net in nets.items():
                for name, parameter in net.named_parameters():
                    if name in allowed[k]:
                        parameter.copy_(torch.as_tensor(current[k].weights[name].copy(), device=parameter.device))
                        opts[k]['value'].state.pop(parameter, None)
    return dict(value_recalibrated=changed, policy_optimizer_retained=True,
                selective_value_adam_reset=changed)


def _execute_update(source_bank, jobs, games, output, config, *, deadline, prior_stage=None,
                    main_value_report=None):
    import torch
    from .phase_checkpoint import create_optimizers, save, restore, algorithm_identity
    from .phase_policy import SCHEMA
    from .episode_replay import dump_episode
    output = Path(output)
    _require_layout(config)
    device = config.get('device', 'cuda')
    if prior_stage is None:
        torch.manual_seed(config.get('seed', 0))
        nets = {k: rt.restore(source_bank, k, device)[0] for k in ROSTER}
        if any(net.policy_schema != SCHEMA for net in nets.values()):
            raise ValueError('Phase stage cannot silently migrate shared policies')
        opts = {k: create_optimizers(net, config.get('learning_rate', .0003)) for k, net in nets.items()}
        pools = {k: PhaseEpisodePool(config.get('episode_pool_limit', 128), config.get('max_row_reuse', 8)) for k in ROSTER}
        rng = np.random.default_rng(config.get('seed', 0))
        transition = dict(value_recalibrated=False, policy_optimizer_retained=False)
    else:
        nets, opts, pools, rng, previous = load_joint_state(prior_stage, device=device)
        for field in ('hard_deadline', 'train_until', 'policy_passes', 'policy_layout'):
            if previous['config'][field] != config[field]:
                raise ValueError('Phase resume changes frozen ' + field)
        for field, default in (('learning_rate', .0003), ('seed', 0), ('max_row_reuse', 8), ('episode_pool_limit', 128)):
            if previous['config'].get(field, default) != config.get(field, default):
                raise ValueError('Phase resume changes frozen ' + field)
        transition = _rebind_value(nets, opts, previous, source_bank, main_value_report)
    before = {k: rt.model(source_bank, k) for k in ROSTER}
    for pool in pools.values():
        pool.begin_fresh_batch()
    for job, game in zip(jobs, games):
        data = dump_episode(output / 'episodes' / (job['id'] + '.json.gz'), job, game)
        for key in {job['policies'][s][1] for s in job['train_sides']}:
            pools[key].add(job['id'], dict(rows=[r for r in data['rows'] if r['key'] == key]))
    health, samples = {}, {}
    for key, net in nets.items():
        stats = []
        samples[key] = [r for j, g in zip(jobs, games) for r in g['rows']
                        if r['key'] == key and r['side'] in j['train_sides']]
        for rows, refs, sampling in pools[key].policy_pass_batches(rng, passes=config['policy_passes'], deadline=deadline):
            if time.time() >= deadline:
                raise TimeoutError('Phase update deadline')
            stats.append(dict(update(net, opts[key], rows), refs=refs, sampling=sampling))
        values = [r for j, g in zip(jobs, games) for r in g['value_rows']
                  if r['key'] == key and r['side'] in j['train_sides'] and r['value_actor'] == 0]
        value_stats = []
        for selection in balanced_slices(len(values), 128):
            if time.time() >= deadline:
                raise TimeoutError('Phase observer value deadline')
            value_stats.append(update(net, opts[key], values[selection], value_only=True))
        coverage = pools[key].coverage_report()
        phase_stats = {phase: [s for s in stats if s['policy_phase'] == phase] for phase in ('opening', 'playing')}
        if (coverage['first_use_coverage'] != 1 or coverage['fresh_policy_rows'] != len(samples[key])
                or any(not items or any(not any(s['policy_gradient_norms'][part] > 0 for s in items)
                                       for part in ('state_net', 'cand_net', 'score')) for items in phase_stats.values())
                or not values or any(not any(s['value_gradient_norms'][part] > 0 for s in stats + value_stats)
                                     for part in ('value_net', 'wdl'))):
            raise ValueError('Both policy phases, observer value gradients and full fresh coverage required')
        changed = [n for n, v in net.state_dict().items() if not np.array_equal(v.detach().cpu().numpy(), before[key].weights[n])]
        phase_changes = dict(opening=[n for n in changed if n.startswith('opening.')],
                             playing=[n for n in changed if n.startswith(('state_net.', 'cand_net.', 'score.'))],
                             value=[n for n in changed if n.startswith(('value_net.', 'wdl.'))])
        if not all(phase_changes.values()):
            raise ValueError('Every phase and main value must actually update')
        if any(not torch.isfinite(v).all() for v in net.state_dict().values()):
            raise ValueError('Nonfinite phase model')
        health[key] = dict(finite=True, gradient=True, coverage=1., policy_rows=len(samples[key]),
            value_rows=len(values), updates=len(stats), phases={p: dict(updates=len(v),
            sample_uses=sum(s['samples'] for s in v)) for p, v in phase_stats.items()},
            changed_policy_arrays=phase_changes['opening'] + phase_changes['playing'],
            changed_value_arrays=phase_changes['value'])
        rr.write(output / (key + '-updates.json'), stats)
        rr.write(output / (key + '-value-updates.json'), value_stats)
    models = output / 'models'
    for key, net in nets.items():
        rt.export(net, models, key, before[key].serving_deck,
                  dict(before[key].manifest['origin'], policy_layout=rt.LAYOUT, hybrid_joint_update=True))
    numeric = rr.numeric_check(nets, models, samples, device)
    folder = output / 'recovery/1'
    for key, net in nets.items():
        save(folder / (key + '.pt'), net, opts[key], directory=models, key=key,
             rng_state=rng.bit_generator.state, metadata=dict(config=config, cycle=1))
    pool_path = folder / 'pools.json'
    temporary = pool_path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump({k: p.export_state() for k, p in pools.items()}, stream,
                  default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a.item())
    temporary.replace(pool_path)
    rr.write(output / 'recovery/pointer.json', dict(folder=str(folder.resolve()), models=str(models.resolve()),
        algorithm_sha256=algorithm_identity(), original_train_until=config['train_until'], hard_deadline=config['hard_deadline']))
    checks = {}
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for key, net in nets.items():
            if time.time() >= deadline:
                raise TimeoutError('Phase restore verification deadline')
            copied, copied_opts, data = restore(folder / (key + '.pt'), directory=models, key=key, device=device)
            control = deepcopy(net)
            control_opts = create_optimizers(control, config.get('learning_rate', .0003))
            for phase in control_opts:
                control_opts[phase].load_state_dict(deepcopy(opts[key][phase].state_dict()))
            start_rng = torch.get_rng_state()
            start_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            phase_samples = [[r for r in samples[key] if row_phase(r) == phase][:8]
                             for phase in ('opening', 'playing')]
            for sample in phase_samples:
                update(control, control_opts, sample)
            expected_rng = torch.get_rng_state()
            expected_cuda = torch.cuda.get_rng_state_all() if start_cuda is not None else None
            torch.set_rng_state(start_rng)
            if start_cuda is not None:
                torch.cuda.set_rng_state_all(start_cuda)
            for sample in phase_samples:
                update(copied, copied_opts, sample)
            random_equal = torch.equal(expected_rng, torch.get_rng_state()) and (
                expected_cuda is None or all(torch.equal(a, b) for a, b in
                    zip(expected_cuda, torch.cuda.get_rng_state_all())))
            if (not all(torch.equal(v, copied.state_dict()[n]) for n, v in control.state_dict().items())
                    or not random_equal or data['rng_state'] != rng.bit_generator.state):
                raise ValueError('Phase next-Adam restoration mismatch')
            checks[key] = dict(adam_next_step_equal=True, rng_equal=True)
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    rr.write(output / 'restore-check.json', checks)
    if time.time() >= deadline:
        raise TimeoutError('Phase export budget expired')
    return dict(models=str(models.resolve()), health=health, numeric=numeric,
                before={k: p.version for k, p in before.items()}, after={k: rt.model(models, k).version for k in ROSTER},
                source_transition=transition, restore_checks=checks)
