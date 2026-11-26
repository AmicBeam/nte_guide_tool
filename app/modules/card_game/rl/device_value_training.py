"""Bounded live GPU value-calibration smoke, before formal Gumbel readiness.

Uses the existing seven-runtime value optimizer. No policy gradient, search
qualification, model selection, serving approval or full training is implied.
Both collection batches use CUDA rule loops and disjoint physical seeds.
"""
from copy import deepcopy
import hashlib
import math
import re
from pathlib import Path
import time


def _check_jobs(jobs, *, deadline):
    if not isinstance(jobs, (list, tuple)) or not jobs:
        raise ValueError('Nonempty explicit pure collection jobs required')
    ids = set()
    for job in jobs:
        if (job.get('pure') is not True or job.get('training') is not False
                or type(job.get('seed')) is not int or job['seed'] < 0
                or not isinstance(job.get('id'), str) or re.fullmatch(r'[A-Za-z0-9_-]+', job['id']) is None or job['id'] in ids
                or job.get('deadline') != deadline):
            raise ValueError('Fixed unique pure jobs and original absolute deadline required')
        ids.add(job['id'])


def run_value_training_smoke(output, *, collector, models, train_jobs, evaluation_jobs, deadline):
    """Collect -> real WDL update -> export/restore -> frozen collection.

    ``collector`` is the real DevicePureCollector, never a list of archived
    episodes. This stage deliberately does not need a qualified Gumbel driver:
    its complete argmax trajectories establish the first value calibration.
    Passing this smoke is not the heldout value qualification for policy search.
    """
    from .batched_search.device_pure_collection import DevicePureCollector
    from .batched_search.soa_pure_jobs import adapt_pure_jobs
    from .batched_search.soa_training_jobs import policy_source_alias
    from .seven_round import write
    from . import seven_runtime as rt
    if not isinstance(collector, DevicePureCollector):
        raise TypeError('Live actual CUDA value collector required')
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError('Future finite absolute smoke deadline required')
    _check_jobs(train_jobs, deadline=deadline); _check_jobs(evaluation_jobs, deadline=deadline)
    if {j['seed'] for j in train_jobs} & {j['seed'] for j in evaluation_jobs}:
        raise ValueError('Training and frozen evaluation seeds must be disjoint')
    if len({m.manifest['deck'] for m in models.values()}) != len(models):
        raise ValueError('One starting source per team required for value smoke exports')
    # Validate every source/build/initializer before output or CUDA collection.
    adapt_pure_jobs(train_jobs, models); adapt_pure_jobs(evaluation_jobs, models)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, stage='collecting_value_train', optimizer_steps=0,
        formal_policy_updates=0, formal_training_complete=False, quality_approved=False,
        training_kind='value_calibration_smoke', deadline=deadline, natural_train_games=0,
        natural_evaluation_games=0, initial_versions={a: m.version for a, m in models.items()})
    def save(): write(root / 'report.json', report)
    save()
    try:
        import numpy as np
        import torch
        from .episode_replay import dump_episode
        games = collector.collect(train_jobs, models)
        report['train_collection'] = deepcopy(collector.last_diagnostics)
        report['natural_train_games'] = len(games)
        by_source = {}
        for job, game in zip(train_jobs, games):
            dump_episode(root / 'train' / (job['id'] + '.json.gz'), job, game)
            for row in game['value_rows']:
                by_source.setdefault(policy_source_alias(*job['policies'][row['side']]), []).append(row)
        if not by_source:
            raise ValueError('Complete device games supplied no actual value training rows')
        report['stage'] = 'updating_value'; save()
        new_bank = {}; updates = {}; folder = root / 'models'
        for alias, model in models.items():
            if time.time() >= deadline:
                raise TimeoutError('Value smoke deadline before update/export')
            key = model.manifest['deck']
            net, build, origin = rt.restore(model.directory, key, str(collector.executor.device))
            before = {n: value.detach().clone() for n, value in net.state_dict().items()}
            rows = by_source.get(alias, [])
            if rows:
                optimizer = torch.optim.Adam(list(net.value_net.parameters()) + list(net.wdl.parameters()), lr=.0003)
                net.zero_grad(set_to_none=True)
                metrics = rt.update(net, optimizer, rows[:128], value_only=True)
                if not all(math.isfinite(float(metrics[n])) for n in ('policy_loss', 'value_loss')):
                    raise ValueError('Nonfinite value smoke update')
                changed = [n for n, value in net.state_dict().items() if not torch.equal(before[n], value)]
                if not changed or any(not n.startswith(('value_net.', 'wdl.')) for n in changed):
                    raise ValueError('Value smoke must change value parameters and preserve every policy tensor')
                updates[key] = dict(metrics=metrics, rows=min(len(rows), 128), changed=changed)
                torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict()), root / (key + '-checkpoint.pt'))
                report['optimizer_steps'] += 1
            rt.export(net, folder, key, build, dict(origin, bounded_value_smoke=True))
            restored = rt.model(folder, key)
            for name, value in net.state_dict().items():
                np.testing.assert_array_equal(value.detach().cpu().numpy(), restored.weights[name])
            new_bank[policy_source_alias(str(folder), key)] = restored
            del net
        report.update(updates=updates, stage='frozen_evaluation', policy_arrays_unchanged=True)
        save()
        eval_jobs = deepcopy(list(evaluation_jobs))
        for job in eval_jobs:
            for side in ('a', 'b'):
                job['policies'][side] = (str(folder), job['policies'][side][1])
        frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        evaluated = collector.collect(eval_jobs, new_bank)
        report['evaluation_collection'] = deepcopy(collector.last_diagnostics)
        for job, game in zip(eval_jobs, evaluated):
            dump_episode(root / 'evaluation' / (job['id'] + '.json.gz'), job, game)
        if frozen != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}:
            raise ValueError('Frozen evaluation changed exported models')
        for alias, model in models.items():
            path = Path(model.directory) / (model.manifest['deck'] + '.npz')
            if hashlib.sha256(path.read_bytes()).hexdigest() != report['initial_versions'][alias]:
                raise ValueError('Original starting weights changed')
        report.update(passed=True, stage='complete', natural_evaluation_games=len(evaluated),
            frozen_weights_unchanged=True, numeric_restore_equal=True, initial_weights_unchanged=True,
            evaluation=[dict(id=g['id'], winner=g['winner'], decisions=g['decisions']) for g in evaluated])
        save(); return report
    except BaseException as error:
        report.update(error=repr(error), failed_stage=report['stage'],
            last_collection=deepcopy(collector.last_diagnostics))
        save(); raise
