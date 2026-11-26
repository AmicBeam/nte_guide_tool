"""Explicit schema dispatch without changing historical seven-model identities."""
import json
from pathlib import Path
from . import seven_runtime as _seven

LAYOUT = 'phase_isolated_v1'


def _phase_schema():
    from .phase_policy import SCHEMA
    return SCHEMA


def model(directory, key):
    manifest = json.loads((Path(directory) / (key + '.json')).read_text(encoding='utf-8'))
    if manifest.get('schema') == _phase_schema():
        from .phase_runtime import model as load
        return load(directory, key)
    return _seven.model(directory, key)


def restore(directory, key, device='cpu'):
    policy = model(directory, key)
    if policy.manifest['schema'] == _phase_schema():
        from .phase_runtime import restore as load
        return load(directory, key, device=device)
    return _seven.restore(directory, key, device=device)


def export(net, directory, key, build, origin):
    if getattr(net, 'policy_schema', None) == _phase_schema():
        from .phase_runtime import export as save
        return save(net, directory, key, build, origin)
    return _seven.export(net, directory, key, build, origin)


def update(net, optimizer, rows, **kwargs):
    if getattr(net, 'policy_schema', None) == _phase_schema():
        if (kwargs.get('value_only') or kwargs.get('warmup')) and not isinstance(optimizer, dict):
            # Existing independent value calibration owns only value parameters.
            return _seven.update(net, optimizer, rows, **kwargs)
        from .phase_update import update as fit
        return fit(net, optimizer, rows, **kwargs)
    return _seven.update(net, optimizer, rows, **kwargs)


def search_game(job):
    if any(model(*spec).manifest['schema'] == _phase_schema() for spec in job['policies'].values()):
        from .phase_rollout import search_game as play
        return play(job)
    return _seven.search_game(job)


def __getattr__(name):
    # Encoding, sampler, main-value admission and old cold-start factory stay exact.
    return getattr(_seven, name)
