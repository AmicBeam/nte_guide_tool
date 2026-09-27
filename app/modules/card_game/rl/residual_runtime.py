"""Strict numeric runtime for the residual recovery experiment; no serving approval."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import residual_policy as architecture
from . import recovery_policy as separated
from . import cross_grounded as grounded
from .cross_lineup import all_features, candidate_names, identity
from .cross_runtime import decision, determinize, passive_count, validate_policy_passes, policy_gradient_ok
from .league_schema import build_hash


class ResidualModel:
    def __init__(self, directory, key):
        root = Path(directory)
        self.manifest = m = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
        path = root / f'{key}.npz'
        self.version = sha256(path.read_bytes()).hexdigest()
        if (m.get('schema') not in (architecture.SCHEMA, separated.RESIDUAL_SCHEMA) or m.get('rule_hash') != identity()
                or m.get('architecture_sha256') != architecture.fingerprint()
                or m.get('candidate_transform_sha256') != grounded.fingerprint()
                or m.get('sha256') != self.version or m.get('deck') != key
                or m.get('features') != all_features() or m.get('candidates') != candidate_names()
                or m.get('wdl_order') != ['loss', 'draw', 'win']
                or m.get('value_context') != ['viewer_is_actor']
                or m.get('automatic_serving_approval') is not False
                or m.get('reward_mode') != 'terminal_wdl_only'):
            raise ValueError('Residual model identity mismatch')
        self.hidden = m['hidden']
        self.separated = m['schema'] == separated.RESIDUAL_SCHEMA
        if self.separated and m.get('value_architecture_sha256') != separated.fingerprint():
            raise ValueError('Value architecture identity mismatch')
        expected = architecture.tensor_shapes(self.hidden)
        if self.separated: expected.update(separated.extra_shapes(self.hidden))
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Residual tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in expected}
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError('Invalid residual tensor')
            self.weights[name].flags.writeable = False
        self.serving_deck = m['build']
        if build_hash(self.serving_deck) != m.get('build_sha256'):
            raise ValueError('Residual build mismatch')

    def scores_only(self, x, c):
        return architecture.scores_numpy(x, c, self.weights)

    scores = scores_only

    def wdl(self, x, is_actor):
        return architecture.wdl_numpy(x, is_actor, separated.value_weights(self.weights) if self.separated else self.weights)

    def scores_value(self, x, c):
        p = self.wdl(x, True)
        return self.scores_only(x, c), float(p[2] - p[0])

    def validate_human(self, build):
        raise ValueError('Residual research model has no website authorization')


@lru_cache(maxsize=16)
def _load(directory, key, modified):
    return ResidualModel(directory, key)


def model(directory, key):
    path = Path(directory) / f'{key}.json'
    return _load(str(directory), key, path.stat().st_mtime_ns)


def value_for(state, viewer, policies):
    from .cross_lineup import encode
    from ..engine.duel_v2 import observe, acting_side
    x, _ = encode(observe(state, viewer, include_previews=False), [])
    p = policies[viewer].wdl(x, acting_side(state) == viewer)
    return float(p[2] - p[0])


def predict_encoded(policy, x, c):
    return policy.scores_value(x, c)


def action_scores(policy, x, c):
    return policy.scores_only(x, c)


create_network = architecture.network


def export(net, directory, key, build, origin):
    from .cross_runtime import _write
    if getattr(net, 'policy_schema', None) not in (architecture.SCHEMA, separated.RESIDUAL_SCHEMA):
        raise ValueError('Not a residual network')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{key}.npz'
    tmp = root / f'{key}.tmp.npz'
    np.savez_compressed(tmp, **{name: tensor.detach().cpu().numpy() for name, tensor in net.state_dict().items()})
    tmp.replace(path)
    _write(root / f'{key}.json', dict(
        schema=net.policy_schema, value_architecture_sha256=separated.fingerprint() if hasattr(net,'value_net') else None, architecture_sha256=architecture.fingerprint(),
        candidate_transform_sha256=grounded.fingerprint(), rule_hash=identity(), deck=key,
        hidden=net.hidden, features=all_features(), candidates=candidate_names(),
        sha256=sha256(path.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'], origin=origin,
        reward_mode='terminal_wdl_only', automatic_serving_approval=False,
        validation={'approved': False}))


def restore(directory, key, device='cpu'):
    import torch
    policy = model(directory, key)
    net = create_network(policy.hidden, device)
    if policy.separated: net = separated.attach(net)
    net.load_state_dict({name: torch.as_tensor(array.copy(), device=device)
                         for name, array in policy.weights.items()})
    return net, policy.serving_deck, policy.manifest['origin']


def update(net, opt, rows, **kwargs):
    from .outcome_runtime import update as update_rows
    return update_rows(net, opt, rows, **kwargs)
