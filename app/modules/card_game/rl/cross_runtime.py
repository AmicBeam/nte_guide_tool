"""NumPy inference and fresh WDL network for the five-preset cross-play contract."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from .league_rollout import determinize as sample
from .league_schema import build_hash
from .cross_lineup import (
    CARD_IDS, SCHEMA, SEATS, all_features, candidate_names, encode, identity,
)
from . import cross_grounded as grounded
from ..engine.duel_v2 import acting_side, observe


class CrossModel:
    def __init__(self, directory, key):
        root = Path(directory)
        manifest = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
        path = root / f'{key}.npz'
        digest = sha256(path.read_bytes()).hexdigest()
        if (manifest.get('schema') != SCHEMA or manifest.get('rule_hash') != identity()
                or manifest.get('sha256') != digest or manifest.get('deck') != key
                or manifest.get('features') != all_features() or manifest.get('candidates') != candidate_names()
                or manifest.get('wdl_order') != ['loss', 'draw', 'win']
                or manifest.get('candidate_transform_sha256') != grounded.fingerprint()
                or manifest.get('automatic_serving_approval') is not False):
            raise ValueError('Cross model identity mismatch')
        self.hidden = manifest['hidden']
        if type(self.hidden) is not int or not 1 <= self.hidden <= 256:
            raise ValueError('Invalid width')
        expected = {
            'state_net.0.weight': (self.hidden, len(all_features())),
            'state_net.0.bias': (self.hidden,),
            'state_net.2.weight': (self.hidden, self.hidden),
            'state_net.2.bias': (self.hidden,),
            'cand_net.0.weight': (self.hidden, grounded.DIM),
            'cand_net.0.bias': (self.hidden,),
            'cand_net.2.weight': (self.hidden, self.hidden),
            'cand_net.2.bias': (self.hidden,),
            'score.weight': (1, self.hidden),
            'score.bias': (1,),
            'value.0.weight': (self.hidden, self.hidden),
            'value.0.bias': (self.hidden,),
            'value.2.weight': (1, self.hidden),
            'value.2.bias': (1,),
            'wdl.weight': (3, self.hidden + 1),
            'wdl.bias': (3,),
        }
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Cross tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in expected}
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError('Invalid cross tensor')
            self.weights[name].flags.writeable = False
        self.serving_deck = manifest['build']
        if build_hash(self.serving_deck) != manifest['build_sha256']:
            raise ValueError('Cross build mismatch')
        self.manifest = manifest
        self.version = digest

    def _linear(self, value, name):
        return np.einsum('...j,ij->...i', value, self.weights[name + '.weight'], optimize=False) + self.weights[name + '.bias']

    def scores_only(self, x, candidates):
        if len(candidates) == 0:
            return np.zeros(0, dtype=np.float32)
        transformed = grounded.transform(x, candidates)
        hidden = np.maximum(self._linear(x, 'state_net.0'), 0)
        hidden = np.maximum(self._linear(hidden, 'state_net.2'), 0)
        cand = np.maximum(self._linear(transformed, 'cand_net.0'), 0)
        cand = np.maximum(self._linear(cand, 'cand_net.2'), 0)
        scores = np.sum(cand * hidden, -1) / np.sqrt(np.float32(self.hidden)) + self._linear(cand, 'score').reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite cross scores')
        return scores

    def scores(self, x, candidates):
        return self.scores_only(x, candidates)

    def wdl(self, x, is_actor):
        hidden = np.maximum(self.weights['state_net.0.weight'] @ x + self.weights['state_net.0.bias'], 0)
        hidden = np.maximum(self.weights['state_net.2.weight'] @ hidden + self.weights['state_net.2.bias'], 0)
        logits = self.weights['wdl.weight'] @ np.append(hidden, np.float32(bool(is_actor))) + self.weights['wdl.bias']
        prob = np.exp(logits - logits.max())
        return prob / prob.sum()

    def scores_value(self, x, candidates):
        prob = self.wdl(x, True)
        return self.scores_only(x, candidates), float(prob[2] - prob[0])

    def validate_human(self, build):
        raise ValueError('Cross research policy has no website authorization')


@lru_cache(maxsize=8)
def _load(directory, key, modified):
    return CrossModel(directory, key)


def model(directory, key):
    path = Path(directory) / f'{key}.json'
    return _load(str(directory), key, path.stat().st_mtime_ns)


def decision(state, side):
    view = observe(state, side, include_previews=False)
    actions = [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']
    if not actions:
        raise ValueError('No legal non-concede action')
    return actions, encode(view, actions)


def determinize(state, side, seed):
    return sample(state, side, seed, encoder=encode, seats=SEATS, cards=CARD_IDS,
                  ignore_summons=True, generated_ids=('AF01', 'A03')
                  if 'anhunqu' in state['sides']['b' if side == 'a' else 'a']['characters'] else ())


def passive_count(state, side):
    return int(state['sides'][side]['characters'].get('zhenhong', {}).get('surplus_passive_triggers', 0))


def value_for(state, viewer, policies):
    encoded, _ = encode(observe(state, viewer, include_previews=False), [])
    prob = policies[viewer].wdl(encoded, acting_side(state) == viewer)
    return float(prob[2] - prob[0])


def predict_encoded(policy, x, c):
    scores = policy.scores_only(x, c)
    prob = policy.wdl(x, True)
    return scores, float(prob[2] - prob[0])


def action_scores(policy, x, c):
    return policy.scores_only(x, c)


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def create_network(hidden, device):
    import torch
    net = grounded.network(hidden)
    with torch.no_grad():
        net.wdl.weight.zero_()
        net.wdl.bias.zero_()
        net.cand_net[2].weight.mul_(0.01)
        net.cand_net[2].bias.mul_(0.01)
        net.score.weight.zero_()
        net.score.bias.zero_()
    for parameter in net.value.parameters():
        parameter.requires_grad_(False)
    return net.to(device)


def export(net, directory, key, build, origin):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{key}.npz'
    temporary = root / f'{key}.tmp.npz'
    np.savez_compressed(temporary, **{name: value.detach().cpu().numpy() for name, value in net.state_dict().items()})
    temporary.replace(path)
    _write(root / f'{key}.json', dict(
        schema=SCHEMA, candidate_transform_sha256=grounded.fingerprint(), rule_hash=identity(), deck=key,
        features=all_features(), candidates=candidate_names(), hidden=int(net.score.in_features),
        wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
        sha256=sha256(path.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        origin=origin, reward_mode='terminal_wdl_only', validation={'approved': False},
        automatic_serving_approval=False))


POLICY_PASSES = (1, 4, 8)


def validate_policy_passes(value):
    if value not in POLICY_PASSES:
        raise ValueError('policy_passes must be 1, 4 or 8')
    return int(value)


def policy_gradient_ok(metrics):
    prefixes = ('cand_net', 'score', 'state_net')
    norms = metrics.get('policy_gradient_norms') or {}
    if metrics.get('policy_samples', 0) <= 0:
        return False
    return all(np.isfinite(norms.get(prefix, 0)) and norms.get(prefix, 0) > 0 for prefix in prefixes)


def update(net, opt, rows, **kwargs):
    from .outcome_runtime import update as update_rows
    return update_rows(net, opt, rows, **kwargs)


def _candidate_columns(contract, transform):
    categories = [('kind', tuple(map(str, range(6)))),
                  ('actor', ('none', *contract.SEATS)),
                  ('card', ('none', *contract.CARD_IDS)),
                  ('side', ('none', 'self', 'opponent')),
                  ('target', ('none', *contract.SEATS))]
    names = [f'{kind}:{value}' for kind, values in categories for value in values]
    names += ['numeric:' + name for name in contract.candidate_names()[5:]]
    names += [f'{role}:{field}' for role in ('actor_state', 'target_state') for field in transform.FIELDS]
    if len(names) != transform.DIM or len(names) != len(set(names)):
        raise ValueError('Invalid named candidate migration')
    return names


def migrate(source, key, device, *, source_rule_hash=None):
    """Copy known named columns into a new schema; never rewrite old manifests."""
    import torch
    from .build_acceptance import check_source
    from . import cross_lineup as target
    check_source(source)
    if key == 'murk':
        from . import murk_lineup as before
        from . import murk_grounded as old_transform
        from .murk_runtime import model as load
    else:
        from . import fixed_lineup as before
        from . import grounded_candidates as old_transform
        from .outcome_runtime import model as load
    if source_rule_hash is None:
        policy = load(source, key)
    else:
        # Explicit research migration across a reviewed rule-source change.
        # Validate the historical identity; do not rewrite it as current.
        from types import SimpleNamespace
        path = Path(source)
        manifest = json.loads((path / f'{key}.json').read_text(encoding='utf-8'))
        digest = sha256((path / f'{key}.npz').read_bytes()).hexdigest()
        if (manifest.get('rule_hash') != source_rule_hash or manifest.get('sha256') != digest
                or manifest.get('schema') != old_transform.SCHEMA or manifest.get('deck') != key
                or manifest.get('features') != before.all_features()
                or manifest.get('candidates') != before.candidate_names()
                or manifest.get('candidate_transform_sha256') != old_transform.fingerprint()
                or manifest.get('wdl_order') != ['loss', 'draw', 'win']
                or manifest.get('value_context') != ['viewer_is_actor']
                or manifest.get('automatic_serving_approval') is not False
                or build_hash(manifest['build']) != manifest.get('build_sha256')):
            raise ValueError('Historical source identity mismatch')
        hidden = manifest['hidden']
        if type(hidden) is not int or not 1 <= hidden <= 256:
            raise ValueError('Invalid historical width')
        expected = old_transform.network(hidden).state_dict()
        with np.load(path / f'{key}.npz', allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Historical tensor set mismatch')
            weights = {name: arrays[name].copy() for name in arrays.files}
        for name, array in weights.items():
            if array.shape != tuple(expected[name].shape) or not np.isfinite(array).all():
                raise ValueError('Historical tensor mismatch')
        policy = SimpleNamespace(hidden=hidden, weights=weights, serving_deck=manifest['build'],
                                 manifest=manifest, version=digest)
    if policy.manifest['schema'] != old_transform.SCHEMA:
        raise ValueError('Migration requires grounded WDL source')
    net = create_network(policy.hidden, device)
    state = net.state_dict()
    reset = set()
    # Murk ordinal seat/card values change in the union; do not mislabel them as
    # semantically unchanged. Categorical action identities map by name below.
    if key == 'murk':
        reset = {'resolving'} | {f'{k}:{s}' for k in ('front', 'last_front', 'extra_genesis_actor') for s in (0, 1)}
        reset |= {f'ch_shape:{s}:{c}' for s in (0, 1) for c in before.SEATS}
    mappings = {
        'state_net.0.weight': (before.all_features(), all_features()),
        'cand_net.0.weight': (_candidate_columns(before, old_transform), _candidate_columns(target, grounded)),
    }
    for name, value in policy.weights.items():
        if name in mappings:
            old, new = mappings[name]
            indexes = {column: i for i, column in enumerate(new)}
            state[name].zero_()
            for i, column in enumerate(old):
                if column not in indexes:
                    raise ValueError('Unsupported source column: ' + column)
                if name != 'state_net.0.weight' or column not in reset:
                    state[name][:, indexes[column]] = torch.as_tensor(value[:, i].copy(), device=device)
        else:
            if tuple(state[name].shape) != value.shape:
                raise ValueError('Source tensor shape mismatch')
            state[name] = torch.as_tensor(value.copy(), device=device)
    net.load_state_dict(state)
    return net, policy.serving_deck, dict(
        initialization='named_grounded_wdl_expansion', source=str(source),
        source_sha256=policy.version, source_schema=policy.manifest['schema'],
        source_rule_hash=policy.manifest['rule_hash'], optimizer_reset=True,
        new_columns_zero_initialized=True, changed_ordinal_columns_reset=sorted(reset),
        automatic_serving_approval=False)
