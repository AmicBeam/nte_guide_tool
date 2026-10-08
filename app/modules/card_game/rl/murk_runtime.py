"""NumPy inference and fresh WDL network for the murk contract."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from .league_rollout import determinize as sample
from .league_schema import build_hash
from .murk_lineup import (
    CARD_IDS, GENERATED, KEY, SCHEMA, SEATS, all_features, candidate_names, encode, identity,
)
from . import murk_grounded as grounded
from ..engine.duel_v2 import acting_side, observe


class MurkModel:
    def __init__(self, directory, key=KEY):
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
            raise ValueError('Murk model identity mismatch')
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
                raise ValueError('Murk tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in expected}
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError('Invalid murk tensor')
            self.weights[name].flags.writeable = False
        self.serving_deck = manifest['build']
        if build_hash(self.serving_deck) != manifest['build_sha256']:
            raise ValueError('Murk build mismatch')
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
            raise ValueError('Non-finite murk scores')
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
        raise ValueError('Murk research policy has no website authorization')


@lru_cache(maxsize=8)
def _load(directory, key, modified):
    return MurkModel(directory, key)


def model(directory, key=KEY):
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
                  ignore_summons=True, generated_ids=GENERATED)


def passive_count(state, side):
    return 0


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


def update(net, opt, rows):
    from .outcome_runtime import update as update_rows
    return update_rows(net, opt, rows)
