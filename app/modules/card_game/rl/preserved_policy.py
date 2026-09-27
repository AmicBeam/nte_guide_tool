"""Keep verified old logits exactly while learning a state-conditioned residual.

The old policy is frozen and embedded with its historical identity. A new
WDL trunk cannot erase it. Only the residual is learned, initially exactly
zero. Unsupported old action identities fail explicitly. Research only.
"""
from hashlib import sha256
from pathlib import Path
import json
import types
import numpy as np
from . import residual_policy as architecture
from . import recovery_policy as separated
from . import cross_grounded as grounded
from .cross_lineup import all_features, candidate_names, identity
from .cross_teacher_distillation import (
    REPO_TEACHERS, verify_old_teacher, teacher_scores, project_observation,
    project_candidates, unsupported_action_mask,
)
from .league_schema import build_hash

SCHEMA = 'cross_five_preserved_residual_wdl_v1'
PARTIAL_SCHEMA = 'cross_five_partial_preserved_residual_wdl_v1'


def encoded_name(name):
    return name.replace('.', '__')


def create_network(key, source=REPO_TEACHERS, hidden=64, device='cpu', *, partial=False):
    import torch
    from torch import nn
    teacher = verify_old_teacher(source, key)
    net = architecture.network(hidden, device)

    class FrozenPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            for name, value in teacher['weights'].items():
                self.register_buffer(encoded_name(name), torch.as_tensor(value.copy()))
            self.register_buffer('observation_index', torch.tensor(teacher['observation_index']), persistent=False)
            self.register_buffer('candidate_index', torch.tensor(teacher['candidate_index']), persistent=False)
            extra = [i for i, name in enumerate(candidate_names()) if name not in teacher['candidates']]
            self.register_buffer('extra_index', torch.tensor(extra, dtype=torch.long), persistent=False)

        def linear(self, x, name):
            return torch.nn.functional.linear(x, getattr(self, encoded_name(name + '.weight')),
                                               getattr(self, encoded_name(name + '.bias')))

        def forward(self, x, c, mask):
            from .fixed_lineup import SEATS, CARD_IDS
            unknown = ((c[..., 1] >= len(SEATS)) | (c[..., 2] >= len(CARD_IDS))
                       | (c[..., 4] >= len(SEATS))
                       | (c[..., self.extra_index].abs() > 1e-8).any(-1))
            unsupported = (unknown & mask).any(-1)
            if bool(unsupported.any()) and not partial:
                raise ValueError('Unsupported frozen policy action')
            # A whole unsupported root gets only the new residual. Never
            # fabricate legacy identity columns or mix incomparable scores.
            result = x.new_zeros(mask.shape)
            supported = ~unsupported
            if not bool(supported.any()):
                return result
            x = x[supported][:, self.observation_index]
            c = c[supported][..., self.candidate_index]
            state = self.linear(x, 'state_net.0').relu()
            state = self.linear(state, 'state_net.2').relu()
            cand = self.linear(c, 'cand_net.0').relu()
            cand = self.linear(cand, 'cand_net.2').relu()
            result[supported] = ((cand * state.unsqueeze(1)).sum(-1) / np.sqrt(np.float32(teacher['hidden']))
                                 + self.linear(cand, 'score').squeeze(-1))
            return result

    net.add_module('legacy', FrozenPolicy().to(device))
    net.source_identity = {name: teacher[name] for name in (
        'key', 'schema', 'sha256', 'source_rule_hash', 'features', 'candidates', 'hidden',
        'observation_index', 'candidate_index', 'added_features')}
    with torch.no_grad():
        net.score[2].weight.zero_(); net.score[2].bias.zero_()
    original = type(net).forward

    def forward(self, x, c, mask):
        residual, value = original(self, x, c, mask)
        result = residual + self.legacy(x, c, mask)
        return result.masked_fill(~mask, torch.finfo(result.dtype).min), value

    net.forward = types.MethodType(forward, net)
    net.policy_schema = PARTIAL_SCHEMA if partial else SCHEMA
    return net, teacher['build']


def export(net, directory, key, build, origin):
    from .cross_runtime import _write
    if net.policy_schema not in (SCHEMA, PARTIAL_SCHEMA, separated.PRESERVED_SCHEMA) or net.source_identity['key'] != key:
        raise ValueError('Preserved model identity mismatch')
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    path = root / f'{key}.npz'; tmp = root / f'{key}.tmp.npz'
    np.savez_compressed(tmp, **{name: value.detach().cpu().numpy() for name, value in net.state_dict().items()})
    tmp.replace(path)
    _write(root / f'{key}.json', dict(
        schema=net.policy_schema, value_architecture_sha256=separated.fingerprint() if hasattr(net,'value_net') else None, unsupported_root='residual_only' if net.policy_schema in (PARTIAL_SCHEMA, separated.PRESERVED_SCHEMA) else 'reject', architecture_sha256=architecture.fingerprint(),
        preservation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_transform_sha256=grounded.fingerprint(), rule_hash=identity(), deck=key,
        hidden=net.hidden, source_identity=net.source_identity, features=all_features(), candidates=candidate_names(),
        sha256=sha256(path.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        origin=origin, reward_mode='terminal_wdl_only', wdl_order=['loss', 'draw', 'win'],
        value_context=['viewer_is_actor'], validation={'approved': False}, automatic_serving_approval=False))


class PreservedModel:
    def __init__(self, directory, key):
        from .fixed_lineup import tensor_shapes, CAND_DIM
        from .cross_teacher_distillation import named_index_map, current_features, current_candidates
        root = Path(directory); path = root / f'{key}.npz'
        self.manifest = m = json.loads((root / f'{key}.json').read_text())
        self.version = sha256(path.read_bytes()).hexdigest()
        if (m.get('schema') not in (SCHEMA, PARTIAL_SCHEMA, separated.PRESERVED_SCHEMA) or m.get('deck') != key or m.get('sha256') != self.version
                or m.get('rule_hash') != identity() or m.get('architecture_sha256') != architecture.fingerprint()
                or m.get('preservation_sha256') != sha256(Path(__file__).read_bytes()).hexdigest()
                or m.get('candidate_transform_sha256') != grounded.fingerprint()
                or m.get('features') != all_features() or m.get('candidates') != candidate_names()
                or m.get('automatic_serving_approval') is not False
                or m.get('wdl_order') != ['loss', 'draw', 'win']
                or m.get('value_context') != ['viewer_is_actor']
                or m.get('reward_mode') != 'terminal_wdl_only'):
            raise ValueError('Preserved numeric identity mismatch')
        self.separated = m['schema'] == separated.PRESERVED_SCHEMA
        if self.separated and m.get('value_architecture_sha256') != separated.fingerprint():
            raise ValueError('Value architecture identity mismatch')
        self.partial = m['schema'] in (PARTIAL_SCHEMA, separated.PRESERVED_SCHEMA)
        if self.partial and m.get('unsupported_root') != 'residual_only':
            raise ValueError('Partial preservation must declare unsupported roots')
        self.hidden = m['hidden']; source = m['source_identity']
        teacher = verify_old_teacher(REPO_TEACHERS, key)
        if any(source.get(name) != teacher[name] for name in source):
            raise ValueError('Frozen source identity mismatch')
        old = tensor_shapes(source['hidden'])
        old['state_net.0.weight'] = (source['hidden'], len(source['features']))
        old['cand_net.0.weight'] = (source['hidden'], CAND_DIM)
        expected = dict(architecture.tensor_shapes(self.hidden))
        if self.separated: expected.update(separated.extra_shapes(self.hidden))
        expected.update({'legacy.' + encoded_name(name): shape for name, shape in old.items()})
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Preserved tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in expected}
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError('Invalid preserved tensor')
            self.weights[name].flags.writeable = False
        # Never allow an exported student to silently change the embedded source.
        for name, value in teacher['weights'].items():
            if not np.array_equal(self.weights['legacy.' + encoded_name(name)], value):
                raise ValueError('Frozen policy weights were modified')
        source = dict(source, weights=teacher['weights'],
                      observation_index=named_index_map(source['features'], current_features()),
                      candidate_index=named_index_map(source['candidates'], current_candidates()))
        self.teacher = source
        self.serving_deck = m['build']
        if build_hash(self.serving_deck) != m.get('build_sha256'):
            raise ValueError('Preserved build mismatch')

    def scores(self, x, c):
        if not len(c):
            return np.zeros(0, dtype=np.float32)
        mapped, unknown = project_candidates(c, self.teacher)
        if unknown.any():
            if not self.partial:
                raise ValueError('Unsupported frozen policy action')
            return architecture.scores_numpy(x, c, self.weights)
        old = teacher_scores(self.teacher, project_observation(x, self.teacher), mapped)
        return old + architecture.scores_numpy(x, c, self.weights)

    scores_only = scores

    def wdl(self, x, is_actor):
        return architecture.wdl_numpy(x, is_actor, separated.value_weights(self.weights) if self.separated else self.weights)

    def scores_value(self, x, c):
        p = self.wdl(x, True)
        return self.scores(x, c), float(p[2] - p[0])

    def validate_human(self, build):
        raise ValueError('Preserved research model has no website authorization')
