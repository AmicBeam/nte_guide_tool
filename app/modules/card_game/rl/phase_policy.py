"""Phase-isolated seven-team residual policy: opening/mulligan vs playing/choice.

Playing parameter names stay on the current seven residual scorer. Opening
copies only state_net/cand_net/score. The independent WDL tower is shared.
No game-rule, search, or reward changes.
"""
from hashlib import sha256
from pathlib import Path
import numpy as np
from . import seven_policy, seven_grounded as grounded
from .seven_lineup import all_features

SCHEMA = 'cross_seven_phase_isolated_residual_wdl_v1'
PHASES = ('opening', 'playing')
PHASE_INDEX = all_features().index('phase')
OPENING_CODE = 0
PLAYING_CODES = (1, 2, 3)


def fingerprint():
    digest = sha256(Path(__file__).read_bytes())
    digest.update((Path(__file__).parent / 'phase_runtime.py').read_bytes())
    digest.update(seven_policy.fingerprint().encode())
    return digest.hexdigest()


def tensor_shapes(hidden):
    shapes = dict(seven_policy.tensor_shapes(hidden))
    shapes.update({f'opening.{name}': shape for name, shape in seven_policy.tensor_shapes(hidden).items()
                   if name.startswith(('state_net.', 'cand_net.', 'score.'))})
    return shapes


def phase_of_observation(x):
    """Map the encoded public phase: mulligan -> opening, else playing/choice."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim == 1:
        return _phase_name(values[PHASE_INDEX])
    if values.ndim != 2:
        raise ValueError('Observation phase requires a vector or batch')
    return np.asarray([_phase_name(value) for value in values[:, PHASE_INDEX]])


def _phase_name(value):
    scaled = float(np.float32(value)) * 10
    if not np.isfinite(scaled) or not np.isclose(scaled, round(scaled), atol=1e-6, rtol=0):
        raise ValueError('Unknown observation phase')
    code = int(round(scaled))
    if code == OPENING_CODE:
        return 'opening'
    if code in PLAYING_CODES:
        return 'playing'
    raise ValueError('Unknown observation phase')


def _opening_rows(x):
    import torch
    code = torch.round(x[:, PHASE_INDEX] * 10)
    if bool((code < 0).any()) or bool((code > 3).any()):
        raise ValueError('Unknown observation phase')
    return code == 0


def scores_numpy(x, candidates, weights, phase):
    if phase not in PHASES:
        raise ValueError("phase must be 'opening' or 'playing'")
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.float32)
    prefix = 'opening.' if phase == 'opening' else ''
    state = seven_policy.tower_numpy(x, weights, prefix + 'state_net')
    cand = seven_policy.tower_numpy(grounded.transform(x, candidates), weights, prefix + 'cand_net')
    context = np.broadcast_to(state, cand.shape)
    joint = np.concatenate((cand, context, cand * context), -1)
    scores = seven_policy._linear(
        seven_policy._silu(seven_policy._linear(joint, weights, prefix + 'score.0')),
        weights, prefix + 'score.2').reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite phase policy scores')
    return seven_policy.canonical_scores_numpy(scores, candidates)


def wdl_numpy(x, is_actor, weights):
    return seven_policy.wdl_numpy(x, is_actor, weights)


def host_policy_metadata(candidates, source_identity=None):
    """Prepare tiny alias/support indexes before H2D; never read CUDA candidates back."""
    width = max(len(c) for c in candidates)
    aliases = np.tile(np.arange(width, dtype=np.int64), (len(candidates), 1))
    supported = []
    if source_identity:
        from .seven_lineup import candidate_names
        from .fixed_lineup import SEATS, CARD_IDS
        extra = [i for i, name in enumerate(candidate_names()) if name not in source_identity['candidates']]
    for i, c in enumerate(candidates):
        aliases[i, :len(c)] = seven_policy.alias_indices(c)
        if source_identity:
            unknown = ((c[:, 0] >= 6) | (c[:, 1] >= len(SEATS)) | (c[:, 2] >= len(CARD_IDS))
                       | (c[:, 4] >= len(SEATS)) | (np.abs(c[:, extra]) > 1e-8).any(-1))
            if not unknown.any():
                supported.append(i)
    return aliases, np.asarray(supported, dtype=np.int64) if source_identity else None


def create_network(hidden=128, device='cpu', key=None, legacy=False):
    import types
    import torch
    from copy import deepcopy
    from torch import nn
    if type(hidden) is not int or not 16 <= hidden <= 256:
        raise ValueError('Residual policy width must be an integer in [16, 256]')
    base = seven_policy.network(hidden, device)

    class PhaseScorer(nn.Module):
        grounded = True
        policy_schema = SCHEMA

        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.state_net = base.state_net
            self.cand_net = base.cand_net
            self.score = base.score
            self.value_net = base.value_net
            self.wdl = base.wdl
            self.register_buffer('actor_indices', base.actor_indices, persistent=False)
            opening = nn.Module()
            opening.state_net = deepcopy(self.state_net)
            opening.cand_net = deepcopy(self.cand_net)
            opening.score = deepcopy(self.score)
            self.add_module('opening', opening)

        candidate_features = type(base).candidate_features

        def _policy_modules(self, phase):
            if phase == 'opening':
                return self.opening.state_net, self.opening.cand_net, self.opening.score
            if phase == 'playing':
                return self.state_net, self.cand_net, self.score
            raise ValueError("phase must be 'opening' or 'playing'")

        def _branch_scores(self, x, c, mask, phase):
            state_net, cand_net, score = self._policy_modules(phase)
            state = state_net(x)
            cand = cand_net(self.candidate_features(x, c))
            context = state.unsqueeze(1).expand_as(cand)
            joint = torch.cat((cand, context, cand * context), -1)
            scores = score(joint).squeeze(-1)
            return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        def _legacy_scores(self, x, c, mask, supported_rows=None):
            module = getattr(self, 'legacy', None)
            if module is None:
                return None
            if supported_rows is None:
                return module(x, c, mask)
            result = x.new_zeros(mask.shape)
            if len(supported_rows) == 0:
                return result
            xx = x[supported_rows][:, module.observation_index]
            cc = c[supported_rows][..., module.candidate_index]
            state = module.linear(xx, 'state_net.0').relu()
            state = module.linear(state, 'state_net.2').relu()
            cand = module.linear(cc, 'cand_net.0').relu()
            cand = module.linear(cand, 'cand_net.2').relu()
            result[supported_rows] = ((cand * state.unsqueeze(1)).sum(-1)
                                      / np.sqrt(np.float32(self.source_identity['hidden']))
                                      + module.linear(cand, 'score').squeeze(-1))
            return result

        def policy_logits(self, x, c, mask, phase, *, aliases=None, legacy_rows=None):
            def canonical(scores):
                if aliases is None:
                    return seven_policy.canonical_scores_tensor(scores, c, mask)
                return scores.gather(1, aliases).masked_fill(~mask, torch.finfo(scores.dtype).min)
            scores = canonical(self._branch_scores(x, c, mask, phase))
            legacy = self._legacy_scores(x, c, mask, legacy_rows)
            return scores if legacy is None else canonical(scores + legacy)

        def _wdl_scalar(self, x):
            actor = torch.ones((len(x), 1), dtype=x.dtype, device=x.device)
            wdl = self.wdl(torch.cat((self.value_net(x), actor), -1)).softmax(-1)
            return wdl[:, 2] - wdl[:, 0]

        def forward_phase(self, x, c, mask, phase):
            return self.policy_logits(x, c, mask, phase), self._wdl_scalar(x)

        def forward(self, x, c, mask):
            opening = _opening_rows(x)
            scores = torch.full(mask.shape, torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
            values = self._wdl_scalar(x)
            if bool(opening.any()):
                scores[opening] = self.policy_logits(x[opening], c[opening], mask[opening], 'opening')
            playing = ~opening
            if bool(playing.any()):
                scores[playing] = self.policy_logits(x[playing], c[playing], mask[playing], 'playing')
            return scores, values

        def policy_parameters(self, phase):
            state_net, cand_net, score = self._policy_modules(phase)
            return list(state_net.parameters()) + list(cand_net.parameters()) + list(score.parameters())

        def value_parameters(self):
            return list(self.value_net.parameters()) + list(self.wdl.parameters())

    net = PhaseScorer().to(device)
    if legacy:
        if key is None:
            raise ValueError('Frozen teacher requires a seven-team key')
        seven_policy.attach_legacy(net, key)
        net.forward = types.MethodType(PhaseScorer.forward, net)
    return net
