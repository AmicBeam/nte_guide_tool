"""Versioned, observation-only residual scorer for learning recovery.

Normal-scale action features and state-conditioned scoring replace the tiny
ReLU action bottleneck. This is a new research architecture, not a migration
that claims to preserve old policy logits. No game rules or play bonuses.
"""
from hashlib import sha256
from pathlib import Path
import numpy as np
from . import cross_grounded as grounded
from .cross_lineup import all_features

SCHEMA = 'cross_five_residual_wdl_v2'
EPSILON = 1e-5


def fingerprint():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def network(hidden=64, device='cpu'):
    import torch
    from torch import nn
    if type(hidden) is not int or not 16 <= hidden <= 256:
        raise ValueError('Residual policy width must be an integer in [16, 256]')

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden, eps=EPSILON)
            self.first = nn.Linear(hidden, hidden)
            self.last = nn.Linear(hidden, hidden)
            # Start as identity; the input projection retains its normal scale.
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

        def forward(self, x):
            return x + self.last(torch.nn.functional.silu(self.first(self.norm(x))))

    def tower(width):
        return nn.Sequential(nn.LayerNorm(width, eps=EPSILON), nn.Linear(width, hidden),
                             nn.SiLU(), Residual(), Residual(), nn.LayerNorm(hidden, eps=EPSILON))

    class Scorer(nn.Module):
        grounded = True
        policy_schema = SCHEMA

        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.state_net = tower(len(all_features()))
            self.cand_net = tower(grounded.DIM)
            self.score = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.wdl = nn.Linear(hidden + 1, 3)
            self.register_buffer('actor_indices', torch.tensor(grounded.INDEX), persistent=False)

        def candidate_features(self, x, c):
            blocks = []
            for column, size in globals()['grounded'].CATEGORIES:
                ids = c[..., column].long() + (column != 0)
                blocks.append(torch.nn.functional.one_hot(ids, size).to(x.dtype))
            numeric = c[..., 5:].clone()
            numeric[..., :4] /= 10
            blocks.append(numeric)
            heroes = x[:, self.actor_indices]
            batch = torch.arange(len(x), device=x.device)[:, None]
            actors = c[..., 1].long(); sides, targets = grounded.context_targets_tensor(x, c)
            blocks.append(heroes[batch, 0, actors.clamp_min(0)] * (actors >= 0).unsqueeze(-1))
            blocks.append(heroes[batch, sides.clamp_min(0), targets.clamp_min(0)]
                          * ((targets >= 0) & (sides >= 0)).unsqueeze(-1))
            return torch.cat(blocks, -1)

        def forward(self, x, c, mask):
            state = self.state_net(x)
            cand = self.cand_net(self.candidate_features(x, c))
            context = state.unsqueeze(1).expand_as(cand)
            joint = torch.cat((cand, context, cand * context), -1)
            scores = self.score(joint).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            actor = torch.ones((len(x), 1), dtype=x.dtype, device=x.device)
            wdl = self.wdl(torch.cat((state, actor), -1)).softmax(-1)
            return scores, wdl[:, 2] - wdl[:, 0]

    return Scorer().to(device)


def tensor_shapes(hidden):
    if type(hidden) is not int or not 16 <= hidden <= 256:
        raise ValueError('Invalid residual width')
    shapes = {}
    for prefix, width in (('state_net', len(all_features())), ('cand_net', grounded.DIM)):
        shapes.update({f'{prefix}.0.weight': (width,), f'{prefix}.0.bias': (width,),
                       f'{prefix}.1.weight': (hidden, width), f'{prefix}.1.bias': (hidden,),
                       f'{prefix}.5.weight': (hidden,), f'{prefix}.5.bias': (hidden,)})
        for block in (3, 4):
            shapes.update({f'{prefix}.{block}.norm.weight': (hidden,),
                           f'{prefix}.{block}.norm.bias': (hidden,)})
            for layer in ('first', 'last'):
                shapes[f'{prefix}.{block}.{layer}.weight'] = (hidden, hidden)
                shapes[f'{prefix}.{block}.{layer}.bias'] = (hidden,)
    shapes.update({'score.0.weight': (hidden, 3 * hidden), 'score.0.bias': (hidden,),
                   'score.2.weight': (1, hidden), 'score.2.bias': (1,),
                   'wdl.weight': (3, hidden + 1), 'wdl.bias': (3,)})
    return shapes


def _linear(x, weights, name):
    return np.einsum('...j,ij->...i', x, weights[name + '.weight'], optimize=False) + weights[name + '.bias']


def _norm(x, weights, name):
    centered = x - x.mean(-1, keepdims=True)
    variance = np.mean(centered * centered, -1, keepdims=True)
    return centered / np.sqrt(variance + EPSILON) * weights[name + '.weight'] + weights[name + '.bias']


def _silu(x):
    # Stable sigmoid, preserving the unbounded positive SiLU branch.
    return x * np.exp(-np.logaddexp(0., -x))


def tower_numpy(x, weights, prefix):
    x = _silu(_linear(_norm(x, weights, prefix + '.0'), weights, prefix + '.1'))
    for block in (3, 4):
        name = f'{prefix}.{block}'
        branch = _silu(_linear(_norm(x, weights, name + '.norm'), weights, name + '.first'))
        x = x + _linear(branch, weights, name + '.last')
    return _norm(x, weights, prefix + '.5')


def scores_numpy(x, candidates, weights):
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.float32)
    state = tower_numpy(x, weights, 'state_net')
    cand = tower_numpy(grounded.transform(x, candidates), weights, 'cand_net')
    context = np.broadcast_to(state, cand.shape)
    joint = np.concatenate((cand, context, cand * context), -1)
    scores = _linear(_silu(_linear(joint, weights, 'score.0')), weights, 'score.2').reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite residual policy scores')
    return scores


def wdl_numpy(x, is_actor, weights):
    state = tower_numpy(x, weights, 'state_net')
    logits = _linear(np.append(state, np.float32(bool(is_actor))), weights, 'wdl')
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite residual WDL')
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()
