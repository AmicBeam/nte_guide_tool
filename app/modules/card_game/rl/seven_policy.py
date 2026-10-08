"""Versioned, observation-only residual scorer for learning recovery.

Normal-scale action features and state-conditioned scoring replace the tiny
ReLU action bottleneck. This is a new research architecture, not a migration
that claims to preserve old policy logits. No game rules or play bonuses.
"""
from hashlib import sha256
from pathlib import Path
import numpy as np
from . import seven_grounded as grounded
from .seven_lineup import all_features

SCHEMA = 'cross_seven_decoupled_residual_wdl_v1'
EPSILON = 1e-5


def fingerprint():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def network(hidden=64, device='cpu'):
    import torch
    from torch import nn
    if type(hidden) is not int or not 16 <= hidden <= 256:
        raise ValueError('Residual policy width must be an integer in [16, 256]')

    class InputNorm(nn.LayerNorm):
        def __init__(self,width):
            super().__init__(width,eps=EPSILON)
            self.register_buffer('active',torch.ones(width))

        def forward(self,x):
            mean=(x*self.active).sum(-1,keepdim=True)/self.active.sum()
            variance=((x-mean).square()*self.active).sum(-1,keepdim=True)/self.active.sum()
            return (x-mean)*torch.rsqrt(variance+EPSILON)*self.weight+self.bias

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
        return nn.Sequential(InputNorm(width), nn.Linear(width, hidden),
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
            from copy import deepcopy
            self.value_net = deepcopy(self.state_net)
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
            wdl = self.wdl(torch.cat((self.value_net(x), actor), -1)).softmax(-1)
            return scores, wdl[:, 2] - wdl[:, 0]

    return Scorer().to(device)


def tensor_shapes(hidden):
    if type(hidden) is not int or not 16 <= hidden <= 256:
        raise ValueError('Invalid residual width')
    shapes = {}
    for prefix, width in (('state_net', len(all_features())), ('value_net', len(all_features())), ('cand_net', grounded.DIM)):
        shapes.update({f'{prefix}.0.active': (width,), f'{prefix}.0.weight': (width,), f'{prefix}.0.bias': (width,),
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
    active=weights.get(name+'.active')
    mean=x.mean(-1,keepdims=True) if active is None else (x*active).sum(-1,keepdims=True)/active.sum()
    centered=x-mean
    variance=np.mean(centered*centered,-1,keepdims=True) if active is None else (centered*centered*active).sum(-1,keepdims=True)/active.sum()
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
    state = tower_numpy(x, weights, 'value_net')
    logits = _linear(np.append(state, np.float32(bool(is_actor))), weights, 'wdl')
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite residual WDL')
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def attach_legacy(net, key):
    """Freeze the verified historical teacher; new identities use residual only."""
    from .preserved_policy import create_network
    from .cross_teacher_distillation import named_index_map
    from .seven_lineup import candidate_names
    import types
    import torch
    old, _ = create_network(key, hidden=net.hidden, partial=True)
    legacy = old.legacy.to(next(net.parameters()).device)
    source = dict(old.source_identity)
    source['observation_index'] = named_index_map(source['features'],all_features())
    source['candidate_index'] = named_index_map(source['candidates'],candidate_names())
    legacy.observation_index = torch.tensor(source['observation_index'],device=next(net.parameters()).device)
    legacy.candidate_index = torch.tensor(source['candidate_index'],device=next(net.parameters()).device)
    legacy.extra_index = torch.tensor([i for i,name in enumerate(candidate_names()) if name not in source['candidates']],
                                     dtype=torch.long,device=next(net.parameters()).device)
    net.add_module('legacy',legacy)
    net.source_identity = source
    original = type(net).forward
    def forward(self,x,c,mask):
        scores,value=original(self,x,c,mask)
        return (scores+self.legacy(x,c,mask)).masked_fill(~mask,torch.finfo(scores.dtype).min),value
    net.forward=types.MethodType(forward,net)
    return net
