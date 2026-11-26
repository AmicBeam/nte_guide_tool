"""Internal chance packet whose TWO exact encoded guards are mandatory.

No caller may use the world until both ordered float32 view pairs validate on
GPU. This callable is not a sampler API or a CPU fallback. Generation reuses
seven_sampler's private canonical transition, without duplicating game rules.
"""
from copy import deepcopy
from ..seven_sampler import _determinize_transition
from ...engine.duel_v2 import observe

REQUIRES_TWO_EXACT_GPU_GUARDS = True


def _public_pair(view):
    actions=[e['action'] for e in view['legal_actions'] if e['action']['type']!='concede']
    return actions,view


def seven_sample_guard_views_packet(state,viewer,seed):
    phase=state['phase']
    if phase not in ('playing','choice','mulligan'):
        raise ValueError('Unsupported information-search root phase')
    proxy=state
    if phase=='mulligan':
        proxy=deepcopy(state);proxy['phase']='playing';proxy['active_side']=viewer
    world,before=_determinize_transition(proxy,viewer,seed)
    after=observe(world,viewer,include_previews=False)
    left=_public_pair(before)
    # Canonical inner check encodes both views with the BEFORE ordered actions.
    inner=(left,(left[0],after))
    outer=inner
    if phase=='mulligan':
        world['phase']='mulligan';world['active_side']=state['active_side']
        outer=(_public_pair(observe(state,viewer,include_previews=False)),
               _public_pair(observe(world,viewer,include_previews=False)))
    return world,(inner,outer)
