"""Internal frozen sampler transition packet, requiring an external exact check.

This is NOT a replacement sample_world API or a qualified sampler. The packet
cannot be consumed as a state by existing graph-call adapters. A collector must
check the complete original float32 observation/candidate arrays when flag=1
before accepting its world. Canonical sample_world remains the batch oracle.
"""
from copy import deepcopy
from .gpu_rule_entries import canonical_seven_determinize

REQUIRES_EXACT_OBSERVATION_VALIDATION = True


def seven_sample_transition_packet(state, viewer, seed):
    """Same canonical chance transition, with a mandatory-check packet flag.

    The encoded mulligan assertion from information_search.sample_world is
    performed by the collector's typed GPU encoder, never omitted or stubbed.
    RNG, original phase and active_side restoration match that source.
    """
    if state['phase'] in ('playing', 'choice'):
        return canonical_seven_determinize(state, viewer, seed), 0.0
    if state['phase'] != 'mulligan':
        raise ValueError('Unsupported information-search root phase')
    proxy = deepcopy(state)
    proxy['phase'] = 'playing'
    proxy['active_side'] = viewer
    world = canonical_seven_determinize(proxy, viewer, seed)
    world['phase'] = 'mulligan'
    world['active_side'] = state['active_side']
    return world, 1.0
