"""Synthetic/offline full-hand comparison. Never a serving or training runtime.

Only current opponent hand identities are exposed in a private diagnostic
view. Deck order remains sampled. No flags on the real rule state change.
"""
from copy import deepcopy
import numpy as np
from . import cross_runtime as base
from .cross_lineup import encode
from ..engine.duel_v2 import observe, acting_side


def diagnostic_view(state, viewer):
    view = observe(state, viewer, include_previews=False)
    foe = 'b' if viewer == 'a' else 'a'
    cards = observe(state, foe, include_previews=False)['sides'][foe]['hand']
    view['sides'][foe]['hand'] = [dict(card, revealed=True) for card in cards]
    return view


def decision(state, side):
    view = diagnostic_view(state, side)
    actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
    return actions, encode(view, actions)


def determinize(state, viewer, seed):
    foe = 'b' if viewer == 'a' else 'a'
    marked = deepcopy(state)
    marked['sides'][foe]['revealed_ids'] = [card['instance_id'] for card in marked['sides'][foe]['hand']]
    world = base.determinize(marked, viewer, seed)
    # Restore actual card metadata (including marks) and real visibility flags.
    # The sampler saw all hand identities as known, but never saw deck order.
    world['sides'][foe]['hand'] = deepcopy(state['sides'][foe]['hand'])
    if 'revealed_ids' in state['sides'][foe]:
        world['sides'][foe]['revealed_ids'] = deepcopy(state['sides'][foe]['revealed_ids'])
    else:
        world['sides'][foe].pop('revealed_ids', None)
    _, (x, c) = decision(state, viewer)
    _, (xx, cc) = decision(world, viewer)
    if not np.array_equal(x, xx) or not np.array_equal(c, cc):
        raise ValueError('Full-hand particle changed declared visible information')
    return world


def value_for(state, viewer, policies):
    x, _ = encode(diagnostic_view(state, viewer), [])
    p = policies[viewer].wdl(x, acting_side(state) == viewer)
    return float(p[2] - p[0])


predict_encoded = base.predict_encoded
action_scores = base.action_scores


def for_viewer(viewer):
    """Give only the searching observer hand access; opponents stay private."""
    if viewer not in ('a', 'b'):
        raise ValueError('Invalid privileged diagnostic observer')
    class Observer:
        def decision(self, state, side):
            return decision(state, side) if side == viewer else base.decision(state, side)

        def determinize(self, state, side, seed):
            if side != viewer:
                raise ValueError('Diagnostic hand privilege belongs to the fixed root observer')
            return determinize(state, side, seed)

        def value_for(self, state, side, policies):
            return value_for(state, side, policies) if side == viewer else base.value_for(state, side, policies)

        predict_encoded = staticmethod(base.predict_encoded)
        action_scores = staticmethod(base.action_scores)
    return Observer()
