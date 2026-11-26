"""Reuse public legal action payloads from one canonical observation.

This callable is a frozen-capture input. It is not a CUDA backend, is not
wired into a runtime, and must not be treated as a readiness or performance
claim. Public views use the current-rule observe surface with
include_previews=False; concede and all other legal actions are kept.
"""
from copy import deepcopy

from app.modules.card_game.engine.duel_v2.projection import observe as canonical_observe


def public_legal_decision_reuse(state, side):
    """Unfiltered current-rule legal actions and public observation.

    Returns (legal_actions, observe(..., include_previews=False)). Action
    payloads are taken from the observation legal entries so
    canonical_legal_actions is not called a second time. Full order is kept
    with no filtering or caps. Returned payloads are deep-copied so return[0]
    does not alias the public legal entries.
    """
    view = canonical_observe(state, side, include_previews=False)
    actions = []
    for entry in view['legal_actions']:
        actions.append(deepcopy(entry['action']))
    return actions, view
