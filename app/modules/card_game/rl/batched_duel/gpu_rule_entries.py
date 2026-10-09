"""Pure source-entry wrappers for canonical Duel V2 formal functions.

These callables are capture inputs for a frozen rule program. They are not a
CUDA backend and must not execute CPU rules under a GPU runtime claim.
Public views use the current-rule observe/legal_actions surface; they do not
encode NumPy features or truncate to a historical seat table.
"""
from app.modules.card_game.engine.duel_v2.flow import (
    acting_side as canonical_acting_side,
    apply_action as canonical_apply_action,
    legal_actions as canonical_legal_actions,
)
from app.modules.card_game.engine.duel_v2.projection import observe as canonical_observe
from app.modules.card_game.engine.duel_v2.simulation import simulate_action as canonical_simulate_action
from app.modules.card_game.engine.duel_v2.state import finished as canonical_finished
from app.modules.card_game.rl.batched_duel.serving_protocol import (
    public_terminal_probe as canonical_public_terminal_probe,
    root_metadata as canonical_root_metadata,
)
from app.modules.card_game.rl.information_search import (
    sample_world as canonical_sample_world,
    terminal_value as canonical_terminal_value,
)
from app.modules.card_game.rl.league_rollout import clean as canonical_clean
from app.modules.card_game.rl.recovery_runtime import (
    decision as canonical_recovery_decision,
    determinize as canonical_recovery_determinize,
)


class _FrozenRecoveryRuntime:
    """Explicit sampler binding matching the existing resident serving backend.

    Class methods are captured as formal source references, rather than relying
    on a dynamically passed module's attributes being discovered by the freezer.
    This retains the recovery sampler's existing supported-roster restrictions.
    """
    decision = staticmethod(canonical_recovery_decision)
    determinize = staticmethod(canonical_recovery_determinize)


def acting_side(state):
    return canonical_acting_side(state)


def finished(state):
    return canonical_finished(state)


def terminal_value(state, viewer):
    return canonical_terminal_value(state, viewer)


def public_legal_decision(state, side):
    """Unfiltered current-rule legal actions and public observation.

    Returns (legal_actions, observe(..., include_previews=False)). No NumPy
    encoding is performed; concede and all other legal actions are kept.
    """
    view = canonical_observe(state, side, include_previews=False)
    return canonical_legal_actions(state, side), view


def apply_action(state, side, action):
    return canonical_apply_action(state, side, action)


def simulate_action(state, side, action):
    return canonical_simulate_action(state, side, action)


def clean(state):
    return canonical_clean(state)


def sample_world(state, viewer, seed):
    return canonical_sample_world(state, viewer, seed, runtime=_FrozenRecoveryRuntime)


def serving_root_metadata(state, side):
    return canonical_root_metadata(state, side)


def public_terminal_probe(state, side, action):
    return canonical_public_terminal_probe(state, side, action)


CANONICAL_FUNCTIONS = {
    'acting_side': canonical_acting_side,
    'finished': canonical_finished,
    'terminal_value': canonical_terminal_value,
    'legal_actions': canonical_legal_actions,
    'public_view': canonical_observe,
    'apply_action': canonical_apply_action,
    'simulate_action': canonical_simulate_action,
    'clean': canonical_clean,
    'sample_world': canonical_sample_world,
    'serving_root_metadata': canonical_root_metadata,
    'public_terminal_probe': canonical_public_terminal_probe,
}

ENTRY_FUNCTIONS = {
    'acting_side': acting_side,
    'finished': finished,
    'terminal_value': terminal_value,
    'public_legal_decision': public_legal_decision,
    'apply_action': apply_action,
    'simulate_action': simulate_action,
    'clean': clean,
    'sample_world': sample_world,
    'serving_root_metadata': serving_root_metadata,
    'public_terminal_probe': public_terminal_probe,
}

FULL_ENTRY_NAMES = tuple(ENTRY_FUNCTIONS)
LIGHT_ENTRY_NAMES = (
    'acting_side',
    'finished',
    'terminal_value',
    'clean',
    'serving_root_metadata',
)
