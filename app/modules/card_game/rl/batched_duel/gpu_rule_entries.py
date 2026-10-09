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
from app.modules.card_game.rl.batched_duel.gpu_schema_contract import (
    DEFAULT_RUNTIME,
    GPU_RUNTIME_CONTRACTS,
    GpuSchemaContractError,
    normalize_runtime,
)
from app.modules.card_game.rl.recovery_runtime import (
    decision as canonical_recovery_decision,
    determinize as canonical_recovery_determinize,
)
from app.modules.card_game.rl.seven_runtime import (
    decision as canonical_seven_decision,
    determinize as canonical_seven_determinize,
)


class _FrozenRecoveryRuntime:
    """Explicit sampler binding matching the existing resident serving backend.

    Class methods are captured as formal source references, rather than relying
    on a dynamically passed module's attributes being discovered by the freezer.
    This retains the recovery sampler's existing supported-roster restrictions.
    """
    runtime = 'recovery'
    decision = staticmethod(canonical_recovery_decision)
    determinize = staticmethod(canonical_recovery_determinize)


class _FrozenSevenRuntime:
    """Explicit sampler binding for the seven-team public encoder/sampler.

    Captured independently of the recovery wrapper so frozen identity cannot
    silently reuse the five-team sampler on collapse/lingke rosters.
    """
    runtime = 'seven'
    decision = staticmethod(canonical_seven_decision)
    determinize = staticmethod(canonical_seven_determinize)


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


def seven_acting_side(state):
    return canonical_acting_side(state)


def seven_finished(state):
    return canonical_finished(state)


def seven_terminal_value(state, viewer):
    return canonical_terminal_value(state, viewer)


def seven_public_legal_decision(state, side):
    view = canonical_observe(state, side, include_previews=False)
    return canonical_legal_actions(state, side), view


def seven_apply_action(state, side, action):
    return canonical_apply_action(state, side, action)


def seven_simulate_action(state, side, action):
    return canonical_simulate_action(state, side, action)


def seven_clean(state):
    return canonical_clean(state)


def seven_sample_world(state, viewer, seed):
    return canonical_sample_world(state, viewer, seed, runtime=_FrozenSevenRuntime)


def seven_serving_root_metadata(state, side):
    return canonical_root_metadata(state, side)


def seven_public_terminal_probe(state, side, action):
    return canonical_public_terminal_probe(state, side, action)


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

SEVEN_ENTRY_FUNCTIONS = {
    'acting_side': seven_acting_side,
    'finished': seven_finished,
    'terminal_value': seven_terminal_value,
    'public_legal_decision': seven_public_legal_decision,
    'apply_action': seven_apply_action,
    'simulate_action': seven_simulate_action,
    'clean': seven_clean,
    'sample_world': seven_sample_world,
    'serving_root_metadata': seven_serving_root_metadata,
    'public_terminal_probe': seven_public_terminal_probe,
}

ENTRY_FUNCTIONS_BY_RUNTIME = {
    'recovery': ENTRY_FUNCTIONS,
    'seven': SEVEN_ENTRY_FUNCTIONS,
}


def entry_functions(runtime=None):
    name = normalize_runtime(runtime)
    try:
        return ENTRY_FUNCTIONS_BY_RUNTIME[name]
    except KeyError as exc:
        raise GpuSchemaContractError('Unknown GPU runtime entries: %s' % name) from exc


def sampler_runtime(runtime=None):
    name = normalize_runtime(runtime)
    return _FrozenSevenRuntime if name == 'seven' else _FrozenRecoveryRuntime
