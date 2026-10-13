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
    new_game as canonical_new_game,
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
from app.modules.card_game.rl.covered_rollout import rule_decision as canonical_rule_decision
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


def rule_decision(state, actor):
    """Canonical public-rule teacher decision. Capture input, not a CUDA runtime."""
    return canonical_rule_decision(state, actor)


def seven_rule_decision(state, actor):
    return canonical_rule_decision(state, actor)


def initialize(parameters):
    """Frozen capture entry for canonical setup; no host game initializer.

    Parameters are batch ingress data: the actual seed, both fixed builds,
    first side and declared mulligan/escalation settings. Initialization must
    execute in the VM and pass independent full-state/RNG/entity parity.
    """
    return canonical_new_game(seed=parameters['seed'],decks=parameters['decks'],
        first_side=parameters['first_side'],
        skip_mulligan=parameters.get('skip_mulligan',False),
        escalation=parameters.get('escalation',True))


def seven_initialize(parameters):
    return initialize(parameters)


def teacher_continue(state):
    """Exact three-step rule continuation used by covered-value leaves."""
    from copy import deepcopy
    world=deepcopy(state)
    for _ in range(3):
        if canonical_finished(world):break
        actor=canonical_acting_side(world)
        legal,index=canonical_rule_decision(world,actor)
        world=canonical_clean(canonical_simulate_action(world,actor,legal[index]))
    return world


def seven_teacher_continue(state):
    return teacher_continue(state)


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
    'rule_decision': canonical_rule_decision,
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
TEACHER_ENTRY_NAMES = ('rule_decision',)
ENTRY_FUNCTIONS['rule_decision'] = rule_decision
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
SEVEN_ENTRY_FUNCTIONS['rule_decision'] = seven_rule_decision

ENTRY_FUNCTIONS_BY_RUNTIME = {
    'recovery': ENTRY_FUNCTIONS,
    'seven': SEVEN_ENTRY_FUNCTIONS,
}

# Explicit device-setup extension. Historical rule-parity entry sets stay
# unchanged; a closed driver must freeze and validate this additional entry.
DEVICE_SETUP_ENTRY_NAMES=('initialize',)
DEVICE_CONTINUATION_ENTRY_NAMES=('teacher_continue',)
ENTRY_FUNCTIONS['initialize']=initialize
SEVEN_ENTRY_FUNCTIONS['initialize']=seven_initialize
ENTRY_FUNCTIONS['teacher_continue']=teacher_continue
SEVEN_ENTRY_FUNCTIONS['teacher_continue']=seven_teacher_continue


def seven_teacher_prepare(state, actor):
    from .gpu_teacher_preview_parts import teacher_prepare
    return teacher_prepare(state, actor)


def seven_teacher_preview(state, actor, action):
    from .gpu_teacher_preview_parts import teacher_preview
    return teacher_preview(state, actor, action)


# Explicit standalone batch-teacher extension. These endpoints execute the
# same formal public sandbox; adding them does not qualify a device driver.
SEVEN_ENTRY_FUNCTIONS['teacher_prepare'] = seven_teacher_prepare
SEVEN_ENTRY_FUNCTIONS['teacher_preview'] = seven_teacher_preview
ENTRY_FUNCTIONS['teacher_prepare'] = seven_teacher_prepare
ENTRY_FUNCTIONS['teacher_preview'] = seven_teacher_preview
DEVICE_BATCH_TEACHER_ENTRY_NAMES = ('teacher_prepare', 'teacher_preview')

from .gpu_teacher_prepared_template import teacher_prepare_with_template,teacher_preview_with_template
for _entries in (SEVEN_ENTRY_FUNCTIONS,ENTRY_FUNCTIONS):
    _entries['teacher_prepare_with_template']=teacher_prepare_with_template
    _entries['teacher_preview_with_template']=teacher_preview_with_template


def entry_functions(runtime=None):
    name = normalize_runtime(runtime)
    try:
        return ENTRY_FUNCTIONS_BY_RUNTIME[name]
    except KeyError as exc:
        raise GpuSchemaContractError('Unknown GPU runtime entries: %s' % name) from exc


def sampler_runtime(runtime=None):
    name = normalize_runtime(runtime)
    return _FrozenSevenRuntime if name == 'seven' else _FrozenRecoveryRuntime
