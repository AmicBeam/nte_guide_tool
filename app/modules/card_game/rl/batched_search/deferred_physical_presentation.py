"""Fail-closed contract for clean-only, raw-recorded physical game loops.

This admission check changes no readiness/qualification flag. Public rooms and
public event consumers must continue using ordinary physical apply_action.
"""
import ast
from contextvars import ContextVar
from hashlib import sha256
import inspect
import textwrap

# Numeric image identities bind the entire frozen program, not same-named
# metadata. Unknown images require renewed source and transition validation.
_IMAGES=frozenset((
    ('00174ebf90f63f10ed6bd4cd4ff3c77bedd0a93fef9785668ed3eb8a1c1333e4',
     'bc881a9da0c4e95bb0f4408eee0ce4019a7f46fde4607f7f7549890097626bbb'),
    ('a7fbbe222097b9197b5bc06f67b7381d67c4810b4097d7815b20326ec106801a',
     '790b4afc3e31b136e2838ec4dab3112260ffe96a61176c6f37d7f79aa2a8f91c')))
_AST=(
    '52cd7a15d2d6558463453f7db63a5796b860288b65adacf728e3593ab3391f69',
    '8570c623cf130d28f66969ea0a68e7989c1a7d61a420ca9b8372a1b98fc634e1',
    'f0a7e86e5ad9250ff416b64823b4defff95ecc9b1e2f032e928e4355ce4cb3e0',
    '35f6e67b573b9b21c4dce3ced7e5e32120d12db7f5e69d37d2174d2f48898f18',
    '78411c9c09f8f0af01db6e2c86e10ad08bc3693632f0103d77a476bb082c81eb',
    '481175d3d52803af6ded4ef15496b2e177b6b01bca119c1a333f0cb3af37bd27',
    'b9f7de404fca4a091477bbceace8affba9fe2fed4d4c5f4b0b3e3b10fc1bfdc0',
    'd145b76526a1440caa6702104cf7b521b80338604163c481ad6ee0364a8c703e',
    '5f0db35d93d624956b50a83267012ac97847e58c2d86d45a10d47c93c608653d',
    'd26d4db9a1606d52b36b00ec5cc82c05d5b75d5357391ae8e2c5bde4b2405d23',
    'd48f074d4827d5d425f8c05a82f3ddc5b518677acb058dd27541c067129bc328',
    'ffa919bfeb0de4c05ce738c116376717d522f5cd7329115fd032312fc2e63f78',
    'feb82fa301361023a8e618b3926906c4b0d009a0a359a264e224503000ddfb7f',
    '8428e0bd3310b9b71be7b0254e426fdb760c3cff434508a287160d597cc20507')


def _current_contract():
    # Setup/static CUDA graph construction only. GPU WHILE replay never calls
    # Python rule methods, inspect, or this source admission function.
    from app.modules.card_game.engine.duel_v2 import flow,state,entities,simulation,presentation
    from ..league_rollout import clean
    from .device_games import DeviceGameLoop
    functions=(flow.apply_action,flow.apply_owned_action,state.log,entities.clone_state,
        simulation.simulate_action,simulation.collecting_patches,simulation._without_patches,
        presentation.capture_public_board,presentation.bump_public_patch,presentation.enrich_log,
        clean,DeviceGameLoop._commit_turn,DeviceGameLoop._cross_alias,DeviceGameLoop._error)
    try:
        actual=tuple(sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))),
            include_attributes=False).encode()).hexdigest() for fn in functions)
    except (TypeError,OSError,SyntaxError) as error:
        raise ValueError('Unknown canonical deferred-presentation source shape') from error
    if actual!=_AST:
        raise ValueError('Unknown canonical deferred-presentation source shape')
    variable=simulation._collect_patches
    if (not isinstance(variable,ContextVar) or variable.name!='duel_v2_collect_presentation_patches'
            or variable.get() is not True or presentation.collecting_patches is not simulation.collecting_patches):
        raise ValueError('Canonical presentation ContextVar boundary changed')
    return actual


def validate_deferred_rules(rules):
    """Static admission only; no rule execution, graph or GPU qualification."""
    from .device_soa_rules import DeviceSoARules
    from .device_tiled_soa_rules import DeviceTiledSoARules
    from .device_tiered_soa_rules import DeviceTieredSoARules
    from .device_raw_action_tape import DeviceRawActionTape
    from ..batched_duel.vm_image import VmImage
    if type(rules) not in (DeviceSoARules,DeviceTiledSoARules,DeviceTieredSoARules):
        raise TypeError('Explicit native SoA adapter required for deferred physical steps')
    raw=getattr(rules,'raw_tape',None)
    if (not isinstance(raw,DeviceRawActionTape) or raw.rows!=rules.rows or raw.device!=rules.device
            or getattr(raw,'closed',False)):
        raise ValueError('Live matching raw action tape required for deferred physical steps')
    if getattr(rules,'closed',False) or getattr(rules.executor,'closed',False):
        raise ValueError('Closed deferred physical adapter')
    if getattr(rules.executor,'execute_only',True) is not False:
        raise ValueError('Full native module required for deferred physical steps')
    image=rules.executor.image
    if not isinstance(image,VmImage) or (image.program_identity,image.identity) not in _IMAGES:
        raise ValueError('Unknown frozen deferred physical rule image')
    image.verify()
    for entry in ('apply_action','simulate_action','clean'):
        index=image.entries.get(entry)
        prefix='func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:seven_'+entry+':'
        if type(index) is not int or not 0<=index<len(image.function_ids) or not image.function_ids[index].startswith(prefix):
            raise ValueError('Exact frozen canonical '+entry+' entry required')
    _current_contract()
    return (image.program_identity,image.identity)


def validate_deferred_binding(rules,loop):
    from .device_games import DeviceGameLoop
    from .device_tiered_soa_rules import DeviceTieredSoARules
    if type(loop) is not DeviceGameLoop or loop.defer_physical_presentation is not True:
        raise ValueError('Explicit enabled physical DeviceGameLoop binding required')
    owner=loop.rules
    if owner is not rules and not (type(owner) is DeviceTieredSoARules and owner.slow is rules):
        raise ValueError('Deferred physical loop owns a different rule adapter')
    contract=validate_deferred_rules(rules)
    if (loop._deferred_physical_contract!=contract or loop._deferred_raw_tape is not rules.raw_tape
            or loop.rows!=rules.rows or loop.device!=rules.device):
        raise ValueError('Deferred physical loop source or cohort binding changed')


def require_deferred_step(rules):
    loop=getattr(rules,'physical_loop',None)
    if loop is None:
        raise ValueError('Bind an enabled physical loop before deferred steps')
    validate_deferred_binding(rules,loop)
    if getattr(loop,'_deferred_step_open',False) is not True:
        raise ValueError('Deferred step is restricted to the immediate-clean physical commit')
