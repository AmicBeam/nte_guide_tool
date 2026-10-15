"""Fixed-plan, observer-only Zhenhong guide for frozen device execution.

This is root guidance, never a leaf value or an environment reward. The three
sampled worlds are for hidden-information robustness, not three teacher moves.
"""
from copy import deepcopy
from app.modules.card_game.engine.ai.zhenhong_guide import _opening_keep, plan_worlds
from app.modules.card_game.engine.duel_v2 import legal_actions, acting_side, observe
from app.modules.card_game.rl.seven_sampler import _determinize_transition


def guide_index(state, side, baseline):
    legal = [a for a in legal_actions(state, side) if a['type'] != 'concede']
    if baseline not in legal:
        raise ValueError('Guide baseline must be in the complete legal root')
    own = state['sides'][side]
    red = own['characters'].get('zhenhong')
    if (state['phase'] not in ('playing', 'mulligan') or acting_side(state) != side
            or not red or red['hp'] <= 0 or red['awakened'] or 'yi' not in own['characters']):
        return legal.index(baseline)
    if state['phase'] == 'mulligan':
        selected, _ = _opening_keep(own['hand'], baseline)
    else:
        # Public deterministic seed, independent of hidden true cards/ordering.
        seed = 1729 + int(state['version']) * 1000003
        worlds = [guide_sample_world(state, side, seed + 104729 * i) for i in range(3)]
        result = dict(selected=deepcopy(baseline), changed=False, plans=0,
                      worlds=0, reason='not_applicable', seconds=0.)
        selected = plan_worlds(state, side, baseline, worlds, result, 24)['selected']
    if selected not in legal:
        raise ValueError('Guide returned an illegal root action')
    return legal.index(selected)


def guide_sample_world(state, side, seed):
    """Canonical playing transition with a stronger full-public-view guard.

    Identical entire raw public views (including ordered legal actions) imply
    identical deterministic observation/candidate encodings. Reject any raw
    difference rather than materialize the large NumPy encoder in the VM.
    This entry is only for playing roots; mulligan guidance samples no worlds.
    """
    if state['phase'] != 'playing':
        raise ValueError('Guide sampling requires a playing root')
    world, before = _determinize_transition(state, side, seed)
    after = observe(world, side, include_previews=False)
    if before != after:
        raise ValueError('Guide sample changed full public information')
    return world
