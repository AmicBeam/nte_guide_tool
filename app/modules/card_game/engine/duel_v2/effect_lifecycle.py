"""Turn-boundary effect expiry and shared natural-expiry notifications.

First remove every due modifier, then notify content in stable entity/effect order.
Explicit removal (leave/down/equipment replacement) never fires natural expiry.
"""
from app.modules.card_game.engine.events import GameEvent
from .modifiers import add_effect, deadline, prepare_clocks, tick, remaining

ULTIMATE_EFFECT = 'status.ultimate'


def schedule_ultimate(state, side, h, turns, phase):
    # A new ultimate replaces only its previous deadline, without ending the ability.
    h['effects'] = [e for e in h.get('effects', []) if e['definition_id'] != ULTIMATE_EFFECT]
    add_effect(h, ULTIMATE_EFFECT, 'status.ultimate', 1,
               source_entity_id=h.get('entity_id'), source_key='ultimate', label='终结', show_marker=False,
               expiry=deadline(state, side, 0 if phase == 'turn_end' else turns, phase),
               clear_on=('down', 'ultimate_end'))


def prepare_effects(state):
    prepare_clocks(state)
    from .flow import ultimate_spec
    for side, team in state['sides'].items():
        for cid, h in team['characters'].items():
            if (h.get('awakened') and h.get('ultimate_turns', 0) > 0
                    and not any(e['definition_id'] == ULTIMATE_EFFECT for e in h.get('effects', []))):
                schedule_ultimate(state, side, h, h['ultimate_turns'], ultimate_spec(cid)['expires'])


def expire_effects(state, side, phase):
    from .context import EffectContext
    from .flow import end_ultimate
    from .deferred import drain
    from .state import finished, log
    from app.modules.card_game.content.duel_v2.registry import fire
    due = tick(state, side, phase)
    for _, holder, effect in due:
        if effect['definition_id'] == ULTIMATE_EFFECT:
            end_ultimate(state, holder['side'], holder['id'])
    # Public countdown compatibility is derived from absolute deadlines.
    for owner, team in state['sides'].items():
        for h in team['characters'].values():
            for effect in h.get('effects', []):
                if effect['definition_id'] == ULTIMATE_EFFECT and effect['expiry']['phase'] == 'turn_start':
                    h['ultimate_turns'] = remaining(effect, state, owner)
    for holder_side, holder, effect in due:
        if finished(state):
            break
        # A shared trigger is observable by both teams; content filters the
        # source/holder entity IDs, rather than inheriting the dispatch actor.
        for observer_side, team in state['sides'].items():
            carrier = next(iter(team['characters']), None)
            if carrier is not None:
                context = EffectContext(state, observer_side, carrier, {'side': observer_side})
                fire(context, GameEvent.V2_EFFECT_EXPIRED,
                     effect=effect, holder_entity_id=holder.get('entity_id'))
        drain(state)
        if not finished(state) and effect.get('visibility') == 'public':
            log(state, f"{effect['label']}到期。", 'effect_expired', side=holder_side,
                actor=f"{holder_side}:{holder.get('id', 'player')}",
                effect_id=effect['effect_id'], definition_id=effect['definition_id'], present=False)
