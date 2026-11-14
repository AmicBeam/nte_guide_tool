"""Harmony resources follow actual action-point payments, independently of combat."""
from app.modules.card_game.content.duel_v2.registry import decide
from .state import alive, collapsed, hero, log, upgrade_harmony_residue


def pay_harmony(c, amount):
    """Record one actual payment; effects may override its base resource yield."""
    if c.operation.get('harmony_payment_resolved'):
        return
    c.operation['harmony_payment_resolved'] = True
    c.operation['tracks_harmony_source'] = True
    from .tutorial import tutorial_flags
    if tutorial_flags(c.state).get('disable_harmony_gain'):
        return
    gained = max(0, int(decide(c, 'action_harmony_gain', int(amount),
                              actor_id=c.cid, ap_paid=int(amount)) or 0))
    if gained:
        c.grant_harmony(c.cid, gained, reason=f'消耗 {amount} 点行动力')


def finish_harmony_action(c):
    if not c.operation.get('tracks_harmony_source') or c.state['active_side'] != c.side:
        return
    upgrade_harmony_residue(c.state)
    team = c.state['sides'][c.side]
    team['last_front'] = c.cid if alive(c.state, c.side, c.cid) else None
    team['last_actor_entity_id'] = c.entity_id if team['last_front'] else None
    team['last_actor_turn'] = c.state['turn']
    log(c.state, '环合来源随行动更新。', 'effect', side=c.side,
        actor=f'{c.side}:{c.cid}', target=f'{c.side}:{c.cid}', present=False)


def clear_turn_source(state, side):
    team = state['sides'][side]
    team.update(last_front=None, last_actor_entity_id=None, last_actor_turn=None)


def source_id(state, side):
    if state.get('harmony_residue_version') != 3 or state['active_side'] != side:
        return None
    team = state['sides'][side]
    if team.get('last_actor_turn') is not None and team['last_actor_turn'] != state['turn']:
        return None
    cid = team.get('last_front')
    if not cid or not alive(state, side, cid) or collapsed(hero(state, side, cid)):
        return None
    entity = team.get('last_actor_entity_id')
    if entity and hero(state, side, cid).get('entity_id') != entity:
        return None
    from .escalation import harmony_cap
    return cid if hero(state, side, cid)['harmony'] >= harmony_cap(state) else None
