"""Serializable extra actions, drained only after the current operation completes."""
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.content.duel_v2.registry import fire
from .state import alive, finished, hero


def enqueue(c, kind):
    c.state.setdefault('deferred_actions', []).append({
        'side': c.side, 'actor': c.cid, 'entity_id': c.entity_id, 'kind': kind,
    })


def drain(state):
    if state.get('_draining_actions') or state.get('_response_lock') or state.get('operation'):
        return
    from .flow import operation, finish_operation
    state['_draining_actions'] = True
    try:
        while state.get('deferred_actions') and not finished(state) and state['phase'] != 'choice':
            action = state['deferred_actions'].pop(0)
            side, cid = action['side'], action['actor']
            if not alive(state, side, cid) or hero(state, side, cid).entity_id != action['entity_id']:
                continue
            c = operation(state, side, cid)
            fire(c, GameEvent.V2_DEFERRED_ACTION, actor_only=True, action=action)
            finish_operation(state)
    finally:
        state.pop('_draining_actions', None)
    if finished(state):
        state['deferred_actions'] = []
