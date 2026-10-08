"""Bounded private turn checkpoints. Transport identity never travels backwards."""
from copy import deepcopy

# Checkpoints contain game mechanics, never checkpoints, event history or receipts.
TRANSIENT = frozenset(('turn_snapshots', 'time_removed', 'events', 'event_seq', '_public_board',
    'version', 'current_action_id', 'current_action_version', 'combat_group', 'next_entity',
    'next_instance', 'operation', 'pending_choice', 'requests'))


def save_turn_start(state, side):
    if state['phase'] != 'playing' or not any(
            h.get('turn_snapshots') for h in state['sides'][side]['characters'].values()):
        return
    snapshot = deepcopy({k: v for k, v in state.items() if k not in TRANSIENT})
    history = state.setdefault('turn_snapshots', {}).setdefault(side, [])
    history.append(snapshot)
    del history[:-2]


def can_rewind(state, side):
    return len(state.get('turn_snapshots', {}).get(side, [])) >= 2


def rewind(state, side, card):
    from .state import log
    from .presentation import public_card
    if not can_rewind(state, side):
        raise ValueError('尚无上个己方回合的快照。')
    snapshot = deepcopy(state['turn_snapshots'][side][-2])
    ledger = state.setdefault('time_removed', {})
    ledger[card['entity_id']] = deepcopy(card)
    kept = {key: state[key] for key in TRANSIENT if key in state}
    histories = {ps: [s for s in rows if s['turn'] <= snapshot['turn']]
                 for ps, rows in state['turn_snapshots'].items()}
    state.clear()
    state.update(snapshot)
    state.update(kept)
    state.update(operation=None, pending_choice=None, phase='playing', turn_snapshots=histories)
    for ps, team in state['sides'].items():
        for zone in ('hand', 'deck', 'discard', 'removed'):
            team[zone] = [c for c in team[zone] if c['entity_id'] not in ledger]
        team['removed'].extend(deepcopy(c) for c in ledger.values() if c['side'] == ps)
    # A rewind updates both private hands; only the appropriate viewer receives each.
    log(state, '时间回溯至上个己方回合开始时。', 'rewind', side=side,
        actor=f'{side}:player')
    for ps, team in state['sides'].items():
        log(state, '回溯后的手牌已恢复。', 'effect', side=ps, present=False,
            private_side=ps, private_hand=[public_card(c) for c in team['hand']])
    log(state, f"「{card['name']}」移出游戏。", 'remove', side=side, card=public_card(card))
