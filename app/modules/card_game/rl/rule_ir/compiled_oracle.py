"""Python-engine oracle helpers for the compiled starter backend. No torch."""
from __future__ import annotations

from typing import Any

from app.modules.card_game.rl.gpu_duel.catalog import (
    ACT_ATTACK, ACT_CHOOSE, ACT_END, ACT_MULLIGAN, ACT_PLAY, ACT_ULTIMATE, CARD_INDEX,
    HAND_SLOTS, SEAT_INDEX,
)
from app.modules.card_game.rl.rule_ir.layout import OFFSETS, ROW_WIDTH
from app.modules.card_game.rl.rule_ir.pack_row import legal_entries, mechan_from_row, pack_python_row


def _instance_int(instance_id: str) -> int:
    if not instance_id or '-' not in str(instance_id):
        return 0
    return int(str(instance_id).rsplit('-', 1)[-1])


def rebuild(backend, row: list[int]) -> list[int]:
    return backend.launch.launch_lists([row], [-1])[0]


def map_python_action(row: list[int], state: dict[str, Any], action: dict[str, Any]) -> int:
    kind = action.get('type')
    entries = legal_entries(row)
    if kind == 'end_turn':
        for item in entries:
            if item['type'] == ACT_END:
                return item['index']
    elif kind == 'mulligan':
        ids = action.get('card_ids') or []
        selected = {_instance_int(card_id) for card_id in ids}
        side = row[OFFSETS['active']]
        mask = 0
        matched = 0
        for hs in range(HAND_SLOTS):
            if row[OFFSETS['hand_kind'] + side * HAND_SLOTS + hs] < 0:
                continue
            inst = row[OFFSETS['hand_inst'] + side * HAND_SLOTS + hs]
            if inst in selected:
                mask |= 1 << hs
                matched += 1
        if matched != len(selected):
            raise ValueError(f'Unknown mulligan hand instances: {action}')
        for item in entries:
            if item['type'] == ACT_MULLIGAN and item['hand'] == mask:
                return item['index']
    elif kind == 'attack':
        actor = SEAT_INDEX[action['character_id']]
        for item in entries:
            if item['type'] == ACT_ATTACK and item['actor'] == actor:
                return item['index']
    elif kind == 'ultimate':
        actor = SEAT_INDEX[action['character_id']]
        for item in entries:
            if item['type'] == ACT_ULTIMATE and item['actor'] == actor:
                return item['index']
    elif kind == 'choose':
        inst = _instance_int(action['choice_id'])
        pending_n = row[OFFSETS['pending_count']]
        slot = None
        for index in range(pending_n):
            if row[OFFSETS['pending_inst'] + index] == inst:
                slot = index
                break
        if slot is None:
            raise ValueError(f'Unknown pending choice instance: {action}')
        for item in entries:
            if item['type'] == ACT_CHOOSE and item['hand'] == slot:
                return item['index']
    elif kind == 'play_card':
        inst = _instance_int(action['card_id'])
        side = 0 if state.get('active_side') == 'a' else 1
        hand_slot = None
        for hs in range(HAND_SLOTS):
            if row[OFFSETS['hand_inst'] + side * HAND_SLOTS + hs] == inst:
                hand_slot = hs
                break
        target = action.get('target_id')
        want_side = -1
        want_seat = -1
        if target:
            tside, tname = str(target).split(':', 1)
            want_side = 0 if tside == 'a' else 1
            want_seat = -1 if tname == 'player' else SEAT_INDEX[tname]
        for item in entries:
            if item['type'] != ACT_PLAY or item['hand'] != hand_slot:
                continue
            if target:
                if item['tside'] == want_side and item['tseat'] == want_seat:
                    return item['index']
            elif item['tseat'] < 0 and item['tside'] < 0:
                return item['index']
    raise AssertionError(f'no compiled legal slot for {action}')


def step_python(backend, state: dict[str, Any], action: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    row = rebuild(backend, pack_python_row(state))
    index = map_python_action(row, state, action)
    nxt = backend.launch.launch_lists([row], [index])[0]
    if nxt[OFFSETS['error']]:
        raise RuntimeError(f'compiled engine error {nxt[OFFSETS["error"]]} on {action}')
    return nxt, mechan_from_row(nxt)


def python_packed_mechan(state: dict[str, Any], backend=None) -> dict[str, Any]:
    row = pack_python_row(state)
    if backend is not None:
        row = rebuild(backend, row)
    return mechan_from_row(row)


def canonical_row(row):
    """Compare all supported gameplay fields, ignoring unused capacity/scratch."""
    from .layout import GPU_STATE_SCALARS, SIDE_SCALARS, CHAR_FIELDS, LEGAL_FIELDS
    from app.modules.card_game.rl.gpu_duel.catalog import N_SEATS, SIDES, HAND_SLOTS, DECK_SLOTS, DISCARD_SLOTS
    out = {name: row[OFFSETS[name]] for name in GPU_STATE_SCALARS
           if name not in ('rng', 'pending_kind', 'pending_side', 'pending_count')}
    for name in SIDE_SCALARS:
        out[name] = row[OFFSETS[name]:OFFSETS[name] + SIDES]
    for name in CHAR_FIELDS:
        out[name] = row[OFFSETS[name]:OFFSETS[name] + SIDES * N_SEATS]
    for zone, capacity in (('hand', HAND_SLOTS), ('deck', DECK_SLOTS), ('discard', DISCARD_SLOTS)):
        for side in range(SIDES):
            n = row[OFFSETS[zone + '_n'] + side]
            out[f'{zone}:{side}'] = [(row[OFFSETS[zone + '_kind'] + side * capacity + k],
                                      row[OFFSETS[zone + '_inst'] + side * capacity + k]) for k in range(n)]
    if row[OFFSETS['phase']] == 0:
        out['pending_count'] = row[OFFSETS['pending_count']]
    if row[OFFSETS['phase']] == 2:
        for name in ('pending_kind', 'pending_side', 'pending_count', 'pending_play_kind', 'pending_play_inst'):
            out[name] = row[OFFSETS[name]]
        for name in ('pending_card', 'pending_inst'):
            out[name] = row[OFFSETS[name]:OFFSETS[name] + row[OFFSETS['pending_count']]]
    out['legal'] = [tuple(row[OFFSETS[name] + k] for name in LEGAL_FIELDS)
                    for k in range(row[OFFSETS['legal_n']]) if row[OFFSETS['legal_type'] + k] >= 0]
    out['error'] = row[OFFSETS['error']]
    return out
