"""Rebuild website-encoder observations from GPU tensors. No hidden deck/RNG."""
from __future__ import annotations

from typing import Any

import torch

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.rl.contracts import PHASES, RULES_VERSION
from app.modules.card_game.rl.encoding import ACTION_DIM, OBSERVATION_DIM, V2Encoder

from .catalog import (
    ACT_ATTACK, ACT_CHOOSE, ACT_CONCEDE, ACT_END, ACT_PLAY, ACT_ULTIMATE,
    CARD_IDS, HAND_SLOTS, MAX_LEGAL, N_SEATS, SEATS, SIDES,
)
from .state import GpuState

_PHASE = PHASES
_WIN = { -1: None, 0: 'a', 1: 'b', 2: 'draw' }
_TYPE_ORDER = {'tactic': 0, 'battle': 1, 'form': 2}


def _instance(kind: int, inst: int) -> str:
    return f'{CARD_IDS[kind]}-{inst}'


def _card_public(kind: int, inst: int) -> dict[str, Any]:
    card_id = CARD_IDS[kind]
    card = CARDS[card_id]
    return {
        'instance_id': _instance(kind, inst),
        'card_id': card_id,
        'character_id': card['character_id'],
        'name': card['name'],
        'type': card['type'],
        'cost': int(card.get('cost') or 0),
        'terminal': bool(card.get('terminal')),
        'copy': False,
        'derived': bool(card.get('derived')),
    }


def _attack(s: GpuState, row: int, side: int, seat: int) -> int:
    form = int(s.ch_shape[row, side, seat])
    base = int(s.ch_base_atk[row, side, seat])
    growth = int(s.ch_growth[row, side, seat])
    buff = int(s.ch_atk_buff[row, side, seat])
    perm = max(0, base - int(s.char_base_atk[seat]))
    if form >= 0:
        form_hp = int(s.catalog_hp[form])
        form_atk = int(s.catalog_atk[form])
        if form_hp > 0 and form_atk > 0:
            return form_atk + perm + buff
    return base + growth + buff


def reconstruct_observation(s: GpuState, row: int, viewer: int) -> dict[str, Any]:
    """Public observation matching V2Encoder keys. Events/logs stay empty."""
    phase = int(s.phase[row])
    active = int(s.active[row])
    winner = _WIN.get(int(s.winner[row]))
    viewer_side = 'ab'[viewer]
    obs: dict[str, Any] = {
        'rules_version': RULES_VERSION,
        'version': 0,
        'phase': _PHASE[phase] if 0 <= phase < len(_PHASE) else 'playing',
        'active_side': 'ab'[active],
        'viewer_side': viewer_side,
        'is_my_turn': viewer == active,
        'turn': int(s.turn[row]),
        'winner': winner,
        'sides': {},
        'pending_choice': None,
        'legal_actions': [],
        'events': [],
        'logs': [],
        'resolving_card': None,
    }
    for side in range(SIDES):
        chars = []
        for seat, cid in enumerate(SEATS):
            if int(s.ch_present[row, side, seat]) <= 0:
                continue
            catalog = CHARACTERS[cid]
            shape = int(s.ch_shape[row, side, seat])
            shape_id = CARD_IDS[shape] if shape >= 0 else None
            chars.append({
                'id': cid,
                'name': catalog['name'],
                'attribute': catalog['attribute'],
                'attack': _attack(s, row, side, seat),
                'hp': int(s.ch_hp[row, side, seat]),
                'max_hp': int(s.ch_max_hp[row, side, seat]),
                'shield': int(s.ch_shield[row, side, seat]),
                'harmony': int(s.ch_harmony[row, side, seat]),
                'energy': int(s.ch_energy[row, side, seat]),
                'awakened': bool(int(s.ch_awakened[row, side, seat])),
                'growth': int(s.ch_growth[row, side, seat]),
                'shape': CARDS[shape_id]['name'] if shape_id else None,
                'shape_id': shape_id,
                'down_turns': int(s.ch_down[row, side, seat]),
                'next_attack_bonus': int(s.ch_next_bonus[row, side, seat]),
            })
        hand = []
        for hs in range(int(s.hand_n[row, side])):
            kind = int(s.hand_kind[row, side, hs])
            if kind < 0:
                continue
            card = _card_public(kind, int(s.hand_inst[row, side, hs]))
            if side != viewer:
                card = {'hidden': True}
            hand.append(card)
        if side == viewer:
            hand.sort(key=lambda card: (
                SEATS.index(card['character_id']) if 'character_id' in card else 9,
                _TYPE_ORDER.get(card.get('type'), 9),
            ))
        discard = []
        for ds in range(int(s.discard_n[row, side])):
            kind = int(s.discard_kind[row, side, ds])
            if kind >= 0:
                discard.append(_card_public(kind, int(s.discard_inst[row, side, ds])))
        front = int(s.front[row, side])
        last = int(s.last_front[row, side])
        obs['sides']['ab'[side]] = {
            'name': '甲方' if side == 0 else '乙方',
            'hp': int(s.hp[row, side]),
            'shield': int(s.shield[row, side]),
            'ap': int(s.ap[row, side]),
            'normal_attack_available': bool(int(s.normal_atk[row, side])),
            'front': SEATS[front] if front >= 0 else None,
            'last_front': SEATS[last] if last >= 0 else None,
            'characters': chars,
            'hand': hand,
            'hand_count': int(s.hand_n[row, side]),
            'deck_count': int(s.deck_n[row, side]),
            'discard': discard,
            'records': [],
            'turn_count': int(s.turn_count[row, side]),
            'fatigue': int(s.fatigue[row, side]),
            'extra_flower': bool(int(s.extra_flower[row, side])),
            'used': {'instant': bool(int(s.used_instant[row, side]))},
            'harmonized': {SEATS[seat]: bool(int(s.ch_harmonized[row, side, seat])) for seat in range(N_SEATS)},
        }
    pending_kind = int(s.pending_kind[row])
    if pending_kind > 0:
        pside = int(s.pending_side[row])
        kind_name = {1: 'inspect_top', 2: 'discard', 3: 'enemy_hand'}.get(pending_kind, 'choose')
        choices = []
        if pside == viewer:
            for i in range(int(s.pending_count[row])):
                kind = int(s.pending_card[row, i])
                if kind < 0:
                    continue
                card = _card_public(kind, i)
                choices.append({'id': card['instance_id'], 'label': card['name'], 'card': card})
        obs['pending_choice'] = {
            'side': 'ab'[pside],
            'kind': kind_name,
            'choices': choices,
        }
    return obs


def reconstruct_actions(s: GpuState, row: int) -> list[dict[str, Any] | None]:
    actions: list[dict[str, Any] | None] = [None] * MAX_LEGAL
    side = int(s.active[row])
    for index in range(MAX_LEGAL):
        typ = int(s.legal_type[row, index])
        if typ < 0:
            continue
        if typ == ACT_END:
            actions[index] = {'type': 'end_turn'}
        elif typ == ACT_ATTACK:
            actions[index] = {'type': 'attack', 'character_id': SEATS[int(s.legal_actor[row, index])]}
        elif typ == ACT_ULTIMATE:
            actions[index] = {'type': 'ultimate', 'character_id': SEATS[int(s.legal_actor[row, index])]}
        elif typ == ACT_CONCEDE:
            actions[index] = {'type': 'concede'}
        elif typ == ACT_CHOOSE:
            slot = int(s.legal_hand[row, index])
            kind = int(s.pending_card[row, slot])
            if kind < 0:
                continue
            actions[index] = {'type': 'choose', 'choice_id': _instance(kind, slot)}
        elif typ == ACT_PLAY:
            hs = int(s.legal_hand[row, index])
            if hs < 0 or hs >= HAND_SLOTS:
                continue
            kind = int(s.hand_kind[row, side, hs])
            if kind < 0:
                continue
            action: dict[str, Any] = {
                'type': 'play_card',
                'card_id': _instance(kind, int(s.hand_inst[row, side, hs])),
            }
            tside = int(s.legal_tside[row, index])
            tseat = int(s.legal_tseat[row, index])
            if tside >= 0:
                if tseat < 0:
                    action['target_id'] = f"{'ab'[tside]}:player"
                else:
                    action['target_id'] = f"{'ab'[tside]}:{SEATS[tseat]}"
            actions[index] = action
    return actions


def encoder_observe(s: GpuState, encoder: V2Encoder | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = encoder or V2Encoder()
    cpu = s.to('cpu') if s.device.type != 'cpu' else s
    n = s.n
    state = torch.zeros(n, OBSERVATION_DIM, dtype=torch.float32)
    cand = torch.zeros(n, MAX_LEGAL, ACTION_DIM, dtype=torch.float32)
    mask = (s.legal_type >= 0)
    for row in range(n):
        viewer = int(cpu.active[row])
        obs = reconstruct_observation(cpu, row, viewer)
        state[row] = torch.tensor(encoder.encode_observation(obs), dtype=torch.float32)
        for index, action in enumerate(reconstruct_actions(cpu, row)):
            if action is None:
                continue
            try:
                cand[row, index] = torch.tensor(encoder.encode_action(obs, action), dtype=torch.float32)
            except ValueError:
                continue
    return state.to(s.device), cand.to(s.device), mask.to(s.device)
