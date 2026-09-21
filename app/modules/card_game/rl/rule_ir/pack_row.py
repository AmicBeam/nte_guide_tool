"""Pack Python engine JSON or a GpuState row into the compiled engine layout."""
from __future__ import annotations
from app.modules.card_game.engine.duel_v2.equipment import permanent_attack, unarmed_max_hp

from typing import Any

from app.modules.card_game.rl.gpu_duel.catalog import (
    CARD_INDEX, CHOICE_SLOTS, DECK_SLOTS, DISCARD_SLOTS, HAND_SLOTS, MAX_LEGAL,
    N_SEATS, SEAT_INDEX, SEATS, SIDES,
)
from .layout import CHAR_FIELDS, GPU_STATE_SCALARS, LEGAL_FIELDS, OFFSETS, ROW_WIDTH, SIDE_SCALARS

_PHASE = {'mulligan': 0, 'playing': 1, 'choice': 2, 'finished': 3}
_WIN = {None: -1, 'a': 0, 'b': 1, 'draw': 2}
_REASON = {None: 0, '': 0, 'player_hp': 1, 'empty_draw': 2, 'concede': 3}


def _instance_int(instance_id: str) -> int:
    if not instance_id or '-' not in str(instance_id):
        return 0
    return int(str(instance_id).rsplit('-', 1)[-1])


def empty_row() -> list[int]:
    row = [0] * ROW_WIDTH
    row[OFFSETS['winner']] = -1
    row[OFFSETS['pending_kind']] = -1
    row[OFFSETS['pending_side']] = -1
    row[OFFSETS['pending_play_kind']] = -1
    for slot in range(CHOICE_SLOTS):
        row[OFFSETS['pending_card'] + slot] = -1
        row[OFFSETS['pending_inst'] + slot] = 0
    for side in range(SIDES):
        row[OFFSETS['front'] + side] = -1
        row[OFFSETS['last_front'] + side] = -1
        row[OFFSETS['extra_genesis_actor'] + side] = -1
        for slot in range(HAND_SLOTS):
            row[OFFSETS['hand_kind'] + side * HAND_SLOTS + slot] = -1
            row[OFFSETS['hand_revealed'] + side * HAND_SLOTS + slot] = 0
        for slot in range(DECK_SLOTS):
            row[OFFSETS['deck_kind'] + side * DECK_SLOTS + slot] = -1
        for slot in range(DISCARD_SLOTS):
            row[OFFSETS['discard_kind'] + side * DISCARD_SLOTS + slot] = -1
        for seat in range(N_SEATS):
            row[OFFSETS['ch_shape'] + (side * N_SEATS + seat)] = -1
            row[OFFSETS['last_hit_side'] + (side * N_SEATS + seat)] = -1
            row[OFFSETS['last_hit_seat'] + (side * N_SEATS + seat)] = -1
            row[OFFSETS['ch_collapse_by'] + (side * N_SEATS + seat)] = -1
    for name in LEGAL_FIELDS:
        base = OFFSETS[name]
        for index in range(MAX_LEGAL):
            row[base + index] = -1
    return row


def pack_python_row(state: dict[str, Any]) -> list[int]:
    row = empty_row()
    row[OFFSETS['phase']] = _PHASE[state['phase']]
    row[OFFSETS['active']] = 0 if state.get('active_side') == 'a' else 1
    row[OFFSETS['first']] = 0 if state.get('first_side') == 'a' else 1
    row[OFFSETS['turn']] = int(state.get('turn') or 0)
    row[OFFSETS['winner']] = _WIN.get(state.get('winner'), -1)
    row[OFFSETS['reason']] = _REASON.get(state.get('reason'), 0)
    row[OFFSETS['rng']] = int(state.get('rng') or 0) & 0x7FFFFFFF
    row[OFFSETS['next_instance']] = int(state.get('next_instance') or 0)
    row[OFFSETS['escalation_enabled']] = 1 if state.get('escalation_enabled') else 0
    if state.get('phase') == 'mulligan':
        done = set(state.get('mulligan_done') or [])
        chooser = next((side for side in ('a', 'b') if side not in done), None)
        if chooser in ('a', 'b'):
            row[OFFSETS['active']] = 0 if chooser == 'a' else 1
        row[OFFSETS['pending_count']] = (1 if 'a' in done else 0) | (2 if 'b' in done else 0)
    pending = state.get('pending_choice')
    if pending:
        row[OFFSETS['pending_kind']] = {'inspect_top': 1, 'discard': 2, 'enemy_hand': 3}.get(pending.get('kind'), 0)
        row[OFFSETS['pending_side']] = 0 if pending.get('side') == 'a' else 1
        cards = pending.get('cards') or []
        row[OFFSETS['pending_count']] = min(len(cards), CHOICE_SLOTS)
        if len(cards) > CHOICE_SLOTS:
            raise ValueError('Choice overflow')
        for index, card in enumerate(cards[:CHOICE_SLOTS]):
            card_id = card.get('card_id') or card.get('id')
            if card_id not in CARD_INDEX:
                raise ValueError(f'Encoder cannot represent card {card_id}')
            row[OFFSETS['pending_card'] + index] = CARD_INDEX[card_id]
            row[OFFSETS['pending_inst'] + index] = _instance_int(card.get('instance_id') or '')
        settling = (state.get('operation') or {}).get('card')
        if settling:
            row[OFFSETS['pending_play_kind']] = CARD_INDEX[settling.get('card_id') or settling.get('id')]
            row[OFFSETS['pending_play_inst']] = _instance_int(settling.get('instance_id') or '')
    for side_i, side in enumerate(('a', 'b')):
        team = state['sides'][side]
        row[OFFSETS['hp'] + side_i] = int(team['hp'])
        row[OFFSETS['shield'] + side_i] = int(team['shield'])
        row[OFFSETS['ap'] + side_i] = int(team.get('ap') or 0)
        row[OFFSETS['front'] + side_i] = SEAT_INDEX[team['front']] if team.get('front') in SEAT_INDEX else -1
        row[OFFSETS['last_front'] + side_i] = SEAT_INDEX[team['last_front']] if state.get('harmony_residue_version') == 1 and team.get('last_front') in SEAT_INDEX else -1
        row[OFFSETS['turn_count'] + side_i] = int(team.get('turn_count') or 0)
        row[OFFSETS['fatigue'] + side_i] = int(team.get('fatigue') or 0)
        row[OFFSETS['normal_atk'] + side_i] = 1 if team.get('normal_attack_available') else 0
        row[OFFSETS['ultimate_ok'] + side_i] = 1 if team.get('ultimate_available') else 0
        row[OFFSETS['ultimates_used'] + side_i] = int(team.get('ultimates_used') or 0)
        row[OFFSETS['used_instant'] + side_i] = 1 if (team.get('used') or {}).get('instant') else 0
        row[OFFSETS['harmony_damage'] + side_i] = 1 if team.get('harmony_damage') else 0
        row[OFFSETS['extra_flower'] + side_i] = 1 if team.get('extra_flower') else 0
        row[OFFSETS['extra_ap'] + side_i] = int(team.get('extra_ap') or 0)
        row[OFFSETS['extra_genesis'] + side_i] = 1 if team.get('extra_genesis_pending') else 0
        actor = team.get('extra_genesis_actor')
        row[OFFSETS['extra_genesis_actor'] + side_i] = SEAT_INDEX.get(actor, -1)
        row[OFFSETS['surplus'] + side_i] = 1 if team.get('surplus') else 0
        row[OFFSETS['nanali_seed'] + side_i] = int(team.get('nanali_family') or 0)
        row[OFFSETS['nanali_seed_played'] + side_i] = int(team.get('nanali_family_played') or 0)
        burn = (team.get('front_debuff') or {}).get('burn') or {}
        delay = (team.get('front_debuff') or {}).get('delay') or {}
        row[OFFSETS['burn_left'] + side_i] = int(burn.get('left') or 0)
        row[OFFSETS['burn_infinite'] + side_i] = 1 if burn.get('infinite') else 0
        burn_by = burn.get('by')
        row[OFFSETS['burn_by'] + side_i] = 0 if burn_by == 'a' else 1 if burn_by == 'b' else -1
        row[OFFSETS['delay_end'] + side_i] = int(delay.get('end') or delay.get('left') or 0)
        weave = (team.get('front_debuff') or {}).get('weave')
        row[OFFSETS['weave'] + side_i] = 1 if weave else 0

        def fill(zone, kind_off, inst_off, limit, count_off):
            cards = team.get(zone) or []
            if len(cards) > limit:
                raise ValueError(f'{zone} overflow')
            row[count_off + side_i] = len(cards)
            for index, card in enumerate(cards):
                card_id = card.get('card_id') or card.get('id')
                if card_id not in CARD_INDEX:
                    raise ValueError(f'Encoder cannot represent card {card_id}')
                row[kind_off + side_i * limit + index] = CARD_INDEX[card_id]
                row[inst_off + side_i * limit + index] = _instance_int(card.get('instance_id') or '')

        fill('hand', OFFSETS['hand_kind'], OFFSETS['hand_inst'], HAND_SLOTS, OFFSETS['hand_n'])
        revealed = set(team.get('revealed_ids') or [])
        for index, card in enumerate((team.get('hand') or [])[:HAND_SLOTS]):
            row[OFFSETS['hand_revealed'] + side_i * HAND_SLOTS + index] = 1 if card.get('instance_id') in revealed else 0
        fill('deck', OFFSETS['deck_kind'], OFFSETS['deck_inst'], DECK_SLOTS, OFFSETS['deck_n'])
        fill('discard', OFFSETS['discard_kind'], OFFSETS['discard_inst'], DISCARD_SLOTS, OFFSETS['discard_n'])
        row[OFFSETS['removed_n'] + side_i] = len(team.get('removed') or [])
        for seat, cid in enumerate(SEATS):
            present = cid in team['characters']
            base = side_i * N_SEATS + seat
            row[OFFSETS['ch_present'] + base] = 1 if present else 0
            if not present:
                continue
            hero = team['characters'][cid]
            flags = hero.get('flags') or {}
            row[OFFSETS['ch_hp'] + base] = int(hero.get('hp') or 0)
            row[OFFSETS['ch_max_hp'] + base] = int(hero.get('max_hp') or 0)
            row[OFFSETS['ch_base_max'] + base] = unarmed_max_hp(hero)
            row[OFFSETS['ch_shield'] + base] = int(hero.get('shield') or 0)
            row[OFFSETS['ch_base_atk'] + base] = permanent_attack(hero)
            row[OFFSETS['ch_growth'] + base] = int(hero.get('growth') or 0)
            row[OFFSETS['ch_harmony'] + base] = int(hero.get('harmony') or 0)
            row[OFFSETS['ch_energy'] + base] = int(hero.get('energy') or 0)
            from app.modules.card_game.content.duel_v2.catalog import energy_max_for_attribute
            row[OFFSETS['ch_energy_max'] + base] = energy_max_for_attribute(hero.get('attribute'))
            row[OFFSETS['ch_down'] + base] = int(hero.get('down_turns') or 0)
            row[OFFSETS['ch_awakened'] + base] = 1 if hero.get('awakened') else 0
            row[OFFSETS['ch_ult_turns'] + base] = int(hero.get('ultimate_turns') or 0)
            row[OFFSETS['ch_ultimate_used_turn'] + base] = int(hero.get('ultimate_used_turn', -1))
            shape = hero.get('shape')
            row[OFFSETS['ch_shape'] + base] = CARD_INDEX.get(shape, -1) if shape else -1
            row[OFFSETS['ch_atk_buff'] + base] = int(flags.get('atk_buff') or 0)
            row[OFFSETS['ch_next_bonus'] + base] = int(flags.get('next_bonus') or 0)
            row[OFFSETS['ch_next_shield'] + base] = int(flags.get('next_shield') or 0)
            row[OFFSETS['ch_next_followup'] + base] = int(flags.get('next_followup') or 0)
            row[OFFSETS['ch_pact'] + base] = 1 if flags.get('pact') else 0
            row[OFFSETS['ch_energy_next'] + base] = 1 if flags.get('energy_next') else 0
            row[OFFSETS['ch_harmonized'] + base] = 1 if (team.get('harmonized') or {}).get(cid) else 0
            row[OFFSETS['ch_allied_hurt'] + base] = 1 if flags.get('allied_hurt') else 0
            row[OFFSETS['ch_pending_atk'] + base] = int(flags.get('pending_atk') or 0)
            row[OFFSETS['ch_share_overflow'] + base] = 1 if flags.get('share_overflow') else 0
            row[OFFSETS['ch_hp_floor'] + base] = 1 if flags.get('hp_floor') else 0
            order = list(team.get('order') or [])
            row[OFFSETS['ch_order'] + base] = order.index(cid) + 1 if cid in order else 0
            collapse = flags.get('collapse') or {}
            row[OFFSETS['ch_collapse_count'] + base] = int(flags.get('collapse_count') or 0)
            if collapse:
                row[OFFSETS['ch_collapse_by'] + base] = 0 if collapse.get('side') == 'a' else 1
                row[OFFSETS['ch_collapse_until'] + base] = int(collapse.get('until') or 0)
            else:
                row[OFFSETS['ch_collapse_by'] + base] = -1
                row[OFFSETS['ch_collapse_until'] + base] = 0
    return row


def pack_gpu_row(state, row: int = 0) -> list[int]:
    packed = empty_row()
    for name in GPU_STATE_SCALARS:
        packed[OFFSETS[name]] = int(getattr(state, name)[row])
    packed[OFFSETS['pending_play_kind']] = int(getattr(state, 'pending_play_kind')[row]) if hasattr(state, 'pending_play_kind') else -1
    packed[OFFSETS['pending_play_inst']] = int(getattr(state, 'pending_play_inst')[row]) if hasattr(state, 'pending_play_inst') else 0
    packed[OFFSETS['error']] = 0
    pending = int(state.pending_count[row])
    for index in range(CHOICE_SLOTS):
        packed[OFFSETS['pending_card'] + index] = int(state.pending_card[row, index])
        packed[OFFSETS['pending_inst'] + index] = int(getattr(state, 'pending_inst')[row, index]) if hasattr(state, 'pending_inst') else 0
    for name in SIDE_SCALARS:
        for side in range(SIDES):
            packed[OFFSETS[name] + side] = int(getattr(state, name)[row, side])
    for side in range(SIDES):
        for slot in range(HAND_SLOTS):
            packed[OFFSETS['hand_kind'] + side * HAND_SLOTS + slot] = int(state.hand_kind[row, side, slot])
            packed[OFFSETS['hand_inst'] + side * HAND_SLOTS + slot] = int(state.hand_inst[row, side, slot])
        for slot in range(DECK_SLOTS):
            packed[OFFSETS['deck_kind'] + side * DECK_SLOTS + slot] = int(state.deck_kind[row, side, slot])
            packed[OFFSETS['deck_inst'] + side * DECK_SLOTS + slot] = int(state.deck_inst[row, side, slot])
        for slot in range(DISCARD_SLOTS):
            packed[OFFSETS['discard_kind'] + side * DISCARD_SLOTS + slot] = int(state.discard_kind[row, side, slot])
            packed[OFFSETS['discard_inst'] + side * DISCARD_SLOTS + slot] = int(state.discard_inst[row, side, slot])
        for seat in range(N_SEATS):
            base = side * N_SEATS + seat
            for name in CHAR_FIELDS:
                packed[OFFSETS[name] + base] = int(getattr(state, name)[row, side, seat])
            packed[OFFSETS['last_hit_side'] + base] = -1
            packed[OFFSETS['last_hit_seat'] + base] = -1
        if hasattr(state, 'hand_revealed'):
            for slot in range(HAND_SLOTS):
                packed[OFFSETS['hand_revealed'] + side * HAND_SLOTS + slot] = int(state.hand_revealed[row, side, slot])
    for name in LEGAL_FIELDS:
        for index in range(MAX_LEGAL):
            packed[OFFSETS[name] + index] = int(getattr(state, name)[row, index])
    packed[OFFSETS['legal_n']] = int(state.legal_n[row])
    return packed


def unpack_gpu_row(state, packed: list[int], row: int = 0) -> None:
    for name in GPU_STATE_SCALARS:
        getattr(state, name)[row] = packed[OFFSETS[name]]
    if hasattr(state, 'pending_play_kind'):
        state.pending_play_kind[row] = packed[OFFSETS['pending_play_kind']]
        state.pending_play_inst[row] = packed[OFFSETS['pending_play_inst']]
    for index in range(CHOICE_SLOTS):
        state.pending_card[row, index] = packed[OFFSETS['pending_card'] + index]
        if hasattr(state, 'pending_inst'):
            state.pending_inst[row, index] = packed[OFFSETS['pending_inst'] + index]
    for name in SIDE_SCALARS:
        for side in range(SIDES):
            getattr(state, name)[row, side] = packed[OFFSETS[name] + side]
    for side in range(SIDES):
        for slot in range(HAND_SLOTS):
            state.hand_kind[row, side, slot] = packed[OFFSETS['hand_kind'] + side * HAND_SLOTS + slot]
            state.hand_inst[row, side, slot] = packed[OFFSETS['hand_inst'] + side * HAND_SLOTS + slot]
            if hasattr(state, 'hand_revealed'):
                state.hand_revealed[row, side, slot] = packed[OFFSETS['hand_revealed'] + side * HAND_SLOTS + slot]
        for slot in range(DECK_SLOTS):
            state.deck_kind[row, side, slot] = packed[OFFSETS['deck_kind'] + side * DECK_SLOTS + slot]
            state.deck_inst[row, side, slot] = packed[OFFSETS['deck_inst'] + side * DECK_SLOTS + slot]
        for slot in range(DISCARD_SLOTS):
            state.discard_kind[row, side, slot] = packed[OFFSETS['discard_kind'] + side * DISCARD_SLOTS + slot]
            state.discard_inst[row, side, slot] = packed[OFFSETS['discard_inst'] + side * DISCARD_SLOTS + slot]
        for seat in range(N_SEATS):
            base = side * N_SEATS + seat
            for name in CHAR_FIELDS:
                getattr(state, name)[row, side, seat] = packed[OFFSETS[name] + base]
    for name in LEGAL_FIELDS:
        for index in range(MAX_LEGAL):
            getattr(state, name)[row, index] = packed[OFFSETS[name] + index]
    state.legal_n[row] = packed[OFFSETS['legal_n']]


def mechan_from_row(packed: list[int]) -> dict[str, Any]:
    sides = []
    for side in range(SIDES):
        chars = []
        for seat in range(N_SEATS):
            base = side * N_SEATS + seat
            chars.append({
                'present': packed[OFFSETS['ch_present'] + base],
                'hp': packed[OFFSETS['ch_hp'] + base],
                'max_hp': packed[OFFSETS['ch_max_hp'] + base],
                'shield': packed[OFFSETS['ch_shield'] + base],
                'base_attack': packed[OFFSETS['ch_base_atk'] + base],
                'harmony': packed[OFFSETS['ch_harmony'] + base],
                'energy': packed[OFFSETS['ch_energy'] + base],
                'down': packed[OFFSETS['ch_down'] + base],
                'awakened': packed[OFFSETS['ch_awakened'] + base],
                'shape': packed[OFFSETS['ch_shape'] + base],
                'atk_buff': packed[OFFSETS['ch_atk_buff'] + base],
                'pact': packed[OFFSETS['ch_pact'] + base],
            })
        sides.append({
            'hp': packed[OFFSETS['hp'] + side], 'shield': packed[OFFSETS['shield'] + side],
            'ap': packed[OFFSETS['ap'] + side], 'front': packed[OFFSETS['front'] + side],
            'last_front': packed[OFFSETS['last_front'] + side],
            'turn_count': packed[OFFSETS['turn_count'] + side],
            'hand_n': packed[OFFSETS['hand_n'] + side], 'deck_n': packed[OFFSETS['deck_n'] + side],
            'discard_n': packed[OFFSETS['discard_n'] + side],
            'normal_atk': packed[OFFSETS['normal_atk'] + side],
            'ultimate_ok': packed[OFFSETS['ultimate_ok'] + side],
            'used_instant': packed[OFFSETS['used_instant'] + side],
            'seed': packed[OFFSETS['nanali_seed'] + side],
            'characters': chars,
        })
    return {
        'phase': packed[OFFSETS['phase']], 'active': packed[OFFSETS['active']],
        'turn': packed[OFFSETS['turn']], 'winner': packed[OFFSETS['winner']],
        'reason': packed[OFFSETS['reason']], 'sides': sides,
        'error': packed[OFFSETS['error']],
        'legal_n': packed[OFFSETS['legal_n']],
    }


def legal_entries(packed: list[int]) -> list[dict[str, int]]:
    n = packed[OFFSETS['legal_n']]
    out = []
    for index in range(n):
        typ = packed[OFFSETS['legal_type'] + index]
        if typ < 0:
            continue
        out.append({
            'index': index,
            'type': typ,
            'actor': packed[OFFSETS['legal_actor'] + index],
            'hand': packed[OFFSETS['legal_hand'] + index],
            'tside': packed[OFFSETS['legal_tside'] + index],
            'tseat': packed[OFFSETS['legal_tseat'] + index],
        })
    return out


def bind_gpu_rows(state, *, device=None):
    """Bind all mutable GpuState fields to one persistent int32 batch.

    No per-game Python packing in the stepping loop. Non-contiguous field views
    preserve the existing state API; the kernel receives the contiguous owner.
    """
    import torch
    if hasattr(state, '_compiled_rows'):
        return state._compiled_rows
    target = torch.device(device or state.device)
    if target != state.device:
        raise ValueError('Compiled state and executor must share a device')
    rows = torch.tensor(empty_row(), dtype=torch.int32, device=target).repeat(state.n, 1)
    names = (*GPU_STATE_SCALARS, 'pending_play_kind', 'pending_play_inst',
             'pending_card', 'pending_inst', *SIDE_SCALARS,
             'hand_kind', 'hand_inst', 'deck_kind', 'deck_inst',
             'discard_kind', 'discard_inst', 'hand_revealed', *CHAR_FIELDS, *LEGAL_FIELDS, 'legal_n')
    for name in names:
        old = getattr(state, name)
        width = old[0].numel()
        view = rows[:, OFFSETS[name]:OFFSETS[name] + width].view(old.shape)
        view.copy_(old)
        setattr(state, name, view)
    state._compiled_rows = rows
    return rows
