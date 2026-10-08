"""Pack Python engine JSON into GPU tensors for oracle diffs."""
from __future__ import annotations
from app.modules.card_game.engine.duel_v2.equipment import permanent_attack, unarmed_max_hp

from typing import Any

from .catalog import (
    CARD_INDEX, DECK_SLOTS, DISCARD_SLOTS, HAND_SLOTS, N_SEATS, PHASE_CHOICE, PHASE_FINISHED,
    PHASE_MULLIGAN, PHASE_PLAYING, SEAT_INDEX, SEATS, SIDES,
)
from .state import GpuState, empty_state

_PHASE = {'mulligan': PHASE_MULLIGAN, 'playing': PHASE_PLAYING, 'choice': PHASE_CHOICE,
          'finished': PHASE_FINISHED}
_WIN = {None: -1, 'a': 0, 'b': 1, 'draw': 2}
_REASON = {None: 0, '': 0, 'player_hp': 1, 'empty_draw': 2, 'concede': 3}


def _instance_int(instance_id: str) -> int:
    if not instance_id or '-' not in str(instance_id):
        return 0
    return int(str(instance_id).rsplit('-', 1)[-1])


def _fill_zone(kind_t, inst_t, cards: list[dict], limit: int) -> int:
    count = min(len(cards), limit)
    for index in range(limit):
        kind_t[index] = -1
        inst_t[index] = 0
    overflow = len(cards) > limit
    if overflow:
        raise ValueError(f'Zone overflow: {len(cards)} > {limit}')
    for index, card in enumerate(cards[:count]):
        card_id = card.get('card_id') or card.get('id')
        kind_t[index] = CARD_INDEX[card_id]
        inst_t[index] = _instance_int(card.get('instance_id') or '')
    return count


def pack_python(states: list[dict[str, Any]], *, device='cpu') -> GpuState:
    gpu = empty_state(len(states), device)
    cpu = gpu.to('cpu') if gpu.device.type != 'cpu' else gpu
    for row, state in enumerate(states):
        cpu.phase[row] = _PHASE[state['phase']]
        cpu.active[row] = 0 if state.get('active_side') == 'a' else 1
        cpu.first[row] = 0 if state.get('first_side') == 'a' else 1
        cpu.turn[row] = int(state.get('turn') or 0)
        cpu.winner[row] = _WIN.get(state.get('winner'), -1)
        cpu.reason[row] = _REASON.get(state.get('reason'), 0)
        cpu.rng[row] = int(state.get('rng') or 0) & 0x7FFFFFFF
        cpu.next_instance[row] = int(state.get('next_instance') or 0)
        pending = state.get('pending_choice')
        if pending:
            cpu.pending_kind[row] = {'inspect_top': 1, 'discard': 2, 'enemy_hand': 3}.get(pending.get('kind'), 0)
            cpu.pending_side[row] = 0 if pending.get('side') == 'a' else 1
            cards = pending.get('cards') or []
            cpu.pending_count[row] = min(len(cards), 10)
            if len(cards) > 10:
                raise ValueError('Choice overflow')
            for index, card in enumerate(cards[:10]):
                cpu.pending_card[row, index] = CARD_INDEX[card.get('card_id') or card.get('id')]
                cpu.pending_inst[row, index] = _instance_int(card.get('instance_id') or '')
            settling = (state.get('operation') or {}).get('card')
            if settling:
                cpu.pending_play_kind[row] = CARD_INDEX[settling.get('card_id') or settling.get('id')]
                cpu.pending_play_inst[row] = _instance_int(settling.get('instance_id') or '')
        for side_i, side in enumerate(('a', 'b')):
            team = state['sides'][side]
            cpu.hp[row, side_i] = int(team['hp'])
            cpu.shield[row, side_i] = int(team['shield'])
            cpu.ap[row, side_i] = int(team.get('ap') or 0)
            cpu.front[row, side_i] = SEAT_INDEX[team['front']] if team.get('front') in SEAT_INDEX else -1
            cpu.last_front[row, side_i] = SEAT_INDEX[team['last_front']] if state.get('harmony_residue_version') == 1 and team.get('last_front') in SEAT_INDEX else -1
            cpu.turn_count[row, side_i] = int(team.get('turn_count') or 0)
            cpu.fatigue[row, side_i] = int(team.get('fatigue') or 0)
            cpu.normal_atk[row, side_i] = 1 if team.get('normal_attack_available') else 0
            cpu.ultimate_ok[row, side_i] = 1 if team.get('ultimate_available') else 0
            cpu.used_instant[row, side_i] = 1 if (team.get('used') or {}).get('instant') else 0
            cpu.extra_flower[row, side_i] = 1 if team.get('extra_flower') else 0
            cpu.extra_ap[row, side_i] = int(team.get('extra_ap') or 0)
            cpu.extra_genesis[row, side_i] = 1 if team.get('extra_genesis_pending') else 0
            actor = team.get('extra_genesis_actor')
            cpu.extra_genesis_actor[row, side_i] = SEAT_INDEX.get(actor, -1)
            cpu.surplus[row, side_i] = 1 if team.get('surplus') else 0
            cpu.nanali_seed[row, side_i] = int(team.get('nanali_family') or 0)
            cpu.nanali_seed_played[row, side_i] = int(team.get('nanali_family_played') or 0)
            burn = (team.get('front_debuff') or {}).get('burn') or {}
            delay = (team.get('front_debuff') or {}).get('delay') or {}
            cpu.burn_left[row, side_i] = int(burn.get('left') or 0)
            cpu.delay_end[row, side_i] = int(delay.get('end') or 0)
            cpu.hand_n[row, side_i] = _fill_zone(cpu.hand_kind[row, side_i], cpu.hand_inst[row, side_i], team.get('hand') or [], HAND_SLOTS)
            cpu.deck_n[row, side_i] = _fill_zone(cpu.deck_kind[row, side_i], cpu.deck_inst[row, side_i], team.get('deck') or [], DECK_SLOTS)
            cpu.discard_n[row, side_i] = _fill_zone(cpu.discard_kind[row, side_i], cpu.discard_inst[row, side_i], team.get('discard') or [], DISCARD_SLOTS)
            order = list(team.get('order') or SEATS)
            for seat, cid in enumerate(SEATS):
                if cid not in team['characters']:
                    cpu.ch_present[row, side_i, seat] = 0
                    continue
                cpu.ch_present[row, side_i, seat] = 1
                hero = team['characters'][cid]
                cpu.ch_hp[row, side_i, seat] = int(hero['hp'])
                cpu.ch_max_hp[row, side_i, seat] = int(hero['max_hp'])
                cpu.ch_base_max[row, side_i, seat] = unarmed_max_hp(hero)
                cpu.ch_shield[row, side_i, seat] = int(hero.get('shield') or 0)
                cpu.ch_base_atk[row, side_i, seat] = permanent_attack(hero)
                cpu.ch_growth[row, side_i, seat] = int(hero.get('growth') or 0)
                cpu.ch_harmony[row, side_i, seat] = int(hero.get('harmony') or 0)
                cpu.ch_energy[row, side_i, seat] = int(hero.get('energy') or 0)
                cpu.ch_energy_max[row, side_i, seat] = int(cpu.char_energy_max[seat])
                cpu.ch_down[row, side_i, seat] = int(hero.get('down_turns') or 0)
                cpu.ch_awakened[row, side_i, seat] = 1 if hero.get('awakened') else 0
                cpu.ch_ult_turns[row, side_i, seat] = int(hero.get('ultimate_turns') or 0)
                cpu.ch_ultimate_used_turn[row, side_i, seat] = int(hero.get('ultimate_used_turn', -1))
                shape = hero.get('shape')
                cpu.ch_shape[row, side_i, seat] = CARD_INDEX[shape] if shape in CARD_INDEX else -1
                flags = hero.get('flags') or {}
                cpu.ch_atk_buff[row, side_i, seat] = int(flags.get('atk_buff') or 0)
                cpu.ch_next_bonus[row, side_i, seat] = int(flags.get('next_bonus') or 0)
                cpu.ch_next_shield[row, side_i, seat] = int(flags.get('next_shield') or 0)
                cpu.ch_next_followup[row, side_i, seat] = int(flags.get('next_followup') or 0)
                cpu.ch_pact[row, side_i, seat] = 1 if flags.get('pact') else 0
                cpu.ch_energy_next[row, side_i, seat] = 1 if flags.get('energy_next') else 0
                cpu.ch_harmonized[row, side_i, seat] = 1 if (team.get('harmonized') or {}).get(cid) else 0
                cpu.ch_allied_hurt[row, side_i, seat] = 1 if flags.get('allied_hurt') else 0
                cpu.ch_pending_atk[row, side_i, seat] = int(flags.get('pending_atk') or 0)
                cpu.ch_share_overflow[row, side_i, seat] = 1 if flags.get('share_overflow') else 0
                collapse = flags.get('collapse') or {}
                cpu.ch_collapse_count[row, side_i, seat] = int(flags.get('collapse_count') or 0)
                if collapse:
                    cpu.ch_collapse_by[row, side_i, seat] = 0 if collapse.get('side') == 'a' else 1
                    cpu.ch_collapse_until[row, side_i, seat] = int(collapse.get('until') or 0)
                else:
                    cpu.ch_collapse_by[row, side_i, seat] = -1
                    cpu.ch_collapse_until[row, side_i, seat] = 0
            weave = (team.get('front_debuff') or {}).get('weave')
            cpu.weave[row, side_i] = 1 if weave else 0
            if order != list(SEATS):
                pass
    return cpu.to(device) if str(device) != 'cpu' else cpu


def snapshot_mechan(gpu: GpuState, row: int = 0) -> dict[str, Any]:
    """Compact mechanical snapshot for diffs (no events/logs)."""
    s = gpu.to('cpu') if gpu.device.type != 'cpu' else gpu
    sides = []
    for side in range(SIDES):
        chars = []
        for seat in range(N_SEATS):
            chars.append({
                'present': int(s.ch_present[row, side, seat]),
                'hp': int(s.ch_hp[row, side, seat]),
                'max_hp': int(s.ch_max_hp[row, side, seat]),
                'shield': int(s.ch_shield[row, side, seat]),
                'base_attack': int(s.ch_base_atk[row, side, seat]),
                'harmony': int(s.ch_harmony[row, side, seat]),
                'energy': int(s.ch_energy[row, side, seat]),
                'down': int(s.ch_down[row, side, seat]),
                'awakened': int(s.ch_awakened[row, side, seat]),
                'shape': int(s.ch_shape[row, side, seat]),
                'atk_buff': int(s.ch_atk_buff[row, side, seat]),
                'pact': int(s.ch_pact[row, side, seat]),
            })
        sides.append({
            'hp': int(s.hp[row, side]), 'shield': int(s.shield[row, side]),
            'ap': int(s.ap[row, side]), 'front': int(s.front[row, side]),
            'last_front': int(s.last_front[row, side]),
            'turn_count': int(s.turn_count[row, side]),
            'hand_n': int(s.hand_n[row, side]), 'deck_n': int(s.deck_n[row, side]),
            'discard_n': int(s.discard_n[row, side]),
            'normal_atk': int(s.normal_atk[row, side]),
            'ultimate_ok': int(s.ultimate_ok[row, side]),
            'used_instant': int(s.used_instant[row, side]),
            'seed': int(s.nanali_seed[row, side]),
            'characters': chars,
        })
    return {
        'phase': int(s.phase[row]), 'active': int(s.active[row]), 'turn': int(s.turn[row]),
        'winner': int(s.winner[row]), 'reason': int(s.reason[row]),
        'sides': sides,
    }


def python_mechan(state: dict[str, Any]) -> dict[str, Any]:
    packed = pack_python([state], device='cpu')
    return snapshot_mechan(packed, 0)
