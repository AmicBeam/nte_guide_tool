"""Batched 创生预组 rules on integer tensors. No JSON, no presentation."""
from __future__ import annotations

import torch

from .catalog import (
    ACT_ATTACK, ACT_CHOOSE, ACT_CONCEDE, ACT_END, ACT_PLAY, ACT_ULTIMATE, CARD_INDEX,
    HAND_SLOTS, HARMONY_GENESIS, MAX_LEGAL, N_SEATS, PHASE_CHOICE, PHASE_FINISHED,
    PHASE_PLAYING, SEAT_INDEX, SIDES, TARGET_NONE, TARGET_PLAYER, TARGET_SEAT,
)
from .state import GpuState, other_side

NANALI, ILOY, ZERO, JIUYUAN = 0, 1, 2, 3
N03 = CARD_INDEX['N03']
N07 = CARD_INDEX.get('N07', -1)
NF01 = CARD_INDEX.get('NF01', -1)
Y03 = CARD_INDEX.get('Y03', -1)
Y02 = CARD_INDEX.get('Y02', -1)
Y06 = CARD_INDEX.get('Y06', -1)
Y08 = CARD_INDEX.get('Y08', -1)
Z02 = CARD_INDEX.get('Z02', -1)
Z07 = CARD_INDEX.get('Z07', -1)
Z08 = CARD_INDEX.get('Z08', -1)
J01 = CARD_INDEX.get('J01', -1)
J02 = CARD_INDEX.get('J02', -1)
J04 = CARD_INDEX.get('J04', -1)
J08 = CARD_INDEX.get('J08', -1)


def _b(s: GpuState) -> torch.Tensor:
    return torch.arange(s.n, device=s.device)


def _live(s: GpuState) -> torch.Tensor:
    return s.phase != PHASE_FINISHED


def _alive(s: GpuState, side: torch.Tensor, seat: int) -> torch.Tensor:
    return s.ch_hp[_b(s), side, seat] > 0


def attack_power(s: GpuState, side: torch.Tensor, seat: int) -> torch.Tensor:
    form = s.ch_shape[_b(s), side, seat]
    idx = form.clamp(min=0)
    form_hp = torch.where(form >= 0, s.catalog_hp[idx], torch.zeros_like(form))
    form_atk = torch.where(form >= 0, s.catalog_atk[idx], torch.zeros_like(form))
    use_form = (form >= 0) & (form_hp > 0) & (form_atk > 0)
    perm = (s.ch_base_atk[_b(s), side, seat] - s.char_base_atk[seat]).clamp(min=0)
    panel = torch.where(use_form, form_atk + perm,
                        s.ch_base_atk[_b(s), side, seat] + s.ch_growth[_b(s), side, seat])
    return (panel + s.ch_atk_buff[_b(s), side, seat]).clamp(min=0)


def _apply_hp(s: GpuState, side: torch.Tensor, is_player: torch.Tensor, seat: int,
              amount: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Absorb shield then HP. Returns HP loss. seat ignored for player."""
    b = _b(s)
    amount = amount.clamp(min=0)
    mask = mask & _live(s)
    sh = torch.where(is_player, s.shield[b, side], s.ch_shield[b, side, seat])
    hp = torch.where(is_player, s.hp[b, side], s.ch_hp[b, side, seat])
    absorb = torch.minimum(sh, amount)
    loss = amount - absorb
    nhp = (hp - loss).clamp(min=0)
    nsh = sh - absorb
    player_hp = torch.where(mask & is_player, nhp, s.hp[b, side])
    player_sh = torch.where(mask & is_player, nsh, s.shield[b, side])
    s.hp[b, side] = player_hp
    s.shield[b, side] = player_sh
    char_mask = mask & ~is_player
    s.ch_hp[b, side, seat] = torch.where(char_mask, nhp, s.ch_hp[b, side, seat])
    s.ch_shield[b, side, seat] = torch.where(char_mask, nsh, s.ch_shield[b, side, seat])
    return torch.where(mask, loss, torch.zeros_like(loss))


def victory(s: GpuState) -> None:
    dead_a = s.hp[:, 0] <= 0
    dead_b = s.hp[:, 1] <= 0
    done = _live(s) & (dead_a | dead_b)
    s.phase = torch.where(done, torch.full_like(s.phase, PHASE_FINISHED), s.phase)
    both = dead_a & dead_b
    s.winner = torch.where(done, torch.where(both, torch.full_like(s.winner, 2),
                                             torch.where(dead_a, torch.ones_like(s.winner), torch.zeros_like(s.winner))),
                           s.winner)
    s.reason = torch.where(done, torch.ones_like(s.reason), s.reason)


def knockdown(s: GpuState) -> None:
    for side in range(SIDES):
        for seat in range(N_SEATS):
            down = (s.ch_hp[:, side, seat] <= 0) & (s.ch_down[:, side, seat] == 0) & _live(s)
            if not bool(down.any()):
                continue
            is_front = s.front[:, side] == seat
            s.last_front[:, side] = torch.where(down & is_front, torch.full_like(s.front[:, side], seat), s.last_front[:, side])
            s.front[:, side] = torch.where(down & is_front, torch.full_like(s.front[:, side], -1), s.front[:, side])
            s.ch_down[:, side, seat] = torch.where(down, torch.full_like(s.ch_down[:, side, seat], 3), s.ch_down[:, side, seat])
            s.ch_shield[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_shield[:, side, seat]), s.ch_shield[:, side, seat])
            s.ch_growth[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_growth[:, side, seat]), s.ch_growth[:, side, seat])
            s.ch_shape[:, side, seat] = torch.where(down, torch.full_like(s.ch_shape[:, side, seat], -1), s.ch_shape[:, side, seat])
            s.ch_awakened[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_awakened[:, side, seat]), s.ch_awakened[:, side, seat])
            s.ch_ult_turns[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_ult_turns[:, side, seat]), s.ch_ult_turns[:, side, seat])
            s.ch_atk_buff[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_atk_buff[:, side, seat]), s.ch_atk_buff[:, side, seat])
            s.ch_max_hp[:, side, seat] = torch.where(down, s.ch_base_max[:, side, seat], s.ch_max_hp[:, side, seat])
            # Iloy Y07: on ally down, if shaped Y07, damage enemy front  — handled if Y07 equipped
            y07 = CARD_INDEX.get('Y07', -1)
            iloy_shape = s.ch_shape[:, side, ILOY] == y07
            iloy_live = s.ch_hp[:, side, ILOY] > 0
            if y07 >= 0:
                foe = 1 - side
                has_f = s.front[:, foe] >= 0
                seat_f = s.front[:, foe].clamp(min=0)
                hit = down & iloy_shape & iloy_live & (torch.arange(N_SEATS, device=s.device)[seat] != ILOY)
                # 2 damage to enemy front or player
                amt = torch.where(hit, torch.full_like(s.hp[:, foe], 2), torch.zeros_like(s.hp[:, foe]))
                for foe_seat in range(N_SEATS):
                    choose = hit & has_f & (s.front[:, foe] == foe_seat)
                    _apply_hp(s, torch.full_like(s.active, foe), torch.zeros_like(hit), foe_seat, amt, choose)
                _apply_hp(s, torch.full_like(s.active, foe), ~has_f, 0, amt, hit & ~has_f)


def _front_target(s: GpuState, side: torch.Tensor):
    foe = other_side(side)
    b = _b(s)
    has = s.front[b, foe] >= 0
    seat = s.front[b, foe].clamp(min=0)
    return foe, has, seat


def _deal_front(s: GpuState, side: torch.Tensor, amount: torch.Tensor, mask: torch.Tensor) -> None:
    foe, has, seat = _front_target(s, side)
    amount = torch.where(mask, amount, torch.zeros_like(amount))
    for foe_seat in range(N_SEATS):
        choose = mask & has & (s.front[_b(s), foe] == foe_seat)
        _apply_hp(s, foe, torch.zeros_like(mask), foe_seat, amount, choose)
    _apply_hp(s, foe, ~has, 0, amount, mask & ~has)
    knockdown(s)
    victory(s)


def _harmony_kind(s: GpuState, side: torch.Tensor, seat: int, payer: torch.Tensor) -> torch.Tensor:
    """创生 if 光+灵. payer is seat index or -1."""
    valid = payer >= 0
    attr_in = s.char_attr[seat]
    attr_pay = s.char_attr[payer.clamp(min=0)]
    pair = torch.minimum(attr_in, attr_pay) + torch.maximum(attr_in, attr_pay) * 8
    # 光=0 灵=1 -> 1
    genesis = valid & ((attr_in == 0) & (attr_pay == 1) | (attr_in == 1) & (attr_pay == 0))
    return torch.where(genesis, torch.full_like(payer, HARMONY_GENESIS), torch.zeros_like(payer))


def _payer(s: GpuState, side: torch.Tensor, seat: int) -> torch.Tensor:
    b = _b(s)
    prev = torch.where(s.front[b, side] >= 0, s.front[b, side], s.last_front[b, side])
    same = prev == seat
    prev = torch.where(same, torch.full_like(prev, -1), prev)
    ok = torch.zeros_like(prev, dtype=torch.bool)
    for pseat in range(N_SEATS):
        cand = (prev == pseat) & (s.ch_hp[b, side, pseat] > 0) & (s.ch_harmony[b, side, pseat] >= 2)
        ok = ok | cand
        prev = torch.where(cand, prev, prev)
    alive_pay = torch.zeros_like(prev, dtype=torch.bool)
    enough = torch.zeros_like(prev, dtype=torch.bool)
    for pseat in range(N_SEATS):
        hit = prev == pseat
        alive_pay = alive_pay | (hit & (s.ch_hp[b, side, pseat] > 0))
        enough = enough | (hit & (s.ch_harmony[b, side, pseat] >= 2))
    return torch.where((prev >= 0) & alive_pay & enough & (s.front[b, side] != seat), prev, torch.full_like(prev, -1))


def resolve_genesis(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor) -> None:
    extra = (s.ch_hp[_b(s), side, JIUYUAN] > 0).to(s.ch_hp.dtype)
    ticks = torch.where(mask, 1 + extra, torch.zeros_like(extra))
    for _ in range(3):
        still = mask & _live(s) & (ticks > 0) & _alive(s, side, seat)
        _deal_front(s, side, torch.ones_like(s.hp[:, 0]), still)
        ticks = ticks - still.to(ticks.dtype)
    flowers = torch.where(mask, s.extra_flower[_b(s), side], torch.zeros_like(s.extra_flower[_b(s), side]))
    s.extra_flower[_b(s), side] = torch.where(mask, torch.zeros_like(s.extra_flower[_b(s), side]), s.extra_flower[_b(s), side])
    for _ in range(2):
        still = mask & _live(s) & (flowers > 0) & _alive(s, side, seat)
        _deal_front(s, side, torch.ones_like(s.hp[:, 0]), still)
        flowers = flowers - still.to(flowers.dtype)
    # J08 on_genesis_damage: +1 and pact if shaped
    j08 = s.ch_shape[_b(s), side, JIUYUAN] == J08
    if J08 >= 0:
        foe, has, _seat = _front_target(s, side)
        hit = mask & j08 & (s.ch_hp[_b(s), side, JIUYUAN] > 0)
        _deal_front(s, side, torch.ones_like(s.hp[:, 0]), hit)
        for foe_seat in range(N_SEATS):
            mark = hit & has & (s.front[_b(s), foe] == foe_seat)
            s.ch_pact[_b(s), foe, foe_seat] = torch.where(mark, torch.ones_like(s.ch_pact[_b(s), foe, foe_seat]), s.ch_pact[_b(s), foe, foe_seat])


def enter(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor, *, support: bool = True) -> None:
    b = _b(s)
    mask = mask & _live(s) & _alive(s, side, seat)
    already = s.front[b, side] == seat
    mask = mask & ~already
    payer = _payer(s, side, seat)
    kind = _harmony_kind(s, side, seat, payer)
    # swap out current front
    for old in range(N_SEATS):
        leaving = mask & (s.front[b, side] == old)
        s.last_front[b, side] = torch.where(leaving, torch.full_like(s.front[b, side], old), s.last_front[b, side])
        s.ch_shield[b, side, old] = torch.where(leaving, torch.zeros_like(s.ch_shield[b, side, old]), s.ch_shield[b, side, old])
    s.front[b, side] = torch.where(mask, torch.full_like(s.front[b, side], seat), s.front[b, side])
    # pay harmony
    for pseat in range(N_SEATS):
        pay = mask & (kind == HARMONY_GENESIS) & (payer == pseat)
        s.ch_harmony[b, side, pseat] = torch.where(pay, s.ch_harmony[b, side, pseat] - 2, s.ch_harmony[b, side, pseat])
    s.ch_harmonized[b, side, seat] = torch.where(mask & (kind == HARMONY_GENESIS), torch.ones_like(s.ch_harmonized[b, side, seat]), s.ch_harmonized[b, side, seat])
    resolve_genesis(s, side, seat, mask & (kind == HARMONY_GENESIS))


def attack_once(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor, *,
                bonus: torch.Tensor | None = None, no_counter: torch.Tensor | None = None) -> None:
    b = _b(s)
    mask = mask & _live(s) & _alive(s, side, seat) & (s.front[b, side] == seat)
    bonus = torch.zeros(s.n, dtype=torch.int32, device=s.device) if bonus is None else bonus
    no_counter = torch.zeros(s.n, dtype=torch.bool, device=s.device) if no_counter is None else no_counter
    atk = attack_power(s, side, seat) + bonus
    foe, has, _ = _front_target(s, side)
    # iloy before_hits already folded into bonus
    for foe_seat in range(N_SEATS):
        tgt = mask & has & (s.front[b, foe] == foe_seat)
        counter = torch.where(tgt & ~no_counter, attack_power(s, foe, foe_seat), torch.zeros_like(atk))
        _apply_hp(s, foe, torch.zeros_like(mask), foe_seat, atk, tgt)
        _apply_hp(s, side, torch.zeros_like(mask), seat, counter, tgt)
    _apply_hp(s, foe, ~has, 0, atk, mask & ~has)
    knockdown(s)
    victory(s)


def _grant_combat_resources(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor) -> None:
    b = _b(s)
    mask = mask & _live(s)
    for ally in range(N_SEATS):
        live = mask & (s.ch_hp[b, side, ally] > 0) & (s.ch_down[b, side, ally] == 0)
        s.ch_energy[b, side, ally] = torch.where(
            live, torch.minimum(s.ch_energy[b, side, ally] + 1, s.ch_energy_max[b, side, ally]),
            s.ch_energy[b, side, ally])
    attacker = mask & _alive(s, side, seat)
    s.ch_energy[b, side, seat] = torch.where(
        attacker, torch.minimum(s.ch_energy[b, side, seat] + 1, s.ch_energy_max[b, side, seat]),
        s.ch_energy[b, side, seat])
    s.ch_harmony[b, side, seat] = torch.where(
        attacker, torch.minimum(s.ch_harmony[b, side, seat] + 1, torch.full_like(s.ch_harmony[b, side, seat], 2)),
        s.ch_harmony[b, side, seat])


def sortie(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor, *,
           card_bonus: torch.Tensor | None = None, followup: torch.Tensor | None = None,
           support: bool = True) -> None:
    b = _b(s)
    mask = mask & _live(s) & _alive(s, side, seat)
    enter(s, side, seat, mask, support=support)
    bonus = s.ch_next_bonus[b, side, seat]
    extra_sh = s.ch_next_shield[b, side, seat]
    s.ch_next_bonus[b, side, seat] = torch.where(mask, torch.zeros_like(bonus), bonus)
    s.ch_next_shield[b, side, seat] = torch.where(mask, torch.zeros_like(extra_sh), extra_sh)
    if extra_sh is not None:
        s.ch_shield[b, side, seat] = torch.where(mask, s.ch_shield[b, side, seat] + extra_sh, s.ch_shield[b, side, seat])
    if card_bonus is not None:
        bonus = bonus + card_bonus
    attack_once(s, side, seat, mask, bonus=bonus)
    fu = torch.zeros(s.n, dtype=torch.int32, device=s.device) if followup is None else followup
    fu = fu + torch.where(mask & (s.ch_awakened[b, side, seat] > 0), torch.ones_like(fu), torch.zeros_like(fu))
    fu = fu + s.ch_next_followup[b, side, seat]
    s.ch_next_followup[b, side, seat] = torch.where(mask, torch.zeros_like(s.ch_next_followup[b, side, seat]), s.ch_next_followup[b, side, seat])
    still = mask & _live(s) & _alive(s, side, seat) & (s.front[b, side] == seat)
    _deal_front(s, side, fu, still & (fu > 0))
    _grant_combat_resources(s, side, seat, still)


def _shift_left(kind: torch.Tensor, inst: torch.Tensor, count: torch.Tensor) -> None:
    kind[:, :-1] = kind[:, 1:].clone()
    inst[:, :-1] = inst[:, 1:].clone()
    kind[:, -1] = -1
    inst[:, -1] = 0
    count.sub_(1).clamp_(min=0)


def draw(s: GpuState, side: torch.Tensor, mask: torch.Tensor, count: int = 1) -> None:
    b = _b(s)
    for _ in range(count):
        still = mask & _live(s)
        empty = still & (s.deck_n[b, side] <= 0)
        foe = other_side(side)
        s.phase = torch.where(empty, torch.full_like(s.phase, PHASE_FINISHED), s.phase)
        s.winner = torch.where(empty, foe, s.winner)
        s.reason = torch.where(empty, torch.full_like(s.reason, 2), s.reason)
        can = still & (s.deck_n[b, side] > 0) & (s.hand_n[b, side] < HAND_SLOTS)
        if not bool(can.any()):
            continue
        # gather deck[0] per-row; loop seats not rows
        for si in range(SIDES):
            pick = can & (side == si)
            if not bool(pick.any()):
                continue
            hn = s.hand_n[:, si]
            kind0 = s.deck_kind[:, si, 0]
            inst0 = s.deck_inst[:, si, 0]
            for slot in range(HAND_SLOTS):
                here = pick & (hn == slot)
                s.hand_kind[:, si, slot] = torch.where(here, kind0, s.hand_kind[:, si, slot])
                s.hand_inst[:, si, slot] = torch.where(here, inst0, s.hand_inst[:, si, slot])
            s.hand_n[:, si] = torch.where(pick, hn + 1, hn)
            # shift deck
            s.deck_kind[:, si, :-1] = torch.where(pick.unsqueeze(-1), s.deck_kind[:, si, 1:], s.deck_kind[:, si, :-1])
            s.deck_inst[:, si, :-1] = torch.where(pick.unsqueeze(-1), s.deck_inst[:, si, 1:], s.deck_inst[:, si, :-1])
            s.deck_kind[:, si, -1] = torch.where(pick, torch.full_like(s.deck_kind[:, si, -1], -1), s.deck_kind[:, si, -1])
            s.deck_n[:, si] = torch.where(pick, s.deck_n[:, si] - 1, s.deck_n[:, si])


def remove_hand_slot(s: GpuState, side: torch.Tensor, slot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return card kind at slot and compact the hand."""
    b = _b(s)
    kind = torch.full((s.n,), -1, dtype=torch.int32, device=s.device)
    for si in range(SIDES):
        pick = mask & (side == si)
        for hs in range(HAND_SLOTS):
            here = pick & (slot == hs)
            kind = torch.where(here, s.hand_kind[:, si, hs], kind)
            # compact: copy j+1 -> j for j>=hs
            for j in range(hs, HAND_SLOTS - 1):
                s.hand_kind[:, si, j] = torch.where(here, s.hand_kind[:, si, j + 1], s.hand_kind[:, si, j])
                s.hand_inst[:, si, j] = torch.where(here, s.hand_inst[:, si, j + 1], s.hand_inst[:, si, j])
            s.hand_kind[:, si, HAND_SLOTS - 1] = torch.where(here, torch.full_like(s.hand_kind[:, si, 0], -1), s.hand_kind[:, si, HAND_SLOTS - 1])
            s.hand_n[:, si] = torch.where(here, (s.hand_n[:, si] - 1).clamp(min=0), s.hand_n[:, si])
    return kind


def begin_turn(s: GpuState, side: torch.Tensor, mask: torch.Tensor) -> None:
    b = _b(s)
    mask = mask & _live(s)
    s.phase = torch.where(mask, torch.full_like(s.phase, PHASE_PLAYING), s.phase)
    s.active = torch.where(mask, side, s.active)
    s.turn = torch.where(mask, s.turn + 1, s.turn)
    for si in range(SIDES):
        here = mask & (side == si)
        s.turn_count[:, si] = torch.where(here, s.turn_count[:, si] + 1, s.turn_count[:, si])
        s.used_instant[:, si] = torch.where(here, torch.zeros_like(s.used_instant[:, si]), s.used_instant[:, si])
        s.surplus[:, si] = torch.where(here, torch.zeros_like(s.surplus[:, si]), s.surplus[:, si])
        for seat in range(N_SEATS):
            s.ch_shield[:, si, seat] = torch.where(here, torch.zeros_like(s.ch_shield[:, si, seat]), s.ch_shield[:, si, seat])
            recovering = here & (s.ch_down[:, si, seat] > 0)
            s.ch_down[:, si, seat] = torch.where(recovering, s.ch_down[:, si, seat] - 1, s.ch_down[:, si, seat])
            revived = recovering & (s.ch_down[:, si, seat] == 0)
            s.ch_hp[:, si, seat] = torch.where(revived, s.ch_max_hp[:, si, seat], s.ch_hp[:, si, seat])
        # return front
        for seat in range(N_SEATS):
            leaving = here & (s.front[:, si] == seat)
            s.last_front[:, si] = torch.where(leaving, torch.full_like(s.front[:, si], seat), s.last_front[:, si])
            s.ch_shield[:, si, seat] = torch.where(leaving, torch.zeros_like(s.ch_shield[:, si, seat]), s.ch_shield[:, si, seat])
        s.front[:, si] = torch.where(here, torch.full_like(s.front[:, si], -1), s.front[:, si])
        # nanali first-turn seed
        first = here & (s.turn_count[:, si] == 1) & (s.ch_hp[:, si, NANALI] > 0) & (s.nanali_seed[:, si] < 3)
        if NF01 >= 0:
            for slot in range(HAND_SLOTS):
                put = first & (s.hand_n[:, si] == slot)
                s.hand_kind[:, si, slot] = torch.where(put, torch.full_like(s.hand_kind[:, si, slot], NF01), s.hand_kind[:, si, slot])
            s.hand_n[:, si] = torch.where(first & (s.hand_n[:, si] < HAND_SLOTS), s.hand_n[:, si] + 1, s.hand_n[:, si])
            s.nanali_seed[:, si] = torch.where(first, s.nanali_seed[:, si] + 1, s.nanali_seed[:, si])
        # iloy energy_next
        give = here & (s.ch_energy_next[:, si, ILOY] > 0)
        s.ch_energy_next[:, si, ILOY] = torch.where(here, torch.zeros_like(s.ch_energy_next[:, si, ILOY]), s.ch_energy_next[:, si, ILOY])
        for seat in range(N_SEATS):
            s.ch_energy[:, si, seat] = torch.where(
                give & (s.ch_hp[:, si, seat] > 0),
                torch.minimum(s.ch_energy[:, si, seat] + 1, s.ch_energy_max[:, si, seat]),
                s.ch_energy[:, si, seat])
        extra = s.extra_ap[:, si]
        s.extra_ap[:, si] = torch.where(here, torch.zeros_like(extra), extra)
        ap = torch.where(s.turn == 1, torch.ones_like(s.ap[:, si]), torch.full_like(s.ap[:, si], 2)) + extra
        s.ap[:, si] = torch.where(here, ap, s.ap[:, si])
        s.normal_atk[:, si] = torch.where(here, torch.ones_like(s.normal_atk[:, si]), s.normal_atk[:, si])
        s.ultimate_ok[:, si] = torch.where(here, torch.ones_like(s.ultimate_ok[:, si]), s.ultimate_ok[:, si])
    not_first_game_turn = mask & (s.turn != 1)
    draw(s, side, not_first_game_turn, 1)


def end_turn(s: GpuState, side: torch.Tensor, mask: torch.Tensor) -> None:
    b = _b(s)
    mask = mask & _live(s)
    knockdown(s)
    victory(s)
    for si in range(SIDES):
        here = mask & (side == si)
        # zero on_turn_end shapes
        front = s.front[:, si]
        z08 = here & (s.ch_shape[:, si, ZERO] == Z08) & (s.ch_hp[:, si, ZERO] > 0) & (front >= 0)
        z07 = here & (s.ch_shape[:, si, ZERO] == Z07) & (s.ch_hp[:, si, ZERO] > 0) & (front >= 0)
        for seat in range(N_SEATS):
            hit = z08 & (front == seat)
            s.ch_atk_buff[:, si, seat] = torch.where(hit, s.ch_atk_buff[:, si, seat] + 2, s.ch_atk_buff[:, si, seat])
            heal = z07 & (front == seat)
            s.ch_hp[:, si, seat] = torch.where(
                heal, torch.minimum(s.ch_hp[:, si, seat] + 2, s.ch_max_hp[:, si, seat]), s.ch_hp[:, si, seat])
        s.normal_atk[:, si] = torch.where(here, torch.zeros_like(s.normal_atk[:, si]), s.normal_atk[:, si])
        s.ultimate_ok[:, si] = torch.where(here, torch.zeros_like(s.ultimate_ok[:, si]), s.ultimate_ok[:, si])
        # tick ultimates
        for seat in range(N_SEATS):
            ticking = here & (s.ch_ult_turns[:, si, seat] > 0)
            s.ch_ult_turns[:, si, seat] = torch.where(ticking, s.ch_ult_turns[:, si, seat] - 1, s.ch_ult_turns[:, si, seat])
            ended = ticking & (s.ch_ult_turns[:, si, seat] <= 0)
            s.ch_awakened[:, si, seat] = torch.where(ended, torch.zeros_like(s.ch_awakened[:, si, seat]), s.ch_awakened[:, si, seat])
    begin_turn(s, other_side(side), mask & _live(s))


def _play_effects(s: GpuState, side: torch.Tensor, seat: int, kind: torch.Tensor, mask: torch.Tensor,
                  tside: torch.Tensor, tseat: torch.Tensor) -> None:
    b = _b(s)
    bonus = s.catalog_atk[kind.clamp(min=0)]
    shield = s.catalog_shield[kind.clamp(min=0)]
    is_battle = s.catalog_type[kind.clamp(min=0)] == 0
    is_form = s.catalog_type[kind.clamp(min=0)] == 2
    s.ch_shield[b, side, seat] = torch.where(mask & is_battle, s.ch_shield[b, side, seat] + shield, s.ch_shield[b, side, seat])
    n03 = mask & (kind == N03)
    fu = torch.where(n03, torch.ones_like(kind), torch.zeros_like(kind))
    do_sortie = mask & is_battle
    sortie(s, side, seat, do_sortie, card_bonus=torch.where(do_sortie, bonus, torch.zeros_like(bonus)), followup=fu)
    # zero after battle: harmony = 2
    s.ch_harmony[b, side, ZERO] = torch.where(
        mask & is_battle & (seat == ZERO) & (s.ch_hp[b, side, ZERO] > 0),
        torch.full_like(s.ch_harmony[b, side, ZERO], 2), s.ch_harmony[b, side, ZERO])
    # forms
    form = mask & is_form
    s.ch_shape[b, side, seat] = torch.where(form, kind, s.ch_shape[b, side, seat])
    panel = s.catalog_hp[kind.clamp(min=0)]
    perm = (s.ch_base_max[b, side, seat] - s.char_base_hp[seat]).clamp(min=0)
    new_max = torch.where(panel > 0, panel + perm, s.ch_max_hp[b, side, seat])
    s.ch_max_hp[b, side, seat] = torch.where(form & (panel > 0), new_max, s.ch_max_hp[b, side, seat])
    s.ch_hp[b, side, seat] = torch.where(form & (panel > 0), new_max, s.ch_hp[b, side, seat])
    # Y03 heal 5 + atk buff
    y03 = mask & (kind == Y03)
    for ally in range(N_SEATS):
        hit = y03 & (tside == side) & (tseat == ally)
        s.ch_hp[b, side, ally] = torch.where(hit, torch.minimum(s.ch_hp[b, side, ally] + 5, s.ch_max_hp[b, side, ally]), s.ch_hp[b, side, ally])
        s.ch_atk_buff[b, side, ally] = torch.where(hit, s.ch_atk_buff[b, side, ally] + 1, s.ch_atk_buff[b, side, ally])
    y03p = y03 & (tseat < 0)
    s.hp[b, side] = torch.where(y03p, torch.minimum(s.hp[b, side] + 5, torch.full_like(s.hp[b, side], 30)), s.hp[b, side])
    # Y06 revive
    y06 = mask & (kind == Y06)
    for ally in range(N_SEATS):
        hit = y06 & (tseat == ally)
        s.ch_down[b, side, ally] = torch.where(hit, torch.zeros_like(s.ch_down[b, side, ally]), s.ch_down[b, side, ally])
        s.ch_hp[b, side, ally] = torch.where(hit, s.ch_max_hp[b, side, ally], s.ch_hp[b, side, ally])
        s.ch_atk_buff[b, side, ally] = torch.where(hit, s.ch_atk_buff[b, side, ally] + 1, s.ch_atk_buff[b, side, ally])
    # J02/J04 pact damage
    j02 = mask & (kind == J02)
    j04 = mask & (kind == J04)
    _deal_front(s, side, torch.ones_like(s.hp[:, 0]), j02)
    for foe_seat in range(N_SEATS):
        foe = other_side(side)
        hit = j04 & (tseat == foe_seat)
        _apply_hp(s, foe, torch.zeros_like(hit), foe_seat, torch.full_like(s.hp[:, 0], 2), hit)
        s.ch_pact[b, foe, foe_seat] = torch.where(hit | (j02 & (s.front[b, foe] == foe_seat)), torch.ones_like(s.ch_pact[b, foe, foe_seat]), s.ch_pact[b, foe, foe_seat])
    knockdown(s)
    victory(s)


def play_card(s: GpuState, side: torch.Tensor, hand_slot: torch.Tensor, mask: torch.Tensor,
              tside: torch.Tensor, tseat: torch.Tensor) -> None:
    b = _b(s)
    kind = remove_hand_slot(s, side, hand_slot, mask)
    owner = torch.where(kind >= 0, s.catalog_owner[kind.clamp(min=0)], torch.zeros_like(kind))
    cost = torch.where(kind >= 0, s.catalog_cost[kind.clamp(min=0)], torch.zeros_like(kind))
    instant = (kind >= 0) & (s.catalog_instant[kind.clamp(min=0)] > 0)
    # zero shaped forms instant
    is_form = (kind >= 0) & (s.catalog_type[kind.clamp(min=0)] == 2)
    zero_shaped = (s.ch_hp[b, side, ZERO] > 0) & (s.ch_shape[b, side, ZERO] >= 0)
    free = mask & (instant | (is_form & zero_shaped)) & (s.used_instant[b, side] == 0)
    for si in range(SIDES):
        here = mask & (side == si)
        s.used_instant[:, si] = torch.where(here & free, torch.ones_like(s.used_instant[:, si]), s.used_instant[:, si])
        s.ap[:, si] = torch.where(here & ~free, s.ap[:, si] - cost, s.ap[:, si])
    for seat in range(N_SEATS):
        _play_effects(s, side, seat, kind, mask & (owner == seat), tside, tseat)


def start_ultimate(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor) -> None:
    b = _b(s)
    mask = mask & _live(s) & _alive(s, side, seat)
    s.ch_energy[b, side, seat] = torch.where(mask, torch.zeros_like(s.ch_energy[b, side, seat]), s.ch_energy[b, side, seat])
    status = s.char_ult_kind[seat] > 0
    s.ch_awakened[b, side, seat] = torch.where(mask & status, torch.ones_like(s.ch_awakened[b, side, seat]), s.ch_awakened[b, side, seat])
    s.ch_ult_turns[b, side, seat] = torch.where(mask & status, s.char_ult_turns[seat].expand(s.n), s.ch_ult_turns[b, side, seat])
    # iloy: energy_next
    if seat == ILOY:
        s.ch_energy_next[b, side, ILOY] = torch.where(mask, torch.ones_like(s.ch_energy_next[b, side, ILOY]), s.ch_energy_next[b, side, ILOY])
    # jiuyuan instant: damage all enemies, +2 if pact
    if seat == JIUYUAN:
        foe = other_side(side)
        for foe_seat in range(N_SEATS):
            hit = mask & (s.ch_hp[b, foe, foe_seat] > 0)
            amt = torch.where(s.ch_pact[b, foe, foe_seat] > 0, torch.full_like(s.hp[:, 0], 3), torch.ones_like(s.hp[:, 0]))
            s.ch_pact[b, foe, foe_seat] = torch.where(hit & (s.ch_pact[b, foe, foe_seat] > 0), torch.zeros_like(s.ch_pact[b, foe, foe_seat]), s.ch_pact[b, foe, foe_seat])
            _apply_hp(s, foe, torch.zeros_like(hit), foe_seat, amt, hit)
        knockdown(s)
        victory(s)


def rebuild_legal(s: GpuState) -> None:
    s.legal_type.fill_(-1)
    s.legal_actor.fill_(-1)
    s.legal_hand.fill_(-1)
    s.legal_tside.fill_(-1)
    s.legal_tseat.fill_(-1)
    s.legal_n.zero_()
    b = _b(s)
    playing = s.phase == PHASE_PLAYING
    for row in range(s.n):
        if int(s.phase[row]) == PHASE_FINISHED:
            continue
        side = int(s.active[row])
        n = 0
        def push(typ, actor=-1, hand=-1, tside=-1, tseat=-1):
            nonlocal n
            if n >= MAX_LEGAL:
                raise ValueError('Legal action overflow')
            s.legal_type[row, n] = typ
            s.legal_actor[row, n] = actor
            s.legal_hand[row, n] = hand
            s.legal_tside[row, n] = tside
            s.legal_tseat[row, n] = tseat
            n += 1
        if int(s.phase[row]) != PHASE_PLAYING:
            s.legal_n[row] = 0
            continue
        push(ACT_END)
        ap = int(s.ap[row, side])
        if ap >= 1 and int(s.normal_atk[row, side]):
            for seat in range(N_SEATS):
                if int(s.ch_hp[row, side, seat]) > 0 and int(s.ch_down[row, side, seat]) == 0:
                    push(ACT_ATTACK, seat)
        if int(s.ultimate_ok[row, side]):
            for seat in range(N_SEATS):
                if int(s.ch_hp[row, side, seat]) > 0 and int(s.ch_energy[row, side, seat]) >= int(s.ch_energy_max[row, side, seat]):
                    push(ACT_ULTIMATE, seat)
        hand_n = int(s.hand_n[row, side])
        for hs in range(hand_n):
            kind = int(s.hand_kind[row, side, hs])
            if kind < 0:
                continue
            owner = int(s.catalog_owner[kind])
            if int(s.ch_hp[row, side, owner]) <= 0:
                continue
            cost = int(s.catalog_cost[kind])
            free = int(s.catalog_instant[kind]) and not int(s.used_instant[row, side])
            if not free and ap < cost:
                continue
            from .catalog import CARD_IDS
            card_id = CARD_IDS[kind]
            if card_id in ('Y03',):
                injured = False
                if int(s.hp[row, side]) < 30:
                    push(ACT_PLAY, owner, hs, side, -1)
                    injured = True
                for ally in range(N_SEATS):
                    if 0 < int(s.ch_hp[row, side, ally]) < int(s.ch_max_hp[row, side, ally]):
                        push(ACT_PLAY, owner, hs, side, ally)
                        injured = True
                if not injured:
                    continue
            elif card_id in ('Y02',):
                for ally in range(N_SEATS):
                    if int(s.ch_hp[row, side, ally]) > 0:
                        push(ACT_PLAY, owner, hs, side, ally)
            elif card_id in ('Y06',):
                for ally in range(N_SEATS):
                    if int(s.ch_down[row, side, ally]) > 0 or int(s.ch_hp[row, side, ally]) <= 0:
                        push(ACT_PLAY, owner, hs, side, ally)
            elif card_id in ('J04',):
                foe = 1 - side
                for enemy in range(N_SEATS):
                    if int(s.ch_hp[row, foe, enemy]) > 0:
                        push(ACT_PLAY, owner, hs, foe, enemy)
            else:
                push(ACT_PLAY, owner, hs, -1, -1)
        s.legal_n[row] = n
        if n > MAX_LEGAL:
            raise ValueError('Legal action overflow')


def step_index(s: GpuState, index: torch.Tensor) -> None:
    rebuild_legal(s)
    b = _b(s)
    idx = index.clamp(min=0)
    valid = _live(s) & (idx < s.legal_n)
    typ = s.legal_type[b, idx]
    actor = s.legal_actor[b, idx]
    hand = s.legal_hand[b, idx]
    tside = s.legal_tside[b, idx]
    tseat = s.legal_tseat[b, idx]
    side = s.active
    end = valid & (typ == ACT_END)
    end_turn(s, side, end)
    for seat in range(N_SEATS):
        atk = valid & (typ == ACT_ATTACK) & (actor == seat)
        for si in range(SIDES):
            here = atk & (side == si)
            s.ap[:, si] = torch.where(here, s.ap[:, si] - 1, s.ap[:, si])
            s.normal_atk[:, si] = torch.where(here, torch.zeros_like(s.normal_atk[:, si]), s.normal_atk[:, si])
        sortie(s, side, seat, atk)
        ult = valid & (typ == ACT_ULTIMATE) & (actor == seat)
        for si in range(SIDES):
            s.ultimate_ok[:, si] = torch.where(ult & (side == si), torch.zeros_like(s.ultimate_ok[:, si]), s.ultimate_ok[:, si])
        start_ultimate(s, side, seat, ult)
    play = valid & (typ == ACT_PLAY)
    play_card(s, side, hand, play, tside, tseat)
    conc = valid & (typ == ACT_CONCEDE)
    s.phase = torch.where(conc, torch.full_like(s.phase, PHASE_FINISHED), s.phase)
    s.winner = torch.where(conc, other_side(side), s.winner)
    s.reason = torch.where(conc, torch.full_like(s.reason, 3), s.reason)
    rebuild_legal(s)
