"""Batched GPU duel rules on integer tensors. No JSON, no presentation."""
from __future__ import annotations

import torch

from .catalog import (
    ACT_ATTACK, ACT_CHOOSE, ACT_CONCEDE, ACT_END, ACT_PLAY, ACT_ULTIMATE, BAICANG,
    BOHE, CARD_INDEX, DISCARD_SLOTS, HAND_SLOTS, HARMONY_GENESIS, HARMONY_OVERLAY,
    ILOY, JIUYUAN, LEGAL_ATK, LEGAL_END, LEGAL_PLAY, LEGAL_ULT, LEGAL_WIDTH,
    MAX_LEGAL, N_SEATS, PHASE_CHOICE, PHASE_FINISHED, PHASE_PLAYING, PLAY_TARGETS,
    POLICY_ALLY, POLICY_DOWN_ALLY, POLICY_ENEMY, POLICY_ENEMY_FRONT, POLICY_INJURED,
    POLICY_NONE, RESPONSE_ALLY, RESPONSE_SELF, SIDES, ZERO,
)
from .state import GpuState, other_side


def _effect_tables(s: GpuState):
    from app.modules.card_game.rl.rule_ir.lower_effects import gpu_tables
    return gpu_tables(str(s.device))


def _hook_card(s: GpuState, key: str) -> int:
    card_id = _effect_tables(s)['hooks'][key]
    return CARD_INDEX.get(card_id, -1)


def _hook_seat(s: GpuState, key: str) -> int:
    from .catalog import SEAT_INDEX
    return SEAT_INDEX[_effect_tables(s)['hooks'][key]]


def _cpu_mask_empty(mask: torch.Tensor) -> bool:
    """Skip CPU no-op work without adding a CUDA-to-host synchronization."""
    return mask.device.type == 'cpu' and not bool(mask.any())


def _b(s: GpuState) -> torch.Tensor:
    return torch.arange(s.n, device=s.device)


def _live(s: GpuState) -> torch.Tensor:
    return s.phase != PHASE_FINISHED


def _alive(s: GpuState, side: torch.Tensor, seat: int) -> torch.Tensor:
    return (s.ch_present[_b(s), side, seat] > 0) & (s.ch_hp[_b(s), side, seat] > 0)


def attack_power(s: GpuState, side: torch.Tensor, seat: int) -> torch.Tensor:
    form = s.ch_shape[_b(s), side, seat]
    idx = form.clamp(min=0)
    form_hp = torch.where(form >= 0, s.catalog_hp[idx], torch.zeros_like(form))
    form_atk = torch.where(form >= 0, s.catalog_atk[idx], torch.zeros_like(form))
    use_form = (form >= 0) & (form_hp > 0) & (form_atk > 0)
    perm = (s.ch_base_atk[_b(s), side, seat] - s.char_base_atk[seat]).clamp(min=0)
    panel = torch.where(use_form, form_atk + perm,
                        s.ch_base_atk[_b(s), side, seat] + s.ch_growth[_b(s), side, seat])
    value = (panel + s.ch_atk_buff[_b(s), side, seat]).clamp(min=0)
    sleep_id = CARD_INDEX.get('M07', -1)
    sleep = (sleep_id >= 0) & (form == sleep_id) & (s.active != side)
    return torch.where(sleep, torch.zeros_like(value), value)


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
    stacked = char_mask & (loss > 0) & (s.ch_present[b, side, seat] > 0)
    if seat == BAICANG:
        s.ch_atk_buff[b, side, seat] = torch.where(
            stacked, s.ch_atk_buff[b, side, seat] + 1, s.ch_atk_buff[b, side, seat])
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
            down = (s.ch_present[:, side, seat] > 0) & (s.ch_hp[:, side, seat] <= 0) & (s.ch_down[:, side, seat] == 0) & _live(s)
            if not bool(down.any()):
                continue
            is_front = s.front[:, side] == seat
            expires = down & (is_front | (s.last_front[:, side] == seat))
            s.last_front[:, side] = torch.where(expires, torch.full_like(s.last_front[:, side], -1), s.last_front[:, side])
            s.front[:, side] = torch.where(down & is_front, torch.full_like(s.front[:, side], -1), s.front[:, side])
            s.ch_down[:, side, seat] = torch.where(down, torch.full_like(s.ch_down[:, side, seat], 3), s.ch_down[:, side, seat])
            s.ch_shield[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_shield[:, side, seat]), s.ch_shield[:, side, seat])
            s.ch_growth[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_growth[:, side, seat]), s.ch_growth[:, side, seat])
            s.ch_shape[:, side, seat] = torch.where(down, torch.full_like(s.ch_shape[:, side, seat], -1), s.ch_shape[:, side, seat])
            s.ch_awakened[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_awakened[:, side, seat]), s.ch_awakened[:, side, seat])
            s.ch_ult_turns[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_ult_turns[:, side, seat]), s.ch_ult_turns[:, side, seat])
            s.ch_atk_buff[:, side, seat] = torch.where(down, torch.zeros_like(s.ch_atk_buff[:, side, seat]), s.ch_atk_buff[:, side, seat])
            s.ch_max_hp[:, side, seat] = torch.where(down, s.ch_base_max[:, side, seat], s.ch_max_hp[:, side, seat])
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


def _trigger_ally_attacked(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor) -> None:
    """Auto-play the first affordable ally_attacked response on the defending side."""
    b = _b(s)
    foe = other_side(side)
    target_seat = s.front[b, foe]
    fired = torch.zeros(s.n, dtype=torch.bool, device=s.device)
    for hs in range(HAND_SLOTS):
        kind = s.hand_kind[b, foe, hs]
        valid = mask & _live(s) & (kind >= 0) & ~fired
        resp = s.catalog_response[kind.clamp(min=0)]
        owner = s.catalog_owner[kind.clamp(min=0)]
        cost = s.catalog_cost[kind.clamp(min=0)]
        ally = valid & (resp == RESPONSE_ALLY) & (target_seat >= 0) & (owner != target_seat)
        self_hit = valid & (resp == RESPONSE_SELF) & (target_seat >= 0) & (owner == target_seat)
        can = ally | self_hit
        for own in range(N_SEATS):
            alive_own = (owner == own) & (s.ch_hp[b, foe, own] > 0)
            afford = (s.ap[b, foe] >= cost) | ((s.catalog_instant[kind.clamp(min=0)] > 0) & (s.used_instant[b, foe] == 0))
            fire = can & alive_own & afford
            if not bool(fire.any()):
                continue
            fired = fired | fire
            s.ap[b, foe] = torch.where(fire, s.ap[b, foe] - cost, s.ap[b, foe])
            remove_hand_slot(s, foe, torch.full_like(s.active, hs), fire)
            enter(s, foe, own, fire, support=False)


def _front_target(s: GpuState, side: torch.Tensor):
    foe = other_side(side)
    b = _b(s)
    has = s.front[b, foe] >= 0
    seat = s.front[b, foe].clamp(min=0)
    return foe, has, seat


def _deal_front(s: GpuState, side: torch.Tensor, amount: torch.Tensor, mask: torch.Tensor) -> None:
    if _cpu_mask_empty(mask):
        # Settlement is part of this helper's contract even for a zero-row hit.
        knockdown(s)
        victory(s)
        return
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
    overlay = valid & ((attr_in == 1) & (attr_pay == 5) | (attr_in == 5) & (attr_pay == 1))
    kind = torch.where(genesis, torch.full_like(payer, HARMONY_GENESIS), torch.zeros_like(payer))
    return torch.where(overlay, torch.full_like(payer, HARMONY_OVERLAY), kind)


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
    y08_id = _hook_card(s, 'extra_genesis_shape')
    y08 = mask & (s.ch_shape[_b(s), side, ILOY] == y08_id) & (s.ch_hp[_b(s), side, ILOY] > 0) & (s.extra_genesis[_b(s), side] == 0)
    s.extra_genesis[_b(s), side] = torch.where(y08, torch.ones_like(s.extra_genesis[_b(s), side]), s.extra_genesis[_b(s), side])
    s.extra_genesis_actor[_b(s), side] = torch.where(y08, torch.full_like(s.extra_genesis_actor[_b(s), side], seat), s.extra_genesis_actor[_b(s), side])
    flowers = torch.where(mask, s.extra_flower[_b(s), side], torch.zeros_like(s.extra_flower[_b(s), side]))
    s.extra_flower[_b(s), side] = torch.where(mask, torch.zeros_like(s.extra_flower[_b(s), side]), s.extra_flower[_b(s), side])
    for _ in range(2):
        still = mask & _live(s) & (flowers > 0) & _alive(s, side, seat)
        _deal_front(s, side, torch.ones_like(s.hp[:, 0]), still)
        flowers = flowers - still.to(flowers.dtype)
    j08_id = _hook_card(s, 'genesis_pact_shape')
    j08 = s.ch_shape[_b(s), side, JIUYUAN] == j08_id
    if j08_id >= 0:
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
    s.ch_harmonized[b, side, seat] = torch.where(
        mask & ((kind == HARMONY_GENESIS) | (kind == HARMONY_OVERLAY)),
        torch.ones_like(s.ch_harmonized[b, side, seat]), s.ch_harmonized[b, side, seat])
    foe = other_side(side)
    s.weave[b, foe] = torch.where(mask & (kind == HARMONY_OVERLAY), torch.ones_like(s.weave[b, foe]), s.weave[b, foe])
    resolve_genesis(s, side, seat, mask & (kind == HARMONY_GENESIS))


def attack_once(s: GpuState, side: torch.Tensor, seat: int, mask: torch.Tensor, *,
                bonus: torch.Tensor | None = None, no_counter: torch.Tensor | None = None) -> None:
    b = _b(s)
    mask = mask & _live(s) & _alive(s, side, seat) & (s.front[b, side] == seat)
    bonus = torch.zeros(s.n, dtype=torch.int32, device=s.device) if bonus is None else bonus
    no_counter = torch.zeros(s.n, dtype=torch.bool, device=s.device) if no_counter is None else no_counter
    _trigger_ally_attacked(s, side, seat, mask)
    atk = attack_power(s, side, seat) + bonus
    foe, has, _ = _front_target(s, side)
    # iloy before_hits already folded into bonus
    overflow_seat = _hook_seat(s, 'overflow_seat')
    for foe_seat in range(N_SEATS):
        tgt = mask & has & (s.front[b, foe] == foe_seat)
        before_hp = s.ch_hp[b, foe, foe_seat]
        before_sh = s.ch_shield[b, foe, foe_seat]
        counter = torch.where(tgt & ~no_counter, attack_power(s, foe, foe_seat), torch.zeros_like(atk))
        _apply_hp(s, foe, torch.zeros_like(mask), foe_seat, atk, tgt)
        overflow = (atk - before_sh - before_hp).clamp(min=0)
        pierce = tgt & (seat == overflow_seat)
        _apply_hp(s, foe, torch.ones_like(mask), 0, overflow, pierce)
        share = tgt & (s.ch_share_overflow[b, foe, foe_seat] > 0)
        _apply_hp(s, foe, torch.ones_like(mask), 0, overflow, share)
        s.ch_share_overflow[b, foe, foe_seat] = torch.where(tgt, torch.zeros_like(s.ch_share_overflow[b, foe, foe_seat]), s.ch_share_overflow[b, foe, foe_seat])
        _apply_hp(s, side, torch.zeros_like(mask), seat, counter, tgt)
        weave_hit = tgt & (s.weave[b, foe] > 0) & ((s.char_attr[seat] == 1) | (s.char_attr[seat] == 5))
        _apply_hp(s, foe, torch.zeros_like(mask), foe_seat, torch.full_like(atk, 2), weave_hit)
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
        can = still & (s.deck_n[b, side] > 0)
        if not bool(can.any()):
            continue
        for si in range(SIDES):
            pick = can & (side == si)
            if not bool(pick.any()):
                continue
            hn = s.hand_n[:, si]
            kind0 = s.deck_kind[:, si, 0]
            inst0 = s.deck_inst[:, si, 0]
            room = pick & (hn < HAND_SLOTS)
            full = pick & (hn >= HAND_SLOTS)
            for slot in range(HAND_SLOTS):
                here = room & (hn == slot)
                s.hand_kind[:, si, slot] = torch.where(here, kind0, s.hand_kind[:, si, slot])
                s.hand_inst[:, si, slot] = torch.where(here, inst0, s.hand_inst[:, si, slot])
            s.hand_n[:, si] = torch.where(room, hn + 1, hn)
            to_discard(s, torch.full_like(s.active, si), kind0, full)
            s.deck_kind[:, si, :-1] = torch.where(pick.unsqueeze(-1), s.deck_kind[:, si, 1:], s.deck_kind[:, si, :-1])
            s.deck_inst[:, si, :-1] = torch.where(pick.unsqueeze(-1), s.deck_inst[:, si, 1:], s.deck_inst[:, si, :-1])
            s.deck_kind[:, si, -1] = torch.where(pick, torch.full_like(s.deck_kind[:, si, -1], -1), s.deck_kind[:, si, -1])
            s.deck_n[:, si] = torch.where(pick, s.deck_n[:, si] - 1, s.deck_n[:, si])


def compact_deck(s: GpuState, si: int, mask: torch.Tensor) -> None:
    if not bool(mask.any()):
        return
    kinds = s.deck_kind[:, si].tolist()
    insts = s.deck_inst[:, si].tolist()
    flags = mask.tolist()
    counts = []
    for row, keep in enumerate(flags):
        if not keep:
            counts.append(s.deck_n[row, si].item())
            continue
        packed_k, packed_i = [], []
        for slot in range(32):
            kind = kinds[row][slot]
            if kind >= 0:
                packed_k.append(kind)
                packed_i.append(insts[row][slot])
        counts.append(len(packed_k))
        packed_k.extend([-1] * (32 - len(packed_k)))
        packed_i.extend([0] * (32 - len(packed_i)))
        kinds[row] = packed_k
        insts[row] = packed_i
    s.deck_kind[:, si] = torch.tensor(kinds, dtype=torch.int32, device=s.device)
    s.deck_inst[:, si] = torch.tensor(insts, dtype=torch.int32, device=s.device)
    s.deck_n[:, si] = torch.tensor(counts, dtype=torch.int32, device=s.device)


def to_discard(s: GpuState, side: torch.Tensor, kind: torch.Tensor, mask: torch.Tensor) -> None:
    for si in range(SIDES):
        here = mask & (side == si) & (kind >= 0)
        n = s.discard_n[:, si]
        for slot in range(DISCARD_SLOTS):
            put = here & (n == slot)
            s.discard_kind[:, si, slot] = torch.where(put, kind, s.discard_kind[:, si, slot])
        s.discard_n[:, si] = torch.where(here & (n < DISCARD_SLOTS), n + 1, n)


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
        s.weave[:, 1 - si] = torch.where(here, torch.zeros_like(s.weave[:, 1 - si]), s.weave[:, 1 - si])
        for seat in range(N_SEATS):
            pending = s.ch_pending_atk[:, si, seat]
            s.ch_atk_buff[:, si, seat] = torch.where(here, s.ch_atk_buff[:, si, seat] + pending, s.ch_atk_buff[:, si, seat])
            s.ch_pending_atk[:, si, seat] = torch.where(here, torch.zeros_like(pending), pending)
            s.ch_allied_hurt[:, si, seat] = torch.where(here, torch.zeros_like(s.ch_allied_hurt[:, si, seat]), s.ch_allied_hurt[:, si, seat])
        for seat in range(N_SEATS):
            s.ch_shield[:, si, seat] = torch.where(here, torch.zeros_like(s.ch_shield[:, si, seat]), s.ch_shield[:, si, seat])
            recovering = here & (s.ch_down[:, si, seat] > 0)
            s.last_front[:, si] = torch.where(recovering & (s.last_front[:, si] == seat), torch.full_like(s.last_front[:, si], -1), s.last_front[:, si])
            s.ch_down[:, si, seat] = torch.where(recovering, s.ch_down[:, si, seat] - 1, s.ch_down[:, si, seat])
            revived = recovering & (s.ch_down[:, si, seat] == 0)
            s.ch_hp[:, si, seat] = torch.where(revived, s.ch_max_hp[:, si, seat], s.ch_hp[:, si, seat])
        # return front
        for seat in range(N_SEATS):
            leaving = here & (s.front[:, si] == seat)
            s.last_front[:, si] = torch.where(leaving, torch.full_like(s.front[:, si], seat), s.last_front[:, si])
            s.ch_shield[:, si, seat] = torch.where(leaving, torch.zeros_like(s.ch_shield[:, si, seat]), s.ch_shield[:, si, seat])
        s.front[:, si] = torch.where(here, torch.full_like(s.front[:, si], -1), s.front[:, si])
        pending = here & (s.extra_genesis[:, si] > 0)
        actor = s.extra_genesis_actor[:, si]
        s.extra_genesis[:, si] = torch.where(here, torch.zeros_like(s.extra_genesis[:, si]), s.extra_genesis[:, si])
        s.extra_genesis_actor[:, si] = torch.where(here, torch.full_like(actor, -1), actor)
        for seat in range(N_SEATS):
            resolve_genesis(s, torch.full_like(s.active, si), seat, pending & (actor == seat) & (s.ch_hp[:, si, seat] > 0))
        seed_card = _hook_card(s, 'seed_card')
        seed_seat = _hook_seat(s, 'seed_seat')
        first = here & (s.turn_count[:, si] == 1) & (s.ch_hp[:, si, seed_seat] > 0) & (s.nanali_seed[:, si] < 3)
        if seed_card >= 0:
            room = first & (s.hand_n[:, si] < HAND_SLOTS)
            full = first & (s.hand_n[:, si] >= HAND_SLOTS)
            for slot in range(HAND_SLOTS):
                put = room & (s.hand_n[:, si] == slot)
                s.hand_kind[:, si, slot] = torch.where(put, torch.full_like(s.hand_kind[:, si, slot], seed_card), s.hand_kind[:, si, slot])
            s.hand_n[:, si] = torch.where(room, s.hand_n[:, si] + 1, s.hand_n[:, si])
            to_discard(s, torch.full_like(s.active, si), torch.full((s.n,), seed_card, dtype=torch.int32, device=s.device), full)
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
        auto_id = CARD_INDEX.get('M08', -1)
        if auto_id >= 0:
            auto = here & (s.ch_shape[:, si, BOHE] == auto_id) & (s.ch_present[:, si, BOHE] > 0) & (s.ch_hp[:, si, BOHE] > 0)
            sortie(s, torch.full_like(s.active, si), BOHE, auto)
    draw(s, side, mask, 1)


def end_turn(s: GpuState, side: torch.Tensor, mask: torch.Tensor) -> None:
    b = _b(s)
    mask = mask & _live(s)
    knockdown(s)
    victory(s)
    for si in range(SIDES):
        here = mask & (side == si)
        front = s.front[:, si]
        z08_id = _hook_card(s, 'turn_end_atk_shape')
        z07_id = _hook_card(s, 'turn_end_heal_shape')
        atk_amt = int(_effect_tables(s)['hooks']['turn_end_atk'])
        heal_amt = int(_effect_tables(s)['hooks']['turn_end_heal'])
        z08 = here & (s.ch_shape[:, si, ZERO] == z08_id) & (s.ch_hp[:, si, ZERO] > 0) & (front >= 0)
        z07 = here & (s.ch_shape[:, si, ZERO] == z07_id) & (s.ch_hp[:, si, ZERO] > 0) & (front >= 0)
        for seat in range(N_SEATS):
            hit = z08 & (front == seat)
            s.ch_atk_buff[:, si, seat] = torch.where(hit, s.ch_atk_buff[:, si, seat] + atk_amt, s.ch_atk_buff[:, si, seat])
            heal = z07 & (front == seat)
            s.ch_hp[:, si, seat] = torch.where(
                heal, torch.minimum(s.ch_hp[:, si, seat] + heal_amt, s.ch_max_hp[:, si, seat]), s.ch_hp[:, si, seat])
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
    tables = _effect_tables(s)
    k = kind.clamp(min=0)
    valid = mask & (kind >= 0)
    perm_atk = tables['perm_atk'][k]
    perm_hp = tables['perm_hp'][k]
    nf = valid & ((perm_atk > 0) | (perm_hp > 0))
    s.ch_base_atk[b, side, seat] = torch.where(nf, s.ch_base_atk[b, side, seat] + perm_atk, s.ch_base_atk[b, side, seat])
    s.ch_max_hp[b, side, seat] = torch.where(nf, s.ch_max_hp[b, side, seat] + perm_hp, s.ch_max_hp[b, side, seat])
    s.ch_base_max[b, side, seat] = torch.where(nf, s.ch_base_max[b, side, seat] + perm_hp, s.ch_base_max[b, side, seat])
    s.ch_hp[b, side, seat] = torch.where(nf, s.ch_hp[b, side, seat] + perm_hp, s.ch_hp[b, side, seat])
    s.nanali_seed_played[b, side] = torch.where(valid & (tables['count_seed'][k] > 0), s.nanali_seed_played[b, side] + 1, s.nanali_seed_played[b, side])
    form = valid & (tables['form'][k] > 0)
    s.ch_shape[b, side, seat] = torch.where(form, kind, s.ch_shape[b, side, seat])
    panel = s.catalog_hp[k]
    perm = (s.ch_base_max[b, side, seat] - s.char_base_hp[seat]).clamp(min=0)
    new_max = torch.where(panel > 0, panel + perm, s.ch_max_hp[b, side, seat])
    s.ch_max_hp[b, side, seat] = torch.where(form & (panel > 0), new_max, s.ch_max_hp[b, side, seat])
    s.ch_hp[b, side, seat] = torch.where(form & (panel > 0), new_max, s.ch_hp[b, side, seat])
    hurt = tables['self_damage'][k]
    self_hit = valid & (hurt > 0)
    _apply_hp(s, side, torch.zeros_like(valid), seat, hurt, self_hit)
    s.ch_allied_hurt[b, side, seat] = torch.where(self_hit, torch.ones_like(s.ch_allied_hurt[b, side, seat]), s.ch_allied_hurt[b, side, seat])
    s.ch_share_overflow[b, side, seat] = torch.where(
        valid & (tables['share_overflow'][k] > 0), torch.ones_like(s.ch_share_overflow[b, side, seat]), s.ch_share_overflow[b, side, seat])
    s.ch_pending_atk[b, side, seat] = torch.where(
        valid & (tables['pending_atk'][k] > 0), s.ch_pending_atk[b, side, seat] + tables['pending_atk'][k], s.ch_pending_atk[b, side, seat])
    team_amt = tables['team_buff'][k]
    team = valid & (team_amt > 0)
    for ally in range(N_SEATS):
        hit = team & (s.ch_present[b, side, ally] > 0) & (s.ch_hp[b, side, ally] > 0)
        s.ch_atk_buff[b, side, ally] = torch.where(hit, s.ch_atk_buff[b, side, ally] + team_amt, s.ch_atk_buff[b, side, ally])
    bonus = s.catalog_atk[k]
    shield = s.catalog_shield[k]
    do_sortie = valid & (tables['sortie'][k] > 0)
    extra = torch.where(do_sortie & (s.ch_allied_hurt[b, side, seat] > 0), tables['sortie_allied_extra'][k], torch.zeros_like(kind))
    fu = torch.where(do_sortie, tables['followup'][k], torch.zeros_like(kind))
    scale_id = CARD_INDEX.get(tables['hooks']['scale_shape'], -1)
    n07 = s.ch_shape[b, side, seat] == scale_id
    played = s.nanali_seed_played[b, side]
    bonus = bonus + extra + torch.where(do_sortie & n07, played, torch.zeros_like(bonus))
    shield = shield + torch.where(do_sortie & n07, played, torch.zeros_like(shield))
    s.ch_shield[b, side, seat] = torch.where(do_sortie, s.ch_shield[b, side, seat] + shield, s.ch_shield[b, side, seat])
    ignore = valid & (tables['ignore_shield'][k] > 0)
    if bool(ignore.any()):
        saved = s.ch_shield[b, other_side(side), :].clone()
        for foe_seat in range(N_SEATS):
            s.ch_shield[b, other_side(side), foe_seat] = torch.where(
                ignore, torch.zeros_like(s.ch_shield[b, other_side(side), foe_seat]), s.ch_shield[b, other_side(side), foe_seat])
        sortie(s, side, seat, do_sortie, card_bonus=torch.where(do_sortie, bonus, torch.zeros_like(bonus)), followup=fu)
        for foe_seat in range(N_SEATS):
            s.ch_shield[b, other_side(side), foe_seat] = torch.where(ignore, saved[:, foe_seat], s.ch_shield[b, other_side(side), foe_seat])
    else:
        sortie(s, side, seat, do_sortie, card_bonus=torch.where(do_sortie, bonus, torch.zeros_like(bonus)), followup=fu)
    heal_hit = tables['heal_after_hit'][k]
    s.ch_hp[b, side, seat] = torch.where(
        do_sortie & (heal_hit > 0),
        torch.minimum(s.ch_hp[b, side, seat] + heal_hit, s.ch_max_hp[b, side, seat]),
        s.ch_hp[b, side, seat])
    zero_seat = _hook_seat(s, 'after_battle_harmony_seat')
    s.ch_harmony[b, side, zero_seat] = torch.where(
        do_sortie & (seat == zero_seat) & (s.ch_hp[b, side, zero_seat] > 0),
        torch.full_like(s.ch_harmony[b, side, zero_seat], 2), s.ch_harmony[b, side, zero_seat])
    front_base = tables['hit_front_base'][k]
    printed = tables['hit_front_printed'][k]
    scaled = valid & (front_base > 0)
    extra_hit = (s.ch_base_atk[b, side, seat] - printed).clamp(min=0)
    _deal_front(s, side, front_base + extra_hit, scaled)
    draw_one = valid & (tables['draw_one'][k] > 0)
    draw(s, side, draw_one, 1)
    follow = valid & (tables['next_followup'][k] > 0)
    s.ch_next_followup[b, side, seat] = torch.where(follow, s.ch_next_followup[b, side, seat] + tables['next_followup'][k], s.ch_next_followup[b, side, seat])
    nb = valid & ((tables['next_bonus_atk'][k] > 0) | (tables['next_bonus_sh'][k] > 0))
    s.ch_next_bonus[b, side, seat] = torch.where(nb, s.ch_next_bonus[b, side, seat] + tables['next_bonus_atk'][k], s.ch_next_bonus[b, side, seat])
    s.ch_next_shield[b, side, seat] = torch.where(nb, s.ch_next_shield[b, side, seat] + tables['next_bonus_sh'][k], s.ch_next_shield[b, side, seat])
    y02 = valid & (tables['draw_owned_target'][k] > 0)
    # draw first deck card owned by tseat
    for si in range(SIDES):
        here = y02 & (side == si)
        for ally in range(N_SEATS):
            want = here & (tseat == ally)
            if not bool(want.any()):
                continue
            for dslot in range(32):
                kind_d = s.deck_kind[:, si, dslot]
                match = want & (kind_d >= 0) & (s.catalog_owner[kind_d.clamp(min=0)] == ally) & (s.catalog_derived[kind_d.clamp(min=0)] == 0)
                # take first match only
                # mark taken
                taken = torch.zeros(s.n, dtype=torch.bool, device=s.device)
                fire = match & ~taken
                # compact is hard; pull into hand then set that deck slot -1 and compact later
                hn = s.hand_n[:, si]
                for hslot in range(HAND_SLOTS):
                    put = fire & (hn == hslot)
                    s.hand_kind[:, si, hslot] = torch.where(put, kind_d, s.hand_kind[:, si, hslot])
                    s.hand_inst[:, si, hslot] = torch.where(put, s.deck_inst[:, si, dslot], s.hand_inst[:, si, hslot])
                s.hand_n[:, si] = torch.where(fire & (hn < HAND_SLOTS), hn + 1, hn)
                s.deck_kind[:, si, dslot] = torch.where(fire, torch.full_like(kind_d, -1), kind_d)
                taken = taken | fire
                want = want & ~fire
    z02 = valid & (tables['draw_forms'][k] > 0)
    for si in range(SIDES):
        here = z02 & (side == si)
        if _cpu_mask_empty(here):
            continue
        found = torch.zeros(s.n, dtype=torch.int32, device=s.device)
        for dslot in range(32):
            kind_d = s.deck_kind[:, si, dslot]
            is_form = (kind_d >= 0) & (s.catalog_type[kind_d.clamp(min=0)] == 2)
            fire = here & is_form & (found < tables['draw_forms'][k].clamp(min=1))
            hn = s.hand_n[:, si]
            for hslot in range(HAND_SLOTS):
                put = fire & (hn == hslot)
                s.hand_kind[:, si, hslot] = torch.where(put, kind_d, s.hand_kind[:, si, hslot])
                s.hand_inst[:, si, hslot] = torch.where(put, s.deck_inst[:, si, dslot], s.hand_inst[:, si, hslot])
            s.hand_n[:, si] = torch.where(fire & (hn < HAND_SLOTS), hn + 1, hn)
            s.deck_kind[:, si, dslot] = torch.where(fire, torch.full_like(kind_d, -1), kind_d)
            found = found + fire.to(found.dtype)
        compact_deck(s, si, here)
    for si in range(SIDES):
        compact_deck(s, si, y02 & (side == si))
    j01 = valid & (tables['inspect_top'][k] > 0)
    for si in range(SIDES):
        here = j01 & (side == si)
        if not bool(here.any()):
            continue
        want_n = tables['inspect_top'][k].clamp(min=1)
        count = torch.minimum(s.deck_n[:, si], want_n)
        s.pending_kind = torch.where(here, torch.ones_like(s.pending_kind), s.pending_kind)
        s.pending_side = torch.where(here, torch.full_like(s.pending_side, si), s.pending_side)
        s.pending_count = torch.where(here, count, s.pending_count)
        for i in range(3):
            take = here & (s.deck_n[:, si] > 0)
            s.pending_card[:, i] = torch.where(take, s.deck_kind[:, si, 0], s.pending_card[:, i])
            # shift deck
            s.deck_kind[:, si, :-1] = torch.where(take.unsqueeze(-1), s.deck_kind[:, si, 1:], s.deck_kind[:, si, :-1])
            s.deck_inst[:, si, :-1] = torch.where(take.unsqueeze(-1), s.deck_inst[:, si, 1:], s.deck_inst[:, si, :-1])
            s.deck_n[:, si] = torch.where(take, s.deck_n[:, si] - 1, s.deck_n[:, si])
        one = here & (count == 1)
        # auto add
        hn = s.hand_n[:, si]
        for hslot in range(HAND_SLOTS):
            put = one & (hn == hslot)
            s.hand_kind[:, si, hslot] = torch.where(put, s.pending_card[:, 0], s.hand_kind[:, si, hslot])
        s.hand_n[:, si] = torch.where(one & (hn < HAND_SLOTS), hn + 1, hn)
        choose = here & (count > 1)
        s.phase = torch.where(choose, torch.full_like(s.phase, PHASE_CHOICE), s.phase)
        s.pending_kind = torch.where(one, torch.full_like(s.pending_kind, -1), s.pending_kind)
    heal_amt = tables['heal_target'][k]
    buff_amt = tables['buff_target'][k]
    y03 = valid & (heal_amt > 0)
    for ally in range(N_SEATS):
        hit = y03 & (tside == side) & (tseat == ally)
        s.ch_hp[b, side, ally] = torch.where(hit, torch.minimum(s.ch_hp[b, side, ally] + heal_amt, s.ch_max_hp[b, side, ally]), s.ch_hp[b, side, ally])
        s.ch_atk_buff[b, side, ally] = torch.where(hit, s.ch_atk_buff[b, side, ally] + buff_amt, s.ch_atk_buff[b, side, ally])
    y03p = valid & (tables['heal_player'][k] > 0) & (tseat < 0)
    s.hp[b, side] = torch.where(y03p, torch.minimum(s.hp[b, side] + tables['heal_player'][k], torch.full_like(s.hp[b, side], 30)), s.hp[b, side])
    y06 = valid & (tables['revive_target'][k] > 0)
    for ally in range(N_SEATS):
        hit = y06 & (tseat == ally)
        s.last_front[b, side] = torch.where(hit & (s.last_front[b, side] == ally), torch.full_like(s.last_front[b, side], -1), s.last_front[b, side])
        s.ch_down[b, side, ally] = torch.where(hit, torch.zeros_like(s.ch_down[b, side, ally]), s.ch_down[b, side, ally])
        s.ch_hp[b, side, ally] = torch.where(hit, s.ch_max_hp[b, side, ally], s.ch_hp[b, side, ally])
        s.ch_atk_buff[b, side, ally] = torch.where(hit, s.ch_atk_buff[b, side, ally] + buff_amt, s.ch_atk_buff[b, side, ally])
    j02 = valid & (tables['pact_front'][k] > 0)
    j04 = valid & (tables['pact_target'][k] > 0)
    _deal_front(s, side, tables['pact_front'][k], j02)
    for foe_seat in range(N_SEATS):
        foe = other_side(side)
        hit = j04 & (tseat == foe_seat)
        _apply_hp(s, foe, torch.zeros_like(hit), foe_seat, tables['pact_target'][k], hit)
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
    nf01 = CARD_INDEX.get('NF01', -1)
    if nf01 >= 0:
        instant = instant | ((kind == nf01) & (s.turn_count[b, side] >= 6))
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
    to_discard(s, side, kind, mask & (kind >= 0))


def choose_pending(s: GpuState, slot: torch.Tensor, mask: torch.Tensor) -> None:
    mask = mask & (s.phase == PHASE_CHOICE) & _live(s)
    if _cpu_mask_empty(mask):
        return
    side = s.pending_side.clamp(min=0)
    b = _b(s)
    selected = torch.zeros(s.n, dtype=torch.int32, device=s.device)
    for i in range(10):
        selected = torch.where(slot == i, s.pending_card[:, i], selected)
    for si in range(SIDES):
        here = mask & (side == si)
        hn = s.hand_n[:, si]
        for hslot in range(HAND_SLOTS):
            put = here & (hn == hslot)
            s.hand_kind[:, si, hslot] = torch.where(put, selected, s.hand_kind[:, si, hslot])
        s.hand_n[:, si] = torch.where(here & (hn < HAND_SLOTS), hn + 1, hn)
        # rest of pending to deck bottom
        for i in range(10):
            rest = here & (i < s.pending_count) & (slot != i) & (s.pending_card[:, i] >= 0)
            n = s.deck_n[:, si]
            for dslot in range(32):
                put = rest & (n == dslot)
                s.deck_kind[:, si, dslot] = torch.where(put, s.pending_card[:, i], s.deck_kind[:, si, dslot])
            s.deck_n[:, si] = torch.where(rest & (n < 32), n + 1, n)
    s.phase = torch.where(mask, torch.full_like(s.phase, PHASE_PLAYING), s.phase)
    s.pending_kind = torch.where(mask, torch.full_like(s.pending_kind, -1), s.pending_kind)
    s.pending_count = torch.where(mask, torch.zeros_like(s.pending_count), s.pending_count)


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
    b = _b(s)
    playing = (s.phase == PHASE_PLAYING) & _live(s)
    choice = (s.phase == PHASE_CHOICE) & _live(s)
    side = s.active
    ap = s.ap[b, side]
    s.legal_type[:, LEGAL_END] = torch.where(playing, torch.full_like(s.legal_type[:, 0], ACT_END), -1)
    for seat in range(N_SEATS):
        can_atk = playing & (ap >= 1) & (s.normal_atk[b, side] > 0) & (s.ch_present[b, side, seat] > 0) & (s.ch_hp[b, side, seat] > 0) & (s.ch_down[b, side, seat] == 0)
        s.legal_type[:, LEGAL_ATK + seat] = torch.where(can_atk, torch.full_like(s.legal_type[:, 0], ACT_ATTACK), -1)
        s.legal_actor[:, LEGAL_ATK + seat] = torch.where(can_atk, torch.full_like(s.legal_actor[:, 0], seat), -1)
        can_ult = playing & (s.ultimate_ok[b, side] > 0) & (s.ch_present[b, side, seat] > 0) & (s.ch_hp[b, side, seat] > 0) & (s.ch_energy[b, side, seat] >= s.ch_energy_max[b, side, seat])
        s.legal_type[:, LEGAL_ULT + seat] = torch.where(can_ult, torch.full_like(s.legal_type[:, 0], ACT_ULTIMATE), -1)
        s.legal_actor[:, LEGAL_ULT + seat] = torch.where(can_ult, torch.full_like(s.legal_actor[:, 0], seat), -1)
    foe = other_side(side)
    for hs in range(HAND_SLOTS):
        kind = torch.where(side == 0, s.hand_kind[b, 0, hs], s.hand_kind[b, 1, hs])
        valid_card = playing & (kind >= 0)
        owner = torch.where(kind >= 0, s.catalog_owner[kind.clamp(min=0)], torch.zeros_like(kind))
        cost = torch.where(kind >= 0, s.catalog_cost[kind.clamp(min=0)], torch.zeros_like(kind))
        policy = torch.where(kind >= 0, s.catalog_policy[kind.clamp(min=0)], torch.zeros_like(kind))
        owner_hp = torch.zeros_like(kind)
        for seat in range(N_SEATS):
            owner_hp = torch.where(owner == seat, s.ch_hp[b, side, seat], owner_hp)
        instant = (kind >= 0) & (s.catalog_instant[kind.clamp(min=0)] > 0)
        nf01 = CARD_INDEX.get('NF01', -1)
        if nf01 >= 0:
            instant = instant | ((kind == nf01) & (s.turn_count[b, side] >= 6))
        is_form = (kind >= 0) & (s.catalog_type[kind.clamp(min=0)] == 2)
        zero_shaped = (s.ch_hp[b, side, ZERO] > 0) & (s.ch_shape[b, side, ZERO] >= 0)
        free = (instant | (is_form & zero_shaped)) & (s.used_instant[b, side] == 0)
        playable = valid_card & (owner_hp > 0) & (free | (ap >= cost))
        base = LEGAL_PLAY + hs * PLAY_TARGETS
        none_ok = playable & ((policy == POLICY_NONE) | ((policy == POLICY_ENEMY_FRONT) & (s.front[b, foe] >= 0)))
        s.legal_type[:, base] = torch.where(none_ok, torch.full_like(kind, ACT_PLAY), -1)
        s.legal_actor[:, base] = torch.where(none_ok, owner, -1)
        s.legal_hand[:, base] = torch.where(none_ok, torch.full_like(kind, hs), -1)
        for ally in range(N_SEATS):
            slot = base + 1 + ally
            ally_ok = playable & (s.ch_present[b, side, ally] > 0) & (
                ((policy == POLICY_ALLY) & (s.ch_hp[b, side, ally] > 0))
                | ((policy == POLICY_INJURED) & (s.ch_hp[b, side, ally] > 0) & (s.ch_hp[b, side, ally] < s.ch_max_hp[b, side, ally]))
                | ((policy == POLICY_DOWN_ALLY) & ((s.ch_down[b, side, ally] > 0) | (s.ch_hp[b, side, ally] <= 0)))
            )
            s.legal_type[:, slot] = torch.where(ally_ok, torch.full_like(kind, ACT_PLAY), -1)
            s.legal_actor[:, slot] = torch.where(ally_ok, owner, -1)
            s.legal_hand[:, slot] = torch.where(ally_ok, torch.full_like(kind, hs), -1)
            s.legal_tside[:, slot] = torch.where(ally_ok, side, -1)
            s.legal_tseat[:, slot] = torch.where(ally_ok, torch.full_like(kind, ally), -1)
        for enemy in range(N_SEATS):
            slot = base + 1 + N_SEATS + enemy
            en_ok = playable & (policy == POLICY_ENEMY) & (s.ch_present[b, foe, enemy] > 0) & (s.ch_hp[b, foe, enemy] > 0)
            s.legal_type[:, slot] = torch.where(en_ok, torch.full_like(kind, ACT_PLAY), -1)
            s.legal_actor[:, slot] = torch.where(en_ok, owner, -1)
            s.legal_hand[:, slot] = torch.where(en_ok, torch.full_like(kind, hs), -1)
            s.legal_tside[:, slot] = torch.where(en_ok, foe, -1)
            s.legal_tseat[:, slot] = torch.where(en_ok, torch.full_like(kind, enemy), -1)
        slot = base + 1 + 2 * N_SEATS
        ply = playable & (policy == POLICY_INJURED) & (s.hp[b, side] < 30)
        s.legal_type[:, slot] = torch.where(ply, torch.full_like(kind, ACT_PLAY), -1)
        s.legal_actor[:, slot] = torch.where(ply, owner, -1)
        s.legal_hand[:, slot] = torch.where(ply, torch.full_like(kind, hs), -1)
        s.legal_tside[:, slot] = torch.where(ply, side, -1)
        s.legal_tseat[:, slot] = torch.where(ply, torch.full_like(kind, -1), -1)
    for cs in range(10):
        ok = choice & (cs < s.pending_count)
        s.legal_type[:, cs] = torch.where(ok, torch.full_like(s.legal_type[:, 0], ACT_CHOOSE), s.legal_type[:, cs])
        s.legal_hand[:, cs] = torch.where(ok, torch.full_like(s.legal_hand[:, 0], cs), s.legal_hand[:, cs])
    valid = s.legal_type >= 0
    # compact? keep sparse fixed layout; legal_n is last valid+1
    slots = torch.arange(MAX_LEGAL, device=s.device).unsqueeze(0).expand(s.n, -1)
    s.legal_n = torch.where(valid, slots + 1, torch.zeros_like(slots)).max(dim=1).values
    if int((s.legal_n > MAX_LEGAL).any()):
        raise ValueError('Legal action overflow')


def step_index(s: GpuState, index: torch.Tensor, *, legal_ready: bool = False) -> None:
    """External callers rebuild; internal callers may reuse an unchanged action table.

    Every successful return leaves the table rebuilt for the resulting state.
    """
    if not legal_ready:
        rebuild_legal(s)
    b = _b(s)
    # Clamp only the gather index. Negative indices are batch no-ops, not slot 0.
    idx = index.clamp(min=0, max=s.legal_type.shape[1] - 1)
    valid = _live(s) & (index >= 0) & (index < s.legal_n) & (index < s.legal_type.shape[1])
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
    choose = valid & (typ == ACT_CHOOSE)
    choose_pending(s, hand, choose)
    conc = valid & (typ == ACT_CONCEDE)
    s.phase = torch.where(conc, torch.full_like(s.phase, PHASE_FINISHED), s.phase)
    s.winner = torch.where(conc, other_side(side), s.winner)
    s.reason = torch.where(conc, torch.full_like(s.reason, 3), s.reason)
    rebuild_legal(s)
