"""GPU-native reset for finished rows. No Python engine on the hot path."""
from __future__ import annotations

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECK

from .catalog import N_SEATS, PHASE_PLAYING, SIDES, deck_kind_ids, present_mask
from .rules import begin_turn, draw, rebuild_legal
from .state import GpuState, empty_state


def _kind_row(deck, device) -> torch.Tensor:
    return torch.tensor(deck_kind_ids(deck), dtype=torch.int32, device=device)


def _shuffle_rows(deck: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    n, width = deck.shape
    b = torch.arange(n, device=deck.device)
    state = rng.to(torch.int64) & 0x7FFFFFFF
    for i in range(width - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = (state % (i + 1)).to(torch.int64)
        cur = deck[b, i].clone()
        swp = deck[b, j].clone()
        deck[b, i] = swp
        deck[b, j] = cur
    return deck


def reset_native(s: GpuState | None, n: int, device, seeds: torch.Tensor,
                 *, deck_a=None, deck_b=None, escalation=True) -> GpuState:
    device = torch.device(device)
    if s is None or s.n != n:
        s = empty_state(n, device)
    seeds = seeds.to(device=device, dtype=torch.int32)
    b = torch.arange(n, device=device)
    s.phase.zero_()
    s.turn.zero_()
    s.winner.fill_(-1)
    s.reason.zero_()
    s.rng.copy_(seeds)
    if hasattr(s, "escalation_enabled"):
        s.escalation_enabled.fill_(1 if escalation else 0)
    s.next_instance.fill_(32)
    s.pending_kind.fill_(-1)
    s.pending_side.fill_(-1)
    s.pending_count.zero_()
    s.pending_card.fill_(-1)
    first = seeds % 2
    s.first.copy_(first)
    s.active.copy_(first)
    s.hp.fill_(30)
    s.shield.zero_()
    s.shield[b, 1 - first] = 5
    s.ap.zero_()
    s.front.fill_(-1)
    s.last_front.fill_(-1)
    s.turn_count.zero_()
    s.fatigue.zero_()
    s.normal_atk.zero_()
    s.ultimate_ok.zero_()
    s.used_instant.zero_()
    s.extra_flower.zero_()
    s.extra_ap.zero_()
    s.extra_genesis.zero_()
    s.extra_genesis_actor.fill_(-1)
    s.surplus.zero_()
    s.nanali_seed.zero_()
    s.nanali_seed_played.zero_()
    s.burn_left.zero_()
    s.delay_end.zero_()
    if hasattr(s, "ultimates_used"):
        s.ultimates_used.zero_()
        s.burn_by.fill_(-1)
        s.burn_infinite.zero_()
        s.harmony_damage.zero_()
        s.hand_revealed.zero_()
    s.weave.zero_()
    s.ch_present.zero_()
    s.ch_allied_hurt.zero_()
    s.ch_pending_atk.zero_()
    s.ch_share_overflow.zero_()
    s.hand_n.zero_()
    s.deck_n.zero_()
    s.discard_n.zero_()
    s.removed_n.zero_()
    s.hand_kind.fill_(-1)
    s.hand_inst.zero_()
    s.deck_kind.fill_(-1)
    s.deck_inst.zero_()
    s.discard_kind.fill_(-1)
    decks = (deck_a or STARTER_DECK, deck_b or STARTER_DECK)
    presents = (
        torch.tensor(present_mask(decks[0]), dtype=torch.int32, device=device),
        torch.tensor(present_mask(decks[1]), dtype=torch.int32, device=device),
    )
    for side in range(SIDES):
        kinds = _kind_row(decks[side], device)
        deck = kinds.unsqueeze(0).repeat(n, 1).contiguous()
        deck = _shuffle_rows(deck, seeds + (side + 1) * 10007)
        s.deck_kind[:, side, :32] = deck
        s.deck_inst[:, side, :32] = torch.arange(1, 33, device=device).unsqueeze(0).expand(n, -1)
        s.deck_n[:, side] = 32
        for seat in range(N_SEATS):
            here = int(presents[side][seat]) > 0
            s.ch_present[:, side, seat] = 1 if here else 0
            zero = torch.zeros((), dtype=s.char_base_hp.dtype, device=device)
            base_hp = s.char_base_hp[seat] if here else zero
            base_atk = s.char_base_atk[seat] if here else zero
            s.ch_hp[:, side, seat] = base_hp
            s.ch_max_hp[:, side, seat] = base_hp
            s.ch_base_max[:, side, seat] = base_hp
            s.ch_shield[:, side, seat] = 0
            s.ch_base_atk[:, side, seat] = base_atk
            s.ch_growth[:, side, seat] = 0
            s.ch_harmony[:, side, seat] = 0
            s.ch_energy[:, side, seat] = 0
            s.ch_energy_max[:, side, seat] = s.char_energy_max[seat] if here else 0
            s.ch_down[:, side, seat] = 0
            s.ch_awakened[:, side, seat] = 0
            s.ch_ult_turns[:, side, seat] = 0
            s.ch_ultimate_used_turn[:, side, seat] = -1
            s.ch_shape[:, side, seat] = -1
            s.ch_atk_buff[:, side, seat] = 0
            s.ch_next_bonus[:, side, seat] = 0
            s.ch_next_shield[:, side, seat] = 0
            s.ch_next_followup[:, side, seat] = 0
            s.ch_pact[:, side, seat] = 0
            s.ch_energy_next[:, side, seat] = 0
            s.ch_harmonized[:, side, seat] = 0
            if hasattr(s, 'ch_hp_floor'):
                s.ch_hp_floor[:, side, seat] = 0
                s.ch_order[:, side, seat] = 0
        if hasattr(s, 'ch_order'):
            from .catalog import SEAT_INDEX
            for rank, cid in enumerate(decks[side]['character_ids']):
                s.ch_order[:, side, SEAT_INDEX[cid]] = rank + 1
        draw(s, torch.full((n,), side, dtype=torch.int32, device=device), torch.ones(n, dtype=torch.bool, device=device), 5)
    s.phase.fill_(PHASE_PLAYING)
    begin_turn(s, first, torch.ones(n, dtype=torch.bool, device=device))
    rebuild_legal(s)
    return s


def reset_rows(s: GpuState, mask: torch.Tensor, seeds: torch.Tensor, *, deck_a=None, deck_b=None) -> None:
    if not bool(mask.any()):
        return
    tmp = reset_native(None, s.n, s.device, seeds, deck_a=deck_a, deck_b=deck_b)
    for name in s.__dataclass_fields__:
        if name in ('n', 'device'):
            continue
        tensor = getattr(s, name)
        other = getattr(tmp, name)
        if torch.is_tensor(tensor) and tensor.shape[0] == s.n:
            tensor[mask] = other[mask]
    rebuild_legal(s)


def reset_public_gpu_state(state: GpuState, seeds, deck_rows_a, deck_rows_b, escalation=True) -> GpuState:
    """Reset device-resident public custom four-person decks.

    seeds: int32 [n]; negative values preserve that row.
    deck_rows_a/b: int32 [n,36] = 4 seat indices in actual team order + 32 card kinds.
    All tensors must already live on ``state.device``. Existing ``reset_gpu_state`` /
    ``reset_native`` exact-preset paths remain unchanged.
    """
    import torch
    from .catalog import N_SEATS, SEATS, decode_public_deck_row
    seeds = torch.as_tensor(seeds, dtype=torch.int32, device=state.device).contiguous()
    deck_rows_a = torch.as_tensor(deck_rows_a, dtype=torch.int32, device=state.device).contiguous()
    deck_rows_b = torch.as_tensor(deck_rows_b, dtype=torch.int32, device=state.device).contiguous()
    if seeds.ndim != 1 or seeds.shape[0] != state.n:
        raise ValueError('public reset seeds must be [n]')
    if deck_rows_a.shape != (state.n, 36) or deck_rows_b.shape != (state.n, 36):
        raise ValueError('public deck rows must be [n,36]')
    if any(t.device != state.device for t in (seeds, deck_rows_a, deck_rows_b)):
        raise ValueError('public reset tensors must already live on the compiled device')
    # Native tensor helper still uses one shared deck pair for the batch. Callers
    # that need mixed rows should use the compiled public_reset kernel.
    deck_a = decode_public_deck_row(deck_rows_a[0].tolist())
    deck_b = decode_public_deck_row(deck_rows_b[0].tolist())
    mask = seeds >= 0
    if bool(mask.all()):
        return reset_native(state, state.n, state.device, seeds, deck_a=deck_a, deck_b=deck_b, escalation=escalation)
    tmp = reset_native(None, state.n, state.device, torch.where(mask, seeds, torch.zeros_like(seeds)),
                       deck_a=deck_a, deck_b=deck_b, escalation=escalation)
    for name in state.__dataclass_fields__:
        if name in ('n', 'device'):
            continue
        tensor = getattr(state, name)
        other = getattr(tmp, name)
        if torch.is_tensor(tensor) and tensor.shape[0] == state.n:
            tensor[mask] = other[mask]
    from .rules import rebuild_legal
    rebuild_legal(state)
    return state
