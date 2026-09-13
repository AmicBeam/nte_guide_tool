"""GPU-native reset for finished rows. No Python engine on the hot path."""
from __future__ import annotations

import torch

from .catalog import CARD_INDEX, N_SEATS, PHASE_PLAYING, SIDES, STARTER_CARD_IDS
from .rules import begin_turn, draw, rebuild_legal
from .state import GpuState, empty_state


def starter_kind_row(device) -> torch.Tensor:
    return torch.tensor([CARD_INDEX[cid] for cid in STARTER_CARD_IDS], dtype=torch.int32, device=device)


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


def reset_native(s: GpuState | None, n: int, device, seeds: torch.Tensor) -> GpuState:
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
    s.hand_n.zero_()
    s.deck_n.zero_()
    s.discard_n.zero_()
    s.removed_n.zero_()
    s.hand_kind.fill_(-1)
    s.hand_inst.zero_()
    s.deck_kind.fill_(-1)
    s.deck_inst.zero_()
    s.discard_kind.fill_(-1)
    kinds = starter_kind_row(device)
    for side in range(SIDES):
        deck = kinds.unsqueeze(0).repeat(n, 1).contiguous()
        deck = _shuffle_rows(deck, seeds + (side + 1) * 10007)
        s.deck_kind[:, side, :32] = deck
        s.deck_inst[:, side, :32] = torch.arange(1, 33, device=device).unsqueeze(0).expand(n, -1)
        s.deck_n[:, side] = 32
        for seat in range(N_SEATS):
            s.ch_hp[:, side, seat] = s.char_base_hp[seat]
            s.ch_max_hp[:, side, seat] = s.char_base_hp[seat]
            s.ch_base_max[:, side, seat] = s.char_base_hp[seat]
            s.ch_shield[:, side, seat] = 0
            s.ch_base_atk[:, side, seat] = s.char_base_atk[seat]
            s.ch_growth[:, side, seat] = 0
            s.ch_harmony[:, side, seat] = 0
            s.ch_energy[:, side, seat] = 0
            s.ch_energy_max[:, side, seat] = s.char_energy_max[seat]
            s.ch_down[:, side, seat] = 0
            s.ch_awakened[:, side, seat] = 0
            s.ch_ult_turns[:, side, seat] = 0
            s.ch_shape[:, side, seat] = -1
            s.ch_atk_buff[:, side, seat] = 0
            s.ch_next_bonus[:, side, seat] = 0
            s.ch_next_shield[:, side, seat] = 0
            s.ch_next_followup[:, side, seat] = 0
            s.ch_pact[:, side, seat] = 0
            s.ch_energy_next[:, side, seat] = 0
            s.ch_harmonized[:, side, seat] = 0
        draw(s, torch.full((n,), side, dtype=torch.int32, device=device), torch.ones(n, dtype=torch.bool, device=device), 5)
    s.phase.fill_(PHASE_PLAYING)
    begin_turn(s, first, torch.ones(n, dtype=torch.bool, device=device))
    rebuild_legal(s)
    return s


def reset_rows(s: GpuState, mask: torch.Tensor, seeds: torch.Tensor) -> None:
    if not bool(mask.any()):
        return
    tmp = reset_native(None, s.n, s.device, seeds)
    for name in s.__dataclass_fields__:
        if name in ('n', 'device'):
            continue
        tensor = getattr(s, name)
        other = getattr(tmp, name)
        if torch.is_tensor(tensor) and tensor.shape[0] == s.n:
            tensor[mask] = other[mask]
    rebuild_legal(s)
