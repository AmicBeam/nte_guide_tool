"""Compact public tensors for GPU PPO. Not the website JSON encoder."""
from __future__ import annotations

import torch

from .catalog import HAND_SLOTS, MAX_LEGAL, N_SEATS
from .state import GpuState, other_side


def compact_observe(s: GpuState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = torch.arange(s.n, device=s.device)
    side = s.active
    foe = other_side(side)
    parts = [
        s.phase, s.active, s.turn,
        s.hp[b, side], s.hp[b, foe], s.shield[b, side], s.shield[b, foe],
        s.ap[b, side], s.front[b, side], s.front[b, foe],
        s.turn_count[b, side], s.hand_n[b, side], s.deck_n[b, side],
        s.normal_atk[b, side], s.ultimate_ok[b, side],
    ]
    for seat in range(N_SEATS):
        parts.extend([
            s.ch_hp[b, side, seat], s.ch_hp[b, foe, seat],
            s.ch_harmony[b, side, seat], s.ch_energy[b, side, seat],
            s.ch_down[b, side, seat], s.ch_awakened[b, side, seat], s.ch_shape[b, side, seat],
        ])
    state = torch.stack(parts, dim=1).to(torch.float32)
    card_kind = torch.full((s.n, MAX_LEGAL), -1, dtype=torch.int32, device=s.device)
    for hs in range(HAND_SLOTS):
        hk = torch.where(side == 0, s.hand_kind[b, 0, hs], s.hand_kind[b, 1, hs])
        card_kind = torch.where(s.legal_hand == hs, hk.unsqueeze(1).expand_as(card_kind), card_kind)
    cost = torch.where(card_kind >= 0, s.catalog_cost[card_kind.clamp(min=0)], torch.zeros_like(card_kind))
    cand = torch.stack([
        s.legal_type, s.legal_actor, s.legal_hand, s.legal_tside, s.legal_tseat, card_kind, cost,
    ], dim=-1).to(torch.float32)
    mask = s.legal_type >= 0
    return state, cand, mask
