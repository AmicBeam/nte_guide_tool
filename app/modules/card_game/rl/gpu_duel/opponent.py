"""GPU rule opponent: same public priority as a simple visible rule policy."""
from __future__ import annotations

import torch

from .catalog import ACT_ATTACK, ACT_CHOOSE, ACT_END, ACT_PLAY, ACT_ULTIMATE, MAX_LEGAL
from .rules import rebuild_legal, step_index
from .state import GpuState


def rule_action(s: GpuState) -> torch.Tensor:
    types = s.legal_type
    valid = types >= 0
    order = (ACT_CHOOSE, ACT_PLAY, ACT_ULTIMATE, ACT_ATTACK, ACT_END)
    chosen = torch.full((s.n,), -1, dtype=torch.int32, device=s.device)
    picked = torch.zeros(s.n, dtype=torch.bool, device=s.device)
    slots = torch.arange(MAX_LEGAL, device=s.device)
    for kind in order:
        hit = valid & (types == kind) & ~picked.unsqueeze(1)
        any_hit = hit.any(dim=1)
        # first matching slot
        masked = torch.where(hit, slots.unsqueeze(0).expand_as(hit), torch.full_like(types, MAX_LEGAL))
        first = masked.min(dim=1).values
        first = torch.where(first < MAX_LEGAL, first, torch.full_like(first, -1))
        chosen = torch.where(any_hit & ~picked, first.to(torch.int32), chosen)
        picked = picked | any_hit
    return chosen


def advance_to_actor(s: GpuState, learner: torch.Tensor, *, max_ops: int = 24,
                     legal_ready: bool = False) -> None:
    if not legal_ready:
        rebuild_legal(s)
    for _ in range(max_ops):
        done = s.phase == 3
        need = ~done & (s.active != learner)
        if not bool(need.any()):
            return
        act = rule_action(s)
        act = torch.where(need, act, torch.full_like(act, -1))
        step_index(s, act, legal_ready=True)
