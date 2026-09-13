"""GPU-resident 创生预组 duel. CPU is used for startup/oracle pack only."""
from __future__ import annotations

from typing import Any

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECK
from app.modules.card_game.engine.duel_v2 import new_game

from .catalog import STARTER_LOCK, assert_starter_deck
from .pack import pack_python, snapshot_mechan
from .rules import rebuild_legal, step_index
from .state import GpuState


class GpuDuelEnv:
    """Batched integer-state environment. device='cuda' keeps the loop on GPU."""

    def __init__(self, n: int, device: str | torch.device | None = None):
        if n < 1:
            raise ValueError('n must be positive')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.n = n
        self.lock = STARTER_LOCK
        self.deck = assert_starter_deck(STARTER_DECK)
        self.state: GpuState | None = None

    def reset_from_python(self, states: list[dict[str, Any]]) -> GpuState:
        if len(states) != self.n:
            raise ValueError(f'expected {self.n} python states, got {len(states)}')
        self.state = pack_python(states, device=self.device)
        rebuild_legal(self.state)
        return self.state

    def reset(self, seeds: list[int] | None = None, *, skip_mulligan: bool = True) -> GpuState:
        seeds = list(seeds or range(self.n))
        if len(seeds) != self.n:
            raise ValueError('seeds must match batch size')
        decks = {'a': self.deck, 'b': self.deck}
        states = [new_game(seed=int(seed), skip_mulligan=skip_mulligan, decks=decks) for seed in seeds]
        return self.reset_from_python(states)

    def legal_count(self) -> torch.Tensor:
        return self.state.legal_n

    def step(self, action_index: torch.Tensor | list[int]) -> GpuState:
        if self.state is None:
            raise RuntimeError('reset first')
        if not torch.is_tensor(action_index):
            action_index = torch.tensor(action_index, dtype=torch.int32, device=self.device)
        action_index = action_index.to(self.device).reshape(self.n)
        step_index(self.state, action_index)
        return self.state

    def snapshot(self, row: int = 0) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError('reset first')
        return snapshot_mechan(self.state, row)

    def outcome(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state is None:
            raise RuntimeError('reset first')
        done = self.state.phase == 3
        return done, self.state.winner
