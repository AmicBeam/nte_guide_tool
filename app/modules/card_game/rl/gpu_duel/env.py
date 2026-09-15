"""GPU-resident 创生/覆纹快攻 duel. CPU is used for startup/oracle pack only."""
from __future__ import annotations

from typing import Any

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECK
from app.modules.card_game.engine.duel_v2 import new_game

from ..episode_return import reset_episode_return
from .catalog import GPU_LOCK, assert_gpu_deck, preset_by_id
from .opponent import advance_to_actor, rule_action
from .pack import pack_python, snapshot_mechan
from .reset import reset_native, reset_rows
from .rules import rebuild_legal, step_index
from .state import GpuState


class GpuDuelEnv:
    """Batched integer-state environment. device='cuda' keeps the loop on GPU."""

    def __init__(self, n: int, device: str | torch.device | None = None, *,
                 deck: dict[str, Any] | str | None = None, matchup: str = 'mirror',
                 compiled_backend: str | None = None, compiled_dir: str | None = None):
        if n < 1:
            raise ValueError('n must be positive')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.n = n
        self.lock = GPU_LOCK
        self.matchup = matchup
        if matchup == 'cross':
            self.deck_a = preset_by_id('starter')
            self.deck_b = preset_by_id('weave-rush')
        else:
            if isinstance(deck, str):
                chosen = preset_by_id(deck)
            else:
                chosen = assert_gpu_deck(deck or STARTER_DECK)
            self.deck_a = chosen
            self.deck_b = chosen
        self.deck = self.deck_a
        self.state: GpuState | None = None
        self.learner = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.compiled = None
        if compiled_backend:
            from .compiled_backend import CompiledStarterBackend, _assert_compiled_deck
            if matchup != 'mirror':
                raise ValueError('compiled backend only supports preset mirror matchups')
            _assert_compiled_deck(self.deck_a)
            _assert_compiled_deck(self.deck_b)
            self.compiled = CompiledStarterBackend(compiled_dir, backend=compiled_backend, deck_id=self.deck_a['id'])

    def reset_from_python(self, states: list[dict[str, Any]]) -> GpuState:
        if len(states) != self.n:
            raise ValueError(f'expected {self.n} python states, got {len(states)}')
        reset_episode_return(self)
        if self.compiled is not None:
            from .state import empty_state
            from ..rule_ir.pack_row import bind_gpu_rows,pack_python_row
            self.state=empty_state(self.n,self.device)
            bind_gpu_rows(self.state).copy_(torch.tensor([pack_python_row(s) for s in states],dtype=torch.int32,device=self.device))
            self.compiled.step_gpu_state(self.state, [-1] * self.n)
        else:
            self.state = pack_python(states, device=self.device)
            rebuild_legal(self.state)
        return self.state

    def reset(self, seeds: list[int] | None = None, *, skip_mulligan: bool = True,
              native: bool = True) -> GpuState:
        seeds = list(seeds or range(self.n))
        if len(seeds) != self.n:
            raise ValueError('seeds must match batch size')
        reset_episode_return(self)
        seed_t = torch.tensor(seeds, dtype=torch.int32, device=self.device)
        # Half the batch learns as first player, half as second.
        self.learner = ((seed_t // 2) % 2).to(torch.int32)
        if self.compiled is not None and native and skip_mulligan:
            from app.modules.card_game.rl.gpu_duel.state import empty_state
            self.state = empty_state(self.n, self.device)
            self.compiled.reset_gpu_state(self.state, seed_t)
            self._advance_to_actor(legal_ready=True)
            return self.state
        if native and skip_mulligan:
            self.state = reset_native(
                self.state, self.n, self.device, seed_t, deck_a=self.deck_a, deck_b=self.deck_b)
            self._advance_to_actor(legal_ready=True)
            return self.state
        decks = {'a': self.deck_a, 'b': self.deck_b}
        states = [new_game(seed=int(seed), skip_mulligan=skip_mulligan, decks=decks) for seed in seeds]
        self.reset_from_python(states)
        self._advance_to_actor(legal_ready=True)
        return self.state

    def reset_public(self, seeds: list[int] | None = None, *,
                     deck_rows_a=None, deck_rows_b=None, escalation: bool = True) -> GpuState:
        """Reset compiled rows from public custom four-person decks.

        Preset ``reset()`` / ``reset_gpu_state`` remain unchanged. Custom decks
        must already be encoded as int32 [n,36] rows on this env device.
        """
        seeds = list(seeds or range(self.n))
        if len(seeds) != self.n:
            raise ValueError('seeds must match batch size')
        if deck_rows_a is None or deck_rows_b is None:
            raise ValueError('public reset requires deck_rows_a and deck_rows_b')
        reset_episode_return(self)
        seed_t = torch.as_tensor(seeds, dtype=torch.int32, device=self.device).contiguous()
        deck_a = torch.as_tensor(deck_rows_a, dtype=torch.int32, device=self.device).contiguous()
        deck_b = torch.as_tensor(deck_rows_b, dtype=torch.int32, device=self.device).contiguous()
        self.learner = ((seed_t // 2) % 2).to(torch.int32)
        from app.modules.card_game.rl.gpu_duel.state import empty_state
        self.state = empty_state(self.n, self.device)
        if self.compiled is not None:
            self.compiled.reset_public_gpu_state(self.state, seed_t, deck_a, deck_b, escalation=escalation)
        else:
            from .reset import reset_public_gpu_state
            reset_public_gpu_state(self.state, seed_t, deck_a, deck_b, escalation=escalation)
        self._advance_to_actor(legal_ready=True)
        return self.state

    def reset_finished(self, seeds: torch.Tensor | None = None) -> None:
        if self.state is None:
            return
        done, _winner = self.outcome()
        if not bool(done.any()):
            return
        if seeds is None:
            seeds = (self.state.rng + 7919) & 0x7FFFFFFF
        if self.compiled is not None:
            self.compiled.reset_gpu_state(self.state, torch.where(done, seeds, -1))
        else:
            reset_rows(self.state, done, seeds, deck_a=self.deck_a, deck_b=self.deck_b)
        reset_episode_return(self, done)
        # keep learner assignment
        self._advance_to_actor()

    def prepare_decision(self) -> None:
        """Keep legality and opponent advancement on the selected executor."""
        if self.compiled is None:
            rebuild_legal(self.state)
        else:
            self.compiled.step_gpu_state(self.state, [-1] * self.n)
        self._advance_to_actor(legal_ready=True)

    def legal_count(self) -> torch.Tensor:
        return self.state.legal_n

    def step(self, action_index: torch.Tensor | list[int]) -> GpuState:
        if self.state is None:
            raise RuntimeError('reset first')
        if not torch.is_tensor(action_index):
            action_index = torch.tensor(action_index, dtype=torch.int32, device=self.device)
        action_index = action_index.to(self.device).reshape(self.n)
        self._dispatch_step(action_index)
        return self.state

    def step_learner(self, action_index: torch.Tensor | list[int]) -> GpuState:
        """Advance to the next decision or terminal, without resetting.

        The collector must record reward/done before calling reset_finished().
        """
        if self.state is None:
            raise RuntimeError('reset first')
        if not torch.is_tensor(action_index):
            action_index = torch.tensor(action_index, dtype=torch.int32, device=self.device)
        action_index = action_index.to(self.device).reshape(self.n)
        done, _winner = self.outcome()
        acting = self.state.active == self.learner
        action_index = torch.where(acting & ~done, action_index, torch.full_like(action_index, -1))
        self._dispatch_step(action_index)
        self._advance_to_actor(legal_ready=True)
        return self.state

    def _dispatch_step(self, action_index: torch.Tensor) -> None:
        if self.compiled is None:
            step_index(self.state, action_index)
            return
        self.compiled.step_gpu_state(self.state, action_index)

    def _advance_to_actor(self, *, legal_ready: bool = False, max_ops: int = 24) -> None:
        if self.compiled is None:
            advance_to_actor(self.state, self.learner, max_ops=max_ops, legal_ready=legal_ready)
            return
        for _ in range(max_ops):
            done = self.state.phase == 3
            need = ~done & (self.state.active != self.learner)
            if not bool(need.any()):
                return
            act = rule_action(self.state)
            act = torch.where(need, act, torch.full_like(act, -1))
            self.compiled.step_gpu_state(self.state, act)

    def snapshot(self, row: int = 0) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError('reset first')
        return snapshot_mechan(self.state, row)

    def outcome(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state is None:
            raise RuntimeError('reset first')
        done = self.state.phase == 3
        return done, self.state.winner
