"""Shared engine-action / game / wall-clock budget for train then eval."""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping


class BudgetStop(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TrainBudget:
    def __init__(
        self, *, max_raw_steps: int, max_games: int, max_seconds: float,
        reserve_raw_steps: int = 0, reserve_games: int = 0, reserve_seconds: float = 0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if min(max_raw_steps, max_games, max_seconds) <= 0:
            raise ValueError('Budget caps must be positive')
        if reserve_raw_steps < 0 or reserve_games < 0 or reserve_seconds < 0:
            raise ValueError('Reserved eval budget cannot be negative')
        if reserve_raw_steps >= max_raw_steps or reserve_games >= max_games or reserve_seconds >= max_seconds:
            raise ValueError('Reserved eval budget must leave room for training')
        self.max_raw_steps = int(max_raw_steps)
        self.max_games = int(max_games)
        self.max_seconds = float(max_seconds)
        self.reserve_raw_steps = int(reserve_raw_steps)
        self.reserve_games = int(reserve_games)
        self.reserve_seconds = float(reserve_seconds)
        self.clock = clock or time.monotonic
        self.phase = 'train'
        self._shared = None
        self.started = self.clock()
        self._raw_steps = 0
        self._games = 0

    def attach_shared(self, shared: Mapping[str, Any] | None) -> None:
        self._shared = shared
        if shared is None:
            return
        self._raw_steps = 0
        self._games = 0

    @property
    def raw_steps(self) -> int:
        if self._shared is None:
            return self._raw_steps
        with self._shared['lock']:
            return int(self._shared['raw'].value)

    @raw_steps.setter
    def raw_steps(self, value: int) -> None:
        if self._shared is None:
            self._raw_steps = int(value)
            return
        with self._shared['lock']:
            self._shared['raw'].value = int(value)

    @property
    def games(self) -> int:
        if self._shared is None:
            return self._games
        with self._shared['lock']:
            return int(self._shared['games'].value)

    @games.setter
    def games(self, value: int) -> None:
        if self._shared is None:
            self._games = int(value)
            return
        with self._shared['lock']:
            self._shared['games'].value = int(value)

    def elapsed(self) -> float:
        start = self._shared['start'].value if self._shared is not None else self.started
        return max(0.0, self.clock() - start)

    def snapshot(self) -> dict[str, Any]:
        return {
            'raw_steps': self.raw_steps, 'games_started': self.games,
            'elapsed_seconds': round(self.elapsed(), 4),
            'max_raw_steps': self.max_raw_steps, 'max_games': self.max_games,
            'max_seconds': self.max_seconds,
            'reserve_raw_steps': self.reserve_raw_steps, 'reserve_games': self.reserve_games,
            'reserve_seconds': self.reserve_seconds, 'phase': self.phase,
        }

    def _limit(self, kind: str) -> tuple[int | float, int | float]:
        reserve = {
            'raw_steps': self.reserve_raw_steps if self.phase == 'train' else 0,
            'games': self.reserve_games if self.phase == 'train' else 0,
            'seconds': self.reserve_seconds if self.phase == 'train' else 0,
        }
        if kind == 'raw_steps':
            return self.raw_steps, self.max_raw_steps - reserve['raw_steps']
        if kind == 'games':
            return self.games, self.max_games - reserve['games']
        return self.elapsed(), self.max_seconds - reserve['seconds']

    def remaining_reason(self) -> str | None:
        used_steps, cap_steps = self._limit('raw_steps')
        if used_steps >= cap_steps:
            return 'max_raw_steps'
        used_time, cap_time = self._limit('seconds')
        if used_time >= cap_time:
            return 'max_seconds'
        return None

    def games_exhausted(self) -> bool:
        used_games, cap_games = self._limit('games')
        return used_games >= cap_games

    def consume_game(self) -> None:
        if self._shared is None:
            reason = self.remaining_reason() or ('max_games' if self.games_exhausted() else None)
            if reason:
                raise BudgetStop(reason)
            self._games += 1
            return
        with self._shared['lock']:
            reserve = self.reserve_games if self.phase == 'train' else 0
            if int(self._shared['games'].value) >= self.max_games - reserve:
                raise BudgetStop('max_games')
            reserve_steps = self.reserve_raw_steps if self.phase == 'train' else 0
            if int(self._shared['raw'].value) >= self.max_raw_steps - reserve_steps:
                raise BudgetStop('max_raw_steps')
            reserve_s = self.reserve_seconds if self.phase == 'train' else 0
            if self.clock() - self._shared['start'].value >= self.max_seconds - reserve_s:
                raise BudgetStop('max_seconds')
            self._shared['games'].value = int(self._shared['games'].value) + 1

    def consume_step(self) -> None:
        if self._shared is None:
            reason = self.remaining_reason()
            if reason:
                raise BudgetStop(reason)
            self._raw_steps += 1
            return
        with self._shared['lock']:
            reserve_steps = self.reserve_raw_steps if self.phase == 'train' else 0
            if int(self._shared['raw'].value) >= self.max_raw_steps - reserve_steps:
                raise BudgetStop('max_raw_steps')
            reserve_s = self.reserve_seconds if self.phase == 'train' else 0
            if self.clock() - self._shared['start'].value >= self.max_seconds - reserve_s:
                raise BudgetStop('max_seconds')
            self._shared['raw'].value = int(self._shared['raw'].value) + 1

    def begin_eval(self) -> None:
        self.phase = 'eval'
        self.reserve_raw_steps = 0
        self.reserve_games = 0
        self.reserve_seconds = 0


_managers: list = []


class _ThreadValue:
    def __init__(self, value: int | float) -> None:
        self.value = value


def make_thread_counters() -> dict[str, Any]:
    import threading
    return {
        'raw': _ThreadValue(0),
        'games': _ThreadValue(0),
        'lock': threading.Lock(),
        'start': _ThreadValue(time.monotonic()),
    }


def make_shared_counters():
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    _managers.append(manager)
    return {
        'raw': manager.Value('q', 0),
        'games': manager.Value('q', 0),
        'lock': manager.Lock(),
        'start': manager.Value('d', time.monotonic()),
    }


class CountingBackend:
    """Counts new_game / apply_action against a shared TrainBudget."""

    def __init__(self, inner, budget: TrainBudget) -> None:
        self.inner = inner
        self.budget = budget

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def new_game(self, *, seed: int = 0, **options):
        self.budget.consume_game()
        return self.inner.new_game(seed=seed, **options)

    def apply_action(self, state, side, action):
        self.budget.consume_step()
        return self.inner.apply_action(state, side, action)
