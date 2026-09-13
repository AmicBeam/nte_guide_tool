"""Bounded decision adapter. No collection loop, trainer, or default opponent."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

from .contracts import ACTION_CAPACITY, Backend, Encoder, FrozenPolicy, Json, Outcome
from .rewards import TERMINAL_LOSS, TERMINAL_WIN, shape_reward


@dataclass(frozen=True)
class Decision:
    revision: int
    observation: Json
    actions: tuple[Json, ...]
    state_features: tuple[float, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]


@dataclass(frozen=True)
class Transition:
    decision: Decision
    reward: float
    terminated: bool
    truncated: bool
    info: Json


def _vector(values: Any, size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError('Encoder returned wrong dimension or nonfinite features')
    return result


def _index(index: Any, count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < count:
        raise ValueError('Illegal candidate index')
    return int(index)


class DuelAdapter:
    def __init__(self, backend: Backend, encoder: Encoder, *, learning_side: str = 'a',
                 opponent: FrozenPolicy | None = None, action_capacity: int = ACTION_CAPACITY,
                 max_game_actions: int = 1000, max_opponent_actions: int = 100,
                 strict_isolation: bool = True, allow_concede: bool = True,
                 capture_opening: bool = False) -> None:
        if learning_side not in ('a', 'b'):
            raise ValueError('learning_side must be a/b')
        if action_capacity != ACTION_CAPACITY:
            raise ValueError('Action candidate capacity is frozen at 256')
        for value in (action_capacity, max_game_actions, max_opponent_actions,
                      encoder.observation_dim, encoder.action_dim):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Capacities, limits and dimensions must be positive integers')
        self.backend, self.encoder = backend, encoder
        self.learning_side, self.opponent = learning_side, opponent
        self.action_capacity = action_capacity
        self.max_game_actions, self.max_opponent_actions = max_game_actions, max_opponent_actions
        self.strict_isolation = bool(strict_isolation)
        self.allow_concede = bool(allow_concede)
        self.capture_opening = bool(capture_opening)
        self._state: Json | None = None
        self._opening_state: Json | None = None
        self._revision = 0
        self._count = 0
        self._truncated = False
        self._truncation_reason = ''
        self._cached: Decision | None = None
        self._reward_seq = 0
        self._episode_shaping = 0.0

    def reset(self, *, seed: int = 0, **options: Any) -> Transition:
        self._state = self.backend.new_game(seed=seed, **options)
        self._opening_state = deepcopy(self._state) if self.capture_opening else None
        self._count, self._truncated, self._cached = 0, False, None
        self._truncation_reason = ''
        self._episode_shaping = 0.0
        self._reward_seq = int((self._state or {}).get('event_seq') or 0)
        self._revision += 1  # Never reuse revisions across resets.
        self._advance_opponent()
        return self._transition(shaped=False)

    def _outcome(self) -> Outcome:
        if self._state is None:
            raise RuntimeError('Call reset before requesting a decision')
        return self.backend.outcome(self._state)

    def _visible(self, side: str) -> tuple[Json, tuple[Json, ...]]:
        assert self._state is not None
        observation = self.backend.observe(self._state, side)
        if self.strict_isolation:
            observation = deepcopy(observation)
        actions = tuple((deepcopy(item['action']) if self.strict_isolation else item['action'])
                        for item in observation['legal_actions'])
        if len(actions) > self.action_capacity:
            raise ValueError('Legal candidate capacity exceeded; no candidates were truncated')
        if not actions and not self._outcome().terminated:
            raise ValueError('Nonterminal acting player has no legal actions')
        if any(not isinstance(action, dict) for action in actions):
            raise ValueError('Each action must be a JSON object')
        if any(action == other for i, action in enumerate(actions) for other in actions[:i]):
            raise ValueError('Duplicate legal action')
        return observation, actions

    def _learner_actions(self, actions: tuple[Json, ...]) -> tuple[Json, ...]:
        if self.allow_concede:
            return actions
        playable = tuple(action for action in actions if action.get('type') != 'concede')
        if not playable and actions and not self._outcome().terminated:
            self._truncated = True
            if not self._truncation_reason:
                self._truncation_reason = 'no_non_concede_actions'
        return playable

    def _apply(self, side: str, action: Json) -> None:
        assert self._state is not None
        payload = deepcopy(action) if self.strict_isolation else action
        state = deepcopy(self._state) if self.strict_isolation else self._state
        self._state = self.backend.apply_action(state, side, payload)
        self._count += 1
        self._revision += 1
        self._cached = None

    def _advance_opponent(self) -> None:
        count = 0
        while not self._outcome().terminated:
            if self._count >= self.max_game_actions:
                self._truncated = True
                self._truncation_reason = 'max_game_actions'
                return
            assert self._state is not None
            side = self.backend.current_player(self._state)
            if side == self.learning_side:
                return
            if side not in ('a', 'b'):
                raise ValueError('Nonterminal game has no valid acting side')
            if count >= self.max_opponent_actions:
                self._truncated = True
                self._truncation_reason = 'max_opponent_actions'
                return
            if self.opponent is None:
                raise RuntimeError('An explicit frozen opponent is required when the other seat acts')
            observation, actions = self._visible(side)
            if self.strict_isolation:
                choice = _index(self.opponent(deepcopy(observation), deepcopy(actions)), len(actions))
            else:
                choice = _index(self.opponent(observation, actions), len(actions))
            self._apply(side, actions[choice])
            count += 1

    def decision(self) -> Decision:
        outcome = self._outcome()
        if self._cached is None:
            assert self._state is not None
            if outcome.terminated:
                observation = self.backend.observe(self._state, self.learning_side)
                if self.strict_isolation:
                    observation = deepcopy(observation)
                # Only true terminals have no next action.
                observation['legal_actions'] = []
                actions: tuple[Json, ...] = ()
            else:
                if not self._truncated and self.backend.current_player(self._state) != self.learning_side:
                    raise RuntimeError('Learning side does not own this decision')
                observation, actions = self._visible(self.learning_side)
                actions = self._learner_actions(actions)
            state_features = _vector(self.encoder.encode_observation(observation), self.encoder.observation_dim)
            action_view = {key: value for key, value in observation.items()
                           if key not in ('legal_actions', 'events', 'logs')}
            rows = [_vector(self.encoder.encode_action(action_view, action), self.encoder.action_dim)
                    for action in actions]
            if self.strict_isolation:
                rows += [(0.0,) * self.encoder.action_dim] * (self.action_capacity - len(rows))
            self._cached = Decision(self._revision, observation, actions, state_features,
                                    tuple(rows), tuple(i < len(actions) for i in range(self.action_capacity)))
        return deepcopy(self._cached) if self.strict_isolation else self._cached

    def step(self, index: int, *, revision: int) -> Transition:
        if self._outcome().terminated or self._truncated:
            raise RuntimeError('Episode already ended; reset before stepping')
        decision = self.decision()
        if isinstance(revision, bool) or not isinstance(revision, Integral) or revision != decision.revision:
            raise ValueError('Stale decision revision')
        index = _index(index, len(decision.actions))
        if self.strict_isolation:
            observation, actions = self._visible(self.learning_side)
            actions = self._learner_actions(actions)
            if observation != decision.observation or actions != decision.actions:
                raise ValueError('Decision changed; reset the adapter')
        self._apply(self.learning_side, decision.actions[index])
        self._advance_opponent()
        return self._transition()

    def _transition(self, *, shaped: bool = True) -> Transition:
        outcome = self._outcome()
        terminal = 0.0
        if outcome.terminated and outcome.winner != 'draw':
            terminal = TERMINAL_WIN if outcome.winner == self.learning_side else TERMINAL_LOSS
        parts: dict = {'harmony': 0, 'awaken': 0, 'damage_hp': 0, 'down': 0, 'collapse': 0, 'shaping': 0.0}
        extra = 0.0
        if shaped and self._state is not None:
            extra, parts = shape_reward(self._state.get('events'), self.learning_side, self._reward_seq)
            self._episode_shaping += extra
        if self._state is not None:
            self._reward_seq = int(self._state.get('event_seq') or 0)
        # Terminal step is exactly ±10 / 0. Mid-game shaping stays on earlier steps.
        reward = terminal if outcome.terminated else extra
        info = {
            'winner': outcome.winner, 'reason': outcome.reason,
            'game_actions': self._count, 'revision': self._revision,
            'truncation_reason': self._truncation_reason,
            'shaping': parts,
            'episode_shaping': round(self._episode_shaping, 6),
            'terminal_reward': terminal,
            'bootstrap_allowed': (not outcome.terminated and
                self.backend.current_player(self._state) == self.learning_side),
        }
        if outcome.terminated or self._truncated:
            info['episode_score'] = {
                'terminal': terminal,
                'shaping': round(self._episode_shaping, 6),
                'winner': outcome.winner,
                'truncated': bool(self._truncated and not outcome.terminated),
                'game_actions': self._count,
            }
        return Transition(self.decision(), reward, outcome.terminated,
                          self._truncated and not outcome.terminated, info)
