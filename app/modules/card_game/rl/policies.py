"""Frozen policies that only see a public observation and legal actions."""
from __future__ import annotations

from typing import Sequence

from .contracts import FrozenPolicy, Json


ACTION_PRIORITY = {
    'choose': 0,
    'mulligan': 1,
    'play_card': 2,
    'awaken': 3,
    'attack': 4,
    'cycle': 5,
    'end_turn': 6,
    'concede': 7,
}


def _require_actions(legal_actions: Sequence[Json]) -> tuple[Json, ...]:
    actions = tuple(legal_actions)
    if not actions:
        raise ValueError('Frozen policy cannot choose from an empty action mask')
    if any(not isinstance(action, dict) or 'type' not in action for action in actions):
        raise ValueError('Frozen policy received a malformed legal action')
    return actions


class FirstLegalPolicy:
    """Deterministic stand-in: first legal candidate, never inspects hidden state."""

    def __call__(self, observation: Json, legal_actions: tuple[Json, ...]) -> int:
        if not isinstance(observation, dict):
            raise ValueError('Frozen policy only accepts an observation object')
        _require_actions(legal_actions)
        return 0


class RuleFrozenPolicy:
    """Tiny visible-information heuristic. No engine state and no hidden cards."""

    def __call__(self, observation: Json, legal_actions: tuple[Json, ...]) -> int:
        if not isinstance(observation, dict):
            raise ValueError('Frozen policy only accepts an observation object')
        actions = _require_actions(legal_actions)
        ranked = sorted(
            enumerate(actions),
            key=lambda item: (ACTION_PRIORITY.get(str(item[1].get('type')), 99), item[0]),
        )
        return ranked[0][0]


def as_frozen_policy(policy: FrozenPolicy) -> FrozenPolicy:
    if not callable(policy):
        raise TypeError('policy must be callable')
    return policy


class MaskablePolicy:
    """Frozen predict-only model interface; never creates or trains a model.

    Caller validates checkpoint metadata before supplying a compatible model.
    """

    def __init__(self, model, encoder, *, action_capacity=256):
        self.model, self.encoder, self.action_capacity = model, encoder, action_capacity

    def __call__(self, observation, legal_actions):
        import numpy as np
        actions = _require_actions(legal_actions)
        if len(actions) > self.action_capacity:
            raise ValueError('Legal candidate capacity exceeded')
        state = self.encoder.encode_observation(observation)
        candidates = [self.encoder.encode_action(observation, action) for action in actions]
        candidates += [[0.0] * self.encoder.action_dim] * (self.action_capacity - len(actions))
        mask = np.arange(self.action_capacity) < len(actions)
        action, _ = self.model.predict({'state': np.asarray(state, dtype=np.float32),
            'candidates': np.asarray(candidates, dtype=np.float32)},
            action_masks=mask, deterministic=True)
        from .adapter import _index
        return _index(np.asarray(action).item(), len(actions))
