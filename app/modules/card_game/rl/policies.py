"""Frozen policies that only see a public observation and legal actions."""
from __future__ import annotations

from random import Random
from typing import Sequence

from .contracts import FrozenPolicy, Json


ACTION_PRIORITY = {
    'choose': 0,
    'mulligan': 1,
    'play_card': 2,
    'ultimate': 3,
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


class RandomMaskedPolicy:
    """Uniform among currently legal candidates. Untrained baseline, not a network."""

    def __init__(self, rng: Random | None = None) -> None:
        self.rng = rng or Random(0)

    def __call__(self, observation: Json, legal_actions: tuple[Json, ...]) -> int:
        if not isinstance(observation, dict):
            raise ValueError('Frozen policy only accepts an observation object')
        actions = tuple(action for action in _require_actions(legal_actions) if action.get('type') != 'concede') or _require_actions(legal_actions)
        return self.rng.randrange(len(actions))


class VisibleEngineRulePolicy:
    """Same public ranking as engine choose_action; never reads hidden state."""

    def __call__(self, observation: Json, legal_actions: tuple[Json, ...]) -> int:
        if not isinstance(observation, dict):
            raise ValueError('Frozen policy only accepts an observation object')
        actions = _require_actions(legal_actions)
        for kind in ('mulligan', 'choose'):
            for index, action in enumerate(actions):
                if action.get('type') == kind:
                    return index
        scores = self.ranking_scores(observation, actions)
        return max(range(len(actions)), key=lambda index: (scores[index], -index))

    def ranking_scores(self, observation: Json, legal_actions: tuple[Json, ...]) -> list[float]:
        """Expose the existing public ranking for offline initialization only.

        This is neither an environment reward nor a learned model score. Equal
        ranks may receive an equal imitation target instead of encoding the
        incidental order of physical cards.
        """
        actions = _require_actions(legal_actions)
        by_key = {_action_key(entry['action']):entry for entry in observation.get('legal_actions') or []
                  if isinstance(entry,dict) and isinstance(entry.get('action'),dict)}
        viewer = observation.get('viewer_side')
        cards = {}
        for card in ((observation.get('sides') or {}).get(viewer) or {}).get('hand') or []:
            if isinstance(card, dict) and card.get('instance_id'):
                cards[card['instance_id']] = card

        def score(index: int) -> float:
            action = actions[index]
            entry = by_key.get(_action_key(action), {})
            preview = entry.get('preview') if isinstance(entry, dict) else {}
            preview = preview if isinstance(preview, dict) else {}
            kind = action.get('type')
            if kind == 'concede':
                return -1000.0
            if kind == 'end_turn':
                return -100.0
            if kind == 'ultimate':
                return 6.0
            if kind == 'cycle':
                return 1.0
            card = cards.get(action.get('card_id')) if kind == 'play_card' else None
            if kind == 'attack' or (kind == 'play_card' and isinstance(card, dict) and card.get('type') == 'battle'):
                return (float(preview.get('attack') or 0) * float(preview.get('攻击次数') or 1)
                        + float(preview.get('创生花伤害') or 0)
                        - float(preview.get('counter') or 0) * 0.4 + 3.0)
            return 2.0

        return [score(index) for index in range(len(actions))]


def _action_key(action: Json) -> str:
    try:
        import json
        return json.dumps(action, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(action)


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
        actions = tuple(action for action in _require_actions(legal_actions) if action.get('type') != 'concede')
        if not actions:
            raise ValueError('Model policy has no non-concede actions')
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
