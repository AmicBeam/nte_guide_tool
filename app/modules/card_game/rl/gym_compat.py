"""Optional genuine Gymnasium Env; imported only from the training loop."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .adapter import DuelAdapter
from .budget import BudgetStop


class DuelV2Env(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, adapter: DuelAdapter):
        super().__init__()
        self.adapter = adapter
        self._last = None
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(0, 1, shape=(adapter.encoder.observation_dim,), dtype=np.float32),
            'candidates': gym.spaces.Box(
                0, 1, shape=(adapter.action_capacity, adapter.encoder.action_dim), dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Discrete(adapter.action_capacity)

    def _observation(self, transition):
        decision = transition.decision
        candidates = np.zeros((self.adapter.action_capacity, self.adapter.encoder.action_dim), dtype=np.float32)
        rows = decision.candidate_features
        if rows:
            filled = np.asarray(rows, dtype=np.float32)
            candidates[:filled.shape[0]] = filled
        return {
            'state': np.asarray(decision.state_features, dtype=np.float32),
            'candidates': candidates,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last = None
        options = dict(options or {})
        try:
            transition = self.adapter.reset(seed=0 if seed is None else seed, **options)
        except BudgetStop as exc:
            raise RuntimeError(f'Training budget exhausted before reset: {exc.reason}') from exc
        if transition.terminated or not any(transition.decision.mask):
            raise RuntimeError('Reset must reach a live learner decision; change setup/opponent/limits')
        if transition.truncated:
            raise RuntimeError('Reset must reach a live learner decision; change setup/opponent/limits')
        self._last = transition
        return self._observation(transition), dict(transition.info)

    def step(self, action):
        if self._last is None:
            raise RuntimeError('Call reset before step')
        try:
            transition = self.adapter.step(action, revision=self._last.decision.revision)
        except BudgetStop as exc:
            info = dict(self._last.info if self._last else {})
            info.update(truncation_reason=str(exc.reason), bootstrap_allowed=False,
                        budget_stop=True, winner=None)
            observation = self._observation(self._last)
            self._last = None
            return observation, 0.0, False, True, info
        self._last = transition
        info = dict(transition.info)
        if transition.truncated and not transition.terminated:
            info['bootstrap_allowed'] = bool(info.get('bootstrap_allowed'))
        return (self._observation(transition), transition.reward, transition.terminated,
                transition.truncated, info)

    def action_masks(self):
        if self._last is None:
            raise RuntimeError('Call reset before requesting action masks')
        mask = np.asarray(self._last.decision.mask, dtype=np.bool_)
        actions = self._last.decision.actions
        concede = [index for index, action in enumerate(actions) if action.get('type') == 'concede']
        if concede:
            mask = mask.copy()
            for index in concede:
                mask[index] = False
        return mask


def make_gym_env(adapter: DuelAdapter):
    return DuelV2Env(adapter)


class DuelV2GymnasiumEnv:
    """Lazy constructor returning a gymnasium.Env instance."""

    def __new__(cls, adapter):
        return make_gym_env(adapter)
