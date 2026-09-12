"""Optional genuine Gymnasium Env; dependencies load only on construction."""
from .adapter import DuelAdapter


def make_gym_env(adapter: DuelAdapter):
    try:
        import numpy as np
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError('Optional dependencies are listed in requirements-rl.txt') from exc

    class _DuelV2Env(gym.Env):
        metadata = {'render_modes': []}

        def __init__(self):
            self.adapter = adapter
            self._last = None
            self.observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(0, 1, shape=(adapter.encoder.observation_dim,), dtype=np.float32),
                'candidates': gym.spaces.Box(0, 1, shape=(adapter.action_capacity, adapter.encoder.action_dim), dtype=np.float32),
            })
            self.action_space = gym.spaces.Discrete(adapter.action_capacity)

        def _observation(self, transition):
            decision = transition.decision
            return {'state': np.asarray(decision.state_features, dtype=np.float32),
                    'candidates': np.asarray(decision.candidate_features, dtype=np.float32)}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._last = None
            transition = adapter.reset(seed=0 if seed is None else seed, **(options or {}))
            if transition.terminated or transition.truncated or not any(transition.decision.mask):
                raise RuntimeError('Reset must reach a live learner decision; change setup/opponent/limits')
            self._last = transition
            return self._observation(transition), dict(transition.info)

        def step(self, action):
            if self._last is None:
                raise RuntimeError('Call reset before step')
            transition = adapter.step(action, revision=self._last.decision.revision)
            self._last = transition
            if transition.truncated and not transition.info['bootstrap_allowed']:
                self._last = None
                raise RuntimeError('Opponent-turn truncation cannot be bootstrapped; raise limits and reset')
            return (self._observation(transition), transition.reward, transition.terminated,
                    transition.truncated, dict(transition.info))

        def action_masks(self):
            if self._last is None:
                raise RuntimeError('Call reset before requesting action masks')
            return np.asarray(self._last.decision.mask, dtype=np.bool_)

    return _DuelV2Env()


class DuelV2GymnasiumEnv:
    """Lazy constructor returning a gymnasium.Env instance."""

    def __new__(cls, adapter):
        return make_gym_env(adapter)
