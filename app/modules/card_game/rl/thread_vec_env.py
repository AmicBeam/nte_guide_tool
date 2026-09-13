"""In-process parallel VecEnv. Avoids Windows spawn pipes for large observations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


class ThreadedVecEnv(DummyVecEnv):
    """DummyVecEnv that steps and resets worker envs on a thread pool."""

    def __init__(self, env_fns, *, max_workers: int | None = None):
        super().__init__(env_fns)
        workers = max(1, int(max_workers or self.num_envs))
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rl-env')

    def step_wait(self) -> VecEnvStepReturn:
        stepped = list(self._pool.map(self._step_one, range(self.num_envs)))
        reset_ids = []
        for env_idx, obs, reward, terminated, truncated, info in stepped:
            self.buf_rews[env_idx] = reward
            self.buf_dones[env_idx] = terminated or truncated
            info['TimeLimit.truncated'] = truncated and not terminated
            self.buf_infos[env_idx] = info
            if self.buf_dones[env_idx]:
                info['terminal_observation'] = obs
                reset_ids.append(env_idx)
            else:
                self._save_obs(env_idx, obs)
        if reset_ids:
            for env_idx, obs, info in self._pool.map(self._reset_one, reset_ids):
                self.reset_infos[env_idx] = info
                self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        ids = list(range(self.num_envs))
        for env_idx, obs, info in self._pool.map(self._reset_seeded, ids):
            self.reset_infos[env_idx] = info
            self._save_obs(env_idx, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().close()

    def _step_one(self, env_idx: int):
        obs, reward, terminated, truncated, info = self.envs[env_idx].step(self.actions[env_idx])
        return env_idx, obs, reward, terminated, truncated, info

    def _reset_one(self, env_idx: int):
        obs, info = self.envs[env_idx].reset()
        return env_idx, obs, info

    def _reset_seeded(self, env_idx: int):
        maybe_options = {'options': self._options[env_idx]} if self._options[env_idx] else {}
        obs, info = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
        return env_idx, obs, info
