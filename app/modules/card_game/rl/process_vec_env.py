"""Multiprocess VecEnv using shared memory for large observations.

Windows named pipes deadlock on the candidate matrix, so workers write
state/candidates/masks into shared arrays and only send small control
payloads over the pipe. Each worker is a process, so CPU game steps
bypass the parent GIL.
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _meta(block: shared_memory.SharedMemory, array: np.ndarray) -> dict[str, Any]:
    return {'name': block.name, 'shape': tuple(array.shape), 'dtype': str(array.dtype)}


def _view(meta: dict[str, Any]) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    block = shared_memory.SharedMemory(name=str(meta['name']))
    array = np.ndarray(tuple(meta['shape']), dtype=np.dtype(meta['dtype']), buffer=block.buf)
    return block, array


def _copy_obs(index: int, observation: dict[str, np.ndarray], buffers: dict[str, np.ndarray]) -> None:
    buffers['state'][index] = np.asarray(observation['state'], dtype=np.float32)
    buffers['candidates'][index] = np.asarray(observation['candidates'], dtype=np.float32)


def _read_obs(index: int, buffers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        'state': np.array(buffers['state'][index], copy=True),
        'candidates': np.array(buffers['candidates'][index], copy=True),
    }


def _small_info(info: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in info.items():
        if key == 'terminal_observation':
            continue
        if isinstance(value, np.ndarray):
            continue
        cleaned[key] = value
    return cleaned


def _worker(index: int, remote, parent_remote, env_fn_wrapper, shm_meta: dict[str, Any]) -> None:
    parent_remote.close()
    handles = []
    buffers = {}
    try:
        for key, meta in shm_meta.items():
            handle, array = _view(meta)
            handles.append(handle)
            buffers[key] = array
        env = _patch_env(env_fn_wrapper.var())
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == 'reset':
                observation, info = env.reset(seed=data[0], **({'options': data[1]} if data[1] else {}))
                _copy_obs(index, observation, buffers)
                remote.send(_small_info(info))
            elif cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                info = dict(info)
                info['TimeLimit.truncated'] = bool(truncated and not terminated)
                if done:
                    _copy_obs(index, observation, {'state': buffers['term_state'], 'candidates': buffers['term_candidates']})
                    observation, reset_info = env.reset()
                    info = {**_small_info(info), 'reset_info': _small_info(reset_info), 'has_terminal': True}
                else:
                    info = _small_info(info)
                _copy_obs(index, observation, buffers)
                buffers['rewards'][index] = float(reward)
                buffers['dones'][index] = 1 if done else 0
                remote.send(info)
            elif cmd == 'env_method':
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(env.get_wrapper_attr(data))
            elif cmd == 'has_attr':
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == 'set_attr':
                setattr(env, data[0], data[1])
                remote.send(True)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(cmd)
    finally:
        for handle in handles:
            handle.close()


class ProcessVecEnv(VecEnv):
    """One game engine per process; observations live in shared memory."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        if len(env_fns) < 1:
            raise ValueError('ProcessVecEnv needs at least one env')
        ctx = mp.get_context('spawn')
        probe = _patch_env(env_fns[0]())
        observation_space = probe.observation_space
        action_space = probe.action_space
        probe.close()
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError('ProcessVecEnv expects a Dict observation space')
        n_envs = len(env_fns)
        state_shape = tuple(observation_space['state'].shape)
        cand_shape = tuple(observation_space['candidates'].shape)
        self._prefix = f'duelv2_{mp.current_process().pid}_{uuid4().hex}'
        self._blocks: list[shared_memory.SharedMemory] = []
        self._buffers: dict[str, np.ndarray] = {}
        specs = {
            'state': ((n_envs, *state_shape), np.float32),
            'candidates': ((n_envs, *cand_shape), np.float32),
            'term_state': ((n_envs, *state_shape), np.float32),
            'term_candidates': ((n_envs, *cand_shape), np.float32),
            'rewards': ((n_envs,), np.float32),
            'dones': ((n_envs,), np.uint8),
        }
        shm_meta = {}
        for key, (shape, dtype) in specs.items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            block = shared_memory.SharedMemory(create=True, name=f'{self._prefix}_{key}', size=nbytes)
            array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
            array[:] = 0
            self._blocks.append(block)
            self._buffers[key] = array
            shm_meta[key] = _meta(block, array)
        self.remotes = []
        self.processes = []
        for index, env_fn in enumerate(env_fns):
            remote, work_remote = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(index, work_remote, remote, CloudpickleWrapper(env_fn), shm_meta),
                daemon=True,
            )
            process.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(process)
        super().__init__(n_envs, observation_space, action_space)
        self.waiting = False
        self.closed = False

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        infos = [remote.recv() for remote in self.remotes]
        self.waiting = False
        for index, info in enumerate(infos):
            info = dict(info)
            if info.pop('has_terminal', False):
                info['terminal_observation'] = _read_obs(index, {
                    'state': self._buffers['term_state'],
                    'candidates': self._buffers['term_candidates'],
                })
                self.reset_infos[index] = dict(info.pop('reset_info', {}) or {})
            infos[index] = info
        obs = {
            'state': np.array(self._buffers['state'], copy=True),
            'candidates': np.array(self._buffers['candidates'], copy=True),
        }
        return obs, np.array(self._buffers['rewards'], copy=True), np.array(self._buffers['dones'], dtype=bool), infos

    def reset(self) -> VecEnvObs:
        for index, remote in enumerate(self.remotes):
            remote.send(('reset', (self._seeds[index], self._options[index])))
        self.reset_infos = [remote.recv() for remote in self.remotes]
        self._reset_seeds()
        self._reset_options()
        return {
            'state': np.array(self._buffers['state'], copy=True),
            'candidates': np.array(self._buffers['candidates'], copy=True),
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except (EOFError, OSError, BrokenPipeError):
                    pass
            self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (EOFError, OSError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for block in self._blocks:
            block.close()
            try:
                block.unlink()
            except FileNotFoundError:
                pass
        self.closed = True

    def has_attr(self, attr_name: str, indices=None) -> bool:
        remotes = self._target(indices)
        for remote in remotes:
            remote.send(('has_attr', attr_name))
        return all(remote.recv() for remote in remotes)

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        remotes = self._target(indices)
        for remote in remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in remotes]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        remotes = self._target(indices)
        for remote in remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        remotes = self._target(indices)
        for remote in remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in remotes]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._target(indices)]

    def _target(self, indices):
        indices = self._get_indices(indices)
        return [self.remotes[index] for index in indices]
