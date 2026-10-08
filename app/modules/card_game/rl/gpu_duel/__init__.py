"""GPU duel for 创生预组 and 覆纹快攻. Website Python engine remains the oracle."""

from .catalog import GPU_LOCK, STARTER_LOCK, assert_gpu_deck, assert_starter_deck

__all__ = [
    'GpuDuelEnv', 'GPU_LOCK', 'STARTER_LOCK', 'assert_gpu_deck', 'assert_starter_deck',
    'pack_python', 'python_mechan', 'snapshot_mechan',
]


def __getattr__(name):
    if name == 'GpuDuelEnv':
        from .env import GpuDuelEnv
        return GpuDuelEnv
    if name in ('pack_python', 'python_mechan', 'snapshot_mechan'):
        from . import pack
        return getattr(pack, name)
    raise AttributeError(name)
