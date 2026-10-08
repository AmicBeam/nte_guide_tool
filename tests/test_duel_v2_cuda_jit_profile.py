"""Compiler profile validation does not require a GPU or start training."""
import os
import importlib.util
import unittest
from unittest.mock import patch
from app.modules.card_game.rl.rule_ir.compile_engine import _cuda_jit_optimization_level, _cuda_fast_compile_level


class CudaJitProfileTest(unittest.TestCase):
    def test_unset_preserves_driver_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_cuda_jit_optimization_level())

    def test_explicit_levels_and_invalid_values(self):
        for value in ('0', '1', '2', '3', '4'):
            with patch.dict(os.environ, {'NTE_CUDA_JIT_OPT_LEVEL': value}):
                self.assertEqual(_cuda_jit_optimization_level(), int(value))
        for value in ('', '-1', '5', 'nan', '1.0'):
            with patch.dict(os.environ, {'NTE_CUDA_JIT_OPT_LEVEL': value}):
                with self.assertRaisesRegex(ValueError, '0 to 4'):
                    _cuda_jit_optimization_level()

    def test_fast_compile_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_cuda_fast_compile_level(), '0')
        for value in ('0', 'min', 'mid', 'max'):
            with patch.dict(os.environ, {'NTE_CUDA_FAST_COMPILE': value}):
                self.assertEqual(_cuda_fast_compile_level(), value)
        with patch.dict(os.environ, {'NTE_CUDA_FAST_COMPILE': 'bad'}):
            with self.assertRaises(ValueError):
                _cuda_fast_compile_level()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_public_reset_compares_concrete_tensor_devices(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        from app.modules.card_game.rl.gpu_duel.state import empty_state
        from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
        state = empty_state(1, 'cpu:0')  # Like cuda/cuda:0, alias differs from the allocated tensor's device.
        self.assertNotEqual(state.device, state.phase.device)
        backend = CompiledStarterBackend.__new__(CompiledStarterBackend)
        backend.closed = False
        backend.backend = 'native'
        backend.launch = SimpleNamespace(public_reset=Mock())
        backend.reset_public_gpu_state(state, torch.zeros(1, dtype=torch.int32),
                                       torch.zeros(1, 36, dtype=torch.int32), torch.zeros(1, 36, dtype=torch.int32))
        backend.launch.public_reset.assert_called_once()
