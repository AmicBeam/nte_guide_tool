"""Exact public-byte hash tests; 1600 rows are not complete game evidence."""
import hashlib
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_search.device_info_key import DeviceInformationKey


class DeviceInformationKeyTest(unittest.TestCase):
    def inputs(self, device='cpu', rows=4, d=13, m=5, width=3):
        import torch
        rng = np.random.default_rng(119)
        x = rng.normal(size=(rows, d)).astype(np.float32)
        c = rng.normal(size=(rows, m, width)).astype(np.float32)
        counts = 1 + np.arange(rows, dtype=np.int64) % m
        active = np.ones(rows, dtype=np.bool_)
        return tuple(torch.as_tensor(v, device=device) for v in (x, c, counts, active))

    def expected(self, args):
        x, c, counts, active = (v.cpu().numpy() for v in args)
        return np.asarray([np.frombuffer(hashlib.sha256(a.tobytes()+b[:n].tobytes()).digest(),
                                         dtype='>u4').astype(np.int64)
                           if on else np.zeros(8, dtype=np.int64)
                           for a, b, n, on in zip(x, c, counts, active)])

    def test_cpu_reference_matches_existing_exact_byte_key_and_ignores_padding(self):
        engine = DeviceInformationKey(device='cpu', allow_cpu_reference=True)
        args = self.inputs()
        result = engine.hash(*args)
        np.testing.assert_array_equal(result.keys.numpy(), self.expected(args))
        self.assertEqual(result.errors.tolist(), [0]*4)
        before = result.keys.clone()
        args[1][0, 1:] = float('nan')
        np.testing.assert_array_equal(engine.hash(*args).keys.numpy(), before.numpy())

    def test_signed_zero_is_preserved_and_different_legal_prefixes_differ(self):
        import torch
        engine = DeviceInformationKey(device='cpu', allow_cpu_reference=True)
        x = torch.zeros((2, 1), dtype=torch.float32)
        x[1, 0] = -0.0
        c = torch.zeros((2, 2, 1), dtype=torch.float32)
        counts = torch.ones(2, dtype=torch.int64)
        active = torch.ones(2, dtype=torch.bool)
        result = engine.hash(x, c, counts, active)
        self.assertFalse(torch.equal(result.keys[0], result.keys[1]))
        x[1, 0] = 0.0
        counts[1] = 2
        result = engine.hash(x, c, counts, active)
        self.assertFalse(torch.equal(result.keys[0], result.keys[1]))

    def test_invalid_count_nonfinite_and_inactive_rows(self):
        engine = DeviceInformationKey(device='cpu', allow_cpu_reference=True)
        args = self.inputs()
        args[2][0] = 0
        args[2][1] = 6
        args[0][2, 0] = float('inf')
        args[3][3] = False
        args[0][3] = float('nan')
        result = engine.hash(*args)
        self.assertEqual(result.errors.tolist(), [1, 1, 2, 0])
        self.assertTrue(bool((result.keys == 0).all()))

    def test_reference_must_be_explicit_and_static_memory_limits_reject(self):
        with self.assertRaises(RuntimeError):
            DeviceInformationKey(device='cpu')
        engine = DeviceInformationKey(device='cpu', allow_cpu_reference=True, max_input_bytes=8)
        with self.assertRaises(ValueError):
            engine.hash(*self.inputs())

    def test_actual_cuda_hash_padding_nonfinite_and_1600_graph_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; numerical hash only')
        engine = DeviceInformationKey(device='cuda')
        self.addCleanup(engine.close)
        args = self.inputs('cuda', rows=1600, d=17, m=8, width=7)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            actual = engine.hash(*args)
        np.testing.assert_array_equal(actual.keys.cpu().numpy(), self.expected(args))
        self.assertFalse(bool((actual.errors != 0).any()))
        # Warm both streams before capture; only entire graph launch/replay is on host.
        for _ in range(3):
            engine.hash(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = engine.hash(*args)
        args[0].add_(.25)
        graph.replay()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(captured.keys.cpu().numpy(), self.expected(args))
        args[2][0] = 0
        args[0][1, 0] = float('inf')
        args[3][2] = False
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(captured.errors[:3].cpu().tolist(), [1, 2, 0])
        self.assertTrue(bool((captured.keys[:3] == 0).all()))


if __name__ == '__main__':
    unittest.main()
