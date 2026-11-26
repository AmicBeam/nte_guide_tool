"""PCG64 RNG primitive tests; this is not game, 1600-root, or training evidence."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from numpy.random import Generator, PCG64, PCG64DXSM

from app.modules.card_game.rl.batched_search import device_random
from app.modules.card_game.rl.batched_search.device_random import (
    DevicePCG64, prepare_numpy_pcg64_states, verify_numpy_pcg64_state,
)


SEEDS = [0, 1, (1 << 300) + 7, 42]


def _states(seeds=SEEDS):
    return prepare_numpy_pcg64_states(seeds)


class DevicePCG64OracleTest(unittest.TestCase):
    def bank(self, states=None, **kwargs):
        opts = dict(device='cpu', allow_cpu_reference=True, max_draws=8, max_choices=8)
        opts.update(kwargs)
        return DevicePCG64(_states() if states is None else states, **opts)

    def test_module_import_does_not_import_torch_or_cuda(self):
        tree = ast.parse(inspect.getsource(device_random))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split('.')[0]]
            self.assertNotIn('torch', names)
            self.assertNotIn('cuda', names)
        self.assertFalse(device_random.DevicePCG64.supports_full_duel_rules)
        self.assertEqual(device_random.SOURCE_VERSION, 'duel_v2_device_pcg64_v1')

    def test_prepare_wide_seed_uses_numpy_pcg64_not_truncation(self):
        wide = (1 << 512) + 99
        prepared = prepare_numpy_pcg64_states([wide])[0]
        numpy_state = PCG64(wide).state
        self.assertEqual(prepared['state']['state'], int(numpy_state['state']['state']))
        self.assertEqual(prepared['state']['inc'], int(numpy_state['state']['inc']))
        self.assertEqual(prepared, prepare_numpy_pcg64_states([numpy_state])[0])
        truncated = prepare_numpy_pcg64_states([wide & ((1 << 256) - 1)])[0]
        self.assertNotEqual(prepared['state']['state'], truncated['state']['state'])
        self.assertNotEqual(prepared['state']['inc'], truncated['state']['inc'])

    def test_verified_state_rejects_dxsm_bool_and_incomplete_limbs(self):
        with self.assertRaises(ValueError):
            verify_numpy_pcg64_state(PCG64DXSM(0).state)
        with self.assertRaises(TypeError):
            prepare_numpy_pcg64_states([True])
        with self.assertRaises(TypeError):
            prepare_numpy_pcg64_states([False])
        bad = PCG64(0).state
        bad = dict(bad); bad['state'] = dict(bad['state']); bad['has_uint32'] = True
        with self.assertRaises(TypeError):
            verify_numpy_pcg64_state(bad)
        with self.assertRaises(ValueError):
            DevicePCG64(_states(), device='cpu', allow_cpu_reference=True, max_rows=1)

    def test_cpu_requires_explicit_oracle_and_cuda_has_no_fallback(self):
        with self.assertRaises(RuntimeError):
            DevicePCG64(_states([0]), device='cpu')
        import torch
        if torch.cuda.is_available():
            bank = DevicePCG64(_states([0]), device='cuda', max_draws=2, max_choices=2)
            self.addCleanup(bank.close)
            self.assertEqual(bank.device.type, 'cuda')
        else:
            with self.assertRaises(RuntimeError):
                DevicePCG64(_states([0]), device='cuda')

    def test_cpu_oracle_raw_uniform_and_final_state_exact(self):
        import torch
        bank = self.bank()
        active = torch.tensor([True, True, False, True])
        before = bank.export_numpy_states()
        raw = bank.raw64(active, draws=4)
        self.assertEqual(raw.errors.tolist(), [0, 0, 0, 0])
        exported = bank.export_numpy_states()
        self.assertEqual(exported[2], before[2])
        np.testing.assert_array_equal(raw.values[2].numpy(), np.zeros(4, dtype=np.uint64))
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            bg = PCG64(seed)
            np.testing.assert_array_equal(raw.values[i].numpy(), bg.random_raw(4))
            self.assertEqual(exported[i], bg.state)
        bank.load_states(before)
        uni = bank.uniform(active, draws=3)
        exported = bank.export_numpy_states()
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            gen = Generator(PCG64(seed))
            np.testing.assert_array_equal(uni.values[i].numpy(), gen.random(3))
            self.assertEqual(exported[i], gen.bit_generator.state)

    def test_cpu_oracle_gumbel_categorical_selection_and_uint32_cache(self):
        import torch
        bank = self.bank()
        active = torch.tensor([True, True, False, True])
        gumbel = bank.gumbel(active, draws=5, loc=0.5, scale=2.0)
        exported = bank.export_numpy_states()
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            gen = Generator(PCG64(seed))
            expected = gen.gumbel(0.5, 2.0, size=5)
            np.testing.assert_array_equal(gumbel.values[i].numpy(), expected)
            self.assertEqual(exported[i], gen.bit_generator.state)
            self.assertEqual(int(np.argmax(gumbel.values[i].numpy())), int(np.argmax(expected)))
        bank.load_states(_states())
        probs = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25],
             [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
        counts = torch.tensor([4, 4, 2, 3], dtype=torch.int64)
        choice = bank.categorical(probs, counts, active)
        exported = bank.export_numpy_states()
        self.assertEqual(int(choice.indices[2]), 0)
        self.assertEqual(int(choice.errors[2]), 0)
        for i, seed in enumerate(SEEDS):
            if i == 2:
                self.assertEqual(exported[i], _states()[i])
                continue
            gen = Generator(PCG64(seed))
            n = int(counts[i])
            expected = int(gen.choice(n, p=probs[i, :n].numpy()))
            self.assertEqual(int(choice.indices[i]), expected)
            self.assertEqual(exported[i], gen.bit_generator.state)
        cached = PCG64(0).state
        cached = dict(cached)
        cached['state'] = dict(cached['state'])
        cached['has_uint32'] = 1
        cached['uinteger'] = 0x12345678
        bank = self.bank([cached], max_draws=2, max_choices=2)
        bank.raw64(torch.tensor([True]), draws=1)
        out = bank.export_numpy_states()[0]
        self.assertEqual(out['has_uint32'], 1)
        self.assertEqual(out['uinteger'], 0x12345678)

    def test_invalid_probabilities_do_not_advance_or_bypass(self):
        import torch
        bank = self.bank()
        before = bank.export_numpy_states()
        probs = torch.tensor(
            [[-0.1, 1.1, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
             [np.nan, 1.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
        counts = torch.tensor([2, 0, 2, 4], dtype=torch.int64)
        result = bank.categorical(probs, counts, torch.tensor([True, True, True, False]))
        self.assertEqual(result.errors.tolist(), [1, 2, 1, 0])
        self.assertEqual(result.indices.tolist(), [-1, -1, -1, 0])
        self.assertEqual(bank.export_numpy_states(), before)
        with self.assertRaises(ValueError):
            bank.gumbel(torch.ones(4, dtype=torch.bool), scale=-1.0)
        self.assertEqual(bank.export_numpy_states(), before)


class DevicePCG64CudaTest(unittest.TestCase):
    def test_actual_cuda_legal_prefix_noise_preserves_padded_rng_state(self):
        import torch
        valid=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1],[0,0,0,0,0,0],[1,0,0,0,0,0]],dtype=torch.bool,device='cuda')
        active=torch.tensor([True,True,False,True],device='cuda')
        result=self.bank.gumbel_device(valid,active)
        exported=self.bank.export_numpy_states()
        for row,count in enumerate([3,6,0,1]):
            generator=Generator(PCG64(SEEDS[row]))
            expected=generator.gumbel(size=count)
            np.testing.assert_allclose(result.values[row,:count].cpu().numpy(),expected,rtol=1e-12,atol=1e-12)
            self.assertEqual(exported[row],generator.bit_generator.state)
            self.assertFalse(bool(result.values[row,count:].any()))

    def test_actual_cuda_gumbel_rejects_zero_uniform_like_numpy(self):
        states=list(_states())
        multiplier=(2549297995355413924<<64)|4865540595714422341
        modulus=1<<128
        forced=dict(states[0]);forced['state']=dict(forced['state'])
        forced['state']['state']=(-forced['state']['inc']*pow(multiplier,-1,modulus))%modulus
        states[0]=forced
        self.bank.load_states(states)
        active=self.torch.tensor([True,False,False,False],device='cuda')
        generator=Generator(PCG64());generator.bit_generator.state=forced
        expected=generator.gumbel(size=1)
        result=self.bank.gumbel(active,draws=1)
        np.testing.assert_allclose(result.values[0].cpu().numpy(),expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(self.bank.export_numpy_states()[0],generator.bit_generator.state)

    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; no CPU random fallback')
        self.torch = torch
        self.bank = DevicePCG64(
            _states(), device='cuda', max_rows=4, max_draws=8, max_choices=8,
        )
        self.addCleanup(self.bank.close)

    def _hot_path(self):
        from contextlib import ExitStack
        t = self.torch
        stack = ExitStack()
        blockers = (
            ('cpu', 'CPU copy'), ('numpy', 'NumPy'), ('item', 'scalar'),
            ('tolist', 'list'), ('__bool__', 'truth'),
        )
        for name, label in blockers:
            stack.enter_context(patch.object(
                t.Tensor, name, side_effect=AssertionError('Hot-path ' + label)))
        return stack

    def test_actual_cuda_raw_uniform_state_exact_without_host_materialize(self):
        t = self.torch
        active = t.tensor([True, True, False, True], device='cuda')
        before = self.bank.export_numpy_states()
        with self._hot_path():
            raw = self.bank.raw64(active, draws=4)
        exported = self.bank.export_numpy_states()
        self.assertEqual(raw.errors.cpu().tolist(), [0, 0, 0, 0])
        self.assertEqual(exported[2], before[2])
        np.testing.assert_array_equal(raw.values[2].cpu().numpy(), np.zeros(4, dtype=np.uint64))
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            bg = PCG64(seed)
            np.testing.assert_array_equal(raw.values[i].cpu().numpy(), bg.random_raw(4))
            self.assertEqual(exported[i], bg.state)
        self.bank.load_states(before)
        with self._hot_path():
            uni = self.bank.uniform(active, draws=3)
        exported = self.bank.export_numpy_states()
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            gen = Generator(PCG64(seed))
            np.testing.assert_array_equal(uni.values[i].cpu().numpy(), gen.random(3))
            self.assertEqual(exported[i], gen.bit_generator.state)

    def test_actual_cuda_gumbel_categorical_masks_and_graph_replay(self):
        t = self.torch
        active = t.tensor([True, True, False, True], device='cuda')
        self.bank.load_states(_states())
        with self._hot_path():
            gumbel = self.bank.gumbel(active, draws=5, loc=0.5, scale=2.0)
        exported = self.bank.export_numpy_states()
        for i, seed in enumerate(SEEDS):
            if i == 2:
                continue
            gen = Generator(PCG64(seed))
            expected = gen.gumbel(0.5, 2.0, size=5)
            got = gumbel.values[i].cpu().numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(int(np.argmax(got)), int(np.argmax(expected)))
            self.assertEqual(exported[i], gen.bit_generator.state)
        self.bank.load_states(_states())
        probs = t.tensor([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=t.float64, device='cuda')
        counts = t.tensor([4, 4, 2, 3], dtype=t.int64, device='cuda')
        with self._hot_path():
            choice = self.bank.categorical(probs, counts, active)
        exported = self.bank.export_numpy_states()
        self.assertEqual(choice.errors.cpu().tolist()[2], 0)
        host_counts = counts.cpu().tolist()
        host_probs = probs.cpu().numpy()
        host_idx = choice.indices.cpu().tolist()
        for i, seed in enumerate(SEEDS):
            if i == 2:
                self.assertEqual(exported[i], _states()[i])
                continue
            gen = Generator(PCG64(seed))
            n = host_counts[i]
            expected = int(gen.choice(n, p=host_probs[i, :n]))
            self.assertEqual(host_idx[i], expected)
            self.assertEqual(exported[i], gen.bit_generator.state)
        alt = _states([7, 8, 9, 10])
        self.bank.load_states(_states())
        for _ in range(3):
            self.bank.raw64(active, draws=2)
        t.cuda.synchronize()
        self.bank.load_states(_states())
        graph = t.cuda.CUDAGraph()
        with t.cuda.graph(graph):
            captured = self.bank.raw64(active, draws=2)
        self.bank.load_states(alt)
        graph.replay()
        t.cuda.synchronize()
        got = captured.values.clone()
        replayed = self.bank.export_numpy_states()
        self.bank.load_states(alt)
        second = self.bank.raw64(active, draws=2)
        np.testing.assert_array_equal(got.cpu().numpy(), second.values.cpu().numpy())
        self.assertEqual(replayed, self.bank.export_numpy_states())


class DevicePCG64ResetMixin:
    def state_vectors(self, seeds):
        t = self.bank.t
        packed = prepare_numpy_pcg64_states(seeds)
        columns = [[] for _ in range(6)]
        for entry in packed:
            state, inc = entry['state']['state'], entry['state']['inc']
            values = [state >> 64, state & device_random.MASK64,
                      inc >> 64, inc & device_random.MASK64,
                      entry['has_uint32'], entry['uinteger']]
            for column, value in zip(columns, values):
                column.append(value)
        return device_random.DevicePCG64State(*[
            t.tensor(values, dtype=dtype, device=self.bank.device)
            for values, dtype in zip(columns, [t.uint64] * 4 + [t.uint32] * 2)
        ])

    def test_reset_is_exact_masked_and_does_not_advance_or_mutate_source(self):
        t = self.bank.t
        active = t.tensor([True, True, False, True], device=self.bank.device)
        self.bank.uniform(active, draws=3)
        before = self.bank.export_numpy_states()
        # Include a cached uint32 to prove all six fields survive reset.
        cached = Generator(PCG64((1 << 300) + 123))
        cached.integers(0, 2**32, dtype=np.uint32)
        states = [0, cached.bit_generator.state, 42, 99]
        vectors = self.state_vectors(states)
        source = [value.clone() for value in vars(vectors).values()]
        errors = self.bank.reset_device(vectors, active)
        self.assertEqual(errors.cpu().tolist(), [0, 0, 0, 0])
        got = self.bank.export_numpy_states()
        expected = prepare_numpy_pcg64_states(states)
        for row in range(4):
            self.assertEqual(got[row], before[row] if row == 2 else expected[row])
        for value, old in zip(vars(vectors).values(), source):
            self.assertTrue(t.equal(value, old))
        result = self.bank.uniform(active, draws=3)
        for row in (0, 1, 3):
            generator = Generator(PCG64())
            generator.bit_generator.state = expected[row]
            np.testing.assert_array_equal(result.values[row].cpu().numpy(), generator.random(3))
            self.assertEqual(self.bank.export_numpy_states()[row], generator.bit_generator.state)

    def test_reset_invalid_state_is_row_local_and_preserves_old_state(self):
        t = self.bank.t
        before = self.bank.export_numpy_states()
        vectors = self.state_vectors([7, 8, 9, 10])
        vectors.inc_lo[0] = 2
        vectors.has_uint32[1] = 2
        vectors.inc_lo[2] = 0  # Inactive invalid input is ignored.
        active = t.tensor([True, True, False, True], device=self.bank.device)
        errors = self.bank.reset_device(vectors, active)
        self.assertEqual(errors.cpu().tolist(), [5, 5, 0, 0])
        got = self.bank.export_numpy_states()
        for row in range(3):
            self.assertEqual(got[row], before[row])
        self.assertEqual(got[3], PCG64(10).state)
        self.bank.uniform(active & (errors == 0),draws=1)
        self.assertEqual(errors.cpu().tolist(),[5,5,0,0])
        self.assertEqual(self.bank.export_numpy_states()[0],before[0])

    def test_reset_rejects_shape_and_dtype_before_launch(self):
        t = self.bank.t
        vectors = self.state_vectors([7, 8, 9, 10])
        values = dict(vars(vectors))
        values['state_hi'] = t.zeros(4, dtype=t.int64, device=self.bank.device)
        active = t.ones(4, dtype=t.bool, device=self.bank.device)
        with self.assertRaises(ValueError):
            self.bank.reset_device(device_random.DevicePCG64State(**values), active)
        with self.assertRaises(TypeError):
            self.bank.reset_device(values, active)


class DevicePCG64ResetOracleTest(DevicePCG64ResetMixin, unittest.TestCase):
    def setUp(self):
        self.bank = DevicePCG64(SEEDS, device='cpu', allow_cpu_reference=True,
                                max_draws=8, max_choices=8)
        self.addCleanup(self.bank.close)


class DevicePCG64ResetCudaTest(DevicePCG64ResetMixin, unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; no CPU reset fallback')
        self.bank = DevicePCG64(SEEDS, device='cuda', max_draws=8, max_choices=8)
        self.addCleanup(self.bank.close)

    def test_reset_and_draw_capture_reads_changed_device_inputs(self):
        from contextlib import ExitStack
        t = self.bank.t
        active = t.tensor([True, True, False, True], device=self.bank.device)
        vectors = self.state_vectors([7, 8, 9, 10])
        self.bank.reset_device(vectors, active)
        self.bank.uniform(active, draws=2)
        t.cuda.synchronize()
        graph = t.cuda.CUDAGraph()
        with ExitStack() as blockers:
            for name in ('cpu', 'numpy', 'item', 'tolist', '__bool__'):
                blockers.enter_context(patch.object(t.Tensor, name, side_effect=AssertionError(name)))
            with t.cuda.graph(graph):
                reset_errors = self.bank.reset_device(vectors, active)
                draw = self.bank.uniform(active, draws=2)
        # Actual step/actor LUT gather can populate the same bound inputs later.
        changed = self.state_vectors([17, 18, 19, 20])
        for destination, source in zip(vars(vectors).values(), vars(changed).values()):
            destination.copy_(source)
        inactive_before = self.bank.export_numpy_states()[2]
        graph.replay()
        t.cuda.synchronize()
        self.assertEqual(reset_errors.cpu().tolist(), [0, 0, 0, 0])
        for row, seed in enumerate([17, 18, 19, 20]):
            if row == 2:
                self.assertEqual(self.bank.export_numpy_states()[row], inactive_before)
                continue
            generator = Generator(PCG64(seed))
            np.testing.assert_array_equal(draw.values[row].cpu().numpy(), generator.random(2))
            self.assertEqual(self.bank.export_numpy_states()[row], generator.bit_generator.state)


if __name__ == '__main__':
    unittest.main()
