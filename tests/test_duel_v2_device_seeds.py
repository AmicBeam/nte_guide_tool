"""Per-root recovery seed tests; 1600 rows are not complete game evidence."""
import hashlib
import inspect
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_search.device_seeds import (
    SEED_ERR_INVALID_ACTOR,
    SEED_ERR_NEGATIVE_STEP,
    DeviceRootSeeds,
)
from app.modules.card_game.rl.batched_search.training_rollout import root_search_seed

INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807
MASK64 = (1 << 64) - 1


def independent_limbs(job_seed, step, actor_id):
    payload = f"recovery:{int(job_seed)}:{int(step)}:{'ab'[int(actor_id)]}".encode('utf-8')
    seed = (1 << 256) | int(hashlib.sha256(payload).hexdigest(), 16)
    words = np.array([(seed >> (64 * i)) & MASK64 for i in range(5)], dtype=np.uint64)
    return words.view(np.int64)


def limbs_to_int(row):
    acc = 0
    for i, word in enumerate(row):
        acc |= (int(word) & MASK64) << (64 * i)
    return acc


def expected_batch(job_seeds, steps, actors, active):
    js, st, ac, on = (v.detach().cpu().numpy() for v in (job_seeds, steps, actors, active))
    limbs = np.zeros((len(js), 5), dtype=np.int64)
    errors = np.zeros(len(js), dtype=np.int32)
    for i in range(len(js)):
        if not on[i]:
            continue
        if st[i] < 0:
            errors[i] = SEED_ERR_NEGATIVE_STEP
            continue
        if ac[i] != 0 and ac[i] != 1:
            errors[i] = SEED_ERR_INVALID_ACTOR
            continue
        limbs[i] = independent_limbs(js[i], st[i], ac[i])
    return limbs, errors


class DeviceRootSeedsTest(unittest.TestCase):
    def engine(self, **kwargs):
        engine = DeviceRootSeeds(device='cpu', allow_cpu_reference=True, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def tensors(self, job_seeds, steps, actors, active, device='cpu'):
        import torch
        return (
            torch.as_tensor(np.asarray(job_seeds, dtype=np.int64), device=device),
            torch.as_tensor(np.asarray(steps, dtype=np.int64), device=device),
            torch.as_tensor(np.asarray(actors, dtype=np.int64), device=device),
            torch.as_tensor(np.asarray(active, dtype=np.bool_), device=device),
        )

    def test_cpu_reference_matches_independent_hashlib_edges(self):
        engine = self.engine()
        job_seeds = [0, 1, -1, INT64_MIN, INT64_MAX, 10, -10]
        steps = [0, 1, 10, 0, INT64_MAX, 2, 3]
        actors = [0, 1, 0, 1, 0, 1, 0]
        active = [True] * 7
        result = engine.derive(*self.tensors(job_seeds, steps, actors, active))
        expected, errors = expected_batch(*self.tensors(job_seeds, steps, actors, active))
        np.testing.assert_array_equal(result.limbs.numpy(), expected)
        np.testing.assert_array_equal(result.errors.numpy(), errors)
        self.assertTrue(bool((result.limbs[:, 4] == 1).all()))
        for i, (job, step, actor) in enumerate(zip(job_seeds, steps, actors)):
            seed = root_search_seed(job, step, 'ab'[actor])
            self.assertEqual(limbs_to_int(result.limbs[i].numpy()), seed)
            self.assertNotEqual(seed >> 256, 0)

    def test_cpu_mixed_mask_inactive_ignores_invalid(self):
        engine = self.engine()
        job_seeds = [INT64_MIN, INT64_MAX, 123, 4, 5, 6]
        steps = [0, INT64_MAX, -3, -1, 0, -9]
        actors = [0, 1, 9, 0, 2, 3]
        active = [True, True, False, True, True, True]
        result = engine.derive(*self.tensors(job_seeds, steps, actors, active))
        expected, errors = expected_batch(*self.tensors(job_seeds, steps, actors, active))
        np.testing.assert_array_equal(result.limbs.numpy(), expected)
        self.assertEqual(result.errors.tolist(), [0, 0, 0, 1, 2, 1])
        np.testing.assert_array_equal(result.errors.numpy(), errors)
        self.assertTrue(bool((result.limbs[2:] == 0).all()))
        self.assertEqual(
            limbs_to_int(result.limbs[0].numpy()),
            root_search_seed(INT64_MIN, 0, 'a'),
        )
        self.assertEqual(
            limbs_to_int(result.limbs[1].numpy()),
            root_search_seed(INT64_MAX, INT64_MAX, 'b'),
        )

    def test_reference_must_be_explicit_and_static_limits_reject(self):
        import torch
        with self.assertRaises(RuntimeError):
            DeviceRootSeeds(device='cpu')
        engine = self.engine(max_rows=2)
        args = self.tensors([1, 2, 3], [0, 0, 0], [0, 1, 0], [True, True, True])
        with self.assertRaises(ValueError):
            engine.derive(*args)
        small = self.tensors([1, 2], [0, 0], [0, 1], [True, True])
        strided = small[0].as_strided((2,), (1,))
        if strided.is_contiguous():
            wide = torch.zeros(4, dtype=torch.int64)
            strided = wide[::2]
        with self.assertRaises(ValueError):
            engine.derive(strided, small[1], small[2], small[3])
        source = inspect.getsource(DeviceRootSeeds.derive)
        from app.modules.card_game.rl.batched_search import device_seeds as mod
        for blob in (source, inspect.getsource(DeviceRootSeeds._reference), mod._CUDA_SOURCE):
            self.assertNotIn('uuid', blob.lower())
            self.assertNotIn('torch.rand', blob)
        for needle in ('.cpu(', '.numpy(', '.item(', '.tolist('):
            self.assertNotIn(needle, source)
            self.assertNotIn(needle, mod._CUDA_SOURCE)

    def _without_host_reads(self, torch):
        stack = ExitStack()
        for name, label in (('cpu', 'CPU copy'), ('numpy', 'NumPy copy'),
                            ('item', 'Scalar copy'), ('tolist', 'List copy')):
            stack.enter_context(patch.object(
                torch.Tensor, name, side_effect=AssertionError(label)))
        return stack

    def test_actual_cuda_1600_conditional_graph_replay_varied_steps_and_invalid_isolation(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; per-root seed derivation only')
        engine = DeviceRootSeeds(device='cuda', max_rows=1600)
        self.addCleanup(engine.close)
        job_seeds = torch.arange(1600, dtype=torch.int64, device='cuda') - 800
        job_seeds[0] = INT64_MIN
        job_seeds[1] = INT64_MAX
        job_seeds[2] = -1
        job_seeds[3] = 0
        steps = torch.arange(1600, dtype=torch.int64, device='cuda')
        actors = (torch.arange(1600, dtype=torch.int64, device='cuda') % 2)
        active = torch.ones(1600, dtype=torch.bool, device='cuda')
        active[::17] = False
        active[0] = True
        args = (job_seeds, steps, actors, active)
        with self._without_host_reads(torch):
            actual = engine.derive(*args)
        want, want_err = expected_batch(*args)
        np.testing.assert_array_equal(actual.limbs.cpu().numpy(), want)
        np.testing.assert_array_equal(actual.errors.cpu().numpy(), want_err)

        for _ in range(3):
            engine.derive(*args)
        torch.cuda.synchronize()
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        predicate = torch.zeros(1600, dtype=torch.uint8, device='cuda')
        captured = {}

        def initialize():
            predicate.fill_(1)

        def body():
            captured['result'] = engine.derive(*args)
            predicate.zero_()

        initialize()
        body()
        initialize()
        torch.cuda.synchronize()
        runner = CudaConditionalWhile(device=str(engine.device), max_ticks=8, seconds=10)
        self.addCleanup(runner.close)
        runner.capture(predicate, body, initialize=initialize)
        with self._without_host_reads(torch):
            runner.replay()
        torch.cuda.synchronize()
        result = captured['result']
        want, want_err = expected_batch(*args)
        np.testing.assert_array_equal(result.limbs.cpu().numpy(), want)
        np.testing.assert_array_equal(result.errors.cpu().numpy(), want_err)

        steps.add_(5)
        actors.copy_((actors + 1) % 2)
        with self._without_host_reads(torch):
            runner.replay()
        torch.cuda.synchronize()
        want, want_err = expected_batch(*args)
        np.testing.assert_array_equal(result.limbs.cpu().numpy(), want)
        np.testing.assert_array_equal(result.errors.cpu().numpy(), want_err)

        steps[5] = -1
        actors[6] = 2
        steps[7] = -4
        actors[7] = 9
        steps[17] = -8
        actors[17] = 4
        with self._without_host_reads(torch):
            runner.replay()
        torch.cuda.synchronize()
        want, want_err = expected_batch(*args)
        np.testing.assert_array_equal(result.limbs.cpu().numpy(), want)
        np.testing.assert_array_equal(result.errors.cpu().numpy(), want_err)
        self.assertEqual(int(result.errors[5].cpu()), SEED_ERR_NEGATIVE_STEP)
        self.assertEqual(int(result.errors[6].cpu()), SEED_ERR_INVALID_ACTOR)
        self.assertEqual(int(result.errors[7].cpu()), SEED_ERR_NEGATIVE_STEP)
        self.assertEqual(int(result.errors[17].cpu()), 0)
        self.assertTrue(bool((result.limbs[5:8] == 0).all()))


if __name__ == '__main__':
    unittest.main()
