"""Per-decision RNG parity; no game or model-readiness evidence."""
import unittest
from unittest.mock import patch
import numpy as np


class RootRandomPlanTest(unittest.TestCase):
    def make(self, seeds, decisions=4, device='cpu'):
        from app.modules.card_game.rl.batched_search.device_random import DevicePCG64
        from app.modules.card_game.rl.batched_search.device_root_random import DeviceRootRandomPlan
        random = DevicePCG64([0] * len(seeds), device=device, max_draws=5,
                             allow_cpu_reference=device == 'cpu')
        plan = DeviceRootRandomPlan(random, seeds, max_decisions=decisions,
            max_bytes=1024 * 1024, allow_cpu_reference=device == 'cpu')
        return random, plan

    def expected(self, seed, step, actor):
        from app.modules.card_game.rl.batched_search.training_rollout import root_search_seed
        return np.random.PCG64(root_search_seed(seed, step, ('a', 'b')[actor])).random_raw(5)

    def test_each_step_and_actor_resets_exact_numpy_seed(self):
        import torch
        seeds = [0, -1, -(1 << 63), (1 << 63) - 1]
        random, plan = self.make(seeds)
        active = torch.ones(4, dtype=torch.bool)
        try:
            for step in (0, 3, 1, 0):
                for actor in (0, 1):
                    errors = plan.reset_device(torch.full((4,), step),
                                               torch.full((4,), actor), active)
                    self.assertFalse(errors.any())
                    actual = random.raw64(active, draws=5).values.numpy()
                    np.testing.assert_array_equal(actual, np.stack([
                        self.expected(seed, step, actor) for seed in seeds]))
            self.assertEqual(plan.allocated_bytes, 4 * 4 * 2 * 6 * 8 + 4 * 8)
        finally:
            random.close()

    def test_inactive_and_invalid_rows_preserve_previous_state(self):
        import torch
        random, plan = self.make([11, 12, 13, 14])
        before = random.export_numpy_states()
        try:
            errors = plan.reset_device(torch.tensor([2, -1, 4, -100]),
                torch.tensor([1, 0, 1, 100]), torch.tensor([True, True, True, False]))
            self.assertEqual(errors.tolist(), [0, 1, 1, 0])
            after = random.export_numpy_states()
            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1:], after[1:])
        finally:
            random.close()

    def test_byte_budget_refuses_before_seed_expansion(self):
        from app.modules.card_game.rl.batched_search.device_root_random import DeviceRootRandomPlan
        random, _ = self.make([1])
        try:
            with patch.object(np.random, 'PCG64', side_effect=AssertionError('Budget checked late')):
                with self.assertRaises(MemoryError):
                    DeviceRootRandomPlan(random, [1], max_decisions=100,
                                         max_bytes=16, allow_cpu_reference=True)
        finally:
            random.close()

    def test_actual_cuda_1600_rows_capture_and_dynamic_step_actor(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        seeds = list(range(1600))
        random, plan = self.make(seeds, decisions=3, device='cuda')
        rows = torch.arange(1600, device='cuda')
        steps = rows.remainder(3)
        actors = rows.remainder(2)
        active = rows.remainder(5) != 0
        predicate = active.to(torch.uint8)
        errors = torch.zeros(1600, dtype=torch.int32, device='cuda')
        output = torch.zeros((1600, 5), dtype=torch.uint64, device='cuda')
        runner = CudaConditionalWhile(max_ticks=3, seconds=10)
        plan.bind_capture_streams(runner.capture_stream, runner.body_stream)
        def body():
            errors.copy_(plan.reset_device(steps, actors, active))
            output.copy_(random.raw64(active & (errors == 0), draws=5).values)
            predicate.zero_()
        try:
            runner.capture(predicate, body, initialize=lambda: predicate.copy_(active.to(torch.uint8)))
            for step, actor in ((None, None), (2, 1)):
                if step is not None:
                    steps.fill_(step);actors.fill_(actor)
                with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Hot host predicate')), \
                     patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Hot host RNG')), \
                     patch.object(np.random, 'PCG64', side_effect=AssertionError('Hot seed expansion')):
                    runner.replay()
                torch.cuda.synchronize()
                actual = output.cpu().numpy()
                for row in range(1600):
                    expected = (self.expected(row, row % 3 if step is None else step,
                        row % 2 if actor is None else actor) if row % 5 else np.zeros(5, np.uint64))
                    np.testing.assert_array_equal(actual[row], expected)
                self.assertFalse(errors.any().item())
                self.assertEqual(runner.errors.item(), 0)
        finally:
            runner.close();random.close()
