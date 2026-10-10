"""Actual device-controlled loop infrastructure; no formal-game evidence."""
import subprocess
import sys
import unittest


class ConditionalGraphTest(unittest.TestCase):
    def test_import_does_not_initialize_torch(self):
        subprocess.run([sys.executable, '-c', "import sys; import app.modules.card_game.rl.batched_search.cuda_conditional_graph; assert 'torch' not in sys.modules"], check=True)

    def test_actual_cuda_1600_row_device_predicate_and_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        from unittest.mock import patch
        target = torch.arange(1600, device='cuda') % 17
        counts = torch.zeros_like(target)
        predicate = torch.zeros(1600, dtype=torch.uint8, device='cuda')
        captured_calls = []
        def initialize():
            counts.zero_()
            predicate.copy_((target > 0).to(torch.uint8))
        def body():
            captured_calls.append(1)
            counts.add_(predicate.to(torch.int64))
            predicate.copy_((counts < target).to(torch.uint8))
        initialize();body();initialize()
        torch.cuda.synchronize()
        runner = CudaConditionalWhile(max_ticks=100, seconds=10)
        try:
            runner.capture(predicate, body, initialize=initialize)
            calls = len(captured_calls)
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('CPU scalar predicate')), \
                 patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU loop fallback')):
                runner.replay()
            torch.cuda.synchronize()
            self.assertEqual(len(captured_calls), calls)
            self.assertTrue(torch.equal(counts, target))
            self.assertEqual(runner.counter.item(), 16)
            self.assertEqual(runner.errors.item(), 0)
            target.fill_(3)
            runner.replay();torch.cuda.synchronize()
            self.assertTrue(torch.equal(counts, target))
            self.assertEqual(runner.counter.item(), 3)
        finally:
            runner.close()

    def test_actual_cuda_guard_refuses_unfinished_rows(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        predicate = torch.ones(4, dtype=torch.uint8, device='cuda')
        counter = torch.zeros(4, dtype=torch.int64, device='cuda')
        runner = CudaConditionalWhile(max_ticks=3, seconds=10)
        try:
            runner.capture(predicate, lambda: counter.add_(1))
            runner.replay();torch.cuda.synchronize()
            self.assertTrue(torch.equal(counter, torch.full_like(counter, 3)))
            self.assertEqual(runner.errors.item(), 1)
        finally:
            runner.close()
