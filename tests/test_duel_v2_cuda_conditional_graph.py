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

    def test_actual_cuda_nested_loops_keep_turns_and_search_on_device(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        from unittest.mock import patch
        rows = torch.arange(1600, device='cuda')
        turns_target = rows.remainder(4)
        search_target = rows.remainder(5) + 1
        turns = torch.zeros_like(rows)
        inner_count = torch.zeros_like(rows)
        work = torch.zeros_like(rows)
        outer_predicate = torch.zeros_like(rows, dtype=torch.uint8)
        inner_predicate = torch.zeros_like(outer_predicate)
        outer = CudaConditionalWhile(max_ticks=10, seconds=20)
        inner = CudaConditionalWhile(max_ticks=10, seconds=10)

        def initialize_outer():
            turns.zero_();work.zero_()
            outer_predicate.copy_((turns_target > 0).to(torch.uint8))

        def initialize_inner():
            inner_count.zero_()
            inner_predicate.copy_(outer_predicate)

        def search_body():
            inner_count.add_(inner_predicate.to(torch.int64))
            inner_predicate.copy_(((outer_predicate != 0)
                & (inner_count < search_target)).to(torch.uint8))

        def turn_body():
            inner.capture_loop(inner_predicate, search_body, initialize=initialize_inner)
            work.add_(inner_count)
            turns.add_(outer_predicate.to(torch.int64))
            outer_predicate.copy_(((turns < turns_target)
                & (inner.errors == 0)).to(torch.uint8))

        try:
            with self.assertRaisesRegex(RuntimeError, 'active CUDA capture'):
                inner.capture_loop(inner_predicate, search_body)
            outer.capture(outer_predicate, turn_body, initialize=initialize_outer)
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host loop predicate')), \
                 patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host root stepping')):
                outer.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(turns, turns_target))
            self.assertTrue(torch.equal(work, turns_target * search_target))
            self.assertEqual(outer.errors.item(), 0)
            self.assertEqual(inner.errors.item(), 0)
            turns_target.fill_(2);search_target.fill_(3)
            outer.replay();torch.cuda.synchronize()
            self.assertTrue(torch.equal(work, torch.full_like(work, 6)))
        finally:
            outer.close()
            inner.close()

class ConditionalStreamOwnershipTest(unittest.TestCase):
    def test_close_keeps_allocator_stream_handles_valid_and_is_idempotent(self):
        from types import SimpleNamespace
        from contextlib import nullcontext
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        destroyed=[];closed=[]
        def destroy(handle):
            destroyed.append(handle)
            raise AssertionError("Allocator-held stream must remain valid")
        loop=object.__new__(CudaConditionalWhile)
        loop.t=SimpleNamespace(cuda=SimpleNamespace(device=lambda device:nullcontext()))
        loop.device=None;loop.runtime=SimpleNamespace(cudaStreamDestroy=destroy)
        loop._owned_streams=[11,12];loop._closed=False;loop.graph=object()
        loop.module=SimpleNamespace(close=lambda:closed.append(True))
        loop.close()
        loop.close()
        self.assertEqual(destroyed,[]);self.assertEqual(closed,[True])
        self.assertEqual(loop._owned_streams,[]);self.assertIsNone(loop.graph)

    def test_actual_cuda_more_than_pool_size_streams_capture_and_replay(self):
        import torch
        from contextlib import ExitStack
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        with ExitStack() as cleanup:
            loops=[CudaConditionalWhile(max_ticks=2,seconds=30) for _ in range(40)]
            for loop in loops:cleanup.callback(loop.close)
            handles=[stream.cuda_stream for loop in loops for stream in (loop.capture_stream,loop.body_stream)]
            self.assertEqual(len(set(handles)),80)
            flags=[torch.zeros(1,dtype=torch.uint8,device='cuda') for _ in loops]
            result=torch.zeros(1,dtype=torch.int64,device='cuda')
            graph=torch.cuda.CUDAGraph();stream=loops[0].capture_stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                result.zero_()
                for loop,flag in zip(loops,flags):
                    def body(flag=flag):
                        result.add_(1);flag.zero_()
                    loop.capture_loop(flag,body,initialize=lambda flag=flag:flag.fill_(1))
            for _ in range(2):
                graph.replay();torch.cuda.synchronize()
                self.assertEqual(result.cpu().tolist(),[40])
                self.assertEqual([int(loop.errors.cpu()[0]) for loop in loops],[0]*40)
            graph=None
            for loop in loops:
                loop.close();loop.close()
                self.assertEqual(loop._owned_streams,[])
