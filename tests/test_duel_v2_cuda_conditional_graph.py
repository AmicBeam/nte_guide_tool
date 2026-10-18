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


class NativeConditionalContractTest(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        events=[]
        parent=SimpleNamespace(cuda_stream=11,device='cuda:0')
        child=SimpleNamespace(cuda_stream=99,device='cuda:0')
        state=SimpleNamespace(stream=parent,capturing=False)
        class Graph:
            @staticmethod
            def get_currently_capturing_graph():return graph
            def begin_capture_to_while_node(self,pred):
                events.append(('begin',pred));state.stream=child
            def set_conditional_handle_for_current_node(self,pred):events.append(('set',pred))
            def end_capture_to_conditional_node(self):events.append(('end',));state.stream=parent
        graph=Graph()
        t=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph,
            current_stream=lambda device:state.stream,
            is_current_stream_capturing=lambda:state.capturing))
        runner=object.__new__(CudaConditionalWhile)
        runner.t=t;runner.device='cuda:0';runner._closed=False;runner._loop_built=False
        runner._condition_scalar=object();runner._condition_bytes=object()
        runner.counter=object();runner.start=object();runner.errors=object()
        runner.max_ticks=7;runner.budget_ns=2500000000;runner._capture_stream_callbacks=[]
        runner.module=SimpleNamespace(launch=lambda name,args,**kw:events.append((name,args,kw)))
        predicate=SimpleNamespace(numel=lambda:1600)
        return runner,predicate,events,state,parent,child

    def test_native_graph_owns_child_and_observer_sees_exact_stream(self):
        runner,pred,events,state,parent,child=self.fixture()
        runner.register_capture_stream_callback(lambda p,b:events.append(('observer',p,b,state.stream)))
        state.capturing=True
        with runner._while_body(pred):events.append(('body',state.stream))
        self.assertEqual([e[0] for e in events],['update_condition','begin','observer','body','update_condition','set','end'])
        self.assertEqual(events[2][1:],(parent,child,child))
        self.assertIs(runner.body_stream,child)
        self.assertIs(state.stream,parent)
        for event,advance in ((events[0],0),(events[4],1)):
            self.assertIs(event[1][0],runner._condition_bytes)
            self.assertIs(event[1][1],pred)
            self.assertEqual(event[1][2:], [1600,runner.counter,runner.start,runner.errors,7,2500000000,advance])
            self.assertEqual(event[2],dict(grid=1,block=1,capture_safe=True))

    def test_capture_exception_always_ends_native_child_without_rearming(self):
        runner,pred,events,state,parent,_=self.fixture();state.capturing=True
        with self.assertRaisesRegex(ValueError,'body failed'):
            with runner._while_body(pred):raise ValueError('body failed')
        self.assertEqual([e[0] for e in events],['update_condition','begin','end'])
        self.assertIs(state.stream,parent)

    def test_missing_native_api_fails_closed(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        runner,_,_,_,_,_=self.fixture()
        CudaConditionalWhile._require_native_api(runner.t)
        with self.assertRaisesRegex(RuntimeError,'native conditional.*no raw fallback'):
            CudaConditionalWhile._require_native_api(SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=object)))

    def test_stream_observers_frozen_before_capture(self):
        runner,_,_,state,_,_=self.fixture()
        with self.assertRaises(TypeError):runner.register_capture_stream_callback(None)
        state.capturing=True
        with self.assertRaisesRegex(RuntimeError,'before static'):runner.register_capture_stream_callback(lambda *_:None)
        state.capturing=False;runner._loop_built=True
        with self.assertRaisesRegex(RuntimeError,'before static'):runner.register_capture_stream_callback(lambda *_:None)

    def test_guard_preserves_iteration_clock_and_unfinished_failure_condition(self):
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import _SOURCE
        self.assertIn('if(advance)(*counter)++',_SOURCE)
        self.assertIn('unfinished&&(*counter>=max_ticks||elapsed>=(unsigned long long)budget_ns)',_SOURCE)
        self.assertIn('*errors|=1',_SOURCE)
        self.assertIn('*proceed=(unsigned char)(unfinished&&!*errors)',_SOURCE)
        self.assertIn('%%globaltimer',_SOURCE)
        self.assertNotIn('cudaGraphSetConditional',_SOURCE)


class NativeChunkStreamContractTest(unittest.TestCase):
    def fixture(self):
        import threading
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.device_chunked_games import DeviceChunkedGameLoop
        class Stream:
            def __init__(self,handle,device='cuda:0'):self.cuda_stream=handle;self.device=device
        parent=Stream(11);body=Stream(99)
        state=SimpleNamespace(stream=body,capturing=True)
        owner=SimpleNamespace(_owner_thread=threading.get_ident(),_capture_depth=1,_capture_streams=frozenset((11,)))
        random=SimpleNamespace(_thread=threading.get_ident(),_capture_streams=frozenset((11,)))
        loop=object.__new__(DeviceChunkedGameLoop);loop.device='cuda:0';loop._counts_as_search=True
        loop.search=SimpleNamespace(statistics=owner);loop.random_plan=random
        loop.t=SimpleNamespace(cuda=SimpleNamespace(Stream=Stream,
            current_stream=lambda device:state.stream,is_current_stream_capturing=lambda:state.capturing))
        return loop,parent,body,state,owner,random

    def test_exact_native_child_registration_retains_existing_owners(self):
        loop,parent,body,_,owner,random=self.fixture()
        loop._register_native_capture_stream(parent,body)
        self.assertEqual(owner._capture_streams,frozenset((11,99)))
        self.assertEqual(random._capture_streams,frozenset((11,99)))
        grandchild=type(body)(101)
        loop.t.cuda.current_stream=lambda device:grandchild
        loop._register_native_capture_stream(body,grandchild)
        self.assertEqual(owner._capture_streams,frozenset((11,99,101)))

    def test_base_and_chunked_prepare_only_known_parent_then_observe_native_child(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        for cls in (DeviceGameLoop,type(self.fixture()[0])):
            with self.subTest(driver=cls.__name__):
                fixture,parent,body,state,owner,random=self.fixture()
                loop=object.__new__(cls);loop.__dict__.update(fixture.__dict__)
                observed=[]
                def bind(target):
                    def apply(a,b):
                        self.assertIs(a,parent);self.assertIs(b,parent)
                        target._capture_streams=frozenset((a.cuda_stream,b.cuda_stream))
                    return apply
                owner.bind_capture_streams=bind(owner);random.bind_capture_streams=bind(random)
                outer=SimpleNamespace(capture_stream=parent,
                    register_capture_stream_callback=lambda cb:observed.append(cb))
                inner=SimpleNamespace(register_capture_stream_callback=lambda cb:observed.append(cb))
                loop._outer=outer;loop._prepare_capture_streams((outer,inner))
                self.assertEqual(owner._capture_streams,frozenset((11,)))
                self.assertEqual(len(observed),2)
                observed[0](parent,body)
                self.assertEqual(random._capture_streams,frozenset((11,99)))

    def test_foreign_parent_thread_scope_device_or_current_stream_rejected_atomically(self):
        for mutation in ('parent','thread','scope','device','current','eager'):
            with self.subTest(mutation=mutation):
                loop,parent,body,state,owner,random=self.fixture()
                if mutation=='parent':parent.cuda_stream=777
                if mutation=='thread':random._thread=-1
                if mutation=='scope':owner._capture_depth=0
                if mutation=='device':body.device='cuda:1'
                if mutation=='current':state.stream=parent
                if mutation=='eager':state.capturing=False
                with self.assertRaises((RuntimeError,ValueError)):
                    loop._register_native_capture_stream(parent,body)
                self.assertEqual(owner._capture_streams,frozenset((11,)))
                self.assertEqual(random._capture_streams,frozenset((11,)))


class NativeAllocatorOwnershipTest(unittest.TestCase):
    def test_actual_cuda_large_nested_temporaries_survive_independent_graph_capture(self):
        import torch
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        outer=CudaConditionalWhile(max_ticks=3,seconds=10)
        inner=CudaConditionalWhile(max_ticks=3,seconds=10)
        x=torch.arange(2**21,device='cuda',dtype=torch.float32)
        out=torch.zeros_like(x)
        outer_pred=torch.ones(1,device='cuda',dtype=torch.uint8)
        inner_pred=torch.ones_like(outer_pred)
        def body():
            # This 8MiB allocation has no Python reference after static capture.
            work=torch.empty_like(x)
            torch.mul(x,2,out=work);work.add_(3);out.copy_(work)
            inner_pred.zero_()
        def turn():
            inner.capture_loop(inner_pred,body,initialize=lambda:inner_pred.fill_(1))
            outer_pred.zero_()
        try:
            outer.capture(outer_pred,turn,initialize=lambda:outer_pred.fill_(1))
            later=torch.cuda.CUDAGraph();later_stream=torch.cuda.Stream()
            later_stream.wait_stream(torch.cuda.current_stream())
            scratch=torch.zeros(1,device='cuda')
            # torch.cuda.graph empties the allocator cache on entry; a raw
            # unowned conditional child's temporary allocation was reclaimed.
            with torch.cuda.graph(later,stream=later_stream,capture_error_mode='thread_local'):
                scratch.add_(1)
            for _ in range(4):
                outer.replay();later.replay();torch.cuda.synchronize()
                self.assertTrue(torch.equal(out,x*2+3))
                self.assertEqual(outer.errors.item(),0)
                self.assertEqual(inner.errors.item(),0)
        finally:
            outer.close();inner.close()
