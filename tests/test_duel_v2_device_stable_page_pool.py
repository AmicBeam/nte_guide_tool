"""Exact page order and every original allocator lifecycle invariant."""
import unittest
from tests import test_duel_v2_device_graph_pool as suite
from app.modules.card_game.rl.batched_search.device_stable_page_pool import stable_free_page_order,DeviceStableGraphPagePool

class StablePageOrderTest(unittest.TestCase):
    def parity(self,device):
        import torch as t
        for n in (1,2,7,64,65536):
            ids=t.arange(n,device=device)
            for free in (ids<0,ids>=0,ids%2==0,ids%7<3):
                expected=t.argsort((~free).long()*n+ids)
                got=stable_free_page_order(free,allow_cpu_reference=device=='cpu')
                self.assertTrue(t.equal(expected,got))
    def test_cpu_exact_order_and_invalid(self):
        import torch as t
        self.parity('cpu')
        for value in (None,t.empty(0,dtype=t.bool),t.zeros(3,dtype=t.int64),t.ones((2,2),dtype=t.bool)):
            with self.assertRaises(ValueError):stable_free_page_order(value,allow_cpu_reference=True)
        with self.assertRaises(ValueError):stable_free_page_order(t.ones(3,dtype=t.bool))
    def test_actual_cuda_exact_order_and_capture(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        self.parity('cuda')
        free=t.arange(65536,device='cuda')%3==0
        stream=t.cuda.Stream();stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream):stable_free_page_order(free)
        t.cuda.current_stream().wait_stream(stream)
        graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph):order=stable_free_page_order(free)
        for divisor in (3,5,7):
            free.copy_(t.arange(65536,device='cuda')%divisor==0);graph.replay();t.cuda.synchronize()
            self.assertTrue(t.equal(order,t.argsort((~free).long()*free.numel()+t.arange(free.numel(),device='cuda'))))

class StablePagePoolLifecycleTest(suite.DeviceGraphPoolTest):
    def pool(self,device='cpu',slots=4,pages=8,per_slot=4):
        return DeviceStableGraphPagePool(slots=slots,pages=pages,page_bytes=16,
            max_pages_per_slot=per_slot,device=device,allow_cpu_reference=device=='cpu')

if __name__=='__main__':unittest.main()
