"""One static captured body for 1600 GPU rows, with inert tail and empty mask."""
import unittest
from app.modules.card_game.rl.batched_search.device_tensor_row_queue import DeviceTensorRowQueue

class TensorRowQueueTest(unittest.TestCase):
    def test_constructor_rejects_invalid_and_cpu(self):
        for rows,tile,seconds in ((0,2,5),(3,0,5),(True,2,5),(3,2,0)):
            with self.assertRaises(ValueError):DeviceTensorRowQueue(rows=rows,scratch_rows=tile,seconds=seconds,device='cpu')
        with self.assertRaises(ValueError):DeviceTensorRowQueue(rows=3,scratch_rows=2,seconds=5,device='cpu')
    def test_actual_cuda_one_body_1600_odd_and_empty(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        for rows in (1600,5):
            queue=DeviceTensorRowQueue(rows=rows,scratch_rows=2,seconds=20)
            active=t.ones(rows,dtype=t.bool,device='cuda');output=t.zeros(rows+1,dtype=t.int64,device='cuda');calls=[]
            def operation(ids,valid,destinations):
                calls.append(1);output.index_copy_(0,destinations,t.where(valid,ids+1,0))
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                output.zero_();guard=queue.capture_device(active,operation)
            self.assertEqual(len(calls),1)
            for pattern in (0,3,-1):
                active.copy_(t.ones_like(active) if pattern==0 else t.zeros_like(active) if pattern==-1 else t.arange(rows,device='cuda')%pattern==0)
                graph.replay();t.cuda.synchronize()
                expected=t.where(active,t.arange(rows,device='cuda')+1,0)
                self.assertTrue(t.equal(output[:rows],expected));self.assertFalse(guard.any().item())
                self.assertEqual(queue.loop.counter.item(),0 if pattern==-1 else (rows+1)//2)
            queue.close()

if __name__=='__main__':unittest.main()
