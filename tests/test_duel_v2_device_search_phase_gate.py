"""Actual CUDA one-shot branches; skipped buffers must keep their contents."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import torch


@unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
class SearchPhaseGateCudaTest(unittest.TestCase):
    def test_empty_then_active_phases_replay_without_host_predicates(self):
        from app.modules.card_game.rl.batched_search.device_search_phase_gate import DeviceSearchPhaseGates
        gates=DeviceSearchPhaseGates(phases=2,device='cuda')
        active=torch.zeros((2,3),dtype=torch.bool,device='cuda')
        counts=torch.tensor([7,11],dtype=torch.int64,device='cuda')
        graph=torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize()
            with ExitStack()as stack:
                for name in ('cpu','numpy','item','tolist','__bool__'):
                    stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Host predicate')))
                with torch.cuda.graph(graph):
                    for phase in range(2):
                        gates.capture(phase,active[phase],lambda phase=phase:counts[phase].add_(1))
                graph.replay()
            torch.cuda.synchronize();self.assertEqual(counts.cpu().tolist(),[7,11])
            active[0,2]=True;graph.replay();torch.cuda.synchronize()
            self.assertEqual(counts.cpu().tolist(),[8,11])
            active[0].zero_();active[1,0]=True;graph.replay();torch.cuda.synchronize()
            self.assertEqual(counts.cpu().tolist(),[8,12])
            for loop in gates.loops:self.assertEqual(loop.errors.cpu().tolist(),[0])
        finally:
            graph=None;gates.close()


if __name__=='__main__':unittest.main()
