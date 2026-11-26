"""Finite/mismatch reductions: complete columns, signed zero and padding."""
import unittest
from app.modules.card_game.rl.batched_search.device_guard_candidate_rows import (
    prepare_candidate_row_checks,candidate_nonfinite_rows,candidate_mismatch_rows)


class GuardRowsTest(unittest.TestCase):
    def check(self,device,rows=8,actions=20,width=37):
        import torch as t
        if device=='cuda':prepare_candidate_row_checks(device)
        left=t.randn((rows,actions,width),device=device);right=left.clone()
        lm=t.zeros((rows,actions),device=device,dtype=t.bool);lm[:,:min(12,actions)]=True;rm=lm.clone()
        left[0,0,0]=-0.;right[0,0,0]=0.
        right[1,0,0]=float('nan');left[2,0,0]=float('inf')
        left[3,-1,0]=float('nan');right[4,0,0]+=1
        left[5,0,0]=t.nextafter(t.tensor(0.,device=device),t.tensor(1.,device=device));right[5,0,0]=0.
        expected=(~t.isfinite(left)).any(-1).any(-1)
        reference=((lm|rm)[:,:,None]&((left!=right)|~t.isfinite(left)|~t.isfinite(right))).any(-1).any(-1)
        self.assertTrue(t.equal(candidate_nonfinite_rows(left),expected))
        actual=candidate_mismatch_rows(left,right,lm,rm)
        self.assertTrue(t.equal(actual,reference));self.assertFalse(actual[0].item());self.assertFalse(actual[3].item())
        if device=='cuda':
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):finite=candidate_nonfinite_rows(left);mismatch=candidate_mismatch_rows(left,right,lm,rm)
            left.fill_(0.);right.fill_(0.);lm.zero_();rm.zero_();right[0,0,0]=float('nan')
            graph.replay();t.cuda.synchronize();self.assertFalse(finite.any().item());self.assertFalse(mismatch.any().item())
            rm[0,0]=True;graph.replay();t.cuda.synchronize();self.assertTrue(mismatch[0].item())
    def test_cpu_explicit_reduction_oracle(self):self.check('cpu')
    def test_actual_cuda_full64x800x1576_rows_and_replay(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        self.check('cuda',64,800,1576)


if __name__=='__main__':unittest.main()
