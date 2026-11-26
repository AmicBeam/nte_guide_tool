"""Full ordered guard encoding through two GPU row queues; no game qualification."""
from dataclasses import fields
from types import SimpleNamespace
import unittest
from tests import test_duel_v2_compact_device_candidates as fixture
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
from app.modules.card_game.rl.batched_search.device_inplace_guard_encoding import prepare_inplace_guard_encoding
from app.modules.card_game.rl.batched_search.device_view_sampler import compare_projected_views_device
from app.modules.card_game.rl.batched_search.device_guard_row_work import DeviceGuardRowWork


def _view(tokens,observations,errors):
    return SimpleNamespace(candidates=tokens,observations=SimpleNamespace(observations=observations),errors=errors)


class GuardRowEncodingTest(unittest.TestCase):
    def test_actual_cuda_full800_two_guard_sites_and_tail(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required; local skip is not GPU evidence')
        prepare_inplace_guard_encoding('cuda');table=encoder.get_card_template_table('cuda')
        cases=fixture._cases()
        for rows in (64,65):
            selected=[cases[i%len(cases)] for i in range(rows)]
            raw=fixture._tokens(selected,device='cuda',capacity=800)
            right=encoder.RawCandidateTokens(**{f.name:getattr(raw,f.name).clone() for f in fields(raw) if f.name!='request_ids'})
            x=fixture._observations(selected,device='cuda');y=x.clone()
            left_errors=t.zeros(rows,dtype=t.int32,device='cuda');right_errors=left_errors.clone()
            left=_view(raw,x,left_errors);bad=_view(right,y,right_errors)
            active=t.ones(rows,dtype=t.bool,device='cuda')
            work=DeviceGuardRowWork(rows=rows,tile=8,seconds=20,device='cuda',compare=compare_projected_views_device)
            try:
                graph=t.cuda.CUDAGraph()
                with t.cuda.graph(graph):
                    inner=work.compare(left,bad,active,table)
                    outer=work.compare(left,left,active,table)
                # Dynamic errors span successive tiles and the final tail row.
                y[0,0]+=1.;right.extra_ap[7,0]+=1.
                right.mask[8,0]=False;right.counts[15]=0
                right.float_tokens[31,0,0]=float('nan')
                right_errors[rows-1]=123;left_errors[rows-2]=119
                for mode in ('all','sparse','empty','restored'):
                    if mode=='all':active.fill_(True)
                    elif mode=='sparse':active.copy_(t.arange(rows,device='cuda')%3==0)
                    elif mode=='empty':active.zero_()
                    else:
                        active.fill_(True);y.copy_(x);left_errors.zero_();right_errors.zero_()
                        for f in fields(raw):
                            if f.name!='request_ids':getattr(right,f.name).copy_(getattr(raw,f.name))
                    graph.replay();t.cuda.synchronize()
                    expected=compare_projected_views_device(left,bad,active,table)
                    outer_expected=compare_projected_views_device(left,left,active,table)
                    self.assertTrue(t.equal(inner,expected),mode)
                    self.assertTrue(t.equal(outer,outer_expected),mode)
                    self.assertEqual(inner.dtype,t.int32);self.assertEqual(tuple(inner.shape),(rows,))
                    if mode in ('empty','restored'):self.assertFalse(inner.any().item())
                with self.assertRaises(RuntimeError):work.compare(left,bad,active,table)
            finally:work.close()
            work.close()
            with self.assertRaises(RuntimeError):work.compare(left,bad,active,table)


if __name__=='__main__':unittest.main()
