"""Exact output bits and input immutability for private guard encoding."""
import unittest
from dataclasses import replace,fields
from tests import test_duel_v2_compact_device_candidates as fixture
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
from app.modules.card_game.rl.batched_search.device_inplace_guard_encoding import (
    prepare_inplace_guard_encoding,encode_guard_tokens_device,_rewrite)


class InplaceGuardEncodingTest(unittest.TestCase):
    def check(self,device):
        import torch as t
        prepare_inplace_guard_encoding(device);cases=fixture._cases()
        raw=fixture._tokens(cases,device=device);active=t.ones(len(cases),dtype=t.bool,device=device)
        raw=replace(raw,float_tokens=raw.float_tokens.to(t.float64))
        raw.float_tokens[0,0,0]=float('nan');raw.float_tokens[1,0,0]=1e300
        active[2]=False
        before={f.name:getattr(raw,f.name).clone() for f in fields(raw) if isinstance(getattr(raw,f.name),t.Tensor)}
        kwargs=dict(device=device,active=active,allow_cpu_reference=device=='cpu')
        old,oe=encoder.encode_tokens_device(raw,**kwargs)
        new,ne=encode_guard_tokens_device(raw,**kwargs)
        self.assertTrue(t.equal(old.candidates.view(t.int32),new.candidates.view(t.int32)))
        self.assertTrue(t.equal(old.mask,new.mask));self.assertTrue(t.equal(old.counts,new.counts));self.assertTrue(t.equal(oe,ne))
        for name,value in before.items():
            t.testing.assert_close(getattr(raw,name),value,rtol=0,atol=0,equal_nan=True)
        if device=='cuda':
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):captured,ce=encode_guard_tokens_device(raw,**kwargs)
            active.copy_(~active);raw.float_tokens.fill_(0.25)
            graph.replay();t.cuda.synchronize()
            expected,ee=encoder.encode_tokens_device(raw,**kwargs)
            self.assertTrue(t.equal(expected.candidates.view(t.int32),captured.candidates.view(t.int32)))
            self.assertTrue(t.equal(ee,ce))
    def test_source_rewrite_is_unique_and_cpu_exact(self):
        with self.assertRaises(RuntimeError):_rewrite('x=1','x=2','x=3')
        with self.assertRaises(RuntimeError):_rewrite('x=1\nx=1','x=1','x=3')
        self.check('cpu')
    def test_actual_cuda_guard_encoding_exact_and_capture(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        self.check('cuda')


if __name__=='__main__':unittest.main()
