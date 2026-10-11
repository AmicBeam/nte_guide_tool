"""Exact full-prefix comparison; synthetic buffers, no model or rule proof."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as e
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceDecision,DeviceStateBatch
from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
from app.modules.card_game.rl.batched_search.device_decision_equality import compare_decisions_device


def fixtures(device='cpu'):
    import torch as t
    b,m=4,8;kw=dict(device=device)
    def make():
        ints=t.full((b,m,7),-1,dtype=t.int64,**kw);ints[:,:,0]=e.ACT_END
        ints[:,1,0]=e.ACT_ATTACK;ints[:,1,1]=0
        counts=t.full((b,),5,dtype=t.int64,**kw);mask=t.arange(m,**kw)[None,:]<counts[:,None]
        raw=e.RawCandidateTokens(ints,t.zeros((b,m,8),**kw),t.zeros((b,m,2),**kw),t.zeros(b,**kw),
            t.full((b,m,e.MAX_MULLIGAN_SELECTIONS),-1,dtype=t.int64,**kw),counts,mask,
            t.full((b,m,2,e.ORDER_SLOTS),-1,dtype=t.int64,**kw),
            t.full((b,m,e.MAX_MULLIGAN_SELECTIONS),-1,dtype=t.int64,**kw),t.zeros((b,m),**kw),())
        compact=CompactDeviceCandidates(raw,scratch_rows=2,device=device,allow_cpu_reference=device=='cpu')
        states=DeviceStateBatch(None,t.zeros(b,dtype=t.int64,**kw),t.zeros(b,dtype=t.int64,**kw),
            t.arange(b,**kw),t.ones(b,dtype=t.bool,**kw),t.zeros(b,dtype=t.int32,**kw))
        return DeviceDecision(states,None,states.slots,states.generations,t.zeros((b,m),dtype=t.int32,**kw),
            t.zeros((b,m),dtype=t.int64,**kw),counts,mask,t.zeros((b,5),**kw),compact,states.errors.clone())
    left,right=make(),make();right.candidates.scratch_rows=3
    return left,right,t.tensor([True,True,True,False],**kw)


class EqualityTest(unittest.TestCase):
    def test_zero_sign_padding_full_width_and_no_host_reads(self):
        import torch as t
        a,b,on=fixtures();b.observations[0,0]=-0.
        b.candidates.tokens.int_tokens[:,6,0]=e.ACT_ATTACK
        with patch.object(t.Tensor,'item',side_effect=AssertionError('host read')),patch.object(t.Tensor,'cpu',side_effect=AssertionError('host read')),patch.object(t.Tensor,'__bool__',side_effect=AssertionError('host read')):
            errors=compare_decisions_device(a,b,on,allow_cpu_reference=True)
        self.assertEqual(errors.tolist(),[0,0,0,0])
        b.candidates.tokens.extra_ap[1,4]=1.
        self.assertEqual(compare_decisions_device(a,b,on,allow_cpu_reference=True).tolist(),[0,1,0,0])

    def test_mismatch_sticky_errors_nonfinite_and_order(self):
        for kind in ('observation','count','mask','order','nan'):
            a,b,on=fixtures()
            if kind=='observation':b.observations[1,2]=1.
            elif kind=='count':b.counts[1]=4
            elif kind=='mask':b.valid[1,4]=False
            elif kind=='order':b.candidates.tokens.int_tokens[1,[0,1]]=b.candidates.tokens.int_tokens[1,[1,0]]
            else:b.observations[1,2]=float('nan')
            a.errors[2]=64;b.errors[2]=2;b.errors[3]=128
            self.assertEqual(compare_decisions_device(a,b,on,allow_cpu_reference=True).tolist(),[0,1,64,0])

    def test_production_rejects_cpu(self):
        a,b,on=fixtures()
        with self.assertRaises(ValueError):compare_decisions_device(a,b,on)

    def test_cuda_capture_replay_observation_mutation(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        a,b,on=fixtures('cuda');stream=t.cuda.Stream();stream.wait_stream(t.cuda.current_stream());g=t.cuda.CUDAGraph()
        with t.cuda.graph(g,stream=stream):errors=compare_decisions_device(a,b,on)
        g.replay();t.cuda.synchronize();self.assertEqual(errors.cpu().tolist(),[0,0,0,0])
        b.observations[1,1]=1.;g.replay();t.cuda.synchronize()
        self.assertEqual(errors.cpu().tolist(),[0,1,0,0])


if __name__=='__main__':unittest.main()
