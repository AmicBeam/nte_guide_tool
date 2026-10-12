"""Complete frozen numerical policy/value outputs with reused norm work."""
import unittest
from tests import test_duel_v2_compact_device_candidates as fixture
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference
from app.modules.card_game.rl.batched_search.device_memory_bounded_inference import DeviceMemoryBoundInference

class MemoryBoundedModelParityTest(unittest.TestCase):
    def test_actual_cuda_models_inputs_and_capture_parity(self):
        import torch as t
        from app.modules.card_game.rl import seven_lineup
        from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        cases=fixture._cases();models={key:fixture.SyntheticSevenModel(key,77+i) for i,key in enumerate(seven_lineup.KEYS)}
        old=DeviceSevenInference(models,device='cuda',max_rows=len(cases));new=DeviceMemoryBoundInference(models,device='cuda',max_rows=len(cases))
        try:
            width=old.weights['cand_net.0.weight'].shape[-1]
            stress=t.randn((8,800,width),device='cuda');stress_ids=t.arange(8,device='cuda')%len(old.keys)
            stress_before=stress.clone();old_norm=old._norm(stress,stress_ids,'cand_net.0',True);new_norm=new._norm(stress,stress_ids,'cand_net.0',True)
            self.assertTrue(t.equal(old_norm,new_norm));self.assertTrue(t.equal(stress,stress_before))
            raw=fixture._tokens(cases,device='cuda');packed,_=encoder.encode_tokens_device(raw,device='cuda')
            x=fixture._observations(cases,device='cuda');ids=t.arange(len(cases),device='cuda',dtype=t.int64)%len(old.keys);context=t.ones((len(cases),1),device='cuda')
            x_before=x.clone();c_before=packed.candidates.clone()
            reference=old.predict_device(x,packed.candidates,packed.mask,ids,context)
            new.predict_device(x,packed.candidates,packed.mask,ids,context)
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):actual=new.predict_device(x,packed.candidates,packed.mask,ids,context)
            graph.replay();t.cuda.synchronize()
            t.testing.assert_close(actual.logits,reference.logits,rtol=1e-5,atol=1e-5);t.testing.assert_close(actual.wdl,reference.wdl,rtol=1e-5,atol=1e-5)
            self.assertTrue(t.equal(actual.invalid_rows,reference.invalid_rows));self.assertTrue(t.equal(x,x_before));self.assertTrue(t.equal(packed.candidates,c_before))
        finally:old.close();new.close()

if __name__=='__main__':unittest.main()
