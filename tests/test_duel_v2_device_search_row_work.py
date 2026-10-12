"""Numerical models and complete encodings agree across device queued rows."""
from types import SimpleNamespace
import unittest
from tests import test_duel_v2_compact_device_candidates as fixture
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference
from app.modules.card_game.rl.batched_search.device_info_key import DeviceInformationKey
from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
from app.modules.card_game.rl.batched_search.device_search_row_work import DeviceSearchRowWork

class SearchRowWorkTest(unittest.TestCase):
    def check_parity(self,tile):
        import torch as t
        from app.modules.card_game.rl import seven_lineup
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        cases=fixture._cases();rows=len(cases);capacity=max(32,max(len(actions) for _,actions in cases))
        models={key:fixture.SyntheticSevenModel(key,41+i) for i,key in enumerate(seven_lineup.KEYS)}
        inference=DeviceSevenInference(models,device='cuda',max_rows=rows,max_candidate_elements=tile*capacity*1576)
        keys=DeviceInformationKey(device='cuda',max_rows=rows)
        factory=DeviceSearchRowWork(inference,inference,keys,rows=rows,max_actions=capacity,tile=tile,seconds=60)
        try:
            tokens=fixture._tokens(cases,device='cuda',capacity=capacity)
            compact=CompactDeviceCandidates(tokens,device='cuda',scratch_rows=tile)
            x=fixture._observations(cases,device='cuda');active=t.arange(rows,device='cuda')%3!=1
            ids=t.arange(rows,dtype=t.int64,device='cuda')%len(inference.keys)
            context=t.ones((rows,1),dtype=t.float32,device='cuda')
            decision=SimpleNamespace(candidates=compact,observations=x,valid=tokens.mask,counts=tokens.counts)
            observed=SimpleNamespace(observations=x,viewer_is_actor=context)
            expected_policy=compact.predict_device(inference,x,tokens.mask,ids,context,active)
            expected_hash=compact.hash_device(keys,x,tokens.counts,active)
            value_parts=[];invalid_parts=[]
            for start in range(0,rows,tile):
                stop=min(start+tile,rows);n=stop-start
                result=inference.predict_device(x[start:stop],t.empty((n,0,inference.candidate_dim),device='cuda'),
                    t.empty((n,0),dtype=t.bool,device='cuda'),ids[start:stop],context[start:stop],value_only=t.ones(n,dtype=t.bool,device='cuda'))
                value_parts.append((result.wdl[:,2]-result.wdl[:,0]).to(t.float64));invalid_parts.append(result.invalid_rows)
            expected_value=t.where(active,t.cat(value_parts),0.)
            expected_invalid=active&t.cat(invalid_parts)
            graph=t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                policy=factory.policy(decision,ids,context,active)
                hashed=factory.hash(decision,active)
                value,invalid=factory.value(observed,ids,active)
                second_value,second_invalid=factory.value(observed,ids,active)
                leaf_value,leaf_invalid=factory.value(observed,ids,active)
            graph.replay();t.cuda.synchronize()
            t.testing.assert_close(policy.logits,expected_policy.logits,rtol=1e-5,atol=1e-5)
            t.testing.assert_close(policy.wdl,expected_policy.wdl,rtol=1e-5,atol=1e-5)
            self.assertTrue(t.equal(policy.invalid_rows,expected_policy.invalid_rows))
            self.assertTrue(t.equal(hashed.keys,expected_hash.keys));self.assertTrue(t.equal(hashed.errors,expected_hash.errors))
            t.testing.assert_close(value,expected_value,rtol=1e-5,atol=1e-5);self.assertTrue(t.equal(invalid,expected_invalid))
            for actual,error in ((second_value,second_invalid),(leaf_value,leaf_invalid)):
                t.testing.assert_close(actual,expected_value,rtol=1e-5,atol=1e-5);self.assertTrue(t.equal(error,expected_invalid))
            active.zero_();graph.replay();t.cuda.synchronize()
            for output in (policy.logits,policy.wdl,hashed.keys,hashed.errors,value,invalid,second_value,second_invalid,leaf_value,leaf_invalid):self.assertFalse(output.any().item())
        finally:factory.close();keys.close();inference.close()

    def test_actual_cuda_fullcandidate_policy_hash_and_value_parity(self):self.check_parity(2)
    def test_actual_cuda_eight_row_policy_hash_value_parity(self):self.check_parity(8)

if __name__=='__main__':unittest.main()
