"""Root row-mask arithmetic is synthetic; it does not prove source equality."""
import unittest
from types import SimpleNamespace as Obj
from dataclasses import replace
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_search.device_root_value_reuse import (
    inspect_root_value_reuse, root_value_rows)


class RootValueReuseTest(unittest.TestCase):
    def test_unknown_source_keeps_original_fixed_value_path(self):
        import torch as t
        from tests import test_duel_v2_device_search as fixtures
        from app.modules.card_game.rl.batched_search.device_root_value_reuse import DeviceRootValueReuse
        factory=fixtures.DeviceSearchTest()
        old,old_rules=factory.engine()
        new,new_rules=factory.engine()
        self.addCleanup(old.close);self.addCleanup(new.close)
        new.root_value_reuse=DeviceRootValueReuse(new_rules,new.inference,new.auxiliary)
        new.verify_root_value_reuse=True
        self.assertFalse(new.root_value_reuse.enabled)
        factory.initialize(old,old_rules)
        with patch.object(new.root_value_reuse,'value_device',side_effect=AssertionError('Unproved reuse')), \
             patch.object(new,'_fixed_value',wraps=new._fixed_value) as fallback:
            factory.initialize(new,new_rules)
        self.assertEqual(fallback.call_count,1)
        self.assertEqual(new.root_value_reuse_rows.item(),0)
        self.assertTrue(t.equal(old.statistics._value[old._base_slots],new.statistics._value[new._base_slots]))

    def inputs(self, device='cpu'):
        import torch as t
        b=10; pool=object(); ids=t.arange(b,device=device,dtype=t.int64)
        states=DeviceStateBatch(pool,ids,ids.clone(),ids.clone(),t.ones(b,device=device,dtype=t.bool),t.zeros(b,device=device,dtype=t.int32))
        decision=Obj(states=replace(states,generations=states.generations.clone(),owners=states.owners.clone()),errors=states.errors.clone(),counts=t.ones_like(ids),observations=t.ones((b,3),device=device))
        meta=Obj(actors=ids%2,finished=t.zeros(b,device=device,dtype=t.bool),errors=states.errors.clone())
        prediction=Obj(wdl=t.tensor([[.1,.2,.7]]*b,device=device,dtype=t.float32),invalid_rows=t.zeros(b,device=device,dtype=t.bool))
        return states,decision,meta,prediction,ids%2,ids%3,ids%3,t.ones(b,device=device,dtype=t.bool)

    def check(self,device,reference=False):
        import torch as t
        args=self.inputs(device)
        value,on=root_value_rows(*args,allow_cpu_reference=reference)
        self.assertTrue(t.equal(value,(args[3].wdl[:,2]-args[3].wdl[:,0]).to(t.float64)))
        self.assertTrue(on.all().item())
        states,decision,meta,prediction,viewers,policy,aux,active=args
        aux[0]+=1; decision.states.generations[1]+=1;decision.states.owners[2]+=1
        meta.actors[3]=1-meta.actors[3];meta.finished[4]=True
        prediction.wdl[5,0]=float('nan');prediction.invalid_rows[6]=True
        decision.errors[7]=1;active[8]=False;decision.observations[9,0]=float('inf')
        value,on=root_value_rows(*args,allow_cpu_reference=reference)
        self.assertFalse(on.any().item());self.assertTrue((value==0).all().item())

    def test_unknown_adapter_rejected_and_cpu_requires_explicit_oracle(self):
        self.assertTrue(inspect_root_value_reuse(Obj(closed=False),Obj(),Obj()))
        with self.assertRaises(ValueError): root_value_rows(*self.inputs())
        self.check('cpu',True)

    def test_actual_cuda_synthetic_rows_and_changing_capture_masks(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        self.check('cuda')
        args=self.inputs('cuda');graph=t.cuda.CUDAGraph()
        with patch.object(t.Tensor,'cpu',side_effect=AssertionError('Host copy')),patch.object(t.Tensor,'item',side_effect=AssertionError('Host selector')):
            with t.cuda.graph(graph):value,on=root_value_rows(*args)
            graph.replay()
        t.cuda.synchronize();self.assertTrue(on.all().item())
        args[6][0]+=1;args[2].finished[1]=True
        graph.replay();t.cuda.synchronize();self.assertEqual(on[:3].cpu().tolist(),[False,False,True])


if __name__=='__main__':unittest.main()
