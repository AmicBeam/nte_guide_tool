"""Root observation reuse preserves independent WDL and full Gumbel32 work.

These CPU tensor trees prove scheduling/ownership, not CUDA rule performance.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SoARootObservationReuseTests(unittest.TestCase):
    def test_factory_default_and_explicit_override(self):
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules,create_soa_search
        rules=object.__new__(DeviceSoARules);rules.device='cpu'
        for options,expected in (({},True),({'reuse_root_observation':False},False)):
            with patch('app.modules.card_game.rl.batched_search.device_search.DeviceGumbelSearch') as search:
                create_soa_search(rules,object(),object(),object(),object(),rows=2,
                    random=object(),zhenhong_prior=False,**options)
                self.assertIs(search.call_args.kwargs['reuse_root_observation'],expected)
                self.assertFalse(search.call_args.kwargs.get('reuse_root_policy_value',False))

    def test_full32_keeps_auxiliary_values_rng_keys_budgets_and_leases(self):
        import torch as t
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        helper=PolicyOpponentTest();results=[];queries=[];rng=[];inference_calls=[]
        for enabled in (False,True):
            search,rules,bank,roots,seeds,ids=helper.make()
            self.addCleanup(bank.close)
            search.reuse_root_observation=enabled
            query=[];original=rules.value_observation_device
            def observed(states,viewers):
                query.append(states.active.clone())
                return original(states,viewers)
            rules.value_observation_device=observed
            calls=[]
            class IndependentAuxiliary:
                candidate_dim=2
                def predict_device(self,x,c,valid,models,actor,*,value_only):
                    calls.append((x.clone(),models.clone(),actor.clone(),value_only.clone()))
                    score=.05*models.float()+.03*x[:,0]+.07*actor[:,0]
                    return SimpleNamespace(wdl=t.stack((-score,t.zeros_like(score),score),-1).softmax(-1),
                        invalid_rows=t.zeros(len(x),dtype=t.bool))
            search.auxiliary=IndependentAuxiliary()
            search.initialize(roots,rules.viewer,seeds,ids,ids+11)
            # Independent auxiliary value must not become policy's zero WDL.
            self.assertTrue(bool((search.statistics._value[search._base_slots][roots.active]>0).all()))
            result=helper.finish(search)
            self.assertEqual(result.completed.tolist(),[32,32,0,32])
            self.assertEqual(result.errors.tolist(),[0]*4)
            results.append({k:getattr(result,k).clone() for k in (
                'policy','visit_policy','actions','visits','information_keys','errors',
                'completed','complete','requested','node_count','phase_work')})
            queries.append(sum(int(active.sum()) for active in query))
            rng.append(bank.export_numpy_states());inference_calls.append(calls)
            self.assertEqual(int(rules.live.sum()),4)
            t.testing.assert_close(rules.data[roots.slots],t.tensor([[0,-1]]*4))
        for key in results[0]:t.testing.assert_close(results[0][key],results[1][key],rtol=0,atol=0)
        self.assertEqual(rng[0],rng[1])
        self.assertEqual(queries[0]-queries[1],3)
        self.assertEqual(len(inference_calls[0]),len(inference_calls[1]))
        # Inactive query rows are deliberately unspecified. Active rows must
        # retain identical auxiliary inputs, IDs, context and inference count.
        for old,new in zip(*inference_calls):
            for a,b in zip(old,new):t.testing.assert_close(a,b,rtol=0,atol=0)

    def test_terminal_other_viewer_fallback_and_no_host_tensor_reads(self):
        import torch as t
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        search,rules,bank,roots,seeds,ids=PolicyOpponentTest().make()
        self.addCleanup(bank.close)
        rules.data[roots.slots[0]]=t.tensor([3,1])
        viewers=rules.viewer.clone();viewers[1]=1-viewers[1]
        meta=rules.metadata_device(roots);decision=rules.decision_device(roots,viewers)
        expected=rules.value_observation_device(roots,viewers)
        with patch.object(t.Tensor,'cpu',side_effect=AssertionError('Host copy')), \
             patch.object(t.Tensor,'item',side_effect=AssertionError('Host scalar')), \
             patch.object(t.Tensor,'__bool__',side_effect=AssertionError('Host selector')):
            got,eligible=search._root_observation(roots,viewers,decision,meta,roots.active)
            values=search._fixed_value(roots,viewers,ids[:,0],roots.active,
                observed=got,reuse_mask=eligible)
        self.assertEqual(eligible.tolist(),[False,False,False,True])
        self.assertEqual(search.errors.tolist(),[0]*4)
        for name in ('observations','viewer_is_actor','terminal','terminal_value','errors'):
            t.testing.assert_close(getattr(got,name)[roots.active],getattr(expected,name)[roots.active])
        self.assertEqual(float(values[0]),1.)

    def test_stale_generation_slot_owner_and_arena_refused(self):
        import torch as t
        from app.modules.card_game.rl.batched_search.device_search import METADATA_ERROR
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        for field in ('slots','generations','owners'):
            search,rules,bank,roots,seeds,ids=PolicyOpponentTest().make()
            try:
                decision=rules.decision_device(roots,rules.viewer)
                wrong=getattr(decision.states,field).clone();wrong[0]+=1
                decision=replace(decision,states=replace(decision.states,**{field:wrong}))
                _,eligible=search._root_observation(roots,rules.viewer,decision,
                    rules.metadata_device(roots),roots.active)
                self.assertFalse(bool(eligible[0]));self.assertEqual(int(search.errors[0]),METADATA_ERROR)
                foreign=replace(decision,states=replace(decision.states,pool=object()))
                with self.assertRaisesRegex(ValueError,'same state arena'):
                    search._root_observation(roots,rules.viewer,foreign,rules.metadata_device(roots),roots.active)
            finally:bank.close()

    def test_root_reuse_errors_stop_search_without_leaking_leases(self):
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        from app.modules.card_game.rl.batched_search.device_search import METADATA_ERROR
        helper=PolicyOpponentTest();search,rules,bank,roots,seeds,ids=helper.make()
        self.addCleanup(bank.close);search.reuse_root_observation=True
        original=rules.decision_device
        def stale(states,viewers):
            decision=original(states,viewers)
            gen=states.generations.clone();gen[0]+=1
            return replace(decision,states=replace(states,generations=gen))
        # Only initial root construction is corrupted; search may still run
        # all remaining independent rows at their original full budgets.
        with patch.object(rules,'decision_device',side_effect=stale):
            search.initialize(roots,rules.viewer,seeds,ids,ids)
        result=helper.finish(search)
        self.assertEqual(result.errors.tolist(),[METADATA_ERROR,0,0,0])
        self.assertEqual(result.completed.tolist(),[0,32,0,32])
        self.assertEqual(int(rules.live.sum()),4)

    def test_independent_auxiliary_failures_are_not_hidden_by_root_reuse(self):
        import torch as t
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        from app.modules.card_game.rl.batched_search.device_search import INFERENCE_ERROR
        helper=PolicyOpponentTest();results=[]
        for enabled in (False,True):
            search,rules,bank,roots,seeds,ids=helper.make()
            self.addCleanup(bank.close);search.reuse_root_observation=enabled
            infer=search.auxiliary.predict_device
            def invalid(*args,**kwargs):
                prediction=infer(*args,**kwargs)
                prediction.invalid_rows[0]=True
                return prediction
            with patch.object(search.auxiliary,'predict_device',side_effect=invalid):
                search.initialize(roots,rules.viewer,seeds,ids,ids)
            result=helper.finish(search)
            self.assertEqual(result.errors.tolist(),[INFERENCE_ERROR,0,0,0])
            self.assertEqual(result.completed.tolist(),[0,32,0,32])
            self.assertEqual(int(rules.live.sum()),4)
            results.append(result)
        for name in ('policy','visits','actions','information_keys','errors','completed'):
            t.testing.assert_close(getattr(results[0],name),getattr(results[1],name),rtol=0,atol=0)

    def test_root_value_verification_retains_independent_observation_query(self):
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        search,rules,bank,roots,seeds,ids=PolicyOpponentTest().make()
        self.addCleanup(bank.close)
        search.reuse_root_observation=True;search.verify_root_value_reuse=True
        with patch.object(search,'_root_observation',side_effect=AssertionError('Verification borrowed observation')), \
             patch.object(rules,'value_observation_device',wraps=rules.value_observation_device) as query:
            search.initialize(roots,rules.viewer,seeds,ids,ids)
        self.assertEqual(query.call_count,1)
        self.assertEqual(query.call_args.args[0].active.tolist(),[True,True,False,True])


if __name__=='__main__':unittest.main()
