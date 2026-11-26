"""Covered teacher equivalence on explicit synthetic/CPU oracles; no CUDA claim."""
from copy import deepcopy
import time
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_duel_v2_batch_traversal import FakeBackend, drive_traversal
from app.modules.card_game.rl import recovery_search
from app.modules.card_game.rl.batched_search.traversal import search_traversal, _teacher_terminal_generator
from app.modules.card_game.rl.batched_search.cooperative import cooperative_search, CooperativeScheduler, RootJob
from app.modules.card_game.rl.batched_search.inference import InferenceResult


class TeacherBackend(FakeBackend):
    backend_kind='python_oracle'
    supports_gpu=False
    def rule_decision(self,state,side):
        return self.decision(state,side)[0],1
    def value_observation(self,state,viewer):
        x,c=self.decision(state,viewer)[1]
        return x,c[:0]


class TeacherEquivalenceTest(unittest.TestCase):
    def test_all_covered_modes_match_sequential_q_pi_choice_and_noise(self):
        backend=TeacherBackend(terminal_depth=8)
        root={'phase':'playing','actor':'a','depth':0,'active_side':'a'}
        logits=np.asarray([.5,-.5]);wdl=np.asarray([.2,.2,.6])
        class Runtime:
            decision=backend.decision
            @staticmethod
            def predict_encoded(policy,x,c):return logits.copy(),.4
            @staticmethod
            def action_scores(policy,x,c):return logits.copy()
            @staticmethod
            def value_for(state,viewer,policies):return .4
        for teacher in ('covered_terminal','covered_expert','covered_value'):
            for noise in (False,True):
                with self.subTest(teacher=teacher,noise=noise), \
                     patch.object(recovery_search.core,'sample_world',side_effect=lambda s,v,seed,**kw:backend.sample_world(s,v,seed)), \
                     patch.object(recovery_search.core,'simulate_action',side_effect=backend.step), \
                     patch.object(recovery_search.core,'acting_side',side_effect=backend.acting_side), \
                     patch.object(recovery_search.core,'clean',side_effect=backend.clean), \
                     patch.object(recovery_search.core,'terminal_value',side_effect=backend.terminal_value), \
                     patch('app.modules.card_game.rl.covered_rollout.rule_decision',side_effect=backend.rule_decision), \
                     patch('app.modules.card_game.rl.covered_rollout.acting_side',side_effect=backend.acting_side), \
                     patch('app.modules.card_game.rl.covered_rollout.simulate_action',side_effect=backend.step), \
                     patch('app.modules.card_game.rl.covered_rollout.clean',side_effect=backend.clean), \
                     patch('app.modules.card_game.rl.covered_rollout.terminal_value',side_effect=backend.terminal_value), \
                     patch('app.modules.card_game.engine.duel_v2.acting_side',side_effect=backend.acting_side), \
                     patch('app.modules.card_game.engine.duel_v2.simulation.simulate_action',side_effect=backend.step):
                    # covered helpers import these canonical names dynamically.
                    common=dict(seed=12345,simulations=32,gumbel_candidates=2,terminal_horizon=3,
                                max_depth=2,noise=noise,q_scale='natural_wdl',teacher_mode=teacher)
                    expected=recovery_search.search(root,'a',{'a':'a','b':'b'},runtime=Runtime(),
                                                    value_policies={'a':'aux','b':'aux'},**common)
                    actual=drive_traversal(search_traversal(root,'a',{'a':('a','v1'),'b':('b','v1')},
                        backend=backend,value_policy_identities_by_side={'a':('aux-a','aux-v'),'b':('aux-b','aux-v')},**common),
                        response_provider=lambda query:(logits.copy() if not query.value_only else np.empty(0),wdl.copy()),limit_steps=5000)
                    self.assertTrue(actual['complete']);self.assertEqual(actual['search_choice'],expected['search_choice'])
                    np.testing.assert_array_equal(actual['visits'],expected['visits'])
                    np.testing.assert_allclose(actual['pi'],expected['pi'])
                    np.testing.assert_allclose(actual['mean_values'],expected['mean_values'])
                    self.assertEqual(actual['leaf_source'],expected['leaf_source'])

    def test_rule_and_leaf_calls_are_queued_in_cooperative_search(self):
        backend=TeacherBackend(terminal_depth=8)
        class Inference:
            def predict(self,requests):
                return [InferenceResult(q.request_id,q.model_key,q.model_version,
                    np.array([.5,-.5]) if not q.value_only else np.empty(0),np.array([.2,.2,.6])) for q in requests]
        class Bulk(TeacherBackend):
            backend_kind='fake_bulk';supports_gpu=False
            def __init__(self):super().__init__(terminal_depth=8);self.queued_rules=0
            def rule_decision(self,*args):raise AssertionError('Teacher escaped cooperative batching')
            def resolve_operations(self,queries):
                from app.modules.card_game.rl.batched_search.cooperative import OperationResponse
                result=[]
                for q in queries:
                    if q.method=='rule_decision':
                        self.queued_rules+=1;value=TeacherBackend.rule_decision(self,*q.args)
                    else:value=getattr(self,q.method)(*q.args)
                    result.append(OperationResponse(q.id,value))
                return result
        backend=Bulk();deadline=time.monotonic()+60
        jobs=[RootJob(str(i),str(i),1,cooperative_search({'phase':'playing','actor':'a','depth':0},'a',
            {'a':('a','v1'),'b':('b','v1')},backend=backend,seed=i,simulations=32,
            gumbel_candidates=2,max_depth=2,terminal_horizon=0,teacher_mode='covered_expert',
            deadline=deadline),deadline) for i in range(3)]
        result=CooperativeScheduler(Inference(),allow_cpu_oracle=True).run(jobs)
        self.assertTrue(all(row['complete'] for row in result['results'].values()))
        self.assertGreater(backend.queued_rules,0)

    def test_leaf_limit_and_invalid_auxiliary_never_guess_terminal(self):
        backend=TeacherBackend(terminal_depth=10000)
        root={'phase':'playing','actor':'a','depth':0}
        result=drive_traversal(_teacher_terminal_generator(root,'a',np.random.default_rng(9),backend=backend,
            query_factory=None,deadline=None,clock=time.monotonic,own_rule=True,q_scale='natural_wdl'))
        self.assertEqual(result,(None,False))
        for extra in (dict(teacher_mode='covered_value'),dict(teacher_mode='covered_expert',q_scale='legacy')):
            with self.assertRaises(ValueError):
                next(search_traversal(root,'a',{'a':('a','v'),'b':('b','v')},backend=backend,seed=1,**extra))


if __name__=='__main__':unittest.main()
