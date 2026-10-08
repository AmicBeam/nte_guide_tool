"""Serving policy parity, including same-model surrogate and shared proof budget."""
from copy import deepcopy
import unittest
import time
from unittest.mock import patch
import numpy as np
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game,acting_side
from app.modules.card_game.engine.ai import advanced_search
from app.modules.card_game.rl import recovery_runtime,recovery_policy
from app.modules.card_game.rl.batched_search.oracle_backend import CPUOracleBackend
from app.modules.card_game.rl.batched_search.cooperative import CooperativeScheduler,RootJob
from app.modules.card_game.rl.batched_search.serving import serving_search
from app.modules.card_game.rl.batched_duel.serving_protocol import root_metadata
from tests.test_duel_v2_batch_real_rules import PublicTestPolicy,NumpyOracleInference


class ServingParityTest(unittest.TestCase):
    def test_expired_queued_root_does_not_start_rules_or_inference(self):
        class ForbiddenInference:
            def predict(self,requests):raise AssertionError('Expired root ran inference')
        class ForbiddenBackend(CPUOracleBackend):
            def acting_side(self,state):raise AssertionError('Expired root ran rules')
        backend=ForbiddenBackend();queued=time.monotonic()-10
        model=PublicTestPolicy('starter')
        gen=serving_search(None,'a',model,backend=backend,budget_mode='wall_clock',queued_at=queued)
        result=CooperativeScheduler(ForbiddenInference(),allow_cpu_oracle=True).run([
            RootJob('r','g',1,gen,deadline=queued+3)])['results']['r']
        self.assertEqual(result['stop_reason'],'deadline')
        self.assertIsNone(result['selected_action']);self.assertEqual(result['total_simulations'],0)

    def test_late_public_win_is_not_counted_or_adopted(self):
        class Clock:
            value=0.
            def __call__(self):return self.value
        clock=Clock()
        class LateProbe(CPUOracleBackend):
            def public_terminal_probe(self,*args):
                result=super().public_terminal_probe(*args);clock.value=4.;return result
        state=new_game(seed=9281,first_side='a',skip_mulligan=True)
        foe=state['sides']['b'];foe['deck'].extend(foe['hand']);foe['hand']=[];foe['hp']=1;foe['shield']=0
        backend=LateProbe(runtime=recovery_runtime);model=PublicTestPolicy('starter')
        gen=serving_search(state,'a',model,backend=backend,budget_mode='wall_clock',
                           queued_at=0.,clock=clock)
        result=CooperativeScheduler(NumpyOracleInference({'starter':model}),allow_cpu_oracle=True).run([
            RootJob('r','g',1,gen)])['results']['r']
        self.assertEqual(result['stop_reason'],'deadline')
        self.assertEqual(result['proof_attempts'],1);self.assertEqual(result['proof_checks'],0)
        self.assertEqual(result['total_simulations'],0);self.assertTrue(result['used_fallback'])

    def run_case(self,state,key,model):
        viewer=acting_side(state);actions,(x,c)=recovery_runtime.decision(state,viewer)
        fallback=actions[int(np.argmax(model.scores_only(x,c)))]
        with patch.object(advanced_search,'SEARCH_SECONDS',1000.):
            expected=advanced_search.choose(state,viewer,model,fallback)
        backend=CPUOracleBackend(runtime=recovery_runtime)
        gen=serving_search(state,viewer,model,backend=backend,budget_mode='simulations',root_id=key)
        run=CooperativeScheduler(NumpyOracleInference({key:model}),allow_cpu_oracle=True).run([
            RootJob(key,'game-'+key,1,gen)])
        result=run['results'][key]
        self.assertEqual(result['selected_action'],expected)
        self.assertLessEqual(result['total_simulations'],32)
        self.assertLessEqual(result['proof_checks'],8)
        self.assertEqual(result['surrogate_model_key'],key)
        return result

    def test_real_recovery_serving_uses_its_own_model_for_both_search_sides(self):
        for i,key in ((0,'starter'),(3,'zhenhong'),(4,'murk')):
            with self.subTest(key=key):
                state=new_game(seed=9200+i,first_side='a',skip_mulligan=True,
                               decks={'a':STARTER_DECKS[i],'b':STARTER_DECKS[(i+1)%5]})
                model=PublicTestPolicy(key);model.schema=recovery_policy.RESIDUAL_SCHEMA
                before=deepcopy(state);result=self.run_case(state,key,model)
                self.assertEqual(state,before)
                self.assertEqual(result['tree_simulations']+result['proof_checks'],result['total_simulations'])

    def test_public_probe_wins_and_never_reads_unknown_hand_identities(self):
        state=new_game(seed=9211,first_side='a',skip_mulligan=True)
        changed=deepcopy(state);changed['sides']['b']['hand'][0]['card_id']='I03'
        self.assertEqual(root_metadata(state,'a'),root_metadata(changed,'a'))
        foe=state['sides']['b'];foe['deck'].extend(foe['hand']);foe['hand']=[];foe['hp']=1;foe['shield']=0
        model=PublicTestPolicy('starter');model.schema=recovery_policy.RESIDUAL_SCHEMA
        result=self.run_case(state,'starter',model)
        self.assertEqual(result['stop_reason'],'public_terminal_win')
        self.assertEqual(result['tree_simulations'],0)
        self.assertGreater(result['proof_checks'],0)


if __name__=='__main__':unittest.main()
