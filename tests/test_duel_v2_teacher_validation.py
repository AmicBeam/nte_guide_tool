import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.teacher_validation import paired_summary,continuation,counterfactuals


class TeacherValidationTest(unittest.TestCase):
    def test_paired_endpoint_counts_worlds_and_excludes_partial_clusters(self):
        def row(seed,a,b,complete=True):
            return dict(world_seed=seed,branches={0:dict(complete=True,value=a),
                1:dict(complete=complete,value=b)})
        records=[row(1,-1,1),row(2,1,-1),row(3,0,1),row(4,-1,1,False)]
        result=paired_summary(records,[0,1],0,expected_worlds=4)
        self.assertFalse(result['complete'])
        self.assertEqual(result['complete_paired_worlds'],3)
        self.assertAlmostEqual(result['branches'][1]['paired_gain'],1/3)
        self.assertEqual(result['branches'][0]['paired_interval'],[0.,0.])
        self.assertFalse(result['quality_approved'])
        self.assertFalse(result['independent_strength_confirmation'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            paired_summary([records[0],records[0]],[0,1],0,expected_worlds=2)
        with self.assertRaisesRegex(ValueError,'Missing'):
            paired_summary([dict(world_seed=1,branches={0:dict(complete=True,value=1)})],
                           [0,1],0,expected_worlds=1)
        with self.assertRaisesRegex(ValueError,'real terminal'):
            paired_summary([row(1,.5,1)],[0,1],0,expected_worlds=1)

    def test_real_world_branches_preserve_root_and_use_paired_randomness(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl import recovery_runtime as rt
        from scripts.train_duel_v2_current_round import preset_decks
        decks=preset_decks()
        state=new_game(seed=5,first_side='a',skip_mulligan=True,
                       decks={'a':decks['zhenhong'],'b':decks['murk']})
        original=deepcopy(state);seen=[]
        def branch(world,viewer,action,policies,runtime,**kwargs):
            seen.append((deepcopy(world),kwargs['seed'],action))
            return dict(complete=True,value=1,actions=[])
        with patch('app.modules.card_game.rl.teacher_validation.continuation',side_effect=branch):
            records=counterfactuals(state,'a',[0,1],{},rt,world_seed=1000,
                continuation_seed=2000,worlds=2,deadline=float('inf'))
        self.assertEqual(state,original)
        self.assertEqual(seen[0][0],seen[1][0])
        self.assertEqual(seen[0][1],seen[1][1])
        self.assertEqual([r['world_seed'] for r in records],[1000,1001])
        self.assertTrue(paired_summary(records,[0,1],0,expected_worlds=2)['complete'])
        legal,_=rt.decision(state,'a')
        result=continuation(state,'a',legal[0],{},rt,seed=1,deadline=0)
        self.assertFalse(result['complete']);self.assertEqual(result['actions'],[])
        self.assertEqual(state,original)

    def test_action_limit_is_incomplete_not_value_estimate(self):
        state={'phase':'playing'}
        class Runtime:
            def decision(self,state,actor):return [{}],(np.zeros(1),np.zeros((1,1)))
            def action_scores(self,policy,x,c):return np.zeros(1)
        with patch('app.modules.card_game.rl.teacher_validation.apply_action',return_value=state),\
             patch('app.modules.card_game.rl.teacher_validation.acting_side',return_value='a'):
            result=continuation(state,'a',{}, {'a':object()},Runtime(),
                seed=1,deadline=float('inf'),max_actions=2)
        self.assertFalse(result['complete']);self.assertEqual(result['reason'],'action_limit')
        self.assertNotIn('value',result)


if __name__=='__main__':unittest.main()
