import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.teacher_validation import paired_summary,continuation,counterfactuals,value_diagnostics,verify_counterfactuals,sampled_build_support


class TeacherValidationTest(unittest.TestCase):
    def test_sampled_card_pool_audit_does_not_change_world_or_claim_capability(self):
        from types import SimpleNamespace
        world=dict(sides={'a':dict(hand=[dict(card_id='N01')],deck=[dict(card_id='N02')],discard=[])})
        original=deepcopy(world)
        policy=SimpleNamespace(serving_deck=dict(card_ids=['N01','N01']))
        report=sampled_build_support(world,{'a':policy})['a']
        self.assertFalse(report['compatible_remaining_pool'])
        self.assertEqual(report['excess_counts'],{'N02':1})
        self.assertFalse(report['policy_capability_proven'])
        self.assertEqual(world,original)
        policy.serving_deck['card_ids']=['N01','N02']
        self.assertTrue(sampled_build_support(world,{'a':policy})['a']['compatible_remaining_pool'])

    def test_value_probe_counts_paired_worlds_and_reports_opposite_order(self):
        record=dict(world_seed=1,branches={
            0:dict(complete=True,value=-1,predicted_value=.8,prediction_source='network'),
            1:dict(complete=True,value=1,predicted_value=-.2,prediction_source='network'),
            2:dict(complete=True,value=1,predicted_value=1,prediction_source='terminal')})
        incomplete=deepcopy(record);incomplete['branches'][1]['complete']=False
        result=value_diagnostics([record,incomplete],[0,1,2])
        self.assertEqual(result['complete_paired_worlds'],1)
        self.assertAlmostEqual(result['branches'][0]['mean_error'],1.8)
        self.assertAlmostEqual(result['branches'][0]['mean_squared_error'],3.24)
        self.assertEqual(result['branches'][2]['network_rows'],0)
        self.assertTrue(result['pairwise_order'][0]['opposite_sign'])
        self.assertFalse(result['calibration_approved'])
        broken=deepcopy(record);del broken['branches'][0]['prediction_source']
        with self.assertRaises(ValueError):value_diagnostics([broken],[0,1,2])

    def test_value_capture_uses_fixed_viewer_after_actor_switch(self):
        state={'phase':'playing'}
        class Runtime:
            def value_for(self,state,viewer,policies):
                self.viewer=viewer;return .6
            def decision(self,state,actor):return [{}],(np.zeros(1),np.zeros((1,1)))
            def action_scores(self,policy,x,c):return np.zeros(1)
        runtime=Runtime()
        terminal={'phase':'finished','winner':'a'}
        with patch('app.modules.card_game.rl.teacher_validation.apply_action',side_effect=[state,terminal]),\
             patch('app.modules.card_game.rl.teacher_validation.acting_side',return_value='b'):
            result=continuation(state,'a',{}, {'b':object()},runtime,
                seed=1,deadline=float('inf'),max_actions=2,capture_value=True)
        self.assertEqual(runtime.viewer,'a');self.assertTrue(result['complete'])
        self.assertEqual(result['value'],1.);self.assertEqual(result['predicted_value'],.6)
        self.assertEqual(result['prediction_source'],'network')

    def test_private_replay_rejects_wrong_branch_and_wrong_terminal(self):
        state={'phase':'playing'};terminal={'phase':'finished','winner':'a'}
        class Runtime:
            def decision(self,state,actor):return [dict(type='end_turn')],(np.zeros(1),np.zeros((1,1)))
        row=dict(world_seed=1,branches={0:dict(complete=True,value=1,winner='a',
            actions=[dict(side='a',action=dict(type='end_turn'))])})
        with patch('app.modules.card_game.rl.teacher_validation.core.sample_world',return_value=state),\
             patch('app.modules.card_game.rl.teacher_validation.acting_side',return_value='a'),\
             patch('app.modules.card_game.rl.teacher_validation.apply_action',return_value=terminal):
            result=verify_counterfactuals(state,'a',[row],Runtime(),deadline=float('inf'))
            self.assertEqual(result,dict(complete=True,verified_branches=1))
            predicted=deepcopy(row);predicted['branches'][0].update(predicted_value=1.,prediction_source='terminal')
            self.assertTrue(verify_counterfactuals(state,'a',[predicted],Runtime(),
                deadline=float('inf'),policies={})['complete'])
            predicted['branches'][0]['predicted_value']=-1.
            with self.assertRaisesRegex(ValueError,'prediction'):
                verify_counterfactuals(state,'a',[predicted],Runtime(),deadline=float('inf'),policies={})
            bad=deepcopy(row);bad['branches'][0]['winner']='b'
            with self.assertRaisesRegex(ValueError,'terminal'):
                verify_counterfactuals(state,'a',[bad],Runtime(),deadline=float('inf'))
            bad=deepcopy(row);bad['branches'][0]['actions'][0]['side']='b'
            with self.assertRaisesRegex(ValueError,'root action'):
                verify_counterfactuals(state,'a',[bad],Runtime(),deadline=float('inf'))
        self.assertFalse(verify_counterfactuals(state,'a',[row],Runtime(),deadline=0)['complete'])

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
