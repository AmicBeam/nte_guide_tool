import unittest
import numpy as np
from app.modules.card_game.rl.rule_foundation import imitation_target,split_training_seeds
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
from app.modules.card_game.rl import cross_grounded as grounded
from app.modules.card_game.rl.cross_lineup import all_features,SEAT_INDEX,CAND_DIM
from app.modules.card_game.rl.league_schema import ACT_ATTACK,ACT_PLAY


class RuleFoundationTest(unittest.TestCase):
    def test_public_rank_ties_do_not_encode_incidental_physical_order(self):
        actions=[dict(type='play_card',card_id='a'),dict(type='play_card',card_id='b'),dict(type='end_turn')]
        view=dict(viewer_side='a',sides={'a':dict(hand=[dict(instance_id='a',type='tactic'),dict(instance_id='b',type='form')])},
                  legal_actions=[dict(action=a,preview={}) for a in actions])
        np.testing.assert_array_equal(imitation_target(view,actions),[.5,.5,0])
        self.assertEqual(VisibleEngineRulePolicy()(view,tuple(actions)),0)
        swapped=[actions[1],actions[0],actions[2]]
        np.testing.assert_array_equal(imitation_target(view,swapped),[.5,.5,0])
        self.assertEqual(VisibleEngineRulePolicy()(view,tuple(swapped)),0)

    def test_public_battle_ranking_and_rule_choice_are_unchanged(self):
        actions=[dict(type='attack',character_id='zhenhong'),dict(type='ultimate',character_id='yi'),dict(type='end_turn')]
        view=dict(viewer_side='a',sides={'a':dict(hand=[])},legal_actions=[
            dict(action=actions[0],preview=dict(attack=4,counter=1)),dict(action=actions[1],preview={}),dict(action=actions[2],preview={})])
        scores=VisibleEngineRulePolicy().ranking_scores(view,tuple(actions))
        self.assertEqual(scores,[6.6,6,-100])
        self.assertEqual(VisibleEngineRulePolicy()(view,tuple(actions)),0)
        np.testing.assert_array_equal(imitation_target(view,actions),[1,0,0])

    def test_internal_split_balances_matchups_and_keeps_initiatives_together(self):
        jobs=[dict(seed=100+match*20+i,first=f,keys={'a':'zhenhong','b':foe})
              for match,foe in enumerate(('murk','starter')) for i in range(10) for f in ('a','b')]
        held=split_training_seeds(jobs,79)
        self.assertEqual(len(held),4)
        for match in range(2):self.assertEqual(sum(100+match*20+i in held for i in range(10)),2)
        self.assertEqual(held,split_training_seeds(jobs,79))

    def test_automatic_combat_gets_visible_front_context_without_mutating_raw_identity(self):
        import torch
        x=np.zeros(len(all_features()),dtype=np.float32)
        x[grounded.FRONT_INDEX]=SEAT_INDEX['zhenhong']/10
        x[all_features().index('ch_hp:1:zhenhong')]=.4
        c=np.zeros((3,CAND_DIM),dtype=np.float32);c[:,1:5]=-1
        c[0,0]=ACT_ATTACK;c[0,1]=SEAT_INDEX['yi']
        c[1,0]=ACT_ATTACK;c[1,3]=0;c[1,4]=SEAT_INDEX['yi']
        c[2,0]=ACT_PLAY;c[2,2]=0
        old=c.copy();s,t=grounded.context_targets_numpy(x,c)
        self.assertEqual((s[0],t[0]),(1,SEAT_INDEX['zhenhong']))
        self.assertEqual((s[1],t[1]),(0,SEAT_INDEX['yi']))
        np.testing.assert_array_equal(c,old)
        ss,tt=grounded.context_targets_tensor(torch.tensor(x)[None],torch.tensor(c)[None])
        np.testing.assert_array_equal(ss.numpy()[0],s);np.testing.assert_array_equal(tt.numpy()[0],t)
        features=grounded.transform(x,c)
        self.assertEqual(features[0,-len(grounded.FIELDS)+grounded.FIELDS.index('ch_hp')],.4)
        x[grounded.FRONT_INDEX]=-.1
        self.assertEqual(grounded.context_targets_numpy(x,c)[1][0],-1)


if __name__=='__main__':unittest.main()
