import unittest
from copy import deepcopy
import numpy as np
from app.modules.card_game.rl import ten_search_runtime as runtime
from app.modules.card_game.rl.information_search import search, sample_world
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action

class Uniform:
    def scores_value(self,x,c): return np.zeros(len(c)),0.
    def scores(self,x,c): return np.zeros(len(c))

class TenSearchTest(unittest.TestCase):
    def test_ten_player_search_preserves_observation_and_hidden_privacy(self):
        builds={d['id']:d for d in STARTER_DECKS}
        for key in ['zhenhong','quick-rush','weave-rush','starter']:
            state=new_game(seed=1441,first_side='b',decks={'a':builds[key],'b':builds['zhenhong']})
            for side in ['a','b']:
                legal,(x,c)=runtime.decision(state,side)
                sampled=sample_world(state,side,19,runtime=runtime)
                after,(xx,cc)=runtime.decision(sampled,side)
                self.assertEqual(legal,after)
                np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)
                result=search(state,side,{'a':Uniform(),'b':Uniform()},seed=98,simulations=8,runtime=runtime)
                self.assertTrue(result['complete'])
                self.assertEqual(result['roots_covered'],len(legal))
                state=apply_action(state,side,legal[0])
            changed=deepcopy(state)
            for card in changed['sides']['a']['hand']+changed['sides']['a']['deck']:
                card.update(card_id='N02',instance_id='a-99999',entity_id='e99999')
            a=search(state,'b',{'a':Uniform(),'b':Uniform()},seed=55,simulations=8,runtime=runtime)
            b=search(changed,'b',{'a':Uniform(),'b':Uniform()},seed=55,simulations=8,runtime=runtime)
            np.testing.assert_array_equal(a['visits'],b['visits'])

    def test_passive_reward_does_not_replace_terminal_label(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.information_search import search_game
        import time
        s={'phase':'playing','active_side':'a','turn':1,'sides':{'a':{'characters':{'zhenhong':{'surplus_passive_triggers':1}}},'b':{'characters':{}}}}
        def act(state,side,action):
            after=deepcopy(state);after.update(phase='finished',winner='a')
            after['sides']['a']['characters']['zhenhong']['surplus_passive_triggers']=3
            return after
        searched=dict(complete=True,actions=[{'type':'test'}],pi=np.array([1.],np.float32),x=np.zeros(1),c=np.zeros((1,1)),search_choice=0,raw_choice=0,roots_covered=1,nodes=1,simulations=1,value=0.)
        job=dict(runtime='ten',seed=1,deadline=time.time()+5,decks={},policies={'a':('unused','zhenhong'),'b':('unused','starter')},training=True,collect=True,zhenhong_passive_reward=.2,train_sides=['a'])
        with patch('app.modules.card_game.rl.information_search.new_game',return_value=s),patch.object(runtime,'model',return_value=Uniform()),patch('app.modules.card_game.rl.information_search.search',return_value=searched),patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):
            result=search_game(job)
        self.assertEqual(result['rows'][0]['z'],1)
        self.assertAlmostEqual(result['rows'][0]['reward_to_go'],10.4)
        self.assertEqual(result['passive_triggers']['a'],3)
