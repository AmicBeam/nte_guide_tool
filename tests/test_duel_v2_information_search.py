import importlib.util
import time
import unittest
from copy import deepcopy
import numpy as np
from app.modules.card_game.rl.information_search import search,sample_world,terminal_value,leaf_value
from app.modules.card_game.rl.league_schema import deck
from app.modules.card_game.rl.league_rollout import decision
from app.modules.card_game.engine.duel_v2 import new_game

class Uniform:
    def scores_value(self,x,c):return np.zeros(len(c)),0.
    def scores(self,x,c):return np.zeros(len(c))

class InformationSearchTest(unittest.TestCase):
    def game(self,opening=False):
        return new_game(seed=983001,first_side='a',skip_mulligan=not opening,
                        decks={'a':deck('quick-rush'),'b':deck('starter')})

    def test_mulligan_sampling_preserves_visible_actions(self):
        s=self.game(True);out=sample_world(s,'a',33)
        self.assertEqual(out['phase'],'mulligan')
        a,(x,c)=decision(s,'a');b,(xx,cc)=decision(out,'a')
        self.assertEqual(a,b);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)

    def test_hidden_information_cannot_change_search(self):
        s=self.game();changed=deepcopy(s)
        changed['rng']=987123;changed['sides']['a']['deck'].reverse();changed['sides']['b']['deck'].reverse()
        for c in changed['sides']['b']['hand']+changed['sides']['b']['deck']:
            c['card_id']='N02';c['instance_id']='b-9999';c['entity_id']='e9999'
        policies={'a':Uniform(),'b':Uniform()}
        x=search(s,'a',policies,seed=9,simulations=12)
        y=search(changed,'a',policies,seed=9,simulations=12)
        np.testing.assert_array_equal(x['visits'],y['visits'])
        np.testing.assert_array_equal(x['pi'],y['pi'])
        self.assertEqual(x['roots_covered'],len(x['actions']));self.assertTrue(x['complete'])

    def test_search_finds_lethal_and_records_visit_distribution(self):
        s=self.game();s['sides']['b']['hp']=1;s['sides']['b']['shield']=0
        r=search(s,'a',{'a':Uniform(),'b':Uniform()},seed=5,simulations=32)
        self.assertTrue(r['complete']);self.assertAlmostEqual(float(r['pi'].sum()),1,places=6)
        from app.modules.card_game.engine.duel_v2 import apply_action
        after=apply_action(s,'a',r['actions'][r['search_choice']])
        self.assertEqual(after.get('winner'),'a')
        np.testing.assert_allclose(r['pi'],r['visits']/r['visits'].sum())

    def test_value_sign_tracks_player_not_action_parity_and_truncation(self):
        class Constant(Uniform):
            def scores_value(self,x,c):return np.zeros(len(c)),5.
        s=self.game();policies={'a':Constant(),'b':Constant()}
        self.assertGreater(leaf_value(s,'a',policies),0)
        self.assertLess(leaf_value(s,'b',policies),0)
        s['phase']='finished';s['winner']='b'
        self.assertEqual(terminal_value(s,'a'),-1);self.assertEqual(terminal_value(s,'b'),1)
        s['winner']='draw';self.assertEqual(terminal_value(s,'a'),0)
        r=search(self.game(),'a',policies,seed=1,deadline=time.time()-1)
        self.assertFalse(r['complete']);self.assertEqual(r['simulations'],0)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
    def test_policy_and_terminal_value_both_train(self):
        import torch
        from app.modules.card_game.rl.information_search import policy_value_update
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([2.,0.]));self.v=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x,c,m):return (c*self.w).sum(-1),self.v.expand(len(x))
        net=Tiny();opt=torch.optim.SGD(net.parameters(),lr=1)
        row=dict(x=np.zeros(1,dtype=np.float32),c=np.eye(2,dtype=np.float32),pi=np.array([.1,.9]),z=1)
        for _ in range(20):policy_value_update(net,opt,[row])
        self.assertGreater(float(net.w[1].detach()),float(net.w[0].detach()))
        self.assertGreater(float(net.v.detach()),0)
        with self.assertRaises(ValueError):policy_value_update(net,opt,[dict(row,z=.5)])

    def test_complete_game_labels_every_actors_search_and_excludes_incomplete(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.information_search import search_game
        tiny={'phase':'playing','active_side':'a','turn':1}
        def act(s,side,action):
            n=dict(s);n['turn']+=1
            if n['turn']>=4:n.update(phase='finished',winner='a')
            elif n['turn']==3:n['active_side']='b'
            return n
        def searched(s,side,policies,**kw):
            return dict(complete=True,actions=[{'type':'test'}],pi=np.array([1.],dtype=np.float32),
                        x=np.zeros(1),c=np.zeros((1,1)),search_choice=0,raw_choice=0,roots_covered=1,nodes=1,simulations=1,value=0.)
        job=dict(seed=1,deadline=time.time()+10,decks={},policies={'a':('none','starter'),'b':('none','weave-rush')},collect=True)
        with patch('app.modules.card_game.rl.information_search.new_game',return_value=tiny),patch('app.modules.card_game.rl.information_search.model',return_value=Uniform()),patch('app.modules.card_game.rl.information_search.search',side_effect=searched),patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):
            result=search_game(job)
            self.assertEqual([r['side'] for r in result['rows']],['a','a','b'])
            self.assertEqual([r['z'] for r in result['rows']],[1,1,-1])
            self.assertEqual(result['searched'],result['decisions'])
            result=search_game(dict(job,max_actions=1))
            self.assertFalse(result['complete']);self.assertEqual(result['rows'],[])

    def test_both_opening_sides_work_even_when_not_the_first_player(self):
        from app.modules.card_game.engine.duel_v2 import apply_action,acting_side
        for first in ('a','b'):
            s=new_game(seed=983002,first_side=first,decks={'a':deck('starter'),'b':deck('weave-rush')})
            for side in ('a','b'):
                self.assertEqual(acting_side(s),side)
                out=sample_world(s,side,77)
                self.assertEqual(out['active_side'],s['active_side'])
                self.assertEqual(decision(out,side)[0],decision(s,side)[0])
                s=apply_action(s,side,decision(s,side)[0][0])
