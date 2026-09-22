import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.information_search import search,Node
from app.modules.card_game.rl.search_policy import completed_q,visit_schedule,choose_root,improved_policy

class Prior:
    def scores_value(self,x,c):return np.array([5.,0.]),0.

class Toy:
    @staticmethod
    def decision(s,side):return [{'type':'wait'},{'type':'finish'}],(np.array([s.get('step',0)],np.float32),np.eye(2,dtype=np.float32))
    @staticmethod
    def determinize(s,viewer,seed):return {**s,'latent':seed%2}

class GumbelSearchTest(unittest.TestCase):
    def test_reference_completed_q_and_schedule(self):
        q=completed_q(np.array([.75,.25]),np.array([2,0]),np.array([2.,0.]),0.)
        np.testing.assert_allclose(q,[5.2,0.])
        self.assertEqual(visit_schedule(2,4),[0,0,1,1])
        self.assertEqual(visit_schedule(1,4),[0,1,2,3])

    def test_exact_budget_and_low_prior_winning_action(self):
        def act(s,side,a):
            s=deepcopy(s)
            if a['type']=='finish':s.update(phase='finished',winner='a')
            else:s['step']=s.get('step',0)+1
            return s
        with patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):
            r=search(dict(phase='playing',active_side='a',turn=1),'a',{'a':Prior(),'b':Prior()},runtime=Toy,seed=1,
                     simulations=16,fast_simulation=False,algorithm='gumbel')
        self.assertEqual(r['simulations'],16);self.assertEqual(r['requested'],16)
        self.assertEqual(r['search_choice'],1);self.assertGreater(r['pi'][1],r['pi'][0]);self.assertEqual(r['policy_source'],'gumbel_q')
        self.assertFalse(np.allclose(r['pi'],r['visit_pi']))

    def test_candidate_cap_does_not_expand_budget(self):
        node=Node.create(np.ones(26)/26,0.);schedule=visit_schedule(4,8)
        for v in schedule:
            a=choose_root(node,np.zeros(26),v);node.visits[a]+=1
        self.assertEqual(node.visits.sum(),8);self.assertEqual((node.visits>0).sum(),4)
        self.assertAlmostEqual(improved_policy(node).sum(),1.)

    def test_low_budget_selfplay_jobs_keep_algorithm_and_bilateral_labels(self):
        from pathlib import Path
        from types import SimpleNamespace
        from app.modules.card_game.rl.selfplay_trial import make_job,FOES
        base={k:SimpleNamespace(serving_deck={'id':k}) for k in ('zhenhong',*FOES)}
        config=dict(seeds={'foundation':100},simulations=32,train_until=1000,
                    search_algorithm='gumbel',gumbel_candidates=16,search_budgets={'gumbel16':16,'gumbel32':32})
        for arm,budget in config['search_budgets'].items():
            for number in range(4):
                job=make_job(arm,number,Path('/tmp/check'),config,base,'current','source','initial')
                self.assertEqual(job['simulations'],budget)
                self.assertEqual(job['search_algorithm'],'gumbel')
                self.assertEqual(job['train_sides'],['a','b'] if number<2 else ['a'])
                self.assertNotIn('search_mix',job)
