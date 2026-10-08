import unittest
import numpy as np
from app.modules.card_game.rl.information_search import Node,search_prior,search
class ExplorationTest(unittest.TestCase):
    def test_floor_is_round_robin_despite_sharp_prior(self):
        n=Node.create(np.array([.999999,.0000005,.0000005]))
        for _ in range(12):
            i=n.choose(4);n.visits[i]+=1;n.total[i]+=100 if i==0 else -100
        np.testing.assert_array_equal(n.visits,[4,4,4])
    def test_uniform_and_tempered_do_not_change_default_prior(self):
        x=np.array([72.,59.,0.]);p=search_prior(x)
        self.assertGreater(p[0],.999)
        np.testing.assert_allclose(search_prior(x,'uniform'),[1/3]*3)
        mixed=search_prior(x,'tempered',4,.25)
        self.assertTrue((mixed>=.25/3).all());self.assertAlmostEqual(mixed.sum(),1)
    def test_real_search_respects_budget_floor_and_hidden_information(self):
        from tests.test_duel_v2_information_search import InformationSearchTest,Uniform
        from copy import deepcopy
        s=InformationSearchTest().game();changed=deepcopy(s);changed['sides']['b']['deck'].reverse()
        policies={'a':Uniform(),'b':Uniform()}
        opts=dict(seed=83,simulations=128,prior_mode='uniform',root_min_visits=4,node_min_visits=2)
        a=search(s,'a',policies,**opts);b=search(changed,'a',policies,**opts)
        self.assertTrue(a['complete']);self.assertGreaterEqual(a['root_min_visits'],4)
        np.testing.assert_array_equal(a['visits'],b['visits'])
        with self.assertRaises(ValueError):search(s,'a',policies,seed=1,simulations=8,root_min_visits=999)
