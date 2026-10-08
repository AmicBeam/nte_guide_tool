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
            if a['type']=='finish' and s.get('step',0)==0:s.update(phase='finished',winner='a')
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

    def test_first_round_is_gumbel_top_k_despite_live_q_completion(self):
        prior=np.array([.40,.20,.15,.10,.08,.07]);noise=np.array([-.2,.3,-.7,.6,.1,1.])
        expected=set(np.argsort(np.log(prior)+noise)[-3:]);node=Node.create(prior,.1)
        selected=[]
        for visit in visit_schedule(3,12):
            a=choose_root(node,noise,visit);selected.append(a)
            node.visits[a]+=1;node.total[a]+=(-1 if a%2 else 1)
        self.assertEqual(set(selected[:3]),expected)
        self.assertEqual(set(selected),expected)

    def test_short_terminal_overrides_a_wrong_value_head(self):
        class Later(Toy):
            @staticmethod
            def decision(s, side):
                step = s.get('step', 0)
                if step == 0:
                    return [{'type': 'stall'}, {'type': 'advance'}], (np.array([0.], np.float32), np.eye(2, dtype=np.float32))
                if step == 1:
                    return [{'type': 'finish'}], (np.array([1.], np.float32), np.ones((1, 2), np.float32))
                return [{'type': 'stall'}], (np.array([2.], np.float32), np.ones((1, 2), np.float32))

        class Biased:
            def scores_value(self, x, c):
                scores = np.zeros(len(c))
                if len(scores) > 1:
                    scores[0] = 1.
                return scores, 0.

        def act(s, side, a):
            s = deepcopy(s)
            if a['type'] == 'finish':
                s.update(phase='finished', winner='a')
            elif a['type'] == 'advance':
                s['step'] = 1
            else:
                s['step'] = 9
            return s

        with patch('app.modules.card_game.rl.information_search.apply_action', side_effect=act):
            blinded = search(dict(phase='playing', active_side='a', turn=1), 'a', {'a': Biased(), 'b': Biased()}, runtime=Later, seed=3,
                             simulations=2, fast_simulation=False, algorithm='gumbel', terminal_horizon=0)
            proved = search(dict(phase='playing', active_side='a', turn=1), 'a', {'a': Biased(), 'b': Biased()}, runtime=Later, seed=3,
                            simulations=2, fast_simulation=False, algorithm='gumbel', terminal_horizon=3)
        self.assertNotEqual(blinded['search_choice'], 1)
        self.assertEqual(proved['search_choice'], 1)
        self.assertGreater(proved['mean_values'][1], 0.5)

    def test_terminal_on_last_depth_is_backed_up_as_actual_win(self):
        def act(s,side,a):
            return {**s,'phase':'finished','winner':'a' if a['type']=='finish' else 'b'}
        with patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):
            r=search(dict(phase='playing',active_side='a',turn=1),'a',{'a':Prior(),'b':Prior()},runtime=Toy,seed=4,
                     simulations=8,max_depth=1,fast_simulation=False,algorithm='gumbel')
        np.testing.assert_array_equal(r['mean_values'],[-1,1])

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
