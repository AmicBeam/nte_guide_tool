import importlib.util
import unittest
import numpy as np
from app.modules.card_game.rl.league_replay import ErrorReplay,OptimizerBudget

class ReplayTest(unittest.TestCase):
    def test_actual_optimizer_budget_never_grants_unearned_ordinary_steps(self):
        b=OptimizerBudget()
        self.assertFalse(b.can_update_ordinary())
        for i in range(40):
            b.record_branch(1)
            if b.can_update_ordinary():b.record_ordinary(1)
            self.assertGreaterEqual(b.branch_steps,3*b.ordinary_steps)
        self.assertEqual(b.ordinary_steps,13)
        with self.assertRaises(ValueError):b.record_ordinary(1)
        with self.assertRaises(ValueError):b.record_ordinary(2)
    def test_large_mistakes_are_prioritized_and_rechecked(self):
        r=ErrorReplay();c=np.eye(3,dtype=np.float32)
        r.add([0],c,[0,1],[0,10],0);r.add([1],c,[0,1],[0,5],0)
        bad=lambda x,c:np.array([5.,0.,-1.])
        self.assertEqual(r.take(bad,0)['x'][0],0)
        self.assertEqual(r.take(bad,0)['x'][0],1)
        self.assertIsNone(r.take(bad,0))
        self.assertIsNone(r.take(lambda x,c:np.array([0.,5.,-1.]),1))
        self.assertEqual(r.take(bad,1)['uses'],2)
        self.assertEqual(r.take(bad,2)['uses'],3)
        self.assertEqual(r.take(bad,3)['x'][0],1)
        self.assertIsNone(r.take(bad,6))
    def test_unknown_actions_ties_expiry_capacity_and_clear(self):
        c=np.eye(3,dtype=np.float32);r=ErrorReplay(capacity=2,ttl=1)
        self.assertFalse(r.add([0],c,[0,1],[4,4],0))
        self.assertFalse(r.add([0],c,[0,1],[4,5],0))
        for i in range(3):r.add([i],c,[0,1],[0,10],i)
        self.assertEqual(len(r.entries),2)
        self.assertIsNone(r.take(lambda x,c:np.array([0.,0.,10.]),2))
        self.assertIsNone(r.take(lambda x,c:np.array([10.,0.,0.]),4))
        r.add([3],c,[0,1],[0,10],5);r.clear();self.assertFalse(r.entries)

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch verification runs on Windows')
class GradientTest(unittest.TestCase):
    def test_long_ordinary_batch_performs_exactly_one_optimizer_step(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.league_learning import ordinary_update
        net=CompactScorer(3,3,8);opt=torch.optim.SGD(net.parameters(),lr=.01)
        calls=[];step=opt.step
        def counted(*a,**kw):calls.append(1);return step(*a,**kw)
        opt.step=counted
        x=np.ones(3,dtype=np.float32);c=np.eye(3,dtype=np.float32)
        games=[{'complete':True,'reward':10,'trajectory':[(x,c,0,-1.1,0)]*600}]
        r=ordinary_update(net,opt,games,11)
        self.assertEqual(len(calls),1);self.assertEqual(r['optimizer_steps'],1);self.assertEqual(r['samples'],512)
    def test_mistaken_anchor_releases_constraint_and_gradient_corrects_choice(self):
        import torch
        from app.modules.card_game.rl.league_learning import branch_update
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([2.,0.]));self.v=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x,c,mask):return (c*self.w).sum(-1).masked_fill(~mask,-1e9),self.v.expand(x.shape[0])
        net=Tiny();opt=torch.optim.SGD(net.parameters(),lr=1)
        x=np.zeros(1,dtype=np.float32);c=np.eye(2,dtype=np.float32)
        for _ in range(12):
            row=branch_update(net,opt,x,c,[-10,10],anchor_logits=np.array([10.,-10.]),all_candidates=c,selected_indices=[0,1])
            self.assertTrue(row['anchor_disabled_for_error']);self.assertEqual(row['anchor_kl'],0)
        self.assertEqual(int(net.w.argmax()),1)
        row=branch_update(net,opt,x,c,[-10,10],anchor_logits=np.array([-10.,10.]),all_candidates=c,selected_indices=[0,1])
        self.assertFalse(row['anchor_disabled_for_error'])
    def test_unassessed_anchor_is_not_declared_wrong(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.league_learning import branch_update
        net=CompactScorer(3,3,8);opt=torch.optim.SGD(net.parameters(),lr=.01);c=np.eye(3,dtype=np.float32)
        row=branch_update(net,opt,np.zeros(3,dtype=np.float32),c[:2],[-10,10],anchor_logits=np.array([0.,0.,10.]),all_candidates=c,selected_indices=[0,1])
        self.assertFalse(row['anchor_disabled_for_error']);self.assertGreaterEqual(row['anchor_kl'],0)


class WeakPreferenceTest(unittest.TestCase):
    def pairs(self, left, right, lethal=()):
        from app.modules.card_game.rl.league_preferences import auxiliary_pairs
        rows=[dict(complete=True,reward=q,remaining_turns=t,own_decisions=d) for q,t,d in left+right]
        return auxiliary_pairs(rows,2,len(left),lethal)

    def test_wins_faster_losses_slower(self):
        self.assertEqual(self.pairs([(10,1,2)],[(10,2,1)]),[(0,1,.05,'winning_speed')])
        self.assertEqual(self.pairs([(-10,1,2)],[(-10,2,1)]),[(1,0,.02,'losing_delay')])
        self.assertFalse(self.pairs([(-10,2,8)],[(-10,2,1)]))
        self.assertFalse(self.pairs([(0,1,1)],[(0,8,9)]))

    def test_win_rate_and_paired_outcomes_take_priority(self):
        self.assertFalse(self.pairs([(10,9,9)],[(-10,1,1)]))
        self.assertFalse(self.pairs([(10,1,1),(-10,9,9)],[(-10,8,8),(10,2,2)]))
        # Faster wins but also faster defeats: no unconditional dominance.
        self.assertFalse(self.pairs([(10,1,1),(-10,1,1)],[(10,2,2),(-10,2,2)]))
        self.assertEqual(self.pairs([(10,1,1),(-10,3,2)],[(10,2,2),(-10,2,2)]),[(0,1,.05,'winning_speed')])

    def test_same_turn_fewer_winning_decisions_and_certain_lethal(self):
        self.assertEqual(self.pairs([(10,0,1)],[(10,0,3)]),[(0,1,.05,'winning_speed')])
        self.assertEqual(self.pairs([(10,0,1)],[(10,0,3)],{0}),[(0,1,.2,'certain_lethal')])
        from app.modules.card_game.rl.league_preferences import auxiliary_pairs
        self.assertFalse(auxiliary_pairs([{'complete':False}],1,1))

    def test_weak_replay_bypasses_gap_and_rechecks_current_choice(self):
        r=ErrorReplay();c=np.eye(2,dtype=np.float32)
        self.assertTrue(r.add([0],c,[0,1],[-10,-10],0,[(1,0,.02,'losing_delay')]))
        row=r.take(lambda x,c:np.array([2.,0.]),0)
        self.assertEqual(row['auxiliary'],((1,0,.02,'losing_delay'),))
        self.assertIsNone(r.take(lambda x,c:np.array([0.,2.]),1))
        with self.assertRaises(ValueError):r.add([0],c,[0,1],[10,-10],0,[(1,0,.02,'losing_delay')])

    def test_certification_is_hidden_information_independent_and_fail_closed(self):
        from copy import deepcopy
        from app.modules.card_game.rl.league_preferences import certified_lethals
        from app.modules.card_game.rl.league_schema import deck
        from app.modules.card_game.rl.league_rollout import decision
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import card_instance
        s=new_game(seed=99113,first_side='a',skip_mulligan=True,decks={'a':deck('quick-rush'),'b':deck('starter')})
        s['sides']['b']['hp']=1;s['sides']['b']['shield']=0
        s['sides']['a']['hand']=[card_instance(s,'U03','a')]
        actions,_=decision(s,'a')
        self.assertFalse(certified_lethals(s,'a',actions))
        changed=deepcopy(s);changed['sides']['b']['hand'][0]['card_id']='N02'
        changed['rng']=8877
        self.assertFalse(certified_lethals(changed,'a',actions))
        s['sides']['b']['hand']=[]
        before=deepcopy(s);lethal=certified_lethals(s,'a',actions)
        self.assertTrue(lethal);self.assertEqual(s,before)
        changed=deepcopy(s);changed['rng']=123;changed['sides']['b']['deck'].reverse();changed['sides']['a']['deck'].reverse()
        self.assertEqual(lethal,certified_lethals(changed,'a',actions))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch verification runs on Windows')
    def test_auxiliary_gradient_preserves_value_and_releases_wrong_anchor(self):
        import torch
        from app.modules.card_game.rl.league_learning import branch_update
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([1.,0.]));self.v=torch.nn.Parameter(torch.tensor(3.))
            def forward(self,x,c,mask):return (c*self.w).sum(-1),self.v.expand(x.shape[0])
        for q,reason,weight in [(10,'winning_speed',.05),(-10,'losing_delay',.02),(10,'certain_lethal',.2)]:
            net=Tiny();opt=torch.optim.SGD(net.parameters(),lr=1)
            c=np.eye(2,dtype=np.float32);x=np.zeros(1,dtype=np.float32)
            row=branch_update(net,opt,x,c,[q,q],anchor_logits=np.array([2.,0.]),all_candidates=c,auxiliary=[(1,0,weight,reason)])
            self.assertTrue(row['anchor_disabled_for_error'])
            self.assertLess(float(net.w[0].detach()),1);self.assertGreater(float(net.w[1].detach()),0)
            self.assertEqual(float(net.v.detach()),3)
            # Absolute .02/.05/.2 remain small even with one pair.
            self.assertLessEqual(1-float(net.w[0].detach()),weight)

    def test_forced_root_lethal_counts_one_decision_and_zero_remaining_turns(self):
        import time
        from copy import deepcopy
        from unittest.mock import patch
        from app.modules.card_game.rl.league_rollout import play,decision
        from app.modules.card_game.rl.league_schema import deck
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import card_instance
        s=new_game(seed=99114,first_side='a',skip_mulligan=True,decks={'a':deck('quick-rush'),'b':deck('starter')})
        s['sides']['b']['hp']=1;s['sides']['b']['shield']=0;s['sides']['b']['hand']=[]
        s['sides']['a']['hand']=[card_instance(s,'U03','a')]
        a=next(a for a in decision(s,'a')[0] if a['type']=='play_card')
        with patch('app.modules.card_game.rl.league_rollout.determinize',side_effect=lambda s,*args:deepcopy(s)):
            g=play(dict(root=s,viewer='a',action=a,seed=1,deadline=time.time()+5,policies={},max_actions=0))
        self.assertTrue(g['complete']);self.assertEqual(g['reward'],10)
        self.assertEqual(g['remaining_turns'],0);self.assertEqual(g['own_decisions'],1)

    def test_certification_rejects_random_and_hidden_deck_dependent_results(self):
        from unittest.mock import patch
        from copy import deepcopy
        from app.modules.card_game.rl.league_preferences import certified_lethals
        from app.modules.card_game.rl.league_schema import deck
        from app.modules.card_game.engine.duel_v2 import new_game
        s=new_game(seed=99115,first_side='a',skip_mulligan=True,decks={'a':deck('starter'),'b':deck('starter')})
        s['sides']['b']['hand']=[]
        for change in ('rng','deck','empty_draw'):
            def unsafe(state,side,action):
                t=deepcopy(state);t['phase']='finished';t['winner']=side
                if change=='rng':t['rng']+=1
                elif change=='deck':t['sides']['a']['deck'].pop()
                else:t['events'].append({'type':'finish','reason':'empty_draw_win'})
                return t
            with patch('app.modules.card_game.engine.duel_v2.apply_action',side_effect=unsafe):
                self.assertFalse(certified_lethals(s,'a',[{'type':'attack','character_id':'zero'}]))


class LeagueBudgetTest(unittest.TestCase):
    def test_all_six_phases_fit_budget_with_evaluation_reserved(self):
        from scripts.train_duel_v2_league import phase_budget
        for seconds in (600,1800,5400):
            phase,reserve=phase_budget(seconds)
            self.assertLessEqual(6*phase+reserve+60,seconds)
        self.assertEqual(phase_budget(1800),(190,600))
        self.assertEqual(phase_budget(5400),(720,900))
        with self.assertRaises(ValueError):phase_budget(599)
