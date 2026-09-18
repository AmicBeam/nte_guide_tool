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
