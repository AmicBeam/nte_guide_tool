import time
import unittest
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.guided_recovery import MODE, apply_guide


class GuidedRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.actions=[{'type':'end_turn'},{'type':'mulligan','card_ids':[]}]
        self.state={'version':3}
        self.job=dict(seed=10,deadline=time.time()+30,guided_policy=MODE,
                      guide_coefficient=.5,train_sides=['a'],
                      policies={'a':('models','zhenhong'),'b':('models','starter')})

    def test_changed_action_has_auditable_mixed_policy_target(self):
        original=np.array([.9,.1],dtype=np.float32)
        with patch('app.modules.card_game.engine.ai.zhenhong_guide.recommend',return_value={
                'selected':self.actions[1],'reason':'mulligan_preserve_setup'}):
            chosen,label,audit=apply_guide(self.state,'a',self.actions,0,self.job,None,pi=original)
        self.assertEqual(chosen,1)
        np.testing.assert_allclose(label['pi'],[.45,.55])
        np.testing.assert_array_equal(label['original_search_pi'],original)
        np.testing.assert_array_equal(label['guide_pi'],[0,1])
        self.assertTrue(audit['changed'])
        self.assertEqual(label['policy_source'],MODE)
        self.assertNotIn('z',label)
        self.assertNotIn('setup_reward',label)

    def test_frozen_opponent_disabled_and_unchanged_keep_original_target(self):
        with patch('app.modules.card_game.engine.ai.zhenhong_guide.recommend') as guide:
            for actor,job in [('b',self.job),('a',{**self.job,'guided_policy':None})]:
                self.assertEqual(apply_guide(self.state,actor,self.actions,0,job,None), (0,{},None))
            guide.assert_not_called()
            guide.return_value={'selected':self.actions[0],'reason':'preserve_winning_action'}
            chosen,label,audit=apply_guide(self.state,'a',self.actions,0,self.job,None,pi=[.9,.1])
            self.assertEqual(chosen,0);self.assertEqual(label,{})
            self.assertFalse(audit['changed'])

    def test_illegal_guide_fails_before_applying_or_labelling(self):
        with patch('app.modules.card_game.engine.ai.zhenhong_guide.recommend',return_value={
                'selected':{'type':'concede'},'reason':'bad'}):
            with self.assertRaisesRegex(ValueError,'legal root'):
                apply_guide(self.state,'a',self.actions,0,self.job,None,pi=[.9,.1])

    def test_real_joint_update_optimizes_effective_policy_without_shaping_wdl(self):
        import torch
        from app.modules.card_game.rl import residual_policy,recovery_policy,recovery_runtime as rt
        from app.modules.card_game.engine.duel_v2 import new_game
        from scripts.train_duel_v2_current_round import preset_decks
        torch.manual_seed(4);torch.set_num_threads(1)
        net=recovery_policy.attach(residual_policy.network(16,'cpu'))
        decks=preset_decks();state=new_game(seed=98,first_side='a',decks={'a':decks['zhenhong'],'b':decks['starter']})
        actions,(x,c)=rt.decision(state,'a')
        with torch.no_grad():
            scores,_=net(torch.tensor(x[None]),torch.tensor(c[None]),torch.ones((1,len(c)),dtype=torch.bool))
            old=int(scores.argmax());chosen=(old+1)%len(c)
        pi=np.full(len(c),.1/len(c),dtype=np.float32);pi[old]+=.9
        with patch('app.modules.card_game.engine.ai.zhenhong_guide.recommend',return_value={
                'selected':actions[chosen],'reason':'guide'}):
            actual,label,_=apply_guide(state,'a',actions,old,self.job,rt,pi=pi)
        row=dict(x=x,c=c,z=-1,selected=actual,value_actor=1,**label)
        opt=torch.optim.Adam(net.parameters(),lr=.003)
        def loss():
            with torch.no_grad():
                logits,_=net(torch.tensor(x[None]),torch.tensor(c[None]),torch.ones((1,len(c)),dtype=torch.bool))
                return float(-(torch.tensor(label['pi'])*logits[0].log_softmax(-1)).sum())
        before=loss()
        for _ in range(8):rt.update(net,opt,[row])
        self.assertLess(loss(),before)
        self.assertEqual(row['z'],-1)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for n,p in net.named_parameters() if n.startswith(('state_net.','cand_net.','score.'))))


if __name__=='__main__':unittest.main()
