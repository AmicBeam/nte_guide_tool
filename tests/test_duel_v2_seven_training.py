"""Versioned seven-team encoding, information sets and numerical learning contract."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game,observe,apply_action,acting_side
from app.modules.card_game.rl import seven_lineup as layout,seven_runtime as rt
from app.modules.card_game.rl.seven_sampler import determinize
from app.modules.card_game.rl.seven_schedule import ROSTER,learning_schedule,matrix_schedule

DECKS={d['id']:d for d in STARTER_DECKS}


class SevenEncodingTest(unittest.TestCase):
    def test_roster_coverage_and_every_starting_build(self):
        self.assertEqual(len(learning_schedule()),42)
        self.assertEqual(len(matrix_schedule(2)),98)
        for key in ROSTER:
            with self.subTest(key=key):
                state=new_game(seed=31,decks={'a':DECKS[key],'b':DECKS['collapse']})
                view=observe(state,'a',include_previews=False);before_view=deepcopy(view)
                actions,(x,c)=rt.decision(state,'a')
                layout.encode(view,actions);self.assertEqual(view,before_view)
                self.assertEqual(x.shape,(len(layout.all_features()),))
                self.assertEqual(c.shape,(len(actions),layout.CAND_DIM))
                self.assertTrue(np.isfinite(c).all())
                proxy=deepcopy(state);proxy['phase']='playing';proxy['active_side']='a'
                sample=determinize(proxy,'a',999)
                aa,(xx,cc)=rt.decision(sample,'a');bb,(yy,dd)=rt.decision(proxy,'a')
                self.assertEqual(aa,bb);np.testing.assert_array_equal(xx,yy);np.testing.assert_array_equal(cc,dd)

    def test_every_training_job_uses_canonical_cards_and_fixed_character_order(self):
        from app.modules.card_game.rl.seven_round import jobs_for
        jobs=jobs_for('numerical-source',DECKS,771,99999999999,pure=True)
        self.assertEqual(len(jobs),56)
        for job in jobs:
            for side in ('a','b'):
                key=job['policies'][side][1];build=job['decks'][side]
                self.assertEqual(build['card_ids'],sorted(DECKS[key]['card_ids']))
                self.assertEqual(build['character_ids'],DECKS[key]['character_ids'])

    def test_ordered_top_bottom_and_exit_are_distinct(self):
        from tests.test_duel_v2_heiyu import game,top,play,witch
        state=game();top(state,['P01','P03','Z01','P05']);state=play(state,'PF01')
        actions,(x,c)=rt.decision(state,'a')
        self.assertTrue(all(a['type']=='choose_deck_order' for a in actions))
        self.assertGreater(len({row.tobytes() for row in c}),4)
        self.assertEqual(set(c[:,0]),{6})
        # Same bottom set, different order must have different features.
        groups={}
        for action,row in zip(actions,c):
            key=tuple(sorted(action.get('bottom_card_ids',[])))
            groups.setdefault(key,[]).append(row)
        self.assertTrue(any(len({r.tobytes() for r in rows})>1 for rows in groups.values()))
        state=witch(game());actions,(x,c)=rt.decision(state,'a')
        self.assertTrue(any(a['type']=='exit_ultimate' and row[0]==7 for a,row in zip(actions,c)))

    def test_opaque_enemy_deck_and_hand_do_not_constrain_samples(self):
        state=new_game(seed=5,skip_mulligan=True,decks={'a':DECKS['collapse'],'b':DECKS['lingke']})
        altered=deepcopy(state)
        altered['sides']['b']['hand'].reverse();altered['sides']['b']['deck'].reverse()
        a=determinize(state,'a',333);b=determinize(altered,'a',333)
        self.assertEqual([c['card_id'] for c in a['sides']['b']['hand']],[c['card_id'] for c in b['sides']['b']['hand']])
        self.assertEqual([c['card_id'] for c in a['sides']['b']['deck']],[c['card_id'] for c in b['sides']['b']['deck']])

    def test_public_resistance_values_are_encoded_without_parsing_labels(self):
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect
        state=new_game(seed=5,skip_mulligan=True,decks={'a':DECKS['lingke'],'b':DECKS['collapse']})
        for holder in (state['sides']['b'],state['sides']['b']['characters']['canhong']):
            add_effect(holder,'lingke.resistance','damage.resistance',-1,
                       conditions=[dict(id='damage_attribute',args=dict(attribute='咒'))])
        _,(x,_)=rt.decision(state,'a')
        names=layout.all_features()
        self.assertAlmostEqual(x[names.index('seven:resistance:1:咒')],-.1)
        self.assertAlmostEqual(x[names.index('seven:resistance:1:canhong:咒')],-.1)

    def test_current_choice_and_generation_roots_keep_information(self):
        from app.modules.card_game.rl.covered_rollout import rule_decision
        for key in ('collapse','lingke','murk'):
            state=new_game(seed=771,skip_mulligan=True,decks={'a':DECKS[key],'b':DECKS['collapse']})
            seen=set()
            for step in range(100):
                if state['phase']=='finished':break
                side=acting_side(state);actions,chosen=rule_decision(state,side)
                seen.add(state['phase'])
                if state['phase']=='choice' or step%10==0:
                    determinize(state,side,880+step)
                state=apply_action(state,side,actions[chosen])
            self.assertIn('playing',seen)


class SevenNumericTest(unittest.TestCase):
    def test_real_rows_joint_update_export_and_value_separation(self):
        try:import torch
        except ImportError:self.skipTest('optional Torch')
        from app.modules.card_game.rl.league_learning import tensors
        torch.set_num_threads(1);torch.manual_seed(322)
        state=new_game(seed=5,decks={'a':DECKS['lingke'],'b':DECKS['collapse']})
        actions,(x,c)=rt.decision(state,'a')
        net=rt.create_network('lingke',hidden=32);opt=torch.optim.Adam(net.parameters(),lr=.0003)
        row=dict(x=x,c=c,pi=np.eye(len(c),dtype=np.float32)[3],z=1,value_actor=1)
        before={n:t.clone() for n,t in net.state_dict().items()}
        metric=rt.update(net,opt,[row])
        self.assertTrue(all(np.isfinite(metric['policy_gradient_norms'][prefix]) and metric['policy_gradient_norms'][prefix]>0 for prefix in ('state_net','cand_net','score')))
        self.assertTrue(any(not torch.equal(v,net.state_dict()[n]) for n,v in before.items() if n.startswith('score.')))
        before={n:t.clone() for n,t in net.state_dict().items() if n.startswith(('state_net.','cand_net.','score.'))}
        rt.update(net,opt,[dict(row,z=-1)],value_only=True)
        self.assertTrue(all(torch.equal(v,net.state_dict()[n]) for n,v in before.items()))
        with tempfile.TemporaryDirectory() as root:
            rt.export(net,root,'lingke',DECKS['lingke'],dict(test_only=True))
            policy=rt.model(root,'lingke');restored,_,_=rt.restore(root,'lingke')
            self.assertEqual(policy.serving_deck['card_ids'],sorted(DECKS['lingke']['card_ids']))
            xx,cc,mask=tensors([(x,c)],'cpu');scores,_=net(xx,cc,mask)
            np.testing.assert_allclose(scores.detach().numpy()[0],policy.scores(x,c),rtol=2e-4,atol=2e-4)
            self.assertTrue(all(torch.equal(t,restored.state_dict()[n]) for n,t in net.state_dict().items()))


class SevenAliasNumericTest(unittest.TestCase):
    def test_equal_features_share_exact_scores_without_merging_actions_or_targets(self):
        import torch
        from app.modules.card_game.rl.seven_policy import canonical_scores_tensor,canonical_scores_numpy
        c=np.asarray([[3,1],[3,1],[4,2]],dtype=np.float32)
        values=torch.tensor([[2.8669748,2.8669751,1.]],requires_grad=True)
        mask=torch.ones(1,3,dtype=torch.bool)
        result=canonical_scores_tensor(values,torch.tensor(c)[None],mask)
        self.assertEqual(result.shape,(1,3))
        self.assertEqual(result[0,0].item(),result[0,1].item())
        self.assertEqual(int(result.argmax()),0)
        pi=torch.tensor([[.1,.8,.1]])
        loss=-(pi*result.log_softmax(-1)).sum();loss.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertNotEqual(values.grad[0,0].item(),0)
        np.testing.assert_array_equal(canonical_scores_numpy(np.asarray([2.8669748,2.8669751,1.]),c),[2.8669748,2.8669748,1.])
