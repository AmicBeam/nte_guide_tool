import tempfile
import unittest
from pathlib import Path
import numpy as np
from app.modules.card_game.rl.episode_replay import EpisodePool
from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.information_search import leaf_value, terminal_value


class EpisodeTest(unittest.TestCase):
    def data(self,z=1):
        return dict(rows=[dict(x=np.zeros(2),c=np.zeros((2,2)),pi=np.array([.2,.8]),side='a',step=0,z=z)])

    def test_sampling_weights_correct_mixture_and_duplicates_rejected(self):
        pool=EpisodePool()
        for i in range(8):pool.add(str(i),self.data(1 if i==0 else -1))
        rows,refs,metrics=pool.sample(np.random.default_rng(4),64,True)
        for row,q in zip(rows,metrics['probabilities']):self.assertAlmostEqual(row['sample_weight'],min(4.,.125/q))
        with self.assertRaises(ValueError):pool.add('0',self.data())
        self.assertGreater(metrics['effective_samples'],0)

    def test_label_refresh_preserves_outcome_and_observation(self):
        pool=EpisodePool();data=self.data();pool.add('a',data)
        new={**data['rows'][0],'pi':np.array([.7,.3]),'reanalysis_model':'new'}
        pool.replace_labels('a',[new]);self.assertEqual(pool.episodes[0]['rows'][0]['z'],1)
        np.testing.assert_array_equal(pool.episodes[0]['rows'][0]['original_pi'],[.2,.8])
        with self.assertRaises(ValueError):pool.replace_labels('a',[{**new,'z':-1}])

    def test_eviction_keeps_complete_episode_and_rare_win(self):
        pool=EpisodePool(limit=4)
        for i in range(7):pool.add(str(i),self.data(1 if i==0 else -1))
        self.assertEqual(len(pool.episodes),4);self.assertIn('0',[e['id'] for e in pool.episodes])

    def test_mirror_retention_is_independent_of_first_label(self):
        retained=[]
        for signs in ((-1,1),(1,-1)):
            pool=EpisodePool(limit=4)
            pool.add('mirror',dict(rows=[dict(z=z,side=side) for side,z in zip(('a','b'),signs)]))
            for i in range(6):pool.add(str(i),self.data(-1))
            ids=[e['id'] for e in pool.episodes]
            self.assertIn('mirror',ids);retained.append(ids)
        self.assertEqual(*retained)

    def test_value_hook_keeps_observer_and_terminal_sign(self):
        class Runtime:
            decision = None
            @staticmethod
            def value_for(state,viewer,policies):return .4 if viewer=='a' else -.2
        state={'phase':'playing'}
        self.assertEqual(leaf_value(state,'a',{},runtime=Runtime),.4)
        self.assertEqual(leaf_value(state,'b',{},runtime=Runtime),-.2)
        self.assertEqual(terminal_value({'winner':'a'},'a'),1)
        self.assertEqual(leaf_value({'phase':'finished','winner':'a'},'b',{},runtime=Runtime),-1)


class WDLTest(unittest.TestCase):
    def test_self_imitation_reinforces_actual_winner_action_only(self):
        try:import torch
        except ImportError:self.skipTest('Torch required')
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.state_net=torch.nn.Linear(2,2);self.wdl=torch.nn.Linear(3,3)
                self.action_logits=torch.nn.Parameter(torch.zeros(2))
                with torch.no_grad():self.wdl.weight.zero_();self.wdl.bias.zero_()
            def forward(self,x,c,mask):return self.action_logits[None].expand(len(x),-1),x[:,0]*0
        for outcome in (-1,1):
            net=Net();opt=torch.optim.SGD(net.parameters(),lr=.1)
            row=dict(x=np.zeros(2,np.float32),c=np.zeros((2,2),np.float32),pi=np.array([.5,.5],np.float32),z=outcome,selected=1)
            result=rt.update(net,opt,[row],self_imitation=.1)
            self.assertEqual(result['self_imitation_samples'],int(outcome==1))
            if outcome==1:self.assertGreater(float(net.action_logits[1].detach()),float(net.action_logits[0].detach()))
            else:torch.testing.assert_close(net.action_logits,torch.zeros(2))
        row.pop('selected')
        with self.assertRaises(ValueError):rt.update(net,opt,[row],self_imitation=.1)

    def test_export_gradient_and_no_auxiliary_labels(self):
        try:import torch
        except ImportError:self.skipTest('Torch required')
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.fixed_lineup import all_features,CAND_DIM
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        net=CompactScorer(len(all_features()),CAND_DIM,8);net.wdl=torch.nn.Linear(9,3)
        x=np.zeros(len(all_features()),np.float32);c=np.zeros((2,CAND_DIM),np.float32)
        rows=[dict(x=x,c=c,pi=np.array([.8,.2]),z=1,value_actor=1),dict(x=x,z=-1,value_actor=0)]
        old={k:v.clone() for k,v in net.state_dict().items()}
        opt=torch.optim.Adam(net.wdl.parameters(),lr=.01);rt.update(net,opt,rows,warmup=True)
        for k,v in old.items():
            if not k.startswith('wdl.'):self.assertTrue(torch.equal(v,net.state_dict()[k]))
        with self.assertRaises(ValueError):rt.update(net,opt,[{**rows[0],'reward_to_go':10.2}])
        build=next(b for b in STARTER_DECKS if b['id']=='zhenhong')
        with tempfile.TemporaryDirectory() as d:
            rt.export(net,d,'zhenhong',build,{})
            model=rt.OutcomeModel(d,'zhenhong')
            with torch.no_grad():p=net.wdl(torch.cat((net.state_net(torch.tensor(x)),torch.tensor([1.])))).softmax(-1).numpy()
            np.testing.assert_allclose(model.wdl(x,1),p,atol=1e-6)
            with torch.no_grad():scores,_=net(torch.tensor(x)[None],torch.tensor(c)[None],torch.ones((1,2),dtype=torch.bool))
            np.testing.assert_allclose(model.scores(x,c),scores[0].numpy(),atol=1e-6)
            self.assertFalse(model.manifest['automatic_serving_approval'])
            from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
            from app.modules.card_game.rl.fixed_lineup import encode
            from app.modules.card_game.engine.duel_v2 import observe
            state=new_game(seed=77,skip_mulligan=True,decks={'a':build,'b':build})
            viewer='b' if acting_side(state)=='a' else 'a'
            value=rt.value_for(state,viewer,{'a':model,'b':model})
            self.assertTrue(np.isfinite(value))
            # Changing the opponent's hidden hand order leaves value unchanged.
            other='b' if viewer=='a' else 'a';state['sides'][other]['hand'][-1],state['sides'][other]['deck'][-1]=state['sides'][other]['deck'][-1],state['sides'][other]['hand'][-1]
            self.assertAlmostEqual(value,rt.value_for(state,viewer,{'a':model,'b':model}))

    def _policy_data(self, n=4, z=1, prefix='new'):
        rows = []
        for i in range(n):
            rows.append(dict(x=np.zeros(2, np.float32), c=np.eye(2, dtype=np.float32),
                             pi=np.array([.25, .75], np.float32), side='a', step=i, z=z))
        return dict(rows=rows)

    def test_fresh_policy_rows_are_used_once_before_repeats(self):
        pool = EpisodePool(max_row_reuse=3)
        pool.add('old', self._policy_data(2, z=-1))
        pool.begin_fresh_batch()
        pool.add('new', self._policy_data(4, z=1))
        rng = np.random.default_rng(7)
        first = list(pool.policy_pass_batches(rng, passes=1, batch_size=8, include_old_replay=False))
        self.assertEqual(len(first), 1)
        rows, refs, stats = first[0]
        self.assertEqual(len(rows), 4)
        self.assertEqual(stats['first_use'], 4)
        self.assertEqual(stats['repeated_use'], 0)
        self.assertEqual(len(set(stats['row_ids'])), 4)
        coverage = pool.coverage_report()
        self.assertEqual(coverage['fresh_policy_rows'], 4)
        self.assertEqual(coverage['unique_covered'], 4)
        self.assertEqual(coverage['first_use_coverage'], 1.0)
        second = list(pool.policy_pass_batches(rng, passes=4, batch_size=8, include_old_replay=False))
        # Fresh ids are already covered, so later passes only repeat those four rows.
        repeated_ids = [row_id for _, _, item in second for row_id in item['row_ids']]
        self.assertTrue(repeated_ids)
        self.assertEqual(set(repeated_ids), set(stats['row_ids']))
        self.assertEqual(pool.coverage_report()['repeated_use'], len(repeated_ids))
        self.assertEqual(pool.coverage_report()['unique_covered'], 4)

    def test_repeats_do_not_count_as_new_experience(self):
        pool = EpisodePool(max_row_reuse=2)
        pool.begin_fresh_batch()
        pool.add('new', self._policy_data(3))
        rng = np.random.default_rng(3)
        list(pool.policy_pass_batches(rng, passes=1, batch_size=8, include_old_replay=False))
        before = pool.coverage_report()
        list(pool.policy_pass_batches(rng, passes=4, batch_size=8, include_old_replay=False))
        after = pool.coverage_report()
        self.assertEqual(after['unique_covered'], before['unique_covered'])
        self.assertEqual(after['first_use'], before['first_use'])
        self.assertGreater(after['repeated_use'], 0)
        restored = EpisodePool.from_state(pool.export_state())
        self.assertEqual(restored.coverage_report()['repeated_use'], after['repeated_use'])
        self.assertEqual(restored.coverage_report()['unique_row_ids'], after['unique_row_ids'])
