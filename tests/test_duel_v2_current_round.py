import unittest
from collections import Counter
from copy import deepcopy
import importlib.util
import tempfile
import numpy as np
from scripts.train_duel_v2_current_round import preset_decks, make_job, policy_update_record, write_training_coverage
from scripts.train_duel_v2_current_round import validate_round_launch, learn
from app.modules.card_game.rl.cross_schedule import ROSTER, learning_schedule, matrix_schedule, actual_cell, coverage
from app.modules.card_game.rl import cross_runtime as rt, cross_lineup as contract
from app.modules.card_game.engine.duel_v2 import new_game, observe, apply_action, acting_side
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.summons import summon_front
from app.modules.card_game.rl.information_search import sample_world, search


class CurrentRoundTest(unittest.TestCase):
    def test_formal_worker_cannot_bypass_disabled_launch(self):
        for config in ({}, {'smoke': False}, {'smoke': 'true'},
                       {'smoke': False, 'formal_launch_approved': True, 'throughput_passed': True}):
            with self.assertRaisesRegex(ValueError, 'worker entry cannot bypass'):
                validate_round_launch(config)
            # Refuse before any imports/model initialization, even when a
            # caller skips main() and invokes the worker function directly.
            with self.assertRaisesRegex(ValueError, 'worker entry cannot bypass'):
                learn('unused', config)
        validate_round_launch({'smoke': True})

    def test_policy_update_record_keeps_unique_ids_without_duplicate_keywords(self):
        metrics = dict(policy_samples=2, unique_row_ids=['game:a:0:pi:0'])
        sampling = dict(source='fresh', row_ids=['game:a:0:pi:0'])
        record = policy_update_record('starter', 1, [('game', 0)], sampling, metrics)
        self.assertEqual(record['unique_row_ids'], metrics['unique_row_ids'])
        self.assertEqual(record['sampling'], sampling)
        self.assertEqual(record['policy_samples'], 2)

    def test_training_coverage_file_writes_after_update_stage(self):
        games = [dict(left='starter', right='murk', first='a', id='one', complete=True,
                      winner='a')]
        with tempfile.TemporaryDirectory() as folder:
            write_training_coverage(folder, games)
            import json
            from pathlib import Path
            rows = json.loads((Path(folder) / 'training-coverage.json').read_text())
            self.assertEqual(len(rows), 25)
            selected = next(row for row in rows if (row['row'], row['column']) == ('starter', 'murk'))
            self.assertTrue(selected['complete'])
            self.assertEqual(selected['completed'], 1)

    def test_pairs_cover_both_initiatives_without_seed_parity(self):
        plan = learning_schedule()
        self.assertEqual(len(plan), 20)
        self.assertEqual(set(actual_cell(j) for j in plan), {(a,b) for a in ROSTER for b in ROSTER if a != b})
        for index in range(0,20,2):
            a,b = plan[index:index+2]
            self.assertEqual((a['left'],a['right']), (b['left'],b['right']))
            self.assertEqual((a['first'],b['first']), ('a','b'))
        for workers in (1,3,4,6):
            for seed in (100,101):
                jobs = [make_job(j, seed=seed+i//2, models='unused', decks=preset_decks(),
                        config=dict(simulations=32,gumbel_candidates=16,terminal_horizon=3),
                        deadline=100, identity=str(i), training=True) for i,j in enumerate(plan)]
                flattened = [j for i in range(0,len(jobs),workers) for j in jobs[i:i+workers]]
                self.assertEqual(Counter(actual_cell(j) for j in flattened), Counter(actual_cell(j) for j in plan))

    def test_matrix_all_cells_and_seats_and_no_false_completeness(self):
        plan = matrix_schedule(4)
        self.assertEqual(len(plan),100)
        for row in ROSTER:
            for column in ROSTER:
                cell = [j for j in plan if actual_cell(j)==(row,column)]
                self.assertEqual(len(cell),4)
                self.assertEqual(Counter(j['first'] for j in cell), {'a':2,'b':2})
        games = [dict(j,id=str(i),complete=True,winner=j['first']) for i,j in enumerate(plan)]
        self.assertTrue(all(c['complete'] and c['first_winrate']==1 for c in coverage(games,plan)))
        games[-1]['complete']=False
        self.assertEqual(sum(c['complete'] for c in coverage(games,plan)),24)
        with self.assertRaises(ValueError): coverage(games+[games[0]],plan)
        self.assertTrue(all(c['first_winrate'] is None for c in coverage([],plan)))

    def test_all_matchups_encode_and_search_hidden_worlds(self):
        decks=preset_decks()
        class Uniform:
            def scores_only(self,x,c): return np.zeros(len(c),np.float32)
            def wdl(self,x,actor): return np.array([.3,.4,.3],np.float32)
        for item in learning_schedule():
            s=new_game(seed=8,first_side=item['first'],decks={'a':decks[item['left']],'b':decks[item['right']]})
            for _ in range(2):
                side=acting_side(s)
                actions,(x,c)=rt.decision(s,side)
                self.assertEqual(c.shape[1],contract.CAND_DIM)
                sampled=sample_world(s,side,19,runtime=rt)
                again,(xx,cc)=rt.decision(sampled,side)
                self.assertEqual(actions,again)
                np.testing.assert_array_equal(x,xx)
                np.testing.assert_array_equal(c,cc)
                s=apply_action(s,side,actions[0])
            side=acting_side(s)
            result=search(s,side,{'a':Uniform(),'b':Uniform()},seed=92,simulations=2,
                          algorithm='gumbel',gumbel_candidates=2,runtime=rt,terminal_horizon=1)
            self.assertTrue(result['complete'])

    def test_old_ten_feature_columns_keep_meaning(self):
        from app.modules.card_game.rl import fixed_lineup as old
        decks=preset_decks()
        s=new_game(seed=3,decks={'a':decks['zhenhong'],'b':decks['quick-rush']})
        view=observe(s,'a',include_previews=False)
        a=[x['action'] for x in view['legal_actions'] if x['action']['type']!='concede']
        before,_=old.encode(view,a);after,_=contract.encode(view,a)
        columns={n:i for i,n in enumerate(contract.all_features())}
        for i,n in enumerate(old.all_features()): self.assertEqual(before[i],after[columns[n]],n)

    def test_mixed_summon_generated_card_and_burn_remain_private(self):
        decks=preset_decks()
        s=new_game(seed=16,first_side='a',skip_mulligan=True,decks={'a':decks['starter'],'b':decks['murk']})
        summon_front(s,'b',name='鬼郎丸',attack=2,hp=4)
        s['sides']['a']['front_debuff']['burn']={'stacks':3,'left':2,'by':'b'}
        s['sides']['b']['hand'].append(card_instance(s,'AF01','b'))
        before,(x,c)=rt.decision(s,'a')
        world=rt.determinize(s,'a',23)
        after,(xx,cc)=rt.decision(world,'a')
        self.assertEqual(before,after)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(c,cc)
        self.assertGreater(x[contract.all_features().index('murk:burn_stacks:0')],0)

    def test_xiaozhi_generated_copies_against_murk(self):
        decks=preset_decks()
        s=new_game(seed=18,first_side='b',skip_mulligan=True,decks={'a':decks['quick-rush'],'b':decks['murk']})
        s['sides']['a']['deck'].append(card_instance(s,'Q05','a'))
        before,(x,c)=rt.decision(s,'b')
        world=rt.determinize(s,'b',31)
        after,(xx,cc)=rt.decision(world,'b')
        self.assertEqual(before,after)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(c,cc)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch optional locally')
    def test_migration_export_and_real_update(self):
        import torch
        from app.modules.card_game.rl import murk_runtime, outcome_runtime, grounded_candidates
        decks=preset_decks()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            for key in ('zhenhong','murk'):
                if key=='murk':
                    net=murk_runtime.create_network(16,'cpu');old=murk_runtime
                else:
                    net=grounded_candidates.network(16);old=outcome_runtime
                old.export(net,folder,key,decks[key],{})
                expanded,build,origin=rt.migrate(folder,key,'cpu')
                rt.export(expanded,folder+'/new',key,build,origin)
                numeric=rt.model(folder+'/new',key)
                s=new_game(seed=9,decks={'a':build,'b':decks['starter']})
                actions,(x,c)=rt.decision(s,'a')
                xx=torch.tensor(x[None]);cc=torch.tensor(c[None]);mask=torch.ones((1,len(c)),dtype=torch.bool)
                scores,_=expanded(xx,cc,mask)
                np.testing.assert_allclose(scores.detach().numpy()[0],numeric.scores(x,c),atol=2e-5,rtol=2e-5)
                wdl=expanded.wdl(torch.cat((expanded.state_net(xx),torch.ones((1,1))),-1)).softmax(-1)
                np.testing.assert_allclose(wdl.detach().numpy()[0],numeric.wdl(x,True),atol=2e-5,rtol=2e-5)
                before=expanded.wdl.weight.detach().clone()
                opt=torch.optim.Adam(expanded.parameters(),lr=1e-3)
                rt.update(expanded,opt,[dict(x=x,c=c,pi=np.ones(len(c))/len(c),selected=0,z=1)])
                self.assertFalse(torch.equal(before,expanded.wdl.weight))
                if key=='murk': self.assertTrue(origin['changed_ordinal_columns_reset'])
                else: self.assertFalse(origin['changed_ordinal_columns_reset'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch optional locally')
    def test_policy_pass_reaches_joint_heads_and_keeps_terminal_value(self):
        import torch
        from app.modules.card_game.rl.episode_replay import EpisodePool
        torch.set_num_threads(1)
        net = rt.create_network(8, 'cpu')
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
        decks = preset_decks()
        state = new_game(seed=11, decks={'a': decks['starter'], 'b': decks['murk']})
        actions, (x, c) = rt.decision(state, 'a')
        pi = np.zeros(len(c), np.float32); pi[0] = 1
        pool = EpisodePool(max_row_reuse=2)
        pool.begin_fresh_batch()
        pool.add('fresh', dict(rows=[
            dict(x=x, c=c, pi=pi, selected=0, z=1, side='a', step=0, key='starter'),
            dict(x=x, z=-1, side='a', step=1, key='starter'),
        ]))
        rows, refs, stats = next(pool.policy_pass_batches(np.random.default_rng(2), passes=1, batch_size=8, include_old_replay=False))
        self.assertEqual(len(rows), 1)
        self.assertIn('pi', rows[0])
        self.assertEqual(rows[0]['z'], 1)
        metrics = rt.update(net, opt, rows)
        self.assertTrue(all(np.isfinite(metrics['policy_gradient_norms'][name])
                            and metrics['policy_gradient_norms'][name] >= 0
                            for name in ('cand_net', 'score', 'state_net')))
        self.assertGreater(sum(metrics['policy_gradient_norms'][name]
                               for name in ('cand_net', 'score', 'state_net')), 0)
        with self.assertRaises(ValueError):
            rt.update(net, opt, [{**rows[0], 'reward_to_go': 4.0}])
        before_wdl = net.wdl.weight.detach().clone()
        value_metrics = rt.update(net, opt, [dict(x=x, z=-1)], value_only=True)
        self.assertEqual(value_metrics['policy_samples'], 0)
        self.assertTrue(value_metrics['value_only'])
        self.assertFalse(torch.equal(before_wdl, net.wdl.weight))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch optional locally')
    def test_p0_still_stops_and_rollback_keeps_best_arrays(self):
        import torch
        from unittest.mock import patch
        from pathlib import Path
        import scripts.train_duel_v2_current_round as train
        metrics = {
            key: dict(finite=True, policy_gradient=True, play_legal=30, play_top1=0.4,
                      reference_play_top1=0.4, first_use_coverage=1.0, unique_covered=4,
                      fresh_policy_rows=4, first_use=4, repeated_use=0)
            for key in ROSTER
        }
        metrics['starter']['play_top1'] = 0
        metrics['starter']['first_use_coverage'] = 0
        metrics['starter']['unique_covered'] = 0
        with patch.object(train, '_pure_report', return_value=metrics):
            decision = train._run_gate('first', Path('.'), dict(device='cpu'), 1, None, None, None, None, False, None, None)
        self.assertEqual(decision['decision'], 'p0')
        self.assertTrue(decision['diagnostic_replay_rows'])
        self.assertFalse(decision['independent_selection'])
        self.assertIn('no_play', {row['reason'] for row in decision['reasons']})
        self.assertIn('no_coverage', {row['reason'] for row in decision['reasons']})
        torch.set_num_threads(1)
        decks = preset_decks()
        nets = {key: rt.create_network(8, 'cpu') for key in ROSTER}
        best_state = {key: {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()} for key, net in nets.items()}
        with tempfile.TemporaryDirectory() as folder:
            best_dir = Path(folder) / 'best'
            origins = {}
            for key, net in nets.items():
                origins[key] = dict(initialization='test')
                rt.export(net, best_dir, key, decks[key], origins[key])
                with torch.no_grad():
                    net.score.weight.add_(1)
            learning_rate, exported = train._rollback(Path(folder), dict(min_learning_rate=1e-5), nets, best_state, best_dir, decks, origins, 3e-5, 1)
            self.assertEqual(learning_rate, 1e-5)
            for key in ROSTER:
                retained = rt.model(best_dir, key)
                rolled = rt.model(exported, key)
                for name, value in retained.weights.items():
                    np.testing.assert_array_equal(value, rolled.weights[name])
