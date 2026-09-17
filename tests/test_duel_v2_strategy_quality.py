import tempfile
import unittest
from app.modules.card_game.rl.strategy_quality import ValidationTracker,tactical_cases,reachable_curriculum,diagnose_network

class StrategyQualityTest(unittest.TestCase):
    def test_validation_keeps_best_not_last_and_rejects_incomplete(self):
        t=ValidationTracker(patience=2)
        self.assertTrue(t.observe(0,.3));self.assertTrue(t.observe(10,.5))
        self.assertFalse(t.observe(20,.4));self.assertFalse(t.observe(30,.5))
        self.assertEqual(t.best['update'],10);self.assertTrue(t.plateau)
        with self.assertRaises(ValueError):t.observe(40,.9,complete=False)

    def test_buff_order_dominance_is_real_in_both_seats(self):
        from app.modules.card_game.engine.duel_v2 import apply_action
        for c in tactical_cases():
            if not c['name'].startswith('buff_before'):continue
            hp=[]
            for order in [('M01','M02'),('M02','M01')]:
                s=c['state'];side=c['side']
                for cid in order:
                    card=next(x for x in s['sides'][side]['hand'] if x['card_id']==cid)
                    s=apply_action(s,side,{'type':'play_card','card_id':card['instance_id']})
                hp.append(s['sides']['b' if side=='a' else 'a']['hp'])
            self.assertEqual(hp[0]-hp[1],1)

    def test_tactical_observation_and_action_parity(self):
        from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.public_schema import state_dim
        from app.modules.card_game.rl.opening_observation import OPENING_CAND_DIM
        with tempfile.TemporaryDirectory() as directory:
            backend=CompiledStarterBackend(directory,backend='native')
            try:r=diagnose_network(CompactScorer(state_dim(),OPENING_CAND_DIM,16),backend=backend)
            finally:backend.close()
        self.assertEqual(len(r['cases']),6)
        self.assertTrue(all(c['parity_actions']>0 for c in r['cases']))

    def test_curriculum_runs_real_ppo_with_outcome_only_reward(self):
        import torch
        from app.modules.card_game.rl.capability import preset_opponent_deck
        from app.modules.card_game.rl.public_training import PublicTrainingEnv,FrozenLeague
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout,rollout_to_batch,ppo_update
        from app.modules.card_game.rl.public_schema import state_dim
        from app.modules.card_game.rl.opening_observation import OPENING_CAND_DIM,opening_observe,OPENING_SCHEMA
        from app.modules.card_game.rl.episode_return import REWARD_MODE
        torch.manual_seed(123)
        build=preset_opponent_deck('starter');foe=preset_opponent_deck('weave-rush')
        rows=reachable_curriculum(build,foe,count=3);self.assertEqual(len(rows),3)
        net=CompactScorer(state_dim(),OPENING_CAND_DIM,16);net._learn_mulligan=True
        league=FrozenLeague(learn_mulligan=True,greedy=True);league.add_pinned(net)
        with tempfile.TemporaryDirectory() as directory:
            env=PublicTrainingEnv(8,deck_id='starter',league=league,device='cpu',compiled_dir=directory,
                learner_deck=build,learn_mulligan=True,random_team_fraction=0,rule_fraction=0,
                fixed_opponent_deck=foe,curriculum_rows=rows,curriculum_fraction=.5)
            try:
                before=net.score.weight.detach().clone();roll=collect_rollout(env,net,4,observe_fn=opening_observe)
                stats=ppo_update(net,torch.optim.Adam(net.parameters()),rollout_to_batch(roll))
                self.assertGreater(stats['optimizer_steps'],0);self.assertGreater(env.curriculum_resets,0)
                self.assertFalse(torch.equal(before,net.score.weight))
                self.assertEqual(REWARD_MODE,'outcome_only_v1')
            finally:env.close()
