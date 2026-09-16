"""Episode return cap: outcome-only rewards, rollout continuity and row resets."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class RewardCapTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from app.modules.card_game.rl.episode_return import cap_episode_reward, reset_episode_return
        self.cap, self.reset = cap_episode_reward, reset_episode_return
        self.env = SimpleNamespace(n=4, device=torch.device('cpu'))
        self.reset(self.env)

    def test_win_loss_draw_and_no_damage_win(self):
        t = self.torch
        self.cap(self.env, t.tensor([3., 3., 3., 0.]))
        emitted = self.cap(self.env, t.tensor([10., -10., 0., 10.]))
        self.assertEqual(emitted.tolist(), [7., -10., 0., 10.])
        self.assertEqual(self.env.episode_return.tolist(), [10., -7., 3., 10.])

    def test_cap_is_cumulative_not_a_per_step_clip(self):
        t = self.torch
        emitted = []
        for _ in range(25):
            emitted.append(self.cap(self.env, t.full((4,), .5)))
        self.assertEqual(t.stack(emitted).sum(0).tolist(), [10.]*4)
        self.assertEqual(emitted[-1].tolist(), [0.]*4)
        self.assertEqual(self.cap(self.env, t.full((4,), -10.)).tolist(), [-10.]*4)
        self.assertEqual(self.env.episode_return.tolist(), [0.]*4)

    def test_partial_reset_does_not_clear_other_games(self):
        t = self.torch
        self.cap(self.env, t.tensor([2., 3., 4., 5.]))
        self.reset(self.env, t.tensor([True, False, True, False]))
        self.assertEqual(self.env.episode_return.tolist(), [0., 3., 0., 5.])
        self.assertEqual(self.cap(self.env, t.full((4,), 10.)).tolist(), [10., 7., 10., 5.])
        self.reset(self.env)
        self.assertEqual(self.env.episode_return.tolist(), [0.]*4)

    def test_damage_knockdown_and_shield_award_zero(self):
        t = self.torch
        from app.modules.card_game.rl.gpu_duel.ppo import _terminal_reward
        env = self.env
        env.learner = t.tensor([0, 1, 0, 1], dtype=t.int32)
        env.state = SimpleNamespace(hp=t.full((4, 2), 30, dtype=t.int32),
                                    ch_hp=t.full((4, 2, 4), 5, dtype=t.int32),
                                    shield=t.full((4, 2), 5, dtype=t.int32))
        env.outcome = lambda: (t.zeros(4, dtype=t.bool), t.full((4,), -1))
        env.state.shield[0, 1] = 0  # Shield loss only: no reward.
        env.state.hp[1, 0] -= 2
        env.state.ch_hp[2, 1, 0] = 0
        env.state.hp[3, 0] -= 20
        env.state.ch_hp[3, 0, :2] = 0
        got = _terminal_reward(env, t.full((4,), 30.), t.zeros(4))
        t.testing.assert_close(got, t.zeros(4))

    def test_collector_keeps_cap_across_rollouts_and_clears_only_terminals(self):
        t = self.torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout
        from app.modules.card_game.rl.gpu_duel.observe import compact_observe
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.engine.duel_v2 import new_game
        env = GpuDuelEnv(4, device='cpu')
        states = [new_game(seed=i, skip_mulligan=True, decks={'a':STARTER_DECK,'b':STARTER_DECK}) for i in range(4)]
        env.reset_from_python(states)
        env.learner = env.state.active.clone()
        env.episode_return.zero_()
        s, c, _ = compact_observe(env.state)
        net = CompactScorer(s.shape[-1], c.shape[-1], hidden=8)
        count = 0
        def advance(state, learner, **kwargs):
            state.active.copy_(learner)
        def step(state, actions):
            nonlocal count
            count += 1
            if count == 1:
                state.hp[0, 1-int(env.learner[0])] -= 20
            elif count == 2:
                state.phase[:3] = 3
                state.winner[0] = env.learner[0]
                state.winner[1] = 1-env.learner[1]
                state.winner[2] = 2
        with patch('app.modules.card_game.rl.gpu_duel.env.step_index', side_effect=step), \
             patch('app.modules.card_game.rl.gpu_duel.env.advance_to_actor', side_effect=advance):
            first = collect_rollout(env, net, 1)
            self.assertEqual(first['reward'][:, 0].tolist(), [0., 0., 0., 0.])
            self.assertEqual(env.episode_return.tolist(), [0., 0., 0., 0.])
            second = collect_rollout(env, net, 1)
            self.assertEqual(second['reward'][:, 0].tolist(), [10., -10., 0., 0.])
            self.assertEqual(second['done'][:, 0].tolist(), [1., 1., 1., 0.])
            self.assertEqual(env.episode_return.tolist(), [0., 0., 0., 0.])
            # New wins in the reset rows must have the full reward available.
            self.assertEqual(self.cap(env, t.tensor([10., 10., 10., 0.])).tolist(), [10., 10., 10., 0.])
        env.reset_from_python(states)
        self.assertEqual(env.episode_return.tolist(), [0.]*4)
        with patch.object(env, '_advance_to_actor'):
            env.episode_return.fill_(8)
            env.reset(seeds=[0,1,2,3])
        self.assertEqual(env.episode_return.tolist(), [0.]*4)

    def test_reward_resume_requires_same_objective(self):
        from app.modules.card_game.rl.episode_return import REWARD_MODE, validate_reward_resume
        for checkpoint in ({}, {'reward_mode': 'shaped'}):
            with self.assertRaisesRegex(ValueError, 'warm-start'):
                validate_reward_resume(checkpoint)
        validate_reward_resume({'reward_mode': REWARD_MODE})

    def test_public_env_resets_selected_episode_totals(self):
        t = self.torch
        from app.modules.card_game.rl.public_training import PublicTrainingEnv
        env = PublicTrainingEnv.__new__(PublicTrainingEnv)
        env.n, env.device = 4, t.device('cpu')
        env.state = object()
        env.terminal_counters = SimpleNamespace(record=Mock(), reset=Mock())
        env.fixed = t.zeros((4,36), dtype=t.int32)
        env.learner = t.tensor([0,1,0,1], dtype=t.int32)
        env.assignment = t.zeros(4, dtype=t.int32)
        env.league = SimpleNamespace(history=[], policies=[])
        env.pool_counters = SimpleNamespace(record=Mock(), reset=Mock())
        env.first_action_recorded = t.zeros(4, dtype=t.bool)
        env.opponent_lineup = t.zeros(4, dtype=t.int32)
        env.random_team_fraction = .2
        env.rule_fraction = .2
        env.learn_mulligan = False
        env.compiled = SimpleNamespace(reset_public_gpu_state=Mock())
        env.prepare_decision = Mock()
        self.reset(env)
        env.episode_return.copy_(t.tensor([2.,3.,4.,5.]))
        with patch('app.modules.card_game.rl.public_training.tensor_preset_opponents', return_value=(env.fixed, env.opponent_lineup)):
            env.reset(t.tensor([False, True, False, True]))
            self.assertEqual(env.episode_return.tolist(), [2.,0.,4.,0.])
            env.reset()
            self.assertEqual(env.episode_return.tolist(), [0.]*4)


if __name__ == '__main__':
    unittest.main()
