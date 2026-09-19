"""Regression tests for batched no-ops and episode boundaries, not card balance."""
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class GpuTransitionTest(unittest.TestCase):
    def setUp(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        self.torch = torch
        self.env = GpuDuelEnv(4, device='cpu')
        # Avoid opponent setup: these tests control the exact acting side.
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.engine.duel_v2 import new_game
        states = [new_game(seed=i, skip_mulligan=True,
                           decks={'a': STARTER_DECK, 'b': STARTER_DECK}) for i in range(4)]
        self.env.reset_from_python(states)
        self.env.learner = self.env.state.active.clone()

    def test_masked_rows_unchanged_in_mixed_batch(self):
        from app.modules.card_game.rl.gpu_duel.rules import step_index
        s, torch = self.env.state, self.torch
        before = {k: getattr(s, k).clone() for k in s.__dataclass_fields__
                  if torch.is_tensor(getattr(s, k))}
        step_index(s, torch.tensor([-1, -2, 100000, 0]))
        for name, old in before.items():
            got = getattr(s, name)
            if old.ndim and old.shape[0] == s.n:
                self.assertTrue(torch.equal(old[:3], got[:3]), name)
        self.assertFalse(torch.equal(before['turn'][3:], s.turn[3:]))

    def test_harmony_residue_capture_uses_lowered_cap_and_batch_mask(self):
        from app.modules.card_game.rl.gpu_duel.rules import begin_turn, enter
        from app.modules.card_game.rl.gpu_duel.catalog import SEAT_INDEX
        s, torch = self.env.state, self.torch
        nana, zero = SEAT_INDEX['nanali'], SEAT_INDEX['zero']
        s.front[:, 0] = nana
        s.last_front.fill_(-1)
        s.ch_harmony[:, 0, nana] = torch.tensor([1, 2, 1, 1])
        s.turn[:] = torch.tensor([18, 18, 19, 19])
        s.escalation_enabled.fill_(1)
        mask = torch.tensor([True, True, True, False])
        begin_turn(s, torch.zeros(4, dtype=torch.long), mask)
        self.assertEqual(s.last_front[:, 0].tolist(), [-1, nana, nana, -1])
        self.assertEqual(s.front[:, 0].tolist(), [-1, -1, -1, nana])
        s.ch_harmony[:, 0, nana] = torch.tensor([2, 2, 1, 1])
        enter(s, torch.zeros(4, dtype=torch.long), zero, mask)
        self.assertEqual(s.ch_harmony[:, 0, nana].tolist(), [2, 0, 0, 1])
        self.assertEqual(s.last_front[:, 0].tolist(), [-1, -1, -1, -1])

    def test_player_shield_expiry_is_owner_and_mask_local(self):
        from app.modules.card_game.rl.gpu_duel.rules import begin_turn
        s, torch = self.env.state, self.torch
        s.shield[:] = torch.tensor([4, 3])
        begin_turn(s, torch.tensor([0, 1, 0, 1]), torch.tensor([True, True, False, False]))
        self.assertEqual(s.shield.tolist(), [[0, 3], [4, 0], [4, 3], [4, 3]])

    def test_family_instant_uses_global_turn_for_legality_and_payment(self):
        from app.modules.card_game.rl.gpu_duel.catalog import CARD_INDEX, ACT_PLAY
        from app.modules.card_game.rl.gpu_duel.rules import rebuild_legal, play_card
        s, torch = self.env.state, self.torch
        s.active[:] = 0
        s.turn[:] = torch.tensor([4, 5, 6, 10])
        s.turn_count[:, 0] = 3
        s.ap[:] = 0
        s.used_instant[:] = 0
        s.hand_kind[:] = -1
        s.hand_kind[:, 0, 0] = CARD_INDEX['NF01']
        s.hand_n[:, 0] = 1
        rebuild_legal(s)
        self.assertEqual((s.legal_type == ACT_PLAY).any(1).tolist(), [False, True, True, True])
        play_card(s, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
                  torch.tensor([False, True, True, True]),
                  torch.zeros(4, dtype=torch.long), torch.full((4,), -1, dtype=torch.long))
        self.assertEqual(s.ap[:, 0].tolist(), [0, 0, 0, 0])
        self.assertEqual(s.used_instant[:, 0].tolist(), [0, 1, 1, 1])

    def test_terminal_retained_until_explicit_reset(self):
        from app.modules.card_game.rl.gpu_duel.catalog import ACT_ATTACK
        from app.modules.card_game.rl.gpu_duel.ppo import _terminal_reward
        env, torch = self.env, self.torch
        b = torch.arange(env.n); foe = 1 - env.learner
        env.state.hp[b, foe] = 1
        env.state.shield[b, foe] = 0
        env.state.front[b, foe] = -1
        hp = env.state.hp[b, foe].float()
        down = (env.state.ch_hp[b, foe] <= 0).sum(1).float()
        matches = env.state.legal_type == ACT_ATTACK
        self.assertTrue(bool(matches.any(1).all()))
        action = matches.int().argmax(1)
        env.step_learner(action)
        done, winner = env.outcome()
        self.assertTrue(bool(done.all()))
        self.assertTrue(torch.equal(winner, env.learner))
        self.assertEqual(_terminal_reward(env, hp, down).tolist(), [10.] * 4)
        with patch('app.modules.card_game.rl.gpu_duel.env.advance_to_actor'):
            env.reset_finished()
        self.assertFalse(bool(env.outcome()[0].any()))

    def test_collector_captures_win_loss_draw_before_reset(self):
        import app.modules.card_game.rl.gpu_duel.env as env_module
        from app.modules.card_game.rl.gpu_duel.observe import compact_observe
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, gae
        env, torch = self.env, self.torch
        state, cand, _ = compact_observe(env.state)
        net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=8)
        # Keep the policy's values nonzero to expose bootstrap across episodes.
        with torch.no_grad():
            for p in net.parameters(): p.zero_()
            net.value[-1].bias.fill_(2.)
        def advance(s, learner, **kwargs):
            from app.modules.card_game.rl.gpu_duel.rules import rebuild_legal
            s.active.copy_(learner)
            rebuild_legal(s)
        calls = 0
        def terminal_transition(s, action):
            nonlocal calls
            calls += 1
            if calls == 1:
                s.phase[:3] = 3
                s.winner[0] = env.learner[0]
                s.winner[1] = 1 - env.learner[1]
                s.winner[2] = 2  # draw
        with patch.object(env_module, 'step_index', side_effect=terminal_transition), \
             patch.object(env_module, 'advance_to_actor', side_effect=advance), \
             patch('app.modules.card_game.rl.gpu_duel.opponent.advance_to_actor', side_effect=advance):
            rollout = collect_rollout(env, net, horizon=2)
        self.assertEqual(rollout['reward'][:, 0].tolist(), [10., -10., 0., 0.])
        self.assertEqual(rollout['done'][:, 0].tolist(), [1., 1., 1., 0.])
        self.assertEqual(rollout['reward'][:, 1].tolist(), [0.] * 4)
        self.assertEqual(rollout['done'][:, 1].tolist(), [0.] * 4)
        self.assertFalse(bool(env.outcome()[0].any()))
        ret, _ = gae(rollout)
        self.assertEqual(ret.reshape(4, 2)[:3, 0].tolist(), [10., -10., 0.])
        self.assertTrue(all(not v.requires_grad for v in rollout.values() if torch.is_tensor(v)))

    def test_collector_rejects_fake_empty_action(self):
        from app.modules.card_game.rl.gpu_duel.observe import compact_observe
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout
        s, c, m = compact_observe(self.env.state)
        net = CompactScorer(s.shape[-1], c.shape[-1], hidden=8)
        def empty(_): return s, c, self.torch.zeros_like(m)
        with patch('app.modules.card_game.rl.gpu_duel.opponent.advance_to_actor'):
            with self.assertRaisesRegex(RuntimeError, 'no legal actions'):
                collect_rollout(self.env, net, 1, observe_fn=empty)


if __name__ == '__main__':
    unittest.main()
