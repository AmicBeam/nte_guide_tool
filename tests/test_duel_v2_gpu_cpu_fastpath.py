"""Equivalence of CPU empty-mask skipping and legal-table reuse."""
import copy
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class CpuFastPathTest(unittest.TestCase):
    def setUp(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.engine.duel_v2 import new_game
        self.torch = torch
        self.env = GpuDuelEnv(4, device='cpu')
        self.env.reset_from_python([new_game(seed=i+17, skip_mulligan=True,
            decks={'a': STARTER_DECK, 'b': STARTER_DECK}) for i in range(4)])

    def assert_state_equal(self, left, right):
        for name in left.__dataclass_fields__:
            a, b = getattr(left, name), getattr(right, name)
            if self.torch.is_tensor(a):
                self.assertTrue(self.torch.equal(a, b), name)
            else:
                self.assertEqual(a, b, name)

    def test_mixed_walk_matches_unoptimized_path(self):
        from app.modules.card_game.rl.gpu_duel import rules
        fast = self.env.state
        slow = copy.deepcopy(fast)
        rng = self.torch.Generator().manual_seed(187)
        for step in range(24):
            action = self.torch.randint(100000, (4,), generator=rng) % fast.legal_n.clamp(min=1)
            action[step % 4] = -1  # inactive rows must not affect other rows
            with patch.object(rules, '_cpu_mask_empty', return_value=False):
                rules.step_index(slow, action)
            rules.step_index(fast, action, legal_ready=True)
            self.assert_state_equal(fast, slow)

    def test_external_step_rebuilds_stale_legal_table(self):
        from app.modules.card_game.rl.gpu_duel import rules
        state = self.env.state
        expected = copy.deepcopy(state)
        rules.step_index(expected, self.torch.zeros(4, dtype=self.torch.int64))
        state.legal_type.fill_(-1)
        state.legal_n.zero_()
        rules.step_index(state, self.torch.zeros(4, dtype=self.torch.int64))
        self.assert_state_equal(state, expected)

    def test_empty_damage_keeps_settlement(self):
        from app.modules.card_game.rl.gpu_duel import rules
        s = self.env.state
        s.ch_hp[0, 0, 0] = 0
        s.ch_down[0, 0, 0] = 0
        expected = copy.deepcopy(s)
        mask = self.torch.zeros(4, dtype=self.torch.bool)
        amount = self.torch.ones(4, dtype=self.torch.int32)
        with patch.object(rules, '_cpu_mask_empty', return_value=False):
            rules._deal_front(expected, expected.active, amount, mask)
        rules._deal_front(s, s.active, amount, mask)
        self.assert_state_equal(s, expected)
        self.assertEqual(int(s.ch_down[0, 0, 0]), 3)


if __name__ == '__main__':
    unittest.main()
