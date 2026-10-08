import unittest
import torch
from app.modules.card_game.rl.cross_runtime import create_network
from app.modules.card_game.rl.factored_action_score import (
    FactoredActionScore, factored_beats, scheme_passes,
)
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY


class FactoredActionScoreTest(unittest.TestCase):
    def test_two_cards_can_differ_without_raising_every_play(self):
        net = create_network(8, 'cpu')
        scored = FactoredActionScore(net)
        with torch.no_grad():
            scored.card_head.bias.zero_()
            scored.card_head.weight.zero_()
            scored.card_head.bias[3] = 0.4
        x = torch.zeros(1, net.state_net[0].in_features)
        candidates = torch.zeros(1, 3, 11)
        candidates[0, 0, 0] = ACT_PLAY
        candidates[0, 0, 2] = 1
        candidates[0, 1, 0] = ACT_PLAY
        candidates[0, 1, 2] = 2
        candidates[0, 2, 0] = ACT_ATTACK
        mask = torch.ones(1, 3, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = scored(x, candidates, mask)
        self.assertGreater(float(logits[0, 1]), float(logits[0, 0]))
        self.assertGreater(float(logits[0, 0]), float(logits[0, 2]) - 1e-6)

    def test_screen_ignores_a_higher_play_rate(self):
        baseline = dict(n=10, attack_n=4, attack_called_play=3, end_n=3, end_called_play=2,
                        target_play_n=4, same_card=1, play_top=2, end_top=5)
        more_play_but_worse_cards = dict(baseline, attack_called_play=1, end_called_play=3, same_card=1, play_top=8)
        self.assertFalse(factored_beats(baseline, more_play_but_worse_cards))
        better = dict(baseline, attack_called_play=1, end_called_play=1, same_card=3, play_top=4)
        self.assertTrue(factored_beats(baseline, better))
        always_play = dict(better, play_top=10)
        self.assertFalse(factored_beats(baseline, always_play))

    def test_coupled_bias_is_not_added_and_padding_is_masked(self):
        net = create_network(8, 'cpu')
        scored = FactoredActionScore(net)
        with torch.no_grad():
            net.score.bias.fill_(10)
            scored.kind_head.bias.zero_()
            scored.kind_head.bias[ACT_END] = 5
        x = torch.zeros(1, net.state_net[0].in_features)
        candidates = torch.zeros(1, 2, 11)
        candidates[0, 0, 0] = ACT_ATTACK
        mask = torch.tensor([[True, False]])
        with torch.no_grad():
            logits, _ = scored(x, candidates, mask)
        self.assertAlmostEqual(float(logits[0, 0]), 0.0, places=5)
        self.assertEqual(float(logits[0, 1]), torch.finfo(logits.dtype).min)
        self.assertAlmostEqual(float(logits.softmax(-1)[0, 0]), 1.0, places=5)

    def test_policy_gradient_reaches_the_new_heads_only(self):
        net = create_network(8, 'cpu')
        with torch.no_grad():
            net.state_net[0].weight.zero_()
            net.state_net[2].weight.zero_()
            net.state_net[0].bias.fill_(0.5)
            net.state_net[2].bias.fill_(0.5)
        scored = FactoredActionScore(net)
        x = torch.zeros(2, net.state_net[0].in_features)
        candidates = torch.zeros(2, 2, 11)
        candidates[:, 0, 0] = ACT_PLAY
        candidates[:, 0, 2] = 1
        candidates[:, 1, 0] = ACT_ATTACK
        mask = torch.ones(2, 2, dtype=torch.bool)
        logits, _ = scored(x, candidates, mask)
        (logits[:, 0].sum() - logits[:, 1].sum()).backward()
        self.assertGreater(float(scored.card_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(scored.kind_head.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in net.cand_net.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in net.score.parameters()))

    def test_scheme_needs_every_selection_team_at_one_rate(self):
        self.assertFalse(scheme_passes({
            1.8e-4: {'starter': True, 'weave-rush': False, 'zhenhong': True},
        }))
        self.assertTrue(scheme_passes({
            1.5e-4: {'starter': False, 'weave-rush': True, 'zhenhong': True},
            1.8e-4: {'starter': True, 'weave-rush': True, 'zhenhong': True, 'quick-rush': False},
        }))


if __name__ == '__main__':
    unittest.main()
