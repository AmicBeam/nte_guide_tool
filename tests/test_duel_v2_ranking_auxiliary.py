import copy
import unittest

import numpy as np

from app.modules.card_game.rl import outcome_runtime as rt
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY


def _require_torch():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest('Torch required')
    return torch


class TinyNet:
    def __new__(cls, torch, state_dim=2, width=3):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.state_net = torch.nn.Linear(state_dim, 2)
                self.wdl = torch.nn.Linear(3, 3)
                self.cand_net = torch.nn.Linear(width, 2)
                self.score = torch.nn.Linear(2, 1)
                with torch.no_grad():
                    self.wdl.weight.zero_()
                    self.wdl.bias.zero_()
                    self.cand_net.weight.zero_()
                    self.cand_net.bias.zero_()
                    self.score.weight.zero_()
                    self.score.bias.zero_()
                    self.state_net.weight.zero_()
                    self.state_net.bias.fill_(0.2)

            def forward(self, x, c, mask):
                hidden = self.cand_net(c)
                logits = self.score(hidden).squeeze(-1) + (hidden * self.state_net(x).unsqueeze(1)).sum(-1)
                return logits.masked_fill(~mask, torch.finfo(logits.dtype).min), x[:, 0] * 0

        return Module()


def _row(c, pi, z=1, truncated=False):
    return dict(
        x=np.zeros(2, np.float32),
        c=np.asarray(c, dtype=np.float32),
        pi=np.asarray(pi, dtype=np.float64),
        z=z,
        value_actor=1,
        truncated=truncated,
    )


class RankingAuxiliaryTest(unittest.TestCase):
    def test_zero_coefficient_matches_old_optimizer_bitwise(self):
        torch = _require_torch()
        torch.manual_seed(0)
        row = _row(
            [[ACT_END, 0, -1], [ACT_PLAY, 0, 1], [ACT_ATTACK, 1, -1]],
            [0.1, 0.7, 0.2],
        )
        baseline = TinyNet(torch)
        ranked = TinyNet(torch)
        ranked.load = None
        ranked.state_net.load_state_dict(baseline.state_net.state_dict())
        ranked.wdl.load_state_dict(baseline.wdl.state_dict())
        ranked.cand_net.load_state_dict(baseline.cand_net.state_dict())
        ranked.score.load_state_dict(baseline.score.state_dict())
        opt_a = torch.optim.SGD(baseline.parameters(), lr=0.05)
        opt_b = torch.optim.SGD(ranked.parameters(), lr=0.05)
        left = rt.update(baseline, opt_a, [copy.deepcopy(row)])
        right = rt.update(ranked, opt_b, [copy.deepcopy(row)], ranking_coefficient=0.)
        self.assertEqual(left['policy_loss'], right['policy_loss'])
        self.assertEqual(left['value_loss'], right['value_loss'])
        self.assertEqual(right['ranking_samples'], 0)
        for name, parameter in baseline.named_parameters():
            other = dict(ranked.named_parameters())[name]
            self.assertTrue(torch.equal(parameter, other), name)

    def test_illegal_and_truncated_rows_are_excluded(self):
        torch = _require_torch()
        logits = torch.tensor([[4., 1., 3., 9.]])
        mask = torch.tensor([[True, True, False, False]])
        rows = [
            _row([[ACT_PLAY, 0, 1], [ACT_ATTACK, 0, -1], [ACT_END, 0, -1]], [0.9, 0.1, 0.0]),
        ]
        rows[0]['c'] = np.asarray([[ACT_PLAY, 0, 1], [ACT_ATTACK, 0, -1], [ACT_END, 0, -1], [ACT_PLAY, 0, 2]], np.float32)
        rows[0]['pi'] = np.asarray([0.05, 0.05, 0.05, 0.85])
        loss, stats = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=0.2, confidence=0.4)
        self.assertEqual(stats['ranking_samples'], 0)
        self.assertEqual(stats['ranking_excluded'], 1)
        self.assertEqual(float(loss.detach()), 0.0)
        rows[0]['pi'] = np.asarray([0.9, 0.1, 0.0, 0.0])
        rows[0]['truncated'] = True
        loss, stats = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=0.2, confidence=0.4)
        self.assertEqual(stats['ranking_samples'], 0)

    def test_low_confidence_gumbel_top_is_skipped(self):
        torch = _require_torch()
        logits = torch.tensor([[0.2, 0.1, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        rows = [_row([[ACT_PLAY, 0, 1], [ACT_PLAY, 0, 2], [ACT_END, 0, -1]], [0.4, 0.35, 0.25])]
        loss, stats = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=0.2, confidence=0.45)
        self.assertEqual(stats['ranking_samples'], 0)
        self.assertEqual(stats['ranking_excluded'], 1)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_card_identity_competes_with_other_legal_cards(self):
        torch = _require_torch()
        logits = torch.tensor([[0.1, 2.0, -1.0]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        rows = [_row([[ACT_PLAY, 0, 1], [ACT_PLAY, 0, 2], [ACT_ATTACK, 1, -1]], [0.8, 0.15, 0.05])]
        loss, stats = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=1.0, confidence=0.45)
        self.assertEqual(stats['ranking_samples'], 1)
        expected = torch.nn.functional.softplus(logits[0, 1] - logits[0, 0])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)

    def test_attack_and_end_targets_are_not_turned_into_play(self):
        torch = _require_torch()
        logits = torch.tensor([[3.0, 0.2, 0.1]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        rows = [_row([[ACT_ATTACK, 0, -1], [ACT_PLAY, 0, 1], [ACT_END, 0, -1]], [0.7, 0.2, 0.1])]
        loss, stats = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=1.0, confidence=0.45)
        self.assertEqual(stats['ranking_samples'], 1)
        expected = torch.nn.functional.softplus(logits[0, 1] - logits[0, 0])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)

        logits = torch.tensor([[0.4, 2.5, 0.1]], requires_grad=True)
        rows = [_row([[ACT_END, 0, -1], [ACT_PLAY, 0, 1], [ACT_ATTACK, 0, -1]], [0.8, 0.1, 0.1])]
        loss, _ = rt.ranking_auxiliary_loss(logits, mask, rows, [0], coefficient=1.0, confidence=0.45)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)

    def test_finite_gradients_reach_state_candidate_and_score(self):
        torch = _require_torch()
        net = TinyNet(torch)
        with torch.no_grad():
            net.score.bias[0] = 0.3
            net.cand_net.bias[0] = 0.4
        row = _row(
            [[ACT_PLAY, 0, 1], [ACT_PLAY, 0, 2], [ACT_ATTACK, 1, -1]],
            [0.75, 0.15, 0.10],
        )
        row['c'] = np.eye(3, dtype=np.float32)
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        metrics = rt.update(net, opt, [row], ranking_coefficient=0.2, ranking_confidence=0.45)
        self.assertEqual(metrics['ranking_samples'], 1)
        self.assertEqual(int(row['z']), 1)
        self.assertTrue(np.isfinite(metrics['policy_gradient_norms']['cand_net']))
        self.assertTrue(np.isfinite(metrics['policy_gradient_norms']['score']))
        self.assertTrue(np.isfinite(metrics['policy_gradient_norms']['state_net']))
        self.assertGreater(metrics['policy_gradient_norms']['cand_net'], 0)
        self.assertGreater(metrics['policy_gradient_norms']['score'], 0)
        self.assertGreater(metrics['policy_gradient_norms']['state_net'], 0)

    def test_value_loss_stays_on_terminal_wdl(self):
        torch = _require_torch()
        net = TinyNet(torch)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        row = _row([[ACT_END, 0, -1], [ACT_PLAY, 0, 1]], [0.8, 0.2], z=-1)
        metrics = rt.update(net, opt, [row], ranking_coefficient=0.2)
        self.assertEqual(metrics['ranking_samples'], 1)
        self.assertGreater(metrics['value_loss'], 0)
        with self.assertRaises(ValueError):
            rt.update(net, opt, [{**row, 'reward_to_go': 4}], ranking_coefficient=0.2)


if __name__ == '__main__':
    unittest.main()
