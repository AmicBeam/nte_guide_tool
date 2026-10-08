import unittest
import numpy as np
from app.modules.card_game.rl.policy_margin import margin_bins, play_margin


class PolicyMarginTest(unittest.TestCase):
    def test_margin_is_best_card_minus_best_other(self):
        scores = np.asarray([0.10, 0.11, 0.10])
        kinds = np.asarray([2, 1, 2])
        self.assertAlmostEqual(play_margin(scores, kinds, 2), -0.01)
        self.assertIsNone(play_margin(np.asarray([1.0]), np.asarray([2]), 2))

    def test_near_zero_negative_margins_are_counted_apart_from_ties(self):
        summary = margin_bins([-0.02, -0.005, 0.0, 0.004, 0.2])
        counted = {item['label']: item['count'] for item in summary['bins']}
        self.assertEqual(summary['n'], 5)
        self.assertEqual(counted['[-0.01, 0)'], 1)
        self.assertEqual(counted['0'], 1)
        self.assertEqual(counted['(0, 0.01)'], 1)
        self.assertEqual(sum(item['count'] for item in summary['bins']), 5)


if __name__ == '__main__':
    unittest.main()
