import unittest
import numpy as np
from scripts.fit_duel_v2_policy_recovery import holdout_seeds, target_rank
from app.modules.card_game.rl.league_schema import ACT_END, ACT_PLAY


class PolicyRecoverySplitTest(unittest.TestCase):
    def test_paired_seed_stays_on_one_side(self):
        seeds = list(range(10))
        held = holdout_seeds(seeds + seeds, every=5)
        self.assertEqual(held, frozenset({0, 5}))
        fit = [seed for seed in seeds if seed not in held]
        self.assertFalse(set(fit) & held)

    def test_play_top1_reads_the_policy_target(self):
        row = dict(c=np.asarray([[ACT_END, 0], [ACT_PLAY, 1]], dtype=np.float32),
                   pi=np.asarray([0.2, 0.8]))
        self.assertTrue(target_rank(row)['play'])
        row['pi'] = np.asarray([0.8, 0.2])
        self.assertFalse(target_rank(row)['play'])
