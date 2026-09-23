import unittest
from collections import Counter
from scripts.train_duel_v2_current_round import ROSTER, cross_pairs, learning_schedule


class CurrentRoundRosterTest(unittest.TestCase):
    def test_five_presets_get_equal_games_per_cycle(self):
        self.assertEqual(ROSTER, ('starter', 'weave-rush', 'quick-rush', 'zhenhong'))
        pairs = cross_pairs()
        self.assertEqual(len(pairs), 12)
        self.assertNotIn('midrange', ROSTER)
        appearances = Counter()
        for left, right in learning_schedule():
            appearances[left] += 1
            if left != 'murk':
                appearances[right] += 1
        self.assertEqual(set(appearances), {*ROSTER, 'murk'})
        self.assertEqual(len(set(appearances.values())), 1)
