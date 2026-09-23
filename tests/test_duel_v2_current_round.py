import unittest
from scripts.train_duel_v2_current_round import ROSTER, cross_pairs


class CurrentRoundRosterTest(unittest.TestCase):
    def test_four_public_presets_cross_every_other_team(self):
        self.assertEqual(ROSTER, ('starter', 'weave-rush', 'quick-rush', 'zhenhong'))
        pairs = cross_pairs()
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(left != right for left, right in pairs))
        self.assertNotIn('midrange', ROSTER)
        self.assertNotIn('murk', ROSTER)
