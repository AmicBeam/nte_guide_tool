import unittest

from scripts.fit_duel_v2_ranking_auxiliary import partitions, ranking_beats, scheme_passes
from scripts.fit_duel_v2_policy_recovery import holdout_seeds


class RankingFitSplitTest(unittest.TestCase):
    def test_confirmation_seeds_stay_out_of_fit_and_selection(self):
        games = [dict(seed=seed, first='a', rows=[]) for seed in range(10)]
        games += [dict(seed=seed, first='b', rows=[]) for seed in range(10)]
        parts = partitions(games)
        held_selection = holdout_seeds(range(10), every=5)
        confirmation = holdout_seeds(sorted(set(range(10)) - held_selection), every=5)
        self.assertEqual(set(parts['selection_seeds']), set(held_selection))
        self.assertEqual(set(parts['confirmation_seeds']), set(confirmation))
        train_seeds = {game['seed'] for game in parts['train']}
        self.assertFalse(train_seeds & set(parts['selection_seeds']))
        self.assertFalse(train_seeds & set(parts['confirmation_seeds']))
        self.assertFalse(set(parts['selection_seeds']) & set(parts['confirmation_seeds']))

    def test_pass_needs_identity_and_nonplay_together(self):
        baseline = dict(n=10, attack_n=3, end_n=3, target_play_n=4, exact_top=2, same_card=1,
                        category_agreement=5, attack_called_play=2, end_called_play=2,
                        always_play=False, always_end=False, never_play=False)
        better_play_only = dict(baseline, exact_top=5, same_card=3, category_agreement=4,
                                attack_called_play=3, end_called_play=1)
        self.assertFalse(ranking_beats(baseline, better_play_only))
        always_play = dict(baseline, exact_top=6, same_card=3, category_agreement=8,
                           attack_called_play=1, end_called_play=1, always_play=True)
        self.assertFalse(ranking_beats(baseline, always_play))
        better = dict(baseline, exact_top=5, same_card=3, category_agreement=8,
                      attack_called_play=1, end_called_play=1)
        self.assertTrue(ranking_beats(baseline, better))
        self.assertIsNone(scheme_passes({0.05: {'starter': True, 'weave-rush': True, 'zhenhong': False}}))
        self.assertEqual(scheme_passes({0.05: {'starter': True, 'weave-rush': True, 'zhenhong': True}}), 0.05)


if __name__ == '__main__':
    unittest.main()
