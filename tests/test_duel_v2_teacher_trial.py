import unittest

from scripts.fit_duel_v2_cross_teacher import selection_pass


class TeacherTrialTest(unittest.TestCase):
    def test_selection_requires_action_gain_without_new_false_plays(self):
        base = dict(n=100, legal_play=80, exact_action=20, category_agreement=60,
                    card_agreement=10, attack_called_play=4, end_called_play=3,
                    teacher_play=32, legal_play_top1=.10, collapse=False)
        improved = dict(base, exact_action=25, category_agreement=65,
                        card_agreement=12, attack_called_play=3,
                        end_called_play=2, legal_play_top1=.40)
        self.assertTrue(selection_pass(base, improved))
        self.assertFalse(selection_pass(base, dict(improved, attack_called_play=5)))
        self.assertFalse(selection_pass(base, dict(improved, card_agreement=9)))
        self.assertFalse(selection_pass(base, dict(improved, collapse=True)))


if __name__ == '__main__':
    unittest.main()
