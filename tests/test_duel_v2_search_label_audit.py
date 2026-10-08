import unittest
from app.modules.card_game.rl.search_label_audit import (
    ap_bucket, classify_leftover, count_ends, question_rows, turn_stage,
)


def _attack(**extra):
    row = dict(kind='attack', ap_cost=1, damage=2, survives=True, counter=0,
               immune_target=False, card_type=None)
    row.update(extra)
    return row


class SearchLabelAuditTest(unittest.TestCase):
    def test_buckets_keep_two_separate_from_more(self):
        self.assertEqual([ap_bucket(value) for value in (0, 1, 2, 3, 5)], ['0', '1', '2', '3+', '3+'])
        self.assertEqual(turn_stage(1, 1), 'first_opening')
        self.assertEqual(turn_stage(1, 2), 'second_opening')

    def test_safe_combat_is_suspicious_and_not_a_verdict(self):
        label, reasons = classify_leftover([_attack()])
        self.assertEqual(label, 'suspicious')
        self.assertEqual(reasons, ['safe_immediate_combat'])

    def test_immunity_or_knockdown_can_explain_the_only_spend(self):
        immune, immune_reasons = classify_leftover([_attack(damage=0, immune_target=True)])
        lethal, lethal_reasons = classify_leftover([_attack(survives=False, counter=4)])
        self.assertEqual(immune, 'explained')
        self.assertIn('immune_front', immune_reasons)
        self.assertEqual(lethal, 'explained')
        self.assertIn('counter_knockdown', lethal_reasons)

    def test_free_cards_and_unsimulated_cards_stay_distinct(self):
        free, free_reasons = classify_leftover([dict(kind='play_card', ap_cost=0, card_type='tactic')])
        card, card_reasons = classify_leftover([dict(kind='play_card', ap_cost=1, card_type='tactic')])
        trade, _ = classify_leftover([_attack(counter=1)])
        self.assertEqual((free, free_reasons), ('explained', ['only_nonspending_action']))
        self.assertEqual(card, 'insufficient')
        self.assertIn('unsimulated_card', card_reasons)
        self.assertEqual(trade, 'insufficient')

    def test_a_card_prevents_an_immunity_explanation(self):
        label, reasons = classify_leftover([
            _attack(damage=0, immune_target=True),
            dict(kind='play_card', ap_cost=1, card_type='awaken'),
        ])
        self.assertEqual(label, 'insufficient')
        self.assertIn('unsimulated_card', reasons)

    def test_policy_preference_is_not_a_second_turn(self):
        rows = [
            dict(corpus='training', game_id='g', step=3, phase='playing', selected='end_turn', ap=2, has_attack_or_play=True),
            dict(corpus='training', game_id='g', step=2, phase='playing', selected='play_card', ap=2, pi_end_first=True, has_attack_or_play=True),
            dict(corpus='argmax', game_id='g', step=3, phase='playing', selected='end_turn', ap=2, has_attack_or_play=True),
        ]
        self.assertEqual(len(count_ends(rows)), 2)
        self.assertEqual(len(question_rows(rows)), 1)
        with self.assertRaises(ValueError):
            count_ends(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main()
