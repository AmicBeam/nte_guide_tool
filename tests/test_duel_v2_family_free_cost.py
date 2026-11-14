"""Family growth becomes intrinsically free from global turn eleven."""
import unittest
from tests.test_duel_v2_latest_card_adjustments import game
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.flow import card_ap_payment, card_cost
from app.modules.card_game.engine.duel_v2.state import card_instance, hero


class FamilyFreeCostTest(unittest.TestCase):
    def test_boundary_and_instant_allowance_are_independent(self):
        for turn in (4, 5, 10, 11, 12):
            for used in (False, True):
                with self.subTest(turn=turn, instant_used=used):
                    s = game();s['turn'] = turn
                    team = s['sides']['a'];team.update(ap=0)
                    team['used']['instant'] = used
                    card = card_instance(s, 'NF01', 'a');team['hand'] = [card]
                    action = dict(type='play_card', card_id=card['instance_id'])
                    free = turn >= 11
                    payable = 0 if free or (turn >= 5 and not used) else 1
                    self.assertEqual(card_cost(s, 'a', card), 0 if free else 1)
                    self.assertEqual(card_ap_payment(s, 'a', card), payable)
                    self.assertEqual(action in legal_actions(s, 'a'), payable == 0)
                    shown = observe(s, 'a', include_previews=False)['sides']['a']['hand'][0]
                    self.assertEqual(shown['action_point_free'], free)
                    if payable:
                        team['ap'] = 1
                    s = apply_action(s, 'a', action)
                    self.assertEqual(s['sides']['a']['ap'], 0)
                    self.assertEqual(s['sides']['a']['used']['instant'], used or (5 <= turn < 11))
                    self.assertEqual(hero(s, 'a', 'nanali')['permanent_growth'], dict(attack=1, max_hp=1))
                    self.assertEqual(hero(s, 'a', 'nanali')['harmony'], payable)

    def test_multiple_free_uses_keep_instant_and_other_cards_still_cost_one(self):
        s = game();s['turn'] = 11;s['sides']['a']['ap'] = 0
        for _ in range(2):
            card = card_instance(s, 'NF01', 'a');s['sides']['a']['hand'] = [card]
            s = apply_action(s, 'a', dict(type='play_card', card_id=card['instance_id']))
        self.assertFalse(s['sides']['a']['used'].get('instant'))
        self.assertEqual(hero(s, 'a', 'nanali')['permanent_growth'], dict(attack=2, max_hp=2))
        self.assertEqual(card_cost(s, 'a', card_instance(s, 'N01', 'a')), 1)
        copied = card_instance(s, 'NF01', 'a')
        copied.update(character_id='zero', original_character_id='nanali')
        self.assertEqual(card_ap_payment(s, 'a', copied), 0)
