import unittest

from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.state import add_hand, card_instance


class GeneratedGainTest(unittest.TestCase):
    def test_first_turn_family_has_one_public_gain_with_card(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        events = [e for e in state['events'] if e['type'] == 'gain' and '家族壮大' in e['text']]
        self.assertEqual(len(events), 1)
        card = events[0]['card']
        self.assertEqual(card['card_id'], 'NF01')
        self.assertTrue(card['derived'])
        self.assertIn(card['instance_id'], [c['instance_id'] for c in state['sides']['a']['hand']])

    def test_all_generated_cards_get_public_gain_only_when_added(self):
        for card_id, copy in [('NF01', False), ('N03', True)]:
            for full in (False, True):
                with self.subTest(card_id=card_id, full=full):
                    state = new_game(seed=9, first_side='a', skip_mulligan=True)
                    state['sides']['a']['hand'] = [card_instance(state, 'N03', 'a') for _ in range(10 if full else 0)]
                    state['events'] = []
                    card = card_instance(state, card_id, 'a', copy=copy)
                    add_hand(state, 'a', card)
                    gains = [e for e in state['events'] if e['type'] == 'gain']
                    self.assertEqual(len(gains), 0 if full else 1)
                    if gains:
                        self.assertEqual(gains[0]['card']['instance_id'], card['instance_id'])


if __name__ == '__main__':
    unittest.main()
