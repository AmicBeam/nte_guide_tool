import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, new_game


class OpeningDrawTest(unittest.TestCase):
    def test_both_first_seats_draw_after_mulligan(self):
        for first in ('a', 'b'):
            with self.subTest(first=first):
                second = 'b' if first == 'a' else 'a'
                state = new_game(seed=12, first_side=first)
                for side in (first, second):
                    self.assertEqual(len(state['sides'][side]['hand']), 5)
                    self.assertEqual(len(state['sides'][side]['deck']), 27)
                state = apply_action(state, first, {'type': 'mulligan', 'card_ids': []})
                self.assertEqual(len(state['sides'][first]['deck']), 27)
                state = apply_action(state, second, {'type': 'mulligan', 'card_ids': []})
                self.assertEqual(len(state['sides'][first]['deck']), 26)
                self.assertEqual(len(state['sides'][second]['deck']), 27)
                self.assertEqual(state['sides'][first]['ap'], 1)
                self.assertEqual(state['sides'][second]['shield'], 5)
                turn_event = next(event for event in state['events'] if event['type'] == 'turn')
                draws = [event for event in state['events']
                         if event['type'] == 'draw' and event['seq'] > turn_event['seq']]
                self.assertEqual(len(draws), 1)
                self.assertEqual(draws[0]['side'], first)
                state = apply_action(state, first, {'type': 'end_turn'})
                self.assertEqual(state['sides'][second]['shield'], 0)
                self.assertEqual(len(state['sides'][second]['deck']), 26)
                self.assertEqual(state['sides'][second]['ap'], 2)
                state = apply_action(state, second, {'type': 'end_turn'})
                self.assertEqual(len(state['sides'][first]['deck']), 25)
                self.assertEqual(state['sides'][first]['ap'], 2)

    def test_skipped_mulligan_still_draws_once(self):
        for first in ('a', 'b'):
            with self.subTest(first=first):
                state = new_game(seed=12, first_side=first, skip_mulligan=True)
                self.assertEqual(len(state['sides'][first]['deck']), 26)
                self.assertEqual(state['sides'][first]['ap'], 1)

    def test_player_shield_expires_only_on_owners_turn_before_start_damage(self):
        from app.modules.card_game.engine.duel_v2.state import damage
        for first in ('a', 'b'):
            with self.subTest(first=first):
                second = 'b' if first == 'a' else 'a'
                state = new_game(seed=12, first_side=first, skip_mulligan=True)
                for team in state['sides'].values():
                    team['hand'] = []  # No automatic responses in this timing fixture.
                damage(state, second, 'player', 2)
                self.assertEqual(state['sides'][second]['hp'], 30)
                self.assertEqual(state['sides'][second]['shield'], 3)
                state['sides'][first]['shield'] = 4
                state['sides'][second]['front_debuff']['burn'] = {'left': 2, 'by': first, 'stacks': 1}
                state = apply_action(state, first, {'type': 'end_turn'})
                self.assertEqual(state['sides'][second]['shield'], 0)
                self.assertEqual(state['sides'][second]['hp'], 30)
                self.assertEqual(state['sides'][first]['shield'], 4)
                state = apply_action(state, second, {'type': 'end_turn'})
                self.assertEqual(state['sides'][first]['shield'], 0)
