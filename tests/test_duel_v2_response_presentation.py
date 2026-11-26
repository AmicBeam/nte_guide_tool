"""Actual automatic responses carry public presentation metadata, not card capability."""
import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.replay_log import public_replay_event
from app.modules.card_game.engine.duel_v2.state import card_instance


class ResponsePresentationTest(unittest.TestCase):
    def game(self, side='a'):
        state = new_game(seed=9, first_side=side, skip_mulligan=True)
        for team in state['sides'].values():
            team['hand'] = []
            team['ap'] = 2
        return state

    def test_response_is_public_in_both_views_and_replay_for_both_seats(self):
        for defender, attacker in [('a', 'b'), ('b', 'a')]:
            with self.subTest(defender=defender):
                state = self.game(attacker)
                team = state['sides'][defender]
                team['front'] = 'zero'
                card = card_instance(state, 'N02', defender)
                secret = card_instance(state, 'N01', defender)
                secret['name'] = 'UNREVEALED_CARD'
                team['hand'] = [card, secret]
                cursor = state['event_seq']
                after = apply_action(state, attacker, {'type': 'attack', 'character_id': 'zero'})
                raw = next(e for e in after['events'] if e['seq'] > cursor and e['type'] == 'play')
                self.assertTrue(raw['response'])
                for viewer in ('a', 'b'):
                    event = next(e for e in observe(after, viewer)['presentation']['events'] if e['seq'] == raw['seq'])
                    replay = public_replay_event(raw, viewer)
                    for item in (event, replay):
                        self.assertIs(item['response'], True)
                        self.assertEqual(item['card']['instance_id'], card['instance_id'])
                        self.assertNotIn('UNREVEALED_CARD', str(item))

    def test_manually_playing_response_card_is_not_a_response(self):
        state = self.game()
        card = card_instance(state, 'N02', 'a')
        state['sides']['a']['hand'] = [card]
        after = apply_action(state, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        event = next(e for e in after['events'] if e['type'] == 'play')
        self.assertTrue(event['card']['response'])
        self.assertIs(event['response'], False)

    def test_unaffordable_response_has_no_play_event_and_stays_in_hand(self):
        state = self.game()
        team = state['sides']['b']
        team['front'] = 'zero'
        team['ap'] = 0
        card = card_instance(state, 'N02', 'b')
        team['hand'] = [card]
        after = apply_action(state, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertFalse(any(e['type'] == 'play' for e in after['events']))
        self.assertEqual(after['sides']['b']['hand'][0]['instance_id'], card['instance_id'])
