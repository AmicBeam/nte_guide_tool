import json
import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.presentation import sanitize_public_event
from app.modules.card_game.engine.duel_v2.state import card_instance


class IloyDrawPrivacyTest(unittest.TestCase):
    def test_selected_owner_draw_is_private_and_animated(self):
        for side in ('a', 'b'):
            with self.subTest(side=side):
                state = new_game(seed=9, first_side=side, skip_mulligan=True)
                team = state['sides'][side]
                played = card_instance(state, 'Y02', side)
                drawn = card_instance(state, 'Z04', side)
                team['hand'] = [played]
                team['deck'] = [drawn]
                team['ap'] = 2
                state['events'] = []
                after = apply_action(state, side, {'type': 'play_card', 'card_id': played['instance_id'], 'target_id': f'{side}:zero'})
                own = observe(after, side)
                enemy = observe(after, 'b' if side == 'a' else 'a')
                draws = [e for e in own['presentation']['events'] if e['type'] == 'draw']
                self.assertEqual(len(draws), 1)
                self.assertEqual(draws[0]['card']['instance_id'], drawn['instance_id'])
                self.assertIn(drawn['name'], draws[0]['text'])
                for events in (enemy['events'], enemy['presentation']['events'],
                               [sanitize_public_event(e, None) for e in after['events']]):
                    draw = next(e for e in events if e['type'] == 'draw')
                    self.assertIn('零的 1 张牌', draw['text'])
                    serialized = json.dumps(events, ensure_ascii=False)
                    for secret in (drawn['name'], drawn['instance_id'], 'private_card'):
                        self.assertNotIn(secret, serialized)
                    self.assertNotIn('card', draw)
                self.assertEqual(after['sides'][side]['hand'][0]['instance_id'], drawn['instance_id'])

    def test_no_matching_card_does_not_animate_a_draw(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        played = card_instance(state, 'Y02', 'a')
        state['sides']['a']['hand'] = [played]
        state['sides']['a']['deck'] = [card_instance(state, 'N03', 'a')]
        state['events'] = []
        after = apply_action(state, 'a', {'type': 'play_card', 'card_id': played['instance_id'], 'target_id': 'a:zero'})
        self.assertFalse(any(e['type'] == 'draw' for e in after['events']))
        self.assertEqual(len(after['sides']['a']['deck']), 1)
