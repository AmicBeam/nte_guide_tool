"""Witch top-card effects reveal only the triggered entity in both views."""
import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.replay_log import public_replay_event
from tests.test_duel_v2_heiyu import game, witch, top, play, confirm


class HeiyuPresentationTest(unittest.TestCase):
    def test_all_witch_top_effects_announce_and_reveal_once_in_both_views(self):
        for card_id in ('P01', 'P02', 'P04', 'P05', 'P06'):
            with self.subTest(card_id=card_id):
                state = witch(game())
                cards = top(state, [card_id, 'Z03'])
                cards[1]['name'] = 'UNREVEALED_NEXT_TOP'
                cursor = state['event_seq']
                after = apply_action(state, 'a', {'type': 'attack', 'character_id': 'heiyu'})
                events = [e for e in after['events'] if e['seq'] > cursor]
                reveals = [e for e in events if e['type'] == 'reveal']
                self.assertEqual(len(reveals), 1)
                reveal = reveals[0]
                self.assertIs(reveal['response'], True)
                self.assertEqual(reveal['card_source'], 'deck')
                self.assertEqual(reveal['card']['entity_id'], cards[0]['entity_id'])
                self.assertLess(events.index(reveal), next(i for i,e in enumerate(events) if e['type'] == 'discard'))
                for event in events:
                    if event['type'] == 'play' and event['card']['entity_id'] == cards[0]['entity_id']:
                        self.assertIs(event['present'], False, 'release must not repeat the public reveal')
                for viewer in ('a', 'b'):
                    projected = [e for e in observe(after, viewer)['presentation']['events'] if e['seq'] > cursor]
                    replay = [public_replay_event(e, viewer) for e in events]
                    for visible in (projected, replay):
                        item = next(e for e in visible if e['seq'] == reveal['seq'])
                        self.assertTrue(item['present'])
                        self.assertIs(item['response'], True)
                        self.assertEqual(item['card_source'], 'deck')
                        self.assertEqual(item['card']['card_id'], card_id)
                        self.assertNotIn('UNREVEALED_NEXT_TOP', str(visible))

    def test_non_effect_top_stays_hidden_without_response(self):
        state = witch(game())
        top(state, ['Z03'])
        cursor = state['event_seq']
        after = apply_action(state, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertFalse(any(e['type'] == 'reveal' or e.get('response') for e in after['events'] if e['seq'] > cursor))
        self.assertEqual(after['sides']['a']['deck'][0]['card_id'], 'Z03')

    def test_basic_divination_reveals_without_witch_response_title(self):
        state = game()
        top(state, ['P02', 'Z03'])
        state = play(state, 'PF01')
        cursor = state['event_seq']
        after = confirm(state, 'P02')
        reveal = next(e for e in after['events'] if e['seq'] > cursor and e['type'] == 'reveal')
        self.assertIs(reveal['response'], False)
        self.assertEqual(reveal['card_source'], 'deck')
        release = next(e for e in after['events'] if e['seq'] > cursor and e['type'] == 'play')
        self.assertIs(release['present'], False)
