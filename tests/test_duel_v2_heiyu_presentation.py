"""Both forms reveal bottom-card benefits without a response announcement."""
import json
import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.replay_log import public_replay_event
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from tests.test_duel_v2_heiyu import game, witch, top, play, confirm, harmony_inspection


class HeiyuPresentationTest(unittest.TestCase):
    def test_all_witch_bottom_effects_reveal_once_without_response_in_both_views(self):
        for card_id in ('P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08'):
            with self.subTest(card_id=card_id):
                state = witch(game())
                cards = top(state, [card_id, 'Z03'])
                cards[1]['name'] = 'UNREVEALED_NEXT_TOP'
                state = harmony_inspection(state)
                self.assertEqual(state['phase'], 'choice')
                cursor = state['event_seq']
                opening = capture_public_board(state)
                after = confirm(state, card_id)
                while after['phase'] == 'choice':
                    after = confirm(json.loads(json.dumps(after)))
                events = [e for e in after['events'] if e['seq'] > cursor]
                reveals = [e for e in events if e['type'] == 'reveal']
                self.assertEqual(len(reveals), 1)
                reveal = reveals[0]
                self.assertIs(reveal['response'], False)
                self.assertEqual(reveal['card_source'], 'deck')
                self.assertEqual(reveal['card']['entity_id'], cards[0]['entity_id'])
                if card_id == 'P05':
                    self.assertIn('当前为1', reveal['card']['description'])
                for event in events:
                    if event['type'] == 'play' and event['card']['entity_id'] == cards[0]['entity_id']:
                        self.assertIs(event['present'], False, 'release must not repeat the public reveal')
                for viewer in ('a', 'b'):
                    projected = [e for e in observe(after, viewer)['presentation']['events'] if e['seq'] > cursor]
                    replay = [public_replay_event(e, viewer) for e in events]
                    for visible in (projected, replay):
                        item = next(e for e in visible if e['seq'] == reveal['seq'])
                        self.assertTrue(item['present'])
                        self.assertIs(item['response'], False)
                        self.assertEqual(item['card_source'], 'deck')
                        self.assertEqual(item['card']['card_id'], card_id)
                        if viewer == 'b' or card_id != 'P03':
                            self.assertNotIn('UNREVEALED_NEXT_TOP', str(visible))
                # Each choice has its own patches; verify the final state's public patch stream.
                self.assertEqual(replay_public_board(opening, events), capture_public_board(after))

    def test_unbottomed_cards_stay_hidden_and_sortie_is_not_repeated(self):
        state = witch(game())
        top(state, ['P01', 'Z03'])
        cursor = state['event_seq']
        state = harmony_inspection(state)
        after = confirm(state)
        self.assertFalse(any(e['type'] == 'reveal' or e.get('response') for e in after['events'] if e['seq'] > cursor))
        self.assertEqual([c['card_id'] for c in after['sides']['a']['deck']], ['P01', 'Z03'])
        after = apply_action(after, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertEqual(after['sides']['b']['hp'], 95)
        self.assertEqual(sum(e['type'] == 'attack' for e in after['events'] if e['seq'] > cursor), 1)

    def test_basic_divination_reveals_released_card_without_response_title(self):
        state = game()
        top(state, ['P02', 'Z03'])
        state = harmony_inspection(state)
        cursor = state['event_seq']
        after = confirm(state, 'P02')
        reveal = next(e for e in after['events'] if e['seq'] > cursor and e['type'] == 'reveal')
        self.assertIs(reveal['response'], False)
        self.assertEqual(reveal['card_source'], 'deck')
        release = next(e for e in after['events'] if e['seq'] > cursor and e['type'] == 'play')
        self.assertIs(release['present'], False)
