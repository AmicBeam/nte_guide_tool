"""One deck draw produces one animation event, including generated deck cards."""
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, project_event
from app.modules.card_game.engine.duel_v2.replay_log import public_replay_event
from app.modules.card_game.engine.duel_v2.state import add_hand, card_instance, draw, other


class DrawPresentationTest(unittest.TestCase):
    def game(self):
        deck = next(d for d in STARTER_DECKS if d['id'] == 'collapse')
        return new_game(seed=123, first_side='a', skip_mulligan=True,
                        decks={'a': deck, 'b': deck}, escalation=False)

    def test_in_turn_draw_has_single_movement_for_normal_derived_and_copied_cards(self):
        for side in ('a', 'b'):
            for card_id, copied in [('P01', False), ('PF01', False), ('P01', True)]:
                with self.subTest(side=side, card=card_id, copied=copied):
                    s = self.game()
                    card = card_instance(s, card_id, side, copy=copied)
                    team = s['sides'][side]
                    team['hand'] = []
                    team['deck'] = [card]
                    s['_public_board'] = capture_public_board(s)
                    seq = s['event_seq']
                    draw(s, side)
                    events = [e for e in s['events'] if e['seq'] > seq]
                    self.assertEqual([e['type'] for e in events], ['draw'])
                    self.assertEqual([c['entity_id'] for c in team['hand']], [card['entity_id']])
                    self.assertEqual(team['deck'], [])
                    self.assertEqual(events[0]['patch']['sides'][side]['hand_count'], 1)
                    self.assertEqual(events[0]['patch']['sides'][side]['deck_count'], 0)
                    own = project_event(events[0], side)
                    self.assertEqual(own['card']['instance_id'], card['instance_id'])
                    for projected in (project_event(events[0], other(side)),
                                      public_replay_event(events[0], other(side))):
                        self.assertNotIn(card['instance_id'], str(projected))
                        self.assertNotIn(card['name'], str(projected))

    def test_direct_generation_still_has_gain_event(self):
        s = self.game()
        card = card_instance(s, 'PF01', 'a')
        s['sides']['a']['hand'] = []
        seq = s['event_seq']
        add_hand(s, 'a', card)
        events = [e for e in s['events'] if e['seq'] > seq]
        self.assertEqual([e['type'] for e in events], ['gain'])
        self.assertEqual(events[0]['card']['instance_id'], card['instance_id'])

    def test_hand_limit_draw_does_not_emit_fake_gain(self):
        s = self.game()
        s['sides']['a']['hand'] = [card_instance(s, 'P01', 'a') for _ in range(10)]
        card = card_instance(s, 'PF01', 'a')
        s['sides']['a']['deck'] = [card]
        seq = s['event_seq']
        draw(s, 'a')
        events = [e for e in s['events'] if e['seq'] > seq]
        self.assertEqual([e['type'] for e in events], ['discard', 'draw'])
        self.assertEqual(len(s['sides']['a']['hand']), 10)
        self.assertEqual(s['sides']['a']['discard'][-1]['entity_id'], card['entity_id'])
