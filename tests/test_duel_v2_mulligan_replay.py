import unittest
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import apply_action
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.replay_log import public_replay_view, append_projected_events
from app.modules.card_game.engine.duel_v2.presentation import replay_public_board


class MulliganReplayTest(unittest.TestCase):
    def test_retained_cards_stay_visible_through_every_replacement(self):
        for viewer in ('a', 'b'):
            for count in range(4):
                with self.subTest(viewer=viewer, count=count):
                    initial = new_game(seed=9, first_side=viewer)
                    old = initial['sides'][viewer]['hand']
                    outgoing = [c['instance_id'] for c in old[:count]]
                    kept = {c['instance_id'] for c in old[count:]}
                    after = apply_action(initial, viewer, {'type': 'mulligan', 'card_ids': outgoing})
                    view = {'opening_board': public_replay_view(initial, viewer), 'events': []}
                    events = [e for e in after['events'] if e['seq'] > initial['event_seq']]
                    append_projected_events(view, after, viewer, events, 'A', 'B')
                    for index in range(1, len(view['events']) + 1):
                        board = replay_public_board(view['opening_board'], view['events'][:index])
                        hand = board['sides'][viewer]['hand']
                        self.assertFalse(any(c.get('hidden') for c in hand))
                        self.assertTrue(kept.issubset({c['instance_id'] for c in hand}))
                        self.assertFalse(set(outgoing) & {c['instance_id'] for c in hand})
                    enemy = 'b' if viewer == 'a' else 'a'
                    self.assertTrue(all(c.get('hidden') for c in board['sides'][enemy]['hand']))
                    self.assertEqual(len(board['sides'][viewer]['hand']), 5)
