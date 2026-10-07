import unittest
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, diff_public
from app.modules.card_game.engine.duel_v2.state import card_instance


class DuelV2ActionPresentationBoundaryTest(unittest.TestCase):
    def fresh(self):
        return new_game(seed=9, first_side='a', skip_mulligan=True)

    def first_event_patch(self, view):
        events = view['presentation']['events']
        self.assertTrue(events)
        return events[0].get('patch') or {}

    def test_old_archive_first_action_keeps_opening_diff(self):
        state = self.fresh()
        state.pop('_public_board', None)
        stale = capture_public_board(state)
        stale['sides']['b']['hp'] = 99
        state['_public_board'] = stale
        after = apply_action(state, 'a', {'type': 'end_turn'})
        view = observe(after, 'a')
        first_patch = self.first_event_patch(view)
        self.assertNotEqual(first_patch.get('sides', {}).get('b', {}).get('hp'), 99)
        self.assertIn('ap', str(view['presentation']['events'][-1].get('patch') or {}))

    def test_end_turn_trailing_ap_refresh_is_patched(self):
        state = self.fresh()
        before_board = capture_public_board(state)
        after = apply_action(state, 'a', {'type': 'end_turn'})
        view = observe(after, 'b')
        patches = [event.get('patch') or {} for event in view['presentation']['events']
                   if event['seq'] > state['event_seq']]
        merged_ap = None
        for patch in patches:
            ap = ((patch.get('sides') or {}).get('b') or {}).get('ap')
            if ap is not None:
                merged_ap = ap
        self.assertEqual(merged_ap, after['sides']['b']['ap'])
        trailing = diff_public(before_board, capture_public_board(after))
        self.assertTrue(trailing)

    def test_pending_choice_trailing_state_is_patched(self):
        state = self.fresh()
        state['sides']['a']['ap'] = 2
        card = card_instance(state, 'J01', 'a')
        state['sides']['a']['hand'] = [card]
        after = apply_action(state, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual(after['phase'], 'choice')
        view = observe(after, 'a')
        last_patch = view['presentation']['events'][-1].get('patch') or {}
        self.assertEqual(last_patch.get('phase'), 'choice')
        self.assertIsNone(observe(state, 'a')['pending_choice'])
        self.assertTrue(view['pending_choice'])

    def test_observe_does_not_mutate_input_state(self):
        state = self.fresh()
        snapshot = deepcopy(state)
        observe(state, 'a')
        apply_action(state, 'a', {'type': 'end_turn'})
        self.assertEqual(state['event_seq'], snapshot['event_seq'])
        self.assertEqual(state['version'], snapshot['version'])
        self.assertEqual(state['sides']['a']['ap'], snapshot['sides']['a']['ap'])


if __name__ == '__main__':
    unittest.main()
