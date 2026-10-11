"""Snapshot-scoped entity action indexing preserves public action identities."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2 import entities, projection


class ActionCardIndexTest(unittest.TestCase):
    def test_shared_index_matches_standalone_for_every_legal_entry(self):
        for skip in (False, True):
            state = new_game(seed=23, skip_mulligan=skip)
            before = deepcopy(state)
            for side in ('a', 'b'):
                actions = projection.legal_actions(state, side)
                index = entities.action_card_index(state)
                saved = deepcopy(index)
                for action in actions:
                    self.assertEqual(entities.entity_action(state, side, action),
                                     entities.entity_action(state, side, action, card_index=index))
                self.assertEqual(index, saved)
            self.assertEqual(state, before)

    def test_observation_reuses_one_index_and_matches_fresh_index_reference(self):
        convert = entities.entity_action
        for skip in (False, True):
            state = new_game(seed=29, skip_mulligan=skip)
            before = deepcopy(state)
            for side in ('a', 'b'):
                for previews in (False, True):
                    # Reference keeps standalone conversion per action.
                    def fresh(s, viewer, action, **ignored):
                        return convert(s, viewer, action)
                    with patch.object(projection, 'entity_action', fresh):
                        expected = projection.observe(state, side, include_previews=previews)
                    with patch.object(projection, 'action_card_index', wraps=entities.action_card_index) as build:
                        actual = projection.observe(state, side, include_previews=previews)
                        self.assertEqual(build.call_count, 1)
                    self.assertEqual(actual, expected)
            self.assertEqual(state, before)

    def test_pending_cards_and_nested_selections_keep_identity_and_no_aliases(self):
        state = new_game(seed=31)
        card = deepcopy(state['sides']['a']['deck'][0])
        card['instance_id'] = 'pending-only'
        card['entity_id'] = 'pending-entity'
        state['pending_choice'] = {'cards': [card]}
        index = entities.action_card_index(state)
        actions = [dict(type='choose', choice_id='pending-only', target_id='pending-only'),
                   dict(type='mulligan', card_ids=['pending-only']),
                   dict(type='choose_deck_order', card_ids=['pending-only'], bottom_card_ids=['pending-only']),
                   dict(type='attack', character_id=next(iter(state['sides']['a']['characters'])), target_id='b:player')]
        for action in actions:
            original = deepcopy(action)
            actual = entities.entity_action(state, 'a', action, card_index=index)
            self.assertEqual(actual, entities.entity_action(state, 'a', action))
            self.assertEqual(action, original)
        self.assertEqual(index['pending-only']['entity_id'], 'pending-entity')
        # A later snapshot builds a fresh identity map rather than caching globally.
        state['pending_choice'] = None
        self.assertNotIn('pending-only', entities.action_card_index(state))


if __name__ == '__main__':
    unittest.main()
