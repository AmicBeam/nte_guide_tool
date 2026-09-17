import unittest
from unittest.mock import patch
from app.modules.card_game.content.duel_v2 import STARTER_DECKS,deck_uses_test_characters
from app.modules.card_game.engine.application import v2_service


class SimpleAiPresetsTest(unittest.TestCase):
    def test_only_public_presets_are_candidates_and_both_can_be_chosen(self):
        expected=[p for p in STARTER_DECKS if not deck_uses_test_characters(p)]
        self.assertEqual({p['id'] for p in expected},{'starter','weave-rush'})
        for selected in expected:
            def choose(pool):
                self.assertEqual(pool,expected)
                return selected
            with patch('secrets.choice',side_effect=choose):
                got=v2_service._solo_ai_deck()
            self.assertEqual(got['character_ids'],selected['character_ids'])
            self.assertEqual(got['card_ids'],selected['card_ids'])
            self.assertIsNot(got['card_ids'],selected['card_ids'])
