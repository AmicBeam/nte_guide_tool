import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.errors import RuleValidationError
from app.modules.card_game.content.duel_v2 import STARTER_DECKS,deck_uses_test_characters
from app.modules.card_game.engine.application import v2_service


class SimpleAiPresetsTest(unittest.TestCase):
    def test_random_pool_stays_public_even_for_a_test_account(self):
        expected=[p for p in STARTER_DECKS if not deck_uses_test_characters(p)]
        self.assertEqual({p['id'] for p in expected},{'starter','weave-rush','quick-rush','zhenhong','murk'})
        tester=SimpleNamespace(shaft_test_whitelisted=True)
        for player in (None, tester):
            for selected in expected:
                def choose(pool, selected=selected):
                    self.assertEqual(pool,expected)
                    return selected
                with patch('secrets.choice',side_effect=choose):
                    got=v2_service._solo_ai_deck(player)
                self.assertEqual(got['character_ids'],selected['character_ids'])
                self.assertEqual(got['card_ids'],selected['card_ids'])
                self.assertIsNot(got['card_ids'],selected['card_ids'])

    def test_public_account_can_select_murk(self):
        tester=SimpleNamespace(shaft_test_whitelisted=True)
        got=v2_service._solo_ai_deck(tester,'murk')
        self.assertEqual(got['character_ids'],['anhunqu','canhong','zaowu','adler'])
        public=SimpleNamespace(shaft_test_whitelisted=False)
        self.assertEqual(v2_service._solo_ai_deck(public,'murk')['character_ids'], got['character_ids'])
        self.assertEqual(v2_service._solo_ai_deck(None,'murk')['character_ids'], got['character_ids'])
