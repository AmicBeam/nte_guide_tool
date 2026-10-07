import unittest
from types import SimpleNamespace

from app.modules.card_game.content.duel_v2.catalog import CARD_ID_ORDER, CHARACTER_ORDER, CARDS
from app.modules.card_game.content.duel_v2.effects import EFFECTS, TARGET_POLICIES, shape_attack_bonus
from app.modules.card_game.content.duel_v2.registry import KITS, copy_owner_id, decide, shape_hp_bonus, text_name
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.flow import new_game


class DuelV2HooksTest(unittest.TestCase):
    def test_kits_cover_catalog_characters_and_cards(self):
        from app.modules.card_game.content.duel_v2.tutorial import TUTORIAL_CHARACTERS, TUTORIAL_CARDS
        self.assertEqual(set(KITS) & set(CHARACTER_ORDER), set(CHARACTER_ORDER))
        self.assertTrue((set(KITS) - set(CHARACTER_ORDER)) <= set(TUTORIAL_CHARACTERS))
        self.assertEqual(text_name('zero'), '零')
        self.assertEqual(set(EFFECTS) & set(CARD_ID_ORDER), set(CARD_ID_ORDER))
        self.assertTrue((set(EFFECTS) - set(CARD_ID_ORDER)) <= set(TUTORIAL_CARDS))
        for card_id, card in CARDS.items():
            self.assertIn(card_id, EFFECTS)
            self.assertEqual(card['effect_id'], card_id)
        self.assertEqual(shape_attack_bonus('N08'), 2)
        self.assertEqual(shape_hp_bonus('N08'), 6)
        self.assertEqual(shape_attack_bonus('N07'), 2)
        self.assertEqual(shape_hp_bonus('N07'), 5)
        self.assertEqual(TARGET_POLICIES['Z01'], 'ally_front')

    def test_legacy_copy_snapshots_keep_their_saved_owner_identity(self):
        # Xun no longer generates copies; this helper hydrates pre-rework snapshots.
        self.assertEqual(copy_owner_id(), 'xun')

    def test_engine_does_not_import_legacy_event_bus(self):
        import app.modules.card_game.engine.duel_v2.combat as combat
        import app.modules.card_game.engine.duel_v2.flow as flow
        self.assertFalse(hasattr(combat, 'redeem'))
        self.assertIn('fire', flow.__dict__)
        self.assertNotIn('event_bus', combat.__name__)
        import inspect
        self.assertNotIn('event_bus', inspect.getsource(combat))
        self.assertNotIn('event_bus', inspect.getsource(flow))

    def test_decide_keeps_default_and_allows_kit_override(self):
        state = new_game(seed=1, first_side='a', skip_mulligan=True)
        context = EffectContext(state, 'a', 'nanali', {'side': 'a'})
        self.assertEqual(decide(context, 'harmony_kind', '创生'), '创生')
        self.assertIsNone(decide(context, 'harmony_kind', None))

        def fill_sync(c, current=None, **_data):
            return current if current else '同频'

        saved = KITS['nanali']
        KITS['nanali'] = SimpleNamespace(id='nanali', harmony_kind=fill_sync)
        try:
            self.assertEqual(decide(context, 'harmony_kind', None), '同频')
            self.assertEqual(decide(context, 'harmony_kind', '创生'), '创生')
            KITS['nanali'].harmony_kind = lambda c, current=None, **_data: False
            self.assertFalse(decide(context, 'harmony_kind', '创生'))
        finally:
            KITS['nanali'] = saved


if __name__ == '__main__':
    unittest.main()
