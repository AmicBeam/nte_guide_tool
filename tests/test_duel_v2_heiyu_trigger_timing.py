"""A renewed application can cause dissonance but is not a harmony trigger."""
import unittest
from tests.test_duel_v2_heiyu import game, witch, top, confirm
from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony, apply_dark_star, resolve_harmony_effect
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation, tick_front_debuff
from app.modules.card_game.engine.duel_v2.state import front_debuff, hero


class HeiyuTriggerTimingTest(unittest.TestCase):
    def test_real_triggers_divine_in_both_forms(self):
        for magical in (False, True):
            for kind in ('浸染', '黯星'):
                with self.subTest(witch=magical, kind=kind):
                    s = witch(game()) if magical else game()
                    top(s, ['Z01', 'Z02', 'Z03', 'Z04'])
                    apply_enter_harmony(operation(s, 'a', 'heiyu'), None, kind,
                                        pay=False, as_support=False)
                    finish_operation(s)
                    self.assertEqual(s['phase'], 'choice')
                    self.assertEqual(s['sides']['a']['heiyu_divinations'], 1)
                    s = confirm(s)
                    self.assertEqual(s['phase'], 'playing')

    def test_renewed_star_does_not_divine_or_count_but_does_dissonance(self):
        s = game();top(s, ['Z01', 'Z02', 'Z03', 'Z04'])
        apply_enter_harmony(operation(s, 'a', 'heiyu'), None, '黯星',
                            pay=False, as_support=False)
        finish_operation(s);s = confirm(s)
        zone = front_debuff(s, 'b')
        self.assertEqual(zone['star']['renewals'], 1)
        zone['burn'] = {'by':'a', 'left':2}
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=40, max_hp=40, shield=0)
        deck = [c['entity_id'] for c in s['sides']['a']['deck']]
        tick_front_debuff(s, 'b');tick_front_debuff(s, 'b')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 38)
        self.assertEqual(hero(s, 'b', 'zero')['flags']['collapse_count'], 1)
        self.assertEqual(zone['star']['left'], 2)
        self.assertEqual(zone['star']['renewals'], 0)
        finish_operation(s)
        self.assertEqual(s['phase'], 'playing')
        self.assertIsNone(s['pending_choice'])
        self.assertFalse(s.get('deferred_actions'))
        self.assertEqual(s['sides']['a']['heiyu_divinations'], 1)
        self.assertEqual(s['sides']['a']['dark_star_count'], 1)
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['deck']], deck)
        tick_front_debuff(s, 'b');tick_front_debuff(s, 'b')
        self.assertNotIn('star', zone)
        self.assertEqual(s['sides']['a']['dark_star_count'], 1)

    def test_pure_applications_never_open_divination(self):
        for kind in ('浸染', '黯星'):
            with self.subTest(kind=kind):
                s = game();c = operation(s, 'a', 'heiyu')
                resolve_harmony_effect(c, kind);finish_operation(s)
                self.assertEqual(s['phase'], 'playing')
                self.assertFalse(s['sides']['a'].get('heiyu_divinations'))
                self.assertEqual(s['sides']['a'].get('dark_star_count', 0), 0)
