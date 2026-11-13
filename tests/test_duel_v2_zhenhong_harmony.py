"""User tabletop rule: only surplus passive attacks skip base combat harmony."""
import unittest
from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.engine.duel_v2 import apply_action
from app.modules.card_game.engine.duel_v2.deferred import enqueue
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import hero, card_instance


class ZhenhongHarmonyTest(unittest.TestCase):
    def test_passive_only_skips_harmony_and_keeps_energy_share(self):
        s = fixtures.SurplusReworkTest().game()
        hero(s, 'a', 'zhenhong')['harmony'] = 1
        c = operation(s, 'a', 'zhenhong')
        enqueue(c, 'surplus')
        finish_operation(s)
        h = hero(s, 'a', 'zhenhong')
        self.assertEqual((h['harmony'], h['energy']), (1, 0))
        self.assertEqual(hero(s, 'a', 'yi')['energy'], 0)
        self.assertFalse(any(e['type']=='attack' for e in s['events']))
        self.assertEqual(s['sides']['b']['hp'],98)

    def test_next_normal_or_battle_attack_still_gains_harmony(self):
        for battle in (False, True):
            with self.subTest(battle=battle):
                s = fixtures.SurplusReworkTest().game()
                c = operation(s, 'a', 'zhenhong'); enqueue(c, 'surplus'); finish_operation(s)
                self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 0)
                if battle:
                    card = card_instance(s, 'R02', 'a'); s['sides']['a']['hand'].append(card)
                    action = dict(type='play_card', card_id=card['instance_id'])
                else:
                    action = dict(type='attack', character_id='zhenhong')
                s = apply_action(s, 'a', action)
                self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 3)
