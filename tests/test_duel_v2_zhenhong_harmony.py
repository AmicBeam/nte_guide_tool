"""User tabletop rule: only surplus passive attacks skip base combat harmony."""
import unittest
from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.engine.duel_v2 import apply_action
from app.modules.card_game.engine.duel_v2.deferred import enqueue
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import hero, card_instance


class ZhenhongHarmonyTest(unittest.TestCase):
    def test_passive_only_skips_harmony_and_keeps_energy_share(self):
        for kind, harmony in (('surplus', 1), ('ultimate_end', 2)):
            with self.subTest(kind=kind):
                s = fixtures.SurplusReworkTest().game()
                hero(s, 'a', 'zhenhong')['harmony'] = 1
                c = operation(s, 'a', 'zhenhong')
                enqueue(c, kind)
                finish_operation(s)
                h = hero(s, 'a', 'zhenhong')
                self.assertEqual((h['harmony'], h['energy'], h['extra_attacks']), (harmony, 2 if kind == 'surplus' else 0, 1))
                self.assertEqual(hero(s, 'a', 'yi')['energy'], 1)
                event = next(e for e in reversed(s['events']) if e['type'] == 'resource')
                self.assertEqual(event['harmony_delta'], 0 if kind == 'surplus' else 1)

    def test_next_normal_or_battle_attack_still_gains_harmony(self):
        for battle in (False, True):
            with self.subTest(battle=battle):
                s = fixtures.SurplusReworkTest().game()
                c = operation(s, 'a', 'zhenhong'); enqueue(c, 'surplus'); finish_operation(s)
                if battle:
                    card = card_instance(s, 'R02', 'a'); s['sides']['a']['hand'].append(card)
                    action = dict(type='play_card', card_id=card['instance_id'])
                else:
                    action = dict(type='attack', character_id='zhenhong')
                s = apply_action(s, 'a', action)
                self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 1)
