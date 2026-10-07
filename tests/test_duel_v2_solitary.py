"""User tabletop changes, 2026-10-07: 独行 and natural ultimate expiry."""
import json
import unittest

from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation, begin_turn
from app.modules.card_game.engine.duel_v2.state import hero, damage, damage_limit, move_out, reduce_max_hp
from app.modules.card_game.engine.duel_v2.summons import summon_front


class SolitaryTest(unittest.TestCase):
    def trigger(self, s):
        fixtures.SurplusReworkTest().delay(s)
        c = operation(s, 'a', 'zero')
        resolve_genesis(c)
        return c

    def test_solitary_heals_current_max_before_queued_attack_and_counts_once(self):
        for front in (None, 'zhenhong'):
            with self.subTest(front=front):
                s = fixtures.SurplusReworkTest().game()
                s['sides']['a']['front'] = front
                h = hero(s, 'a', 'zhenhong')
                h.update(hp=1, max_hp=8)
                c = self.trigger(s)
                self.assertEqual(h['hp'], 8)
                self.assertEqual(h['surplus_passive_triggers'], 1)
                self.assertEqual(h.get('extra_attacks', 0), 0)
                self.assertEqual(damage_limit(s, 'a', 'zhenhong'), 1)
                heal_event = next(e for e in s['events'] if e['type'] == 'heal')
                self.assertEqual((heal_event['target'], heal_event['amount']), ('a:zhenhong', 7))
                finish_operation(s)
                self.assertEqual(h['extra_attacks'], 1)
                self.assertEqual(h['surplus_passive_triggers'], 1)

    def test_downed_character_does_not_trigger_heal_or_count(self):
        s = fixtures.SurplusReworkTest().game()
        damage(s, 'a', 'zhenhong', 99)
        self.trigger(s)
        finish_operation(s)
        h = hero(s, 'a', 'zhenhong')
        self.assertEqual(h['hp'], 0)
        self.assertEqual(h.get('surplus_passive_triggers', 0), 0)
        self.assertEqual(h.get('extra_attacks', 0), 0)

    def test_r06_counts_trigger_even_if_attack_is_cancelled_and_survives_json(self):
        s = fixtures.SurplusReworkTest().game()
        self.trigger(s)
        h = hero(s, 'a', 'zhenhong')
        # A later part of the same operation can defeat the source by reducing
        # its life to zero; the already-fired mechanism still counts.
        reduce_max_hp(s, 'a', 'zhenhong', h['max_hp'])
        finish_operation(s)
        self.assertEqual(h.get('extra_attacks', 0), 0)
        s = json.loads(json.dumps(s))
        h = hero(s, 'a', 'zhenhong')
        h.update(hp=6, down_turns=0)
        h['extra_attacks'] = 9  # Historical attacks are not R06's counter.
        s = fixtures.SurplusReworkTest().play(s, 'R06')
        h = hero(s, 'a', 'zhenhong')
        self.assertEqual(h['shield'], 1)
        self.assertEqual(next(e for e in reversed(s['events']) if e['type'] == 'attack')['amount'], 3)

    def test_natural_expiry_hits_player_bench_front_summon_and_uses_shields(self):
        s = fixtures.SurplusReworkTest().game()
        hero(s, 'a', 'zhenhong')['energy'] = 5
        summon = summon_front(s, 'b', name='测试召唤物', attack=0, hp=4)
        hero(s, 'b', 'yi')['shield'] = 2
        damage(s, 'b', 'zero', 99)
        s['sides']['b']['shield'] = 1
        s = apply_action(s, 'a', dict(type='ultimate', character_id='zhenhong'))
        begin_turn(s, 'a')
        self.assertEqual(hero(s, 'b', summon)['hp'], 4)
        seq = s['event_seq']
        begin_turn(s, 'a')
        self.assertEqual(hero(s, 'b', summon)['hp'], 1)
        self.assertEqual(hero(s, 'b', 'yi')['hp'], 4)
        self.assertEqual(hero(s, 'b', 'jiuyuan')['hp'], 1)
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)
        self.assertEqual(s['sides']['b']['hp'], 98)
        h = hero(s, 'a', 'zhenhong')
        self.assertEqual((h['energy'], h['harmony']), (0, 1))
        self.assertEqual(h.get('surplus_passive_triggers', 0), 0)
        self.assertEqual(h.get('extra_attacks', 0), 0)
        events = [e for e in s['events'] if e['seq'] > seq]
        self.assertFalse(any(e['type'] in ('attack', 'resource') for e in events))
        self.assertEqual(len([e for e in events if e['type'] == 'damage']), 5)

    def test_early_end_never_deals_expiry_damage(self):
        for down in (False, True):
            with self.subTest(down=down):
                s = fixtures.SurplusReworkTest().game()
                hero(s, 'a', 'zhenhong')['energy'] = 5
                s = apply_action(s, 'a', dict(type='ultimate', character_id='zhenhong'))
                if down:
                    damage(s, 'a', 'zhenhong', 99)
                else:
                    move_out(s, 'a', 'zhenhong')
                begin_turn(s, 'a')
                begin_turn(s, 'a')
                self.assertFalse(any('真红终结自然结束' in e['text'] for e in s['events']))

    def test_catalog_and_public_preview_use_solitary(self):
        self.assertIn('则触发「独行」', CHARACTERS['zhenhong']['passive'])
        self.assertIn('对所有对方角色造成 3 点伤害', CHARACTERS['zhenhong']['awakened_passive'])
        self.assertIn('「独行」', CARDS['R06']['description'])
        s = fixtures.SurplusReworkTest().game()
        h = hero(s, 'a', 'zhenhong')
        h['surplus_passive_triggers'] = 2
        h['extra_attacks'] = 7
        card = fixtures.SurplusReworkTest().hold(s, 'R06')
        entry = next(e for e in observe(s, 'a')['legal_actions']
                     if e['action'].get('card_id') == card['instance_id'])
        self.assertEqual(entry['preview']['attack'], 4)
