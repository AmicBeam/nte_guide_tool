"""User tabletop changes, 2026-10-07: 独行 and natural ultimate expiry."""
import json
import unittest

from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.combat import resolve_genesis
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation, begin_turn
from app.modules.card_game.engine.duel_v2.state import hero, damage, damage_limit, move_out, reduce_max_hp
from app.modules.card_game.engine.duel_v2.summons import summon_front


class SolitaryTest(unittest.TestCase):
    def test_solitary_support_only_comes_from_actual_owner_turn_entry_harmony(self):
        cases = [(None, 0, 'a', False), ('zhenhong', 2, 'a', False),
                 ('jiuyuan', 0, 'a', False), ('jiuyuan', 2, 'a', True),
                 ('jiuyuan', 2, 'b', False)]
        for front, harmony, active, support in cases:
            with self.subTest(front=front, harmony=harmony, active=active):
                s = fixtures.SurplusReworkTest().game(team=('zhenhong', 'xun', 'zero', 'jiuyuan'))
                s['active_side'] = active
                s['sides']['a']['front'] = front
                hero(s, 'a', 'jiuyuan')['harmony'] = harmony
                s['sides']['b']['front'] = 'zero'
                hero(s, 'b', 'zero').update(hp=50, max_hp=50)
                seq = s['event_seq']
                c = operation(s, 'a', 'zhenhong')
                fire(c, 'on_surplus', actor_only=True)
                finish_operation(s)
                events = [e for e in s['events'] if e['seq'] > seq]
                attack = next(e for e in events if e['type'] == 'attack')
                self.assertEqual(attack['counter'], 2)
                self.assertFalse(any(e.get('counter_immunity') == 'support' for e in events))
                self.assertEqual(hero(s, 'a', 'xun').get('arid', 0), int(support))
                self.assertEqual(hero(s, 'b', 'zero')['flags'].get('collapse_count', 0), int(support))
                self.assertEqual(hero(s, 'a', 'zhenhong')['hp'], 5)
                self.assertEqual(hero(s, 'a', 'jiuyuan')['harmony'], 0 if support else harmony)
                self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 1)

    def test_normal_entry_harmony_still_blocks_counter(self):
        s = fixtures.SurplusReworkTest().game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=50, max_hp=50)
        s = apply_action(s, 'a', dict(type='attack', character_id='zhenhong'))
        attack = next(e for e in reversed(s['events']) if e['type'] == 'attack')
        self.assertEqual(attack['counter'], 0)
        self.assertTrue(any(e.get('counter_immunity') == 'support' for e in s['events']))
        self.assertEqual(hero(s, 'a', 'zhenhong')['hp'], 6)

    def test_solitary_does_not_restore_counter_for_collapsed_defender(self):
        from app.modules.card_game.engine.duel_v2.state import add_collapse
        s = fixtures.SurplusReworkTest().game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=50, max_hp=50)
        add_collapse(s, 'b', 'zero', 5, by='a')
        c = operation(s, 'a', 'zhenhong')
        fire(c, 'on_surplus', actor_only=True)
        finish_operation(s)
        attack = next(e for e in reversed(s['events']) if e['type'] == 'attack')
        self.assertEqual(attack['counter'], 0)
        self.assertFalse(any(e.get('counter_immunity') == 'support' for e in s['events']))

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
        self.assertNotIn('视为援护技', CHARACTERS['zhenhong']['mechanisms'][0]['description'])
        s = fixtures.SurplusReworkTest().game()
        h = hero(s, 'a', 'zhenhong')
        h['surplus_passive_triggers'] = 2
        h['extra_attacks'] = 7
        card = fixtures.SurplusReworkTest().hold(s, 'R06')
        entry = next(e for e in observe(s, 'a')['legal_actions']
                     if e['action'].get('card_id') == card['instance_id'])
        self.assertEqual(entry['preview']['attack'], 4)
