"""Elyms's test-only named placeholders and authored passive mechanisms."""
import json
from pathlib import Path
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog, validate_deck
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.combat import resolve_harmony_effect
from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
from app.modules.card_game.engine.duel_v2.flow import operation, tick_front_debuff, begin_turn
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board
from app.modules.card_game.engine.duel_v2.state import hero, front_debuff, damage, attack_value
from app.modules.card_game.engine.duel_v2.modifiers import resistance_values
from tests.test_duel_v2_latest_card_adjustments import game, play


class ElymsTest(unittest.TestCase):
    def game(self):
        return game(('elyms', 'zero', 'nanali', 'hathor'))

    def test_catalog_access_names_and_three_empty_weapons(self):
        self.assertNotIn('elyms', {c['id'] for c in get_catalog(include_test_characters=False)['characters']})
        self.assertIn('elyms', {c['id'] for c in get_catalog(include_test_characters=True)['characters']})
        self.assertEqual((CHARACTERS['elyms']['attack'], CHARACTERS['elyms']['max_hp']), (2, 5))
        cards = [c for c in CARDS.values() if c['character_id'] == 'elyms']
        self.assertEqual([c['name'] for c in cards], [
            '404 遥望一座孤岛', '200 节点彼此相连', '504 访问响应超时',
            '302 祈愿重新定向', '304 今日一如往昔', '千年虫之谜', '花海追想', '微风精灵'])
        self.assertEqual([c['type'] for c in cards], ['tactic'] * 5 + ['form'] * 3)
        self.assertTrue(all(c['description'] == '' for c in cards))
        for field in ('avatar', 'portrait'):
            self.assertTrue((Path(__file__).resolve().parents[1] / 'app' / CHARACTERS['elyms'][field].lstrip('/')).is_file())
        deck = dict(id='elyms-test', name='曳尔沐测试', character_ids=['elyms', 'zero', 'nanali', 'hathor'],
                    card_ids=[key for cid in ('elyms', 'zero', 'nanali', 'hathor') for key, c in CARDS.items()
                              if c['character_id'] == cid and not c.get('derived')])
        self.assertEqual(len(validate_deck(deck)['card_ids']), 32)

    def test_cards_only_pay_or_equip_no_signal_or_extra_effect(self):
        for index in range(1, 9):
            with self.subTest(index=index):
                s = self.game()
                s = play(s, f'MU{index:02d}')
                h = hero(s, 'a', 'elyms')
                self.assertEqual(h['signal_level'], 0)
                self.assertEqual(h['harmony'], 1)
                self.assertEqual(h['energy'], 0)
                self.assertIsNone(s['sides']['a']['front'])
                self.assertEqual(s['sides']['b']['hp'], 100)
                self.assertEqual(h['shape'], f'MU{index:02d}' if index >= 6 else None)
                self.assertEqual(attack_value(h, s, 'a'), 2)

    def test_signal_is_public_on_both_views_and_replay_and_survives_down(self):
        s = self.game()
        hero(s, 'a', 'elyms')['signal_level'] = 2
        damage(s, 'a', 'elyms', 100)
        s = json.loads(json.dumps(s)); hydrate_entities(s)
        for view in (observe(s, 'a', include_previews=False), observe(s, 'b', include_previews=False), capture_public_board(s)):
            h = next(c for c in view['sides']['a']['characters'] if c['id'] == 'elyms')
            self.assertEqual(h['signal_level'], 2)

    def test_delay_reapplies_once_after_json_without_harmony_payment(self):
        s = self.game()
        resolve_harmony_effect(operation(s, 'a', 'zero'), '延滞')
        self.assertEqual(front_debuff(s, 'b')['delay']['renewals'], 1)
        s = json.loads(json.dumps(s)); hydrate_entities(s)
        s['turn'] += 2
        tick_front_debuff(s, 'a')
        renewed = front_debuff(s, 'b')['delay']
        self.assertEqual(renewed['left'], 1)
        self.assertEqual(renewed['slow_turn'], s['turn'])
        self.assertEqual(renewed.get('renewals', 0), 0)
        self.assertFalse(s['sides']['a'].get('support_count'))
        tick_front_debuff(s, 'a')
        self.assertNotIn('delay', front_debuff(s, 'b'))

    def test_no_new_renewal_while_down_and_pause_keeps_pending_renewal(self):
        s = game(('elyms', 'zero', 'nanali', 'yi'))
        hero(s, 'a', 'yi')['beast_fangs'] = 2
        resolve_harmony_effect(operation(s, 'a', 'zero'), '延滞')
        tick_front_debuff(s, 'a')
        self.assertEqual(front_debuff(s, 'b')['delay']['renewals'], 1)
        self.assertEqual(front_debuff(s, 'b')['delay']['left'], 1)
        damage(s, 'a', 'elyms', 100)
        front_debuff(s, 'b').clear()
        resolve_harmony_effect(operation(s, 'a', 'zero'), '延滞')
        self.assertNotIn('renewals', front_debuff(s, 'b')['delay'])

    def test_surplus_chooses_one_role_light_only_refreshes_and_expires(self):
        s = self.game()
        for cid in s['sides']['b']['characters']:
            damage(s, 'b', cid, 100)
        c = operation(s, 'a', 'zero')
        fire(c, GameEvent.V2_SURPLUS)
        fire(c, GameEvent.V2_SURPLUS)
        self.assertEqual(resistance_values(s['sides']['b']).get('光'), -1)
        self.assertEqual(resistance_values(s['sides']['b']).get('灵', 0), 0)
        marker = next(m for m in observe(s, 'a', include_previews=False)['sides']['b']['effect_markers']
                      if m['name'] == '光抗性 -1')
        self.assertIn('曳尔沐下个己方回合开始', marker['timing'])
        s['operation'] = None
        begin_turn(s, 'a')
        self.assertEqual(resistance_values(s['sides']['b']).get('光', 0), 0)

    def test_ultimate_consumes_signal_buffs_team_and_expires(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                s = self.game(); h = hero(s, 'a', 'elyms')
                h.update(signal_level=level, energy=5)
                s = apply_action(s, 'a', dict(type='ultimate', character_id='elyms'))
                h = hero(s, 'a', 'elyms')
                self.assertEqual((h['signal_level'], h['energy']), (0, 0))
                for cid in s['sides']['a']['characters']:
                    self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), CHARACTERS[cid]['attack'] + level)
                begin_turn(s, 'a'); begin_turn(s, 'a')
                self.assertEqual(attack_value(h, s, 'a'), 2)
