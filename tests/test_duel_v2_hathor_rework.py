"""Hathor's special rules are cleared while base battle/equipment behavior remains."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import hero, front_debuff, card_instance, attack_value
from tests import test_duel_v2_surplus_rework as fixtures


class HathorReworkTest(unittest.TestCase):
    def game(self):
        return fixtures.SurplusReworkTest().game(team=('hathor', 'xun', 'zero', 'jiuyuan'))

    def test_tactics_have_no_target_or_special_result(self):
        for key in ('H01', 'H02', 'H05', 'H06'):
            with self.subTest(key=key):
                s = self.game(); h = hero(s, 'a', 'hathor'); h['hp'] = 2
                card = card_instance(s, key, 'a'); s['sides']['a']['hand'].append(card)
                before = (h['hp'], h['energy'], len(s['sides']['a']['deck']))
                action = dict(type='play_card', card_id=card['instance_id'])
                after = apply_action(s, 'a', action)
                h = hero(after, 'a', 'hathor')
                self.assertEqual((h['hp'], h['energy'], len(after['sides']['a']['deck'])), before)
                self.assertEqual(after['phase'], 'playing')
                self.assertEqual(after['sides']['a']['ap'], 19)
                self.assertFalse(after['sides']['a'].get('extra_delay'))
                self.assertFalse(after['sides']['a']['hand'])

    def test_battles_keep_frames_but_remove_low_life_bonus_and_max_life_damage(self):
        s = self.game(); s['sides']['b']['hp'] = 3
        s = fixtures.SurplusReworkTest().play(s, 'H03')
        self.assertEqual(s['sides']['b']['hp'], 1)
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=10, max_hp=10, shield=3)
        s = fixtures.SurplusReworkTest().play(s, 'H04')
        self.assertEqual((hero(s, 'b', 'zero')['hp'], hero(s, 'b', 'zero')['max_hp'],
                          hero(s, 'b', 'zero')['shield']), (9, 10, 0))
        self.assertEqual(hero(s, 'a', 'hathor')['hp'], 4)
        self.assertTrue(any(e['type'] == 'shield' and e.get('target') == 'a:hathor'
                            and e.get('amount') == CARDS['H04']['shield'] for e in s['events']))

    def test_weapons_keep_panels_and_old_snapshots_do_not_restore_delay_override(self):
        for key in ('H07', 'H08'):
            with self.subTest(key=key):
                s = fixtures.SurplusReworkTest().play(self.game(), key)
                h = hero(s, 'a', 'hathor')
                self.assertEqual(h['shape'], key)
                self.assertEqual(attack_value(h, s, 'a'), 2 + CARDS[key]['attack'])
                h['awakened'] = True
                s = json.loads(json.dumps(s))
                hero(s, 'a', 'zero')['harmony'] = 2
                c = operation(s, 'a', 'hathor')
                apply_enter_harmony(c, 'zero', '延滞')
                finish_operation(s)
                self.assertEqual((front_debuff(s, 'b')['delay']['left'],
                                  front_debuff(s, 'b')['delay']['slow']), (1, 1))

    def test_old_entity_description_is_cleared_and_ultimate_has_no_special_buff(self):
        s = self.game(); card = card_instance(s, 'H02', 'a')
        card['description'] = '旧效果：给予能量和环合值。'
        s['sides']['a']['hand'].append(card)
        view = observe(s, 'a', include_previews=False)
        self.assertEqual(view['sides']['a']['hand'][0]['description'], '')
        for key in (f'H{i:02d}' for i in range(1, 9)):
            self.assertEqual(CARDS[key]['description'], '')
        h = hero(s, 'a', 'hathor'); h['energy'] = 5
        before = (h['hp'], attack_value(h, s, 'a'), h['harmony'])
        s = apply_action(s, 'a', dict(type='ultimate', character_id='hathor'))
        h = hero(s, 'a', 'hathor')
        self.assertEqual((h['hp'], attack_value(h, s, 'a'), h['harmony']), before)
        self.assertEqual(h['energy'], 0)
