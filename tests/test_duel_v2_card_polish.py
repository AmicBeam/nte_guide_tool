"""2026-10-08 user tabletop wording, instant, targeting and auto-cost rules."""
import unittest
from pathlib import Path

from tests import test_duel_v2_surplus_rework as surplus
from tests import test_duel_v2_continuous_cast as continuous
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.state import hero, knockdowns, card_instance, front_debuff, attack_value
from app.modules.card_game.engine.duel_v2.summons import summon_front


class CardPolishTest(unittest.TestCase):
    def test_n05_first_instant_is_free_but_keeps_condition_and_target(self):
        s = continuous.ContinuousCastTest().game()
        team = s['sides']['a']
        team['ap'] = 0
        held = card_instance(s, 'N05', 'a')
        team['hand'] = [held]
        action = dict(type='play_card', card_id=held['instance_id'], target_id='b:zero')
        self.assertNotIn(action, legal_actions(s, 'a'))
        team['harmony_damage'] = True
        self.assertIn(action, legal_actions(s, 'a'))
        s = apply_action(s, 'a', action)
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertTrue(s['sides']['a']['used']['instant'])
        held = card_instance(s, 'N05', 'a')
        s['sides']['a']['hand'] = [held]
        self.assertFalse(any(a.get('card_id') == held['instance_id'] for a in legal_actions(s, 'a')))

    def test_all_response_wording_names_automatic_use_without_generic_battle_text(self):
        for card in CARDS.values():
            if card.get('response'):
                self.assertIn('自动使用', card['description'], card['id'])
                self.assertNotIn('自动打出', card['description'], card['id'])
        self.assertNotIn('进入战斗区', CARDS['N02']['description'])
        self.assertNotIn('反击方', CARDS['N02']['description'])

    def test_i06_aims_all_reduction_at_selected_enemy_esper_and_expires(self):
        s = surplus.SurplusReworkTest().game()
        s['sides']['a']['delay_turns'] = 4
        hero(s, 'b', 'zero')['base_attack'] = 10
        hero(s, 'b', 'yi').update(hp=0, down_turns=3)
        summon_front(s, 'b', name='测试召唤物', attack=2, hp=4)
        held = surplus.SurplusReworkTest().hold(s, 'I06')
        actions = [a for a in legal_actions(s, 'a') if a.get('card_id') == held['instance_id']]
        self.assertEqual({a['target_id'] for a in actions}, {'b:zhenhong', 'b:zero', 'b:jiuyuan'})
        s = apply_action(s, 'a', dict(type='play_card', card_id=held['instance_id'], target_id='b:zero'))
        self.assertEqual(attack_value(hero(s, 'b', 'zero'), s, 'b'), 6)
        self.assertEqual(attack_value(hero(s, 'b', 'jiuyuan'), s, 'b'), 3)
        begin_turn(s, 'b')
        self.assertEqual(attack_value(hero(s, 'b', 'zero'), s, 'b'), 6)
        begin_turn(s, 'b')
        self.assertEqual(attack_value(hero(s, 'b', 'zero'), s, 'b'), 10)

    def test_c07_automatic_hand_and_deck_release_costs_zero_and_manual_costs_one(self):
        for zone in ('hand', 'deck'):
            with self.subTest(zone=zone):
                s = continuous.ContinuousCastTest().game()
                team = s['sides']['a']
                team.update(ap=0, hand=[], deck=[])
                team[zone] = [card_instance(s, 'C07', 'a')]
                front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 20}
                hero(s, 'a', 'canhong')['hp'] = 0
                knockdowns(s)
                self.assertEqual(hero(s, 'a', 'canhong')['shape'], 'C07')
                self.assertEqual(team['ap'], 0)
                self.assertFalse(team['used'].get('instant'))
        s = continuous.ContinuousCastTest().game()
        team = s['sides']['a']
        team.update(ap=1, hand=[card_instance(s, 'C07', 'a')])
        held = team['hand'][0]
        s = apply_action(s, 'a', dict(type='play_card', card_id=held['instance_id']))
        self.assertEqual(s['sides']['a']['ap'], 0)

    def test_private_mechanism_words_and_s07_condition(self):
        self.assertTrue(CARDS['S07']['description'].startswith('装备时和己方回合开始时，若已施加 4 种持续伤害'))
        descriptions = [m['description'] for c in CHARACTERS.values() for m in c.get('mechanisms', [])]
        self.assertFalse(any('减半是倒计时衰减' in d or '按施加方回合结束次数递减' in d for d in descriptions))

    def test_gulang_uses_dedicated_local_portrait(self):
        s = continuous.ContinuousCastTest().play(continuous.ContinuousCastTest().game(), 'S01')
        h = hero(s, 'a', s['sides']['a']['front'])
        self.assertEqual(h['name'], '鬼郎丸')
        self.assertEqual(h['portrait'], '/static/images/characters/portrait/鬼郎丸.webp')
        self.assertTrue((Path(__file__).resolve().parents[1] / 'app' / h['portrait'].lstrip('/')).is_file())
