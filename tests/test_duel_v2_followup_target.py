"""Combat followups keep the resolved attack target; they never retarget on a KO."""
import unittest

from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, move_out
from app.modules.card_game.content.duel_v2.characters.common import deal_combat_followup


class FollowupTargetTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=52, first_side='a', skip_mulligan=True, escalation=False)
        for team in s['sides'].values():
            team.update(hand=[], ap=20, shield=0)
        return s

    def strike(self, s, card=False):
        if card:
            instance = card_instance(s, 'N03', 'a')
            s['sides']['a']['hand'].append(instance)
            action = {'type': 'play_card', 'card_id': instance['instance_id']}
        else:
            action = {'type': 'attack', 'character_id': 'nanali'}
        return apply_action(s, 'a', action)

    def test_attack_knockout_never_redirects_card_ultimate_or_weapon_followup(self):
        for card, ultimate, shape in ((True, False, None), (False, True, None),
                                      (False, False, 'N08'), (True, True, 'N08')):
            with self.subTest(card=card, ultimate=ultimate, shape=shape):
                s = self.game(); s['sides']['b']['front'] = 'zero'
                hero(s, 'b', 'zero')['hp'] = 1
                hero(s, 'a', 'nanali').update(awakened=ultimate, shape=shape)
                s = self.strike(s, card)
                self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)
                self.assertEqual(s['sides']['b']['hp'], 30)
                self.assertFalse(any(e['type'] == 'followup' for e in s['events']))

    def test_followup_can_knock_out_original_target_without_hitting_player(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 4
        s = self.strike(s, True)
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)
        self.assertEqual(s['sides']['b']['hp'], 30)
        self.assertEqual([e['target'] for e in s['events'] if e['type'] == 'followup'], ['b:zero'])

    def test_original_player_target_still_receives_followup(self):
        s = self.strike(self.game(), True)
        self.assertEqual(s['sides']['b']['hp'], 26)
        self.assertEqual([e['target'] for e in s['events'] if e['type'] == 'followup'], ['b:player'])

    def test_followup_absorbed_by_original_targets_shield(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['shield'] = 5
        hp = hero(s, 'b', 'zero')['hp']
        s = self.strike(s, True)
        self.assertEqual((hero(s, 'b', 'zero')['hp'], hero(s, 'b', 'zero')['shield']), (hp, 1))
        self.assertEqual(s['sides']['b']['hp'], 30)

    def test_departed_target_is_not_replaced_by_new_front_or_player(self):
        for replacement in (None, 'jiuyuan'):
            with self.subTest(replacement=replacement):
                s = self.game(); s['sides']['b']['front'] = 'zero'
                hero(s, 'a', 'nanali')['awakened'] = True
                operation = {'side': 'a', 'last_attack_target': ['b', 'zero']}
                move_out(s, 'b', 'zero')
                s['sides']['b']['front'] = replacement
                deal_combat_followup(EffectContext(s, 'a', 'nanali', operation))
                self.assertFalse(any(e['type'] == 'followup' for e in s['events']))
                self.assertEqual(s['sides']['b']['hp'], 30)

    def test_no_attack_means_no_combat_followup(self):
        s = self.game(); hero(s, 'a', 'nanali')['awakened'] = True
        deal_combat_followup(EffectContext(s, 'a', 'nanali', {'side': 'a'}))
        self.assertEqual(s['sides']['b']['hp'], 30)
        self.assertFalse(any(e['type'] == 'followup' for e in s['events']))

    def test_mimicked_nanali_hook_without_weapon_does_not_lookup_none_card(self):
        s = self.game()
        hero(s, 'a', 'zero')['shape'] = None
        operation = {'side': 'a', 'last_attack_target': ['b', 'player']}
        deal_combat_followup(EffectContext(s, 'a', 'zero', operation))
        self.assertFalse(any(e['type'] == 'followup' for e in s['events']))
