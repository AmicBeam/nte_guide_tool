"""User-authored tabletop rules (2026-09-18), not official combat values."""
from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags
import json
import unittest
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, get_catalog
from app.modules.card_game.engine.duel_v2.flow import (
    apply_action, begin_turn, end_ultimate, legal_actions, new_game, tick_front_debuff,
)
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero, knockdowns
from app.modules.card_game.engine.duel_v2.projection import observe, preview
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class NewTestCharacters(unittest.TestCase):
    def game(self):
        ids = ['xiaozhi', 'haiyue', 'zero', 'nanali']
        deck = {'id': 'test', 'name': '测试', 'character_ids': ids,
                'card_ids': [key for cid in ids for key, card in CARDS.items()
                             if card['character_id'] == cid and not card.get('derived')
                             for _ in range(card['starter_copies'])]}
        s = new_game(5, decks={'a': deck, 'b': deck}, first_side='a', skip_mulligan=True, escalation=False)
        for side in ('a', 'b'):
            s['sides'][side].update(hand=[], shield=0, ap=10)
        return s

    def play(self, s, cid, side='a', **kwargs):
        card = card_instance(s, cid, side)
        s['sides'][side]['hand'].append(card)
        if card.get('play_options'):
            kwargs.setdefault('option_id', 'base')
        return apply_action(s, side, {'type': 'play_card', 'card_id': card['instance_id'], **kwargs})

    def attack(self, s, cid='xiaozhi', side='a'):
        return apply_action(s, side, {'type': 'attack', 'character_id': cid})

    def test_access_and_images(self):
        from pathlib import Path
        full = get_catalog()
        public = get_catalog(include_test_characters=False)
        buildable = [c for c in full['cards'] if not c.get('derived')]
        self.assertEqual(len(buildable), 8 * len(full['characters']))
        for cid in ('xiaozhi', 'haiyue'):
            self.assertEqual(len([c for c in buildable if c['character_id'] == cid]), 8)
        for cid in ('xiaozhi', 'haiyue'):
            c = next(c for c in full['characters'] if c['id'] == cid)
            self.assertEqual(c['access_level'], 'public')
            self.assertIn(cid, [c['id'] for c in public['characters']])
            for key in ('avatar', 'portrait'):
                self.assertTrue(Path('app' + c[key]).is_file())

    def test_jingu_start_player_damage_and_knockdown(self):
        s = self.game()
        self.assertEqual(hero(s, 'a', 'xiaozhi')['base_stats']['attack'], 3)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 1)
        self.assertEqual(hero(s, 'b', 'xiaozhi')['jingu'], 0)
        s = self.attack(s)
        self.assertEqual(next(e for e in reversed(s['events']) if e['type'] == 'attack')['amount'], 3)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 2)
        damage(s, 'b', 'zero', 1, source='a:xiaozhi')
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 2)
        h = hero(s, 'a', 'xiaozhi');h.update(jingu=-3, energy=2, harmony=1)
        damage(s, 'a', 'xiaozhi', 99)
        self.assertEqual((h['jingu'], h['energy'], h['harmony']), (-3, 2, 1))

    def test_shield_absorbed_player_damage_does_not_gain_jingu(self):
        s = self.game();s['sides']['b']['shield'] = 9
        s = self.attack(s)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 1)

    def test_ultimate_negative_and_rng_persist(self):
        from app.modules.card_game.engine.duel_v2.state import add_hand
        s = self.game(); hero(s, 'a', 'xiaozhi').update(jingu=-2, energy=5)
        add_hand(s, 'a', card_instance(s, 'Q04', 'a'))
        rng = Random(s['rng']); delta = rng.choice((-1, 0, 1))
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'xiaozhi'})
        card = s['sides']['a']['hand'][0]
        self.assertEqual(card['jingu_mark']['delta'], delta)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 1)
        s = apply_action(json.loads(json.dumps(s)), 'a', {'type':'play_card', 'card_id':card['instance_id']})
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 2 + delta)
        end_ultimate(s, 'a', 'xiaozhi')
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], -1 + delta)
        end_ultimate(s, 'a', 'xiaozhi')
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], -1 + delta)

    def test_ultimate_down_preserves_resource_and_settles_loan(self):
        s = self.game();h = hero(s, 'a', 'xiaozhi');h.update(awakened=True, ultimate_turns=2, jingu=-7, jingu_loan=True)
        damage(s, 'a', 'xiaozhi', 99)
        self.assertEqual(h['jingu'], -10)
        self.assertFalse(h.get('jingu_loan'))

    def test_a_bonus_only_when_actively_played(self):
        s = self.game();hero(s, 'a', 'xiaozhi')['jingu'] = 3
        s = self.play(s, 'Q01', option_id='boost')
        self.assertEqual(s['sides']['b']['hp'], 23)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 1)

    def test_c_b_a_chain_no_extra_ap_cards_or_resources(self):
        s = self.game();hero(s, 'a', 'xiaozhi')['jingu'] = 9
        start = s['event_seq'];s = self.play(s, 'Q03', option_id='release_ba')
        attacks = [e for e in s['events'] if e['seq'] > start and e['type'] == 'attack']
        self.assertEqual([e['amount'] for e in attacks], [4, 3, 4])
        self.assertEqual(s['sides']['b']['hp'], 17)  # third hit additionally consumes -2 resistance
        self.assertEqual(s['sides']['a']['ap'], 9)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 6)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['energy'], 2)
        self.assertEqual([c['card_id'] for c in s['sides']['a']['discard']], ['Q03'])

    def test_chain_stops_when_attacker_dies(self):
        s = self.game();hero(s, 'a', 'xiaozhi')['jingu'] = 9
        s['sides']['b']['front'] = 'zero';hero(s, 'b', 'zero').update(base_attack=20, hp=40)
        start=s['event_seq'];s=self.play(s,'Q03',option_id='release_ba')
        self.assertEqual(len([e for e in s['events'] if e['seq']>start and e['type']=='attack']),1)
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],3)

    def test_b_resistance_is_targeted_one_shot_and_attribute_specific(self):
        s=self.game();s['sides']['b']['front']='zero';hero(s,'b','zero').update(hp=30,max_hp=30)
        s=self.play(s,'Q02');h=hero(s,'b','zero')
        before=h['hp'];damage(s,'b','zero',1,source='a:nanali');self.assertEqual(h['hp'],before-1)
        damage(s,'b','zero',1,source='a:zero');self.assertEqual(h['hp'],before-4)
        damage(s,'b','zero',1,source='a:zero');self.assertEqual(h['hp'],before-5)

    def test_d_response_counter_and_no_resources(self):
        s=self.game();s['sides']['b']['front']='xiaozhi'
        s['sides']['b']['hand']=[card_instance(s,'Q04','b')]
        s=self.attack(s,'zero')
        self.assertEqual(hero(s,'a','zero')['down_turns'],3)
        self.assertEqual(hero(s,'b','xiaozhi')['energy'],0)
        self.assertEqual(s['sides']['b']['ap'],9)

    def test_shuffle_tactic_empty_deck_and_instant(self):
        s=self.game();s['sides']['a']['deck']=[];s['sides']['a']['ap']=0
        s=self.play(s,'Q05')
        self.assertEqual(s['phase'],'playing')
        self.assertEqual(s['sides']['a']['hand'][0]['card_id'],'Q05')
        self.assertFalse(s['sides']['a']['hand'][0]['copy'])
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],2)
        self.assertTrue(s['sides']['a']['used']['instant'])

    def test_surplus_gate_and_energy_fill(self):
        s=self.game();card=card_instance(s,'Q06','a');s['sides']['a']['hand']=[card]
        action={'type':'play_card','card_id':card['instance_id']}
        self.assertNotIn(action,legal_actions(s,'a'))
        s['sides']['a']['surplus']=True;s=apply_action(s,'a',action)
        self.assertEqual(hero(s,'a','xiaozhi')['energy'],5)

    def test_jingu_weapon_damage_and_free_extra_attack(self):
        from app.modules.card_game.engine.duel_v2.state import attack_value
        s=self.game();h=hero(s,'a','xiaozhi');h.update(shape='Q08',jingu=6)
        self.assertEqual((attack_value(h), CARDS['Q08']['hp']), (3, 5))
        s=self.play(s,'Q04');self.assertEqual(s['sides']['b']['hp'],22)
        s=self.game();h=hero(s,'a','xiaozhi');h.update(shape='Q07',jingu=3)
        self.assertEqual((attack_value(h), CARDS['Q07']['hp']), (3, 5))
        s['sides']['a'].update(ap=0,normal_attack_available=False)
        action={'type':'attack','character_id':'xiaozhi'}
        self.assertEqual(preview(s,'a',action)['ap_cost'],0)
        s=self.attack(s);self.assertEqual(s['sides']['b']['hp'],25)
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],1)
        self.assertNotIn(action,legal_actions(s,'a'))

    def test_haiyue_attack_returns_and_ultimate_keeps_front(self):
        s=self.game();s=self.attack(s,'haiyue')
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s,'a','haiyue')['energy'],2)
        s=self.game();s['sides']['a']['front']='zero';hero(s,'a','zero')['harmony']=2
        s['sides']['b']['front']='zero';hero(s,'b','zero').update(base_attack=9,hp=30,max_hp=30)
        hero(s,'a','haiyue').update(awakened=True,ultimate_turns=1)
        p=preview(s,'a',{'type':'attack','character_id':'haiyue'})
        self.assertFalse(p['will_switch']);self.assertEqual(p['counter'],0)
        self.assertEqual(p['入场'], '不入场')
        from app.modules.card_game.engine.duel_v2.combat import harmony_marks
        self.assertNotIn('haiyue', harmony_marks(s, 'a')['ready'])
        s=self.attack(s,'haiyue')
        self.assertEqual(s['sides']['a']['front'],'zero')
        self.assertEqual(hero(s,'a','zero')['harmony'],2)
        self.assertEqual(hero(s,'a','haiyue')['hp'],4)

    def test_haiyue_ultimate_expires_at_this_turn_end(self):
        s = self.game()
        hero(s, 'a', 'haiyue')['energy'] = 6
        hero(s, 'a', 'xiaozhi').update(awakened=True, ultimate_turns=2)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'haiyue'})
        self.assertTrue(hero(s, 'a', 'haiyue')['awakened'])
        for board in (observe(s, 'a'), capture_public_board(s)):
            h = next(h for h in board['sides']['a']['characters'] if h['id'] == 'haiyue')
            self.assertEqual(h['ultimate_expires'], 'turn_end')
        # Persistence must retain the effect during the same turn.
        s = self.attack(json.loads(json.dumps(s)), 'haiyue')
        self.assertTrue(hero(s, 'a', 'haiyue')['awakened'])
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(s['active_side'], 'b')
        self.assertFalse(hero(s, 'a', 'haiyue')['awakened'])
        self.assertEqual(hero(s, 'a', 'haiyue')['ultimate_turns'], 0)
        self.assertTrue(hero(s, 'a', 'xiaozhi')['awakened'])
        # The following own turn no longer bypasses entry or retaliation.
        s = apply_action(s, 'b', {'type': 'end_turn'})
        s['sides']['b']['front'] = 'zero'
        start = s['event_seq']
        s = self.attack(s, 'haiyue')
        events = [e for e in s['events'] if e['seq'] > start]
        self.assertTrue(any(e['type'] == 'enter' and e.get('actor') == 'a:haiyue' for e in events))
        self.assertEqual(next(e['counter'] for e in events if e['type'] == 'attack'), 2)

    def test_wangri_fuge_equips_four_attack_four_health(self):
        from app.modules.card_game.engine.duel_v2.state import attack_value
        s = self.play(self.game(), 'U06')
        h = hero(s, 'a', 'haiyue')
        self.assertEqual((CARDS['U06']['attack'], CARDS['U06']['hp']), (4, 4))
        self.assertEqual((attack_value(h), h['hp'], h['max_hp']), (4, 4, 4))

    def test_haiyue_response_checks_enemy_front(self):
        for front in (None,'zero'):
            s=self.game();s['sides']['a']['front']=front;s['sides']['b']['ap']=0
            s['sides']['b']['hand']=[card_instance(s,'U01','b')]
            s=apply_action(s,'a',{'type':'end_turn'})
            plays=[e for e in s['events'] if e['type']=='play' and e.get('side')=='b']
            self.assertEqual(len(plays),int(front is None))
            if front is None:self.assertEqual(s['sides']['a']['hp'],25)

    def test_haiyue_direct_and_bench_target(self):
        s=self.game();s['sides']['b']['front']='zero';s=self.play(s,'U03')
        self.assertEqual(s['sides']['b']['hp'],27)
        self.assertEqual(hero(s,'b','zero')['hp'],5)
        s=self.game();s['sides']['b']['front']='zero'
        s=self.play(s,'U04',target_id='b:nanali')
        self.assertEqual(hero(s,'b','nanali')['hp'],0)
        self.assertEqual(hero(s,'a','haiyue')['hp'],4)
        self.assertEqual(hero(s,'b','zero')['hp'],5)

    def test_conditional_instant_and_weapon_attack(self):
        s=self.game();s=self.attack(s,'zero');s['sides']['a']['ap']=0
        s=self.play(s,'U07');self.assertEqual(hero(s,'a','haiyue')['shape'],'U07')
        self.assertTrue(s['sides']['a']['used']['instant'])
        s=self.game();s=self.attack(s,'zero');s['sides']['a']['ap']=0;hero(s,'a','haiyue')['shape']='U06'
        s=self.attack(s,'haiyue');self.assertTrue(s['sides']['a']['used']['instant'])
        self.assertFalse(s['sides']['a']['normal_attack_available'])
        self.assertNotIn({'type':'attack','character_id':'haiyue'},legal_actions(s,'a'))

    def test_star_bonus_and_attack_growth(self):
        s=self.game();hero(s,'a','haiyue')['shape']='U05'
        s['sides']['b']['front_debuff']['star']={'by':'a','left':1}
        tick_front_debuff(s,'b');self.assertEqual(s['sides']['b']['hp'],27)
        s=self.game();hero(s,'a','haiyue')['shape']='U08';s=self.play(s,'U02')
        self.assertEqual(compatibility_flags(hero(s,'a','haiyue'))['atk_buff'],1)
        s=self.play(s,'U02');self.assertEqual(compatibility_flags(hero(s,'a','haiyue'))['atk_buff'],2)

    def test_response_requires_available_instant_or_ap(self):
        s = self.game()
        s['sides']['b'].update(ap=0, used={'instant': True})
        s['sides']['b']['hand'] = [card_instance(s, 'U01', 'b')]
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertFalse(any(e['type'] == 'play' and e.get('side') == 'b' for e in s['events']))
        self.assertTrue(any(c['card_id'] == 'U01' for c in s['sides']['b']['hand']))

    def test_u02_free_after_teammate_and_condition_resets_next_turn(self):
        s = self.game()
        s = self.attack(s, 'zero')
        s['sides']['a']['ap'] = 0
        card = card_instance(s, 'U02', 'a'); s['sides']['a']['hand'].append(card)
        action = {'type': 'play_card', 'card_id': card['instance_id']}
        self.assertEqual(preview(s, 'a', action)['ap_cost'], 0)
        s = apply_action(s, 'a', action)
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertTrue(s['sides']['a']['used']['instant'])
        begin_turn(s, 'a')
        s['sides']['a']['ap'] = 0
        card = card_instance(s, 'U02', 'a'); s['sides']['a']['hand'] = [card]
        self.assertNotIn({'type': 'play_card', 'card_id': card['instance_id']}, legal_actions(s, 'a'))

    def test_terminal_duration_and_reactivation_settle_loan(self):
        s = self.game(); h = hero(s, 'a', 'xiaozhi'); h.update(jingu=0, energy=5)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'xiaozhi'})
        begin_turn(s, 'a')
        self.assertEqual(hero(s, 'a', 'xiaozhi')['ultimate_turns'], 1)
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 4)
        hero(s, 'a', 'xiaozhi')['energy'] = 5
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'xiaozhi'})
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 4)
        begin_turn(s, 'a'); begin_turn(s, 'a')
        self.assertFalse(hero(s, 'a', 'xiaozhi')['awakened'])
        self.assertEqual(hero(s, 'a', 'xiaozhi')['jingu'], 3)

    def test_target_policy_rejects_front_and_downed_bench(self):
        s = self.game(); s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'nanali').update(hp=0, down_turns=3)
        card = card_instance(s, 'U04', 'a'); s['sides']['a']['hand'] = [card]
        actions = [a for a in legal_actions(s, 'a') if a.get('card_id') == card['instance_id']]
        self.assertEqual({a['target_id'] for a in actions}, {'b:xiaozhi', 'b:haiyue'})

    def test_jingu_damage_bonus_applies_to_response(self):
        s = self.game(); s['sides']['b']['front'] = 'xiaozhi'
        hero(s, 'b', 'xiaozhi').update(shape='Q08', jingu=4)
        s['sides']['b']['hand'] = [card_instance(s, 'Q04', 'b')]
        hero(s, 'a', 'zero').update(hp=30, max_hp=30)
        s = self.attack(s, 'zero')
        attack = [e for e in s['events'] if e['type'] == 'attack'][-1]
        self.assertEqual(attack['counter'], 7)

    def test_signed_jingu_projection_and_replay(self):
        s=self.game();hero(s,'a','xiaozhi')['jingu']=-5
        for board in (observe(s,'b'),capture_public_board(s)):
            h=next(h for h in board['sides']['a']['characters'] if h['id']=='xiaozhi')
            self.assertEqual(h['jingu'],-5)


if __name__ == '__main__':
    unittest.main()
