import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import (new_game, apply_action, legal_actions,
                                                observe, acting_side, choose_action)
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, attack_value


class DuelV2EngineTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=9, first_side='a', skip_mulligan=True)
        s['sides']['a']['ap'] = 2
        s['sides']['b']['shield'] = 0
        return s

    def hand(self, s, cid, side='a', copy=False):
        card = card_instance(s, cid, side, copy=copy)
        s['sides'][side]['hand'] = [card]
        return card

    def play(self, s, cid, target=None, side='a', copy=False):
        card = self.hand(s, cid, side, copy)
        action = {'type': 'play_card', 'card_id': card['instance_id']}
        if target:
            action['target_id'] = target
        return apply_action(s, side, action)

    def record(self, cid):
        card = CARDS[cid]
        return {'card_id': cid, **{k: card[k] for k in ('name', 'type', 'character_id')}}

    def next_own_turn(self, s):
        s = apply_action(s, 'a', {'type': 'end_turn'})
        return apply_action(s, 'b', {'type': 'end_turn'})

    def test_setup_mulligan_and_first_second_action_points(self):
        s = new_game(seed=12, first_side='b')
        self.assertEqual(s['sides']['a']['shield'], 5)
        self.assertTrue(all(t['front'] is None for t in s['sides'].values()))
        ids = [c['instance_id'] for c in s['sides']['a']['hand'][:3]]
        old = deepcopy(s)
        s = apply_action(s, 'a', {'type': 'mulligan', 'card_ids': ids[::-1]})
        self.assertEqual(old['version'], 0)
        self.assertFalse(set(ids) & {c['instance_id'] for c in s['sides']['a']['hand']})
        s = apply_action(s, 'b', {'type': 'mulligan', 'card_ids': []})
        self.assertEqual(s['sides']['b']['ap'], 1)
        self.assertEqual(len(s['sides']['b']['hand']), 6)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['ap'], 2)

    def test_illegal_atomic_and_json_replay(self):
        s = self.game()
        old = deepcopy(s)
        with self.assertRaises(ValueError):
            apply_action(s, 'b', {'type': 'attack', 'character_id': 'zero'})
        with self.assertRaises(ValueError):
            apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero', 'free': True})
        self.assertEqual(s, old)
        action = {'type': 'attack', 'character_id': 'zero'}
        self.assertEqual(apply_action(s, 'a', action), apply_action(json.loads(json.dumps(s)), 'a', action))

    def test_zero_resources_personal_and_battle_extra_sortie(self):
        s = apply_action(self.game(), 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (2, 1))
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 0)
        self.assertFalse(any(a['type'] == 'attack' for a in legal_actions(s, 'a')))
        s = self.play(s, 'N03')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['normal_attack_available'])

    def test_manual_genesis_growth_and_record_example(self):
        s = self.game()
        s['sides']['a']['front'] = 'zero'
        hero(s, 'a', 'nanali')['harmony'] = 2
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 6
        s = self.play(s, 'N03')
        n = hero(s, 'a', 'nanali')
        self.assertEqual((n['hp'], n['harmony'], n['energy'], n['growth']), (7, 1, 1, 1))
        self.assertEqual(attack_value(n), 4)
        self.assertIsNone(s['sides']['b']['front'])
        self.assertEqual(hero(s, 'b', 'zero')['down_turns'], 3)
        self.assertEqual(hero(s, 'a', 'jiuyuan')['energy'], 0)
        self.assertEqual(s['sides']['a']['records'], [self.record('N03')])

    def test_growth_cap_shape_and_awaken_does_not_refresh(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        n.update(shape='N02', growth=1, energy=4)
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['shield'], 1)
        s = apply_action(s, 'a', {'type': 'awaken', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['growth'], 2)
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N03')
        self.assertEqual(hero(s, 'a', 'nanali')['growth'], 2)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 0)
        s = self.next_own_turn(s)
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['growth'], 3)

    def test_n05_form_attack_and_once_heal(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(shape='N05', hp=3)
        s = self.play(s, 'N04')
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 4)
        self.assertEqual(attack_value(hero(s, 'a', 'nanali')), 5)
        s = self.play(s, 'N04')
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 4)

    def test_simultaneous_knockdown_energy_retained_revive_three_starts(self):
        s = self.game()
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'a', 'nanali').update(hp=3, energy=3, harmony=2, shape='N02', growth=0)
        hero(s, 'b', 'nanali').update(hp=3, energy=4)
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 0)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 3)
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        self.assertIsNone(hero(s, 'a', 'nanali')['shape'])
        for remaining in (2, 1, 0):
            s = self.next_own_turn(s)
            self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], remaining)
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 7)
        self.assertIsNone(s['sides']['a']['front'])

    def test_entry_uses_own_harmony_and_last_front(self):
        s = self.game()
        s['sides']['a']['last_front'] = 'nanali'
        hero(s, 'a', 'zero')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertTrue(s['sides']['a']['harmonized']['zero'])
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        before = sum(e['type'] == 'harmony' for e in s['events'])
        s = self.play(s, 'Z06')
        self.assertEqual(sum(e['type'] == 'harmony' for e in s['events']), before)
        s = self.game()
        s['sides']['a']['last_front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertFalse(s['sides']['a']['harmonized'])

    def test_flowers_lock_followup_target_and_retarget_next_flower(self):
        s = self.game()
        s['sides']['a']['front'] = 'nanali'
        hero(s, 'a', 'zero')['harmony'] = 2
        hero(s, 'a', 'jiuyuan').update(awakened=True, shape='J02')
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun')['hp'] = 1
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        flowers = [e for e in s['events'] if e['type'] == 'flower']
        self.assertEqual([e['target'] for e in flowers], ['b:xun', 'b:player'])
        self.assertFalse(any(e['type'] == 'followup' for e in s['events']))
        self.assertEqual(s['sides']['b']['hp'], 26)  # flower 1 + attack 3
        self.assertTrue(s['sides']['a']['used']['J02'])

    def test_xun_zero_harmony_stays_front_redeems_and_copy_casts_as_xun(self):
        s = self.game()
        s['sides']['a']['front'] = 'xun'
        s['sides']['a']['records'] = [self.record('N03')]
        s = self.play(s, 'X08')
        self.assertEqual(s['sides']['b']['hp'], 24)
        self.assertEqual(s['sides']['a']['records'], [])
        self.assertFalse(s['sides']['a']['harmonized'])
        copies = [c for c in s['sides']['a']['hand'] if c.get('copy')]
        self.assertEqual(len(copies), 1)
        copy = copies[0]
        self.assertEqual((copy['character_id'], copy['cost'], copy['expires_turn']), ('xun', 1, 2))
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 1)
        s = self.next_own_turn(s)
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': copy['instance_id']})
        self.assertEqual(s['sides']['b']['hp'], 21)
        self.assertEqual(hero(s, 'a', 'nanali')['growth'], 0)
        self.assertEqual(s['sides']['a']['ap'], 1)
        self.assertTrue(any(c['instance_id'] == copy['instance_id'] for c in s['sides']['a']['removed']))
        self.assertFalse(s['sides']['a']['records'])

    def test_awakened_xun_two_records_shape_expiry_and_no_recursive_copy(self):
        s = self.game()
        s['sides']['a']['front'] = 'xun'
        hero(s, 'a', 'xun').update(awakened=True, shape='X05')
        s['sides']['a']['records'] = [self.record('N03'), self.record('Z06')]
        s = self.play(s, 'X08')
        self.assertEqual(s['sides']['b']['hp'], 21)  # 5 + 4, only first gets shape bonus
        self.assertEqual(len([c for c in s['sides']['a']['hand'] if c['copy']]), 2)
        s = self.next_own_turn(s)
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertFalse(any(c['copy'] for c in s['sides']['a']['hand']))
        self.assertEqual(len(s['sides']['a']['removed']), 2)

    def test_xun_death_clears_records_and_unplayed_copies(self):
        s = self.game()
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'a', 'xun')['hp'] = 1
        s['sides']['a']['records'] = [self.record('N03')]
        copy = card_instance(s, 'N03', 'a', copy=True)
        s['sides']['a']['hand'].append(copy)
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'xun'})
        self.assertFalse(s['sides']['a']['records'])
        self.assertFalse(any(c.get('copy') for c in s['sides']['a']['hand']))
        self.assertEqual(hero(s, 'a', 'xun')['harmony'], 0)

    def test_record_first_only_and_distinct_authors_after_awaken(self):
        s = self.game()
        hero(s, 'a', 'xun')['awakened'] = True
        s = self.play(s, 'N01')
        s = self.play(s, 'J03')
        if s['phase'] == 'choice':
            s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
        self.assertEqual(s['sides']['a']['records'], [self.record('N01')])
        s = self.next_own_turn(s)
        s = self.play(s, 'Z02')
        s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
        self.assertEqual(s['sides']['a']['records'], [self.record('N01'), self.record('Z02')])

    def test_return_after_sortie_keeps_combat_resources_and_clears_shield(self):
        s = self.game()
        s = self.play(s, 'X07')
        hero(s, 'a', 'xun')['shield'] = 2
        before = len(s['sides']['a']['deck'])
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'xun'})
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s, 'a', 'xun')['shield'], 0)
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 1)
        self.assertEqual(len(s['sides']['a']['deck']), before - 1)

    def test_pending_draw_discard_one_payment_and_private_inspection(self):
        s = self.game()
        s = self.play(s, 'Z02')
        self.assertEqual(s['phase'], 'choice')
        self.assertEqual(s['sides']['a']['ap'], 1)
        self.assertEqual(len(s['sides']['a']['hand']), 2)
        self.assertEqual({a['type'] for a in legal_actions(s)}, {'concede', 'choose'})
        self.assertEqual(observe(s, 'b')['pending_choice']['choices'], [])
        s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
        self.assertEqual(s['sides']['a']['ap'], 1)
        self.assertEqual(len(s['sides']['a']['hand']), 1)
        s = self.play(s, 'J03')
        own, enemy = observe(s, 'a'), observe(s, 'b')
        secret = own['pending_choice']['choices'][0]['id']
        self.assertNotIn(secret, json.dumps(enemy, ensure_ascii=False))
        self.assertEqual(len(own['pending_choice']['choices']), 2)

    def test_j06_private_hand_then_public_selection_and_copy_removal(self):
        s = self.game()
        copy = self.hand(s, 'N01', side='b', copy=True)
        s = self.play(s, 'J06')
        self.assertEqual(len(observe(s, 'a')['pending_choice']['choices']), 1)
        self.assertEqual(observe(s, 'b')['pending_choice']['choices'], [])
        deck_count = len(s['sides']['b']['deck'])
        s = apply_action(s, 'a', {'type': 'choose', 'choice_id': copy['instance_id']})
        self.assertEqual(len(s['sides']['b']['deck']), deck_count)
        self.assertEqual(len(s['sides']['b']['removed']), 1)

    def test_projection_and_preview_do_not_depend_on_hidden_card_identity(self):
        s = self.game()
        s['sides']['a']['front'] = 'nanali'
        hero(s, 'a', 'zero')['harmony'] = 2
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['shape'] = 'Z03'
        before = deepcopy(s)
        public = observe(s, 'a')
        changed = deepcopy(s)
        for card in changed['sides']['b']['hand'] + changed['sides']['b']['deck'] + changed['sides']['a']['deck']:
            card.update(card_id='X08', effect_id='X08', character_id='xun', name='SECRET-MUTATION')
        self.assertEqual(observe(changed, 'a'), public)
        self.assertEqual(s, before)
        serialized = json.dumps(public)
        for hidden in ('"rng"', '"deck":', 'SECRET-MUTATION'):
            self.assertNotIn(hidden, serialized)

    def test_shield_bypass_counter_and_penetration(self):
        s = self.game()
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun').update(hp=3, shield=5)
        s = self.play(s, 'Z06')
        self.assertEqual(hero(s, 'b', 'xun')['hp'], 0)
        self.assertEqual(s['sides']['b']['hp'], 30)
        s = self.game()
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun').update(hp=1, shield=1)
        s = self.play(s, 'J08')
        self.assertEqual(s['sides']['b']['hp'], 28)
        self.assertEqual(hero(s, 'a', 'jiuyuan')['hp'], 6)

    def test_fatigue_empty_inspect_and_finished_stops_remaining_effects(self):
        s = self.game()
        s['sides']['a']['deck'] = []
        s = self.play(s, 'J03')
        self.assertEqual(s['sides']['a']['fatigue'], 0)
        s['sides']['a']['hp'] = 1
        s['sides']['a']['shield'] = 5
        s = self.play(s, 'N01')
        self.assertEqual(s['phase'], 'finished')
        self.assertEqual(s['winner'], 'b')
        self.assertEqual(s['sides']['a']['shield'], 5)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 0)

    def test_cycle_only_downed_owner_and_resolving_card_not_recoverable(self):
        s = self.game()
        card = self.hand(s, 'N01')
        self.assertFalse(any(a['type'] == 'cycle' for a in legal_actions(s)))
        hero(s, 'a', 'nanali').update(hp=0, down_turns=3)
        s = apply_action(s, 'a', {'type': 'cycle', 'card_id': card['instance_id']})
        self.assertFalse(any(a['type'] == 'cycle' for a in legal_actions(s)))
        self.hand(s, 'X04')
        self.assertFalse(any(a['type'] == 'play_card' for a in legal_actions(s)))
        old = card_instance(s, 'X04', 'a')
        s['sides']['a']['discard'].append(old)
        s = self.play(s, 'X04', target=old['instance_id'])
        self.assertEqual(s['sides']['a']['hand'][0]['instance_id'], old['instance_id'])

    def test_every_card_has_executable_legal_effect(self):
        for cid in CARDS:
            with self.subTest(card=cid):
                s = self.game()
                s['sides']['b']['front'] = 'zero'
                s['sides']['a']['discard'].append(card_instance(s, 'X01', 'a'))
                card = self.hand(s, cid)
                a = next(a for a in legal_actions(s) if a['type'] == 'play_card' and a['card_id'] == card['instance_id'])
                s = apply_action(s, 'a', a)
                if s['phase'] == 'choice':
                    s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
                self.assertEqual(s['phase'], 'playing')
                self.assertEqual(s['sides']['a']['ap'], 2 - CARDS[cid]['cost'])
                self.assertTrue(any(c['instance_id'] == card['instance_id'] for c in s['sides']['a']['discard']))
                json.dumps(observe(s, 'a'))

    def test_rule_opponent_and_bounded_game_completion(self):
        s = self.game()
        self.assertIn(choose_action(s, 'a'), legal_actions(s, 'a'))
        # A deterministic turn/fatigue regression, not rollout collection/training.
        for _ in range(80):
            if s['phase'] == 'finished':
                break
            s = apply_action(s, acting_side(s), {'type': 'end_turn'})
        self.assertEqual(s['phase'], 'finished')
        self.assertEqual(legal_actions(s), [])
        self.assertIsNone(acting_side(s))


if __name__ == '__main__':
    unittest.main()
