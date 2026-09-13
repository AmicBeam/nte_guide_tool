import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import (new_game, apply_action, legal_actions,
                                                observe, acting_side, choose_action)
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, attack_value, new_character, reduce_max_hp


class DuelV2EngineTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=9, first_side='a', skip_mulligan=True)
        s['sides']['a']['ap'] = 2
        s['sides']['b']['shield'] = 0
        for side in ('a', 'b'):
            s['sides'][side]['hand'] = [c for c in s['sides'][side]['hand'] if not c.get('response')]
            if 'xun' not in s['sides'][side]['characters']:
                s['sides'][side]['characters']['xun'] = new_character('xun')
                s['sides'][side]['order'].append('xun')
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

    def test_observe_exposes_first_side(self) -> None:
        s = new_game(seed=12, first_side='b')
        view = observe(s, 'a')
        self.assertEqual(view['first_side'], 'b')
        self.assertEqual(view['viewer_side'], 'a')

    def test_observe_hand_follows_character_order(self) -> None:
        s = self.game()
        order = list(s['sides']['a']['order'])
        first = next(card_id for card_id, card in CARDS.items() if card['character_id'] == order[0])
        last = next(card_id for card_id, card in CARDS.items() if card['character_id'] == order[-1])
        s['sides']['a']['hand'] = [
            card_instance(s, last, 'a'),
            card_instance(s, first, 'a'),
            card_instance(s, last, 'a'),
        ]
        view = observe(s, 'a')
        owners = [card['character_id'] for card in view['sides']['a']['hand']]
        self.assertEqual(owners, [order[0], order[-1], order[-1]])

    def test_lingke_turns_unpaired_switch_into_sync(self) -> None:
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        deck = next(item for item in STARTER_DECKS if item['id'] == 'sync-curse')
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        s['sides']['a']['ap'] = 2
        s['sides']['a']['front'] = 'canhong'
        hero(s, 'a', 'canhong')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'iloy'})
        self.assertTrue(any(e['type'] == 'harmony' and '同频' in e['text'] for e in s['events']))
        self.assertEqual(hero(s, 'a', 'iloy')['shield'], 1)

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
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (1, 2))
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 0)
        self.assertFalse(any(a['type'] == 'attack' for a in legal_actions(s, 'a')))
        s = self.play(s, 'N03')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['normal_attack_available'])

    def test_response_battle_joins_as_counter_not_new_combat(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hand'] = [card_instance(s, 'N02', 'b')]
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'xun'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 5)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 5)
        self.assertEqual(hero(s, 'b', 'nanali')['shield'], 0)
        self.assertEqual(hero(s, 'a', 'xun')['hp'], 3)
        attacks = [e for e in s['events'] if e['type'] == 'attack']
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]['attack'], 2)
        self.assertEqual(attacks[0]['counter'], 2)
        self.assertTrue(any('作为反击方' in e['text'] for e in s['events'] if e['type'] == 'enter'))
        self.assertEqual((hero(s, 'b', 'nanali')['harmony'], hero(s, 'b', 'nanali')['energy']), (0, 0))
        self.assertEqual((hero(s, 'a', 'xun')['harmony'], hero(s, 'a', 'xun')['energy']), (1, 2))
        self.assertTrue(any(c['card_id'] == 'N02' for c in s['sides']['b']['discard']))
        self.assertFalse(any(e['type'] == 'harmony' for e in s['events']))

        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s = self.play(s, 'N02')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertEqual((hero(s, 'a', 'nanali')['harmony'], hero(s, 'a', 'nanali')['energy']), (1, 2))
        self.assertEqual(len([e for e in s['events'] if e['type'] == 'attack']), 1)

    def test_counter_does_not_grant_combat_resources(self):
        s = self.game()
        s['sides']['b']['hand'] = []
        s['sides']['b']['front'] = 'nanali'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (1, 2))
        self.assertEqual((hero(s, 'b', 'nanali')['harmony'], hero(s, 'b', 'nanali')['energy']), (0, 0))
        self.assertFalse(any(e['type'] == 'resource' and e.get('side') == 'b' for e in s['events']))

        s = self.game()
        s['sides']['b']['hand'] = []
        s['sides']['b']['front'] = 'zero'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual((hero(s, 'a', 'nanali')['harmony'], hero(s, 'a', 'nanali')['energy']), (1, 2))
        self.assertEqual((hero(s, 'b', 'zero')['harmony'], hero(s, 'b', 'zero')['energy']), (0, 0))

    def test_nanali_family_card_and_n01_draws_owned(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (2, 5, 5))
        self.assertTrue(any(c['card_id'] == 'NF01' for c in s['sides']['a']['hand']))
        family = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': family['instance_id']})
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (3, 6, 6))
        s['sides']['a']['deck'] = [card_instance(s, 'N03', 'a')] + s['sides']['a']['deck']
        s['sides']['a']['ap'] = 0
        s = self.play(s, 'N01')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertTrue(any(c['card_id'] == 'N03' for c in s['sides']['a']['hand']))
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 1
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N03')
        self.assertTrue(any(c['card_id'] == 'NF01' for c in s['sides']['a']['hand']))
        self.assertEqual(s['sides']['a']['nanali_family'], 2)
        self.assertTrue(any(e['type'] == 'followup' for e in s['events']))

    def test_growth_cap_shape_and_awaken_does_not_refresh(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=4)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        n = hero(s, 'a', 'nanali')
        self.assertTrue(n['awakened'])
        self.assertEqual(s['sides']['a']['front'], 'nanali')

    def test_ultimate_is_once_per_turn_and_free(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=4)
        s['sides']['a']['ap'] = 0
        self.assertTrue(s['sides']['a']['ultimate_available'])
        self.assertIn('ultimate', {a['type'] for a in legal_actions(s, 'a') if a.get('character_id') == 'nanali'})
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['ultimate_available'])
        hero(s, 'a', 'zero').update(energy=4)
        self.assertFalse(any(a['type'] == 'ultimate' for a in legal_actions(s, 'a')))

    def test_ultimate_from_bench_lasts_two_own_turns(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=4, growth=3)
        self.assertIsNone(s['sides']['a']['front'])
        self.assertIn('ultimate', {a['type'] for a in legal_actions(s, 'a') if a.get('character_id') == 'nanali'})
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        n = hero(s, 'a', 'nanali')
        self.assertTrue(n['awakened'])
        self.assertEqual(n['ultimate_turns'], 2)
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        s = self.next_own_turn(s)
        self.assertTrue(hero(s, 'a', 'nanali')['awakened'])
        self.assertEqual(hero(s, 'a', 'nanali')['ultimate_turns'], 1)
        s = self.next_own_turn(s)
        n = hero(s, 'a', 'nanali')
        self.assertFalse(n['awakened'])
        self.assertEqual(n['ultimate_turns'], 0)

    def test_form_hp_and_battle_shield_value(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        self.assertEqual((n['hp'], n['max_hp']), (5, 5))
        s = self.play(s, 'N08')
        n = hero(s, 'a', 'nanali')
        self.assertEqual((n['shape'], attack_value(n), n['hp'], n['max_hp']), ('N08', 2, 6, 6))
        s = self.game()
        s = self.play(s, 'N04')
        self.assertEqual(hero(s, 'a', 'nanali')['shield'], 2)

    def test_n08_scales_battle_frame_with_family(self):
        s = self.game()
        family = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': family['instance_id']})
        s['sides']['a']['ap'] = 2
        s = self.play(s, 'N07')
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N04')
        self.assertEqual(hero(s, 'a', 'nanali')['shield'], 3)

    def test_instant_first_card_skips_action_cost(self):
        s = self.game()
        card = self.hand(s, 'N01')
        s['sides']['a']['ap'] = 0
        self.assertTrue(any(a['type'] == 'play_card' and a['card_id'] == card['instance_id']
                            for a in legal_actions(s, 'a')))
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertTrue(s['sides']['a']['used'].get('instant'))

    def test_n05_form_attack_and_once_heal(self):
        s = self.game()
        s = self.play(s, 'N04')
        self.assertEqual(hero(s, 'a', 'nanali')['shield'], 2)

    def test_simultaneous_knockdown_keeps_resources_revive_three_starts(self):
        s = self.game()
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'a', 'nanali').update(hp=1, energy=3, harmony=2, shape='N07', growth=0)
        hero(s, 'b', 'nanali').update(hp=1, energy=4)
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 0)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 3)
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 2)
        self.assertEqual(hero(s, 'b', 'nanali')['energy'], 4)
        self.assertIsNone(hero(s, 'a', 'nanali')['shape'])
        self.assertFalse(any('环合失去' in e['text'] or '能量失去' in e['text']
                             for e in s['events'] if e['type'] == 'down'))
        for remaining in (2, 1, 0):
            s = self.next_own_turn(s)
            self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], remaining)
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 5)
        self.assertIsNone(s['sides']['a']['front'])

    def test_entry_consumes_previous_harmony_not_enterer(self):
        s = self.game()
        s['sides']['a']['last_front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertTrue(s['sides']['a']['harmonized']['zero'])
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 1)
        before = sum(e['type'] == 'harmony' for e in s['events'])
        s = self.play(s, 'Z03')
        self.assertEqual(sum(e['type'] == 'harmony' for e in s['events']), before)
        s = self.game()
        s['sides']['a']['last_front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertFalse(s['sides']['a']['harmonized'])
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 2)

    def test_downed_previous_keeps_harmony_but_does_not_pay(self):
        s = self.game()
        s['sides']['a']['front'] = 'zero'
        hero(s, 'a', 'zero').update(hp=1, harmony=2)
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'a', 'zero')['base_attack'] = 0
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'zero')['down_turns'], 3)
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 2)
        self.assertFalse(any('环合失去' in e['text'] for e in s['events'] if e['type'] == 'down'))
        s['sides']['a']['ap'] = 1
        s['sides']['a']['normal_attack_available'] = True
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertFalse(s['sides']['a']['harmonized'])
        self.assertFalse(any(e['type'] == 'harmony' for e in s['events']))

    def test_turn_start_returns_front_unless_collapsed(self):
        s = self.game()
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertIsNone(s['sides']['b']['front'])
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(s['sides']['a']['last_front'], 'nanali')
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        hero(s, 'a', 'zero').setdefault('flags', {})['collapse'] = {'side': 'b', 'until': 99}
        s = apply_action(s, 'a', {'type': 'end_turn'})
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['front'], 'zero')

    def test_observe_marks_harmony_ready_from_last_front(self):
        s = self.game()
        s['sides']['a']['last_front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        view = observe(s, 'a')
        chars = {c['id']: c for c in view['sides']['a']['characters']}
        self.assertTrue(view['sides']['a']['harmony_available'])
        self.assertTrue(chars['zero']['harmony_source'])
        self.assertFalse(chars['zero']['harmony_ready'])
        self.assertTrue(chars['nanali']['harmony_ready'])
        self.assertTrue(chars['jiuyuan']['harmony_ready'])
        hero(s, 'a', 'zero').update(hp=0, down_turns=3)
        view = observe(s, 'a')
        chars = {c['id']: c for c in view['sides']['a']['characters']}
        self.assertFalse(view['sides']['a']['harmony_available'])
        self.assertFalse(chars['zero']['harmony_source'])
        self.assertFalse(chars['nanali']['harmony_ready'])

    def test_downed_character_does_not_gain_resources(self):
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        from app.modules.card_game.engine.duel_v2.state import energy, knockdowns
        s = self.game()
        hero(s, 'a', 'nanali').update(hp=0, energy=3, harmony=2)
        knockdowns(s)
        self.assertEqual((hero(s, 'a', 'nanali')['harmony'], hero(s, 'a', 'nanali')['energy']), (2, 3))
        energy(s, 'a', 'nanali', 2)
        EffectContext(s, 'a', 'nanali', {'side': 'a'}).grant_harmony('nanali', 1)
        self.assertEqual((hero(s, 'a', 'nanali')['harmony'], hero(s, 'a', 'nanali')['energy']), (2, 3))

    def test_flowers_lock_followup_target_and_retarget_next_flower(self):
        s = self.game()
        s['sides']['a']['front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = 2
        hero(s, 'a', 'jiuyuan').update(awakened=True, shape='J08')
        s['sides']['a']['extra_flower'] = True
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun')['hp'] = 1
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        flowers = [e for e in s['events'] if e['type'] == 'flower']
        self.assertGreaterEqual(len(flowers), 1)

    def test_xun_zero_harmony_stays_front_redeems_and_copy_casts_as_xun(self):
        s = self.game()
        s['sides']['a']['front'] = 'xun'
        s['sides']['a']['records'] = [self.record('N03')]
        s = self.play(s, 'X05')
        self.assertEqual(s['sides']['b']['hp'], 20)
        self.assertEqual(s['sides']['a']['records'], [])
        self.assertFalse(s['sides']['a']['harmonized'])
        copies = [c for c in s['sides']['a']['hand'] if c.get('copy')]
        self.assertEqual(len(copies), 1)
        copy = copies[0]
        self.assertEqual((copy['character_id'], copy['cost'], copy['expires_turn']), ('xun', 1, 2))
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 2)
        s = self.next_own_turn(s)
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': copy['instance_id']})
        self.assertEqual(s['sides']['b']['hp'], 17)
        self.assertEqual(hero(s, 'a', 'nanali')['growth'], 0)
        self.assertEqual(s['sides']['a']['ap'], 1)
        self.assertTrue(any(c['instance_id'] == copy['instance_id'] for c in s['sides']['a']['removed']))
        self.assertFalse(s['sides']['a']['records'])

    def test_awakened_xun_two_records_shape_expiry_and_no_recursive_copy(self):
        s = self.game()
        s['sides']['a']['front'] = 'xun'
        hero(s, 'a', 'xun').update(awakened=True, shape='X07')
        s['sides']['a']['records'] = [self.record('N03'), self.record('Z06')]
        s = self.play(s, 'X05')
        self.assertEqual(s['sides']['b']['hp'], 18)
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

    def test_iloy_door_shape_does_not_recurse_on_knockdown_damage(self):
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        s = self.game()
        hero(s, 'a', 'iloy').update(shape='Y08')
        hero(s, 'b', 'iloy').update(shape='Y08', hp=2)
        s['sides']['b']['front'] = 'iloy'
        hero(s, 'a', 'nanali')['hp'] = 0
        knockdowns(s)
        self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], 3)
        self.assertEqual(hero(s, 'b', 'iloy')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'iloy')['down_turns'], 3)

    def test_record_first_only_and_distinct_authors_after_awaken(self):
        s = self.game()
        hero(s, 'a', 'xun')['awakened'] = True
        s['sides']['b']['front'] = 'zero'
        s = self.play(s, 'N01')
        s = self.play(s, 'J02')
        if s['phase'] == 'choice':
            s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
        self.assertEqual(s['sides']['a']['records'], [self.record('N01')])
        s = self.next_own_turn(s)
        s = self.play(s, 'Z03')
        if s['phase'] == 'choice':
            s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
        self.assertEqual(s['sides']['a']['records'], [self.record('N01'), self.record('Z03')])

    def test_return_after_sortie_keeps_combat_resources_and_clears_shield(self):
        s = self.game()
        s = self.play(s, 'X03')
        hero(s, 'a', 'xun')['shield'] = 2
        before = len(s['sides']['a']['deck'])
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'xun'})
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s, 'a', 'xun')['shield'], 0)
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 2)
        self.assertEqual(len(s['sides']['a']['deck']), before - 1)

    def test_front_combat_energy_doubles_bench_cards_stay_one(self):
        s = apply_action(self.game(), 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 2)
        self.assertEqual(observe(s, 'a')['sides']['a']['characters'][1]['energy_max'], 4)
        s = self.play(s, 'N01')
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 0)
        hero(s, 'a', 'zero')['energy'] = 3
        s['sides']['a']['ap'] = 1
        s['sides']['a']['normal_attack_available'] = True
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 4)

    def test_dark_attribute_energy_cap_is_five(self):
        deck = next(item for item in STARTER_DECKS if item['id'] == 'requiem')
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        s['sides']['a']['ap'] = 2
        s['sides']['b']['shield'] = 0
        self.assertEqual(observe(s, 'a')['sides']['a']['characters'][0]['energy_max'], 5)
        hero(s, 'a', 'anhunqu')['energy'] = 4
        self.assertFalse(any(a['type'] == 'ultimate' and a.get('character_id') == 'anhunqu'
                             for a in legal_actions(s, 'a')))
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'anhunqu'})
        self.assertEqual(hero(s, 'a', 'anhunqu')['energy'], 5)
        s['sides']['a']['ap'] = 1
        self.assertTrue(any(a['type'] == 'ultimate' and a.get('character_id') == 'anhunqu'
                            for a in legal_actions(s, 'a')))

    def test_pending_draw_discard_one_payment_and_private_inspection(self):
        s = self.game()
        s = self.play(s, 'N01')
        self.assertEqual(s['phase'], 'playing')

    def test_j06_private_hand_then_public_selection_and_copy_removal(self):
        s = self.game()
        s['sides']['a']['records'] = [self.record('N01')]
        s = self.play(s, 'X05')
        self.assertTrue(any(c.get('copy') for c in s['sides']['a']['hand']))

    def test_projection_and_preview_do_not_depend_on_hidden_card_identity(self):
        s = self.game()
        s['sides']['a']['front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = 2
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['shape'] = 'Z07'
        before = deepcopy(s)
        public = observe(s, 'a')
        changed = deepcopy(s)
        for card in changed['sides']['b']['hand'] + changed['sides']['b']['deck'] + changed['sides']['a']['deck']:
            card.update(card_id='X05', effect_id='X05', character_id='xun', name='SECRET-MUTATION')
        self.assertEqual(observe(changed, 'a'), public)
        self.assertEqual(s, before)
        serialized = json.dumps(public)
        for hidden in ('"rng"', '"deck":', 'SECRET-MUTATION'):
            self.assertNotIn(hidden, serialized)

    def test_max_hp_ratio_rounds_half_up(self):
        s = self.game()
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun').update(hp=5, max_hp=7)
        reduce_max_hp(s, 'b', 'xun', 1)
        self.assertEqual((hero(s, 'b', 'xun')['hp'], hero(s, 'b', 'xun')['max_hp']), (4, 6))
        reduce_max_hp(s, 'b', 'xun', 1)
        self.assertEqual((hero(s, 'b', 'xun')['hp'], hero(s, 'b', 'xun')['max_hp']), (3, 5))

    def test_max_hp_cut_rounds_and_skips_shield(self):
        s = self.game()
        s['sides']['b']['front'] = 'xun'
        hero(s, 'b', 'xun').update(hp=5, max_hp=7, shield=5)
        reduce_max_hp(s, 'b', 'xun', 1)
        xun = hero(s, 'b', 'xun')
        self.assertEqual((xun['hp'], xun['max_hp'], xun['shield']), (4, 6, 5))
        reduce_max_hp(s, 'b', 'xun', 1)
        xun = hero(s, 'b', 'xun')
        self.assertEqual((xun['hp'], xun['max_hp'], xun['shield']), (3, 5, 5))

    def test_delay_zone_slow_one(self):
        from app.modules.card_game.engine.duel_v2.state import front_debuff
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['delay'] = {'end': 9, 'slow': 1, 'by': 'a'}
        before = hero(s, 'a', 'nanali')['hp']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], before - 1)

    def test_front_debuff_persists_and_overlay_followup(self):
        from app.modules.card_game.engine.duel_v2.state import front_debuff, move_out
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a'}
        front_debuff(s, 'b')['weave'] = {'by': 'a'}
        move_out(s, 'b', 'zero', force=True)
        s['sides']['b']['front'] = 'xun'
        self.assertEqual(front_debuff(s, 'b')['burn']['left'], 2)
        hp = hero(s, 'b', 'xun')['hp']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertTrue(any(e['type'] == 'overlay' for e in s['events']))
        self.assertEqual(hero(s, 'b', 'xun')['hp'], hp - 4)

    def test_fatigue_empty_inspect_and_finished_stops_remaining_effects(self):
        s = self.game()
        s['sides']['a']['deck'] = []
        s['sides']['a']['hp'] = 1
        s['sides']['a']['shield'] = 0
        before_harmony = hero(s, 'a', 'xun')['harmony']
        s = self.play(s, 'X01')
        self.assertEqual(s['phase'], 'finished')
        self.assertEqual(s['winner'], 'b')
        self.assertEqual(hero(s, 'a', 'xun')['harmony'], before_harmony)
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 0)

    def test_downed_owner_cards_unplayable_and_cycle_removed(self):
        s = self.game()
        card = self.hand(s, 'N01')
        hero(s, 'a', 'nanali').update(hp=0, down_turns=3)
        view = observe(s, 'a')
        self.assertEqual(view['sides']['a']['hand'][0].get('unavailable_reason'), '所属角色倒地')
        self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], 3)
        self.assertFalse(any(a['type'] == 'cycle' for a in legal_actions(s)))
        with self.assertRaises(ValueError):
            apply_action(s, 'a', {'type': 'cycle', 'card_id': card['instance_id']})
        self.hand(s, 'N03')
        self.assertFalse(any(a['type'] == 'play_card' for a in legal_actions(s)))
        revive = self.hand(s, 'N06')
        self.assertTrue(any(a['type'] == 'play_card' and a['card_id'] == revive['instance_id']
                            for a in legal_actions(s)))
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': revive['instance_id']})
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], hero(s, 'a', 'nanali')['max_hp'])
        self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], 0)

    def test_every_card_has_executable_legal_effect(self):
        for cid in CARDS:
            with self.subTest(card=cid):
                s = self.game()
                s['sides']['b']['front'] = 'zero'
                s['sides']['a']['discard'].append(card_instance(s, 'X01', 'a'))
                owner = CARDS[cid]['character_id']
                for side in ('a', 'b'):
                    if owner not in s['sides'][side]['characters']:
                        s['sides'][side]['characters'][owner] = new_character(owner)
                        s['sides'][side]['order'].append(owner)
                card = self.hand(s, cid)
                s['sides']['a']['harmony_damage'] = True
                s['sides']['a']['surplus'] = True
                s['sides']['a']['front'] = 'nanali'
                dummy = 'zaowu' if owner != 'zaowu' else 'canhong'
                for side in ('a', 'b'):
                    if dummy not in s['sides'][side]['characters']:
                        s['sides'][side]['characters'][dummy] = new_character(dummy)
                        s['sides'][side]['order'].append(dummy)
                    hero(s, side, dummy).update(hp=0, down_turns=3)
                a = next(a for a in legal_actions(s) if a['type'] == 'play_card' and a['card_id'] == card['instance_id'])
                s = apply_action(s, 'a', a)
                if s['phase'] == 'choice':
                    s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] == 'choose'))
                self.assertEqual(s['phase'], 'playing')
                spent = 0 if CARDS[cid].get('instant') else CARDS[cid]['cost']
                self.assertEqual(s['sides']['a']['ap'], 2 - spent)
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
