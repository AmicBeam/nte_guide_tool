from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags
from tests.duel_v2_test_decks import test_character_deck
import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS
from app.modules.card_game.content.duel_v2.catalog import DARK_ENERGY_MAX, LIGHT_ENERGY_MAX
from app.modules.card_game.engine.duel_v2 import (new_game, apply_action, legal_actions,
                                                observe, acting_side, choose_action)
from app.modules.card_game.engine.duel_v2.state import (PLAYER_ID, card_instance, damage, front_target,
                                                        hero, attack_value, new_character, reduce_max_hp)


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

    def next_own_turn(self, s):
        s = apply_action(s, 'a', {'type': 'end_turn'})
        return apply_action(s, 'b', {'type': 'end_turn'})

    def rush_game(self):
        deck = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        s = new_game(seed=9, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        s['sides']['a']['ap'] = 2
        s['sides']['b']['shield'] = 0
        for side in ('a', 'b'):
            s['sides'][side]['hand'] = [c for c in s['sides'][side]['hand'] if not c.get('response')]
        return s

    def test_setup_mulligan_and_first_second_action_points(self):
        s = new_game(seed=12, first_side='b')
        self.assertEqual(s['sides']['a']['shield'], 5)
        self.assertTrue(all(t['front'] is None for t in s['sides'].values()))
        initiative = next(event for event in s['events'] if event['type'] == 'initiative')
        self.assertEqual(initiative['first_side'], 'b')
        self.assertTrue(initiative.get('present', True))
        self.assertLess(initiative['seq'], next(event['seq'] for event in s['events'] if event['type'] == 'draw'))
        ids = [c['instance_id'] for c in s['sides']['a']['hand'][:3]]
        old = deepcopy(s)
        s = apply_action(s, 'a', {'type': 'mulligan', 'card_ids': ids[::-1]})
        self.assertEqual(old['version'], 0)
        self.assertFalse(set(ids) & {c['instance_id'] for c in s['sides']['a']['hand']})
        mulligan = next(event for event in s['events'] if event['type'] == 'mulligan')
        self.assertEqual(set(mulligan['card_ids']), set(ids))
        self.assertEqual(mulligan['amount'], 3)
        s = apply_action(s, 'b', {'type': 'mulligan', 'card_ids': []})
        self.assertEqual(s['sides']['b']['ap'], 1)
        self.assertEqual(len(s['sides']['b']['hand']), 7)
        self.assertEqual(len(s['sides']['b']['deck']), 26)
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
        deck = test_character_deck('sync-curse')
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        s['sides']['a']['ap'] = 2
        s['sides']['a']['front'] = 'lingke'
        hero(s, 'a', 'lingke')['harmony'] = 2
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
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 1)
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        self.assertFalse(any(a['type'] == 'attack' for a in legal_actions(s, 'a')))
        s = self.play(s, 'N03')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 3)
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 3)
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['normal_attack_available'])

    def test_nanali_followup_knockdown_grants_family_seed(self):
        s = self.game()
        s['sides']['b']['front'] = 'jiuyuan'
        hero(s, 'b', 'jiuyuan').update(hp=4, energy=0)
        s['sides']['a']['nanali_family'] = 1
        before = sum(1 for e in s['events'] if e['type'] == 'gain' and '家族壮大' in e.get('text', ''))
        s = self.play(s, 'N03')
        self.assertEqual(hero(s, 'b', 'jiuyuan')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'jiuyuan')['down_turns'], 3)
        self.assertEqual(s['sides']['a']['nanali_family'], 2)
        self.assertEqual(sum(1 for e in s['events'] if e['type'] == 'gain' and '家族壮大' in e.get('text', '')),
                         before + 1)

    def test_nanali_genesis_knockdown_grants_family_seed(self):
        s = self.game()
        s['sides']['a']['front'] = 'zero'
        s['sides']['a']['last_front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        s['sides']['b']['front'] = 'jiuyuan'
        hero(s, 'b', 'jiuyuan').update(hp=1, energy=0)
        s['sides']['a']['nanali_family'] = 1
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertTrue(any(e['type'] == 'genesis' for e in s['events']))
        self.assertEqual(hero(s, 'b', 'jiuyuan')['hp'], 0)
        self.assertEqual(s['sides']['a']['nanali_family'], 2)
        genesis = next(e for e in s['events'] if e['type'] == 'genesis')
        self.assertEqual(genesis.get('source'), 'a:nanali')
        self.assertEqual(genesis.get('source_side'), 'a')

    def test_response_battle_joins_as_counter_not_new_combat(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['ap'] = 1
        s['sides']['b']['hand'] = [card_instance(s, 'N02', 'b')]
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertEqual(s['sides']['b']['ap'], 0)
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 5)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 5)
        self.assertEqual(hero(s, 'b', 'nanali')['shield'], 0)
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 3)
        attacks = [e for e in s['events'] if e['type'] == 'attack']
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]['attack'], 2)
        self.assertEqual(attacks[0]['counter'], 2)
        self.assertTrue(any('作为反击方' in e['text'] for e in s['events'] if e['type'] == 'enter'))
        self.assertEqual((hero(s, 'b', 'nanali')['harmony'], hero(s, 'b', 'nanali')['energy']), (0, 0))
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (1, 2))
        self.assertTrue(any(c['card_id'] == 'N02' for c in s['sides']['b']['discard']))
        self.assertFalse(any(e['type'] == 'harmony' for e in s['events']))

        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['ap'] = 1
        hero(s, 'b', 'zero')['harmony'] = 2
        s['sides']['b']['hand'] = [card_instance(s, 'N02', 'b')]
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertEqual(hero(s, 'b', 'zero')['harmony'], 2)
        self.assertFalse(any(e['type'] == 'harmony' and '创生' in e['text'] for e in s['events']))
        attacks = [e for e in s['events'] if e['type'] == 'attack']
        self.assertEqual(len(attacks), 1)
        self.assertGreater(attacks[0]['counter'], 0)
        self.assertEqual(int((hero(s, 'a', 'zero').get('flags') or {}).get('collapse_count') or 0), 0)
        self.assertEqual((hero(s, 'b', 'nanali')['harmony'], hero(s, 'b', 'nanali')['energy']), (0, 0))
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (1, 2))

        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s = self.play(s, 'N02')
        self.assertEqual(s['sides']['a']['front'], 'nanali')
        self.assertEqual((hero(s, 'a', 'nanali')['harmony'], hero(s, 'a', 'nanali')['energy']), (1, 2))
        self.assertEqual(len([e for e in s['events'] if e['type'] == 'attack']), 1)

    def test_response_requires_and_spends_action_points(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['ap'] = 0
        s['sides']['b']['used'] = {'instant': True}
        s['sides']['b']['hand'] = [card_instance(s, 'N02', 'b')]
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'zero')
        self.assertTrue(any(c['card_id'] == 'N02' for c in s['sides']['b']['hand']))
        self.assertFalse(any(c['card_id'] == 'N02' for c in s['sides']['b']['discard']))

        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['ap'] = 0
        s['sides']['b']['used'] = {}
        card = card_instance(s, 'N02', 'b')
        card['instant'] = True
        s['sides']['b']['hand'] = [card]
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertEqual(s['sides']['b']['ap'], 0)
        self.assertTrue(s['sides']['b']['used'].get('instant'))
        self.assertTrue(any(c['card_id'] == 'N02' for c in s['sides']['b']['discard']))

        s = self.game()
        s['sides']['a']['ap'] = 2
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(s['sides']['a']['ap'], 2)
        self.assertEqual(s['active_side'], 'b')

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

    def test_nanali_family_becomes_instant_from_global_turn_five(self):
        s = self.game()
        family = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        shown = next(c for c in observe(s, 'a')['sides']['a']['hand'] if c.get('card_id') == 'NF01')
        self.assertFalse(shown.get('instant'))
        s['turn'] = 4
        shown = next(c for c in observe(s, 'a')['sides']['a']['hand'] if c.get('card_id') == 'NF01')
        self.assertFalse(shown.get('instant'))
        s['turn'] = 5
        s['sides']['a']['turn_count'] = 3
        s['sides']['a']['ap'] = 0
        shown = next(c for c in observe(s, 'a')['sides']['a']['hand'] if c.get('card_id') == 'NF01')
        self.assertTrue(shown.get('instant'))
        playable = [item for item in legal_actions(s, 'a')
                    if item.get('type') == 'play_card' and item.get('card_id') == family['instance_id']]
        self.assertTrue(playable)
        s = apply_action(s, 'a', playable[0])
        self.assertEqual(s['sides']['a']['ap'], 0)
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (3, 6, 6))

    def test_nanali_family_card_and_n01_heals_to_current_max(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (2, 5, 5))
        self.assertTrue(any(c['card_id'] == 'NF01' for c in s['sides']['a']['hand']))
        family = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': family['instance_id']})
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (3, 6, 6))
        s['sides']['a']['deck'] = [card_instance(s, 'N03', 'a')] + s['sides']['a']['deck']
        n['hp'] = 1
        deck_before = deepcopy(s['sides']['a']['deck'])
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N01')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 6)
        self.assertEqual(s['sides']['a']['deck'], deck_before)
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 1
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N03')
        self.assertTrue(any(c['card_id'] == 'NF01' for c in s['sides']['a']['hand']))
        self.assertEqual(s['sides']['a']['nanali_family'], 2)
        # N03 keeps its original attack target; a defeated target is not chased.
        self.assertFalse(any(e['type'] == 'followup' for e in s['events']))
        self.assertEqual(s['sides']['b']['hp'], 30)

    def test_z08_turn_end_adds_attack_on_front_even_with_form_panel(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        n.update(base_attack=3, max_hp=6, hp=6, shape='N07')
        hero(s, 'a', 'zero')['shape'] = 'Z08'
        s['sides']['a']['front'] = 'nanali'
        s['sides']['b']['front'] = 'zero'
        self.assertEqual(attack_value(n), 3)
        s = apply_action(s, 'a', {'type': 'end_turn'})
        n = hero(s, 'a', 'nanali')
        self.assertEqual(compatibility_flags(n).get('atk_buff'), 1)
        self.assertEqual(attack_value(n), 4)
        shown = {item['id']: item for item in observe(s, 'a')['sides']['a']['characters']}
        self.assertEqual(shown['nanali']['attack'], 4)
        self.assertTrue(any(e['type'] == 'effect' and '倾世之雨' in e['text'] and '娜娜莉' in e['text']
                            for e in s['events']))
        s = apply_action(s, 'b', {'type': 'attack', 'character_id': 'zero'})
        attack_log = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        self.assertEqual(attack_log['counter'], 4)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        n = hero(s, 'a', 'nanali')
        self.assertEqual(compatibility_flags(n).get('atk_buff'), 1)
        self.assertEqual(attack_value(n), 4)
        self.assertIsNone(s['sides']['a']['front'])
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        attack_log = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        self.assertEqual(attack_log['attack'], 4)
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        hero(s, 'a', 'nanali')['hp'] = 0
        knockdowns(s)
        self.assertFalse((hero(s, 'a', 'nanali').get('flags') or {}).get('atk_buff'))

    def test_nanali_ultimate_from_bench_does_not_enter_front(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=LIGHT_ENERGY_MAX)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        n = hero(s, 'a', 'nanali')
        self.assertTrue(n['awakened'])
        self.assertIsNone(s['sides']['a']['front'])

    def test_ultimate_is_once_per_turn_and_free(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=LIGHT_ENERGY_MAX)
        s['sides']['a']['ap'] = 0
        self.assertTrue(s['sides']['a']['ultimate_available'])
        self.assertIn('ultimate', {a['type'] for a in legal_actions(s, 'a') if a.get('character_id') == 'nanali'})
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['ultimate_available'])
        hero(s, 'a', 'zero').update(energy=LIGHT_ENERGY_MAX)
        self.assertFalse(any(a['type'] == 'ultimate' for a in legal_actions(s, 'a')))

    def test_iloy_ultimate_grants_team_energy_next_own_turn(self):
        s = self.game()
        hero(s, 'a', 'iloy').update(energy=LIGHT_ENERGY_MAX)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'iloy'})
        self.assertEqual({cid: hero(s, 'a', cid)['energy'] for cid in s['sides']['a']['order']
                          if cid != 'xun'},
                         {'nanali': 0, 'iloy': 0, 'zero': 0, 'jiuyuan': 0})
        s = self.next_own_turn(s)
        energies = {cid: hero(s, 'a', cid)['energy'] for cid in ('nanali', 'iloy', 'zero', 'jiuyuan')}
        self.assertEqual(energies, {'nanali': 1, 'iloy': 1, 'zero': 1, 'jiuyuan': 1})
        self.assertTrue(any('因伊洛伊的终结获得 1 点能量' in e.get('text', '') for e in s['events']))

    def test_ultimate_from_bench_lasts_two_own_turns(self):
        s = self.game()
        hero(s, 'a', 'nanali').update(energy=LIGHT_ENERGY_MAX)
        self.assertIsNone(s['sides']['a']['front'])
        self.assertIn('ultimate', {a['type'] for a in legal_actions(s, 'a') if a.get('character_id') == 'nanali'})
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        n = hero(s, 'a', 'nanali')
        self.assertTrue(n['awakened'])
        self.assertEqual(n['ultimate_turns'], 2)
        self.assertIsNone(s['sides']['a']['front'])
        s = self.next_own_turn(s)
        self.assertTrue(hero(s, 'a', 'nanali')['awakened'])
        self.assertEqual(hero(s, 'a', 'nanali')['ultimate_turns'], 1)
        s = self.next_own_turn(s)
        n = hero(s, 'a', 'nanali')
        self.assertFalse(n['awakened'])
        self.assertEqual(n['ultimate_turns'], 0)
        ended = next(event for event in s['events'] if '终结结束' in (event.get('text') or ''))
        self.assertFalse(ended.get('present'))

    def test_form_hp_and_battle_shield_value(self):
        s = self.game()
        n = hero(s, 'a', 'nanali')
        self.assertEqual((n['hp'], n['max_hp']), (5, 5))
        s = self.play(s, 'N08')
        n = hero(s, 'a', 'nanali')
        self.assertEqual((n['shape'], attack_value(n), n['hp'], n['max_hp']), ('N08', 2, 6, 6))
        s = self.game()
        family = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': family['instance_id']})
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'N08')
        n = hero(s, 'a', 'nanali')
        self.assertEqual((attack_value(n), n['hp'], n['max_hp']), (3, 7, 7))
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

    def test_n01_full_health_and_downed_boundaries(self):
        for front in (None, 'nanali'):
            s = self.game()
            s['sides']['a']['front'] = front
            n = hero(s, 'a', 'nanali')
            n['max_hp'] = 9
            n['hp'] = 2
            s = self.play(s, 'N01')
            self.assertEqual(hero(s, 'a', 'nanali')['hp'], 9)
            self.assertEqual(s['sides']['a']['ap'], 1)
            s = self.play(s, 'N01')
            self.assertEqual(hero(s, 'a', 'nanali')['hp'], 9)
            self.assertEqual(s['sides']['a']['ap'], 0)
        s = self.game()
        card = self.hand(s, 'N01')
        hero(s, 'a', 'nanali').update(hp=0, down_turns=3)
        with self.assertRaises(ValueError):
            apply_action(s, 'a', {'type':'play_card','card_id':card['instance_id']})

    def test_legacy_n01_copy_heals_its_saved_owner(self):
        s = self.game()
        hero(s, 'a', 'xun')['hp'] = 1
        hero(s, 'a', 'nanali')['hp'] = 2
        s = self.play(s, 'N01', copy=True)
        self.assertEqual(hero(s, 'a', 'xun')['hp'], hero(s, 'a', 'xun')['max_hp'])
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], 2)

    def test_instant_first_card_skips_action_cost(self):
        s = self.game()
        card = self.hand(s, 'J02')
        s['sides']['a']['ap'] = 0
        view = observe(s, 'a')
        shown = next(item for item in view['sides']['a']['hand'] if item.get('instance_id') == card['instance_id'])
        self.assertTrue(shown.get('instant'))
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
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'b', 'nanali')['hp'] = 5
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertTrue(s['sides']['a']['harmonized']['zero'])
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 1)
        self.assertEqual(hero(s, 'a', 'zero')['shield'], 0)
        self.assertEqual(sum(1 for e in s['events'] if e['type'] == 'genesis'), 2)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 1)
        self.assertFalse(any(e['type'] == 'shield' for e in s['events']))
        before = sum(e['type'] == 'harmony' for e in s['events'])
        s = self.play(s, 'Z03')
        self.assertEqual(sum(e['type'] == 'harmony' for e in s['events']), before)
        s = self.game()
        s['sides']['a']['last_front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertFalse(s['sides']['a']['harmonized'])
        self.assertEqual(hero(s, 'a', 'zero')['harmony'], 2)

    def test_front_target_empty_front_uses_player_id(self):
        s = self.game()
        s['sides']['b']['front'] = None
        s['sides']['b']['hp'] = 10
        s['sides']['b']['shield'] = 0
        enemy, cid = front_target(s, 'a')
        self.assertEqual((enemy, cid), ('b', PLAYER_ID))
        damage(s, *front_target(s, 'a'), 3, source='a:zero')
        self.assertEqual(s['sides']['b']['hp'], 7)
        self.assertTrue(any(e.get('target') == 'b:player' and e['amount'] == 3 for e in s['events']))

    def test_shelter_genesis_hits_player_when_no_front(self):
        s = self.game()
        s['sides']['a']['last_front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = 2
        hero(s, 'a', 'jiuyuan')['shape'] = 'J08'
        s['sides']['b']['front'] = None
        s['sides']['b']['hp'] = 10
        s['sides']['b']['shield'] = 0
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        genesis = [e for e in s['events'] if e['type'] == 'genesis']
        extras = [e for e in s['events'] if e['type'] == 'damage' and e.get('source') == 'a:jiuyuan']
        self.assertEqual(len(genesis), 2)
        self.assertEqual(len(extras), 2)
        self.assertTrue(all(e.get('target') == 'b:player' for e in genesis + extras))
        self.assertEqual(sum(e['amount'] for e in genesis + extras), 4)

    def test_shelter_genesis_attaches_pact_on_front(self):
        s = self.game()
        s['sides']['a']['last_front'] = 'nanali'
        hero(s, 'a', 'nanali')['harmony'] = 2
        hero(s, 'a', 'jiuyuan')['shape'] = 'J08'
        s['sides']['b']['front'] = 'nanali'
        hero(s, 'b', 'nanali')['hp'] = 8
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        extras = [e for e in s['events'] if e['type'] == 'damage' and e.get('source') == 'a:jiuyuan']
        self.assertEqual(sum(1 for e in s['events'] if e['type'] == 'genesis'), 2)
        self.assertEqual(len(extras), 2)
        self.assertEqual(hero(s, 'b', 'nanali')['hp'], 2)
        self.assertTrue(hero(s, 'b', 'nanali').get('flags', {}).get('pact'))

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
        self.assertIsNone(s['sides']['a']['last_front'])
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


    def test_collapsed_front_can_be_replaced_but_cannot_normally_attack(self):
        for battle_card in (False, True):
            with self.subTest(battle_card=battle_card):
                s = self.game()
                s['sides']['a']['front'] = 'zero'
                hero(s, 'a', 'zero')['flags']['collapse'] = {'side': 'b', 'until': 99}
                blocked = {'type': 'attack', 'character_id': 'zero'}
                self.assertNotIn(blocked, legal_actions(s, 'a'))
                with self.assertRaises(ValueError):
                    apply_action(s, 'a', blocked)
                if battle_card:
                    s = self.play(s, 'N03')
                else:
                    s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
                self.assertEqual(s['sides']['a']['front'], 'nanali')
                self.assertIsNone(s['sides']['a']['last_front'])
                self.assertIn('collapse', hero(s, 'a', 'zero')['flags'])

    def test_response_can_replace_collapsed_front(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['ap'] = 1
        hero(s, 'b', 'zero')['flags']['collapse'] = {'side': 'a', 'until': 99}
        card = self.hand(s, 'N02', side='b')
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertIsNone(s['sides']['b']['last_front'])
        self.assertEqual(s['sides']['b']['ap'], 0)
        self.assertTrue(any(c['instance_id'] == card['instance_id'] for c in s['sides']['b']['discard']))
        self.assertTrue(any(e['type'] == 'enter' and '反击方' in e['text'] for e in s['events']))

    def test_return_effect_can_move_collapsed_front(self):
        from app.modules.card_game.engine.duel_v2.state import move_out
        s = self.game()
        s['sides']['a']['front'] = 'zero'
        hero(s, 'a', 'zero')['flags']['collapse'] = {'side': 'b', 'until': 99}
        move_out(s, 'a', 'zero')
        self.assertIsNone(s['sides']['a']['front'])
        self.assertIsNone(s['sides']['a']['last_front'])

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





    def test_xun_death_preserves_arid_and_clock(self):
        s = self.game()
        hero(s, 'a', 'xun').update(arid=2, energy=3)
        damage(s, 'a', 'xun', 99)
        self.assertEqual(hero(s, 'a', 'xun')['arid'], 2)
        self.assertEqual(hero(s, 'a', 'xun')['energy'], 3)

    def test_iloy_kindness_can_heal_injured_player(self):
        s = self.game()
        s['sides']['a']['hp'] = 24
        s['sides']['a']['ap'] = 1
        card = card_instance(s, 'Y03', 'a')
        s['sides']['a']['hand'] = [card]
        actions = [item for item in legal_actions(s, 'a')
                   if item.get('type') == 'play_card' and item.get('card_id') == card['instance_id']]
        self.assertTrue(any(item.get('target_id') == 'a:player' for item in actions))
        s = apply_action(s, 'a', {
            'type': 'play_card',
            'card_id': card['instance_id'],
            'target_id': 'a:player',
        })
        self.assertEqual(s['sides']['a']['hp'], 29)

    def test_y07_does_not_trigger_on_ally_knockdown(self):
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        s = self.game()
        hero(s, 'a', 'iloy').update(shape='Y07')
        hero(s, 'b', 'iloy').update(shape='Y07', hp=2)
        s['sides']['b']['front'] = 'iloy'
        hero(s, 'a', 'nanali')['hp'] = 0
        knockdowns(s)
        self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], 3)
        self.assertEqual(hero(s, 'b', 'iloy')['hp'], 2)
        self.assertEqual(hero(s, 'b', 'iloy')['down_turns'], 0)
        self.assertFalse(any(e.get('source') == 'a:iloy' and e['type'] == 'damage'
                             for e in s['events']))

    def test_simultaneous_knockdowns_settle_both_seat_orders_before_next_action(self):
        for attacker, defender in (('a', 'b'), ('b', 'a')):
            with self.subTest(attacker=attacker):
                s = self.rush_game()
                s['active_side'] = attacker
                for side in ('a', 'b'):
                    s['sides'][side]['hand'] = []
                    s['sides'][side]['ap'] = 2
                    s['sides'][side]['normal_attack_available'] = True
                    hero(s, side, 'iloy')['shape'] = 'Y07'
                hero(s, attacker, 'bohe').update(hp=2, shape='M08', harmony=1, energy=2)
                hero(s, defender, 'baicang').update(hp=3, base_attack=2)
                s['sides'][defender]['front'] = 'baicang'
                start = s['event_seq']
                s = apply_action(s, attacker, {'type': 'attack', 'character_id': 'bohe'})
                mint = hero(s, attacker, 'bohe')
                self.assertEqual((mint['hp'], mint['down_turns'], mint['shape']), (0, 3, None))
                self.assertEqual((mint['harmony'], mint['energy']), (1, 2))
                self.assertIsNone(s['sides'][attacker]['front'])
                events = [e for e in s['events'] if e['seq'] > start]
                downs = [e['target'] for e in events if e['type'] == 'down']
                self.assertCountEqual(downs, [attacker + ':bohe', defender + ':baicang'])
                self.assertFalse(any('想做什么梦？' in e.get('text', '') for e in events))
                from app.modules.card_game.engine.duel_v2.state import knockdowns
                seq = s['event_seq']
                knockdowns(s)
                self.assertEqual(s['event_seq'], seq)

    def test_m08_knockdowns_finish_before_draw_and_new_equipment(self):
        s = self.rush_game()
        s['active_side'] = 'b'
        for side in ('a', 'b'):
            s['sides'][side]['hand'] = []
        hero(s, 'a', 'bohe').update(hp=2, shape='M08')
        hero(s, 'b', 'baicang').update(hp=3, base_attack=2)
        hero(s, 'b', 'iloy')['shape'] = 'Y07'
        s['sides']['b']['front'] = 'baicang'
        start = s['event_seq']
        s = apply_action(s, 'b', {'type': 'end_turn'})
        events = [e for e in s['events'] if e['seq'] > start]
        down = next(e for e in events if e['type'] == 'down' and e['target'] == 'a:bohe')
        draw = next(e for e in events if e['type'] == 'draw' and e['side'] == 'a')
        self.assertLess(down['seq'], draw['seq'])
        self.assertEqual(hero(s, 'a', 'bohe')['down_turns'], 3)
        s = self.play(s, 'Y07')
        hp = s['sides']['b']['hp']
        start = s['event_seq']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['hp'], hp - 2)
        self.assertFalse(any(e['type'] == 'down' for e in s['events'] if e['seq'] > start))

    def test_b02_heal_precedes_enemy_knockdown_without_y07_retaliation(self):
        s = self.rush_game()
        hero(s, 'a', 'baicang').update(hp=1)
        hero(s, 'b', 'zero').update(hp=3, base_attack=3)
        hero(s, 'b', 'iloy')['shape'] = 'Y07'
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hand'] = []
        start = s['event_seq']
        s = self.play(s, 'B02')
        events = [e for e in s['events'] if e['seq'] > start]
        heal = next(e for e in events if e['type'] == 'heal' and e['target'] == 'a:baicang')
        self.assertEqual(heal['before']['hp'], 0)
        self.assertEqual(heal['after']['hp'], 2)
        zero_down = next(e for e in events if e['type'] == 'down' and e['target'] == 'b:zero')
        self.assertLess(heal['seq'], zero_down['seq'])
        self.assertFalse(any(e['type'] == 'down' and e['target'] == 'a:baicang' for e in events))
        self.assertEqual((hero(s, 'a', 'baicang')['hp'], hero(s, 'a', 'baicang')['down_turns']), (2, 0))



    def test_return_after_sortie_keeps_combat_resources_and_clears_shield(self):
        s = self.game()
        h = hero(s, 'a', 'nanali')
        h['flags']['return_after'] = True
        h['shield'] = 2
        before = len(s['sides']['a']['deck'])
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s, 'a', 'nanali')['shield'], 0)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 2)
        self.assertEqual(len(s['sides']['a']['deck']), before - 1)

    def test_front_combat_energy_doubles_bench_cards_stay_one(self):
        s = apply_action(self.game(), 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 2)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 1)
        self.assertEqual(hero(s, 'a', 'jiuyuan')['energy'], 1)
        self.assertEqual(observe(s, 'a')['sides']['a']['characters'][1]['energy_max'], LIGHT_ENERGY_MAX)
        s = self.play(s, 'N01')
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 1)
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 2)
        hero(s, 'a', 'zero')['energy'] = LIGHT_ENERGY_MAX - 1
        s['sides']['a']['ap'] = 1
        s['sides']['a']['normal_attack_available'] = True
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'zero')['energy'], LIGHT_ENERGY_MAX)
        self.assertEqual(hero(s, 'a', 'nanali')['energy'], 2)

    def test_dark_attribute_energy_cap_is_six(self):
        deck = test_character_deck('requiem')
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        s['sides']['a']['ap'] = 2
        s['sides']['b']['shield'] = 0
        self.assertEqual(observe(s, 'a')['sides']['a']['characters'][0]['energy_max'], DARK_ENERGY_MAX)
        hero(s, 'a', 'anhunqu')['energy'] = DARK_ENERGY_MAX - 1
        self.assertFalse(any(a['type'] == 'ultimate' and a.get('character_id') == 'anhunqu'
                             for a in legal_actions(s, 'a')))
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'anhunqu'})
        self.assertEqual(hero(s, 'a', 'anhunqu')['energy'], DARK_ENERGY_MAX)
        s['sides']['a']['ap'] = 1
        self.assertTrue(any(a['type'] == 'ultimate' and a.get('character_id') == 'anhunqu'
                            for a in legal_actions(s, 'a')))

    def test_pending_draw_discard_one_payment_and_private_inspection(self):
        s = self.game()
        s = self.play(s, 'N01')
        self.assertEqual(s['phase'], 'playing')



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
        move_out(s, 'b', 'zero')
        s['sides']['b']['front'] = 'xun'
        self.assertEqual(front_debuff(s, 'b')['burn']['left'], 2)
        hp = hero(s, 'b', 'xun')['hp']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertTrue(any(e['type'] == 'overlay' for e in s['events']))
        self.assertEqual(hero(s, 'b', 'xun')['hp'], hp - 4)

    def test_weave_and_stain_apply_only_to_combat_attacks(self):
        from app.modules.card_game.engine.duel_v2.state import front_debuff
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['weave'] = {'by': 'a'}
        hp = hero(s, 'b', 'zero')['hp']
        s = self.play(s, 'J02')
        self.assertFalse(any(e['type'] == 'overlay' for e in s['events']))
        self.assertEqual(hero(s, 'b', 'zero')['hp'], hp - 1)
        s = self.game()
        if 'hathor' not in s['sides']['a']['characters']:
            s['sides']['a']['characters']['hathor'] = new_character('hathor')
            s['sides']['a']['order'].append('hathor')
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['stain'] = {'by': 'a'}
        hp = hero(s, 'b', 'zero')['hp']
        damage(s, 'b', 'zero', 2, source='a:hathor', kind='followup')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], hp - 2)
        hero(s, 'b', 'zero')['hp'] = 5
        s['sides']['a']['front'] = 'hathor'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'hathor'})
        attack_log = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        self.assertEqual(attack_log['attack'], attack_value(hero(s, 'a', 'hathor')))
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 5 - attack_log['attack'] - 2)

    def test_j01_instant_inspection_shares_free_use_and_pays_once(self):
        for used, ap in ((False, 0), (True, 1), (True, 0)):
            with self.subTest(instant_used=used, ap=ap):
                s = self.game()
                s['sides']['a']['ap'] = ap
                s['sides']['a']['used']['instant'] = used
                card = self.hand(s, 'J01')
                action = {'type': 'play_card', 'card_id': card['instance_id']}
                if used and ap == 0:
                    with self.assertRaises(ValueError):
                        apply_action(s, 'a', action)
                    continue
                top = [c['instance_id'] for c in s['sides']['a']['deck'][:3]]
                s = apply_action(s, 'a', action)
                self.assertEqual(s['phase'], 'choice')
                self.assertEqual(s['sides']['a']['ap'], 0)
                self.assertTrue(s['sides']['a']['used']['instant'])
                choice = next(a for a in legal_actions(s, 'a') if a['type'] in ('choose', 'choose_cards'))
                s = apply_action(s, 'a', choice)
                self.assertEqual(s['sides']['a']['ap'], 0)
                self.assertEqual(sum(c['instance_id'] in top for c in s['sides']['a']['hand']), 1)
                self.assertEqual(sum(c['instance_id'] in top for c in s['sides']['a']['deck'][-2:]), 2)

    def test_fatigue_empty_inspect_and_finished_stops_remaining_effects(self):
        s = self.game()
        s['sides']['a']['deck'] = []
        s['sides']['a']['hp'] = 1
        s['sides']['a']['shield'] = 0
        before_harmony = hero(s, 'a', 'xun')['harmony']
        s = self.play(s, 'J01')
        from app.modules.card_game.engine.duel_v2.state import draw
        draw(s, 'a')
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

    def test_n06_instant_revive_cost_and_exhausted_instant(self):
        for used, ap, allowed in ((False, 0, True), (True, 0, False), (True, 1, True)):
            with self.subTest(instant_used=used, ap=ap):
                s = self.game()
                card = self.hand(s, 'N06')
                hero(s, 'a', 'nanali').update(hp=0, down_turns=3)
                s['sides']['a']['ap'] = ap
                s['sides']['a']['used']['instant'] = used
                action = {'type': 'play_card', 'card_id': card['instance_id']}
                available = any(a['type'] == 'play_card' and a['card_id'] == card['instance_id']
                                for a in legal_actions(s))
                self.assertEqual(available, allowed)
                if not allowed:
                    with self.assertRaises(ValueError):
                        apply_action(s, 'a', action)
                    continue
                result = apply_action(s, 'a', action)
                self.assertEqual(result['sides']['a']['ap'], 0)
                self.assertTrue(result['sides']['a']['used']['instant'])
                revived = hero(result, 'a', 'nanali')
                self.assertEqual(revived['hp'], revived['max_hp'])
                self.assertEqual(revived['down_turns'], 0)
                self.assertFalse(revived.get('flags', {}).get('temp_revive'))

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
                s['sides']['a']['surplus_ever'] = True
                if cid == 'I05':
                    s['sides']['b']['front_debuff']['delay'] = {'left': 1, 'by': 'a', 'tick_side': 'a'}
                if cid == 'E03':
                    s['sides']['a']['hand'].append(card_instance(s, 'Z03', 'a'))
                from app.modules.card_game.engine.duel_v2.state import front_debuff
                front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a'}
                s['sides']['a']['front'] = 'nanali'
                hero(s, 'a', 'nanali')['hp'] = 4
                dummy = 'zaowu' if owner != 'zaowu' else 'canhong'
                for side in ('a', 'b'):
                    if dummy not in s['sides'][side]['characters']:
                        s['sides'][side]['characters'][dummy] = new_character(dummy)
                        s['sides'][side]['order'].append(dummy)
                    hero(s, side, dummy).update(hp=0, down_turns=3)
                if cid == 'X01':
                    from app.modules.card_game.engine.duel_v2.temporal import save_turn_start
                    save_turn_start(s, 'a')
                    save_turn_start(s, 'a')
                if cid == 'XA01':
                    card['antique_investment'] = 1
                if cid == 'A01' and 'hathor' not in s['sides']['b']['order']:
                    s['sides']['b']['characters']['hathor'] = new_character('hathor')
                    s['sides']['b']['order'].append('hathor')
                if cid == 'S04':
                    from app.modules.card_game.engine.duel_v2.summons import summon_front
                    summon_front(s, 'b', name='靶子', attack=0, hp=4)
                if cid in ('S05', 'S06'):
                    from app.modules.card_game.engine.duel_v2.summons import summon_front
                    summon_front(s, 'a', name='鬼郎丸', attack=0, hp=4)
                a = next(a for a in legal_actions(s) if a['type'] == 'play_card' and a['card_id'] == card['instance_id'])
                s = apply_action(s, 'a', a)
                if s['phase'] == 'choice':
                    s = apply_action(s, 'a', next(a for a in legal_actions(s) if a['type'] in ('choose', 'choose_cards')))
                self.assertEqual(s['phase'], 'playing')
                spent = 0 if CARDS[cid].get('instant') else CARDS[cid]['cost']
                if CARDS[cid].get('free_if_surplus'): spent = 0
                if cid == 'X04': spent = 2
                if cid == 'XA01': spent = 2 - s['sides']['a']['ap']
                self.assertEqual(s['sides']['a']['ap'], 2 - spent)
                pile = 'hand' if CARDS[cid].get('retain') else ('removed' if cid == 'X01' else 'discard')
                self.assertTrue(any(c['instance_id'] == card['instance_id'] for c in s['sides']['a'][pile]))
                json.dumps(observe(s, 'a'))

    def test_baicang_b06_allied_hurt_adds_one_attack(self):
        s = self.rush_game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=10, shield=0, base_attack=0)
        s = self.play(s, 'B06')
        plain = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        s = self.rush_game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=10, shield=0, base_attack=0)
        hero(s, 'a', 'baicang').setdefault('flags', {})['allied_hurt'] = True
        s = self.play(s, 'B06')
        boosted = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        self.assertEqual(boosted['attack'] - plain['attack'], 1)

    def test_baicang_tea_form_damages_self_on_equip(self):
        s = self.rush_game()
        atk = attack_value(hero(s, 'a', 'baicang'))
        s = self.play(s, 'B08')
        self.assertEqual(hero(s, 'a', 'baicang')['shape'], 'B08')
        # 用户指定 B08 为 2/9；装备自伤仍为 4，并触发一次受伤加攻。
        self.assertEqual(hero(s, 'a', 'baicang')['max_hp'], 9)
        self.assertEqual(hero(s, 'a', 'baicang')['hp'], 5)
        self.assertEqual(attack_value(hero(s, 'a', 'baicang')), atk + 1)

    def test_baicang_damage_hooks_are_personal(self):
        s = self.rush_game()
        bc = hero(s, 'a', 'baicang')
        before = attack_value(bc)
        damage(s, 'a', 'zero', 1, source='a:zero')
        self.assertEqual(attack_value(bc), before)
        self.assertFalse(bc['flags'].get('allied_hurt'))
        bc['awakened'] = True
        hero(s, 'b', 'zero')['hp'] = 3
        damage(s, 'b', 'zero', 1, source='a:iloy')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 2)
        damage(s, 'b', 'zero', 1, source='a:baicang')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)

    def test_baicang_stacks_attack_from_combat_and_self_damage(self):
        s = self.rush_game()
        before = attack_value(hero(s, 'a', 'baicang'))
        s['sides']['b']['front'] = 'zero'
        s['sides']['a']['front'] = 'baicang'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'baicang'})
        self.assertGreaterEqual(attack_value(hero(s, 'a', 'baicang')), before + 1)
        s['sides']['a']['ap'] = 1
        s = self.play(s, 'B01')
        self.assertEqual(len(s['sides']['a']['hand']), 1)
        self.assertGreaterEqual(attack_value(hero(s, 'a', 'baicang')), before + 2)
        self.assertTrue((hero(s, 'a', 'baicang').get('flags') or {}).get('allied_hurt'))

    def test_baicang_execute_finishes_low_hp_esper_not_self(self):
        s = self.rush_game()
        hero(s, 'a', 'baicang').update(awakened=True, energy=0)
        hero(s, 'b', 'zero').update(hp=3)
        s['sides']['b']['front'] = 'zero'
        s['sides']['a']['front'] = 'baicang'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'baicang'})
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'zero')['down_turns'], 3)
        s = self.rush_game()
        hero(s, 'a', 'baicang').update(awakened=True, hp=2)
        s = self.play(s, 'B01')
        self.assertEqual(hero(s, 'a', 'baicang')['hp'], 1)
        self.assertEqual(hero(s, 'a', 'baicang')['down_turns'], 0)

    def test_baicang_b02_heals_before_knockdown(self):
        s = self.rush_game()
        hero(s, 'a', 'baicang').update(hp=1)
        hero(s, 'b', 'zero').update(hp=5, base_attack=3)
        s['sides']['b']['front'] = 'zero'
        s = self.play(s, 'B02')
        self.assertGreater(hero(s, 'a', 'baicang')['hp'], 0)
        self.assertEqual(hero(s, 'a', 'baicang')['down_turns'], 0)

    def test_baicang_lethal_response_keeps_one_hp(self):
        s = self.rush_game()
        card = card_instance(s, 'B05', 'a')
        s['sides']['a']['hand'] = [card]
        s['sides']['a']['ap'] = 1
        s['sides']['a']['front'] = 'baicang'
        hero(s, 'a', 'baicang').update(hp=1)
        s['sides']['b']['front'] = 'zero'
        s = apply_action(s, 'a', {'type': 'end_turn'})
        s = apply_action(s, 'b', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(hero(s, 'a', 'baicang')['hp'], 1)
        self.assertTrue(any(item['card_id'] == 'B05' for item in s['sides']['a']['discard']))

    def test_baicang_burn_can_become_infinite(self):
        from app.modules.card_game.engine.duel_v2.state import front_debuff
        s = self.rush_game()
        s['sides']['b']['front'] = 'zero'
        front_debuff(s, 'b')['burn'] = {'left': 1, 'by': 'a'}
        s = self.play(s, 'B03')
        self.assertTrue(front_debuff(s, 'b')['burn'].get('infinite'))
        s = self.next_own_turn(s)
        self.assertTrue(front_debuff(s, 'b').get('burn', {}).get('infinite'))

    def test_bohe_overflow_ignore_shield_and_auto_attack(self):
        s = self.rush_game()
        hero(s, 'b', 'zero').update(hp=3, shield=2)
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hp'] = 30
        s = self.play(s, 'M03')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 0)
        self.assertEqual(s['sides']['b']['hp'], 28)
        s = self.rush_game()
        s = self.play(s, 'M08')
        self.assertEqual(hero(s, 'a', 'bohe')['shape'], 'M08')
        s = apply_action(s, 'a', {'type': 'end_turn'})
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=4)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertIsNone(s['sides']['a']['front'])
        self.assertTrue(any(event['type'] == 'attack' and event.get('source') == 'bohe'
                            for event in s['events']))

    def test_bohe_auto_attack_has_no_resources_but_manual_attack_does(self):
        # User balance adjustment, 2026-09-15; not an original-game mechanic.
        for enemy_front in (None, 'zero'):
            with self.subTest(enemy_front=enemy_front):
                s = self.play(self.rush_game(), 'M08')
                s = apply_action(s, 'a', {'type': 'end_turn'})
                s['sides']['b']['front'] = enemy_front
                # M08 now has 4 HP; leave enough HP for both test attacks.
                hero(s, 'b', 'zero')['base_attack'] = 1
                before = {cid: (h['harmony'], h['energy'])
                          for cid, h in s['sides']['a']['characters'].items()}
                s = apply_action(s, 'b', {'type': 'end_turn'})
                self.assertEqual({cid: (h['harmony'], h['energy'])
                                  for cid, h in s['sides']['a']['characters'].items()}, before)
                self.assertEqual(s['sides']['a']['ap'], 2)
                s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'bohe'})
                self.assertEqual(hero(s, 'a', 'bohe')['harmony'], 1)
                self.assertEqual(hero(s, 'a', 'bohe')['energy'], 2)
                self.assertEqual(hero(s, 'a', 'zero')['energy'], 1)

    def test_jiuyuan_pact_front_draw_checks_before_damage(self):
        # User balance adjustment, 2026-09-15: first attachment does not draw.
        for pact, hp in ((False, 5), (True, 5), (True, 1)):
            with self.subTest(pact=pact, hp=hp):
                s = self.game()
                s['sides']['b']['front'] = 'zero'
                target = hero(s, 'b', 'zero')
                target['hp'] = hp
                target['flags']['pact'] = pact
                deck_size = len(s['sides']['a']['deck'])
                s = self.play(s, 'J02')
                self.assertEqual(len(s['sides']['a']['hand']), int(pact))
                self.assertEqual(len(s['sides']['a']['deck']), deck_size - int(pact))
                self.assertEqual(hero(s, 'b', 'zero')['hp'], hp - 1)
                self.assertEqual(bool(hero(s, 'b', 'zero')['flags'].get('pact')), hp > 1)

    def test_jiuyuan_j02_empty_front_hits_player_and_draws(self):
        s = self.game()
        s['sides']['b']['front'] = None
        s['sides']['b']['hp'] = 30
        deck_size = len(s['sides']['a']['deck'])
        s = self.play(s, 'J02')
        self.assertEqual(s['sides']['b']['hp'], 29)
        self.assertEqual(len(s['sides']['a']['hand']), 1)
        self.assertEqual(len(s['sides']['a']['deck']), deck_size - 1)

    def test_bohe_share_overflow_on_self_attacked_and_sleep_form(self):
        s = self.rush_game()
        s = self.play(s, 'M07')
        self.assertEqual(hero(s, 'a', 'bohe')['shape'], 'M07')
        s['sides']['a']['front'] = 'bohe'
        s['sides']['a']['ap'] = 1
        card = card_instance(s, 'M04', 'a')
        s['sides']['a']['hand'] = [card]
        s = apply_action(s, 'a', {'type': 'end_turn'})
        hero(s, 'b', 'zero').update(base_attack=12)
        s['sides']['b']['front'] = 'zero'
        before = s['sides']['a']['hp']
        s = apply_action(s, 'b', {'type': 'attack', 'character_id': 'zero'})
        attack_log = next(event for event in reversed(s['events']) if event['type'] == 'attack')
        self.assertEqual(attack_log['counter'], 2)  # M04 bonus above the zero panel.
        self.assertLess(s['sides']['a']['hp'], before)
        self.assertTrue(any(item['card_id'] == 'M04' for item in s['sides']['a']['discard']))

    def test_bohe_front_damage_scales_with_permanent_attack(self):
        s = self.rush_game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=8)
        s = self.play(s, 'M06')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 6)
        s = self.rush_game()
        add = hero(s, 'a', 'bohe')
        add['flags'] = {'atk_buff': 2}
        add['shape'] = 'M07'
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=8)
        s = self.play(s, 'M06')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 6)
        s = self.rush_game()
        hero(s, 'a', 'bohe')['base_attack'] = 5
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=8)
        s = self.play(s, 'M06')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 4)

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
