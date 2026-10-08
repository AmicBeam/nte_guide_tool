"""User-authored Heiyu tabletop rules; numbers are original playtest values."""
import json
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game, observe
from app.modules.card_game.engine.duel_v2.combat import attack
from app.modules.card_game.engine.duel_v2.entities import clone_state, entity_action
from app.modules.card_game.engine.duel_v2.flow import finish_operation, operation
from app.modules.card_game.engine.duel_v2.state import attack_value, card_instance, hero, knockdowns
from app.modules.card_game.engine.duel_v2.summons import summon_front


TEAM = ['heiyu', 'zero', 'nanali', 'jiuyuan']


def deck_for(team=TEAM):
    cards = [cid for owner in team for cid, card in CARDS.items()
             if card['character_id'] == owner and not card.get('derived')]
    return validate_deck(dict(id='heiyu-test', name='黑羽测试', character_ids=list(team), card_ids=cards))


def raw_game(team=TEAM):
    deck = deck_for(team)
    s = new_game(seed=42, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck}, escalation=False)
    for team_state in s['sides'].values():
        team_state.update(ap=10, shield=0, hp=80)
        for h in team_state['characters'].values():
            h.update(hp=h['max_hp'])
    card = card_instance(s, 'P02', 'a')
    s['sides']['a']['hand'].append(card)
    return apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})


def resolve_order(s, bottom=None):
    while s.get('phase') == 'choice' and (s.get('pending_choice') or {}).get('kind') == 'heiyu_divination':
        pending = s['pending_choice']
        side = pending['side']
        ids = [card['instance_id'] for card in pending['cards']]
        bottom_id = None
        if bottom:
            match = next((card for card in pending['cards']
                          if card['card_id'] == bottom or card['instance_id'] == bottom), None)
            if match:
                bottom_id = match['instance_id']
        if bottom_id:
            action = {'type': 'choose_deck_order',
                      'card_ids': [item for item in ids if item != bottom_id],
                      'bottom_card_id': bottom_id}
        else:
            action = {'type': 'choose_deck_order', 'card_ids': ids, 'bottom_card_id': None}
        s = apply_action(s, side, action)
    return s


def game(team=TEAM):
    s = resolve_order(raw_game(team))
    for team_state in s['sides'].values():
        team_state['hand'] = []
        team_state['ap'] = 10
        team_state['used']['instant'] = False
    return s


def give(s, cid, side='a'):
    card = card_instance(s, cid, side)
    s['sides'][side]['hand'].append(card)
    return card


def play(s, cid, target=None, side='a'):
    card = give(s, cid, side)
    action = {'type': 'play_card', 'card_id': card['instance_id']}
    if target:
        action['target_id'] = target
    return apply_action(s, side, action), card


def set_top(s, cids, side='a'):
    team = s['sides'][side]
    cards = [card_instance(s, cid, side) for cid in cids]
    team['deck'] = cards
    return cards



def pass_to_own_inspect(s):
    s = apply_action(s, 'a', {'type': 'end_turn'})
    s = resolve_order(s)
    s = apply_action(s, 'b', {'type': 'end_turn'})
    return play(s, 'P02')[0]


class HeiyuTest(unittest.TestCase):
    def test_catalog_stats_cards_and_hidden_access(self):
        c = CHARACTERS['heiyu']
        self.assertEqual((c['name'], c['attribute'], c['attack'], c['max_hp'], c['energy_max']),
                         ('黑羽', '魂', 2, 5, 6))
        self.assertEqual([CARDS[f'P{i:02d}']['name'] for i in range(1, 9)],
                         ['叛逃者', '漫游记', '失明症', '坏骨头', '烂柯人', '死与生', '牧星者', '不凋羽'])
        self.assertIn('救世魔女期间，黑羽的主动攻击获得穿透', CARDS['P08']['description'])
        self.assertNotIn('heiyu', {item['id'] for item in get_catalog(include_test_characters=False)['characters']})
        self.assertIn('heiyu', {item['id'] for item in get_catalog()['characters']})
        for field in ('avatar', 'portrait'):
            self.assertTrue(Path('app' + c[field]).is_file(), c[field])

    def test_inspection_uses_living_roster_and_excludes_summons(self):
        s = raw_game()
        pending = s['pending_choice']
        self.assertEqual(pending['kind'], 'heiyu_divination')
        self.assertEqual(pending['snapshot_n'], 4)
        self.assertEqual(len(pending['cards']), 4)
        s = resolve_order(s)
        hero(s, 'a', 'nanali').update(hp=0, down_turns=3)
        hero(s, 'a', 'jiuyuan').update(hp=0, down_turns=3)
        s = pass_to_own_inspect(s)
        pending = s['pending_choice']
        self.assertEqual(pending['snapshot_n'], 2)
        self.assertEqual(len(pending['cards']), 2)
        s = resolve_order(s)
        summon_front(s, 'a', name='鸦', attack=0, hp=3)
        s = pass_to_own_inspect(s)
        self.assertEqual(s['pending_choice']['snapshot_n'], 2)

    def test_recovery_updates_future_n_and_witch_skips_auto_inspect(self):
        s = resolve_order(raw_game())
        hero(s, 'a', 'nanali').update(hp=0, down_turns=1)
        s = pass_to_own_inspect(s)
        pending = s['pending_choice']
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], hero(s, 'a', 'nanali')['max_hp'])
        self.assertEqual(pending['snapshot_n'], 4)
        s = resolve_order(s)
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        s = apply_action(s, 'a', {'type': 'end_turn'})
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertNotEqual((s.get('pending_choice') or {}).get('kind'), 'heiyu_divination')
        self.assertTrue(hero(s, 'a', 'heiyu')['awakened'])

    def test_bottom_or_keep_single_card_and_private_projection(self):
        s = raw_game()
        s = resolve_order(s)
        s['sides']['a']['deck'] = [card_instance(s, 'P01', 'a')]
        s['sides']['a']['hand'] = []
        s, _ = play(s, 'P02')
        pending = s['pending_choice']
        self.assertEqual(len(pending['cards']), 1)
        actions = [item for item in legal_actions(s, 'a') if item['type'] == 'choose_deck_order']
        self.assertEqual(len(actions), 2)
        owner = observe(s, 'a')
        foe = observe(s, 'b')
        self.assertEqual(owner['pending_choice']['kind'], 'heiyu_divination')
        self.assertEqual(len(owner['pending_choice']['choices']), 1)
        self.assertEqual(foe['pending_choice']['choices'], [])
        self.assertEqual(owner['sides']['a']['deck_count'], 1)
        self.assertEqual(foe['sides']['a']['deck_count'], 1)
        card = pending['cards'][0]
        old = deepcopy(s)
        with self.assertRaises(ValueError):
            apply_action(s, 'a', {'type': 'choose_deck_order', 'card_ids': [], 'bottom_card_id': 'nope'})
        self.assertEqual(s['pending_choice']['cards'][0]['instance_id'], old['pending_choice']['cards'][0]['instance_id'])
        roundtrip = clone_state(json.loads(json.dumps(s)))
        s = apply_action(roundtrip, 'a', {'type': 'choose_deck_order', 'card_ids': [],
                                         'bottom_card_id': card['instance_id']})
        self.assertEqual(s['sides']['a']['deck'][-1]['instance_id'], card['instance_id'])
        self.assertTrue(any('公开占卜' in event['text'] for event in s['events']))

    def test_omen_inserted_on_active_sortie_without_harmony_fill(self):
        s = game()
        before = [c['entity_id'] for c in s['sides']['a']['deck']]
        hero(s, 'a', 'heiyu')['harmony'] = 0
        s, _ = play(s, 'P01')
        self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], 1)
        deck = s['sides']['a']['deck']
        self.assertEqual(len(deck), len(before) + 1)
        self.assertEqual([c['entity_id'] for c in deck if c['entity_id'] in before], before)
        self.assertTrue(any(c['card_id'] == 'PF01' for c in deck))

    def test_omen_draw_waits_for_confirmation(self):
        s = game()
        top = set_top(s, ['P01', 'P03', 'P04', 'Z01'])
        s, _ = play(s, 'PF01')
        self.assertEqual(len(s['sides']['a']['hand']), 0)
        s = resolve_order(clone_state(json.loads(json.dumps(s))))
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['hand']], [top[0]['entity_id']])
        self.assertEqual(len(s['sides']['a']['deck']), 3)

    def test_witch_retains_harmony_strengthening_and_star_renews_once(self):
        from app.modules.card_game.engine.duel_v2.combat import resolve_harmony_effect
        from app.modules.card_game.engine.duel_v2.flow import tick_front_debuff, expire_turn_zone_states
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        from app.modules.card_game.engine.duel_v2.state import front_debuff
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        c = EffectContext(s, 'a', 'zero', {'side': 'a', 'actor': 'zero'})
        resolve_harmony_effect(c, '浸染')
        expire_turn_zone_states(s, 'a')
        self.assertIn('stain', front_debuff(s, 'b'))
        expire_turn_zone_states(s, 'b')
        self.assertNotIn('stain', front_debuff(s, 'b'))
        resolve_harmony_effect(c, '黯星')
        tick_front_debuff(s, 'b')
        tick_front_debuff(s, 'b')
        self.assertEqual(front_debuff(s, 'b')['star']['left'], 2)
        tick_front_debuff(s, 'b')
        tick_front_debuff(s, 'b')
        self.assertNotIn('star', front_debuff(s, 'b'))

    def test_ultimate_exit_keeps_team_buff_and_costs_nothing(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s['sides']['a']['ultimates_used'] = 0
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        self.assertTrue(hero(s, 'a', 'heiyu')['awakened'])
        self.assertEqual(hero(s, 'a', 'heiyu')['energy'], 0)
        buffed = attack_value(hero(s, 'a', 'zero'), s, 'a')
        self.assertEqual(buffed, attack_value(hero(s, 'a', 'heiyu'), s, 'a'))
        actions = legal_actions(s, 'a')
        self.assertTrue(any(item['type'] == 'exit_ultimate' and item['character_id'] == 'heiyu' for item in actions))
        self.assertFalse(any(item['type'] == 'ultimate' and item['character_id'] == 'heiyu' for item in actions))
        view = observe(s, 'a')
        exit_action = next(item for item in view['legal_actions'] if item['action']['type'] == 'exit_ultimate')
        self.assertEqual(exit_action['label'], '退出救世魔女')
        self.assertEqual(exit_action['interaction']['kind'], 'ultimate')
        self.assertEqual(exit_action['interaction']['actor_id'], 'a:heiyu')
        used = s['sides']['a']['ultimates_used']
        ap = s['sides']['a']['ap']
        energy = hero(s, 'a', 'heiyu')['energy']
        addressed = entity_action(s, 'a', {'type': 'exit_ultimate', 'character_id': 'heiyu'})
        s = apply_action(s, 'a', addressed)
        self.assertFalse(hero(s, 'a', 'heiyu')['awakened'])
        self.assertEqual(s['sides']['a']['ultimates_used'], used)
        self.assertEqual(s['sides']['a']['ap'], ap)
        self.assertEqual(hero(s, 'a', 'heiyu')['energy'], energy)
        self.assertEqual(attack_value(hero(s, 'a', 'zero'), s, 'a'), buffed)

    def test_p04_instant_after_divination_and_p05_p06_normals(self):
        s = game()
        s['sides']['a']['ap'] = 0
        card = give(s, 'P04')
        self.assertTrue(any(item.get('card_id') == card['instance_id'] and item['type'] == 'play_card'
                            for item in legal_actions(s, 'a')))
        s = game()
        s, _ = play(s, 'P05', target='b:zero')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], hero(s, 'b', 'zero')['max_hp'] - 3)
        hero(s, 'a', 'nanali').update(hp=0, down_turns=2)
        s, _ = play(s, 'P06', target='a:nanali')
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], hero(s, 'a', 'nanali')['max_hp'])

    def test_p01_magic_chain_has_no_basic_resources(self):
        s = game()
        hero(s, 'a', 'heiyu').update(energy=6, harmony=0)
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        set_top(s, ['P01', 'P01', 'P01'])
        s['sides']['a']['front'] = 'heiyu'
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hp'] = 80
        before_h = hero(s, 'a', 'heiyu')['harmony']
        before_e = {cid: hero(s, 'a', cid)['energy'] for cid in TEAM}
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        attacks = [event for event in s['events'] if event['type'] == 'attack' and event.get('actor') == 'a:heiyu']
        self.assertGreaterEqual(len(attacks), 4)
        self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], before_h + 1)
        for cid in TEAM:
            if cid == 'heiyu':
                self.assertEqual(hero(s, 'a', cid)['energy'], before_e[cid] + 2)
            else:
                self.assertEqual(hero(s, 'a', cid)['energy'], before_e[cid] + 1)

    def test_p02_magic_choice_resumes_and_keeps_operation(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        set_top(s, ['P02'])
        held = give(s, 'P04')
        s['sides']['a']['front'] = 'heiyu'
        s['sides']['b']['front'] = 'zero'
        s['sides']['a']['ap'] = 10
        s['sides']['a']['normal_attack_available'] = True
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertEqual(s['phase'], 'choice')
        self.assertEqual(s['pending_choice']['kind'], 'kit')
        self.assertEqual(s['sides']['a']['ap'], 9)
        self.assertFalse(s['sides']['a']['normal_attack_available'])
        dumped = json.dumps(s)
        s = apply_action(clone_state(json.loads(dumped)), 'a', {'type': 'choose', 'choice_id': held['instance_id']})
        self.assertEqual(s['phase'], 'playing')
        self.assertEqual(s['sides']['a']['deck'][0]['entity_id'], held['entity_id'])
        self.assertFalse(any(card.get('entity_id') == held['entity_id'] for card in s['sides']['a']['hand']))
        self.assertEqual(s['sides']['a']['ap'], 9)

    def test_p03_defense_only_when_targeted_and_expires_per_battle(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        tops = set_top(s, ['P03'])
        top_entity_id = tops[0]['entity_id']
        hero(s, 'b', 'zero').update(hp=20, max_hp=20)
        s['sides']['a']['front'] = 'nanali'
        s['sides']['b']['front'] = 'zero'
        ctx = operation(s, 'b', 'zero')
        attack(ctx, {}, 0)
        self.assertEqual(s['sides']['a']['deck'][0]['entity_id'], top_entity_id)
        s['sides']['a']['front'] = 'heiyu'
        hp = hero(s, 'a', 'heiyu')['hp']
        attack(ctx, {}, 0)
        self.assertEqual(hero(s, 'a', 'heiyu')['hp'], hp)
        self.assertTrue(any(card.get('entity_id') == top_entity_id for card in s['sides']['a']['discard']))
        attack(ctx, {}, 0)
        self.assertLess(hero(s, 'a', 'heiyu')['hp'], hp)
        finish_operation(s)

    def test_p05_growth_caps_at_four_and_reuses_later_event(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        first = set_top(s, ['P05'])[0]
        s['sides']['a']['front'] = 'heiyu'
        s['sides']['b']['front'] = 'zero'
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertEqual(s['sides']['a']['heiyu_p05_magic_bonus'], 1)
        entity = first['entity_id']
        self.assertTrue(any(card.get('entity_id') == entity for card in s['sides']['a']['deck']))
        for expected in (2, 2):
            card = next(item for item in s['sides']['a']['deck'] if item.get('entity_id') == entity)
            s['sides']['a']['deck'] = [card] + [item for item in s['sides']['a']['deck'] if item is not card]
            s['sides']['a']['normal_attack_available'] = True
            s['sides']['a']['ap'] = 10
            s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
            self.assertEqual(s['sides']['a']['heiyu_p05_magic_bonus'], expected)
        self.assertTrue(any(card.get('entity_id') == entity for card in s['sides']['a']['deck']))

    def test_p07_doubles_after_modifiers_and_p08_penetrates_active_attacks(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        s, _ = play(s, 'P08')
        s['sides']['a']['front'] = 'heiyu'
        hero(s, 'b', 'zero').update(hp=1, shield=0)
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hp'] = 80
        s['sides']['a']['normal_attack_available'] = True
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertLess(s['sides']['b']['hp'], 80)
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        set_top(s, ['P07'])
        s['sides']['a']['front'] = 'heiyu'
        hero(s, 'b', 'zero').update(hp=20, shield=0)
        s['sides']['b']['front'] = 'zero'
        s['sides']['b']['hp'] = 80
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 20 - 6)

    def test_p03_normal_is_combat_only_and_json_has_no_sets(self):
        s = game()
        s['sides']['a']['front'] = 'heiyu'
        s['sides']['b']['front'] = 'zero'
        s, _ = play(s, 'P03')
        dumped = json.dumps(s)
        self.assertNotIn('set()', dumped)
        clone_state(json.loads(dumped))

    def test_distinct_magic_cards_add_and_double_each_real_resolution(self):
        s = game()
        hero(s, 'a', 'heiyu')['energy'] = 6
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'heiyu'})
        set_top(s, ['P04', 'P05', 'P07', 'P07'])
        s['sides']['b']['front'] = None
        before = s['sides']['b']['hp']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'heiyu'})
        self.assertEqual(s['sides']['b']['hp'], before - 24)  # (2 + team 1 + P04 1 + P05 2) * 2 * 2
        self.assertEqual(sum(c['card_id'] == 'P07' for c in s['sides']['a']['discard']), 2)

    def test_divination_attack_adds_to_other_prepared_attack_sources(self):
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        s = game()
        set_top(s, ['P01', 'Z03', 'Z04', 'Z05'])
        s, _ = play(s, 'P02')
        s = resolve_order(s, bottom='P01')
        EffectContext(s, 'a', 'heiyu', {'side': 'a'}).grant_next_attack(hero(s, 'a', 'zero'), 2)
        s['sides']['b']['front'] = None
        before = s['sides']['b']['hp']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['hp'], before - 5)
