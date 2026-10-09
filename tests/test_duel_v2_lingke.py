"""User-defined Lingke rules: real actions, response timing and public replay."""
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, new_character, effective_card
from app.modules.card_game.engine.duel_v2.flow import begin_turn, card_plays_instant
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.modifiers import resistance_values
from app.modules.card_game.engine.duel_v2.simulation import simulate_action


def game(first='a'):
    ids = ['lingke', 'zero', 'baicang', 'iloy']
    cards = [key for cid in ids for key, card in CARDS.items()
             if card['character_id'] == cid and not card.get('derived')]
    deck = validate_deck(dict(id='lingke-test', name='灵可测试', character_ids=ids, card_ids=cards))
    s = new_game(seed=39, first_side=first, skip_mulligan=True,
                 decks={'a': deck, 'b': deck}, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[], ap=20, shield=0, hp=200)
        for h in team['characters'].values():
            h.update(hp=40, max_hp=40)
    return s


def give(s, side, cid):
    card = card_instance(s, cid, side)
    s['sides'][side]['hand'].append(card)
    return dict(type='play_card', card_id=card['instance_id'])


def play(s, cid, target=None, side='a'):
    action = give(s, side, cid)
    if target: action['target_id'] = target
    return apply_action(s, side, action)


class LingkeTest(unittest.TestCase):
    def test_borrowed_owner_passive_and_weapon_trigger_after_hit_without_changing_attacker(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); hero(s, 'a', 'zero')['shape'] = 'Z07'
        calls = []
        def own_support(c, actor_id):
            if actor_id == c.cid:
                hits = [e for e in c.state['events'] if e['type'] == 'combat' and e.get('source') == 'a:lingke']
                calls.append((c.cid, len(hits), c.operation['actor']))
        with patch.object(KITS['zero'], 'on_support', staticmethod(own_support), create=True), \
             patch.object(KITS['zero'], 'weapon_hooks', {'Z07': {'on_support': own_support}}, create=True):
            s = play(s, 'K05', 'a:zero')
        self.assertEqual(calls, [('zero', 1, 'lingke'), ('zero', 1, 'lingke')])
        hit = next(e for e in s['events'] if e['type'] == 'combat' and e.get('source') == 'a:lingke')
        self.assertEqual(hit['attribute'], '光')
        self.assertEqual((hero(s, 'a', 'zero')['harmony'], hero(s, 'a', 'zero')['energy']), (0, 0))

    def test_same_attribute_uses_selected_entity_and_fixed_spirit_mark_has_no_borrowed_owner(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); s['sides']['a']['characters']['jiuyuan'] = new_character('jiuyuan')
        s['sides']['a']['order'].append('jiuyuan')
        calls = []
        def own_support(c, actor_id):
            if actor_id == c.cid: calls.append(c.cid)
        with ExitStack() as stack:
            for cid in ('lingke', 'iloy', 'jiuyuan'):
                stack.enter_context(patch.object(KITS[cid], 'on_support', staticmethod(own_support), create=True))
            s = play(s, 'K05', 'a:iloy')
            self.assertEqual(calls, ['lingke', 'iloy'])
            calls.clear(); s = play(s, 'K04', 'b:iloy'); s = play(s, 'Z03')
            self.assertEqual(calls, ['lingke'])
            calls.clear(); s = play(s, 'K05', 'a:lingke')
            self.assertEqual(calls, ['lingke'])

    def test_k06_notifies_each_borrowed_owner_once_even_for_same_attribute(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); s['sides']['a']['characters']['jiuyuan'] = new_character('jiuyuan')
        s['sides']['a']['order'].append('jiuyuan')
        s['sides']['b']['front'] = 'iloy'
        calls = []
        def own_support(c, actor_id):
            if actor_id == c.cid: calls.append(c.cid)
        with ExitStack() as stack:
            for cid in ('zero', 'baicang', 'iloy', 'jiuyuan'):
                stack.enter_context(patch.object(KITS[cid], 'on_support', staticmethod(own_support), create=True))
            s = play(s, 'K06')
        self.assertEqual(calls, ['zero', 'baicang', 'iloy', 'jiuyuan'])

    def test_three_card_use_trigger_precedes_effect_and_preserves_owner_identity(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); calls = []
        def own_support(c, actor_id):
            if actor_id == c.cid: calls.append(c.cid)
        with patch.object(KITS['zero'], 'on_support', staticmethod(own_support), create=True):
            for _ in range(2): s = play(s, 'Z07')
            seq = s['event_seq']
            s = play(s, 'Z07')
        self.assertEqual(calls, ['zero'])
        events = [e for e in s['events'] if e['seq'] > seq]
        hit_index = next(i for i, e in enumerate(events) if e['type'] == 'combat' and e.get('source') == 'a:lingke')
        shape_index = next(i for i, e in enumerate(events) if e['type'] == 'shape')
        self.assertLess(hit_index, shape_index)
        self.assertFalse(s.get('deferred_actions'))

    def test_third_battle_card_sync_precedes_frame_and_attack(self):
        s = game()
        for _ in range(2): s = play(s, 'Z07')
        s['sides']['b']['front'] = 'zero'
        seq = s['event_seq']; opening = capture_public_board(s)
        action = give(s, 'a', 'Z03')
        after = apply_action(s, 'a', action)
        events = [e for e in after['events'] if e['seq'] > seq]
        sync = next(i for i, e in enumerate(events) if e['type'] == 'attack' and e.get('actor') == 'a:lingke')
        frame = next(i for i, e in enumerate(events) if e['type'] == 'shield' and e.get('target') == 'a:zero')
        attack = next(i for i, e in enumerate(events) if e['type'] == 'attack' and e.get('actor') == 'a:zero')
        self.assertLess(sync, frame); self.assertLess(sync, attack)
        self.assertEqual(replay_public_board(opening, events), capture_public_board(after))
        self.assertEqual(capture_public_board(simulate_action(s, 'a', action)), capture_public_board(after))

    def test_use_trigger_terminal_stops_the_third_card_effect(self):
        s = game()
        for _ in range(2): s = play(s, 'Z07')
        s['sides']['b']['hp'] = 1
        seq = s['event_seq']; s = play(s, 'Z07')
        self.assertEqual(s['phase'], 'finished')
        events = [e for e in s['events'] if e['seq'] > seq]
        self.assertFalse(any(e['type'] == 'shape' for e in events))

    def test_team_observer_does_not_trigger_twice_when_it_is_borrowed_owner(self):
        s = game(); s['sides']['a']['characters']['edgar'] = new_character('edgar')
        s['sides']['a']['order'].append('edgar'); hero(s, 'a', 'edgar')['truth_keys'] = 0
        s = play(s, 'K05', 'a:edgar')
        self.assertEqual(hero(s, 'a', 'edgar')['truth_keys'], 1)
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(hero(s, 'a', 'edgar')['truth_keys'], 2)

    def test_borrowed_owner_survives_json_while_third_card_waits_for_choice(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); s['sides']['a']['characters']['edgar'] = new_character('edgar')
        s['sides']['a']['order'].append('edgar'); hero(s, 'a', 'edgar')['truth_keys'] = 0
        s = play(s, 'E04'); s = play(s, 'E04'); give(s, 'a', 'Z07')
        s = play(s, 'E03')
        self.assertEqual(s['phase'], 'choice')
        self.assertFalse(s.get('deferred_actions'))
        self.assertEqual(hero(s, 'a', 'edgar')['truth_keys'], 1)
        hits = [e for e in s['events'] if e['type'] == 'combat' and e.get('source') == 'a:lingke']
        self.assertEqual(len(hits), 1)
        observed = []; original = KITS['edgar'].on_support
        def capture(c, actor_id):
            observed.append(actor_id); original(c, actor_id)
        with patch.object(KITS['edgar'], 'on_support', staticmethod(capture)):
            s = apply_action(json.loads(json.dumps(s)), 'a', dict(type='choose_cards', card_ids=[]))
        self.assertEqual(observed, [])
        self.assertEqual(hero(s, 'a', 'edgar')['truth_keys'], 1)

    def test_stale_or_downed_borrowed_entity_does_not_receive_own_trigger(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        from app.modules.card_game.content.duel_v2.characters.lingke import _sync
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        for mode in ('stale', 'down'):
            s = game(); owner = hero(s, 'a', 'zero'); calls = []
            identity = dict(character_id='zero', entity_id=owner.entity_id)
            if mode == 'stale':
                s['sides']['a']['characters']['zero'] = new_character('zero')
            else:
                owner.update(hp=0, down_turns=3)
            def own_support(c, actor_id):
                if actor_id == c.cid: calls.append(c.cid)
            with patch.object(KITS['zero'], 'on_support', staticmethod(own_support), create=True):
                _sync(EffectContext(s, 'a', 'lingke', {'side': 'a'}), '光', support_actor=identity)
            self.assertEqual(calls, [], mode)
            self.assertTrue(any(e['type'] == 'combat' and e.get('source') == 'a:lingke' for e in s['events']))

    def test_catalog_three_weapons_no_battle_and_mechanism(self):
        cards = [c for c in CARDS.values() if c['character_id'] == 'lingke' and not c.get('derived')]
        self.assertEqual(len(cards), 8)
        self.assertEqual([c['id'] for c in cards if c['type'] == 'form'], ['K03', 'K07', 'K08'])
        self.assertFalse(any(c['type'] == 'battle' for c in cards))
        self.assertEqual(CARDS['K06']['cost'], 1)
        self.assertEqual(CARDS['K06']['additional_ap_cost'], 1)
        self.assertEqual((CHARACTERS['lingke']['attack'], CHARACTERS['lingke']['max_hp']), (3, 4))
        self.assertEqual(CHARACTERS['lingke']['mechanisms'][0]['name'], '同频合击')

    def test_sync_is_support_accumulates_collapse_and_notifies_allies_without_resources(self):
        s = game(); s['sides']['b']['front'] = 'iloy'
        s['sides']['a']['characters']['xun'] = new_character('xun')
        s['sides']['a']['order'].append('xun')
        hero(s, 'a', 'xun').update(arid=0, arid_contributors=[])
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(hero(s, 'b', 'iloy')['flags']['collapse_count'], 1)
        self.assertEqual(hero(s, 'a', 'xun')['arid'], 1)
        self.assertEqual((hero(s, 'a', 'lingke')['energy'], hero(s, 'a', 'lingke')['harmony']), (0, 0))
        self.assertIsNone(s['sides']['a']['front'])
        for _ in range(4): s = play(s, 'K05', 'a:zero')
        self.assertTrue(hero(s, 'b', 'iloy')['flags'].get('collapse'))

    def test_selected_attribute_player_resistance_refreshes_without_stacking_and_expires(self):
        s = game(); before = deepcopy(s)
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(s['sides']['b']['hp'], 196)
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -1})
        self.assertIsNone(s['sides']['a']['front'])
        for cid, h in s['sides']['a']['characters'].items():
            self.assertEqual((h['harmony'], h['energy']),
                             (before['sides']['a']['characters'][cid]['harmony'], before['sides']['a']['characters'][cid]['energy']))
        hits = [e for e in s['events'] if e['type'] == 'combat' and e.get('source') == 'a:lingke']
        self.assertEqual((hits[-1]['attribute'], hits[-1]['additional_attack']), ('光', True))
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(s['sides']['b']['hp'], 192)
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -1})
        for board in (observe(s, 'a', include_previews=False), observe(s, 'b', include_previews=False), capture_public_board(s)):
            marks = board['sides']['b']['effect_markers']
            self.assertEqual(len(marks), 1)
            self.assertTrue(all(m['kind'] == 'debuff' and '下次受到' not in m['timing'] for m in marks))
        s = json.loads(json.dumps(s)); begin_turn(s, 'b')
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -1})
        begin_turn(s, 'a')
        self.assertEqual(resistance_values(s['sides']['b']), {})

    def test_different_attribute_resistance_is_independent_but_copied_sources_do_not_stack(self):
        s = game()
        s = play(s, 'K05', 'a:zero')
        s = play(s, 'K05', 'a:iloy')
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -1, '灵': -1})
        s['sides']['a']['characters']['anhunqu'] = new_character('anhunqu')
        s['sides']['a']['order'].append('anhunqu')
        hero(s, 'a', 'anhunqu')['flags']['mimic'] = {'id': 'lingke'}
        s = play(s, 'K05', 'a:zero')
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -1, '灵': -1})

    def test_opponent_turn_sync_uses_shared_decay_at_next_lingke_turn_start(self):
        s = game('b'); s['sides']['a']['front'] = 'baicang'
        actor = hero(s, 'a', 'baicang'); actor['hp'] = 1
        hero(s, 'a', 'lingke')['lingke_chain'] = dict(entity_id=actor.entity_id,
                                                    character_id='baicang', count=2)
        give(s, 'a', 'B05')
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        target = hero(s, 'b', 'zero')
        self.assertEqual(resistance_values(target), {'咒': -1})
        effect = next(e for e in target['effects'] if e['definition_id'] == 'lingke.resistance')
        self.assertEqual(effect['expiry'], dict(side='a', turn=s['sides']['a']['turn_count'] + 1, phase='turn_start'))
        for view in (observe(s, 'a', include_previews=False), observe(s, 'b', include_previews=False), capture_public_board(s)):
            shown = next(h for h in view['sides']['b']['characters'] if h['id'] == 'zero')
            self.assertTrue(any(m['timing'] == '持续到灵可下个己方回合开始时' for m in shown['effect_markers']))
        s = apply_action(json.loads(json.dumps(s)), 'b', dict(type='end_turn'))
        self.assertEqual(s['active_side'], 'a')
        self.assertEqual(resistance_values(hero(s, 'b', 'zero')), {})

    def test_weave_follows_attack_attribute_and_extra_damage_weapon(self):
        for weapon, expected in ((None, [4, 4]), ('K08', [5, 5])):
            s = game(); hero(s, 'a', 'lingke')['shape'] = weapon
            s['sides']['b']['front_debuff']['weave'] = {'by': 'a', 'expires_side': 'a'}
            seq = s['event_seq']; s = play(s, 'K05', 'a:lingke')
            hits = [e for e in s['events'] if e['seq'] > seq and e['type'] in ('combat', 'overlay') and e.get('source') == 'a:lingke']
            self.assertEqual([e['amount'] for e in hits], expected)
            self.assertTrue(all(e['attribute'] == '灵' and e['additional_attack'] for e in hits))

    def test_passive_cross_turn_chain_ignores_lingke_and_resets_different_owner(self):
        s = game(); s = play(s, 'Z07'); s = play(s, 'K02'); s = play(s, 'Z07')
        self.assertEqual(hero(s, 'a', 'lingke')['lingke_chain']['count'], 2)
        begin_turn(s, 'b'); begin_turn(s, 'a'); s['sides']['a']['ap'] = 20
        seq = s['event_seq']; s = play(s, 'Z07')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'combat' and e.get('source') == 'a:lingke']
        self.assertEqual(len(hits), 1); self.assertEqual(hits[0]['attribute'], '光')
        self.assertEqual(hero(s, 'a', 'lingke')['lingke_chain']['count'], 0)
        s = play(s, 'Z07'); s = play(s, 'B07')
        chain = hero(s, 'a', 'lingke')['lingke_chain']
        self.assertEqual((chain['character_id'], chain['count']), ('baicang', 1))

    def test_responses_count_and_trigger_on_opponent_turn(self):
        s = game('b'); s['sides']['a']['front'] = 'baicang'
        hero(s, 'a', 'lingke')['lingke_chain'] = dict(entity_id=hero(s, 'a', 'baicang').entity_id, character_id='baicang', count=2)
        # B05 responds to lethal damage; use a real response invocation.
        give(s, 'a', 'B05')
        hero(s, 'a', 'baicang')['hp'] = 1
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        self.assertEqual(hero(s, 'a', 'lingke')['lingke_chain']['count'], 0)
        self.assertTrue(any(e['type'] == 'combat' and e.get('source') == 'a:lingke' for e in s['events']))
        self.assertEqual(s['active_side'], 'b')

    def test_next_instant_consumed_by_zero_cost_without_spending_instant(self):
        s = game(); s = play(s, 'K02')
        s['sides']['a']['characters']['yi'] = new_character('yi')
        s['sides']['a']['order'].append('yi')
        free = card_instance(s, 'I04', 'a')
        self.assertTrue(card_plays_instant(s, 'a', free))
        s['sides']['a']['hand'].append(free)
        s = apply_action(s, 'a', dict(type='play_card', card_id=free['instance_id']))
        self.assertFalse(s['sides']['a']['used'].get('instant', False))
        self.assertFalse(card_plays_instant(s, 'a', card_instance(s, 'Z03', 'a')))

    def test_k03_substitution_preserves_physical_card_and_instant_opportunity(self):
        self.assertNotIn('KF01', CARDS)
        self.assertIn('瞬发。获得 1 点行动力。', CARDS['K03']['description'])
        for physical_id in ('K01', 'K03', 'K06'):
            with self.subTest(physical_id=physical_id):
                s = game(); s = play(s, 'K03'); s['sides']['a']['ap'] = 0
                action = give(s, 'a', physical_id); physical = s['sides']['a']['hand'][-1]
                card = effective_card(s, 'a', physical)
                self.assertEqual((card['card_id'], card['type'], card['cost'], card['instant']),
                                 ('K03', 'tactic', 1, True))
                self.assertEqual(card['entity_id'], physical['entity_id'])
                self.assertEqual(card['name'], physical['name'])
                actions = [a for a in legal_actions(s, 'a')
                           if a['type'] == 'play_card' and a['card_id'] == physical['instance_id']]
                self.assertEqual(actions, [action])  # no original K01 target requirement
                before_hp = hero(s, 'a', 'lingke')['hp'] = 2
                s = apply_action(json.loads(json.dumps(s)), 'a', action)
                self.assertEqual(s['phase'], 'playing')
                self.assertEqual(s['sides']['a']['ap'], 1)
                self.assertTrue(s['sides']['a']['used']['instant'])
                self.assertEqual(s['sides']['a']['discard'][-1]['card_id'], physical_id)
                self.assertEqual(hero(s, 'a', 'lingke')['hp'], before_hp)  # no re-equip/heal
                self.assertTrue(any('「星间漫游」：获得 1 点行动力。' in e.get('text', '')
                                    for e in s['events']))
                s['sides']['a']['ap'] = 0
                self.assertEqual(effective_card(s, 'a', card_instance(s, 'K01', 'a'))['card_id'], 'K01')

    def test_k01_random_three_distinct_then_one_private_and_json_restorable(self):
        s = game(); ids = [c['entity_id'] for c in s['sides']['a']['deck']]
        s = play(s, 'K01', 'a:zero')
        pending = s['pending_choice']; self.assertEqual(pending['kind'], 'kit')
        self.assertEqual(len(pending['cards']), 3)
        self.assertNotIn('max_select', pending)
        self.assertFalse(any(a['type'] == 'choose_cards' for a in legal_actions(s, 'a')))
        self.assertEqual(observe(s, 'b', include_previews=False)['pending_choice']['choices'], [])
        self.assertEqual(len({c['card_id'] for c in pending['cards']}), len(pending['cards']))
        count = observe(s, 'a', include_previews=False)['sides']['a']['deck_count']
        self.assertEqual(count, len(ids))
        chosen = s['pending_choice']['cards'][0]['instance_id']
        s = apply_action(json.loads(json.dumps(s)), 'a', dict(type='choose', choice_id=chosen))
        self.assertEqual(s['phase'], 'playing')
        self.assertTrue(any(c['instance_id'] == chosen for c in s['sides']['a']['hand']))
        self.assertEqual(len(s['sides']['a']['deck']), len(ids)-1)
        all_ids = [c['entity_id'] for c in s['sides']['a']['deck'] + s['sides']['a']['hand']]
        self.assertEqual(set(all_ids), set(ids)); self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_k01_random_candidates_are_saved_and_not_fixed_by_deck_order(self):
        base = game()
        results = []
        for seed in range(6):
            s = deepcopy(base)
            s['rng'] = seed
            s = play(s, 'K01', 'a:zero')
            cards = s['pending_choice']['cards']
            self.assertEqual(len(cards), 3)
            self.assertEqual(len({c['card_id'] for c in cards}), 3)
            self.assertTrue(all(c['character_id'] == 'zero' for c in cards))
            results.append(tuple(c['card_id'] for c in cards))
            restored = json.loads(json.dumps(s))
            self.assertEqual(observe(s, 'a', include_previews=False)['pending_choice'],
                             observe(restored, 'a', include_previews=False)['pending_choice'])
        self.assertGreater(len(set(results)), 1)
        same = deepcopy(base)
        same['rng'] = 0
        self.assertEqual(tuple(c['card_id'] for c in play(same, 'K01', 'a:zero')['pending_choice']['cards']),
                         results[0])

    def test_k01_short_or_empty_owned_pool(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                s = game()
                team = s['sides']['a']
                team['deck'] = [card_instance(s, cid, 'a') for cid in ('Z01', 'Z01', 'Z02')[:count + 1]] if count else []
                before = len(team['deck'])
                s = play(s, 'K01', 'a:zero')
                if count == 0:
                    self.assertEqual(s['phase'], 'playing')
                    continue
                self.assertEqual(len(s['pending_choice']['cards']), count)
                chosen = s['pending_choice']['cards'][0]['instance_id']
                s = apply_action(s, 'a', dict(type='choose', choice_id=chosen))
                self.assertEqual(len(s['sides']['a']['deck']), before - 1)

    def test_mark_attacks_selected_bench_before_battle_and_grants_energy(self):
        s = game(); s = play(s, 'K04', 'b:iloy'); energy = hero(s, 'a', 'lingke')['energy']
        seq = s['event_seq']; s = play(s, 'Z03')
        attacks = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'attack']
        self.assertEqual((attacks[0]['actor'], attacks[0]['target']), ('a:lingke', 'b:iloy'))
        self.assertEqual(hero(s, 'b', 'iloy')['hp'], 36)
        self.assertEqual(hero(s, 'a', 'lingke')['energy'], energy+2)  # mark + team share
        self.assertFalse(any(e['definition_id'] == 'lingke.taunt' for e in hero(s, 'b', 'iloy').get('effects', [])))

    def test_k06_pays_two_and_uses_ally_attributes_then_own_spirit(self):
        s = game(); s['sides']['b']['front'] = 'iloy'
        ap = s['sides']['a']['ap']; seq=s['event_seq']; s=play(s,'K06')
        hits=[e for e in s['events'] if e['seq']>seq and e['type']=='combat' and e.get('source')=='a:lingke']
        self.assertEqual([e['attribute'] for e in hits], ['光','咒','灵','灵'])
        self.assertEqual(s['sides']['a']['ap'], ap-2)
        s = game(); s['sides']['a']['ap']=1; action=give(s,'a','K06')
        self.assertNotIn(action, legal_actions(s,'a'))

    def test_k07_curse_applies_weave_and_light_triggers_genesis_without_entry(self):
        s=game();s=play(s,'K07');seq=s['event_seq'];s=play(s,'K05','a:baicang')
        self.assertTrue(s['sides']['b']['front_debuff'].get('weave'))
        self.assertTrue(any(e['type']=='overlay' and e.get('attribute')=='咒' for e in s['events'] if e['seq']>seq))
        seq=s['event_seq'];s=play(s,'K05','a:zero')
        self.assertTrue(any(e['type']=='genesis' for e in s['events'] if e['seq']>seq))
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s,'a','lingke')['harmony'],0)

    def test_k06_skips_empty_front_without_restricting_later_sync(self):
        s = game(); hp = s['sides']['b']['hp']; seq = s['event_seq']
        s = play(s, 'K06')
        self.assertEqual(s['sides']['b']['hp'], hp)
        self.assertFalse(any(e['type'] == 'attack' for e in s['events'] if e['seq'] > seq))
        self.assertEqual(resistance_values(s['sides']['b']), {})
        s = play(s, 'K05', 'a:zero')
        self.assertLess(s['sides']['b']['hp'], hp)

    def test_k06_remaining_sync_skips_after_front_is_defeated_and_replays_match(self):
        s = game(); s['sides']['b']['front'] = 'iloy'
        hero(s, 'b', 'iloy')['hp'] = 1
        action = give(s, 'a', 'K06'); hp = s['sides']['b']['hp']
        opening = capture_public_board(s); seq = s['event_seq']
        after = apply_action(json.loads(json.dumps(s)), 'a', action)
        hits = [e for e in after['events'] if e['seq'] > seq and e['type'] == 'attack']
        self.assertEqual([e['target'] for e in hits], ['b:iloy'])
        self.assertEqual(after['sides']['b']['hp'], hp)
        self.assertEqual((hero(after, 'a', 'lingke')['harmony'], hero(after, 'a', 'lingke')['energy']), (0, 0))
        self.assertEqual(capture_public_board(simulate_action(s, 'a', action)), capture_public_board(after))
        self.assertEqual(replay_public_board(opening, [e for e in after['events'] if e['seq'] > seq]),
                         capture_public_board(after))

    def test_k06_does_not_retarget_player_when_front_returns_in_response(self):
        s = game(); s['sides']['b']['front'] = 'zero'
        s['sides']['b']['characters']['xun'] = new_character('xun')
        s['sides']['b']['order'].append('xun')
        give(s, 'b', 'X05')
        hp = s['sides']['b']['hp']; seq = s['event_seq']
        s = play(s, 'K06')
        self.assertIsNone(s['sides']['b']['front'])
        self.assertEqual(s['sides']['b']['hp'], hp)
        self.assertFalse(any(e['type'] == 'attack' for e in s['events'] if e['seq'] > seq))

    def test_k06_does_not_attack_player_when_k07_genesis_defeats_front(self):
        s = game(); hero(s, 'a', 'lingke')['shape'] = 'K07'
        s['sides']['b']['front'] = 'iloy'; hero(s, 'b', 'iloy')['hp'] = 1
        hp = s['sides']['b']['hp']; seq = s['event_seq']
        s = play(s, 'K06')
        self.assertEqual(s['sides']['b']['hp'], hp)
        self.assertFalse(any(e['type'] == 'attack' for e in s['events'] if e['seq'] > seq))

    def test_k06_can_attack_front_summon(self):
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        s = game(); front = summon_front(s, 'b', name='测试召唤物', attack=0, hp=40)
        seq = s['event_seq']; s = play(s, 'K06')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'attack']
        self.assertEqual([e['target'] for e in hits], [f'b:{front}'] * 4)

    def test_k06_skips_downed_allies_and_finishes_with_own_spirit_without_borrowing_self(self):
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = game(); s['sides']['b']['front'] = 'iloy'
        hero(s, 'a', 'baicang').update(hp=0, down_turns=3)
        calls = []
        def own_support(c, actor_id):
            if actor_id == c.cid: calls.append(c.cid)
        with ExitStack() as stack:
            for cid in ('lingke', 'zero', 'baicang', 'iloy'):
                stack.enter_context(patch.object(KITS[cid], 'on_support', staticmethod(own_support), create=True))
            seq = s['event_seq']; s = play(s, 'K06')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'combat' and e.get('source') == 'a:lingke']
        self.assertEqual([e['attribute'] for e in hits], ['光', '灵', '灵'])
        self.assertEqual(calls, ['lingke', 'zero', 'lingke', 'iloy', 'lingke'])

    def test_k06_with_all_other_allies_downed_still_uses_own_spirit(self):
        s = game(); s['sides']['b']['front'] = 'iloy'
        for cid in ('zero', 'baicang', 'iloy'):
            hero(s, 'a', cid).update(hp=0, down_turns=3)
        seq = s['event_seq']; s = play(s, 'K06')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'combat' and e.get('source') == 'a:lingke']
        self.assertEqual([e['attribute'] for e in hits], ['灵'])

    def test_ultimate_support_reduces_resistance_and_keeps_normal_resources(self):
        s=game();hero(s,'a','lingke')['energy']=5
        s=apply_action(s,'a',dict(type='ultimate',character_id='lingke'))
        s['sides']['a']['front']='iloy';hero(s,'a','iloy')['harmony']=2
        s['sides']['b']['front']='iloy';seq=s['event_seq']
        s=apply_action(s,'a',dict(type='attack',character_id='zero'))
        h=hero(s,'a','zero')
        self.assertEqual((h['harmony'],h['energy']),(1,2))
        self.assertEqual(resistance_values(hero(s,'b','iloy')).get('光'),-1)
        hit=next(e for e in s['events'] if e['seq']>seq and e['type']=='combat' and e.get('source')=='a:zero')
        self.assertTrue(hit['additional_attack'])
        self.assertEqual(hero(s,'a','lingke')['energy'],1)
        s = apply_action(s, 'a', dict(type='end_turn'))
        s = apply_action(s, 'b', dict(type='end_turn'))
        self.assertEqual(resistance_values(hero(s, 'b', 'iloy')), {})
        self.assertTrue(hero(s, 'a', 'lingke')['awakened'], '减抗衰减不结束仍在持续的共鸣领域')

    def test_simulation_same_rules_and_input_unchanged(self):
        s=game();action=give(s,'a','K06');before=deepcopy(s)
        slow=apply_action(s,'a',action);fast=simulate_action(s,'a',action)
        self.assertEqual(s,before)
        for state in (slow,fast):
            state.pop('_public_board',None)
            for e in state['events']:e.pop('patch',None)
        self.assertEqual(slow,fast)

    def test_player_resistance_and_chain_replay_patches_match_final_board(self):
        s = game(); opening = capture_public_board(s); s['_public_board'] = deepcopy(opening)
        seq = s['event_seq']; s = play(s, 'K05', 'a:zero'); s = play(s, 'Z07')
        replay = replay_public_board(opening, [e for e in s['events'] if e['seq'] > seq])
        expected = capture_public_board(s)
        self.assertEqual(replay['sides']['b']['effect_markers'], expected['sides']['b']['effect_markers'])
        actual_h = next(h for h in replay['sides']['a']['characters'] if h['id'] == 'lingke')
        expected_h = next(h for h in expected['sides']['a']['characters'] if h['id'] == 'lingke')
        self.assertEqual(actual_h['lingke_chain'], expected_h['lingke_chain'])

if __name__ == '__main__': unittest.main()
