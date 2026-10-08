"""User-defined Lingke rules: real actions, response timing and public replay."""
import json
import unittest
from copy import deepcopy
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
    def test_catalog_three_weapons_no_battle_and_mechanism(self):
        cards = [c for c in CARDS.values() if c['character_id'] == 'lingke' and not c.get('derived')]
        self.assertEqual(len(cards), 8)
        self.assertEqual([c['id'] for c in cards if c['type'] == 'form'], ['K03', 'K07', 'K08'])
        self.assertFalse(any(c['type'] == 'battle' for c in cards))
        self.assertEqual(CARDS['K06']['cost'], 3)
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

    def test_selected_attribute_player_resistance_stacks_and_expires_both_views(self):
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
        self.assertEqual(s['sides']['b']['hp'], 191)
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -2})
        for board in (observe(s, 'a', include_previews=False), observe(s, 'b', include_previews=False), capture_public_board(s)):
            marks = board['sides']['b']['effect_markers']
            self.assertEqual(len(marks), 2)
            self.assertTrue(all(m['kind'] == 'debuff' and '下次受到' not in m['timing'] for m in marks))
        s = json.loads(json.dumps(s)); begin_turn(s, 'b')
        self.assertEqual(resistance_values(s['sides']['b']), {'光': -2})
        begin_turn(s, 'a')
        self.assertEqual(resistance_values(s['sides']['b']), {})

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
        s = game(); s = play(s, 'K03'); s['sides']['a']['ap'] = 0
        action = give(s, 'a', 'K06'); physical = s['sides']['a']['hand'][-1]
        card = effective_card(s, 'a', physical)
        self.assertEqual((card['card_id'], card['cost'], card['instant']), ('KF01', 1, True))
        s = apply_action(s, 'a', action)
        self.assertEqual(s['sides']['a']['ap'], 1)
        self.assertTrue(s['sides']['a']['used']['instant'])
        self.assertEqual(s['sides']['a']['discard'][-1]['card_id'], 'K06')
        s['sides']['a']['ap'] = 0
        self.assertEqual(effective_card(s, 'a', card_instance(s, 'K01', 'a'))['card_id'], 'K01')

    def test_k01_three_distinct_then_one_private_and_json_restorable(self):
        s = game(); ids = [c['entity_id'] for c in s['sides']['a']['deck']]
        s = play(s, 'K01', 'a:zero')
        pending = s['pending_choice']; self.assertEqual(pending['kind'], 'kit_multi')
        self.assertEqual(pending['min_select'], 3)
        self.assertEqual(observe(s, 'b', include_previews=False)['pending_choice']['choices'], [])
        self.assertEqual(len({c['card_id'] for c in pending['cards']}), len(pending['cards']))
        action = next(a for a in legal_actions(s, 'a') if a['type'] == 'choose_cards')
        s = apply_action(json.loads(json.dumps(s)), 'a', action)
        count = observe(s, 'a', include_previews=False)['sides']['a']['deck_count']
        self.assertEqual(count, len(ids))
        chosen = s['pending_choice']['cards'][0]['instance_id']
        s = apply_action(json.loads(json.dumps(s)), 'a', dict(type='choose', choice_id=chosen))
        self.assertEqual(s['phase'], 'playing')
        self.assertTrue(any(c['instance_id'] == chosen for c in s['sides']['a']['hand']))
        self.assertEqual(len(s['sides']['a']['deck']), len(ids)-1)
        all_ids = [c['entity_id'] for c in s['sides']['a']['deck'] + s['sides']['a']['hand']]
        self.assertEqual(set(all_ids), set(ids)); self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_mark_attacks_selected_bench_before_battle_and_grants_energy(self):
        s = game(); s = play(s, 'K04', 'b:iloy'); energy = hero(s, 'a', 'lingke')['energy']
        seq = s['event_seq']; s = play(s, 'Z03')
        attacks = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'attack']
        self.assertEqual((attacks[0]['actor'], attacks[0]['target']), ('a:lingke', 'b:iloy'))
        self.assertEqual(hero(s, 'b', 'iloy')['hp'], 36)
        self.assertEqual(hero(s, 'a', 'lingke')['energy'], energy+2)  # mark + team share
        self.assertFalse(any(e['definition_id'] == 'lingke.taunt' for e in hero(s, 'b', 'iloy').get('effects', [])))

    def test_k06_pays_three_and_uses_all_three_ally_attributes(self):
        s = game(); ap = s['sides']['a']['ap']; seq=s['event_seq']; s=play(s,'K06')
        hits=[e for e in s['events'] if e['seq']>seq and e['type']=='combat' and e.get('source')=='a:lingke']
        self.assertEqual([e['attribute'] for e in hits], ['光','咒','灵'])
        self.assertEqual(s['sides']['a']['ap'], ap-3)
        s = game(); s['sides']['a']['ap']=2; action=give(s,'a','K06')
        self.assertNotIn(action, legal_actions(s,'a'))

    def test_k07_curse_applies_weave_and_light_triggers_genesis_without_entry(self):
        s=game();s=play(s,'K07');seq=s['event_seq'];s=play(s,'K05','a:baicang')
        self.assertTrue(s['sides']['b']['front_debuff'].get('weave'))
        self.assertTrue(any(e['type']=='overlay' and e.get('attribute')=='咒' for e in s['events'] if e['seq']>seq))
        seq=s['event_seq'];s=play(s,'K05','a:zero')
        self.assertTrue(any(e['type']=='genesis' for e in s['events'] if e['seq']>seq))
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(hero(s,'a','lingke')['harmony'],0)

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
