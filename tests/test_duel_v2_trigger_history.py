"""Triggered effects must explain public changes at their actual resolution."""
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, legal_actions
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import hero, attack_value, new_character, card_instance


class TriggerHistoryTest(unittest.TestCase):
    def game(self):
        deck = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
        s = new_game(seed=9, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        for side in ('a', 'b'):
            s['sides'][side]['hand'] = []
        s['sides']['a']['front'] = 'zero'
        hero(s, 'a', 'zero')['harmony'] = 2
        s['sides']['b']['front'] = 'baicang'
        return s

    def test_genesis_buff_is_ordered_public_and_replayable(self):
        s = self.game()
        opening = capture_public_board(s)
        s['_public_board'] = opening
        start = len(s['events'])
        result = apply_action(s, 'a', {'type': 'attack', 'character_id': 'iloy'})
        events = result['events'][start:]
        triggers = [e for e in events if '白藏受到伤害，触发异能' in e['text']]
        self.assertEqual(len(triggers), 2)  # Genesis, then the actual attack.
        trigger = triggers[0]
        self.assertEqual(trigger['before'], {'attack': 2})
        self.assertEqual(trigger['after'], {'attack': 3})
        self.assertEqual((trigger['actor'], trigger['target']), ('b:baicang', 'b:baicang'))
        self.assertFalse(trigger['present'])  # History, without an extra animation pause.
        self.assertLess(next(e['seq'] for e in events if e['type'] == 'genesis'), trigger['seq'])
        self.assertLess(trigger['seq'], next(e['seq'] for e in events if e['type'] == 'attack'))
        board = replay_public_board(opening, [e for e in events if e['seq'] <= trigger['seq']])
        bc = next(h for h in board['sides']['b']['characters'] if h['id'] == 'baicang')
        self.assertEqual((bc['hp'], bc['attack']), (5, 3))
        for viewer in ('a', 'b'):
            view = observe(result, viewer)
            self.assertTrue(any(e['seq'] == trigger['seq'] and '攻击 +1' in e['text'] for e in view['events']))
        self.assertEqual(replay_public_board(opening, events), capture_public_board(result))
        self.assertEqual(attack_value(hero(s, 'b', 'baicang')), 2)  # Input snapshot is immutable.

    def test_support_counter_immunity_is_explicit_in_damage_slot(self):
        s = self.game()
        result = apply_action(s, 'a', {'type': 'attack', 'character_id': 'iloy'})
        for viewer in ('a', 'b'):
            events = observe(result, viewer)['presentation']['events']
            immune = next(e for e in events if e.get('counter_immunity') == 'support')
            attack = next(e for e in events if e['type'] == 'attack')
            self.assertEqual(immune['type'], 'combat')
            self.assertEqual(immune['target'], 'a:iloy')
            self.assertEqual(immune['group_id'], attack['group_id'])
            self.assertEqual(immune['amount'], 0)
            self.assertEqual(immune['before'], immune['after'])
            self.assertIn('免疫反击', immune['text'])
        for defender, harmony in (('baicang', 0), (None, 2)):
            s = self.game()
            s['sides']['b']['front'] = defender
            hero(s, 'a', 'zero')['harmony'] = harmony
            if defender:
                hero(s, 'b', defender)['base_attack'] = 0
            result = apply_action(s, 'a', {'type': 'attack', 'character_id': 'iloy'})
            self.assertFalse(any(e.get('counter_immunity') for e in result['events']))

    def test_absorbed_and_lethal_genesis_do_not_claim_buff(self):
        for hp, shield in ((6, 2), (1, 0)):
            with self.subTest(hp=hp, shield=shield):
                s = self.game()
                hero(s, 'b', 'baicang').update(hp=hp, shield=shield)
                result = apply_action(s, 'a', {'type': 'attack', 'character_id': 'iloy'})
                self.assertFalse(any('白藏受到伤害，触发异能' in e['text'] for e in result['events']))

    def context(self, s, cid):
        if cid not in s['sides']['a']['characters']:
            s['sides']['a']['characters'][cid] = new_character(cid)
            s['sides']['a']['order'].append(cid)
        return EffectContext(s, 'a', cid, {'side': 'a', 'actor': cid})

    def test_delayed_attack_and_action_point_triggers(self):
        s = self.game()
        hero(s, 'a', 'bohe')['flags']['pending_atk'] = 4
        c = self.context(s, 'bohe')
        fire(c, 'on_turn_begin', actor_only=True)
        event = s['events'][-1]
        self.assertIn('「第一直觉」触发', event['text'])
        self.assertEqual(event['after']['attack'] - event['before']['attack'], 4)
        seq = s['event_seq']
        fire(c, 'on_turn_begin', actor_only=True)
        self.assertEqual(s['event_seq'], seq)
        hero(s, 'a', 'zero')['shape'] = 'Z06'
        fire(self.context(s, 'zero'), 'on_turn_end', actor_only=True)
        self.assertIn('获得 1 点行动力', s['events'][-1]['text'])
        self.assertNotIn('下个己方回合', s['events'][-1]['text'])
        self.assertEqual(s['sides']['a']['ap'], 2)
        self.assertEqual(s['sides']['a'].get('extra_ap', 0), 0)

    def test_healing_attack_is_distinct_from_heal_result(self):
        s = self.game()
        hero(s, 'a', 'baicang')['hp'] = 2
        card = card_instance(s, 'Y03', 'a')
        s['sides']['a']['hand'] = [card]
        result = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id'],
                                       'target_id': 'a:baicang'})
        self.assertEqual(sum(e['type'] == 'heal' for e in result['events']), 1)
        buffs = [e for e in result['events'] if '伊洛伊的治疗效果' in e['text']]
        self.assertEqual(len(buffs), 1)
        self.assertEqual(buffs[0]['actor'], 'a:iloy')
        self.assertEqual(buffs[0]['target'], 'a:baicang')
        self.assertEqual(buffs[0]['after'], {'attack': 3})

    def test_marks_and_gaze_only_log_actual_increase(self):
        s = self.game()
        c = self.context(s, 'zhenhong')
        c.add_gaze(2)
        self.assertIn('获得 2 层凝视', s['events'][-1]['text'])
        seq = s['event_seq']
        self.assertEqual(c.add_gaze(1), 0)
        self.assertEqual(s['event_seq'], seq)

    def test_pact_energy_and_delayed_genesis_are_explained(self):
        s = self.game()
        c = self.context(s, 'jiuyuan')
        c.character['shape'] = 'J07'
        hero(s, 'b', 'baicang')['flags']['pact'] = True
        fire(c, 'on_ultimate', actor_only=True)
        gains = [e for e in s['events'] if '「使命必达」触发' in e['text']]
        self.assertEqual(len(gains), 1)
        self.assertEqual(gains[0]['after'], {'energy': 1})
        c = self.context(s, 'iloy')
        c.character['shape'] = 'Y08'
        fire(c, 'after_genesis', actor_only=True, entering='zero')
        self.assertIn('下个己方回合开始时追加一次创生', s['events'][-1]['text'])
        seq = s['event_seq']
        fire(c, 'after_genesis', actor_only=True, entering='zero')
        self.assertEqual(s['event_seq'], seq)

    def test_no_hidden_card_or_preview_side_effects(self):
        s = self.game()
        c = self.context(s, 'baicang')
        c.operation['card'] = {'name': 'SECRET-HIDDEN-RESPONSE'}
        c.card = c.operation['card']
        fire(c, 'on_damage_taken', actor_only=True, amount=1, source='b:zero')
        self.assertNotIn('SECRET-HIDDEN-RESPONSE', str(s['events']))
        before = deepcopy(s)
        for _ in range(2):
            observe(s, 'a')
            legal_actions(s)
        self.assertEqual(s, before)


if __name__ == '__main__':
    unittest.main()
