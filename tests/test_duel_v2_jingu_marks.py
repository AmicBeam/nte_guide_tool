"""User 2026-09-18: persistent gold, visible hand marks and once-per-character ultimates."""
import json
import unittest
from copy import deepcopy
from random import Random

from tests import test_duel_v2_xiaozhi_haiyue as fixtures
from app.modules.card_game.engine.duel_v2.flow import apply_action, begin_turn, end_ultimate, legal_actions
from app.modules.card_game.engine.duel_v2.state import add_hand, card_instance, damage, hero
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2.replay_log import build_viewer_log, assemble_replay_game, leak_markers


class JinguRulesTest(unittest.TestCase):
    game = fixtures.NewTestCharacters.game

    def hand(self, s, *ids, side='a'):
        cards = [card_instance(s, cid, side) for cid in ids]
        for card in cards:
            add_hand(s, side, card)
        return cards

    def ult(self, s, side='a', cid='xiaozhi'):
        h = hero(s, side, cid); h['energy'] = 6 if h['attribute'] in ('魂', '咒', '暗') else 5
        return apply_action(s, side, {'type':'ultimate', 'character_id':cid})

    def play(self, s, card, side='a', option_id='base'):
        action = {'type':'play_card', 'card_id':card['instance_id']}
        if card.get('play_options'): action['option_id'] = option_id
        return apply_action(s, side, action)

    def test_all_hand_marked_new_draws_marked_and_other_cards_do_not_reroll(self):
        s = self.game(); self.hand(s, 'N01', 'Q01', 'U07')
        s = self.ult(s)
        cards = s['sides']['a']['hand']
        self.assertTrue(all(c['jingu_mark']['delta'] in (-1,0,1) for c in cards))
        expected = deepcopy(cards[1]['jingu_mark'])
        mark = cards[0]['jingu_mark']['delta']; before = hero(s,'a','xiaozhi')['jingu']
        s = self.play(s,cards[0])
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],before+mark)
        self.assertEqual(s['sides']['a']['hand'][0]['jingu_mark'],expected)
        self.assertNotIn('jingu_mark',s['sides']['a']['discard'][-1])
        added=self.hand(s,'Z03')[0]
        self.assertIn('jingu_mark',added)

    def test_turn_begin_rerolls_with_persisted_rng_and_end_removes_marks(self):
        s=self.game();self.hand(s,'Q01','N01');s=self.ult(s)
        expected=[];rng_state=s['rng']
        for _ in s['sides']['a']['hand']:
            rng=Random(rng_state);expected.append(rng.choice((-1,0,1)));rng_state=rng.getrandbits(63)
        begin_turn(s,'a')
        self.assertEqual([c['jingu_mark']['delta'] for c in s['sides']['a']['hand'][:2]],expected)
        end_ultimate(s,'a','xiaozhi')
        self.assertTrue(all('jingu_mark' not in c for c in s['sides']['a']['hand']))

    def test_battle_uses_gold_before_mark_then_settles_after_full_chain(self):
        s=self.game();self.hand(s,'Q01');s=self.ult(s)
        hero(s,'a','xiaozhi')['jingu']=2
        card=s['sides']['a']['hand'][0];card['jingu_mark']['delta']=1
        s=self.play(s,card)
        attack=[e for e in s['events'] if e['type']=='attack'][-1]
        self.assertEqual(attack['amount'],4) # +1 mark cannot pay the 3-gold condition retroactively
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],4) # hit player +1, then mark +1
        s=self.game();self.hand(s,'Q03');s=self.ult(s);hero(s,'a','xiaozhi')['jingu']=6
        card=s['sides']['a']['hand'][0];card['jingu_mark']['delta']=-1
        start=s['event_seq'];s=self.play(s,card,option_id='release_ba')
        attacks=[e for e in s['events'] if e['seq']>start and e['type']=='attack']
        self.assertEqual(len(attacks),3)
        mark_event=next(e for e in s['events'] if e['seq']>start and '手牌金谷标注结算' in e['text'])
        self.assertGreater(mark_event['seq'],attacks[-1]['seq'])
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],2)

    def test_choose_base_never_forces_gold_payment(self):
        for cid in ('Q01', 'Q02', 'Q03'):
            s=self.game();hero(s,'a','xiaozhi')['jingu']=9
            card=self.hand(s,cid)[0];start=s['event_seq'];s=self.play(s,card)
            self.assertEqual(hero(s,'a','xiaozhi')['jingu'],10)
            self.assertEqual(len([e for e in s['events'] if e['seq']>start and e['type']=='attack']),1)

    def test_paid_choices_release_leftmost_only_and_validate_cost_before_damage(self):
        for card_id,option,gold,attacks in [('Q01','boost',3,[7]),('Q02','release_a',3,[3,4]),
                                           ('Q03','release_b',3,[4,3]),('Q03','release_ba',6,[4,3,4])]:
            s=self.game();hero(s,'a','xiaozhi')['jingu']=gold
            card=self.hand(s,card_id)[0];start=s['event_seq'];s=self.play(s,card,option_id=option)
            self.assertEqual([e['amount'] for e in s['events'] if e['seq']>start and e['type']=='attack'],attacks)
            self.assertEqual(hero(s,'a','xiaozhi')['jingu'],len(attacks))
        s=self.game();hero(s,'a','xiaozhi')['jingu']=5;card=self.hand(s,'Q03')[0]
        before=deepcopy(s)
        with self.assertRaises(ValueError):self.play(s,card,option_id='release_ba')
        self.assertEqual(s,before)
        with self.assertRaises(ValueError):self.play(s,card,option_id='unknown')
        # Automatic play without a selected option must use base, despite ample gold.
        from app.modules.card_game.engine.duel_v2.flow import resolve_play
        s=self.game();hero(s,'a','xiaozhi')['jingu']=9;card=card_instance(s,'Q03','a')
        resolve_play(s,'a',card,pay=False,option_id='release_ba')
        self.assertEqual(len([e for e in s['events'] if e['type']=='attack']),1)
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],10)

    def test_response_mark_applies_after_counter_not_before_weapon_scaling(self):
        s=self.game();begin_turn(s,'b');s['sides']['b']['hand']=[];self.hand(s,'Q04',side='b');s=self.ult(s,side='b')
        h=hero(s,'b','xiaozhi');h.update(jingu=2,shape='Q08')
        card=s['sides']['b']['hand'][0];card['jingu_mark']['delta']=-1
        s['active_side']='a';s['sides']['b']['front']='xiaozhi'
        hero(s,'a','zero').update(hp=30,max_hp=30)
        s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        attack=[e for e in s['events'] if e['type']=='attack'][-1]
        self.assertEqual(attack['counter'],6) # set 5 + weapon 1 using pre-mark gold 2
        self.assertEqual(hero(s,'b','xiaozhi')['jingu'],1)
        self.assertFalse(s.get('_deferred_card_marks'))

    def test_response_mark_settles_without_enemy_turn_entry_harmony(self):
        s=self.game();begin_turn(s,'b');s['sides']['b']['hand']=[]
        self.hand(s,'N02',side='b');s=self.ult(s,side='b')
        s['sides']['b']['hand'][0]['jingu_mark']['delta']=1
        s['sides']['b']['front']='zero';hero(s,'b','zero')['harmony']=2
        before=hero(s,'b','xiaozhi')['jingu']
        s['active_side']='a';hero(s,'a','zero')['hp']=1
        s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        self.assertEqual(hero(s,'a','zero')['down_turns'],3)
        self.assertEqual(hero(s,'b','zero')['harmony'],2)
        self.assertFalse(any(e['type']=='harmony' and e['side']=='b' for e in s['events']))
        self.assertEqual(hero(s,'b','xiaozhi')['jingu'],before+1)
        self.assertFalse(s.get('_deferred_card_marks'))

    def test_fixed_attack_ignores_character_growth_and_weapon_panel(self):
        for cid,amount in [('Q01',4),('Q02',3),('Q03',4),('Q04',5)]:
            s=self.game();h=hero(s,'a','xiaozhi');h.update(base_attack=20,jingu=-3,shape='Q07')
            cards=self.hand(s,cid);s=self.play(s,cards[0])
            self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],amount)

    def test_gold_survives_down_and_recovery_runs_before_begin_effect(self):
        s=self.game();h=hero(s,'a','xiaozhi');h.update(jingu=-2,down_turns=1,hp=0)
        start=s['event_seq'];begin_turn(s,'a')
        self.assertEqual((h['hp'],h['down_turns'],h['jingu']),(4,0,-1))
        events=[e for e in s['events'] if e['seq']>start]
        revival=next(e for e in events if e['type']=='revive')
        gain=next(e for e in events if '小吱触发异能' in e['text'])
        self.assertLess(revival['seq'],gain['seq'])
        damage(s,'a','xiaozhi',99)
        self.assertEqual(h['jingu'],-1)

    def test_once_per_character_survives_refill_death_and_other_ultimates_allowed(self):
        s=self.game();s['escalation_enabled']=True;s['turn']=6
        s=self.ult(s,cid='haiyue');h=hero(s,'a','haiyue');h['energy']=6
        action={'type':'ultimate','character_id':'haiyue'}
        self.assertNotIn(action,legal_actions(s,'a'))
        damage(s,'a','haiyue',99);h.update(hp=4,down_turns=0)
        self.assertNotIn(action,legal_actions(s,'a'))
        hero(s,'a','zero')['energy']=5
        self.assertIn({'type':'ultimate','character_id':'zero'},legal_actions(s,'a'))
        s=json.loads(json.dumps(s))
        self.assertNotIn(action,legal_actions(s,'a'))
        begin_turn(s,'a')
        self.assertIn(action,legal_actions(s,'a'))

    def test_marks_are_owner_only_and_replay_patches_capture_them(self):
        s=self.game();self.hand(s,'Q01','N01');opening=deepcopy(s);s=self.ult(s)
        own=observe(s,'a');foe=observe(s,'b')
        self.assertTrue(all('jingu_mark' in c for c in own['sides']['a']['hand']))
        self.assertTrue(all(c=={'hidden':True} for c in foe['sides']['a']['hand']))
        for viewer in ('a','b'):
            log=build_viewer_log(opening,s,viewer,name_a='测试甲',name_b='测试乙')
            self.assertFalse(leak_markers(log))
            hand=assemble_replay_game(log['opening_board'],log['events'])['sides']['a']['hand']
            self.assertEqual('jingu_mark' in hand[0],viewer=='a')
            if viewer=='b':self.assertNotIn('private_hand',json.dumps(log))


class CompiledUltimateGateTest(unittest.TestCase):
    def test_native_gate_matches_python_after_refill(self):
        # Isolated generated-kernel contract; does not approve complete training
        # support (the repository still has unrelated content/IR gaps).
        import tempfile
        from app.modules.card_game.rl.rule_ir.compile_engine import compile_native_engine
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row, legal_entries
        from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import ACT_ULTIMATE, SEAT_INDEX
        from app.modules.card_game.engine.duel_v2.flow import new_game
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        state['turn'] = 6
        for team in state['sides'].values(): team['hand'] = []
        for cid in ('nanali', 'zero'): hero(state, 'a', cid)['energy'] = 5
        with tempfile.TemporaryDirectory() as path:
            from unittest.mock import patch
            # Only this unit fixture stubs unrelated completeness checks. The
            # production compiler and model loader keep their strict guards.
            with patch('app.modules.card_game.rl.rule_ir.emit_engine.validate_compiled_deck'):
                engine = compile_native_engine(path)
            try:
                packed = engine.launch_lists([pack_python_row(state)], [-1])[0]
                action = {'type':'ultimate', 'character_id':'nanali'}
                index = map_python_action(packed, state, action)
                packed = engine.launch_lists([packed], [index])[0]
                packed[OFFSETS['ch_energy'] + SEAT_INDEX['nanali']] = 5
                packed = engine.launch_lists([packed], [-1])[0]
                actors = {entry['actor'] for entry in legal_entries(packed) if entry['type'] == ACT_ULTIMATE}
                self.assertNotIn(SEAT_INDEX['nanali'], actors)
                self.assertIn(SEAT_INDEX['zero'], actors)
                self.assertEqual(packed[OFFSETS['ch_ultimate_used_turn'] + SEAT_INDEX['nanali']], 6)
                state = apply_action(state, 'a', action)
                hero(state, 'a', 'nanali')['energy'] = 5
                pyrow = engine.launch_lists([pack_python_row(state)], [-1])[0]
                self.assertEqual(legal_entries(packed), legal_entries(pyrow))
            finally:
                engine.close()


if __name__=='__main__':unittest.main()
