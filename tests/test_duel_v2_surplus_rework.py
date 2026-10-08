"""September 20 user rules: delay episodes, surplus actions, Yi attachments."""
from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags
import json
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, legal_actions
from app.modules.card_game.engine.duel_v2.flow import begin_turn, trigger_responses, operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import (
    hero, front_debuff, card_instance, damage, energy, move_out, attack_value, effective_card,
)
from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony, resolve_genesis
from app.modules.card_game.engine.duel_v2.deferred import drain
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


class SurplusReworkTest(unittest.TestCase):
    def game(self, team=('zhenhong', 'yi', 'zero', 'jiuyuan'), escalation=False):
        deck = dict(id='rework', name='规则测试', character_ids=list(team), card_ids=[key for cid in team for key, c in CARDS.items()
                    if c['character_id'] == cid and not c.get('derived')])
        s = new_game(seed=61, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck}, escalation=escalation)
        for t in s['sides'].values():
            t.update(hand=[], ap=20, shield=0, hp=100)
        return s

    def hold(self, s, key, side='a'):
        c = card_instance(s, key, side); s['sides'][side]['hand'].append(c); return c

    def play(self, s, key, side='a'):
        c = self.hold(s, key, side)
        return apply_action(s, side, dict(type='play_card', card_id=c['instance_id']))

    def delay(self, s, left=1):
        front_debuff(s, 'b')['delay'] = dict(left=left, by='a', tick_side='a', slow=1, slow_turn=s['turn'])

    def test_delay_owner_clock_first_global_turn_and_episode_refresh(self):
        s = self.game()
        c = operation(s, 'a', 'yi'); hero(s,'a','zero')['harmony']=2
        apply_enter_harmony(c, 'zero', '延滞')
        finish_operation(s)
        status=front_debuff(s,'b')['delay']; first=status['slow_turn']
        self.assertEqual((status['left'],status['tick_side']), (1,'a'))
        begin_turn(s,'b'); self.assertEqual(status['left'],1)
        # Refresh during the same episode cannot restart its first-turn slow.
        c=operation(s,'a','yi'); hero(s,'a','zero')['harmony']=2
        apply_enter_harmony(c,'zero','延滞');finish_operation(s)
        self.assertEqual(front_debuff(s,'b')['delay']['slow_turn'],first)
        begin_turn(s,'a');self.assertNotIn('delay',front_debuff(s,'b'))
        c=operation(s,'a','yi');hero(s,'a','zero')['harmony']=2
        apply_enter_harmony(c,'zero','延滞');finish_operation(s)
        self.assertEqual(front_debuff(s,'b')['delay']['slow_turn'],s['turn'])

    def test_delay_counter_damage_only_first_turn(self):
        from app.modules.card_game.engine.duel_v2.combat import _zone_slow
        s=self.game();self.delay(s,2);s['sides']['b']['front']='zero'
        self.assertEqual(_zone_slow(s,'b','zero'),1)
        begin_turn(s,'b');s['sides']['b']['front']='zero'
        self.assertEqual(_zone_slow(s,'b','zero'),0)
        self.assertIn('delay',front_debuff(s,'b'))

    def test_surplus_doubled_genesis_not_two_notifications_and_action_after_resources(self):
        s=self.game();self.delay(s);s['sides']['a']['front']='zero';hero(s,'a','zero')['harmony']=2
        s['sides']['b']['front']='yi';hero(s,'b','yi').update(hp=60,max_hp=60)
        seq=s['event_seq'];s=apply_action(s,'a',dict(type='attack',character_id='jiuyuan'))
        h=hero(s,'a','zhenhong')
        self.assertEqual(compatibility_flags(h)['atk_buff'],1)
        self.assertEqual(h['extra_attacks'],1)
        self.assertEqual((h['harmony'],h['energy']),(1,2))
        events=[e for e in s['events'] if e['seq']>seq]
        attacks=[e['actor'] for e in events if e['type']=='attack']
        self.assertEqual(attacks,['a:jiuyuan','a:zhenhong'])
        first_resource=next(i for i,e in enumerate(events) if e['type']=='resource' and e.get('actor')=='a:jiuyuan')
        extra_attack=next(i for i,e in enumerate(events) if e['type']=='attack' and e.get('actor')=='a:zhenhong')
        self.assertLess(first_resource,extra_attack)
        self.assertTrue(any(e.get('counter_immunity')=='support' for e in events))

    def test_extra_entry_can_trigger_genesis_and_enqueue_second_attack(self):
        s=self.game();self.delay(s);s['sides']['a']['front']='jiuyuan';hero(s,'a','jiuyuan')['harmony']=2
        s['sides']['b']['front']='yi';hero(s,'b','yi').update(hp=70,max_hp=70)
        c=operation(s,'a','jiuyuan');resolve_genesis(c);finish_operation(s)
        h=hero(s,'a','zhenhong')
        self.assertEqual((h['extra_attacks'],compatibility_flags(h)['atk_buff']), (2,2))
        self.assertEqual(h['harmony'],1)
        self.assertEqual(h['energy'],5)
        self.assertFalse(s.get('deferred_actions'))

    def test_energy_gate_and_escalation_excludes_bench(self):
        from app.modules.card_game.engine.duel_v2.escalation import begin_escalation
        s=self.game(escalation=True);s['turn']=6;h=hero(s,'a','zhenhong')
        for cid in ('yi','zero','jiuyuan'):hero(s,'a',cid)['energy']=1
        energy(s,'a','zhenhong',5);self.assertEqual(h['energy'],0)
        begin_escalation(s,'a');self.assertEqual(h['energy'],0)
        self.assertEqual(sum(hero(s,'a',cid)['energy'] for cid in ('yi','zero','jiuyuan')),4)
        s['sides']['a']['front']='zhenhong';energy(s,'a','zhenhong');self.assertEqual(h['energy'],1)
        h['awakened']=True;energy(s,'a','zhenhong');self.assertEqual(h['energy'],1)

    def test_iloy_charges_before_front_returns(self):
        s=self.game(team=('zhenhong','iloy','zero','yi'))
        s['sides']['a']['front']='zhenhong';hero(s,'a','iloy')['flags']['energy_next']=True
        begin_turn(s,'a');self.assertEqual(hero(s,'a','zhenhong')['energy'],1)
        self.assertIsNone(s['sides']['a']['front'])

    def test_typed_battles_pay_ap_and_use_reduced_attack(self):
        for key, surplus, amount in [('R02',False,1),('R02',True,1),('R03',True,2),('R04',True,3)]:
            with self.subTest(key=key,surplus=surplus):
                s=self.game();s['sides']['a']['surplus']=surplus
                s['sides']['a']['ap']=1
                s=self.play(s,key)
                self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],amount)
                self.assertFalse(s['sides']['a']['used'].get('instant'))
                self.assertEqual(s['sides']['a']['ap'],0)

    def test_growth_battle_counts_solitary_triggers_and_survives_down(self):
        s=self.game();h=hero(s,'a','zhenhong');h['surplus_passive_triggers']=3;h['extra_attacks']=8
        damage(s,'a','zhenhong',99);self.assertEqual(h['surplus_passive_triggers'],3)
        h.update(hp=5,down_turns=0)
        s=self.play(s,'R06');h=hero(s,'a','zhenhong')
        self.assertEqual(h['shield'],3)
        self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],5)
        self.assertEqual(h['surplus_passive_triggers'],3)

    def test_downed_response_discards_only_owned_cards_and_retains_other_cards(self):
        s=self.game();self.hold(s,'R05');self.hold(s,'R02');other=self.hold(s,'I01')
        count=len(s['sides']['a']['deck']);damage(s,'a','zhenhong',99)
        self.assertEqual(len(s['sides']['a']['deck']),count-1)
        self.assertIn(other,s['sides']['a']['hand'])
        self.assertEqual({c['card_id'] for c in s['sides']['a']['discard']},{'R05','R02'})
        self.assertEqual(hero(s,'a','zhenhong')['down_turns'],3)

    def test_downed_response_unaffordable_and_wrong_owner(self):
        s=self.game();c=self.hold(s,'R05');s['sides']['a']['ap']=0;s['sides']['a']['used']['instant']=True
        damage(s,'a','zhenhong',99);self.assertIn(c,s['sides']['a']['hand'])
        s=self.game();c=self.hold(s,'R05');damage(s,'a','yi',99);self.assertIn(c,s['sides']['a']['hand'])

    def test_ultimate_transforms_identity_legality_preview_and_restores(self):
        s=self.game();physical=self.hold(s,'I03');self.hold(s,'R05');h=hero(s,'a','zhenhong');h['energy']=5
        s=apply_action(s,'a',dict(type='ultimate',character_id='zhenhong'))
        view=observe(s,'a');hand=view['sides']['a']['hand']
        self.assertEqual({c['card_id'] for c in hand},{'RF01'})
        shown = next(c for c in hand if c['instance_id'] == physical['instance_id'])
        self.assertEqual(shown['entity_id'],physical['entity_id'])
        self.assertEqual(shown['hand_face']['card_id'], 'I03')
        self.assertNotIn('_physical_card',str(hand))
        self.assertTrue(all(c.get('hidden') for c in observe(s,'b')['sides']['a']['hand']))
        s=apply_action(s,'a',dict(type='play_card',card_id=physical['instance_id']))
        self.assertEqual(s['sides']['a']['discard'][-1]['card_id'],'I03')
        self.assertEqual(hero(s,'a','zhenhong')['energy'],0)
        self.assertEqual(hero(s,'a','zhenhong')['harmony'],2)
        self.assertEqual(hero(s,'a','yi')['energy'],1)
        self.assertFalse(hero(s,'a','yi').get('beast_fangs'))
        move_out(s,'a','zhenhong')
        self.assertEqual(observe(s,'a',include_previews=False)['sides']['a']['hand'][0]['card_id'],'R05')

    def test_ultimate_stays_then_natural_expiry_deals_all_character_damage(self):
        s=self.game();h=hero(s,'a','zhenhong');h['energy']=5;s['sides']['a']['front']='zhenhong'
        s=apply_action(s,'a',dict(type='ultimate',character_id='zhenhong'))
        begin_turn(s,'a');self.assertEqual(s['sides']['a']['front'],'zhenhong')
        self.assertEqual(hero(s,'a','zhenhong')['ultimate_turns'],1)
        before=s['sides']['b']['hp'];seq=s['event_seq']
        begin_turn(s,'a');h=hero(s,'a','zhenhong')
        self.assertEqual((h.get('extra_attacks',0),h['energy'],h['harmony']),(0,0,1))
        self.assertEqual(s['sides']['b']['hp'],before-3)
        self.assertFalse(any(e['type']=='attack' for e in s['events'] if e['seq']>seq))
        self.assertFalse(h['awakened']);self.assertIsNone(s['sides']['a']['front'])

    def test_voluntary_switch_ends_ultimate_without_final_attack(self):
        s = self.game()
        hero(s, 'a', 'zhenhong')['energy'] = 5
        s = apply_action(s, 'a', dict(type='ultimate', character_id='zhenhong'))
        seq = s['event_seq']
        s = apply_action(s, 'a', dict(type='attack', character_id='zero'))
        h = hero(s, 'a', 'zhenhong')
        self.assertFalse(h['awakened'])
        self.assertFalse(h.get('extra_attacks'))
        attacks = [e['actor'] for e in s['events'] if e['seq'] > seq and e['type'] == 'attack']
        self.assertEqual(attacks, ['a:zero'])

    def test_death_ends_ultimate_without_final_attack(self):
        s=self.game();h=hero(s,'a','zhenhong');h.update(awakened=True,ultimate_turns=2)
        s['sides']['a']['front']='zhenhong';damage(s,'a','zhenhong',99);drain(s)
        self.assertFalse(h.get('extra_attacks'));self.assertFalse(h['awakened'])

    def test_surplus_weapon_triggers_in_front_and_on_bench(self):
        for in_front, expected in [(False,2),(True,2)]:
            s=self.game();self.delay(s);h=hero(s,'a','zhenhong');h['shape']='R07'
            s['sides']['a']['front']='zhenhong' if in_front else 'zero'
            s['sides']['b']['front']='yi';hero(s,'b','yi').update(hp=60,max_hp=60)
            hp=hero(s,'b','zero')['hp']
            c=operation(s,'a','zero');resolve_genesis(c);finish_operation(s)
            self.assertEqual(hp-hero(s,'b','zero')['hp'],expected)

    def test_intimidation_does_not_increase_panel_consumes_on_sortie(self):
        s=self.game();s=self.play(s,'R08');h=hero(s,'a','zhenhong')
        from app.modules.card_game.content.duel_v2.registry import fire
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        c=EffectContext(s,'a','zhenhong',{'side':'a'})
        fire(c,'on_turn_end',actor_only=True);fire(c,'on_turn_end',actor_only=True)
        self.assertEqual(h['intimidation'],2);self.assertEqual(attack_value(h,s,'a'),2)
        self.assertEqual((h['hp'],h['max_hp']),(8,8))
        s=apply_action(s,'a',dict(type='attack',character_id='zhenhong'));h=hero(s,'a','zhenhong')
        self.assertEqual((h['intimidation'],h['harmony']),(0,2))
        self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],4)

    def test_fangs_pause_before_decrement_and_survive_death(self):
        s=self.game();self.delay(s);s=self.play(s,'I01')
        self.assertEqual(front_debuff(s,'b')['delay']['left'],2)
        damage(s,'a','yi',99)
        for remaining in (3, 2, 1, 0):
            begin_turn(s,'a');self.assertEqual(hero(s,'a','yi')['beast_fangs'],remaining)
            self.assertEqual(front_debuff(s,'b')['delay']['left'],2)
        begin_turn(s,'a');self.assertEqual(front_debuff(s,'b')['delay']['left'],1)

    def test_yi_attack_extends_delay_one_turn_capped_at_two(self):
        # User tabletop parameters, 2026-09-21; not original-game values.
        for left, expected in ((1, 2), (2, 2), (3, 3), (4, 4)):
            with self.subTest(left=left):
                s=self.game();self.delay(s,left)
                original=deepcopy(front_debuff(s,'b')['delay'])
                before=capture_public_board(s);s['_public_board']=deepcopy(before)
                seq=s['event_seq']
                s=apply_action(s,'a',dict(type='attack',character_id='yi'))
                self.assertEqual(front_debuff(s,'b')['delay'],dict(original,left=expected))
                events=[e for e in observe(s,'a')['presentation']['events'] if e['seq']>seq]
                history=[e for e in events if '翳的攻击延长延滞' in e.get('text','')]
                self.assertEqual(len(history),int(left<2))
                if history:
                    self.assertIn('至 2 个己方回合',history[0]['text'])
                    self.assertFalse(history[0]['present'])
                board=replay_public_board(before,events)
                self.assertEqual(board['sides']['b']['front_debuff'],front_debuff(s,'b'))
                s=json.loads(json.dumps(s))
                begin_turn(s,'b')
                self.assertEqual(front_debuff(s,'b')['delay']['left'],expected)
                begin_turn(s,'a')
                self.assertEqual(front_debuff(s,'b')['delay']['left'],expected-1)

    def test_yi_attack_does_not_create_or_extend_opponent_delay(self):
        for enemy_owned in (False,True):
            with self.subTest(enemy_owned=enemy_owned):
                s=self.game()
                if enemy_owned:
                    self.delay(s);front_debuff(s,'b')['delay']['by']='b'
                original=deepcopy(front_debuff(s,'b'))
                s=apply_action(s,'a',dict(type='attack',character_id='yi'))
                self.assertEqual(front_debuff(s,'b'),original)

    def test_fangs_no_innate_damage_weapon_each_turn_front_only(self):
        s=self.game();s=self.play(s,'I03');s['sides']['b']['front']='zero';hp=hero(s,'b','zero')['hp']
        begin_turn(s,'a');self.assertEqual(hero(s,'b','zero')['hp'],hp)
        s=self.play(s,'I07');begin_turn(s,'a');self.assertEqual(hero(s,'b','zero')['hp'],hp-1)
        begin_turn(s,'b');self.assertEqual(hero(s,'b','zero')['hp'],hp-2)
        player=s['sides']['b']['hp'];begin_turn(s,'a');self.assertEqual(s['sides']['b']['hp'],player)

    def test_attachment_bonus_only_fang_and_pact(self):
        s=self.game();s=self.play(s,'I07');h=hero(s,'a','yi');h.update(energy=5)
        s=apply_action(s,'a',dict(type='ultimate',character_id='yi'))
        s['sides']['b']['front']='zero';hero(s,'b','zero').update(hp=30,max_hp=30)
        begin_turn(s,'a');self.assertEqual(hero(s,'b','zero')['hp'],28)
        h=hero(s,'a','jiuyuan');h['energy']=5;hero(s,'b','yi')['flags']['pact']=True
        s=apply_action(s,'a',dict(type='ultimate',character_id='jiuyuan'))
        self.assertEqual(hero(s,'b','yi')['hp'],1) # 1 base + (2+1) pact
        self.assertEqual(hero(s,'b','zhenhong')['hp'],5) # only base 1

    def test_skip_normal_attack_expires_next_opponent_turn_only(self):
        s=self.game();c=self.hold(s,'I05');self.assertNotIn(c['instance_id'],[a.get('card_id') for a in legal_actions(s,'a')])
        self.delay(s);s=apply_action(s,'a',dict(type='play_card',card_id=c['instance_id']))
        begin_turn(s,'b');self.assertFalse(s['sides']['b']['normal_attack_available'])
        begin_turn(s,'b');self.assertTrue(s['sides']['b']['normal_attack_available'])

    def test_weaken_two_enemy_starts_and_no_permanent_debuff(self):
        s=self.game();s['sides']['a']['delay_turns']=4
        card=self.hold(s,'I06')
        s=apply_action(s,'a',dict(type='play_card',card_id=card['instance_id'],target_id='b:zero'))
        def layers():return sum(abs(e['amount']) for h in s['sides']['b']['characters'].values()
                                for e in compatibility_flags(h, s, 'b').get('timed_attack',[]))
        self.assertEqual(layers(),4);begin_turn(s,'a');self.assertEqual(layers(),4)
        begin_turn(s,'b');self.assertEqual(layers(),4);begin_turn(s,'b');self.assertEqual(layers(),0)

    def test_i08_only_itself_instant_without_existing_weapon(self):
        from app.modules.card_game.engine.duel_v2.flow import card_plays_instant
        s=self.game();self.delay(s);s['sides']['a']['ap']=0
        card=self.hold(s,'I08');action=dict(type='play_card',card_id=card['instance_id'])
        self.assertIsNone(hero(s,'a','yi')['shape'])
        self.assertTrue(card_plays_instant(s,'a',card))
        self.assertIn(action,legal_actions(s,'a'))
        self.assertTrue(next(c for c in observe(s,'a')['sides']['a']['hand'] if c['card_id']=='I08')['instant'])
        s=apply_action(s,'a',action)
        self.assertEqual(hero(s,'a','yi')['shape'],'I08')
        self.assertEqual(s['sides']['a']['ap'],0)
        self.assertTrue(s['sides']['a']['used']['instant'])
        s['sides']['a']['used']['instant']=False
        for key in ('I01','I02','I04','I05','I06','I07'):
            self.assertFalse(card_plays_instant(s,'a',self.hold(s,key)),key)
        self.assertTrue(card_plays_instant(s,'a',self.hold(s,'I03')))

    def test_i08_requires_ap_without_delay_or_after_instant_spent(self):
        for delayed, spent in ((False,False),(True,True)):
            s=self.game();s['sides']['a'].update(ap=0)
            s['sides']['a']['used']['instant']=spent
            if delayed:self.delay(s)
            card=self.hold(s,'I08');action=dict(type='play_card',card_id=card['instance_id'])
            self.assertNotIn(action,legal_actions(s,'a'))
            s['sides']['a']['ap']=1
            self.assertIn(action,legal_actions(s,'a'))
            s=apply_action(s,'a',action)
            self.assertEqual(s['sides']['a']['ap'],0)
            self.assertEqual(bool(s['sides']['a']['used'].get('instant')),spent)

    def test_yi_weapon_instant_and_resource_replay(self):
        s=self.game();s=self.play(s,'I08');self.delay(s);s['sides']['a']['ap']=0
        before=capture_public_board(s);seq=s['event_seq'];s=self.play(s,'I03')
        self.assertTrue(s['sides']['a']['used']['instant']);self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)
        events=[e for e in observe(s,'a')['presentation']['events'] if e['seq']>seq]
        board=replay_public_board(before,events)
        self.assertEqual(next(c for c in board['sides']['a']['characters'] if c['id']=='yi')['beast_fangs'],4)

    def test_yi_buff_free_tactic_does_not_spend_instant_even_with_weapon(self):
        from app.modules.card_game.engine.duel_v2.flow import card_is_free_instant
        for spent in (False, True):
            s=self.game();self.delay(s);hero(s,'a','yi')['shape']='I08'
            s['sides']['a']['ap']=0;s['sides']['a']['used']['instant']=spent
            card=self.hold(s,'I04')
            self.assertFalse(card_is_free_instant(s,'a',card))
            view=observe(s,'a');held=next(c for c in view['sides']['a']['hand'] if c['card_id']=='I04')
            self.assertTrue(held['action_point_free'])
            action=dict(type='play_card',card_id=card['instance_id'])
            self.assertIn(action,legal_actions(s,'a'))
            s=apply_action(s,'a',action)
            self.assertEqual(s['sides']['a']['ap'],0)
            self.assertEqual(s['sides']['a']['used']['instant'],spent)
            self.assertEqual(hero(s,'a','yi')['beast_fangs'],2)
            if not spent:
                s=self.play(s,'I03')
                self.assertTrue(s['sides']['a']['used']['instant'])
                self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)

    def test_yi_battle_and_instant_tactic_four_turns(self):
        for key in ('I01','I02'):
            s=self.play(self.game(),key)
            self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)
        s=self.game();s['sides']['a']['ap']=0
        s=self.play(s,'I03')
        self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)
        self.assertEqual(s['sides']['a']['ap'],0)
        card=self.hold(s,'I03')
        self.assertNotIn(dict(type='play_card',card_id=card['instance_id']),legal_actions(s,'a'))

    def test_yi_tactics_draw_once_refresh_max_and_empty_deck_finishes(self):
        for key, turns in (('I03',4),('I04',2)):
            for previous in (0,6):
                s=self.game();s['sides']['a']['ap']=0
                hero(s,'a','yi')['beast_fangs']=previous
                s['sides']['a']['deck']=[card_instance(s,'Z03','a')]
                s=self.play(s,key)
                self.assertEqual(hero(s,'a','yi')['beast_fangs'],max(previous,turns))
                self.assertEqual([c['card_id'] for c in s['sides']['a']['hand']],['Z03'])
                self.assertEqual(s['sides']['a']['ap'],0)
                self.assertEqual(bool(s['sides']['a']['used'].get('instant')),key=='I03')
                other_view=observe(s,'b',include_previews=False)
                self.assertTrue(all(c.get('hidden') for c in other_view['sides']['a']['hand']))
            s=self.game();s['sides']['a']['deck']=[]
            s=self.play(s,key)
            self.assertEqual(s['winner'],'b')
            self.assertEqual(hero(s,'a','yi')['beast_fangs'],turns)

    def test_catalog_sources_access_and_images(self):
        self.assertEqual((CHARACTERS['zhenhong']['attack'], CHARACTERS['zhenhong']['max_hp']), (2, 6))
        self.assertEqual((CARDS['R07']['attack'], CARDS['R07']['hp']), (3, 6))
        self.assertEqual((CARDS['R08']['attack'], CARDS['R08']['hp']), (2, 8))
        self.assertEqual((CHARACTERS['yi']['attribute'],CHARACTERS['yi']['attack'],CHARACTERS['yi']['max_hp']),('相',2,5))
        self.assertIn('yi',{h['id'] for h in get_catalog(include_test_characters=False)['characters']})
        self.assertEqual([CARDS[k]['name'] for k in ('I07','I08')],['潜影追击','群犬吠形'])
        self.assertTrue(CARDS['RF01']['derived'])
        for field in ('avatar','portrait'):self.assertTrue(Path('app'+CHARACTERS['yi'][field]).is_file())


    def test_json_restore_preserves_transformed_card_and_deferred_identity(self):
        from app.modules.card_game.engine.duel_v2.entities import clone_state
        from app.modules.card_game.content.duel_v2.registry import fire
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        s=self.game();c=self.hold(s,'I04');hero(s,'a','zhenhong').update(awakened=True,ultimate_turns=2)
        fire(EffectContext(s,'a','zhenhong',{'side':'a'}),'on_surplus',actor_only=True)
        restored=clone_state(json.loads(json.dumps(s)));drain(restored)
        self.assertEqual(hero(restored,'a','zhenhong')['extra_attacks'],1)
        restored=apply_action(restored,'a',dict(type='play_card',card_id=c['instance_id']))
        self.assertEqual(restored['sides']['a']['discard'][-1]['card_id'],'I04')
        self.assertEqual(restored['sides']['a']['discard'][-1]['entity_id'],c.entity_id)

    def test_preview_uses_transformed_actor_and_no_side_effects(self):
        s=self.game();c=self.hold(s,'I07');hero(s,'a','zhenhong').update(awakened=True,ultimate_turns=2)
        before=deepcopy(s);v=observe(s,'a')
        action=next(a for a in v['legal_actions'] if a['action'].get('card_id')==c['instance_id'])
        self.assertEqual(action['interaction']['actor_id'],'a:zhenhong')
        self.assertEqual(action['preview']['attack'],6)
        self.assertEqual(s,before)

    def test_down_response_waits_for_nested_response_to_finish(self):
        s=self.game();self.hold(s,'R05');self.hold(s,'R03');s['_response_lock']=True
        damage(s,'a','zhenhong',99)
        self.assertEqual(len(s['sides']['a']['hand']),2)
        self.assertEqual(s['_pending_down_responses'],[('a','zhenhong')])
        s['_response_lock']=False
        for side,cid in s.pop('_pending_down_responses'):
            trigger_responses(s,side,'self_downed',target_id=cid)
        self.assertEqual(len(s['sides']['a']['hand']),1)

    def test_player_defeat_stops_downed_response(self):
        s=self.game();card=self.hold(s,'R05');self.hold(s,'R03')
        s['sides']['a']['hp']=0;damage(s,'a','zhenhong',99)
        self.assertEqual(s['reason'],'player_hp');self.assertIn(card,s['sides']['a']['hand'])

    def test_fangs_no_weapon_at_zero_and_downed_cannot_damage(self):
        s=self.game();hero(s,'a','yi').update(shape='I07',beast_fangs=0)
        s['sides']['b']['front']='zero';hp=hero(s,'b','zero')['hp']
        begin_turn(s,'a');self.assertEqual(hero(s,'b','zero')['hp'],hp)
        hero(s,'a','yi')['beast_fangs']=3;damage(s,'a','yi',99)
        begin_turn(s,'a');self.assertEqual(hero(s,'b','zero')['hp'],hp)

    def test_fangs_max_refresh_and_delay_turn_count_once_per_global_turn(self):
        s=self.game();s=self.play(s,'I03');s=self.play(s,'I04')
        self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)
        from app.modules.card_game.engine.duel_v2.state import record_delay_turn
        self.delay(s);record_delay_turn(s,'a');record_delay_turn(s,'a')
        self.assertEqual(s['sides']['a']['delay_turns'],1)
        begin_turn(s,'b');self.assertEqual(s['sides']['a']['delay_turns'],2)
        begin_turn(s,'a');self.assertEqual(s['sides']['a']['delay_turns'],3)

    def test_explicit_growth_and_surplus_state_survive_but_attack_buff_clears_on_down(self):
        s=self.game();h=hero(s,'a','zhenhong');h['flags']['atk_buff']=3
        s['sides']['a']['front']='zhenhong';move_out(s,'a','zhenhong')
        self.assertEqual(attack_value(h,s,'a'),5)
        s=self.play(s,'R08');h=hero(s,'a','zhenhong');self.assertEqual(attack_value(h,s,'a'),5)
        damage(s,'a','zhenhong',99);self.assertNotIn('atk_buff',h['flags'])


    def test_delayed_genesis_gives_energy_to_current_front_not_benched_source(self):
        s=self.game();self.delay(s);s['sides']['a']['front']='zero'
        c=operation(s,'a','jiuyuan');resolve_genesis(c,arm_delay=False)
        self.assertEqual(hero(s,'a','zero')['energy'],2)
        self.assertEqual(hero(s,'a','jiuyuan')['energy'],0)
        finish_operation(s)

if __name__=='__main__':unittest.main()
