"""Action-paid harmony, independent genesis layers, and author-specified kits."""
import json
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, legal_actions
from app.modules.card_game.engine.duel_v2.combat import resolve_genesis, harmony_marks, resolve_harmony_effect
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation, begin_turn, end_turn
from app.modules.card_game.engine.duel_v2.genesis import add_layers, on_sortie, tick_layers
from app.modules.card_game.engine.duel_v2.state import hero, card_instance, damage, front_debuff, move_out, attack_value
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


class HarmonyReworkTest(unittest.TestCase):
    def game(self, ids=('zero','nanali','hathor','edgar')):
        deck = dict(id='rework',name='rework',character_ids=list(ids),card_ids=[
            key for cid in ids for key,c in CARDS.items() if c['character_id']==cid and not c.get('derived')])
        s = new_game(seed=23,decks={'a':deck,'b':deck},first_side='a',skip_mulligan=True,escalation=False)
        for t in s['sides'].values():
            t.update(hand=[],ap=20,hp=100,shield=0)
        return s

    def play(self,s,key,target=None):
        card=card_instance(s,key,'a');s['sides']['a']['hand'].append(card)
        action=dict(type='play_card',card_id=card['instance_id'])
        if target: action['target_id']=target
        return apply_action(s,'a',action)

    def attack(self,s,cid):
        s['sides']['a']['normal_attack_available']=True
        return apply_action(s,'a',dict(type='attack',character_id=cid))

    def test_paid_tactic_and_normal_sortie_set_source_without_retreat(self):
        s=self.game();s=self.play(s,'N01')
        self.assertEqual(hero(s,'a','nanali')['harmony'],1)
        self.assertEqual(s['sides']['a']['last_front'],'nanali')
        self.assertIsNone(s['sides']['a']['front'])
        s=self.attack(s,'nanali');s=self.attack(s,'nanali')
        self.assertEqual(hero(s,'a','nanali')['harmony'],3)
        self.assertEqual(harmony_marks(s,'a')['payer'],'nanali')
        self.assertTrue(next(c for c in observe(s,'a')['sides']['a']['characters'] if c['id']=='nanali')['harmony_source'])

    def test_battle_card_counts_payment_once_zero_still_fills(self):
        s=self.game();s=self.play(s,'N04')
        self.assertEqual(hero(s,'a','nanali')['harmony'],1)
        s=self.play(s,'Z03')
        self.assertEqual(hero(s,'a','zero')['harmony'],3)

    def test_free_and_extra_ap_and_effect_override(self):
        s=self.game(('lingke','zero','nanali','hathor'))
        s=self.play(s,'K06')
        self.assertEqual(hero(s,'a','lingke')['harmony'],2)
        s=self.play(s,'K02')
        self.assertEqual(hero(s,'a','lingke')['harmony'],3)
        s=self.play(s,'N01')  # next non-Lingke card is instant
        self.assertEqual(hero(s,'a','nanali')['harmony'],0)
        with patch('app.modules.card_game.engine.duel_v2.harmony.decide',return_value=2):
            s=self.attack(s,'nanali')
        self.assertEqual(hero(s,'a','nanali')['harmony'],2)

    def test_source_switch_and_no_self_pairing(self):
        s=self.game();s=self.attack(s,'zero');hero(s,'a','zero')['harmony']=3
        s=self.play(s,'N01')
        self.assertEqual(s['sides']['a']['last_front'],'nanali')
        self.assertFalse(harmony_marks(s,'a')['available'])
        hero(s,'a','nanali')['harmony']=3
        s=self.attack(s,'nanali')
        self.assertFalse(s['sides']['a']['harmonized'].get('nanali'))

    def test_death_keeps_resource_but_clears_source(self):
        s=self.game();s=self.attack(s,'zero');hero(s,'a','zero')['harmony']=3
        damage(s,'a','zero',100)
        self.assertEqual(hero(s,'a','zero')['harmony'],3)
        self.assertIsNone(s['sides']['a']['last_front'])

    def test_move_out_never_creates_source_but_keeps_existing_action(self):
        s=self.game();s['sides']['a']['front']='zero';hero(s,'a','zero')['harmony']=3
        move_out(s,'a','zero');self.assertIsNone(s['sides']['a']['last_front'])
        s=self.attack(s,'zero');move_out(s,'a','zero')
        self.assertEqual(s['sides']['a']['last_front'],'zero')

    def test_cap_stays_three_in_white_heat(self):
        from app.modules.card_game.engine.duel_v2.escalation import harmony_cap
        s=self.game();s['escalation_enabled']=True
        for turn in (1,12,13,20):
            s['turn']=turn;self.assertEqual(harmony_cap(s),3)

    def test_genesis_immediate_and_each_layer_once_per_turn(self):
        s=self.game();c=operation(s,'a','zero');hp=s['sides']['b']['hp']
        resolve_genesis(c);resolve_genesis(c);resolve_genesis(c)
        self.assertEqual(s['sides']['b']['hp'],hp-3)
        on_sortie(c)
        self.assertEqual(s['sides']['b']['hp'],hp-6)
        on_sortie(operation(s,'a','nanali'))
        self.assertEqual(s['sides']['b']['hp'],hp-6)
        s['turn']+=2;on_sortie(operation(s,'a','nanali'))
        self.assertEqual(s['sides']['b']['hp'],hp-9)

    def test_layers_expire_independently_and_cap_does_not_refresh_old(self):
        s=self.game();add_layers(operation(s,'a','zero'));tick_layers(s,'a')
        add_layers(operation(s,'a','zero'));add_layers(operation(s,'a','zero'));add_layers(operation(s,'a','zero'))
        self.assertEqual([x['left'] for x in s['sides']['a']['genesis_layers']],[1,2,2])
        tick_layers(s,'a')
        self.assertEqual([x['left'] for x in s['sides']['a']['genesis_layers']],[1,1])
        tick_layers(s,'a');self.assertEqual(s['sides']['a']['genesis_layers'],[])

    def test_jiuyuan_two_stacks_per_bead_keeps_three_ordinary_slots(self):
        s=self.game(('jiuyuan','zero','nanali','hathor'));c=operation(s,'a','zero')
        for _ in range(4):add_layers(c)
        self.assertEqual(len(s['sides']['a']['genesis_layers']),3)
        self.assertTrue(all(bead['stacks']==2 for bead in s['sides']['a']['genesis_layers']))
        add_layers(c,uncapped=True)
        self.assertEqual(len(s['sides']['a']['genesis_layers']),4)
        self.assertEqual(sum(not x['uncapped'] for x in s['sides']['a']['genesis_layers']),3)

    def test_every_genesis_layer_can_return_surplus_energy_and_jingu(self):
        s=self.game(('xiaozhi','zero','nanali','hathor'));s['sides']['a']['front']='zero'
        front_debuff(s,'b')['delay']=dict(left=1,by='a',slow=1,tick_side='a',slow_turn=s['turn'])
        c=operation(s,'a','zero');add_layers(c);add_layers(c);add_layers(c)
        on_sortie(c)
        self.assertEqual(hero(s,'a','zero')['energy'],3)
        self.assertEqual(hero(s,'a','xiaozhi')['jingu'],4) # opening 1 plus 3 surplus events

    def test_nanali_participation_as_payer_and_actor_shares_three_limit(self):
        s=self.game();base=s['sides']['a']['nanali_family'];self.assertEqual(base,1)
        hero(s,'a','nanali')['harmony']=3;s['sides']['a'].update(last_front='nanali',last_actor_entity_id=hero(s,'a','nanali').entity_id)
        s=self.attack(s,'zero');self.assertEqual(s['sides']['a']['nanali_family'],2)
        hero(s,'a','zero')['harmony']=3
        s=self.attack(s,'nanali');self.assertEqual(s['sides']['a']['nanali_family'],3)
        hero(s,'a','nanali')['harmony']=3;s=self.attack(s,'zero')
        self.assertEqual(s['sides']['a']['nanali_family'],3)

    def test_nanali_ultimate_requires_zone_and_applies_to_counter_battle(self):
        s=self.game();hero(s,'a','nanali')['awakened']=True
        s=self.attack(s,'nanali')
        self.assertFalse(any(e['type']=='followup' for e in s['events']))
        s=self.game();hero(s,'b','nanali')['awakened']=True
        s['sides']['b']['front']='nanali'
        add_layers(operation(s,'b','zero'));s['operation']=None
        s=self.attack(s,'zero')
        follow=[e for e in s['events'] if e['type']=='followup']
        self.assertEqual(len(follow),1)
        self.assertEqual((follow[0]['amount'],follow[0]['target']),(2,'a:zero'))

    def test_jiuyuan_ultimate_waits_and_liquidates_first_segment_only(self):
        s=self.game(('jiuyuan','zero','nanali','hathor'));hero(s,'a','jiuyuan')['energy']=5
        hp={cid:hero(s,'b',cid)['hp'] for cid in s['sides']['b']['characters']}
        hero(s,'b','hathor')['flags']['pact']=True
        s=apply_action(s,'a',dict(type='ultimate',character_id='jiuyuan'))
        self.assertEqual(hp,{cid:hero(s,'b',cid)['hp'] for cid in hp})
        resolve_genesis(operation(s,'a','zero'))
        self.assertEqual(hero(s,'b','hathor')['hp'],hp['hathor']-2)
        self.assertFalse(hero(s,'b','hathor')['flags'].get('pact'))
        self.assertEqual(hero(s,'b','zero')['hp'],hp['zero'])
        before=hero(s,'b','zero')['hp'];resolve_genesis(operation(s,'a','zero'))
        self.assertEqual(hero(s,'b','zero')['hp'],before)

    def test_iloy_passive_and_equipped_healing_only(self):
        s=self.game(('iloy','zero','nanali','hathor'));target=hero(s,'a','zero');target['hp']=2
        before=attack_value(target,s,'a');s=self.play(s,'Y03','a:zero')
        self.assertEqual(attack_value(hero(s,'a','zero'),s,'a'),before)
        s=self.play(s,'Y08');hero(s,'a','zero')['hp']=2;s=self.play(s,'Y03','a:zero')
        self.assertEqual(attack_value(hero(s,'a','zero'),s,'a'),before+2)
        resolve_genesis(operation(s,'a','zero'));finish_operation(s)
        self.assertTrue(s['sides']['a']['extra_genesis_pending'])
        begin_turn(s,'a')
        self.assertTrue(any(x['uncapped'] for x in s['sides']['a']['genesis_layers']))

    def test_iloy_ultimate_two_start_triggers(self):
        s=self.game(('iloy','zero','nanali','hathor'));hero(s,'a','iloy')['energy']=5
        s=apply_action(s,'a',dict(type='ultimate',character_id='iloy'))
        begin_turn(s,'a');self.assertEqual(hero(s,'a','zero')['energy'],1)
        begin_turn(s,'a');self.assertEqual(hero(s,'a','zero')['energy'],2)
        self.assertFalse(hero(s,'a','iloy')['awakened'])
        begin_turn(s,'a');self.assertEqual(hero(s,'a','zero')['energy'],2)

    def test_baicang_refresh_and_burn_execution(self):
        from app.modules.card_game.engine.duel_v2.continuous import tick_owner_end
        s=self.game(('baicang','anhunqu','zero','hathor'));front_debuff(s,'b')['burn']=dict(left=1,by='a',stacks=1,source='a:anhunqu')
        damage(s,'a','baicang',1,source='a:baicang')
        self.assertEqual(front_debuff(s,'b')['burn']['left'],2)
        hero(s,'a','baicang')['awakened']=True
        # Only one living enemy makes the continuous allocation deterministic.
        for cid in s['sides']['b']['characters']:
            hero(s,'b',cid).update(hp=3 if cid=='zero' else 0,down_turns=3 if cid!='zero' else 0)
        tick_owner_end(s,'a')
        self.assertEqual(hero(s,'b','zero')['hp'],0)

    def test_bohe_refreshes_all_independent_layers_on_overflow(self):
        s=self.game(('bohe','zero','nanali','hathor'));c=operation(s,'a','zero');add_layers(c);tick_layers(s,'a');add_layers(c)
        s['sides']['b']['front']='zero';hero(s,'b','zero')['hp']=1
        s=self.attack(s,'bohe')
        self.assertTrue(all(x['left']==2 for x in s['sides']['a']['genesis_layers']))
        self.assertLess(s['sides']['b']['hp'],100)

    def test_haiyue_remote_requires_star_not_ultimate_alone(self):
        s=self.game(('haiyue','zero','nanali','hathor'));hero(s,'a','haiyue')['energy']=6
        s=apply_action(s,'a',dict(type='ultimate',character_id='haiyue'))
        s=self.attack(s,'haiyue');self.assertEqual(s['sides']['b']['hp'],97)
        resolve_harmony_effect(operation(s,'a','zero'),'黯星');finish_operation(s)
        s=self.attack(s,'haiyue');self.assertEqual(s['sides']['b']['hp'],92)

    def test_ultimate_movement_neither_triggers_harmony_nor_replaces_action_source(self):
        s=self.game(('zhenhong','jiuyuan','zero','hathor'))
        s=self.attack(s,'jiuyuan');hero(s,'a','jiuyuan')['harmony']=3
        hero(s,'a','zhenhong')['energy']=5
        before=s['sides']['b']['hp']
        s=apply_action(s,'a',dict(type='ultimate',character_id='zhenhong'))
        self.assertEqual(s['sides']['a']['front'],'zhenhong')
        self.assertEqual(s['sides']['a']['last_front'],'jiuyuan')
        self.assertEqual(hero(s,'a','jiuyuan')['harmony'],3)
        self.assertEqual(s['sides']['b']['hp'],before)
        self.assertFalse(s['sides']['a'].get('genesis_layers'))

    def test_zhenhong_first_attack_and_solitary_damage_not_sortie(self):
        from app.modules.card_game.content.duel_v2.registry import fire
        from app.modules.card_game.engine.events import GameEvent
        s=self.game(('zhenhong','zero','nanali','hathor'));self.assertEqual(hero(s,'a','zhenhong')['harmony'],2)
        s=self.attack(s,'zhenhong');self.assertEqual(hero(s,'a','zhenhong')['harmony'],3)
        self.assertEqual(s['sides']['b']['hp'],98)
        s=self.attack(s,'zhenhong');self.assertEqual(s['sides']['b']['hp'],96)
        c=operation(s,'a','zero');fire(c,GameEvent.V2_SURPLUS);finish_operation(s)
        self.assertEqual(s['sides']['b']['hp'],94)
        self.assertEqual(s['sides']['a']['front'],'zhenhong')
        self.assertEqual(s['sides']['a']['last_front'],'zhenhong')

    def test_heiyu_has_no_stones_or_attack_divination_and_harmony_works_in_witch(self):
        s=self.game(('heiyu','zero','nanali','hathor'));self.assertNotIn('PF01',CARDS)
        before=len(s['sides']['a']['deck']);s=self.attack(s,'heiyu')
        self.assertEqual(len(s['sides']['a']['deck']),before)
        self.assertFalse(s['sides']['a'].get('heiyu_divinations'))
        for witch in (False,True):
            t=self.game(('heiyu','zero','nanali','hathor'));hero(t,'a','heiyu')['awakened']=witch
            from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony
            apply_enter_harmony(operation(t,'a','zero'),None,'浸染',pay=False,as_support=False);finish_operation(t)
            self.assertEqual(t['phase'],'choice')
            self.assertEqual(t['pending_choice']['actor'],'heiyu')
            restored=json.loads(json.dumps(t));action=next(a for a in legal_actions(restored,'a') if a['type']=='choose_deck_order' and not a['bottom_card_ids'])
            restored=apply_action(restored,'a',action)
            self.assertEqual(restored['phase'],'playing')
            self.assertEqual(restored['sides']['a']['heiyu_divinations'],1)

    def test_public_layers_and_source_replay_round_trip(self):
        s=self.game();s=self.play(s,'Z03')
        opening=capture_public_board(s);seq=s['event_seq']
        s=self.attack(s,'nanali')
        events=[e for e in s['events'] if e['seq']>seq]
        self.assertEqual(replay_public_board(opening,events),capture_public_board(s))
        self.assertEqual(observe(s,'b')['sides']['a']['genesis_layers'],s['sides']['a']['genesis_layers'])
        self.assertTrue(capture_public_board(s)['sides']['a']['genesis_layers'])
        self.assertNotIn('rng',json.dumps(observe(s,'b')['sides']['a']['genesis_layers']))
