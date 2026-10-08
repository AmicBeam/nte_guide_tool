"""Unified modifiers: query purity, clocks, stacking, source and public boundaries."""
import json
import unittest
from copy import deepcopy

from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn, card_cost
from app.modules.card_game.engine.duel_v2.state import (
    hero, attack_value, add_atk_buff, damage, card_instance, can_gain_energy, move_out,
)
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.modifiers import (
    add_effect, deadline, evaluate, tick, clear, remove_source,
    compatibility_flags, public_effects, contribution, reduce_modifiers,
)
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class ModifiersTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=91, first_side='a', skip_mulligan=True, escalation=False)
        for side in s['sides'].values():
            side.update(hand=[], shield=0, ap=10)
        return s

    def test_independent_expiry_and_other_buffs_preserved_across_json(self):
        s = self.game(); h = hero(s, 'a', 'zero')
        add_atk_buff(h, 3)
        for turns, amount in ((1, 2), (2, -1)):
            add_effect(h, 'test.timed', 'attack.panel', amount,
                       expiry=deadline(s, 'a', turns))
        baseline = deepcopy(s)
        self.assertEqual(attack_value(h, s, 'a'), 6)
        self.assertEqual(s, baseline, 'query is pure')
        s = json.loads(json.dumps(s)); h = hero(s, 'a', 'zero')
        begin_turn(s, 'b'); self.assertEqual(attack_value(h, s, 'a'), 6)
        begin_turn(s, 'a'); self.assertEqual(attack_value(h, s, 'a'), 4)
        begin_turn(s, 'b'); begin_turn(s, 'a')
        self.assertEqual(attack_value(h, s, 'a'), 5)
        self.assertEqual(len(h['effects']), 1)

    def test_all_turn_durations_expire_before_start_effects(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.registry import KITS
        s = self.game(); h = hero(s, 'a', 'zero')
        add_effect(h, 'test.positive', 'attack.panel', 2, expiry=deadline(s, 'a', 1))
        add_effect(h, 'test.negative', 'attack.panel', -1, expiry=deadline(s, 'a', 1))
        observed = []
        with patch.object(KITS['zero'], 'on_turn_begin', staticmethod(
                lambda c: observed.append(attack_value(c.character, c.state, c.side))), create=True):
            begin_turn(s, 'a')
        self.assertEqual(observed, [2])
        self.assertFalse(h['effects'])

    def test_turn_end_duration_remains_distinct_from_next_turn_start(self):
        s = self.game(); h = hero(s, 'a', 'zero')
        add_effect(h, 'test.end', 'attack.panel', 4, expiry=deadline(s, 'a', 0, 'turn_end'))
        tick(s, 'a', 'turn_start'); self.assertEqual(attack_value(h, s, 'a'), 6)
        tick(s, 'a', 'turn_end'); self.assertEqual(attack_value(h, s, 'a'), 2)

    def test_aura_tracks_source_front_death_revival_and_both_sides(self):
        s = self.game(); source = hero(s, 'a', 'nanali'); target = hero(s, 'a', 'zero')
        add_effect(source, 'test.front_aura', 'attack.panel', 2, kind='aura', scope='other_allies',
                   source_entity_id=source['entity_id'], conditions=[{'id':'source_front'}])
        self.assertEqual(attack_value(target, s, 'a'), 2)
        s['sides']['a']['front'] = 'nanali'
        self.assertEqual(attack_value(target, s, 'a'), 4)
        self.assertEqual(attack_value(source, s, 'a'), 2)
        self.assertEqual(attack_value(hero(s,'b','zero'), s, 'b'), 2)
        damage(s, 'a', 'nanali', 99)
        self.assertEqual(attack_value(target, s, 'a'), 2)
        source.update(hp=5, down_turns=0); s['sides']['a']['front'] = 'nanali'
        self.assertEqual(attack_value(target, s, 'a'), 4)
        move_out(s, 'a', 'nanali')
        self.assertEqual(attack_value(target, s, 'a'), 2)

    def test_persistent_effect_survives_source_death_but_clears_with_target(self):
        s = self.game(); source = hero(s,'a','nanali'); target = hero(s,'a','zero')
        add_effect(target, 'test.gift', 'attack.panel', 2, source_entity_id=source['entity_id'])
        damage(s,'a','nanali',99)
        self.assertEqual(attack_value(target,s,'a'),4)
        damage(s,'a','zero',99)
        self.assertFalse(target['effects'])

    def test_weapon_removal_only_removes_linked_source_not_other_instances(self):
        s=self.game(); source=hero(s,'a','nanali'); target=hero(s,'a','zero')
        for key in ('N07','N08'):
            add_effect(target,'test.weapon','attack.panel',1,source_entity_id=source['entity_id'],
                       source_key=key,clear_on=('down','source_removed'))
        source['shape']='N07'
        c=EffectContext(s,'a','nanali',{'side':'a','card':card_instance(s,'N08','a')})
        c.set_shape()
        self.assertEqual([e['source_key'] for e in target['effects']],['N08'])
        self.assertEqual(attack_value(target,s,'a'),3)

    def test_stacking_and_reapply_do_not_overwrite_other_sources(self):
        s=self.game(); h=hero(s,'a','zero')
        for policy, expected in [('independent',6),('refresh',2),('max',4),('replace',4)]:
            h['effects']=[]
            first=add_effect(h,'test.stack','attack.panel',2,source_key='one',stacking=policy)
            second=add_effect(h,'test.stack','attack.panel',4,source_key='one',stacking=policy)
            self.assertEqual(evaluate(s,'a',h,'attack.panel',0)['value'],expected)
            self.assertEqual(first['effect_id']==second['effect_id'], policy!='independent')
            add_effect(h,'test.stack','attack.panel',1,source_key='two',stacking=policy)
            self.assertEqual(evaluate(s,'a',h,'attack.panel',0)['value'],expected+1)

    def test_nonattack_cost_damage_and_rule_aura_share_evaluator(self):
        s=self.game(); source=hero(s,'a','nanali'); target=hero(s,'a','zero')
        add_effect(source,'test.cost','card.cost',-1,kind='aura',scope='allies')
        self.assertEqual(card_cost(s,'a',card_instance(s,'Z03','a')),0)
        self.assertEqual(card_cost(s,'b',card_instance(s,'Z03','b')),1)
        add_effect(source,'test.energy','rule.gain_energy',True,op='forbid',kind='aura',scope='allies')
        self.assertFalse(can_gain_energy(s,'a','zero'))
        add_effect(target,'test.reduction','damage.amount',0.5,op='multiply')
        damage(s,'a','zero',4)
        self.assertEqual(target['hp'],3)
        damage(s,'a','nanali',99)
        self.assertEqual(card_cost(s,'a',card_instance(s,'Z03','a')),1)
        self.assertTrue(can_gain_energy(s,'a','zero'))

    def test_legacy_expiring_slice_is_not_counted_twice(self):
        s=self.game(); h=hero(s,'a','zero')
        h['flags'].update(atk_buff=5,atk_buff_expiring={'amount':2,'until':2},
                          timed_attack=[{'amount':-1,'left':2}])
        self.assertEqual(attack_value(h,s,'a'),6)
        add_atk_buff(h,1)
        self.assertNotIn('atk_buff',h['flags'])
        self.assertEqual(attack_value(h,s,'a'),7)
        begin_turn(s,'a');self.assertEqual(attack_value(h,s,'a'),5)
        begin_turn(s,'a');self.assertEqual(attack_value(h,s,'a'),6)
        self.assertEqual(compatibility_flags(h,s,'a')['atk_buff'],4)

    def test_next_attack_is_consumed_only_by_attack_not_panel_or_preview(self):
        s=self.game(); h=hero(s,'a','zero')
        h['flags']['next_attack']=2
        c=EffectContext(s,'a','nanali',{'side':'a'})
        c.grant_next_attack(h,1)
        self.assertEqual(attack_value(h,s,'a'),2)
        baseline=deepcopy(s); observe(s,'a'); self.assertEqual(s,baseline)
        s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],4)
        self.assertEqual(compatibility_flags(hero(s,'a','zero'))['next_attack'],0)

    def test_layer_order_explanations_and_unknown_conditions(self):
        effects=[contribution('add','attack.hit',3,layer=1),
                 contribution('set','attack.hit',4,op='set',layer=0),
                 contribution('multiply','attack.hit',2,op='multiply',layer=2)]
        result=reduce_modifiers(99,effects)
        self.assertEqual(result['value'],14)
        self.assertEqual([e['after'] for e in result['contributions']],[4,7,14])
        with self.assertRaises(ValueError):
            contribution('bad','attack.panel',1,conditions=[{'id':'eval(code)'}])
        with self.assertRaises(ValueError):contribution('bad','attack.panel',float('nan'))

    def test_private_effect_metadata_is_not_projected_to_opponent_or_replay(self):
        s=self.game(); h=hero(s,'a','zero')
        add_effect(h,'private.name','attack.panel',0,visibility='private',source_key='hidden-card')
        add_effect(h,'public.name','attack.panel',1,label='公开增益')
        baseline=deepcopy(s)
        for viewer in ('a','b'):
            view=observe(s,viewer,include_previews=False)
            self.assertNotIn('hidden-card',str(view))
            self.assertNotIn('private.name',str(view))
        board=capture_public_board(s)
        self.assertNotIn('hidden-card',str(board))
        self.assertEqual(s,baseline)
        self.assertEqual(len(public_effects(h)),1)

    def test_natural_expiry_uses_shared_hook_after_all_due_modifiers_are_removed(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.registry import KITS
        s=self.game(); h=hero(s,'a','zero')
        for amount in (1,2):
            add_effect(h,'test.expire','attack.panel',amount,source_entity_id=h['entity_id'],
                       expiry=deadline(s,'a',1))
        seen=[]
        def on_expired(c,effect,holder_entity_id):
            if holder_entity_id == c.entity_id:
                seen.append((effect['value'],attack_value(c.character,c.state,c.side),holder_entity_id))
        with patch.object(KITS['zero'],'on_effect_expired',staticmethod(on_expired),create=True):
            begin_turn(s,'a');begin_turn(s,'b');begin_turn(s,'a')
        self.assertEqual(seen,[(1,2,h['entity_id']),(2,2,h['entity_id'])])
        self.assertEqual(len([e for e in s['events'] if e['type']=='effect_expired']),2)

    def test_manual_removal_and_refresh_do_not_fire_expiry(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.registry import KITS
        s=self.game(); h=hero(s,'a','zero')
        add_effect(h,'test.remove','attack.panel',1,source_key='weapon',
                   expiry=deadline(s,'a',1),clear_on=('leave',))
        seen=[]
        with patch.object(KITS['zero'],'on_effect_expired',staticmethod(lambda *a,**kw:seen.append(kw)),create=True):
            s['sides']['a']['front']='zero';move_out(s,'a','zero');begin_turn(s,'a')
        self.assertEqual(seen,[])


    def test_conditional_one_shot_is_consumed_only_when_it_contributes(self):
        s=self.game();h=hero(s,'a','zero')
        add_effect(h,'test.conditional_hit','attack.hit',3,consume_on='attack',
                   conditions=[{'id':'resource_at_least','args':{'resource':'energy','amount':1}}])
        s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        self.assertTrue(hero(s,'a','zero')['effects'])
        card=card_instance(s,'Z04','a');s['sides']['a']['hand'].append(card)
        s=apply_action(s,'a',{'type':'play_card','card_id':card['instance_id']})
        self.assertEqual([e for e in s['events'] if e['type']=='attack'][-1]['amount'],7)
        self.assertFalse(hero(s,'a','zero')['effects'])


    def test_independent_stack_cap_replaces_oldest_without_merging_deadlines(self):
        s=self.game();h=hero(s,'a','zero')
        ids=[]
        for turns in (1,2,3):
            ids.append(add_effect(h,'test.cap','attack.panel',1,max_stacks=2,
                                  expiry=deadline(s,'a',turns))['effect_id'])
        self.assertEqual([e['effect_id'] for e in h['effects']],ids[1:])
        begin_turn(s,'a');self.assertEqual(attack_value(h,s,'a'),4)
        begin_turn(s,'a');self.assertEqual(attack_value(h,s,'a'),3)

    def test_clock_queries_exclude_expired_values_without_mutating_snapshots(self):
        s=self.game();h=hero(s,'a','zero')
        add_effect(h,'test.clock','attack.panel',3,expiry=deadline(s,'a',1))
        s['sides']['a']['turn_count']+=1
        before=deepcopy(s)
        self.assertEqual(attack_value(h,s,'a'),2)
        self.assertEqual(s,before)


    def test_max_stack_does_not_expose_private_winning_source(self):
        s=self.game();h=hero(s,'a','zero')
        add_effect(h,'private','attack.hit',3,stack_key='shared',stacking='max',
                   visibility='private',source_key='hidden-source')
        add_effect(h,'public','attack.hit',1,stack_key='shared',stacking='max',source_key='public-source')
        self.assertEqual(evaluate(s,'a',h,'attack.hit',0)['value'],3)
        self.assertEqual(public_effects(h),[])


    def test_expiry_trigger_can_be_observed_from_either_team(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.registry import KITS
        s=self.game();h=hero(s,'a','zero')
        add_effect(h,'test.observe','attack.panel',1,source_entity_id=h['entity_id'],expiry=deadline(s,'a',1))
        seen=[]
        with patch.object(KITS['nanali'],'on_effect_expired',staticmethod(
                lambda c,**kw:seen.append((c.side,kw['effect']['source_entity_id']))),create=True):
            begin_turn(s,'a')
        self.assertEqual(seen,[('a',h['entity_id']),('b',h['entity_id'])])


    def test_all_expired_ultimates_exit_before_any_expiry_listener(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.registry import KITS
        s=self.game()
        for cid in ('zero','nanali'):
            hero(s,'a',cid).update(awakened=True,ultimate_turns=1)
        seen=[]
        def inspect(c,**kw):
            if c.side=='a':
                seen.append(tuple(hero(c.state,'a',cid)['awakened'] for cid in ('zero','nanali')))
        with patch.object(KITS['nanali'],'on_effect_expired',staticmethod(inspect),create=True):
            begin_turn(s,'a')
        self.assertEqual(seen,[(False,False),(False,False)])
