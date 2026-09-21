"""Weapon bases, permanent growth and equip sources preserve existing card rules."""
import json
import unittest
from copy import deepcopy
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.state import hero, card_instance, attack_value, damage, move_out, add_atk_buff
from app.modules.card_game.engine.duel_v2.equipment import (
    stat_layers, permanent_attack, unarmed_max_hp, grow_permanently, equip_weapon,
    refill_after_equipping, weapon_source, public_equipment,
)
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.duel_v2.modifiers import add_effect
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class EquipmentTest(unittest.TestCase):
    def game(self):
        s=new_game(seed=41,first_side='a',skip_mulligan=True,escalation=False)
        for t in s['sides'].values():t.update(hand=[],shield=0,ap=10)
        return s

    def equip(self,s,cid,key):
        c=EffectContext(s,'a',cid,{'side':'a','card':card_instance(s,key,'a')})
        c.set_shape()
        return c

    def test_growth_and_weapon_bases_are_independent_and_queries_are_pure(self):
        s=self.game();h=hero(s,'a','nanali')
        grow_permanently(h,1,1)
        self.assertEqual(stat_layers(h),{'base':{'attack':2,'max_hp':5},'permanent_growth':{'attack':1,'max_hp':1}})
        self.assertNotIn('base_attack',h)
        self.equip(s,'nanali','N08')
        self.assertEqual((attack_value(h,s,'a'),h['max_hp'],h['hp']),(3,7,7))
        add_atk_buff(h,2);h['hp']=2
        self.equip(s,'nanali','N07')
        self.assertEqual((attack_value(h,s,'a'),h['max_hp'],h['hp']),(5,6,6))
        self.assertEqual(permanent_attack(h),3)
        before=deepcopy(s);observe(s,'a');capture_public_board(s)
        self.assertEqual(s,before)

    def test_equip_and_refill_are_separate_and_do_not_emit_healing(self):
        s=self.game();h=hero(s,'a','nanali');h['hp']=1
        equip_weapon(s,h,card_instance(s,'N08','a'))
        self.assertEqual((h['hp'],h['max_hp']),(1,6))
        refill_after_equipping(h);self.assertEqual(h['hp'],6)
        h['hp']=1;seq=s['event_seq'];self.equip(s,'nanali','N07')
        self.assertEqual(h['hp'],5)
        self.assertFalse(any(e['type']=='heal' for e in s['events'] if e['seq']>seq))

    def test_old_snapshot_import_preserves_wounds_and_imports_growth_once(self):
        s=self.game();h=hero(s,'a','nanali')
        h.pop('base_stats');h.pop('permanent_growth')
        h.update(base_attack=4,base_max_hp=7,max_hp=8,hp=2,shape='N08')
        before=deepcopy(h);self.assertEqual(attack_value(h,s,'a'),4);self.assertEqual(h,before)
        s=json.loads(json.dumps(s));h=hero(s,'a','nanali')
        self.equip(s,'nanali','N07')
        self.assertEqual((h['permanent_growth'],h['hp'],h['max_hp']),({'attack':2,'max_hp':2},7,7))
        self.equip(s,'nanali','N08')
        self.assertEqual((permanent_attack(h),h['hp'],h['max_hp']),(4,8,8))

    def test_same_weapon_re_equip_changes_source_and_only_removes_linked_effects(self):
        s=self.game();h=hero(s,'a','zero');target=hero(s,'a','nanali')
        self.equip(s,'zero','Z08');old=weapon_source(h)
        add_effect(target,'linked','attack.panel',2,source_entity_id=h['entity_id'],source_key=old,
                   clear_on=('down','source_removed'))
        add_atk_buff(target,1,source_entity_id=h['entity_id'],source_key=old)
        self.equip(s,'zero','Z08')
        self.assertNotEqual(old,weapon_source(h))
        self.assertEqual(attack_value(target,s,'a'),3)
        self.assertNotIn('source_card_entity_id',public_equipment(h))

    def test_knockdown_removes_weapon_and_temporary_buffs_but_keeps_growth(self):
        s=self.game();h=hero(s,'a','nanali');grow_permanently(h,2,2)
        self.equip(s,'nanali','N08');add_atk_buff(h,3)
        damage(s,'a','nanali',99)
        self.assertEqual((h['shape'],h['max_hp'],permanent_attack(h)),(None,7,4))
        self.assertNotIn('equipment',h)
        self.assertEqual(h['permanent_growth'],{'attack':2,'max_hp':2})
        self.assertFalse(h.get('effects'))

    def test_registered_weapon_hook_switches_immediately_and_retains_granted_growth(self):
        s=self.game();s['sides']['a']['front']='nanali';h=hero(s,'a','nanali')
        self.equip(s,'zero','Z08');key=weapon_source(hero(s,'a','zero'))
        c=EffectContext(s,'a','zero',{'side':'a'})
        fire(c,'on_turn_end',actor_only=True)
        self.assertEqual(attack_value(h,s,'a'),3)
        self.assertEqual(h['effects'][0]['source_key'],key)
        self.equip(s,'zero','Z07');h['hp']=1
        fire(c,'on_turn_end',actor_only=True)
        self.assertEqual(h['hp'],3);self.assertEqual(attack_value(h,s,'a'),3)
        s=json.loads(json.dumps(s));view=observe(s,'a',include_previews=False)
        shown=next(c for c in view['sides']['a']['characters'] if c['id']=='zero')
        self.assertEqual(shown['equipment']['card_id'],'Z07')


    def test_growth_while_equipped_keeps_missing_life_until_next_equip(self):
        s=self.game();h=hero(s,'a','nanali');self.equip(s,'nanali','N08');h['hp']=2
        grow_permanently(h,1,1)
        self.assertEqual((h['hp'],h['max_hp'],attack_value(h,s,'a')),(3,7,3))
        self.equip(s,'nanali','N07')
        self.assertEqual((h['hp'],h['max_hp'],attack_value(h,s,'a')),(6,6,3))

    def test_weapon_source_survives_json_roundtrip_without_private_card_id(self):
        s=self.game();h=hero(s,'a','zero');self.equip(s,'zero','Z08');source=weapon_source(h)
        s=json.loads(json.dumps(s));h=hero(s,'a','zero')
        for view in (observe(s,'a',include_previews=False),observe(s,'b',include_previews=False),capture_public_board(s)):
            shown=next(c for c in view['sides']['a']['characters'] if c['id']=='zero')
            self.assertEqual(shown['equipment']['equipment_id'],source)
            self.assertNotIn('source_card_entity_id',shown['equipment'])

    def test_ir_adapter_exports_derived_base_values_not_equipped_maximum(self):
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        from app.modules.card_game.rl.gpu_duel.catalog import SEAT_INDEX
        s=self.game();h=hero(s,'a','nanali');grow_permanently(h,1,2)
        self.equip(s,'nanali','N08')
        row=pack_python_row(s);seat=SEAT_INDEX['nanali']
        self.assertEqual(row[OFFSETS['ch_base_atk']+seat],3)
        self.assertEqual(row[OFFSETS['ch_base_max']+seat],7)
        self.assertEqual(row[OFFSETS['ch_max_hp']+seat],8)


    def test_stale_weapon_callback_does_not_bind_to_replacement_instance(self):
        from app.modules.card_game.content.duel_v2.registry import _kit_handlers
        s=self.game();s['sides']['a']['front']='nanali'
        c=self.equip(s,'zero','Z08')
        old=list(_kit_handlers(c,'on_turn_end',actor_only=True))[-1][1]
        self.equip(s,'zero','Z08')
        old(c)
        self.assertEqual(attack_value(hero(s,'a','nanali'),s,'a'),2)
        fire(c,'on_turn_end',actor_only=True)
        self.assertEqual(attack_value(hero(s,'a','nanali'),s,'a'),3)
