"""Current Elyms and signature-arc mechanics, including independent resonance toggles."""
import unittest
import copy
import json
import subprocess
from pathlib import Path
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis

CID='char_elyms'
ARC='arc_elyms_waiting_ringtone'
class ElymsAdjustmentsTest(unittest.TestCase):
    def result(self, names, nodes=(), resonances=None, arc='', refine=1, loop=False, others=()):
        member=dict(slot=0,character_id=CID,arc_id=arc,arc_refinement=refine,cartridge_id='',awakening_nodes=list(nodes))
        if resonances is not None:member['awakening_resonances']=resonances
        axis=dict(team=[member,*others],steps=[dict(id=str(i),slot=slot,action_id=action if action.startswith('action_') else 'action_elyms_'+action,start_tick=tick)
          for i,(slot,action,tick) in enumerate(names)],options=dict(loop_enabled=loop))
        return simulate_shaft_axis(axis)['result']
    def own(self, names, **kwargs):return self.result([(0,name,tick) for name,tick in names],**kwargs)
    def effects(self, d):return {b['rule_id']:b['effects'] for b in d['applied_buffs']}

    def test_b_q_bonus_and_three_resonance_are_separate(self):
        for nodes,resonances,expected_signal,qfactor in [([2],[],1,1.8),([], [3],2,2),([2],[3],2,2.3)]:
            with self.subTest(nodes=nodes,resonances=resonances):
                r=self.own([('skill2',0),('ultraskilla1',10)],nodes=nodes,resonances=resonances)
                self.assertEqual(r['details'][0]['personal_resources_after']['信号等级'],expected_signal)
                self.assertAlmostEqual(r['details'][1]['formula_parts']['base_multiplier_factor'],qfactor)

    def test_c_grants_three_caps_four_and_no_longer_grants_on_reactions(self):
        r=self.own([('skill2',0),('goodbye',10),('goodbye',20)],nodes=[3],resonances=[])
        c=next(b for b in r['details'][1]['triggered_buffs'] if b['rule_id']=='character_elyms_illusion_c')
        self.assertEqual(c['stack_count'],3)
        self.assertEqual(r['details'][2]['formula_parts']['base_multiplier_factor'],7)
        r=self.own([('skill2',0),('goodbye',10)],nodes=[3],resonances=[3])
        self.assertEqual(r['details'][1]['formula_parts']['base_multiplier_factor'],9)
        rules=load_shaft_catalog()['buffs']
        self.assertFalse(any(b['id']=='character_elyms_illusion_reaction' for b in rules))
        self.assertTrue(all(b['stacking']['max_stacks']==4 for b in rules if b.get('stacking',{}).get('key')=='character_elyms_illusion'))

    def test_a_and_f_clear_accumulated_consumption_and_old_bonus_on_cast(self):
        for node,rule,effect,expected in [(1,'character_elyms_a','crit_dmg',.1),(6,'character_elyms_f','follow_dmg',.25)]:
            r=self.own([('skill2',0),('goodbye',10),('ultraskilla1',20),('melee1',30),('ultraskilla1',40),('melee1',50)],nodes=[node],resonances=[])
            self.assertAlmostEqual(self.effects(r['details'][3])[rule][effect],expected)
            self.assertNotIn(rule,self.effects(r['details'][4]))
            self.assertEqual(self.effects(r['details'][5])[rule],{})

    def test_d_and_six_resonance_are_permanent_before_q(self):
        base=self.own([('melee1',0),('melee1',250)],resonances=[])
        d=self.own([('melee1',0),('melee1',250)],nodes=[4],resonances=[])
        for a,b in zip(base['details'],d['details']):self.assertAlmostEqual(b['panel']['crit_dmg']-a['panel']['crit_dmg'],.2)
        r=self.own([('xietongskilla1',0)],resonances=[6])
        self.assertEqual(self.effects(r['details'][0])['character_elyms_six'], {'res_down': .12})
        self.assertEqual(r['details'][0]['formula_parts']['base_multiplier_factor'], 1)

    def test_entry_grants_all_arc_effects_for_twenty_seconds_and_e_buffs_team(self):
        teammate=dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id='')
        for refine in range(1,6):
            r=self.result([(0,'melee1',0),(1,'action_982c67944f',10),(0,'melee1',210),(0,'skill1',230),(1,'action_982c67944f',250),(0,'melee1',280)],arc=ARC,refine=refine,others=[teammate])
            initial=r['initial_buffs']
            self.assertEqual({b['rule_id'] for b in initial},{'arc_elyms_team_def','arc_elyms_self_def','arc_elyms_light_pen'})
            self.assertTrue(all(b['trigger_tick']==0 and b['end_tick']==200 for b in initial))
            pen=.08+.03*(refine-1)
            self.assertAlmostEqual(self.effects(r['details'][0])['arc_elyms_light_pen']['res_down_光'],pen)
            self.assertAlmostEqual(self.effects(r['details'][1])['arc_elyms_light_pen']['res_down_光'],pen)
            self.assertAlmostEqual(r['details'][1]['formula_parts']['res_down'],pen)
            self.assertIn('arc_elyms_team_def',self.effects(r['details'][1]))
            self.assertNotIn('arc_elyms_team_def',self.effects(r['details'][2]))
            self.assertAlmostEqual(self.effects(r['details'][4])['arc_elyms_light_pen']['res_down_光'],pen)
            self.assertAlmostEqual(r['details'][4]['formula_parts']['res_down'],pen)
            self.assertAlmostEqual(self.effects(r['details'][5])['arc_elyms_light_pen']['res_down_光'],pen)
        r=self.own([('melee1',0),('melee1',250)],arc=ARC,loop=True)
        self.assertFalse(r['initial_buffs'])
        self.assertNotIn('arc_elyms_team_def',self.effects(r['details'][0]))

    def test_team_light_resistance_ignore_is_unique_and_uses_highest_refinement(self):
        teammate=dict(slot=1,character_id='char_dd034941ef',arc_id=ARC,arc_refinement=5,cartridge_id='')
        result=self.result([(0,'melee1',0),(1,'action_982c67944f',10)],arc=ARC,refine=1,others=[teammate])
        for detail in result['details']:
            buffs=[b for b in detail['applied_buffs'] if b['rule_id']=='arc_elyms_light_pen']
            self.assertEqual(len(buffs),1)
            self.assertAlmostEqual(buffs[0]['effects']['res_down_光'],.2)
            self.assertAlmostEqual(detail['formula_parts']['res_down'],.2)


class ElymsTeamDamageBuffsTest(unittest.TestCase):
    def simulate(self, axis, catalog):
        root = Path(__file__).resolve().parents[1]
        output = subprocess.check_output(
            ['node', str(root / 'scripts/shaft_compute.js')],
            input=json.dumps({'axis': axis, 'catalog': catalog}), text=True, timeout=15)
        return json.loads(output)['result']

    def catalog_with_probes(self):
        catalog = copy.deepcopy(load_shaft_catalog())
        for element in ['光', '灵', '咒', '暗', '魂', '相', '心灵']:
            catalog['actions'].append(dict(
                id='probe_' + element, character_id='char_dd034941ef', name='属性伤害',
                action_type='普攻', damage_type='普攻', damage_element=element,
                multipliers={'atk': 1}, hit_count=1, duration_ticks=0,
                is_background_damage=True, skill_level_category='basic'))
        catalog['actions'].append(dict(
            id='probe_follow', character_id='char_dd034941ef', name='追加攻击',
            action_type='被动', damage_type='被动', tags=['追加攻击'],
            multipliers={'atk': 1}, hit_count=1, duration_ticks=0,
            is_background_damage=True, skill_level_category='none'))
        return catalog

    def axis(self, steps, resonances=(), loop=False):
        return dict(team=[dict(slot=0, character_id=CID, arc_id='', cartridge_id='',
                               awakening_nodes=[], awakening_resonances=list(resonances)),
                          dict(slot=1, character_id='char_dd034941ef', arc_id='', cartridge_id='')],
                    steps=[dict(id=str(i), slot=slot, action_id=aid, start_tick=tick)
                           for i, (slot, aid, tick) in enumerate(steps)],
                    options={'loop_enabled': loop})

    def test_six_resonance_all_attributes_on_teammate_without_q(self):
        catalog = self.catalog_with_probes()
        steps = [(1, 'probe_' + element, i*10)
                 for i, element in enumerate(['光', '灵', '咒', '暗', '魂', '相', '心灵'])]
        steps += [(1, 'action_none_dd034941ef', 100)]
        base = self.simulate(self.axis(steps), catalog)
        for loop in [False, True]:
            with self.subTest(loop=loop):
                result = self.simulate(self.axis(steps, [6], loop), catalog)
                for before, after in zip(base['details'][:7], result['details'][:7]):
                    self.assertAlmostEqual(after['formula_parts']['res_down'] - before['formula_parts']['res_down'], .12)
                    self.assertGreater(after['direct_damage'], before['direct_damage'])
                    buff = next(b for b in after['applied_buffs'] if b['rule_id']=='character_elyms_six')
                    self.assertFalse(buff['display_as_line'])
                    self.assertEqual(buff['effects'], {'res_down': .12})

    def test_both_q_forms_grant_timed_team_follow_bonus_and_preserve_attack(self):
        catalog = self.catalog_with_probes()
        control = copy.deepcopy(catalog)
        next(b for b in control['buffs'] if b['id']=='character_elyms_q_atk')['effects'] = {}
        for form, follow in [('ultraskilla1', 'xietongskilla1'), ('ultraskillb1', 'xietongskillb1')]:
            with self.subTest(form=form):
                steps = [(0, 'action_elyms_' + form, 0), (1, 'probe_光', 10),
                         (1, 'probe_follow', 20), (0, 'action_elyms_' + follow, 30),
                         (1, 'probe_follow', 204), (1, 'probe_follow', 205),
                         (1, 'action_none_dd034941ef', 230)]
                axis = self.axis(steps)
                before = self.simulate(axis, control)
                result = self.simulate(axis, catalog)
                buff = next(b for b in result['details'][0]['triggered_buffs'] if b['rule_id']=='character_elyms_q_atk')
                self.assertEqual(buff['end_tick'] - buff['trigger_tick'], 200)
                self.assertAlmostEqual(buff['effects']['flat_atk'], 132)
                self.assertAlmostEqual(buff['effects']['follow_dmg'], .1)
                for i in [2, 3, 4]:
                    a, b = before['details'][i], result['details'][i]
                    self.assertAlmostEqual(b['formula_parts']['dmg_bonus'] - a['formula_parts']['dmg_bonus'], .1)
                    self.assertAlmostEqual(b['panel']['atk'], a['panel']['atk'])
                    self.assertGreater(b['direct_damage'], a['direct_damage'])
                for i in [0, 1, 5]:
                    self.assertAlmostEqual(result['details'][i]['direct_damage'], before['details'][i]['direct_damage'])
                self.assertEqual(result['details'][5]['start_tick'], buff['end_tick'])

    def test_q_follow_window_refreshes_without_stacking_and_carries_in_loop(self):
        catalog = self.catalog_with_probes()
        axis = self.axis([(0, 'action_elyms_ultraskilla1', 0),
                          (0, 'action_elyms_ultraskilla1', 100),
                          (1, 'probe_follow', 110), (1, 'probe_follow', 305),
                          (1, 'action_none_dd034941ef', 320)])
        result = self.simulate(axis, catalog)
        renewed = next(b for b in result['details'][1]['triggered_buffs'] if b['rule_id']=='character_elyms_q_atk')
        buffs = [b for b in result['details'][2]['applied_buffs'] if b['rule_id']=='character_elyms_q_atk']
        self.assertEqual(len(buffs), 1)
        self.assertEqual(buffs[0]['effects']['follow_dmg'], .1)
        self.assertEqual(renewed['end_tick'] - renewed['trigger_tick'], 200)
        self.assertFalse(any(b['rule_id']=='character_elyms_q_atk' for b in result['details'][3]['applied_buffs']))
        loop = self.axis([(1, 'probe_follow', 0), (0, 'action_elyms_ultraskilla1', 50),
                          (1, 'action_none_dd034941ef', 100)], loop=True)
        first = self.simulate(loop, catalog)['details'][0]
        buff = next(b for b in first['applied_buffs'] if b['rule_id']=='character_elyms_q_atk')
        self.assertEqual(buff['effects']['follow_dmg'], .1)
