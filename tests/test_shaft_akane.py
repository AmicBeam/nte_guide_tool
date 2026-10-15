"""NTEData resources + Oct 11 user notice and latest Tencent bracketed multipliers (Oct 12)."""
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.domain.legacy_actions import migrate_lingke_joint_steps
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis, DEFAULT_UNPUBLISHED_CHARACTERS

CID = 'char_akane'
ARC = 'arc_akane_moon_antipode'


class AkaneTest(unittest.TestCase):
    def payload(self, steps, nodes=(), arc='', **options):
        return {'team': [dict(slot=0, character_id=CID, arc_id=arc, cartridge_id='',
                             awakening_nodes=list(nodes),
                             skill_levels=dict(basic=1, skill=1, ultimate=1, support=1))],
                'steps': [dict(id=str(i), slot=0, action_id='action_akane_'+name, start_tick=tick)
                          for i, (name, tick) in enumerate(steps)], 'options': options}

    def sim(self, steps, **kwargs):
        return simulate_shaft_axis(self.payload(steps, **kwargs))['result']

    def test_user_supplied_bond_bonus_adds_eight_percent_crit_damage(self):
        # User 2026-10-15; this bonus is not Nanoka/NTEData-verified.
        character = next(c for c in load_shaft_catalog()['characters'] if c['id'] == CID)
        self.assertEqual(character['bond_bonus'], {'label': '8暴伤', 'modifiers': {'crit_dmg': .08}})
        panels = []
        for enabled in (False, True):
            axis = self.payload([('a1', 0)])
            axis['team'][0]['bond_full'] = enabled
            normalized = normalize_axis_payload(axis)
            self.assertEqual(normalized['team'][0]['bond_full'], enabled)
            self.assertEqual(normalized['character_builds'][CID]['bond_full'], enabled)
            panels.append(simulate_shaft_axis(axis)['result']['details'][0]['panel'])
        self.assertAlmostEqual(panels[1]['crit_dmg'] - panels[0]['crit_dmg'], .08)
        self.assertEqual(panels[1]['crit_rate'], panels[0]['crit_rate'])
        self.assertEqual(panels[1]['atk'], panels[0]['atk'])

    def test_catalog_and_sources(self):
        c = load_shaft_catalog()
        actions = {a['name']: a for a in c['actions_by_character'][CID]}
        self.assertEqual(DEFAULT_UNPUBLISHED_CHARACTERS[CID], '明音凛')
        self.assertEqual(next(a['id'] for a in c['arcs'] if a['adaptation'] == '液态'), ARC)
        self.assertEqual(actions['a1']['hit_count'], 2)
        self.assertAlmostEqual(actions['a1']['multipliers']['atk'], .298)
        self.assertEqual(actions['a2']['hit_count'], 2)
        self.assertAlmostEqual(actions['a2']['multipliers']['atk'], .438)
        self.assertEqual(actions['a3']['hit_count'], 7)
        self.assertAlmostEqual(actions['a3']['multipliers']['atk'], .849)
        self.assertAlmostEqual(actions['a3']['energy_gain'], 5.442)
        self.assertEqual(actions['a3']['hit_profile']['cumulative'][3]['damage'], .309)
        self.assertEqual(actions['援护']['damage_type'], '援护')
        self.assertEqual(actions['闪反']['damage_type'], '闪反')
        self.assertAlmostEqual(actions['援护']['multipliers']['atk'], 2)
        self.assertAlmostEqual(actions['援护']['energy_gain'], 6.5)
        self.assertAlmostEqual(actions['闪反']['multipliers']['atk'], 1.375)
        self.assertAlmostEqual(actions['q1']['multipliers']['atk'], 6.48)
        self.assertEqual(actions['q1']['hit_count'],10)
        self.assertEqual(actions['q1']['cooldown_ticks'],200)
        self.assertEqual(actions['q1']['energy_cost'], 100)
        self.assertAlmostEqual(actions['q1']['stagger'], 1.5)
        self.assertAlmostEqual(actions['q2']['multipliers']['atk'], 9.18)
        self.assertAlmostEqual(actions['短强E']['multipliers']['atk'], 11.25)
        self.assertAlmostEqual(actions['长强E']['multipliers']['atk'], 10.125)
        self.assertEqual(actions['长强E']['duration_ticks'],15)
        self.assertNotIn('同调共振',actions)
        c_awaken=c['awakenings']['明音凛'][2]
        self.assertIn('音域范围扩大30%',c_awaken['description'])
        self.assertFalse(any(a.get('source_row') in (31, 38) for a in actions.values()))

    def test_resource_cap_and_awakening_refund(self):
        r = self.sim([('e', 0), ('enhanced_e', 12), ('a3', 24)], nodes=range(1, 7))
        self.assertEqual(r['details'][0]['personal_resources_after']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_consumed']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_after']['弦能'], 10)
        self.assertEqual(r['details'][2]['personal_resources_after']['弦能'], 12)
        self.assertTrue(r['healing_events'])
        self.assertEqual(r['healing_events'][0]['healing'], 636*10)
        poor = self.sim([('enhanced_e', 0)])['details'][0]
        self.assertTrue(any('弦能' in w for w in poor['warnings']))
        self.assertTrue(any('音域' in w for w in poor['warnings']))

    def test_manual_enhanced_variants_keep_own_multiplier_duration_and_buffs(self):
        for form,name,base,duration in [('e','short_enhanced_e',11.25,7),
                                        ('hold_e','long_enhanced_e',10.125,15)]:
            with self.subTest(action=name):
                p=self.payload([(form,0),(name,20),('a1',50)],nodes=range(1,7))
                rows=simulate_shaft_axis(p)['result']['details']
                d=rows[1]
                self.assertEqual(d['action_id'],'action_akane_'+name)
                self.assertEqual(d['duration_ticks'],duration)
                self.assertAlmostEqual(d['formula_parts']['raw_base']/d['panel']['atk'],base)
                self.assertAlmostEqual(d['formula_parts']['post_skill_level_base']/d['panel']['atk'],8)
                self.assertEqual(d['personal_resources_consumed']['弦能'],30)
                self.assertEqual(d['personal_resources_after']['弦能'],10)
                self.assertTrue({'character_akane_a','character_akane_d','character_akane_six'} <= {
                    b['rule_id'] for b in rows[2]['applied_buffs']})
                wrong_form='hold_e' if form=='e' else 'e'
                wrong=self.sim([(wrong_form,0),(name,20)])['details'][1]
                self.assertEqual(wrong['action_id'],d['action_id'])
                required='泛音列' if form=='e' else '音域'
                self.assertTrue(any(required in w for w in wrong['warnings']))
                self.assertAlmostEqual(wrong['formula_parts']['raw_base']/wrong['panel']['atk'],base)

    def test_retired_enhanced_migration_is_ordered_idempotent_and_matches_browser(self):
        catalog=load_shaft_catalog()
        axis=self.payload([('enhanced_e',200),('e',0),('enhanced_e',20),
                           ('hold_e',160),('short_enhanced_e',220)])
        axis['steps'][0].update(interrupted=True,interrupt_duration_ticks=5,interrupt_hit_count=1)
        original=deepcopy(axis['steps'])
        expected=migrate_lingke_joint_steps(original,axis['team'],catalog)
        self.assertEqual(axis['steps'],original)
        self.assertEqual(expected[0]['action_id'],'action_akane_long_enhanced_e')
        self.assertEqual(expected[2]['action_id'],'action_akane_short_enhanced_e')
        self.assertEqual(expected[4]['action_id'],'action_akane_short_enhanced_e')
        self.assertTrue(expected[0]['interrupted'])
        for before,after in zip(original,expected):
            self.assertEqual(before['id'],after['id'])
            self.assertEqual(before['start_tick'],after['start_tick'])
        self.assertEqual(migrate_lingke_joint_steps(expected,axis['team'],catalog),expected)
        engine=Path(__file__).resolve().parents[1]/'app/modules/shaft/static/js/shaft_engine.js'
        script=('const x=JSON.parse(require("fs").readFileSync(0,"utf8"));'
                'const e=require('+json.dumps(str(engine))+');'
                'console.log(JSON.stringify(e.migrateLingkeJointSteps(x.steps,x.team,x.catalog)));')
        output=subprocess.run(['node','-e',script],input=json.dumps({
            'steps':original,'team':axis['team'],'catalog':catalog}),text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(output.stdout),expected)
        fallback=normalize_axis_payload(self.payload([('enhanced_e',0)]))
        self.assertEqual(fallback['steps'][0]['action_id'],'action_akane_long_enhanced_e')

    def test_unpacked_resource_totals_and_one_hit_hold_e(self):
        # SkillDamageData_Akane.xlsx / SkillDamage rows 6, 8, 9, 10, 14.
        cases={'a4':(3.439,5.847,.697), 'counter':(2.5,4.2,2.5),
               'e':(3.6,6,.5), 'hold_e':(4.9,8.2,1), 'q1':(0,0,1.5)}
        for name,values in cases.items():
            with self.subTest(action=name):
                d=self.sim([(name,0)])['details'][0]
                for field,value in zip(('energy_gain','harmony','stagger_amount'),values):
                    self.assertAlmostEqual(d[field],value)
        basics=self.sim([('a1',0),('a2',10),('a3',20),('a4',40)])['details']
        self.assertAlmostEqual(sum(d['harmony'] for d in basics),23.999)
        p=self.payload([('hold_e',0)])
        p['steps'][0].update(interrupted=True,interrupt_duration_ticks=5,interrupt_hit_count=1)
        d=simulate_shaft_axis(p)['result']['details'][0]
        self.assertAlmostEqual(d['harmony'],4.1)
        self.assertAlmostEqual(d['energy_gain'],2.45)
        self.assertAlmostEqual(d['stagger_amount'],.5)
        self.assertEqual(d['personal_resources_after']['弦能'],6)

    def test_zero_hit_and_partial_interruption(self):
        partial = self.payload([('e', 0), ('a3', 20)])
        partial['steps'][1].update(interrupted=True, interrupt_duration_ticks=2, interrupt_hit_count=3)
        partial_result = simulate_shaft_axis(partial)['result']
        before = partial_result['details'][0]['personal_resources_after']['弦能']
        self.assertEqual(partial_result['details'][1]['personal_resources_after']['弦能'],before + 2)
        detail = partial_result['details'][1]
        self.assertAlmostEqual(detail['energy_gain'], 2.046)
        self.assertAlmostEqual(detail['harmony'], 3.42)
        self.assertAlmostEqual(detail['stagger_amount'], .414)
        empty = self.payload([('e', 0), ('a3', 20)])
        empty['steps'][1].update(interrupted=True, interrupt_duration_ticks=2, interrupt_hit_count=0)
        empty_result = simulate_shaft_axis(empty)['result']
        self.assertEqual(empty_result['details'][1]['personal_resources_after']['弦能'],
                         empty_result['details'][0]['personal_resources_after']['弦能'])
        enhanced = self.payload([('e', 0), ('enhanced_e', 20)], nodes=range(1, 7))
        enhanced['steps'][1].update(interrupted=True, interrupt_duration_ticks=2, interrupt_hit_count=0)
        enhanced_result = simulate_shaft_axis(enhanced)['result']
        self.assertEqual(enhanced_result['details'][1]['personal_resources_after']['弦能'], 30)
        self.assertFalse(enhanced_result['healing_events'])
        self.assertEqual(enhanced_result['details'][1]['direct_damage'],0)
        self.assertTrue({'character_akane_a', 'character_akane_d', 'character_akane_six'}.isdisjoint(
            {b['rule_id'] for b in enhanced_result['details'][1]['triggered_buffs']}))

    def test_modes_exclude_each_other_and_expire(self):
        r = self.sim([('e', 0), ('a1', 12), ('hold_e', 170), ('a1', 184), ('a1', 440)], nodes=[2,5])
        single, area, expired = [r['details'][i] for i in [1,3,4]]
        self.assertAlmostEqual(single['formula_parts']['base_multiplier_factor'], 1.64)
        self.assertAlmostEqual(area['formula_parts']['base_multiplier_factor'], 1)
        self.assertAlmostEqual(area['formula_parts']['dmg_bonus'] - expired['formula_parts']['dmg_bonus'], .9)
        self.assertNotIn('character_akane_transmit_single', {b['rule_id'] for b in area['applied_buffs']})
        self.assertFalse(any('character_akane_transmit' in b['rule_id'] for b in expired['applied_buffs']))

    def test_first_hit_gives_full_integer_string_energy_once(self):
        for name, hits, gain in [('a1',2,1),('a2',2,1),('a3',7,2),('hold_e',2,6)]:
            for kept in (0,1,hits):
                with self.subTest(action=name, kept=kept):
                    p=self.payload([(name,0)])
                    p['steps'][0].update(interrupted=True,interrupt_duration_ticks=1,interrupt_hit_count=kept)
                    d=simulate_shaft_axis(p)['result']['details'][0]
                    self.assertEqual(d['personal_resources_after']['弦能'],gain if kept else 0)
                    self.assertEqual(d['duration_ticks'],1)

    def test_initial_string_energy_is_integer_without_rounding_other_resources(self):
        p=self.payload([('a3',0)],loop_enabled=True,loop_initial_resources={
            CID:{'personal_resources':{'弦能':28.9}},
            'char_heiyu':{'personal_resources':{'恶意':7.25}}})
        p['team'].append(dict(slot=1,character_id='char_heiyu',arc_id='',cartridge_id=''))
        normalized=normalize_axis_payload(p)
        configured=normalized['options']['loop_initial_resources']
        self.assertEqual(configured[CID]['personal_resources']['弦能'],28)
        self.assertEqual(configured['char_heiyu']['personal_resources']['恶意'],7.25)
        engine=Path(__file__).resolve().parents[1]/'app/modules/shaft/static/js/shaft_engine.js'
        script=('const x=JSON.parse(require("fs").readFileSync(0,"utf8"));'
                'const e=require('+json.dumps(str(engine))+');'
                'console.log(JSON.stringify(e.simulateAxis(x.axis,x.catalog)));')
        output=subprocess.run(['node','-e',script],input=json.dumps({
            'axis':p,'catalog':load_shaft_catalog()}),text=True,capture_output=True,check=True)
        result=json.loads(output.stdout)
        resources={r['slot']:r['initial_personal_resources'] for r in result['resources_by_slot']}
        self.assertEqual(resources[0]['弦能'],28)
        self.assertEqual(resources[1]['恶意'],7.25)
        self.assertEqual(result['details'][0]['personal_resources_after']['弦能'],30)

    def test_shared_cd_and_enhanced_does_not_refresh_mode(self):
        r = self.sim([('e',0),('hold_e',20),('enhanced_e',40),('enhanced_e',280)], nodes=range(1,7))
        self.assertTrue(any('CD' in w for w in r['details'][1]['warnings']))
        self.assertFalse(any('CD' in w for w in r['details'][2]['warnings']))
        self.assertTrue(any('音域' in w for w in r['details'][3]['warnings']))

    def test_enhanced_e_locks_teammate_switch_until_action_ends(self):
        catalog=load_shaft_catalog()
        teammate=next(c for c in catalog['characters'] if c['name']=='真红')
        entry=next(a for a in catalog['actions_by_character'][teammate['id']] if a['name']=='a1')
        for form, name, duration in [('e','short_enhanced_e',7),('hold_e','long_enhanced_e',15)]:
            for switch_tick in (23,20+duration):
                with self.subTest(form=form, switch_tick=switch_tick):
                    p=self.payload([(form,0),(name,20)],nodes=range(1,7))
                    p['team'].append(dict(slot=1,character_id=teammate['id'],arc_id='',cartridge_id=''))
                    p['steps'].append(dict(id='switch',slot=1,action_id=entry['id'],start_tick=switch_tick))
                    details=simulate_shaft_axis(p)['result']['details']
                    cast=details[1]
                    switched=next(d for d in details if d['step_id']=='switch')
                    self.assertEqual(cast['duration_ticks'],duration)
                    self.assertEqual(switched['start_tick'],cast['end_tick'])

    def test_q2_is_manual_without_cadenza_time_window(self):
        early = self.sim([('q1', 0), ('q2', 50)])
        self.assertFalse(any('不能使用' in warning or '华彩' in warning for warning in early['details'][1]['warnings']))
        late = self.sim([('q1', 0), ('q2', 110)])
        q2 = late['details'][1]
        self.assertFalse(any('不能使用' in warning or '华彩连段' in warning for warning in q2['warnings']))
        self.assertAlmostEqual(q2['formula_parts']['raw_base'] / q2['panel']['atk'], 9.18)
        self.assertAlmostEqual(q2['stagger_amount'], 1.5)
        self.assertAlmostEqual(late['details'][0]['stagger_amount'], 1.5)
        self.assertAlmostEqual(late['summary']['character_damage'], sum(d['direct_damage'] for d in late['details']))
        missing = self.sim([('q2', 0)])['details'][0]
        self.assertFalse(any('华彩连段' in warning for warning in missing['warnings']))
        after_window = self.sim([('q1', 0), ('q2', 400)])['details'][1]
        self.assertFalse(any('华彩' in warning for warning in after_window['warnings']))
        self.assertFalse(any('cadenza' in b['rule_id'] for b in late['details'][0]['applied_buffs']))

    def test_q2_repeated_per_q1_warns_and_new_q1_resets(self):
        details=self.sim([('q1',0),('q2',110),('q2',130),('q1',270),('q2',390)])['details']
        self.assertFalse(any('第二段只能' in w for w in details[1]['warnings']))
        self.assertTrue(any('第二段只能布置1次' in w for w in details[2]['warnings']))
        self.assertFalse(any('第二段只能' in w for w in details[4]['warnings']))
        self.assertGreater(details[2]['direct_damage'],0)  # Self-check does not block trial calculation.

    def test_manual_early_q2_counts_for_duplicate_self_check(self):
        details=self.sim([('q1',0),('q2',50),('q2',110)])['details']
        self.assertFalse(any('不能使用' in w for w in details[1]['warnings']))
        self.assertTrue(any('第二段只能' in w for w in details[2]['warnings']))

    def test_curtain_uses_type_three_attack_and_migrates_old_zero_bonus(self):
        p=self.payload([('a1',0)])
        p['team'][0].update(cartridge_id='cartridge_f4282fad3f',substat_counts={},
                            curtain_bonus={'value':0,'stat':'暴伤','passive_type':'type2'})
        axis=normalize_axis_payload(p)
        self.assertEqual(axis['team'][0]['curtain_bonus'],{'value':10,'stat':'攻击','passive_type':'type3'})
        panel=simulate_shaft_axis(p)['result']['build_panels_by_slot'][0]
        self.assertAlmostEqual(panel['zones']['atk_pct'],.4)
        self.assertEqual(panel['build_options']['curtain_bonus']['layers'],4)

    def test_arc_is_permanent_and_e_grants_both_timed_bonuses(self):
        base_crit=self.sim([('a1',0)])['details'][0]['panel']['crit_rate']
        for level, bonus in [(1,.28),(2,.35),(3,.42),(4,.49),(5,.56)]:
            p=self.payload([('a1',0),('e',10),('a1',25),('a1',170)],arc=ARC)
            p['team'][0]['arc_refinement']=level
            details=simulate_shaft_axis(p)['result']['details']
            for d in details:
                self.assertAlmostEqual(d['panel']['crit_rate']-base_crit,.15+.03*(level-1))
                b=next(b for b in d['applied_buffs'] if b['rule_id']=='arc_akane_skill')
                self.assertEqual(b['stack_count'],1)
                self.assertAlmostEqual(b['effects']['skill_dmg'],bonus)
            for i in (0,3):
                ids={b['rule_id'] for b in details[i]['applied_buffs']}
                self.assertNotIn('arc_akane_element',ids)
                self.assertNotIn('arc_akane_basic',ids)
            active={b['rule_id']:b for b in details[2]['applied_buffs']}
            self.assertAlmostEqual(active['arc_akane_basic']['effects']['basic_dmg'],bonus)
            self.assertAlmostEqual(active['arc_akane_element']['effects']['element_dmg_相'],.20+.05*(level-1))

    def test_arc_e_refreshes_the_fifteen_second_window(self):
        r=self.sim([('e',0),('e',160),('a1',300),('a1',311)],arc=ARC)
        active={b['rule_id'] for b in r['details'][2]['applied_buffs']}
        expired={b['rule_id'] for b in r['details'][3]['applied_buffs']}
        self.assertTrue({'arc_akane_basic','arc_akane_element'} <= active)
        self.assertTrue({'arc_akane_basic','arc_akane_element'}.isdisjoint(expired))

    def test_six_resonance_grants_sixty_four_crit_damage_then_expires(self):
        for resonances, bonus in [([],0),([6],.64)]:
            with self.subTest(resonances=resonances):
                p=self.payload([('e',0),('enhanced_e',12),('a1',24),('a1',280)],nodes=[1,2,3,5,6])
                p['team'][0]['awakening_resonances']=resonances
                during,after=simulate_shaft_axis(p)['result']['details'][2:]
                self.assertAlmostEqual(during['panel']['crit_dmg']-after['panel']['crit_dmg'],bonus)
                active={b['rule_id']:b for b in during['applied_buffs']}
                self.assertEqual('character_akane_six' in active,bool(resonances))
                if resonances:
                    buff=active['character_akane_six']
                    self.assertEqual(buff['end_tick']-buff['start_tick'],250)
                self.assertNotIn('character_akane_six',{b['rule_id'] for b in after['applied_buffs']})

    def test_talent_is_twenty_percent_attack_and_f_is_defense_ignore_for_twenty_seconds(self):
        # Oct 12 user notice lowers attack to 20%; F and the existing 20s window remain.
        for nodes, ignore in [((),0),([6],.16)]:
            details=self.sim([('q1',0),('a1',20),('a1',230)],nodes=nodes)['details']
            during,after=details[1:]
            self.assertAlmostEqual(during['panel']['atk']-after['panel']['atk'],636*.2)
            self.assertAlmostEqual(during['formula_parts']['dmg_bonus']-after['formula_parts']['dmg_bonus'],0)
            self.assertAlmostEqual(during['panel']['crit_rate'],after['panel']['crit_rate'])
            self.assertAlmostEqual(during['panel']['crit_dmg'],after['panel']['crit_dmg'])
            self.assertAlmostEqual(during['formula_parts']['def_ignore'],ignore)
            self.assertAlmostEqual(after['formula_parts']['def_ignore'],0)
            b=next(b for b in during['applied_buffs'] if b['rule_id']=='character_akane_talent2')
            self.assertEqual(b['effects'],{'atk_pct':.2})
            self.assertEqual(b['end_tick']-b['start_tick'],200)
            if nodes:
                f=next(b for b in during['applied_buffs'] if b['rule_id']=='character_akane_f')
                self.assertEqual(f['effects'],{'def_ignore':.16})
                self.assertEqual(f['end_tick']-f['start_tick'],200)

    def test_delay_unlock_pulses_remaining_time_and_mode_switch(self):
        p=self.payload([('e',30),('hold_e',100),('a1',210)])
        p['team'].append(dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''))
        p['steps'][:0]=[dict(id='front',slot=1,action_id='action_none_dd034941ef',start_tick=0),
                       dict(id='support',slot=0,action_id='action_akane_support',start_tick=10)]
        r=simulate_shaft_axis(p)['result']
        hits=[e for e in r['periodic_damage_events'] if e['reaction']=='绝对音准']
        self.assertEqual(hits[0]['tick'],60)
        self.assertAlmostEqual(hits[0]['remaining_seconds'],1.3)
        self.assertAlmostEqual(hits[0]['atk_multiplier'],1.89)
        ticks=[e['start_tick'] for e in r['reaction_effects']]
        self.assertNotIn(120,ticks)  # Old single mode cadence was cancelled.
        self.assertIn(130,ticks)
        self.assertEqual(len(ticks),len(set(ticks)))
        no_unlock=self.sim([('e',0),('a1',200)])
        self.assertFalse(no_unlock['reaction_effects'])

    def test_single_mode_enhanced_attack_is_higher_than_area_mode(self):
        single = self.sim([('e', 0), ('enhanced_e', 20)])['details'][1]
        area = self.sim([('hold_e', 0), ('enhanced_e', 20)])['details'][1]
        self.assertAlmostEqual(single['formula_parts']['base'] / single['panel']['atk'], 11.25)
        self.assertAlmostEqual(area['formula_parts']['base'] / area['panel']['atk'], 10.125)

    def test_new_c_is_additive_and_resonance_can_be_toggled_independently(self):
        # User notice: C +1000%, enhanced E +800%, Q2 +1000%, AFTER level scaling.
        for level in (1, 10):
            for form, enhanced_base in [('e', 11.25), ('hold_e', 10.125)]:
                for resonances in ([], [3]):
                    with self.subTest(level=level, form=form, resonances=resonances):
                        p=self.payload([(form,0),('enhanced_e',20),('q2',30)],nodes=[1,2,3])
                        p['team'][0].update(awakening_resonances=resonances,
                            skill_levels=dict(basic=level,skill=level,ultimate=level,support=1))
                        rows=simulate_shaft_axis(p)['result']['details']
                        fixed=[10 if form=='e' else 0, 8 if resonances else 0,
                               10 if resonances else 0]
                        for d, original, addition in zip(rows,
                                [3.6 if form=='e' else 3.24, enhanced_base, 9.18], fixed):
                            f=d['formula_parts'];atk=d['panel']['atk']
                            self.assertAlmostEqual(f['raw_base']/atk,original)
                            self.assertAlmostEqual(f['post_skill_level_base']/atk,addition)
                            self.assertAlmostEqual(f['skill_level_multiplier'],
                                1.08**(level-1+bool(resonances)))
                            self.assertAlmostEqual(f['base']/atk,
                                original*f['skill_level_multiplier']+addition)
                        if level==1 and resonances:
                            self.assertAlmostEqual(rows[1]['formula_parts']['skill_level_multiplier'],1.08)

    def test_final_awakening_multiplier_vanishes_on_zero_hit(self):
        for name in ('e', 'enhanced_e'):
            with self.subTest(action=name):
                p=self.payload([(name,0)],nodes=[1,2,3])
                p['team'][0].update(awakening_resonances=[3],
                    skill_levels=dict(basic=10,skill=10,ultimate=10,support=1))
                p['steps'][0].update(interrupted=True,interrupt_duration_ticks=1,
                                    interrupt_hit_count=0)
                d=simulate_shaft_axis(p)['result']['details'][0]
                self.assertEqual(d['formula_parts']['post_skill_level_base'],0)
                self.assertEqual(d['direct_damage'],0)

    def test_new_a_and_d_expire_and_b_is_phase_only_in_both_forms(self):
        for form in ['e','hold_e']:
            r=self.sim([(form,0),('enhanced_e',20),('a1',35),('a1',180)],nodes=[1,2,4])
            during,after=r['details'][2:]
            self.assertAlmostEqual(during['panel']['interrupt_resistance'],.6)
            self.assertAlmostEqual(after['panel']['interrupt_resistance'],0)
            self.assertAlmostEqual(during['panel']['def']-after['panel']['def'],921*.5)
            self.assertAlmostEqual(during['formula_parts']['def_ignore'],0)
            self.assertAlmostEqual(after['formula_parts']['def_ignore'],0)
            self.assertAlmostEqual(during['panel']['crit_rate'],after['panel']['crit_rate'])
            self.assertAlmostEqual(during['panel']['crit_dmg']-after['panel']['crit_dmg'],.3)
            d=next(x for x in during['applied_buffs'] if x['rule_id']=='character_akane_d')
            self.assertEqual(d['effects'],{'crit_dmg':.3})
            self.assertEqual(d['end_tick']-d['start_tick'],150)
            b=next(x for x in during['applied_buffs'] if x['rule_id'].startswith('character_akane_b_'))
            self.assertEqual(b['effects'],{'res_down_相':.12})
            a=next(x for x in during['applied_buffs'] if x['rule_id']=='character_akane_a')
            self.assertEqual(a['end_tick']-a['start_tick'],150)
            without_a=self.sim([(form,0),('enhanced_e',20),('a1',35)],nodes=[2,4])['details'][2]
            self.assertAlmostEqual(without_a['panel']['def'],after['panel']['def'])
            self.assertNotIn('character_akane_a',{x['rule_id'] for x in without_a['applied_buffs']})

    def test_counter_gets_transmission_and_passive_is_skill_damage(self):
        for form,factor,bonus in [('e',1.64,0),('hold_e',1,.9)]:
            rows=self.sim([(form,0),('counter',20),('counter',270)],nodes=[5])['details']
            d,expired=rows[1:]
            self.assertAlmostEqual(d['formula_parts']['base_multiplier_factor'],factor)
            self.assertAlmostEqual(d['formula_parts']['dmg_bonus']-expired['formula_parts']['dmg_bonus'],bonus)
        p=self.payload([('e',30),('a1',120)],arc=ARC)
        p['team'].append(dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''))
        p['steps'][:0]=[dict(id='front',slot=1,action_id='action_none_dd034941ef',start_tick=0),dict(id='support',slot=0,action_id='action_akane_support',start_tick=10)]
        events=simulate_shaft_axis(p)['result']['periodic_damage_events']
        hit=next(x for x in events if x['reaction']=='绝对音准')
        self.assertEqual(hit['damage_type'],'E')
        self.assertEqual(hit['formula_parts']['skill_level_category'],'skill')
        self.assertGreaterEqual(hit['formula_parts']['damage_bonus'],.28)

    def test_a_and_d_expire_at_fifteen_second_boundary_in_both_forms(self):
        # Oct 11 user values; the triggering hit remains at tick 20.
        for form, enhanced in [('e','short_enhanced_e'),('hold_e','long_enhanced_e')]:
            rows=self.sim([(form,0),(enhanced,20),('a1',169),('a1',170)],nodes=[1,4])['details']
            before,at_end=rows[2:]
            self.assertAlmostEqual(before['panel']['interrupt_resistance'],.6)
            self.assertAlmostEqual(at_end['panel']['interrupt_resistance'],0)
            self.assertAlmostEqual(before['panel']['crit_dmg']-at_end['panel']['crit_dmg'],.3)
            self.assertTrue({'character_akane_a','character_akane_d'}.isdisjoint(
                {b['rule_id'] for b in at_end['applied_buffs']}))

    def test_enhanced_e_eight_hundred_bonus_requires_three_not_six_resonance(self):
        for form, enhanced in [('e','short_enhanced_e'),('hold_e','long_enhanced_e')]:
            for resonances in ([],[3],[6],[3,6]):
                with self.subTest(form=form,resonances=resonances):
                    p=self.payload([(form,0),(enhanced,20)])
                    p['team'][0]['awakening_resonances']=resonances
                    d=simulate_shaft_axis(p)['result']['details'][1]
                    self.assertAlmostEqual(d['formula_parts']['post_skill_level_base']/d['panel']['atk'],
                                           8 if 3 in resonances else 0)

    def test_oct11_document_bracketed_multipliers_and_interrupted_damage(self):
        # Latest user document; long E's accidental ×2 is explicitly corrected to one hit.
        expected = {
            'a1': [.149,.149], 'a2': [.219,.219], 'a3': [.103]*3+[.135]*4,
            'a4': [4.247], 'e': [3.6], 'hold_e': [1.62,1.62],
            'short_enhanced_e': [11.25], 'long_enhanced_e': [10.125],
            'q1': [.648]*10, 'q2': [9.18],
        }
        catalog=load_shaft_catalog()
        for name,hits in expected.items():
            a=next(a for a in catalog['actions'] if a['id']=='action_akane_'+name)
            self.assertEqual(a['hit_count'],len(hits))
            self.assertAlmostEqual(a['multipliers']['atk'],sum(hits))
            if a['duration_ticks']==0:
                continue
            for kept in range(len(hits)+1):
                with self.subTest(action=name,kept=kept):
                    p=self.payload([(name,0)])
                    p['steps'][0].update(interrupted=True,interrupt_duration_ticks=1,interrupt_hit_count=kept)
                    d=simulate_shaft_axis(p)['result']['details'][0]
                    self.assertAlmostEqual(d['formula_parts']['raw_base']/d['panel']['atk'],sum(hits[:kept]))

    def test_latest_document_does_not_restore_old_passives_awakenings_or_durations(self):
        c=load_shaft_catalog()
        self.assertEqual(next(x for x in c['characters'] if x['id']==CID)['reaction_reapplication']['atk_multiplier'],1.5)
        buffs={b['id']:b for b in c['buffs']}
        self.assertEqual(buffs['character_akane_talent2']['effects'],{'atk_pct':.2})
        self.assertEqual(buffs['character_akane_d']['effects'],{'crit_dmg':.3})
        self.assertEqual(buffs['character_akane_f']['effects'],{'def_ignore':.16})
        self.assertEqual(buffs['character_akane_e_single']['effects'],{'base_multiplier_pct':.24})
        self.assertEqual([next(x for x in c['actions'] if x['id']=='action_akane_'+a)['duration_ticks'] for a in ('a1','a2','a3','a4')],[5,5,10,8])
