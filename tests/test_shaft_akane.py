"""User workbook + Tencent document, 2026-09-21. Estimates are not official data."""
import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis, DEFAULT_UNPUBLISHED_CHARACTERS

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

    def test_catalog_and_sources(self):
        c = load_shaft_catalog()
        actions = {a['name']: a for a in c['actions_by_character'][CID]}
        self.assertEqual(DEFAULT_UNPUBLISHED_CHARACTERS[CID], '明音凛')
        self.assertEqual(next(a['id'] for a in c['arcs'] if a['adaptation'] == '液态'), ARC)
        self.assertEqual(actions['a1']['hit_count'], 2)
        self.assertAlmostEqual(actions['a1']['multipliers']['atk'], .788)
        self.assertEqual(actions['a2']['hit_count'], 2)
        self.assertAlmostEqual(actions['a2']['multipliers']['atk'], 1.158)
        self.assertEqual(actions['a3']['hit_count'], 7)
        self.assertAlmostEqual(actions['a3']['multipliers']['atk'], 2.253)
        self.assertAlmostEqual(actions['a3']['energy_gain'], 5.625)
        self.assertEqual(actions['a3']['hit_profile']['cumulative'][3]['damage'], .825)
        self.assertEqual(actions['援护']['damage_type'], '援护')
        self.assertEqual(actions['闪反']['damage_type'], '闪反')
        self.assertAlmostEqual(actions['援护']['multipliers']['atk'], 1.29)
        self.assertAlmostEqual(actions['闪反']['multipliers']['atk'], .25)
        self.assertAlmostEqual(actions['q1']['multipliers']['atk'], 1.2)
        self.assertAlmostEqual(actions['q1']['stagger'], .15)
        self.assertAlmostEqual(actions['q2']['multipliers']['atk'], 10)
        self.assertAlmostEqual(actions['同调共振']['multipliers']['atk'], 8)
        self.assertFalse(any(a.get('source_row') in (31, 38) for a in actions.values()))

    def test_resource_cap_and_awakening_refund(self):
        r = self.sim([('e', 0), ('enhanced_e', 12), ('a3', 24)], nodes=range(1, 7))
        self.assertEqual(r['details'][0]['personal_resources_after']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_consumed']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_after']['弦能'], 10)
        self.assertEqual(r['details'][2]['personal_resources_after']['弦能'], 14)
        self.assertTrue(r['healing_events'])
        self.assertEqual(r['healing_events'][0]['healing'], 660)
        poor = self.sim([('enhanced_e', 0)])['details'][0]
        self.assertTrue(any('弦能' in w for w in poor['warnings']))
        self.assertTrue(any('泛音列或音域' in w for w in poor['warnings']))

    def test_zero_hit_and_partial_interruption(self):
        partial = self.payload([('e', 0), ('a3', 20)])
        partial['steps'][1].update(interrupted=True, interrupt_duration_ticks=2, interrupt_hit_count=3)
        partial_result = simulate_shaft_axis(partial)['result']
        before = partial_result['details'][0]['personal_resources_after']['弦能']
        self.assertAlmostEqual(partial_result['details'][1]['personal_resources_after']['弦能'],
                               before + 4 * .825 / 2.253)
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

    def test_modes_exclude_each_other_and_expire(self):
        r = self.sim([('e', 0), ('a1', 12), ('hold_e', 170), ('a1', 184), ('a1', 440)], nodes=[2,5])
        single, area, expired = [r['details'][i] for i in [1,3,4]]
        self.assertAlmostEqual(single['formula_parts']['base_multiplier_factor'], 1.5)
        self.assertAlmostEqual(area['formula_parts']['base_multiplier_factor'], 1)
        self.assertAlmostEqual(area['formula_parts']['dmg_bonus'] - expired['formula_parts']['dmg_bonus'], .725)
        self.assertNotIn('character_akane_transmit_single', {b['rule_id'] for b in area['applied_buffs']})
        self.assertFalse(any('character_akane_transmit' in b['rule_id'] for b in expired['applied_buffs']))

    def test_shared_cd_and_enhanced_does_not_refresh_mode(self):
        r = self.sim([('e',0),('hold_e',20),('enhanced_e',40),('enhanced_e',280)], nodes=range(1,7))
        self.assertTrue(any('CD' in w for w in r['details'][1]['warnings']))
        self.assertFalse(any('CD' in w for w in r['details'][2]['warnings']))
        self.assertTrue(any('泛音列或音域' in w for w in r['details'][3]['warnings']))

    def test_q2_is_manual_during_the_second_half(self):
        early = self.sim([('q1', 0), ('q2', 50)])
        self.assertTrue(any('不能使用' in warning for warning in early['details'][1]['warnings']))
        late = self.sim([('q1', 0), ('q2', 110)])
        q2 = late['details'][1]
        self.assertFalse(any('不能使用' in warning or '华彩连段' in warning for warning in q2['warnings']))
        self.assertAlmostEqual(q2['formula_parts']['raw_base'] / q2['panel']['atk'], 10)
        self.assertAlmostEqual(q2['stagger_amount'], 1.5)
        self.assertAlmostEqual(late['details'][0]['stagger_amount'], .15)
        self.assertAlmostEqual(late['summary']['character_damage'], sum(d['direct_damage'] for d in late['details']))
        missing = self.sim([('q2', 0)])['details'][0]
        self.assertTrue(any('华彩连段' in warning for warning in missing['warnings']))

    def test_arc_stacks_expiry_and_refinements(self):
        for level in [1,5]:
            p=self.payload([('e',0),('a1',12),('a2',18),('a1',30),('a1',36),('enhanced_e',45),('a1',210)],arc=ARC)
            p['team'][0]['arc_refinement']=level
            r=simulate_shaft_axis(p)['result'];d=r['details'][5]
            b=next(b for b in d['applied_buffs'] if b['rule_id']=='arc_akane_skill')
            self.assertEqual(b['stack_count'],4)
            self.assertAlmostEqual(b['effects']['skill_dmg'],.28 if level==1 else .56)
            self.assertIn('arc_akane_element',{b['rule_id'] for b in d['applied_buffs']})
            self.assertNotIn('arc_akane_element',{b['rule_id'] for b in r['details'][-1]['applied_buffs']})

    def test_delay_unlock_pulses_remaining_time_and_mode_switch(self):
        p=self.payload([('e',30),('hold_e',100),('a1',210)])
        p['team'].append(dict(slot=1,character_id='char_dd034941ef',arc_id='',cartridge_id=''))
        p['steps'][:0]=[dict(id='front',slot=1,action_id='action_none_dd034941ef',start_tick=0),
                       dict(id='support',slot=0,action_id='action_akane_support',start_tick=10)]
        r=simulate_shaft_axis(p)['result']
        hits=[e for e in r['periodic_damage_events'] if e['reaction']=='绝对音准']
        self.assertEqual(hits[0]['tick'],60)
        self.assertAlmostEqual(hits[0]['remaining_seconds'],1.3)
        self.assertAlmostEqual(hits[0]['atk_multiplier'],2.52)
        ticks=[e['start_tick'] for e in r['reaction_effects']]
        self.assertNotIn(120,ticks)  # Old single mode cadence was cancelled.
        self.assertIn(130,ticks)
        self.assertEqual(len(ticks),len(set(ticks)))
        no_unlock=self.sim([('e',0),('a1',200)])
        self.assertFalse(no_unlock['reaction_effects'])

    def test_single_mode_enhanced_attack_is_higher_than_area_mode(self):
        single = self.sim([('e', 0), ('enhanced_e', 20)])['details'][1]
        area = self.sim([('hold_e', 0), ('enhanced_e', 20)])['details'][1]
        self.assertAlmostEqual(single['formula_parts']['base'] / single['panel']['atk'], 10)
        self.assertAlmostEqual(area['formula_parts']['base'] / area['panel']['atk'], 8)
