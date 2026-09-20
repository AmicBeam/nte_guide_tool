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
        self.assertEqual(actions['a2']['hit_count'], 2)
        self.assertAlmostEqual(actions['a2']['multipliers']['atk'], .764)
        self.assertEqual(actions['a3']['hit_count'], 7)
        self.assertAlmostEqual(actions['a3']['multipliers']['atk'], 1.49)
        self.assertAlmostEqual(actions['a3']['energy_gain'], 5.625)
        self.assertEqual(actions['a3']['hit_profile']['cumulative'][3]['damage'], .546)
        self.assertEqual(actions['援护']['damage_type'], '援护')
        self.assertEqual(actions['闪反']['damage_type'], '闪反')
        self.assertEqual(actions['援护']['multipliers']['atk'], 1)
        self.assertNotIn('q2', actions)  # Automatic, cannot double-count a hand-placed copy.
        self.assertFalse(any(a.get('source_row') in (31, 38) for a in actions.values()))

    def test_resource_cap_and_awakening_refund(self):
        r = self.sim([('e', 0), ('enhanced_e', 12), ('a3', 24)], nodes=range(1, 7))
        self.assertEqual(r['details'][0]['personal_resources_after']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_consumed']['弦能'], 30)
        self.assertEqual(r['details'][1]['personal_resources_after']['弦能'], 11)
        self.assertEqual(r['details'][2]['personal_resources_after']['弦能'], 18)
        self.assertTrue(r['healing_events'])
        self.assertEqual(r['healing_events'][0]['healing'], 660)
        poor = self.sim([('enhanced_e', 0)])['details'][0]
        self.assertTrue(any('弦能' in w for w in poor['warnings']))
        self.assertTrue(any('泛音列或音域' in w for w in poor['warnings']))

    def test_zero_hit_and_partial_interruption(self):
        for key, hits, expected in [('a3', 3, 3), ('a3', 0, 0), ('enhanced_e', 0, 30)]:
            p = self.payload([('e', 0), (key, 20)], nodes=range(1, 7) if key == 'enhanced_e' else [])
            p['steps'][1].update(interrupted=True, interrupt_duration_ticks=2, interrupt_hit_count=hits)
            r = simulate_shaft_axis(p)['result']
            before = r['details'][0]['personal_resources_after']['弦能']
            self.assertEqual(r['details'][1]['personal_resources_after']['弦能'],
                             expected if key == 'enhanced_e' else before + expected)
            if key == 'enhanced_e': self.assertFalse(r['healing_events'])

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

    def test_q2_delayed_dynamic_panel_and_outside_axis(self):
        short = self.sim([('q1',0)])
        long = self.sim([('q1',0),('a1',100)])
        q2, = [e for e in long['periodic_damage_events'] if e['reaction']=='q2']
        self.assertEqual(q2['tick'],80)
        self.assertEqual(q2['damage_type'],'Q')
        self.assertAlmostEqual(q2['formula_parts']['crit_rate'],.05)
        self.assertAlmostEqual(q2['formula_parts']['skill_level_multiplier'],1)
        self.assertAlmostEqual(short['summary']['character_damage'],sum(d['direct_damage'] for d in short['details']))
        self.assertAlmostEqual(long['summary']['character_damage'],sum(d['direct_damage'] for d in long['details'])+q2['damage'])
        self.assertAlmostEqual(long['summary']['total_stagger'],3.2)
        self.assertAlmostEqual(long['summary']['effective_stagger'],3.2)

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

    def test_loop_q2_inherits_remaining_delay(self):
        r=self.sim([('q1',10),('a1',55)],loop_enabled=True)
        hits=[e for e in r['periodic_damage_events'] if e['reaction']=='q2']
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['tick'],10+80-r['summary']['duration_ticks'])
        self.assertGreater(hits[0]['damage'],0)
