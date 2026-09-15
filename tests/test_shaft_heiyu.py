import unittest
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout
from app.errors import AppError

CID = 'char_heiyu'
ARC = 'arc_heiyu_crime_punishment'
DARK = 'char_c78f7a08d5'

class HeiyuTest(unittest.TestCase):
    def simulate(self, steps, others=(), **options):
        axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': '',
                          'skill_levels': dict(basic=1, skill=1, ultimate=1, support=1)}, *others],
                'steps': [dict(id=str(i), slot=slot, action_id=action, start_tick=tick)
                          for i, (slot, action, tick) in enumerate(steps)], 'options': options}
        return simulate_shaft_axis(axis)['result']

    def test_source_rows_and_missing_durations(self):
        catalog = load_shaft_catalog()
        actions = {a['name']: a for a in catalog['actions_by_character'][CID]}
        self.assertAlmostEqual(actions['a5']['multipliers']['atk'], 1.735)
        self.assertAlmostEqual(actions['a5']['energy_gain'], 5.785)
        self.assertAlmostEqual(actions['二a5']['harmony'], 14.142)
        self.assertEqual(actions['e']['harmony'], 100)
        self.assertEqual(actions['q1']['energy_cost'], 100)
        self.assertEqual(actions['q2']['energy_cost'], 0)
        self.assertEqual(actions['失乐鸟']['hit_count'], 5)
        self.assertEqual(actions['a1']['duration_ticks'], 0)
        self.assertFalse(actions['a1'].get('duration_implemented'))
        self.assertEqual(len([a for a in actions.values() if a.get('source_row') == 19]), 1)

    def test_kongmu_access_and_topology(self):
        self.assertNotIn(CID, [c['id'] for c in get_kongmu_catalog_payload()['characters']])
        c = next(c for c in get_kongmu_catalog_payload(include_test_characters=True)['characters'] if c['id'] == CID)
        self.assertEqual(c['element'], '魂')
        self.assertIn('暴击率增加8%', c['kongmu_passive']['text'])
        with self.assertRaises(AppError): plan_kongmu_layout(CID, 'attack')
        result = plan_kongmu_layout(CID, 'attack', include_test_characters=True)
        slots = result['character']['equip_slots']['slots']
        self.assertEqual(sum(v != -1 for row in slots for v in row), 20)
        self.assertEqual(slots[2], [0, 0, -1, 0, 0])

    def test_resource_cap_spend_and_q_exit_preserves(self):
        r = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                           (0, 'action_heiyu_form_z1', 20), (0, 'action_heiyu_q2', 30)])
        d = r['details']
        self.assertEqual(d[0]['personal_resources_after']['恶意'], 60)
        self.assertAlmostEqual(d[2]['personal_resources_after']['恶意'], 56.2)
        self.assertGreater(d[2]['resource_damage'], 0)
        self.assertEqual(d[3]['personal_resources_after'], d[2]['personal_resources_after'])
        self.assertEqual(d[3]['resource_damage'], 0)
        r = self.simulate([(0, 'action_heiyu_form_e', 0), (0, 'action_heiyu_form_z1', 10)])
        self.assertTrue(any('激活' in w for w in r['details'][0]['warnings']))
        self.assertEqual(r['details'][1]['resource_damage'], 0)
        self.assertTrue(any('恶意 不足' in w for w in r['details'][1]['warnings']))

    def test_form_pauses_off_field_and_q2_clears(self):
        other = {'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''}
        r = self.simulate([(0, 'action_heiyu_q1', 0), (1, 'action_none_c78f7a08d5', 20),
                           (0, 'action_heiyu_form_z1', 300), (0, 'action_heiyu_q2', 320),
                           (0, 'action_heiyu_form_z1', 330)], [other])
        self.assertFalse(any('二形态 状态' in w for w in r['details'][2]['warnings']))
        self.assertTrue(any('二形态 状态' in w for w in r['details'][4]['warnings']))
        r = self.simulate([(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_form_z1', 200)])
        self.assertTrue(any('二形态 状态' in w for w in r['details'][1]['warnings']))

    def test_assist_five_charges_and_expiration(self):
        r = self.simulate([(0, 'action_heiyu_e', 0)] + [(0, 'action_heiyu_assist', t) for t in [10, 30, 50, 70, 90, 110]])
        self.assertFalse(any('失乐鸟' in w for w in r['details'][5]['warnings']))
        self.assertTrue(any('失乐鸟' in w for w in r['details'][6]['warnings']))
        r = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_assist', 160)])
        self.assertTrue(any('失乐鸟' in w for w in r['details'][1]['warnings']))

    def test_dark_star_excludes_support_stagger_and_own_damage(self):
        other = {'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''}
        r = self.simulate([(1, 'action_none_c78f7a08d5', 0), (0, 'action_heiyu_support', 2),
                           (0, 'action_heiyu_e', 20), (1, 'action_dodge_c78f7a08d5', 120)], [other])
        extra = [e for e in r['reaction_damage_events'] if e.get('heiyu_accumulation')]
        self.assertEqual(len(extra), 1)
        self.assertAlmostEqual(extra[0]['damage'], r['details'][2]['direct_damage'] * .2)
        self.assertTrue(extra[0]['formula_parts']['excludes_stagger'])
        self.assertEqual(len(r['reaction_effects']), 2)
        self.assertTrue(r['reaction_effects'][1]['heiyu_extra'])

    def test_known_arc_refinement_only(self):
        axis = normalize_axis_payload({'team': [{'slot': 0, 'character_id': CID, 'arc_id': ARC,
            'arc_refinement': 5, 'cartridge_id': ''}], 'steps': []})
        self.assertEqual(axis['team'][0]['arc_refinement'], 1)
        r = simulate_shaft_axis({**axis, 'steps': [dict(id=str(i), slot=0,
            action_id='action_heiyu_'+a, start_tick=t) for i,(a,t) in enumerate([('q1',0),('q2',10),('q1',220)])]})['result']
        ember = [[b for b in d['applied_buffs'] if b['rule_id'] == 'arc_heiyu_ember'] for d in r['details']]
        self.assertEqual([bool(x) for x in ember], [False, True, False])

if __name__ == '__main__': unittest.main()
