import unittest
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout
from app.errors import AppError

CID = 'char_heiyu'
ARC = 'arc_heiyu_crime_punishment'
DARK = 'char_c78f7a08d5'

class HeiyuTest(unittest.TestCase):
    def simulate(self, steps, others=(), awakening_nodes=(), **options):
        axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': list(awakening_nodes),
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

    def test_arc_refinements_use_user_approved_interpolation(self):
        axis = normalize_axis_payload({'team': [{'slot': 0, 'character_id': CID, 'arc_id': ARC,
            'arc_refinement': 5, 'cartridge_id': ''}], 'steps': []})
        self.assertEqual(axis['team'][0]['arc_refinement'], 5)
        levels = load_shaft_catalog()['arc_refinements']['arcs'][ARC]['levels']
        self.assertEqual([levels[str(i)]['panel_modifiers']['harmony_strength'] for i in range(1, 6)], [72, 90, 108, 126, 144])
        self.assertEqual([levels[str(i)]['buff_effects']['arc_heiyu_ember']['crit_dmg'] for i in range(1, 6)], [.24, .3, .36, .42, .48])
        r = simulate_shaft_axis({**axis, 'steps': [dict(id=str(i), slot=0,
            action_id='action_heiyu_'+a, start_tick=t) for i,(a,t) in enumerate([('q1',0),('q2',10),('q1',220)])]})['result']
        ember = [[b for b in d['applied_buffs'] if b['rule_id'] == 'arc_heiyu_ember'] for d in r['details']]
        self.assertEqual([bool(x) for x in ember], [False, True, False])
        self.assertEqual(ember[1][0]['effects']['crit_dmg'], .48)

    def test_a_consumes_all_only_after_unlock_and_e_multiplier_is_separate(self):
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_q2', 20)]
        base = self.simulate(steps)['details'][1]
        awakened = self.simulate(steps, awakening_nodes=[1])['details'][1]
        self.assertEqual(base['personal_resources_after']['恶意'], 20)
        self.assertEqual(awakened['personal_resources_after']['恶意'], 0)
        self.assertEqual(awakened['personal_resources_consumed']['恶意'], 20)
        self.assertGreater(awakened['resource_damage'], 0)
        self.assertAlmostEqual(awakened['direct_damage'] - awakened['resource_damage'], base['direct_damage'])
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_form_a5', 10), (0, 'action_heiyu_form_e', 20)]
        base = self.simulate(steps)['details'][-1]
        awakened = self.simulate(steps, awakening_nodes=[1])['details'][-1]
        self.assertAlmostEqual(awakened['resource_damage'], base['resource_damage'])
        self.assertAlmostEqual(awakened['direct_damage'] - awakened['resource_damage'],
                               3 * (base['direct_damage'] - base['resource_damage']))

    def test_b_multiplier_and_assist_resource_only_in_normal_form(self):
        steps = [(0, 'action_heiyu_e', 0)]
        self.assertAlmostEqual(self.simulate(steps, awakening_nodes=[2])['details'][0]['direct_damage'],
                               2 * self.simulate(steps)['details'][0]['direct_damage'])
        normal = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                               (0, 'action_heiyu_q2', 20), (0, 'action_heiyu_assist', 40)], awakening_nodes=[1,2])['details'][-1]
        self.assertEqual(normal['personal_resources_after']['恶意'], 4)
        form = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                             (0, 'action_heiyu_form_z1', 20), (0, 'action_heiyu_assist', 40)], awakening_nodes=[2])['details'][-1]
        self.assertAlmostEqual(form['personal_resources_after']['恶意'], 56.2)

    def test_c_only_boosts_resource_damage(self):
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_form_z1', 10)]
        base = self.simulate(steps)['details'][-1]
        awakened = self.simulate(steps, awakening_nodes=[3])['details'][-1]
        bonus = base['formula_parts']['dmg_bonus']
        self.assertAlmostEqual(awakened['resource_damage'], base['resource_damage'] * (1 + bonus + .5) / (1 + bonus))
        self.assertTrue(any('仅恶意追加' in b['name'] for b in awakened['applied_buffs']))
        self.assertAlmostEqual(awakened['direct_damage'] - awakened['resource_damage'], base['direct_damage'] - base['resource_damage'])

    def test_d_replaces_attack_and_f_expires(self):
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_a1', 200), (0, 'action_heiyu_a1', 300)]
        r = self.simulate(steps, awakening_nodes=[4])['details']
        buffs = r[1]['applied_buffs']
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_q_attack' for b in buffs))
        self.assertEqual(next(b for b in buffs if b['rule_id'] == 'character_heiyu_d_attack')['effects']['atk_pct'], .3)
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_d_attack' for b in r[2]['applied_buffs']))
        r = self.simulate(steps, awakening_nodes=[6])['details']
        self.assertEqual(next(b for b in r[1]['applied_buffs'] if b['rule_id'] == 'character_heiyu_f_soul')['effects']['element_dmg_魂'], .25)
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_f_soul' for b in r[2]['applied_buffs']))

    def test_e_resistance_reduction_starts_after_hit_and_expires(self):
        r = self.simulate([(0, 'action_heiyu_a1', 0), (0, 'action_heiyu_a1', 10),
                           (0, 'action_heiyu_a1', 300)], awakening_nodes=[5])['details']
        self.assertFalse(any(b['rule_id'].startswith('character_heiyu_e_res') for b in r[0]['applied_buffs']))
        effect = next(b for b in r[1]['applied_buffs'] if b['rule_id'].startswith('character_heiyu_e_res'))['effects']
        self.assertEqual(effect, {'res_down_魂': .08, 'res_down_相': .08, 'res_down_咒': .08})
        self.assertFalse(any(b['rule_id'].startswith('character_heiyu_e_res') for b in r[2]['applied_buffs']))

    def test_three_node_resonance_and_six_node_team_crit(self):
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_q2', 20)]
        base = self.simulate(steps)['details']
        r = self.simulate(steps, awakening_nodes=[1,2,3])['details']
        self.assertEqual(r[0]['personal_resources_after']['恶意'], 40)
        self.assertAlmostEqual(r[0]['direct_damage'], base[0]['direct_damage'] * 2 * 1.08)
        self.assertAlmostEqual(r[1]['direct_damage'] - r[1]['resource_damage'], base[1]['direct_damage'] * 1.5 * 1.08)
        other = {'slot':1,'character_id':DARK,'arc_id':'','cartridge_id':''}
        r = self.simulate([(1,'action_dodge_c78f7a08d5',0)], [other], awakening_nodes=[1,2,3,4,5,6])['details'][0]
        self.assertEqual(next(b for b in r['applied_buffs'] if b['rule_id']=='character_heiyu_six_crit')['effects']['crit_dmg'], .3)

    def test_cross_loop_accumulation_is_explicitly_out_of_scope(self):
        other = {'slot':1,'character_id':DARK,'arc_id':'','cartridge_id':''}
        r = self.simulate([(1,'action_none_c78f7a08d5',0),(0,'action_heiyu_support',40),
                           (0,'action_heiyu_a1',60)], [other], loop_enabled=True)
        self.assertFalse(any('跨轮' in warning for d in r['details'] for warning in d['warnings']))
        self.assertTrue(all(e['formula_parts']['base'] >= 0 for e in r['reaction_damage_events'] if e.get('heiyu_accumulation')))

if __name__ == '__main__': unittest.main()
