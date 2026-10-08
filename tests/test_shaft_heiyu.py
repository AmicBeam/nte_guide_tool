import unittest
from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout

CID = 'char_heiyu'
ARC = 'arc_heiyu_crime_punishment'
DARK = 'char_c78f7a08d5'

class HeiyuTest(unittest.TestCase):
    def simulate(self, steps, others=(), awakening_nodes=(), team_panel_bonus=None, **options):
        axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': list(awakening_nodes),
                          'skill_levels': dict(basic=1, skill=1, ultimate=1, support=1)}, *others],
                'team_panel_bonus': team_panel_bonus or {},
                'steps': [dict(id=str(i), slot=slot, action_id=action, start_tick=tick)
                          for i, (slot, action, tick) in enumerate(steps)], 'options': options}
        return simulate_shaft_axis(axis)['result']

    def test_user_supplied_bond_bonus_adds_eight_percent_crit_damage(self):
        # User 2026-09-18; Everness has no matching Heiyu entry.
        character = next(c for c in load_shaft_catalog()['characters'] if c['id'] == CID)
        self.assertEqual(character['bond_bonus'], {'label': '8暴伤', 'modifiers': {'crit_dmg': .08}})
        panels = []
        for enabled in (False, True):
            axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '',
                              'cartridge_id': '', 'bond_full': enabled}],
                    'steps': [{'id': 'a', 'slot': 0, 'action_id': 'action_heiyu_a1', 'start_tick': 0}]}
            normalized = normalize_axis_payload(axis)
            self.assertEqual(normalized['team'][0]['bond_full'], enabled)
            panels.append(simulate_shaft_axis(axis)['result']['details'][0]['panel'])
        self.assertAlmostEqual(panels[1]['crit_dmg'] - panels[0]['crit_dmg'], .08)
        self.assertEqual(panels[1]['crit_rate'], panels[0]['crit_rate'])
        self.assertEqual(panels[1]['atk'], panels[0]['atk'])

    def test_source_rows_and_estimated_durations(self):
        catalog = load_shaft_catalog()
        actions = {a['name']: a for a in catalog['actions_by_character'][CID]}
        self.assertAlmostEqual(actions['a5']['multipliers']['atk'], 1.82)
        self.assertAlmostEqual(actions['a5']['energy_gain'], 6.445)
        self.assertAlmostEqual(actions['魔a5']['harmony'], 11.778)
        self.assertEqual(actions['e']['harmony'], 10.4)
        self.assertEqual(actions['e']['harmony_on_start'], 100)
        self.assertEqual(actions['q1']['energy_cost'], 100)
        self.assertEqual(actions['q2']['energy_cost'], 0)
        # 用户2026-09-18口径：一个动作对应整套5次协同中的1次。
        assist = actions['失乐鸟']
        self.assertEqual(assist['hit_count'], 1)
        self.assertEqual(assist['multipliers']['atk'], 2)
        self.assertEqual(assist['source_formula'], '{0}%')
        self.assertEqual(assist['energy_gain'], 2)
        self.assertAlmostEqual(assist['harmony'], 3.34)
        self.assertAlmostEqual(assist['stagger'], .4)
        self.assertEqual(actions['a1']['duration_ticks'], 4)
        self.assertEqual(actions['a1']['duration_source'], 'user-authorized-estimate')
        self.assertEqual(sum(actions[f'a{i}']['duration_ticks'] for i in range(1, 6)), 40)
        # 用户2026-09-19提供耗时，非Nanoka核实数据。
        for name, ticks in {'魔a1': 5, '魔a2': 7, '魔a3': 12, '魔a4': 4,
                            '魔a5': 14, '魔e': 10, '魔z2': 11}.items():
            self.assertEqual(actions[name]['duration_ticks'], ticks)
            self.assertEqual(actions[name]['duration_seconds'], ticks / 10)
            self.assertEqual(actions[name]['duration_source'], 'user-supplied')
        self.assertEqual(sum(actions[f'魔a{i}']['duration_ticks'] for i in range(1, 6)), 42)
        self.assertEqual(len([a for a in actions.values() if a.get('source_row') == 19]), 1)

    def test_ntedata_energy_matches_damage_rows_and_interruptions(self):
        import json
        from pathlib import Path
        profiles = json.loads((Path(__file__).resolve().parents[1] /
            'app/modules/shaft/static/data/action_hit_profiles.json').read_text())
        expected = dict(a1=.838, a2=3.452, a3=4.264, a4=4.565, a5=6.445,
                        z=4.08, plunge=1.68, counter=3, form_a1=1.948,
                        form_a2=3.325, form_a3=3.78, form_a4=4.066, form_a5=8.484,
                        form_z1=6.51, form_z2=6.09, form_plunge=1.68,
                        form_counter=2.52, e=7.56, assist=2, form_e=5.04,
                        q1=0, q2=0, support=7.56, form_support=6, parry=0)
        actions = [a for a in load_shaft_catalog()['actions_by_character'][CID]
                   if a['id'].startswith('action_heiyu_')]
        self.assertEqual(len(actions), len(expected))
        for action in actions:
            energy = expected[action['id'].removeprefix('action_heiyu_')]
            with self.subTest(action=action['id']):
                self.assertAlmostEqual(action['energy_gain'], energy)
                if action['id'] in profiles:
                    self.assertAlmostEqual(profiles[action['id']]['cumulative'][-1]['energy'],
                                           action['energy_gain'])
        self.assertAlmostEqual(profiles['action_heiyu_form_a2']['cumulative'][2]['energy'], 1.33)

    def test_user_form_q_and_malice_rebalance(self):
        # User 2026-09-22 includes branches, plunge and counter in the doubling.
        import json
        from pathlib import Path
        expected = dict(form_a1=.65, form_a2=1.11, form_a3=1.26,
                        form_a4=1.355, form_a5=2.826, form_z1=2.16,
                        form_z2=2.02, form_plunge=.56, form_counter=1.26,
                        q1=4.203, q2=6.3)
        catalog = load_shaft_catalog()
        actions = {a['id']: a for a in catalog['actions_by_character'][CID]}
        profiles = json.loads((Path(__file__).resolve().parents[1] /
            'app/modules/shaft/static/data/action_hit_profiles.json').read_text())
        for suffix, multiplier in expected.items():
            aid = 'action_heiyu_' + suffix
            with self.subTest(action=aid):
                self.assertAlmostEqual(actions[aid]['multipliers']['atk'], multiplier)
                if aid in profiles:
                    curve = profiles[aid]['cumulative']
                    for hit in curve:
                        self.assertAlmostEqual(hit['damage'],
                            multiplier * hit['hit_count'] / curve[-1]['hit_count'])
        for action in actions.values():
            if action.get('resource_damage'):
                self.assertEqual(action['resource_damage']['atk_per_point'], .28)
        for suffix in ('q1', 'q2'):
            detail = self.simulate([(0, 'action_heiyu_' + suffix, 0)])['details'][0]
            self.assertAlmostEqual(detail['formula_parts']['raw_base'] / detail['panel']['atk'],
                                   expected[suffix])

    def test_normal_form_document_multipliers_and_partial_hits(self):
        # Tencent user document, new multiplier column read on 2026-09-22.
        expected = dict(a1=.237, a2=.974, a3=1.204, a4=1.29, a5=1.82,
                        z=1.139, plunge=.476, counter=1.275)
        for suffix, multiplier in expected.items():
            action_id = 'action_heiyu_' + suffix
            with self.subTest(action=action_id):
                detail = self.simulate([(0, action_id, 0)])['details'][0]
                self.assertAlmostEqual(detail['formula_parts']['raw_base'] / detail['panel']['atk'], multiplier)
        axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': '',
                          'skill_levels': dict(basic=1, skill=1, ultimate=1, support=1)}],
                'steps': [{'id': 'a', 'slot': 0, 'action_id': 'action_heiyu_a4',
                           'start_tick': 0, 'interrupted': True, 'interrupt_duration_ticks': 4,
                           'interrupt_hit_count': 2}]}
        detail = simulate_shaft_axis(axis)['result']['details'][0]
        self.assertAlmostEqual(detail['formula_parts']['raw_base'] / detail['panel']['atk'], .516)

    def test_form_a5_optional_six_tick_detach_preserves_damage_and_resources(self):
        # User 2026-09-23: optional detach occupies 0.6s; full cast stays 1.4s.
        action = next(a for a in load_shaft_catalog()['actions_by_character'][CID] if a['name'] == '魔a5')
        self.assertEqual(action['detached_duration_seconds'], 0.6)
        self.assertEqual(action['detached_duration_ticks'], 6)
        results = []
        for detached in (False, True):
            axis = {'team': [{'slot': 0, 'character_id': CID, 'arc_id': '', 'cartridge_id': ''}],
                    'steps': [{'id': 'q', 'slot': 0, 'action_id': 'action_heiyu_q1', 'start_tick': 0},
                              {'id': 'a', 'slot': 0, 'action_id': 'action_heiyu_form_a5',
                               'start_tick': 10, 'detached': detached}]}
            normalized = normalize_axis_payload(axis)
            detail = simulate_shaft_axis(normalized)['result']['details'][-1]
            self.assertEqual(detail['duration_ticks'], 6 if detached else 14)
            self.assertEqual(detail['is_detached'], detached)
            results.append(detail)
        for key in ('direct_damage', 'energy_gain', 'harmony', 'personal_resources_consumed'):
            self.assertEqual(results[0][key], results[1][key])

    def test_basic_malice_is_proportional_and_chain_totals_are_preserved(self):
        actions = {a['name']: a for a in load_shaft_catalog()['actions_by_character'][CID]}
        gains = [0.857919, 3.525792, 4.358372, 4.669685, 6.588232]
        costs = [1.083182, 1.849745, 2.099706, 2.25802, 4.709347]
        for prefix, amount, field, expected in [
                ('', 20, 'personal_resource_gain', gains),
                ('魔', 12, 'personal_resource_cost', costs)]:
            chain = [actions[f'{prefix}a{i}'] for i in range(1, 6)]
            self.assertAlmostEqual(sum(a[field]['恶意'] for a in chain), amount)
            for action, value in zip(chain, expected):
                self.assertAlmostEqual(action[field]['恶意'], value)
                self.assertNotIn('validation_note', action)
        r = self.simulate([(0, f'action_heiyu_a{i+1}', t) for i,t in enumerate([0,4,10,18,28])])
        self.assertAlmostEqual(r['details'][-1]['personal_resources_after']['恶意'], 40)
        self.assertEqual(r['summary']['duration_ticks'], 40)
        r = self.simulate([(0, 'action_heiyu_q1', 0)] + [(0, f'action_heiyu_form_a{i+1}', t) for i,t in enumerate([10,15,22,34,38])])
        self.assertAlmostEqual(r['details'][-1]['personal_resources_after']['恶意'], 8)
        self.assertTrue(all(d['resource_damage'] > 0 for d in r['details'][1:]))

    def test_interrupted_basic_scales_malice_by_retained_hits(self):
        actions = {a['id']: a for a in load_shaft_catalog()['actions_by_character'][CID]}
        for action_id, hits, key, ratio in [('action_heiyu_a2',1,'personal_resource_gain',.5),
                                           ('action_heiyu_form_a2',2,'personal_resource_cost',.4),
                                           ('action_heiyu_form_a2',0,'personal_resource_cost',0)]:
            steps = [{'id':'q','slot':0,'action_id':'action_heiyu_q1','start_tick':0}] if 'form_' in action_id else []
            steps.append({'id':'cut','slot':0,'action_id':action_id,'start_tick':20,
                          'interrupted':True,'interrupt_duration_ticks':2,'interrupt_hit_count':hits})
            r = simulate_shaft_axis({'team':[{'slot':0,'character_id':CID,'arc_id':'','cartridge_id':''}], 'steps':steps})['result']
            d = r['details'][-1]
            self.assertTrue(d['is_interrupted'])
            amount = actions[action_id][key]['恶意'] * ratio
            self.assertAlmostEqual(d['personal_resources_after']['恶意'], 20 + (amount if key.endswith('gain') else -amount))
            if key.endswith('cost'):
                self.assertAlmostEqual(d['personal_resources_consumed'].get('恶意', 0), amount)
                self.assertEqual(d['resource_damage'] > 0, hits > 0)

    def test_kongmu_access_and_topology(self):
        c = next(c for c in get_kongmu_catalog_payload()['characters'] if c['id'] == CID)
        self.assertEqual(c['element'], '魂')
        self.assertIn('暴击率增加8%', c['kongmu_passive']['text'])
        result = plan_kongmu_layout(CID, 'attack')
        slots = result['character']['equip_slots']['slots']
        self.assertEqual(sum(v != -1 for row in slots for v in row), 20)
        self.assertEqual(slots[2], [0, 0, -1, 0, 0])

    def test_resource_cap_spend_and_q_exit_preserves(self):
        r = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                           (0, 'action_heiyu_form_z1', 20), (0, 'action_heiyu_q2', 30)])
        d = r['details']
        self.assertEqual(d[0]['personal_resources_after']['恶意'], 60)
        self.assertAlmostEqual(d[2]['personal_resources_after']['恶意'], 56.4005)
        self.assertGreater(d[2]['resource_damage'], 0)
        self.assertEqual(d[3]['personal_resources_after'], d[2]['personal_resources_after'])
        self.assertEqual(d[3]['resource_damage'], 0)
        r = self.simulate([(0, 'action_heiyu_form_e', 0), (0, 'action_heiyu_form_z1', 10)])
        self.assertTrue(any('激活' in w for w in r['details'][0]['warnings']))
        # 魔E消耗13，魔z1消耗3.5995；初始20点足够连续支付。
        self.assertAlmostEqual(r['details'][0]['personal_resources_after']['恶意'], 7)
        self.assertAlmostEqual(r['details'][1]['personal_resources_after']['恶意'], 3.4005)
        self.assertGreater(r['details'][1]['resource_damage'], 0)
        self.assertFalse(any('恶意 不足' in w for w in r['details'][1]['warnings']))

    def test_partial_malice_cost_is_optional_and_damage_uses_available_amount(self):
        # 用户2026-09-19：恶意不足不报自检错误，仍只消耗实际可用量。
        r = self.simulate([(0, 'action_heiyu_form_z1', 0)], loop_enabled=True,
                          loop_initial_resources={CID: {'energy': 0, 'harmony': 0,
                                                        'personal_resources': {'恶意': 2}}})
        detail = r['details'][0]
        self.assertNotIn('个人资源 恶意 不足。', detail['warnings'])
        self.assertAlmostEqual(detail['personal_resources_consumed']['恶意'], 2)
        self.assertEqual(detail['personal_resources_after']['恶意'], 0)
        self.assertAlmostEqual(detail['resource_atk_multiplier'], 2 * .28)
        self.assertGreater(detail['resource_damage'], 0)
        self.assertTrue(any('二形态' in w for w in detail['warnings']))

    def test_resource_multiplier_uses_actual_consumption_and_basic_level(self):
        r = self.simulate([(0, 'action_heiyu_q1', 0),
                           (0, 'action_heiyu_form_e', 10),
                           (0, 'action_heiyu_form_z1', 30)])
        self.assertAlmostEqual(r['details'][1]['resource_atk_multiplier'], 13 * .28)
        self.assertAlmostEqual(r['details'][1]['personal_resources_consumed']['恶意'], 13)
        self.assertAlmostEqual(r['details'][1]['personal_resources_after']['恶意'], 7)
        self.assertAlmostEqual(r['details'][2]['resource_atk_multiplier'], 3.5995 * .28)
        self.assertAlmostEqual(r['details'][2]['personal_resources_after']['恶意'], 3.4005)

    def test_malice_contribution_is_separate_basic_damage_and_preserves_totals(self):
        for nodes in ((), (1,)):
            with self.subTest(awakening_nodes=nodes):
                result = self.simulate([
                    (0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 20),
                    (0, 'action_heiyu_form_a1', 30),
                    (0, 'action_heiyu_form_e', 40),
                    (0, 'action_heiyu_form_support', 60),
                    (0, 'action_heiyu_q2', 80),
                ], awakening_nodes=nodes)
                contribution = result['damage_by_action_by_slot'][0]
                entries = {a['action_id']: a for a in contribution['actions']}
                extra = next(a for a in entries.values() if a['action_name'] == '恶意追加')
                self.assertEqual(extra['action_type'], '普攻')
                self.assertEqual(extra['damage_type'], '普攻')
                self.assertEqual(extra['damage_element'], '魂')
                self.assertAlmostEqual(extra['damage'], sum(d['resource_damage'] for d in result['details']))
                self.assertEqual(result['details'][-1]['resource_damage'] > 0, bool(nodes))
                for detail in result['details']:
                    self.assertAlmostEqual(entries[detail['action_id']]['damage'],
                                           detail['direct_damage'] - detail['resource_damage'])
                self.assertAlmostEqual(sum(a['damage'] for a in entries.values()), contribution['total_damage'])
                self.assertAlmostEqual(contribution['total_damage'], result['damage_by_slot'][0]['damage'])
                self.assertAlmostEqual(contribution['total_damage'], result['summary']['character_damage'])

        empty = self.simulate([(0, 'action_heiyu_form_z1', 0)], loop_enabled=True)
        self.assertFalse(any(a['action_name'] == '恶意追加'
                             for a in empty['damage_by_action_by_slot'][0]['actions']))

    def test_q1_checks_form_before_start_and_q2_allows_reentry(self):
        warning = '当前形态不能使用该动作。'
        r = self.simulate([(0, 'action_heiyu_q1', 0),
                           (0, 'action_heiyu_q1', 20),
                           (0, 'action_heiyu_q2', 40),
                           (0, 'action_heiyu_q1', 240)])
        self.assertNotIn(warning, r['details'][0]['warnings'])
        self.assertIn(warning, r['details'][1]['warnings'])
        self.assertNotIn(warning, r['details'][3]['warnings'])

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
        # 五次合法协同合计1000%，单次伤害为同属性同类型400%普通E的一半。
        assists = r['details'][1:6]
        for detail in assists:
            self.assertAlmostEqual(detail['direct_damage'], r['details'][0]['direct_damage'] / 2)
        self.assertAlmostEqual(sum(d['direct_damage'] for d in assists), r['details'][0]['direct_damage'] * 2.5)
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

    def test_accumulated_damage_is_other_not_harmony_or_character_damage(self):
        other = {'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''}
        r = self.simulate([(1, 'action_none_c78f7a08d5', 0), (0, 'action_heiyu_support', 2),
                           (0, 'action_heiyu_e', 20), (1, 'action_dodge_c78f7a08d5', 120)], [other])
        extra = sum(e['damage'] for e in r['reaction_damage_events'] if e.get('heiyu_accumulation'))
        dark_star = sum(e['damage'] for e in r['reaction_damage_events']
                        if e['reaction'] == '黯星' and not e.get('heiyu_accumulation')
                        and e.get('_counted_in_total') is not False)
        summary = r['summary']
        self.assertGreater(extra, 0)
        self.assertAlmostEqual(summary['other_damage'], extra)
        self.assertAlmostEqual(summary['harmony_damage'], dark_star)
        self.assertAlmostEqual(summary['total_damage'], sum(summary[k] for k in
                               ('character_damage', 'harmony_damage', 'stagger_damage', 'other_damage')))
        self.assertAlmostEqual(sum(c['damage'] for c in r['damage_by_slot']), summary['character_damage'])
        self.assertAlmostEqual(sum(c['damage'] for c in r['harmony_contributions_by_slot']), dark_star)
        contribution = next(c for c in r['other_contributions_by_slot'] if c['damage'] > 0)
        self.assertEqual(contribution['character_id'], CID)
        self.assertEqual(contribution['percent'], 100)
        self.assertAlmostEqual(contribution['damage'], extra)
        self.assertEqual(contribution['sources'][0]['source'], '黑羽黯星额外伤害')
        sources = {c['source']: c['damage'] for c in r['damage_by_source']}
        self.assertAlmostEqual(sources['其他'], extra)
        self.assertAlmostEqual(sources['黯星'], dark_star)
        ordinary = self.simulate([(0, 'action_heiyu_a1', 0)])
        self.assertEqual(ordinary['summary']['other_damage'], 0)
        self.assertNotIn('其他', [c['source'] for c in ordinary['damage_by_source']])

    def test_dark_star_recast_starts_a_fresh_damage_window(self):
        other = {'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''}
        for second_window_hit in (False, True):
            with self.subTest(second_window_hit=second_window_hit):
                steps = [(1, 'action_none_c78f7a08d5', 0),
                         (0, 'action_heiyu_support', 2), (0, 'action_heiyu_e', 20)]
                if second_window_hit:
                    steps.append((0, 'action_heiyu_a1', 70))
                steps.append((1, 'action_dodge_c78f7a08d5', 120))
                r = self.simulate(steps, [other])
                effects = [e for e in r['reaction_effects'] if e['reaction'] == '黯星']
                extras = {e['effect_id']: e for e in r['reaction_damage_events']
                          if e.get('heiyu_accumulation')}
                self.assertEqual(len(effects), 2)
                self.assertEqual(effects[1]['start_tick'], effects[0]['end_tick'])
                self.assertAlmostEqual(extras[effects[0]['id']]['formula_parts']['base'],
                                       r['details'][2]['direct_damage'])
                if second_window_hit:
                    self.assertAlmostEqual(extras[effects[1]['id']]['formula_parts']['base'],
                                           r['details'][3]['direct_damage'])
                else:
                    self.assertNotIn(effects[1]['id'], extras)

    def test_dark_star_ledger_excludes_explicit_dark_star_damage_sources(self):
        # 用户2026-09-18：每个实例只累计期间的非黯星伤害，包括手工伤害来源过滤。
        others = [{'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''},
                  {'slot': 2, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''}]
        r = self.simulate([(1, 'action_none_c78f7a08d5', 0),
                           (0, 'action_heiyu_support', 2), (0, 'action_heiyu_e', 20),
                           (2, 'action_bdc49de5c5', 30),
                           (1, 'action_dodge_c78f7a08d5', 120)], others)
        self.assertEqual(r['details'][3]['damage_source'], '黯星')
        self.assertGreater(r['details'][3]['direct_damage'], 0)
        extras = [e for e in r['reaction_damage_events'] if e.get('heiyu_accumulation')]
        self.assertEqual(len(extras), 1)
        self.assertAlmostEqual(extras[0]['formula_parts']['base'], r['details'][2]['direct_damage'])

    def test_arc_refinements_use_user_approved_interpolation(self):
        axis = normalize_axis_payload({'team': [{'slot': 0, 'character_id': CID, 'arc_id': ARC,
            'arc_refinement': 5, 'cartridge_id': ''}], 'steps': []})
        self.assertEqual(axis['team'][0]['arc_refinement'], 5)
        levels = load_shaft_catalog()['arc_refinements']['arcs'][ARC]['levels']
        self.assertEqual([levels[str(i)]['panel_modifiers']['harmony_strength'] for i in range(1, 6)], [72, 90, 108, 126, 144])
        self.assertEqual([levels[str(i)]['buff_effects']['arc_heiyu_self_crit']['crit_dmg'] for i in range(1, 6)], [.32, .4, .48, .56, .64])
        axis['team'].append({'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''})
        # 5精：装备者同时获得32%全队与64%自身暴伤，普攻也享受；重复Q刷新15秒。
        steps = [(0, 'q1', 0), (0, 'a1', 20), (1, 'other', 40),
                 (0, 'q2', 100), (0, 'a1', 180), (0, 'a1', 280)]
        axis['steps'] = [dict(id=str(i), slot=slot, start_tick=t,
            action_id='action_dodge_c78f7a08d5' if a == 'other' else 'action_heiyu_'+a)
            for i, (slot, a, t) in enumerate(steps)]
        details = simulate_shaft_axis(axis)['result']['details']
        for index in (0, 1, 3, 4):
            buffs = {b['rule_id']: b for b in details[index]['applied_buffs']}
            self.assertEqual(buffs['arc_heiyu_self_crit']['effects']['crit_dmg'], .64)
            self.assertEqual(buffs['arc_heiyu_team_crit']['effects']['crit_dmg'], .32)
        teammate = {b['rule_id'] for b in details[2]['applied_buffs']}
        self.assertIn('arc_heiyu_team_crit', teammate)
        self.assertNotIn('arc_heiyu_self_crit', teammate)
        self.assertFalse(any(b['rule_id'].startswith('arc_heiyu_') for b in details[-1]['applied_buffs']))

    def test_form_no_longer_grants_crit_damage(self):
        detail = self.simulate([(0, 'action_heiyu_q1', 0),
                                (0, 'action_heiyu_form_a1', 10)])['details'][-1]
        normal = self.simulate([(0, 'action_heiyu_a1', 0)])['details'][0]
        self.assertAlmostEqual(detail['panel']['crit_dmg'], normal['panel']['crit_dmg'])

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
        self.assertEqual(next(b['effects']['base_multiplier_pct'] for b in awakened['applied_buffs']
                              if b['rule_id'] == 'character_heiyu_a_e'), 1.5)
        self.assertAlmostEqual(awakened['direct_damage'] - awakened['resource_damage'],
                               2.5 * (base['direct_damage'] - base['resource_damage']))

    def test_b_multiplier_and_assist_resource_only_in_normal_form(self):
        steps = [(0, 'action_heiyu_e', 0)]
        self.assertAlmostEqual(self.simulate(steps, awakening_nodes=[2])['details'][0]['direct_damage'],
                               2 * self.simulate(steps)['details'][0]['direct_damage'])
        normal = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                               (0, 'action_heiyu_q2', 20), (0, 'action_heiyu_assist', 40)], awakening_nodes=[1,2])['details'][-1]
        self.assertEqual(normal['personal_resources_after']['恶意'], 0)
        granted = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                                 (0, 'action_heiyu_q2', 20), (0, 'action_heiyu_assist', 40)], awakening_nodes=[1,6])['details'][-1]
        self.assertEqual(granted['personal_resources_after']['恶意'], 4)
        form = self.simulate([(0, 'action_heiyu_e', 0), (0, 'action_heiyu_q1', 10),
                             (0, 'action_heiyu_form_z1', 20), (0, 'action_heiyu_assist', 40)], awakening_nodes=[2])['details'][-1]
        self.assertAlmostEqual(form['personal_resources_after']['恶意'], 56.4005)
        healed = self.simulate([(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_form_a5', 10),
                                (0, 'action_heiyu_form_e', 30)], awakening_nodes=[2])
        self.assertEqual(healed['healing_events'][0]['healing'], 668 * 4)
        self.assertEqual(next(b['effects']['def_pct'] for b in healed['details'][-1]['applied_buffs']
                              if b['rule_id'] == 'character_heiyu_b_def'), .2)

    def test_c_only_boosts_resource_damage(self):
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_form_z1', 10)]
        # 已有100%通伤时仍应独立乘1.5，不能退回增伤区加算。
        base = self.simulate(steps, team_panel_bonus={'all_dmg': 1})['details'][-1]
        awakened = self.simulate(steps, awakening_nodes=[3], team_panel_bonus={'all_dmg': 1})['details'][-1]
        self.assertAlmostEqual(awakened['resource_damage'], base['resource_damage'] * 1.5)
        self.assertAlmostEqual(awakened['resource_atk_multiplier'], base['resource_atk_multiplier'] * 1.5)
        self.assertTrue(any('仅恶意追加' in b['name'] for b in awakened['applied_buffs']))
        self.assertAlmostEqual(awakened['direct_damage'] - awakened['resource_damage'], base['direct_damage'] - base['resource_damage'])

    def test_d_keeps_base_attack_adds_damage_and_f_expires(self):
        # User 2026-09-22: replace only the extra 10% attack with 12% all damage.
        steps = [(0, 'action_heiyu_q1', 0), (0, 'action_heiyu_a1', 200), (0, 'action_heiyu_a1', 300)]
        r = self.simulate(steps, awakening_nodes=[4])['details']
        buffs = r[1]['applied_buffs']
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_q_attack' for b in buffs))
        self.assertEqual(next(b for b in buffs if b['rule_id'] == 'character_heiyu_d_attack')['effects'], {'atk_pct': .2, 'all_dmg': .12})
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_d_attack' for b in r[2]['applied_buffs']))
        r = self.simulate(steps, awakening_nodes=[6])['details']
        self.assertEqual(next(b for b in r[1]['applied_buffs'] if b['rule_id'] == 'character_heiyu_f_soul')['effects']['element_dmg_魂'], .25)
        self.assertFalse(any(b['rule_id'] == 'character_heiyu_f_soul' for b in r[2]['applied_buffs']))

    def test_e_resistance_reduction_starts_after_hit_and_expires(self):
        r = self.simulate([(0, 'action_heiyu_a1', 0), (0, 'action_heiyu_a1', 10),
                           (0, 'action_heiyu_a1', 300)], awakening_nodes=[5])['details']
        self.assertFalse(any(b['rule_id'].startswith('character_heiyu_e_res') for b in r[0]['applied_buffs']))
        effect = next(b for b in r[1]['applied_buffs'] if b['rule_id'].startswith('character_heiyu_e_res'))['effects']
        self.assertEqual(effect, {'res_down_魂': .08, 'res_down_相': .08, 'res_down_暗': .08})
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

    def test_new_dark_star_settles_previous_and_reports_accumulation_at_replacement(self):
        other = {'slot': 1, 'character_id': DARK, 'arc_id': '', 'cartridge_id': ''}
        for second_tick in (42, 60):
            with self.subTest(second_tick=second_tick):
                steps = [(1, 'action_none_c78f7a08d5', 0),
                         (0, 'action_heiyu_support', 2),
                         (0, 'action_heiyu_e', 20),
                         (1, 'action_none_c78f7a08d5', second_tick - 2),
                         (0, 'action_heiyu_support', second_tick)]
                if second_tick == 60:
                    # 原黯星6.5秒到期，新黯星的环合生效节点为7.3秒。
                    # 6.8秒后台伤害只属于旧黯星的续生窗口。
                    steps.append((0, 'action_heiyu_assist', 68))
                steps.append((1, 'action_dodge_c78f7a08d5', 180))
                r = self.simulate(steps, [other])
                effects = [e for e in r['reaction_effects'] if e['reaction'] == '黯星']
                replacement = next(e for e in effects if e.get('source_step_id') == '4')
                replaced = next(e for e in effects if e.get('settled_by_conflict')
                                and e['start_tick'] < replacement['start_tick'])
                self.assertEqual(replaced['end_tick'], replacement['start_tick'])
                events = [e for e in r['reaction_damage_events'] if e['effect_id'] == replaced['id']]
                self.assertEqual(len(events), 2)  # 一次本体、一次累计追加
                self.assertTrue(all(e['tick'] == replacement['start_tick'] for e in events))
                extra = next(e for e in events if e.get('heiyu_accumulation'))
                self.assertTrue(extra['conflict_settlement'])
                self.assertEqual(extra['damage_category'], 'other')
                hit = next(d for d in r['details'] if d['action_id'] ==
                           ('action_heiyu_assist' if second_tick == 60 else 'action_heiyu_e'))
                expected_base = hit['direct_damage']
                if second_tick == 42:
                    # 新援护直伤发生在5.5秒环合节点之前，仍计入旧窗口。
                    expected_base += next(d['direct_damage'] for d in r['details'] if d['step_id'] == '4')
                self.assertAlmostEqual(extra['damage'], expected_base * .2)
                self.assertEqual(extra['contributor_character_id'], CID)
                for tick in range(181):
                    self.assertLessEqual(sum(e['start_tick'] <= tick < e['end_tick'] for e in effects), 1)
                self.assertAlmostEqual(r['summary']['other_damage'], sum(
                    e['damage'] for e in r['reaction_damage_events'] if e.get('heiyu_accumulation')))
                if second_tick == 60:
                    self.assertTrue(replaced['heiyu_extra'])
                    self.assertFalse(any(e['effect_id'] == replacement['id'] and e.get('heiyu_accumulation')
                                         for e in r['reaction_damage_events']))

if __name__ == '__main__': unittest.main()
