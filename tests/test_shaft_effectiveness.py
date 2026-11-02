"""Marginal team DPS keeps the original compiled clock, including removed masks."""
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from tests import test_shaft_lingke_support as lingke_tests

ROOT = Path(__file__).resolve().parents[1]


class ShaftEffectivenessTest(unittest.TestCase):
    def evaluate(self, axis, slot, catalog=None):
        script = """
          const e = require('./app/modules/shaft/static/js/shaft_engine.js');
          const {axis, catalog, slot} = JSON.parse(require('fs').readFileSync(0, 'utf8'));
          const before = JSON.stringify({axis, catalog});
          const full = e.simulateAxis(axis, catalog);
          const without = e.simulateAxis(axis, catalog, {excluded_slot: slot});
          const analysis = e.analyzeCharacterEffectiveness(axis, catalog, {slot});
          console.log(JSON.stringify({full, without, analysis,
            unchanged: before === JSON.stringify({axis, catalog})}));
        """
        return json.loads(subprocess.check_output(
            ['node', '-e', script], cwd=ROOT, text=True, timeout=30,
            input=json.dumps(dict(axis=axis, catalog=catalog or load_shaft_catalog(), slot=slot))))

    def assert_fixed_clock(self, result):
        full, without, analysis = (result[k] for k in ('full', 'without', 'analysis'))
        self.assertTrue(result['unchanged'])
        self.assertEqual(full['time_axis'], without['time_axis'])
        self.assertEqual(full['summary']['duration_seconds'], without['summary']['duration_seconds'])
        for detail in without['details']:
            original = next(d for d in full['details'] if d['step_id'] == detail['step_id'])
            for key in ('start_tick', 'end_tick', 'duration_ticks'):
                self.assertEqual(original[key], detail[key])
        self.assertAlmostEqual(analysis['rdps'], full['summary']['dps'] - without['summary']['dps'])

    def test_removed_mask_and_tail_keep_damage_timing_and_loop_period(self):
        catalog = deepcopy(load_shaft_catalog())
        template = catalog['characters'][0]
        catalog['characters'] = [{**template, 'id': 'test_a', 'name': 'A'},
                                 {**template, 'id': 'test_b', 'name': 'B'}]
        catalog['buffs'] = []
        catalog['actions'] = [
            dict(id='hit', character_id='test_a', name='hit', action_type='普攻',
                 damage_type='普攻', duration_ticks=40, hit_count=1, multipliers={'atk': 1}),
            dict(id='mask', character_id='test_b', name='mask', action_type='Q',
                 damage_type='Q', duration_ticks=0, is_time_stop_zero=True, hit_count=0),
            dict(id='tail', character_id='test_b', name='tail', action_type='普攻',
                 damage_type='普攻', duration_ticks=20, hit_count=1, multipliers={'atk': 1}),
        ]
        axis = dict(team=[dict(slot=1, character_id='test_a', arc_id='', cartridge_id=''),
                          dict(slot=3, character_id='test_b', arc_id='', cartridge_id='')],
                    steps=[dict(id='a', slot=1, action_id='hit', start_tick=0),
                           dict(id='b', slot=3, action_id='mask', start_tick=5),
                           dict(id='c', slot=3, action_id='tail', start_tick=50)])
        for loop in (False, True):
            axis['options'] = {'loop_enabled': loop}
            result = self.evaluate(axis, 3, catalog)
            self.assert_fixed_clock(result)
            self.assertTrue(result['full']['time_axis']['frozen_intervals'])
            self.assertLess(result['full']['details'][0]['duration_ticks'], 40)
            self.assertEqual(len(result['without']['details']), 1)
            self.assertAlmostEqual(result['without']['summary']['total_damage'],
                                   result['full']['details'][0]['direct_damage'])
            analysis = result['analysis']
            self.assertAlmostEqual(analysis['increase_percent'],
                                   analysis['rdps'] / analysis['without_dps'] * 100)

    def test_lingke_joint_supports_removed_including_variant_and_side_effects(self):
        for window in ('e', 'q'):
            for variant in (False, True):
                axis = lingke_tests.LingkeSupportTest().axis(window=window, variant=variant)
                # A teammate's later ordinary assist must survive outside the joint window.
                assist_id = axis['steps'][1]['action_id']
                axis['steps'].append(dict(id='ordinary', slot=1, action_id=assist_id, start_tick=200))
                for loop in (False, True):
                    axis['options'] = {'loop_enabled': loop}
                    result = self.evaluate(axis, 0)
                    self.assert_fixed_clock(result)
                    self.assertEqual(next(d for d in result['full']['details']
                                          if d['step_id'] == 'assist')['lingke_joint_source_slot'], 0)
                    self.assertEqual([d['step_id'] for d in result['without']['details']], ['ordinary'])
                    self.assertFalse(any(e.get('source_step_id') == 'assist'
                                         for e in result['without']['energy_events']))

    def test_passive_support_without_actions_and_zero_denominator(self):
        catalog = load_shaft_catalog()
        axis = lingke_tests.LingkeSupportTest().axis()
        elyms = next(c for c in catalog['characters'] if c['name'] == '曳尔沐')
        axis['team'][0] = dict(slot=0, character_id=elyms['id'], awakening_nodes=[4],
                               arc_id='', cartridge_id='')
        axis['steps'] = [axis['steps'][1]]
        result = self.evaluate(axis, 0)
        self.assert_fixed_clock(result)
        self.assertGreater(result['analysis']['rdps'], 0)
        result = self.evaluate(axis, 1)
        self.assert_fixed_clock(result)
        self.assertEqual(result['analysis']['without_dps'], 0)
        self.assertIsNone(result['analysis']['increase_percent'])

    def test_removed_support_retains_harmony_with_level80_zero_strength_proxy(self):
        catalog = deepcopy(load_shaft_catalog())
        template = catalog['characters'][0]
        catalog['characters'] = [{**template, 'id': 'test_a', 'name': 'A', 'element': '光', 'level': 80},
                                 {**template, 'id': 'test_b', 'name': 'B', 'element': '灵', 'level': 80}]
        catalog['buffs'] = []
        catalog['actions'] = [
            dict(id='hit', character_id='test_a', name='hit', action_type='普攻',
                 damage_type='普攻', duration_ticks=10, harmony=100, hit_count=1, multipliers={'atk': 1}),
            dict(id='support', character_id='test_b', name='support', action_type='援护',
                 damage_type='援护', duration_ticks=15, hit_count=1, multipliers={'atk': 2}),
            dict(id='tail', character_id='test_a', name='tail', action_type='无',
                 is_instant_switch=True, duration_ticks=0, hit_count=0),
        ]
        from tests.test_shaft_simulator import ShaftSimulatorValidationTestCase
        bonus = deepcopy(ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS)
        # A same-strength tie chooses the support, allowing the proxy's stats to be inspected.
        axis = dict(team=[dict(slot=1, character_id='test_a', arc_id='', cartridge_id=''),
                          dict(slot=3, character_id='test_b', arc_id='', cartridge_id='',
                               substat_counts={'harmony_strength': 100})],
                    team_panel_bonus=bonus,
                    steps=[dict(id='a', slot=1, action_id='hit', start_tick=0),
                           dict(id='b', slot=3, action_id='support', start_tick=20),
                           dict(id='c', slot=1, action_id='tail', start_tick=200)])
        for loop in (False, True):
            axis['options'] = {'loop_enabled': loop}
            result = self.evaluate(axis, 3, catalog)
            self.assert_fixed_clock(result)
            without = result['without']
            effects = [e for e in without['reaction_effects'] if e['source_step_id'] == 'b']
            self.assertTrue(effects)
            self.assertEqual(effects[0]['reaction'], '创生')
            self.assertEqual(effects[0]['contributor_slot'], 3)
            events = [e for e in without['reaction_damage_events'] if e['damage'] > 0]
            self.assertTrue(events)
            self.assertTrue(all(e['formula_parts']['harmony_strength'] == 0 for e in events))
            self.assertTrue(all(e['formula_parts']['base'] == 9000 for e in events))
            costs = [e for e in without['harmony_events'] if e['kind'] == 'reaction_cost']
            self.assertEqual(costs[0]['slot'], 1)
            self.assertAlmostEqual(costs[0]['amount'], -100)
            self.assertNotIn('b', [d['step_id'] for d in without['details']])
            self.assertGreater(without['summary']['harmony_damage'], 0)
            self.assertGreater(result['full']['summary']['harmony_damage'], without['summary']['harmony_damage'])
            stronger_teammate = deepcopy(axis)
            stronger_teammate['team'][0]['substat_counts'] = {'harmony_strength': 20}
            shared = self.evaluate(stronger_teammate, 3, catalog)['without']
            effect = next(e for e in shared['reaction_effects'] if e['source_step_id'] == 'b')
            self.assertEqual(effect['contributor_slot'], 1)
            event = next(e for e in shared['reaction_damage_events'] if e['damage'] > 0)
            self.assertEqual(event['formula_parts']['harmony_strength'], 120)
            for first, second, reaction in [('光', '相', '延滞'), ('灵', '咒', '覆纹'),
                                            ('咒', '暗', '浊燃'), ('暗', '魂', '黯星'),
                                            ('魂', '相', '浸染')]:
                pair_catalog = deepcopy(catalog)
                pair_catalog['characters'][0]['element'] = first
                pair_catalog['characters'][1]['element'] = second
                pair_result = self.evaluate(axis, 3, pair_catalog)
                self.assert_fixed_clock(pair_result)
                retained = next(e for e in pair_result['without']['reaction_effects']
                                if e['source_step_id'] == 'b')
                self.assertEqual(retained['reaction'], reaction)
                if reaction in ('浊燃', '黯星'):
                    event = next(e for e in pair_result['without']['reaction_damage_events']
                                 if e['damage'] > 0)
                    self.assertEqual(event['formula_parts']['harmony_strength'], 0)



    def test_negative_team_effect_is_not_clamped(self):
        catalog = deepcopy(load_shaft_catalog())
        axis = lingke_tests.LingkeSupportTest().axis()
        axis['steps'] = [axis['steps'][1]]
        provider_id = axis['team'][0]['character_id']
        catalog['buffs'] = [dict(
            id='test_penalty', name='negative marginal test',
            providers=[dict(kind='character', id=provider_id)],
            trigger=dict(event='passive'), target=dict(scope='team'),
            duration=dict(type='permanent'), effects=dict(all_dmg=-0.5))]
        # No reactions or personal resources in this fixture: only the team penalty changes.
        support = next(a for a in catalog['actions'] if a['id'] == axis['steps'][0]['action_id'])
        support['stagger'] = 0
        result = self.evaluate(axis, 0, catalog)
        self.assert_fixed_clock(result)
        self.assertLess(result['analysis']['rdps'], 0)
        self.assertLess(result['analysis']['increase_percent'], 0)

    def test_unknown_slot_and_empty_axis_rejected(self):
        script = """
          const e = require('./app/modules/shaft/static/js/shaft_engine.js');
          const cases = [{}, {slot: 8}, {slot: 'bad'}, {slot: 0.5}, {slot: 0}];
          console.log(JSON.stringify(cases.map(options => {
            try { e.analyzeCharacterEffectiveness({team: [{slot: 0}], steps: []}, {}, options); return false; }
            catch (_) { return true; }
          })));
        """
        self.assertTrue(all(json.loads(subprocess.check_output(['node', '-e', script], text=True))))
