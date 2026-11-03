"""Checked level-1 coefficients and the existing relative level curve."""
import copy
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENTS = {
    'skill1': 1.701,
    'goodbye': 1.277,
    'ultraskilla1': 4.05,
    'ultraskillb1': 4.5,
    'xietongskilla1': 1.08,
    'xietongskillb1': 1.2,
}


class ElymsSkillLevelsTest(unittest.TestCase):
    def simulate(self, names, level=10, nodes=(), resonances=(), isolated=False, interrupt=None):
        catalog = copy.deepcopy(load_shaft_catalog())
        if isolated:
            catalog['buffs'] = []
        axis = {
            'team': [dict(slot=0, character_id='char_elyms', arc_id='', cartridge_id='',
                          awakening_nodes=list(nodes), awakening_resonances=list(resonances),
                          skill_levels=dict(basic=level, skill=level, ultimate=level, support=level))],
            'steps': [dict(id=str(i), slot=0, action_id='action_elyms_' + name, start_tick=i*20)
                      for i, name in enumerate(names)],
        }
        if interrupt is not None:
            axis['steps'][0].update(interrupted=True, interrupt_duration_ticks=5,
                                    interrupt_hit_count=interrupt)
        output = subprocess.check_output(
            ['node', str(ROOT / 'scripts/shaft_compute.js')],
            input=json.dumps({'axis': axis, 'catalog': catalog}), text=True, timeout=15)
        return json.loads(output)['result']

    def test_six_coefficients_at_one_and_relative_levels(self):
        for name, coefficient in COEFFICIENTS.items():
            for level, resonances, expected_level in [(1, [], 1), (10, [], 10), (10, [3], 11)]:
                with self.subTest(action=name, level=expected_level):
                    detail = self.simulate([name], level, resonances=resonances, isolated=True)['details'][0]
                    self.assertEqual(detail['formula_parts']['skill_level'], expected_level)
                    self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'],
                                           coefficient * 1.08 ** (expected_level - 1))

    def test_unmodified_action_keeps_one_level_reference(self):
        detail = self.simulate(['melee1'], isolated=True)['details'][0]
        self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'], .224 * 1.08**9)

    def test_e1_partial_hits_use_actual_one_level_segments_and_keep_resources(self):
        damage = [0, .269, .505, .741, .977, 1.213, 1.701]
        energy = [0, .808, 1.515, 2.222, 2.929, 3.636, 5.1]
        harmony = [0, 1.331, 2.495, 3.659, 4.823, 5.987, 8.399]
        for hits in range(7):
            with self.subTest(hits=hits):
                detail = self.simulate(['skill1'], level=1, isolated=True, interrupt=hits)['details'][0]
                self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'], damage[hits])
                self.assertAlmostEqual(detail['base_energy_gain'], energy[hits])
                self.assertAlmostEqual(detail['harmony'], harmony[hits])
                self.assertEqual(detail['harmony_on_start'], 100)

    def test_e3_each_illusion_adds_one_hundred_percent(self):
        cases = [(['goodbye'], [], [], 10, 0),
                 (['skill2', 'goodbye'], [], [], 10, 1),
                 (['skill2', 'goodbye', 'meleeb3', 'goodbye'], [3], [], 10, 3),
                 (['skill2', 'goodbye'], [3], [3], 9, 4)]
        for names, nodes, resonances, level, count in cases:
            with self.subTest(layers=count):
                detail = self.simulate(names, level, nodes, resonances)['details'][-1]
                self.assertEqual(detail['formula_parts']['skill_level'], 10)
                self.assertEqual(detail['formula_parts']['base_multiplier_factor'], 1 + count)
                self.assertAlmostEqual(detail['formula_parts']['base'] / detail['panel']['atk'], 1.277 * 1.08**9 * (1 + count))

    def test_six_extra_uses_its_own_one_level_coefficient(self):
        for resonances, count, effective_level in [([6], 1, 10), ([3, 6], 4, 11)]:
            with self.subTest(resonances=resonances):
                result = self.simulate(['skill2', 'goodbye'], nodes=[3], resonances=resonances)
                extra, = [e for e in result['periodic_damage_events']
                          if e.get('action_id') == 'action_elyms_six_extra']
                self.assertAlmostEqual(extra['formula_parts']['scaling_multipliers']['atk'],
                                       1.68 * count * 1.08 ** (effective_level - 1))
                self.assertEqual(extra['formula_parts']['base_multiplier_factor'], 1)
        result = self.simulate(['goodbye'], resonances=[6])
        self.assertFalse(any(e.get('action_id') == 'action_elyms_six_extra'
                             for e in result['periodic_damage_events']))
