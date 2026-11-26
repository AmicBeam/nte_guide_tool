"""开始净空: Nanoka 1.4.6 fork_PaperPlane, verified 2026-10-13."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload

ARC = 'arc_edbcee2b65'
ROOT = Path(__file__).resolve().parents[1]


class PaperPlaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_shaft_catalog()
        cls.without_effect = deepcopy(cls.catalog)
        # Keep base attack and the ATK secondary stat identical in both cases.
        cls.without_effect['buffs'] = [b for b in cls.without_effect['buffs'] if not any(
            provider.get('id') == ARC
            for provider in b.get('providers', [b.get('provider', {})])
        )]
        for arc in cls.without_effect['arcs']:
            if arc['id'] == ARC:
                arc['modifiers'].update(skill_dmg=0, ultimate_dmg=0)
        for level in cls.without_effect['arc_refinements']['arcs'][ARC]['levels'].values():
            level['panel_modifiers'].update(skill_dmg=0, ultimate_dmg=0)

    def detail(self, character_id, name, refinement=5, enabled=True, damage_element=None):
        catalog = deepcopy(self.catalog if enabled else self.without_effect)
        action = next(a for a in catalog['actions'] if a['character_id'] == character_id and a['name'] == name)
        if damage_element:
            action['damage_element'] = damage_element
        payload = {'team': [{'slot': 0, 'character_id': character_id, 'arc_id': ARC,
                             'arc_refinement': refinement, 'cartridge_id': '', 'awakening_nodes': [],
                             'awakening_resonances': [],
                             'skill_levels': {'basic': 1, 'skill': 1, 'ultimate': 1, 'support': 1}}],
                   'steps': [{'id': 'probe', 'slot': 0, 'action_id': action['id'], 'start_tick': 0}],
                   'options': {}}
        result = subprocess.run(['node', str(ROOT / 'scripts/shaft_compute.js')],
                                input=json.dumps({'axis': normalize_axis_payload(payload), 'catalog': catalog}),
                                capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        self.assertTrue(output['ok'], output)
        return output['result']['details'][0]

    def test_akane_phase_e_and_q_do_not_receive_spirit_bonus(self):
        for refinement in range(1, 6):
            for name in ('e', 'q1'):
                with self.subTest(refinement=refinement, action=name):
                    active = self.detail('char_akane', name, refinement)
                    control = self.detail('char_akane', name, refinement, enabled=False)
                    self.assertEqual(active['damage_element'], '相')
                    self.assertAlmostEqual(active['panel']['atk'], control['panel']['atk'])
                    self.assertAlmostEqual(active['formula_parts']['dmg_bonus'], control['formula_parts']['dmg_bonus'])
                    self.assertAlmostEqual(active['direct_damage'], control['direct_damage'])

    def test_spirit_e_and_q_keep_the_bonus_at_every_refinement(self):
        for refinement, bonus in enumerate((.2, .2375, .275, .3125, .35), start=1):
            for name in ('e', 'q'):
                with self.subTest(refinement=refinement, action=name):
                    active = self.detail('char_dda9fae1ef', name, refinement)
                    control = self.detail('char_dda9fae1ef', name, refinement, enabled=False)
                    self.assertEqual(active['damage_element'], '灵')
                    self.assertAlmostEqual(active['panel']['atk'], control['panel']['atk'])
                    self.assertAlmostEqual(active['formula_parts']['dmg_bonus'] - control['formula_parts']['dmg_bonus'], bonus)

    def test_spirit_basic_and_support_do_not_receive_e_q_bonus(self):
        for name in ('a1', '援护'):
            with self.subTest(action=name):
                active = self.detail('char_dda9fae1ef', name)
                control = self.detail('char_dda9fae1ef', name, enabled=False)
                self.assertAlmostEqual(active['formula_parts']['dmg_bonus'], control['formula_parts']['dmg_bonus'])
                self.assertAlmostEqual(active['direct_damage'], control['direct_damage'])

    def test_bonus_uses_actual_damage_element(self):
        active = self.detail('char_akane', 'e', damage_element='灵')
        control = self.detail('char_akane', 'e', enabled=False, damage_element='灵')
        self.assertEqual(active['damage_element'], '灵')
        self.assertAlmostEqual(active['formula_parts']['dmg_bonus'] - control['formula_parts']['dmg_bonus'], .35)
        active = self.detail('char_dda9fae1ef', 'e', damage_element='相')
        control = self.detail('char_dda9fae1ef', 'e', enabled=False, damage_element='相')
        self.assertEqual(active['damage_element'], '相')
        self.assertAlmostEqual(active['formula_parts']['dmg_bonus'], control['formula_parts']['dmg_bonus'])

    def test_refinement_import_preserves_elemental_condition(self):
        spec = importlib.util.spec_from_file_location('paperplane_refinement_import',
            ROOT / 'scripts/import_shaft_nanoka_refinements.py')
        importer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(importer)
        weapon = {'id': 'fork_PaperPlane',
                  'stats': [{'id_stats': 'AtkUp', 'values': [25], 'b_is_percent': True}],
                  'effect': {'values': [{'values': ['20%', '23.75%', '27.5%', '31.25%', '35%']}]}}
        for level, bonus in enumerate((.2, .2375, .275, .3125, .35), start=1):
            panel, elements = importer.level_panel(weapon, level)
            self.assertEqual(panel, {'atk_pct': .25})
            self.assertEqual(elements, {})
            self.assertEqual(importer.buff_values(weapon, 'arc_paperplane_spirit_eq', level), {'all_dmg': bonus})
