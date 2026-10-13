"""Food limits and provisional blessing candidates supplied by user, 2026-10-13."""
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis

ROOT = Path(__file__).resolve().parents[1]
FOOD = dict(food_crit_rate=.12, food_crit_dmg=.24, food_flat_atk=150)
BLESSINGS = dict(crit_rate=.1, crit_dmg=.2, all_dmg=.15, atk_pct=.15,
                 stagger_strength=60, harmony_strength=60)


class ShaftFoodBlessingTest(unittest.TestCase):
    def simulate(self, bonus):
        return simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_d38b672525', 'arc_id': '',
                      'cartridge_id': '', 'substat_counts': {}, 'bond_full': False,
                      'skill_levels': {'basic': 1}}],
            'steps': [{'id': 'a', 'slot': 0, 'action_id': 'action_48204bed53', 'start_tick': 0}],
            'team_panel_bonus': {'version': 3, **bonus},
        })['result']

    def test_defaults_caps_and_invalid_selection(self):
        bonus = normalize_axis_payload({'team_panel_bonus': {
            'version': 3, 'furniture_flat_atk': 13, 'training_flat_atk': 0,
        }})['team_panel_bonus']
        self.assertEqual({k: bonus[k] for k in FOOD}, dict.fromkeys(FOOD, 0))
        self.assertEqual(bonus['blessing'], 'none')
        self.assertEqual(bonus['furniture_flat_atk'], 13)
        self.assertEqual(bonus['training_flat_atk'], 0)
        for value, expected in [(999, FOOD), (-1, dict.fromkeys(FOOD, 0))]:
            bonus = normalize_axis_payload({'team_panel_bonus': dict.fromkeys(FOOD, value)})['team_panel_bonus']
            self.assertEqual({k: bonus[k] for k in FOOD}, expected)
        for raw in [None, [], ['crit_rate', 'crit_dmg'], {}, 'unknown', '__proto__', 'constructor']:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_axis_payload({'team_panel_bonus': {'blessing': raw}})
                                 ['team_panel_bonus']['blessing'], 'none')

    def test_food_and_each_blessing_add_only_the_selected_stats(self):
        baseline = self.simulate({})
        base = baseline['build_panels_by_slot'][0]
        for selection in ['none', *BLESSINGS]:
            with self.subTest(selection=selection):
                result = self.simulate({**FOOD, 'blessing': selection, 'blessing_crit_dmg': 999})
                panel = result['build_panels_by_slot'][0]['panel']
                expected = dict(base['panel'])
                expected['crit_rate'] += .12
                expected['crit_dmg'] += .24
                expected['atk'] += 150
                if selection == 'atk_pct':
                    expected['atk'] += base['base_stats']['atk'] * .15
                elif selection != 'none':
                    expected[selection] += BLESSINGS[selection]
                for stat in expected:
                    self.assertAlmostEqual(panel[stat], expected[stat], msg=stat)
                self.assertGreater(result['details'][0]['direct_damage'], baseline['details'][0]['direct_damage'])
                self.assertEqual(result['summary']['total_stagger'], baseline['summary']['total_stagger'])
        # Selecting again replaces the first effect; it cannot retain an old blessing.
        switched = normalize_axis_payload({'team_panel_bonus': {**FOOD, 'blessing': 'crit_dmg'}})
        switched['team_panel_bonus']['blessing'] = 'atk_pct'
        loaded = normalize_axis_payload(json.loads(json.dumps(switched)))
        self.assertEqual(loaded['team_panel_bonus']['blessing'], 'atk_pct')

    def test_raw_input_normalization_agrees_in_ui_engine_and_server(self):
        source = (ROOT / 'app/modules/shaft/static/js/shaft.js').read_text()
        constants = source[source.index('  const DEFAULT_TEAM_PANEL_BONUS ='):source.index('  const SKILL_LEVEL_FIELDS =')]
        normalizer = source[source.index('  function normalizeTeamPanelBonus('):source.index('  function teamPanelBonusMods(')]
        cases = [
            {}, {'version': 3, 'food_crit_rate': .0065, 'food_crit_dmg': 999, 'food_flat_atk': 2.5},
            *[{'version': 3, **FOOD, 'blessing': choice} for choice in ['none', *BLESSINGS, [], {}, 'constructor']],
        ]
        ui_script = ('const numberValue=(v,d=0)=>Number.isFinite(Number(v))&&v!=null?Number(v):d;'
                     + constants + normalizer + '\nconsole.log(JSON.stringify(' + json.dumps(cases)
                     + '.map(normalizeTeamPanelBonus)));')
        ui = json.loads(subprocess.check_output(['node', '-e', ui_script], text=True))
        engine_script = ('const data=JSON.parse(require("fs").readFileSync(0,"utf8"));'
                         'const engine=require(' + json.dumps(str(ROOT / 'app/modules/shaft/static/js/shaft_engine.js')) + ');'
                         'console.log(JSON.stringify(data.axes.map(axis=>engine.simulateAxis(axis,data.catalog).build_panels_by_slot)));')
        axes = []
        expected = []
        for raw, frontend in zip(cases, ui):
            normalized = normalize_axis_payload({'team_panel_bonus': raw})
            self.assertEqual(frontend, normalized['team_panel_bonus'])
            expected.append(simulate_shaft_axis(normalized)['result']['build_panels_by_slot'])
            normalized['team_panel_bonus'] = raw
            axes.append(normalized)
        output = subprocess.run(['node', '-e', engine_script], check=True, capture_output=True, text=True,
                                input=json.dumps({'axes': axes, 'catalog': load_shaft_catalog()}))
        self.assertEqual(json.loads(output.stdout), expected)
