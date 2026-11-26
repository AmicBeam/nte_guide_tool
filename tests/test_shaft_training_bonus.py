"""噗咔特训 defaults/caps are user-specified on 2026-10-09, not upstream data."""
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from app.modules.shaft.domain.catalog import load_shaft_catalog
from tests import test_shaft_simulator as simulator_tests


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(training_crit_rate=.02, training_crit_dmg=.04,
                training_flat_atk=23, training_flat_def=25, training_flat_hp=340,
                training_harmony_strength=10, training_stagger_strength=10)
CAPS = dict(training_crit_rate=.025, training_crit_dmg=.05,
            training_flat_atk=28, training_flat_def=30, training_flat_hp=420,
            training_harmony_strength=12, training_stagger_strength=12)


class ShaftTrainingBonusTest(unittest.TestCase):
    def test_old_axis_gets_training_defaults_without_resetting_furniture(self):
        axis = normalize_axis_payload({'team_panel_bonus': {
            'version': 3, 'furniture_crit_dmg': .026, 'furniture_flat_atk': 12,
            'furniture_flat_def': 17,
        }})
        bonus = axis['team_panel_bonus']
        self.assertEqual({k: bonus[k] for k in DEFAULTS}, DEFAULTS)
        self.assertEqual(bonus['furniture_crit_dmg'], .026)
        self.assertEqual(bonus['furniture_flat_atk'], 12)
        self.assertEqual(bonus['furniture_flat_def'], 17)

    def test_zero_values_caps_and_serialization(self):
        for value, expected in [(0, dict.fromkeys(CAPS, 0)), (-10, dict.fromkeys(CAPS, 0)), (999, CAPS)]:
            with self.subTest(value=value):
                normalized = normalize_axis_payload({'team_panel_bonus': {'version': 3, **dict.fromkeys(CAPS, value)}})
                self.assertEqual({k: normalized['team_panel_bonus'][k] for k in CAPS}, expected)
                loaded = normalize_axis_payload(json.loads(json.dumps(normalized)))
                self.assertEqual(loaded['team_panel_bonus'], normalized['team_panel_bonus'])

    def test_all_seven_bonuses_add_to_each_team_panel(self):
        payload = {'team_panel_bonus': deepcopy(simulator_tests.ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS)}
        baseline = simulate_shaft_axis(payload)['result']['build_panels_by_slot']
        payload['team_panel_bonus'].update(DEFAULTS)
        boosted = simulate_shaft_axis(payload)['result']['build_panels_by_slot']
        deltas = dict(crit_rate=.02, crit_dmg=.04, atk=23, def_=25, hp=340,
                      harmony_strength=10, stagger_strength=10)
        self.assertTrue(baseline)
        for before, after in zip(baseline, boosted):
            for key, delta in deltas.items():
                stat = 'def' if key == 'def_' else key
                self.assertAlmostEqual(after['panel'][stat] - before['panel'][stat], delta)

    def test_ui_and_engine_normalization_match_server(self):
        # Exercise the real frontend normalizer with its actual defaults/fields.
        source = (ROOT / 'app/modules/shaft/static/js/shaft.js').read_text()
        constants = source[source.index('  const DEFAULT_TEAM_PANEL_BONUS ='):source.index('  const SKILL_LEVEL_FIELDS =')]
        normalizer = source[source.index('  function normalizeTeamPanelBonus('):source.index('  function teamPanelBonusMods(')]
        helpers = 'const numberValue=(v,d=0)=>Number.isFinite(Number(v))&&v!=null?Number(v):d;\n'
        raw = {'version': 3, **CAPS, 'training_flat_atk': 2.5, 'training_crit_rate': .0065}
        script = helpers + constants + normalizer + '\nconsole.log(JSON.stringify(normalizeTeamPanelBonus(' + json.dumps(raw) + ')));'
        ui = json.loads(subprocess.check_output(['node', '-e', script], text=True))
        server = normalize_axis_payload({'team_panel_bonus': raw})['team_panel_bonus']
        self.assertEqual(ui, server)
        payload = normalize_axis_payload({'team_panel_bonus': raw})
        expected = simulate_shaft_axis(payload)['result']['build_panels_by_slot']
        payload['team_panel_bonus'] = raw
        script = ('const data=JSON.parse(require("fs").readFileSync(0,"utf8"));'
                  'const engine=require(' + json.dumps(str(ROOT / 'app/modules/shaft/static/js/shaft_engine.js')) + ');'
                  'console.log(JSON.stringify(engine.simulateAxis(data.axis,data.catalog).build_panels_by_slot));')
        result = subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True,
                                input=json.dumps({'axis': payload, 'catalog': load_shaft_catalog()}))
        self.assertEqual(json.loads(result.stdout), expected)
