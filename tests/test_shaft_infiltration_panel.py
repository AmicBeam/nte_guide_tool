import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.service import simulate_shaft_axis


class InfiltrationPanelTest(unittest.TestCase):
    def render(self, details):
        source = (Path(__file__).resolve().parents[1] / 'app/modules/shaft/static/js/shaft.js').read_text()
        helper = source[source.index('  function infiltrationVulnerabilityRow('):source.index('  function renderActionDetail(')]
        script = 'const formatNumber = (n, digits) => n.toFixed(digits);\n' + helper
        script += '\nconsole.log(JSON.stringify(' + json.dumps(details) + '.map(infiltrationVulnerabilityRow)));'
        return json.loads(subprocess.check_output(['node', '-e', script], text=True))

    def test_actual_reaction_and_expiry(self):
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_912dbfe17c', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'start', 'slot': 0, 'action_id': 'action_none_caa6c2e5a8', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_f229587fd2', 'start_tick': 20},
                {'id': 'inside', 'slot': 1, 'action_id': 'action_61c61302f2', 'start_tick': 60},
                {'id': 'expired', 'slot': 1, 'action_id': 'action_61c61302f2', 'start_tick': 200},
            ],
        })['result']
        inside, expired = [next(d for d in result['details'] if d['step_id']==id) for id in ('inside', 'expired')]
        rendered = self.render([inside, expired])
        actual = inside['formula_parts']['non_fuwen_amplification']
        self.assertGreater(actual, 1)
        self.assertIn('浸染易伤', rendered[0])
        self.assertIn(f'+{(actual - 1) * 100:.1f}%', rendered[0])
        self.assertEqual(rendered[1], '')

    def test_marker_value_and_unaffected_or_legacy_results(self):
        # Markers use the same formula_parts through renderDamageEventDetail.
        rows = self.render([{'formula_parts': {'non_fuwen_amplification': 1.35}},
                            {'formula_parts': {'non_fuwen_amplification': 1, 'fuwen_amplification': 1.3}}, {},
                            {'formula_parts': {'non_fuwen_amplification': 'invalid'}}])
        self.assertIn('+35.0%', rows[0])
        self.assertEqual(rows[1:], ['', '', ''])
