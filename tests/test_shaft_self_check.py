import json
import shutil
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog


ROOT = Path(__file__).resolve().parents[1]
SELF_CHECK_JS = ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft_self_check.js'


@unittest.skipUnless(shutil.which('node'), 'node is required for shaft frontend self-check tests')
class ShaftSelfCheckTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_shaft_catalog()

    def inspect(self, steps: list[dict], details: list[dict] | None = None) -> list[str]:
        script = """
const selfCheck = require(process.argv[1]);
const payload = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(selfCheck.inspectAxis(payload.axis, payload.catalog, payload.details)));
"""
        payload = {
            'axis': {'steps': steps},
            'catalog': {'actions': self.catalog['actions']},
            'details': details or [],
        }
        completed = subprocess.run(
            ['node', '-e', script, str(SELF_CHECK_JS), json.dumps(payload, ensure_ascii=False)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def action(self, character: str, name: str) -> dict:
        return next(
            action for action in self.catalog['actions']
            if action.get('character_name') == character and action.get('name') == name
        )

    def step(self, action: dict, tick: int, slot: int = 0) -> dict:
        return {'id': f"{action['id']}-{tick}", 'action_id': action['id'], 'slot': slot, 'start_tick': tick}

    def test_warns_for_adjacent_descending_basic_stages_except_reset_to_first_stage(self) -> None:
        a4 = self.action('真红', 'a4')
        a2 = self.action('真红', 'a2')
        a1 = self.action('真红', 'a1')

        self.assertEqual(self.inspect([self.step(a4, 0), self.step(a2, 2)]), ['真红存在普攻段数异常'])
        self.assertEqual(self.inspect([self.step(a4, 0), self.step(a1, 2)]), [])

    def test_other_action_between_basic_stages_prevents_warning(self) -> None:
        a4 = self.action('真红', 'a4')
        a2 = self.action('真红', 'a2')
        q = self.action('真红', 'q')

        self.assertEqual(self.inspect([self.step(a4, 0), self.step(q, 2), self.step(a2, 4)]), [])

    def test_requiem_does_not_require_manual_nightmare_action(self) -> None:
        actions = [
            self.action('安魂曲', 'a1近'),
            self.action('安魂曲', 'e'),
            self.action('安魂曲', 'q'),
        ]
        steps = [self.step(action, index * 10) for index, action in enumerate(actions)]
        self.assertEqual(self.inspect(steps), [])

    def test_warns_when_character_returns_to_foreground_within_twelve_ticks(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        zhenhong_a2 = self.action('真红', 'a2')

        warnings = self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(zhenhong_a2, 10, slot=0),
        ])

        self.assertEqual(
            warnings,
            ['真红「a2」：角色切换 CD 尚未结束，需等到 1.4s。'],
        )

    def test_allows_character_to_return_after_twelve_ticks(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        zhenhong_a2 = self.action('真红', 'a2')

        self.assertEqual(self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(zhenhong_a2, 14, slot=0),
        ]), [])

    def test_zero_duration_foreground_q_between_switches_allows_early_return(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        nanali_q = self.action('娜娜莉', 'q')
        zhenhong_a2 = self.action('真红', 'a2')

        self.assertEqual(self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(nanali_q, 4, slot=1),
            self.step(zhenhong_a2, 6, slot=0),
        ]), [])

    def test_other_characters_q_on_departure_node_allows_early_return(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_q = self.action('娜娜莉', 'q')
        zhenhong_a2 = self.action('真红', 'a2')

        self.assertEqual(self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_q, 2, slot=1),
            self.step(zhenhong_a2, 6, slot=0),
        ]), [])

    def test_any_characters_zero_duration_q_between_switches_allows_early_return(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        iloy_q = self.action('伊洛伊', 'q')
        zhenhong_a2 = self.action('真红', 'a2')

        self.assertEqual(self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(iloy_q, 4, slot=2),
            self.step(zhenhong_a2, 6, slot=0),
        ]), [])

    def test_positive_duration_q_does_not_allow_early_return(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        zhenhong_traverse = self.action('真红', '穿梭')

        warnings = self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(zhenhong_traverse, 6, slot=0),
        ])

        self.assertIn('真红「穿梭」：角色切换 CD 尚未结束，需等到 1.4s。', warnings)

    def test_zero_duration_q_on_return_node_is_not_between_switch_nodes(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_e = self.action('娜娜莉', 'e')
        zhenhong_q = self.action('真红', 'q')

        warnings = self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(nanali_e, 2, slot=1),
            self.step(zhenhong_q, 6, slot=0),
        ])

        self.assertIn('真红「q」：角色切换 CD 尚未结束，需等到 1.4s。', warnings)

    def test_departure_q_uses_calculation_ticks_and_exempts_switch_cooldown(self) -> None:
        requiem_a4 = self.action('安魂曲', 'a4脱手')
        iloy_q = self.action('伊洛伊', 'q')
        iloy_z1 = self.action('伊洛伊', 'z1')
        requiem_a1 = self.action('安魂曲', 'a1远')
        steps = [
            self.step(requiem_a4, 61, slot=0),
            self.step(iloy_q, 66, slot=2),
            self.step(iloy_z1, 71, slot=2),
            self.step(requiem_a1, 73, slot=0),
        ]
        calculation_ticks = [23, 28, 28, 30]
        details = [
            {
                'step_id': step['id'],
                'start_tick': calculation_tick,
                'visual_start_tick': step['start_tick'],
                'calculation_start_sequence': 0,
            }
            for step, calculation_tick in zip(steps, calculation_ticks)
        ]

        self.assertEqual(self.inspect(steps, details), [])


if __name__ == '__main__':
    unittest.main()
