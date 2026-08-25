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

    def inspect(
        self,
        steps: list[dict],
        details: list[dict] | None = None,
        *,
        team: list[dict] | None = None,
        options: dict | None = None,
        reaction_effects: list[dict] | None = None,
    ) -> list[str]:
        script = """
const selfCheck = require(process.argv[1]);
const payload = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(selfCheck.inspectAxis(payload.axis, payload.catalog, payload.result)));
"""
        payload = {
            'axis': {'steps': steps, 'team': team or [], 'options': options or {}},
            'catalog': {
                'actions': self.catalog['actions'],
                'characters': self.catalog['characters'],
                'formula_constants': self.catalog['formula_constants'],
            },
            'result': {'details': details or [], 'reaction_effects': reaction_effects or []},
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
        self.assertEqual(
            self.inspect(steps),
            ['安魂曲「a1近」：该动作的时长未实装。'],
        )

    def test_warns_once_for_each_zero_duration_non_q_foreground_action(self) -> None:
        requiem_a2 = self.action('安魂曲', 'a2近')
        xun_falling = self.action('浔', '跳A')

        self.assertEqual(
            self.inspect([
                self.step(requiem_a2, 0),
                self.step(requiem_a2, 5),
                self.step(xun_falling, 10, slot=1),
            ]),
            [
                '安魂曲「a2近」：该动作的时长未实装。',
                '浔「跳A」：该动作的时长未实装。',
            ],
        )

    def test_zero_duration_q_and_instant_switch_do_not_warn_for_missing_duration(self) -> None:
        zhenhong_q = self.action('真红', 'q')
        instant_switch = self.action('真红', '无')

        self.assertEqual(self.inspect([
            self.step(zhenhong_q, 0),
            self.step(instant_switch, 5),
        ]), [])

    def test_background_instant_switch_still_participates_in_return_cooldown(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        nanali_switch = self.action('娜娜莉', '无')
        zhenhong_a2 = self.action('真红', 'a2')
        switch_step = self.step(nanali_switch, 2, slot=1)
        switch_step['placement'] = 'background'

        self.assertIn(
            '真红「a2」：角色切换 CD 尚未结束，需等到 1.4s。',
            self.inspect([
                self.step(zhenhong_a1, 0, slot=0),
                switch_step,
                self.step(zhenhong_a2, 6, slot=0),
            ]),
        )

    def test_manual_background_zero_duration_action_does_not_warn_for_missing_duration(self) -> None:
        requiem_a2 = self.action('安魂曲', 'a2近')
        step = self.step(requiem_a2, 0)
        step['placement'] = 'background'

        self.assertEqual(self.inspect([step]), [])

    def test_invalid_background_placement_cannot_hide_missing_foreground_duration(self) -> None:
        xun_falling = self.action('浔', '跳A')
        step = self.step(xun_falling, 0)
        step['placement'] = 'background'

        self.assertEqual(
            self.inspect([step]),
            ['浔「跳A」：该动作的时长未实装。'],
        )

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

    def test_lingke_time_stop_select_masks_early_return_without_being_q(self) -> None:
        zhenhong_a1 = self.action('真红', 'a1')
        lingke_e = self.action('灵可', 'e')
        time_stop_select = self.action('灵可', '时停选人')
        zhenhong_a2 = self.action('真红', 'a2')

        self.assertEqual(time_stop_select['action_type'], '无')
        self.assertTrue(time_stop_select['is_timeline_mask'])
        self.assertEqual(self.inspect([
            self.step(zhenhong_a1, 0, slot=0),
            self.step(lingke_e, 2, slot=1),
            self.step(time_stop_select, 4, slot=1),
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
        requiem_a4 = self.action('安魂曲', 'a4远')
        iloy_q = self.action('伊洛伊', 'q')
        iloy_z1 = self.action('伊洛伊', 'z1')
        requiem_a1 = self.action('安魂曲', 'a1远')
        steps = [
            self.step(requiem_a4, 61, slot=0),
            self.step(iloy_q, 66, slot=2),
            self.step(iloy_z1, 71, slot=2),
            self.step(requiem_a1, 73, slot=0),
        ]
        steps[0]['detached'] = True
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

    def test_warns_when_multiple_characters_carry_the_same_non_overlapping_reaction(self) -> None:
        team = [
            {'slot': 0, 'character_id': 'char_076a1f4e53'},
            {'slot': 1, 'character_id': 'char_dd034941ef'},
        ]
        warnings = self.inspect([], team=team, options={
            'loop_enabled': True,
            'loop_initial_resources': {
                'char_076a1f4e53': {'reaction': '浊燃'},
                'char_dd034941ef': {'reaction': '浊燃'},
            },
        })

        self.assertEqual(len(warnings), 1)
        self.assertIn('循环轴自带环合「浊燃」冲突', warnings[0])
        self.assertIn('最多同时存在 1 个实例', warnings[0])

    def test_warns_when_axis_also_produces_a_carried_reaction_even_if_it_can_be_refreshed(self) -> None:
        team = [{'slot': 0, 'character_id': 'char_076a1f4e53'}]
        warnings = self.inspect(
            [],
            details=[{
                'step_id': 'support',
                'triggered_reaction': {
                    'reaction': '浊燃',
                    'trigger_character_id': 'char_076a1f4e53',
                    'trigger_character_name': '残虹',
                    'frequency_multiplier': 3,
                    'loop_initial': True,
                },
            }],
            team=team,
            options={
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_076a1f4e53': {'reaction': '浊燃'},
                },
            },
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn('残虹自带，且轴内由残虹产生', warnings[0])
        self.assertIn('刷新或额外加层能力不放宽此自检', warnings[0])

    def test_genesis_allows_three_instances_but_warns_on_the_fourth(self) -> None:
        team = [
            {'slot': index, 'character_id': character_id}
            for index, character_id in enumerate([
                'char_076a1f4e53',
                'char_dd034941ef',
                'char_c78f7a08d5',
                'char_1895e259be',
            ])
        ]
        three_resources = {
            member['character_id']: {'reaction': '创生'}
            for member in team[:3]
        }
        self.assertEqual(self.inspect([], team=team, options={
            'loop_enabled': True,
            'loop_initial_resources': three_resources,
        }), [])

        four_resources = {
            member['character_id']: {'reaction': '创生'}
            for member in team
        }
        warnings = self.inspect([], team=team, options={
            'loop_enabled': True,
            'loop_initial_resources': four_resources,
        })
        self.assertEqual(len(warnings), 1)
        self.assertIn('最多同时存在 3 个实例', warnings[0])

        generated_warnings = self.inspect(
            [],
            details=[{
                'step_id': 'support',
                'triggered_reaction': {
                    'reaction': '创生',
                    'trigger_character_name': '主角',
                },
            }],
            team=team[:3],
            options={
                'loop_enabled': True,
                'loop_initial_resources': three_resources,
            },
        )
        self.assertEqual(len(generated_warnings), 1)
        self.assertIn('最多同时存在 3 个实例', generated_warnings[0])

    def test_does_not_warn_for_distinct_carried_reactions_or_non_loop_axis(self) -> None:
        team = [
            {'slot': 0, 'character_id': 'char_076a1f4e53'},
            {'slot': 1, 'character_id': 'char_dd034941ef'},
        ]
        resources = {
            'char_076a1f4e53': {'reaction': '浊燃'},
            'char_dd034941ef': {'reaction': '创生'},
        }
        self.assertEqual(self.inspect([], team=team, options={
            'loop_enabled': True,
            'loop_initial_resources': resources,
        }), [])
        self.assertEqual(self.inspect([], team=team, options={
            'loop_enabled': False,
            'loop_initial_resources': {
                'char_076a1f4e53': {'reaction': '浊燃'},
                'char_dd034941ef': {'reaction': '浊燃'},
            },
        }), [])


if __name__ == '__main__':
    unittest.main()
