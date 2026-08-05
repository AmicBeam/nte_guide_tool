import unittest
from copy import deepcopy

from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis
from app.modules.shaft.domain.buffs import activate_buff, registered_buff_rules
from app.modules.shaft.domain.catalog import load_shaft_catalog


class ShaftSimulatorValidationTestCase(unittest.TestCase):
    ZERO_TEAM_PANEL_BONUS = {
        'version': 3,
        'furniture_crit_dmg': 0,
        'furniture_flat_atk': 0,
        'furniture_flat_def': 0,
        'small_flat_atk': 0,
        'small_flat_hp': 0,
    }

    def test_every_character_has_zero_second_noop_switch_action(self) -> None:
        catalog = load_shaft_catalog()
        actions_by_character = catalog['actions_by_character']

        for character in catalog['characters']:
            with self.subTest(character=character['name']):
                noop = next(
                    action
                    for action in actions_by_character[character['id']]
                    if action['name'] == '无'
                )
                self.assertEqual(noop['action_type'], '无')
                self.assertEqual(noop['damage_type'], '无')
                self.assertEqual(noop['duration_ticks'], 0)
                self.assertEqual(noop['hit_count'], 0)
                self.assertTrue(noop['is_instant_switch'])
                self.assertNotIn('triggers_reaction_on_switch', noop)
                self.assertEqual(actions_by_character[character['id']][-1]['name'], '无')

    def test_characters_without_bond_bonus_cannot_enable_bond(self) -> None:
        for character_id in ('char_dd034941ef', 'char_076a1f4e53'):
            with self.subTest(character_id=character_id):
                normalized = normalize_axis_payload({
                    'team': [{
                        'slot': 0,
                        'character_id': character_id,
                        'arc_id': '',
                        'cartridge_id': '',
                        'bond_level': 1,
                        'bond_full': True,
                    }],
                    'steps': [],
                })
                member = normalized['team'][0]
                build = normalized['character_builds'][character_id]
                self.assertEqual(member['bond_level'], 0)
                self.assertFalse(member['bond_full'])
                self.assertEqual(build['bond_level'], 0)
                self.assertFalse(build['bond_full'])

    def test_canhong_starts_non_loop_axis_with_full_harmony_only_once(self) -> None:
        base_payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'noop',
                'slot': 0,
                'action_id': 'action_none_076a1f4e53',
                'start_tick': 0,
            }],
            'initial_energy': 0,
        }
        non_loop = simulate_shaft_axis(base_payload)['result']['resources_by_slot'][0]
        self.assertEqual(non_loop['initial_harmony'], 100)
        self.assertEqual(non_loop['harmony'], 100)

        loop_payload = deepcopy(base_payload)
        loop_payload['options'] = {'loop_enabled': True}
        loop = simulate_shaft_axis(loop_payload)['result']['resources_by_slot'][0]
        self.assertEqual(loop['initial_harmony'], 0)

        configured_payload = deepcopy(loop_payload)
        configured_payload['options']['loop_initial_resources'] = {
            'char_076a1f4e53': {'energy': 0, 'harmony': 40, 'personal_resources': {}},
        }
        configured = simulate_shaft_axis(configured_payload)['result']['resources_by_slot'][0]
        self.assertEqual(configured['initial_harmony'], 40)

    def test_canhong_e_variants_share_five_second_cooldown_in_self_check(self) -> None:
        def simulate_pair(first_action_id: str, second_action_id: str, second_start_tick: int) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_076a1f4e53',
                    'arc_id': '',
                    'cartridge_id': '',
                }],
                'steps': [
                    {'id': 'first', 'slot': 0, 'action_id': first_action_id, 'start_tick': 0},
                    {'id': 'second', 'slot': 0, 'action_id': second_action_id, 'start_tick': second_start_tick},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']['details'][-1]

        early_enhanced_e = simulate_pair('action_canhong_e1', 'action_canhong_E1', 30)
        ready_illusion_e = simulate_pair('action_canhong_e1', 'action_canhong_E2', 50)
        early_normal_e = simulate_pair('action_canhong_E2', 'action_canhong_e1', 40)

        self.assertTrue(any('CD 尚未结束' in warning for warning in early_enhanced_e['warnings']))
        self.assertFalse(any('CD 尚未结束' in warning for warning in ready_illusion_e['warnings']))
        self.assertTrue(any('CD 尚未结束' in warning for warning in early_normal_e['warnings']))

    def test_canhong_illusion_attacks_can_be_manually_placed_in_background(self) -> None:
        action_ids = (
            'action_canhong_illusion_a1',
            'action_canhong_illusion_a2',
            'action_canhong_illusion_a3',
            'action_canhong_illusion_a4',
        )
        actions = {action['id']: action for action in load_shaft_catalog()['actions']}

        for action_id in action_ids:
            with self.subTest(action_id=action_id):
                self.assertTrue(actions[action_id]['can_background_override'])
                payload = {
                    'team': [{
                        'slot': 0,
                        'character_id': 'char_076a1f4e53',
                        'arc_id': '',
                        'cartridge_id': '',
                    }],
                    'steps': [{
                        'id': 'illusion-a',
                        'slot': 0,
                        'action_id': action_id,
                        'start_tick': 0,
                        'placement': 'background',
                    }],
                    'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                }

                normalized = normalize_axis_payload(payload)
                detail = simulate_shaft_axis(payload)['result']['details'][0]

                self.assertEqual(normalized['steps'][0]['placement'], 'background')
                self.assertTrue(detail['is_background_damage'])
                self.assertTrue(detail['is_basic_background'])
                self.assertGreater(detail['direct_damage'], 0)

    def test_canhong_hunt_covers_all_damage_and_restores_after_illusion(self) -> None:
        base_payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 500,
        }
        action_ids = (
            'action_canhong_a1',
            'action_canhong_illusion_a1',
            'action_canhong_e1',
            'action_canhong_E2',
            'action_canhong_q',
            'action_canhong_q_enhanced',
            'action_canhong_q_blood_banquet',
        )
        for action_id in action_ids:
            with self.subTest(action_id=action_id):
                payload = deepcopy(base_payload)
                payload['steps'] = [{
                    'id': action_id,
                    'slot': 0,
                    'action_id': action_id,
                    'start_tick': 0,
                }]
                detail = simulate_shaft_axis(payload)['result']['details'][0]
                hunt = next(
                    buff for buff in detail['applied_buffs']
                    if buff['rule_id'] == 'character_canhong_hunt'
                )
                self.assertEqual(hunt['effects']['all_dmg'], 0.25)

        illusion_payload = deepcopy(base_payload)
        illusion_payload['steps'] = [
            {'id': 'enter', 'slot': 0, 'action_id': 'action_canhong_e1', 'start_tick': 0},
            {'id': 'during', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 50},
            {'id': 'after', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 90},
        ]
        details = {
            detail['step_id']: detail
            for detail in simulate_shaft_axis(illusion_payload)['result']['details']
        }
        self.assertTrue(any(
            buff['rule_id'] == 'character_canhong_hunt'
            and buff['effects'].get('all_dmg') == 0.25
            for buff in details['during']['applied_buffs']
        ))
        self.assertTrue(any(
            buff['rule_id'] == 'character_canhong_hunt_natural_restore'
            for buff in details['after']['applied_buffs']
        ))

    def test_canhong_leaving_foreground_clears_illusion_state_immediately(self) -> None:
        def simulate(awakening_nodes: list[int] | None = None) -> dict:
            return simulate_shaft_axis({
                'team': [
                    {
                        'slot': 0,
                        'character_id': 'char_076a1f4e53',
                        'awakening_nodes': awakening_nodes or [],
                        'arc_id': '',
                        'cartridge_id': '',
                    },
                    {
                        'slot': 1,
                        'character_id': 'char_c78f7a08d5',
                        'arc_id': '',
                        'cartridge_id': '',
                    },
                ],
                'steps': [
                    {'id': 'enter', 'slot': 0, 'action_id': 'action_canhong_e1', 'start_tick': 0},
                    {'id': 'leave', 'slot': 1, 'action_id': 'action_2745f804a5', 'start_tick': 20},
                    {
                        'id': 'illusion-background',
                        'slot': 0,
                        'action_id': 'action_canhong_illusion_a1',
                        'start_tick': 21,
                        'placement': 'background',
                    },
                    {
                        'id': 'illusion-foreground',
                        'slot': 0,
                        'action_id': 'action_canhong_illusion_a1',
                        'start_tick': 30,
                    },
                ],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        result = simulate()
        enter = next(detail for detail in result['details'] if detail['step_id'] == 'enter')
        leave = next(detail for detail in result['details'] if detail['step_id'] == 'leave')
        background = next(detail for detail in result['details'] if detail['step_id'] == 'illusion-background')
        foreground = next(detail for detail in result['details'] if detail['step_id'] == 'illusion-foreground')
        illusion_state = next(
            buff for buff in enter['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_illusion_state'
        )
        restored_hunt = next(
            buff for buff in leave['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_hunt'
            and buff['start_tick'] == 20
        )
        delayed_delusion = next(
            buff for buff in leave['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_delusion'
            and buff['start_tick'] == 20
        )
        clear = next(
            buff for buff in leave['triggered_buffs']
            if buff['rule_id'] == 'character_canhong_illusion_state_foreground_leave_clear'
        )

        self.assertEqual(illusion_state['end_tick'], 20)
        self.assertGreater(restored_hunt['end_tick'], 100000000)
        self.assertEqual(delayed_delusion['end_tick'], 100)
        self.assertIn('character_canhong_illusion_state', clear['cleared_buff_keys'])
        self.assertFalse(any(
            buff['definition_id'] == 'character_canhong_illusion_state'
            for buff in background['applied_buffs']
        ))
        self.assertNotIn('动作需要处于 幻境状态 状态。', background['warnings'])
        self.assertIn('动作需要处于 幻境状态 状态。', foreground['warnings'])

        c_result = simulate([3])
        c_leave = next(detail for detail in c_result['details'] if detail['step_id'] == 'leave')
        c_regeneration = next(
            buff for buff in c_leave['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_c_delusion_regeneration'
            and buff['start_tick'] == 20
        )
        self.assertEqual(c_regeneration['end_tick'], 100)

    def test_canhong_active_and_natural_illusion_exit_clear_state(self) -> None:
        active_result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'enter', 'slot': 0, 'action_id': 'action_canhong_e1', 'start_tick': 0},
                {'id': 'exit', 'slot': 0, 'action_id': 'action_canhong_E2', 'start_tick': 50},
                {'id': 'after', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 51},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        active_details = {detail['step_id']: detail for detail in active_result['details']}
        self.assertNotIn('动作需要处于 幻境状态 状态。', active_details['exit']['warnings'])
        self.assertIn('动作需要处于 幻境状态 状态。', active_details['after']['warnings'])

        natural_result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'enter', 'slot': 0, 'action_id': 'action_canhong_e1', 'start_tick': 0},
                {'id': 'after', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 81},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        natural_after = next(detail for detail in natural_result['details'] if detail['step_id'] == 'after')
        self.assertIn('动作需要处于 幻境状态 状态。', natural_after['warnings'])

    def test_noop_only_switches_front_without_triggering_reaction_or_q_cover(self) -> None:
        catalog = load_shaft_catalog()
        protagonist_noop = next(
            action for action in catalog['actions_by_character']['char_dd034941ef']
            if action['name'] == '无'
        )
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'away', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 15},
                {'id': 'noop', 'slot': 0, 'action_id': protagonist_noop['id'], 'start_tick': 30},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 30},
            ],
            'options': {'switch_gap_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['noop']['switch_gap_ticks'], 0)
        self.assertEqual(details['noop']['duration_ticks'], 0)
        self.assertIsNone(details['noop']['triggered_reaction'])
        self.assertFalse(details['noop']['q_instant_release'])
        self.assertEqual(details['noop']['q_cover_target_step_ids'], [])
        self.assertEqual(details['noop']['raw_start_tick'], 30)
        self.assertEqual(details['noop']['visual_start_tick'], 30)
        self.assertEqual(details['support']['triggered_reaction']['reaction'], '创生')
        self.assertEqual(details['support']['triggered_reaction']['previous_slot'], 0)

    def test_character_awakening_effects_are_gated_by_their_nodes(self) -> None:
        catalog = load_shaft_catalog()
        rules = {rule['id']: rule for rule in catalog['buffs']}
        exact_nodes = {
            'character_edgar_q_team_def': 6,
            'character_mint_light_reaction_atk': 3,
            'character_fadia_enemy_slayer_crit': 5,
            'character_nanali_offhand_damage': 4,
            'character_haiyue_a1_huacai_crit': 1,
            'character_haiyue_a3_huacai_q_damage': 3,
            'character_haniya_a3_ace_crit': 3,
            'character_haniya_a5_ace_soul_damage': 5,
            'character_hathor_a6_emergency_crit_dmg': 6,
            'character_nanali_a1_pursuit_energy': 1,
            'character_chaos_a1_license_resource_gain': 1,
            'character_chaos_a6_refill_sin': 6,
            'character_chaos_a2_license_def_ignore': 2,
            'character_chaos_a5_license_damage': 5,
            'character_yi_a3_single_locked_target_damage': 3,
            'character_zhenhong_a5_ascendant_def': 5,
        }
        for rule_id, node in exact_nodes.items():
            with self.subTest(rule_id=rule_id):
                self.assertIn(
                    {'type': 'awakening_min', 'min': node},
                    rules[rule_id]['trigger']['conditions'],
                )

        full_awakening_rules = {
            'character_zhenhong_full_crit_def_ignore',
            'character_xun_a6_team_atk',
            'character_iloy_a6_team_crit_dmg',
            'character_nanali_a6_damage',
            'character_baizang_a6_team_atk',
            'character_haniya_ace_team_crit_dmg',
            'character_hathor_full_emergency_def_ignore',
            'character_daphne_full_dark_res_down',
            'character_chaos_full_license_res_down',
            'character_yi_full_locked_target_res_down',
            'character_jiuyuan_rose_a6_res',
            'character_sagiri_a6_negative_damage',
        }
        for rule_id in full_awakening_rules:
            with self.subTest(rule_id=rule_id):
                self.assertIn(
                    {'type': 'awakening_count_min', 'min': 6},
                    rules[rule_id]['trigger']['conditions'],
                )

        protagonist_e = next(action for action in catalog['actions'] if action['id'] == 'action_982c67944f')
        protagonist_extra = next(action for action in catalog['actions'] if action['id'] == 'action_4e1fab68e7')
        self.assertEqual(protagonist_e['energy_return'], 0)
        self.assertEqual(protagonist_e['energy_return_by_awakening_node'], {'2': 8})
        self.assertEqual(protagonist_extra['required_awakening'], 1)

    def test_simple_awakening_action_values_only_apply_with_their_nodes(self) -> None:
        def result_for(character_id: str, action_id: str, awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': character_id,
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [{'id': 'action', 'slot': 0, 'action_id': action_id, 'start_tick': 0}],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        adler_base = result_for('char_6f46705fd1', 'action_95733904eb', [])
        adler_a5 = result_for('char_6f46705fd1', 'action_95733904eb', [5])
        self.assertEqual(adler_base['resources_by_slot'][0]['energy'], 0)
        self.assertEqual(adler_a5['resources_by_slot'][0]['energy'], 20)

        mint_base = result_for('char_dda9fae1ef', 'action_6b96f24d8c', [])
        mint_full = result_for('char_dda9fae1ef', 'action_6b96f24d8c', [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(mint_base['resources_by_slot'][0]['harmony'], 10.2)
        self.assertAlmostEqual(mint_full['resources_by_slot'][0]['harmony'], 12.2)

        catalog = load_shaft_catalog()
        actions = {action['id']: action for action in catalog['actions']}
        self.assertEqual(actions['action_28759c6e0b']['required_awakening'], 4)
        self.assertEqual(actions['action_85906994df']['required_awakening'], 3)
        self.assertEqual(actions['action_b646db1098']['required_awakening'], 2)
        self.assertEqual(actions['action_db30565b20']['cooldown_ticks_by_awakening_node'], {'6': 80})

        def haniya_second_e_warnings(awakening_nodes: list[int]) -> list[str]:
            result = simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_e0a4292b4e',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [
                    {'id': 'first', 'slot': 0, 'action_id': 'action_db30565b20', 'start_tick': 0},
                    {'id': 'second', 'slot': 0, 'action_id': 'action_db30565b20', 'start_tick': 80},
                ],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']
            return next(detail for detail in result['details'] if detail['step_id'] == 'second')['warnings']

        self.assertTrue(any('CD 尚未结束' in warning for warning in haniya_second_e_warnings([])))
        self.assertFalse(any('CD 尚未结束' in warning for warning in haniya_second_e_warnings([6])))

    def test_nanali_a1_pursuit_energy_respects_shared_one_second_cooldown(self) -> None:
        def simulate(awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [
                    {'id': 'first', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 0},
                    {'id': 'within-cooldown', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 5},
                    {'id': 'after-cooldown', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 10},
                ],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 0,
            })['result']

        self.assertEqual(simulate([])['resources_by_slot'][0]['energy'], 0)
        active = simulate([1])
        details = {detail['step_id']: detail for detail in active['details']}
        self.assertAlmostEqual(details['first']['energy_after'], 2.5)
        self.assertAlmostEqual(details['within-cooldown']['energy_after'], 2.5)
        self.assertAlmostEqual(details['after-cooldown']['energy_after'], 5)
        self.assertAlmostEqual(active['resources_by_slot'][0]['energy'], 5)

    def test_chaos_a1_and_a6_use_existing_license_and_personal_resource_system(self) -> None:
        def simulate(awakening_nodes: list[int], include_q: bool = False) -> dict:
            steps = [
                {'id': 'license', 'slot': 0, 'action_id': 'action_2d3f8642dd', 'start_tick': 0},
                {'id': 'attack', 'slot': 0, 'action_id': 'action_40d0576f80', 'start_tick': 20},
            ]
            if include_q:
                steps.append({'id': 'q', 'slot': 0, 'action_id': 'action_dbd682321e', 'start_tick': 30})
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_d38b672525',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': steps,
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        base = {detail['step_id']: detail for detail in simulate([])['details']}
        a1 = {detail['step_id']: detail for detail in simulate([1])['details']}
        self.assertEqual(base['attack']['personal_resources_after']['罪状'], 122)
        self.assertAlmostEqual(a1['attack']['personal_resources_after']['罪状'], 130.8)

        a6 = {detail['step_id']: detail for detail in simulate([6], include_q=True)['details']}
        self.assertEqual(a6['q']['personal_resources_after']['罪状'], 1000)

    def test_edgar_truth_keys_and_awakened_e_cooldown_affect_self_check(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_31c5130304',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [1, 2],
                },
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'edgar-e', 'slot': 0, 'action_id': 'action_4a29818248', 'start_tick': 0},
                {'id': 'team-support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 40},
                {'id': 'too-early-e', 'slot': 0, 'action_id': 'action_4a29818248', 'start_tick': 100},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['edgar-e']['personal_resources_after']['真理之匙'], 1)
        self.assertTrue(any('19.0s' in warning for warning in details['too-early-e']['warnings']))
        edgar_resources = next(
            resource for resource in details['team-support']['resources_after_by_slot']
            if resource['slot'] == 0
        )
        self.assertEqual(edgar_resources['personal_resources']['真理之匙'], 2)

    def test_mint_a1_counter_resets_e_cooldown_for_self_check(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_dda9fae1ef',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1],
            }],
            'steps': [
                {'id': 'first-e', 'slot': 0, 'action_id': 'action_6b96f24d8c', 'start_tick': 0},
                {'id': 'counter', 'slot': 0, 'action_id': 'action_cc06c5dd84', 'start_tick': 20},
                {'id': 'second-e', 'slot': 0, 'action_id': 'action_6b96f24d8c', 'start_tick': 45},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        second_e = next(detail for detail in result['details'] if detail['step_id'] == 'second-e')
        self.assertFalse(any('CD 尚未结束' in warning for warning in second_e['warnings']))

    def test_awakened_healing_tags_are_exposed_to_buff_triggers(self) -> None:
        requiem = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': 'arc_bab179ec33',
                'cartridge_id': '',
                'awakening_nodes': [5],
            }],
            'steps': [{
                'id': 'nightmare-heal',
                'slot': 0,
                'action_id': 'action_requiem_a5_nightmare_heal',
                'start_tick': 0,
            }],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']['details'][0]
        self.assertIn('heal', requiem['action_tags'])
        self.assertIn(
            'arc_wrong_door_heal_spirit',
            {buff['definition_id'] for buff in requiem['triggered_buffs']},
        )

        for character_id, awakening_node, action_id in (
            ('char_701295143d', 4, 'action_222f577087'),
            ('char_b52cc8f160', 5, 'action_7773821d79'),
        ):
            with self.subTest(character_id=character_id):
                detail = simulate_shaft_axis({
                    'team': [{
                        'slot': 0,
                        'character_id': character_id,
                        'arc_id': '',
                        'cartridge_id': '',
                        'awakening_nodes': [awakening_node],
                    }],
                    'steps': [{'id': 'heal', 'slot': 0, 'action_id': action_id, 'start_tick': 0}],
                    'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                    'initial_energy': 200,
                })['result']['details'][0]
                self.assertIn('heal', detail['action_tags'])

    def test_daphne_awakenings_strengthen_layered_e_and_stagger_formula(self) -> None:
        def simulate(awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_e8ad982185',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [{'id': 'e', 'slot': 0, 'action_id': 'action_5206179376', 'start_tick': 0}],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        base = simulate([])
        a1 = simulate([1])
        a4 = simulate([4])
        a5 = simulate([5])
        self.assertAlmostEqual(a1['details'][0]['formula_parts']['raw_base'], base['details'][0]['formula_parts']['raw_base'] * 2)
        self.assertAlmostEqual(
            a4['summary']['stagger_damage_per_trigger'] / base['summary']['stagger_damage_per_trigger'],
            1.125,
        )
        self.assertAlmostEqual(
            a5['summary']['stagger_damage_per_trigger'] / base['summary']['stagger_damage_per_trigger'],
            1.5,
        )
        self.assertAlmostEqual(a4['stagger_contributions_by_slot'][0]['average_stagger_damage_bonus'], 0.35)
        self.assertAlmostEqual(a5['stagger_contributions_by_slot'][0]['average_stagger_damage_bonus'], 0.2)

    def test_hathor_haiyue_and_iloy_new_awakenings_use_existing_timeline_systems(self) -> None:
        hathor = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_912dbfe17c',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1, 2],
            }],
            'steps': [
                {'id': 'long-e', 'slot': 0, 'action_id': 'action_e3ab099393', 'start_tick': 0},
                {'id': 'review', 'slot': 0, 'action_id': 'action_hathor_a2_good_review', 'start_tick': 30},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        self.assertEqual(hathor['details'][0]['personal_resources_after']['闪送之力'], 6)
        self.assertEqual(hathor['details'][0]['energy_after'], 30)

        loop_hathor = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_912dbfe17c',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1],
            }],
            'steps': [{'id': 'long-e', 'slot': 0, 'action_id': 'action_e3ab099393', 'start_tick': 0}],
            'options': {'loop_enabled': True, 'loop_duration_ticks': 30},
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        self.assertNotIn('闪送之力', loop_hathor['details'][0]['personal_resources_after'])

        haiyue = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_699966e2e7',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [2],
            }],
            'steps': [
                {'id': 'support-1', 'slot': 0, 'action_id': 'action_4c11b71469', 'start_tick': 0},
                {'id': 'support-2', 'slot': 0, 'action_id': 'action_4c11b71469', 'start_tick': 30},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        huacai = next(
            buff for buff in haiyue['details'][-1]['triggered_buffs']
            if buff['definition_id'] == 'character_haiyue_huacai'
        )
        self.assertEqual(huacai['end_tick'], 115)

        iloy = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_a01c39f576',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [1, 4, 6],
                },
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'long-e', 'slot': 0, 'action_id': 'action_iloy_long_e', 'start_tick': 0},
                {'id': 'away', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 20},
                {'id': 'regen', 'slot': 0, 'action_id': 'action_iloy_a1_background_imagination', 'start_tick': 25},
                {'id': 'companion', 'slot': 0, 'action_id': 'action_iloy_regressed_morpheus_1', 'start_tick': 30},
                {'id': 'back', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 40},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        details = {detail['step_id']: detail for detail in iloy['details']}
        self.assertAlmostEqual(details['regen']['personal_resources_after']['臆想'], 21.25)
        self.assertAlmostEqual(details['companion']['formula_parts']['raw_base'] / iloy['details'][0]['panel']['atk'], 0.755)
        self.assertGreaterEqual(details['back']['harmony_after'], 50)
        self.assertGreaterEqual(details['back']['energy_after'], 50)

    def test_simple_awakening_durations_and_stack_caps_follow_nodes(self) -> None:
        catalog = load_shaft_catalog()
        rules = {rule['id']: rule for rule in catalog['buffs']}

        self.assertEqual(
            rules['character_adler_e_shield']['duration']['ticks_by_awakening_node'],
            {'6': 120},
        )
        self.assertEqual(
            rules['character_nanali_authority']['duration']['ticks_by_awakening_node'],
            {'6': 150},
        )
        self.assertEqual(
            rules['character_baizang_word_support']['stacking'],
            {
                'key': 'character_baizang_word',
                'mode': 'add_stack',
                'max_stacks': 3,
                'max_stacks_by_awakening_node': {'3': 4},
            },
        )
        self.assertEqual(
            rules['character_baizang_a6_q_crit']['duration']['ticks_by_awakening_node'],
            {'5': 160},
        )

    def test_initial_resistance_applies_to_every_damage_element(self) -> None:
        normalized = normalize_axis_payload({
            'enemy': {'initial_resistance': 0.42},
        })

        self.assertEqual(normalized['enemy']['initial_resistance'], 0.42)
        self.assertEqual(
            normalized['enemy']['resistances'],
            {element: 0.42 for element in ('光', '灵', '咒', '暗', '魂', '相', '心灵')},
        )

    def test_starter_builds_match_configured_character_defaults(self) -> None:
        starter = load_shaft_catalog()['starter_axis']
        builds = starter['character_builds']

        zhenhong = builds['char_b52cc8f160']['substat_counts']
        self.assertEqual(
            [zhenhong[key] for key in ('all_dmg', 'crit_rate', 'crit_dmg', 'atk_pct')],
            [20, 20, 20, 20],
        )
        requiem = builds['char_c78f7a08d5']['substat_counts']
        self.assertEqual(requiem['harmony_strength'], 1)

        nanali = builds['char_bdc43f82c6']
        self.assertEqual((nanali['arc_name'], nanali['arc_id']), ('预备备', 'arc_ce53905d70'))
        self.assertEqual(
            [nanali['substat_counts'][key] for key in ('all_dmg', 'crit_rate', 'crit_dmg', 'atk_pct')],
            [20, 20, 20, 20],
        )
        jiuyuan = builds['char_b2e3b2bf7a']
        self.assertEqual(
            (jiuyuan['cartridge_name'], jiuyuan['cartridge_id']),
            ('森林萤火之心', 'cartridge_29793225a0'),
        )
        self.assertEqual(
            [jiuyuan['substat_counts'][key] for key in ('all_dmg', 'crit_rate', 'crit_dmg', 'atk_pct')],
            [20, 20, 20, 20],
        )
        baizang = builds['char_701295143d']
        self.assertEqual((baizang['arc_name'], baizang['arc_id']), ('茶花会', 'arc_6e7753edf5'))
        haniya = builds['char_e0a4292b4e']
        self.assertEqual((haniya['arc_name'], haniya['arc_id']), ('引爆全场', 'arc_a5b483cca6'))

    def test_character_sheet_modifiers_are_removed_and_base_crit_is_fixed(self) -> None:
        catalog = load_shaft_catalog()
        for character in catalog['characters']:
            with self.subTest(character=character['name']):
                modifiers = character['modifiers']
                self.assertEqual(modifiers['crit_rate'], 0.05)
                self.assertEqual(modifiers['crit_dmg'], 0.5)
                self.assertTrue(all(
                    value == 0
                    for key, value in modifiers.items()
                    if key not in {'crit_rate', 'crit_dmg'}
                ))

        result = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''}],
            'steps': [{'id': 'q', 'slot': 0, 'action_id': 'action_9307778f01', 'start_tick': 0}],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        detail = result['details'][0]
        self.assertEqual(detail['panel']['crit_rate'], 0.05)
        self.assertEqual(detail['panel']['crit_dmg'], 0.5)
        self.assertAlmostEqual(detail['panel']['atk'], 644.0 * 1.2)
        passive = next(buff for buff in detail['applied_buffs'] if buff['rule_id'] == 'character_baizang_atk')
        self.assertEqual(passive['effects'], {'atk_pct': 0.2})
        self.assertFalse(passive['display_as_line'])
        self.assertEqual(passive['line_hidden_reason'], 'passive')

    def test_canhong_curtain_bonus_adds_sixteen_percent_crit_damage_per_type3_drive(self) -> None:
        base_payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'basic',
                'slot': 0,
                'action_id': 'action_canhong_a1',
                'start_tick': 0,
            }],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        }
        baseline = simulate_shaft_axis(base_payload)['result']['build_panels_by_slot'][0]
        equipped_payload = deepcopy(base_payload)
        equipped_payload['team'][0]['cartridge_id'] = 'cartridge_505ea4a58f'
        result = simulate_shaft_axis({
            **equipped_payload,
        })['result']

        build = result['build_panels_by_slot'][0]
        self.assertEqual(build['build_options']['curtain_bonus'], {
            'value': 16,
            'stat': '暴伤',
            'passive_type': 'type3',
            'layers': 4,
        })
        self.assertAlmostEqual(
            build['panel']['crit_dmg'] - baseline['panel']['crit_dmg'],
            0.16 * 4,
        )

    def test_iloy_bond_bonus_adds_five_percent_attack_not_crit(self) -> None:
        def panel_with_bond(enabled: bool) -> dict:
            result = simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_a01c39f576',
                    'arc_id': '',
                    'cartridge_id': '',
                    'bond_full': enabled,
                }],
                'steps': [{'id': 'z1', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 0}],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']
            return result['details'][0]['panel']

        without_bond = panel_with_bond(False)
        with_bond = panel_with_bond(True)
        self.assertAlmostEqual(with_bond['atk'] - without_bond['atk'], 596.0 * 0.05)
        self.assertEqual(with_bond['crit_rate'], without_bond['crit_rate'])

    def test_character_timed_buff_triggers_and_applies_to_later_action(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_dd034941ef',
                'arc_id': '',
                'cartridge_id': '',
                'awakening': 5,
            }],
            'steps': [
                {'id': 'support', 'slot': 0, 'action_id': 'action_33c0ac631e', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 14},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertIn(
            'character_protagonist_a5_support_atk',
            {buff['rule_id'] for buff in details['support']['triggered_buffs']},
        )
        applied = {
            buff['rule_id']: buff
            for buff in details['q']['applied_buffs']
        }
        self.assertEqual(applied['character_protagonist_a5_support_atk']['effects'], {'atk_pct': 0.1})

    def test_iloy_e_grants_team_twenty_percent_of_her_base_atk_on_start(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'iloy_e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'teammate_action', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        expected_flat_atk = 596.0 * 0.2

        triggered = {buff['rule_id']: buff for buff in details['iloy_e']['triggered_buffs']}
        self.assertEqual(
            triggered['character_iloy_e_team_flat_atk']['name'],
            '伊洛伊·自我认同延伸加攻',
        )
        self.assertAlmostEqual(
            triggered['character_iloy_e_team_flat_atk']['effects']['flat_atk'],
            expected_flat_atk,
        )

        for step_id in ('iloy_e', 'teammate_action'):
            applied = {buff['rule_id']: buff for buff in details[step_id]['applied_buffs']}
            self.assertAlmostEqual(
                applied['character_iloy_e_team_flat_atk']['effects']['flat_atk'],
                expected_flat_atk,
            )

    def test_iloy_a5_healing_grants_team_twelve_point_five_percent_base_atk(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_a01c39f576',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [5],
                },
                {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'iloy_e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'teammate_action', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        expected_flat_atk = 596.0 * 0.125
        triggered = {buff['rule_id']: buff for buff in details['iloy_e']['triggered_buffs']}
        applied = {buff['rule_id']: buff for buff in details['teammate_action']['applied_buffs']}

        self.assertAlmostEqual(
            triggered['character_iloy_a5_heal_team_flat_atk']['effects']['flat_atk'],
            expected_flat_atk,
        )
        self.assertAlmostEqual(
            applied['character_iloy_a5_heal_team_flat_atk']['effects']['flat_atk'],
            expected_flat_atk,
        )

    def test_iloy_a5_flat_atk_is_resolved_during_loop_warmup(self) -> None:
        def simulate(awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [
                    {
                        'slot': 0,
                        'character_id': 'char_a01c39f576',
                        'arc_id': '',
                        'cartridge_id': '',
                        'awakening_nodes': awakening_nodes,
                    },
                    {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
                ],
                'steps': [
                    {'id': 'iloy_e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                    {'id': 'teammate_action', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 20},
                ],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
                'options': {'loop_enabled': True},
            })['result']

        inactive = simulate([])
        active = simulate([5])
        active_details = {detail['step_id']: detail for detail in active['details']}
        active_buffs = {
            buff['rule_id']: buff
            for buff in active_details['teammate_action']['applied_buffs']
        }

        self.assertAlmostEqual(
            active_buffs['character_iloy_e_team_flat_atk']['effects']['flat_atk'],
            596.0 * 0.2,
        )
        self.assertAlmostEqual(
            active_buffs['character_iloy_a5_heal_team_flat_atk']['effects']['flat_atk'],
            596.0 * 0.125,
        )
        self.assertGreater(active['summary']['total_damage'], inactive['summary']['total_damage'])

    def test_iloy_long_e_marks_three_regressed_teammates_for_full_awakening_damage(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_a01c39f576',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [1, 2, 3, 4, 5, 6],
                },
                {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'long_e', 'slot': 0, 'action_id': 'action_iloy_long_e', 'start_tick': 0},
                {'id': 'after', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        triggered = {buff['rule_id']: buff for buff in details['long_e']['triggered_buffs']}
        applied = {buff['rule_id']: buff for buff in details['after']['applied_buffs']}

        regression = triggered['character_iloy_regressed_teammates']
        self.assertEqual(regression['name'], '伊洛伊·退行（3名队友）')
        self.assertEqual(regression['duration_ticks'], 9990)
        self.assertTrue(regression['display_as_line'])
        self.assertEqual(
            applied['character_iloy_full_regressed_teammate_damage']['effects'],
            {'all_dmg': 0.45},
        )
        self.assertEqual(
            applied['character_iloy_a6_team_crit_dmg']['effects'],
            {'crit_dmg': 0.25},
        )
        self.assertEqual(details['after']['formula_parts']['skill_level'], 11)

    def test_iloy_normal_e_remains_separate_and_does_not_apply_regression(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
            }],
            'steps': [
                {'id': 'normal_e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'after', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        triggered = {buff['rule_id']: buff for buff in details['normal_e']['triggered_buffs']}
        applied = {buff['rule_id']: buff for buff in details['after']['applied_buffs']}

        self.assertIn('character_iloy_e_team_flat_atk', triggered)
        self.assertNotIn('character_iloy_regressed_teammates', triggered)
        self.assertNotIn('character_iloy_full_regressed_teammate_damage', triggered)
        self.assertNotIn('character_iloy_full_regressed_teammate_damage', applied)

    def test_iloy_e_uses_online_sheet_nine_hit_damage_and_resource_totals(self) -> None:
        catalog = load_shaft_catalog()
        iloy_actions = {
            action['name']: action
            for action in catalog['actions']
            if action['character_id'] == 'char_a01c39f576'
        }

        for action_name in ('e', '长e'):
            with self.subTest(action_name=action_name):
                action = iloy_actions[action_name]
                self.assertEqual(action['hit_count'], 9)
                self.assertEqual(action['source_formula'], '{0}%+{1}%*7+{2}%')
                self.assertAlmostEqual(action['multipliers']['atk'], 4.003)
                self.assertAlmostEqual(action['energy_gain'], 4.798)
                self.assertAlmostEqual(action['harmony'], 7.998)
                self.assertAlmostEqual(action['stagger'], 1.003)

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'normal_e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        detail = result['details'][0]

        self.assertEqual(detail['hit_count'], 9)
        self.assertAlmostEqual(detail['harmony'], 7.998)
        self.assertAlmostEqual(detail['stagger_amount'], 1.003)

    def test_iloy_healing_actions_grant_visible_team_def_ignore_line(self) -> None:
        healing_actions = (
            'action_01209221c1',
            'action_afea1e6fb2',
            'action_iloy_long_e',
            'action_307ad00e8e',
        )
        for action_id in healing_actions:
            with self.subTest(action_id=action_id):
                result = simulate_shaft_axis({
                    'team': [
                        {'slot': 0, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
                        {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
                    ],
                    'steps': [
                        {'id': 'healing_action', 'slot': 0, 'action_id': action_id, 'start_tick': 0},
                        {'id': 'teammate_action', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 20},
                    ],
                    'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                    'initial_energy': 200,
                })['result']
                details = {detail['step_id']: detail for detail in result['details']}
                triggered = {
                    buff['rule_id']: buff
                    for buff in details['healing_action']['triggered_buffs']
                }
                applied = {
                    buff['rule_id']: buff
                    for buff in details['teammate_action']['applied_buffs']
                }

                buff = triggered['character_iloy_heal_team_def_ignore']
                self.assertEqual(buff['effects'], {'def_ignore': 0.05})
                self.assertEqual(buff['duration_ticks'], 200)
                self.assertTrue(buff['display_as_line'])
                self.assertEqual(
                    applied['character_iloy_heal_team_def_ignore']['effects'],
                    {'def_ignore': 0.05},
                )

    def test_action_energy_is_shared_to_teammates_but_return_energy_is_not(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2]},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']

        resources = {item['slot']: item for item in result['resources_by_slot']}
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertAlmostEqual(details['main_e']['energy_gain'], 8.801 + 8)
        self.assertAlmostEqual(details['main_e']['base_energy_gain'], 8.801)
        self.assertEqual(details['main_e']['energy_return'], 8)
        self.assertEqual(details['main_e']['energy_gain_timing'], 'uniform')
        self.assertEqual(details['main_e']['energy_after'], 0)
        self.assertAlmostEqual(resources[0]['energy'], 8.801 + 8)
        self.assertAlmostEqual(resources[1]['energy'], 8.801 * 0.6)
        actor_events = [
            event
            for event in result['energy_events']
            if event['slot'] == 0 and event['source_step_id'] == 'main_e'
        ]
        self.assertEqual([event['tick'] for event in actor_events], list(range(1, 16)))
        self.assertTrue(all(
            abs(event['amount'] - (8.801 + 8) / 15) < 1e-9
            for event in actor_events
        ))

    def test_action_resource_snapshots_include_teammate_energy_and_reaction_consumption(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2]},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        gain_resources = {item['slot']: item for item in details['gain']['resources_after_by_slot']}
        support_resources = {item['slot']: item for item in details['support']['resources_after_by_slot']}
        self.assertEqual(gain_resources[0]['energy'], 0)
        self.assertEqual(gain_resources[1]['energy'], 0)
        self.assertAlmostEqual(support_resources[0]['energy'], 8.801 + 8)
        self.assertAlmostEqual(support_resources[1]['energy'], 8.801 * 0.6)
        self.assertEqual(gain_resources[0]['harmony'], 100)
        self.assertEqual(support_resources[0]['harmony'], 0)

    def test_action_energy_available_mid_action_is_only_the_elapsed_fraction(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2]},
                {'slot': 1, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2]},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'midpoint', 'slot': 1, 'action_id': 'action_0b76fe2aaa', 'start_tick': 5},
                {'id': 'complete', 'slot': 1, 'action_id': 'action_0b76fe2aaa', 'start_tick': 15},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        midpoint = {item['slot']: item for item in details['midpoint']['resources_after_by_slot']}
        complete = {item['slot']: item for item in details['complete']['resources_after_by_slot']}

        self.assertAlmostEqual(midpoint[0]['energy'], (8.801 + 8) / 3)
        self.assertAlmostEqual(midpoint[1]['energy'], 8.801 * 0.6 / 3)
        self.assertAlmostEqual(complete[0]['energy'], 8.801 + 8)
        self.assertAlmostEqual(complete[1]['energy'], 8.801 * 0.6)

    def test_overlapping_action_energy_streams_accumulate_independently(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'protagonist_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanali_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 5},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        resources = {item['slot']: item for item in result['resources_by_slot']}
        protagonist_ticks = {
            event['tick']
            for event in result['energy_events']
            if event['slot'] == 0 and event['source_step_id'] == 'protagonist_e'
        }
        nanali_share_ticks = {
            event['tick']
            for event in result['energy_events']
            if event['slot'] == 0 and event['source_step_id'] == 'nanali_e'
        }

        self.assertTrue(protagonist_ticks & nanali_share_ticks)
        self.assertAlmostEqual(resources[0]['energy'], 8.801 + 6.3 * 0.6)
        self.assertAlmostEqual(resources[1]['energy'], 8.801 * 0.6 + 6.3)

    def test_zero_duration_action_energy_remains_instant(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'instant_gain', 'slot': 0, 'action_id': 'action_5e62f427cb', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']
        detail = result['details'][0]

        self.assertEqual(detail['energy_gain_timing'], 'instant')
        self.assertEqual(detail['energy_after'], 10)
        self.assertEqual(result['resources_by_slot'][0]['energy'], 10)
        self.assertEqual(result['energy_events'][0]['tick'], 0)

    def test_q_energy_check_only_uses_energy_accumulated_by_its_start_tick(self) -> None:
        def simulate(q_tick: int) -> dict:
            return simulate_shaft_axis({
                'team': [
                    {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                ],
                'steps': [
                    {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                    {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': q_tick},
                ],
                'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 92,
            })['result']

        midway = {detail['step_id']: detail for detail in simulate(5)['details']}
        completed = {detail['step_id']: detail for detail in simulate(15)['details']}

        self.assertIn('终结技能量不足。', midway['q']['warnings'])
        self.assertNotIn('终结技能量不足。', completed['q']['warnings'])

    def test_q_masked_action_keeps_full_energy_and_q_owner_receives_same_tick_share(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'nanali_e', 'slot': 0, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'protagonist_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 5},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_bdc43f82c6': {'energy': 0, 'harmony': 0},
                    'char_dd034941ef': {'energy': 100, 'harmony': 0},
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        resources = {item['slot']: item for item in result['resources_by_slot']}
        same_tick_events = [
            event
            for event in result['energy_events']
            if event['slot'] == 1 and event['tick'] == 5
        ]

        self.assertTrue(details['nanali_e']['q_instant_release'])
        self.assertEqual(details['nanali_e']['duration_ticks'], 5)
        self.assertAlmostEqual(details['nanali_e']['energy_gain'], 6.3)
        self.assertAlmostEqual(resources[0]['energy'], 6.3)
        self.assertAlmostEqual(resources[1]['energy'], 6.3 * 0.6 / 5)
        self.assertEqual(
            [event['kind'] for event in same_tick_events],
            ['action_cost', 'action_gain'],
        )
        self.assertAlmostEqual(same_tick_events[-1]['amount'], 6.3 * 0.6 / 5)

    def test_character_energy_is_capped_and_q_consumes_the_capped_pool(self) -> None:
        catalog = load_shaft_catalog()
        protagonist = next(character for character in catalog['characters'] if character['id'] == 'char_dd034941ef')
        self.assertEqual(protagonist['energy_capacity'], 100)

        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain_near_cap', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 95,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        resources = result['resources_by_slot'][0]
        self.assertEqual(resources['initial_energy'], 95)
        self.assertEqual(resources['energy_capacity'], 100)
        self.assertAlmostEqual(details['gain_near_cap']['energy_gain'], 8.801)
        self.assertAlmostEqual(details['gain_near_cap']['energy_after'], 95)
        self.assertAlmostEqual(details['q']['energy_after'], 0)
        self.assertAlmostEqual(resources['energy'], 0)

    def test_initial_energy_above_character_capacity_is_wasted(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'normal_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        resources = result['resources_by_slot'][0]
        self.assertEqual(resources['energy_capacity'], 60)
        self.assertEqual(resources['initial_energy'], 60)
        self.assertEqual(result['details'][0]['energy_after'], 60)
        self.assertEqual(resources['energy'], 60)

    def test_default_initial_energy_fills_each_character_to_capacity(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        }

        normalized = normalize_axis_payload(payload)
        self.assertEqual(normalized['initial_energy'], 1000)

        resources = {
            item['character_id']: item
            for item in simulate_shaft_axis(payload)['result']['resources_by_slot']
        }
        self.assertEqual(resources['char_701295143d']['energy_capacity'], 120)
        self.assertEqual(resources['char_701295143d']['initial_energy'], 120)
        self.assertEqual(resources['char_dd034941ef']['energy_capacity'], 100)
        self.assertEqual(resources['char_dd034941ef']['initial_energy'], 100)

    def test_zhenhong_ascendant_energy_cap_dragon_e_cost_and_exit_clear(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'enter', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'gain_ten', 'slot': 0, 'action_id': 'action_5e62f427cb', 'start_tick': 5},
                {'id': 'gain_to_cap', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 7},
                {'id': 'dragon_e', 'slot': 0, 'action_id': 'action_7773821d79', 'start_tick': 12},
                {'id': 'gain_before_exit', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 30},
                {'id': 'exit', 'slot': 0, 'action_id': 'action_e3711f0cf5', 'start_tick': 35},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 60,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual(details['enter']['energy_after'], 0)
        self.assertEqual(details['enter']['energy_capacity_after'], 12)
        self.assertAlmostEqual(details['gain_ten']['energy_after'], 10)
        self.assertAlmostEqual(details['gain_to_cap']['energy_gain'], 2.415)
        self.assertAlmostEqual(details['gain_to_cap']['energy_after'], 10)
        self.assertEqual(details['dragon_e']['warnings'], [])
        self.assertEqual(details['dragon_e']['energy_after'], 0)
        self.assertGreater(details['gain_before_exit']['energy_gain'], 0)
        self.assertEqual(details['gain_before_exit']['energy_after'], 0)
        self.assertEqual(details['exit']['energy_after'], 0)
        self.assertEqual(details['exit']['energy_capacity_after'], 60)
        self.assertEqual(result['resources_by_slot'][0]['energy'], 0)
        self.assertEqual(result['resources_by_slot'][0]['energy_capacity'], 60)

    def test_zhenhong_non_front_does_not_receive_energy_or_shared_energy(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2]},
                {'slot': 1, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'zhenhong_yingxu', 'slot': 1, 'action_id': 'action_5e62f427cb', 'start_tick': 10},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']

        resources = {item['slot']: item for item in result['resources_by_slot']}
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertAlmostEqual(details['zhenhong_yingxu']['energy_gain'], 0)
        self.assertAlmostEqual(resources[0]['energy'], 8.801 + 8 + 10 * 0.6)
        self.assertAlmostEqual(resources[1]['energy'], 0)

    def test_zhenhong_front_can_receive_teammate_background_action_shared_energy(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'zhenhong_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'yi_background', 'slot': 1, 'action_id': 'action_8380c01e01', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']

        resources = {item['slot']: item for item in result['resources_by_slot']}
        self.assertAlmostEqual(resources[0]['energy'], 5.2 + 1.814 * 0.6)
        self.assertAlmostEqual(resources[1]['energy'], 5.2 * 0.6 + 1.814)

    def test_nanali_talent_changes_genesis_to_ten_one_second_flowers(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        effect = details['support']['triggered_reaction']
        self.assertEqual(effect['reaction'], '创生')
        self.assertEqual(effect['support_slot'], 1)
        self.assertEqual(effect['trigger_slot'], effect['contributor_slot'])
        self.assertEqual(effect['trigger_character_name'], effect['contributor_character_name'])
        self.assertEqual(effect['start_tick'], details['support']['visual_end_tick'] - 2)
        self.assertEqual(details['support']['warnings'], [])
        self.assertEqual(result['resources_by_slot'][0]['harmony'], 0)
        self.assertEqual([event['tick'] for event in result['reaction_damage_events']], [41, 51, 61, 71, 81, 91, 101, 111, 121, 131])
        self.assertTrue(all(event['damage'] > 0 for event in result['reaction_damage_events']))
        self.assertEqual(result['summary']['duration_ticks'], 33)

    def test_genesis_without_nanali_keeps_five_two_second_flowers(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_01209221c1', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        self.assertEqual(result['details'][-1]['triggered_reaction']['reaction'], '创生')
        genesis_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生'
        ]
        clone_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生复制体'
        ]
        self.assertEqual([event['tick'] for event in genesis_events], [53, 73, 93, 113, 133])
        self.assertEqual(len(clone_events), 20)
        self.assertEqual([event['tick'] for event in clone_events], list(range(68, 168, 5)))
        self.assertTrue(all(
            event['contributor_character_name'] == genesis_events[0]['contributor_character_name']
            for event in clone_events
        ))
        self.assertAlmostEqual(clone_events[0]['damage'], genesis_events[0]['damage'] * 0.375)
        damage_by_source = {
            item['source']: item
            for item in result['damage_by_source']
        }
        self.assertNotIn('创生复制体', damage_by_source)
        self.assertAlmostEqual(
            damage_by_source['创生']['damage'],
            sum(event['damage'] for event in genesis_events + clone_events),
        )
        harmony_sources = result['harmony_contributions_by_slot'][0]['sources']
        self.assertEqual([item['source'] for item in harmony_sources], ['创生'])
        self.assertAlmostEqual(harmony_sources[0]['damage'], damage_by_source['创生']['damage'])
        clone_effect = next(
            effect for effect in result['reaction_effects']
            if effect['reaction'] == '创生复制体'
        )
        genesis_effect = result['details'][-1]['triggered_reaction']
        self.assertEqual(clone_effect['start_tick'], genesis_effect['start_tick'] + 30)
        self.assertEqual(clone_effect['contributor_slot'], genesis_effect['contributor_slot'])

    def test_nanali_does_not_change_iloy_genesis_clone_flowers(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_01209221c1', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        genesis_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生'
        ]
        clone_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生复制体'
        ]
        self.assertEqual(len(genesis_events), 10)
        self.assertEqual(len(clone_events), 20)
        self.assertEqual(
            [event['tick'] for event in clone_events],
            list(range(clone_events[0]['tick'], clone_events[0]['tick'] + 100, 5)),
        )
        clone_effect = next(
            effect for effect in result['reaction_effects']
            if effect['reaction'] == '创生复制体'
        )
        self.assertEqual(clone_effect['generation_delay_ticks'], 30)
        self.assertEqual(clone_events[0]['tick'], clone_effect['start_tick'] + 5)
        self.assertTrue(all(event['formula_parts']['damage_scale'] == 0.375 for event in clone_events))
        self.assertAlmostEqual(clone_events[0]['damage'], genesis_events[0]['damage'] * 0.375)

    def test_jiuyuan_talent_adds_independent_frequency_multiplier_to_genesis_and_clone(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_01209221c1', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        genesis_effect = next(
            effect for effect in result['reaction_effects']
            if effect['reaction'] == '创生'
        )
        clone_effect = next(
            effect for effect in result['reaction_effects']
            if effect['reaction'] == '创生复制体'
        )
        genesis_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生'
        ]
        clone_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生复制体'
        ]

        self.assertEqual(genesis_effect['frequency_multiplier'], 2)
        self.assertEqual(clone_effect['frequency_multiplier'], 2)
        self.assertEqual(len(genesis_events), 10)
        self.assertEqual(len(clone_events), 20)
        self.assertEqual(genesis_events[-1]['tick'], genesis_effect['end_tick'])
        self.assertEqual(clone_events[-1]['tick'], clone_effect['end_tick'])
        self.assertEqual(clone_effect['duration_ticks'], 100)
        self.assertTrue(all(event['frequency_multiplier'] == 2 for event in genesis_events))
        self.assertTrue(all(event['frequency_multiplier'] == 2 for event in clone_events))
        self.assertTrue(all(event['formula_parts']['frequency_multiplier'] == 2 for event in genesis_events))
        first_formula = genesis_events[0]['formula_parts']
        self.assertAlmostEqual(
            genesis_events[0]['damage'],
            first_formula['base']
            * first_formula['strength']
            * first_formula['frequency_multiplier']
            * first_formula['defense']
            * first_formula['critical']
            * first_formula['resistance']
            * first_formula['final_multiplier']
            * first_formula['damage_scale'],
        )
        self.assertAlmostEqual(clone_events[0]['damage'], genesis_events[0]['damage'] * 0.375)

    def test_axis_duration_ignores_background_actions_after_foreground_end(self) -> None:
        payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_b52cc8f160',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'front', 'slot': 0, 'action_id': 'action_a01a545590', 'start_tick': 0},
                {
                    'id': 'background',
                    'slot': 0,
                    'action_id': 'action_c6ff824434',
                    'start_tick': 50,
                    'placement': 'background',
                },
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        }

        normalized = normalize_axis_payload(payload)
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(normalized['duration_ticks'], 5)
        self.assertEqual(result['summary']['duration_ticks'], 5)
        self.assertEqual(result['summary']['timeline_ticks'], 5)
        self.assertFalse(details['front']['is_background_damage'])
        self.assertTrue(details['background']['is_background_damage'])
        self.assertEqual(details['background']['visual_end_tick'], 60)

    def test_invalid_or_underfunded_support_is_reported_without_reaction(self) -> None:
        underfunded = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
            ],
        })['result']
        self.assertFalse(underfunded['reaction_effects'])
        self.assertIn('环合值不足', underfunded['details'][-1]['warnings'][0])

        invalid_pair = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_222f577087', 'start_tick': 20},
            ],
        })['result']
        self.assertFalse(invalid_pair['reaction_effects'])
        self.assertIn('无法产生异能环合', invalid_pair['details'][-1]['warnings'][0])

    def test_all_element_pairs_resolve_to_documented_reactions(self) -> None:
        cases = [
            ('延滞', 'char_dd034941ef', 'action_982c67944f', 1, 'char_912dbfe17c', 'action_f229587fd2', 0),
            ('覆纹', 'char_b2e3b2bf7a', 'action_39d4605011', 4, 'char_701295143d', 'action_222f577087', 0),
            ('浊燃', 'char_1895e259be', 'action_4c5142a40f', 4, 'char_c78f7a08d5', 'action_2635f721a8', 15),
            ('黯星', 'char_c78f7a08d5', 'action_6d2645f71e', 7, 'char_caa6c2e5a8', 'action_b0e5fd6662', 1),
            ('浸染', 'char_caa6c2e5a8', 'action_441d3aa300', 4, 'char_912dbfe17c', 'action_f229587fd2', 0),
        ]
        for reaction, previous_character, gain_action, repeats, support_character, support_action, damage_events in cases:
            with self.subTest(reaction=reaction):
                steps = [
                    {'id': f'gain_{index}', 'slot': 0, 'action_id': gain_action, 'start_tick': index * 2}
                    for index in range(repeats)
                ]
                steps.append({'id': 'support', 'slot': 1, 'action_id': support_action, 'start_tick': 40})
                result = simulate_shaft_axis({
                    'team': [
                        {'slot': 0, 'character_id': previous_character, 'arc_id': '', 'cartridge_id': ''},
                        {'slot': 1, 'character_id': support_character, 'arc_id': '', 'cartridge_id': ''},
                    ],
                    'steps': steps,
                })['result']
                self.assertEqual(result['details'][-1]['triggered_reaction']['reaction'], reaction)
                self.assertEqual(len(result['reaction_damage_events']), damage_events)

    def test_loop_axis_primes_previous_reaction_without_extending_loop_height(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
            ],
            'options': {'loop_enabled': True},
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        primed = [effect for effect in result['reaction_effects'] if effect['loop_primed']]
        self.assertEqual(len(primed), 1)
        self.assertEqual(primed[0]['start_tick'], -2)
        self.assertEqual(result['summary']['duration_ticks'], 33)
        self.assertEqual([event['tick'] for event in result['reaction_damage_events']], [8, 18, 28])

    def test_loop_axis_resource_display_excludes_priming_passes(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
            ],
            'options': {'loop_enabled': True},
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        self.assertEqual(result['details'][0]['harmony'], 100)
        self.assertEqual(result['resources_by_slot'][0]['harmony'], 100)

    def test_loop_axis_uses_per_character_initial_energy_and_harmony(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'zhenhong_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_b52cc8f160': {'energy': 999, 'harmony': 130, 'personal_resources': {'未知': 20}},
                    'char_dd034941ef': {'energy': 25, 'harmony': 40, 'personal_resources': {}},
                    'unknown_character': {'energy': 50, 'harmony': 50},
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        }
        normalized = normalize_axis_payload(payload)
        self.assertEqual(normalized['options']['loop_initial_resources'], {
            'char_b52cc8f160': {'energy': 60, 'harmony': 100, 'reaction': '', 'personal_resources': {}},
            'char_dd034941ef': {'energy': 25, 'harmony': 40, 'reaction': '', 'personal_resources': {}},
        })

        result = simulate_shaft_axis(payload)['result']
        resources = {item['character_id']: item for item in result['resources_by_slot']}
        self.assertEqual(resources['char_b52cc8f160']['initial_energy'], 60)
        self.assertEqual(resources['char_b52cc8f160']['initial_harmony'], 100)
        self.assertEqual(resources['char_dd034941ef']['initial_energy'], 25)
        self.assertEqual(resources['char_dd034941ef']['initial_harmony'], 40)

        non_loop_payload = deepcopy(payload)
        non_loop_payload['options']['loop_enabled'] = False
        non_loop = simulate_shaft_axis(non_loop_payload)['result']
        self.assertEqual(non_loop['resources_by_slot'][0]['initial_energy'], 0)
        self.assertEqual(non_loop['resources_by_slot'][0]['initial_harmony'], 0)

    def test_loop_axis_can_seed_one_sustained_reaction_per_character(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_076a1f4e53', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 30},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_076a1f4e53': {
                        'energy': 0,
                        'harmony': 0,
                        'reaction': '浊燃',
                        'personal_resources': {},
                    },
                    'char_dd034941ef': {
                        'energy': 0,
                        'harmony': 0,
                        'reaction': '失谐',
                        'personal_resources': {},
                    },
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        }
        normalized = normalize_axis_payload(payload)
        self.assertEqual(
            normalized['options']['loop_initial_resources']['char_076a1f4e53']['reaction'],
            '浊燃',
        )
        self.assertEqual(
            normalized['options']['loop_initial_resources']['char_dd034941ef']['reaction'],
            '',
        )

        result = simulate_shaft_axis(payload)['result']
        initial_effects = [effect for effect in result['reaction_effects'] if effect.get('loop_initial')]
        self.assertEqual(len(initial_effects), 1)
        self.assertEqual(initial_effects[0]['reaction'], '浊燃')
        self.assertEqual(initial_effects[0]['contributor_slot'], 0)
        self.assertEqual(initial_effects[0]['start_tick'], 0)
        self.assertEqual(initial_effects[0]['end_tick'], 150)
        self.assertTrue(all(event.get('loop_initial') for event in result['reaction_damage_events']))
        resources = {item['slot']: item for item in result['resources_by_slot']}
        self.assertEqual(resources[0]['initial_reaction'], '浊燃')
        self.assertEqual(resources[1]['initial_reaction'], '')

        non_loop_payload = deepcopy(payload)
        non_loop_payload['options']['loop_enabled'] = False
        non_loop = simulate_shaft_axis(non_loop_payload)['result']
        self.assertFalse(any(effect.get('loop_initial') for effect in non_loop['reaction_effects']))
        self.assertTrue(all(not item['initial_reaction'] for item in non_loop['resources_by_slot']))

    def test_loop_axis_carries_reaction_layer_mutations_from_the_previous_cycle(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'illusion-a1', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 10},
                {'id': 'illusion-a2', 'slot': 0, 'action_id': 'action_canhong_illusion_a2', 'start_tick': 20},
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 50},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_076a1f4e53': {
                        'energy': 120,
                        'harmony': 0,
                        'reaction': '浊燃',
                        'personal_resources': {},
                    },
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        turbid_effects = [effect for effect in result['reaction_effects'] if effect['reaction'] == '浊燃']
        turbid_events = [event for event in result['reaction_damage_events'] if event['reaction'] == '浊燃']
        self.assertEqual(len(turbid_effects), 1)
        self.assertTrue(turbid_effects[0]['looped'])
        self.assertEqual(turbid_effects[0]['stack_count'], 3)
        self.assertEqual(turbid_effects[0]['frequency_multiplier'], 3)
        self.assertTrue(turbid_events)
        self.assertTrue(all(event['damage'] > 0 for event in turbid_events))
        self.assertTrue(all(event['formula_parts']['frequency_multiplier'] == 3 for event in turbid_events))

    def test_loop_axis_uses_per_character_initial_personal_resources(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_d38b672525', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'iloy_attack', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 0},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_a01c39f576': {
                        'energy': 0,
                        'harmony': 0,
                        'personal_resources': {'臆想': 999, '未知': 50},
                    },
                    'char_d38b672525': {
                        'energy': 0,
                        'harmony': 0,
                        'personal_resources': {'罪状': 135},
                    },
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        }
        normalized = normalize_axis_payload(payload)
        self.assertEqual(
            normalized['options']['loop_initial_resources']['char_a01c39f576']['personal_resources'],
            {'臆想': 100},
        )
        self.assertEqual(
            normalized['options']['loop_initial_resources']['char_d38b672525']['personal_resources'],
            {'罪状': 135},
        )

        result = simulate_shaft_axis(payload)['result']
        resources = {item['character_id']: item for item in result['resources_by_slot']}
        self.assertEqual(resources['char_a01c39f576']['initial_personal_resources'], {'臆想': 100})
        self.assertEqual(resources['char_a01c39f576']['personal_resources']['臆想'], 80)
        self.assertEqual(resources['char_d38b672525']['initial_personal_resources'], {'罪状': 135})

        payload['options']['loop_enabled'] = False
        non_loop = simulate_shaft_axis(payload)['result']
        self.assertEqual(non_loop['resources_by_slot'][0]['initial_personal_resources'], {})

    def test_harmony_is_capped_at_one_hundred_for_every_character(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'overflow_harmony', 'slot': 0, 'action_id': 'action_28759c6e0b', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 0,
        })['result']

        self.assertGreater(result['details'][0]['harmony'], 100)
        self.assertEqual(result['details'][0]['harmony_after'], 100)
        self.assertEqual(result['resources_by_slot'][0]['harmony'], 100)
        self.assertEqual(result['summary']['total_harmony'], 100)

    def test_loop_axis_opening_support_uses_configured_initial_harmony(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 0},
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 20},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_dd034941ef': {'energy': 0, 'harmony': 0},
                    'char_bdc43f82c6': {'energy': 0, 'harmony': 0},
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual(
            details['support']['warnings'],
            ['主角环合值不足：需要 100，当前 0.0。'],
        )
        self.assertIsNone(details['support']['triggered_reaction'])
        self.assertEqual(result['resources_by_slot'][0]['harmony'], 100)

    def test_zhenhong_to_yi_support_warns_when_loop_initial_harmony_is_zero(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'zhenhong_q2', 'slot': 0, 'action_id': 'action_e3711f0cf5', 'start_tick': 0},
                {'id': 'yi_support', 'slot': 3, 'action_id': 'action_31c54a4e71', 'start_tick': 5},
            ],
            'options': {
                'loop_enabled': True,
                'loop_initial_resources': {
                    'char_b52cc8f160': {'energy': 60, 'harmony': 0},
                    'char_7578b18979': {'energy': 100, 'harmony': 0},
                },
            },
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        support = next(detail for detail in result['details'] if detail['step_id'] == 'yi_support')
        self.assertEqual(
            support['warnings'],
            ['真红环合值不足：需要 100，当前 0.0。'],
        )
        self.assertIsNone(support['triggered_reaction'])

    def test_passive_damage_actions_do_not_scale_with_skill_level(self) -> None:
        payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
                'skill_levels': {'basic': 1, 'skill': 1, 'ultimate': 1, 'support': 1},
            }],
            'steps': [{'id': 'dissonance', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 0}],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        }
        low = simulate_shaft_axis(payload)['result']['details'][0]
        payload['team'][0]['skill_levels'] = {'basic': 10, 'skill': 10, 'ultimate': 10, 'support': 10}
        high = simulate_shaft_axis(payload)['result']['details'][0]

        self.assertEqual(low['formula_parts']['raw_base'], low['panel']['atk'] * 4)
        self.assertEqual(low['formula_parts']['skill_level_category'], '')
        self.assertEqual(low['formula_parts']['skill_level_multiplier'], 1)
        self.assertAlmostEqual(low['direct_damage'], high['direct_damage'])

    def test_iloy_actions_and_b_awakening_gate_are_available(self) -> None:
        catalog = load_shaft_catalog()
        iloy_actions = {
            action['name']: action
            for action in catalog['actions']
            if action['character_id'] == 'char_a01c39f576'
        }
        self.assertEqual(iloy_actions['援护']['duration_ticks'], 15)
        self.assertEqual(iloy_actions['z1']['duration_ticks'], 15)
        self.assertEqual(iloy_actions['e']['duration_ticks'], 4)
        self.assertEqual(iloy_actions['长e']['duration_ticks'], 4)
        self.assertNotEqual(iloy_actions['e']['id'], iloy_actions['长e']['id'])
        self.assertEqual(iloy_actions['q']['duration_ticks'], 0)
        self.assertEqual(iloy_actions['B觉']['multipliers']['atk'], 1.5)
        self.assertEqual(iloy_actions['B觉']['action_type'], '被动')

        base_payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
                'awakening': 0,
            }],
            'steps': [{'id': 'b', 'slot': 0, 'action_id': 'action_0b76fe2aaa', 'start_tick': 0}],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        }
        locked = simulate_shaft_axis(base_payload)['result']['details'][0]
        self.assertIn('动作需要 B 觉醒节点。', locked['warnings'])
        base_payload['team'][0]['awakening_nodes'] = [2]
        unlocked = simulate_shaft_axis(base_payload)['result']['details'][0]
        self.assertNotIn('动作需要 B 觉醒节点。', unlocked['warnings'])
        base_payload['team'][0]['awakening_nodes'] = [1]
        wrong_node = simulate_shaft_axis(base_payload)['result']['details'][0]
        self.assertIn('动作需要 B 觉醒节点。', wrong_node['warnings'])

    def test_nanali_basic_and_plunge_action_durations_match_measured_values(self) -> None:
        catalog = load_shaft_catalog()
        nanali_actions = {
            action['name']: action
            for action in catalog['actions']
            if action['character_id'] == 'char_bdc43f82c6'
        }

        expected_duration_ticks = {
            'a1': 3,
            'a2': 3,
            'a3': 8,
            'a4': 10,
            'a5': 13,
            '下落': 7,
        }
        self.assertEqual(
            {
                name: nanali_actions[name]['duration_ticks']
                for name in expected_duration_ticks
            },
            expected_duration_ticks,
        )
        self.assertTrue(all(
            not nanali_actions[name].get('duration_pending')
            for name in expected_duration_ticks
        ))

    def test_iloy_z1_uses_three_hits_and_consumes_twenty_reverie(self) -> None:
        catalog = load_shaft_catalog()
        action = next(
            action
            for action in catalog['actions']
            if action['character_id'] == 'char_a01c39f576' and action['name'] == 'z1'
        )

        self.assertEqual(action['action_type'], '普攻')
        self.assertEqual(action['duration_ticks'], 15)
        self.assertEqual(action['hit_count'], 3)
        self.assertAlmostEqual(action['multipliers']['atk'], 1.104)
        self.assertAlmostEqual(action['energy_gain'], 1.599)
        self.assertAlmostEqual(action['stagger'], 0.3)
        self.assertAlmostEqual(action['harmony'], 2.7)
        self.assertEqual(action['personal_resource_cost'], {'臆想': 20})
        self.assertEqual(action['personal_resource_gain'], {})

        detail = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'z1',
                'slot': 0,
                'action_id': 'action_iloy_z1',
                'start_tick': 0,
            }],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']['details'][0]

        self.assertEqual(detail['hit_count'], 3)
        self.assertIn('个人资源 臆想 不足。', detail['warnings'])
        self.assertEqual(detail['resources_after_by_slot'][0]['personal_resources']['臆想'], 0)

    def test_iloy_lucid_dream_uses_thirty_five_hits_and_consumes_ten_reverie(self) -> None:
        catalog = load_shaft_catalog()
        action = next(
            action
            for action in catalog['actions']
            if action['id'] == 'action_iloy_lucid_dream'
        )

        self.assertEqual(action['name'], '清明梦')
        self.assertEqual(action['action_type'], '普攻')
        self.assertEqual(action['duration_ticks'], 70)
        self.assertEqual(action['hit_count'], 35)
        self.assertAlmostEqual(action['multipliers']['atk'], 0.237 * 35)
        self.assertAlmostEqual(action['energy_gain'], 0.766 * 35)
        self.assertAlmostEqual(action['harmony'], 0.2 * 35)
        self.assertAlmostEqual(action['stagger'], 0.186 * 35)
        self.assertEqual(action['personal_resource_cost'], {'臆想': 10})
        self.assertEqual(action['personal_resource_gain'], {})
        self.assertEqual(action['tags'], ['蓄力', '蓄力重击', '重击'])

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'lucid_dream', 'slot': 0, 'action_id': 'action_iloy_lucid_dream', 'start_tick': 20},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertNotIn('个人资源 臆想 不足。', details['lucid_dream']['warnings'])
        self.assertEqual(details['lucid_dream']['personal_resources_after']['臆想'], 6)

    def test_iloy_support_and_ultimate_resources_match_current_online_sheet(self) -> None:
        actions = {
            action['id']: action
            for action in load_shaft_catalog()['actions']
            if action['character_id'] == 'char_a01c39f576'
        }

        self.assertAlmostEqual(actions['action_01209221c1']['energy_gain'], 3.9)
        self.assertAlmostEqual(actions['action_01209221c1']['stagger'], 2.5)
        self.assertAlmostEqual(actions['action_307ad00e8e']['stagger'], 1.5)
        self.assertAlmostEqual(actions['action_c371c893ce']['stagger'], 0.125)

    def test_iloy_first_three_basic_attacks_match_current_online_sheet(self) -> None:
        actions = {
            action['name']: action
            for action in load_shaft_catalog()['actions']
            if action['character_id'] == 'char_a01c39f576'
        }
        expected = {
            'a1': (3, 0.76, 1, 2.431, 4.048, 0.484, 2, False, 442),
            'a2': (7, 0.477, 3, 0.996, 1.659, 0.198, 3, True, 443),
            'a3': (7, 0.824, 8, 0.642, 1.069, 0.128, 3, True, 444),
        }

        for name, values in expected.items():
            with self.subTest(name=name):
                action = actions[name]
                duration, multiplier, hits, energy, harmony, stagger, reverie, background, source_row = values
                self.assertEqual(action['duration_ticks'], duration)
                self.assertAlmostEqual(action['multipliers']['atk'], multiplier)
                self.assertEqual(action['hit_count'], hits)
                self.assertAlmostEqual(action['energy_gain'], energy)
                self.assertAlmostEqual(action['harmony'], harmony)
                self.assertAlmostEqual(action['stagger'], stagger)
                self.assertEqual(action['personal_resource_gain'], {'臆想': reverie})
                self.assertEqual(action['can_background_override'], background)
                self.assertEqual(action['source_row'], source_row)

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'a1', 'slot': 0, 'action_id': 'action_5ce7a08ca0', 'start_tick': 0},
                {'id': 'a2', 'slot': 0, 'action_id': 'action_d3328cde94', 'start_tick': 3},
                {'id': 'a3', 'slot': 0, 'action_id': 'action_ace58eb0ce', 'start_tick': 10},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['a1']['personal_resources_after']['臆想'], 2)
        self.assertEqual(details['a2']['personal_resources_after']['臆想'], 5)
        self.assertEqual(details['a3']['personal_resources_after']['臆想'], 8)

    def test_iloy_e_builds_reverie_and_q_dream_waives_heavy_attack_cost(self) -> None:
        catalog = load_shaft_catalog()
        iloy_actions = {
            action['name']: action
            for action in catalog['actions']
            if action['character_id'] == 'char_a01c39f576'
        }
        self.assertEqual(iloy_actions['e']['personal_resource_gain'], {'臆想': 16})
        self.assertEqual(iloy_actions['长e']['personal_resource_gain'], {'臆想': 16})
        self.assertEqual(iloy_actions['q']['personal_resource_threshold'], {'臆想': 30})
        self.assertEqual(iloy_actions['q持续']['required_buff_key'], 'character_iloy_q_ivory_dream')

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'e1', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'e2', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 120},
                {'id': 'e3', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 240},
                {'id': 'q', 'slot': 0, 'action_id': 'action_307ad00e8e', 'start_tick': 260},
                {'id': 'dream-z1', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 270},
                {'id': 'dream-e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 280},
                {'id': 'awake-z1', 'slot': 0, 'action_id': 'action_iloy_z1', 'start_tick': 320},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['e1']['personal_resources_after']['臆想'], 16)
        self.assertEqual(details['e2']['personal_resources_after']['臆想'], 32)
        self.assertEqual(details['e3']['personal_resources_after']['臆想'], 48)
        dream_buffs = {
            buff['definition_id']: buff
            for buff in details['q']['triggered_buffs']
        }
        self.assertEqual(dream_buffs['character_iloy_q_dream']['duration_ticks'], 48)
        self.assertIn('character_iloy_q_ivory_dream', dream_buffs)
        self.assertNotIn('个人资源 臆想 不足。', details['dream-z1']['warnings'])
        self.assertEqual(details['dream-z1']['personal_resources_after']['臆想'], 43)
        self.assertIn('臆想 正在持续消耗，期间无法积攒。', details['dream-e']['warnings'])
        self.assertEqual(details['dream-e']['personal_resources_after']['臆想'], 33)
        self.assertIn('个人资源 臆想 不足。', details['awake-z1']['warnings'])

    def test_iloy_q_under_thirty_reverie_does_not_open_ivory_dream(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_a01c39f576',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_afea1e6fb2', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_307ad00e8e', 'start_tick': 20},
                {'id': 'dot', 'slot': 0, 'action_id': 'action_c371c893ce', 'start_tick': 25},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        dream_buffs = {
            buff['definition_id']: buff
            for buff in details['q']['triggered_buffs']
        }

        self.assertEqual(dream_buffs['character_iloy_q_dream']['duration_ticks'], 16)
        self.assertNotIn('character_iloy_q_ivory_dream', dream_buffs)
        self.assertIn('动作需要处于 象牙门梦境 状态。', details['dot']['warnings'])

    def test_iloy_zero_reverie_dream_has_hidden_sixteen_tick_floor_and_waives_attack_costs(self) -> None:
        catalog = load_shaft_catalog()
        rules = registered_buff_rules(
            [{'slot': 0, 'character_id': 'char_a01c39f576', 'arc_id': '', 'cartridge_id': ''}],
            catalog,
        )
        dream_rule = next(rule for rule in rules if rule['id'] == 'character_iloy_q_dream')
        self.assertFalse(dream_rule['display']['line'])

        for action_id in ('action_iloy_z1', 'action_iloy_lucid_dream'):
            with self.subTest(action_id=action_id):
                result = simulate_shaft_axis({
                    'team': [{
                        'slot': 0,
                        'character_id': 'char_a01c39f576',
                        'arc_id': '',
                        'cartridge_id': '',
                    }],
                    'steps': [
                        {'id': 'q', 'slot': 0, 'action_id': 'action_307ad00e8e', 'start_tick': 0},
                        {'id': 'dream-attack', 'slot': 0, 'action_id': action_id, 'start_tick': 1},
                    ],
                    'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                })['result']
                details = {detail['step_id']: detail for detail in result['details']}
                dream_buffs = {
                    buff['definition_id']: buff
                    for buff in details['q']['triggered_buffs']
                }

                self.assertEqual(dream_buffs['character_iloy_q_dream']['duration_ticks'], 16)
                self.assertNotIn('个人资源 臆想 不足。', details['dream-attack']['warnings'])
                self.assertEqual(details['dream-attack']['personal_resources_after'].get('臆想', 0), 0)

    def test_daphne_shift_follow_up_returns_one_hundred_twenty_extra_energy(self) -> None:
        catalog = load_shaft_catalog()
        action = next(
            action
            for action in catalog['actions']
            if action['id'] == 'action_016c708ddb'
        )
        self.assertEqual(action['energy_gain'], 0)
        self.assertEqual(action['energy_return'], 120)

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_e8ad982185',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'shift-follow-up',
                'slot': 0,
                'action_id': 'action_016c708ddb',
                'start_tick': 0,
            }],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']
        detail = result['details'][0]

        self.assertEqual(detail['energy_return'], 120)
        self.assertEqual(detail['energy_after'], 120)

    def test_edgar_and_iloy_e_q_healing_actions_trigger_wrong_door(self) -> None:
        cases = (
            ('char_31c5130304', 'action_4a29818248'),
            ('char_31c5130304', 'action_16e181ce7c'),
            ('char_a01c39f576', 'action_afea1e6fb2'),
            ('char_a01c39f576', 'action_iloy_long_e'),
            ('char_a01c39f576', 'action_307ad00e8e'),
        )

        for character_id, action_id in cases:
            with self.subTest(character_id=character_id, action_id=action_id):
                result = simulate_shaft_axis({
                    'team': [{
                        'slot': 0,
                        'character_id': character_id,
                        'arc_id': 'arc_bab179ec33',
                        'arc_refinement': 1,
                        'cartridge_id': '',
                    }],
                    'steps': [{
                        'id': 'healing_action',
                        'slot': 0,
                        'action_id': action_id,
                        'start_tick': 0,
                    }],
                    'initial_energy': 200,
                    'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
                })['result']
                triggered_rule_ids = {
                    buff['rule_id']
                    for buff in result['details'][0]['triggered_buffs']
                }
                self.assertIn('arc_wrong_door_heal_spirit', triggered_rule_ids)
                self.assertIn('arc_wrong_door_heal_team_damage', triggered_rule_ids)

    def test_xun_has_no_energy_or_cooldown_restrictions(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_15f458f7ef',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first_e', 'slot': 0, 'action_id': 'action_0c4da30973', 'start_tick': 0},
                {'id': 'second_e', 'slot': 0, 'action_id': 'action_0c4da30973', 'start_tick': 20},
                {'id': 'q', 'slot': 0, 'action_id': 'action_50de5ed4be', 'start_tick': 40},
            ],
            'initial_energy': 0,
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        details = {detail['step_id']: detail for detail in result['details']}
        self.assertNotIn('动作 CD 尚未结束', '；'.join(details['second_e']['warnings']))
        self.assertNotIn('终结技能量不足。', details['q']['warnings'])
        self.assertEqual(details['q']['energy_after'], 0)
        self.assertEqual(result['resources_by_slot'][0]['initial_energy'], 0)
        self.assertEqual(result['resources_by_slot'][0]['energy'], 0)

    def test_xun_q_terminal_enhancement_is_baked_into_action_multiplier(self) -> None:
        detail = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_15f458f7ef',
                'arc_id': '',
                'cartridge_id': '',
                'skill_levels': {'ultimate': 10},
            }],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_50de5ed4be', 'start_tick': 0},
            ],
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']['details'][0]

        all_rule_ids = {
            buff['rule_id']
            for buff in [*detail['triggered_buffs'], *detail['applied_buffs']]
        }
        self.assertNotIn('character_xun_q_damage', all_rule_ids)
        self.assertNotIn('character_xun_q_record_damage', all_rule_ids)
        self.assertNotIn('character_xun_q_time_stop_damage', all_rule_ids)
        self.assertNotIn('character_xun_q_def_ignore', all_rule_ids)
        self.assertIn('character_xun_q_consume_records', all_rule_ids)
        self.assertAlmostEqual(
            detail['formula_parts']['base'] / detail['panel']['atk'],
            23 * (1.08 ** 9),
        )
        self.assertAlmostEqual(detail['panel']['all_dmg'], 0.0)
        self.assertAlmostEqual(detail['formula_parts']['defense'], 0.5)

    def test_native_background_action_multiplier_overlaps_damage_at_same_tick(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_4f917797cb',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {
                    'id': 'background',
                    'slot': 0,
                    'action_id': 'action_ed6a3e3c26',
                    'start_tick': 0,
                    'repeat': 3,
                },
            ],
        }

        multiplied = simulate_shaft_axis(payload)
        single = simulate_shaft_axis({
            **payload,
            'steps': [{**payload['steps'][0], 'repeat': 1}],
        })
        detail = multiplied['result']['details'][0]
        single_detail = single['result']['details'][0]

        self.assertEqual(multiplied['axis']['steps'][0]['repeat'], 3)
        self.assertEqual(detail['action_multiplier'], 3)
        self.assertEqual(detail['start_tick'], single_detail['start_tick'])
        self.assertEqual(detail['end_tick'], single_detail['end_tick'])
        self.assertEqual(detail['duration_ticks'], 0)
        self.assertEqual(detail['visual_end_tick'] - detail['visual_start_tick'], 5)
        self.assertEqual(detail['display_duration_ticks'], 0)
        self.assertEqual(detail['tick_duration_ticks'], 5)
        self.assertEqual(detail['hit_count'], single_detail['hit_count'] * 3)
        self.assertAlmostEqual(detail['direct_damage'], single_detail['direct_damage'] * 3)

    def test_native_background_action_multiplier_is_capped_at_999(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_4f917797cb',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {
                    'id': 'background',
                    'slot': 0,
                    'action_id': 'action_ed6a3e3c26',
                    'start_tick': 0,
                    'repeat': 1000,
                },
            ],
        }

        multiplied = simulate_shaft_axis(payload)
        single = simulate_shaft_axis({
            **payload,
            'steps': [{**payload['steps'][0], 'repeat': 1}],
        })
        detail = multiplied['result']['details'][0]
        single_detail = single['result']['details'][0]

        self.assertEqual(multiplied['axis']['steps'][0]['repeat'], 999)
        self.assertEqual(detail['action_multiplier'], 999)
        self.assertEqual(detail['hit_count'], single_detail['hit_count'] * 999)
        self.assertAlmostEqual(detail['direct_damage'], single_detail['direct_damage'] * 999)

    def test_native_zero_background_action_ignores_switch_gap_and_q_release_with_fixed_footprint(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_4f917797cb', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'native_bg', 'slot': 1, 'action_id': 'action_ed6a3e3c26', 'start_tick': 2},
            ],
            'options': {'switch_loss_ticks': 2, 'switch_gap_ticks': 2},
            'initial_energy': 200,
            'team_panel_bonus': self.ZERO_TEAM_PANEL_BONUS,
        })['result']

        detail = {item['step_id']: item for item in result['details']}['native_bg']
        self.assertTrue(detail['is_background_damage'])
        self.assertFalse(detail['q_instant_release'])
        self.assertEqual(detail['switch_gap_ticks'], 0)
        self.assertEqual(detail['duration_ticks'], 0)
        self.assertEqual(detail['end_tick'], detail['start_tick'])
        self.assertEqual(detail['display_duration_ticks'], 0)
        self.assertEqual(detail['visual_start_tick'], 2)
        self.assertEqual(detail['visual_end_tick'], 7)
        self.assertEqual(detail['tick_duration_ticks'], 5)

    def test_native_background_catalog_actions_are_instant_except_declared_exceptions(self) -> None:
        catalog = load_shaft_catalog()
        offenders = []
        for action in catalog['actions']:
            marker_text = f'{action.get("name") or ""} {action.get("extra_tag") or ""}'
            is_background = bool(action.get('is_background_damage')) or '后台' in marker_text
            is_declared_exception = (
                str(action.get('action_type') or '') == '援护'
                or bool(action.get('pre_input_node'))
                or bool(action.get('periodic_damage'))
            )
            if is_background and not is_declared_exception and int(action.get('duration_ticks') or 0) != 0:
                offenders.append(f'{action.get("character_name")} {action.get("name")} {action.get("id")}')

        self.assertEqual(offenders, [])

    def test_manual_background_basic_cannot_keep_action_multiplier(self) -> None:
        normalized = normalize_axis_payload({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_d38b672525',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {
                    'id': 'manual_background_basic',
                    'slot': 0,
                    'action_id': 'action_48204bed53',
                    'start_tick': 0,
                    'placement': 'background',
                    'repeat': 7,
                },
            ],
        })

        self.assertEqual(normalized['steps'][0]['placement'], 'background')
        self.assertEqual(normalized['steps'][0]['repeat'], 1)

    def test_team_furniture_bonus_uses_actual_caps(self) -> None:
        normalized = normalize_axis_payload({})

        self.assertEqual(
            normalized['team_panel_bonus'],
            {
                'version': 3,
                'furniture_crit_dmg': 0.04,
                'furniture_flat_atk': 20.0,
                'furniture_flat_def': 30.0,
                'small_flat_atk': 420.0,
                'small_flat_hp': 5200.0,
            },
        )

    def test_team_furniture_bonus_is_editable_and_clamped_to_caps(self) -> None:
        normalized = normalize_axis_payload({
            'team_panel_bonus': {
                'version': 3,
                'furniture_crit_dmg': 0.0264,
                'furniture_flat_atk': 12.8,
                'furniture_flat_def': 99,
            },
        })

        self.assertEqual(normalized['team_panel_bonus']['furniture_crit_dmg'], 0.026)
        self.assertEqual(normalized['team_panel_bonus']['furniture_flat_atk'], 13.0)
        self.assertEqual(normalized['team_panel_bonus']['furniture_flat_def'], 30.0)

        minimum = normalize_axis_payload({
            'team_panel_bonus': {
                'version': 3,
                'furniture_crit_dmg': -1,
                'furniture_flat_atk': -1,
                'furniture_flat_def': -1,
            },
        })['team_panel_bonus']
        self.assertEqual(minimum['furniture_crit_dmg'], 0.0)
        self.assertEqual(minimum['furniture_flat_atk'], 0.0)
        self.assertEqual(minimum['furniture_flat_def'], 0.0)

    def test_support_owner_can_chain_q_without_switch_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'other_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'support', 'slot': 0, 'action_id': 'action_33c0ac631e', 'start_tick': 2},
                {'id': 'owner_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 16},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['support']['visual_start_tick'], 2)
        self.assertEqual(details['support']['visual_end_tick'], 16)
        self.assertEqual(details['owner_q']['visual_start_tick'], 16)

    def test_different_character_can_start_when_support_ends_without_switch_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_7578b18979',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'yi_support', 'slot': 0, 'action_id': 'action_31c54a4e71', 'start_tick': 0},
                {'id': 'main_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 15},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['yi_support']['visual_end_tick'], 15)
        self.assertEqual(details['main_q']['raw_start_tick'], 15)
        self.assertEqual(details['main_q']['visual_start_tick'], 15)
        self.assertEqual(details['main_q']['start_tick'], 15)

    def test_different_character_zero_duration_q_does_not_pay_switch_loss(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'yi_q', 'slot': 1, 'action_id': 'action_6ece34aff8', 'start_tick': 15},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['yi_q']['raw_start_tick'], 15)
        self.assertEqual(details['yi_q']['visual_start_tick'], 15)
        self.assertEqual(details['yi_q']['start_tick'], 15)

    def test_zhenhong_opening_places_yi_q_at_three_point_two_seconds(self) -> None:
        payload = {
            'team': [
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'yi_support', 'slot': 3, 'action_id': 'action_31c54a4e71', 'start_tick': 0},
                {'id': 'main_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 15},
                {'id': 'main_e', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 20},
                {'id': 'nanali_support', 'slot': 2, 'action_id': 'action_482b5d9df7', 'start_tick': 22},
                {'id': 'nanali_q', 'slot': 2, 'action_id': 'action_0f1e29b8ca', 'start_tick': 35},
                {'id': 'nanali_e', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 40},
                {'id': 'yi_q', 'slot': 3, 'action_id': 'action_6ece34aff8', 'start_tick': 42},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        yi_q = next(detail for detail in result['details'] if detail['step_id'] == 'yi_q')

        self.assertEqual(yi_q['visual_start_tick'], 42)
        self.assertEqual(yi_q['start_tick'], 32)

    def test_support_locks_other_characters_until_its_end_without_post_lock_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'support', 'slot': 0, 'action_id': 'action_33c0ac631e', 'start_tick': 0},
                {'id': 'intruding_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 5},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['support']['visual_start_tick'], 0)
        self.assertEqual(details['support']['visual_end_tick'], 14)
        self.assertEqual(details['intruding_e']['raw_start_tick'], 5)
        self.assertEqual(details['intruding_e']['visual_start_tick'], 14)
        self.assertEqual(details['intruding_e']['foreground_lock_ticks'], 9)
        self.assertEqual(details['intruding_e']['switch_gap_ticks'], 0)

    def test_zero_duration_q_locks_other_characters_for_its_visual_ticks(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'same_character', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 2},
                {'id': 'other_character', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 2},
            ],
            'options': {'switch_gap_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual(details['same_character']['visual_start_tick'], 5)
        self.assertEqual(details['same_character']['foreground_lock_ticks'], 3)
        self.assertEqual(details['other_character']['visual_start_tick'], 7)
        self.assertEqual(details['other_character']['foreground_lock_ticks'], 3)
        self.assertEqual(details['other_character']['switch_gap_ticks'], 2)

    def test_different_foreground_starts_with_short_gap_still_compute(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'front_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanaly_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 1},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['details']), 2)

    def test_different_foreground_starts_allow_two_tick_gap(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'front_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanaly_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 2},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['front_e']['start_tick'], 0)
        self.assertEqual(details['nanaly_e']['start_tick'], 2)

    def test_q_display_ticks_drive_forest_firefly_buff_line(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 2,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanaly_q', 'slot': 2, 'action_id': 'action_0f1e29b8ca', 'start_tick': 7},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        nanaly_q = details['nanaly_q']
        firefly_buffs = [
            buff for buff in nanaly_q['triggered_buffs']
            if buff.get('rule_id') == 'cartridge_forest_firefly_crit_stack'
        ]

        self.assertEqual(nanaly_q['raw_start_tick'], 7)
        self.assertEqual(nanaly_q['visual_start_tick'], 7)
        self.assertEqual(nanaly_q['display_start_tick'], nanaly_q['visual_start_tick'])
        self.assertEqual(nanaly_q['display_visual_end_tick'], nanaly_q['visual_end_tick'])
        self.assertEqual(nanaly_q['q_cover_target_step_ids'], ['main_e'])
        self.assertEqual(len(firefly_buffs), 1)
        self.assertEqual(firefly_buffs[0]['trigger_tick'], nanaly_q['start_tick'])
        self.assertEqual(firefly_buffs[0]['display_start_tick'], nanaly_q['display_start_tick'])
        self.assertEqual(firefly_buffs[0]['visual_start_tick'], nanaly_q['display_start_tick'])

    def test_foreground_starts_allow_two_internal_sequence_gap_at_same_real_time(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'main_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'zhenhong_q', 'slot': 1, 'action_id': 'action_c32b4b9417', 'start_tick': 2},
                {'id': 'dragon_a1', 'slot': 1, 'action_id': 'action_40e3b63106', 'start_tick': 4},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['main_q']['start_tick'], 0)
        self.assertEqual(details['zhenhong_q']['start_tick'], 0)
        self.assertEqual(details['dragon_a1']['start_tick'], 0)
        self.assertEqual(details['dragon_a1']['end_tick'], 3)
        self.assertEqual(details['dragon_a1']['duration_ticks'], 3)
        self.assertEqual(details['dragon_a1']['visual_start_tick'], details['zhenhong_q']['visual_end_tick'])
        self.assertEqual(details['dragon_a1']['visual_end_tick'] - details['dragon_a1']['visual_start_tick'], 3)

    def test_foreground_starts_same_visual_tick_with_q_still_compute(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'main_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'dragon_a1', 'slot': 1, 'action_id': 'action_40e3b63106', 'start_tick': 0},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['details']), 2)

    def test_zero_duration_q_end_can_switch_character_without_switch_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_27dc4a7281',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'zhenhong_q2', 'slot': 0, 'action_id': 'action_e3711f0cf5', 'start_tick': 0},
                {'id': 'nanaly_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 5},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['zhenhong_q2']['start_tick'], 0)
        self.assertEqual(details['zhenhong_q2']['end_tick'], 0)
        self.assertEqual(details['nanaly_e']['start_tick'], 0)

    def test_non_q_end_switches_character_with_switch_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'front_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanaly_e', 'slot': 1, 'action_id': 'action_5870d8ba67', 'start_tick': 15},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['front_e']['end_tick'], 15)
        self.assertEqual(details['nanaly_e']['start_tick'], 15)
        self.assertEqual(details['nanaly_e']['visual_start_tick'], 15)
        self.assertEqual(details['nanaly_e']['switch_gap_ticks'], 0)

    def test_q_starts_can_share_visual_column_for_tolerant_compute(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
            ],
            'steps': [
                {'id': 'main_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 0},
                {'id': 'zhenhong_q', 'slot': 1, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['details']), 2)


class ShaftSimulatorQInstantReleaseTestCase(unittest.TestCase):
    def test_frozen_q_interval_has_one_real_time_and_preserves_action_tick_lengths(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'covered_e', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 7},
            ],
            'options': {'switch_gap_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        covered_e = details['covered_e']
        q_interval = result['time_axis']['frozen_intervals'][0]

        self.assertEqual(covered_e['visual_end_tick'] - covered_e['visual_start_tick'], covered_e['original_duration_ticks'])
        self.assertEqual(details['zhenhong_q']['visual_end_tick'], covered_e['visual_end_tick'])
        frozen_start = q_interval['start_tick']
        frozen_end = q_interval['end_tick']

        def real_tick(visual_tick: int) -> int:
            frozen_ticks = sum(
                max(0, min(visual_tick, interval['end_tick']) - interval['start_tick'])
                for interval in result['time_axis']['frozen_intervals']
                if visual_tick > interval['start_tick']
            )
            return visual_tick - frozen_ticks

        self.assertTrue(all(
            real_tick(visual_tick) == real_tick(frozen_start)
            for visual_tick in range(frozen_start, frozen_end + 1)
        ))
        self.assertEqual(covered_e['start_tick'], real_tick(covered_e['visual_start_tick']))
        self.assertEqual(covered_e['end_tick'], real_tick(covered_e['visual_end_tick']))

    def test_expanded_q_lock_delays_switch_and_carries_the_character_sequence(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'covered_e', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 7},
                {'id': 'switched_e', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 12},
                {'id': 'same_character_next', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 23},
            ],
            'options': {'switch_gap_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual(details['q']['visual_end_tick'], 15)
        self.assertEqual(details['q']['visual_end_tick'], details['switched_e']['visual_start_tick'])
        self.assertEqual(details['switched_e']['visual_start_tick'], 15)
        self.assertEqual(details['switched_e']['foreground_lock_ticks'], 3)
        self.assertEqual(details['same_character_next']['visual_start_tick'], 26)
        self.assertEqual(details['switched_e']['start_tick'], 7)

    def test_true_red_opening_yi_e_starts_at_expanded_q_right_edge(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_7578b18979', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'main_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 15},
                {'id': 'nanaly_q', 'slot': 2, 'action_id': 'action_0f1e29b8ca', 'start_tick': 35},
                {'id': 'nanaly_e', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 40},
                {'id': 'yi_q', 'slot': 3, 'action_id': 'action_6ece34aff8', 'start_tick': 42},
                {'id': 'yi_e', 'slot': 3, 'action_id': 'action_2e072f2b0b', 'start_tick': 47},
                {'id': 'true_red_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 49},
            ],
            'options': {'switch_gap_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual((details['yi_q']['visual_start_tick'], details['yi_q']['visual_end_tick']), (42, 51))
        self.assertEqual(details['yi_q']['start_tick'], 32)
        self.assertEqual(details['yi_e']['visual_start_tick'], details['yi_q']['visual_end_tick'])
        self.assertEqual(details['yi_e']['visual_start_tick'], 51)
        self.assertEqual(details['yi_e']['start_tick'], 32)
        self.assertEqual(details['true_red_e']['visual_start_tick'], 53)

    def test_buff_duration_and_cooldown_use_real_time_across_q_freeze(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'first_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 16},
                {'id': 'inside_real_duration', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 24},
                {'id': 'too_early_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 32},
                {'id': 'ready_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 35},
            ],
            'buff_rules': [
                {
                    'id': 'real_time_buff',
                    'name': '真实时间增益',
                    'trigger': {'slot': 0, 'action_id': 'action_3987d8ff2d'},
                    'targets': {'slots': [0]},
                    'duration_ticks': 20,
                    'modifiers': {'atk_pct': 0.5},
                },
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertEqual(details['inside_real_duration']['start_tick'], 19)
        self.assertIn('real_time_buff', {
            buff['rule_id'] for buff in details['inside_real_duration']['applied_buffs']
        })
        self.assertEqual(details['too_early_e']['start_tick'], 27)
        self.assertTrue(any('CD' in warning for warning in details['too_early_e']['warnings']))

        payload['steps'] = [
            step for step in payload['steps']
            if step['id'] != 'too_early_e'
        ]
        ready_result = simulate_shaft_axis(payload)['result']
        ready_e = next(detail for detail in ready_result['details'] if detail['step_id'] == 'ready_e')
        self.assertEqual(ready_e['start_tick'], 30)
        self.assertFalse(any('CD' in warning for warning in ready_e['warnings']))

    def test_q_does_not_cover_other_foreground_support_or_q(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 2,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'main_support', 'slot': 0, 'action_id': 'action_33c0ac631e', 'start_tick': 0},
                {'id': 'zhenhong_q2', 'slot': 1, 'action_id': 'action_e3711f0cf5', 'start_tick': 0},
                {'id': 'nanaly_q', 'slot': 2, 'action_id': 'action_0f1e29b8ca', 'start_tick': 4},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertFalse(details['main_support']['q_instant_release'])
        self.assertEqual(details['main_support']['original_duration_ticks'], 14)
        self.assertEqual(details['main_support']['visual_end_tick'] - details['main_support']['visual_start_tick'], 14)
        self.assertEqual(details['main_support']['duration_ticks'], 14)
        self.assertEqual(details['zhenhong_q2']['visual_start_tick'], 14)
        self.assertFalse(details['zhenhong_q2']['q_instant_release'])
        self.assertEqual(details['zhenhong_q2']['duration_ticks'], 0)
        self.assertEqual(details['nanaly_q']['visual_start_tick'], 19)
        self.assertEqual(details['nanaly_q'].get('q_cover_target_step_ids', []), [])

    def test_zhenhong_traverse_is_an_ordinary_action_but_q2_has_q_cover(self) -> None:
        base_payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }
        traverse_payload = {
            **base_payload,
            'steps': [
                {'id': 'traverse', 'slot': 1, 'action_id': 'action_a17ac7c5b7', 'start_tick': 0},
                {'id': 'main_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 2},
            ],
        }
        q2_payload = {
            **base_payload,
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'zhenhong_q2', 'slot': 1, 'action_id': 'action_e3711f0cf5', 'start_tick': 2},
            ],
        }
        q2_avoid_payload = {
            **base_payload,
            'steps': [
                {'id': 'zhenhong_q2', 'slot': 1, 'action_id': 'action_e3711f0cf5', 'start_tick': 0},
                {'id': 'main_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 2},
            ],
        }

        traverse_result = simulate_shaft_axis(traverse_payload)['result']
        traverse_details = {detail['step_id']: detail for detail in traverse_result['details']}
        q2_result = simulate_shaft_axis(q2_payload)['result']
        q2_details = {detail['step_id']: detail for detail in q2_result['details']}
        q2_avoid_result = simulate_shaft_axis(q2_avoid_payload)['result']
        q2_avoid_details = {detail['step_id']: detail for detail in q2_avoid_result['details']}

        self.assertTrue(traverse_details['traverse']['q_instant_release'])
        self.assertEqual(traverse_details['traverse']['duration_ticks'], 2)
        self.assertFalse(q2_details['zhenhong_q2']['q_instant_release'])
        self.assertTrue(q2_details['main_e']['q_instant_release'])
        self.assertEqual(q2_details['zhenhong_q2']['q_cover_target_step_ids'], ['main_e'])
        self.assertFalse(q2_avoid_details['zhenhong_q2']['q_instant_release'])

    def test_q_instant_release_keeps_visual_length_and_elapsed_until_q(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_27dc4a7281',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': 'arc_a5b483cca6',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 2,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'foreground_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'same_column_e', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 2},
                {'id': 'q_anchor', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 4},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        foreground = details['foreground_e']
        self.assertTrue(foreground['q_instant_release'])
        self.assertEqual(foreground['q_instant_release_kind'], 'column')
        self.assertEqual(foreground['end_tick'], 4)
        self.assertEqual(foreground['duration_ticks'], 4)
        self.assertEqual(foreground['calculation_start_sequence'], 0)
        self.assertEqual(foreground['calculation_end_sequence'], 1)
        self.assertEqual(foreground['visual_end_tick'], 10)
        self.assertEqual(foreground['original_duration_ticks'], 10)
        self.assertEqual(foreground['q_instant_release_calculation_tick'], 4)
        self.assertEqual(foreground['q_instant_release_anchor_tick'], 4)
        self.assertEqual(foreground['q_instant_release_anchor_step_id'], 'q_anchor')
        self.assertEqual(foreground['q_instant_release_start_sequence'], 0)
        self.assertEqual(foreground['q_instant_release_end_sequence'], 1)

        q_anchor = details['q_anchor']
        self.assertEqual(q_anchor['raw_start_tick'], 4)
        self.assertEqual(q_anchor['start_tick'], 4)
        self.assertEqual(q_anchor['duration_ticks'], 0)
        self.assertEqual(q_anchor['visual_end_tick'], 13)
        self.assertEqual(q_anchor['q_cover_visual_end_tick'], 13)
        self.assertEqual(q_anchor['q_cover_target_step_ids'], ['foreground_e', 'same_column_e'])

        same_column = details['same_column_e']
        self.assertTrue(same_column['q_instant_release'])
        self.assertEqual(same_column['q_instant_release_kind'], 'column')
        self.assertEqual(same_column['end_tick'], 4)
        self.assertEqual(same_column['duration_ticks'], 2)
        self.assertEqual(same_column['calculation_start_sequence'], 0)
        self.assertEqual(same_column['calculation_end_sequence'], 1)
        self.assertEqual(same_column['visual_end_tick'], 13)
        self.assertEqual(same_column['original_duration_ticks'], 11)
        self.assertEqual(same_column['q_instant_release_calculation_tick'], 4)
        self.assertEqual(same_column['q_instant_release_anchor_tick'], 4)
        self.assertEqual(same_column['q_instant_release_anchor_step_id'], 'q_anchor')
        self.assertEqual(same_column['q_instant_release_start_sequence'], 0)
        self.assertEqual(same_column['q_instant_release_end_sequence'], 1)

    def test_q_instant_release_keeps_q_visual_width(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': 'arc_a5b483cca6',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_27dc4a7281',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 2,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'foreground_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'same_column_e', 'slot': 2, 'action_id': 'action_5870d8ba67', 'start_tick': 2},
                {'id': 'q_anchor', 'slot': 1, 'action_id': 'action_e3711f0cf5', 'start_tick': 4},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        q_anchor = details['q_anchor']
        self.assertEqual(q_anchor['start_tick'], 4)
        self.assertEqual(q_anchor['end_tick'], 4)
        self.assertEqual(q_anchor['duration_ticks'], 0)
        self.assertEqual(q_anchor['visual_end_tick'], 15)
        self.assertEqual(q_anchor['q_cover_visual_end_tick'], 15)
        self.assertEqual(q_anchor['q_cover_target_step_ids'], ['foreground_e', 'same_column_e'])
        self.assertFalse(q_anchor['q_instant_release'])

        foreground = details['foreground_e']
        self.assertEqual(foreground['q_instant_release_tick'], 4)
        self.assertEqual(foreground['end_tick'], 4)
        self.assertEqual(foreground['duration_ticks'], 4)
        self.assertEqual(foreground['calculation_start_sequence'], 0)
        self.assertEqual(foreground['calculation_end_sequence'], 1)
        self.assertEqual(foreground['visual_end_tick'], 15)
        self.assertEqual(foreground['q_instant_release_calculation_tick'], 4)
        self.assertEqual(foreground['q_instant_release_anchor_tick'], 4)
        self.assertEqual(foreground['q_instant_release_start_sequence'], 0)
        self.assertEqual(foreground['q_instant_release_end_sequence'], 1)

        same_column = details['same_column_e']
        self.assertEqual(same_column['q_instant_release_tick'], 4)
        self.assertEqual(same_column['end_tick'], 4)
        self.assertEqual(same_column['duration_ticks'], 2)
        self.assertEqual(same_column['calculation_start_sequence'], 0)
        self.assertEqual(same_column['calculation_end_sequence'], 1)
        self.assertEqual(same_column['visual_end_tick'], 13)
        self.assertEqual(same_column['q_instant_release_calculation_tick'], 4)
        self.assertEqual(same_column['q_instant_release_anchor_tick'], 4)
        self.assertEqual(same_column['q_instant_release_start_sequence'], 0)
        self.assertEqual(same_column['q_instant_release_end_sequence'], 1)

    def test_q_instant_release_covers_connected_basic_background_attacks(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_d38b672525',
                    'arc_id': 'arc_27dc4a7281',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'ka_a2', 'slot': 1, 'action_id': 'action_48204bed53', 'start_tick': 0, 'placement': 'background'},
                {'id': 'q_anchor', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 2},
                {'id': 'ka_a3', 'slot': 1, 'action_id': 'action_09bd76720d', 'start_tick': 6, 'placement': 'background'},
                {'id': 'ka_a4', 'slot': 1, 'action_id': 'action_90b1619681', 'start_tick': 11},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        q_anchor = details['q_anchor']
        self.assertEqual(q_anchor['visual_end_tick'], 21)
        self.assertEqual(q_anchor['q_cover_visual_end_tick'], 21)
        self.assertEqual(q_anchor['q_cover_target_step_ids'], ['ka_a2', 'ka_a3', 'ka_a4'])

        first_basic = details['ka_a2']
        self.assertTrue(first_basic['is_basic_background'])
        self.assertTrue(first_basic['q_instant_release'])
        self.assertEqual(first_basic['q_instant_release_kind'], 'column')
        self.assertEqual(first_basic['duration_ticks'], 2)
        self.assertEqual(first_basic['end_tick'], 2)
        self.assertEqual(first_basic['calculation_start_sequence'], 0)
        self.assertEqual(first_basic['calculation_end_sequence'], 1)
        self.assertEqual(first_basic['visual_end_tick'], 6)
        self.assertEqual(first_basic['q_instant_release_calculation_tick'], 2)
        self.assertEqual(first_basic['q_instant_release_start_sequence'], 0)
        self.assertEqual(first_basic['q_instant_release_end_sequence'], 1)

        chained_basic = details['ka_a3']
        self.assertTrue(chained_basic['is_basic_background'])
        self.assertTrue(chained_basic['q_instant_release'])
        self.assertEqual(chained_basic['q_instant_release_kind'], 'basic-background')
        self.assertEqual(chained_basic['duration_ticks'], 0)
        self.assertEqual(chained_basic['start_tick'], 2)
        self.assertEqual(chained_basic['end_tick'], 2)
        self.assertEqual(chained_basic['calculation_start_sequence'], 1)
        self.assertEqual(chained_basic['calculation_end_sequence'], 2)
        self.assertEqual(chained_basic['visual_start_tick'], 6)
        self.assertEqual(chained_basic['visual_end_tick'], 11)
        self.assertEqual(chained_basic['q_instant_release_calculation_tick'], 2)
        self.assertEqual(chained_basic['q_instant_release_start_sequence'], 1)
        self.assertEqual(chained_basic['q_instant_release_end_sequence'], 2)

        foreground_capable_basic = details['ka_a4']
        self.assertFalse(foreground_capable_basic['is_basic_background'])
        self.assertTrue(foreground_capable_basic['q_instant_release'])
        self.assertEqual(foreground_capable_basic['q_instant_release_kind'], 'basic-background')
        self.assertEqual(foreground_capable_basic['duration_ticks'], 0)
        self.assertEqual(foreground_capable_basic['start_tick'], 2)
        self.assertEqual(foreground_capable_basic['end_tick'], 2)
        self.assertEqual(foreground_capable_basic['calculation_start_sequence'], 2)
        self.assertEqual(foreground_capable_basic['calculation_end_sequence'], 3)
        self.assertEqual(foreground_capable_basic['visual_start_tick'], 11)
        self.assertEqual(foreground_capable_basic['visual_end_tick'], 21)
        self.assertEqual(foreground_capable_basic['q_instant_release_calculation_tick'], 2)
        self.assertEqual(foreground_capable_basic['q_instant_release_start_sequence'], 2)
        self.assertEqual(foreground_capable_basic['q_instant_release_end_sequence'], 3)

    def test_q_instant_release_does_not_reach_back_to_completed_basic_attacks(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
            ],
            'steps': [
                {'id': 'completed_a2', 'slot': 0, 'action_id': 'action_8724db7bac', 'start_tick': 0},
                {'id': 'completed_a3', 'slot': 0, 'action_id': 'action_8fd4a72c7d', 'start_tick': 3},
                {'id': 'covered_a5', 'slot': 0, 'action_id': 'action_74adab43b7', 'start_tick': 20, 'placement': 'background'},
                {'id': 'q_anchor', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 22},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertFalse(details['completed_a2']['q_instant_release'])
        self.assertFalse(details['completed_a3']['q_instant_release'])
        self.assertTrue(details['covered_a5']['q_instant_release'])
        self.assertEqual(details['q_anchor']['q_cover_target_step_ids'], ['covered_a5'])

    def test_q_instant_release_uses_sequence_for_zhenhong_dragon_chain(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'dragon_a4', 'slot': 0, 'action_id': 'action_5cd7ad2380', 'start_tick': 0, 'placement': 'background'},
                {'id': 'teammate_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 2},
                {'id': 'dragon_a5', 'slot': 0, 'action_id': 'action_cf3b21dac1', 'start_tick': 8, 'placement': 'background'},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        dragon_a4 = details['dragon_a4']
        self.assertTrue(dragon_a4['q_instant_release'])
        self.assertEqual(dragon_a4['q_instant_release_kind'], 'column')
        self.assertEqual(dragon_a4['start_tick'], 0)
        self.assertEqual(dragon_a4['end_tick'], 2)
        self.assertEqual(dragon_a4['duration_ticks'], 2)
        self.assertEqual(dragon_a4['calculation_start_sequence'], 0)
        self.assertEqual(dragon_a4['calculation_end_sequence'], 1)
        self.assertEqual(dragon_a4['visual_start_tick'], 0)
        self.assertEqual(dragon_a4['visual_end_tick'], 8)
        self.assertEqual(dragon_a4['q_instant_release_calculation_tick'], 2)
        self.assertEqual(dragon_a4['q_instant_release_start_sequence'], 0)
        self.assertEqual(dragon_a4['q_instant_release_end_sequence'], 1)

        dragon_a5 = details['dragon_a5']
        self.assertTrue(dragon_a5['q_instant_release'])
        self.assertEqual(dragon_a5['q_instant_release_kind'], 'basic-background')
        self.assertEqual(dragon_a5['start_tick'], 2)
        self.assertEqual(dragon_a5['end_tick'], 2)
        self.assertEqual(dragon_a5['duration_ticks'], 0)
        self.assertEqual(dragon_a5['calculation_start_sequence'], 1)
        self.assertEqual(dragon_a5['calculation_end_sequence'], 2)
        self.assertEqual(dragon_a5['visual_start_tick'], 8)
        self.assertEqual(dragon_a5['visual_end_tick'], 18)
        self.assertEqual(dragon_a5['q_instant_release_calculation_tick'], 2)
        self.assertEqual(dragon_a5['q_instant_release_start_sequence'], 1)
        self.assertEqual(dragon_a5['q_instant_release_end_sequence'], 2)

        teammate_q = details['teammate_q']
        self.assertFalse(teammate_q['q_instant_release'])
        self.assertEqual(teammate_q['start_tick'], 2)
        self.assertEqual(teammate_q['end_tick'], 2)
        self.assertEqual(result['summary']['duration_ticks'], 2)

    def test_q_cover_width_expands_over_background_chain_and_keeps_followup_sequence(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'zhenhong_a4', 'slot': 0, 'action_id': 'action_5cd7ad2380', 'start_tick': 0},
                {'id': 'main_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 2},
                {'id': 'zhenhong_a5', 'slot': 0, 'action_id': 'action_cf3b21dac1', 'start_tick': 8, 'placement': 'background'},
                {'id': 'main_e', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 18},
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 20},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        main_q = details['main_q']
        self.assertFalse(main_q['q_instant_release'])
        self.assertEqual(main_q['start_tick'], 2)
        self.assertEqual(main_q['end_tick'], 2)
        self.assertEqual(main_q['visual_start_tick'], 2)
        self.assertEqual(main_q['visual_end_tick'], 18)
        self.assertEqual(main_q['q_cover_visual_end_tick'], 18)
        self.assertEqual(main_q['q_cover_target_step_ids'], ['zhenhong_a4', 'zhenhong_a5'])

        zhenhong_a4 = details['zhenhong_a4']
        self.assertTrue(zhenhong_a4['q_instant_release'])
        self.assertEqual(zhenhong_a4['end_tick'], 2)
        self.assertEqual(zhenhong_a4['duration_ticks'], 2)
        self.assertEqual(zhenhong_a4['calculation_start_sequence'], 0)
        self.assertEqual(zhenhong_a4['calculation_end_sequence'], 1)
        self.assertEqual(zhenhong_a4['visual_end_tick'], 8)

        zhenhong_a5 = details['zhenhong_a5']
        self.assertTrue(zhenhong_a5['q_instant_release'])
        self.assertEqual(zhenhong_a5['start_tick'], 2)
        self.assertEqual(zhenhong_a5['end_tick'], 2)
        self.assertEqual(zhenhong_a5['duration_ticks'], 0)
        self.assertEqual(zhenhong_a5['calculation_start_sequence'], 1)
        self.assertEqual(zhenhong_a5['calculation_end_sequence'], 2)
        self.assertEqual(zhenhong_a5['visual_end_tick'], 18)

        main_e = details['main_e']
        self.assertEqual(main_e['start_tick'], 2)
        self.assertTrue(main_e['q_instant_release'])
        self.assertEqual(main_e['q_instant_release_anchor_step_id'], 'zhenhong_q')
        self.assertEqual(main_e['end_tick'], 4)
        self.assertEqual(main_e['duration_ticks'], 2)

        zhenhong_q = details['zhenhong_q']
        self.assertFalse(zhenhong_q['q_instant_release'])
        self.assertEqual(zhenhong_q['start_tick'], 4)
        self.assertEqual(zhenhong_q['end_tick'], 4)
        self.assertEqual(zhenhong_q['visual_start_tick'], 20)
        self.assertGreaterEqual(zhenhong_q['visual_start_tick'], zhenhong_a5['visual_end_tick'])
        self.assertEqual(result['summary']['duration_ticks'], 4)

    def test_foreground_basic_starting_after_q_sequence_keeps_duration(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'other_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 2},
                {'id': 'dragon_a1', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 4},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        dragon_a1 = details['dragon_a1']

        self.assertFalse(dragon_a1['q_instant_release'])
        self.assertEqual(dragon_a1['duration_ticks'], 3)
        self.assertEqual(dragon_a1['end_tick'], dragon_a1['start_tick'] + 3)
        self.assertEqual(dragon_a1['visual_end_tick'] - dragon_a1['visual_start_tick'], 3)
        self.assertEqual(dragon_a1['original_duration_ticks'], 3)

    def test_support_does_not_receive_switch_loss(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': 'arc_d75aa15b91',
                    'cartridge_id': 'cartridge_29793225a0',
                },
            ],
            'steps': [
                {'id': 'front_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 5},
            ],
            'options': {'switch_loss_ticks': 2},
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['front_e']['start_tick'], 0)
        self.assertEqual(details['support']['start_tick'], 5)
        self.assertEqual(details['support']['duration_ticks'], 13)


class ShaftEquipmentBuffTestCase(unittest.TestCase):
    def test_speed_cotton_stacks_each_front_second_and_clears_on_leave(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': 'arc_126824ee1b',
                    'arc_refinement': 1,
                    'cartridge_id': '',
                },
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'start', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'one_second', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 10},
                {'id': 'five_seconds', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 50},
                {'id': 'leave', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 60},
                {'id': 'return', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 80},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertNotIn('arc_speed_cotton_front_atk', {buff['rule_id'] for buff in details['start']['applied_buffs']})
        one_stack = next(buff for buff in details['one_second']['applied_buffs'] if buff['rule_id'] == 'arc_speed_cotton_front_atk')
        five_stacks = next(buff for buff in details['five_seconds']['applied_buffs'] if buff['rule_id'] == 'arc_speed_cotton_front_atk')
        self.assertEqual(one_stack['effects'], {'atk_pct': 0.05})
        self.assertEqual(five_stacks['stack_count'], 5)
        self.assertEqual(five_stacks['effects'], {'atk_pct': 0.25})
        self.assertNotIn('arc_speed_cotton_front_atk', {buff['rule_id'] for buff in details['return']['applied_buffs']})

    def test_new_arc_note_conditions_for_sagiri_shield_and_front_action(self) -> None:
        good_dog = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_1895e259be',
                'arc_id': 'arc_5167a86a24',
                'arc_refinement': 5,
                'cartridge_id': '',
            }],
            'steps': [{'id': 'q', 'slot': 0, 'action_id': 'action_0db78d1f01', 'start_tick': 0}],
            'initial_energy': 200,
        })['result']['details'][0]
        triggered = {buff['rule_id']: buff for buff in good_dog['triggered_buffs']}
        self.assertEqual(triggered['arc_good_dog_q_team_atk']['effects'], {'atk_pct': 0.16})
        self.assertEqual(triggered['arc_good_dog_q_control_extra_atk']['effects'], {'atk_pct': 0.1})

        thinking_cat = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_4f917797cb',
                'arc_id': 'arc_112b3492d8',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [{'id': 'basic', 'slot': 0, 'action_id': 'action_605c800d26', 'start_tick': 0}],
            'initial_energy': 200,
        })['result']['details'][0]
        thinking_buff = next(
            buff for buff in thinking_cat['applied_buffs']
            if buff['rule_id'] == 'arc_thinking_cat_fons_light'
        )

        self.assertEqual(thinking_buff['effects'], {'element_dmg': 0.25})

        spring = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_6f46705fd1',
                'arc_id': 'arc_5fdc5b1847',
                'arc_refinement': 1,
                'cartridge_id': '',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
            }],
            'steps': [{'id': 'e', 'slot': 0, 'action_id': 'action_25fea9f7cb', 'start_tick': 0}],
            'initial_energy': 200,
        })['result']['details'][0]
        spring_triggered = {buff['rule_id'] for buff in spring['triggered_buffs']}
        self.assertIn('character_adler_e_shield', spring_triggered)
        self.assertIn('character_adler_e_team_curse', spring_triggered)
        self.assertIn('arc_spring_shield_atk', spring_triggered)
        spring_buff = next(buff for buff in spring['triggered_buffs'] if buff['rule_id'] == 'arc_spring_shield_atk')
        self.assertEqual(spring_buff['effects'], {'atk_pct': 0.18})

        danger = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_6f46705fd1',
                'arc_id': 'arc_18855bc4f1',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first', 'slot': 0, 'action_id': 'action_25fea9f7cb', 'start_tick': 0},
                {'id': 'cooldown', 'slot': 0, 'action_id': 'action_55c8843d1c', 'start_tick': 100},
                {'id': 'ready', 'slot': 0, 'action_id': 'action_25fea9f7cb', 'start_tick': 200},
            ],
            'initial_energy': 200,
        })['result']
        danger_details = {detail['step_id']: detail for detail in danger['details']}
        self.assertIn('arc_danger_game_stagger', {buff['rule_id'] for buff in danger_details['first']['triggered_buffs']})
        self.assertNotIn('arc_danger_game_stagger', {buff['rule_id'] for buff in danger_details['cooldown']['triggered_buffs']})
        self.assertIn('arc_danger_game_stagger', {buff['rule_id'] for buff in danger_details['ready']['triggered_buffs']})

    def test_team_unique_buff_keeps_independent_triggers_and_applies_highest_value_once(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_7578b18979',
                    'arc_id': 'arc_5167a86a24',
                    'arc_refinement': 1,
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_1895e259be',
                    'arc_id': 'arc_5167a86a24',
                    'arc_refinement': 5,
                    'cartridge_id': '',
                },
                {'slot': 2, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'low_q', 'slot': 0, 'action_id': 'action_6ece34aff8', 'start_tick': 0},
                {'id': 'high_q', 'slot': 1, 'action_id': 'action_0db78d1f01', 'start_tick': 20},
                {'id': 'inspect', 'slot': 2, 'action_id': 'action_982c67944f', 'start_tick': 40},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertIn(
            'arc_good_dog_q_team_atk',
            {buff['rule_id'] for buff in details['low_q']['triggered_buffs']},
        )
        self.assertIn(
            'arc_good_dog_q_team_atk',
            {buff['rule_id'] for buff in details['high_q']['triggered_buffs']},
        )
        applied = [
            buff
            for buff in details['inspect']['applied_buffs']
            if buff['rule_id'] == 'arc_good_dog_q_team_atk'
        ]
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]['owner_slot'], 1)
        self.assertEqual(applied[0]['effects'], {'atk_pct': 0.16})

    def test_declared_team_unique_sources_have_unique_keys(self) -> None:
        catalog = load_shaft_catalog()
        rules = {rule['id']: rule for rule in catalog['buffs']}
        expected_rule_ids = {
            'arc_wrong_door_heal_team_damage',
            'arc_bit_game_background_front_atk',
            'arc_bit_game_background_damage_front_atk_stack',
            'arc_bit_game_foreground_psyche_dmg',
            'arc_bit_game_basic_psyche_stack',
            'arc_good_dog_q_team_atk',
            'arc_good_dog_q_control_extra_atk',
            'cartridge_sonic_q_team_atk',
        }

        for rule_id in expected_rule_ids:
            with self.subTest(rule_id=rule_id):
                self.assertIn(rule_id, rules)
                self.assertEqual(
                    rules[rule_id]['calculation']['team_unique_key'],
                    rule_id,
                )

    def test_loop_carry_uses_foreground_axis_end_instead_of_late_background_actions(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b2e3b2bf7a',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_a01c39f576',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_823bb4c4eb',
                },
                {
                    'slot': 2,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'opening_jiuyuan_q', 'slot': 0, 'action_id': 'action_5501c87ce7', 'start_tick': 0},
                {'id': 'tail_iloy_q', 'slot': 1, 'action_id': 'action_307ad00e8e', 'start_tick': 50},
                {'id': 'foreground_end', 'slot': 2, 'action_id': 'action_7d6ec164ca', 'start_tick': 60},
                {
                    'id': 'late_background_a5',
                    'slot': 2,
                    'action_id': 'action_74adab43b7',
                    'start_tick': 300,
                    'placement': 'background',
                },
            ],
            'options': {'loop_enabled': True, 'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertLess(
            result['summary']['duration_ticks'],
            details['late_background_a5']['end_tick'],
        )
        self.assertIn(
            'cartridge_sonic_q_team_atk',
            {buff['rule_id'] for buff in details['opening_jiuyuan_q']['applied_buffs']},
        )

    def test_time_outside_records_team_actions_and_q_consumes_three_stacks(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': 'arc_7d0ca08e3d',
                    'arc_refinement': 1,
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'owner_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'team_e_1', 'slot': 1, 'action_id': 'action_3987d8ff2d', 'start_tick': 20},
                {'id': 'team_e_2', 'slot': 1, 'action_id': 'action_3987d8ff2d', 'start_tick': 40},
                {'id': 'team_support', 'slot': 1, 'action_id': 'action_e98dde3662', 'start_tick': 60},
                {'id': 'owner_q', 'slot': 0, 'action_id': 'action_b356b46da2', 'start_tick': 80},
                {'id': 'after_q_team_e', 'slot': 1, 'action_id': 'action_3987d8ff2d', 'start_tick': 100},
                {'id': 'owner_post_q_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 120},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        q_buff = next(
            buff
            for buff in details['owner_q']['triggered_buffs']
            if buff['rule_id'] == 'arc_time_outside_recorded_crit_def'
        )
        self.assertAlmostEqual(q_buff['effects']['crit_dmg'], 0.48)
        self.assertEqual(q_buff['effects']['def_ignore'], 0.12)
        self.assertAlmostEqual(
            details['owner_q']['panel']['crit_dmg'] - details['owner_post_q_e']['panel']['crit_dmg'],
            q_buff['effects']['crit_dmg'],
        )
        self.assertNotIn(
            'arc_time_outside_stack',
            {buff['rule_id'] for buff in details['after_q_team_e']['triggered_buffs']},
        )

    def test_time_outside_buffs_are_not_registered_without_its_arc(self) -> None:
        catalog = load_shaft_catalog()
        rules = registered_buff_rules(
            [
                {
                    'slot': 0,
                    'character_id': 'char_dd034941ef',
                    'arc_id': 'arc_a5b483cca6',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            catalog,
        )

        self.assertFalse(any(str(rule['id']).startswith('arc_time_outside_') for rule in rules))

    def test_xun_records_each_teammate_once_and_q_consumes_records(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_15f458f7ef',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [1, 2, 6],
                },
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'xun_e', 'slot': 0, 'action_id': 'action_0c4da30973', 'start_tick': 0},
                {'id': 'main_e_1', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 10},
                {'id': 'main_e_2', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 20},
                {'id': 'nanali_support', 'slot': 2, 'action_id': 'action_482b5d9df7', 'start_tick': 30},
                {'id': 'xun_q', 'slot': 0, 'action_id': 'action_50de5ed4be', 'start_tick': 40},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertNotIn(
            'character_xun_record_stack',
            {buff['rule_id'] for buff in details['main_e_2']['triggered_buffs']},
        )
        record_buff = next(
            buff
            for buff in details['xun_q']['triggered_buffs']
            if buff['rule_id'] == 'character_xun_q_record_damage'
        )
        time_stop_buff = next(
            buff
            for buff in details['xun_q']['triggered_buffs']
            if buff['rule_id'] == 'character_xun_q_time_stop_damage'
        )
        def_ignore_buff = next(
            buff
            for buff in details['xun_q']['triggered_buffs']
            if buff['rule_id'] == 'character_xun_q_def_ignore'
        )
        self.assertEqual(record_buff['effects'], {'all_dmg': 0.24})
        self.assertEqual(time_stop_buff['effects'], {'all_dmg': 0.2})
        self.assertEqual(def_ignore_buff['effects'], {'def_ignore': 0.3})

    def test_zhenhong_yingxu_passive_stacks_solitary(self) -> None:
        result = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'yingxu_1', 'slot': 0, 'action_id': 'action_5e62f427cb', 'start_tick': 0},
                {'id': 'yingxu_2', 'slot': 0, 'action_id': 'action_5e62f427cb', 'start_tick': 10},
                {'id': 'e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        e_detail = next(detail for detail in result['details'] if detail['step_id'] == 'e')
        solitary = next(buff for buff in e_detail['applied_buffs'] if buff['rule_id'] == 'character_zhenhong_solitary_stack')
        self.assertEqual(solitary['stack_count'], 2)
        self.assertEqual(solitary['effects'], {'atk_pct': 0.1})

    def test_character_buff_can_trigger_from_reaction_event(self) -> None:
        payload = {
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [1, 2, 3, 4, 5, 6]},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'main_e', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'nanali_support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
                {'id': 'nanali_basic', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 40},
            ],
            'initial_energy': 200,
        }
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertIn(
            'character_protagonist_light_reaction_team_atk',
            {buff['rule_id'] for buff in details['nanali_support']['triggered_buffs']},
        )
        team_atk = next(
            buff for buff in details['nanali_basic']['applied_buffs']
            if buff['rule_id'] == 'character_protagonist_light_reaction_team_atk'
        )
        self.assertEqual(team_atk['effects'], {'atk_pct': 0.12})

        payload['team'][0]['awakening_nodes'] = [1, 2, 3, 4, 6]
        five_node_result = simulate_shaft_axis(payload)['result']
        five_node_details = {detail['step_id']: detail for detail in five_node_result['details']}
        self.assertNotIn(
            'character_protagonist_light_reaction_team_atk',
            {buff['rule_id'] for buff in five_node_details['nanali_support']['triggered_buffs']},
        )
        self.assertNotIn(
            'character_protagonist_light_reaction_team_atk',
            {buff['rule_id'] for buff in five_node_details['nanali_basic']['applied_buffs']},
        )

    def test_extra_awakening_two_buffs_require_six_active_nodes(self) -> None:
        catalog = load_shaft_catalog()
        rules_by_id = {
            rule['id']: rule
            for rule in catalog['buffs']
        }
        expected_rule_ids = {
            'character_protagonist_light_reaction_team_atk',
            'character_xiaozhi_light_res_down',
            'character_adler_e_team_curse',
            'character_fadia_team_hp',
            'character_haiyue_huacai_team_soul',
            'character_canhong_six_node_resonance_attack',
        }

        for rule_id in expected_rule_ids:
            rule = rules_by_id[rule_id]
            conditions = [
                *(rule.get('trigger', {}).get('conditions') or []),
                *(rule.get('target', {}).get('conditions') or []),
            ]
            self.assertIn(
                {'type': 'awakening_count_min', 'min': 6},
                conditions,
                rule_id,
            )

    def test_enemy_resistances_are_element_specific_and_xiaozhi_reduces_only_light(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_4f917797cb', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [1, 2, 3, 4, 5, 6]},
                {'slot': 1, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'enemy': {
                'weakness_elements': [],
                'resistances': {'光': 0.5, '灵': 0.1, '咒': 0.2, '暗': 0.3, '魂': 0.4, '相': 0.6, '心灵': 0.7},
            },
            'steps': [
                {'id': 'xiaozhi_hit', 'slot': 0, 'action_id': 'action_605c800d26', 'start_tick': 0},
                {'id': 'zhenhong_hit', 'slot': 1, 'action_id': 'action_c5f19361cb', 'start_tick': 5},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertAlmostEqual(details['xiaozhi_hit']['formula_parts']['resistance'], 0.5)
        self.assertAlmostEqual(details['zhenhong_hit']['formula_parts']['resistance'], 0.6)
        self.assertAlmostEqual(details['xiaozhi_hit']['formula_parts']['settled_resistance'], 0.5)
        self.assertAlmostEqual(details['zhenhong_hit']['formula_parts']['settled_resistance'], 0.4)
        self.assertGreater(details['xiaozhi_hit']['formula_parts']['settled_defense'], 0)
        light_down = next(
            buff for buff in details['zhenhong_hit']['applied_buffs']
            if buff['rule_id'] == 'character_xiaozhi_light_res_down'
        )
        self.assertEqual(light_down['effects'], {'res_down_光': 0.1})

        inactive_result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_4f917797cb', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [1, 2, 3, 4, 6]},
                {'slot': 1, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
            ],
            'enemy': {
                'weakness_elements': [],
                'resistances': {'光': 0.5, '灵': 0.1, '咒': 0.2, '暗': 0.3, '魂': 0.4, '相': 0.6, '心灵': 0.7},
            },
            'steps': [
                {'id': 'xiaozhi_hit', 'slot': 0, 'action_id': 'action_605c800d26', 'start_tick': 0},
                {'id': 'zhenhong_hit', 'slot': 1, 'action_id': 'action_c5f19361cb', 'start_tick': 5},
            ],
            'initial_energy': 200,
        })['result']
        inactive_zhenhong = next(detail for detail in inactive_result['details'] if detail['step_id'] == 'zhenhong_hit')
        self.assertNotIn(
            'character_xiaozhi_light_res_down',
            {buff['rule_id'] for buff in inactive_zhenhong['applied_buffs']},
        )

    def test_chaos_license_and_state_action_buffs_follow_notes(self) -> None:
        chaos = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_d38b672525', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [2, 5]}],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_2d3f8642dd', 'start_tick': 0},
                {'id': 'basic', 'slot': 0, 'action_id': 'action_40d0576f80', 'start_tick': 10},
            ],
            'initial_energy': 200,
        })['result']
        basic = next(detail for detail in chaos['details'] if detail['step_id'] == 'basic')
        license_buff = next(buff for buff in basic['applied_buffs'] if buff['rule_id'] == 'character_chaos_pursuit_license')
        self.assertEqual(license_buff['effects'], {})
        applied = {buff['rule_id']: buff for buff in basic['applied_buffs']}
        self.assertEqual(applied['character_chaos_a2_license_def_ignore']['effects'], {'def_ignore': 0.2})
        self.assertEqual(applied['character_chaos_a5_license_damage']['effects'], {'all_dmg': 0.2})
        self.assertNotIn('character_chaos_full_license_res_down', applied)

        fadia = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [5]}],
            'steps': [{'id': 'q5a', 'slot': 0, 'action_id': 'action_441d3aa300', 'start_tick': 0}],
            'initial_energy': 200,
        })['result']['details'][0]
        self.assertIn('character_fadia_enemy_slayer_crit', {buff['rule_id'] for buff in fadia['applied_buffs']})

        haiyue = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_699966e2e7', 'arc_id': '', 'cartridge_id': '', 'awakening_nodes': [1, 3]}],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_952b4b561f', 'start_tick': 0},
                {'id': 'e', 'slot': 0, 'action_id': 'action_36a6d1d153', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        haiyue_e = next(detail for detail in haiyue['details'] if detail['step_id'] == 'e')
        self.assertIn('character_haiyue_huacai', {buff['rule_id'] for buff in haiyue_e['applied_buffs']})

    def test_arc_refinement_overrides_new_nanoka_buff_values(self) -> None:
        catalog = load_shaft_catalog()
        team = [{
            'slot': 0,
            'character_id': 'char_b52cc8f160',
            'arc_id': 'arc_f317782734',
            'arc_refinement': 5,
            'cartridge_id': '',
        }]

        rule = next(
            rule
            for rule in registered_buff_rules(team, catalog)
            if rule['id'] == 'arc_reality_shelter_q_attach'
        )

        self.assertEqual(rule['effects'], {'attach_dmg': 0.15})
        self.assertEqual(rule['duration']['ticks'], 60)

    def test_last_rose_e_and_dot_share_one_ten_layer_buff(self) -> None:
        catalog = load_shaft_catalog()
        rules = registered_buff_rules(
            [{
                'slot': 0,
                'character_id': 'char_b52cc8f160',
                'arc_id': 'arc_dcd5900afc',
                'arc_refinement': 5,
                'cartridge_id': '',
            }],
            catalog,
        )
        e_rule = next(rule for rule in rules if rule['id'] == 'arc_last_rose_e_full_stack')
        dot_rule = next(rule for rule in rules if rule['id'] == 'arc_last_rose_crit_stack')
        self.assertEqual(e_rule['name'], '最后一朵玫瑰：暴伤叠层')
        self.assertEqual(e_rule['name'], dot_rule['name'])
        active_buffs = []

        e_instance = activate_buff(active_buffs, e_rule, 0, 10)
        dot_instance = activate_buff(active_buffs, dot_rule, 10)

        self.assertIs(dot_instance, e_instance)
        self.assertEqual(len(active_buffs), 1)
        self.assertEqual(dot_instance['stack_count'], 10)
        self.assertEqual(dot_instance['end_tick'], 40)
        self.assertEqual(dot_instance['rule']['id'], 'arc_last_rose_crit_stack')
        self.assertEqual(dot_instance['rule']['effects'], {'crit_dmg': 0.12})

    def test_manual_dot_catalog_excludes_automatically_calculated_requiem_nightmare(self) -> None:
        catalog = load_shaft_catalog()
        dot_actions = {
            (action['character_name'], action['name'])
            for action in catalog['actions']
            if 'DOT' in set(action.get('tags') or []) | {str(action.get('extra_tag') or '')}
        }

        self.assertEqual(dot_actions, {('白藏', 'q的dot'), ('阿德勒', 'dot10跳')})
        self.assertNotIn(
            ('安魂曲', 'dot'),
            {(action['character_name'], action['name']) for action in catalog['actions']},
        )

    def test_last_rose_shared_stack_runs_in_browser_engine(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': 'arc_dcd5900afc',
                'arc_refinement': 5,
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_2745f804a5', 'start_tick': 0},
                {'id': 'basic', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 11},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        e_buff = next(buff for buff in details['e']['triggered_buffs'] if buff['rule_id'] == 'arc_last_rose_e_full_stack')
        nightmare_tick = next(
            event
            for event in result['periodic_damage_events']
            if event['reaction'] == '噩梦' and event['tick'] == 10 and event.get('triggered_buffs')
        )
        dot_buff = next(
            buff
            for buff in nightmare_tick['triggered_buffs']
            if buff['rule_id'] == 'arc_last_rose_crit_stack'
        )
        rose_buffs = [
            buff
            for buff in details['basic']['applied_buffs']
            if buff['rule_id'] in {'arc_last_rose_e_full_stack', 'arc_last_rose_crit_stack'}
        ]
        self.assertEqual(e_buff['stack_count'], 10)
        self.assertEqual(e_buff['definition_id'], 'arc_last_rose_dark_thorn')
        self.assertEqual(e_buff['name'], '最后一朵玫瑰：暴伤叠层')
        self.assertEqual(dot_buff['stack_count'], 10)
        self.assertEqual(dot_buff['definition_id'], e_buff['definition_id'])
        self.assertEqual(dot_buff['name'], e_buff['name'])
        self.assertEqual(dot_buff['effects'], {'crit_dmg': 1.2})
        self.assertEqual(len(rose_buffs), 1)

    def test_owner_settled_corrosion_refreshes_last_rose_with_fixed_dot_crit_rate(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_1895e259be',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': 'arc_dcd5900afc',
                    'arc_refinement': 5,
                    'cartridge_id': '',
                },
            ],
            'steps': [
                *[
                    {
                        'id': f'gain_{index}',
                        'slot': 0,
                        'action_id': 'action_4c5142a40f',
                        'start_tick': index * 2,
                    }
                    for index in range(4)
                ],
                {
                    'id': 'support',
                    'slot': 1,
                    'action_id': 'action_2635f721a8',
                    'start_tick': 40,
                },
                {
                    'id': 'extend_axis',
                    'slot': 0,
                    'action_id': 'action_4c5142a40f',
                    'start_tick': 100,
                },
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        corrosion_events = [
            event
            for event in result['reaction_damage_events']
            if event['reaction'] == '浊燃' and event['damage'] > 0
        ]

        self.assertGreaterEqual(len(corrosion_events), 2)
        first, second = corrosion_events[:2]
        self.assertEqual(first['contributor_slot'], 1)
        self.assertEqual(first['extra_tag'], 'DOT')
        self.assertEqual(first['tags'], ['DOT'])
        self.assertEqual(first['formula_parts']['crit_rate'], 0.5)
        self.assertAlmostEqual(first['formula_parts']['critical'], 1.25)
        self.assertEqual(second['formula_parts']['crit_rate'], 0.5)
        self.assertAlmostEqual(second['formula_parts']['critical'], 1.31)

        first_rose = next(
            buff for buff in first['triggered_buffs']
            if buff['rule_id'] == 'arc_last_rose_crit_stack'
        )
        second_rose = next(
            buff for buff in second['triggered_buffs']
            if buff['rule_id'] == 'arc_last_rose_crit_stack'
        )
        self.assertEqual(first_rose['stack_count'], 1)
        self.assertEqual(first_rose['end_tick'], first['tick'] + 30)
        self.assertEqual(second_rose['stack_count'], 2)
        self.assertEqual(second_rose['end_tick'], second['tick'] + 30)

    def test_stagger_frequency_uses_team_stagger_axis_duration_and_recovery(self) -> None:
        catalog = load_shaft_catalog()
        character = next(
            character
            for character in catalog['characters']
            if character['adaptation'] == '液态' and any(
                action['character_id'] == character['id'] and float(action.get('stagger') or 0) > 0
                for action in catalog['actions']
            )
        )
        action = next(
            action
            for action in catalog['actions']
            if action['character_id'] == character['id'] and float(action.get('stagger') or 0) > 0
        )

        def simulate(arc_id: str):
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': character['id'],
                    'arc_id': arc_id,
                    'cartridge_id': '',
                }],
                'steps': [
                    {'id': 'first', 'slot': 0, 'action_id': action['id'], 'start_tick': 0},
                    {'id': 'second', 'slot': 0, 'action_id': action['id'], 'start_tick': 100},
                ],
                'initial_energy': 200,
            })['result']

        normal = simulate('')
        rose = simulate('arc_dcd5900afc')
        summary = normal['summary']
        expected_frequency = 1 / (
            50 / summary['total_stagger']
            + 10 / summary['duration_seconds']
        )

        self.assertAlmostEqual(summary['stagger_frequency'], expected_frequency)
        self.assertEqual(summary['stagger_recovery_seconds'], 10)
        self.assertEqual(rose['summary']['stagger_recovery_seconds'], 13)
        self.assertLess(rose['summary']['stagger_frequency'], summary['stagger_frequency'])

    def test_stagger_damage_has_four_character_contributions_without_entering_character_share(self) -> None:
        catalog = load_shaft_catalog()
        arc_character = next(
            character
            for character in catalog['characters']
            if character['adaptation'] == '液态' and any(
                action['character_id'] == character['id'] and float(action.get('stagger') or 0) > 0
                for action in catalog['actions']
            )
        )
        selected = []
        for character in [arc_character, *catalog['characters']]:
            if any(existing[0]['id'] == character['id'] for existing in selected):
                continue
            action = next((
                action
                for action in catalog['actions']
                if action['character_id'] == character['id'] and float(action.get('stagger') or 0) > 0
            ), None)
            if action:
                selected.append((character, action))
            if len(selected) == 4:
                break
        self.assertEqual(len(selected), 4)
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': slot,
                    'character_id': character['id'],
                    'arc_id': 'arc_60702143b6' if slot == 0 else '',
                    'cartridge_id': '',
                }
                for slot, (character, _action) in enumerate(selected)
            ],
            'steps': [
                {
                    'id': f'step_{slot}',
                    'slot': slot,
                    'action_id': action['id'],
                    'start_tick': slot * 20,
                }
                for slot, (_character, action) in enumerate(selected)
            ],
            'initial_energy': 200,
        })['result']

        contributions = result['stagger_contributions_by_slot']
        self.assertEqual(len(contributions), 4)
        self.assertAlmostEqual(
            sum(item['damage'] for item in contributions),
            result['summary']['stagger_damage'],
        )
        self.assertEqual(contributions[0]['average_stagger_strength'], 60)
        self.assertGreater(contributions[0]['damage_per_trigger'], contributions[1]['damage_per_trigger'])
        self.assertAlmostEqual(
            sum(item['damage'] for item in result['damage_by_slot']),
            result['summary']['character_damage'],
        )
        self.assertAlmostEqual(
            sum(item['percent'] for item in result['damage_by_slot']),
            100,
        )
        for contribution in result['damage_by_action_by_slot']:
            self.assertTrue(all(
                action['action_type'] not in {'倾陷', '环合'}
                for action in contribution['actions']
            ))
            self.assertAlmostEqual(
                sum(action['damage'] for action in contribution['actions']),
                contribution['total_damage'],
            )
        self.assertTrue(all(item['damage'] > 0 for item in contributions))

    def test_stagger_damage_bonus_adds_strength_daphne_and_canhong_awakening(self) -> None:
        def simulate(canhong_awakening_nodes: list[int]) -> dict:
            result = simulate_shaft_axis({
                'team': [
                    {
                        'slot': 0,
                        'character_id': 'char_e8ad982185',
                        'arc_id': '',
                        'cartridge_id': '',
                    },
                    {
                        'slot': 1,
                        'character_id': 'char_076a1f4e53',
                        'awakening_nodes': canhong_awakening_nodes,
                        'arc_id': '',
                        'cartridge_id': '',
                    },
                ],
                'steps': [{
                    'id': 'daphne-support',
                    'slot': 0,
                    'action_id': 'action_5d4664d859',
                    'start_tick': 0,
                }],
                'initial_energy': 200,
            })['result']
            return next(
                item for item in result['stagger_contributions_by_slot']
                if item['character_id'] == 'char_e8ad982185'
            )

        daphne_only = simulate([])
        with_canhong = simulate([5])

        self.assertEqual(daphne_only['average_stagger_strength'], 0)
        strength_bonus = daphne_only['average_stagger_strength'] / 300
        self.assertAlmostEqual(daphne_only['average_stagger_damage_bonus'], strength_bonus + 0.2)
        self.assertAlmostEqual(with_canhong['average_stagger_damage_bonus'], strength_bonus + 0.2 + 3)
        self.assertAlmostEqual(
            with_canhong['damage_per_trigger'] / daphne_only['damage_per_trigger'],
            (1 + strength_bonus + 0.2 + 3) / (1 + strength_bonus + 0.2),
        )

    def test_independent_buff_drops_oldest_layer_at_cap(self) -> None:
        catalog = load_shaft_catalog()
        rule = next(
            rule
            for rule in registered_buff_rules(
                [{
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_4d19ef5194',
                    'arc_refinement': 1,
                    'cartridge_id': '',
                }],
                catalog,
            )
            if rule['id'] == 'arc_ora_basic_stack'
        )
        active_buffs = []

        for tick in range(11):
            activate_buff(active_buffs, rule, tick)

        self.assertEqual(len(active_buffs), 10)
        self.assertEqual([instance['start_tick'] for instance in active_buffs], list(range(1, 11)))
        self.assertTrue(all(instance['end_tick'] - instance['start_tick'] == 100 for instance in active_buffs))

    def test_independent_buff_creates_one_timer_per_layer_and_evicts_earliest_expiry(self) -> None:
        base_rule = {
            'id': 'independent_test',
            'name': '独立层测试',
            'owner_slot': 0,
            'duration': {'type': 'time', 'ticks': 100},
            'stacking': {'key': 'independent_test_stack', 'mode': 'independent', 'max_stacks': 2},
            'effects': {'atk_pct': 0.1},
        }
        active_buffs = []

        activate_buff(active_buffs, base_rule, 0)
        short_rule = deepcopy(base_rule)
        short_rule['duration']['ticks'] = 10
        activate_buff(active_buffs, short_rule, 1)
        replacement_rule = deepcopy(base_rule)
        replacement_rule['duration']['ticks'] = 50
        activate_buff(active_buffs, replacement_rule, 2)

        self.assertEqual([(buff['start_tick'], buff['end_tick']) for buff in active_buffs], [(0, 100), (2, 52)])

        active_buffs.clear()
        latest = activate_buff(active_buffs, base_rule, 5, 10)
        self.assertEqual(len(active_buffs), 2)
        self.assertTrue(all(buff['stack_count'] == 1 for buff in active_buffs))
        self.assertIs(latest, active_buffs[-1])

    def test_forest_firefly_independent_layers_expose_stacking_metadata_for_buff_line(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b52cc8f160', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': 'cartridge_29793225a0'},
            ],
            'steps': [
                {'id': 'nanali_1', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 0},
                {'id': 'nanali_2', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 10},
                {'id': 'inspect', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        inspect = next(detail for detail in result['details'] if detail['step_id'] == 'inspect')
        layers = [
            buff
            for buff in inspect['applied_buffs']
            if buff['rule_id'] == 'cartridge_forest_firefly_crit_stack'
        ]

        self.assertEqual(len(layers), 2)
        self.assertTrue(all(layer['stack_count'] == 1 for layer in layers))
        self.assertTrue(all(layer['stacking_mode'] == 'independent' for layer in layers))
        self.assertTrue(all(layer['max_stacks'] == 7 for layer in layers))

    def test_cross_element_cartridge_suppresses_two_piece_element_damage_but_keeps_four_piece_trigger(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_29793225a0',
                },
                {
                    'slot': 1,
                    'character_id': 'char_bdc43f82c6',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'spirit_hit', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 0},
                {'id': 'wearer_hit', 'slot': 0, 'action_id': 'action_c5f19361cb', 'start_tick': 10},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        wearer_hit = next(detail for detail in result['details'] if detail['step_id'] == 'wearer_hit')
        firefly_buffs = [
            buff
            for buff in wearer_hit['applied_buffs']
            if buff['rule_id'] == 'cartridge_forest_firefly_crit_stack'
        ]

        self.assertAlmostEqual(wearer_hit['panel']['element_dmg'], 0.0)
        self.assertEqual(len(firefly_buffs), 1)
        self.assertAlmostEqual(firefly_buffs[0]['effects']['crit_dmg'], 0.08)

    def test_ready_tally_triggers_ten_second_boss_damage_once_when_second_stack_is_reached(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_699966e2e7',
                'arc_id': 'arc_ce53905d70',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first_e', 'slot': 0, 'action_id': 'action_36a6d1d153', 'start_tick': 0},
                {'id': 'full_q', 'slot': 0, 'action_id': 'action_952b4b561f', 'start_tick': 20},
                {'id': 'third_e', 'slot': 0, 'action_id': 'action_36a6d1d153', 'start_tick': 40},
                {'id': 'inside', 'slot': 0, 'action_id': 'action_ec98ddd32d', 'start_tick': 60},
                {'id': 'expired', 'slot': 0, 'action_id': 'action_ec98ddd32d', 'start_tick': 130},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        full_q_triggered = {
            buff['rule_id']: buff
            for buff in details['full_q']['triggered_buffs']
        }
        third_e_triggered = {
            buff['rule_id']: buff
            for buff in details['third_e']['triggered_buffs']
        }
        inside_applied = {
            buff['rule_id']: buff
            for buff in details['inside']['applied_buffs']
        }
        expired_applied = {
            buff['rule_id']: buff
            for buff in details['expired']['applied_buffs']
        }

        self.assertEqual(
            full_q_triggered['arc_ready_eq_basic_dodge']['stack_count'],
            2,
        )
        self.assertEqual(
            full_q_triggered['arc_ready_tally_boss_damage']['duration_ticks'],
            100,
        )
        self.assertEqual(
            full_q_triggered['arc_ready_tally_boss_damage']['effects'],
            {'all_dmg': 0.1},
        )
        self.assertNotIn('arc_ready_tally_boss_damage', third_e_triggered)
        self.assertEqual(
            inside_applied['arc_ready_eq_basic_dodge']['effects'],
            {'basic_dmg': 0.3, 'dodge_counter_dmg': 0.3},
        )
        self.assertEqual(
            inside_applied['arc_ready_tally_boss_damage']['effects'],
            {'all_dmg': 0.1},
        )
        self.assertNotIn('arc_ready_tally_boss_damage', expired_applied)

    def test_little_adventure_q_creates_ten_independent_hp_layers(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_caa6c2e5a8',
                'arc_id': '',
                'cartridge_id': 'cartridge_868dbc2c5c',
            }],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_122b6bf962', 'start_tick': 0},
                {'id': 'e', 'slot': 0, 'action_id': 'action_72051426d5', 'start_tick': 1},
            ],
            'initial_energy': 200,
        })['result']
        detail = next(item for item in result['details'] if item['step_id'] == 'e')
        layers = [
            buff
            for buff in detail['applied_buffs']
            if buff['definition_id'] == 'cartridge_little_adventure_hp'
        ]

        self.assertEqual(len(layers), 10)
        self.assertTrue(all(layer['stack_count'] == 1 for layer in layers))
        self.assertTrue(all(layer['effects'] == {'hp_pct': 0.04} for layer in layers))

    def test_nanali_authority_delay_preserves_pre_start_switch_but_resets_after_start(self) -> None:
        team = [
            {'slot': 0, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            {'slot': 1, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
        ]
        catalog = load_shaft_catalog()
        nanali_noop = next(
            action for action in catalog['actions_by_character']['char_bdc43f82c6']
            if action['name'] == '无'
        )
        teammate_noop = next(
            action for action in catalog['actions_by_character']['char_caa6c2e5a8']
            if action['name'] == '无'
        )
        before_start = simulate_shaft_axis({
            'team': team,
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'switch', 'slot': 1, 'action_id': 'action_122b6bf962', 'start_tick': 2},
                {'id': 'pursuit', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 13},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        before_details = {item['step_id']: item for item in before_start['details']}
        before_pursuit = next(item for item in before_start['details'] if item['step_id'] == 'pursuit')
        self.assertFalse(any('一代目的权柄' in warning for warning in before_pursuit['warnings']))
        before_authority = next(
            buff for buff in before_details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_nanali_authority'
        )
        self.assertEqual(before_authority['end_tick'], 124)

        starts_in_background = simulate_shaft_axis({
            'team': team,
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'leave_before_start', 'slot': 1, 'action_id': teammate_noop['id'], 'start_tick': 2},
                {'id': 'stay_in_background', 'slot': 1, 'action_id': teammate_noop['id'], 'start_tick': 6},
                {'id': 'pursuit', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 10},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        background_details = {item['step_id']: item for item in starts_in_background['details']}
        self.assertFalse(any(
            '一代目的权柄' in warning
            for warning in background_details['pursuit']['warnings']
        ))
        background_authority = next(
            buff for buff in background_details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_nanali_authority'
        )
        self.assertEqual(background_authority['end_tick'], 124)

        after_start = simulate_shaft_axis({
            'team': team,
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'switch', 'slot': 1, 'action_id': 'action_122b6bf962', 'start_tick': 5},
                {'id': 'pursuit', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 10},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        after_details = {item['step_id']: item for item in after_start['details']}
        after_pursuit = after_details['pursuit']
        self.assertTrue(any('一代目的权柄' in warning for warning in after_pursuit['warnings']))
        after_authority = next(
            buff for buff in after_details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_nanali_authority'
        )
        self.assertEqual(after_authority['end_tick'], after_details['switch']['start_tick'])
        self.assertEqual(after_authority['visual_end_tick'], after_details['switch']['visual_start_tick'])

        returns_then_leaves = simulate_shaft_axis({
            'team': team,
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_5870d8ba67', 'start_tick': 0},
                {'id': 'leave_before_start', 'slot': 1, 'action_id': teammate_noop['id'], 'start_tick': 2},
                {'id': 'return_front', 'slot': 0, 'action_id': nanali_noop['id'], 'start_tick': 6},
                {'id': 'leave_after_return', 'slot': 1, 'action_id': teammate_noop['id'], 'start_tick': 8},
                {'id': 'pursuit', 'slot': 0, 'action_id': 'action_8510ca415f', 'start_tick': 10},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        transition_details = {item['step_id']: item for item in returns_then_leaves['details']}
        transition_authority = next(
            buff for buff in transition_details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_nanali_authority'
        )
        self.assertTrue(any(
            '一代目的权柄' in warning
            for warning in transition_details['pursuit']['warnings']
        ))
        self.assertEqual(
            transition_authority['end_tick'],
            transition_details['leave_after_return']['start_tick'],
        )
        self.assertEqual(
            transition_authority['visual_end_tick'],
            transition_details['leave_after_return']['visual_start_tick'],
        )

    def test_haniya_e_and_q_grant_independent_team_base_atk_buffs(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_e0a4292b4e',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [2],
                },
                {'slot': 1, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_db30565b20', 'start_tick': 0},
                {'id': 'e_only', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 20},
                {'id': 'q', 'slot': 0, 'action_id': 'action_3529c3d915', 'start_tick': 30},
                {'id': 'stacked', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 40},
                {'id': 'after_q', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 150},
                {'id': 'expired', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 250},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        details = {item['step_id']: item for item in result['details']}
        expected_flat_atk = 493.0 * 0.2

        e_trigger = next(
            buff for buff in details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_haniya_e_team_flat_atk'
        )
        q_trigger = next(
            buff for buff in details['q']['triggered_buffs']
            if buff['rule_id'] == 'character_haniya_q_team_flat_atk'
        )
        self.assertEqual(e_trigger['duration_ticks'], 120)
        self.assertEqual(q_trigger['duration_ticks'], 100)
        self.assertAlmostEqual(e_trigger['effects']['flat_atk'], expected_flat_atk)
        self.assertAlmostEqual(q_trigger['effects']['flat_atk'], expected_flat_atk)

        e_only = {buff['rule_id'] for buff in details['e_only']['applied_buffs']}
        stacked = {buff['rule_id'] for buff in details['stacked']['applied_buffs']}
        after_q = {buff['rule_id'] for buff in details['after_q']['applied_buffs']}
        expired = {buff['rule_id'] for buff in details['expired']['applied_buffs']}
        self.assertIn('character_haniya_e_team_flat_atk', e_only)
        self.assertNotIn('character_haniya_q_team_flat_atk', e_only)
        self.assertTrue({
            'character_haniya_e_team_flat_atk',
            'character_haniya_q_team_flat_atk',
        }.issubset(stacked))
        self.assertIn('character_haniya_e_team_flat_atk', after_q)
        self.assertNotIn('character_haniya_q_team_flat_atk', after_q)
        self.assertFalse({
            'character_haniya_e_team_flat_atk',
            'character_haniya_q_team_flat_atk',
        } & expired)

    def test_haniya_ace_pauses_guguzhi_duration_in_both_cast_orders(self) -> None:
        def simulate(steps):
            return simulate_shaft_axis({
                'team': [
                    {'slot': 0, 'character_id': 'char_e0a4292b4e', 'arc_id': '', 'cartridge_id': ''},
                    {'slot': 1, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
                ],
                'steps': steps,
                'options': {'switch_loss_ticks': 0},
                'initial_energy': 200,
            })['result']

        e_before_q = simulate([
            {'id': 'e', 'slot': 0, 'action_id': 'action_db30565b20', 'start_tick': 0},
            {'id': 'q', 'slot': 0, 'action_id': 'action_3529c3d915', 'start_tick': 40},
            {'id': 'during_pause', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 120},
            {'id': 'after_resume', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 170},
            {'id': 'expired', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 200},
        ])
        before_details = {item['step_id']: item for item in e_before_q['details']}
        for step_id in ('during_pause', 'after_resume'):
            self.assertIn('character_haniya_e_team_flat_atk', {
                buff['rule_id'] for buff in before_details[step_id]['applied_buffs']
            })
        self.assertNotIn('character_haniya_e_team_flat_atk', {
            buff['rule_id'] for buff in before_details['expired']['applied_buffs']
        })

        e_during_q = simulate([
            {'id': 'q', 'slot': 0, 'action_id': 'action_3529c3d915', 'start_tick': 0},
            {'id': 'e', 'slot': 0, 'action_id': 'action_db30565b20', 'start_tick': 20},
            {'id': 'after_resume', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 170},
            {'id': 'expired', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 200},
        ])
        during_details = {item['step_id']: item for item in e_during_q['details']}
        e_trigger = next(
            buff for buff in during_details['e']['triggered_buffs']
            if buff['rule_id'] == 'character_haniya_e_team_flat_atk'
        )
        self.assertEqual(e_trigger['end_tick'], 180)
        self.assertIn('character_haniya_e_team_flat_atk', {
            buff['rule_id'] for buff in during_details['after_resume']['applied_buffs']
        })
        self.assertNotIn('character_haniya_e_team_flat_atk', {
            buff['rule_id'] for buff in during_details['expired']['applied_buffs']
        })

    def test_haiyue_haniya_and_hathor_new_character_buffs(self) -> None:
        haiyue = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_699966e2e7', 'arc_id': '', 'cartridge_id': '', 'awakening': 6}],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_36a6d1d153', 'start_tick': 0},
                {'id': 'basic', 'slot': 0, 'action_id': 'action_18cf1ad6b6', 'start_tick': 10},
            ],
            'initial_energy': 200,
        })['result']
        haiyue_basic = next(item for item in haiyue['details'] if item['step_id'] == 'basic')
        jianqiang = next(buff for buff in haiyue_basic['applied_buffs'] if buff['definition_id'] == 'character_haiyue_jianqiang')
        self.assertEqual(jianqiang['stack_count'], 10)
        self.assertEqual(jianqiang['effects'], {'atk_pct': 0.1})

        haniya = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_e0a4292b4e', 'arc_id': '', 'cartridge_id': '', 'awakening': 6},
                {'slot': 1, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_3529c3d915', 'start_tick': 0},
                {'id': 'ensemble', 'slot': 0, 'action_id': 'action_8cadfa76a4', 'start_tick': 1},
                {'id': 'first', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 2},
                {'id': 'second', 'slot': 1, 'action_id': 'action_72051426d5', 'start_tick': 3},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        haniya_details = {item['step_id']: item for item in haniya['details']}
        self.assertIn('character_haniya_ace_team_crit_dmg', {
            buff['rule_id'] for buff in haniya_details['first']['applied_buffs']
        })
        self.assertIn('character_haniya_ace_ensemble_next_attack', {
            buff['rule_id'] for buff in haniya_details['first']['applied_buffs']
        })
        self.assertNotIn('character_haniya_ace_ensemble_next_attack', {
            buff['rule_id'] for buff in haniya_details['second']['applied_buffs']
        })

        hathor = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_912dbfe17c', 'arc_id': '', 'cartridge_id': '', 'awakening': 6}],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_cc530277d3', 'start_tick': 0},
                {'id': 'e1', 'slot': 0, 'action_id': 'action_0dfd7dfab8', 'start_tick': 1},
                {'id': 'e2', 'slot': 0, 'action_id': 'action_85ee761950', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        hathor_details = {item['step_id']: item for item in hathor['details']}
        self.assertIn('character_hathor_emergency_delivery', {
            buff['rule_id'] for buff in hathor_details['e1']['applied_buffs']
        })
        self.assertNotIn('character_hathor_next_e_crit', {
            buff['rule_id'] for buff in hathor_details['e1']['applied_buffs']
        })
        self.assertIn('character_hathor_next_e_crit', {
            buff['rule_id'] for buff in hathor_details['e2']['applied_buffs']
        })

    def test_adler_karma_consumption_and_expected_murk_debuff(self) -> None:
        karma = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_6f46705fd1',
                'arc_id': '',
                'cartridge_id': '',
                'awakening': 2,
            }],
            'steps': [
                {'id': 'karma_1', 'slot': 0, 'action_id': 'action_adler_karma_2', 'start_tick': 0},
                {'id': 'karma_2', 'slot': 0, 'action_id': 'action_adler_karma_2', 'start_tick': 1},
                {'id': 'karma_3', 'slot': 0, 'action_id': 'action_adler_karma_2', 'start_tick': 2},
                {'id': 'e', 'slot': 0, 'action_id': 'action_25fea9f7cb', 'start_tick': 3},
                {'id': 'after', 'slot': 0, 'action_id': 'action_95733904eb', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        karma_details = {item['step_id']: item for item in karma['details']}
        consumed = next(
            buff for buff in karma_details['e']['applied_buffs']
            if buff['rule_id'] == 'character_adler_a2_e_consume_karma'
        )
        self.assertEqual(consumed['effects'], {'final_dmg': 0.18})
        self.assertAlmostEqual(karma_details['e']['panel']['final_dmg'], 0.18)
        self.assertAlmostEqual(karma_details['e']['formula_parts']['final_multiplier'], 1.18)
        self.assertNotIn('character_adler_karma', {
            buff['definition_id'] for buff in karma_details['after']['applied_buffs']
        })

        murk = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_6f46705fd1', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                *[
                    {'id': f'gain_{index}', 'slot': 0, 'action_id': 'action_6d2645f71e', 'start_tick': index}
                    for index in range(15)
                ],
                {'id': 'support', 'slot': 1, 'action_id': 'action_55c8843d1c', 'start_tick': 20},
                {'id': 'after_murk', 'slot': 1, 'action_id': 'action_25fea9f7cb', 'start_tick': 33},
            ],
            'initial_energy': 200,
        })['result']
        after_murk = next(item for item in murk['details'] if item['step_id'] == 'after_murk')
        expected = next(
            buff for buff in after_murk['applied_buffs']
            if buff['rule_id'] == 'character_adler_murk_expected_debuff'
        )
        self.assertAlmostEqual(expected['effects']['res_down'], 1 / 30)
        self.assertAlmostEqual(expected['effects']['stagger_multiplier'], 1 / 30)

    def test_requiem_demons_gift_applies_on_cast_and_refreshes_without_stacking(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'black_book', 'slot': 0, 'action_id': 'action_7af75245df', 'start_tick': 0},
                {'id': 'before_refresh', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 40},
                {'id': 'tomato', 'slot': 0, 'action_id': 'action_0b958faf88', 'start_tick': 100},
                {'id': 'after_refresh', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 200},
                {'id': 'expired', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 260},
            ],
            'initial_energy': 200,
        })['result']
        details = {item['step_id']: item for item in result['details']}
        rule_id = 'character_requiem_demons_gift_atk_steal'

        self.assertIn(rule_id, {
            buff['rule_id'] for buff in details['black_book']['applied_buffs']
        })
        self.assertIn(rule_id, {
            buff['rule_id'] for buff in details['before_refresh']['applied_buffs']
        })
        refreshed = [
            buff for buff in details['after_refresh']['applied_buffs']
            if buff['rule_id'] == rule_id
        ]
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0]['stack_count'], 1)
        self.assertEqual(refreshed[0]['effects'], {'flat_atk': 200})
        self.assertNotIn(rule_id, {
            buff['rule_id'] for buff in details['expired']['applied_buffs']
        })
        self.assertEqual(
            details['after_refresh']['panel']['atk'] - details['expired']['panel']['atk'],
            200,
        )

    def test_requiem_nightmare_has_independent_three_second_layers_and_settlement(self) -> None:
        catalog = load_shaft_catalog()
        requiem_q = next(
            action
            for action in catalog['actions']
            if action['character_id'] == 'char_c78f7a08d5' and action['name'] == 'q'
        )
        self.assertEqual(requiem_q['source_formula'], '{0}%*4+{1}%')
        self.assertEqual(requiem_q['hit_count'], 5)
        self.assertEqual(requiem_q['nightmare_stacks'], 5)
        self.assertEqual(requiem_q['personal_resource_gain'], {'噩梦': 5})

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
                'awakening': 2,
            }],
            'steps': [
                {'id': 'first', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'second', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 15},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 16},
            ],
            'initial_energy': 200,
        })['result']
        inspect = next(item for item in result['details'] if item['step_id'] == 'inspect')
        layers = [
            buff for buff in inspect['applied_buffs']
            if buff['definition_id'] == 'character_requiem_nightmare'
        ]
        self.assertEqual(len(layers), 4)
        self.assertTrue(all(layer['stack_count'] == 1 for layer in layers))
        self.assertEqual(
            sorted(event['tick'] for event in result['periodic_damage_events']),
            [10, 20, 30, 40],
        )
        self.assertEqual(
            [event['stack_count'] for event in result['periodic_damage_events']],
            [2, 4, 4, 2],
        )
        self.assertEqual(
            [event['formula_parts']['periodic_scale'] for event in result['periodic_damage_events']],
            [2, 4, 4, 2],
        )
        first_periodic = result['periodic_damage_events'][0]
        self.assertAlmostEqual(first_periodic['atk_multiplier'], 0.06 * 1.33)
        self.assertEqual(first_periodic['formula_parts']['crit_rate'], 0.5)
        self.assertEqual(first_periodic['formula_parts']['skill_category'], 'basic')
        self.assertEqual(first_periodic['formula_parts']['skill_level'], 10)
        self.assertAlmostEqual(
            first_periodic['formula_parts']['skill_multiplier'],
            1.08 ** 9,
        )
        self.assertNotIn('噩梦', {
            item['source']
            for item in result['damage_by_source']
        })
        contribution = result['damage_by_action_by_slot'][0]
        nightmare_contribution = next(
            action
            for action in contribution['actions']
            if action['action_name'] == '噩梦'
        )
        self.assertEqual(nightmare_contribution['action_type'], '普攻')
        self.assertEqual(nightmare_contribution['damage_type'], '普攻')
        self.assertGreater(nightmare_contribution['damage'], 0)
        self.assertAlmostEqual(
            sum(action['damage'] for action in contribution['actions']),
            contribution['total_damage'],
        )

        q_result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'requiem_q',
                'slot': 0,
                'action_id': 'action_1104de864b',
                'start_tick': 0,
            }],
            'initial_energy': 200,
        })['result']
        q_detail = q_result['details'][0]
        self.assertEqual(q_detail['hit_count'], 5)
        self.assertEqual(q_detail['nightmare_stacks'], 5)
        q_nightmare = next(
            buff
            for buff in q_detail['triggered_buffs']
            if buff['definition_id'] == 'character_requiem_nightmare'
        )
        self.assertEqual(q_nightmare['stacking_mode'], 'independent')
        self.assertEqual(q_nightmare['stack_count'], 5)

    def test_requiem_awakening_nodes_adjust_nightmare_without_f_damage_bonus(self) -> None:
        catalog = load_shaft_catalog()
        self.assertEqual(
            catalog['awakenings']['安魂曲'][0]['description'],
            '「噩梦」的伤害倍率提升33%。',
        )

        def simulate(awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [
                    {'id': 'gain', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                    {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 70},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        base = simulate([])
        awakening_a = simulate([1])
        awakening_d = simulate([4])
        awakening_e = simulate([5])
        awakening_f = simulate([6])

        base_tick = base['periodic_damage_events'][0]
        self.assertAlmostEqual(
            awakening_a['periodic_damage_events'][0]['damage'],
            base_tick['damage'] * 1.33,
        )
        self.assertEqual(
            [event['tick'] for event in awakening_d['periodic_damage_events']],
            [10, 20, 30, 40, 50, 60],
        )
        awakening_d_nightmare = next(
            buff
            for buff in awakening_d['details'][0]['triggered_buffs']
            if buff['definition_id'] == 'character_requiem_nightmare'
        )
        self.assertEqual(awakening_d_nightmare['duration_ticks'], 60)
        self.assertAlmostEqual(
            awakening_e['periodic_damage_events'][0]['damage'],
            base_tick['damage'] * 2.05,
        )
        self.assertAlmostEqual(
            awakening_f['periodic_damage_events'][0]['damage'],
            base_tick['damage'],
        )

    def test_requiem_special_actions_do_not_add_nightmare_stacks(self) -> None:
        catalog = load_shaft_catalog()
        excluded_action_ids = {
            'action_2635f721a8',
            'action_b9b3237c74',
            'action_7af75245df',
            'action_0b958faf88',
        }
        actions = {
            action['id']: action
            for action in catalog['actions']
            if action['id'] in excluded_action_ids
        }
        self.assertEqual(set(actions), excluded_action_ids)
        self.assertEqual(
            {action['name']: action['hit_count'] for action in actions.values()},
            {'援护': 1, '失谐强化': 1, '黑之书': 2, '番茄酱恶魔': 32},
        )
        self.assertTrue(all(action['nightmare_stacks'] == 0 for action in actions.values()))
        self.assertTrue(all(action['personal_resource_gain'] == {} for action in actions.values()))

        rules = registered_buff_rules(
            [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            catalog,
        )
        nightmare_rule = next(
            rule for rule in rules
            if rule['id'] == 'character_requiem_nightmare_stack'
        )
        self.assertTrue(
            excluded_action_ids.isdisjoint(nightmare_rule['trigger']['source']['action_ids'])
        )
        for action_id in excluded_action_ids:
            result = simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': '',
                }],
                'steps': [{
                    'id': action_id,
                    'slot': 0,
                    'action_id': action_id,
                    'start_tick': 0,
                }],
                'initial_energy': 200,
            })['result']
            detail = next(
                item for item in result['details']
                if item['action_id'] == action_id
            )
            self.assertNotIn(
                'character_requiem_nightmare',
                {
                    buff['definition_id']
                    for buff in detail['triggered_buffs']
                },
                actions[action_id]['name'],
            )
            self.assertEqual(result['periodic_damage_events'], [], actions[action_id]['name'])

        two_layers = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'gain_two', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 40},
            ],
            'initial_energy': 200,
        })['result']
        five_layers = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'gain_five', 'slot': 0, 'action_id': 'action_1104de864b', 'start_tick': 0},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 40},
            ],
            'initial_energy': 200,
        })['result']
        self.assertEqual(
            [event['tick'] for event in two_layers['periodic_damage_events']],
            [10, 20, 30],
        )
        self.assertEqual(
            [event['tick'] for event in five_layers['periodic_damage_events']],
            [10, 20, 30],
        )
        self.assertAlmostEqual(
            five_layers['periodic_damage_events'][0]['damage'],
            two_layers['periodic_damage_events'][0]['damage'] * 2.5,
        )

        restarted = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first_window', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'second_window', 'slot': 0, 'action_id': 'action_00edea34a8', 'start_tick': 40},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 80},
            ],
            'initial_energy': 200,
        })['result']
        self.assertEqual(
            [event['tick'] for event in restarted['periodic_damage_events']],
            [10, 20, 30, 50, 60, 70],
        )

        settlement = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_c78f7a08d5',
                'arc_id': '',
                'cartridge_id': '',
                'awakening': 3,
            }],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_2745f804a5', 'start_tick': 0},
                {'id': 'settle', 'slot': 0, 'action_id': 'action_a7443f657f', 'start_tick': 5},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 6},
            ],
            'initial_energy': 200,
        })['result']
        settlement_events = [
            event for event in settlement['periodic_damage_events']
            if event['kind'] == 'buff_periodic_settlement'
        ]
        self.assertEqual(len(settlement_events), 1)
        self.assertEqual(settlement_events[0]['stack_count'], 9)
        self.assertAlmostEqual(settlement_events[0]['remaining_seconds'], 24.5)
        self.assertAlmostEqual(settlement_events[0]['formula_parts']['periodic_scale'], 24.5)
        inspect_after = next(item for item in settlement['details'] if item['step_id'] == 'inspect')
        self.assertNotIn('character_requiem_nightmare', {
            buff['definition_id'] for buff in inspect_after['applied_buffs']
        })

    def test_fadia_darkstar_background_action_adds_capped_team_flat_hp(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_caa6c2e5a8', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                *[
                    {'id': f'drain_{index}', 'slot': 0, 'action_id': 'action_bdc49de5c5', 'start_tick': index}
                    for index in range(6)
                ],
                {'id': 'inspect', 'slot': 1, 'action_id': 'action_7d6ec164ca', 'start_tick': 10},
            ],
            'initial_energy': 200,
        })['result']
        inspect = next(item for item in result['details'] if item['step_id'] == 'inspect')
        drain = next(
            buff for buff in inspect['applied_buffs']
            if buff['rule_id'] == 'character_fadia_darkstar_hp_drain'
        )
        self.assertEqual(drain['stack_count'], 5)
        self.assertAlmostEqual(drain['effects']['flat_hp'], 5160.5)

    def test_arc_refinement_uses_nanoka_panel_and_buff_values(self) -> None:
        payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_b52cc8f160',
                'arc_id': 'arc_27dc4a7281',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
            ],
            'initial_energy': 200,
        }

        refined_one = simulate_shaft_axis(payload)['result']['details'][0]
        payload['team'][0]['arc_refinement'] = 5
        refined_five = simulate_shaft_axis(payload)['result']['details'][0]
        payload['team'][0]['arc_refinement'] = 0
        current_default = simulate_shaft_axis(payload)['result']['details'][0]

        buff_one = next(buff for buff in refined_one['applied_buffs'] if buff['rule_id'] == 'arc_crimson_mirage_q_light_def')
        buff_five = next(buff for buff in refined_five['applied_buffs'] if buff['rule_id'] == 'arc_crimson_mirage_q_light_def')
        self.assertEqual(buff_one['effects'], {'element_dmg': 0.32, 'def_ignore': 0.12})
        self.assertEqual(buff_five['effects'], {'element_dmg': 0.512, 'def_ignore': 0.192})
        self.assertGreater(refined_five['panel']['atk'], refined_one['panel']['atk'])
        self.assertAlmostEqual(current_default['panel']['atk'], refined_one['panel']['atk'])

    def test_arc_q_buff_is_not_constant_and_applies_from_q_start(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_27dc4a7281',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'before_q_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'q_trigger', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 20},
                {'id': 'after_q_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 22},
            ],
            'initial_energy': 200,
        }

        baseline_payload = deepcopy(payload)
        baseline_payload['team'][0]['arc_id'] = ''
        baseline_details = {
            detail['step_id']: detail
            for detail in simulate_shaft_axis(baseline_payload)['result']['details']
        }
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        before_q = details['before_q_e']
        self.assertEqual(before_q['applied_buffs'], [])
        self.assertAlmostEqual(before_q['formula_parts']['dmg_bonus'], 0.0)

        q_trigger = details['q_trigger']
        self.assertIn(
            'arc_crimson_mirage_q_light_def',
            {buff['rule_id'] for buff in q_trigger['triggered_buffs']},
        )
        self.assertIn(
            'arc_crimson_mirage_q_light_def',
            {buff['rule_id'] for buff in q_trigger['applied_buffs']},
        )
        self.assertAlmostEqual(
            q_trigger['formula_parts']['dmg_bonus'] - baseline_details['q_trigger']['formula_parts']['dmg_bonus'],
            0.32,
        )

        after_q = details['after_q_e']
        self.assertIn(
            'arc_crimson_mirage_q_light_def',
            {buff['rule_id'] for buff in after_q['applied_buffs']},
        )
        self.assertAlmostEqual(
            after_q['formula_parts']['dmg_bonus'] - baseline_details['after_q_e']['formula_parts']['dmg_bonus'],
            0.32,
        )

    def test_cartridge_e_buff_applies_from_e_start(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_097acfc8f6',
                },
            ],
            'steps': [
                {'id': 'first_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'second_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 16},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        first_e = details['first_e']
        second_e = details['second_e']
        self.assertIn(
            'cartridge_shadow_creed_e_atk',
            {buff['rule_id'] for buff in first_e['triggered_buffs']},
        )
        self.assertIn(
            'cartridge_shadow_creed_e_atk',
            {buff['rule_id'] for buff in first_e['applied_buffs']},
        )
        self.assertIn(
            'cartridge_shadow_creed_e_atk',
            {buff['rule_id'] for buff in second_e['applied_buffs']},
        )
        self.assertAlmostEqual(second_e['panel']['atk'], first_e['panel']['atk'])

    def test_bit_game_refine1_background_attack_applies_to_front_character(self) -> None:
        base_payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'zhenhong_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
            ],
            'initial_energy': 200,
        }
        bit_game_payload = {
            **base_payload,
            'team': [
                base_payload['team'][0],
                {
                    **base_payload['team'][1],
                    'arc_id': 'arc_a5b483cca6',
                },
            ],
        }

        base_detail = simulate_shaft_axis(base_payload)['result']['details'][0]
        bit_game_detail = simulate_shaft_axis(bit_game_payload)['result']['details'][0]

        self.assertIn(
            'arc_bit_game_background_front_atk',
            {buff['rule_id'] for buff in bit_game_detail['triggered_buffs']},
        )
        self.assertIn(
            'arc_bit_game_background_front_atk',
            {buff['rule_id'] for buff in bit_game_detail['applied_buffs']},
        )
        self.assertGreater(bit_game_detail['direct_damage'], base_detail['direct_damage'])

    def test_zhenhong_q_matches_xlsx_low_bit_game_after_yi_q(self) -> None:
        catalog = load_shaft_catalog()
        starter = catalog['starter_axis']
        payload = {
            'team': deepcopy(starter['team']),
            'character_builds': deepcopy(starter.get('character_builds') or {}),
            'team_panel_bonus': deepcopy(starter.get('team_panel_bonus') or {}),
            'options': {'switch_loss_ticks': 0, 'front_state_enabled': True},
            'steps': [
                {'id': 'yi_q', 'slot': 3, 'action_id': 'action_6ece34aff8', 'start_tick': 0},
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 2},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        detail = next(item for item in result['details'] if item['step_id'] == 'zhenhong_q')

        self.assertAlmostEqual(detail['direct_damage'], 59679.53450844564)
        self.assertAlmostEqual(detail['formula_parts']['resistance'], 0.7)
        self.assertAlmostEqual(detail['formula_parts']['defense'], 0.6134969325153374)
        self.assertIn(
            'cartridge_lost_light_q_def_ignore',
            {buff['rule_id'] for buff in detail['applied_buffs']},
        )

    def test_def_down_is_reserved_for_sagiri_character_buff(self) -> None:
        catalog = load_shaft_catalog()
        def_down_buffs = [
            buff for buff in catalog['buffs']
            if (buff.get('effects') or {}).get('def_down')
        ]
        self.assertEqual(
            [buff['id'] for buff in def_down_buffs],
            ['character_sagiri_control_def_down'],
        )

        rules = registered_buff_rules(
            [
                {
                    'slot': 0,
                    'character_id': 'char_1895e259be',
                    'character_name': '早雾',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            catalog,
        )
        sagiri_rule = next(rule for rule in rules if rule['id'] == 'character_sagiri_control_def_down')
        self.assertEqual(sagiri_rule['provider_kind'], 'character')
        self.assertEqual(sagiri_rule['effects']['def_down'], 0.1)

    def test_bit_game_refine1_psyche_damage_stacks_and_resets_on_switch(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_e0a4292b4e',
                    'arc_id': 'arc_a5b483cca6',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'first_basic', 'slot': 0, 'action_id': 'action_519f3d978a', 'start_tick': 0},
                {'id': 'second_basic', 'slot': 0, 'action_id': 'action_519f3d978a', 'start_tick': 20},
                {'id': 'switch_to_zhenhong', 'slot': 1, 'action_id': 'action_3987d8ff2d', 'start_tick': 40},
                {'id': 'after_switch_basic', 'slot': 0, 'action_id': 'action_519f3d978a', 'start_tick': 60},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        panel = next(item for item in result['build_panels_by_slot'] if item['slot'] == 0)

        self.assertAlmostEqual(panel['panel']['element_dmg'], 0.0)
        self.assertIn(
            'arc_bit_game_foreground_psyche_dmg',
            {buff['rule_id'] for buff in details['first_basic']['applied_buffs']},
        )
        self.assertNotIn(
            'arc_bit_game_basic_psyche_stack',
            {buff['rule_id'] for buff in details['first_basic']['applied_buffs']},
        )
        self.assertAlmostEqual(details['first_basic']['formula_parts']['dmg_bonus'], 0.12)
        self.assertIn(
            'arc_bit_game_basic_psyche_stack',
            {buff['rule_id'] for buff in details['second_basic']['applied_buffs']},
        )
        self.assertAlmostEqual(details['second_basic']['formula_parts']['dmg_bonus'], 0.14)
        self.assertAlmostEqual(details['after_switch_basic']['formula_parts']['dmg_bonus'], 0.12)

    def test_legacy_axis_buff_rules_still_use_runtime_buff_path(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'first_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 0},
                {'id': 'second_e', 'slot': 0, 'action_id': 'action_3987d8ff2d', 'start_tick': 16},
            ],
            'buff_rules': [
                {
                    'id': 'legacy_manual_atk',
                    'name': '旧手动攻击增益',
                    'trigger': {'slot': 0, 'action_id': 'action_3987d8ff2d'},
                    'targets': {'slots': [0]},
                    'duration_ticks': 100,
                    'modifiers': {'atk_pct': 0.5},
                },
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['first_e']['applied_buffs'], [])
        self.assertIn(
            'legacy_manual_atk',
            {buff['rule_id'] for buff in details['first_e']['triggered_buffs']},
        )
        self.assertIn(
            'legacy_manual_atk',
            {buff['rule_id'] for buff in details['second_e']['applied_buffs']},
        )
        self.assertGreater(details['second_e']['panel']['atk'], details['first_e']['panel']['atk'])

    def test_catalog_records_hit_count_for_every_action(self) -> None:
        catalog = load_shaft_catalog()
        self.assertTrue(catalog['actions'])
        self.assertTrue(all('hit_count' in action for action in catalog['actions']))
        xiaozhi_a4 = next(action for action in catalog['actions'] if action['id'] == 'action_bc6c8ca82b')
        self.assertEqual(xiaozhi_a4['hit_count'], 5)

        actions = {
            (action['character_name'], action['name']): action
            for action in catalog['actions']
        }
        expected_hit_counts = {
            ('真红', 'e'): 3, ('真红', '龙e'): 2, ('真红', 'q'): 9,
            ('真红', '穿梭'): 4, ('真红', 'q2'): 11,
            ('浔', 'e'): 7, ('浔', 'q'): 10,
            ('小吱', '蓄力'): 2, ('小吱', '分支'): 2, ('小吱', '闪反'): 2,
            ('小吱', 'e1'): 2, ('小吱', 'e2'): 2, ('小吱', 'e3'): 4, ('小吱', 'q'): 6,
            ('埃德嘉', 'e'): 7,
            ('娜娜莉', 'e'): 5, ('娜娜莉', 'q'): 7,
            ('九原', '5a'): 11, ('九原', '6枪'): 6, ('九原', 'e'): 6, ('九原', 'q'): 12,
            ('薄荷', 'e'): 2, ('薄荷', 'q'): 7,
            ('白藏', '单E'): 4, ('白藏', '单E额外'): 3,
            ('早雾', '5a'): 10, ('早雾', 'e'): 2, ('早雾', '长e'): 3, ('早雾', 'q'): 6,
            ('阿德勒', 'e'): 3,
            ('达芙蒂尔', 'e'): 8, ('达芙蒂尔', '1层e'): 8,
            ('达芙蒂尔', '2层e'): 7, ('达芙蒂尔', 'q'): 14,
            ('法帝娅', 'e'): 3, ('法帝娅', 'q'): 6, ('法帝娅', 'q后5a'): 13,
            ('哈尼娅', 'e'): 2, ('哈尼娅', '强化后台'): 4,
            ('海月', '水母闪反'): 4, ('海月', 'q'): 6,
            ('卡厄斯', 'z'): 2, ('卡厄斯', 'z2'): 2, ('卡厄斯', 'e'): 4, ('卡厄斯', 'q'): 5,
            ('哈索尔', '援护'): 2, ('哈索尔', 'e'): 3, ('哈索尔', '长e尾刀'): 10,
            ('哈索尔', 'e持续'): 20, ('哈索尔', 'q'): 3,
            ('哈索尔', 'E1'): 7, ('哈索尔', 'E2'): 3, ('哈索尔', 'E3'): 2,
            ('翳', 'e'): 6, ('翳', 'q'): 4,
        }
        for key, expected in expected_hit_counts.items():
            with self.subTest(character=key[0], action=key[1]):
                self.assertEqual(actions[key]['hit_count'], expected)

    def test_zhenhong_support_and_air_attack_match_tencent_ground_truth(self) -> None:
        catalog = load_shaft_catalog()
        actions = {
            (action['character_name'], action['name']): action
            for action in catalog['actions']
        }

        support = actions[('真红', '援护')]
        self.assertEqual(support['multipliers']['atk'], 2.0)
        self.assertEqual(support['energy_gain'], 5.3)
        self.assertEqual(support['harmony'], 0)
        self.assertEqual(support['stagger'], 2.5)
        self.assertEqual(support['source_formula'], '{0}%')
        self.assertEqual(support['hit_count'], 1)

        air_attack = actions[('真红', '跳A')]
        self.assertEqual(air_attack['multipliers']['atk'], 0.77)
        self.assertEqual(air_attack['energy_gain'], 1.56)
        self.assertEqual(air_attack['harmony'], 2.56)
        self.assertEqual(air_attack['stagger'], 0.8)
        self.assertEqual(air_attack['source_formula'], '{0}%')
        self.assertEqual(air_attack['hit_count'], 1)

    def test_zhenhong_ascendant_red_and_awakened_q1_resonance(self) -> None:
        catalog = load_shaft_catalog()
        actions = {
            action['id']: action
            for action in catalog['actions']
            if action.get('character_name') == '真红'
        }
        ascendant_red_action_ids = {
            'action_c32b4b9417',
            'action_40e3b63106',
            'action_c2c019771f',
            'action_f0625f9268',
            'action_5cd7ad2380',
            'action_cf3b21dac1',
            'action_7773821d79',
            'action_a17ac7c5b7',
            'action_e3711f0cf5',
        }
        self.assertEqual(
            {
                actions[action_id]['name']
                for action_id in ascendant_red_action_ids
            },
            {'q', '龙a1', '龙a2', '龙a3', '龙a4', '龙a5', '龙e', '穿梭', 'q2'},
        )
        for action_id in ascendant_red_action_ids:
            self.assertAlmostEqual(actions[action_id]['self_modifiers']['all_dmg'], 0.0)

        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'q1_trigger', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'dragon_a', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 2},
                {'id': 'q2_reset', 'slot': 0, 'action_id': 'action_e3711f0cf5', 'start_tick': 5},
                {'id': 'after_q2_dragon_a', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 7},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        q1_triggered_buffs = {
            buff['rule_id']: buff
            for buff in details['q1_trigger']['triggered_buffs']
        }
        self.assertAlmostEqual(details['q1_trigger']['formula_parts']['dmg_bonus'], 0.3)
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in details['q1_trigger']['applied_buffs']},
        )
        self.assertFalse(q1_triggered_buffs['character_zhenhong_ascendant_red']['display_as_line'])
        self.assertAlmostEqual(details['q1_trigger']['panel']['all_dmg'], 0.3)
        self.assertAlmostEqual(details['q1_trigger']['panel']['element_dmg'], 0.0)
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in details['q1_trigger']['triggered_buffs']},
        )
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in details['dragon_a']['applied_buffs']},
        )
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in details['dragon_a']['applied_buffs']},
        )
        self.assertAlmostEqual(details['dragon_a']['formula_parts']['dmg_bonus'], 0.3)
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in details['q2_reset']['applied_buffs']},
        )
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in details['q2_reset']['applied_buffs']},
        )
        self.assertAlmostEqual(details['q2_reset']['formula_parts']['dmg_bonus'], 0.3)
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in details['after_q2_dragon_a']['applied_buffs']},
        )
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in details['after_q2_dragon_a']['applied_buffs']},
        )
        self.assertAlmostEqual(details['after_q2_dragon_a']['formula_parts']['dmg_bonus'], 0.3)

        awakening_payload = deepcopy(payload)
        awakening_payload['team'][0]['awakening'] = 3
        awakening_result = simulate_shaft_axis(awakening_payload)['result']
        awakening_details = {detail['step_id']: detail for detail in awakening_result['details']}
        awakening_q1_triggered_buffs = {
            buff['rule_id']: buff
            for buff in awakening_details['q1_trigger']['triggered_buffs']
        }
        self.assertIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in awakening_details['q1_trigger']['triggered_buffs']},
        )
        self.assertTrue(awakening_q1_triggered_buffs['character_zhenhong_q1_self_all_dmg']['display_as_line'])
        self.assertIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in awakening_details['dragon_a']['applied_buffs']},
        )
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in awakening_details['dragon_a']['applied_buffs']},
        )
        self.assertAlmostEqual(awakening_details['dragon_a']['formula_parts']['dmg_bonus'], 0.6)
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in awakening_details['q2_reset']['applied_buffs']},
        )
        self.assertAlmostEqual(awakening_details['q2_reset']['formula_parts']['dmg_bonus'], 0.3)
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in awakening_details['after_q2_dragon_a']['applied_buffs']},
        )
        self.assertAlmostEqual(awakening_details['after_q2_dragon_a']['formula_parts']['dmg_bonus'], 0.3)

        leave_front_payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'awakening': 3,
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'q1_trigger', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'leave_front', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 2},
                {'id': 'after_leave_dragon_a', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 4},
            ],
            'initial_energy': 200,
        }

        leave_front_result = simulate_shaft_axis(leave_front_payload)['result']
        leave_front_details = {detail['step_id']: detail for detail in leave_front_result['details']}
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in leave_front_details['after_leave_dragon_a']['applied_buffs']},
        )
        self.assertAlmostEqual(leave_front_details['after_leave_dragon_a']['formula_parts']['dmg_bonus'], 0.3)

        duration_payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'awakening': 3,
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'q1_trigger', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'within_duration', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 127},
                {'id': 'after_duration', 'slot': 0, 'action_id': 'action_5cd7ad2380', 'start_tick': 136},
            ],
            'initial_energy': 200,
        }

        duration_result = simulate_shaft_axis(duration_payload)['result']
        duration_details = {detail['step_id']: detail for detail in duration_result['details']}
        self.assertIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in duration_details['within_duration']['applied_buffs']},
        )
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in duration_details['within_duration']['applied_buffs']},
        )
        self.assertAlmostEqual(duration_details['within_duration']['formula_parts']['dmg_bonus'], 0.6)
        self.assertNotIn(
            'character_zhenhong_q1_self_all_dmg',
            {buff['rule_id'] for buff in duration_details['after_duration']['applied_buffs']},
        )
        self.assertIn(
            'character_zhenhong_ascendant_red',
            {buff['rule_id'] for buff in duration_details['after_duration']['applied_buffs']},
        )
        self.assertAlmostEqual(duration_details['after_duration']['formula_parts']['dmg_bonus'], 0.3)

    def test_zhenhong_action_detail_panel_exposes_active_damage_bonus(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': [1, 2, 3, 4, 5, 6],
                },
            ],
            'steps': [
                {'id': 'q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
            ],
            'initial_energy': 200,
        })['result']

        detail = result['details'][0]
        self.assertAlmostEqual(detail['formula_parts']['dmg_bonus'], 0.3)
        self.assertAlmostEqual(detail['panel']['all_dmg'], 0.3)
        self.assertAlmostEqual(detail['formula_parts']['base_multiplier_factor'], 1.3)
        self.assertAlmostEqual(detail['panel']['element_dmg'], 0.0)

    def test_fons_condition_defaults_to_full_stacks(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_4f917797cb',
                    'arc_id': 'arc_112b3492d8',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'light_e', 'slot': 0, 'action_id': 'action_09ad16cdd4', 'start_tick': 0},
            ],
            'initial_energy': 200,
        }

        baseline_payload = deepcopy(payload)
        baseline_payload['team'][0]['arc_id'] = ''
        baseline = simulate_shaft_axis(baseline_payload)['result']['details'][0]
        result = simulate_shaft_axis(payload)['result']
        detail = result['details'][0]
        self.assertIn(
            'arc_thinking_cat_fons_light',
            {buff['rule_id'] for buff in detail['triggered_buffs']},
        )
        self.assertIn(
            'arc_thinking_cat_fons_light',
            {buff['rule_id'] for buff in detail['applied_buffs']},
        )
        self.assertAlmostEqual(
            detail['formula_parts']['dmg_bonus'] - baseline['formula_parts']['dmg_bonus'],
            0.25,
        )

    def test_enemy_debuff_applied_triggers_team_reaction_buff(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_912dbfe17c',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_9229a5376c',
                },
                {
                    'slot': 1,
                    'character_id': 'char_d38b672525',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'delay_apply', 'slot': 1, 'action_id': 'action_be26cd6a7c', 'start_tick': 0},
                {'id': 'owner_after_delay', 'slot': 0, 'action_id': 'action_61c61302f2', 'start_tick': 1},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertIn('延滞', details['delay_apply']['applied_enemy_debuffs'])
        self.assertIn(
            'cartridge_street_boxer_delay_reaction',
            {buff['rule_id'] for buff in details['delay_apply']['triggered_buffs']},
        )
        self.assertIn(
            'cartridge_street_boxer_delay_reaction',
            {buff['rule_id'] for buff in details['owner_after_delay']['applied_buffs']},
        )

    def test_diabolos_doubles_dark_resistance_ignore_after_owner_joins_dark_reaction(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_1895e259be',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_2f40014b04',
                },
            ],
            'steps': [
                *[
                    {
                        'id': f'gain_{index}',
                        'slot': 0,
                        'action_id': 'action_4c5142a40f',
                        'start_tick': index * 2,
                    }
                    for index in range(4)
                ],
                {'id': 'support', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 40},
                {'id': 'owner_after_reaction', 'slot': 1, 'action_id': 'action_00edea34a8', 'start_tick': 50},
            ],
            'initial_energy': 200,
        }
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['support']['triggered_reaction']['reaction'], '浊燃')
        self.assertAlmostEqual(details['support']['formula_parts']['settled_resistance'], 0.18)
        self.assertIn(
            'cartridge_diabolos_dark_reaction',
            {buff['rule_id'] for buff in details['support']['triggered_buffs']},
        )
        self.assertIn(
            'cartridge_diabolos_dark_reaction',
            {buff['rule_id'] for buff in details['owner_after_reaction']['applied_buffs']},
        )
        self.assertAlmostEqual(
            details['owner_after_reaction']['formula_parts']['settled_resistance'],
            0.06,
        )

        loop_payload = deepcopy(payload)
        loop_payload['options'] = {'loop_enabled': True}
        loop_result = simulate_shaft_axis(loop_payload)['result']
        loop_details = {detail['step_id']: detail for detail in loop_result['details']}
        self.assertIn(
            'cartridge_diabolos_dark_reaction',
            {buff['rule_id'] for buff in loop_details['support']['applied_buffs']},
        )
        self.assertAlmostEqual(
            loop_details['support']['formula_parts']['settled_resistance'],
            0.06,
        )

    def test_self_hp_loss_tag_triggers_hp_loss_buff(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_caa6c2e5a8',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_868dbc2c5c',
                },
            ],
            'steps': [
                {'id': 'hp_loss', 'slot': 0, 'action_id': 'action_bdc49de5c5', 'start_tick': 0},
                {'id': 'after_hp_loss', 'slot': 0, 'action_id': 'action_72051426d5', 'start_tick': 1},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertIn('self_hp_loss', details['hp_loss']['action_tags'])
        self.assertIn(
            'cartridge_little_adventure_hp_stack',
            {buff['rule_id'] for buff in details['hp_loss']['triggered_buffs']},
        )
        self.assertIn(
            'cartridge_little_adventure_hp_stack',
            {buff['rule_id'] for buff in details['after_hp_loss']['applied_buffs']},
        )
        self.assertGreater(details['after_hp_loss']['panel']['hp'], details['hp_loss']['panel']['hp'])

    def test_critical_hit_condition_uses_expected_stacks_from_hit_count(self) -> None:
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': 'arc_1ddecc32f3',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'multi_hit', 'slot': 0, 'action_id': 'action_f0625f9268', 'start_tick': 0},
            ],
            'initial_energy': 200,
        }

        result = simulate_shaft_axis(payload)['result']
        detail = result['details'][0]
        triggered = {
            buff['rule_id']: buff
            for buff in detail['triggered_buffs']
        }
        self.assertEqual(detail['hit_count'], 6)
        self.assertGreater(detail['expected_critical_hits'], 0)
        self.assertIn('arc_fierce_cotton_crit', triggered)
        self.assertAlmostEqual(triggered['arc_fierce_cotton_crit']['stack_count'], detail['expected_critical_hits'])

    def test_true_red_dragon_e_cooldown_is_one_second(self) -> None:
        catalog = load_shaft_catalog()
        action = next(action for action in catalog['actions'] if action['id'] == 'action_7773821d79')
        self.assertEqual(action['cooldown_ticks'], 10)

    def test_true_red_normal_e_cooldown_is_three_seconds(self) -> None:
        catalog = load_shaft_catalog()
        action = next(action for action in catalog['actions'] if action['id'] == 'action_3987d8ff2d')
        self.assertEqual(action['cooldown_ticks'], 30)

    def test_jiuyuan_rose_transitions_and_settlement_bonuses(self) -> None:
        result = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'awakening': 6, 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                {'id': 'marked_dot', 'slot': 0, 'action_id': 'action_d8430d0b9e', 'start_tick': 20},
                {'id': 'settlement', 'slot': 0, 'action_id': 'action_4edb91866d', 'start_tick': 60},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        marked = {buff['rule_id']: buff for buff in details['marked_dot']['applied_buffs']}
        settlement = {buff['rule_id']: buff for buff in details['settlement']['applied_buffs']}
        self.assertEqual(
            marked['character_jiuyuan_rose_a2_mark_damage']['effects'],
            {'base_multiplier_pct': 1},
        )
        self.assertAlmostEqual(details['marked_dot']['formula_parts']['base_multiplier_factor'], 2)
        self.assertIn('character_jiuyuan_deadly_rose_mark', marked)
        self.assertNotIn('character_jiuyuan_deadly_rose_mark', settlement)
        self.assertIn('character_jiuyuan_deadly_rose_settle_window', settlement)
        self.assertEqual(
            settlement['character_jiuyuan_rose_a4_settlement']['effects'],
            {'base_multiplier_pct': 0.5},
        )
        self.assertAlmostEqual(details['settlement']['formula_parts']['base_multiplier_factor'], 1.5)
        self.assertEqual(
            marked['character_jiuyuan_rose_a5_damage']['effects'],
            {'final_dmg': 0.08},
        )
        self.assertAlmostEqual(details['marked_dot']['panel']['final_dmg'], 0.08)
        self.assertAlmostEqual(details['marked_dot']['formula_parts']['final_multiplier'], 1.08)

    def test_jiuyuan_rose_buff_line_transitions_then_q_submits_it_early(self) -> None:
        natural = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
            ],
            'initial_energy': 200,
        })['result']
        natural_buffs = {
            buff['definition_id']: buff
            for buff in natural['details'][0]['triggered_buffs']
        }
        mark = natural_buffs['character_jiuyuan_rose_mark']
        settle = natural_buffs['character_jiuyuan_rose_settle']
        self.assertEqual(mark['name'], '九原·致命玫约')
        self.assertEqual(mark['start_tick'], 0)
        self.assertEqual(mark['end_tick'], 50)
        self.assertEqual(settle['name'], '九原·致命玫约可清算')
        self.assertEqual(settle['start_tick'], mark['end_tick'])
        self.assertEqual(settle['end_tick'], settle['start_tick'] + 250)

        submitted = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_5501c87ce7', 'start_tick': 20},
            ],
            'initial_energy': 200,
        })['result']
        submitted_details = {detail['step_id']: detail for detail in submitted['details']}
        original_buffs = {
            buff['definition_id']: buff
            for buff in submitted_details['mark']['triggered_buffs']
        }
        q_settle = next(
            buff for buff in submitted_details['q']['triggered_buffs']
            if buff['definition_id'] == 'character_jiuyuan_rose_settle'
        )
        self.assertEqual(original_buffs['character_jiuyuan_rose_mark']['end_tick'], 20)
        self.assertTrue(original_buffs['character_jiuyuan_rose_settle']['cancelled'])
        self.assertEqual(q_settle['start_tick'], 20)
        self.assertEqual(q_settle['end_tick'], 270)

    def test_team_genesis_hit_marks_once_without_refreshing_jiuyuan_rose_states(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_bdc43f82c6', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'gain', 'slot': 0, 'action_id': 'action_982c67944f', 'start_tick': 0},
                {'id': 'support', 'slot': 1, 'action_id': 'action_482b5d9df7', 'start_tick': 20},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        genesis_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '创生'
        ]
        first_event = genesis_events[0]
        first_buffs = {
            buff['definition_id']: buff
            for buff in first_event['triggered_buffs']
        }
        mark = first_buffs['character_jiuyuan_rose_mark']
        initial_settle = first_buffs['character_jiuyuan_rose_settle']
        self.assertEqual(mark['start_tick'], first_event['tick'])
        self.assertEqual(mark['end_tick'], first_event['tick'] + 50)
        self.assertEqual(initial_settle['start_tick'], mark['end_tick'])
        self.assertFalse(any(event.get('triggered_buffs') for event in genesis_events[1:5]))

        repeated_rose_events = [
            event for event in genesis_events[1:]
            if any(
                buff['definition_id'] in {
                    'character_jiuyuan_rose_mark',
                    'character_jiuyuan_rose_settle',
                }
                for buff in event.get('triggered_buffs', [])
            )
        ]
        self.assertEqual(repeated_rose_events, [])
        self.assertEqual(initial_settle['end_tick'], initial_settle['start_tick'] + 250)

    def test_jiuyuan_hits_do_not_reapply_or_refresh_either_rose_state(self) -> None:
        result = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'mark', 'slot': 0, 'action_id': 'action_fbc91559f8', 'start_tick': 0},
                {'id': 'during_mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 20},
                {'id': 'during_settle', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 60},
            ],
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        original_buffs = {
            buff['definition_id']: buff
            for buff in details['mark']['triggered_buffs']
        }
        rose_keys = {
            'character_jiuyuan_rose_mark',
            'character_jiuyuan_rose_settle',
        }
        self.assertEqual(
            rose_keys.intersection(
                buff['definition_id']
                for buff in details['during_mark']['triggered_buffs']
            ),
            set(),
        )
        self.assertEqual(
            rose_keys.intersection(
                buff['definition_id']
                for buff in details['during_settle']['triggered_buffs']
            ),
            set(),
        )
        self.assertEqual(original_buffs['character_jiuyuan_rose_mark']['end_tick'], 50)
        self.assertEqual(original_buffs['character_jiuyuan_rose_settle']['start_tick'], 50)
        self.assertEqual(original_buffs['character_jiuyuan_rose_settle']['end_tick'], 300)

    def test_loop_carried_jiuyuan_settlement_blocks_next_cycle_mark_until_consumed(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_dd034941ef', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'early_mark_attempt', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 8},
                {'id': 'settlement', 'slot': 0, 'action_id': 'action_7809042596', 'start_tick': 94},
                {'id': 'late_mark', 'slot': 0, 'action_id': 'action_fbc91559f8', 'start_tick': 325},
                {'id': 'foreground_end', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 400},
            ],
            'options': {'loop_enabled': True, 'switch_loss_ticks': 0},
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        rose_keys = {
            'character_jiuyuan_rose_mark',
            'character_jiuyuan_rose_settle',
        }

        self.assertEqual(
            rose_keys.intersection(
                buff['definition_id']
                for buff in details['early_mark_attempt']['triggered_buffs']
            ),
            set(),
        )
        self.assertIn(
            'character_jiuyuan_deadly_rose_settle_window',
            {buff['rule_id'] for buff in details['early_mark_attempt']['applied_buffs']},
        )
        self.assertNotIn(
            '动作需要处于 致命玫约可清算 状态。',
            details['settlement']['warnings'],
        )
        self.assertEqual(
            rose_keys.intersection(
                buff['definition_id']
                for buff in details['late_mark']['triggered_buffs']
            ),
            rose_keys,
        )

    def test_jiuyuan_settlement_actions_require_and_consume_only_settle_window(self) -> None:
        for action_id in ('action_4edb91866d', 'action_7809042596'):
            with self.subTest(action_id=action_id, state='missing'):
                missing = simulate_shaft_axis({
                    'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
                    'steps': [
                        {'id': 'settlement', 'slot': 0, 'action_id': action_id, 'start_tick': 0},
                    ],
                    'initial_energy': 200,
                })['result']
                self.assertIn(
                    '动作需要处于 致命玫约可清算 状态。',
                    missing['details'][0]['warnings'],
                )

            with self.subTest(action_id=action_id, state='base_mark'):
                base_mark = simulate_shaft_axis({
                    'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
                    'steps': [
                        {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                        {'id': 'settlement', 'slot': 0, 'action_id': action_id, 'start_tick': 20},
                        {'id': 'after', 'slot': 0, 'action_id': 'action_d8430d0b9e', 'start_tick': 30},
                    ],
                    'initial_energy': 200,
                })['result']
                base_details = {detail['step_id']: detail for detail in base_mark['details']}
                self.assertIn(
                    '动作需要处于 致命玫约可清算 状态。',
                    base_details['settlement']['warnings'],
                )
                self.assertIn(
                    'character_jiuyuan_deadly_rose_mark',
                    {buff['rule_id'] for buff in base_details['after']['applied_buffs']},
                )

            with self.subTest(action_id=action_id, state='settle_window'):
                consumed = simulate_shaft_axis({
                    'team': [{'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''}],
                    'steps': [
                        {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                        {'id': 'settlement', 'slot': 0, 'action_id': action_id, 'start_tick': 60},
                        {'id': 'after', 'slot': 0, 'action_id': 'action_d8430d0b9e', 'start_tick': 70},
                    ],
                    'initial_energy': 200,
                })['result']
                consumed_details = {detail['step_id']: detail for detail in consumed['details']}
                self.assertNotIn(
                    '动作需要处于 致命玫约可清算 状态。',
                    consumed_details['settlement']['warnings'],
                )
                self.assertNotIn(
                    'character_jiuyuan_deadly_rose_settle_window',
                    {buff['rule_id'] for buff in consumed_details['after']['applied_buffs']},
                )
                settle_line = next(
                    buff for buff in consumed_details['mark']['triggered_buffs']
                    if buff['definition_id'] == 'character_jiuyuan_rose_settle'
                )
                self.assertEqual(settle_line['end_tick'], 60)
                clear_summary = next(
                    buff for buff in consumed_details['settlement']['triggered_buffs']
                    if buff['rule_id'] == 'character_jiuyuan_rose_settlement_clear'
                )
                self.assertEqual(
                    clear_summary['cleared_buff_keys'],
                    ['character_jiuyuan_rose_settle'],
                )

    def test_jiuyuan_q_settlement_is_four_times_normal_before_awakening_multiplier(self) -> None:
        catalog = load_shaft_catalog()
        actions = {action['id']: action for action in catalog['actions']}
        normal = actions['action_4edb91866d']
        q_settlement = actions['action_7809042596']

        self.assertAlmostEqual(q_settlement['multipliers']['atk'], normal['multipliers']['atk'] * 4)
        self.assertEqual(q_settlement['source_formula'], '普通清算倍率*4')
        self.assertIn('用户 2026-07-19 补充', q_settlement['source_note'])
        self.assertIn('Nanoka', q_settlement['source_note'])

    def test_jiuyuan_c_awakening_adds_four_stagger_only_to_q_settlement(self) -> None:
        def simulate(awakening_nodes: list[int]) -> dict:
            result = simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_b2e3b2bf7a',
                    'arc_id': '',
                    'cartridge_id': '',
                    'awakening_nodes': awakening_nodes,
                }],
                'steps': [
                    {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                    {'id': 'settlement', 'slot': 0, 'action_id': 'action_7809042596', 'start_tick': 60},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']
            return next(
                detail for detail in result['details']
                if detail['step_id'] == 'settlement'
            )

        self.assertEqual(simulate([])['stagger_amount'], 0)
        self.assertEqual(simulate([3])['stagger_amount'], 4)

    def test_jiuyuan_f_awakening_is_manual_background_damage_with_mark_and_cooldown_checks(self) -> None:
        catalog = load_shaft_catalog()
        actions = {action['id']: action for action in catalog['actions']}
        recoil = actions['action_jiuyuan_f_secret_recoil']
        self.assertTrue(recoil['is_background_damage'])
        self.assertEqual(recoil['required_awakening'], 6)
        self.assertEqual(recoil['multipliers']['atk'], 2)
        self.assertEqual(recoil['cooldown_ticks'], 50)
        self.assertEqual(
            recoil['required_buff_any_keys'],
            ['character_jiuyuan_rose_mark', 'character_jiuyuan_rose_settle'],
        )

        missing = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_b2e3b2bf7a',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
            }],
            'steps': [
                {'id': 'missing', 'slot': 0, 'action_id': recoil['id'], 'start_tick': 0},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        self.assertIn(
            '动作需要处于 致命玫约或致命玫约可清算 状态。',
            missing['details'][0]['warnings'],
        )

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_b2e3b2bf7a',
                'arc_id': '',
                'cartridge_id': '',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
            }],
            'steps': [
                {'id': 'mark', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 0},
                {'id': 'first', 'slot': 0, 'action_id': recoil['id'], 'start_tick': 10},
                {'id': 'cooldown', 'slot': 0, 'action_id': recoil['id'], 'start_tick': 40},
                {'id': 'ready', 'slot': 0, 'action_id': recoil['id'], 'start_tick': 90},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        self.assertNotIn(
            '动作需要处于 致命玫约或致命玫约可清算 状态。',
            details['first']['warnings'],
        )
        self.assertTrue(any('CD 尚未结束' in warning for warning in details['cooldown']['warnings']))
        self.assertFalse(any('CD 尚未结束' in warning for warning in details['ready']['warnings']))
        self.assertGreater(details['first']['direct_damage'], 0)

    def test_explicit_damage_multiplier_rules_do_not_use_all_damage_bonus(self) -> None:
        catalog = load_shaft_catalog()
        rules = {rule['id']: rule for rule in catalog['buffs']}
        expected = {
            'character_protagonist_a6_attach_damage': 0.5,
            'character_zhenhong_a6_skill_multiplier': 0.3,
            'character_yi_a1_beast_damage': 0.6,
            'character_yi_a6_extra_damage': 0.2,
            'character_jiuyuan_rose_a2_mark_damage': 1.0,
            'character_jiuyuan_rose_a4_settlement': 0.5,
        }

        for rule_id, multiplier_increase in expected.items():
            with self.subTest(rule_id=rule_id):
                self.assertEqual(
                    rules[rule_id]['effects'],
                    {'base_multiplier_pct': multiplier_increase},
                )

    def test_baizang_words_are_consumed_by_q_and_empower_next_e(self) -> None:
        result = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_701295143d', 'awakening': 2, 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'jin', 'slot': 0, 'action_id': 'action_c234af7127', 'start_tick': 0},
                {'id': 'q', 'slot': 0, 'action_id': 'action_9307778f01', 'start_tick': 20},
                {'id': 'next_e', 'slot': 0, 'action_id': 'action_9166cd44f2', 'start_tick': 40},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        q_triggered = {buff['rule_id']: buff for buff in details['q']['triggered_buffs']}
        next_e = {buff['rule_id']: buff for buff in details['next_e']['applied_buffs']}
        self.assertEqual(q_triggered['character_baizang_a1_next_skill']['effects'], {'all_dmg': 0.3})
        self.assertEqual(next_e['character_baizang_a1_next_skill']['effects'], {'all_dmg': 0.3})
        self.assertEqual(next_e['character_baizang_a2_word_atk']['effects'], {})

    def test_sagiri_counts_only_registered_negative_states_and_converts_base_atk(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_1895e259be', 'awakening': 6, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_701295143d', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 2, 'character_id': 'char_c78f7a08d5', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 3, 'character_id': 'char_6f46705fd1', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                *[
                    {'id': f'gain_{index}', 'slot': 2, 'action_id': 'action_6d2645f71e', 'start_tick': index}
                    for index in range(15)
                ],
                {'id': 'murk_support', 'slot': 3, 'action_id': 'action_55c8843d1c', 'start_tick': 20},
                {'id': 'baizang_dot', 'slot': 1, 'action_id': 'action_10c15dd4d1', 'start_tick': 33},
                {'id': 'second_dot', 'slot': 1, 'action_id': 'action_10c15dd4d1', 'start_tick': 50},
                {'id': 'sagiri_q', 'slot': 0, 'action_id': 'action_0db78d1f01', 'start_tick': 70},
                {'id': 'teammate_after_q', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 90},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        dot_buffs = {buff['rule_id']: buff for buff in details['second_dot']['applied_buffs']}
        sagiri_buffs = {buff['rule_id']: buff for buff in details['sagiri_q']['applied_buffs']}
        teammate_buffs = {buff['rule_id']: buff for buff in details['teammate_after_q']['applied_buffs']}
        self.assertEqual(dot_buffs['character_sagiri_dot_amplification']['effects'], {'final_dmg': 0.5})
        self.assertAlmostEqual(details['second_dot']['panel']['final_dmg'], 0.5)
        self.assertAlmostEqual(details['second_dot']['formula_parts']['final_multiplier'], 1.5)
        self.assertEqual(sagiri_buffs['character_sagiri_a6_negative_damage']['effects'], {'all_dmg': 0.06})
        self.assertAlmostEqual(teammate_buffs['character_sagiri_q_base_team_flat_atk']['effects']['flat_atk'], 186)
        self.assertAlmostEqual(teammate_buffs['character_sagiri_q_team_flat_atk']['effects']['flat_atk'], 186)
        self.assertAlmostEqual(
            teammate_buffs['character_sagiri_q_base_team_flat_atk']['effects']['flat_atk']
            + teammate_buffs['character_sagiri_q_team_flat_atk']['effects']['flat_atk'],
            372,
        )

    def test_baizang_and_adler_dot_actions_schedule_their_own_periodic_damage(self) -> None:
        catalog = load_shaft_catalog()
        actions = {action['id']: action for action in catalog['actions']}
        baizang_action = actions['action_10c15dd4d1']
        adler_action = actions['action_881b816d9f']

        self.assertEqual(baizang_action['duration_ticks'], 160)
        self.assertEqual(
            baizang_action['periodic_damage'],
            {
                'interval_ticks': 10,
                'tick_count': 16,
                'multipliers': {'atk': 0.15, 'hp': 0, 'def': 0, 'flat': 0},
            },
        )
        self.assertEqual(adler_action['duration_ticks'], 150)
        self.assertEqual(
            adler_action['periodic_damage'],
            {
                'interval_ticks': 15,
                'tick_count': 10,
                'multipliers': {'atk': 0, 'hp': 0, 'def': 0.2, 'flat': 0},
            },
        )

        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_701295143d', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_6f46705fd1', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'baizang_dot', 'slot': 0, 'action_id': 'action_10c15dd4d1', 'start_tick': 0},
                {'id': 'adler_dot', 'slot': 1, 'action_id': 'action_881b816d9f', 'start_tick': 0},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}
        baizang_events = [
            event for event in result['periodic_damage_events']
            if event.get('action_id') == 'action_10c15dd4d1'
        ]
        adler_events = [
            event for event in result['periodic_damage_events']
            if event.get('action_id') == 'action_881b816d9f'
        ]

        self.assertEqual(details['baizang_dot']['direct_damage'], 0)
        self.assertEqual(details['adler_dot']['direct_damage'], 0)
        self.assertEqual([event['tick'] for event in baizang_events], list(range(10, 161, 10)))
        self.assertEqual([event['tick'] for event in adler_events], list(range(15, 151, 15)))
        self.assertTrue(all(event['damage'] > 0 for event in baizang_events + adler_events))
        self.assertTrue(all(
            event['formula_parts']['crit_rate'] == 0.5
            for event in baizang_events + adler_events
        ))

        buff_rule_ids = {buff['id'] for buff in catalog['buffs']}
        self.assertNotIn('character_sagiri_baizang_dot_marker', buff_rule_ids)
        self.assertNotIn('character_sagiri_adler_dot_marker', buff_rule_ids)

    def test_sagiri_reads_active_periodic_dot_actions_without_own_marker_buffs(self) -> None:
        def first_baizang_dot_event(include_sagiri):
            team = [
                {'slot': 0, 'character_id': 'char_701295143d', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
            ]
            if include_sagiri:
                team.append(
                    {'slot': 1, 'character_id': 'char_1895e259be', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                )
            result = simulate_shaft_axis({
                'team': team,
                'steps': [
                    {'id': 'baizang_dot', 'slot': 0, 'action_id': 'action_10c15dd4d1', 'start_tick': 0},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']
            return next(
                event
                for event in result['periodic_damage_events']
                if event.get('action_id') == 'action_10c15dd4d1'
            )

        solo = first_baizang_dot_event(False)
        with_sagiri = first_baizang_dot_event(True)
        self.assertEqual(with_sagiri['formula_parts']['final_multiplier'], 1)
        self.assertAlmostEqual(with_sagiri['damage'], solo['damage'])

    def test_sagiri_passive_amplifies_requiem_nightmare_periodic_damage(self) -> None:
        def simulate(team):
            return simulate_shaft_axis({
                'team': team,
                'steps': [
                    {'id': 'requiem_q', 'slot': 0, 'action_id': 'action_1104de864b', 'start_tick': 0},
                    {'id': 'inspect', 'slot': 0, 'action_id': 'action_b9b3237c74', 'start_tick': 40},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        requiem = {
            'slot': 0,
            'character_id': 'char_c78f7a08d5',
            'awakening': 0,
            'arc_id': '',
            'cartridge_id': '',
        }
        solo = simulate([requiem])
        with_sagiri = simulate([
            requiem,
            {
                'slot': 1,
                'character_id': 'char_1895e259be',
                'awakening': 0,
                'arc_id': '',
                'cartridge_id': '',
            },
        ])

        solo_nightmare = solo['periodic_damage_events'][0]
        sagiri_nightmare = with_sagiri['periodic_damage_events'][0]
        self.assertEqual(solo_nightmare['reaction'], '噩梦')
        self.assertEqual(sagiri_nightmare['formula_parts']['final_multiplier'], 1)
        self.assertAlmostEqual(sagiri_nightmare['damage'], solo_nightmare['damage'])

    def test_sagiri_talent_one_uses_dot_count_as_final_multiplier_only_during_corrosion(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_1895e259be',
                    'awakening': 0,
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_c78f7a08d5',
                    'awakening': 0,
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                *[
                    {
                        'id': f'gain_{index}',
                        'slot': 0,
                        'action_id': 'action_4c5142a40f',
                        'start_tick': index * 2,
                    }
                    for index in range(4)
                ],
                {
                    'id': 'support',
                    'slot': 1,
                    'action_id': 'action_2635f721a8',
                    'start_tick': 40,
                },
                {
                    'id': 'requiem_q',
                    'slot': 1,
                    'action_id': 'action_1104de864b',
                    'start_tick': 65,
                },
                {
                    'id': 'extend_axis',
                    'slot': 0,
                    'action_id': 'action_4c5142a40f',
                    'start_tick': 100,
                },
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        corrosion_events = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '浊燃' and event['damage'] > 0
        ]
        nightmare_events = [
            event for event in result['periodic_damage_events']
            if event['reaction'] == '噩梦' and event['damage'] > 0
        ]

        self.assertGreaterEqual(len(corrosion_events), 2)
        self.assertGreaterEqual(len(nightmare_events), 1)
        self.assertEqual(corrosion_events[0]['formula_parts']['final_multiplier'], 1.25)
        self.assertEqual(corrosion_events[1]['formula_parts']['final_multiplier'], 1.5)
        self.assertEqual(nightmare_events[0]['formula_parts']['final_multiplier'], 1.5)

    def test_sagiri_counts_canhong_dot_types_by_damage_tag_without_layer_duplication(self) -> None:
        def simulate(include_poison_fire: bool):
            steps = [
                {'id': 'canhong', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                {'id': 'turbid-burn', 'slot': 2, 'action_id': 'action_2635f721a8', 'start_tick': 10},
                {'id': 'corrosion-heart', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 25},
            ]
            if include_poison_fire:
                steps.append({'id': 'poison-fire', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 30})
            steps.append({'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 60})
            return simulate_shaft_axis({
                'team': [
                    {'slot': 0, 'character_id': 'char_076a1f4e53', 'arc_id': '', 'cartridge_id': ''},
                    {'slot': 1, 'character_id': 'char_1895e259be', 'arc_id': '', 'cartridge_id': ''},
                    {'slot': 2, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
                ],
                'steps': steps,
                'initial_energy': 200,
            })['result']

        corrosion_only = simulate(False)
        both_dot_types = simulate(True)
        corrosion_event = next(
            event for event in corrosion_only['periodic_damage_events']
            if event['reaction'] == '蚀心'
        )
        corrosion_with_poison = next(
            event for event in both_dot_types['periodic_damage_events']
            if event['reaction'] == '蚀心'
        )
        poison_event = next(
            event for event in both_dot_types['periodic_damage_events']
            if event['reaction'] == '鸩火'
        )

        self.assertEqual(corrosion_event['stack_count'], 2)
        self.assertEqual(corrosion_event['formula_parts']['final_multiplier'], 1.5)
        self.assertEqual(corrosion_with_poison['formula_parts']['final_multiplier'], 1.75)
        self.assertEqual(poison_event['formula_parts']['final_multiplier'], 1.75)

    def test_sagiri_q_base_team_flat_atk_does_not_require_awakening_four(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_1895e259be', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_701295143d', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'sagiri_q', 'slot': 0, 'action_id': 'action_0db78d1f01', 'start_tick': 0},
                {'id': 'teammate_after_q', 'slot': 1, 'action_id': 'action_c234af7127', 'start_tick': 20},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        teammate = next(detail for detail in result['details'] if detail['step_id'] == 'teammate_after_q')
        teammate_buffs = {buff['rule_id']: buff for buff in teammate['applied_buffs']}

        self.assertAlmostEqual(
            teammate_buffs['character_sagiri_q_base_team_flat_atk']['effects']['flat_atk'],
            186,
        )
        self.assertNotIn('character_sagiri_q_team_flat_atk', teammate_buffs)

    def test_requiem_f_e_window_reuses_pre_e_front_for_reaction(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_1895e259be', 'awakening': 0, 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_c78f7a08d5', 'awakening_nodes': [6], 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'previous_front', 'slot': 0, 'action_id': 'action_4c5142a40f', 'start_tick': 0},
                {'id': 'requiem_e', 'slot': 1, 'action_id': 'action_2745f804a5', 'start_tick': 20},
                {'id': 'free_support', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 40},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        support = next(detail for detail in result['details'] if detail['step_id'] == 'free_support')
        self.assertEqual(support['warnings'], [])
        self.assertEqual(support['triggered_reaction']['reaction'], '浊燃')
        self.assertEqual(support['triggered_reaction']['previous_slot'], 0)

    def test_daphne_insight_stacks_and_yi_attachment_scales_with_elapsed_time(self) -> None:
        daphne = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_e8ad982185', 'awakening': 3, 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'q1', 'slot': 0, 'action_id': 'action_f5e921e6d8', 'start_tick': 0},
                {'id': 'q2', 'slot': 0, 'action_id': 'action_f5e921e6d8', 'start_tick': 50},
            ],
            'initial_energy': 200,
        })['result']
        insight = next(
            buff for buff in daphne['details'][-1]['applied_buffs']
            if buff['definition_id'] == 'character_daphne_insight'
        )
        self.assertEqual(insight['stack_count'], 2)
        self.assertEqual(insight['end_tick'], 345)

        yi = simulate_shaft_axis({
            'team': [{'slot': 0, 'character_id': 'char_7578b18979', 'awakening': 4, 'arc_id': '', 'cartridge_id': ''}],
            'steps': [
                {'id': 'e', 'slot': 0, 'action_id': 'action_2e072f2b0b', 'start_tick': 0},
                {'id': 'fang', 'slot': 0, 'action_id': 'action_ff4f07b530', 'start_tick': 30},
            ],
            'initial_energy': 200,
        })['result']
        attachment = next(
            buff for buff in yi['details'][-1]['applied_buffs']
            if buff['rule_id'] == 'character_yi_a4_attachment_state'
        )
        self.assertEqual(attachment['effects'], {'attach_dmg': 0.06})

    def test_canhong_corrosion_heart_and_signature_arc_are_calculated(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': 'arc_canhong_devouring_heart',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'enhanced-e', 'slot': 0, 'action_id': 'action_canhong_E1', 'start_tick': 0},
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 31},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        enhanced_e = result['details'][0]
        triggered_ids = {buff['definition_id'] for buff in enhanced_e['triggered_buffs']}
        self.assertIn('character_canhong_corrosion_heart', triggered_ids)
        self.assertIn('arc_canhong_devouring_heart_crit_damage', triggered_ids)
        corrosion_buff = next(
            buff for buff in enhanced_e['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_corrosion_heart'
        )
        self.assertEqual(corrosion_buff['stack_count'], 5)
        corrosion_ticks = [
            event for event in result['periodic_damage_events']
            if event['reaction'] == '蚀心'
        ]
        self.assertEqual([event['tick'] for event in corrosion_ticks[:3]], [10, 20, 30])
        self.assertEqual(corrosion_ticks[-1]['tick'], 300)
        self.assertEqual(len(corrosion_ticks), 30)
        self.assertTrue(all(event['stack_count'] == 5 for event in corrosion_ticks))
        self.assertTrue(all(event['atk_multiplier'] == 0.015 for event in corrosion_ticks))
        self.assertTrue(all(event['action_type'] == '普攻' for event in corrosion_ticks))
        self.assertTrue(all(event['damage_type'] == '普攻' for event in corrosion_ticks))

        blood_banquet = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'blood-banquet',
                'slot': 0,
                'action_id': 'action_canhong_q_blood_banquet',
                'start_tick': 0,
            }],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        poison_ticks = [
            event for event in blood_banquet['periodic_damage_events']
            if event['reaction'] == '鸩火'
        ]
        self.assertEqual(len(poison_ticks), 30)
        self.assertTrue(all(event['stack_count'] == 5 for event in poison_ticks))
        self.assertTrue(all(event['action_type'] == 'Q' for event in poison_ticks))
        self.assertTrue(all(event['damage_type'] == 'Q' for event in poison_ticks))
        self.assertFalse(any(
            event['reaction'] == '蚀心'
            for event in blood_banquet['periodic_damage_events']
        ))
        self.assertGreater(poison_ticks[0]['formula_parts']['critical'], 1.3)

    def test_canhong_fuwen_bonus_is_reported_separately_from_direct_damage(self) -> None:
        def simulate(ignore_harmony_requirement: bool) -> dict:
            return simulate_shaft_axis({
                'team': [
                    {'slot': 0, 'character_id': 'char_076a1f4e53', 'arc_id': '', 'cartridge_id': ''},
                    {'slot': 1, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''},
                ],
                'steps': [
                    {'id': 'canhong', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                    {
                        'id': 'jiuyuan-support',
                        'slot': 1,
                        'action_id': 'action_8a8000d4a4',
                        'start_tick': 10,
                        'ignore_harmony_requirement': ignore_harmony_requirement,
                    },
                    {'id': 'blood', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 30},
                ],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']

        baseline = simulate(False)
        fuwen = simulate(True)
        baseline_blood = next(detail for detail in baseline['details'] if detail['step_id'] == 'blood')
        fuwen_blood = next(detail for detail in fuwen['details'] if detail['step_id'] == 'blood')
        damage_by_source = {item['source']: item for item in fuwen['damage_by_source']}

        self.assertAlmostEqual(fuwen_blood['direct_damage'], baseline_blood['direct_damage'])
        self.assertGreater(fuwen_blood['fuwen_damage'], 0)
        expected_amplification = 1.2 * (1 + 0.2 * 100 / (100 + 180))
        self.assertAlmostEqual(
            fuwen_blood['formula_parts']['reaction_amplification'],
            expected_amplification,
        )
        self.assertAlmostEqual(
            fuwen_blood['fuwen_damage'],
            baseline_blood['direct_damage'] * (expected_amplification - 1),
        )
        self.assertAlmostEqual(
            fuwen_blood['direct_damage'] + fuwen_blood['fuwen_damage'],
            baseline_blood['direct_damage'] * fuwen_blood['formula_parts']['reaction_amplification'],
        )
        self.assertAlmostEqual(damage_by_source['覆纹']['damage'], fuwen_blood['fuwen_damage'])
        self.assertAlmostEqual(
            fuwen['summary']['character_damage'] + fuwen['summary']['harmony_damage'],
            fuwen['summary']['direct_damage'],
        )

    def test_fuwen_without_harmony_strength_keeps_the_base_twenty_percent_amplification(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_b2e3b2bf7a', 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                *[
                    {'id': f'gain-{index}', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': index * 2}
                    for index in range(4)
                ],
                {'id': 'support', 'slot': 1, 'action_id': 'action_222f577087', 'start_tick': 40},
                {'id': 'fuwen-hit', 'slot': 0, 'action_id': 'action_39d4605011', 'start_tick': 60},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
        })['result']

        fuwen_hit = next(detail for detail in result['details'] if detail['step_id'] == 'fuwen-hit')
        self.assertEqual(fuwen_hit['panel']['harmony_strength'], 0)
        self.assertAlmostEqual(fuwen_hit['formula_parts']['reaction_amplification'], 1.2)
        self.assertAlmostEqual(fuwen_hit['fuwen_damage'], fuwen_hit['direct_damage'] * 0.2)

    def test_canhong_three_node_resonance_increases_skill_level_and_all_ultimate_multipliers(self) -> None:
        def simulate(action_id: str, awakening_nodes: list[int]) -> dict:
            return simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_076a1f4e53',
                    'awakening_nodes': awakening_nodes,
                    'arc_id': '',
                    'cartridge_id': '',
                }],
                'steps': [{
                    'id': 'ultimate',
                    'slot': 0,
                    'action_id': action_id,
                    'start_tick': 0,
                }],
                'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                'initial_energy': 200,
            })['result']['details'][0]

        single_c_node = simulate('action_canhong_q', [3])
        self.assertEqual(single_c_node['formula_parts']['skill_level'], 10)
        self.assertNotIn(
            'character_canhong_three_node_resonance_ultimate_multiplier',
            {buff['rule_id'] for buff in single_c_node['applied_buffs']},
        )

        for action_id in (
            'action_canhong_q',
            'action_canhong_q_enhanced',
            'action_canhong_q_blood_banquet',
        ):
            with self.subTest(action_id=action_id):
                before = simulate(action_id, [1, 2])
                awakened = simulate(action_id, [1, 2, 4])
                buffs = {buff['rule_id']: buff for buff in awakened['applied_buffs']}

                self.assertEqual(before['formula_parts']['skill_level'], 10)
                self.assertEqual(awakened['formula_parts']['skill_level'], 11)
                self.assertEqual(
                    buffs['character_canhong_three_node_resonance_ultimate_multiplier']['effects'],
                    {'base_multiplier_pct': 0.2},
                )
                self.assertAlmostEqual(
                    awakened['direct_damage'] / before['direct_damage'],
                    1.08 * 1.2,
                )

    def test_canhong_six_node_resonance_attack_buff_triggers_after_damage_and_refreshes_without_stacking(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                {'id': 'second', 'slot': 0, 'action_id': 'action_canhong_a2', 'start_tick': 100},
                {'id': 'third', 'slot': 0, 'action_id': 'action_canhong_a3', 'start_tick': 250},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertFalse(any(
            buff['rule_id'] == 'character_canhong_six_node_resonance_attack'
            for buff in details['first']['applied_buffs']
        ))
        self.assertAlmostEqual(details['first']['panel']['atk'], 660)
        self.assertAlmostEqual(details['second']['panel']['atk'], 660 * 1.4)
        self.assertAlmostEqual(details['third']['panel']['atk'], 660 * 1.4)
        for step_id in ('second', 'third'):
            active = [
                buff for buff in details[step_id]['applied_buffs']
                if buff['rule_id'] == 'character_canhong_six_node_resonance_attack'
            ]
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]['effects'], {'atk_pct': 0.4})

        refreshed = {
            step_id: next(
                buff for buff in details[step_id]['triggered_buffs']
                if buff['rule_id'] == 'character_canhong_six_node_resonance_attack'
            )
            for step_id in ('first', 'second', 'third')
        }
        self.assertEqual(refreshed['first']['end_tick'], 200)
        self.assertEqual(refreshed['second']['end_tick'], 300)
        self.assertEqual(refreshed['third']['end_tick'], 450)

    def test_canhong_six_node_resonance_attack_buff_is_refreshed_by_own_periodic_damage(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'awakening_nodes': [1, 2, 3, 4, 5, 6],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'illusion', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 0},
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 12},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        first_periodic = result['periodic_damage_events'][0]
        refreshed = next(
            buff for buff in first_periodic['triggered_buffs']
            if buff['rule_id'] == 'character_canhong_six_node_resonance_attack'
        )

        self.assertEqual(first_periodic['tick'], 10)
        self.assertEqual(refreshed['end_tick'], 210)

    def test_loop_priming_keeps_signature_arc_stacks_across_axis_end(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': 'arc_canhong_devouring_heart',
                'arc_refinement': 1,
                'cartridge_id': '',
            }],
            'steps': [
                {
                    'id': f'illusion_{index}',
                    'slot': 0,
                    'action_id': 'action_canhong_illusion_a1',
                    'start_tick': index * 3,
                }
                for index in range(7)
            ] + [{
                'id': 'axis_end',
                'slot': 0,
                'action_id': 'action_none_076a1f4e53',
                'start_tick': 30,
            }],
            'options': {'loop_enabled': True, 'switch_loss_ticks': 0},
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        first_hit = result['details'][0]
        carried_arc = next(
            buff for buff in first_hit['applied_buffs']
            if buff['rule_id'] == 'arc_canhong_devouring_heart_crit_damage'
        )
        refreshed_arc = next(
            buff for buff in first_hit['triggered_buffs']
            if buff['rule_id'] == 'arc_canhong_devouring_heart_crit_damage'
        )

        self.assertEqual(carried_arc['stack_count'], 7)
        self.assertEqual(refreshed_arc['stack_count'], 7)

    def test_poison_fire_can_reach_ten_layers_without_exceeding_cap(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'blood_banquet',
                'slot': 0,
                'action_id': 'action_canhong_q_blood_banquet',
                'start_tick': 0,
            }] + [
                {
                    'id': f'illusion_{index}',
                    'slot': 0,
                    'action_id': 'action_canhong_illusion_a1',
                    'start_tick': 5 + index * 3,
                }
                for index in range(7)
            ] + [{
                'id': 'axis_end',
                'slot': 0,
                'action_id': 'action_none_076a1f4e53',
                'start_tick': 40,
            }],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        poison_ticks = [
            event for event in result['periodic_damage_events']
            if event['reaction'] == '鸩火'
        ]

        self.assertEqual(max(event['stack_count'] for event in poison_ticks), 10)
        self.assertTrue(all(event['stack_count'] <= 10 for event in poison_ticks))

    def test_dot_spread_replaces_earliest_expiring_independent_layer_at_cap(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'blood-1', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 0},
                {'id': 'blood-2', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 10},
                {'id': 'illusion', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 100},
                {'id': 'inspect', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 101},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 500,
        })['result']
        illusion = next(detail for detail in result['details'] if detail['step_id'] == 'illusion')
        inspect = next(detail for detail in result['details'] if detail['step_id'] == 'inspect')
        spread = next(
            buff for buff in illusion['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_illusion_dot_spread'
        )
        poison_addition = next(
            addition for addition in spread['dot_layer_additions']
            if addition['name'] == '鸩火'
        )
        poison_layers = [
            buff for buff in inspect['applied_buffs']
            if buff['definition_id'] == 'character_canhong_poison_fire'
        ]

        self.assertEqual(poison_addition['stack_count'], 10)
        self.assertEqual(len(poison_layers), 10)
        self.assertEqual(
            [(buff['start_tick'], buff['end_tick']) for buff in poison_layers],
            [(0, 300)] * 4 + [(5, 305)] * 5 + [(90, 390)],
        )

    def test_delusion_increases_requiem_team_dot_critical_damage(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_076a1f4e53',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'requiem_e', 'slot': 0, 'action_id': 'action_2745f804a5', 'start_tick': 0},
                {'id': 'canhong_e', 'slot': 1, 'action_id': 'action_canhong_e1', 'start_tick': 12},
                {'id': 'axis_end', 'slot': 1, 'action_id': 'action_none_076a1f4e53', 'start_tick': 40},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']
        nightmare_ticks = [
            event for event in result['periodic_damage_events']
            if event['reaction'] == '噩梦'
        ]

        self.assertGreaterEqual(len(nightmare_ticks), 2)
        self.assertAlmostEqual(
            nightmare_ticks[1]['formula_parts']['critical']
            - nightmare_ticks[0]['formula_parts']['critical'],
            0.125,
        )

    def test_canhong_illusion_attacks_add_one_layer_to_every_active_dot(self) -> None:
        expected_layers = {
            'action_canhong_illusion_a1': 2,
            'action_canhong_illusion_a2': 4,
            'action_canhong_illusion_a3': 5,
            'action_canhong_illusion_a4': 4,
        }
        for action_id, layer_count in expected_layers.items():
            with self.subTest(action_id=action_id):
                hit_result = simulate_shaft_axis({
                    'team': [{
                        'slot': 0,
                        'character_id': 'char_076a1f4e53',
                        'arc_id': '',
                        'cartridge_id': '',
                    }],
                    'steps': [
                        {'id': 'illusion-hit', 'slot': 0, 'action_id': action_id, 'start_tick': 0},
                        {'id': 'inspect-hit', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 20},
                    ],
                    'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
                    'initial_energy': 200,
                })['result']
                illusion_hit = next(detail for detail in hit_result['details'] if detail['step_id'] == 'illusion-hit')
                inspect_hit = next(detail for detail in hit_result['details'] if detail['step_id'] == 'inspect-hit')
                corrosion_trigger = next(
                    buff for buff in illusion_hit['triggered_buffs']
                    if buff['rule_id'] == 'character_canhong_corrosion_heart'
                    and buff.get('trigger_event') == 'action_hit'
                )
                spread_triggers = [
                    buff for buff in illusion_hit['triggered_buffs']
                    if buff['definition_id'] == 'character_canhong_illusion_dot_spread'
                ]
                corrosion_layers = [
                    buff for buff in inspect_hit['applied_buffs']
                    if buff['definition_id'] == 'character_canhong_corrosion_heart'
                ]

                self.assertEqual(corrosion_trigger['stack_count'], layer_count - 1)
                self.assertEqual(len(spread_triggers), 1)
                self.assertEqual(spread_triggers[0]['dot_layer_additions'][0]['stack_count'], layer_count)
                self.assertEqual(len(corrosion_layers), layer_count)

        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_076a1f4e53',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 2,
                    'character_id': 'char_701295143d',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'external-dot', 'slot': 2, 'action_id': 'action_10c15dd4d1', 'start_tick': 0},
                {'id': 'turbid-burn', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 0},
                {'id': 'illusion-a', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 15},
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 30},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        illusion = next(detail for detail in result['details'] if detail['step_id'] == 'illusion-a')
        spread = next(
            buff for buff in illusion['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_illusion_dot_spread'
        )
        additions = {(item['kind'], item['name']): item['stack_count'] for item in spread['dot_layer_additions']}
        self.assertEqual(additions[('buff_periodic', '蚀心')], 2)
        self.assertEqual(additions[('reaction', '浊燃')], 2)
        self.assertEqual(additions[('action_periodic', 'action_10c15dd4d1')], 2)

        corrosion = next(event for event in result['periodic_damage_events'] if event['reaction'] == '蚀心')
        external_dot = [
            event for event in result['periodic_damage_events']
            if event['reaction'] == 'q的dot' and event['tick'] > 15
        ]
        turbid_burn = [
            event for event in result['reaction_damage_events']
            if event['reaction'] == '浊燃' and event['tick'] > 15
        ]
        turbid_effects = [
            effect for effect in result['reaction_effects']
            if effect['reaction'] == '浊燃'
        ]
        self.assertEqual(corrosion['stack_count'], 2)
        self.assertEqual(corrosion['formula_parts']['periodic_scale'], 2)
        self.assertIn(20, [event['tick'] for event in external_dot])
        self.assertIn(25, [event['tick'] for event in external_dot])
        self.assertIn(175, [event['tick'] for event in external_dot])
        self.assertTrue(all(event['formula_parts']['periodic_scale'] == 1 for event in external_dot))
        self.assertIn(23, [event['tick'] for event in turbid_burn])
        self.assertNotIn(25, [event['tick'] for event in turbid_burn])
        self.assertEqual(len({event['effect_id'] for event in turbid_burn}), 1)
        self.assertTrue(all(event['damage'] > 0 for event in turbid_burn))
        self.assertTrue(all(event['formula_parts']['frequency_multiplier'] == 2 for event in turbid_burn))
        self.assertEqual(
            [
                (
                    effect['start_tick'],
                    effect['end_tick'],
                    effect['stack_count'],
                    effect['frequency_multiplier'],
                )
                for effect in turbid_effects
            ],
            [(13, 165, 2, 2)],
        )

    def test_canhong_illusion_attack_adds_and_projects_requiem_nightmare_layer(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_076a1f4e53',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_c78f7a08d5',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'nightmare', 'slot': 1, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'illusion-a', 'slot': 0, 'action_id': 'action_canhong_illusion_a1', 'start_tick': 15},
            ],
            'team_panel_bonus': ShaftSimulatorValidationTestCase.ZERO_TEAM_PANEL_BONUS,
            'initial_energy': 200,
        })['result']

        illusion = next(detail for detail in result['details'] if detail['step_id'] == 'illusion-a')
        spread = next(
            buff for buff in illusion['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_illusion_dot_spread'
        )
        nightmare_addition = next(
            addition for addition in spread['dot_layer_additions']
            if addition['name'] == '噩梦'
        )
        projected_nightmare = next(
            buff for buff in illusion['triggered_buffs']
            if buff['definition_id'] == 'character_requiem_nightmare'
            and buff['start_tick'] == 15
        )
        requiem_resources = next(
            resources for resources in illusion['resources_after_by_slot']
            if resources['slot'] == 1
        )
        nightmare_events = {
            event['tick']: event
            for event in result['periodic_damage_events']
            if event['reaction'] == '噩梦'
        }

        self.assertEqual(nightmare_addition['stack_count'], 3)
        self.assertEqual(projected_nightmare['owner_slot'], 1)
        self.assertEqual(projected_nightmare['stack_count'], 1)
        self.assertEqual(requiem_resources['personal_resources']['噩梦'], 3)
        self.assertEqual(nightmare_events[10]['stack_count'], 2)
        self.assertEqual(nightmare_events[20]['stack_count'], 3)

    def test_canhong_a_awakening_spreads_existing_dot_with_reality_attack(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_076a1f4e53', 'awakening_nodes': [1], 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'nightmare', 'slot': 1, 'action_id': 'action_00edea34a8', 'start_tick': 0},
                {'id': 'reality-a', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 15},
            ],
            'initial_energy': 200,
        })['result']

        reality_attack = next(detail for detail in result['details'] if detail['step_id'] == 'reality-a')
        spread = next(
            buff for buff in reality_attack['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_a_dot_spread'
        )
        nightmare = next(item for item in spread['dot_layer_additions'] if item['name'] == '噩梦')

        self.assertEqual(nightmare['stack_count'], 3)

    def test_canhong_b_awakening_stores_blaze_and_consumes_it_on_fentian(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_076a1f4e53', 'awakening_nodes': [2], 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'canhong', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                {'id': 'reaction', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 10},
                {'id': 'fentian', 'slot': 0, 'action_id': 'action_canhong_q', 'start_tick': 30},
            ],
            'initial_energy': 200,
        })['result']

        reaction = next(detail for detail in result['details'] if detail['step_id'] == 'reaction')
        blaze = next(
            buff for buff in reaction['triggered_buffs']
            if buff['definition_id'] == 'character_canhong_b_blaze_stack'
        )
        fentian = next(detail for detail in result['details'] if detail['step_id'] == 'fentian')

        self.assertEqual(blaze['stack_count'], 1)
        self.assertEqual(fentian['formula_parts']['final_multiplier'], 2)
        self.assertFalse(any(
            buff['definition_id'] == 'character_canhong_b_blaze_stack'
            for buff in fentian['applied_buffs']
        ))

    def test_loop_priming_preserves_reaction_buff_order_before_carrying_survivors(self) -> None:
        for axis_end_tick in (None, 50):
            with self.subTest(axis_end_tick=axis_end_tick):
                steps = [
                    {'id': 'canhong', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                    {'id': 'reaction', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 10},
                    {'id': 'fentian', 'slot': 0, 'action_id': 'action_canhong_q', 'start_tick': 30},
                ]
                if axis_end_tick is not None:
                    steps.append({
                        'id': 'axis-end',
                        'slot': 0,
                        'action_id': 'action_none_076a1f4e53',
                        'start_tick': axis_end_tick,
                    })
                result = simulate_shaft_axis({
                    'team': [
                        {'slot': 0, 'character_id': 'char_076a1f4e53', 'awakening_nodes': [2], 'arc_id': '', 'cartridge_id': ''},
                        {'slot': 1, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
                    ],
                    'steps': steps,
                    'initial_energy': 200,
                    'options': {
                        'loop_enabled': True,
                        'loop_initial_resources': {
                            'char_076a1f4e53': {'energy': 200, 'harmony': 100, 'personal_resources': {}},
                        },
                    },
                })['result']

                canhong = next(detail for detail in result['details'] if detail['step_id'] == 'canhong')
                reaction = next(detail for detail in result['details'] if detail['step_id'] == 'reaction')
                fentian = next(detail for detail in result['details'] if detail['step_id'] == 'fentian')

                self.assertFalse(any(
                    buff['definition_id'] == 'character_canhong_b_blaze_stack'
                    for buff in canhong['applied_buffs']
                ))
                blaze = next(
                    buff for buff in reaction['triggered_buffs']
                    if buff['definition_id'] == 'character_canhong_b_blaze_stack'
                )
                self.assertEqual(blaze['stack_count'], 1)
                self.assertEqual(fentian['formula_parts']['final_multiplier'], 2)
                self.assertFalse(any(
                    buff['definition_id'] == 'character_canhong_b_blaze_stack'
                    for buff in fentian['applied_buffs']
                ))

    def test_canhong_c_awakening_adds_interrupt_resistance_and_periodic_healing(self) -> None:
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'awakening_nodes': [3],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'attack', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                {'id': 'enter-illusion', 'slot': 0, 'action_id': 'action_canhong_e1', 'start_tick': 10},
                {'id': 'axis-end', 'slot': 0, 'action_id': 'action_none_076a1f4e53', 'start_tick': 45},
            ],
            'initial_energy': 200,
        })['result']

        self.assertEqual(result['details'][0]['panel']['interrupt_resistance'], 0.5)
        self.assertEqual([event['tick'] for event in result['healing_events']], [20, 30, 40])
        self.assertTrue(all(event['healing'] == 620.52 for event in result['healing_events']))

    def test_canhong_d_awakening_auto_activates_blood_banquet_at_full_energy(self) -> None:
        def warnings(nodes: list[int]) -> list[str]:
            result = simulate_shaft_axis({
                'team': [{
                    'slot': 0,
                    'character_id': 'char_076a1f4e53',
                    'awakening_nodes': nodes,
                    'arc_id': '',
                    'cartridge_id': '',
                }],
                'steps': [{'id': 'blood', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 0}],
                'initial_energy': 200,
            })['result']
            return result['details'][0]['warnings']

        self.assertIn('血宴尚未激活。', warnings([]))
        self.assertNotIn('血宴尚未激活。', warnings([4]))

    def test_canhong_d_awakening_blood_banquet_does_not_consume_blaze(self) -> None:
        result = simulate_shaft_axis({
            'team': [
                {'slot': 0, 'character_id': 'char_076a1f4e53', 'awakening_nodes': [2, 4], 'arc_id': '', 'cartridge_id': ''},
                {'slot': 1, 'character_id': 'char_c78f7a08d5', 'arc_id': '', 'cartridge_id': ''},
            ],
            'steps': [
                {'id': 'canhong', 'slot': 0, 'action_id': 'action_canhong_a1', 'start_tick': 0},
                {'id': 'reaction', 'slot': 1, 'action_id': 'action_2635f721a8', 'start_tick': 10},
                {'id': 'blood', 'slot': 0, 'action_id': 'action_canhong_q_blood_banquet', 'start_tick': 30},
                {'id': 'fentian', 'slot': 0, 'action_id': 'action_canhong_q', 'start_tick': 240},
            ],
            'initial_energy': 200,
        })['result']
        blood = next(detail for detail in result['details'] if detail['step_id'] == 'blood')
        fentian = next(detail for detail in result['details'] if detail['step_id'] == 'fentian')

        self.assertEqual(blood['formula_parts']['final_multiplier'], 2)
        self.assertEqual(fentian['formula_parts']['final_multiplier'], 2)

    def test_canhong_e_awakening_forces_stagger_only_once_per_target(self) -> None:
        payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'awakening_nodes': [5],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first', 'slot': 0, 'action_id': 'action_canhong_E1', 'start_tick': 0},
                {'id': 'second', 'slot': 0, 'action_id': 'action_canhong_E1', 'start_tick': 60},
            ],
            'initial_energy': 200,
        }
        result = simulate_shaft_axis(payload)['result']

        self.assertTrue(result['details'][0]['forced_stagger'])
        self.assertEqual(result['details'][0]['stagger_amount'], 50)
        self.assertFalse(result['details'][1]['forced_stagger'])

        loop_payload = deepcopy(payload)
        loop_payload['options'] = {'loop_enabled': True, 'loop_duration_ticks': 120}
        loop_result = simulate_shaft_axis(loop_payload)['result']
        self.assertFalse(any(detail['forced_stagger'] for detail in loop_result['details']))

    def test_canhong_f_awakening_scales_annihilation_last_hit_by_dot_layers(self) -> None:
        payload = {
            'team': [{
                'slot': 0,
                'character_id': 'char_076a1f4e53',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{'id': 'annihilation', 'slot': 0, 'action_id': 'action_canhong_E2', 'start_tick': 0}],
            'initial_energy': 200,
        }
        baseline = simulate_shaft_axis(payload)['result']['details'][0]
        awakened_payload = deepcopy(payload)
        awakened_payload['team'][0]['awakening_nodes'] = [6]
        awakened = simulate_shaft_axis(awakened_payload)['result']['details'][0]
        expected_bonus = 5 * 0.12 * (2.498 / 4.498)

        self.assertEqual(awakened['formula_parts']['active_dot_layer_count'], 5)
        self.assertAlmostEqual(awakened['formula_parts']['base_multiplier_factor'], 1 + expected_bonus)
        self.assertAlmostEqual(awakened['direct_damage'] / baseline['direct_damage'], 1 + expected_bonus)

    def test_canhong_missing_energy_and_stagger_values_use_documented_estimates(self) -> None:
        catalog = load_shaft_catalog()
        actions = {
            action['id']: action
            for action in catalog['actions']
            if action.get('character_id') == 'char_076a1f4e53'
        }
        expected = {
            'action_canhong_support': (6.5, 2.5),
            'action_canhong_a1': (2.4, 0.3),
            'action_canhong_a2': (2.1, 0.3),
            'action_canhong_a3': (3.3, 0.7),
            'action_canhong_a4': (4.2, 1.1),
            'action_canhong_a5': (4.8, 1.3),
            'action_canhong_z1': (4.5, 1.4),
            'action_canhong_illusion_a1': (1.2, 0.2),
            'action_canhong_illusion_a2': (2.8, 0.6),
            'action_canhong_illusion_a3': (3.0, 0.7),
            'action_canhong_illusion_a4': (4.8, 1.6),
            'action_canhong_illusion_z1': (4.2, 1.3),
            'action_canhong_plunge': (2.8, 0.7),
            'action_canhong_dodge_counter': (4.0, 2.5),
            'action_canhong_e1': (6.3, 1.4),
            'action_canhong_E1': (7.0, 2.0),
            'action_canhong_E2': (9.5, 6.0),
            'action_canhong_q': (0.0, 2.5),
            'action_canhong_q_enhanced': (0.0, 2.5),
            'action_canhong_q_blood_banquet': (0.0, 2.5),
        }

        self.assertEqual(set(expected), set(actions) - {'action_none_076a1f4e53', 'action_canhong_illusion_enter'})
        for action_id, (energy, stagger) in expected.items():
            with self.subTest(action_id=action_id):
                self.assertEqual(actions[action_id].get('energy_gain', 0), energy)
                self.assertEqual(actions[action_id].get('stagger', 0), stagger)
                note = actions[action_id].get('source_note', '')
                self.assertTrue('估值' in note or '用户口径' in note)

    def test_canhong_basic_attack_harmony_uses_documented_estimates(self) -> None:
        actions = {
            action['id']: action
            for action in load_shaft_catalog()['actions']
            if action.get('character_id') == 'char_076a1f4e53'
        }
        expected = {
            'action_canhong_a1': 2.5,
            'action_canhong_a2': 2.0,
            'action_canhong_a3': 4.0,
            'action_canhong_a4': 6.0,
            'action_canhong_a5': 7.0,
            'action_canhong_z1': 6.5,
            'action_canhong_illusion_a1': 1.5,
            'action_canhong_illusion_a2': 3.5,
            'action_canhong_illusion_a3': 4.0,
            'action_canhong_illusion_a4': 7.5,
            'action_canhong_illusion_z1': 6.5,
            'action_canhong_plunge': 4.6,
            'action_canhong_dodge_counter': 6.3,
        }

        for action_id, harmony in expected.items():
            with self.subTest(action_id=action_id):
                self.assertEqual(actions[action_id].get('harmony'), harmony)
                self.assertIn('经验估值', actions[action_id].get('source_note', ''))


if __name__ == '__main__':
    unittest.main()
