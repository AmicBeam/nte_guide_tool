"""User 2026-09-18: teammate assists borrow Lingke joint panel, retain identity/effects."""
import unittest
from copy import deepcopy

from app.modules.shaft.service import simulate_shaft_axis
from app.modules.shaft.domain.catalog import load_shaft_catalog


class LingkeSupportTest(unittest.TestCase):
    def axis(self, nodes=(), window='e', tick=None, variant=False):
        catalog = load_shaft_catalog()
        lingke = next(c for c in catalog['characters'] if c['name'] == '灵可')
        heiyu = next(c for c in catalog['characters'] if c['name'] == '黑羽')
        assist = next(a for a in catalog['actions_by_character'][heiyu['id']]
                      if a['name'] == ('魔援护' if variant else '援护'))
        steps = [{'id': 'opener', 'slot': 0, 'action_id': 'action_lingke_'+window, 'start_tick': 0}]
        steps.append({'id': 'assist', 'slot': 1, 'action_id': assist['id'],
                      'start_tick': tick if tick is not None else (10 if window == 'e' else 5)})
        if window == 'q':
            steps.append({'id': 'end', 'slot': 0, 'action_id': 'action_lingke_q_multi', 'start_tick': 30})
        return {'team': [
            {'slot': 0, 'character_id': lingke['id'], 'arc_id': '', 'cartridge_id': '',
             'awakening_nodes': list(nodes), 'skill_levels': {'support': 3}},
            {'slot': 1, 'character_id': heiyu['id'], 'arc_id': '', 'cartridge_id': '',
             'skill_levels': {'support': 9}},
        ], 'steps': steps, 'initial_energy': 0}

    def run_axis(self, axis):
        result = simulate_shaft_axis(axis)['result']
        return result, {d['step_id']: d for d in result['details']}

    def test_e_window_keeps_identity_resources_and_uses_lingke_panel(self):
        axis = self.axis()
        result, details = self.run_axis(axis)
        d = details['assist']
        self.assertEqual(d['duration_ticks'], 0)
        self.assertEqual(d['lingke_joint_source_slot'], 0)
        self.assertFalse(d['triggered_reaction'])
        self.assertEqual(d['energy_gain'], 0)
        self.assertEqual(d['formula_parts']['skill_level'], 3)
        self.assertAlmostEqual(d['panel']['atk'], details['opener']['panel']['atk'])
        self.assertEqual(d['personal_resources_after']['恶意'], 40)
        self.assertEqual(d['slot'], 1)
        reference = deepcopy(axis)
        reference['steps'][1].update(slot=0, action_id='action_lingke_joint_soul')
        expected = self.run_axis(reference)[1]['assist']
        self.assertEqual(d['panel'], expected['panel'])
        self.assertAlmostEqual(d['direct_damage'], expected['direct_damage'])
        self.assertIn('character_lingke_precise_tuning_soul', {b['rule_id'] for b in d['triggered_buffs']})
        increased = deepcopy(axis)
        increased['team'][1]['substat_counts'] = {'atk_pct': 100}
        self.assertAlmostEqual(self.run_axis(increased)[1]['assist']['direct_damage'], d['direct_damage'])
        increased['team'][0]['substat_counts'] = {'atk_pct': 100}
        self.assertGreater(self.run_axis(increased)[1]['assist']['direct_damage'], d['direct_damage'])

    def test_e_boundary_and_q_closing_restore_normal_assist(self):
        _, outside = self.run_axis(self.axis(tick=11))
        self.assertEqual(outside['assist']['duration_ticks'], 15)
        self.assertGreater(outside['assist']['energy_gain'], 0)
        for tick, expected in ((5, 0), (20, 0), (35, 15)):
            with self.subTest(tick=tick):
                _, details = self.run_axis(self.axis(window='q', tick=tick))
                self.assertEqual(details['assist']['duration_ticks'], expected)
                if expected == 0:
                    self.assertFalse(details['assist']['triggered_reaction'])
                    self.assertEqual(details['assist']['energy_gain'], 0)

    def test_f_awakening_preserves_variant_energy_and_extra_damage(self):
        for variant, energy in ((False, 6.3), (True, 5)):
            with self.subTest(variant=variant):
                _, base = self.run_axis(self.axis(window='q', variant=variant))
                _, awakened = self.run_axis(self.axis(nodes=[6], window='q', variant=variant))
                d = awakened['assist']
                self.assertEqual(d['duration_ticks'], 0)
                self.assertAlmostEqual(d['energy_gain'], energy)
                self.assertAlmostEqual(d['panel']['crit_rate']-base['assist']['panel']['crit_rate'], .25)
                if variant:
                    self.assertAlmostEqual(d['personal_resources_after']['恶意'], 16.5)
                    self.assertGreater(d['resource_damage'], 0)

    def test_e_awakening_keeps_exception_but_q_does_not_trigger_reaction(self):
        catalog = load_shaft_catalog()
        curse = next(c for c in catalog['characters'] if c['name'] == '白藏')
        assist = next(a for a in catalog['actions_by_character'][curse['id']] if a['name'] == '援护')
        for window, expected in (('e', True), ('q', False)):
            axis = self.axis(nodes=[5], window=window)
            axis['team'][1]['character_id'] = curse['id']
            axis['steps'][1]['action_id'] = assist['id']
            _, details = self.run_axis(axis)
            self.assertEqual(bool(details['assist']['triggered_reaction']), expected)
            self.assertFalse(any('环合值不足' in w for w in details['assist']['warnings']))

    def test_consecutive_q_assists_are_zero_and_stop_after_q_multi(self):
        axis = self.axis(window='q')
        assist = deepcopy(axis['steps'][1])
        assist.update(id='second', start_tick=10)
        axis['steps'].append(assist)
        _, details = self.run_axis(axis)
        for key in ('assist', 'second'):
            self.assertEqual(details[key]['duration_ticks'], 0)
            self.assertEqual(details[key]['start_tick'], 0)

    def test_editor_insertion_reserves_zero_time_visual_span(self):
        import json
        import subprocess
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        source = (root / 'app/modules/shaft/static/js/shaft.js').read_text()
        resolver = source[source.index("  let jointActionCacheKey"):source.index('  function stepStartTick(')]
        insertion = source[source.index('  function reserveTimeStopZeroInsertionSpan('):source.index('  function addStep()')]
        axis = self.axis(window='q')
        candidate = axis['steps'].pop(1)
        script = '''
const window = {ShaftEngine: require('./app/modules/shaft/static/js/shaft_engine.js')};
const state = {axis: AXIS};
const actions = new Map(ACTIONS.map(a => [a.id, a]));
const getActionMap = () => actions;
const pushUndoSnapshot = () => {};
const prepareInsertionTick = t => t;
const isTimeStopZeroForegroundStep = (s,a) => a.is_time_stop_zero && a.duration_ticks === 0;
const actionVisualDurationTicks = a => a.duration_ticks || 5;
const normalizeEditedSteps = () => {};
const selectStep = () => {};
const syncAddTimeInput = () => {};
const closeContextMenu = () => {};
const scheduleSimulation = () => {};
const revealTimelineTick = () => {};
''' .replace('AXIS', json.dumps(axis)).replace('ACTIONS', json.dumps(load_shaft_catalog()['actions']))
        script += resolver + insertion
        script += f"addActionAt(1, {json.dumps(candidate['action_id'])}, 5);"
        script += "console.log(JSON.stringify({cursor:state.cursorTick,end:state.axis.steps.find(s=>s.id==='end').start_tick,duration:actionForStep(state.axis.steps.at(-1)).duration_ticks}));"
        value = json.loads(subprocess.check_output(['node', '-e', script], cwd=root, text=True))
        self.assertEqual(value, {'cursor': 10, 'end': 35, 'duration': 0})
