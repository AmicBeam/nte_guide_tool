import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import simulate_shaft_axis

ROOT = Path(__file__).resolve().parents[1]


class LoopPassiveReactionTest(unittest.TestCase):
    def axis(self, form='e', initial=True, elyms=False, active=False, duplicate=False):
        team = [dict(slot=0, character_id='char_akane', arc_id='', cartridge_id='')]
        steps = [dict(id='e', slot=0, action_id='action_akane_' + form, start_tick=0),
                 dict(id='tail', slot=0, action_id='action_none_akane', start_tick=180)]
        resources = {'char_akane': {'reaction': '延滞'}} if initial else {}
        if elyms:
            team.append(dict(slot=len(team), character_id='char_elyms', arc_id='', cartridge_id=''))
        if active or duplicate:
            slot = len(team)
            team.append(dict(slot=slot, character_id='char_dd034941ef', arc_id='', cartridge_id=''))
            if active:
                steps.insert(1, dict(id='active', slot=slot, action_id='action_33c0ac631e', start_tick=30))
            if duplicate:
                resources['char_dd034941ef'] = {'reaction': '延滞'}
        return dict(team=team, steps=steps, options=dict(loop_enabled=True, loop_initial_resources=resources))

    def check(self, axis):
        result = simulate_shaft_axis(axis)['result']
        script = "const s=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.inspectAxis(p.axis,p.catalog,p.result)));"
        warnings = json.loads(subprocess.check_output(['node', '-e', script,
            str(ROOT/'app/modules/shaft/static/js/shaft_self_check.js')],
            input=json.dumps(dict(axis=axis, catalog=load_shaft_catalog(), result=result)), text=True))
        return result, [w for w in warnings if '自带环合「延滞」冲突' in w]

    def test_own_initial_delay_unlocks_passive_without_a_conflict(self):
        for form in ('e', 'hold_e'):
            with self.subTest(form=form):
                result, warnings = self.check(self.axis(form=form))
                pulses = [e for e in result['reaction_effects'] if e.get('is_reapplication')]
                self.assertTrue(pulses)
                self.assertTrue(all(e['contributor_character_id'] == 'char_akane' for e in pulses))
                self.assertEqual(warnings, [])

    def test_no_initial_or_active_delay_does_not_unlock_the_passive(self):
        result, warnings = self.check(self.axis(initial=False))
        self.assertFalse(any(e['reaction'] == '延滞' for e in result['reaction_effects']))
        self.assertEqual(warnings, [])

    def test_active_delay_without_initial_delay_unlocks_the_passive(self):
        result, warnings = self.check(self.axis(initial=False, active=True))
        self.assertTrue(any(e.get('is_reapplication') for e in result['reaction_effects']))
        self.assertEqual(warnings, [])

    def test_another_carriers_initial_delay_does_not_hide_a_new_source(self):
        axis = self.axis(duplicate=True)
        del axis['options']['loop_initial_resources']['char_akane']
        result, warnings = self.check(axis)
        self.assertTrue(any(e.get('is_reapplication') for e in result['reaction_effects']))
        self.assertTrue(warnings)

    def test_derived_echo_is_not_a_second_independent_source(self):
        result, warnings = self.check(self.axis(elyms=True))
        self.assertTrue(any(e.get('is_echo') for e in result['reaction_effects']))
        self.assertTrue(any(e.get('is_reapplication') for e in result['reaction_effects']))
        self.assertEqual(warnings, [])

    def test_real_active_trigger_and_duplicate_initial_carriers_still_conflict(self):
        for options in (dict(active=True), dict(duplicate=True), dict(active=True, elyms=True)):
            with self.subTest(options=options):
                result, warnings = self.check(self.axis(**options))
                self.assertTrue(warnings)
                if options.get('active'):
                    self.assertTrue(any((d.get('triggered_reaction') or {}).get('reaction') == '延滞'
                                        for d in result['details']))
