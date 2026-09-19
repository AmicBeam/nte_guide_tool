"""Automatic 未迟到的正义, Everness verified 2026-09-19."""
import unittest
import json
import subprocess
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis


class ChaosDelayTest(unittest.TestCase):
    def payload(self, hathor=False):
        ids = ['char_dd034941ef', 'char_d38b672525']
        if hathor:
            ids.append('char_912dbfe17c')
        support = next(a['id'] for a in load_shaft_catalog()['actions_by_character'][ids[1]] if a['name'] == '援护')
        return {'team': [dict(slot=i, character_id=id, arc_id='', cartridge_id='') for i, id in enumerate(ids)],
                'steps': [dict(id='start', slot=0, action_id='action_982c67944f', start_tick=0),
                          dict(id='support', slot=1, action_id=support, start_tick=30),
                          dict(id='end', slot=0, action_id='action_none_dd034941ef', start_tick=200)]}

    def events(self, result):
        return [e for e in result['periodic_damage_events'] if e.get('kind') == 'passive_settlement']

    def test_natural_expiry_uses_chaos_panel_and_duration(self):
        for hathor, duration, multiplier in ((False, 50, 8), (True, 120, 32)):
            with self.subTest(hathor=hathor):
                result = simulate_shaft_axis(self.payload(hathor))['result']
                events = self.events(result)
                self.assertEqual(len(events), 1)
                event = events[0]
                effect = next(e for e in result['reaction_effects'] if e['reaction'] == '延滞')
                self.assertEqual(event['tick'], effect['start_tick'] + duration)
                self.assertEqual(event['atk_multiplier'], multiplier)
                self.assertEqual(event['contributor_slot'], 1)
                self.assertEqual(event['damage_element'], '相')
                self.assertGreater(event['damage'], 0)
                formula = event['formula_parts']
                self.assertEqual(formula['skill_level_multiplier'], 1)
                self.assertAlmostEqual(formula['base'], event['panel']['atk'] * multiplier)
                self.assertAlmostEqual(event['damage'], formula['base'] * (1 + formula['damage_bonus'])
                                       * formula['crit'] * formula['defense'] * formula['resistance']
                                       * formula['final_multiplier'])
                self.assertAlmostEqual(result['summary']['character_damage'],
                                       sum(d['direct_damage'] for d in result['details']) + event['damage'])

    def test_reapplication_cancels_old_settlement(self):
        payload = self.payload(True)
        payload['steps'][2:2] = [dict(id='switch', slot=0, action_id='action_none_dd034941ef', start_tick=60),
                                dict(id='again', slot=1, action_id=payload['steps'][1]['action_id'], start_tick=80)]
        result = simulate_shaft_axis(payload)['result']
        events = self.events(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['tick'], result['reaction_effects'][-1]['end_tick'])
        self.assertEqual(events[0]['atk_multiplier'], 32)

    def test_no_chaos_no_settlement_and_retired_steps_removed(self):
        payload = self.payload(True)
        payload['team'] = [payload['team'][0], {**payload['team'][2], 'slot': 1}]
        payload['steps'][1]['action_id'] = 'action_f229587fd2'
        self.assertEqual(self.events(simulate_shaft_axis(payload)['result']), [])
        payload = self.payload(True)
        baseline = simulate_shaft_axis(payload)['result']
        for id in ('action_be26cd6a7c', 'action_82c5415d69'):
            payload['steps'].append(dict(id=id, slot=1, action_id=id, start_tick=190))
        normalized = normalize_axis_payload(payload)
        self.assertEqual(len(normalized['steps']), 3)
        result = simulate_shaft_axis(payload)['result']
        self.assertEqual(result['summary'], baseline['summary'])
        choices = load_shaft_catalog()['actions_by_character']['char_d38b672525']
        self.assertFalse(any(a['id'] in ('action_be26cd6a7c', 'action_82c5415d69') for a in choices))

    def test_loop_remaining_time_preserves_full_duration_multiplier(self):
        payload = self.payload(True)
        payload['steps'] = [dict(id='end', slot=0, action_id='action_none_dd034941ef', start_tick=100)]
        payload['options'] = {'loop_enabled': True, 'loop_initial_resources': {
            'char_dd034941ef': {'reaction': '延滞'},
        }}
        event, = self.events(simulate_shaft_axis(payload)['result'])
        self.assertEqual(event['tick'], 20)
        self.assertEqual(event['delay_duration_ticks'], 120)
        self.assertEqual(event['atk_multiplier'], 32)

    def test_projection_after_axis_end_does_not_inflate_total(self):
        payload = self.payload(True)
        payload['steps'][-1]['start_tick'] = 60
        result = simulate_shaft_axis(payload)['result']
        self.assertEqual(len(self.events(result)), 1)
        self.assertAlmostEqual(result['summary']['character_damage'], sum(d['direct_damage'] for d in result['details']))

    def test_triangle_tooltip_uses_settlement_snapshot(self):
        event, = self.events(simulate_shaft_axis(self.payload(True))['result'])
        source = (Path(__file__).resolve().parents[1] / 'app/modules/shaft/static/js/shaft.js').read_text()
        functions = source[source.index('  function damageMarkerTooltip('):source.index('  function renderTimeline(')]
        script = 'const formatNumber = n => String(n); const ticksToSeconds = n => n / 10; const memberName = () => "卡厄斯";\n'
        script += functions + '\nconst event = ' + json.dumps(event, ensure_ascii=False) + ';'
        script += 'console.log(JSON.stringify({groups: groupedTimelineDamageEvents([event]), tooltip: damageMarkerTooltip(event)}));'
        rendered = json.loads(subprocess.check_output(['node', '-e', script], text=True))
        self.assertEqual(len(rendered['groups']), 1)
        self.assertEqual(rendered['groups'][0]['contributor_slot'], 1)
        self.assertIn('延滞持续 12s', rendered['tooltip'])
        self.assertIn('倍率 3200%', rendered['tooltip'])
        self.assertNotIn('层数', rendered['tooltip'])

    def test_zero_awakening_license_increases_damage_and_expires(self):
        payload = {'team': [dict(slot=0, character_id='char_d38b672525', arc_id='', cartridge_id='', awakening_nodes=[])],
                   'steps': [dict(id='e', slot=0, action_id='action_2d3f8642dd', start_tick=0),
                             dict(id='inside', slot=0, action_id='action_40d0576f80', start_tick=30),
                             dict(id='expired', slot=0, action_id='action_40d0576f80', start_tick=180)]}
        result = simulate_shaft_axis(payload)['result']
        details = {d['step_id']: d for d in result['details']}
        inside, expired = details['inside'], details['expired']
        buff, = [b for b in inside['applied_buffs'] if b['rule_id'] == 'character_chaos_pursuit_license']
        self.assertEqual(buff['effects'], {'all_dmg': .2})
        self.assertEqual(buff['duration_ticks'], 150)
        self.assertNotIn('character_chaos_pursuit_license', {b['rule_id'] for b in expired['applied_buffs']})
        self.assertAlmostEqual(inside['formula_parts']['dmg_bonus'] - expired['formula_parts']['dmg_bonus'], .2)
        self.assertAlmostEqual(inside['direct_damage'] / expired['direct_damage'],
                               (1 + inside['formula_parts']['dmg_bonus']) / (1 + expired['formula_parts']['dmg_bonus']))
        payload['team'][0]['awakening_nodes'] = [5]
        awakened = simulate_shaft_axis(payload)['result']['details'][1]
        self.assertAlmostEqual(awakened['formula_parts']['dmg_bonus'] - inside['formula_parts']['dmg_bonus'], .2)
