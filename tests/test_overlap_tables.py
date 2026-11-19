import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('overlap_generator', ROOT / 'app/modules/overlap/generator.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class OverlapTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((generator.MODULE / 'data/rules.json').read_text())
        cls.characters = json.loads((generator.SOURCE / 'characters.json').read_text())
        cls.actions = json.loads((generator.SOURCE / 'actions.json').read_text())
        cls.payload = generator.generate(cls.characters, cls.actions, cls.rules)

    def character(self, name):
        return next(c for c in self.payload['characters'] if c['name'] == name)

    def test_recovery_catalog_defaults_and_sources(self):
        configured = {(c['name'], a['name']): a['recovery_seconds']
                      for c in self.payload['characters'] for a in c['actions'] if a['recovery_seconds']}
        self.assertEqual(configured, {('安魂曲', 'a4远'): .5, ('黑羽', '魔a3'): .5, ('Exe','a4'):.5})
        self.assertFalse(any('linger_seconds' in a for c in self.payload['characters'] for a in c['actions']))

    def test_recovery_slots_delay_continuation_without_extra_benefits(self):
        for character, names, operation, index, base_duration, next_start in [
            ('黑羽', [f'魔a{i}' for i in range(1, 6)], '魔a2预魔a3 / 魔a4预魔a5', 2, 1.0, 2.3),
            ('安魂曲', [f'a{i}远' for i in range(1, 6)], 'a3预a4 / a5打满', 3, 1.2, 2.7),
        ]:
            rows = [r for r in self.character(character)['rows'] if r['original_actions'] == names
                    and r['operation'] == operation and not r['has_dodge']]
            self.assertEqual(len(rows), 6)
            for row, extra in zip(rows, (0, .1, .2, .3, .4, .5)):
                step, following = row['timeline'][index:index + 2]
                self.assertEqual(step['recovery_seconds'], .5)
                self.assertEqual(step['recovery_used_seconds'], extra)
                self.assertEqual(step['base_duration_seconds'], base_duration)
                self.assertAlmostEqual(step['duration_seconds'], base_duration + extra)
                self.assertAlmostEqual(following['start_seconds'], next_start + extra)
                self.assertEqual((step['placement'], following['placement']), ('background', 'foreground'))
                for field in ('multipliers', 'energy_gain', 'energy_return', 'energy_cost', 'harmony',
                              'harmony_on_start', 'personal_resources', 'stagger'):
                    self.assertEqual(row[field], rows[0][field])
            self.assertEqual(len({r['id'] for r in rows}), 6)

    def test_recovery_wait_only_on_background_to_foreground_not_dodge_or_tail(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                for index, step in enumerate(row['timeline']):
                    if step['recovery_used_seconds']:
                        self.assertFalse(step['dodge_replaces'])
                        self.assertEqual(step['placement'], 'background')
                        self.assertEqual(row['timeline'][index + 1]['placement'], 'foreground')
                        self.assertLessEqual(step['recovery_used_seconds'], step['recovery_seconds'])
        self.assertFalse(any(r['operation'] == '魔a3正常切人 / 魔a4预魔a5'
                             for r in self.character('黑羽')['rows']))

    def test_new_recovery_action_can_enable_return_without_character_branches(self):
        rules = copy.deepcopy(self.rules)
        rules['characters']['明音凛'].setdefault('actions', {})['a3'] = {
            'recovery_seconds': .3, 'source': 'synthetic recovery configuration'}
        payload = generator.generate(self.characters, self.actions, rules)
        rows = [r for c in payload['characters'] if c['name'] == '明音凛' for r in c['rows']
                if r['original_actions'] == ['a1','a2','a3','a4']
                and r['operation'] == 'a2预a3 / a4打满']
        self.assertEqual([r['timeline'][2]['recovery_used_seconds'] for r in rows], [.1,.2,.3])
        self.assertEqual(rows[0]['window_1']['duration_seconds'], 1.2)

    def test_scope_and_no_melee_requiem(self):
        self.assertEqual([c['name'] for c in self.payload['characters']], ['安魂曲','残虹','黑羽','明音凛','曳尔沐','Exe'])
        self.assertEqual({c['name'] for c in self.payload['characters']}, {'安魂曲', '残虹', '黑羽', '明音凛', '曳尔沐', 'Exe'})
        self.assertEqual({a['name'] for a in self.character('安魂曲')['actions']}, {*(f'a{i}远' for i in range(1, 6)), 'e'})

    def test_exe_basic_preinputs_recovery_and_full_damage(self):
        c=self.character('Exe')
        self.assertEqual(c['start_actions'],['a1'])
        self.assertFalse(c['dodge_preserves_combo'])
        self.assertEqual([a['name']for a in c['actions']],['a1','a2','a3','a4','a5','z'])
        self.assertEqual([a['duration_seconds']for a in c['actions']],[.6,.6,.7,1.3,.8,.7])
        self.assertEqual([a['recovery_seconds']for a in c['actions']],[0,0,0,.5,0,0])
        self.assertEqual([a['successors'][0]['can_preinput']for a in c['actions'][:5]],[True,False,True,False,False])
        self.assertTrue(all(not r['has_dodge'] and r['original_actions'][0]=='a1'for r in c['rows']))
        self.assertEqual({i for r in c['rows']for i in r['preinput_action_indices']},{0,2})
        full=next(r for r in c['rows']if r['switch_method']=='full'and r['actions']==['a1','a2','a3','a4','a5'])
        self.assertEqual(full['release_end_seconds'],4)
        self.assertAlmostEqual(full['multipliers']['atk'],4.168584)
        self.assertFalse(full['personal_resources'])
        rows=[r for r in c['rows']if r['operation']=='a3预a4 / a5打满']
        self.assertEqual([r['timeline'][3]['recovery_used_seconds']for r in rows],[0,.1,.2,.3,.4,.5])
        for row,extra in zip(rows,[0,.1,.2,.3,.4,.5]):
            self.assertEqual(row['foreground_seconds'],2.4)
            self.assertAlmostEqual(row['timeline'][4]['start_seconds'],3.2+extra)
            self.assertAlmostEqual(row['release_end_seconds'],4+extra)
            self.assertEqual(row['multipliers'],full['multipliers'])
            self.assertEqual(row['harmony'],full['harmony'])
            self.assertEqual(row['energy_gain'],full['energy_gain'])
            self.assertEqual(row['stagger'],full['stagger'])
        self.assertTrue(all(r['original_actions']==r['actions']for r in c['rows']))

    def test_exe_z_resets_chain_and_never_leaves_foreground(self):
        c=self.character('Exe');z=next(a for a in c['actions']if a['name']=='z')
        self.assertTrue(z['combo_reset'])
        self.assertFalse(z['successors']);self.assertFalse(z['switch_points']);self.assertFalse(z['gaps'])
        self.assertEqual({a['name']for a in c['actions']if any(e['to']=='z'for e in a['successors'])},{'a1','a2','a3','a4','a5'})
        rows=[r for r in c['rows']if 'z'in r['actions']]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row['actions'][-1],'z')
            step=row['timeline'][-1]
            self.assertEqual(step['placement'],'foreground')
            self.assertEqual(step['input_seconds'],step['start_seconds'])
            self.assertAlmostEqual(step['end_seconds']-step['start_seconds'],.7)
            self.assertFalse(row['ends_with_switch'])
            self.assertTrue(all(row['actions'][i+1]!='z'for i in row['preinput_action_indices']))
        self.assertTrue(all(any(r['original_actions']==[*(f'a{j}'for j in range(1,i+1)),'z']for r in rows)for i in range(1,6)))

    def test_elyms_a2_preinput_a3_starts_from_a1_and_preserves_whole_chain(self):
        character=self.character('曳尔沐')
        self.assertEqual(character['start_actions'],['强a1'])
        self.assertEqual([a['duration_seconds'] for a in character['actions']],[.6,.8,.6])
        row=next(r for r in character['rows'] if r['switch_method']=='preinput')
        self.assertEqual(row['actions'],['强a1','强a2','强a3'])
        self.assertEqual(row['operation'],'强a2预强a3')
        self.assertEqual(row['foreground_seconds'],1)
        self.assertEqual(row['release_end_seconds'],2)
        self.assertEqual(row['background_seconds'],1)
        self.assertEqual(row['timeline'][2]['input_seconds'],1)
        self.assertEqual(row['timeline'][2]['start_seconds'],1.4)
        self.assertEqual(row['preinput_action_indices'],[1])
        self.assertAlmostEqual(row['multipliers']['atk'],.259+.376+.788)
        self.assertTrue(all(r['original_actions'][0]=='强a1' for r in character['rows']))
        self.assertFalse(any(r['has_dodge'] or r.get('overlap_windows') for r in character['rows']))
        self.assertEqual(len([r for r in character['rows'] if r['ends_with_switch']]),1)

    def test_normal_switch_preserves_full_requiem_benefits(self):
        row = next(r for r in self.character('安魂曲')['rows'] if r['original_actions'] == ['a1远', 'a2远', 'a3远', 'a4远'] and r['switch_method'] == 'normal')
        self.assertAlmostEqual(row['foreground_seconds'], 1.7)
        self.assertAlmostEqual(row['release_end_seconds'], 2.7)
        self.assertAlmostEqual(row['background_seconds'], 1.0)
        self.assertAlmostEqual(row['multipliers']['atk'], 4.401)
        self.assertAlmostEqual(row['timeline'][-1]['input_seconds'], 1.2)
        self.assertAlmostEqual(row['timeline'][-1]['start_seconds'], 1.5)
        self.assertTrue(all(not step['preinput_before_switch'] for step in row['timeline']))
        pre = next(r for r in self.character('安魂曲')['rows'] if r['original_actions']==['a1远','a2远','a3远','a4远'] and r['switch_method']=='preinput')
        self.assertAlmostEqual(pre['foreground_seconds'],1.2)
        self.assertEqual(pre['operation'],'a3预a4')
        self.assertEqual(pre['multipliers'],row['multipliers'])

    def test_requiem_normal_e_and_detached_a4_connect_to_a5(self):
        character = self.character('安魂曲')
        self.assertEqual(character['start_actions'],['a1远','e'])
        simple = next(r for r in character['rows'] if r['original_actions']==['e','a5远'] and r['switch_method']=='normal')
        self.assertEqual(simple['actions'],['e','a5远'])
        self.assertAlmostEqual(simple['foreground_seconds'],1.0)
        self.assertAlmostEqual(simple['release_end_seconds'],2.8)
        self.assertEqual(simple['timeline'][-1]['input_seconds'],simple['timeline'][0]['end_seconds'])
        names = ['a1远','a2远','a3远','a4远','e','a5远']
        row = next(r for r in character['rows'] if r['original_actions']==names and r['switch_method']=='normal' and r['kind']=='合轴')
        a4, e, a5 = row['timeline'][-3:]
        self.assertTrue(a4['detached'])
        self.assertAlmostEqual(a4['duration_seconds'],0.5)
        self.assertAlmostEqual(e['start_seconds'],2.0)
        self.assertEqual(e['input_seconds'],a4['end_seconds'])
        self.assertFalse(e['preinput'])
        self.assertAlmostEqual(row['foreground_seconds'],3.0)
        self.assertAlmostEqual(a5['start_seconds'],2.8)
        originals = [a for a in self.actions if a['character_name']=='安魂曲' and a['name'] in names]
        self.assertAlmostEqual(row['multipliers']['atk'],sum(a['multipliers']['atk'] for a in originals))
        self.assertAlmostEqual(row['energy_gain'],sum(a.get('energy_gain',0) for a in originals))
        self.assertAlmostEqual(row['harmony'],sum(a.get('harmony',0) for a in originals))
        e_action = next(a for a in character['actions'] if a['name']=='e')
        self.assertFalse(any(p['id']=='detached' for p in e_action['switch_points']))
        self.assertFalse(any(step['detached'] for r in character['rows'] for step in r['timeline'] if step['action']=='e'))

    def test_single_action_routes_are_excluded_but_action_inventory_remains(self):
        for character in self.payload['characters']:
            self.assertTrue(character['rows'])
            self.assertTrue(all(len(row['actions'])>=2 for row in character['rows']))
            self.assertEqual(character['coverage']['rows'],len(character['rows']))
            self.assertTrue(all(any(a['name']==name for a in character['actions']) for name in character['start_actions']))

    def test_preinput_now_uses_point_four_seconds(self):
        row = next(r for r in self.character('残虹')['rows'] if r['original_actions']==['a1','a2','a3','a4'] and r['switch_method']=='preinput')
        self.assertEqual(row['operation'], 'a3预a4')
        self.assertAlmostEqual(row['foreground_seconds'], 1.6)
        self.assertEqual([s['preinput_before_switch'] for s in row['timeline']], [False,False,True,False])

    def test_normal_switch_supports_akane_intermediate_dodge_route(self):
        row = next(r for r in self.character('明音凛')['rows'] if r['original_actions']==['a1','a2','a3','a4'] and r['has_dodge'] and r['switch_method']=='normal')
        self.assertEqual(row['actions'], ['a1','a2','a3闪','a4'])
        self.assertAlmostEqual(row['foreground_seconds'],1.9)
        self.assertAlmostEqual(row['release_end_seconds'],2.5)
        self.assertFalse(row['timeline'][2]['preinput'])
        self.assertEqual(row['timeline'][3]['input_seconds'], row['timeline'][2]['end_seconds'])
        self.assertEqual(row['overlap_windows'],[])
        self.assertEqual(row['foreground_segments'],[{'number':1,'start_seconds':0,'end_seconds':1.9,'duration_seconds':1.9}])
        self.assertTrue(row['timeline'][-1]['terminal_switch'])

    def test_terminal_switch_marker_only_marks_active_last_action(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                marked=[i for i,step in enumerate(row['timeline']) if step['terminal_switch']]
                self.assertEqual(marked,[len(row['timeline'])-1] if row['switch_method'] in {'normal','detached'} else [])

    def test_hold_candidates_follow_terminal_or_cycle_rule(self):
        c=self.character('明音凛')
        held=next(r for r in c['rows'] if r['switch_method']=='hold' and r['original_actions']==['a1','a2','a3','a4'] and not r['has_dodge'] and r['kind']=='打满')
        self.assertEqual(held['operation'],'a4打满')
        self.assertEqual(held['hold_reason'],'cycle_to_start')
        self.assertAlmostEqual(held['foreground_seconds'],2.8)
        self.assertIsNone(held['switch_seconds'])
        self.assertFalse(held['ends_with_switch'])
        self.assertFalse(held['timeline'][-1]['terminal_switch'])
        self.assertEqual(held['timeline'][-1]['action'],'a4')
        self.assertFalse(any(r['switch_method']=='hold' and r['original_actions']==['a1','a2'] for r in c['rows']))
        cycle=next(r for r in self.character('黑羽')['rows'] if r['switch_method']=='hold'
                   and r['original_actions'][0]=='魔e' and r['original_actions'][-1]=='魔a5' and r['kind']=='打满' and not r['has_dodge'])
        self.assertEqual(cycle['hold_reason'],'cycle_to_start')
        self.assertIn('魔e',cycle['operation'])
        self.assertTrue(cycle['operation'].endswith('魔a5打满'))

    def test_basic_terminal_restarts_same_form_without_preinput(self):
        expected = {'安魂曲': [('a5远','a1远')], '残虹': [('a5','a1'),('幻a4','幻a1')],
                    '黑羽': [('魔a5','魔a1')], '明音凛': [('a4','a1')]}
        for name, pairs in expected.items():
            character = self.character(name)
            for terminal, opener in pairs:
                action = next(a for a in character['actions'] if a['name'] == terminal)
                restart = next(e for e in action['successors'] if e['to'] == opener)
                self.assertFalse(restart['can_preinput'])
                self.assertTrue(restart['basic_chain_restart'])
                self.assertTrue(all(r['original_actions'].count(opener) <= 1 for r in character['rows']))
                self.assertFalse(any((terminal, opener) in list(zip(r['original_actions'],r['original_actions'][1:]))
                                     for r in character['rows']))
        for kind in ('打满','分段合轴'):
            held = next(r for r in self.character('黑羽')['rows'] if r['switch_method']=='hold'
                        and r['original_actions']==[f'魔a{i}' for i in range(1,6)]
                        and r['kind']==kind and not r['has_dodge'])
            self.assertEqual(held['hold_reason'],'cycle_to_start')
            self.assertTrue(held['operation'].endswith('魔a5打满'))
        self.assertEqual(generator.hold_eligibility(['a1','a2'], {'a1': [], 'a2': []}), 'no_successor')

    def test_hold_finish_is_supported_after_return(self):
        row=next(r for r in self.character('黑羽')['rows'] if r['switch_method']=='hold'
                 and r['kind']=='分段合轴' and not r['has_dodge']
                 and r['original_actions']==[f'魔a{i}' for i in range(1,6)])
        self.assertEqual(row['operation'],'魔a2预魔a3 / 魔a4 → 魔a5打满')
        self.assertEqual([s['duration_seconds'] for s in row['foreground_segments']],[1.1,1.3])
        self.assertAlmostEqual(row['foreground_seconds'],2.4)
        self.assertFalse(row['timeline'][-1]['terminal_switch'])

    def test_normal_final_switch_can_follow_a_return(self):
        row = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==[f'魔a{i}' for i in range(1,6)]
                   and r['kind']=='分段合轴' and r['switch_method']=='normal' and not r['has_dodge'])
        self.assertEqual(row['operation'],'魔a2预魔a3 / 魔a4 → 魔a5正常切人')
        self.assertEqual(row['window_1']['duration_seconds'],1.2)
        self.assertEqual(row['window_1']['return_start_seconds'],2.3)
        self.assertEqual(row['window_1']['end_seconds'],2.3)
        self.assertAlmostEqual(row['foreground_seconds'],1.9)
        self.assertEqual([s['preinput_before_switch'] for s in row['timeline']],[False,True,False,False,False])

    def test_updated_akane_window_is_too_short_to_return(self):
        row = next(r for r in self.character('明音凛')['rows'] if r['original_actions']==['a1','a2','a3']
                   and r['switch_method']=='preinput')
        self.assertAlmostEqual(row['foreground_seconds'],0.9)
        self.assertAlmostEqual(row['release_end_seconds'],2.0)
        self.assertAlmostEqual(row['release_end_seconds']-row['foreground_seconds'],1.1)
        self.assertFalse(any(r['kind']=='分段合轴' for r in self.character('明音凛')['rows']))

    def test_return_preinput_cannot_be_issued_from_background_tail(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                self.assertTrue(all(segment['duration_seconds']>0 for segment in row['foreground_segments']))
                self.assertTrue(all(step['placement']=='foreground' for step in row['timeline'] if step['preinput_before_switch']))
        self.assertFalse(any(r['operation']=='魔a3预魔a4 / 魔a4预魔a5' for r in self.character('黑羽')['rows']))

    def test_dodge_requires_foreground_and_full_execution(self):
        for c in self.payload['characters']:
            for row in c['rows']:
                for index, step in enumerate(row['timeline']):
                    if not step['dodge_replaces']:
                        continue
                    self.assertEqual(step['placement'],'foreground')
                    self.assertTrue(any(left<=step['start_seconds']+1e-9 and right>=step['end_seconds']-1e-9
                                        for left,right in row['foreground_intervals']))
                    self.assertEqual(step['input_seconds'],step['start_seconds'])
                    self.assertFalse(step['preinput_before_switch'])
                    if index+1<len(row['timeline']):
                        self.assertEqual(row['timeline'][index+1]['input_seconds'],step['end_seconds'])

    def test_only_legal_start_actions_generate_routes(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                self.assertIn(row['original_actions'][0], character['start_actions'])
        self.assertFalse(any(r['original_actions']==['a4远'] for r in self.character('安魂曲')['rows']))

    def test_heiyu_e_cannot_be_preinput_but_can_switch_normally(self):
        for row in self.character('黑羽')['rows']:
            self.assertNotIn('魔e预', row['operation'])
            self.assertNotIn('预魔e', row['operation'])
            for index, step in enumerate(row['timeline']):
                if step['action'] != '魔e':
                    continue
                self.assertFalse(step['preinput'])
                self.assertEqual(step['input_seconds'], step['start_seconds'])
                required_end = step['start_seconds']+0.2 if index==len(row['timeline'])-1 and row['switch_method']=='normal' else step['end_seconds']
                self.assertTrue(any(left <= step['start_seconds']+1e-9 and right >= required_end-1e-9
                                    for left,right in row['foreground_intervals']))
                if index+1 < len(row['timeline']):
                    following=row['timeline'][index+1]
                    self.assertEqual(following['action'], '魔z2')
                    self.assertEqual(following['input_seconds'], step['end_seconds'])
                    self.assertEqual(following['start_seconds'], step['end_seconds'])
                    self.assertFalse(following['preinput'])

    def test_only_configured_actions_allow_normal_switch(self):
        for character in self.payload['characters']:
            configured=set(self.rules['characters'][character['name']].get('normal_switch_actions',[]))
            for action in character['actions']:
                if action['name'] not in configured:
                    self.assertFalse(any(p['id']=='normal' for p in action['switch_points']))
                    continue
                normal = next(p for p in action['switch_points'] if p['id']=='normal')
                self.assertEqual(normal['after_seconds'],0.2)
                self.assertFalse(action['locks_foreground_switch'])
        names = [f'魔a{i}' for i in range(1,6)] + ['魔e']
        rows = [r for r in self.character('黑羽')['rows'] if r['original_actions']==names
                and r['switch_method']=='normal' and not r['has_dodge']]
        self.assertTrue(any(r['kind']=='合轴' for r in rows))
        self.assertTrue(any(r['kind']=='分段合轴' for r in rows))
        for row in rows:
            self.assertAlmostEqual(row['switch_seconds']-row['timeline'][-1]['start_seconds'],0.2)
            self.assertAlmostEqual(row['release_end_seconds']-row['switch_seconds'],0.8)
            self.assertEqual(row['timeline'][-1]['duration_seconds'],1.0)
            self.assertTrue(row['timeline'][-1]['terminal_switch'])
            full = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==names
                        and r['kind']=='连招' and not r['has_dodge'])
            for field in ('multipliers','energy_gain','harmony','personal_resources'):
                self.assertEqual(row[field],full[field])

    def test_e_cannot_preinput_followup_for_any_character(self):
        for character in self.payload['characters']:
            e_names = {a['name'] for a in character['actions'] if a['type']=='E'}
            for action in character['actions']:
                if action['name'] in e_names:
                    self.assertTrue(all(e['can_preinput'] is False for e in action['successors']))
            for row in character['rows']:
                for index, step in enumerate(row['timeline']):
                    if step['action'] in e_names:
                        self.assertFalse(step['preinput_before_switch'])
                        if index+1<len(row['timeline']):
                            self.assertEqual(row['timeline'][index+1]['input_seconds'],step['end_seconds'])
        self.assertFalse(any(r['operation']=='e预a5' for r in self.character('安魂曲')['rows']))

    def test_heavy_attacks_cannot_receive_preinput(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                self.assertIsNone(re.search(r'预[幻魔]?z', row['operation']))
                for index, step in enumerate(row['timeline']):
                    if not re.fullmatch(r'.*z\d*', step['action']):
                        continue
                    self.assertFalse(step['preinput'])
                    self.assertEqual(step['input_seconds'], step['start_seconds'])
                    if index:
                        self.assertEqual(step['input_seconds'], row['timeline'][index-1]['end_seconds'])

    def test_akane_exception_cannot_reappear_in_candidates(self):
        c = self.character('明音凛')
        self.assertFalse(any(r['original_actions'][-2:] == ['a3', 'a4'] and r['switch_label'] == '预输入合轴' for r in c['rows']))
        edge = next(a for a in c['actions'] if a['name'] == 'a3')['successors'][0]
        self.assertFalse(edge['can_preinput'])

    def test_dodge_drops_all_skipped_benefits_and_only_intermediate_long_actions(self):
        c = self.character('明音凛')
        original = next(r for r in c['rows'] if r['original_actions'] == ['a1','a2','a3','a4'] and not r['has_dodge'])
        dodged = next(r for r in c['rows'] if r['original_actions'] == ['a1','a2','a3','a4'] and r['has_dodge'])
        self.assertAlmostEqual(original['foreground_seconds'] - dodged['foreground_seconds'], 0.3)
        a3 = next(a for a in self.actions if a['character_name'] == '明音凛' and a['name'] == 'a3')
        self.assertAlmostEqual(original['multipliers']['atk'] - dodged['multipliers']['atk'], a3['multipliers']['atk'])
        self.assertAlmostEqual(original['energy_gain'] - dodged['energy_gain'], a3['energy_gain'])
        self.assertEqual(original['personal_resources']['弦能'] - dodged['personal_resources']['弦能'], 2)
        self.assertEqual(dodged['actions'][2], 'a3闪')
        self.assertEqual(dodged['timeline'][2]['dodge_replaces'], 'a3')
        self.assertFalse(any(r['has_dodge'] for r in self.character('安魂曲')['rows']))
        for c in self.payload['characters']:
            for row in c['rows']:
                if row['has_dodge']:
                    self.assertIsNone(row['timeline'][0]['dodge_replaces'])
                    self.assertIsNone(row['timeline'][-1]['dodge_replaces'])

    def test_heiyu_z_returns_to_the_correct_combo_stage(self):
        paths = [r['original_actions'] for r in self.character('黑羽')['rows']]
        self.assertTrue(all(n.startswith('魔') for names in paths for n in names))
        self.assertIn(['魔a1', '魔a2', '魔z1', '魔z2', '魔a3'], paths)

    def test_heiyu_z1_preinput_waits_until_point_eight_to_switch(self):
        character = self.character('黑羽')
        action = next(a for a in character['actions'] if a['name']=='魔z1')
        self.assertEqual(action['duration_seconds'], 1.4)
        self.assertTrue(action['preinput_forbidden'])
        path = ['魔a1','魔a2','魔z1','魔a3']
        pre = next(r for r in character['rows'] if r['original_actions']==path
                   and r['kind']=='合轴' and r['switch_method']=='preinput')
        full = next(r for r in character['rows'] if r['original_actions']==path and r['kind']=='连招')
        z1, a3 = pre['timeline'][-2:]
        self.assertEqual(pre['operation'], '魔z1预魔a3')
        self.assertAlmostEqual(a3['input_seconds'] - z1['start_seconds'], 0.4)
        self.assertAlmostEqual(pre['switch_seconds'] - z1['start_seconds'], 0.8)
        self.assertEqual(z1['duration_seconds'], 1.4)
        self.assertEqual(a3['start_seconds'], z1['end_seconds'])
        self.assertAlmostEqual(pre['foreground_seconds'], 2.1)
        self.assertAlmostEqual(pre['release_end_seconds'], 3.7)
        self.assertEqual(a3['placement'], 'background')
        for metric in ('multipliers','harmony','energy_gain','personal_resources'):
            self.assertEqual(pre[metric], full[metric])
        return_path = [*path,'魔a4','魔a5']
        returned = next(r for r in character['rows'] if r['original_actions']==return_path
                        and r['kind']=='分段合轴' and r['switch_method']=='preinput'
                        and r['operation'].startswith('魔z1预魔a3 /'))
        self.assertAlmostEqual(returned['window_1']['start_seconds'], 2.1)
        self.assertAlmostEqual(returned['window_1']['duration_seconds'], 1.6)
        self.assertAlmostEqual(returned['foreground_seconds'], 2.5)
        self.assertIn(['魔a1','魔a2','魔z1','魔z2'], [r['original_actions'] for r in character['rows']])

    def test_canhong_second_stage_heavy_finishes_at_each_forms_last_stage(self):
        paths = [r['original_actions'] for r in self.character('残虹')['rows']]
        self.assertIn(['a1','a2','z','a5'], paths)
        self.assertIn(['幻a1','幻a2','幻z','幻a4'], paths)

    def test_no_cross_form_paths_or_repeated_actions(self):
        for c in self.payload['characters']:
            ids = [r['id'] for r in c['rows']]
            self.assertEqual(len(ids), len(set(ids)))
            for row in c['rows']:
                names = row['original_actions']
                for name in set(names):
                    self.assertLessEqual(names.count(name), self.rules['characters'][c['name']].get('max_uses', {}).get(name, 1))
                if any(n.startswith('魔') for n in names):
                    self.assertTrue(all(n.startswith('魔') for n in names))
                if any(n.startswith('幻') for n in names):
                    self.assertTrue(all(n.startswith('幻') for n in names))

    def test_advanced_window_and_foreground_occupancy(self):
        row = next(r for r in self.character('黑羽')['rows'] if r['original_actions'] == ['魔a1','魔a2','魔a3','魔a4','魔a5'] and r['kind']=='分段合轴' and r['switch_method']=='preinput' and not r['has_dodge'])
        self.assertEqual(row['operation'], '魔a2预魔a3 / 魔a4预魔a5')
        self.assertEqual(row['window_1'], {'start_seconds':1.1,'end_seconds':2.3,'duration_seconds':1.2,'return_start_seconds':2.3,'return_end_seconds':2.3})
        self.assertAlmostEqual(row['foreground_seconds'], 1.5)
        self.assertAlmostEqual(row['switch_seconds'], 2.7)
        self.assertAlmostEqual(row['release_end_seconds'], 3.6)
        self.assertEqual(row['timeline'][3]['input_seconds'], 2.3)
        self.assertEqual([s['placement'] for s in row['timeline']], ['foreground','foreground','background','foreground','background'])
        self.assertEqual([s['preinput_before_switch'] for s in row['timeline']], [False,True,False,True,False])
        self.assertEqual([w['duration_seconds'] for w in row['overlap_windows']],[1.2])
        self.assertEqual([w['requires_return'] for w in row['overlap_windows']],[True])
        self.assertEqual([s['duration_seconds'] for s in row['foreground_segments']],[1.1,0.4])
        self.assertEqual(row['overlap_windows'][0]['return_start_seconds'],2.3)
        for c in self.payload['characters']:
            for r in c['rows']:
                if r['kind']=='分段合轴':
                    self.assertGreaterEqual(r['window_1']['duration_seconds'], 1.2)

    def test_each_off_field_window_is_numbered_and_totals_match(self):
        for character in self.payload['characters']:
            for row in character['rows']:
                windows=row['overlap_windows']
                self.assertEqual([w['number'] for w in windows],list(range(1,len(windows)+1)))
                if row['foreground_seconds'] is not None:
                    self.assertAlmostEqual(sum(s['duration_seconds'] for s in row['foreground_segments']),row['foreground_seconds'],places=5)
                expected=sum(row['foreground_intervals'][i+1][0]-interval[1]
                             for i,interval in enumerate(row['foreground_intervals'][:-1]))
                self.assertAlmostEqual(sum(w['duration_seconds'] for w in windows),expected,places=5)
                for window in windows:
                    self.assertTrue(window['requires_return'])
                    self.assertGreater(window['duration_seconds'],0)
                    if window['requires_return']:
                        self.assertGreaterEqual(window['duration_seconds'],1.2)
                        self.assertLessEqual(window['return_start_seconds'],window['end_seconds'])

    def test_short_return_cut_requires_three_characters_but_hold_does_not(self):
        names=[f'魔a{i}' for i in range(1,6)]
        rows=[r for r in self.character('黑羽')['rows'] if r['original_actions']==names
              and r['kind']=='分段合轴' and not r['has_dodge']]
        pre=next(r for r in rows if r['switch_method']=='preinput')
        normal=next(r for r in rows if r['switch_method']=='normal')
        held=next(r for r in rows if r['switch_method']=='hold')
        self.assertEqual(pre['foreground_segments'][1]['duration_seconds'],0.4)
        self.assertEqual(normal['foreground_segments'][1]['duration_seconds'],0.8)
        for row in (pre,normal):
            self.assertTrue(row['requires_three_characters'])
            self.assertEqual(row['three_character_segments'],[2])
        self.assertFalse(held['requires_three_characters'])
        self.assertEqual(held['three_character_segments'],[])
        self.assertTrue(all(not r['requires_three_characters'] for c in self.payload['characters']
                            for r in c['rows'] if not r.get('window_1')))

    def test_three_character_rule_exempts_first_segment_and_full_finish(self):
        def requirements(durations, cut=True):
            return generator.return_requirements({'foreground_segments':[
                {'number':i+1,'duration_seconds':duration} for i,duration in enumerate(durations)
            ], 'ends_with_switch':cut})
        self.assertFalse(requirements([0.2])['requires_three_characters'])
        self.assertFalse(requirements([0.2,1.2])['requires_three_characters'])
        self.assertEqual(requirements([2,1.199])['three_character_segments'],[2])
        self.assertFalse(requirements([0.2,0.4],False)['requires_three_characters'])
        self.assertEqual(requirements([2,0.4,0.8],False)['three_character_segments'],[2])

    def test_heiyu_e_uses_normal_a5_and_z2_can_preinput_a3(self):
        names = ['魔a1','魔a2','魔a3','魔a4','魔a5','魔e','魔z2','魔a3']
        row = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==names and not r['has_dodge'] and r['kind']=='连招')
        a5 = row['timeline'][4]
        e = row['timeline'][5]
        self.assertEqual(a5['duration_seconds'], 0.7)
        self.assertAlmostEqual(e['start_seconds'], a5['end_seconds'])
        z2, a3 = row['timeline'][-2:]
        self.assertFalse(a3['automatic'])
        self.assertAlmostEqual(a3['input_seconds'], z2['start_seconds'] + 0.4)
        self.assertEqual(a3['start_seconds'], z2['end_seconds'])
        for prefix in (['魔a1','魔a2','魔z1'], ['魔e'], names[:-2]):
            path = [*prefix, '魔z2', '魔a3']
            pre = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==path
                       and not r['has_dodge'] and r['kind']=='合轴' and r['switch_method']=='preinput')
            full = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==path
                        and not r['has_dodge'] and r['kind']=='连招')
            self.assertEqual(pre['operation'], '魔z2预魔a3')
            self.assertEqual(pre['preinput_action_indices'], [len(path)-2])
            self.assertAlmostEqual(pre['switch_seconds'], pre['timeline'][-2]['start_seconds'] + 0.4)
            self.assertEqual(pre['timeline'][-1]['placement'], 'background')
            self.assertEqual(pre['release_end_seconds'], full['release_end_seconds'])
            for metric in ('multipliers','energy_gain','harmony','personal_resources'):
                self.assertEqual(pre[metric], full[metric])
        returned = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==names
                        and not r['has_dodge'] and r['kind']=='分段合轴' and r['switch_method']=='preinput')
        self.assertTrue(returned['operation'].endswith('魔z2预魔a3'))

    def test_heiyu_z2_can_finish_without_forcing_a3(self):
        rows = self.character('黑羽')['rows']
        for path in (['魔e','魔z2'], ['魔a1','魔a2','魔z1','魔z2']):
            full = next(r for r in rows if r['original_actions']==path and r['switch_method']=='full')
            normal = next(r for r in rows if r['original_actions']==path and r['switch_method']=='normal'
                          and r['kind']=='合轴')
            self.assertEqual(normal['timeline'][-1]['action'], '魔z2')
            self.assertTrue(normal['timeline'][-1]['terminal_switch'])
            self.assertAlmostEqual(normal['switch_seconds'], normal['timeline'][-1]['start_seconds'] + 0.2)
            self.assertEqual(normal['multipliers'], full['multipliers'])
            originals = [next(a for a in self.actions if a['character_name']=='黑羽' and a['name']==n) for n in path]
            self.assertAlmostEqual(normal['multipliers']['atk'], sum(a['multipliers']['atk'] for a in originals))
        returned = next(r for r in rows if r['original_actions'][-1]=='魔z2' and r['kind']=='分段合轴'
                        and r['switch_method']=='normal')
        self.assertTrue(returned['operation'].endswith('魔z2正常切人'))

    def test_return_title_keeps_intermediate_skill_and_dodge_actions(self):
        names=['魔a1','魔a2','魔a3','魔a4','魔a5','魔e','魔z2','魔a3','魔a4','魔a5']
        row=next(r for r in self.character('黑羽')['rows'] if r['original_actions']==names
                 and r['kind']=='分段合轴' and r['switch_method']=='preinput'
                 and r['window_1']['start_seconds']==1.1 and r['actions'][7]=='魔a3闪')
        self.assertEqual(row['operation'],'魔a2预魔a3 / 魔a4 → 魔a5 → 魔e → 魔z2 → 魔a3闪 → 魔a4预魔a5')

    def test_delayed_switch_point_configuration(self):
        rules = copy.deepcopy(self.rules)
        rules['characters']['明音凛']['transitions'].append({'from':'a1','to':'a2','switch_delay_seconds':0.3,'source':'synthetic timing fixture'})
        payload = generator.generate(self.characters, self.actions, rules)
        row = next(r for c in payload['characters'] if c['name']=='明音凛' for r in c['rows'] if r['original_actions']==['a1','a2'] and r['switch_label']=='预输入合轴')
        self.assertAlmostEqual(row['foreground_seconds'], 0.7)

    def test_missing_duration_is_not_free_zero_seconds(self):
        actions = copy.deepcopy(self.actions)
        next(a for a in actions if a['character_name']=='明音凛' and a['name']=='a1').update(duration_seconds=0,duration_pending=True)
        payload = generator.generate(self.characters, actions, self.rules)
        row = next(r for c in payload['characters'] if c['name']=='明音凛' for r in c['rows'] if r['original_actions']==['a1','a2'])
        self.assertIsNone(row['foreground_seconds'])
        self.assertIsNone(row['energy_rate'])

    def test_unknown_transition_rejected(self):
        rules = copy.deepcopy(self.rules)
        rules['characters']['明音凛']['transitions'].append({'from':'a4','to':'a5','source':'test'})
        with self.assertRaises(ValueError):
            generator.generate(self.characters, self.actions, rules)

    def test_generator_is_deterministic_and_checked_in_output_is_current(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'tables.json'
            generator.generate_files(output=output)
            self.assertEqual(output.read_bytes(), (generator.MODULE / 'static/data/tables.json').read_bytes())
            initial = output.read_bytes()
            generator.generate_files(output=output)
            self.assertEqual(output.read_bytes(), initial)

    def test_anonymous_module_route_and_static_data(self):
        # Load only this blueprint: no account DB or sibling-module services involved.
        from app.modules.overlap import blueprint
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'), static_folder=str(ROOT / 'app/static'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/', endpoint='main.default_page', view_func=lambda: '')
        app.add_url_rule('/login', endpoint='main.login_page', view_func=lambda: '')
        with app.test_client() as client:
            response = client.get('/overlap')
            self.assertEqual(response.status_code, 200)
            self.assertNotIn('window.location.replace', response.get_data(as_text=True))
            self.assertIn('/overlap?page=pair', response.get_data(as_text=True))
            pair = client.get('/overlap?page=pair')
            self.assertEqual(pair.status_code, 200)
            self.assertNotIn('window.location.replace', pair.get_data(as_text=True))
            self.assertIn('pair-results', pair.get_data(as_text=True))
            self.assertIn('pair_worker.js', pair.get_data(as_text=True))
            self.assertIn('pair_multi_engine.js', pair.get_data(as_text=True))
            self.assertIn('pair-group-count', pair.get_data(as_text=True))
            self.assertIn('pair-first-group', pair.get_data(as_text=True))
            self.assertIn('pair-no-foreground-idle', pair.get_data(as_text=True))
            response = client.get('/static/overlap/data/tables.json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.get_json()['characters']), 6)
            response.close()


if __name__ == '__main__':
    unittest.main()
