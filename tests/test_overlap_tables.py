import copy
import importlib.util
import json
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

    def test_scope_and_no_melee_requiem(self):
        self.assertEqual({c['name'] for c in self.payload['characters']}, {'安魂曲', '残虹', '黑羽', '明音凛'})
        self.assertEqual([a['name'] for a in self.character('安魂曲')['actions']], [f'a{i}远' for i in range(1, 6)])

    def test_preinput_foreground_and_full_background_benefits(self):
        row = next(r for r in self.character('安魂曲')['rows'] if r['original_actions'] == ['a1远', 'a2远', 'a3远', 'a4远'] and r['switch_label'] == '预输入合轴')
        self.assertEqual(row['operation'], 'a3预a4')
        self.assertAlmostEqual(row['foreground_seconds'], 1.0)
        self.assertAlmostEqual(row['release_end_seconds'], 2.7)
        self.assertAlmostEqual(row['background_seconds'], 1.7)
        self.assertAlmostEqual(row['multipliers']['atk'], 4.401)
        self.assertAlmostEqual(row['timeline'][-1]['input_seconds'], 1.0)
        self.assertAlmostEqual(row['timeline'][-1]['start_seconds'], 1.5)
        self.assertEqual([s['placement'] for s in row['timeline']], ['foreground','foreground','foreground','background'])

    def test_akane_exception_cannot_reappear_in_candidates(self):
        c = self.character('明音凛')
        self.assertFalse(any(r['original_actions'][-2:] == ['a3', 'a4'] and r['switch_label'] == '预输入合轴' for r in c['rows']))
        edge = next(a for a in c['actions'] if a['name'] == 'a3')['successors'][0]
        self.assertFalse(edge['can_preinput'])

    def test_dodge_drops_all_skipped_benefits_and_only_intermediate_long_actions(self):
        c = self.character('明音凛')
        original = next(r for r in c['rows'] if r['original_actions'] == ['a1','a2','a3','a4'] and not r['has_dodge'])
        dodged = next(r for r in c['rows'] if r['original_actions'] == ['a1','a2','a3','a4'] and r['has_dodge'])
        self.assertAlmostEqual(original['foreground_seconds'] - dodged['foreground_seconds'], 0.4)
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
        row = next(r for r in self.character('黑羽')['rows'] if r['original_actions'] == ['魔a1','魔a2','魔a3','魔a4','魔a5'] and r['kind']=='分段合轴' and not r['has_dodge'])
        self.assertEqual(row['operation'], '魔a2预魔a3 / 魔a4预魔a5')
        self.assertEqual(row['window_1'], {'start_seconds':0.7,'end_seconds':2.4,'duration_seconds':1.7,'return_start_seconds':1.9,'return_end_seconds':2.4})
        self.assertAlmostEqual(row['foreground_seconds'], 0.9)
        self.assertAlmostEqual(row['switch_seconds'], 2.6)
        self.assertAlmostEqual(row['release_end_seconds'], 4.2)
        self.assertEqual(row['timeline'][3]['input_seconds'], 2.4)
        self.assertEqual([s['placement'] for s in row['timeline']], ['foreground','foreground','background','foreground','background'])
        for c in self.payload['characters']:
            for r in c['rows']:
                if r['kind']=='分段合轴':
                    self.assertGreaterEqual(r['window_1']['duration_seconds'], 1.2)

    def test_heiyu_e_forces_detached_a5_and_z2_automatically_returns_a3(self):
        names = ['魔a1','魔a2','魔a3','魔a4','魔a5','魔e','魔z2','魔a3']
        row = next(r for r in self.character('黑羽')['rows'] if r['original_actions']==names and not r['has_dodge'] and r['kind']=='连招')
        a5 = row['timeline'][4]
        e = row['timeline'][5]
        self.assertEqual(a5['duration_seconds'], 0.6)
        self.assertAlmostEqual(e['start_seconds'], a5['end_seconds'])
        self.assertTrue(row['timeline'][-1]['automatic'])
        self.assertIsNone(row['timeline'][-1]['input_seconds'])

    def test_delayed_switch_point_configuration(self):
        rules = copy.deepcopy(self.rules)
        rules['characters']['明音凛']['transitions'].append({'from':'a1','to':'a2','switch_delay_seconds':0.3,'source':'synthetic timing fixture'})
        payload = generator.generate(self.characters, self.actions, rules)
        row = next(r for c in payload['characters'] if c['name']=='明音凛' for r in c['rows'] if r['original_actions']==['a1','a2'] and r['switch_label']=='预输入合轴')
        self.assertAlmostEqual(row['foreground_seconds'], 0.5)

    def test_missing_duration_is_not_free_zero_seconds(self):
        actions = copy.deepcopy(self.actions)
        next(a for a in actions if a['character_name']=='明音凛' and a['name']=='a1').update(duration_seconds=0,duration_pending=True)
        payload = generator.generate(self.characters, actions, self.rules)
        row = next(r for c in payload['characters'] if c['name']=='明音凛' for r in c['rows'] if r['original_actions']==['a1'])
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
            response = client.get('/static/overlap/data/tables.json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.get_json()['characters']), 4)
            response.close()


if __name__ == '__main__':
    unittest.main()
