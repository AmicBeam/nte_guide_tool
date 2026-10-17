"""Downloaded CLI defaults to level-ten skills and optional per-hit details."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit
from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class CliSkillDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()
        spec = importlib.util.spec_from_file_location('skill_defaults_cli', ROOT / 'app/modules/guide/cli.py')
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def run_cli(self, *args):
        def opening(url, **kwargs):
            parsed = urlsplit(url)
            result = self.client.get(parsed.path + '?' + parsed.query)
            self.assertEqual(result.status_code, 200, result.text[:200])
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = result.data
            response.headers.get_content_type.return_value = result.mimetype
            return response
        with patch.object(self.cli.urllib.request, 'urlopen', side_effect=opening), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli.main(list(args)), 0)
        return output.getvalue()

    def test_default_ten_and_explicit_detail_and_all_level_flags(self):
        args = ['get', 'characters', '1042', '--ability', 'GA_BlackBird_Melee']
        source = service.detail('characters', '1042')['abilities'][0]
        before = json.dumps(source)
        result = json.loads(self.run_cli(*args))['data']
        skill = result['abilities'][0]
        self.assertEqual(skill['param_rows'][0]['values']['level'], 10)
        self.assertEqual(skill['param_rows'][0]['values']['value'], source['param_rows'][0]['values'][9])
        self.assertNotIn('hit_details', json.dumps(skill))
        self.assertFalse(any('环合值' in row['label'] for row in skill['param_rows']))
        keys = {row['key'] for row in skill['param_rows']}
        for group in skill['groups']:
            self.assertTrue(set(group.get('parameter_keys', [])).issubset(keys))
        self.assertEqual(result['skill_parameters'], {'level': 10, 'hit_details': False})
        first = json.loads(self.run_cli(*args, '--level', '1'))['data']['abilities'][0]
        self.assertEqual(first['param_rows'][0]['values']['value'], source['param_rows'][0]['values'][0])
        detailed = json.loads(self.run_cli(*args, '--hit-details'))['data']['abilities'][0]
        self.assertEqual(detailed['param_rows'][0]['hit_details'][0]['energy_gain'], source['param_rows'][0]['hit_details'][0]['energy_gain'])
        self.assertTrue(any('环合值' in row['label'] for row in detailed['param_rows']))
        all_levels = json.loads(self.run_cli(*args, '--all-levels'))['data']['abilities'][0]
        self.assertEqual(all_levels['param_rows'][0]['values'], source['param_rows'][0]['values'])
        self.assertNotIn('hit_details', json.dumps(all_levels))
        full = json.loads(self.run_cli(*args, '--all-levels', '--hit-details'))['data']['abilities'][0]
        self.assertEqual(full['param_rows'], source['param_rows'])
        self.assertEqual(before, json.dumps(source))

    def test_default_character_keeps_growth_and_other_collections_unchanged(self):
        result = json.loads(self.run_cli('get', 'characters', '1042'))['data']
        original = service.detail('characters', '1042')
        self.assertEqual(result['stats'], original['stats'])
        self.assertEqual(result['experience'], original['experience'])
        self.assertEqual(result['materials'], original['materials'])
        self.assertNotIn('hit_details', json.dumps(result['abilities']))
        arc = json.loads(self.run_cli('get', 'arcs', 'fork_twinbirds'))['data']
        self.assertEqual(arc['effect'], service.detail('arcs', 'fork_twinbirds')['effect'])

    def test_markdown_defaults_and_api_backwards_compatibility(self):
        args = ['get', 'characters', '1042', '--ability', 'GA_BlackBird_Melee', '--format', 'markdown']
        basic = self.run_cli(*args)
        self.assertIn('47.4%', basic)
        self.assertNotIn('基础回能', basic)
        self.assertNotIn('基础倾陷', basic)
        self.assertNotIn('23.7%', basic)
        detailed = self.run_cli(*args, '--hit-details')
        self.assertIn('基础回能', detailed)
        self.assertIn('基础倾陷', detailed)
        raw = self.client.get('/api/guide/characters/1042?section=abilities').json['data']
        self.assertIsInstance(raw['abilities'][0]['param_rows'][0]['values'], list)
        self.assertIn('hit_details', raw['abilities'][0]['param_rows'][0])
        for query in ['skill_level=0', 'skill_level=14', 'hit_details=2', 'hit_details=abc']:
            self.assertEqual(self.client.get('/api/guide/characters/1042?' + query).status_code, 400)
        self.assertEqual(self.client.get('/api/guide/arcs/fork_twinbirds?skill_level=10').status_code, 400)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli.main(['get', 'characters', '1042', '--level', '10', '--all-levels'])

    def test_every_character_skill_default_preserves_exact_level_values(self):
        for character in service.catalog()['characters']:
            with self.subTest(character=character['id']):
                result = self.client.get('/api/guide/characters/' + character['id'] + '?section=abilities&skill_level=10&hit_details=0').json['data']
                original = {ability['id']: ability for ability in character['abilities']}
                self.assertNotIn('hit_details', json.dumps(result['abilities']))
                for ability in result['abilities']:
                    identities = {(row['key'], row['label']) for row in ability['param_rows']}
                    rows = [row for row in original[ability['id']]['param_rows']
                            if (row['key'], row['label']) in identities]
                    self.assertEqual(len(rows), len(ability['param_rows']))
                    for row, source in zip(ability['param_rows'], rows):
                        values = source['values']
                        if values:
                            self.assertEqual(row['values']['value'], values[min(10, len(values)) - 1])
