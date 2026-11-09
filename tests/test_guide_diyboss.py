import importlib.util
import json
from pathlib import Path
import unittest

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class DiyBossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()
        cls.records = json.loads((ROOT / 'app/modules/guide/data/diyboss.json').read_text())

    def test_configurations_and_historical_values_are_independent(self):
        self.assertEqual([r['stage_count'] for r in self.records], [7, 8, 8])
        life = [next(g for g in r['stages'][0]['buff_groups'] if '生命' in g['name']) for r in self.records]
        self.assertNotEqual(life[0]['options'], life[2]['options'])
        for record in self.records:
            self.assertIsNone(record['begin_at'])
            self.assertIsNone(record['end_at'])
            for stage in record['stages']:
                self.assertTrue(stage['enemies'])
                for enemy in stage['enemies']:
                    path = ROOT / 'app/modules/guide/static' / enemy['image'].removeprefix('/static/guide/')
                    raw = path.read_bytes()
                    self.assertEqual(raw[:4], b'RIFF')
                    self.assertEqual(raw[8:12], b'WEBP')
                for group in stage['buff_groups']:
                    self.assertEqual(len(group['options']), 3)
                    self.assertTrue(all(isinstance(o['score'], (int, float)) for o in group['options']))
        latest = self.records[-1]
        composite = next(s for s in latest['stages'] if s['id'] == '8')
        self.assertEqual(composite['enemies'][0]['name'], '玛门&瓦楞城堡')
        self.assertEqual(sum(len(s['targets']) for s in latest['stages']), 24)
        self.assertTrue(all(t['score'] is None and t['description'] is None for s in latest['stages'] for t in s['targets']))

    def test_public_routes_search_and_single_stage(self):
        self.assertEqual(self.client.get('/api/guide/diyboss').json['total'], 3)
        self.assertIn('争锋赏宴', self.client.get('/guide?kind=diyboss').text)
        for record in self.records:
            page = self.client.get('/guide/diyboss/' + record['id'])
            self.assertEqual(page.status_code, 200)
            self.assertIn('自选强化', page.text)
            self.assertIn('scope="row"', page.text)
        one = self.client.get('/api/guide/diyboss/config-3?stage=7').json['data']
        self.assertEqual(len(one['stages']), 1)
        self.assertEqual(one['stages'][0]['name'], '心脏与羽毛')
        md = self.client.get('/guide/diyboss/config-3.md?stage=7').text
        self.assertIn('自选强化', md)
        self.assertIn('未提供', md)
        self.assertEqual(self.client.get('/api/guide/diyboss/config-3?stage=999').status_code, 404)
        self.assertEqual(self.client.get('/guide/diyboss/missing').status_code, 404)
        self.assertTrue(any(r['kind'] == 'diyboss' for r in self.client.get('/api/guide/search?q=心脏与羽毛').json['items']))
        with self.client.get('/guide/cli.py') as response:
            self.assertIn('diyboss', response.text)

    def test_importer_rejects_empty_input(self):
        spec = importlib.util.spec_from_file_location('sync_diyboss', ROOT / 'scripts/sync_guide_diyboss.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.json'
            path.write_text('[{"Rows": {}}]')
            with self.assertRaises(ValueError):
                module.rows(path)
