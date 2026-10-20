"""Permanent bonuses share public formats with explicit upgrade semantics."""
import importlib.util
from pathlib import Path
import sys
import unittest
from flask import Flask
from app.modules.guide import blueprint, service, cli, chat_tools

ROOT = Path(__file__).resolve().parents[1]


class GuidePermanentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_collection_search_formats_and_cli(self):
        response = self.client.get('/guide?kind=permanent')
        self.assertEqual(response.status_code, 200)
        self.assertIn('常驻加成', response.text)
        self.assertIn('异象家具', response.text)
        self.assertIn('噗咔特训', response.text)
        self.assertEqual(self.client.get('/api/guide/permanent').json['total'], 2)
        self.assertEqual(service.search('permanent', query='锻刀石')[0]['id'], 'furniture')
        self.assertEqual(service.search('permanent', query='噗卡特训')[0]['id'], 'puka-training')
        self.assertIn('permanent', cli.KINDS)
        self.assertIn('permanent', chat_tools.KINDS['enum'])
        for identifier in ('furniture', 'puka-training'):
            self.assertEqual(self.client.get('/guide/permanent/' + identifier).status_code, 200)
            self.assertEqual(self.client.get('/api/guide/permanent/' + identifier).status_code, 200)
            self.assertIn('加成', self.client.get('/guide/permanent/' + identifier + '.md').text)
        self.assertIn('/api/guide/permanent', self.client.get('/guide/llms.txt').text)
        self.assertEqual(self.client.get('/api/guide/permanent/missing').status_code, 404)

    def test_training_costs_and_total_bonuses_are_not_summed(self):
        data = self.client.get('/api/guide/permanent/puka-training?section=levels').json['data']
        rows = data['levels']
        self.assertEqual([r['level'] for r in rows], list(range(1, 21)))
        self.assertEqual(rows[0]['required_puka_level'], 7)
        self.assertEqual(rows[-1]['required_puka_level'], 45)
        self.assertEqual(rows[-1]['total_fons'], 11000000)
        self.assertEqual(sum(row['fons_cost'] for row in rows), rows[-1]['total_fons'])
        self.assertEqual(rows[4]['fons_cost'], 600000)
        self.assertEqual({b['name']: b['value'] for b in rows[-1]['bonuses']},
                         {'攻击力': 28, '防御力': 30, '生命值': 420, '暴击率': 2.5,
                          '暴击伤害': 5, '环合强度': 12, '倾陷强度': 12})
        self.assertNotIn('notes', data)
        html = self.client.get('/guide/permanent/puka-training').text
        self.assertIn('11,000,000', html)
        self.assertIn('本级总加成', html)

    def test_furniture_next_level_cost_and_max_level(self):
        entries = service.detail('permanent', 'furniture')['entries']
        self.assertEqual(len(entries), 7)
        rows = {row['id']: row for row in entries}
        sword = rows['SF_0011']
        self.assertEqual(sword['levels'][0]['upgrade_costs'], [{'name': '锻刀石', 'amount': 2}])
        self.assertEqual(sword['levels'][0]['next_level'], 2)
        self.assertEqual(sword['levels'][-1]['bonuses'][0]['value'], 20)
        self.assertEqual(rows['SF_0012']['levels'][-1]['bonuses'][0]['value'], 4)
        self.assertEqual(rows['SF_0013']['levels'][-1]['bonuses'][0]['value'], 30)
        for row in entries:
            self.assertIsNone(row['levels'][-1]['next_level'])
            self.assertEqual(row['levels'][-1]['upgrade_costs'], [])
        self.assertIn('本性像素消耗进度', str(rows['SF_0009']))
        html = self.client.get('/guide/permanent/furniture').text
        self.assertIn('查看1–10级加成与升级要求', html)
        self.assertIn('已满级', html)

    def test_importer_rejects_unknown_or_conditional_modifiers(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            spec = importlib.util.spec_from_file_location('permanent_import', ROOT / 'scripts/sync_guide_permanent_bonuses.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.assertRaises(ValueError):
                module.bonuses({'ConditionArray': ['condition'], 'ModifyData': []})
            with self.assertRaises(ValueError):
                module.bonuses({'ConditionArray': [], 'ModifyData': [{'PropName': 'AtkAdd', 'PropValue': 2, 'ModifierOp': 'Multiply'}]})
        finally:
            sys.path.pop(0)
