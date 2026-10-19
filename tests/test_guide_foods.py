"""Food function grouping, local assets and shared public read formats."""
from pathlib import Path
import importlib.util
import unittest

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class GuideFoodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_foods_grouping_filter_and_effect_search(self):
        foods = service.catalog()['foods']
        self.assertEqual(len(foods), 142)
        self.assertEqual(len({r['id'] for r in foods}), len(foods))
        self.assertEqual({r['function'] for r in foods},
                         {'Attack', 'Defense', 'AddHPInstant', 'AddHPConstant', 'Revive', 'Explore'})
        filtered = self.client.get('/api/guide/foods?function=Revive&limit=100').json
        self.assertEqual(filtered['total'], 12)
        self.assertTrue(all(r['function'] == 'Revive' for r in filtered['items']))
        self.assertTrue(service.search('foods', query='攻击力增加75'))
        html = self.client.get('/guide?kind=foods&function=Revive').text
        self.assertIn('料理功能', html)
        self.assertIn('data-food-function="Revive"', html)
        self.assertIn('12 条结果', html)
        self.assertIn('data-guide-filter-group', html)

    def test_detail_formats_and_all_local_images(self):
        data = self.client.get('/api/guide/foods/Food_001').json['data']
        self.assertEqual(data['name'], '极点')
        self.assertIn('攻击力增加75', data['effect'])
        self.assertIn('持续180秒', data['effect'])
        html = self.client.get('/guide/foods/Food_001').text
        self.assertIn('料理效果', html)
        self.assertIn(data['effect'], html)
        md = self.client.get('/guide/foods/Food_001.md').text
        self.assertIn(data['effect'], md)
        self.assertIn('料理效果', md)
        self.assertNotIn('弧盘效果', md)
        self.assertIn('foods', self.client.get('/api/guide/schema').json['collections'])
        self.assertIn('/api/guide/foods', self.client.get('/guide/llms.txt').text)
        for food in service.catalog()['foods']:
            path = ROOT / 'app/modules/guide/static' / food['image'].removeprefix('/static/guide/')
            content = path.read_bytes()
            self.assertEqual(content[:4], b'RIFF')
            self.assertEqual(content[8:12], b'WEBP')

    def test_import_rejects_unknown_functions_and_missing_effects(self):
        spec = importlib.util.spec_from_file_location('sync_foods', ROOT / 'scripts/sync_guide_foods.py')
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        row = {'ItemEffectType': 'EItemEffectType::Attack', 'ItemName': {'LocalizedString': '菜'},
               'UseContext': {'LocalizedString': '攻击力增加75'}, 'Description': {},
               'ItemQuality': 'EItemQuality::ITEM_QUALITY_BLUE', 'ItemCD': 0}
        tables = {'foods': {'Food_1': row}, 'types': {'Attack': {'EffectTypeName': '攻击'}}}
        self.assertEqual(module.build_foods(tables)[0]['function_name'], '攻击')
        row['UseContext'] = {}
        with self.assertRaises(ValueError):
            module.build_foods(tables)
        row['ItemEffectType'] = 'unknown'
        with self.assertRaises(ValueError):
            module.build_foods(tables)
