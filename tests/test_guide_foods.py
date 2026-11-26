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
        self.assertIn('料理加成', html)
        self.assertIn('data-food-function="Revive"', html)
        self.assertIn('12 条结果', html)
        self.assertIn('data-guide-filter-group', html)

    def test_story_food_filter_defaults_on_only_for_web(self):
        foods = service.catalog()['foods']
        by_id = {r['id']: r for r in foods}
        self.assertTrue(by_id['Food_108']['story_only'])
        self.assertTrue(by_id['Food_060_01']['story_only'])
        self.assertTrue(by_id['Food_103']['story_only'])
        self.assertFalse(by_id['Food_115']['story_only'])
        self.assertFalse(by_id['Food_122']['story_only'])
        html = self.client.get('/guide?kind=foods').text
        self.assertIn('129 条结果', html)
        self.assertIn('name="hide_story" value="1" checked', html)
        html = self.client.get('/guide?kind=foods&hide_story=0').text
        self.assertIn('142 条结果', html)
        self.assertNotIn('name="hide_story" value="1" checked', html)
        self.assertEqual(self.client.get('/api/guide/foods').json['total'], 142)
        self.assertEqual(self.client.get('/api/guide/foods?hide_story=1').json['total'], 129)
        self.assertEqual(self.client.get('/guide?kind=foods&hide_story=0&hide_story=1').status_code, 200)

    def test_healing_sort_uses_22000_hp(self):
        names = [r['name'] for r in service.search('foods', bonus='heal')]
        self.assertLess(names.index('魔法膨胀全麦面包'), names.index('经典三重奏'))
        self.assertLess(names.index('扭扭饼干'), names.index('经典三重奏'))

    def test_detail_formats_and_all_local_images(self):
        data = self.client.get('/api/guide/foods/Food_001').json['data']
        self.assertEqual(data['name'], '极点')
        self.assertIn('攻击力增加75', data['effect'])
        self.assertIn('持续180秒', data['effect'])
        self.assertEqual(data['bonus_group'], 'attack')
        self.assertEqual(data['fons_price'], 450)
        self.assertIn('2-Four便利店(桥间地)', data['sources'])
        filtered = self.client.get('/api/guide/foods?bonus=crit_rate&limit=100').json
        self.assertGreater(filtered['total'], 0)
        self.assertTrue(all(r['bonus_group'] == 'crit_rate' for r in filtered['items']))
        self.assertTrue(all('暴击率增加' in r['effect'] for r in filtered['items']))
        self.assertEqual(self.client.get('/api/guide/foods/Food_001').json['data']['fons_price'], 450)
        response = self.client.get('/guide/foods/Food_001')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, '/guide?kind=foods&food=Food_001')
        html = self.client.get(response.location).text
        self.assertIn('food-detail-dialog', html)
        self.assertIn('获取途径', html)
        self.assertNotIn('href="/guide/foods/', html)
        self.assertIn('450 <small>方斯</small>', html)
        self.assertIn('class="guide-food-bonus">75</strong>', html)
        self.assertNotIn('class="guide-food-bonus">180</strong>', html)
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
        food = module.build_foods(tables)[0]
        self.assertEqual(food['function_name'], '攻击')
        self.assertEqual(food['bonus_group'], 'attack')
        self.assertIsNone(food['fons_price'])
        self.assertEqual(food['sources'], [])
        self.assertEqual(module.estimated_healing('恢复生命值上限的12%，并额外恢复850点生命值。'), 3490)
        self.assertEqual(module.estimated_healing('为当前角色恢复生命值上限的10%，并在之后的30秒内，每5秒恢复260点生命值。'), 3760)
        self.assertEqual(module.estimated_healing('当前角色在之后的30秒内，每5秒回复455生命值'), 2730)
        self.assertEqual(module.estimated_healing('复活选中角色，并恢复生命值上限的30%。'), 6600)
        segments = module.effect_segments('暴击率增加6.4%，持续180秒')
        self.assertEqual([r['text'] for r in segments if r['highlight']], ['6.4%'])
        row['OutputSourceID'] = 'food_source'
        tables.update(sources={'food_source': {'SourceArray': ['shop_a ']}},
                      source_details={'shop_a': {'ContentDesc': {'LocalizedString': '便利店'}}},
                      goods={'a': {'ItemID': 'food_1', 'PriceType': 'PriceShowType::Cost',
                                   'CurrencyID': 'Fons', 'Price': 450, 'OwnerShopID': 'shop_a'},
                             'b': {'ItemID': 'Food_1', 'PriceType': 'PriceShowType::Cost',
                                   'CurrencyID': 'Fons', 'Price': 300, 'OwnerShopID': 'shop_a'},
                             'c': {'ItemID': 'Food_1', 'PriceType': 'PriceShowType::Cost',
                                   'CurrencyID': 'other', 'Price': 1, 'OwnerShopID': 'shop_a'}})
        food = module.build_foods(tables)[0]
        self.assertEqual(food['fons_price'], 300)
        self.assertEqual(food['sources'], ['便利店'])
        self.assertEqual(len(food['offers']), 2)
        tables['goods']['a']['Price'] = -1
        with self.assertRaises(ValueError):
            module.build_foods(tables)
        tables['goods']['a']['Price'] = 450
        row['UseContext'] = {}
        with self.assertRaises(ValueError):
            module.build_foods(tables)
        row['ItemEffectType'] = 'unknown'
        with self.assertRaises(ValueError):
            module.build_foods(tables)
