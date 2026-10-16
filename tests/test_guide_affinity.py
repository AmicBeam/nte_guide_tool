"""Affinity import joins, shop semantics, and public format integration."""
import importlib.util
import json
from pathlib import Path
import unittest

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class GuideAffinityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_character_gifts_across_formats(self):
        response = self.client.get('/api/guide/characters/1042?section=affinity')
        self.assertEqual(response.status_code, 200)
        data = response.json['data']
        self.assertEqual(set(data), {'id', 'name', 'affinity'})
        affinity = data['affinity']
        self.assertEqual(len(affinity['levels']), 10)
        gifts = {g['name']: g for g in affinity['gifts']}
        self.assertEqual(gifts['归乡谣']['affinity_gain'], 200)
        self.assertEqual(gifts['结的艺术']['offers'][0]['price'], 5000)
        self.assertEqual(gifts['溶月茶具']['offers'][0]['currency_name'], '方斯')
        self.assertEqual(gifts['天星仪']['affinity_gain'], 1200)
        self.assertTrue(gifts['天星仪']['exclusive'])
        self.assertEqual(gifts['天星仪']['offers'], [])
        self.assertGreater(len(gifts['浮冰影券']['offers']), 1)
        md = self.client.get('/guide/characters/1042.md?section=affinity').text
        self.assertIn('赠送好感值: 1200', md)
        self.assertIn('商店标价: 5000', md)
        html = self.client.get('/guide/characters/1042?section=affinity').text
        for phrase in ['喜好礼物与总花费', '咕比奇原味薯片', '专属礼物', '5,000', 'href="#section-affinity"']:
            self.assertIn(phrase, html)
        self.assertTrue(any(r['id'] == '1042' for r in service.search('characters', '天星仪')))
        self.assertIn('affinity', service.schema()['character_sections'])
        self.assertIn('section=affinity', self.client.get('/guide/llms.txt').text)

    def test_total_costs_round_up_fons_only_and_sort(self):
        from app.modules.guide.affinity import with_gift_costs
        affinity = service.detail('characters', '1042')['affinity']
        self.assertEqual(affinity['total_affinity'], 56100)
        gifts = {g['name']: g for g in affinity['gifts']}
        tea = gifts['溶月茶具']
        self.assertEqual(tea['required_gifts'], 141)
        self.assertEqual(tea['costs'][0]['total_cost'], 2115000)
        self.assertEqual(gifts['天星仪']['costs'], [])
        self.assertEqual(gifts['天星仪']['required_gifts'], 47)
        data = {'levels': [{'max_value': 100}], 'gifts': [{'affinity_gain': 30,
            'offers': [{'currency_id': 'Fons', 'currency_name': '方斯', 'price': 40},
                       {'currency_id': 'Fons', 'currency_name': '方斯', 'price': 20},
                       {'currency_id': 'b', 'currency_name': '币B', 'price': 5}]}]}
        result = with_gift_costs(data)['gifts'][0]
        self.assertEqual(result['required_gifts'], 4)
        self.assertEqual([row['total_cost'] for row in result['costs']], [80])
        self.assertNotIn('costs', data['gifts'][0])
        data['levels'] = []
        result = with_gift_costs(data)['gifts'][0]
        self.assertIsNone(result['required_gifts'])
        self.assertTrue(all(row['total_cost'] is None for row in result['costs']))
        html = self.client.get('/guide/characters/1042?section=affinity').text
        self.assertIn('2,115,000 方斯', html)
        self.assertIn('总花费</th>', html)
        self.assertNotIn('购买渠道与标价', html)
        self.assertNotIn('Oops！Chest', html)
        self.assertNotIn('630 黑铁螺钉', html)
        ticket_row = html.split('浮冰影券', 1)[1].split('</tr>', 1)[0]
        self.assertNotIn('共 141 份', ticket_row)
        self.assertEqual(ticket_row.count('>—</span>'), 2)
        self.assertEqual(gifts['浮冰影券']['costs'], [])
        costs = [g['costs'][0]['total_cost'] for g in affinity['gifts'] if g['costs']]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(affinity['gifts'][0]['name'], '咕比奇原味薯片')
        self.assertEqual(affinity['gifts'][-1]['costs'], [])

    def test_all_character_coverage_and_missing_configuration(self):
        for character in service.catalog()['characters']:
            affinity = character['affinity']
            self.assertEqual(bool(affinity['levels']), affinity['available'])
            for gift in affinity['gifts']:
                self.assertNotEqual(gift['name'], gift['id'])
                self.assertGreater(gift['affinity_gain'], 0)
                for offer in gift['offers']:
                    self.assertGreaterEqual(offer['price'], 0)
                    self.assertTrue(offer['shop_ids'])
                    self.assertTrue(offer['goods_ids'])
                    if offer['discount_price'] is not None:
                        self.assertGreaterEqual(offer['discount_price'], 0)
        html = self.client.get('/guide/characters/1046?section=affinity').text
        self.assertIn('尚未提供好感度等级配置', html)
        self.assertIn('尚未提供喜好礼物配置', html)
        self.assertNotIn('>免费<', html)

    def test_import_preserves_multi_currency_and_unpriced_gifts(self):
        spec = importlib.util.spec_from_file_location('sync_affinity', ROOT / 'scripts/sync_guide_affinity.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def goods(currency, price, shop='a'):
            return {'PriceType': 'PriceShowType::Cost', 'OwnerShopID': shop,
                    'CurrencyID': currency, 'ItemID': 'gift', 'Price': price,
                    'bUseOff': False, 'OffPrice': 1, 'bUseOffTime': False,
                    'UseLimitStartTime': False, 'UseLimitEndTime': False, 'UnLockID': 'None'}
        gift = {'SpeicalGiftDatas': [{'RoleId': '1', 'LikeabilityValue': 200}],
                'ExclusiveLikeabilityID': 'None', 'DefaultLikeabilityValue': 50, 'bUnlimit': False}
        discounted = {**goods('coin', 40), 'bUseOff': True, 'OffPrice': 30, 'bUseOffTime': True}
        tables = {'roles': {'1': {'LevelInfo': [{'MaxValue': 100}]}},
                  'items': {'gift': {'ItemName': {'LocalizedString': '礼物'}}},
                  'shops': {'a': {'ShopName': {'LocalizedString': '商店'}}},
                  'gifts': {'gift': gift, 'unpriced': {**gift, 'ExclusiveLikeabilityID': '1'},
                            'other': {**gift, 'ExclusiveLikeabilityID': '2'}},
                  'goods': {'1': goods('coin', 40), '2': goods('token', 60),
                            '3': goods('coin', 40), '4': discounted,
                            '5': {**goods('coin', 999), 'PriceType': 'PriceShowType::Gain'}}}
        affinity = module.build_affinity(['1', '2'], tables)['characters'][0]['affinity']
        self.assertEqual(len(affinity['gifts']), 2)
        offers = affinity['gifts'][0]['offers']
        self.assertEqual(len(offers), 3)
        self.assertEqual(offers[0]['goods_ids'], ['1', '3'])
        self.assertIsNone(offers[0]['discount_price'])
        self.assertEqual(offers[1]['currency_id'], 'token')
        self.assertEqual(offers[2]['discount_price'], 30)
        self.assertTrue(offers[2]['discount_time_limited'])
        self.assertEqual(affinity['gifts'][1]['offers'], [])
        tables['goods']['1']['Price'] = -1
        with self.assertRaises(ValueError):
            module.build_affinity(['1'], tables)
        tables['goods']['1']['Price'] = 40
        tables['roles']['1']['LevelInfo'][0]['MaxValue'] = -1
        with self.assertRaises(ValueError):
            module.build_affinity(['1'], tables)


if __name__ == '__main__':
    unittest.main()
