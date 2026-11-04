"""Arc selection, costs, recommended links and saved equipment prefill."""
import unittest
from pathlib import Path
from unittest.mock import patch
from flask import Flask
from app.modules.guide import blueprint, service, cultivation, arc_cultivation


class ArcCultivationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        app = Flask(__name__, template_folder=str(root / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_character_pages_with_gbk_default_encoding(self):
        original_open = Path.open

        def gbk_open(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
            if 'b' not in mode and encoding in (None, 'locale'):
                encoding = 'gbk'
            return original_open(path, mode, buffering, encoding, errors, newline)

        arc_cultivation.config.cache_clear()
        self.addCleanup(arc_cultivation.config.cache_clear)
        with patch.object(Path, 'open', gbk_open):
            for character in service.catalog()['characters']:
                with self.subTest(character=character['id']):
                    response = self.client.get('/guide/characters/' + character['id'])
                    self.assertEqual(response.status_code, 200, response.text[:300])
                    self.assertIn(character['name'], response.text)
            response = self.client.get('/guide/characters/1042/cultivation')
            self.assertEqual(response.status_code, 200, response.text[:300])
            response = self.client.post('/api/guide/characters/1042/cultivation',
                                        json={'include_arc': True, 'arc_id': 'fork_twinbirds'})
            self.assertEqual(response.status_code, 200, response.text[:300])
            self.assertEqual(response.json['data']['arc']['name'], service.detail('arcs', 'fork_twinbirds')['name'])

    def test_recommendations_and_compatible_choices_cover_roster(self):
        for character in service.catalog()['characters']:
            first = arc_cultivation.recommended(character)
            choices = arc_cultivation.choices(character)
            if character['id'] == '1074':
                self.assertEqual(choices, [])
                self.assertIsNone(first)
                continue
            self.assertTrue(choices)
            if first is None:
                self.assertNotIn(character['id'], arc_cultivation.config()['recommendations'])
                continue
            self.assertIn(first['id'], [arc['id'] for arc in choices])
            page = self.client.get('/guide/characters/' + character['id'])
            self.assertIn('href="/guide/arcs/' + first['id'] + '"', page.text)
        self.assertEqual(arc_cultivation.recommended({'id': '1042'})['id'], 'fork_twinbirds')

    def test_arc_costs_all_arcs_and_noop(self):
        for arc in service.catalog()['arcs']:
            result = arc_cultivation.calculate_arc(arc, {'arc_current_level': 1, 'arc_target_level': 80})
            self.assertGreater(result['experience'], 0)
            self.assertEqual(result['total_gold'], result['level_up_gold'] + sum(x['cost_gold'] for x in arc['materials']))
            self.assertEqual(arc_cultivation.calculate_arc(arc, {'arc_current_level': 80, 'arc_target_level': 80})['items'], [])
            boundary = arc_cultivation.calculate_arc(arc, {'arc_current_level': 20, 'arc_target_level': 20, 'arc_target_ascended': True})
            self.assertEqual(boundary['experience'], 0)
            self.assertEqual(boundary['total_gold'], arc['materials'][0]['cost_gold'])
            expected = {x['id'].casefold(): x['amount'] for x in arc['materials'][0]['items']}
            expected['gold'] = arc['materials'][0]['cost_gold']
            self.assertEqual({x['id'].casefold(): x['amount'] for x in boundary['items']}, expected)

    def test_first_upgrade_experience_and_dye_fee(self):
        arc = service.detail('arcs', 'fork_twinbirds')
        result = arc_cultivation.calculate_arc(arc, {'arc_current_level': 1, 'arc_target_level': 2})
        self.assertEqual(result['experience'], 120)
        self.assertEqual(result['book_experience'], 500)
        self.assertEqual(result['total_gold'], 150)
        self.assertEqual({x['id']: x['amount'] for x in result['items']},
                         {'gold': 150, 'WeaponUpMaterial_lv1': 1})

    def test_combined_api_aggregates_costs_and_rejects_invalid_arc(self):
        url = '/api/guide/characters/1042/cultivation'
        character = self.client.post(url, json={}).json['data']
        values = {'include_arc': True, 'arc_id': 'fork_twinbirds'}
        response = self.client.post(url, json=values)
        self.assertEqual(response.status_code, 200)
        result = response.json['data']
        self.assertEqual(result['total_gold'], character['total_gold'] + result['arc']['total_gold'])
        expected = {}
        for group in result['breakdown']:
            for item in group['items']:
                key = item['id'].casefold()
                expected[key] = expected.get(key, 0) + item['amount']
        self.assertEqual(expected, {x['id'].casefold(): x['amount'] for x in result['items']})
        for change in [{'arc_id': 'unknown'}, {'arc_id': 'fork_PaperPlane'},
                       {'arc_current_level': 40, 'arc_target_level': 20}, {'arc_current_level': 0},
                       {'arc_target_level': 81}, {'arc_current_level': True}, {'include_arc': 'yes'}]:
            self.assertEqual(self.client.post(url, json={**values, **change}).status_code, 400, change)
        page = self.client.get('/guide/characters/1042/cultivation?calculate=1&include_arc=1&arc_id=fork_twinbirds')
        self.assertEqual(page.status_code, 200)
        self.assertIn('已合并', page.text)
        self.assertNotIn('prefill-confirm', page.text)

    def test_anonymous_ui_has_optional_arc_below_character_levels(self):
        html = self.client.get('/guide/characters/1042/cultivation').text
        self.assertLess(html.index('当前练度 · 角色等级'), html.index('一并计算弧盘'))
        self.assertLess(html.index('一并计算弧盘'), html.index('<h2>技能与天赋</h2>'))
        self.assertIn('id="arc-fields" disabled hidden', html)
        self.assertNotIn('我已核对同步', html)


if __name__ == '__main__':
    unittest.main()
