import importlib.util
import io
import re
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GuideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_anonymous_html_contains_data_without_script(self):
        for path, text in [('/guide', '娜娜莉'), ('/guide/characters/1042', '葬羽冥'),
                ('/guide/arcs/fork_PaperPlane', '纸翼战队')]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(text, response.text)
            self.assertNotIn('window.location.replace', response.text)
        with self.client.get('/static/guide/guide.css') as response:
            self.assertEqual(response.status_code, 200)

    def test_search_pagination_and_empty_result(self):
        result = self.client.get('/api/guide/characters?limit=2').json
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['next_offset'], 2)
        second = self.client.get('/api/guide/characters?limit=2&offset=2').json
        self.assertNotEqual(result['items'][0]['id'], second['items'][0]['id'])
        matches = self.client.get('/api/guide/search?q=黑羽').json['items']
        self.assertTrue(any(x['id'] == '1042' for x in matches))
        self.assertTrue(all(x['element_name'] == '魂' for x in self.client.get('/api/guide/characters?element=魂').json['items']))
        self.assertIn('没有找到相关资料', self.client.get('/guide?q=不存在的关键词XYZ&rarity=').text)
        self.assertEqual(self.client.get('/api/guide/search?q=不存在的关键词XYZ').json['total'], 0)

    def test_narrow_reads_and_numbers(self):
        full = self.client.get('/api/guide/characters/1042').json['data']
        narrowed = self.client.get('/api/guide/characters/1042?section=abilities').json['data']
        self.assertEqual(narrowed['abilities'], full['abilities'])
        self.assertNotIn('stats', narrowed)
        challenge = self.client.get('/api/guide/challenges/Abyss_14?stage=12').json['data']
        self.assertEqual([x['id'] for x in challenge['stages']], [12])
        stats = challenge['stages'][0]['halves'][0]['waves'][0]['monsters'][0]['stats']
        self.assertGreater(stats['hp'], 0)
        self.assertIsInstance(stats['resistances']['cosmos'], (int, float))
        md = self.client.get('/guide/characters/1042.md?section=awaken')
        self.assertEqual(md.mimetype, 'text/markdown')
        self.assertIn('叛逃者', md.text)
        self.assertNotIn('<NumGreen>', md.text)
        self.assertEqual(self.client.get('/guide/challenges/Abyss_14?stage=').status_code, 200)
        stage_html = self.client.get('/guide/challenges/Abyss_14?stage=12').text
        self.assertIn('魂', stage_html)
        self.assertNotIn('>psyche<', stage_html)
        self.assertNotIn('1 · 始发站', stage_html)

    def test_equipment_and_materials_across_formats(self):
        equipment = self.client.get('/api/guide/characters/1042?section=equip_slots').json['data']['equip_slots']
        self.assertEqual(equipment['owner_grid_count'], 3)
        self.assertEqual(equipment['stats'][0]['value'], 8)
        self.assertEqual(len(equipment['slots']), 7)
        self.assertEqual(equipment['slots'][1], [-1, 0, 0, 0, 0, -1, -1])
        html = self.client.get('/guide/characters/1042?section=equip_slots').text
        self.assertIn('每装配一个Ⅲ型驱动，使暴击率增加8%', html)
        self.assertIn('aria-label="空幕格位"', html)
        self.assertNotIn('{0}', html.split('<details')[0])
        materials = self.client.get('/api/guide/characters/1042?section=materials').json['data']['materials']
        first = materials['breakthrough'][0]
        self.assertEqual(first['level'], '30')
        self.assertEqual(first['items'][0]['name'], '模糊数符')
        self.assertEqual(first['items'][0]['amount'], 5)
        self.assertEqual(first['items'][1]['name'], '甲硬币')
        skill = materials['skills'][0]['levels'][0]
        self.assertEqual(skill['require_breakthrough_level'], 2)
        self.assertEqual(skill['items'][1]['name'], '水波的迟疑')
        md = self.client.get('/guide/characters/1042.md?section=materials').text
        self.assertIn('角色突破', md)
        self.assertIn('技能升级', md)
        self.assertIn('模糊数符', md)
        arc = self.client.get('/api/guide/arcs/fork_PaperPlane?section=materials').json['data']['materials'][0]
        self.assertEqual(arc['cost_gold'], 16000)
        self.assertEqual(arc['items'][0]['name'], '无味冷点心')

    def test_human_level_tables_preserve_full_data(self):
        html = self.client.get('/guide/characters/1042?section=abilities').text
        self.assertIn('type="range"', html)
        self.assertIn('aria-label="掠影技能等级"', html)
        self.assertNotIn('查看全部等级数据', html)
        self.assertIn('data-guide-toc', html)
        self.assertIn('href="#skill-1"', html)
        self.assertIn('一段伤害倍率', html)
        self.assertIn('技能目录', html)
        abilities = self.client.get('/api/guide/characters/1042?section=abilities').json['data']['abilities']
        self.assertEqual(len(abilities[0]['param_rows'][0]['values']), 13)
        arc = self.client.get('/guide/arcs/fork_PaperPlane?section=effect').text
        self.assertIn('灵属性异能伤害提升20%', re.sub(r'<[^>]*>', '', arc))
        self.assertIn('guide-effect-number', arc)
        self.assertNotIn('效果 1', arc)
        unsafe = service.effect_html('<img src=x>{0}', [{'id_value': 0, 'values': ['<script>1</script>']}], 1)
        self.assertNotIn('<img', unsafe)
        self.assertNotIn('<script>', unsafe)
        self.assertIn('&lt;img', unsafe)
        self.assertLess(self.client.get('/guide/arcs/fork_PaperPlane').text.index('id="section-stats"'), self.client.get('/guide/arcs/fork_PaperPlane').text.index('id="section-effect"'))
        self.assertIn('23.75%', arc)
        self.assertIn('35%', arc)
        materials = self.client.get('/guide/characters/1042?section=materials').text
        self.assertIn('guide-material-grid', materials)
        self.assertIn('模糊数符', materials)
        self.assertNotIn('材料 ID', materials)

    def test_hit_details_and_move_groups(self):
        characters = service.catalog()['characters']
        blackbird = next(c for c in characters if c['id'] == '1042')
        melee = blackbird['abilities'][0]
        group = next(g for g in melee['groups'] if g['name'] == '普通攻击：掠影')
        self.assertTrue(any('五段伤害倍率' in row['label'] for row in group['param_rows']))
        first = group['param_rows'][0]['hit_details'][0]
        self.assertEqual(first['attack_values'][0], '23.7%')
        self.assertEqual(first['attack_values'][12], '59.7%')
        self.assertEqual(first['harmony_gain'], 1.161)
        self.assertEqual(first['energy_gain'], 0.838)
        self.assertEqual(first['stagger'], 0.141)
        self.assertEqual(group['param_rows'][1]['hit_details'][0]['repeat'], 2)
        for character in characters:
            for ability in character['abilities']:
                grouped = [r['key'] for g in ability['groups'] for r in g['param_rows']]
                self.assertCountEqual(grouped, [r['key'] for r in ability['param_rows']])
                self.assertFalse(any(r['hit_detail_status'] == 'not_provided' for r in ability['param_rows']))
        nanally = next(c for c in characters if c['id'] == '1010')
        third = nanally['abilities'][0]['param_rows'][2]
        self.assertEqual(len(third['hit_details']), 3)
        html = self.client.get('/guide/characters/1042?section=abilities').text
        self.assertIn('详细命中数据', html)
        self.assertIn('基础回能', html)
        self.assertIn('0.838', html)

    def test_character_arc_compatibility_and_badges(self):
        character = self.client.get('/api/guide/characters/1042').json['data']
        self.assertEqual(character['arc_type']['name'], '气态')
        self.assertEqual([tag['name'] for tag in character['char_tags']], ['输出', '即瞬环合', '主力输出', '输出增益'])
        for record in service.catalog()['characters']:
            self.assertIn(record['arc_type']['name'], ['固态', '液态', '等离子', '气态', '聚合'])
            self.assertTrue((ROOT / 'app' / record['rarity_icon'].removeprefix('/')).is_file())
            for badge in [record['arc_type'], *record['char_tags']]:
                self.assertTrue(badge['icon'].endswith('.webp'))
                self.assertTrue((ROOT / 'app' / badge['icon'].removeprefix('/')).is_file())
        html = self.client.get('/guide/characters/1042').text
        self.assertIn('气态弧盘', html)
        self.assertIn('aria-label="角色定位"', html)
        self.assertIn('alt="5星角色"', html)
        self.assertIn('alt="魂属性"', html)
        self.assertNotIn('魂 · 5 星', html)
        self.assertIn('/static/images/character-tags/', html)

    def test_all_characters_have_local_webp_art(self):
        characters = service.catalog()['characters']
        self.assertEqual(len(characters), 25)
        for character in characters:
            for kind in ('avatar', 'portrait', 'fullbody'):
                url = character['assets'][kind]
                self.assertTrue(url.startswith('/static/images/characters/'))
                self.assertTrue(url.endswith('.webp'))
                path = ROOT / 'app' / url.removeprefix('/')
                with path.open('rb') as stream:
                    header = stream.read(12)
                self.assertEqual(header[:4], b'RIFF')
                self.assertEqual(header[8:12], b'WEBP')
        by_id = {c['id']: c for c in characters}
        self.assertNotEqual(by_id['1046']['assets'], by_id['1051']['assets'])
        html = self.client.get('/guide/characters/1042').text
        self.assertIn('黑羽高清立绘', html)
        self.assertIn('黑羽头像', html)
        self.assertNotIn('默认立绘预览', html)

    def test_errors_and_discovery(self):
        for path, status in [('/api/guide/unknown',404),('/api/guide/arcs/missing',404),
                ('/api/guide/arcs?limit=0',400),('/api/guide/arcs?offset=-1',400),
                ('/api/guide/arcs?limit=abc',400),('/api/guide/characters?rarity=9',400),
                ('/api/guide/characters/1042?section=missing',404),
                ('/api/guide/characters/1042?stage=1',400),
                ('/api/guide/challenges/Abyss_14?stage=999',404)]:
            response = self.client.get(path)
            self.assertEqual(response.status_code,status,path)
            self.assertIn('error',response.json)
        self.assertIn('/api/guide/search', self.client.get('/llms.txt').text)
        self.assertEqual(self.client.get('/api/guide/schema').json['schema_version'],1)
        download = self.client.get('/guide/cli.py')
        self.assertIn('attachment',download.headers['Content-Disposition'])
        self.assertNotIn('from app',download.text)
        download.close()

    def test_importer_preserves_combat_and_cleans_markup(self):
        importer = load_file('guide_refresh', 'scripts/refresh_guide_data.py')
        value = {'desc':'<Hun>魂属性</>提升<lv>45%</>\\r\\n下一行', 'values':[0.3, 15], 'icon':'unused'}
        self.assertEqual(importer.clean(value), {'desc':'魂属性提升45%\n下一行','values':[0.3,15]})
        self.assertEqual(importer.clean({'materials': [{'amount': [5]}]}), {'materials': [{'amount': [5]}]})
        for group in service.KINDS:
            records = service.catalog()[group]
            self.assertEqual(len(records),len({x['id'] for x in records}))
            self.assertNotIn('/Game/',json.dumps(records,ensure_ascii=False))
        for season in service.catalog()['challenges']:
            self.assertEqual(len(season['stages']),season['stage_count'])

    def test_experience_and_kongmu_link(self):
        rows = self.client.get('/api/guide/characters/1042?section=experience').json['data']['experience']
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows[1], {'level': 2, 'required_exp': 300})
        self.assertEqual(rows[-1]['required_exp'], 322380)
        self.assertIn('经验需求', self.client.get('/guide/characters/1042.md?section=experience').text)
        html = self.client.get('/guide/characters/1042?section=equip_slots').text
        self.assertIn('href="/kongmu"', html)
        self.assertIn('前往空幕计算', html)

    def test_cultivation_full_roster_and_cost_totals(self):
        for character in service.catalog()['characters']:
            response = self.client.post(f"/api/guide/characters/{character['id']}/cultivation", json={})
            self.assertEqual(response.status_code, 200, character['name'])
            result = response.json['data']
            self.assertEqual(result['experience'], 6492840)
            self.assertEqual(result['level_up_gold'], 1624000)
            self.assertEqual(result['total_gold'], 3967000)
            self.assertEqual(result['overflow_experience'], 3160)
            summed = {}
            for group in result['breakdown']:
                for item in group['items']:
                    key = item['id'].casefold()
                    summed[key] = summed.get(key, 0) + item['amount']
            self.assertEqual(summed, {item['id'].casefold(): item['amount'] for item in result['items']})
            self.assertTrue(all(item['amount'] > 0 and item['name'] for item in result['items']))

    def test_material_cards_and_local_asset_coverage(self):
        from app.modules.guide.presentation import material_metadata
        metadata = material_metadata()
        material_ids = {'gold', 'fons'}

        def collect(value):
            if isinstance(value, dict):
                material_ids.update(str(item['id']).casefold() for item in value.get('items', []))
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        for kind in ('characters', 'arcs'):
            for record in service.catalog()[kind]:
                collect(record['materials'])
        material_ids.update(book['id'].casefold() for book in service.catalog()['cultivation']['exp_books'])
        self.assertFalse(material_ids - metadata.keys())
        for item in metadata.values():
            self.assertIn(item['quality'], ('gray', 'green', 'blue', 'purple', 'orange', 'red'))
            data = (ROOT / 'app' / item['icon'].lstrip('/')).read_bytes()
            self.assertEqual(data[:4], b'RIFF')
            self.assertEqual(data[8:12], b'WEBP')
        for url in ('/guide/characters/1042', '/guide/arcs/fork_PaperPlane',
                    '/guide/characters/1042/cultivation?calculate=1'):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn('guide-material-grid', response.text)
            self.assertIn('guide-quality-purple', response.text)
            self.assertIn('loading="lazy" width="128" height="128"', response.text)

    def test_cultivation_crossed_levels_noop_and_boundaries(self):
        from app.modules.guide import cultivation
        record = service.detail('characters', '1042')
        values = cultivation.defaults(record, service.catalog()['cultivation'])
        for key in list(values):
            if key.startswith('target_'):
                values[key] = values[key.replace('target_', 'current_')]
        url = '/api/guide/characters/1042/cultivation'
        self.assertEqual(self.client.post(url, json=values).json['data']['items'], [])
        values['target_level'] = 2
        result = self.client.post(url, json=values).json['data']
        self.assertEqual(result['experience'], 300)
        self.assertEqual(result['total_gold'], 250)
        self.assertEqual(result['overflow_experience'], 700)
        values['current_exp'] = 200
        self.assertEqual(self.client.post(url, json=values).json['data']['experience'], 300)
        values.update(current_level=20, target_level=20, current_ascended=False, target_ascended=True)
        result = self.client.post(url, json=values).json['data']
        self.assertEqual(result['experience'], 0)
        self.assertEqual(result['total_gold'], 25000)
        self.assertEqual(len(result['breakdown']), 1)
        values['current_ascended'] = True
        self.assertEqual(self.client.post(url, json=values).json['data']['items'], [])
        values['target_level'] = 21
        self.assertGreater(self.client.post(url, json=values).json['data']['experience'], 0)
        automatic = self.client.post(url, json={**values, 'target_level': 31, 'target_breakthrough': 0}).json['data']
        self.assertEqual(automatic['state']['target_breakthrough'], 2)
        self.assertFalse(automatic['state']['target_ascended'])
        for override in [{'current_ascended': 'invalid'}, {'target_level': 20, 'target_ascended': False},
                         {'target_level': 19}, {'target_level': 20.5}, {'current_level': True},
                         {'target_skill_0': 10}, {'target_skill_0': 99}, {'target_level': None}]:
            response = self.client.post(url, json={**values, **override})
            self.assertEqual(response.status_code, 400, override)
        self.assertEqual(self.client.post(url, json=[]).status_code, 400)
        page = self.client.get('/guide/characters/1042/cultivation')
        self.assertEqual(page.status_code, 200)
        self.assertNotIn('name="current_exp"', page.text)
        self.assertNotIn('已突破次数', page.text)
        self.assertIn('<legend>普通攻击</legend>', page.text)
        self.assertIn('<legend>变轨攻击</legend>', page.text)
        self.assertNotIn('掠影', page.text)
        page = self.client.get('/guide/characters/1042/cultivation?calculate=1')
        self.assertIn('3,967,000', page.text)
        self.assertIn('记忆的永恒', page.text)
        invalid = self.client.get('/guide/characters/1042/cultivation?calculate=1&target_level=999')
        self.assertEqual(invalid.status_code, 400)
        self.assertIn('role="alert"', invalid.text)

    def test_cultivation_skill_unlocks_and_city_currency(self):
        from app.modules.guide import cultivation
        record = service.detail('characters', '1042')
        values = cultivation.defaults(record, service.catalog()['cultivation'])
        for key in list(values):
            if key.startswith('target_'):
                values[key] = values[key.replace('target_', 'current_')]
        url = '/api/guide/characters/1042/cultivation'
        values.update(current_level=30, target_level=30, current_ascended=True, target_ascended=True)
        values['target_skill_0'] = 2
        result = self.client.post(url, json=values).json['data']
        self.assertEqual(result['total_gold'], 2000)
        self.assertEqual(result['experience'], 0)
        self.assertEqual(len(result['breakdown']), 1)
        values['target_skill_0'] = 1
        values['target_skill_4'] = 1
        result = self.client.post(url, json=values).json['data']
        self.assertEqual(result['total_gold'], 30000)
        self.assertEqual(len(result['breakdown']), 1)
        values['target_skill_4'] = 0
        values['target_skill_6'] = 1
        result = self.client.post(url, json=values).json['data']
        self.assertEqual(result['total_gold'], 0)
        self.assertEqual({i['id'].casefold(): i['amount'] for i in result['items']},
                         {'fons': 500, 'cityability_upmaterial': 2})

    def test_experience_import_validation(self):
        importer = load_file('guide_refresh_experience', 'scripts/refresh_guide_data.py')
        payload = [{'Rows': {f'character_Upgrade_{i}': {'NeedExp': i * 100} for i in range(1, 81)}}]
        self.assertEqual(importer.character_experience(payload)[-1]['required_exp'], 8000)
        payload[0]['Rows']['character_Upgrade_80']['NeedExp'] = -1
        with self.assertRaises(ValueError):
            importer.character_experience(payload)
        del payload[0]['Rows']['character_Upgrade_80']
        with self.assertRaises(KeyError):
            importer.character_experience(payload)

    def test_cli_encodes_query_and_reports_errors(self):
        cli = load_file('guide_cli', 'app/modules/guide/cli.py')
        response = self.client.get('/api/guide/search?q=黑羽')
        class FakeResponse(io.BytesIO):
            headers = response.headers
        # Use email-style content type access as urllib does.
        from email.message import Message
        headers = Message()
        headers['Content-Type'] = 'application/json'
        fake = FakeResponse(response.data)
        fake.headers = headers
        with patch.object(cli.urllib.request, 'urlopen', return_value=fake) as opening, patch('sys.stdout',new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(['search','黑羽']),0)
            self.assertIn('%E9%BB%91%E7%BE%BD',opening.call_args.args[0])
            self.assertIn('1042',output.getvalue())
        with patch.object(cli.urllib.request,'urlopen',side_effect=OSError('offline')), patch('sys.stderr',new_callable=io.StringIO) as output:
            self.assertEqual(cli.main(['list','arcs']),1)
            self.assertIn('offline',output.getvalue())


if __name__ == '__main__':
    unittest.main()
