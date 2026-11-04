"""Announcements share discovery but never expose incomplete character tools."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from app.modules.guide import service, previews
from app.modules.guide.chat_links import entry_links
from pathlib import Path
from flask import Flask
from app.modules.guide import blueprint


class GuidePreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'app/templates'),
                    static_folder=str(Path(__file__).resolve().parents[1] / 'app/static'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_discovery_filters_and_local_assets(self):
        for identifier, name, element in [('preview-exe', 'Exe', '暗'), ('1090', '曳尔沐', '光')]:
            rows = self.client.get('/api/guide/characters?q=' + name + '&element=' + element + '&rarity=5').json['items']
            row = next(row for row in rows if row['id'] == identifier)
            self.assertEqual(row['status'], 'preview')
            for asset in (row['preview_image'], row['assets']['avatar']):
                with self.client.get(asset) as response:
                    self.assertEqual(response.status_code, 200)
        html = self.client.get('/guide').text
        self.assertEqual(html.count('class="guide-preview-badge"'), 2)
        self.assertEqual(len(service.catalog()['characters']), 25)

    def test_detail_is_preview_only_and_readable_in_public_formats(self):
        for identifier in ('1090', 'preview-exe'):
            path = '/guide/characters/' + identifier
            html = self.client.get(path).text
            self.assertIn('guide-character-intro' if identifier == '1090' else 'class="guide-preview-hero', html)
            self.assertEqual('guide-preview-hero' in html, identifier == 'preview-exe')
            self.assertIn('所需升级材料', html)
            self.assertIn('空心花籽', html)
            hidden = ['角色定位', '养成计算器']
            if identifier == 'preview-exe':
                hidden.extend(['属性数值', 'guide-toc', 'type="range"'])
            for text in hidden:
                self.assertNotIn(text, html)
            self.assertEqual(self.client.get(path + '.md').status_code, 200)
            overview = self.client.get('/api/guide/characters/' + identifier + '?section=overview').json['data']
            expected = {'id', 'name', 'status', 'preview_image', 'materials'}
            if identifier == '1090':
                expected.add('available_sections')
            self.assertEqual(set(overview), expected)
            self.assertEqual(self.client.get('/api/guide/characters/' + identifier + '?ability_id=missing').status_code, 404)
            self.assertEqual(set(entry_links('characters', service.detail('characters', identifier))), {'detail'})
            for suffix in ('/cultivation', '/builds', '/axes/1/compare'):
                self.assertEqual(self.client.get(path + suffix).status_code, 404)
            self.assertEqual(self.client.post('/api/guide/characters/' + identifier + '/cultivation', json={}).status_code, 404)

    def test_elyms_verified_skill_sections_keep_incomplete_tools_closed(self):
        response = self.client.get('/api/guide/characters/1090?section=reference_details')
        self.assertEqual(response.status_code, 200)
        data = response.json['data']['reference_details']
        self.assertEqual(len(data['abilities']), 7)
        self.assertEqual(len(data['awaken']), 6)
        self.assertEqual([r['resonance_level'] for r in data['resonance']], [3, 6])
        skill = next(a for a in data['abilities'] if a['id'] == 'GA_Elyms_Skill')
        row = next(r for r in skill['param_rows'] if r['label'] == '「离去的人们」伤害倍率')
        self.assertEqual(len(row['values']), 13)
        self.assertTrue(row['hit_details'])
        html = self.client.get('/guide/characters/1090').text
        for text in ('沉入电子梦', '错频故障', '404 遥望一座孤岛', '拜启，我的旧友', '基础面板暂未提供'):
            self.assertIn(text, html)
        self.assertIn('guide-character-header', html)
        self.assertIn('guide-character-overview', html)
        self.assertIn('guide-toc', html)
        self.assertIn('alt="光属性"', html)
        self.assertIn('alt="5星角色"', html)
        self.assertNotIn('guide-preview-hero', html)
        self.assertNotIn('曳尔沐 · 预告', html)
        self.assertIn('沉入电子梦', self.client.get('/guide/characters/1090.md').text)
        for ability in data['abilities']:
            self.assertEqual(self.client.get(ability['icon']).status_code, 200)

    def test_elyms_includes_both_breakthrough_and_skill_material_series(self):
        record = self.client.get('/api/guide/characters/1090?section=overview').json['data']
        materials = record['materials']
        breakthrough = materials['breakthrough'][0]['items']
        skills = materials['skills'][0]['levels'][0]['items']
        self.assertEqual([(i['id'],i['name']) for i in breakthrough], [
            ('worldboss_material_07','海之泪'),
            ('ordinarymonmaterial_02_lv1','褪色掠影'),
            ('ordinarymonmaterial_02_lv2','失焦掠影'),
            ('ordinarymonmaterial_02_lv3','晦暗掠影')])
        self.assertEqual([(i['id'],i['name']) for i in skills], [
            ('weeklycloneboss_a04_01','空心花籽'),
            ('skillupmaterial_01_lv1','雏鸟的希冀'),
            ('skillupmaterial_01_lv2','白鸽的振翅'),
            ('skillupmaterial_01_lv3','「橄榄枝」')])
        self.assertTrue(all('amount' not in i for i in breakthrough+skills))
        html = self.client.get('/guide/characters/1090').text
        markdown = self.client.get('/guide/characters/1090.md').text
        for name in ('海之泪','晦暗掠影','「橄榄枝」','空心花籽'):
            self.assertIn(name,html)
        for item in breakthrough+skills:
            self.assertIn(item['name'],markdown)
        from app.modules.guide.presentation import material_display
        for item in breakthrough+skills:
            display = material_display(item)
            self.assertTrue(display['icon'])
            self.assertEqual(self.client.get(display['icon']).status_code,200)

    def test_optional_materials_and_other_fields_never_render(self):
        rows = deepcopy(previews.characters())
        row = rows[0]
        row.update(abilities=[{'name': '隐藏技能'}], stats=[{'name': '隐藏属性'}],
                   char_tags=[{'name': '隐藏定位'}])
        row['materials'] = {'breakthrough': [{'items': [
            {'id': 'test-material', 'name': '已确认材料', 'amount': 9}]}], 'skills': []}
        with patch.object(previews, 'characters', return_value=rows):
            html = self.client.get('/guide/characters/' + row['id']).text
        self.assertIn('所需升级材料', html)
        self.assertIn('已确认材料', html)
        self.assertNotIn('×9', html)
        for text in ('隐藏技能', '隐藏属性', '隐藏定位', '养成计算器'):
            self.assertNotIn(text, html)

    def test_complete_record_replaces_matching_preview(self):
        row = {**previews.characters()[0], 'id': '1042'}
        with patch.object(previews, 'characters', return_value=[row]):
            records = [r for r in service.public_records('characters') if r['id'] == '1042']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['name'], '黑羽')


if __name__ == '__main__':
    unittest.main()
