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
        for identifier, name, element in [('preview-exe', 'Exe', '暗')]:
            rows = self.client.get('/api/guide/characters?q=' + name + '&element=' + element + '&rarity=5').json['items']
            row = next(row for row in rows if row['id'] == identifier)
            self.assertEqual(row['status'], 'preview')
            for asset in (row['preview_image'], row['assets']['avatar']):
                with self.client.get(asset) as response:
                    self.assertEqual(response.status_code, 200)
        html = self.client.get('/guide').text
        self.assertEqual(html.count('class="guide-preview-badge"'), 1)
        self.assertEqual(len(service.catalog()['characters']), 26)

    def test_detail_is_preview_only_and_readable_in_public_formats(self):
        for identifier in ('preview-exe',):
            path = '/guide/characters/' + identifier
            html = self.client.get(path).text
            self.assertIn('class="guide-preview-hero', html)
            self.assertIn('guide-preview-hero', html)
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
            self.assertEqual(set(overview), expected)
            self.assertEqual(self.client.get('/api/guide/characters/' + identifier + '?ability_id=missing').status_code, 404)
            self.assertEqual(set(entry_links('characters', service.detail('characters', identifier))), {'detail'})
            for suffix in ('/cultivation', '/builds', '/axes/1/compare'):
                self.assertEqual(self.client.get(path + suffix).status_code, 404)
            self.assertEqual(self.client.post('/api/guide/characters/' + identifier + '/cultivation', json={}).status_code, 404)

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
