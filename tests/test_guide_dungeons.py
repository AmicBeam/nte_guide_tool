"""Dungeon navigation, offline rewards and empirical yield semantics."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from urllib.parse import urlsplit

from flask import Flask
from app.modules.guide import blueprint, service, dungeons, presentation, cli, chat_tools

ROOT = Path(__file__).resolve().parents[1]


class GuideDungeonsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'), static_folder=str(ROOT / 'app/static'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_directory_search_and_all_detail_formats(self):
        self.assertTrue('副本 <span>6</span>' in self.client.get('/guide').text)
        self.assertEqual(self.client.get('/guide/dungeons').location, '/guide?kind=dungeons')
        self.assertEqual(self.client.get('/api/guide/dungeons').json['total'], 6)
        self.assertIn('dungeons', cli.KINDS)
        self.assertIn('dungeons', chat_tools.KINDS['enum'])
        self.assertIn('/api/guide/dungeons', self.client.get('/guide/llms.txt').text)
        self.assertEqual(service.search('dungeons', query='橄榄枝')[0]['id'], 'skills')
        for record in dungeons.records():
            path = '/guide/dungeons/' + record['id']
            self.assertEqual(self.client.get(path).status_code, 200, path)
            self.assertEqual(self.client.get('/api' + path).json['data']['name'], record['name'])
            self.assertIn('单倍领取数量参考', self.client.get(path + '.md').text)
        for path in ['/guide/dungeons/missing', '/api/guide/dungeons/missing',
                     '/guide/dungeons/Trailclone_exp?stage=999']:
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_material_links_and_multiple_locations(self):
        target = dungeons.material_url('SkillUpMaterial_01_lv3')
        self.assertIn('/guide/dungeons/skills?', target)
        self.assertIn('material=skillupmaterial_01_lv3', target)
        matches = self.client.get('/api/guide/dungeons?material=SkillUpMaterial_01_lv3').json
        self.assertEqual([r['id'] for r in matches['items']], ['skills'])
        self.assertEqual(self.client.get('/api/guide/dungeons?category=技能材料').json['total'], 1)
        self.assertIsNone(dungeons.material_url('not-a-material'))
        for character in service.catalog()['characters']:
            html = self.client.get('/guide/characters/' + character['id']).text
            for item in presentation.upgrade_material_preview(character):
                url = dungeons.material_url(item['id'])
                if url:
                    self.assertIn(url.replace('&', '&amp;'), html)
                    self.assertEqual(self.client.get(urlsplit(url).path + '?' + urlsplit(url).query).status_code, 200)
        gold_url = dungeons.material_url('GOLD')
        self.assertIn('kind=dungeons', gold_url)
        html = self.client.get(gold_url).text
        self.assertIn('正在查找', html)
        self.assertIn('硬币记', html)
        self.assertNotIn('异象追猎「塞润尼缇」', html)
        empty = self.client.get('/guide?kind=dungeons&material=unknown').text
        self.assertIn('没有找到对应副本', empty)

    def test_fixed_counts_hidden_counts_and_stage_selection(self):
        record = service.detail('dungeons', 'Trailclone_gold')
        self.assertEqual([next(r['amount'] for r in s['rewards'] if r['id'] == 'gold')
                          for s in record['stages']], [40000, 60000, 80000, 96000, 108000, 120000])
        material = service.detail('dungeons', 'DLC_talent_001', stage=7)['stages'][0]
        self.assertEqual(material['stamina'], 40)
        self.assertTrue(all(r['amount'] is None for r in material['rewards'] if r['id'].startswith('skillup')))
        html = self.client.get('/guide/dungeons/Trailclone_gold?stage=4').text
        self.assertIn('×96,000', html)
        self.assertNotIn('×120,000', html)
        weekly = service.detail('dungeons', 'WeeklyClone_Boss05')
        self.assertTrue(all(s['stamina'] == 60 for s in weekly['stages']))

    def test_quantities_are_merged_and_ranges_are_not_means(self):
        for identifier, level, expected in [('DLC_forkbreakthrough_001', 40, '约1–3个'),
                ('DLC_talent_001', 40, '约2–3个'), ('worldboss_01_drop', 40, '约6–7个'),
                ('WeeklyClone_Boss05', 30, '约2–3个')]:
            context = dungeons.page_context(dungeons.find(identifier), material='')
            stage = next(s for s in context['selected_encounter']['stages'] if s['required_level'] == level)
            quantities = [r['quantity'] for r in stage['rewards']]
            self.assertIn(expected, quantities)
            self.assertNotIn('6.5', str(quantities))
        html = self.client.get('/guide/dungeons/skills?encounter=DLC_talent_001&stage=5').text
        self.assertIn('约2–3个', html)
        self.assertNotIn('实测参考', html)
        data = self.client.get('/api/guide/dungeons/skills?encounter=DLC_talent_001&stage=5').json['data']
        self.assertEqual(len(data['encounters']), 1)
        self.assertEqual(data['encounters'][0]['stages'][0]['required_level'], 40)
        self.assertEqual(self.client.get('/guide/dungeons/skills?encounter=WeeklyClone_Boss05').status_code, 404)
        self.assertEqual(self.client.get('/api/guide/dungeons/skills?stage=5').status_code, 400)

    def test_importer_does_not_turn_preview_numbers_into_fixed_rewards(self):
        spec = importlib.util.spec_from_file_location('dungeon_import', ROOT / 'scripts/sync_guide_dungeons.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = {'ItemID': 'material', 'Num': 1, 'SequenceNumType': 'EDropSequenceNumType::DROPSEQUENCENUMTYPE_FIXED', 'LimitLevel': ''}
        group = {'SequenceId': 'sequence', 'HiddenCounts': False, 'DropConditions': [],
                 'SequenceWeight': 1, 'MinLevel': 0, 'MaxLevel': 100, 'ModifyNum': 10}
        tables = {'clones': {'clone': {'CloneType': 'ECloneType::Talent_Clone', 'CloneName': {'LocalizedString': '技能本'},
                    'CloneSystemSubNodes': [{'DropID': 'drop', 'DifficultyLevel': 1, 'TeamLevel': 30,
                                            'StaminaCount': 40, 'CloneDesArray': []}]}},
                  'groups': {'drop_0': group}, 'sequences': {'sequence_0': entry},
                  'items': {'material': {'ItemName': {'LocalizedString': '材料'}}},
                  'equipment': {}, 'currencies': {}, 'sources': {}, 'details': {}}
        self.assertEqual(module.build(tables)[0]['stages'][0]['rewards'][0]['amount'], 10)
        for field, value in [('HiddenCounts', True), ('DropConditions', ['condition']), ('SequenceWeight', .5)]:
            variant = deepcopy(tables)
            variant['groups']['drop_0'][field] = value
            self.assertIsNone(module.build(variant)[0]['stages'][0]['rewards'][0]['amount'])
        variant = deepcopy(tables)
        variant['sequences'] = {}
        with self.assertRaises(ValueError):
            module.build(variant)
