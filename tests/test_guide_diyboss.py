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
            self.assertRegex(record['begin_date'], r'^2026-\d{2}-\d{2}$')
            self.assertTrue(record['end_at'].endswith('05:59:00+08:00'))
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
        self.assertEqual([r['name'] for r in self.records], ['争锋赏宴', '争锋赏宴', '争锋赏宴·攻坚'])
        self.assertEqual([r['end_date'] for r in self.records], ['2026-07-02', '2026-09-24', '2026-12-17'])
        latest = self.records[-1]
        composite = next(s for s in latest['stages'] if s['id'] == '8')
        self.assertEqual(composite['enemies'][0]['name'], '玛门&瓦楞城堡')
        self.assertEqual(sum(len(s['targets']) for s in latest['stages']), 24)
        self.assertTrue(all(t['score'] > 0 and bool(t['description']) for s in latest['stages'] for t in s['targets']))

    def test_public_routes_search_and_single_stage(self):
        self.assertEqual(self.client.get('/api/guide/diyboss').json['total'], 3)
        index = self.client.get('/guide?kind=diyboss').text
        self.assertIn('2026-06-12 → 2026-07-02', index)
        self.assertIn('2026-08-13 → 2026-09-24', index)
        self.assertIn('2026-11-05 → 2026-12-17', index)
        self.assertNotIn('<code>config-', index)
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

    def test_difficulty_stats_and_multi_enemy_stages(self):
        first = self.records[0]['stages'][0]['difficulties']
        second = self.records[1]['stages'][0]['difficulties']
        self.assertEqual([d['monsters'][0]['level'] for d in first], [42, 52, 62, 72])
        self.assertEqual(first[0]['monsters'][0]['stats']['hp'], 222176)
        self.assertEqual(first[-1]['monsters'][0]['stats']['hp'], 1114432)
        self.assertEqual(second[0]['monsters'][0]['stats']['hp'], 502176)
        self.assertEqual(first[0]['monsters'][0]['weaknesses'], ['lakshana'])
        latest = {stage['id']: stage for stage in self.records[2]['stages']}
        judge = latest['7']['difficulties'][0]['monsters']
        self.assertEqual([m['name'] for m in judge], ['审判官一阶段', '审判官二阶段'])
        self.assertEqual([m['stats']['hp'] for m in judge], [1914598, 2871897])
        self.assertEqual([m['name'] for m in latest['8']['difficulties'][0]['monsters']], ['玛门', '瓦楞城堡'])
        for record in self.records:
            for stage in record['stages']:
                for difficulty in stage['difficulties']:
                    for monster in difficulty['monsters']:
                        self.assertGreater(monster['level'], 0)
                        self.assertGreater(monster['stats']['hp'], 0)

    def test_page_selection_preserves_all_options(self):
        from html.parser import HTMLParser
        class Panels(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stages, self.visible, self.links = [], [], []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'data-diy-stage' in attrs:
                    self.stages.append(attrs)
                    if 'hidden' not in attrs: self.visible.append(attrs)
                if 'data-diy-stage-link' in attrs: self.links.append(attrs)
        response = self.client.get('/guide/diyboss/config-3?stage=7&difficulty=4')
        self.assertEqual(response.status_code, 200)
        page = Panels()
        page.feed(response.text)
        self.assertEqual(len(page.stages), 8)
        self.assertEqual([s['data-diy-stage'] for s in page.visible], ['7'])
        self.assertEqual(len(page.links), 8)
        self.assertIn('id="monsters-7-4" data-diy-difficulty="4" >', response.text)
        self.assertIn('2,542,242', response.text)
        self.assertIn('敌方承伤属性提高20%（包括生命值、防御力、全属性异能伤害抗性）', response.text)
        self.assertIn('目标 1 · 700 分', response.text)
        self.assertNotIn('各项目标的独立积分暂未提供', response.text)
        self.assertNotIn('暂不展示具体效果', response.text)
        self.assertEqual(self.client.get('/guide/diyboss/config-3?difficulty=99').status_code, 404)
        self.assertEqual(self.client.get('/guide/diyboss/config-3?stage=99').status_code, 404)

    def test_importer_uses_case_insensitive_bindings_and_rejects_growth(self):
        import copy
        spec = importlib.util.spec_from_file_location('sync_diyboss_stats', ROOT / 'scripts/sync_guide_diyboss.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = {'AITemplate': {'AssetPathName': '/Game/Monster/Boss_Test.Boss_Test_C'},
                 'AIConfigOverrideID': 'None', 'AISettingsOverrideID': 'None',
                 'bOverrideLevelType': False, 'Count': 1}
        pack = {key: 0 for key in ('HPMaxUp', 'HPMaxAdd', 'AtkUp', 'AtkAdd', 'DefUp', 'DefAdd')}
        pack.update(HPMaxBase=502176, AtkBase=354, DefBase=852)
        for element in ('Normal', 'Cosmos', 'Nature', 'Incantation', 'Chaos', 'Psyche', 'Lakshana', 'Psychically'):
            pack['DamageResist' + element + 'Base'] = .2
        tables = {'clones': {'Stage1': {'CloneSystemSubNodes': [{'DifficultyLevel': 1, 'SpawnID': 'SPAWN'}]}},
                  'spawns': {'spawn': {'SpawnWaveEntriesOverride': [{'SpawnEntries': [entry]}]}},
                  'monsters': {'boss_test': {'bIsCloneDifficultyLevel': True, 'CloneDifficultyLevelArray': [
                      {'MonsterCloneLevel': 42, 'MonsterClonePropModifyID': 'PACK'}],
                      'UnbalBarTypeList': ['UNBAL_BAR_LAKSHANA'], 'TextName': {}, 'Comment': '自定义BOSS-塔吉多'}},
                  'packs': {'pack': pack}}
        resolved = module.difficulty_monsters('stage1', '1', tables)[0]
        self.assertEqual((resolved['name'], resolved['level'], resolved['stats']['hp']), ('塔吉多', 42, 502176))
        self.assertEqual(resolved['weaknesses'], ['lakshana'])
        bad = copy.deepcopy(tables)
        bad['packs']['pack']['HPMaxUp'] = 1
        with self.assertRaisesRegex(ValueError, 'growth'):
            module.difficulty_monsters('stage1', '1', bad)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            module.difficulty_monsters('stage1', '1', None)

    def test_mechanism_descriptions_follow_actual_references(self):
        spec = importlib.util.spec_from_file_location('sync_diyboss_targets', ROOT / 'scripts/sync_guide_diyboss.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        choices = [{'OptionID': 'DiyBoss_Shenpanguan_03'}, {'OptionID': 'DiyBoss_Shenpanguan_01'}]
        options = {choice['OptionID']: {'OptionType': 'EDiyBossOptionType::MechanismBuff', 'OptionNode': None}
                   for choice in choices}
        descriptions = {'diyboss_shenpanguan_01': '效果一', 'diyboss_shenpanguan_03': '效果三'}
        targets = module.mechanism_targets(choices, options, descriptions)
        self.assertEqual([target['description'] for target in targets], ['效果三', '效果一'])
        self.assertTrue(all(target['score'] is None for target in targets))
        with self.assertRaisesRegex(ValueError, 'Missing mechanism description'):
            module.mechanism_targets(choices, options, {})

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
