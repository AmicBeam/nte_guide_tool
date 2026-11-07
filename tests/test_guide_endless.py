"""Endless Trial coverage, floor boundaries and shared guide reading interfaces."""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import unittest
from flask import Flask
from app.modules.guide import blueprint, service
from scripts.sync_guide_endless import build, floor_rules

ROOT = Path(__file__).resolve().parents[1]


class Stages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.visible = []
        self.all = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article' and 'data-challenge-stage' in attrs:
            self.all.append(attrs['data-challenge-stage'])
            if 'hidden' not in attrs:
                self.visible.append(attrs['data-challenge-stage'])


class GuideEndlessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_complete_floor_coverage_and_group_boundaries(self):
        groups = service.catalog()['endless']
        self.assertEqual([(g['floor_begin'], g['floor_end']) for g in groups], [(1, 50), (51, 100), (101, 150), (151, 200)])
        self.assertEqual([s['id'] for g in groups for s in g['stages']], list(range(1, 201)))
        for group in groups:
            self.assertEqual(group['stage_count'], 50)
            for stage in group['stages']:
                self.assertEqual(stage['maximum_time'], 120)
                for wave in stage['halves'][0]['waves']:
                    self.assertTrue(wave['monsters'])
                    for monster in wave['monsters']:
                        self.assertGreater(monster['count'], 0)
                        self.assertGreater(monster['level'], 0)
                        for key in ('hp', 'attack', 'defense'):
                            self.assertGreater(monster['stats'][key], 0)
                        self.assertEqual(len(monster['stats']['resistances']), 8)
                        if monster['image']:
                            response = self.client.get(monster['image'])
                            self.assertEqual(response.status_code, 200)
                            response.close()

    def test_specific_floor_stats_and_individual_hp(self):
        expected = [(1, 'EndlessTrial_1_50', 155100, 2677, 1026, 4),
                    (51, 'EndlessTrial_51_100', 1079316, 2833, 1056, 4),
                    (200, 'EndlessTrial_151_200', 15739284, 5506, 1200, 1)]
        for floor, group, hp, attack, defense, count in expected:
            monster = service.detail('endless', group, stage=floor)['stages'][0]['halves'][0]['waves'][0]['monsters'][0]
            self.assertEqual((monster['stats']['hp'], monster['stats']['attack'], monster['stats']['defense'], monster['count']), (hp, attack, defense, count))
        self.assertIn('单体生命：<strong>155,100</strong>', self.client.get('/guide/endless/EndlessTrial_1_50').text)

    def test_html_defaults_switches_and_deep_links(self):
        for start in (1, 51, 101, 151):
            group = f'EndlessTrial_{start}_{start + 49}'
            for selected in (None, start + 49):
                url = '/guide/endless/' + group + (f'?stage={selected}' if selected else '')
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                parser = Stages()
                parser.feed(response.text)
                self.assertEqual(parser.all, list(map(str, range(start, start + 50))))
                self.assertEqual(parser.visible, [str(selected or start)])
                self.assertNotIn('轮换环线', response.text)
                self.assertNotIn('限时 0 秒', response.text)
                self.assertIn('推荐属性', response.text)
                self.assertIn('选择层数', response.text)
        self.assertEqual(self.client.get('/guide/endless/EndlessTrial_1_50?stage=51').status_code, 404)

    def test_archived_season_entry_and_all_floors(self):
        listing = self.client.get('/guide?kind=endless').text
        self.assertEqual(listing.count('href="/guide/endless/EndlessTrial_Season1"'), 1)
        self.assertNotIn('href="/guide/endless/EndlessTrial_1_50"', listing)
        self.assertIn('第一期', listing)
        self.assertIn('已结束', listing)
        self.assertIn('2026-09-28 05:00', listing)
        self.assertIn('2026-11-05 05:59', listing)
        for floor in (1, 50, 51, 100, 101, 150, 151, 200):
            url = f'/guide/endless/EndlessTrial_Season1?stage={floor}'
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            parser = Stages()
            parser.feed(response.text)
            self.assertEqual(parser.all, list(map(str, range(1, 201))))
            self.assertEqual(parser.visible, [str(floor)])
            self.assertIn('已结束', response.text)
        data = self.client.get('/api/guide/endless/EndlessTrial_Season1?stage=200').json['data']
        self.assertEqual(data['season'], 1)
        self.assertEqual(data['status'], 'ended')
        self.assertEqual(data['stage_count'], 200)
        self.assertEqual(data['begin_at'], '2026-09-28T05:00:00+08:00')
        self.assertEqual(data['end_at'], '2026-11-05T05:59:00+08:00')
        md = self.client.get('/guide/endless/EndlessTrial_Season1.md?stage=200').text
        self.assertIn('第一期', md)
        self.assertIn('2026-11-05T05:59:00+08:00', md)
        self.assertIn('第 200 层', md)
        self.assertEqual(self.client.get('/guide/endless/EndlessTrial_Season1?stage=201').status_code, 404)

    def test_list_search_json_markdown_and_discovery(self):
        listing = self.client.get('/api/guide/endless?limit=2').json
        self.assertEqual(listing['total'], 1)
        self.assertIsNone(listing['next_offset'])
        self.assertEqual(listing['items'][0]['id'], 'EndlessTrial_Season1')
        self.assertEqual(self.client.get('/api/guide/endless?q=讨债人').json['total'], 1)
        response = self.client.get('/api/guide/endless/EndlessTrial_51_100?stage=100').json['data']
        self.assertEqual([s['id'] for s in response['stages']], [100])
        md = self.client.get('/guide/endless/EndlessTrial_51_100.md?stage=100').text
        self.assertIn('讨债人', md)
        self.assertIn('5967427', md)
        self.assertNotIn('第 99 层', md)
        self.assertIn('endless', self.client.get('/api/guide/schema').json['collections'])
        self.assertIn('/api/guide/endless', self.client.get('/guide/llms.txt').text)
        with self.client.get('/guide/cli.py') as response:
            self.assertIn('endless', response.text)
        self.assertEqual(self.client.get('/api/guide/endless/EndlessTrial_1_50?stage=0').status_code, 404)
        self.assertEqual(self.client.get('/api/guide/endless/no-such-group').status_code, 404)

    def test_per_floor_resistance_and_paired_benefit_cost(self):
        record = service.detail('endless', 'EndlessTrial_1_50', stage=1)['stages'][0]
        self.assertEqual(record['rules']['enhanced_resistances'], ['psyche', 'lakshana'])
        self.assertEqual(record['halves'][0]['recommended_elements'], ['nature', 'incantation'])
        self.assertFalse(record['rules']['buff_selection']['enabled'])
        self.assertEqual(record['objectives'][0]['text'], '在60秒时间内通关')
        self.assertEqual(record['objectives'][0]['score'], 200)
        stage = service.detail('endless', 'EndlessTrial_1_50', stage=10)['stages'][0]
        selection = stage['rules']['buff_selection']
        self.assertTrue(stage['rules']['node_last'])
        self.assertTrue(selection['enabled'])
        self.assertEqual(selection['default_count'], 2)
        self.assertEqual(selection['count_bonuses'], [{'floor': 5, 'count': 1}])
        self.assertEqual(len(selection['options']), 5)
        option = selection['options'][0]
        self.assertEqual(option['positive'][0]['description'], '全队的暗属性伤害增加10%')
        self.assertEqual(option['negative'][0]['description'], '怪物的全属性抗性增加8%')
        self.assertEqual(option['negative'][0]['target'], 'monster')
        self.assertEqual(sum(s['rules']['buff_selection']['enabled'] for g in service.catalog()['endless'] for s in g['stages']), 20)
        html = self.client.get('/guide/endless/EndlessTrial_1_50?stage=10').text
        selected = html.split('id="stage-10"', 1)[1].split('<article class="guide-challenge-stage"', 1)[0]
        self.assertIn('关底正向效果与反向效果', selected)
        self.assertIn('<th>正向效果</th><th>反向效果</th>', selected)
        self.assertIn('默认正向效果数量：2', selected)
        self.assertIn('怪物的全属性抗性增加8%', selected)
        self.assertIn('限时 120 秒', selected)
        self.assertIn('强化抗性', selected)
        md = self.client.get('/guide/endless/EndlessTrial_1_50.md?stage=10').text
        self.assertIn('怪物的全属性抗性增加8%', md)
        self.assertIn('正向效果', md)
        self.assertIn('反向效果', md)

    def test_importer_rejects_missing_floor_pack_and_growth(self):
        pack = {key: 0 for key in ('HPMaxUp', 'HPMaxAdd', 'AtkUp', 'AtkAdd', 'DefUp', 'DefAdd')}
        pack.update(HPMaxBase=123, AtkBase=45, DefBase=67, UnbalMax=10)
        pack.update({'DamageResist' + element.title() + 'Base': .1 for element in ('normal', 'cosmos', 'nature', 'incantation', 'chaos', 'psyche', 'lakshana', 'psychically')})
        entry = {'AITemplate': {'AssetPathName': '/Game/Test.Test_C'}, 'AISettingsOverrideID': 'test-pack',
                 'bOverrideLevelType': True, 'SpawnLevelTypeOverride': 'EAISpawnLevelType::PermanentLevel', 'LevelOverride': 71, 'Count': 4}
        wave = {'SpawnEntries': [entry], 'WaveDelayTime': 0}
        spawns = [{'Type': 'DataTable', 'Rows': {f'EndlessTrial_Floor_{i}': {'SpawnWaveEntriesOverride': [deepcopy(wave)]} for i in range(1, 201)}}]
        packs = [{'Type': 'DataTable', 'Rows': {'test-pack': pack}}]
        identity = {'/Game/Test': {'name': 'Test', 'image': None, 'swallowable': None}}
        result = build(spawns, packs, identity)
        self.assertEqual(result['groups'][0]['stages'][0]['halves'][0]['waves'][0]['monsters'][0]['stats']['hp'], 123)
        self.assertEqual(result['groups'][-1]['stages'][-1]['id'], 200)
        broken = deepcopy(spawns)
        del broken[0]['Rows']['EndlessTrial_Floor_200']
        with self.assertRaises(ValueError):
            build(broken, packs, identity)
        with self.assertRaises(KeyError):
            build(spawns, [{'Type': 'DataTable', 'Rows': {}}], identity)
        pack['HPMaxUp'] = 1
        with self.assertRaises(ValueError):
            build(spawns, packs, identity)


if __name__ == '__main__':
    unittest.main()
