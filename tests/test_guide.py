from tests.guide_release_fixtures import published_character_data
import importlib.util
import io
import re
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@published_character_data
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

    def test_combat_details_require_a_valid_login(self):
        paths = (
            ('/guide/characters/1010', '判定与时长'),
            ('/guide/characters/1010.md?ability_id=GA_Nanally_Skill', '判定与时长'),
            ('/api/guide/characters/1010?section=abilities&skill_level=1&hit_details=0', 'combat_details'),
            ('/api/guide/characters/1010?ability_id=GA_Nanally_Skill', 'combat_details'),
        )
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', side_effect={'valid': player}.get):
            for path, marker in paths:
                with self.subTest(path=path):
                    for token in ('valid', None, 'expired'):
                        headers = {'Authorization': f'Bearer {token}'} if token else {}
                        response = self.client.get(path, headers=headers)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.headers['Cache-Control'], 'no-store')
                        self.assertEqual(marker in response.text, token == 'valid')
            self.assertNotIn('判定半径 400', self.client.get('/guide?kind=characters').text)

    def test_special_combo_followups_render_in_combat_details(self):
        endpoint = '/api/guide/characters/1042?ability_id=GA_BlackBird_UltraSkill'
        followup = '接魔女形态「祭火」第 3 段。'
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', return_value=player):
            headers = {'Authorization': 'Bearer valid'}
            authorized = self.client.get(endpoint, headers=headers).json['data']
            groups = authorized['abilities'][0]['groups']
            combat = next(group['combat_details'] for group in groups if group['name'] == '恳请众魂安息')
            self.assertIn(followup, combat['notes'])
            self.assertIn('3.15–5.667', combat['rows'][0])
            self.assertIn(followup, self.client.get('/guide/characters/1042', headers=headers).text)
            self.assertIn(followup, self.client.get('/guide/characters/1042.md', headers=headers).text)
            self.assertNotIn(followup, self.client.get(endpoint).text)

    def test_akane_combat_timing_is_bound_to_combat_moves(self):
        path = '/api/guide/characters/1057?section=abilities&hit_details=0'
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', return_value=player):
            headers = {'Authorization': 'Bearer valid'}
            response = self.client.get(path, headers=headers)
            html = self.client.get('/guide/characters/1057', headers=headers).text
            markdown = self.client.get('/guide/characters/1057.md?ability_id=GA_Akane_Melee', headers=headers).text
        groups = {group['name']: group['combat_details'] for ability in response.json['data']['abilities']
                  for group in ability['groups'] if 'combat_details' in group}
        self.assertEqual(set(groups), {'普通攻击：弦刃', '普通攻击：舞台跳水', '极限反击：羽刃回响',
                                      '变轨技能：律动音浪', '音障爆鸣', '变奏和弦'})
        for table in groups.values():
            self.assertEqual(table['columns'][0], '动作')
            self.assertEqual(table['columns'][-1], '伤害判定')
            self.assertTrue(all(len(row) == len(table['columns']) for row in table['rows']))
        normal = groups['普通攻击：弦刃']
        window = normal['columns'].index('连段窗口（秒）')
        self.assertEqual([row[window] for row in normal['rows']],
                         ['0.150–0.889', '0.172–0.494', '0.073–1.036', '0.500–0.760', None])
        self.assertEqual(normal['rows'][-1], ['完整循环', None, '2.775', '0.180', None])
        self.assertIn('解除暂停事件（秒）', groups['音障爆鸣']['columns'])
        self.assertEqual(groups['音障爆鸣']['rows'][0][:3], ['终结第一段', '4.25', '0.020*10'])
        self.assertIn('切人锁定（秒）', groups['变奏和弦']['columns'])
        self.assertIn('<th scope="row">强化短 E</th>', html)
        self.assertIn('| 第 3 段 | 0.073–1.036 | 1.080 | 0.020*7 | 盒体半尺寸 600 × 270 × 200 |', markdown)
        self.assertNotIn('combat_details', self.client.get(path).text)

    def test_exe_assist_combat_details_include_all_fixed_hit_pauses(self):
        path = '/guide/characters/1074.md?ability_id=GA_Exe_QTE'
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', return_value=player):
            markdown = self.client.get(path, headers={'Authorization': 'Bearer valid'}).text
        self.assertIn('判定与时长', markdown)
        self.assertIn('| 援护攻击 | 1.767 |', markdown)
        self.assertIn('| 0.033*2 / 0.300 | 0–1.189 | 盒体半尺寸 350 × 350 × 300 |', markdown)
        self.assertNotIn('判定与时长', self.client.get(path).text)

    def test_exe_normal_combo_uses_hit_adjusted_transition_time(self):
        table = service._skill_combat_details()['1074']['GA_Exe_Melee']['普通攻击：物理层侵入']
        columns, rows = table['columns'], table['rows']
        self.assertNotIn('连段占时推算（秒）', columns)
        transition = columns.index('动作占时（秒）')
        by_action = {row[0]: row[transition] for row in rows}
        self.assertEqual(by_action['第 1 段'], '0.45')
        self.assertEqual(by_action['第 2 段'], '0.633')
        self.assertEqual(by_action['第 5 段'], '0.619')

    def test_action_occupancy_uses_combo_first_and_includes_fixed_spawn_pauses(self):
        tables = service._skill_combat_details()['1042']
        ultimate = tables['GA_BlackBird_UltraSkill']['恳请众魂安息']
        self.assertEqual(ultimate['rows'][0][ultimate['columns'].index('动作占时（秒）')], '5.653')
        attacks = tables['GA_BlackBird_Melee']['普通攻击：祭火']
        turn = dict(zip(attacks['columns'], next(row for row in attacks['rows'] if row[0] == '第 5 段')))
        self.assertEqual(turn['动作占时（秒）'], '1.358')
        self.assertEqual(turn['命中顿帧（秒）'], '0.03*6')

    def test_directional_counter_durations_match_shaft_rounding(self):
        from decimal import Decimal, ROUND_CEILING

        tables = service._skill_combat_details()
        actions = json.loads((ROOT / 'app/modules/shaft/static/data/actions.json').read_text())
        counters = {(action['character_name'], action['name']): action['duration_ticks'] for action in actions}
        for role, cid, ability, group, action in (
            ('Exe', '1074', 'GA_Exe_Melee', '极限反击：零延迟响应', '闪反'),
            ('明音凛', '1057', 'GA_Akane_Melee', '极限反击：羽刃回响', '闪反'),
            ('黑羽', '1042', 'GA_BlackBird_Melee', '极限反击：生于灾厄', '魔闪反'),
            ('九原', '1055', 'GA_Kuhara_Melee', '极限反击：局势逆转', '闪反'),
        ):
            with self.subTest(role=role, action=action):
                table = tables[cid][ability][group]
                self.assertEqual([row[0] for row in table['rows']], ['前闪反击', '后闪反击'])
                duration = table['columns'].index('动作占时（秒）')
                slower = max(Decimal(row[duration]) for row in table['rows'])
                self.assertEqual(counters[role, action], int((slower * 10).to_integral_value(rounding=ROUND_CEILING)))
        self.assertEqual([row[1] for row in tables['1074']['GA_Exe_Melee']['极限反击：零延迟响应']['rows']],
                         ['1.194', '1.277'])
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', return_value=player):
            markdown = self.client.get('/guide/characters/1074.md?ability_id=GA_Exe_Melee',
                                       headers={'Authorization': 'Bearer valid'}).text
        self.assertIn('| 前闪反击 | 1.194 |', markdown)
        self.assertIn('| 后闪反击 | 1.277 |', markdown)
        self.assertNotIn('触发延迟', markdown)

    def test_combat_snapshot_covers_formal_characters(self):
        tables = service._skill_combat_details()
        characters = service.catalog()['characters']
        self.assertEqual(set(tables), {str(character['id']) for character in characters})
        for character in characters:
            bindings = {ability['id']: {group['name'] for group in ability['groups']}
                        for ability in character['abilities']}
            skills = tables[str(character['id'])]
            with self.subTest(character=character['name']):
                for ability in character['abilities']:
                    if ability['type_name'] not in ('Melee', 'Skill', 'UltraSkill', 'QTE'):
                        continue
                    columns = {column for table in skills[ability['id']].values() for column in table['columns']}
                    self.assertTrue(any('秒' in column for column in columns))
                    self.assertTrue(columns & {'伤害判定', '盒体半尺寸', '球形判定半径', '球形半径'})
                for ability, groups in skills.items():
                    self.assertTrue(groups.keys() <= bindings[ability])
                    for table in groups.values():
                        self.assertEqual(table['columns'][0], '动作')
                        self.assertFalse({'动作衔接（秒）', '连段占时推算（秒）'} & set(table['columns']))
                        self.assertNotIn('动画总长（秒）', table['columns'])
                        self.assertTrue(table['rows'])
                        self.assertTrue(all(len(row) == len(table['columns']) for row in table['rows']))

    def test_combat_snapshot_includes_separate_abilities_and_named_moves(self):
        tables = service._skill_combat_details()
        for cid, skill, groups in (
            ('1023', 'GA_Cang_Melee', {'普通攻击：无状之言', '普通攻击：无讳之语'}),
            ('1025', 'GA_Hathor_Skill', {'变轨技能：轮转打击', '持续'}),
            ('1036', 'GA_Zankou_Melee', {'普通攻击：魇舞', '普通攻击：落月'}),
            ('1042', 'GA_BlackBird_Melee', {'普通攻击：祭火', '普通攻击：灼痕',
                                          '普通攻击：予尔等之恩泽', '普通攻击：赐众生之福音'}),
            ('1042', 'GA_BlackBird_Skill', {'渡世死焰'}),
            ('1042', 'GA_BlackBird_UltraSkill', {'万火归一'}),
            ('1054', 'GA_Daffodill_Skill', {'变轨技能：同振', '变轨技能：如响斯应'}),
            ('1070', 'GA_Mitsuki_Melee', {'普通攻击：顿音'}),
            ('1071', 'GA_Chaos071_Melee', {'普通攻击：裁断'}),
            ('1075', 'GA_Oneiroi_Melee', {'无意识响应'}),
            ('1090', 'GA_Elyms_Melee', {'错频故障'}),
        ):
            with self.subTest(skill=skill):
                self.assertTrue(groups <= tables[cid][skill].keys())

        elyms = tables['1090']['GA_Elyms_Melee']['错频故障']
        self.assertEqual([row[:3] for row in elyms['rows'][:3]], [
            ['第 1 段', '0.085–0.568', '0.451'],
            ['第 2 段', '0.082–0.54', '0.382'],
            ['第 3 段', None, '0.934'],
        ])
        self.assertEqual([row[0] for row in tables['1090']['GA_Elyms_UltraSkill']['一如昨日光景']['rows']],
                         ['对群终结', '对单终结'])

    def test_character_icon_filters_combine_and_reset(self):
        page = self.client.get('/guide?kind=characters&element=魂&rarity=5&arc_type=4')
        self.assertEqual(page.status_code, 200)
        self.assertIn('name="arc_type"', page.text)
        self.assertIn('type="radio"', page.text)
        self.assertNotIn('<select name="element"', page.text)
        self.assertEqual(page.text.count('data-character-card'), 29)
        self.assertIn('data-filter-text=', page.text)
        result = self.client.get('/api/guide/characters?element=魂&rarity=5&arc_type=4').json
        self.assertGreater(result['total'], 0)
        for item in result['items']:
            self.assertEqual(item['element_name'], '魂')
            self.assertEqual(item['rarity'], 5)
            self.assertEqual(service.detail('characters', item['id'])['arc_type']['id'], '4')
        self.assertEqual(self.client.get('/api/guide/characters?arc_type=').json['total'], 29)
        self.assertEqual(self.client.get('/api/guide/characters?arc_type=unknown').json['total'], 0)
        self.assertTrue(any(item['id'] == '1042' for item in
            self.client.get('/api/guide/characters?q=黑羽&arc_type=4').json['items']))

    def test_arc_icon_filters_and_rarity_images(self):
        html = self.client.get('/guide?kind=arcs&rarity=3&arc_type=1').text
        self.assertEqual(html.count('data-filter-card '), 53)
        self.assertIn('aria-label="3星"', html)
        self.assertIn('data-guide-filter-group', html)
        self.assertNotIn('class="guide-arc-types"', html)
        matches = self.client.get('/api/guide/arcs?rarity=3&arc_type=1').json['items']
        self.assertGreater(len(matches), 0)
        for arc in matches:
            self.assertEqual(arc['rarity'], 3)
            self.assertEqual(arc['type_name'], '固态')
        for arc in service.catalog()['arcs']:
            self.assertTrue((ROOT / 'app' / arc['rarity_icon'].removeprefix('/')).is_file())
            self.assertIn(arc['rarity_icon'], self.client.get('/guide/arcs/' + arc['id']).text)
        self.assertNotRegex(self.client.get('/guide?kind=arcs').text, r'[345]★')

    def test_narrow_reads_and_numbers(self):
        full = self.client.get('/api/guide/characters/1042').json['data']
        narrowed = self.client.get('/api/guide/characters/1042?section=abilities').json['data']
        self.assertEqual(narrowed['abilities'], full['abilities'])
        self.assertNotIn('stats', narrowed)
        challenge = self.client.get('/api/guide/challenges/Abyss_12?stage=12').json['data']
        self.assertEqual([x['id'] for x in challenge['stages']], [12])
        stats = challenge['stages'][0]['halves'][0]['waves'][0]['monsters'][0]['stats']
        self.assertGreater(stats['hp'], 0)
        self.assertIsInstance(stats['resistances']['cosmos'], (int, float))
        md = self.client.get('/guide/characters/1042.md?section=awaken')
        self.assertEqual(md.mimetype, 'text/markdown')
        self.assertIn('叛逃者', md.text)
        self.assertNotIn('<NumGreen>', md.text)
        self.assertEqual(self.client.get('/guide/challenges/Abyss_12?stage=').status_code, 200)
        stage_html = self.client.get('/guide/challenges/Abyss_12?stage=12').text
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
        self.assertIn('扭曲数符', materials)
        self.assertNotIn('id="section-materials"', materials)
        self.assertNotIn('材料 ID', materials)

    def test_character_overview_merges_stats_and_highest_materials(self):
        from app.modules.guide import presentation
        record = service.detail('characters', '1042')
        before = json.dumps(record['materials'], ensure_ascii=False)
        preview = presentation.upgrade_material_preview(record)
        self.assertEqual({item['name'] for item in preview}, {'扭曲数符', '斑斓的票根', '「第二人」', '记忆的永恒'})
        self.assertEqual(before, json.dumps(record['materials'], ensure_ascii=False))
        self.assertTrue(all(set(item) == {'id', 'name'} for item in preview))
        html = self.client.get('/guide/characters/1042').text
        self.assertEqual(html.count('id="section-stats"'), 1)
        self.assertEqual(html.count('id="stat-level"'), 1)
        self.assertIn('value="80"', html)
        self.assertIn('暴击伤害', html)
        self.assertNotIn('id="section-materials"', html)
        self.assertNotIn('id="section-experience"', html)
        self.assertNotIn('href="#section-materials"', html)
        self.assertNotIn('href="#section-experience"', html)
        section = html.split('class="guide-overview-materials"', 1)[1].split('</section>', 1)[0]
        self.assertNotIn('数量', section)
        self.assertNotIn('模糊数符', section)
        self.assertIn('记忆的永恒', section)
        for character in service.catalog()['characters']:
            items = presentation.upgrade_material_preview(character)
            self.assertTrue(items)
            self.assertEqual(len({item['id'].lower().rsplit('_lv', 1)[0] for item in items}), len(items))
        self.assertIn('experience', self.client.get('/api/guide/characters/1042?section=experience').json['data'])

    def test_arc_uses_shared_overview_without_duplicate_sections(self):
        from app.modules.guide import presentation
        character = service.detail('characters', '1042')
        html = self.client.get('/guide/characters/1042').text
        overview = html.split('class="guide-character-overview"', 1)[1].split('</section>', 1)[0]
        self.assertNotIn(character['ability_name'], overview)
        self.assertNotIn('<span>异能</span>', overview)
        for arc in service.catalog()['arcs']:
            preview = presentation.upgrade_material_preview(arc)
            self.assertEqual(len(preview), 2)
            self.assertTrue(all(item['id'].lower().endswith('_lv3') for item in preview))
        page = self.client.get('/guide/arcs/fork_PaperPlane').text
        self.assertEqual(page.count('id="stat-level"'), 1)
        self.assertEqual(page.count('id="section-stats"'), 1)
        self.assertNotIn('id="section-materials"', page)
        self.assertNotIn('href="#section-materials"', page)
        self.assertIn('弧盘属性与升级材料', page)
        self.assertIn('guide-overview-materials', page)
        self.assertLess(page.index('id="section-stats"'), page.index('id="section-effect"'))
        self.assertIn('value="80"', page)
        api = self.client.get('/api/guide/arcs/fork_PaperPlane?section=materials').json['data']
        self.assertTrue(api['materials'])

    def test_skill_default_level_matches_server_values_and_short_caps(self):
        page = self.client.get('/guide/characters/1042?section=abilities').text
        skill = page.split('id="skill-1"', 1)[1].split('</article>', 1)[0]
        self.assertRegex(skill, r'id="skill-level-1"[^>]*value="10"')
        ability = service.detail('characters', '1042')['abilities'][0]
        row = ability['groups'][0]['param_rows'][0]
        self.assertIn('>' + row['values'][9] + '</td>', skill)
        self.assertIn('data-level-inline>10</span>', skill)
        from flask import render_template_string
        row = {'label': '倍率', 'values': ['10%', '20%', '30%'], 'hit_details': []}
        short = {'name': '短等级技能', 'type_name': 'Skill', 'param_rows': [row],
                 'groups': [{'name': '招式', 'description': '', 'param_rows': [row]}]}
        with self.client.application.app_context():
            html = render_template_string("{% from 'guide/components.html' import skill %}{{ skill(ability, 1) }}", ability=short)
        self.assertIn('value="3"', html)
        self.assertIn('>30%</td>', html)

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
        melee = nanally['abilities'][0]
        punch = next(g for g in melee['groups'] if g['name'] == '普通攻击：重拳出击！')
        keys = [f'GA_Nanally_ExtraAtk{stage}_num' for stage in range(1, 4)]
        self.assertEqual([r['key'] for r in punch['param_rows']], keys)
        self.assertEqual([len(r['hit_details']) for r in punch['param_rows']], [2, 2, 1])
        self.assertNotIn('强力普攻', [g['name'] for g in melee['groups']])
        importer = load_file('guide_punch_grouping', 'scripts/guide_skill_details.py')
        for row in punch['param_rows']:
            phase = importer.select_group(row, melee['phases'])
            self.assertEqual(melee['phases'][phase]['name'], punch['name'])
        html = self.client.get('/guide/characters/1042?section=abilities').text
        self.assertIn('详细命中数据', html)
        self.assertIn('基础回能', html)
        self.assertIn('0.838', html)

    def test_lacrimosa_normal_attacks_group_stats_and_combat_data_by_form(self):
        importer = load_file('guide_lacrimosa_attack_grouping', 'scripts/guide_skill_details.py')
        ability_id = 'GA_Lacrimosa_Melee'
        path = f'/api/guide/characters/1004?ability_id={ability_id}'
        player = SimpleNamespace(shaft_test_whitelisted=False)
        with patch('app.dao.get_player_by_token', return_value=player):
            response = self.client.get(path, headers={'Authorization': 'Bearer valid'})
        groups = {group['name']: group for group in response.json['data']['abilities'][0]['groups']}
        for name, prefix, damage_rows in (
            ('普通攻击：茄汁金属乐', '近战普攻', 6),
            ('普通攻击：茄汁打击乐', '远程普攻', 5),
        ):
            with self.subTest(name=name):
                self.assertNotIn(prefix, groups)
                group = groups[name]
                self.assertEqual(len(group['parameter_keys']), damage_rows)
                self.assertEqual([row['label'] for row in group['param_rows']], [prefix + '环合值'])
                self.assertEqual([row[0] for row in group['combat_details']['rows']][:5],
                                 [f'第 {i} 段' for i in range(1, 6)])
        self.assertNotIn('近战普攻爆炸', groups)
        switch = groups['风味变奏']['combat_details']
        self.assertEqual(switch['columns'],
                         ['动作', '动作占时（秒）', '伤害通知（秒）', '命中顿帧（秒）', '伤害判定'])
        self.assertEqual([row[:2] for row in switch['rows']], [['近切远', '1.506'], ['远切近', '0.76']])
        self.assertTrue(all(row[-1].startswith('盒体半尺寸 ') for row in switch['rows']))
        raw = next(c for c in service.catalog()['characters'] if c['id'] == '1004')
        ability = dict(next(a for a in raw['abilities'] if a['id'] == ability_id))
        importer.group_ability_parameters(ability)
        self.assertEqual(ability['groups'], next(a['groups'] for a in raw['abilities'] if a['id'] == ability_id))

    def test_lacrimosa_gift_forms_and_refresh(self):
        import copy
        importer = load_file('guide_gift_details', 'scripts/guide_skill_details.py')
        character = service.detail('characters', '1004')
        ability = next(a for a in character['abilities'] if a['id'] == 'GA_Lacrimosa_Skill')
        group = next(g for g in ability['groups'] if g['name'] == '变轨技能：恶魔的礼物')
        row = next(r for r in group['param_rows'] if r.get('details_only'))
        hits = row['hit_details']
        self.assertEqual(len({h['form_ability_id'] for h in hits}), 46)
        self.assertEqual(len(hits), 52)
        first = hits[0]
        self.assertEqual(first['hit_label'], '分解者')
        self.assertEqual(first['effect_id'], 'GE_mon_14_act05_Dmg01_Steal_BP')
        self.assertEqual(first['attack_values'][0], '48%')
        self.assertEqual(first['attack_values'][9], '96%')
        self.assertIn('至多8次', first['hit_description'])
        self.assertTrue(all(h['repeat'] is None for h in hits))
        self.assertTrue(all(len(h['attack_values']) == 13 for h in hits))
        unnamed = [h for h in hits if h['hit_label'] == '无明众' or h['hit_label'].startswith('无明众 ·')]
        self.assertEqual(len({h['hit_description'] for h in unnamed}), 3)
        html = self.client.get('/guide/characters/1004').text
        self.assertIn('分解者', html)
        self.assertIn('96%', html)
        self.assertIn('次数未提供', html)
        self.assertIn('变身攻击说明', self.client.get('/guide/characters/1004.md').text)
        # Supplemental rows are rebuilt from explicit bindings and damage arrays.
        damages = {h['effect_id']: {'GAName': ability['id'], 'AtkRateBaseArray': [0.25] * 13}
                   for h in hits}
        original = copy.deepcopy(ability)
        original['param_rows'] = [copy.deepcopy(row)]
        tables = {'DT_GameplayAbilityTipsData': {ability['id']: {}}, 'DT_SkillDamageData': damages}
        importer.enrich_ability(original, tables)
        before = json.dumps(original, ensure_ascii=False)
        importer.enrich_ability(original, tables)
        self.assertEqual(json.dumps(original, ensure_ascii=False), before)
        self.assertEqual(original['param_rows'][0]['hit_details'][0]['attack_values'][9], '25%')
        del damages[first['effect_id']]
        with self.assertRaises(ValueError):
            importer.enrich_ability(original, tables)

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
        self.assertEqual(len(characters), 27)
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
        self.assertNotIn('黑羽头像', html)
        self.assertNotIn('查看高清立绘', html)
        self.assertNotIn('Markdown ↗', html)
        self.assertNotIn('JSON ↗', html)
        self.assertNotIn('默认立绘预览', html)

    def test_errors_and_discovery(self):
        for path, status in [('/api/guide/unknown',404),('/api/guide/arcs/missing',404),
                ('/api/guide/arcs?limit=0',400),('/api/guide/arcs?offset=-1',400),
                ('/api/guide/arcs?limit=abc',400),('/api/guide/characters?rarity=9',400),
                ('/api/guide/characters/1042?section=missing',404),
                ('/api/guide/characters/1042?stage=1',400),
                ('/api/guide/challenges/Abyss_12?stage=999',404)]:
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
        self.assertIn('href="/kongmu?character=1042&amp;cartridge=psyche"', html)
        self.assertIn('空幕计算', html)

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
