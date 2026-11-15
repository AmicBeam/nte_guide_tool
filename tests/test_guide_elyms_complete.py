"""Elyms must satisfy the whole normal character page and all four tools."""
from tests.guide_release_fixtures import published_character_data
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import unittest
import tempfile
from datetime import datetime
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from PIL import Image
from flask import Flask, template_rendered
from app.modules.guide import blueprint, service, presentation, infographic

ROOT = Path(__file__).resolve().parents[1]
URL = '/guide/characters/1090'


@published_character_data
class ElymsCharacterModulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__, template_folder=str(ROOT/'app/templates'), static_folder=str(ROOT/'app/static'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/login', endpoint='main.login_page', view_func=lambda: 'login')
        cls.app, cls.client = app, app.test_client()

    def test_normal_identity_header_and_all_template_modules(self):
        templates = []
        def rendered(sender, template, **kwargs): templates.append(template.name)
        template_rendered.connect(rendered, self.app)
        try: response = self.client.get(URL)
        finally: template_rendered.disconnect(rendered, self.app)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(templates, ['guide/detail.html'])
        html = response.text
        for section in ('stats', 'abilities', 'awaken', 'resonance', 'nouns', 'equip_slots', 'affinity', 'aliases'):
            with self.subTest(section=section):
                self.assertIn(f'id="section-{section}"', html)
                if section != 'stats': self.assertIn(f'href="#section-{section}"', html)
        for module in ('guide-character-header', 'guide-profile-icons', 'guide-role-badges',
                       'guide-character-art', 'guide-character-overview', 'guide-character-tools',
                       'data-guide-toc', '数值阅读约定', '所需升级材料'):
            with self.subTest(module=module): self.assertIn(module, html)
        for text in ('曳尔沐', '固态', '增益', '输出增益', '追加攻击', '光属性', '5星角色',
                     '生命值', '攻击力', '防御力', '暴击率', '暴击伤害',
                     '信号强度', '错视幻觉', '好感度'):
            self.assertIn(text, html)
        self.assertNotIn('guide-preview-hero', html)
        self.assertNotIn('基础面板暂未提供', html)
        self.assertNotIn('预告', html)
        card = next(c for c in self.client.get('/api/guide/characters?q=曳尔沐').json['items'] if c['id']=='1090')
        self.assertNotEqual(card.get('status'), 'preview')
        self.assertEqual(self.client.get('/guide').text.count('class="guide-preview-badge"'), 3)

    def test_real_attributes_skills_costs_equipment_and_explicit_affinity_state(self):
        record = service.detail('characters', '1090')
        stats = {row['id_stats']: row['values'] for row in record['stats']}
        for key, first, last in [('HPMaxBase',1270,15392),('AtkBase',83,660),('DefBase',75,909),
                                 ('CritBase',5,5),('CritDamageBase',50,50)]:
            with self.subTest(stat=key):
                self.assertEqual(len(stats[key]),80)
                self.assertEqual((stats[key][0],stats[key][-1]),(first,last))
                self.assertTrue(all(isinstance(v,(int,float)) and v>0 for v in stats[key]))
        self.assertTrue(record['char_tags'])
        self.assertEqual(record['arc_type']['name'],'固态')
        self.assertEqual(len(record['abilities']),9)
        by_id={a['id']:a for a in record['abilities']}
        self.assertEqual(by_id['GA_Elyms_Skill']['base_cooldown'],16)
        self.assertEqual(by_id['GA_Elyms_UltraSkill']['base_cooldown'],20)
        self.assertEqual(len(record['awaken']),6)
        self.assertEqual([r['resonance_level'] for r in record['resonance']],[3,6])
        self.assertEqual(len(record['nouns']),2)
        self.assertEqual(record['equip_slots']['owner_grid_count'],3)
        self.assertEqual(sum(v>=0 for row in record['equip_slots']['slots'] for v in row),20)
        self.assertEqual(record['equip_slots']['stats'][0]['value'],8)
        self.assertEqual(len(record['materials']['breakthrough']),6)
        self.assertEqual(len(record['materials']['skills']),8)
        for stage in record['materials']['breakthrough']:
            self.assertTrue(stage['items'])
            self.assertTrue(all(i['name'] and i['amount']>0 for i in stage['items']))
        # The live role binding now includes all ten levels and the actual full-affinity bonus.
        self.assertTrue(record['affinity']['available'])
        self.assertEqual(len(record['affinity']['levels']),10)
        self.assertEqual(record['affinity']['full_affinity_bonus'][0]['text'],'攻击力＋5%')
        self.assertEqual(len(record['affinity']['gifts']),14)
        self.assertEqual(record['affinity']['total_affinity'],56100)
        self.assertNotIn('total_affinity_estimated',record['affinity'])
        phone=next(g for g in record['affinity']['gifts']if g['id']=='SpecialGift_1090')
        self.assertEqual((phone['name'],phone['affinity_gain']),('复古风手机壳',1200))
        self.assertTrue(phone['exclusive'])
        self.assertEqual(phone['required_gifts'],47)
        self.assertIn('复古风手机壳',self.client.get(URL).text)
        self.assertNotIn('该角色尚未提供喜好礼物配置',self.client.get(URL).text)
        self.assertNotIn('尚未提供好感度等级配置', self.client.get(URL).text)
        for section in ('stats','abilities','awaken','resonance','nouns','equip_slots','affinity','materials'):
            response=self.client.get('/api/guide/characters/1090?section='+section)
            self.assertEqual(response.status_code,200)
            self.assertIn(section,response.json['data'])
        self.assertEqual(self.client.get('/api/guide/characters/1090?section=reference_details').status_code,404)
        self.assertIn('信号强度',self.client.get(URL+'.md').text)

    def test_four_actual_tool_links_and_editor_assets(self):
        html=self.client.get(URL).text
        tools=html.split('class="guide-character-tools"',1)[1].split('</nav>',1)[0]
        for label,path in [('养成计算器',URL+'/cultivation'),('排轴与配装',URL+'/builds'),
                           ('空幕计算','/kongmu?character=1090&amp;cartridge=cosmos'),
                           ('制作角色一图流攻略',URL+'/infographic')]:
            with self.subTest(tool=label):
                self.assertIn(label,tools);self.assertIn('href="'+path+'"',tools)
                if label != '空幕计算':
                    self.assertEqual(self.client.get(path.replace('&amp;','&')).status_code,200)
        self.assertIn('/overlap?character=char_elyms',self.client.get(URL+'/builds').text)
        opts=infographic.options('1090')
        self.assertIn('1090',{c['id']for c in opts['characters']})
        self.assertTrue(opts['arcs'])
        self.assertEqual(opts['recommended_core'],'cosmos')
        for mode in ('cultivation','builds','kongmu','infographic'):
            self.assertIn('曳尔沐',self.client.get('/guide/tools?tool='+mode).text)

    def test_refresh_preserves_the_complete_local_character_without_duplicates(self):
        spec=importlib.util.spec_from_file_location('elyms_refresh',ROOT/'scripts/refresh_guide_data.py')
        refresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(refresh)
        local=json.loads((ROOT/'app/modules/guide/data/ntedata_characters.json').read_text())
        data={'characters':[],'cultivation':{'level_caps':{}}}
        refresh.merge_local_characters(data,local)
        refresh.merge_local_characters(data,local)
        self.assertEqual([c['id']for c in data['characters']],['1090'])
        self.assertEqual(data['cultivation']['level_caps']['1090'],[20,30,40,50,60,70,80])
        self.assertTrue(data['characters'][0]['stats'])
        broken=deepcopy(local);broken['characters'][0]['stats']=[]
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            refresh.merge_local_characters({'characters':[],'cultivation':{'level_caps':{}}},broken)

    def test_avatar_is_the_deterministic_portrait_crop_not_the_round_original(self):
        avatar=ROOT/'app'/service.detail('characters','1090')['assets']['avatar'].removeprefix('/')
        with Image.open(ROOT/'app/static/images/characters/portrait/曳尔沐.webp')as portrait:
            expected=portrait.convert('RGBA').crop((340,100,780,540)).resize((256,256),Image.Resampling.LANCZOS)
        with Image.open(avatar)as actual:
            self.assertEqual(actual.size,(256,256))
            self.assertEqual(actual.convert('RGBA').tobytes(),expected.tobytes())
            # The portrait crop includes neck/shoulder pixels at the lower edge, without a rounded bottom mask.
            self.assertGreater(actual.getchannel('A').crop((0,240,256,256)).getextrema()[1],0)
        self.assertEqual(avatar.read_bytes(),(ROOT/'app/modules/guide/static/character-previews/elyms-avatar.webp').read_bytes())
        for path in service.detail('characters','1090')['assets'].values():
            self.assertEqual(self.client.get(path).status_code,200)


@published_character_data
class ElymsToolFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = importlib.import_module('app.db')
        self.original_database = self.db.db.database
        if not self.db.db.is_closed(): self.db.db.close()
        self.db.db.init(str(Path(self.temp.name)/'elyms-tools.db'))
        self.app = importlib.import_module('app').create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        if not self.db.db.is_closed(): self.db.db.close()
        self.db.db.init(self.original_database)
        self.temp.cleanup()

    def _issue_login_and_get_token(self, uid):
        dao = importlib.import_module('app.dao')
        with self.db.atomic_transaction(): dao.issue_mock_login(uid,uid,'654321')
        response = self.client.post('/api/auth/login',json={'player_uid':uid,'code':'654321'})
        self.assertEqual(response.status_code,200)
        return response.json['token']

    def test_normal_wiki_does_not_open_test_only_axis_access(self):
        tester=self._issue_login_and_get_token('elyms-axis-tester')
        regular=self._issue_login_and_get_token('elyms-axis-regular')
        invited=self._issue_login_and_get_token('elyms-axis-invited')
        models=importlib.import_module('app.models')
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid=='elyms-axis-tester').execute()
        models.Player.update(shaft_invited=True).where(models.Player.player_uid=='elyms-axis-invited').execute()
        tester_headers={'Authorization':'Bearer '+tester}
        axis={'team':[{'slot':0,'character_id':'char_elyms','arc_id':'','cartridge_id':''}],
              'steps':[{'id':'e1','slot':0,'action_id':'action_elyms_skill1','start_tick':0}]}
        shaft=importlib.import_module('app.modules.shaft.service')
        computed=shaft.simulate_shaft_axis(axis)['result']
        saved=self.client.post('/api/shaft/axes',headers=tester_headers,
            json={'title':'曳尔沐权限回归轴','axis':axis,'result':computed})
        self.assertEqual(saved.status_code,200)
        identifier=saved.json['id']
        published=self.client.post(f'/api/shaft/axes/{identifier}/publish',headers=tester_headers,json={})
        self.assertEqual(published.status_code,200)
        public_id=published.json['id']
        for instant,opened in [('2026-11-05T09:59:59+08:00',False),('2026-11-05T10:00:00+08:00',True)]:
            with self.subTest(instant=instant),patch.object(service.character_release,'current_time',return_value=datetime.fromisoformat(instant)):
                self.assertEqual(self.client.get('/api/guide/characters/1090').json['data']['content_released'],opened)
                allowed=self.client.get('/api/guide/axes?character=1090',headers=tester_headers)
                self.assertEqual(allowed.status_code,200)
                self.assertIn(public_id,{row['id']for row in allowed.json['items']})
                for headers in ({},{'Authorization':'Bearer '+regular},{'Authorization':'Bearer '+invited}):
                    with self.subTest(authenticated=bool(headers)):
                        self.assertEqual(self.client.get(URL).status_code,200)
                        self.assertEqual(self.client.get(URL+'/builds').status_code,200)
                        filtered=self.client.get('/api/guide/axes?character=1090',headers=headers)
                        self.assertEqual(filtered.status_code,200)
                        self.assertNotIn(public_id,{row['id']for row in filtered.json['items']})
                        self.assertEqual(self.client.get('/api/guide/axes/'+str(public_id),headers=headers).status_code,400)

    def test_cultivation_kongmu_editor_and_related_articles(self):
        token=self._issue_login_and_get_token('elyms-complete-tools')
        models=importlib.import_module('app.models')
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid=='elyms-complete-tools').execute()
        headers={'Authorization':'Bearer '+token}
        result=self.client.post('/api/guide/characters/1090/cultivation',json={})
        self.assertEqual(result.status_code,200)
        self.assertTrue(result.json['data']['items'])
        arc=infographic.options('1090')['arcs'][0]['id']
        with_arc=self.client.post('/api/guide/characters/1090/cultivation',json={'include_arc':True,'arc_id':arc})
        self.assertEqual(with_arc.status_code,200)
        self.assertEqual(self.client.get('/api/guide/axes?character=1090',headers=headers).status_code,200)
        self.assertEqual(self.client.get('/kongmu?character=1090&cartridge=cosmos').status_code,200)
        for identifier in ('1090','char_elyms'):
            planned=self.client.post('/api/kongmu/plan',headers=headers,
                json={'character_id':identifier,'cartridge_id':'cosmos'})
            self.assertEqual(planned.status_code,200)
            self.assertEqual(planned.json['character']['id'],'1090')
            self.assertGreater(planned.json['result']['total_solution_count'],0)
        self.assertEqual(self.client.post('/api/kongmu/plan',json={'character_id':'1090','cartridge_id':'cosmos'}).status_code,200)
        self.assertEqual(self.client.get('/api/guide/characters/1090/infographic').status_code,401)
        editor=self.client.get('/api/guide/characters/1090/infographic',headers=headers)
        self.assertEqual(editor.status_code,200)
        self.assertIn('曳尔沐',{c['name']for c in editor.json['data']['characters']})
        ident=str(uuid4())
        saved=self.client.put('/api/guide/me/articles/'+ident,headers=headers,json={
            'revision':0,'title':'曳尔沐完整模板验收攻略','markdown':'已验证正文',
            'target_kind':'characters','target_id':'1090'})
        self.assertEqual(saved.status_code,200)
        published=self.client.post('/api/guide/me/articles/'+ident+'/publish',headers=headers,
            json={'revision':saved.json['data']['revision']})
        self.assertEqual(published.status_code,200)
        html=self.client.get(URL).text
        self.assertIn('guide-related-articles',html)
        self.assertIn('曳尔沐完整模板验收攻略',html)
