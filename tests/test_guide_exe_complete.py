"""Exe's full Wiki template and the server-time 26 November noon release."""
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

from PIL import Image
from flask import Flask
from app.modules.guide import blueprint, service, character_release, arc_cultivation, infographic, chat_tools
from app.modules.kongmu import service as kongmu
from tests.guide_release_fixtures import published_character_data
from tests.test_solo_room_flow import RoomFlowTestCase

ROOT=Path(__file__).resolve().parents[1]
CID='1074'
ARC='fork_exe'
BEFORE='2026-11-26T11:59:59+08:00'
OPEN='2026-11-26T12:00:00+08:00'


class ExeReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=Flask(__name__,template_folder=str(ROOT/'app/templates'),static_folder=str(ROOT/'app/static'))
        app.register_blueprint(blueprint);app.add_url_rule('/login',endpoint='main.login_page',view_func=lambda:'login');cls.client=app.test_client()

    def clock(self,value):return patch.object(character_release,'current_time',return_value=datetime.fromisoformat(value))

    def test_noon_boundary_utc_and_cache_without_browser_clock_authority(self):
        raw=deepcopy(service.catalog())
        for instant,opened in [(BEFORE,False),(OPEN,True),('2026-11-26T03:59:59+00:00',False),('2026-11-26T04:00:00+00:00',True)]:
            with self.subTest(instant=instant),self.clock(instant):
                for identifier in (CID,'preview-exe'):
                    record=service.detail('characters',identifier)
                    self.assertEqual(record['id'],CID);self.assertEqual(record['content_released'],opened)
                    self.assertEqual('abilities'in record,opened)
                    state=self.client.get('/api/guide/characters/'+identifier+'/release?now=2099-01-01').json['data']
                    self.assertEqual(state['content_released'],opened)
                    self.assertEqual(state['release_at'],'2026-11-26T12:00:00+08:00')
                    self.assertEqual(state['check_after_ms'],0 if opened else 1000)
                self.assertEqual('effect'in service.detail('arcs',ARC),opened)
        self.assertEqual(service.catalog(),raw)

    def test_all_public_formats_search_and_assistant_redact_until_noon(self):
        with self.clock(BEFORE):
            for identifier in (CID,'preview-exe'):
                for path in ['/guide/characters/'+identifier,'/guide/characters/'+identifier+'.md','/api/guide/characters/'+identifier]:
                    response=self.client.get(path);self.assertEqual(response.status_code,200)
                    text=json.dumps(response.json,ensure_ascii=False)if response.is_json else response.text
                    self.assertNotIn('数据穿透',text);self.assertNotIn('量化算法',text);self.assertNotIn('离线留言终端',text)
                    self.assertEqual(response.headers['Cache-Control'],'no-store')
                for field in character_release.AFTER_SKILL_FIELDS:
                    self.assertEqual(self.client.get('/api/guide/characters/'+identifier+'?section='+field).status_code,404)
            self.assertEqual(self.client.get('/api/guide/characters/1074?ability_id=GA_Exe_Skill').status_code,404)
            self.assertEqual(self.client.get('/api/guide/search?q=数据穿透').json['total'],0)
            overview=self.client.get('/api/guide/characters/1074?section=overview').json['data']
            self.assertEqual(overview['available_sections'],['stats','materials'])
            html=self.client.get('/guide/characters/1074').text
            self.assertIn('2026-11-26 12:00',html);self.assertNotIn('id="section-abilities"',html)
            result,_=chat_tools.Toolkit(None,None).execute('wiki_get',{'kind':'characters','id':CID})
            self.assertNotIn('数据穿透',json.dumps(result,ensure_ascii=False))
        with self.clock(OPEN):
            self.assertGreater(self.client.get('/api/guide/search?q=数据穿透').json['total'],0)
            self.assertEqual(self.client.get('/api/guide/characters/1074?ability_id=GA_Exe_Skill&level=10').status_code,200)

    def test_signature_arc_effect_has_same_release_and_auto_refresh(self):
        for instant,opened in [(BEFORE,False),(OPEN,True)]:
            with self.clock(instant):
                for path in ['/guide/arcs/'+ARC,'/guide/arcs/'+ARC+'.md','/api/guide/arcs/'+ARC]:
                    response=self.client.get(path);self.assertEqual(response.status_code,200)
                    self.assertEqual('缥缈'in(json.dumps(response.json,ensure_ascii=False)if response.is_json else response.text),opened)
                self.assertEqual(self.client.get('/api/guide/arcs/'+ARC+'?section=effect').status_code,200 if opened else 404)
                html=self.client.get('/guide/arcs/'+ARC).text
                self.assertEqual('data-character-release-url="/api/guide/characters/1074/release"'in html,not opened)
                self.assertEqual('id="section-effect"'in html,opened)

    def test_before_release_cultivation_excludes_skills_and_keeps_arc_costs(self):
        results=[]
        for instant in (BEFORE,OPEN):
            with self.clock(instant):
                result=self.client.post('/api/guide/characters/1074/cultivation',json={'include_arc':True,'arc_id':ARC}).json['data']
                self.assertEqual(result['arc']['name'],'虚线彼方');self.assertEqual(result['arc']['experience'],3640330)
                self.assertEqual(result['arc']['total_gold'],1512450);results.append(result)
        self.assertFalse(results[0]['skills_included'])
        self.assertLess(results[0]['total_gold'],results[1]['total_gold'])
        self.assertNotIn('数据穿透',json.dumps(results[0],ensure_ascii=False))


@published_character_data
class ExeCompleteTest(ExeReleaseTest):
    # Boundary tests above set their own independent server clock.
    def test_full_template_identity_every_section_tools_and_unknown_city_text(self):
        role=service.detail('characters',CID)
        self.assertEqual((role['name'],role['element_name'],role['rarity'],role['arc_type']['name']),('Exe','暗',5,'等离子'))
        self.assertEqual([tag['name']for tag in role['char_tags']],['输出','主力输出','召唤'])
        stats={s['id_stats']:s['values']for s in role['stats']}
        for key,first,last in [('HPMaxBase',1260,15271),('AtkBase',85,676),('DefBase',73,884),('CritBase',5,5),('CritDamageBase',50,50)]:
            self.assertEqual(len(stats[key]),80);self.assertEqual((stats[key][0],stats[key][-1]),(first,last))
        self.assertEqual(len(role['abilities']),9);self.assertEqual(len(role['awaken']),6);self.assertEqual(len(role['resonance']),2);self.assertEqual(len(role['nouns']),4)
        self.assertEqual(role['affinity']['total_affinity'],56100);self.assertEqual(len(role['affinity']['levels']),10)
        self.assertEqual(len(role['affinity']['gifts']),11)
        self.assertIn('离线留言终端',[g['name']for g in role['affinity']['gifts']])
        html=self.client.get('/guide/characters/1074').text
        for section in ('stats','abilities','awaken','resonance','nouns','equip_slots','affinity','aliases'):
            self.assertIn('id="section-'+section+'"',html)
        for text in ['养成计算器','排轴与配装','空幕计算','制作角色一图流攻略','数据穿透','评分','都市技能效果暂未提供。','虚线彼方']:
            self.assertIn(text,html)
        self.assertNotIn('guide-preview-hero',html);self.assertNotIn('guide-preview-badge',html)
        self.assertNotIn('<!>10',html)
        from app.modules.guide.articles import fields
        self.assertEqual(fields(dict(title='Exe',markdown='攻略',target_kind='characters',target_id='preview-exe'))['target_id'],CID)
        self.assertEqual(len([c for c in service.public_records('characters')if c['name']=='Exe']),1)

    def test_real_skill_values_costs_materials_and_machine_reads(self):
        role=service.detail('characters',CID)
        e=next(a for a in role['abilities']if a['id']=='GA_Exe_Skill');q=next(a for a in role['abilities']if a['id']=='GA_Exe_UltraSkill')
        self.assertEqual((e['base_cooldown'],q['base_cooldown']),(15,20))
        cost=next(p for p in q['param_rows']if p['label']=='消耗终结能量');self.assertEqual(cost['values'],['120']*13)
        damage=next(p for p in e['param_rows']if p.get('hit_details'))
        self.assertEqual([h['repeat']for h in damage['hit_details']],[3,3])
        self.assertEqual(damage['values'][0],'36.5148%*3+60.48%*3')
        self.assertEqual(len(role['materials']['breakthrough']),6);self.assertEqual(len(role['materials']['skills']),8)
        for section in ('stats','materials','abilities','awaken','resonance','nouns','equip_slots','affinity','experience'):
            self.assertEqual(self.client.get('/api/guide/characters/1074?section='+section).status_code,200)
        self.assertIn('数据穿透',self.client.get('/guide/characters/1074.md').text)

    def test_per_hit_frames_scores_are_in_every_public_detail_format(self):
        role=service.detail('characters',CID)
        expected={'Melee1':(.4,5),'Melee2':(.8,5),'Melee3':(.8,5),
                  'Melee4':(.8,10),'Melee4_1':(.8,10),'Melee4_2':(.8,10),
                  'Melee5':(0,0),'Branch':(2,100),'AirAttack':(.5,10),
                  'PerfectEvadeAttack':(5,150),'CloneAtk':(3,50),
                  'Skill':(1.67,50),'Skill1':(1.67,50),'UltraSkill1':(1,25),
                  'UltraSkill2':(1.2,30),'QTE':(4,100),'QTE1':(3,100)}
        found={}
        for ability in role['abilities']:
            for row in ability['param_rows']:
                for hit in row.get('hit_details',[]):
                    key=hit['effect_id'].removeprefix('GE_Player_Exe_').removesuffix('_Damage')
                    found[key]=(hit['frame_gain'],hit['score_gain'])
        self.assertEqual(found,expected)
        e=next(a for a in role['abilities']if a['id']=='GA_Exe_Skill')['param_rows'][0]['hit_details']
        self.assertEqual([h['repeat']for h in e],[3,3])
        self.assertAlmostEqual(sum(h['repeat']*h['frame_gain']for h in e),10.02)
        self.assertEqual(sum(h['repeat']*h['score_gain']for h in e),300)
        reply=self.client.get('/api/guide/characters/1074?ability_id=GA_Exe_Skill&level=10').json['data']
        self.assertIn('frame_gain',json.dumps(reply))
        text=self.client.get('/guide/characters/1074.md?section=abilities').text
        self.assertIn('基础帧数',text);self.assertIn('基础评分',text)
        html=self.client.get('/guide/characters/1074').text
        self.assertIn('<th scope="col">基础帧数</th>',html)
        self.assertIn('<th scope="col">基础评分</th>',html)
        self.assertIn('重复 ×3',html)
        other=self.client.get('/guide/characters/1042').text
        self.assertNotIn('<th scope="col">基础帧数</th>',other)
        self.assertNotIn('<th scope="col">基础评分</th>',other)
        self.assertNotIn('frame_gain',json.dumps(self.client.get('/api/guide/characters/1074?section=abilities&hit_details=0').json))

    def test_rating_table_is_in_score_noun_and_public_formats(self):
        score=service.detail('characters',CID)['nouns']['exescore']
        table=score['rating_table']
        self.assertEqual([r['rating']for r in table],['B','A','S','SS','SSS','Ω'])
        self.assertEqual([r['score_threshold']for r in table],[0,100,300,500,3000,5000])
        self.assertEqual([r['frame_multiplier']for r in table],[1,1.2,1.5,1.8,2,3])
        self.assertEqual([r['score_decay_per_second']for r in table],[0,10,30,100,300,0])
        self.assertIn('六次共鸣',score['rating_note'])
        html=self.client.get('/guide/characters/1074').text
        self.assertEqual(html.count('aria-label="评分评级标准"'),1)
        for label in ['分数门槛','帧数获取倍率','每秒掉分']:self.assertIn(label,html)
        self.assertIn('rating_table',json.dumps(self.client.get('/api/guide/characters/1074?section=nouns').json))
        self.assertIn('评分评级标准',self.client.get('/guide/characters/1074.md?section=nouns').text)

    def test_support_final_multiplier_growth_and_resonance_update(self):
        record=service.detail('characters',CID)
        support=next(a for a in record['abilities']if a['id']=='GA_Exe_QTE')['param_rows'][0]
        self.assertEqual(support['values'][0],'20%*2+160%*1')
        self.assertEqual([h['repeat']for h in support['hit_details']],[2,1])
        self.assertEqual([h['energy_gain']for h in support['hit_details']],[.5,4])
        self.assertEqual([h['stagger']for h in support['hit_details']],[.25,2])
        passive=next(a for a in record['abilities']if a['id']=='GA_Exe_Passive_2')
        self.assertIn('无法减至2秒以下',passive['groups'][0]['description'])
        self.assertIn('抗打断能力提高50%',record['resonance'][-1]['desc'])

    def test_signature_arc_and_actual_defaults(self):
        arc=service.detail('arcs',ARC)
        self.assertEqual((arc['name'],arc['type_name'],arc['rarity']),('虚线彼方','等离子',5))
        self.assertEqual(arc['exclusive_character_ids'],[CID])
        self.assertEqual([s['values'][-1]for s in arc['stats']],[570,48])
        self.assertEqual(arc['effect']['values'][0]['values'],['28%','31%','34%','37%','40%'])
        self.assertEqual(arc['effect']['values'][2]['values'],['2%','2.5%','3%','3.5%','4%'])
        self.assertEqual(arc_cultivation.recommended({'id':CID}),{'id':ARC,'name':'虚线彼方'})
        self.assertEqual(infographic.options(CID)['recommended_arc'],ARC)
        html=self.client.get('/guide/characters/1074/cultivation').text
        self.assertIn('value="fork_exe" selected',html)
        self.assertIn('href="/guide/arcs/fork_exe"',self.client.get('/guide/characters/1074').text)

    def test_bound_images_decode_and_avatar_pixels_match_expected_crop(self):
        role=service.detail('characters',CID)
        paths=[*role['assets'].values(),'/static/images/arcs/fork_exe.webp']
        paths += [a['icon']for a in role['abilities']if a.get('icon')]
        paths += [a['icon']for a in role['awaken']if a.get('icon')]
        for url in paths:
            with Image.open((ROOT/'app/modules/guide/static'/url.removeprefix('/static/guide/') if url.startswith('/static/guide/') else ROOT/'app'/url.removeprefix('/')))as image:
                self.assertEqual(image.format,'WEBP');self.assertGreater(image.width,0)
            with self.client.get(url)as response:self.assertEqual(response.status_code,200)
        with Image.open(ROOT/'app'/role['assets']['avatar'].removeprefix('/'))as image:
            self.assertEqual(image.size,(256,256));self.assertEqual(image.mode,'RGBA')
            self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(),'c9405bf1c4fed07b61372dbd0ed0a800ed02abbe16e8ee726084059a8f045462')
        with Image.open(ROOT/'app'/role['assets']['fullbody'].removeprefix('/'))as image:self.assertEqual(image.size,(2160,2560))

    def test_kongmu_uses_real_grid_and_requires_test_access(self):
        tester=SimpleNamespace(shaft_test_whitelisted=True)
        self.assertNotIn(CID,{c['id']for c in kongmu.get_kongmu_catalog_payload()['characters']})
        record,detail=kongmu.get_character_detail(CID,player=tester)
        self.assertEqual(record['name'],'Exe');self.assertEqual(sum(v>=0 for row in detail['equip_slots']['slots']for v in row),20)
        self.assertEqual(detail['equip_slots']['stats'][0]['value'],8)
        self.assertEqual(kongmu.get_character_detail('char_exe',player=tester),kongmu.get_character_detail(CID,player=tester))
        result=kongmu.plan_kongmu_layout(CID,'chaos',player=tester)
        self.assertTrue(result['result']['solutions']);self.assertEqual(result['character']['id'],CID)


class ExeWikiAxisAccessTest(RoomFlowTestCase):
    # Keep loaded guide module identities intact for the rest of the suite.
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app_module = importlib.import_module('app')
        self.dao_module = importlib.import_module('app.dao')
        self.db_module = importlib.import_module('app.db')
        self.original_database = self.db_module.db.database
        if not self.db_module.db.is_closed(): self.db_module.db.close()
        self.db_module.db.init(str(Path(self.temp_dir.name) / 'test.db'))
        self.app = self.app_module.create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        if not self.db_module.db.is_closed(): self.db_module.db.close()
        self.db_module.db.init(self.original_database)
        self.temp_dir.cleanup()

    def test_wiki_release_does_not_open_shaft_account_access(self):
        tokens={k:self._issue_login_and_get_token('wiki-exe-'+k)for k in ['formal','invited','test']}
        models=importlib.import_module('app.models')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid=='wiki-exe-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid=='wiki-exe-test').execute()
        axis={'team':[dict(slot=0,character_id='char_exe')],'steps':[dict(id='e',slot=0,action_id='action_exe_e',start_tick=0)]}
        saved=self._post('/api/shaft/axes',dict(title='Exe Wiki access',axis=axis,result=self._shaft_client_result(axis)),token=tokens['test'])
        public=self._post(f"/api/shaft/axes/{saved['id']}/publish",{},token=tokens['test'])
        for time in [BEFORE,OPEN]:
            with patch.object(character_release,'current_time',return_value=datetime.fromisoformat(time)):
                for name,token in tokens.items():
                    reply=self._get('/api/guide/axes?character=1074',token=token)
                    self.assertEqual(public['id']in{row['id']for row in reply['items']},name=='test')
        self.assertEqual(next(c for c in importlib.import_module('app.modules.shaft.domain.catalog').load_shaft_catalog()['characters']if c['id']=='char_exe')['base_stats']['atk'],676)
