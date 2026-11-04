"""Exe uses the formal template without fabricating missing master-table fields."""
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from flask import Flask, template_rendered
from PIL import Image
from tests.guide_release_fixtures import published_character_data
from app.modules.guide import blueprint, service, infographic, arc_cultivation, presentation

ROOT = Path(__file__).resolve().parents[1]
URL = '/guide/characters/1074'


@published_character_data
class ExeCharacterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=Flask(__name__,template_folder=str(ROOT/'app/templates'),static_folder=str(ROOT/'app/static'))
        cls.app.register_blueprint(blueprint)
        cls.app.add_url_rule('/login',endpoint='main.login_page',view_func=lambda:'login')
        cls.client=cls.app.test_client()

    def test_formal_header_and_every_section_or_explicit_missing_state(self):
        templates=[]
        def rendered(sender,template,**kwargs):templates.append(template.name)
        template_rendered.connect(rendered,self.app)
        try:response=self.client.get(URL)
        finally:template_rendered.disconnect(rendered,self.app)
        self.assertEqual(templates,['guide/detail.html'])
        self.assertEqual(response.status_code,200)
        for section in ('stats','abilities','awaken','resonance','nouns','equip_slots','affinity'):
            self.assertIn(f'id="section-{section}"',response.text)
            if section!='stats':self.assertIn(f'href="#section-{section}"',response.text)
        for text in ('暗属性','5星角色','生命值','攻击力','防御力','暴击率','暴击伤害',
                     '弧盘适配暂未提供','角色定位暂未提供','机制词条暂未提供',
                     '所需升级材料','空心花籽','数值阅读约定','guide-character-tools','data-guide-toc'):
            self.assertIn(text,response.text)
        for text in ('预告','guide-preview-hero','真红','Shinku','None'):
            self.assertNotIn(text,response.text)
        rows=self.client.get('/api/guide/characters?q=Exe').json['items']
        self.assertEqual([r['id'] for r in rows],['1074'])
        self.assertNotEqual(rows[0].get('status'),'preview')
        self.assertEqual(self.client.get('/guide').text.count('class="guide-preview-badge"'),0)

    def test_verified_numbers_and_unknowns_are_shared_by_machine_formats(self):
        record=service.detail('characters','1074')
        for row,last in zip(record['stats'][:3],[14011,591,811]):
            self.assertEqual(row['values'],[None]*80)
            self.assertEqual(row['growth_values'][-1],last)
        self.assertNotIn('arc_type',record)
        self.assertEqual(record['char_tags'],[])
        self.assertEqual(len(record['abilities']),9)
        hits=[h for a in record['abilities'] for row in a['param_rows'] for h in row['hit_details']]
        self.assertEqual(len(hits),18)
        self.assertTrue(all(h['effect_id'].startswith('GE_Player_Exe_') and h['repeat'] is None for h in hits))
        first=next(h for h in hits if h['effect_id']=='GE_Player_Exe_Melee1_Damage')
        self.assertEqual(first['attack_values'][0],'52.3%')
        self.assertEqual((first['harmony_gain'],first['energy_gain'],first['stagger']),(2.901,1.743,.35))
        self.assertEqual(len(record['awaken']),6)
        self.assertEqual([r['resonance_level'] for r in record['resonance']],[3,6])
        self.assertTrue(all(r['icon'] is None and '未提供' in r['desc'] for r in record['awaken']+record['resonance']))
        self.assertEqual(sum(c!=-1 for row in record['equip_slots']['slots'] for c in row),20)
        self.assertNotIn('owner_grid_count',record['equip_slots'])
        self.assertEqual(len(record['affinity']['gifts']),11)
        self.assertEqual(record['affinity']['total_affinity'],56100)
        for path in ['/api/guide/characters/1074',URL+'.md']:
            text=self.client.get(path).get_data(as_text=True)
            self.assertIn('GE_Player_Exe_Melee1_Damage',text)
            self.assertNotIn('Shinku',text)
            self.assertNotIn('真红',text)
        payload=self.client.get('/api/guide/characters/1074?section=stats').json['data']
        self.assertIsNone(payload['stats'][0]['values'][0])

    def test_tools_and_real_cultivation_costs(self):
        page=self.client.get(URL).text
        for suffix in ('cultivation','builds','infographic'):
            self.assertIn('href="'+URL+'/'+suffix+'"',page)
            self.assertEqual(self.client.get(URL+'/'+suffix).status_code,200)
        self.assertIn('aria-disabled="true"',page)
        self.assertIn('该角色的空幕计算资料暂未接入',page)
        self.assertIsNone(presentation.kongmu_entry({'id':'1074'}))
        self.assertEqual(arc_cultivation.choices(service.detail('characters','1074')),[])
        result=self.client.post('/api/guide/characters/1074/cultivation',json={}).json['data']
        self.assertEqual((result['experience'],result['total_gold']),(6492840,3967000))
        items={i['id'].casefold():i['amount'] for i in result['items']}
        self.assertEqual(items['weeklycloneboss_a04_01'],35)
        self.assertGreater(items['skillupmaterial_05_lv3'],0)
        opts=infographic.options('1074')
        self.assertEqual(opts['character_id'],'1074')
        self.assertEqual(opts['arcs'],[])
        self.assertIn('1074',{c['id'] for c in opts['characters']})
        self.assertEqual(self.client.get('/api/guide/characters/1074/infographic').status_code,401)

    def test_avatar_is_an_active_crop_and_all_assets_decode(self):
        record=service.detail('characters','1074')
        paths={k:ROOT/'app'/v.removeprefix('/') for k,v in record['assets'].items()}
        with Image.open(paths['fullbody']) as source:
            self.assertEqual(source.size,(1440,1952))
            expected=source.convert('RGBA').crop((340,30,820,510)).resize((256,256),Image.Resampling.LANCZOS)
        with Image.open(paths['avatar']) as avatar:
            self.assertEqual(avatar.size,(256,256))
            self.assertEqual(avatar.mode,'RGBA')
            self.assertEqual(avatar.tobytes(),expected.tobytes())
        with Image.open(paths['portrait']) as portrait:self.assertEqual(portrait.size,(1024,1024))
        for url in record['assets'].values():
            with self.client.get(url) as response:self.assertEqual(response.status_code,200)

    def test_unreleased_kit_stays_hidden_for_all_public_formats(self):
        with patch('app.modules.guide.character_release.current_time',return_value=datetime.fromisoformat('2026-11-26T09:59:59+08:00')):
            for path in [URL,URL+'.md','/api/guide/characters/1074']:
                response=self.client.get(path)
                self.assertEqual(response.status_code,200)
                self.assertNotIn('GE_Player_Exe_',response.text)
                self.assertNotIn('GA_Exe_',response.text)
            page=self.client.get(URL).text
            self.assertIn('guide-character-header',page)
            self.assertNotIn('id="section-abilities"',page)
            self.assertIn('2026-11-26 10:00',page)
            result=self.client.post('/api/guide/characters/1074/cultivation',json={}).json['data']
            cultivation_page=self.client.get(URL+'/cultivation').text
            self.assertNotIn('<h2>技能与天赋</h2>',cultivation_page)
            self.assertIn('弧盘适配暂未提供，当前仅计算角色养成',cultivation_page)
            self.assertFalse(result['skills_included'])
            self.assertNotIn('weeklycloneboss_a04_01',{i['id'].casefold() for i in result['items']})


class ExeImportTest(unittest.TestCase):
    def test_import_uses_independent_damage_and_never_borrowed_descriptions(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        self.addCleanup(lambda:sys.path.remove(str(ROOT/'scripts')))
        from sync_guide_exe import assemble
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            def put(name,rows):(folder/(name+'.json')).write_text(json.dumps([{'Rows':rows}]))
            growth={f'Exe_lv_{i}':{'ConditionArray':[],'ModifyData':[{
                'PropName':'AtkBase','PropValue':i-1,'ModifierOp':'EModifyModOp::MODIFY_MODOP_ADDITIVE'}]} for i in range(1,81)}
            growth.update({f'Exe_stage_{i}':{'ConditionArray':[],'ModifyData':[]} for i in range(1,7)})
            put('DT_PlayerModifyPackData',growth)
            put('DT_CharacterBreakthroughDataTable',{f'Exe_Breakthrough_{i}':{
                'MaxCharacterLevel':cap,'NeedItems':'OrdinaryMonMaterial_03_lv1:5','NeedGolds':'gold:25000'}
                for i,cap in enumerate([20,30,40,50,60,70,80])})
            put('DT_CharacterAbilityConfig',{'1074':{'CharacterAbilityList':[{'Key':'GA_Exe_Melee','Value':{
                'AbilityIndex':1,'AbilityType':'ECharacterAbilityType::Proactive','LevelsCostItems':[]}}],
                'PassiveAbilityList':[],'PeculiarityList':[],'CityList':[]}})
            damage={'GAName':'GA_Exe_Melee','HeterochromeAdd':2.901,'ChargeAdd':1.743,'UnbalValue':.35,
                    'AtkRateBaseArray':[.523]*13,'DefRateBaseArray':[],'HPRateBaseArray':[]}
            put('DT_SkillDamageData',{'GE_Player_Exe_Melee1_Damage':damage,'GE_Player_Shinku_Melee1_Damage':damage})
            put('DT_GameplayAbilityTipsData',{'GA_Exe_Melee':{'Description':'真红错误占位说明'}})
            put('1074AwakenEffect',{'1074':{'AwakenEffectStructList':[{'Desc':'真红错误占位觉醒'}]*8}})
            put('DT_EquipmentPlanData',{'1074':{'EquipmentSlots':['0,0,-1']}})
            record,caps=assemble(folder,service.catalog())
            serialized=json.dumps(record,ensure_ascii=False)
            self.assertNotIn('真红',serialized)
            self.assertNotIn('Shinku',serialized)
            self.assertEqual(record['stats'][1]['values'],[None]*80)
            self.assertEqual(record['stats'][1]['growth_values'][-1],79)
            self.assertEqual(record['abilities'][0]['param_rows'][0]['values'][0],'52.3%')
            self.assertEqual(len(record['abilities'][0]['param_rows']),1)
            self.assertEqual(caps[-1],80)
            growth['Exe_lv_1']['ModifyData'][0]['ModifierOp']='MULTIPLY'
            put('DT_PlayerModifyPackData',growth)
            with self.assertRaisesRegex(ValueError,'Nonadditive'):assemble(folder,service.catalog())
