"""A promoted character includes its actual signature arc and complete tool bindings."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest
from flask import Flask
from PIL import Image
from tests.guide_release_fixtures import published_character_data
from app.modules.guide import blueprint, service, arc_cultivation, infographic

ROOT=Path(__file__).resolve().parents[1]
ARC='fork_elyms'


@published_character_data
class ElymsArcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=Flask(__name__,template_folder=str(ROOT/'app/templates'),static_folder=str(ROOT/'app/static'))
        cls.app.register_blueprint(blueprint)
        cls.app.add_url_rule('/login',endpoint='main.login_page',view_func=lambda:'login')
        cls.client=cls.app.test_client()

    def test_signature_identity_full_stats_and_all_refinement_parameters(self):
        arc=service.detail('arcs',ARC)
        self.assertEqual((arc['name'],arc['rarity'],arc['type_name']),('静候彩铃',5,'固态'))
        self.assertEqual(arc['exclusive_character_ids'],['1090'])
        self.assertIn('曳尔沐',arc['exclusive_desc'])
        for row,first,last in zip(arc['stats'],[37,9.6],[570,24]):
            self.assertEqual(len(row['values']),80)
            self.assertEqual((row['values'][0],row['values'][-1]),(first,last))
        params=arc['effect']['values']
        self.assertEqual(len(params),6)
        self.assertEqual(params[0]['values'],['15%','18.75%','22.5%','26.25%','30%'])
        self.assertEqual(params[4]['values'],['8%','11%','14%','17%','20%'])
        self.assertEqual(params[3]['values'],['20']*5)
        self.assertEqual(params[5]['values'],['20']*5)
        self.assertIn('装备者在进入战斗时直接触发以上所有效果',arc['effect']['description'])
        for path in ['/guide/arcs/'+ARC,'/guide/arcs/'+ARC+'.md','/api/guide/arcs/'+ARC]:
            response=self.client.get(path)
            self.assertEqual(response.status_code,200)
            text=json.dumps(response.json,ensure_ascii=False) if response.is_json else response.get_data(as_text=True)
            self.assertIn('静候彩铃',text)
        with Image.open(ROOT/'app/static/images/arcs/fork_elyms.webp') as image:
            self.assertEqual(image.size,(256,256))
            self.assertEqual(image.mode,'RGBA')
            self.assertEqual(image.format,'WEBP')
        with self.client.get('/static/images/arcs/fork_elyms.webp') as response:self.assertEqual(response.status_code,200)

    def test_recommendation_header_editor_and_combined_cultivation(self):
        self.assertEqual(arc_cultivation.recommended({'id':'1090'}),{'id':ARC,'name':'静候彩铃'})
        self.assertIn('href="/guide/arcs/'+ARC+'"',self.client.get('/guide/characters/1090').text)
        page=self.client.get('/guide/characters/1090/cultivation').text
        self.assertIn('value="fork_elyms" selected',page)
        opts=infographic.options('1090')
        self.assertEqual(opts['recommended_arc'],ARC)
        self.assertIn(ARC,{a['id'] for a in opts['arcs']})
        url='/api/guide/characters/1090/cultivation'
        character=self.client.post(url,json={}).json['data']
        combined=self.client.post(url,json={'include_arc':True,'arc_id':ARC}).json['data']
        arc=combined['arc']
        self.assertEqual(arc['name'],'静候彩铃')
        self.assertEqual((arc['experience'],arc['total_gold']),(3640330,1512450))
        self.assertEqual(combined['total_gold'],character['total_gold']+1512450)
        for mat in ['WeaponBreakMaterial_02_lv3','OrdinaryMonMaterial_02_lv3']:
            self.assertIn(mat,{i['id'] for i in arc['items']})
        self.assertEqual(self.client.post(url,json={'include_arc':True,'arc_id':'fork_exe'}).status_code,400)

    def test_updated_character_mechanics_affinity_and_preview_state(self):
        role=service.detail('characters','1090')
        talent=next(a for a in role['abilities'] if a['id']=='GA_Elyms_Passive1')
        text=json.dumps(talent,ensure_ascii=False)
        self.assertIn('持续30秒',text)
        self.assertNotIn('持续3秒',text)
        self.assertIn('或脱战后',json.dumps(next(a for a in role['abilities'] if a['id']=='GA_Elyms_UltraSkill'),ensure_ascii=False))
        self.assertIn('并依次触发',role['resonance'][0]['desc'])
        self.assertEqual(role['affinity']['total_affinity'],56100)
        self.assertEqual(len(role['affinity']['levels']),10)
        self.assertEqual(role['affinity']['full_affinity_bonus'][0]['text'],'攻击力＋5%')
        self.assertEqual(service.detail('characters','preview-exe')['status'],'preview')
        self.assertNotIn('1074',{c['id'] for c in service.catalog()['characters']})


class ArcImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'scripts'))
        cls.addClassCleanup(lambda:sys.path.remove(str(ROOT/'scripts')))
        import sync_guide_ntedata_arcs, refresh_guide_data
        cls.importer=sync_guide_ntedata_arcs;cls.refresh=refresh_guide_data

    def test_piecewise_curve_interpolation_and_invalid_inputs(self):
        curve={'InterpMode':'ERichCurveInterpMode::RCIM_Linear','Keys':[
            {'Time':1,'Value':.15},{'Time':4,'Value':.2625},{'Time':5,'Value':.3}]}
        self.assertEqual(self.importer.curve_value(curve,2),Decimal('.1875'))
        self.assertEqual(self.importer.curve_value(curve,5),Decimal('.3'))
        for broken in [{**curve,'InterpMode':'unknown'},{**curve,'Keys':[]},
                       {**curve,'Keys':[{'Time':1,'Value':1},{'Time':1,'Value':2}]}]:
            with self.assertRaises(ValueError):self.importer.curve_value(broken,2)
        with self.assertRaises(ValueError):self.importer.curve_value(curve,6)

    def test_refresh_preserves_local_arc_without_duplicate_or_remote_override(self):
        payload=json.loads((ROOT/'app/modules/guide/data/ntedata_arcs.json').read_text())
        data={'arcs':[]}
        self.refresh.merge_local_arcs(data,payload);self.refresh.merge_local_arcs(data,payload)
        self.assertEqual([a['id'] for a in data['arcs']],[ARC])
        remote={'arcs':[{'id':ARC,'name':'new remote data'}]}
        self.refresh.merge_local_arcs(remote,payload)
        self.assertEqual(remote['arcs'],[{'id':ARC,'name':'new remote data'}])
        broken=deepcopy(payload);broken['arcs'][0]['materials']=[]
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            self.refresh.merge_local_arcs({'arcs':[]},broken)
