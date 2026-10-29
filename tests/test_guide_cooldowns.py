import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guide_skill_cooldowns', ROOT/'scripts/guide_skill_cooldowns.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def asset(identifier, properties=None, parent=None):
    return [dict(Type='BlueprintGeneratedClass', Super={'ObjectPath':parent} if parent else {}),
            dict(Name=f'Default__{identifier}_C', Properties=properties or {})]


class CooldownImporterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.put('CoolDwonCurveTable', [{'Rows': {'shared': {'Keys':[{'Time':1,'Value':16},{'Time':13,'Value':16}]}}}])

    def put(self, name, payload):
        (self.root/(name+'.json')).write_text(json.dumps(payload))

    def source(self):
        return sync.CooldownData(self.root)

    def test_explicit_curve_alias_and_inheritance(self):
        duration={'Curve':{'CurveTable':{'ObjectPath':'/Game/Blueprints/Abilities/CoolDown/CoolDwonCurveTable.0'},'RowName':'shared'}}
        self.put('GA_Parent',asset('GA_Parent',{'CooldownDuration':duration}))
        self.put('GA_Child',asset('GA_Child',{'CancelCooldownDuration':{'Value':.3}},'/Game/Blueprints/Test/GA_Parent.0'))
        self.assertEqual(self.source().resolve('GA_Child'),16)
        duration['Value']=.5
        self.put('GA_Parent',asset('GA_Parent',{'CooldownDuration':duration}))
        self.assertEqual(self.source().resolve('GA_Child'),8)

    def test_zero_is_explicit_missing_stays_unknown_and_cancel_is_not_base(self):
        self.put('GA_Zero',asset('GA_Zero',{'CooldownDuration':{'Value':0}}))
        self.put('GA_Unknown',asset('GA_Unknown',{'CancelCooldownDuration':{'Value':.3}}))
        source=self.source()
        self.assertEqual(source.resolve('GA_Zero'),0)
        self.assertIsNone(source.resolve('GA_Unknown'))
        self.assertIsNone(source.resolve('GA_Absent'))

    def test_invalid_or_variable_curve_and_cyclic_parents_fail(self):
        for value in (True,-1,float('inf'),'16'):
            with self.subTest(value=value),self.assertRaises(ValueError):sync.seconds(value)
        self.put('GA_Loop',asset('GA_Loop',parent='/Game/Test/GA_Loop.0'))
        with self.assertRaisesRegex(ValueError,'Cyclic'):self.source().resolve('GA_Loop')
        self.put('CoolDwonCurveTable',[{'Rows':{'shared':{'Keys':[{'Value':16},{'Value':12}]}}}])
        duration={'Curve':{'CurveTable':{'ObjectPath':'/Game/Blueprints/Abilities/CoolDown/CoolDwonCurveTable.0'},'RowName':'shared'}}
        with self.assertRaisesRegex(ValueError,'Level-dependent'):self.source().scalable_seconds(duration)

    def test_refresh_replaces_old_cooldown_without_rebuilding_groups_or_modifiers(self):
        self.put('GA_Test',asset('GA_Test',{'CooldownDuration':{'Value':16}}))
        row={'key':'cd','label':'冷却时间','values':['15秒']*13,'hit_details':[]}
        extra={'key':'extra','label':'额外每段延长的冷却时间','values':['1秒']*13,'hit_details':[]}
        ability={'id':'GA_Test','type_name':'Skill','base_cooldown':99,'param_rows':[row,extra],
                 'groups':[{'name':'主技能','param_rows':[copy.deepcopy(row),copy.deepcopy(extra)]}]}
        data={'characters':[{'abilities':[ability]}]}
        sync.enrich_cooldowns(data,self.source())
        self.assertEqual(ability['base_cooldown'],16)
        self.assertEqual(row['values'],['16秒']*13)
        self.assertEqual(ability['groups'][0]['param_rows'][0]['values'],row['values'])
        self.assertEqual(extra['values'],['1秒']*13)
        self.assertEqual(len(ability['param_rows']),2)
        first=copy.deepcopy(data)
        self.assertEqual(sync.enrich_cooldowns(data,self.source()),first)

    def test_remote_index_is_scoped_and_must_not_be_truncated(self):
        def fetch(path):
            if path=='contents/Blueprints/Abilities':return [{'name':'Player','sha':'test'}]
            self.assertEqual(path,'git/trees/test?recursive=1')
            return {'truncated':True,'tree':[]}
        with self.assertRaisesRegex(ValueError,'Incomplete'):sync.CooldownData(fetcher=fetch).ability_index()


class GuideCooldownPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app=Flask(__name__,template_folder=str(ROOT/'app/templates'))
        app.register_blueprint(blueprint)
        cls.client=app.test_client()

    def test_all_base_skills_have_a_verified_value_or_explicit_unknown(self):
        abilities=[a for c in service.catalog()['characters'] for a in c['abilities'] if a['type_name'] in ('Skill','UltraSkill')]
        self.assertEqual(len(abilities),50)
        self.assertTrue(all('base_cooldown'in a for a in abilities))
        unknown=[a['id'] for a in abilities if a['base_cooldown'] is None]
        self.assertEqual(unknown,['GA_Jin_UltraSkill'])
        by_id={a['id']:a for a in abilities}
        for identifier,value in [('GA_Lacrimosa_Skill',16),('GA_Lacrimosa_UltraSkill',20),('GA_Hathor_Skill',16),
                                 ('GA_Hathor_UltraSkill',15),('GA_Fadia_UltraSkill',20),('GA_Mint019_UltraSkill',15),
                                 ('GA_Female046_Skill',16),('GA_Female051_Skill',16)]:
            self.assertEqual(by_id[identifier]['base_cooldown'],value)
        for ability in abilities:
            for row in ability['param_rows']:
                if row['label']=='冷却时间':self.assertEqual(set(row['values']),{f'{ability["base_cooldown"]}秒'})

    def test_html_markdown_json_and_level_projection_keep_cooldown(self):
        page=self.client.get('/guide/characters/1003').text
        self.assertIn('基础冷却：16秒',page)
        self.assertIn('基础冷却：20秒',page)
        unknown=self.client.get('/guide/characters/1052').text
        self.assertNotIn('基础冷却：未提供',unknown)
        url='/api/guide/characters/1003?section=abilities&ability_id=GA_Sagiri_Skill&skill_level=10&hit_details=0'
        payload=self.client.get(url).json['data']['abilities'][0]
        self.assertEqual(payload['base_cooldown'],16)
        markdown=self.client.get('/guide/characters/1003.md?section=abilities').text
        self.assertIn('基础冷却（秒）: 16',markdown)
        self.assertNotIn('基础冷却（秒）: 未提供',self.client.get('/guide/characters/1052.md?section=abilities').text)
