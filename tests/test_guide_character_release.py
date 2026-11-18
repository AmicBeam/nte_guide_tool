"""Server UTC+8 release boundaries apply to every public character reader."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from flask import Flask
from app.modules.guide import blueprint, service, previews, chat_tools, character_release as release

ROOT=Path(__file__).resolve().parents[1]
PROTECTED=('abilities','awaken','resonance','nouns','equip_slots','affinity','aliases','experience','reference_details')


class CharacterReleaseTest(unittest.TestCase):
    def setUp(self):
        app=Flask(__name__,template_folder=str(ROOT/'app/templates'),static_folder=str(ROOT/'app/static'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/login',endpoint='main.login_page',view_func=lambda:'login')
        self.client=app.test_client()
        self.raw=next(c for c in service.catalog()['characters']if c['id']=='1090')
        self.original=deepcopy(self.raw)

    def clock(self,value):
        return patch.object(release,'current_time',return_value=datetime.fromisoformat(value))

    def test_test_permission_previews_without_changing_public_cache_or_deadline(self):
        from types import SimpleNamespace
        users = {'tester': SimpleNamespace(shaft_test_whitelisted=True),
                 'regular': SimpleNamespace(shaft_test_whitelisted=False)}
        paths = ['/guide/characters/1074', '/api/guide/characters/1074',
                 '/guide/characters/1074.md', '/api/guide/characters/1074?section=abilities',
                 '/guide/arcs/fork_exe', '/api/guide/arcs/fork_exe?section=effect']
        original = deepcopy(service.catalog())
        with self.clock('2026-11-26T11:59:59+08:00'), patch('app.dao.get_player_by_token', side_effect=users.get):
            for token, allowed in [('tester', True), ('regular', False), ('expired', False), ('', False)]:
                headers = {'Authorization': 'Bearer ' + token} if token else {}
                with self.subTest(token=token):
                    state = self.client.get('/api/guide/characters/1074/release', headers=headers).json['data']
                    self.assertEqual(state['content_released'], allowed)
                    self.assertEqual(state['release_at'], '2026-11-26T12:00:00+08:00')
                    for path in paths:
                        response = self.client.get(path, headers=headers)
                        self.assertEqual(response.status_code, 200 if allowed or '?section=' not in path else 404, path)
                        self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    page = self.client.get(paths[0], headers=headers).text
                    self.assertEqual('id="section-abilities"' in page, allowed)
                    self.assertIn('test_preview.js', page)
                    api = self.client.get(paths[1], headers=headers).json['data']
                    self.assertEqual('abilities' in api, allowed)
                    # A privileged request must never make subsequent anonymous reads public.
                    self.assertNotIn('abilities', self.client.get(paths[1]).json['data'])
        self.assertEqual(service.catalog(), original)

    def test_before_release_preserves_header_but_redacts_all_later_sections(self):
        with self.clock('2026-11-05T09:59:59+08:00'):
            row=service.detail('characters','1090')
            self.assertFalse(row['content_released'])
            self.assertEqual(row['release_at'],'2026-11-05T10:00:00+08:00')
            self.assertTrue(row['stats']);self.assertTrue(row['char_tags']);self.assertTrue(row['assets'])
            for key in PROTECTED:self.assertNotIn(key,row)
            self.assertEqual(row['materials']['skills'],[])
            self.assertNotIn('GA_Elyms_',json.dumps(row))
            html=self.client.get('/guide/characters/1090').text
            for text in ('guide-character-header','guide-character-overview','15392','660','909',
                         '空心花籽','guide-character-tools','2026-11-05 10:00','data-character-release-url'):
                self.assertIn(text,html)
            for key in PROTECTED:self.assertNotIn('id="section-'+key+'"',html)
            self.assertNotIn('data-guide-toc',html)
            self.assertNotIn('数值阅读约定',html)
            bookmark=self.client.get('/guide/characters/1090?section=affinity#section-affinity')
            self.assertEqual(bookmark.status_code,200)
            self.assertIn('guide-character-header',bookmark.text)
            self.assertNotIn('id="section-affinity"',bookmark.text)
            for path in ['/api/guide/characters/1090','/api/guide/characters/1090?section=overview',
                         '/guide/characters/1090.md','/guide?kind=characters','/guide/account']:
                response=self.client.get(path)
                self.assertEqual(response.status_code,200,path)
                self.assertEqual(response.headers['Cache-Control'],'no-store')
                for secret in ('中继节点','沉入电子梦','GA_Elyms_','信号强度','复古风手机壳'):
                    self.assertNotIn(secret,response.text,path)
            overview=self.client.get('/api/guide/characters/1090?section=overview').json['data']
            self.assertEqual(overview['available_sections'],['stats','materials'])
            for field in PROTECTED:
                for base in ['/api/guide/characters/1090','/guide/characters/1090.md']:
                    blocked=self.client.get(base+'?section='+field)
                    self.assertEqual(blocked.status_code,404)
                    self.assertIn('2026-11-05 10:00',blocked.json['error']['message'])
            for query in ('?ability_id=GA_Elyms_Skill','?section=abilities&skill_level=13&hit_details=1'):
                self.assertEqual(self.client.get('/api/guide/characters/1090'+query).status_code,404)
        self.assertEqual(self.raw,self.original)

    def test_exact_boundary_and_utc_equivalence_without_cache_invalidation(self):
        service.catalog()  # Warm full-data cache before crossing the deadline.
        for time,expected in [('2026-11-05T01:59:59+00:00',False),('2026-11-05T02:00:00+00:00',True),
                              ('2026-11-05T10:00:01+08:00',True),('2026-11-05T09:59:59+08:00',False)]:
            with self.subTest(time=time),self.clock(time):
                data=service.detail('characters','1090')
                self.assertEqual(data['content_released'],expected)
                self.assertEqual('abilities'in data,expected)
                response=self.client.get('/api/guide/characters/1090?section=abilities')
                self.assertEqual(response.status_code,200 if expected else 404)
                if expected:self.assertTrue(response.json['data']['content_released'])
                html=self.client.get('/guide/characters/1090').text
                self.assertEqual('id="section-affinity"'in html,expected)
                self.assertEqual('id="section-abilities"'in html,expected)
                self.assertEqual('data-character-release-url'in html,not expected)
                state=self.client.get('/api/guide/characters/1090/release?now=2099-01-01').json['data']
                self.assertEqual(state['content_released'],expected)
                self.assertEqual(state['check_after_ms'],0 if expected else 1000)
                states={row['name']:row for row in release.publication_states()}
                self.assertEqual(states['曳尔沐']['content_released'],expected)
                self.assertFalse(states['Exe']['content_released'])
        self.assertEqual(self.raw,self.original)

    def test_search_filter_text_and_assistant_share_the_gate(self):
        for time,visible in [('2026-11-05T09:59:59+08:00',False),('2026-11-05T10:00:00+08:00',True)]:
            with self.clock(time):
                api=self.client.get('/api/guide/search?kind=characters&q=中继节点').json
                self.assertEqual(any(r['id']=='1090'for r in api['items']),visible)
                items=service.filter_index_items('characters')
                own=next(r for r in items if r['id']=='1090')
                self.assertEqual('中继节点'in own['filter_text'],visible)
                toolkit=chat_tools.Toolkit(None,None)
                result,_=toolkit.execute('wiki_get',{'kind':'characters','id':'1090'})
                self.assertEqual('中继节点'in json.dumps(result,ensure_ascii=False),visible)
                self.assertEqual(result['content_released'],visible)
                found,_=toolkit.execute('wiki_search',{'kind':'characters','query':'中继节点'})
                self.assertEqual(any(r['id']=='1090'for r in found['items']),visible)
                if not visible:
                    with self.assertRaises(LookupError):
                        toolkit.execute('wiki_get',{'kind':'characters','id':'1090','ability_id':'GA_Elyms_Skill'})

    def test_cultivation_keeps_public_growth_without_kit_or_silent_skill_costs(self):
        with self.clock('2026-11-05T09:59:59+08:00'):
            page=self.client.get('/guide/characters/1090/cultivation')
            self.assertEqual(page.status_code,200)
            self.assertIn('当前仅计算角色与弧盘养成',page.text)
            self.assertNotIn('GA_Elyms_',page.text)
            response=self.client.post('/api/guide/characters/1090/cultivation',json={'target_skill_0':10})
            self.assertEqual(response.status_code,200)
            value=response.json['data']
            self.assertFalse(value['skills_included'])
            self.assertTrue(value['items'])
            self.assertNotIn('GA_Elyms_',response.text)
            toolkit=chat_tools.Toolkit(None,None)
            result,_=toolkit.execute('calculate_cultivation',{'character':'1090','values':{}})
            self.assertFalse(result['skills_included'])
            self.assertNotIn('沉入电子梦',json.dumps(result,ensure_ascii=False))

    def test_exe_reference_data_is_gated_even_when_added_to_preview(self):
        records=deepcopy(previews.characters());row=deepcopy(records[0]);row.update(id='synthetic-exe-preview',name='Exe');records.append(row)
        row['reference_details']={'abilities':[{'id':'GA_Exe_hidden','name':'EXE_SECRET',
            'type_name':'Melee','phases':[{'name':'EXE_SECRET','desc':'EXE_PRIVATE_EFFECT'}],
            'param_rows':[],'groups':[{'name':'EXE_SECRET','description':'EXE_PRIVATE_EFFECT','param_rows':[]}]}],
            'awaken':[],'resonance':[]}
        row['assets']['fullbody']=row['preview_image']
        for time,visible in [('2026-11-26T11:59:59+08:00',False),('2026-11-26T04:00:00+00:00',True)]:
            with patch.object(previews,'characters',return_value=records),self.clock(time):
                result=service.detail('characters','synthetic-exe-preview')
                self.assertEqual(result['content_released'],visible)
                for path in ['/api/guide/characters/synthetic-exe-preview','/guide/characters/synthetic-exe-preview.md',
                             '/guide/characters/synthetic-exe-preview']:
                    response=self.client.get(path)
                    self.assertEqual(response.status_code,200)
                    self.assertEqual('EXE_SECRET'in response.text,visible)
                self.assertEqual(self.client.get('/api/guide/characters/synthetic-exe-preview?section=reference_details').status_code,
                                 200 if visible else 404)
        for record in [{'id':'synthetic-new-exe','name':'Exe'},
                       {'id':'synthetic-new-exe','name':'测试角色','aliases':['Exe']}]:
            self.assertFalse(release.metadata(record,datetime.fromisoformat('2026-11-26T11:59:59+08:00'))['content_released'])

    def test_unconfigured_characters_and_timezone_validation(self):
        with self.clock('2026-11-01T00:00:00+08:00'):
            black=service.detail('characters','1042')
            self.assertIn('abilities',black)
            self.assertNotIn('release_at',black)
            self.assertTrue(self.client.get('/api/guide/characters/1042/release').json['data']['content_released'])
        with self.assertRaises(ValueError):release.metadata(self.raw,datetime(2026,11,5,10))
