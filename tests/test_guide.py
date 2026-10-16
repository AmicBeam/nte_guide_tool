import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_narrow_reads_and_numbers(self):
        full = self.client.get('/api/guide/characters/1042').json['data']
        narrowed = self.client.get('/api/guide/characters/1042?section=abilities').json['data']
        self.assertEqual(narrowed['abilities'], full['abilities'])
        self.assertNotIn('stats', narrowed)
        challenge = self.client.get('/api/guide/challenges/Abyss_14?stage=12').json['data']
        self.assertEqual([x['id'] for x in challenge['stages']], [12])
        stats = challenge['stages'][0]['halves'][0]['waves'][0]['monsters'][0]['stats']
        self.assertGreater(stats['hp'], 0)
        self.assertIsInstance(stats['resistances']['cosmos'], (int, float))
        md = self.client.get('/guide/characters/1042.md?section=awaken')
        self.assertEqual(md.mimetype, 'text/markdown')
        self.assertIn('叛逃者', md.text)
        self.assertNotIn('<NumGreen>', md.text)
        self.assertEqual(self.client.get('/guide/challenges/Abyss_14?stage=').status_code, 200)
        stage_html = self.client.get('/guide/challenges/Abyss_14?stage=12').text
        self.assertIn('魂', stage_html)
        self.assertNotIn('>psyche<', stage_html)
        self.assertNotIn('1 · 始发站', stage_html)

    def test_errors_and_discovery(self):
        for path, status in [('/api/guide/unknown',404),('/api/guide/arcs/missing',404),
                ('/api/guide/arcs?limit=0',400),('/api/guide/arcs?offset=-1',400),
                ('/api/guide/arcs?limit=abc',400),('/api/guide/characters?rarity=9',400),
                ('/api/guide/characters/1042?section=missing',404),
                ('/api/guide/characters/1042?stage=1',400),
                ('/api/guide/challenges/Abyss_14?stage=999',404)]:
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
        for group in service.KINDS:
            records = service.catalog()[group]
            self.assertEqual(len(records),len({x['id'] for x in records}))
            self.assertNotIn('/Game/',json.dumps(records,ensure_ascii=False))
        for season in service.catalog()['challenges']:
            self.assertEqual(len(season['stages']),season['stage_count'])

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
