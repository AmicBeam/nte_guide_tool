import io
import json
from email.message import Message
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from flask import Flask
from app.modules.guide import blueprint, cli


class BeginnerGuideTest(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'app/templates'))
        app.register_blueprint(blueprint)
        self.client = app.test_client()

    def test_discovery_detail_and_shared_toc(self):
        listing = self.client.get('/api/guide/beginner').json
        self.assertEqual(listing['total'], 1)
        item = listing['items'][0]
        self.assertEqual(item['id'], 'getting-started')
        self.assertIn('萌新攻略', self.client.get('/guide').text)
        self.assertIn(item['url'], self.client.get('/guide?kind=beginner').text)
        self.assertTrue(any(row['kind'] == 'beginner' for row in
                            self.client.get('/api/guide/search?q=新手入坑').json['items']))
        page = self.client.get(item['url'])
        self.assertEqual(page.status_code, 200)
        self.assertIn('data-guide-toc', page.text)
        for key in ('gameplay', 'currencies', 'progression', 'wishes'):
            self.assertIn(f'href="#section-{key}"', page.text)
            self.assertIn(f'id="section-{key}"', page.text)
        self.assertIn('阅读完整图文', page.text)
        self.assertNotIn('数值阅读约定', page.text)
        self.assertEqual(self.client.get('/api/guide/beginner/missing').status_code, 404)

    def test_sections_markdown_and_invalid_fields(self):
        full = self.client.get('/api/guide/beginner/getting-started').json['data']
        short = self.client.get('/api/guide/beginner/getting-started?section=wishes').json['data']
        self.assertEqual(short['wishes'], full['wishes'])
        self.assertNotIn('gameplay', short)
        self.assertEqual(short['article_url'], full['article_url'])
        md = self.client.get('/guide/beginner/getting-started.md?section=wishes')
        self.assertIn('## 自选与抽取规划', md.text)
        self.assertIn(full['article_url'], md.text)
        self.assertNotIn('## 游戏玩法', md.text)
        self.assertEqual(self.client.get('/api/guide/beginner/getting-started?section=bad').status_code, 404)
        self.assertEqual(self.client.get('/api/guide/beginner/getting-started?stage=1').status_code, 400)

    def test_cli_reads_real_endpoint_in_both_formats(self):
        class Reply(io.BytesIO):
            def __init__(self, response):
                super().__init__(response.data)
                self.headers = Message()
                self.headers['Content-Type'] = response.content_type

        def opening(url, **kwargs):
            parts = urlsplit(url)
            response = self.client.get(parts.path + ('?' + parts.query if parts.query else ''))
            self.assertEqual(response.status_code, 200)
            return Reply(response)

        for args in (['beginner'], ['beginner', '--format', 'markdown'],
                     ['beginner', '--section', 'wishes'], ['list', 'beginner'],
                     ['get', 'beginner', 'getting-started', '--format', 'markdown']):
            with self.subTest(args=args), patch.object(cli.urllib.request, 'urlopen', side_effect=opening), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(cli.main(args), 0)
                self.assertIn('新手入坑', output.getvalue())
                if args == ['beginner']:
                    self.assertEqual(json.loads(output.getvalue())['kind'], 'beginner')
