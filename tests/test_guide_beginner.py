import io
import json
from email.message import Message
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from flask import Flask
from app.modules.guide import blueprint, cli, service
from PIL import Image


class BeginnerGuideTest(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'app/templates'))
        app.register_blueprint(blueprint)
        self.client = app.test_client()

    def test_discovery_detail_and_shared_toc(self):
        listing = self.client.get('/api/guide/beginner').json
        self.assertEqual(listing['total'], 3)
        item = listing['items'][0]
        self.assertEqual(item['id'], 'getting-started')
        self.assertIn('萌新攻略', self.client.get('/guide').text)
        self.assertIn(item['url'], self.client.get('/guide?kind=beginner').text)
        self.assertTrue(any(row['kind'] == 'beginner' for row in
                            self.client.get('/api/guide/search?q=新手入坑').json['items']))
        page = self.client.get(item['url'])
        self.assertEqual(page.status_code, 200)
        self.assertIn('data-guide-toc', page.text)
        for key in service.BEGINNER_SECTIONS:
            self.assertIn(f'href="#section-{key}"', page.text)
            self.assertIn(f'id="section-{key}"', page.text)
        self.assertIn('作者原帖', page.text)
        self.assertIn('href="#article-hunter"', page.text)
        self.assertIn('id="article-hunter"', page.text)
        self.assertNotIn('数值阅读约定', page.text)
        self.assertEqual(self.client.get('/api/guide/beginner/missing').status_code, 404)

    def test_sections_markdown_and_invalid_fields(self):
        full = self.client.get('/api/guide/beginner/getting-started').json['data']
        short = self.client.get('/api/guide/beginner/getting-started?section=wishes').json['data']
        self.assertEqual(short['wishes'], full['wishes'])
        self.assertNotIn('gameplay', short)
        self.assertEqual(short['article_url'], full['article_url'])
        md = self.client.get('/guide/beginner/getting-started.md?section=wishes')
        self.assertIn('## 四.抽卡资源规划', md.text)
        self.assertIn(full['article_url'], md.text)
        self.assertNotIn('## 一.游戏玩法讲解', md.text)
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

    def test_full_text_and_local_images_are_complete(self):
        record = self.client.get('/api/guide/beginner/getting-started').json['data']
        blocks = [block for key in service.BEGINNER_SECTIONS for block in record[key]]
        text = '\n'.join(block.get('text', '') for block in blocks)
        self.assertGreater(len(text), 12000)
        for phrase in ('前言', '结语'):
            self.assertIn(phrase, self.client.get('/guide/beginner/getting-started.md').text)
        for phrase in ('2700*1.09*24=70632', '3.2异环1.4版本活动简述', '祝大家都能抽出自己想要的角色和盲盒，不要碰刮刮乐喔！'):
            self.assertIn(phrase, text)
        figures = [block for block in blocks if block['type'] == 'image']
        self.assertEqual(len(figures), 65)
        self.assertEqual(sum(block['type'] == 'heading' for block in blocks), 9)
        images = figures + [run for block in blocks for run in block.get('runs', []) if 'src' in run]
        self.assertEqual(len(images), 72)
        folder = Path(__file__).resolve().parents[1] / 'app/modules/guide/static'
        for image in images:
            self.assertTrue(image['src'].startswith('/static/guide/beginner/getting-started/'))
            with Image.open(folder / image['src'].removeprefix('/static/guide/')) as picture:
                self.assertEqual(picture.size, (image['width'], image['height']))
                picture.verify()
        md = self.client.get('/guide/beginner/getting-started.md').text
        self.assertEqual(md.count('![攻略配图 '), 65)
        for image in figures:
            self.assertIn(image['src'], md)
        response = self.client.get('/guide/beginner/getting-started')
        self.assertEqual(response.text.count('class="guide-article-figure"'), 65)
        self.assertNotIn('>TapTap<', response.text)
        with self.client.get(figures[0]['src']) as image_response:
            self.assertEqual(image_response.status_code, 200)
        self.assertEqual(self.client.get('/api/guide/beginner/getting-started?section=events').json['data']['events'], record['events'])
        self.assertNotIn('gameplay', self.client.get('/api/guide/beginner/getting-started?section=events').json['data'])

    def test_combat_video_embeds_official_player_without_autoplay(self):
        page = self.client.get('/guide/beginner/combat-mechanics')
        self.assertEqual(page.status_code, 200)
        self.assertIn('战斗机制深入攻略', page.text)
        self.assertIn('https://player.bilibili.com/player.html?bvid=BV1JFeb6vE6o&amp;page=1&amp;autoplay=0', page.text)
        self.assertIn('allowfullscreen', page.text)
        self.assertIn('title="战斗机制深入攻略"', page.text)
        self.assertIn('在哔哩哔哩观看', page.text)
        self.assertIn('/guide/beginner/combat-mechanics', self.client.get('/guide?kind=beginner').text)
        data = self.client.get('/api/guide/beginner/combat-mechanics').json['data']
        self.assertEqual(data['gameplay'][0]['bvid'], 'BV1JFeb6vE6o')
        md = self.client.get('/guide/beginner/combat-mechanics.md').text
        self.assertIn('[战斗机制深入攻略](https://www.bilibili.com/video/BV1JFeb6vE6o)', md)
        self.assertNotIn('<iframe', md)

    def test_nga_guide_is_second_in_web_and_api(self):
        rows = self.client.get('/api/guide/beginner').json['items']
        self.assertEqual([row['id'] for row in rows], ['getting-started', 'damage-mechanics', 'combat-mechanics'])
        index = self.client.get('/guide?kind=beginner').text
        self.assertLess(index.index('href="/guide/beginner/getting-started"'), index.index('href="/guide/beginner/damage-mechanics"'))
        self.assertLess(index.index('href="/guide/beginner/damage-mechanics"'), index.index('href="/guide/beginner/combat-mechanics"'))
        page = self.client.get('/guide/beginner/damage-mechanics')
        self.assertEqual(page.status_code, 200)
        self.assertIn('异环数据机制导论', page.text)
        self.assertIn('https://bbs.nga.cn/read.php?tid=46659382', page.text)
        self.assertIn('666bj', page.text)
        self.assertNotIn('在哔哩哔哩观看', page.text)
        self.assertIn('https://bbs.nga.cn/read.php?tid=46659382', self.client.get('/guide/beginner/damage-mechanics.md').text)
