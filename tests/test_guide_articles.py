"""Private autosaved drafts, explicit public snapshots and safe Markdown rendering."""
import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4


class GuideArticlesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = importlib.import_module('app.db')
        self.original_database = self.db.db.database
        if not self.db.db.is_closed():
            self.db.db.close()
        self.db.db.init(str(Path(self.temp.name) / 'articles.db'))
        self.app = importlib.import_module('app').create_app()
        self.client = self.app.test_client()
        self.articles = importlib.import_module('app.modules.guide.articles')
        self.rendering = importlib.import_module('app.modules.guide.article_markdown')
        self.service = importlib.import_module('app.modules.guide.service')
        dao = importlib.import_module('app.dao')
        self.tokens = []
        for uid in ('synthetic-author', 'synthetic-other'):
            with self.db.atomic_transaction():
                dao.issue_mock_login(uid, uid, '654321')
            self.tokens.append(self.client.post('/api/auth/login', json={'player_uid': uid, 'code': '654321'}).json['token'])
        self.headers = {'Authorization': 'Bearer ' + self.tokens[0]}
        self.other = {'Authorization': 'Bearer ' + self.tokens[1]}
        self.ident = str(uuid4())

    def tearDown(self):
        if not self.db.db.is_closed():
            self.db.db.close()
        self.db.db.init(self.original_database)
        self.temp.cleanup()

    def save(self, *, revision=0, title='测试攻略', markdown='# 开始\n\n正文', target_kind='', target_id='', ident=None, headers=None):
        return self.client.put('/api/guide/me/articles/' + (ident or self.ident), json={
            'revision': revision, 'title': title, 'markdown': markdown,
            'target_kind': target_kind, 'target_id': target_id}, headers=headers or self.headers)

    def publish(self, revision=1, ident=None, headers=None):
        return self.client.post('/api/guide/me/articles/' + (ident or self.ident) + '/publish',
                                json={'revision': revision}, headers=headers or self.headers)

    def test_private_drafts_require_owner_and_survive_reconnect(self):
        path = '/api/guide/me/articles/' + self.ident
        self.assertEqual(self.client.put(path, json={}).status_code, 401)
        response = self.save()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['data']['revision'], 1)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.db.db.close()
        self.assertEqual(self.client.get(path, headers=self.headers).json['data']['markdown'], '# 开始\n\n正文')
        self.assertEqual(self.client.get('/api/guide/me/articles', headers=self.other).json['items'], [])
        self.assertEqual(self.client.get(path, headers=self.other).status_code, 404)
        self.assertEqual(self.save(revision=1, headers=self.other).status_code, 404)
        self.assertEqual(self.publish(headers=self.other).status_code, 404)
        self.assertEqual(self.client.get('/guide/articles/' + self.ident).status_code, 404)
        self.assertEqual(self.client.get('/api/guide/articles').json['items'], [])

    def test_stale_saves_and_publishes_cannot_replace_newer_work(self):
        self.assertEqual(self.save().status_code, 200)
        self.assertEqual(self.save(revision=1, markdown='最新草稿').json['data']['revision'], 2)
        stale = self.save(revision=1, markdown='过期标签页')
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.publish(revision=1).status_code, 409)
        self.assertEqual(self.client.get('/api/guide/me/articles/' + self.ident, headers=self.headers).json['data']['markdown'], '最新草稿')

    def test_autosaved_edits_do_not_change_published_snapshot(self):
        self.save(title='旧公开标题', markdown='旧公开正文')
        self.assertEqual(self.publish().json['data']['revision'], 2)
        self.assertEqual(self.save(revision=2, title='新标题', markdown='尚未发布的秘密').status_code, 200)
        page = self.client.get('/guide/articles/' + self.ident)
        self.assertIn('旧公开正文', page.text)
        self.assertNotIn('尚未发布的秘密', page.text)
        self.assertNotIn('尚未发布的秘密', self.client.get('/api/guide/articles/' + self.ident).text)
        self.assertEqual(self.client.get('/api/guide/search?q=尚未发布的秘密').json['total'], 0)
        self.assertEqual(self.publish(revision=3).status_code, 200)
        self.assertIn('尚未发布的秘密', self.client.get('/guide/articles/' + self.ident).text)
        self.assertIn('新标题', self.client.get('/guide?kind=beginner').text)

    def test_targets_and_attachment_placement_for_all_four_kinds(self):
        choices = self.client.get('/api/guide/article-targets').json['items']
        self.assertEqual({item['kind'] for item in choices}, set(self.articles.TARGET_KINDS))
        self.assertFalse(any(item['id'] == 'preview-exe' for item in choices))
        for kind in self.articles.TARGET_KINDS:
            target = next(item for item in choices if item['kind'] == kind)
            ident = str(uuid4())
            self.assertEqual(self.save(title=kind+'挂载攻略', ident=ident, target_kind=kind, target_id=target['id']).status_code, 200)
            self.assertEqual(self.publish(ident=ident).status_code, 200)
            page = self.client.get(f'/guide/{kind}/{target["id"]}')
            self.assertEqual(page.status_code, 200)
            self.assertIn('aria-label="相关攻略"', page.text)
            self.assertIn(kind+'挂载攻略', page.text)
            related = page.text.index('aria-label="相关攻略"')
            if kind == 'characters':
                self.assertLess(page.text.index('guide-character-tools'), related)
            if kind == 'challenges':
                self.assertLess(related, page.text.index('id="section-resonance"'))
            self.assertNotIn('数值按等级或精炼档位顺序列出；敌人抗性为小数比例。完整字段可通过 JSON 查看。', page.text)

    def test_invalid_targets_and_fields_do_not_mutate_saved_draft(self):
        self.save()
        self.assertEqual(self.save(revision=1, target_kind='characters', target_id='missing').status_code, 400)
        self.assertEqual(self.save(revision=1, target_kind='characters', target_id='preview-exe').status_code, 400)
        self.assertEqual(self.save(revision=True).status_code, 400)
        self.assertEqual(self.save(ident='invalid').status_code, 404)
        self.assertEqual(self.save(revision=1, markdown='x'*150001).status_code, 400)
        self.assertEqual(self.client.get('/api/guide/me/articles/' + self.ident, headers=self.headers).json['data']['revision'], 1)
        self.save(revision=1, title='', markdown='')
        self.assertEqual(self.publish(revision=2).status_code, 400)

    def test_preview_escapes_html_and_only_generates_valid_bilibili_iframes(self):
        text = ('# 第一节\n\n<script>alert(1)</script>\n\n<iframe src="https://evil.invalid"></iframe>\n\n'
                '[坏链接](javascript:alert%281%29)\n\n![图片](data:image/svg+xml,attack)\n\n'
                '```bilibili\nhttps://www.bilibili.com/video/BV1JFeb6vE6o\n```\n\n'
                '```bilibili\nhttps://evil.invalid/video/BV1JFeb6vE6o\n```')
        response = self.client.post('/api/guide/articles/preview', json={'markdown': text}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        html = response.json['html']
        self.assertNotIn('<script>', html)
        self.assertNotIn('<iframe src="https://evil', html)
        self.assertNotIn('href="javascript:', html)
        self.assertNotIn('src="data:', html)
        self.assertEqual(html.count('<iframe'), 1)
        self.assertIn('player.bilibili.com/player.html?bvid=BV1JFeb6vE6o', html)
        self.assertIn('autoplay=0', html)
        self.assertEqual(response.json['toc'][0]['title'], '第一节')
        self.assertEqual(self.client.post('/api/guide/articles/preview', json={'markdown':'private'}).status_code, 401)
        for link in ('javascript:alert(1)', 'https://[invalid', '//evil.invalid', 'data:text/html,test'):
            self.assertEqual(self.rendering.safe_url(link), '')

    def test_listing_search_and_group_labels_hide_internal_identifiers(self):
        self.save(title='独立示例攻略', markdown='只出现在正文中的检索词')
        self.publish()
        page = self.client.get('/guide?kind=beginner')
        self.assertIn('我要撰写', page.text)
        self.assertIn('通用攻略', page.text)
        self.assertIn('自建攻略', page.text)
        self.assertNotIn('萌新攻略档案', page.text)
        self.assertNotIn('<code>damage-mechanics</code>', page.text)
        self.assertNotIn('<code>getting-started</code>', page.text)
        self.assertIn('独立示例攻略', self.client.get('/guide?kind=beginner&q=检索词').text)
        result = self.client.get('/api/guide/search?q=检索词').json
        self.assertEqual(result['items'][0]['kind'], 'articles')
        self.assertIn('只出现在正文中的检索词', self.client.get('/guide/articles/'+self.ident+'.md').text)
        self.assertNotIn('aria-label="相关攻略"', self.client.get('/guide/characters/1042').text)

    def test_draft_attachment_changes_only_take_effect_when_republished(self):
        self.save(target_kind='characters', target_id='1042')
        self.publish()
        self.save(revision=2, target_kind='endless', target_id='EndlessTrial_1_50')
        self.assertIn('/guide/articles/'+self.ident, self.client.get('/guide/characters/1042').text)
        self.assertNotIn('/guide/articles/'+self.ident, self.client.get('/guide/endless/EndlessTrial_1_50').text)
        self.publish(revision=3)
        self.assertNotIn('/guide/articles/'+self.ident, self.client.get('/guide/characters/1042').text)
        self.assertIn('/guide/articles/'+self.ident, self.client.get('/guide/endless/EndlessTrial_1_50').text)
