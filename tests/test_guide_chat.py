import base64
import importlib
import importlib.util
import io
import os
import json
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


class HistoryTest(unittest.TestCase):
    def test_six_hour_boundary_and_multimodal_input(self):
        history = importlib.import_module('app.modules.guide.chat_history')
        now = 1000000
        rows = [{'role': 'user', 'text': '旧问题', 'timestamp': (now - 21601) * 1000},
                {'role': 'user', 'text': '新问题', 'timestamp': now * 1000}]
        self.assertEqual(len(history.prepare({'messages': rows}, now)), 1)
        rows[0]['timestamp'] = (now - 21600) * 1000
        self.assertEqual(len(history.prepare({'messages': rows}, now)), 2)
        image = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\nsynthetic').decode()
        rows[-1]['images'] = [image]
        messages = history.prepare({'messages': rows}, now)
        self.assertEqual(messages[-1]['content'][1]['image_url']['url'], image)

    def test_history_cannot_supply_system_tools_remote_images_or_future_times(self):
        history = importlib.import_module('app.modules.guide.chat_history')
        base = {'role': 'user', 'text': 'hello', 'timestamp': 1000000}
        for change in [{'role': 'system'}, {'role': 'tool'}, {'timestamp': float('nan')},
                       {'timestamp': 9999999999999}, {'images': ['https://private.invalid/image']},
                       {'images': ['data:image/png;base64,' + base64.b64encode(b'not-image').decode()]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                history.prepare({'messages': [{**base, **change}]}, now=1000)


class GuideChatTest(unittest.TestCase):
    def setUp(self):
        # Keep already-imported blueprint/filter identities intact for other guide tests.
        self.temp = tempfile.TemporaryDirectory()
        self.db_module = importlib.import_module('app.db')
        self.original_database = self.db_module.db.database
        if not self.db_module.db.is_closed():
            self.db_module.db.close()
        self.db_module.db.init(str(Path(self.temp.name) / 'chat.db'))
        self.dao_module = importlib.import_module('app.dao')
        self.app = importlib.import_module('app').create_app()
        self.client = self.app.test_client()
        self.chat_model = importlib.import_module('app.modules.guide.chat_model')
        self.chat_tools = importlib.import_module('app.modules.guide.chat_tools')
        self.settings = {'base_url': 'https://model.invalid/v1', 'model': 'synthetic-vision',
                         'api_key': 'synthetic-api-secret', 'timeout': 20}
        self.app.config['GUIDE_CHAT_SETTINGS'] = self.settings
        with self.db_module.atomic_transaction():
            self.dao_module.issue_mock_login('chat-owner', 'chat-owner', '654321')
        self.token = self.client.post('/api/auth/login', json={
            'player_uid': 'chat-owner', 'code': '654321'}).json['token']

    def tearDown(self):
        if not self.db_module.db.is_closed():
            self.db_module.db.close()
        self.db_module.db.init(self.original_database)
        self.temp.cleanup()

    def post_chat(self, text='请查明音凛', **kwargs):
        return self.client.post('/api/guide/chat', json={
            'messages': [{'role': 'user', 'text': text, 'timestamp': time.time() * 1000}],
            'page': '/guide/characters/1057?stage=1&token=synthetic-private',
            'page_title': '明音凛 · 角色', **kwargs}, headers={'Authorization': 'Bearer ' + self.token})

    def test_login_only_config_and_unconfigured_model(self):
        self.assertEqual(self.client.get('/api/guide/chat/config').status_code, 401)
        self.assertEqual(self.client.post('/api/guide/chat', json={}).status_code, 401)
        response = self.client.get('/api/guide/chat/config', headers={'Authorization': 'Bearer ' + self.token})
        self.assertTrue(response.json['available'])
        self.assertNotIn('synthetic-api-secret', response.text)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.app.config['GUIDE_CHAT_SETTINGS'] = {}
        self.assertEqual(self.post_chat().status_code, 503)

    def test_real_tool_dispatch_and_ephemeral_page_context(self):
        seen = []
        def model(settings, messages, tools):
            seen.append(json.loads(json.dumps(messages)))
            if len(seen) == 1:
                return {'content': None, 'tool_calls': [{'id': 'call-one', 'type': 'function',
                    'function': {'name': 'wiki_search', 'arguments': '{"query":"明音凛","kind":"characters"}'}}]}
            self.assertIn('1057', messages[-1]['content'])
            return {'content': '可以查看[明音凛资料](/guide/characters/1057)。'}
        with patch.object(self.chat_model, 'complete', side_effect=model):
            response = self.post_chat()
            events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(response.status_code, 200)
        answer = next(row for row in events if row['type'] == 'answer')
        self.assertIn('明音凛', answer['text'])
        self.assertTrue(any(row['url'] == '/guide/characters/1057' for row in answer['sources']))
        self.assertIn('明音凛 · 角色', seen[0][0]['content'])
        self.assertNotIn('synthetic-private', seen[0][0]['content'])
        self.assertNotIn('synthetic-api-secret', response.text)
        self.assertEqual(len(seen[0]), 2)
        self.assertNotIn('当前页面', seen[0][1]['content'][0]['text'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_images_reach_model_and_failures_are_safe(self):
        image = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\nsynthetic').decode()
        def model(settings, messages, tools):
            self.assertEqual(messages[-1]['content'][1]['image_url']['url'], image)
            return {'content': '已读取图片。'}
        with patch.object(self.chat_model, 'complete', side_effect=model):
            response = self.post_chat(messages=[{'role': 'user', 'text': '', 'images': [image], 'timestamp': time.time() * 1000}])
            self.assertIn('已读取图片', response.text)
        with patch.object(self.chat_model, 'complete', side_effect=OSError('synthetic-api-secret')):
            response = self.post_chat()
            self.assertNotIn('synthetic-api-secret', response.text)
            self.assertIn('error', response.text)
        service = importlib.import_module('app.modules.guide.chat_service')
        player = self.dao_module.get_player_by_token(self.token)
        self.assertTrue(service.acquire(player.id))
        service.release(player.id)

    def test_private_profile_tools_are_bound_to_authenticated_player(self):
        player = self.dao_module.get_player_by_token(self.token)
        toolkit = self.chat_tools.Toolkit(player, self.app.extensions['guide_axis_reader'])
        with patch.object(self.chat_tools.read_tools.user_data, 'profiles', return_value=[]) as profiles:
            value, images = toolkit.execute('my_profiles', {})
            self.assertEqual(value, {'data': []})
            self.assertEqual(profiles.call_args.args[0].id, player.id)
        for arguments in [{'user_id': 'someone-else'}, {'token': 'secret'}, {'command': 'sh'}]:
            with self.assertRaises(ValueError):
                toolkit.execute('my_profiles', arguments)
        with self.assertRaises(ValueError):
            toolkit.execute('shell', {'command': 'sh'})

    def test_axis_api_preserves_owner_permissions_and_character_selection(self):
        models = importlib.import_module('app.models')
        owner = self.dao_module.get_player_by_token(self.token)
        other = models.Player.create(player_uid='chat-other')
        mine = models.ShaftAxis.create(owner=owner, title='Mine', visibility='private')
        hidden = models.ShaftAxis.create(owner=other, title='Other private', visibility='private')
        public = models.ShaftAxis.create(owner=other, title='Visible', visibility='public')
        models.ShaftAxisCharacter.create(axis=public, character_id='char_heiyu', slot=0)
        headers = {'Authorization': 'Bearer ' + self.token}
        result = self.client.get('/api/guide/axes?character=1042').json
        self.assertEqual([row['id'] for row in result['items']], [public.id])
        result = self.client.get('/api/guide/axes?scope=mine', headers=headers).json
        self.assertEqual([row['id'] for row in result['items']], [mine.id])
        self.assertEqual(self.client.get('/api/guide/axes?scope=mine').status_code, 401)
        self.assertEqual(self.client.get('/api/guide/axes?character=missing').status_code, 400)
        self.assertEqual(self.client.get(f'/api/guide/axes/{hidden.id}', headers=headers).status_code, 400)
        self.assertEqual(self.client.get(f'/api/guide/axes/{mine.id}', headers=headers).status_code, 200)


class TapTapTest(unittest.TestCase):
    def test_search_extracts_real_moment_ids_not_topic_or_author_ids(self):
        taptap = importlib.import_module('app.modules.guide.taptap')
        payload = {'success': True, 'data': {'list': [{'list': [{'moment': {
            'id_str': '859774296539532888', 'author': {'user': {'id': 123, 'name': '作者'}},
            'topic': {'id_str': '54510255', 'title': '<em>明音凛</em>攻略', 'summary': '配队与手法'}}}]}]}}
        with patch.object(taptap, 'fetch', return_value=json.dumps(payload).encode()) as fetch:
            result = taptap.search('明音凛')
        self.assertEqual(result['items'][0]['id'], '859774296539532888')
        self.assertEqual(result['items'][0]['author'], '作者')
        self.assertEqual(result['items'][0]['title'], '明音凛攻略')
        self.assertIn('/search/v4/agg-search?', fetch.call_args.args[0])
        for address in ['http://127.0.0.1', 'https://evil.invalid/test', 'https://www.taptap.cn@evil.invalid/a']:
            with self.assertRaises(taptap.TapTapError):
                taptap.fetch(address)

    def test_article_reads_first_post_and_distinct_images_not_only_summary(self):
        taptap = importlib.import_module('app.modules.guide.taptap')
        body = '这段正文不是摘要。' * 40
        payload = {'success': True, 'data': {'moment': {'id_str': '859774296539532888',
            'topic': {'title': '一图流', 'summary': '仅摘要', 'footer_images': [
                {'original_url': 'https://img2.tapimg.com/moment/one.png', 'url': 'https://img2.tapimg.com/moment/one-small.jpg'},
                {'original_url': 'https://img2.tapimg.com/moment/two.png'}]}},
            'first_post': {'contents': {'json': [{'children': [{'text': body}]}]}}}}
        def fetch(url, limit=0):
            return json.dumps(payload).encode() if '/webapiv2/' in url else b'\x89PNG\r\n\x1a\nsynthetic'
        with patch.object(taptap, 'fetch', side_effect=fetch) as opening:
            value, images = taptap.read('859774296539532888')
        self.assertEqual(value['text'], body)
        self.assertEqual(value['summary'], '仅摘要')
        self.assertEqual(len(images), 2)
        self.assertEqual([call.args[0] for call in opening.call_args_list[1:]],
            ['https://img2.tapimg.com/moment/one.png', 'https://img2.tapimg.com/moment/two.png'])


class AxisCliTest(unittest.TestCase):
    def test_cli_online_axis_filters_and_auth_are_not_exposed(self):
        from contextlib import redirect_stdout
        from urllib.parse import parse_qs, urlsplit
        path = Path(__file__).resolve().parents[1] / 'app/modules/guide/cli.py'
        spec = importlib.util.spec_from_file_location('chat_axis_cli', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"items":[],"page":1}'
        response.headers.get_content_type.return_value = 'application/json'
        output = io.StringIO()
        with patch.dict(os.environ, {'NTE_GUIDE_TOKEN': 'synthetic-site-secret'}), \
                patch.object(cli.urllib.request, 'build_opener') as opener, redirect_stdout(output):
            opener.return_value.open.return_value = response
            self.assertEqual(cli.main(['axes', '--character', '1057', '--character', '黑羽', '--scope', 'mine']), 0)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(parse_qs(urlsplit(request.full_url).query)['character'], ['1057', '黑羽'])
            self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-site-secret')
            self.assertNotIn('synthetic-site-secret', request.full_url + output.getvalue())


if __name__ == '__main__':
    unittest.main()
