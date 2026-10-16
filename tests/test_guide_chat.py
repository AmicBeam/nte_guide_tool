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


def streamed_model(model):
    def stream(*args):
        message = model(*args) if callable(model) else model
        if message.get('content'):
            yield {'type': 'delta', 'text': message['content']}
        yield {'type': 'message', 'message': message}
    return stream


class HistoryTest(unittest.TestCase):
    def test_parallel_tool_replies_precede_external_images(self):
        service = importlib.import_module('app.modules.guide.chat_service')
        calls = [{'id': f'call-{i}', 'type': 'function', 'function': {
            'name': 'read_taptap', 'arguments': '{}'}} for i in range(2)]
        seen = []
        def complete(settings, messages, tools):
            seen.append(json.loads(json.dumps(messages)))
            return {'tool_calls': calls} if len(seen) == 1 else {'content': '已读取。'}
        with patch.object(service.chat_model, 'stream', side_effect=streamed_model(complete)), \
                patch.object(service.chat_tools, 'Toolkit') as toolkit:
            toolkit.return_value.execute.return_value = ({'images_read': 1}, ['data:synthetic'])
            toolkit.return_value.sources = []
            list(service.events({}, [{'role': 'user', 'content': '读取两篇攻略'}], None, None, '/guide'))
        self.assertEqual([row['role'] for row in seen[1]],
                         ['system', 'user', 'assistant', 'tool', 'tool', 'user', 'user'])
        self.assertIn('call-0', seen[1][-2]['content'][0]['text'])

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

    def test_hourly_quota_and_rolling_expiry(self):
        from datetime import datetime, timedelta
        from app.modules.guide import chat_quota, chat_service
        now = datetime(2026, 10, 17, 4)
        with patch.object(chat_quota, 'datetime') as clock, \
                patch.object(self.chat_model, 'stream', side_effect=streamed_model({'content': '回答'})) as model:
            clock.fromisoformat.side_effect = datetime.fromisoformat
            clock.utcnow.return_value = now
            for index in range(10):
                clock.utcnow.return_value = now + timedelta(seconds=index)
                response = self.post_chat()
                self.assertEqual(response.status_code, 200)
                self.assertIn('回答', response.text)
            blocked = self.post_chat()
            self.assertEqual(blocked.status_code, 429)
            self.assertEqual(blocked.json['error']['code'], 'chat_rate_limited')
            self.assertEqual(blocked.headers['Retry-After'], '3591')
            self.assertEqual(blocked.headers['Cache-Control'], 'no-store')
            self.assertIn('10条', blocked.json['error']['message'])
            self.assertEqual(model.call_count, 10)
            # Reopening the database keeps usage, just as a service restart does.
            self.db_module.db.close()
            self.assertEqual(self.post_chat().status_code, 429)
            player = self.dao_module.get_player_by_token(self.token)
            self.assertTrue(chat_service.acquire(player.id))
            chat_service.release(player.id)
            clock.utcnow.return_value = now + timedelta(hours=1)
            response = self.post_chat()
            self.assertEqual(response.status_code, 200)
            self.assertIn('回答', response.text)
            self.assertEqual(self.post_chat().status_code, 429)

    def test_quota_is_per_account_and_excludes_rejected_requests(self):
        from app.modules.guide import chat_quota, chat_service
        from app.models import Player
        owner = self.dao_module.get_player_by_token(self.token)
        self.assertEqual(self.post_chat(messages=[]).status_code, 400)
        self.app.config['GUIDE_CHAT_SETTINGS'] = {}
        self.assertEqual(self.post_chat().status_code, 503)
        self.app.config['GUIDE_CHAT_SETTINGS'] = self.settings
        self.assertTrue(chat_service.acquire(owner.id))
        try:
            self.assertEqual(self.post_chat().status_code, 429)
        finally:
            chat_service.release(owner.id)
        self.assertEqual(chat_quota.GuideChatQuota.select().count(), 0)
        for _ in range(10):
            self.assertEqual(chat_quota.consume(owner.id), 0)
        self.assertGreater(chat_quota.consume(owner.id), 0)
        other = Player.create(player_uid='quota-other')
        self.assertEqual(chat_quota.consume(other.id), 0)
        self.assertEqual(len(json.loads(chat_quota.GuideChatQuota.get(player=owner.id).timestamps)), 10)

    def test_quota_failure_releases_concurrency_slot(self):
        from app.modules.guide import chat_quota, chat_service
        with patch.object(chat_quota, 'consume', side_effect=RuntimeError('quota unavailable')):
            response = self.post_chat()
            self.assertEqual(response.status_code, 500)
        owner = self.dao_module.get_player_by_token(self.token)
        self.assertTrue(chat_service.acquire(owner.id))
        chat_service.release(owner.id)

    def test_parallel_quota_reservations_do_not_exceed_limit(self):
        from concurrent.futures import ThreadPoolExecutor
        from app.modules.guide import chat_quota
        owner = self.dao_module.get_player_by_token(self.token)

        def reserve(_):
            try:
                return chat_quota.consume(owner.id)
            finally:
                self.db_module.db.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(reserve, range(16)))
        self.assertEqual(results.count(0), 10)
        self.assertEqual(sum(value > 0 for value in results), 6)

    def test_real_tool_dispatch_and_ephemeral_page_context(self):
        seen = []
        def model(settings, messages, tools):
            seen.append(json.loads(json.dumps(messages)))
            if len(seen) == 1:
                return {'content': None, 'tool_calls': [{'id': 'call-one', 'type': 'function',
                    'function': {'name': 'wiki_search', 'arguments': '{"query":"明音凛","kind":"characters"}'}}]}
            self.assertIn('1057', messages[-1]['content'])
            return {'content': '可以查看[明音凛资料](/guide/characters/1057)。'}
        with patch.object(self.chat_model, 'stream', side_effect=streamed_model(model)):
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

    def test_answer_deltas_are_delivered_before_final_message(self):
        def model(*args):
            yield {'type': 'delta', 'text': '先收到'}
            yield {'type': 'delta', 'text': '后收到'}
            yield {'type': 'message', 'message': {'content': '先收到后收到'}}
        with patch.object(self.chat_model, 'stream', side_effect=model):
            response = self.post_chat()
            chunks = iter(response.response)
            self.assertEqual(json.loads(next(chunks))['type'], 'status')
            self.assertEqual(json.loads(next(chunks)), {'type': 'answer_delta', 'text': '先收到'})
            self.assertEqual(json.loads(next(chunks)), {'type': 'answer_delta', 'text': '后收到'})
            self.assertEqual(json.loads(next(chunks))['type'], 'answer')
            response.close()

    def test_tools_supply_verified_html_links(self):
        player = self.dao_module.get_player_by_token(self.token)
        toolkit = self.chat_tools.Toolkit(player, self.app.extensions['guide_axis_reader'])
        result, _ = toolkit.execute('wiki_search', {'query': '黑羽', 'kind': 'characters'})
        links = result['items'][0]['links']
        self.assertEqual(links['detail'], '/guide/characters/1042')
        self.assertEqual(links['cultivation'], '/guide/characters/1042/cultivation')
        self.assertEqual(links['builds'], '/guide/characters/1042/builds')
        for url in links.values():
            self.assertEqual(self.client.get(url).status_code, 200, url)
        result, _ = toolkit.execute('wiki_get', {'kind': 'characters', 'id': '1042', 'section': 'abilities'})
        self.assertEqual(result['links'], links)
        result, _ = toolkit.execute('wiki_get', {'kind': 'arcs', 'id': 'fork_twinbirds'})
        self.assertEqual(result['links'], {'detail': '/guide/arcs/fork_twinbirds'})
        result, _ = toolkit.execute('wiki_get', {'kind': 'endless', 'id': 'EndlessTrial_51_100', 'stage': 100})
        self.assertEqual(result['links']['detail'], '/guide/endless/EndlessTrial_51_100?stage=100')

    def test_images_reach_model_and_failures_are_safe(self):
        image = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\nsynthetic').decode()
        def model(settings, messages, tools):
            self.assertEqual(messages[-1]['content'][1]['image_url']['url'], image)
            return {'content': '已读取图片。'}
        with patch.object(self.chat_model, 'stream', side_effect=streamed_model(model)):
            response = self.post_chat(messages=[{'role': 'user', 'text': '', 'images': [image], 'timestamp': time.time() * 1000}])
            self.assertIn('已读取图片', response.text)
        with patch.object(self.chat_model, 'stream', side_effect=OSError('synthetic-api-secret')):
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
            self.assertEqual(value, {'data': [], 'links': {'account': '/guide/account'}})
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
