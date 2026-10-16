"""Private snapshot ownership and SMS orchestration, using synthetic accounts only."""
import importlib
import json
from unittest.mock import patch

from tests.test_solo_room_flow import RoomFlowTestCase


class GuideUserDataTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.token = self._issue_login_and_get_token('guide-owner')
        self.other = self._issue_login_and_get_token('guide-other')
        self.headers = {'Authorization': 'Bearer ' + self.token}
        self.data = importlib.import_module('app.modules.guide.user_data')
        self.sync = importlib.import_module('app.modules.guide.game_sync')
        self.provider = importlib.import_module('app.modules.guide.game_provider')
        self.raw = [{'id': '1042', 'name': '黑羽', 'alev': 20, 'slev': 3, 'awakenLev': 1,
            'accessToken': 'must-not-store', 'mobile': 'not-stored',
            'skills': [{'id': 'GA_BlackBird_Melee', 'level': 6},
                       {'id': 'GA_BlackBird_Skill', 'level': 12}],
            'fork': {'id': 'fork_test', 'name': '测试弧盘', 'alev': '40', 'token': 'must-not-store'},
            'suit': {'name': '测试卡带', 'core': [{'id': 'equip_a', 'lev': 15,
                     'mainProperties': [{'id': 'attack', 'value': '20%'}]}]}}]

    def post(self, path, payload, token=None):
        return self.client.post('/api/guide/me/' + path, json=payload,
            headers={'Authorization': 'Bearer ' + (token or self.token)})

    def test_missing_login_dependency_fails_before_sending_sms(self):
        with patch.object(self.provider.importlib, 'import_module', side_effect=ModuleNotFoundError('synthetic')), \
                patch.object(self.provider.GameClient, '_laohu') as upstream:
            response = self.post('sync/sms', {'mobile': '13800000000'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json['error']['code'], 'login_dependency_unavailable')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        upstream.assert_not_called()
        self.assertEqual(self.sync._sessions, {})

    def test_native_cipher_failure_is_safe(self):
        with patch.object(self.provider.importlib, 'import_module', side_effect=OSError('synthetic native failure')):
            with self.assertRaises(self.provider.LoginDependencyError):
                self.provider.GameClient().login('13800000000', '123456')

    def test_unexpected_sync_failure_returns_json_and_preserves_snapshot(self):
        player = self.dao_module.get_player_by_uid('guide-owner')
        role = {'role_id': 'one', 'role_name': '旧快照'}
        self.data.save_profile(player, role, self.raw)
        with patch.object(self.sync, 'GameClient') as factory:
            client = factory.return_value
            session = self.post('sync/sms', {'mobile': '13800000000'}).json['data']['session_id']
            client.login.return_value = [role]
            client.characters.side_effect = RuntimeError('synthetic-private-secret')
            with self.assertLogs(self.app.logger, level='ERROR') as logs:
                response = self.post('sync/verify', {'session_id': session, 'code': '123456'})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json['error']['code'], 'private_data_unavailable')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertNotIn('synthetic-private-secret', response.text + str(logs.output))
        self.assertNotIn(session, str(logs.output))
        self.assertIn('RuntimeError', str(logs.output))
        self.assertEqual(self.data.profiles(player)[0]['role_name'], '旧快照')

    def test_skill_id_matching_ignores_case_but_rejects_ambiguous_ids(self):
        character = self.data.service.detail('characters', '1042')
        saved = self.data.normalize(self.raw)[0]
        saved['skills'] = [
            {'id': 'ga_blackbird_melee', 'group': 'skills', 'level': 8},
            {'id': 'Ga_BlackBird_Skill', 'group': 'skills', 'level': 6},
            {'id': 'city_ability_blackbird_01', 'group': 'citySkills', 'level': 3}]
        result = self.data.prefill(character, saved)
        self.assertEqual(result['values']['current_skill_0'], 8)
        self.assertEqual(result['values']['current_skill_1'], 6)
        self.assertEqual(result['values']['current_skill_6'], 3)
        saved['skills'].append({'id': 'GA_BlackBird_Melee', 'group': 'skills', 'level': 7})
        self.assertEqual(self.data.prefill(character, saved)['values']['current_skill_0'], 1)
        saved['skills'] = [{'id': 'unknown', 'name': '掠影', 'group': 'skills', 'level': 8}]
        self.assertEqual(self.data.prefill(character, saved)['values']['current_skill_0'], 1)

    def test_prefill_equipped_arc_level_and_boundary_breakthrough(self):
        character = self.data.service.detail('characters', '1042')
        saved = self.data.normalize(self.raw)[0]
        saved['equipment']['fork'] = {'id': 'FORK_TWINBIRDS', 'alev': '40', 'blev': '3'}
        result = self.data.prefill(character, saved)['values']
        self.assertEqual(result['arc_id'], 'fork_twinbirds')
        self.assertEqual(result['arc_current_level'], 40)
        self.assertTrue(result['arc_current_ascended'])
        self.assertNotIn('include_arc', result)
        saved['equipment']['fork']['blev'] = '2'
        self.assertFalse(self.data.prefill(character, saved)['values']['arc_current_ascended'])
        saved['equipment']['fork']['id'] = 'unknown'
        self.assertNotIn('arc_id', self.data.prefill(character, saved)['values'])

    def test_anonymous_calculator_but_private_sync(self):
        for path in ['profiles', 'sync/sms', 'sync/verify', 'sync/select-role']:
            response = self.client.get('/api/guide/me/' + path) if path == 'profiles' else self.client.post('/api/guide/me/' + path, json={})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.client.get('/guide/characters/1042/cultivation').status_code, 200)
        self.assertEqual(self.client.post('/api/guide/characters/1042/cultivation', json={}).status_code, 200)

    def test_sync_save_read_reopen_and_ownership(self):
        with patch.object(self.sync, 'GameClient') as factory:
            client = factory.return_value
            client.login.return_value = [{'role_id': 'synthetic-role', 'role_name': '测试角色'}]
            client.characters.return_value = self.raw
            response = self.post('sync/sms', {'mobile': '13800000000'})
            self.assertEqual(response.status_code, 200)
            session = response.json['data']['session_id']
            self.assertGreater(len(session), 32)
            denied = self.post('sync/verify', {'session_id': session, 'code': '123456'}, self.other)
            self.assertEqual(denied.status_code, 400)
            result = self.post('sync/verify', {'session_id': session, 'code': '123456'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json['data']['status'], 'success')
            self.assertNotIn(session, self.sync._sessions)
        self.db_module.db.close()
        stored = self.client.get('/api/guide/me/profiles', headers=self.headers)
        self.assertEqual(len(stored.json['data']), 1)
        self.assertNotIn('must-not-store', stored.text)
        self.assertNotIn('13800000000', stored.text)
        character = stored.json['data'][0]['characters'][0]
        self.assertEqual(character['equipment']['suit']['core'][0]['level'], 15)
        self.assertEqual(character['equipment']['fork']['name'], '测试弧盘')
        self.assertEqual(self.client.get('/api/guide/me/profiles', headers={'Authorization': 'Bearer ' + self.other}).json['data'], [])
        url = '/api/guide/me/profiles/synthetic-role/characters/1042/prefill'
        self.assertEqual(self.client.get(url, headers={'Authorization': 'Bearer ' + self.other}).status_code, 404)
        fill = self.client.get(url, headers=self.headers).json['data']
        self.assertEqual(fill['values']['current_level'], 20)
        self.assertEqual(fill['values']['current_skill_0'], 6)
        self.assertEqual(fill['values']['current_skill_1'], 1)  # no clamping a display level
        self.assertTrue(any('突破' in x for x in fill['warnings']))
        self.assertTrue(any('基础等级' in x for x in fill['warnings']))
        self.client.delete('/api/guide/me/profiles/synthetic-role', headers={'Authorization': 'Bearer ' + self.other})
        self.assertEqual(len(self.client.get('/api/guide/me/profiles', headers=self.headers).json['data']), 1)
        self.client.delete('/api/guide/me/profiles/synthetic-role', headers=self.headers)
        self.assertEqual(self.client.get('/api/guide/me/profiles', headers=self.headers).json['data'], [])

    def test_multiple_roles_retries_and_failed_import_preserves_old_snapshot(self):
        player = self.dao_module.get_player_by_uid('guide-owner')
        role = {'role_id': 'one', 'role_name': '旧快照'}
        self.data.save_profile(player, role, self.raw)
        with patch.object(self.sync, 'GameClient') as factory:
            client = factory.return_value
            client.login.return_value = [role, {'role_id': 'two', 'role_name': '另一个'}]
            session = self.post('sync/sms', {'mobile': '13800000000'}).json['data']['session_id']
            response = self.post('sync/verify', {'session_id': session, 'code': '123456'})
            self.assertEqual(response.json['data']['status'], 'select_role')
            self.assertFalse(client.characters.called)
            self.assertEqual(self.post('sync/select-role', {'session_id': session, 'role_id': 'foreign'}).status_code, 400)
            client.characters.side_effect = self.provider.ProviderError('服务暂不可用')
            self.assertEqual(self.post('sync/select-role', {'session_id': session, 'role_id': 'one'}).status_code, 502)
            self.assertEqual(self.data.profiles(player)[0]['role_name'], '旧快照')
            client.characters.side_effect = None
            client.characters.return_value = []
            self.assertEqual(self.post('sync/select-role', {'session_id': session, 'role_id': 'one'}).status_code, 400)
            self.assertEqual(len(self.data.profiles(player)[0]['characters']), 1)
            client.characters.return_value = self.raw
            self.assertEqual(self.post('sync/select-role', {'session_id': session, 'role_id': 'two'}).status_code, 200)
            self.assertEqual(len(self.data.profiles(player)), 2)

    def test_limits_and_expiration(self):
        self.assertEqual(self.post('sync/sms', {'mobile': 'invalid'}).status_code, 400)
        with patch.object(self.sync, 'GameClient'):
            session = self.post('sync/sms', {'mobile': '13800000000'}).json['data']['session_id']
            self.assertEqual(self.post('sync/sms', {'mobile': '13800000000'}).status_code, 400)
            for _ in range(6):
                with patch.object(self.sync._sessions[session].client, 'login', side_effect=self.provider.ProviderError('无效验证码')) if session in self.sync._sessions else patch.object(self.sync, 'GameClient'):
                    response = self.post('sync/verify', {'session_id': session, 'code': '123456'})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(session, self.sync._sessions)
            self.sync._sessions['expired'] = self.sync.Session(
                self.dao_module.get_player_by_uid('guide-owner').id, '', self.sync.GameClient(), 0)
            self.assertEqual(self.post('sync/verify', {'session_id': 'expired', 'code': '123456'}).status_code, 400)
            self.assertNotIn('expired', self.sync._sessions)

    def test_provider_protocol_and_credentials_not_returned(self):
        client = self.provider.GameClient(device='synthetic-device')
        replies = [{}, {'userId': '123', 'token': 'temporary-laohu'}, {'accessToken': 'temporary-access'},
                   {'roles': [{'roleId': 'one', 'roleName': '测试'}]}, self.raw]
        with patch.object(self.provider, 'request_json', side_effect=replies) as network:
            roles = client.login('13800000000', '123456')
            self.assertEqual(roles, [{'role_id': 'one', 'role_name': '测试'}])
            sms_payload = network.call_args_list[1].args[1]
            self.assertNotEqual(sms_payload['cellphone'], '13800000000')
            self.assertNotEqual(sms_payload['captcha'], '123456')
            self.assertEqual(client.characters('one'), self.raw)
            self.assertIn('roleId=one', network.call_args.args[0])
        normalized = self.data.normalize(self.raw)
        self.assertNotIn('must-not-store', json.dumps(normalized))
        with self.assertRaises(ValueError):
            self.data.normalize(self.raw + self.raw)
