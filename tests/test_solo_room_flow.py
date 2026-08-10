import importlib
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


DEFAULT_ITEMS = [
    'genesis_urban_energy',
    'genesis_urban_energy',
    'genesis_urban_energy',
    'genesis_refresh_charge',
    'genesis_refresh_charge',
    'genesis_refresh_charge',
    'genesis_breakfast_bag',
    'genesis_breakfast_bag',
    'genesis_breakfast_bag',
    'genesis_marble_soda',
    'genesis_marble_soda',
    'genesis_eborn_cake',
    'genesis_eborn_cake',
    'genesis_eborn_cake',
    'genesis_chip_washer',
    'genesis_chip_washer',
    'genesis_nest_shard',
    'genesis_nest_shard',
    'delay_commute_bag',
    'delay_commute_bag',
]

DEFAULT_ESPERS = [
    'nanali',
    'bohe',
    'xun',
    'jiuyuan',
]


def _purge_app_modules() -> None:
    stale_modules = [
        module_name
        for module_name in sys.modules
        if module_name == 'app' or module_name.startswith('app.')
    ]
    for module_name in stale_modules:
        sys.modules.pop(module_name, None)


class RoomFlowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ['NTE_DATABASE_PATH'] = str(Path(self.temp_dir.name) / 'test.db')
        os.environ['NTE_LOG_DIR'] = str(Path(self.temp_dir.name) / 'logs')
        _purge_app_modules()

        self.app_module = importlib.import_module('app')
        self.dao_module = importlib.import_module('app.dao')
        self.db_module = importlib.import_module('app.db')
        self.app = self.app_module.create_app()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if not self.db_module.db.is_closed():
            self.db_module.db.close()
        self.temp_dir.cleanup()
        _purge_app_modules()

    def _issue_login_and_get_token(self, player_uid: str) -> str:
        with self.db_module.atomic_transaction():
            self.dao_module.issue_mock_login(player_uid, player_uid, '654321')
        payload = self._post('/api/auth/login', {
            'player_uid': player_uid,
            'code': '654321',
        })
        return payload['token']

    def _save_default_build(self, token: str) -> None:
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'item_ids': DEFAULT_ITEMS,
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)

    def _choose_selection(self, state: dict, token: str) -> dict:
        selection = state.get('selection')
        self.assertIsInstance(selection, dict)
        pick_count = int(selection['pick_count'])
        return self._post('/api/game/choose-cards', {
            'card_instance_ids': [card['instance_id'] for card in selection['cards'][:pick_count]],
        }, token=token)

    def _choose_if_needed(self, state: dict, token: str) -> dict:
        if state.get('selection'):
            return self._choose_selection(state, token)
        return state

    def _post_error(
        self,
        path: str,
        payload: dict | None = None,
        *,
        token: str | None = None,
        expected_status: int = 400,
    ) -> dict:
        headers = self._auth_headers(token)
        response = self.client.post(path, json=payload, headers=headers)
        self.assertEqual(response.status_code, expected_status, response.get_data(as_text=True))
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('error', data)
        return data

    def _post(self, path: str, payload: dict | None = None, *, token: str | None = None, expected_status: int = 200) -> dict:
        headers = self._auth_headers(token)
        response = self.client.post(path, json=payload, headers=headers)
        self.assertEqual(response.status_code, expected_status, response.get_data(as_text=True))
        data = response.get_json()
        self.assertIsInstance(data, dict)
        return data

    def _get(self, path: str, *, token: str | None = None, expected_status: int = 200) -> dict:
        headers = self._auth_headers(token)
        response = self.client.get(path, headers=headers)
        self.assertEqual(response.status_code, expected_status, response.get_data(as_text=True))
        data = response.get_json()
        self.assertIsInstance(data, dict)
        return data

    def _auth_headers(self, token: str | None) -> dict[str, str]:
        if token is None:
            return {}
        return {'Authorization': f'Bearer {token}'}

    def _shaft_client_result(self, axis: dict) -> dict:
        duration_ticks = max(0, int(axis.get('duration_ticks') or 0))
        return {
            'enemy': dict(axis.get('enemy') or {}),
            'summary': {
                'duration_ticks': duration_ticks,
                'duration_seconds': duration_ticks / 10,
                'direct_damage': 0,
                'harmony_damage': 0,
                'stagger_damage': 0,
                'total_damage': 0,
                'dps': 0,
            },
            'damage_by_slot': [],
            'damage_by_source': [],
            'details': [],
        }

    def _solo_test_run_context(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch('app.modules.card_game.engine.setup.snapshot_factory.random.shuffle', side_effect=lambda value: None))
        stack.enter_context(self._fixed_battlefield_patch())
        return stack

    def _fixed_battlefield_patch(self, trait_id: str = 'mirror_archive'):
        battlefields = importlib.import_module('app.modules.card_game.engine.setup.battlefields')
        trait = next(
            item for item in battlefields.BATTLEFIELD_TRAITS
            if item['id'] == trait_id
        )
        return patch('app.modules.card_game.engine.setup.snapshot_factory.BATTLEFIELD_TRAITS', [trait])


class SoloRoomFlowTest(RoomFlowTestCase):
    def test_canhong_selection_requires_shaft_test_account(self) -> None:
        public_catalog = self._get('/api/shaft/catalog')
        public_canhong = next(character for character in public_catalog['characters'] if character['name'] == '残红')
        self.assertTrue(public_canhong['selection_disabled'])

        regular_token = self._issue_login_and_get_token('regular-shaft-player')
        regular_catalog = self._get('/api/shaft/catalog', token=regular_token)
        regular_canhong = next(character for character in regular_catalog['characters'] if character['name'] == '残红')
        self.assertTrue(regular_canhong['selection_disabled'])

        test_token = self._issue_login_and_get_token('shaft-whitelisted-player')
        models_module = importlib.import_module('app.models')
        models_module.Player.update(shaft_test_whitelisted=True).where(
            models_module.Player.player_uid == 'shaft-whitelisted-player'
        ).execute()
        test_catalog = self._get('/api/shaft/catalog', token=test_token)
        test_canhong = next(character for character in test_catalog['characters'] if character['name'] == '残红')
        self.assertFalse(test_canhong['selection_disabled'])

        restricted_axis = dict(public_catalog['starter_axis'])
        restricted_axis['team'] = [{
            **restricted_axis['team'][0],
            'character_id': public_canhong['id'],
            'arc_id': '',
            'cartridge_id': '',
        }]
        restricted_axis['character_builds'] = {}
        restricted_axis['steps'] = []
        denied = self._post('/api/shaft/axes', {
            'title': '普通账号残红',
            'axis': restricted_axis,
            'result': self._shaft_client_result(restricted_axis),
        }, token=regular_token, expected_status=400)
        self.assertIn('仅对测试账号开放', denied['error'])

        allowed = self._post('/api/shaft/axes', {
            'title': '测试账号残红',
            'axis': restricted_axis,
            'result': self._shaft_client_result(restricted_axis),
        }, token=test_token)
        self.assertEqual(allowed['team'][0]['character_id'], public_canhong['id'])

        published = self._post(
            f"/api/shaft/axes/{allowed['id']}/publish",
            {},
            token=test_token,
        )
        self.assertEqual(published['visibility'], 'public')

        anonymous_market = self._get('/api/shaft/market')
        regular_market = self._get('/api/shaft/market', token=regular_token)
        test_market = self._get('/api/shaft/market', token=test_token)
        self.assertNotIn('测试账号残红', [axis['title'] for axis in anonymous_market['items']])
        self.assertNotIn('测试账号残红', [axis['title'] for axis in regular_market['items']])
        self.assertIn('测试账号残红', [axis['title'] for axis in test_market['items']])

        for token in (None, regular_token):
            hidden_detail = self.client.get(
                f"/api/shaft/axes/{published['id']}",
                headers=self._auth_headers(token),
            )
            self.assertEqual(hidden_detail.status_code, 400)
            self.assertIn('排轴不存在', hidden_detail.get_json()['error'])
        visible_detail = self._get(f"/api/shaft/axes/{published['id']}", token=test_token)
        self.assertEqual(visible_detail['id'], published['id'])

        denied_favorite = self._post(
            f"/api/shaft/axes/{published['id']}/favorite",
            {},
            token=regular_token,
            expected_status=400,
        )
        self.assertIn('排轴不存在', denied_favorite['error'])
        models_module.Player.update(shaft_test_whitelisted=True).where(
            models_module.Player.player_uid == 'regular-shaft-player'
        ).execute()
        self._post(
            f"/api/shaft/axes/{published['id']}/favorite",
            {},
            token=regular_token,
        )
        visible_favorites = self._get('/api/shaft/me/favorites', token=regular_token)
        self.assertIn('测试账号残红', [axis['title'] for axis in visible_favorites['items']])

        models_module.Player.update(shaft_test_whitelisted=False).where(
            models_module.Player.player_uid == 'regular-shaft-player'
        ).execute()
        hidden_favorites = self._get('/api/shaft/me/favorites', token=regular_token)
        self.assertNotIn('测试账号残红', [axis['title'] for axis in hidden_favorites['items']])
        models_module.Player.update(shaft_test_whitelisted=False).where(
            models_module.Player.player_uid == 'shaft-whitelisted-player'
        ).execute()
        denied_publish = self._post(
            f"/api/shaft/axes/{allowed['id']}/publish",
            {},
            token=test_token,
            expected_status=400,
        )
        self.assertIn('仅对测试账号开放', denied_publish['error'])

    def test_yiloyi_is_released_and_persisted_as_public(self) -> None:
        models_module = importlib.import_module('app.models')
        publication = models_module.ShaftCharacterPublication.get(
            models_module.ShaftCharacterPublication.character_name == '伊洛伊'
        )
        publication.is_published = False
        publication.save(only=[models_module.ShaftCharacterPublication.is_published])

        shaft_service = importlib.import_module('app.modules.shaft.service')
        shaft_service.initialize_shaft_character_publications()
        publication = models_module.ShaftCharacterPublication.get_by_id(publication.id)
        self.assertTrue(publication.is_published)

        public_catalog = self._get('/api/shaft/catalog')
        public_yiloyi = next(
            character for character in public_catalog['characters']
            if character['name'] == '伊洛伊'
        )
        self.assertFalse(public_yiloyi['selection_disabled'])

    def test_portal_and_shaft_plaza_routes_render(self) -> None:
        portal = self.client.get('/')
        self.assertEqual(portal.status_code, 200)
        portal_html = portal.get_data(as_text=True)
        self.assertIn('卡牌桌游', portal_html)
        self.assertIn('空幕', portal_html)
        self.assertIn('预配队', portal_html)
        self.assertIn('排轴计算', portal_html)
        self.assertIn('/card-game', portal_html)
        self.assertIn('/shaft/rotation', portal_html)
        self.assertEqual(portal_html.count('data-login-link'), 1)
        self.assertNotIn('href="/profile">账号设置</a>', portal_html)
        portal_script = (
            Path(__file__).resolve().parents[1] / 'app' / 'static' / 'common' / 'js' / 'portal.js'
        ).read_text(encoding='utf-8')
        self.assertIn(
            "loginLink.textContent = '账号设置';",
            portal_script,
        )

        profile = self.client.get('/profile')
        self.assertEqual(profile.status_code, 200)
        profile_html = profile.get_data(as_text=True)
        self.assertIn('href="/">主界面</a>', profile_html)
        self.assertNotIn('>构筑页</a>', profile_html)
        self.assertIn('id="logout-btn"', profile_html)
        self.assertIn(
            'id="password-input" name="password" type="password" autocomplete="new-password"',
            profile_html,
        )

        card_game = self.client.get('/card-game')
        self.assertEqual(card_game.status_code, 200)
        self.assertNotIn('id="logout-btn"', card_game.get_data(as_text=True))
        self.assertIn('异象对决', card_game.get_data(as_text=True))
        self.assertEqual(self.client.get('/home').status_code, 404)

        legacy_plaza = self.client.get('/shaft/market')
        self.assertEqual(legacy_plaza.status_code, 200)
        plaza_html = legacy_plaza.get_data(as_text=True)
        self.assertIn('data-active-page="plaza"', plaza_html)
        self.assertIn('排轴广场', plaza_html)

    def test_shaft_empty_axis_can_be_saved_backed_up_and_deleted(self) -> None:
        token = self._issue_login_and_get_token('shaft-empty-axis')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []

        simulate_response = self.client.post('/api/shaft/simulate', json=axis)
        self.assertEqual(simulate_response.status_code, 404)

        missing_result = self._post('/api/shaft/axes', {
            'title': '缺少本地结果',
            'axis': axis,
        }, token=token, expected_status=400)
        self.assertIn('缺少本地计算结果', missing_result['error'])

        with patch(
            'app.modules.shaft.service._simulate_axis_with_js',
            side_effect=AssertionError('保存链路不应启动 Node'),
        ):
            saved = self._post('/api/shaft/axes', {
                'title': '空白排轴',
                'visibility': 'private',
                'axis': axis,
                'result': self._shaft_client_result(axis),
            }, token=token)
        self.assertEqual(saved['axis']['steps'], [])
        axis_id = saved['id']

        mine = self._get('/api/shaft/me/axes', token=token)
        self.assertEqual([item['id'] for item in mine['items']], [axis_id])

        backup = self._post(f'/api/shaft/axes/{axis_id}/backup', token=token)
        self.assertNotEqual(backup['id'], axis_id)
        self.assertEqual(backup['title'], '空白排轴-副本')
        self.assertEqual(backup['visibility'], 'private')
        self.assertEqual(backup['axis'], saved['axis'])

        mine_with_backup = self._get('/api/shaft/me/axes', token=token)
        self.assertEqual([item['id'] for item in mine_with_backup['items']], [backup['id'], axis_id])

        backup_delete_response = self.client.delete(
            f'/api/shaft/axes/{backup["id"]}',
            headers=self._auth_headers(token),
        )
        self.assertEqual(backup_delete_response.status_code, 200, backup_delete_response.get_data(as_text=True))

        response = self.client.delete(f'/api/shaft/axes/{axis_id}', headers=self._auth_headers(token))
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json(), {'ok': True})

        mine_after = self._get('/api/shaft/me/axes', token=token)
        self.assertEqual(mine_after['items'], [])

    def test_shaft_private_axis_share_link_is_stable_public_and_revoked_on_delete(self) -> None:
        owner_token = self._issue_login_and_get_token('shaft-share-owner')
        other_token = self._issue_login_and_get_token('shaft-share-other')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []

        saved = self._post('/api/shaft/axes', {
            'title': '可分享私有轴',
            'description': '初始版本',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=owner_token)

        hidden = self.client.get(f'/api/shaft/axes/{saved["id"]}')
        self.assertEqual(hidden.status_code, 400)

        share = self._post(f'/api/shaft/axes/{saved["id"]}/share', token=owner_token)
        self.assertRegex(share['share_token'], r'^[A-Za-z0-9_-]{20,64}$')
        self.assertEqual(
            share['share_path'],
            f'/shaft/rotation?share={share["share_token"]}',
        )
        repeated = self._post(f'/api/shaft/axes/{saved["id"]}/share', token=owner_token)
        self.assertEqual(repeated['share_token'], share['share_token'])

        denied = self._post(
            f'/api/shaft/axes/{saved["id"]}/share',
            token=other_token,
            expected_status=400,
        )
        self.assertIn('没有分享', denied['error'])

        shared = self._get(f'/api/shaft/shared/{share["share_token"]}')
        self.assertTrue(shared['shared'])
        self.assertTrue(shared['read_only'])
        self.assertEqual(shared['title'], '可分享私有轴')
        self.assertEqual(shared['description'], '初始版本')
        self.assertEqual(shared['axis'], saved['axis'])
        authenticated_shared = self._get(
            f'/api/shaft/shared/{share["share_token"]}',
            token=other_token,
        )
        self.assertFalse(authenticated_shared['read_only'])
        if self.app.config['SHAFT_LOGIN_REQUIRED']:
            shared_page = self.client.get(share['share_path'])
            self.assertEqual(shared_page.status_code, 200)
            self.assertNotIn(
                "if (!window.localStorage.getItem('nte_token'))",
                shared_page.get_data(as_text=True),
            )
            invalid_shared_page = self.client.get('/shaft/rotation?share=invalid-token')
            self.assertIn(
                "if (!window.localStorage.getItem('nte_token'))",
                invalid_shared_page.get_data(as_text=True),
            )

        update_response = self.client.put(
            f'/api/shaft/axes/{saved["id"]}',
            json={
                'title': '可分享私有轴',
                'description': '保存后的版本',
                'axis': axis,
                'result': self._shaft_client_result(axis),
            },
            headers=self._auth_headers(owner_token),
        )
        self.assertEqual(update_response.status_code, 200, update_response.get_data(as_text=True))
        updated_shared = self._get(f'/api/shaft/shared/{share["share_token"]}')
        self.assertEqual(updated_shared['description'], '保存后的版本')

        deleted = self.client.delete(
            f'/api/shaft/axes/{saved["id"]}',
            headers=self._auth_headers(owner_token),
        )
        self.assertEqual(deleted.status_code, 200, deleted.get_data(as_text=True))
        expired = self._get(
            f'/api/shaft/shared/{share["share_token"]}',
            expected_status=400,
        )
        self.assertIn('无效或已失效', expired['error'])

    def test_shaft_owner_can_unpublish_snapshot_without_deleting_private_source(self) -> None:
        token = self._issue_login_and_get_token('shaft-unpublish-owner')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []

        saved = self._post('/api/shaft/axes', {
            'title': '待下架快照',
            'description': '用于筛选的下架备注',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token)
        snapshot = self._post(f'/api/shaft/axes/{saved["id"]}/publish', token=token)

        filtered = self._get('/api/shaft/me/axes?q=下架备注', token=token)
        self.assertEqual([item['id'] for item in filtered['items']], [saved['id']])
        self.assertEqual(filtered['items'][0]['published_snapshot_id'], snapshot['id'])

        delete_response = self.client.delete(
            f'/api/shaft/axes/{snapshot["id"]}',
            headers=self._auth_headers(token),
        )
        self.assertEqual(delete_response.status_code, 200, delete_response.get_data(as_text=True))
        self.assertTrue(delete_response.get_json()['ok'])

        market = self._get('/api/shaft/market')
        self.assertNotIn(snapshot['id'], [item['id'] for item in market['items']])
        mine = self._get('/api/shaft/me/axes?sort=new', token=token)
        self.assertIn(saved['id'], [item['id'] for item in mine['items']])
        private_source = self._get(f'/api/shaft/axes/{saved["id"]}', token=token)
        self.assertEqual(private_source['visibility'], 'private')

    def test_shaft_market_can_search_title_or_description(self) -> None:
        token = self._issue_login_and_get_token('shaft-market-search-owner')
        catalog = self._get('/api/shaft/catalog')

        title_axis = dict(catalog['starter_axis'])
        title_source = self._post('/api/shaft/axes', {
            'title': 'market-name-token',
            'description': '普通介绍',
            'axis': title_axis,
            'result': self._shaft_client_result(title_axis),
        }, token=token)
        title_snapshot = self._post(f'/api/shaft/axes/{title_source["id"]}/publish', token=token)

        description_axis = dict(catalog['starter_axis'])
        description_axis['steps'] = []
        description_axis['buff_rules'] = []
        description_source = self._post('/api/shaft/axes', {
            'title': '另一条公开轴',
            'description': '包含 market-note-token 的备注',
            'axis': description_axis,
            'result': self._shaft_client_result(description_axis),
        }, token=token)
        description_snapshot = self._post(
            f'/api/shaft/axes/{description_source["id"]}/publish',
            token=token,
        )

        name_response = self.client.get(
            '/api/shaft/market?q=market-name-token',
            headers={'X-Shaft-Visitor-Key': 'market-title-search'},
        )
        self.assertEqual(name_response.status_code, 200, name_response.get_data(as_text=True))
        name_results = name_response.get_json()
        self.assertEqual([item['id'] for item in name_results['items']], [title_snapshot['id']])

        note_response = self.client.get(
            '/api/shaft/market?q=market-note-token',
            headers={'X-Shaft-Visitor-Key': 'market-description-search'},
        )
        self.assertEqual(note_response.status_code, 200, note_response.get_data(as_text=True))
        note_results = note_response.get_json()
        self.assertEqual(
            [item['id'] for item in note_results['items']],
            [description_snapshot['id']],
        )

    def test_shaft_market_search_is_rate_limited_to_once_per_three_seconds(self) -> None:
        headers = {'X-Shaft-Visitor-Key': 'market-rate-limit-visitor'}

        first = self.client.get('/api/shaft/market?q=第一次', headers=headers)
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))

        limited = self.client.get('/api/shaft/market?q=第二次', headers=headers)
        self.assertEqual(limited.status_code, 429, limited.get_data(as_text=True))
        payload = limited.get_json()
        self.assertEqual(payload['code'], 'shaft_market_search_rate_limited')
        self.assertGreater(payload['retry_after_ms'], 0)
        self.assertLessEqual(payload['retry_after_ms'], 3000)

        unfiltered = self.client.get('/api/shaft/market', headers=headers)
        self.assertEqual(unfiltered.status_code, 200, unfiltered.get_data(as_text=True))

    def test_shaft_private_axis_limit_is_fifty_without_exposing_limit_in_my_axes(self) -> None:
        token = self._issue_login_and_get_token('shaft-axis-limit-owner')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        models_module = importlib.import_module('app.models')
        player = models_module.Player.get(
            models_module.Player.player_uid == 'shaft-axis-limit-owner'
        )
        with self.db_module.atomic_transaction():
            records = [
                models_module.ShaftAxis.create(
                    owner=player,
                    title=f'数量限制轴 {index + 1}',
                    visibility='private',
                )
                for index in range(50)
            ]

        mine = self._get('/api/shaft/me/axes', token=token)
        self.assertEqual(mine['total'], 50)
        self.assertEqual(len(mine['items']), 50)
        self.assertNotIn('limit', mine)
        self.assertNotIn('max', mine)

        create_payload = {
            'title': '第 51 条轴',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }
        denied_create = self._post(
            '/api/shaft/axes',
            create_payload,
            token=token,
            expected_status=400,
        )
        self.assertIn('最多创建 50 个排轴', denied_create['error'])

        denied_backup = self._post(
            f'/api/shaft/axes/{records[0].id}/backup',
            token=token,
            expected_status=400,
        )
        self.assertIn('最多创建 50 个排轴', denied_backup['error'])

        update_response = self.client.put(
            f'/api/shaft/axes/{records[0].id}',
            json={
                **create_payload,
                'title': '已存在轴更新后',
            },
            headers=self._auth_headers(token),
        )
        self.assertEqual(update_response.status_code, 200, update_response.get_data(as_text=True))

        delete_response = self.client.delete(
            f'/api/shaft/axes/{records[1].id}',
            headers=self._auth_headers(token),
        )
        self.assertEqual(delete_response.status_code, 200, delete_response.get_data(as_text=True))
        created_after_delete = self._post('/api/shaft/axes', create_payload, token=token)
        self.assertEqual(created_after_delete['title'], '第 51 条轴')

    def test_public_shaft_snapshot_can_be_saved_as_private_author_suffixed_copy(self) -> None:
        owner_token = self._issue_login_and_get_token('shaft-preview-author')
        recipient_token = self._issue_login_and_get_token('shaft-preview-recipient')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []

        source = self._post('/api/shaft/axes', {
            'title': '广场预览轴',
            'description': '公开轴完整备注',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=owner_token)
        snapshot = self._post(f'/api/shaft/axes/{source["id"]}/publish', token=owner_token)

        preview = self._get(f'/api/shaft/axes/{snapshot["id"]}')
        author_name = preview['owner']['nickname']
        copied = self._post('/api/shaft/axes', {
            'title': f'{preview["title"]} - {author_name}',
            'description': preview['description'],
            'axis': preview['axis'],
            'result': preview['result'],
        }, token=recipient_token)

        self.assertEqual(copied['title'], f'广场预览轴 - {author_name}')
        self.assertEqual(copied['description'], '公开轴完整备注')
        self.assertEqual(copied['visibility'], 'private')
        self.assertEqual(copied['owner']['player_uid'], 'shaft-preview-recipient')
        self.assertEqual(copied['axis'], preview['axis'])
        self.assertEqual(copied['result'], preview['result'])
        self.assertIsNone(copied['source_axis_id'])

        repeated_preview = self._get(f'/api/shaft/axes/{snapshot["id"]}', token=recipient_token)
        self.assertEqual(repeated_preview['local_copy_id'], copied['id'])

        updated = self.client.put(
            f'/api/shaft/axes/{copied["id"]}',
            json={
                'title': f'{preview["title"]} - {author_name}',
                'description': '在线轴更新后的备注',
                'axis': preview['axis'],
                'result': preview['result'],
                'source_axis_id': snapshot['id'],
            },
            headers=self._auth_headers(recipient_token),
        )
        self.assertEqual(updated.status_code, 200, updated.get_data(as_text=True))
        updated_payload = updated.get_json()
        self.assertEqual(updated_payload['id'], copied['id'])
        self.assertEqual(updated_payload['source_axis_id'], snapshot['id'])
        self.assertEqual(updated_payload['description'], '在线轴更新后的备注')

        linked_preview = self._get(f'/api/shaft/axes/{snapshot["id"]}', token=recipient_token)
        self.assertEqual(linked_preview['local_copy_id'], copied['id'])
        mine = self._get('/api/shaft/me/axes', token=recipient_token)
        self.assertEqual(mine['total'], 1)

    def test_deleting_private_source_keeps_uploaded_snapshot(self) -> None:
        token = self._issue_login_and_get_token('shaft-delete-source-owner')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []
        saved = self._post('/api/shaft/axes', {
            'title': '删除私有源',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token)
        snapshot = self._post(f'/api/shaft/axes/{saved["id"]}/publish', token=token)

        delete_response = self.client.delete(
            f'/api/shaft/axes/{saved["id"]}',
            headers=self._auth_headers(token),
        )
        self.assertEqual(delete_response.status_code, 200, delete_response.get_data(as_text=True))
        market = self._get('/api/shaft/market')
        market_snapshot = next(item for item in market['items'] if item['id'] == snapshot['id'])
        self.assertIsNone(market_snapshot['source_axis_id'])

    def test_shaft_publish_is_snapshot_and_same_name_save_can_overwrite(self) -> None:
        token = self._issue_login_and_get_token('shaft-snapshot-axis')
        catalog = self._get('/api/shaft/catalog')
        axis = dict(catalog['starter_axis'])
        axis['steps'] = []
        axis['buff_rules'] = []
        axis['options'] = {**axis.get('options', {}), 'loop_enabled': True}

        saved = self._post('/api/shaft/axes', {
            'title': '快照来源',
            'description': '上传时版本',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token)
        snapshot = self._post(f'/api/shaft/axes/{saved["id"]}/publish', token=token)
        self.assertNotEqual(snapshot['id'], saved['id'])
        self.assertEqual(snapshot['visibility'], 'public')
        self.assertEqual(snapshot['like_count'], 0)
        self.assertEqual(snapshot['dislike_count'], 0)

        voter_token = self._issue_login_and_get_token('shaft-reaction-voter')
        liked = self._post(f'/api/shaft/axes/{snapshot["id"]}/like', token=voter_token)
        self.assertTrue(liked['liked'])
        self.assertFalse(liked['disliked'])
        self.assertEqual(liked['like_count'], 1)
        self.assertEqual(liked['dislike_count'], 0)

        disliked = self._post(f'/api/shaft/axes/{snapshot["id"]}/dislike', token=voter_token)
        self.assertFalse(disliked['liked'])
        self.assertTrue(disliked['disliked'])
        self.assertEqual(disliked['like_count'], 0)
        self.assertEqual(disliked['dislike_count'], 1)

        undislike_response = self.client.delete(
            f'/api/shaft/axes/{snapshot["id"]}/dislike',
            headers=self._auth_headers(voter_token),
        )
        self.assertEqual(undislike_response.status_code, 200, undislike_response.get_data(as_text=True))
        undisliked = undislike_response.get_json()
        self.assertFalse(undisliked['liked'])
        self.assertFalse(undisliked['disliked'])
        self.assertEqual(undisliked['dislike_count'], 0)

        update_response = self.client.put(
            f'/api/shaft/axes/{saved["id"]}',
            json={
                'title': '本地修改版',
                'description': '保存后的版本',
                'axis': axis,
                'result': self._shaft_client_result(axis),
            },
            headers=self._auth_headers(token),
        )
        self.assertEqual(update_response.status_code, 200, update_response.get_data(as_text=True))

        market = self._get('/api/shaft/market')
        market_snapshot = next(item for item in market['items'] if item['id'] == snapshot['id'])
        self.assertEqual(market_snapshot['title'], '快照来源')
        self.assertEqual(market_snapshot['description'], '上传时版本')
        self.assertTrue(market_snapshot['loop_enabled'])
        self.assertEqual(market_snapshot['duration_seconds'], market_snapshot['duration_ticks'] / 10)
        self.assertEqual(
            market_snapshot['harmony_damage'],
            int(market_snapshot['summary'].get('harmony_damage') or 0),
        )

        mine = self._get('/api/shaft/me/axes?sort=new', token=token)
        self.assertEqual([item['id'] for item in mine['items']], [saved['id']])
        self.assertEqual(mine['items'][0]['title'], '本地修改版')
        self.assertEqual(mine['items'][0]['published_snapshot_id'], snapshot['id'])

        refreshed_snapshot = self._post(f'/api/shaft/axes/{saved["id"]}/publish', token=token)
        self.assertEqual(refreshed_snapshot['id'], snapshot['id'])
        self.assertEqual(refreshed_snapshot['title'], '本地修改版')
        refreshed_market = self._get('/api/shaft/market')
        refreshed_market_snapshot = next(
            item for item in refreshed_market['items'] if item['id'] == snapshot['id']
        )
        self.assertEqual(refreshed_market_snapshot['description'], '保存后的版本')

        conflict = self._post('/api/shaft/axes', {
            'title': '本地修改版',
            'description': '将覆盖同名轴',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token, expected_status=409)
        self.assertEqual(conflict['code'], 'axis_name_conflict')
        self.assertEqual(conflict['existing_axis_id'], saved['id'])

        overwritten = self._post('/api/shaft/axes', {
            'title': '本地修改版',
            'description': '将覆盖同名轴',
            'conflict_action': 'overwrite',
            'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token)
        self.assertEqual(overwritten['id'], saved['id'])
        self.assertEqual(overwritten['description'], '将覆盖同名轴')

        character_id = axis['team'][0]['character_id']
        filtered = self._get(
            f'/api/shaft/me/axes?sort=dps&character_id={character_id}',
            token=token,
        )
        self.assertEqual([item['id'] for item in filtered['items']], [saved['id']])
        no_match = self._get(
            '/api/shaft/me/axes?character_id=missing-character',
            token=token,
        )
        self.assertEqual(no_match['items'], [])

        models_module = importlib.import_module('app.models')
        models_module.ShaftAxis.update(source_version='异环云配队 V0.2.7').execute()
        self.app_module.create_app()
        preserved_mine = self._get('/api/shaft/me/axes', token=token)
        preserved_market = self._get('/api/shaft/market')
        self.assertEqual(preserved_mine['items'][0]['source_version'], '异环云配队 V0.2.7')
        self.assertEqual(preserved_market['items'][0]['source_version'], '异环云配队 V0.2.7')

        resave_response = self.client.put(
            f'/api/shaft/axes/{saved["id"]}',
            json={
                'title': '本地修改版',
                'description': '按新版本再次保存',
                'axis': axis,
                'result': self._shaft_client_result(axis),
            },
            headers=self._auth_headers(token),
        )
        self.assertEqual(resave_response.status_code, 200, resave_response.get_data(as_text=True))
        resaved_mine = self._get('/api/shaft/me/axes', token=token)
        unchanged_market = self._get('/api/shaft/market')
        self.assertEqual(resaved_mine['items'][0]['source_version'], '异环云配队 1.0.2')
        self.assertEqual(unchanged_market['items'][0]['source_version'], '异环云配队 V0.2.7')

    def test_balance_analytics_requires_login(self) -> None:
        payload = self._get('/api/analytics/balance', expected_status=401)

        self.assertIn('error', payload)

    def test_balance_analytics_reports_missing_data(self) -> None:
        token = self._issue_login_and_get_token('analytics-missing-data')
        analytics_module = importlib.import_module('app.modules.card_game.engine.application.analytics_service')
        missing_path = Path(self.temp_dir.name) / 'missing-analytics.json'

        with patch.object(analytics_module, 'ANALYTICS_DATA_PATH', missing_path):
            payload = self._get('/api/analytics/balance', token=token)

        self.assertFalse(payload['available'])
        self.assertEqual(payload['data_path'], str(missing_path))
        self.assertIn('duel_balance_eval.py', payload['message'])

    def test_balance_analytics_returns_latest_dashboard_data(self) -> None:
        token = self._issue_login_and_get_token('analytics-latest-data')
        analytics_module = importlib.import_module('app.modules.card_game.engine.application.analytics_service')
        data_path = Path(self.temp_dir.name) / 'duel_analytics_latest.json'
        data_path.write_text(
            (
                '{"focus":"triad","samples":8,'
                '"decks":[{"id":"genesis_bloom","label":"创生","games":32}],'
                '"cards":[{"id":"genesis_urban_energy","name":"都市活力"}],'
                '"espers":[{"id":"nanali","name":"娜娜莉"}]}'
            ),
            encoding='utf-8',
        )

        with patch.object(analytics_module, 'ANALYTICS_DATA_PATH', data_path):
            payload = self._get('/api/analytics/balance', token=token)

        self.assertTrue(payload['available'])
        self.assertEqual(payload['data_path'], str(data_path))
        self.assertEqual(payload['data']['focus'], 'triad')
        self.assertEqual(payload['data']['samples'], 8)
        self.assertEqual(payload['data']['decks'][0]['label'], '创生')

    def test_legacy_starter_fields_are_saved_as_plain_deck(self) -> None:
        token = self._issue_login_and_get_token('legacy-starter-build')

        self._post('/api/build/save', {
            'character_id': 'nanali',
            'starter_item_ids': DEFAULT_ITEMS[:4],
            'reserve_item_ids': DEFAULT_ITEMS[4:],
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)

        catalog = self._get('/api/catalog', token=token)
        saved_build = catalog['saved_build']
        self.assertEqual(catalog['min_build_size'], 10)
        self.assertEqual(catalog['build_size'], 20)
        self.assertFalse(catalog['fixed_opening_hand_enabled'])
        self.assertEqual(saved_build['item_ids'], DEFAULT_ITEMS)
        self.assertEqual(saved_build['starter_item_ids'], [])
        self.assertEqual(saved_build['reserve_item_ids'], DEFAULT_ITEMS)
        self.assertEqual(saved_build['esper_card_ids'], DEFAULT_ESPERS)

    def test_prebuilt_preview_decks_use_standard_card_order(self) -> None:
        token = self._issue_login_and_get_token('prebuilt-preview-order')
        build_service = importlib.import_module('app.modules.card_game.engine.application.build_service')

        catalog = self._get('/api/catalog', token=token)

        self.assertEqual([deck['id'] for deck in catalog['decks']], ['genesis_bloom', 'delay_lock', 'murk_burn'])
        for deck in catalog['decks']:
            self.assertEqual(deck['card_ids'], build_service.sort_card_ids(deck['card_ids']), deck['id'])
            self.assertEqual(deck['esper_card_ids'], build_service.sort_card_ids(deck['esper_card_ids']), deck['id'])
            item_costs = [build_service.card_by_id()[card_id]['cost'] for card_id in deck['card_ids']]
            self.assertEqual(item_costs, sorted(item_costs), deck['id'])
        genesis = next(deck for deck in catalog['decks'] if deck['id'] == 'genesis_bloom')
        self.assertGreater(
            genesis['card_ids'].index('genesis_eborn_cake'),
            genesis['card_ids'].index('genesis_chip_washer'),
        )
        item_ids = {item['id'] for item in catalog['items']}
        character_ids = {character['id'] for character in catalog['characters']}
        self.assertIn('darkstar_ringstone', item_ids)
        self.assertIn('darkstar_nature_pixel', item_ids)
        self.assertIn('surplus_fons', item_ids)
        self.assertIn('discord_tomato_bucket', item_ids)
        self.assertIn('fatiya', character_ids)
        self.assertIn('xiaozhi', character_ids)
        self.assertIn('dafutier', character_ids)

    def test_legacy_starter_fields_do_not_fix_opening_hand(self) -> None:
        token = self._issue_login_and_get_token('legacy-starter-random')
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'starter_item_ids': DEFAULT_ITEMS[:4],
            'reserve_item_ids': DEFAULT_ITEMS[4:],
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        def reverse_shuffle(value: list) -> None:
            value.reverse()

        with self._fixed_battlefield_patch():
            with patch('app.modules.card_game.engine.setup.snapshot_factory.random.shuffle', side_effect=reverse_shuffle):
                state = self._post('/api/room/start', token=token)

        hand_ids = [card['definition_id'] for card in state['player']['hand']]
        self.assertEqual(len(hand_ids), 4)
        self.assertEqual(hand_ids, list(reversed(DEFAULT_ITEMS[-4:])))
        self.assertNotEqual(hand_ids, DEFAULT_ITEMS[:4])

    def test_ten_card_build_starts_with_random_opening_hand(self) -> None:
        token = self._issue_login_and_get_token('ten-card-build')
        ten_card_items = DEFAULT_ITEMS[:10]
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'item_ids': ten_card_items,
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)

        self.assertEqual(len(state['player']['hand']), 4)
        self.assertEqual(state['player']['deck_count'], 6)

    def test_create_solo_room_and_start_run(self) -> None:
        token = self._issue_login_and_get_token('solo-room-start')
        self._save_default_build(token)

        room_payload = self._post('/api/room/create', {'mode': 'solo'}, token=token)
        room_code = room_payload['room_code']
        self.assertEqual(room_payload['mode'], 'solo')
        self.assertEqual(room_payload['status'], 'ready')
        self.assertEqual(len(room_payload['members']), 1)
        self.assertTrue(room_payload['members'][0]['is_host'])
        self.assertTrue(room_payload['members'][0]['is_ready'])

        with self._solo_test_run_context():
            run_payload = self._post('/api/room/start', token=token)

        self.assertEqual(run_payload['status'], 'playing')
        self.assertEqual(run_payload['game_id'], 'anomaly_snap_duel')
        self.assertEqual(run_payload['room']['mode'], 'solo')
        self.assertEqual(run_payload['room']['room_code'], room_code)
        self.assertEqual(run_payload['turn'], 1)
        self.assertEqual(run_payload['phase'], 'planning')
        self.assertIsNone(run_payload['selection'])
        self.assertEqual(len(run_payload['player']['hand']), 4)
        self.assertEqual(run_payload['player']['deck_count'], 16)
        self.assertEqual(run_payload['turn_energy'], 1)
        self.assertEqual(run_payload['energy_remaining'], 1)
        self.assertFalse(run_payload['can_undo_turn'])
        self.assertEqual(run_payload['player_seat'], 'a')
        self.assertTrue(run_payload['opponent']['is_ai'])
        self.assertEqual(len(run_payload['locations']), 1)
        self.assertEqual(run_payload['locations'][0]['id'], 'main_battlefield')
        constants = importlib.import_module('app.modules.card_game.content.common.constants')
        self.assertEqual(constants.LOCATION_CARD_LIMIT, 7)
        self.assertEqual(run_payload['locations'][0]['capacity'], constants.LOCATION_CARD_LIMIT)
        self.assertEqual(run_payload['current_actor_uid'], 'solo-room-start')

        room_state = self._get('/api/room/state', token=token)
        self.assertEqual(room_state['room_code'], room_code)
        self.assertEqual(room_state['status'], 'playing')
        self.assertEqual(room_state['run_status'], 'playing')

    def test_hollow_theater_grants_first_turn_extra_normal_draw(self) -> None:
        token = self._issue_login_and_get_token('hollow-theater-draw')
        self._save_default_build(token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)
        battlefields = importlib.import_module('app.modules.card_game.engine.setup.battlefields')
        hollow_theater = next(
            trait for trait in battlefields.BATTLEFIELD_TRAITS
            if trait['id'] == 'hollow_theater'
        )

        with self._solo_test_run_context():
            with patch('app.modules.card_game.engine.setup.snapshot_factory.BATTLEFIELD_TRAITS', [hollow_theater]):
                state = self._post('/api/room/start', token=token)

        self.assertEqual(state['locations'][0]['name'], '战场：呼啸环线')
        self.assertEqual(state['locations'][0]['description'], '首个回合双方额外执行 1 次通常抽卡。')
        self.assertEqual(len(state['player']['hand']), 5)
        self.assertEqual(state['player']['deck_count'], 15)
        self.assertEqual(state['opponent']['hand_count'], 5)
        self.assertEqual(state['opponent']['deck_count'], 15)
        self.assertTrue(any('呼啸环线' in line and '额外执行 1 次通常抽卡' in line for line in state['log']))

    def test_chip_washer_requires_ally_item_target_before_deploying(self) -> None:
        token = self._issue_login_and_get_token('solo-chip-washer-target')
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'item_ids': DEFAULT_ITEMS,
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        def chip_washer_first(value: list) -> None:
            value.sort(key=lambda card: 0 if card.get('definition_id') == 'genesis_chip_washer' else 1)

        with self._fixed_battlefield_patch():
            with patch('app.modules.card_game.engine.setup.snapshot_factory.random.shuffle', side_effect=chip_washer_first):
                self._post('/api/room/start', token=token)
                state = self._post('/api/game/end-turn', token=token)

        washer = next(
            card for card in state['player']['hand']
            if card.get('definition_id') == 'genesis_chip_washer'
        )
        error = self._post_error('/api/game/end-turn', {
            'planning_actions': [{
                'kind': 'play_card',
                'card_instance_id': washer['instance_id'],
                'location_id': state['locations'][0]['id'],
            }],
        }, token=token)

        self.assertIn('需要可选择的己方道具', error['error'])
        state_after = self._get('/api/game/state', token=token)
        self.assertTrue(any(card['instance_id'] == washer['instance_id'] for card in state_after['player']['hand']))
        self.assertEqual(state_after['energy_remaining'], 2)
        self.assertFalse(state_after['locations'][0]['slots']['player'])
        self.assertIsNone(state_after.get('pending_target'))

    def test_declaration_previews_are_prefetched_only_during_planning(self) -> None:
        token = self._issue_login_and_get_token('solo-declaration-preview')
        self._save_default_build(token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)

        location_id = state['locations'][0]['id']
        source = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_urban_energy')
        preview_payload = self._get('/api/game/declaration-previews', token=token)
        preview_key = f"{source['instance_id']}:{location_id}"

        self.assertEqual(preview_payload['phase'], 'planning')
        self.assertIn(preview_key, preview_payload['previews'])
        preview = preview_payload['previews'][preview_key]
        self.assertEqual(preview['kind'], 'declaration')
        self.assertTrue(preview['cards'])

        selection_payload = self._post('/api/game/declaration-preview', {
            'card_instance_id': source['instance_id'],
            'location_id': location_id,
        }, token=token)
        self.assertIsNotNone(selection_payload['selection'])
        self.assertEqual(selection_payload['selection']['source_instance_id'], source['instance_id'])

        blocked_payload = self._get('/api/game/declaration-previews', token=token)
        self.assertEqual(blocked_payload['phase'], 'planning')
        self.assertIsInstance(blocked_payload['previews'], dict)

        choice = selection_payload['selection']['cards'][0]
        ended_state = self._post('/api/game/end-turn', {
            'planning_actions': [{
                'kind': 'play_card',
                'card_instance_id': source['instance_id'],
                'location_id': location_id,
            }],
            'declaration_choices': [{
                'source_instance_id': source['instance_id'],
                'location_id': location_id,
                'card_instance_ids': [choice['instance_id']],
                'card_names': [choice['name']],
            }],
        }, token=token)
        self.assertEqual(ended_state['turn'], 2)
        self.assertTrue(any(f"{source['name']} 宣言了 {choice['name']}" in line for line in ended_state['log']))

    def test_target_previews_use_board_declaration_predicate(self) -> None:
        token = self._issue_login_and_get_token('solo-target-preview-refresh-only')
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'item_ids': [
                'genesis_refresh_charge',
                'genesis_marble_soda',
                'genesis_eborn_cake',
                'genesis_chip_washer',
                'genesis_breakfast_bag',
                'genesis_breakfast_bag',
                'genesis_urban_energy',
                'genesis_urban_energy',
                'delay_commute_bag',
                'genesis_nest_shard',
            ],
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)
            location_id = state['locations'][0]['id']
            refresh = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_refresh_charge')
            state = self._post('/api/game/end-turn', {
                'planning_actions': [{
                    'kind': 'play_card',
                    'card_instance_id': refresh['instance_id'],
                    'location_id': location_id,
                }],
            }, token=token)

        marble = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_marble_soda')
        refresh_on_board = next(card for card in state['locations'][0]['slots']['player'] if card['definition_id'] == 'genesis_refresh_charge')
        preview_payload = self._get('/api/game/declaration-previews', token=token)
        preview_key = f"{marble['instance_id']}:{location_id}"

        self.assertEqual(preview_payload['phase'], 'planning')
        self.assertIn('target_previews', preview_payload)
        self.assertNotIn(preview_key, preview_payload['target_previews'])

        error = self._post_error('/api/game/end-turn', {
            'planning_actions': [{
                'kind': 'play_card',
                'card_instance_id': marble['instance_id'],
                'location_id': location_id,
                'selected_target_instance_id': refresh_on_board['instance_id'],
            }],
        }, token=token)
        self.assertIn('请选择合法的战场目标', error['error'])
        state_after_error = self._get('/api/game/state', token=token)
        self.assertTrue(any(card['instance_id'] == marble['instance_id'] for card in state_after_error['player']['hand']))
        self.assertEqual(
            [card['definition_id'] for card in state_after_error['locations'][0]['slots']['player']],
            ['genesis_refresh_charge'],
        )

    def test_target_previews_include_only_legal_board_targets(self) -> None:
        token = self._issue_login_and_get_token('solo-target-preview-legal-consumable')
        self._post('/api/build/save', {
            'character_id': 'nanali',
            'item_ids': [
                'genesis_urban_energy',
                'genesis_marble_soda',
                'genesis_refresh_charge',
                'genesis_eborn_cake',
                'genesis_breakfast_bag',
                'genesis_breakfast_bag',
                'genesis_refresh_charge',
                'genesis_chip_washer',
                'delay_commute_bag',
                'genesis_nest_shard',
            ],
            'esper_card_ids': DEFAULT_ESPERS,
        }, token=token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)
            location_id = state['locations'][0]['id']
            urban = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_urban_energy')
            selection_payload = self._post('/api/game/declaration-preview', {
                'card_instance_id': urban['instance_id'],
                'location_id': location_id,
            }, token=token)
            choice = selection_payload['selection']['cards'][0]
            state = self._post('/api/game/end-turn', {
                'planning_actions': [{
                    'kind': 'play_card',
                    'card_instance_id': urban['instance_id'],
                    'location_id': location_id,
                }],
                'declaration_choices': [{
                    'source_instance_id': urban['instance_id'],
                    'location_id': location_id,
                    'card_instance_ids': [choice['instance_id']],
                    'card_names': [choice['name']],
                }],
            }, token=token)

        marble = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_marble_soda')
        urban_on_board = next(card for card in state['locations'][0]['slots']['player'] if card['definition_id'] == 'genesis_urban_energy')
        preview_payload = self._get('/api/game/declaration-previews', token=token)
        preview_key = f"{marble['instance_id']}:{location_id}"
        preview = preview_payload['target_previews'][preview_key]

        self.assertEqual(preview['prompt'], '选择 1 张己方表侧耗材道具。')
        self.assertEqual(preview['target_instance_ids'], [urban_on_board['instance_id']])

    def test_solo_room_can_finish_run_and_reset(self) -> None:
        token = self._issue_login_and_get_token('solo-room-victory')
        self._save_default_build(token)
        room_payload = self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)
            self.assertEqual(state['status'], 'playing')
            self.assertEqual(state['phase'], 'planning')
            self.assertEqual(len(state['player']['hand']), 4)
            first_card = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_refresh_charge')
            first_location = state['locations'][0]
            state = self._post('/api/game/end-turn', {
                'planning_actions': [{
                    'kind': 'play_card',
                    'card_instance_id': first_card['instance_id'],
                    'location_id': first_location['id'],
                }],
            }, token=token)
            self.assertEqual(state['turn'], 2)
            self.assertEqual(state['phase'], 'planning')
            self.assertIsNone(state['selection'])
            self.assertTrue(any('标记' in line or '战力' in line for line in state['log']))
            self.assertTrue(any(action['kind'] == 'reveal_phase_begin' for action in state['action_queue']))
            self.assertTrue(any(action['kind'] == 'draw_card' for action in state['action_queue']))
            refreshed_state = self._get('/api/game/state', token=token)
            self.assertEqual(refreshed_state['action_queue'], [])
            self.assertEqual(refreshed_state['banner_queue'], [])

            for _ in range(5):
                state = self._post('/api/game/end-turn', token=token)
                state = self._choose_if_needed(state, token)

        self.assertIn(state['status'], {'victory', 'defeat', 'draw'})
        self.assertIn(state['phase'], {'victory', 'defeat', 'draw'})

        game_state = self._get('/api/game/state', token=token)
        self.assertEqual(game_state['status'], state['status'])
        self.assertEqual(game_state['room']['room_code'], room_payload['room_code'])

        room_state = self._get('/api/room/state', token=token)
        self.assertEqual(room_state['status'], state['status'])
        self.assertEqual(room_state['run_status'], state['status'])

        reset_payload = self._post('/api/game/reset', token=token)
        self.assertTrue(reset_payload['ok'])

        reset_room_state = self._get('/api/room/state', token=token)
        self.assertEqual(reset_room_state['status'], 'ready')
        self.assertIsNone(reset_room_state['run_status'])

    def test_deployment_write_compat_routes_are_removed(self) -> None:
        token = self._issue_login_and_get_token('solo-room-removed-routes')
        self._save_default_build(token)
        self._post('/api/room/create', {'mode': 'solo'}, token=token)

        with self._solo_test_run_context():
            state = self._post('/api/room/start', token=token)
            first_card = next(card for card in state['player']['hand'] if card['definition_id'] == 'genesis_refresh_charge')
            first_location = state['locations'][0]

            for path, payload in [
                ('/api/game/play-card', {'card_instance_id': first_card['instance_id'], 'location_id': first_location['id']}),
                ('/api/game/play-esper', {'card_instance_id': 'missing', 'location_id': first_location['id'], 'material_instance_ids': []}),
                ('/api/game/return-card', {'card_instance_id': first_card['instance_id']}),
                ('/api/game/move-card', {'card_instance_id': first_card['instance_id'], 'location_id': first_location['id']}),
                ('/api/game/choose-target', {'target_instance_id': first_card['instance_id']}),
                ('/api/game/cancel-target', {}),
                ('/api/game/undo-turn', {}),
            ]:
                response = self.client.post(path, headers=self._auth_headers(token), json=payload)
                self.assertEqual(response.status_code, 404, path)


class _RuleModules:
    def __init__(self) -> None:
        self.runtime = importlib.import_module('app.modules.card_game.engine.flow.turn_flow')
        self.board = importlib.import_module('app.modules.card_game.engine.rules.board_state')
        self.declarations = importlib.import_module('app.modules.card_game.engine.rules.declarations')
        self.harmony = importlib.import_module('app.modules.card_game.engine.rules.harmony')
        self.materials = importlib.import_module('app.modules.card_game.engine.rules.materials')
        self.reveal = importlib.import_module('app.modules.card_game.engine.flow.reveal_flow')
        self.build = importlib.import_module('app.modules.card_game.engine.application.build_service')
        self.run_state = importlib.import_module('app.modules.card_game.engine.application.run_state')
        self.factory = importlib.import_module('app.modules.card_game.engine.setup.snapshot_factory')
        self.projection = importlib.import_module('app.modules.card_game.engine.projection.public_state')
        self.errors = importlib.import_module('app.errors')

    def __getattr__(self, name: str):
        mapping = {
            '_card_by_id': self.build.card_by_id,
            '_card_instance': self.factory.card_instance,
            '_declaration_selection_preview': self.declarations.declaration_selection_preview,
            '_prepare_declaration_selection': self.declarations.prepare_declaration_selection,
            '_prepare_declaration_target': self.declarations.prepare_declaration_target,
            '_public_pending_target': self.run_state.public_pending_target,
            '_public_location': self.run_state.public_location,
            '_reveal_card': self.reveal.reveal_card,
            '_resolve_pending_material_consumption': self.reveal.resolve_pending_material_consumption,
            '_recompute_scores': self.board.recompute_scores,
            '_location_occupied_card_count': self.board.location_occupied_card_count,
            '_open_locations': self.board.open_locations,
            '_location_has_room_after_materials': self.materials.location_has_room_after_materials,
            '_material_cards_for_esper': self.materials.material_cards_for_esper,
            '_decay_harmony_layers_at_turn_start': self.harmony.decay_harmony_layers_at_turn_start,
            '_resolve_harmony_end_of_turn': self.harmony.resolve_harmony_end_of_turn,
            'SidePerspective': self.projection.SidePerspective,
            'RuleValidationError': self.errors.RuleValidationError,
        }
        if name in mapping:
            return mapping[name]
        return getattr(self.runtime, name)


class DuelRuleTimingTest(RoomFlowTestCase):
    def _rules(self) -> _RuleModules:
        return _RuleModules()

    def _card_instance(self, definition_id: str, suffix: str, *, revealed: bool = True, turn: int = 1) -> dict:
        rules = self._rules()
        definition = rules._card_by_id()[definition_id]
        card = rules._card_instance(definition, rules.SIDE_A, suffix)
        card['revealed'] = revealed
        card['played_turn'] = turn
        card['location_id'] = 'mirror_archive'
        return card

    def _snapshot_with_cards(self, cards: list[dict]) -> dict:
        rules = self._rules()
        return {
            'schema_version': rules.SCHEMA_VERSION,
            'game_id': rules.GAME_ID,
            'mode': 'solo',
            'scenario': 'standard',
            'status': 'playing',
            'phase': 'revealing',
            'turn': 2,
            'max_turns': 6,
            'locations': [{
                'id': 'mirror_archive',
                'name': '本初环线',
                'short_name': '本初',
                'reveal_turn': 1,
                'revealed': True,
                'art': '',
                'description': '',
                'effect': '',
                'cards': {rules.SIDE_A: cards, rules.SIDE_B: []},
                'power': {rules.SIDE_A: 0, rules.SIDE_B: 0},
                'winner_side': None,
            }],
            'sides': {
                rules.SIDE_A: {
                    'side': rules.SIDE_A,
                    'uid': 'rules-a',
                    'nickname': 'A',
                    'is_ai': False,
                    'deck': [],
                    'hand': [],
                    'esper_standby': [],
                    'discard': [],
                    'selection': None,
                    'pending_target': None,
                    'combo': {},
                    'energy_used': 0,
                    'ended_turn': False,
                },
                rules.SIDE_B: {
                    'side': rules.SIDE_B,
                    'uid': 'rules-b',
                    'nickname': 'B',
                    'is_ai': True,
                    'deck': [],
                    'hand': [],
                    'esper_standby': [],
                    'discard': [],
                    'selection': None,
                    'pending_target': None,
                    'combo': {},
                    'energy_used': 0,
                    'ended_turn': False,
                },
            },
            'winner_side': None,
            'log': [],
            'action_queue': [],
            'banner_queue': [],
        }

    def test_declaration_config_can_select_board_consumables(self) -> None:
        rules = self._rules()
        source = self._card_instance('genesis_marble_soda', 'source', revealed=False, turn=2)
        source['staged'] = True
        consumable = self._card_instance('genesis_urban_energy', 'consumable')
        material = self._card_instance('delay_first_wish', 'material')
        snapshot = self._snapshot_with_cards([source, consumable, material])
        snapshot['phase'] = 'planning'

        prepared = rules._prepare_declaration_target(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertTrue(prepared)
        pending = rules._public_pending_target(snapshot['sides'][rules.SIDE_A], snapshot)
        self.assertEqual(pending['target_instance_ids'], [consumable['instance_id']])

    def test_tomato_counts_as_murk_material_for_adler(self) -> None:
        rules = self._rules()
        tomato = self._card_instance('discord_tomato', 'tomato', turn=1)
        tomato_100 = self._card_instance('discord_tomato_100', 'tomato-100', turn=1)
        adler = self._card_instance('adler', 'adler', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([tomato, tomato_100])
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 2
        location = snapshot['locations'][0]

        selected = rules._material_cards_for_esper(snapshot, rules.SIDE_A, location, adler)

        self.assertEqual([card['definition_id'] for card in selected], ['discord_tomato', 'discord_tomato_100'])

    def test_draw_skips_when_hand_limit_is_reached(self) -> None:
        rules = self._rules()
        snapshot = self._snapshot_with_cards([])
        snapshot['sides'][rules.SIDE_A]['hand'] = [
            self._card_instance('genesis_refresh_charge', f'hand-{index}', revealed=False)
            for index in range(rules.MAX_HAND_SIZE)
        ]
        deck_card = self._card_instance('genesis_urban_energy', 'deck-top', revealed=False)
        snapshot['sides'][rules.SIDE_A]['deck'] = [deck_card]

        drawn = rules._draw_cards(snapshot, rules.SIDE_A, 1, reason='回合补牌')

        self.assertEqual(drawn, [])
        self.assertEqual(len(snapshot['sides'][rules.SIDE_A]['hand']), rules.MAX_HAND_SIZE)
        self.assertEqual(snapshot['sides'][rules.SIDE_A]['deck'], [deck_card])
        self.assertTrue(any('手牌已达上限 8' in line for line in snapshot['log']))

    def test_declaration_preview_is_read_only_and_excludes_source_in_hand(self) -> None:
        rules = self._rules()
        source = self._card_instance('genesis_eborn_cake', 'source', revealed=False, turn=2)
        consumable = self._card_instance('genesis_urban_energy', 'consumable')
        material = self._card_instance('delay_first_wish', 'material')
        snapshot = self._snapshot_with_cards([])
        snapshot['phase'] = 'planning'
        snapshot['sides'][rules.SIDE_A]['hand'] = [source]
        snapshot['sides'][rules.SIDE_A]['deck'] = [material]
        snapshot['sides'][rules.SIDE_A]['discard'] = [consumable]

        preview = rules._declaration_selection_preview(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertIsNotNone(preview)
        self.assertIsNone(snapshot['sides'][rules.SIDE_A]['selection'])
        self.assertEqual([card['instance_id'] for card in preview['cards']], [consumable['instance_id']])

    def test_commute_bag_end_turn_boosts_this_turn_revealed_cards(self) -> None:
        rules = self._rules()
        source = self._card_instance('delay_commute_bag', 'source', revealed=True, turn=2)
        target = self._card_instance('genesis_urban_energy', 'target', revealed=True, turn=2)
        old_card = self._card_instance('genesis_marble_soda', 'old-card', revealed=True, turn=1)
        snapshot = self._snapshot_with_cards([source, target, old_card])

        rules._dispatch_revealed_card_turn_end(snapshot)

        self.assertEqual(source['computed_power'], 4)
        self.assertEqual(target['computed_power'], 3)
        self.assertEqual(old_card['computed_power'], 3)
        self.assertTrue(any('通勤公文包 在回合结束时使本回合揭示的' in line for line in snapshot['log']))

    def test_declaration_config_selects_discard_drink_or_consumable_cards(self) -> None:
        rules = self._rules()
        source = self._card_instance('genesis_eborn_cake', 'source', revealed=False, turn=2)
        source['staged'] = True
        urban_deck = self._card_instance('genesis_urban_energy', 'urban-deck')
        cake_deck = self._card_instance('genesis_eborn_cake', 'cake-deck')
        urban_discard = self._card_instance('genesis_urban_energy', 'urban-discard')
        soda_discard = self._card_instance('genesis_marble_soda', 'soda-discard')
        snapshot = self._snapshot_with_cards([source])
        snapshot['phase'] = 'planning'
        snapshot['sides'][rules.SIDE_A]['deck'] = [urban_deck, cake_deck]
        snapshot['sides'][rules.SIDE_A]['discard'] = [urban_discard, soda_discard]

        prepared = rules._prepare_declaration_selection(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertTrue(prepared)
        selection = snapshot['sides'][rules.SIDE_A]['selection']
        self.assertEqual(
            [card['definition_id'] for card in selection['cards']],
            ['genesis_urban_energy'],
        )
        self.assertEqual(
            len({card['definition_id'] for card in selection['cards']}),
            len(selection['cards']),
        )

    def test_declaration_config_can_select_opponent_discard(self) -> None:
        rules = self._rules()
        source = self._card_instance('delay_nestling_hope', 'source', revealed=False, turn=2)
        source['staged'] = True
        discarded = self._card_instance('genesis_refresh_charge', 'discarded')
        discarded['side'] = rules.SIDE_B
        opponent_esper = self._card_instance('bohe', 'opponent-esper', revealed=True, turn=2)
        opponent_esper['side'] = rules.SIDE_B
        opponent_esper['consumed_material_names'] = ['畅爽焕能']
        snapshot = self._snapshot_with_cards([source])
        snapshot['phase'] = 'planning'
        snapshot['locations'][0]['cards'][rules.SIDE_B] = [opponent_esper]
        snapshot['sides'][rules.SIDE_B]['discard'] = [discarded]

        prepared = rules._prepare_declaration_selection(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertTrue(prepared)
        selection = snapshot['sides'][rules.SIDE_A]['selection']
        self.assertEqual([card['instance_id'] for card in selection['cards']], [discarded['instance_id']])

        rules.declarations.resolve_declaration_selection(snapshot, rules.SIDE_A, [discarded['instance_id']])
        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)

        self.assertEqual(snapshot['sides'][rules.SIDE_B]['discard'], [])
        self.assertEqual(
            [card['definition_id'] for card in snapshot['sides'][rules.SIDE_A]['hand']],
            ['delay_common_role_material_box'],
        )
        self.assertTrue(any('老旧信箱 将对手墓地的 畅爽焕能 除外' in line for line in snapshot['log']))
        self.assertTrue(any('老旧信箱 生成 1 张「初级角色异能材料自选箱」加入手牌' in line for line in snapshot['log']))

    def test_declaration_preview_requires_minimum_candidate_count(self) -> None:
        rules = self._rules()
        source = self._card_instance('murk_basic_material_box', 'source', revealed=False, turn=2)
        only_candidate = self._card_instance('murk_fantasy_delusion', 'only')
        snapshot = self._snapshot_with_cards([])
        snapshot['phase'] = 'planning'
        snapshot['sides'][rules.SIDE_A]['deck'] = [only_candidate]

        preview = rules._declaration_selection_preview(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertIsNone(preview)

    def test_ai_orders_murk_material_box_declaration_by_grave_and_hand_value(self) -> None:
        rules = self._rules()
        source = self._card_instance('murk_basic_material_box', 'source', revealed=False, turn=2)
        fantasy = self._card_instance('murk_fantasy_delusion', 'fantasy')
        blur = self._card_instance('murk_blur_number', 'blur')
        shadow = self._card_instance('murk_faded_shadow', 'shadow')
        snapshot = self._snapshot_with_cards([source])
        snapshot['sides'][rules.SIDE_A]['deck'] = [blur, fantasy, shadow]
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('adler', 'adler', revealed=False, turn=2),
            self._card_instance('requiem', 'requiem', revealed=False, turn=2),
        ]

        prepared = rules._prepare_declaration_selection(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertTrue(prepared)
        rules._resolve_ai_pending_choices(snapshot, rules.SIDE_A)
        self.assertEqual(source['declared_card_instance_ids'][0], fantasy['instance_id'])
        self.assertEqual(len(source['declared_card_instance_ids']), 2)
        self.assertNotEqual(source['declared_card_instance_ids'][1], fantasy['instance_id'])

    def test_board_declaration_uses_card_predicate_for_legal_targets(self) -> None:
        rules = self._rules()
        source = self._card_instance('genesis_marble_soda', 'source', revealed=False, turn=2)
        source['staged'] = True
        legal = self._card_instance('genesis_urban_energy', 'legal')
        illegal = self._card_instance('genesis_eborn_cake', 'illegal')
        illegal['computed_power'] = 4
        snapshot = self._snapshot_with_cards([source, legal, illegal])
        snapshot['phase'] = 'planning'

        prepared = rules._prepare_declaration_target(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            source,
        )

        self.assertTrue(prepared)
        pending = rules._public_pending_target(snapshot['sides'][rules.SIDE_A], snapshot)
        self.assertEqual(pending['target_instance_ids'], [legal['instance_id']])

    def test_common_role_material_box_is_tagged_and_not_constructible(self) -> None:
        rules = self._rules()
        loader = importlib.import_module('app.modules.card_game.content.loader')
        card = rules._card_by_id()['delay_common_role_material_box']

        ok, message = loader.validate_duel_deck_card_ids(['delay_common_role_material_box'] * 10)

        self.assertFalse(ok)
        self.assertIn('不能放入构筑', message)
        self.assertEqual(card.get('name'), '初级角色异能材料自选箱')
        self.assertEqual(card.get('cost'), 0)
        self.assertEqual(card.get('power'), 3)
        self.assertEqual(card.get('display_tags'), ['不可构筑'])
        self.assertFalse(card.get('deck_buildable'))
        self.assertNotIn('不可', card.get('description', ''))

    def test_ai_values_harmony_setup_before_ace_window(self) -> None:
        picked = self._run_delay_ai_esper_pick(turn=4)

        self.assertEqual(picked, 'hasuoer')

    def test_yi_esper_score_drops_without_current_delay(self) -> None:
        rules = self._rules()
        scoring = importlib.import_module('app.modules.card_game.engine.ai.scoring')
        yi = self._card_instance('yi', 'yi', revealed=False)
        material = self._card_instance('delay_mind_sync', 'mind', turn=1)
        snapshot = self._snapshot_with_cards([material])
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 3
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [yi]
        current_location = snapshot['locations'][0]
        remote_location = {
            'id': 'remote_archive',
            'name': '远端环线',
            'short_name': '远端',
            'reveal_turn': 1,
            'revealed': True,
            'art': '',
            'description': '',
            'effect': '',
            'cards': {rules.SIDE_A: [], rules.SIDE_B: []},
            'power': {rules.SIDE_A: 0, rules.SIDE_B: 0},
            'marks': {rules.SIDE_A: {rules.TAG_DELAY: 2}, rules.SIDE_B: {}},
            'winner_side': None,
        }
        snapshot['locations'].append(remote_location)

        no_current_delay = scoring.esper_contribution(
            snapshot,
            rules.SIDE_A,
            current_location,
            yi,
            [material],
        )
        current_location['marks'] = {rules.SIDE_A: {rules.TAG_DELAY: 1}, rules.SIDE_B: {}}
        with_current_delay = scoring.esper_contribution(
            snapshot,
            rules.SIDE_A,
            current_location,
            yi,
            [material],
        )

        self.assertLess(no_current_delay.total, 0.25)
        self.assertGreater(with_current_delay.total - no_current_delay.total, 25)
        self.assertEqual(no_current_delay.rounded()['breakdown']['condition'], -28.0)

    def test_yi_esper_score_accepts_same_turn_protagonist_delay_setup(self) -> None:
        rules = self._rules()
        scoring = importlib.import_module('app.modules.card_game.engine.ai.scoring')
        protagonist = self._card_instance('protagonist', 'protagonist', revealed=False)
        yi = self._card_instance('yi', 'yi', revealed=False)
        wish = self._card_instance('delay_first_wish', 'wish', turn=1)
        mind = self._card_instance('delay_mind_sync', 'mind', turn=1)
        water = self._card_instance('delay_water_hesitation', 'water', turn=1)
        protagonist['staged'] = True
        protagonist['played_turn'] = 3
        protagonist['play_sequence'] = 1
        protagonist['pending_material_ids'] = [wish['instance_id'], mind['instance_id']]
        snapshot = self._snapshot_with_cards([wish, mind, water, protagonist])
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 3
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [yi]

        contribution = scoring.esper_contribution(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            yi,
            [water],
        )

        self.assertGreater(contribution.total, 0.25)
        self.assertNotIn('condition', contribution.rounded()['breakdown'])

    def test_ai_defers_yi_until_current_delay_exists(self) -> None:
        no_delay_pick = self._run_delay_ai_yi_pick(delay_layers=0)
        with_delay_pick = self._run_delay_ai_yi_pick(delay_layers=1)

        self.assertEqual(no_delay_pick, 'hasuoer')
        self.assertEqual(with_delay_pick, 'yi')

    def test_ai_chains_protagonist_then_yi_when_delay_is_queued(self) -> None:
        rules = self._rules()
        materials = [
            self._card_instance('delay_mind_sync', 'mind', turn=1),
            self._card_instance('delay_water_hesitation', 'water', turn=1),
            self._card_instance('delay_first_wish', 'wish', turn=1),
        ]
        snapshot = self._snapshot_with_cards(materials)
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 3
        snapshot['sides'][rules.SIDE_A]['is_ai'] = True
        snapshot['sides'][rules.SIDE_A]['ai_plan'] = {
            'esper_priority_ids': ['protagonist', 'yi'],
            'ace_esper_ids': [],
        }
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('protagonist', 'protagonist', revealed=False),
            self._card_instance('yi', 'yi', revealed=False),
        ]

        rules._run_ai_turn(snapshot, rules.SIDE_A)

        picked_cards = [
            card
            for card in snapshot['locations'][0]['cards'][rules.SIDE_A]
            if card.get('type') == 'esper'
        ]
        self.assertEqual([card['definition_id'] for card in picked_cards], ['protagonist', 'yi'])
        for card in picked_cards:
            self.assertIn('ai_contribution', card)
            self.assertGreater(card['ai_contribution']['score'], 0)

    def test_ai_prioritizes_ace_esper_on_turn_five(self) -> None:
        picked = self._run_delay_ai_esper_pick(turn=5, delay_layers=1)

        self.assertEqual(picked, 'chaos')

    def test_murk_ai_prioritizes_requiem_when_ace_is_ready(self) -> None:
        rules = self._rules()
        loader = importlib.import_module('app.modules.card_game.content.loader')
        murk_deck = loader.get_duel_deck('murk_burn')
        self.assertIsNotNone(murk_deck)
        materials = [
            self._card_instance('discord_tomato_bucket', 'bucket', turn=1),
            self._card_instance('discord_tomato_100', 'tomato-100', turn=1),
            self._card_instance('murk_faded_shadow', 'shadow', turn=1),
            self._card_instance('murk_lost_whisper', 'whisper', turn=1),
        ]
        snapshot = self._snapshot_with_cards(materials)
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 4
        snapshot['sides'][rules.SIDE_A]['is_ai'] = True
        snapshot['sides'][rules.SIDE_A]['ai_plan'] = murk_deck['ai_plan']
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('adler', 'adler', revealed=False),
            self._card_instance('requiem', 'requiem', revealed=False),
        ]

        rules._run_ai_turn(snapshot, rules.SIDE_A)

        picked = [
            card['definition_id']
            for card in snapshot['locations'][0]['cards'][rules.SIDE_A]
            if card.get('type') == 'esper'
        ]
        self.assertEqual(picked, ['requiem'])

    def test_ai_prefers_new_esper_over_reactivation_when_lineup_member_is_ready(self) -> None:
        rules = self._rules()
        hasuoer = self._card_instance('hasuoer', 'hasuoer', revealed=True, turn=2)
        materials = [
            self._card_instance('delay_mind_sync', 'mind', turn=1),
            self._card_instance('delay_water_hesitation', 'water', turn=1),
            self._card_instance('delay_first_wish', 'wish', turn=1),
        ]
        snapshot = self._snapshot_with_cards([hasuoer, *materials])
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 5
        snapshot['locations'][0]['marks'] = {
            rules.SIDE_A: {rules.TAG_DELAY: 1},
            rules.SIDE_B: {},
        }
        snapshot['sides'][rules.SIDE_A]['is_ai'] = True
        snapshot['sides'][rules.SIDE_A]['ai_plan'] = {
            'esper_priority_ids': ['hasuoer', 'chaos'],
            'ace_esper_ids': ['chaos'],
        }
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('chaos', 'chaos', revealed=False),
        ]

        rules._run_ai_turn(snapshot, rules.SIDE_A)

        chaos_cards = [
            card
            for card in snapshot['locations'][0]['cards'][rules.SIDE_A]
            if card.get('definition_id') == 'chaos'
        ]
        self.assertEqual(len(chaos_cards), 1)
        self.assertTrue(chaos_cards[0].get('pending_material_ids'))
        self.assertFalse(hasuoer.get('pending_material_ids'))

    def _run_delay_ai_esper_pick(self, *, turn: int, delay_layers: int = 0) -> str:
        rules = self._rules()
        materials = [
            self._card_instance('delay_mind_sync', 'mind', turn=1),
            self._card_instance('delay_water_hesitation', 'water', turn=1),
            self._card_instance('delay_first_wish', 'wish', turn=1),
        ]
        snapshot = self._snapshot_with_cards(materials)
        snapshot['phase'] = 'planning'
        snapshot['turn'] = turn
        if delay_layers:
            snapshot['locations'][0]['marks'] = {
                rules.SIDE_A: {rules.TAG_DELAY: delay_layers},
                rules.SIDE_B: {},
            }
        snapshot['sides'][rules.SIDE_A]['is_ai'] = True
        snapshot['sides'][rules.SIDE_A]['ai_plan'] = {
            'esper_priority_ids': ['hasuoer', 'chaos'],
            'ace_esper_ids': ['chaos'],
        }
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('hasuoer', 'hasuoer', revealed=False),
            self._card_instance('chaos', 'chaos', revealed=False),
        ]

        rules._run_ai_turn(snapshot, rules.SIDE_A)

        picked_cards = [
            card
            for card in snapshot['locations'][0]['cards'][rules.SIDE_A]
            if card.get('type') == 'esper'
        ]
        picked = [card['definition_id'] for card in picked_cards]
        self.assertEqual(len(picked), 1)
        self.assertIn('ai_contribution', picked_cards[0])
        self.assertGreater(picked_cards[0]['ai_contribution']['score'], 0)
        return picked[0]

    def _run_delay_ai_yi_pick(self, *, delay_layers: int) -> str:
        rules = self._rules()
        materials = [
            self._card_instance('delay_mind_sync', 'mind', turn=1),
            self._card_instance('delay_first_wish', 'wish', turn=1),
        ]
        snapshot = self._snapshot_with_cards(materials)
        snapshot['phase'] = 'planning'
        snapshot['turn'] = 3
        if delay_layers:
            snapshot['locations'][0]['marks'] = {
                rules.SIDE_A: {rules.TAG_DELAY: delay_layers},
                rules.SIDE_B: {},
            }
        snapshot['sides'][rules.SIDE_A]['is_ai'] = True
        snapshot['sides'][rules.SIDE_A]['ai_plan'] = {
            'esper_priority_ids': ['yi', 'hasuoer'],
            'ace_esper_ids': [],
        }
        snapshot['sides'][rules.SIDE_A]['esper_standby'] = [
            self._card_instance('yi', 'yi', revealed=False),
            self._card_instance('hasuoer', 'hasuoer', revealed=False),
        ]

        rules._run_ai_turn(snapshot, rules.SIDE_A)

        picked_cards = [
            card
            for card in snapshot['locations'][0]['cards'][rules.SIDE_A]
            if card.get('type') == 'esper'
        ]
        picked = [card['definition_id'] for card in picked_cards]
        self.assertEqual(len(picked), 1)
        self.assertIn('ai_contribution', picked_cards[0])
        self.assertGreater(picked_cards[0]['ai_contribution']['score'], 0)
        return picked[0]

    def test_delay_nestling_wish_penalizes_next_turn_when_target_disappears(self) -> None:
        rules = self._rules()
        source = self._card_instance('delay_nestling_wish', 'source', revealed=False, turn=2)
        source['staged'] = True
        source['selected_target_instance_id'] = 'missing-target'
        snapshot = self._snapshot_with_cards([source])

        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)
        snapshot['turn'] = 3
        rules._begin_turn(snapshot)

        self.assertEqual(snapshot['sides'][rules.SIDE_B]['energy_used'], 1)
        self.assertEqual(rules._energy_remaining(snapshot, rules.SIDE_B), 2)

    def test_darkstar_nature_pixel_stops_revealing_after_first_match(self) -> None:
        rules = self._rules()
        source = self._card_instance('darkstar_nature_pixel', 'source', revealed=False, turn=2)
        source['staged'] = True
        first_nonmatch = self._card_instance('genesis_urban_energy', 'first')
        first_match = self._card_instance('darkstar_ringstone', 'match')
        uninspected = self._card_instance('darkstar_mining_permit', 'uninspected')
        snapshot = self._snapshot_with_cards([source])
        snapshot['sides'][rules.SIDE_A]['deck'] = [first_nonmatch, first_match, uninspected]

        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)

        self.assertEqual([card['definition_id'] for card in snapshot['sides'][rules.SIDE_A]['hand']], ['darkstar_ringstone'])
        self.assertEqual(
            [card['definition_id'] for card in snapshot['sides'][rules.SIDE_A]['deck']],
            ['darkstar_mining_permit', 'genesis_urban_energy'],
        )

    def test_tomato_bucket_stops_when_chain_mill_empties_deck(self) -> None:
        rules = self._rules()
        source = self._card_instance('discord_tomato_bucket', 'source', revealed=False, turn=2)
        source['staged'] = True
        blur = self._card_instance('murk_blur_number', 'blur')
        fantasy = self._card_instance('murk_fantasy_delusion', 'fantasy')
        snapshot = self._snapshot_with_cards([source])
        snapshot['sides'][rules.SIDE_A]['deck'] = [blur, fantasy]

        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)

        self.assertFalse(snapshot['sides'][rules.SIDE_A]['deck'])
        self.assertIn(blur, snapshot['sides'][rules.SIDE_A]['discard'])

    def test_silver_butler_uses_capped_nature_pixel_count_and_repeats(self) -> None:
        rules = self._rules()
        source = self._card_instance('darkstar_silver_butler', 'source', revealed=False, turn=2)
        source['staged'] = True
        board_pixel = self._card_instance('darkstar_nature_pixel', 'board-pixel')
        high_enemy = self._card_instance('genesis_muscle_faith', 'enemy-high')
        low_enemy = self._card_instance('genesis_eborn_cake', 'enemy-low')
        high_enemy['side'] = rules.SIDE_B
        low_enemy['side'] = rules.SIDE_B
        high_enemy['base_power'] = 10
        high_enemy['computed_power'] = 10
        low_enemy['base_power'] = 9
        low_enemy['computed_power'] = 9
        snapshot = self._snapshot_with_cards([source, board_pixel])
        snapshot['locations'][0]['cards'][rules.SIDE_B] = [high_enemy, low_enemy]
        snapshot['sides'][rules.SIDE_A]['hand'] = [
            self._card_instance('darkstar_nature_pixel', f'hand-{index}')
            for index in range(3)
        ]
        snapshot['sides'][rules.SIDE_A]['discard'] = [
            self._card_instance('darkstar_nature_pixel', f'discard-{index}')
            for index in range(3)
        ]

        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)

        self.assertEqual(high_enemy['computed_power'], 4)
        self.assertEqual(low_enemy['computed_power'], 3)

    def test_pending_material_consumption_does_not_sweep_reserved_materials(self) -> None:
        rules = self._rules()
        material_a = self._card_instance('genesis_refresh_charge', 'material-a')
        material_b = self._card_instance('genesis_marble_soda', 'material-b')
        material_a['bonus_power'] = -3
        material_a['computed_power'] = 1
        esper = self._card_instance('xun', 'esper-a', revealed=False, turn=2)
        esper['staged'] = True
        esper['summoned_from'] = 'esper_standby'
        esper['pending_material_ids'] = [material_a['instance_id'], material_b['instance_id']]
        material_a['reserved_as_material_for'] = esper['instance_id']
        material_b['reserved_as_material_for'] = esper['instance_id']
        snapshot = self._snapshot_with_cards([material_a, material_b, esper])
        snapshot['locations'][0]['effect'] = 'first_card_plus_two'

        rules._resolve_pending_material_consumption(snapshot)

        side = snapshot['sides'][rules.SIDE_A]
        self.assertEqual([card['definition_id'] for card in side['discard']], ['genesis_refresh_charge', 'genesis_marble_soda'])
        self.assertEqual(side['esper_standby'], [])
        self.assertEqual([card['definition_id'] for card in snapshot['locations'][0]['cards'][rules.SIDE_A]], ['xun'])
        self.assertFalse(any('素材不足' in line for line in snapshot['log']))

    def test_material_consumption_happens_before_new_cards_reveal(self) -> None:
        rules = self._rules()
        material_a = self._card_instance('genesis_refresh_charge', 'material-a')
        material_b = self._card_instance('genesis_marble_soda', 'material-b')
        esper = self._card_instance('xun', 'esper-a', revealed=False, turn=2)
        esper['staged'] = True
        esper['summoned_from'] = 'esper_standby'
        esper['pending_material_ids'] = [material_a['instance_id'], material_b['instance_id']]
        material_a['reserved_as_material_for'] = esper['instance_id']
        material_b['reserved_as_material_for'] = esper['instance_id']
        new_card = self._card_instance('genesis_breakfast_bag', 'new-card', revealed=False, turn=2)
        new_card['staged'] = True
        snapshot = self._snapshot_with_cards([material_a, material_b, esper, new_card])

        rules._resolve_pending_material_consumption(snapshot)

        remaining = [card['definition_id'] for card in snapshot['locations'][0]['cards'][rules.SIDE_A]]
        self.assertEqual(remaining, ['xun', 'genesis_breakfast_bag'])
        self.assertTrue(snapshot['locations'][0]['cards'][rules.SIDE_A][0]['staged'])
        self.assertFalse(snapshot['locations'][0]['cards'][rules.SIDE_A][1]['revealed'])
        consume_actions = [action for action in snapshot['action_queue'] if action['kind'] == 'consume_material']
        self.assertEqual([action['card']['definition_id'] for action in consume_actions], ['genesis_refresh_charge', 'genesis_marble_soda'])

    def test_reserved_materials_do_not_occupy_slots_or_score(self) -> None:
        rules = self._rules()
        materials = [
            self._card_instance('genesis_refresh_charge', f'material-{index}')
            for index in range(rules.LOCATION_CARD_LIMIT)
        ]
        esper = self._card_instance('jiuyuan', 'esper-a', revealed=False, turn=2)
        esper['staged'] = True
        esper['summoned_from'] = 'esper_standby'
        esper['pending_material_ids'] = [materials[0]['instance_id'], materials[1]['instance_id']]
        materials[0]['reserved_as_material_for'] = esper['instance_id']
        materials[1]['reserved_as_material_for'] = esper['instance_id']
        snapshot = self._snapshot_with_cards([*materials, esper])

        rules._recompute_scores(snapshot)
        location = snapshot['locations'][0]
        public_location = rules._public_location(
            location,
            rules.SidePerspective(own=rules.SIDE_A, opponent=rules.SIDE_B),
        )

        self.assertEqual(rules._location_occupied_card_count(location, rules.SIDE_A), rules.LOCATION_CARD_LIMIT - 1)
        self.assertEqual(len(public_location['slots']['player']), rules.LOCATION_CARD_LIMIT - 1)
        self.assertEqual(public_location['occupied']['player'], rules.LOCATION_CARD_LIMIT - 1)
        self.assertNotIn(materials[0]['instance_id'], [card['instance_id'] for card in public_location['slots']['player']])
        self.assertTrue(rules._location_has_room_after_materials(location, rules.SIDE_A, materials[:2]))
        self.assertIn(location, rules._open_locations(snapshot, rules.SIDE_A))
        expected_power = sum(int(card['computed_power']) for card in materials[2:])
        self.assertEqual(location['power'][rules.SIDE_A], expected_power)

    def test_refresh_charge_effect_deploy_uses_shared_location_capacity(self) -> None:
        rules = self._rules()
        source = self._card_instance('genesis_refresh_charge', 'source', revealed=False, turn=2)
        source['staged'] = True
        urban = self._card_instance('genesis_urban_energy', 'urban')
        filler_a = self._card_instance('genesis_breakfast_bag', 'filler-a')
        filler_b = self._card_instance('genesis_marble_soda', 'filler-b')
        hand_card = self._card_instance('genesis_breakfast_bag', 'hand-card')
        snapshot = self._snapshot_with_cards([source, urban, filler_a, filler_b])
        snapshot['locations'][0]['capacity'] = rules.LOCATION_CARD_LIMIT
        snapshot['sides'][rules.SIDE_A]['hand'] = [hand_card]

        rules._reveal_card(snapshot, rules.SIDE_A, snapshot['locations'][0], source)

        self.assertEqual(len(snapshot['locations'][0]['cards'][rules.SIDE_A]), 5)
        self.assertEqual(snapshot['locations'][0]['cards'][rules.SIDE_A][-1]['definition_id'], 'genesis_breakfast_bag')
        self.assertFalse(snapshot['sides'][rules.SIDE_A]['hand'])
        self.assertFalse(any('战场已满' in line for line in snapshot['log']))

    def test_multi_requirement_esper_material_accepts_matching_anomaly_items(self) -> None:
        rules = self._rules()
        material_a = self._card_instance('genesis_urban_energy', 'material-a')
        material_b = self._card_instance('surplus_fons', 'material-b')
        esper = self._card_instance('xiaozhi', 'esper-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([material_a, material_b])

        selected = rules._material_cards_for_esper(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            esper,
            [material_a['instance_id'], material_b['instance_id']],
        )

        self.assertEqual([card['definition_id'] for card in selected], ['genesis_urban_energy', 'surplus_fons'])

    def test_material_matching_accepts_any_valid_assignment_order(self) -> None:
        rules = self._rules()
        drink_material = self._card_instance('genesis_refresh_charge', 'material-a')
        light_material = self._card_instance('genesis_urban_energy', 'material-b')
        esper = self._card_instance('xun', 'esper-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([drink_material, light_material])

        selected = rules._material_cards_for_esper(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            esper,
            [drink_material['instance_id'], light_material['instance_id']],
        )
        auto_selected = rules._material_cards_for_esper(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            esper,
        )

        self.assertEqual([card['definition_id'] for card in selected], ['genesis_refresh_charge', 'genesis_urban_energy'])
        self.assertCountEqual([card['definition_id'] for card in auto_selected], ['genesis_refresh_charge', 'genesis_urban_energy'])

    def test_continuous_damage_marks_decay_at_turn_start(self) -> None:
        rules = self._rules()
        snapshot = self._snapshot_with_cards([])
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_NIGHTMARE: 2,
                rules.TAG_ZHUE_HUCHI: 1,
            },
            rules.SIDE_B: {},
        }

        rules._decay_harmony_layers_at_turn_start(snapshot)

        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_NIGHTMARE], 1)
        self.assertNotIn(rules.TAG_ZHUE_HUCHI, location['marks'][rules.SIDE_A])

    def test_base_harmony_marks_do_not_resolve_by_themselves(self) -> None:
        rules = self._rules()
        ally = self._card_instance('genesis_urban_energy', 'ally')
        enemy = self._card_instance('genesis_muscle_faith', 'enemy')
        enemy['side'] = rules.SIDE_B
        snapshot = self._snapshot_with_cards([ally])
        location = snapshot['locations'][0]
        location['cards'][rules.SIDE_B] = [enemy]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_GENESIS: 3,
                rules.TAG_MURK: 3,
            },
            rules.SIDE_B: {},
        }

        rules._resolve_harmony_end_of_turn(snapshot)

        self.assertEqual(ally['computed_power'], 2)
        self.assertEqual(enemy['computed_power'], 7)
        self.assertFalse(snapshot['action_queue'])

    def test_genesis_gain_creates_capped_surplus_mark(self) -> None:
        rules = self._rules()
        genesis_sources = [
            self._card_instance('protagonist', f'protagonist-{index}', revealed=False, turn=2)
            for index in range(3)
        ]
        for card in genesis_sources:
            card['consumed_material_attributes'] = ['灵']
        snapshot = self._snapshot_with_cards(genesis_sources)
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {rules.TAG_DELAY: 2},
            rules.SIDE_B: {},
        }

        for card in genesis_sources:
            rules._reveal_card(snapshot, rules.SIDE_A, location, card)

        marks = location['marks'][rules.SIDE_A]
        self.assertEqual(marks[rules.TAG_GENESIS], 3)
        self.assertEqual(marks[rules.TAG_SURPLUS], 2)

    def test_murk_or_darkstar_gain_creates_capped_discord_mark(self) -> None:
        rules = self._rules()
        fatiya_cards = [
            self._card_instance('fatiya', f'fatiya-{index}', revealed=False, turn=2)
            for index in range(3)
        ]
        snapshot = self._snapshot_with_cards(fatiya_cards)
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {rules.TAG_MURK: 2},
            rules.SIDE_B: {},
        }

        for card in fatiya_cards:
            rules._reveal_card(snapshot, rules.SIDE_A, location, card)

        marks = location['marks'][rules.SIDE_A]
        self.assertEqual(marks[rules.TAG_DARKSTAR], 3)
        self.assertEqual(marks[rules.TAG_DISCORD], 2)

    def test_hasuoer_creates_hamster_ball_when_delay_is_set(self) -> None:
        rules = self._rules()
        hasuoer = self._card_instance('hasuoer', 'hasuoer-a', revealed=True, turn=1)
        protagonist = self._card_instance('protagonist', 'protagonist-a', revealed=False, turn=2)
        protagonist['consumed_material_attributes'] = ['相']
        snapshot = self._snapshot_with_cards([hasuoer, protagonist])
        location = snapshot['locations'][0]

        rules._reveal_card(snapshot, rules.SIDE_A, location, protagonist)

        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_DELAY], 1)
        generated = [
            card
            for card in location['cards'][rules.SIDE_A]
            if card.get('definition_id') == 'delay_hamster_ball'
        ]
        self.assertEqual(len(generated), 1)
        self.assertTrue(generated[0]['revealed'])
        self.assertEqual(generated[0]['attribute'], '相')
        self.assertFalse(generated[0].get('deck_buildable'))

    def test_yi_requires_existing_delay_before_setting_delay(self) -> None:
        rules = self._rules()
        yi_without_delay = self._card_instance('yi', 'yi-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([yi_without_delay])
        location = snapshot['locations'][0]

        rules._reveal_card(snapshot, rules.SIDE_A, location, yi_without_delay)

        self.assertNotIn(rules.TAG_DELAY, location.get('marks', {}).get(rules.SIDE_A, {}))

        yi_with_delay = self._card_instance('yi', 'yi-b', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([yi_with_delay])
        location = snapshot['locations'][0]
        location['marks'] = {rules.SIDE_A: {rules.TAG_DELAY: 1}, rules.SIDE_B: {}}

        rules._reveal_card(snapshot, rules.SIDE_A, location, yi_with_delay)

        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_DELAY], 2)
        self.assertTrue(yi_with_delay.get('yi_delay_support_used'))

        rules._reveal_card(snapshot, rules.SIDE_A, location, yi_with_delay)

        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_DELAY], 2)

    def test_one_material_limited_espers_match_design(self) -> None:
        rules = self._rules()
        card_by_id = rules._card_by_id()
        for card_id, attribute in [
            ('bohe', '灵'),
            ('yi', '相'),
            ('baicang', '咒'),
            ('haniya', '魂'),
        ]:
            card = card_by_id[card_id]
            self.assertEqual(card['power'], 3)
            self.assertEqual(card['material_cost'], 1)
            self.assertEqual(card['material_requirements'], [{'attribute': attribute, 'count': 1}])
            self.assertIn('限1次', card['description'])
            self.assertEqual(rules._esper_material_cost(card), 1)

    def test_bohe_consumes_one_spirit_material_and_limited_boost(self) -> None:
        rules = self._rules()
        spirit = self._card_instance('genesis_refresh_charge', 'spirit-a')
        light = self._card_instance('genesis_urban_energy', 'light-a')
        bohe = self._card_instance('bohe', 'bohe-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([spirit, light, bohe])
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {rules.TAG_GENESIS: 1},
            rules.SIDE_B: {},
        }

        selected = rules._material_cards_for_esper(
            snapshot,
            rules.SIDE_A,
            location,
            bohe,
        )
        self.assertEqual([card['definition_id'] for card in selected], ['genesis_refresh_charge'])

        rules._reveal_card(snapshot, rules.SIDE_A, location, bohe)

        self.assertEqual(bohe['computed_power'], 6)
        self.assertTrue(bohe.get('bohe_genesis_boost_used'))

        rules._reveal_card(snapshot, rules.SIDE_A, location, bohe)

        self.assertEqual(bohe['computed_power'], 6)

    def test_haniya_counts_darkstar_layers_not_card_tags_and_limited_boost(self) -> None:
        rules = self._rules()
        haniya = self._card_instance('haniya', 'haniya-a', revealed=False, turn=2)
        fatiya = self._card_instance('fatiya', 'fatiya-a', revealed=True, turn=1)
        snapshot = self._snapshot_with_cards([haniya, fatiya])
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {rules.TAG_DARKSTAR: 2},
            rules.SIDE_B: {},
        }

        rules._reveal_card(snapshot, rules.SIDE_A, location, haniya)

        self.assertEqual(haniya['computed_power'], 5)
        self.assertEqual(fatiya['computed_power'], int(fatiya['base_power']) + 2)
        self.assertTrue(haniya.get('haniya_darkstar_support_used'))

        rules._reveal_card(snapshot, rules.SIDE_A, location, haniya)

        self.assertEqual(haniya['computed_power'], 5)
        self.assertEqual(fatiya['computed_power'], int(fatiya['base_power']) + 2)

    def test_fusion_marks_clamp_after_base_harmony_is_consumed(self) -> None:
        rules = self._rules()
        snapshot = self._snapshot_with_cards([])
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_MURK: 2,
                rules.TAG_DARKSTAR: 2,
                rules.TAG_DISCORD: 2,
            },
            rules.SIDE_B: {},
        }

        rules._consume_location_mark(location, rules.SIDE_A, rules.TAG_DARKSTAR, 1)

        marks = location['marks'][rules.SIDE_A]
        self.assertEqual(marks[rules.TAG_MURK], 2)
        self.assertEqual(marks[rules.TAG_DARKSTAR], 1)
        self.assertEqual(marks[rules.TAG_DISCORD], 1)

    def test_nightmare_and_panyu_qiu_resolve_at_end_of_turn(self) -> None:
        rules = self._rules()
        high_enemy = self._card_instance('genesis_muscle_faith', 'enemy-high')
        low_enemy = self._card_instance('genesis_urban_energy', 'enemy-low')
        for card in (high_enemy, low_enemy):
            card['side'] = rules.SIDE_B
        snapshot = self._snapshot_with_cards([])
        location = snapshot['locations'][0]
        location['cards'][rules.SIDE_B] = [high_enemy, low_enemy]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_NIGHTMARE: 3,
                rules.TAG_PANYU_QIU: 3,
            },
            rules.SIDE_B: {},
        }

        rules._resolve_harmony_end_of_turn(snapshot)

        self.assertEqual(high_enemy['computed_power'], 6)
        self.assertEqual([card['definition_id'] for card in location['cards'][rules.SIDE_B]], ['genesis_muscle_faith'])
        self.assertEqual([card['definition_id'] for card in snapshot['sides'][rules.SIDE_B]['discard']], ['genesis_urban_energy'])
        self.assertTrue(any(action['title'] == '噩梦' for action in snapshot['action_queue']))
        self.assertTrue(any(action['title'] == '判予秋' for action in snapshot['action_queue']))

    def test_zaowu_boosts_other_esper_after_resonance(self) -> None:
        rules = self._rules()
        zaowu = self._card_instance('zaowu', 'zaowu-a')
        adler = self._card_instance('adler', 'adler-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([zaowu, adler])
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_NIGHTMARE: 2,
            },
            rules.SIDE_B: {},
        }

        rules._reveal_card(snapshot, rules.SIDE_A, location, adler)

        self.assertEqual(adler['computed_power'], 7)
        self.assertTrue(any('早雾 使 阿德勒 +3' in line for line in snapshot['log']))
        self.assertTrue(any(action.get('subtitle') == '4 + 3 = 7' for action in snapshot['action_queue']))

    def test_zaowu_no_longer_adds_damage_to_end_phase_damage_packets(self) -> None:
        rules = self._rules()
        zaowu = self._card_instance('zaowu', 'zaowu-a')
        enemy = self._card_instance('genesis_muscle_faith', 'enemy-a')
        enemy['side'] = rules.SIDE_B
        snapshot = self._snapshot_with_cards([zaowu])
        location = snapshot['locations'][0]
        location['cards'][rules.SIDE_B] = [enemy]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_MURK: 1,
                rules.TAG_NIGHTMARE: 2,
                rules.TAG_ZHUE_HUCHI: 1,
            },
            rules.SIDE_B: {},
        }

        rules._resolve_harmony_end_of_turn(snapshot)

        self.assertEqual(enemy['computed_power'], 6)
        self.assertFalse(any('早雾 使本次结束阶段伤害' in line for line in snapshot['log']))
        self.assertTrue(any(action.get('subtitle') == '7 - 1 = 6' for action in snapshot['action_queue']))

    def test_delay_missed_call_caps_hand_gap_penalty_at_three(self) -> None:
        rules = self._rules()
        source = self._card_instance('delay_missed_call', 'source-a', revealed=False, turn=2)
        enemy = self._card_instance('genesis_eborn_cake', 'enemy-a')
        enemy['side'] = rules.SIDE_B
        snapshot = self._snapshot_with_cards([source])
        location = snapshot['locations'][0]
        location['cards'][rules.SIDE_B] = [enemy]
        snapshot['sides'][rules.SIDE_A]['hand'] = []
        snapshot['sides'][rules.SIDE_B]['hand'] = [
            self._card_instance('genesis_urban_energy', f'hand-{index}', revealed=False)
            for index in range(5)
        ]

        rules._reveal_card(snapshot, rules.SIDE_A, location, source)

        self.assertEqual(enemy['computed_power'], 1)
        self.assertTrue(any('无人来电 依据对手手牌多出的数量，使对手所有表侧卡牌 -3' in line for line in snapshot['log']))

    def test_hamster_ball_reduces_enemy_when_consumed_as_material(self) -> None:
        rules = self._rules()
        light_material = self._card_instance('delay_first_wish', 'light-a', revealed=True, turn=1)
        hamster = self._card_instance('delay_hamster_ball', 'hamster-a', revealed=True, turn=1)
        ally = self._card_instance('delay_missed_call', 'ally-a', revealed=True, turn=1)
        enemy = self._card_instance('genesis_urban_energy', 'enemy-a', revealed=True, turn=1)
        enemy['side'] = rules.SIDE_B
        hasuoer = self._card_instance('hasuoer', 'hasuoer-a', revealed=False, turn=2)
        hasuoer['staged'] = True
        hasuoer['summoned_from'] = 'esper_standby'
        hasuoer['pending_material_ids'] = [light_material['instance_id'], hamster['instance_id']]
        light_material['reserved_as_material_for'] = hasuoer['instance_id']
        hamster['reserved_as_material_for'] = hasuoer['instance_id']
        snapshot = self._snapshot_with_cards([light_material, hamster, ally, hasuoer])
        snapshot['locations'][0]['cards'][rules.SIDE_B] = [enemy]

        rules._resolve_pending_material_consumption(snapshot)

        self.assertEqual(ally['computed_power'], 4)
        self.assertEqual(enemy['computed_power'], 1)
        self.assertEqual(hasuoer['absorbed_material_power'], 4)
        self.assertTrue(any('仓鼠球 被当作素材消耗，使对手 都市活力 -1' in line for line in snapshot['log']))

    def test_xun_copies_highest_harmony_and_restores_consumed_light_material(self) -> None:
        rules = self._rules()
        xun = self._card_instance('xun', 'xun-a', revealed=False, turn=2)
        light_material = self._card_instance('genesis_urban_energy', 'light-a', revealed=True, turn=1)
        light_material['location_id'] = None
        xun['consumed_material_instance_ids'] = [light_material['instance_id']]
        snapshot = self._snapshot_with_cards([xun])
        snapshot['sides'][rules.SIDE_A]['discard'] = [light_material]
        location = snapshot['locations'][0]
        location['marks'] = {
            rules.SIDE_A: {
                rules.TAG_GENESIS: 2,
                rules.TAG_DELAY: 2,
            },
            rules.SIDE_B: {},
        }

        rules._reveal_card(snapshot, rules.SIDE_A, location, xun)

        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_GENESIS], 3)
        self.assertEqual(location['marks'][rules.SIDE_A][rules.TAG_DELAY], 2)
        self.assertEqual(xun['computed_power'], 6)
        self.assertEqual(snapshot['sides'][rules.SIDE_A]['discard'], [])
        restored = [
            card
            for card in location['cards'][rules.SIDE_A]
            if card.get('instance_id') == light_material['instance_id']
        ]
        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0]['revealed'])
        self.assertTrue(xun.get('xun_light_restore_used'))

    def test_xun_requires_light_and_drink_materials(self) -> None:
        rules = self._rules()
        light = self._card_instance('genesis_urban_energy', 'light-a')
        drink = self._card_instance('genesis_refresh_charge', 'drink-a')
        cake = self._card_instance('genesis_eborn_cake', 'cake-a')
        xun = self._card_instance('xun', 'xun-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([light, drink, cake])

        selected = rules._material_cards_for_esper(
            snapshot,
            rules.SIDE_A,
            snapshot['locations'][0],
            xun,
        )

        self.assertCountEqual(
            [card['definition_id'] for card in selected],
            ['genesis_urban_energy', 'genesis_refresh_charge'],
        )

    def test_murk_item_pressure_does_not_create_murk_mark(self) -> None:
        rules = self._rules()
        murk_item = self._card_instance('murk_lost_whisper', 'murk-a', revealed=False, turn=2)
        target_definition = rules._card_by_id()['genesis_muscle_faith']
        target = rules._card_instance(target_definition, rules.SIDE_B, 'target-b')
        target['revealed'] = True
        target['played_turn'] = 1
        target['location_id'] = 'mirror_archive'
        snapshot = self._snapshot_with_cards([murk_item])
        location = snapshot['locations'][0]
        location['cards'][rules.SIDE_B].append(target)

        rules._reveal_card(snapshot, rules.SIDE_A, location, murk_item)

        self.assertEqual(location.get('marks', {}).get(rules.SIDE_B, {}).get(rules.TAG_MURK, 0), 0)
        self.assertEqual(target['bonus_power'], -1)
        self.assertFalse(any(action.get('kind') == 'spawn_mark' and action.get('mark') == rules.TAG_MURK for action in snapshot['action_queue']))

    def test_non_positive_anomaly_item_is_not_valid_esper_material(self) -> None:
        rules = self._rules()
        material_a = self._card_instance('genesis_refresh_charge', 'material-a')
        material_b = self._card_instance('genesis_marble_soda', 'material-b')
        material_a['bonus_power'] = -2
        material_a['computed_power'] = 0
        esper = self._card_instance('bohe', 'esper-a', revealed=False, turn=2)
        snapshot = self._snapshot_with_cards([material_a, material_b])

        with self.assertRaisesRegex(rules.RuleValidationError, '战力为正'):
            rules._material_cards_for_esper(
                snapshot,
                rules.SIDE_A,
                snapshot['locations'][0],
                esper,
                [material_a['instance_id']],
            )


class DuoRoomFlowTest(RoomFlowTestCase):
    def test_duo_room_requires_both_players_ready_before_start(self) -> None:
        host_token = self._issue_login_and_get_token('duo-host-waiting')
        guest_token = self._issue_login_and_get_token('duo-guest-waiting')
        self._save_default_build(host_token)
        self._save_default_build(guest_token)

        room_payload = self._post('/api/room/create', {'mode': 'duo'}, token=host_token)
        room_code = room_payload['room_code']
        self.assertEqual(room_payload['status'], 'waiting')
        self.assertFalse(room_payload['members'][0]['is_ready'])

        joined_payload = self._post('/api/room/join', {'room_code': room_code}, token=guest_token)
        self.assertEqual(len(joined_payload['members']), 2)
        self.assertEqual(joined_payload['status'], 'waiting')

        start_error = self._post_error('/api/room/start', token=host_token)
        self.assertEqual(start_error['error'], '仍有玩家未准备，无法开始游戏。')

    def test_duo_room_only_host_can_start(self) -> None:
        host_token = self._issue_login_and_get_token('duo-host-only')
        guest_token = self._issue_login_and_get_token('duo-guest-only')
        self._save_default_build(host_token)
        self._save_default_build(guest_token)

        room_payload = self._post('/api/room/create', {'mode': 'duo'}, token=host_token)
        room_code = room_payload['room_code']
        self._post('/api/room/join', {'room_code': room_code}, token=guest_token)
        self._post('/api/room/ready', {'is_ready': True}, token=host_token)
        self._post('/api/room/ready', {'is_ready': True}, token=guest_token)

        guest_start_error = self._post_error('/api/room/start', token=guest_token)
        self.assertEqual(guest_start_error['error'], '只有房主可以开始游戏。')

    def test_duo_room_can_start_and_advance_after_both_players_act(self) -> None:
        host_token = self._issue_login_and_get_token('duo-host-start')
        guest_token = self._issue_login_and_get_token('duo-guest-start')
        self._save_default_build(host_token)
        self._save_default_build(guest_token)

        room_payload = self._post('/api/room/create', {'mode': 'duo'}, token=host_token)
        room_code = room_payload['room_code']
        self._post('/api/room/join', {'room_code': room_code}, token=guest_token)

        ready_payload = self._post('/api/room/ready', {'is_ready': True}, token=host_token)
        self.assertEqual(ready_payload['status'], 'waiting')
        ready_payload = self._post('/api/room/ready', {'is_ready': True}, token=guest_token)
        self.assertEqual(ready_payload['status'], 'ready')

        with self._duo_test_run_context():
            start_payload = self._post('/api/room/start', token=host_token)
            self.assertEqual(start_payload['status'], 'playing')
            self.assertEqual(start_payload['game_id'], 'anomaly_snap_duel')
            self.assertEqual(start_payload['room']['room_code'], room_code)
            self.assertEqual(start_payload['room']['mode'], 'duo')
            self.assertEqual(start_payload['turn'], 1)
            self.assertEqual(start_payload['phase'], 'planning')
            self.assertIsNone(start_payload['selection'])
            self.assertEqual(len(start_payload['player']['hand']), 4)
            self.assertEqual(start_payload['current_actor_uid'], 'duo-host-start')
            self.assertEqual(len(start_payload['players_overview']), 2)

            room_state = self._get('/api/room/state', token=host_token)
            self.assertEqual(room_state['status'], 'playing')
            self.assertEqual(room_state['run_status'], 'playing')

            guest_start_view = self._get('/api/game/state', token=guest_token)
            self.assertEqual(guest_start_view['phase'], 'planning')
            self.assertIsNone(guest_start_view['selection'])

            host_after_end = self._post('/api/game/end-turn', token=host_token)
            self.assertEqual(host_after_end['turn'], 1)
            self.assertEqual(host_after_end['phase'], 'waiting')
            self.assertTrue(host_after_end['player']['ended_turn'])

            guest_view_before_end = self._get('/api/game/state', token=guest_token)
            self.assertEqual(guest_view_before_end['turn'], 1)
            self.assertEqual(guest_view_before_end['phase'], 'planning')
            self.assertEqual(guest_view_before_end['current_actor_uid'], 'duo-guest-start')

            guest_after_end = self._post('/api/game/end-turn', token=guest_token)
            self.assertEqual(guest_after_end['turn'], 2)
            self.assertEqual(guest_after_end['phase'], 'planning')
            self.assertEqual(guest_after_end['current_actor_uid'], 'duo-guest-start')
            self.assertIsNone(guest_after_end['selection'])

        host_view_next_turn = self._get('/api/game/state', token=host_token)
        self.assertEqual(host_view_next_turn['turn'], 2)
        self.assertEqual(host_view_next_turn['phase'], 'planning')
        self.assertEqual(host_view_next_turn['current_actor_uid'], 'duo-host-start')

    def _duo_test_run_context(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch('app.modules.card_game.engine.setup.snapshot_factory.random.shuffle', side_effect=lambda value: None))
        stack.enter_context(self._fixed_battlefield_patch())
        return stack


if __name__ == '__main__':
    unittest.main()
