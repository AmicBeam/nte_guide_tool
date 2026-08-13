import importlib
from pathlib import Path

from tests.test_solo_room_flow import RoomFlowTestCase


ROOT = Path(__file__).resolve().parents[1]


class ModuleRoutesTest(RoomFlowTestCase):
    def _assert_asset(self, path: str) -> None:
        response = self.client.get(path)
        try:
            self.assertEqual(response.status_code, 200)
        finally:
            response.close()

    def test_card_game_module_page_and_asset(self) -> None:
        page = self.client.get('/card-game')
        self.assertEqual(page.status_code, 200)
        self.assertIn('异象对决', page.get_data(as_text=True))
        self._assert_asset('/static/card_game/js/home.js')

    def test_kongmu_module_page_catalog_and_asset(self) -> None:
        page = self.client.get('/kongmu')
        self.assertEqual(page.status_code, 200)
        page_text = page.get_data(as_text=True)
        self.assertLess(
            page_text.index('/static/common/js/common.js'),
            page_text.index('/static/kongmu/js/kongmu.js'),
        )
        catalog = self.client.get('/api/kongmu/catalog')
        self.assertEqual(catalog.status_code, 200)
        self.assertTrue(catalog.is_json)
        zero = next(character for character in catalog.get_json()['characters'] if character['name'] == '「零」')
        self.assertEqual(zero['avatar'], '/static/images/characters/avatar/男主.webp')
        avatar = self.client.get(zero['avatar'])
        try:
            self.assertEqual(avatar.status_code, 200)
            self.assertIn('max-age=31536000', avatar.headers.get('Cache-Control', ''))
            self.assertIn('immutable', avatar.headers.get('Cache-Control', ''))
        finally:
            avatar.close()
        self.assertEqual(catalog.headers.get('Cache-Control'), 'private, no-store')
        self.assertIn('Authorization', catalog.headers.get('Vary', ''))
        frontend = (
            ROOT / 'app' / 'modules' / 'kongmu' / 'static' / 'js' / 'kongmu.js'
        ).read_text(encoding='utf-8')
        self.assertNotIn('pickCharacterAvatar', frontend)
        self.assertNotIn('avatarChoiceIndexes', frontend)
        self.assertIn("{cache: 'no-store'}", frontend)
        canhong = next(character for character in catalog.get_json()['characters'] if character['name'] == '残红')
        self.assertEqual(canhong['avatar'], '/static/images/characters/avatar/残红.png')
        self._assert_asset(canhong['avatar'])
        self.assertIn('headers.Authorization = `Bearer ${token}`', frontend)
        self._assert_asset('/static/kongmu/js/kongmu.js')

    def test_kongmu_half_open_character_is_public(self) -> None:
        anonymous_catalog = self.client.get('/api/kongmu/catalog').get_json()
        anonymous_canhong = next(character for character in anonymous_catalog['characters'] if character['name'] == '残红')
        self.assertEqual(anonymous_canhong['avatar'], '/static/images/characters/avatar/残红.png')

        anonymous_plan = self.client.post('/api/kongmu/plan', json={
            'character_id': anonymous_canhong['id'],
            'cartridge_id': 'attack',
        })
        self.assertEqual(anonymous_plan.status_code, 200)

        token = self._issue_login_and_get_token('kongmu-tester')
        regular_headers = {'Authorization': f'Bearer {token}'}
        regular_catalog = self.client.get('/api/kongmu/catalog', headers=regular_headers).get_json()
        regular_canhong = next(character for character in regular_catalog['characters'] if character['name'] == '残红')

        allowed_without_permission = self.client.post('/api/kongmu/plan', json={
            'character_id': regular_canhong['id'],
            'cartridge_id': 'attack',
        }, headers=regular_headers)
        self.assertEqual(allowed_without_permission.status_code, 200)

        models_module = importlib.import_module('app.models')
        models_module.Player.update(shaft_test_whitelisted=True).where(
            models_module.Player.player_uid == 'kongmu-tester'
        ).execute()
        allowed_catalog = self.client.get('/api/kongmu/catalog', headers=regular_headers).get_json()
        canhong = next(character for character in allowed_catalog['characters'] if character['name'] == '残红')
        self.assertEqual(canhong['owner_grid_count'], 3)
        self.assertIn('16%', canhong['kongmu_passive']['text'])

        allowed = self.client.post('/api/kongmu/plan', json={
            'character_id': canhong['id'],
            'cartridge_id': 'attack',
        }, headers=regular_headers)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            allowed.get_json()['character']['equip_slots']['slots'],
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0, 0, -1, 0, -1],
                [-1, 0, 0, -1, -1, 0, -1],
                [-1, 0, 0, 0, 0, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
        )

        invited_token = self._issue_login_and_get_token('kongmu-invited')
        models_module.Player.update(shaft_invited=True).where(
            models_module.Player.player_uid == 'kongmu-invited'
        ).execute()
        invited_headers = {'Authorization': f'Bearer {invited_token}'}
        invited_catalog = self.client.get('/api/kongmu/catalog', headers=invited_headers).get_json()
        self.assertIn('残红', [character['name'] for character in invited_catalog['characters']])

    def test_preteam_module_page_and_asset(self) -> None:
        page = self.client.get('/preteam')
        self.assertEqual(page.status_code, 200)
        page_text = page.get_data(as_text=True)
        self.assertIn('异环预配队', page_text)
        self.assertIn('预配队即将下线', page_text)
        self.assertIn('排轴模块', page_text)
        self._assert_asset('/static/preteam/单位.jpg')

    def test_portal_shows_community_links_record_and_preteam_sunset(self) -> None:
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        page_text = page.get_data(as_text=True)
        self.assertIn('https://github.com/AmicBeam/nte_board_game', page_text)
        self.assertIn('在 GitHub 查看 NTE Tools', page_text)
        self.assertIn('https://space.bilibili.com/9412490', page_text)
        self.assertIn('https://space.bilibili.com/3546651192986524', page_text)
        self.assertIn('津ICP备2026003916号', page_text)
        self.assertIn('即将下线，并由排轴模块取代', page_text)

    def test_shaft_module_page_catalog_and_asset(self) -> None:
        self.assertEqual(self.client.get('/shaft/rotation').status_code, 200)
        catalog = self.client.get('/api/shaft/catalog')
        self.assertEqual(catalog.status_code, 200)
        self.assertTrue(catalog.is_json)
        self._assert_asset('/static/shaft/js/shaft.js')
        shaft_axis_columns = {
            column.name
            for column in self.db_module.db.get_columns('shaftaxis')
        }
        self.assertIn('dislike_count', shaft_axis_columns)
        self.assertTrue(self.db_module.db.table_exists('shaftaxisdislike'))

    def test_module_owned_files_are_not_left_in_legacy_directories(self) -> None:
        for legacy_path in (
            ROOT / 'app' / 'content',
            ROOT / 'app' / 'engine',
            ROOT / 'app' / 'shaft',
            ROOT / 'app' / 'templates' / 'card_game',
            ROOT / 'app' / 'templates' / 'kongmu',
            ROOT / 'app' / 'templates' / 'preteam',
            ROOT / 'app' / 'templates' / 'shaft',
            ROOT / 'app' / 'static' / 'card_game',
            ROOT / 'app' / 'static' / 'kongmu',
            ROOT / 'app' / 'static' / 'preteam',
            ROOT / 'app' / 'static' / 'shaft',
        ):
            self.assertFalse(legacy_path.exists(), str(legacy_path))
