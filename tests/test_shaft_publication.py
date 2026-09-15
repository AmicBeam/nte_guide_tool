import importlib
from copy import deepcopy

from tests.test_solo_room_flow import RoomFlowTestCase


class ShaftPublicationTest(RoomFlowTestCase):
    def test_heiyu_is_test_only_across_catalog_save_and_market(self):
        models = importlib.import_module('app.models')
        regular = self._issue_login_and_get_token('heiyu-regular')
        invited = self._issue_login_and_get_token('heiyu-invited')
        tester = self._issue_login_and_get_token('heiyu-tester')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid == 'heiyu-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid == 'heiyu-tester').execute()
        for token in (None, regular, invited, tester):
            headers = {'Authorization': f'Bearer {token}'} if token else {}
            catalog = self.client.get('/api/shaft/catalog', headers=headers).get_json()
            character = next(c for c in catalog['characters'] if c['id'] == 'char_heiyu')
            self.assertEqual(character['selection_disabled'], token != tester)
            kongmu = self.client.get('/api/kongmu/catalog', headers=headers).get_json()
            self.assertEqual('char_heiyu' in [c['id'] for c in kongmu['characters']], token == tester)
            plan = self.client.post('/api/kongmu/plan', json={'character_id': 'char_heiyu', 'cartridge_id': 'attack'}, headers=headers)
            self.assertEqual(plan.status_code, 200 if token == tester else 400)
        axis = {'team': [{'slot': 0, 'character_id': 'char_heiyu', 'arc_id': '', 'cartridge_id': ''}], 'steps': []}
        payload = {'title': '黑羽测试权限', 'axis': axis, 'result': self._shaft_client_result(axis)}
        for token in (regular, invited):
            response = self.client.post('/api/shaft/axes', json=payload, headers={'Authorization': f'Bearer {token}'})
            self.assertEqual(response.status_code, 400)
        saved = self._post('/api/shaft/axes', payload, token=tester)
        published = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=tester)
        for token in (None, regular, invited):
            market = self._get('/api/shaft/market', token=token)
            self.assertNotIn(published['id'], [a['id'] for a in market['items']])

    def test_lingke_release_migrates_and_allows_regular_account_workflows(self):
        models = importlib.import_module('app.models')
        service = importlib.import_module('app.modules.shaft.service')
        model = models.ShaftCharacterPublication
        model.update(is_published=False, access_level='test').where(
            model.character_name == '灵可'
        ).execute()
        service.initialize_shaft_character_publications()
        publication = model.get(model.character_name == '灵可')
        self.assertTrue(publication.is_published)
        self.assertEqual(publication.access_level, 'public')
        catalog = self._get('/api/shaft/catalog')
        lingke = next(item for item in catalog['characters'] if item['name'] == '灵可')
        self.assertFalse(lingke['selection_disabled'])
        token = self._issue_login_and_get_token('lingke-public-player')
        axis = deepcopy(catalog['starter_axis'])
        axis['team'] = [{**axis['team'][0], 'character_id': lingke['id'], 'arc_id': '', 'cartridge_id': ''}]
        axis['character_builds'] = {}
        axis['steps'] = []
        saved = self._post('/api/shaft/axes', {
            'title': '灵可公开权限回归', 'axis': axis,
            'result': self._shaft_client_result(axis),
        }, token=token)
        published = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=token)
        market = self._get('/api/shaft/market')
        self.assertIn(published['id'], [item['id'] for item in market['items']])
        self._get(f"/api/shaft/axes/{published['id']}")
        viewer = self._issue_login_and_get_token("lingke-public-viewer")
        self._post(f"/api/shaft/axes/{published['id']}/favorite", {}, token=viewer)
        favorites = self._get('/api/shaft/me/favorites', token=viewer)
        self.assertIn(published['id'], [item['id'] for item in favorites['items']])
