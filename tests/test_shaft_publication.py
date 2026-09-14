import importlib
from copy import deepcopy

from tests.test_solo_room_flow import RoomFlowTestCase


class ShaftPublicationTest(RoomFlowTestCase):
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
