import importlib
import json
from copy import deepcopy

from tests.test_solo_room_flow import RoomFlowTestCase


class ShaftPublicationTest(RoomFlowTestCase):
    def test_online_copy_requires_explicit_overwrite_when_matched_by_source(self):
        owner = self._issue_login_and_get_token('copy-source-owner')
        viewer = self._issue_login_and_get_token('copy-viewer')
        axis = self._get('/api/shaft/catalog', token=owner)['starter_axis']
        payload = {'title': 'Online original', 'axis': axis, 'result': self._shaft_client_result(axis)}
        saved = self._post('/api/shaft/axes', payload, token=owner)
        public = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=owner)
        copy_payload = {**payload, 'title': 'Local copy', 'source_axis_id': public['id'], 'description': 'custom changes'}
        local = self._post('/api/shaft/axes', copy_payload, token=viewer)
        replacement = {**copy_payload, 'description': 'online replacement'}
        response = self.client.post('/api/shaft/axes', json=replacement, headers={'Authorization': f'Bearer {viewer}'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()['existing_axis_id'], local['id'])
        self.assertEqual(self._get(f"/api/shaft/axes/{local['id']}", token=viewer)['description'], 'custom changes')
        updated = self._post('/api/shaft/axes', {**replacement, 'conflict_action': 'overwrite'}, token=viewer)
        self.assertEqual(updated['id'], local['id'])
        self.assertEqual(updated['description'], 'online replacement')

    def test_retired_lingke_joint_migrates_on_save_share_and_public_snapshot(self):
        token = self._issue_login_and_get_token('legacy-lingke-owner')
        axis = {'team': [{'slot': 0, 'character_id': 'char_0846d632e0', 'arc_id': '', 'cartridge_id': ''},
                         {'slot': 1, 'character_id': 'char_701295143d', 'arc_id': '', 'cartridge_id': ''}],
                'steps': [{'id': 'old-joint', 'slot': 0, 'action_id': 'action_lingke_joint_curse', 'start_tick': 15}]}
        saved = self._post('/api/shaft/axes', {
            'title': '旧同频兼容', 'axis': axis, 'result': self._shaft_client_result(axis),
        }, token=token)
        # Emulate an existing persisted snapshot created before the migration existed.
        models = importlib.import_module('app.models')
        models.ShaftAxis.update(axis_json=json.dumps(axis)).where(models.ShaftAxis.id == saved['id']).execute()
        shared = self._post(f"/api/shaft/axes/{saved['id']}/share", {}, token=token)
        published = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=token)
        for payload in (
            self._get(f"/api/shaft/axes/{saved['id']}", token=token),
            self._get(f"/api/shaft/shared/{shared['share_token']}"),
            self._get(f"/api/shaft/axes/{published['id']}"),
        ):
            self.assertTrue(payload['legacy_actions_migrated'])
            self.assertEqual(payload['result'], {})
            step = payload['axis']['steps'][0]
            self.assertEqual((step['id'], step['slot'], step['start_tick'], step['action_id']),
                             ('old-joint', 1, 15, 'action_222f577087'))

    def test_heiyu_release_migrates_and_is_public_across_workflows(self):
        models = importlib.import_module('app.models')
        service = importlib.import_module('app.modules.shaft.service')
        model = models.ShaftCharacterPublication
        model.update(is_published=False, access_level='test').where(
            model.character_id == 'char_heiyu'
        ).execute()
        service.initialize_shaft_character_publications()
        publication = model.get(model.character_id == 'char_heiyu')
        self.assertTrue(publication.is_published)
        self.assertEqual(publication.access_level, 'public')
        regular = self._issue_login_and_get_token('heiyu-regular')
        invited = self._issue_login_and_get_token('heiyu-invited')
        tester = self._issue_login_and_get_token('heiyu-tester')
        models.Player.update(shaft_invited=True).where(models.Player.player_uid == 'heiyu-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid == 'heiyu-tester').execute()
        for token in (None, regular, invited, tester):
            headers = {'Authorization': f'Bearer {token}'} if token else {}
            catalog = self.client.get('/api/shaft/catalog', headers=headers).get_json()
            character = next(c for c in catalog['characters'] if c['id'] == 'char_heiyu')
            self.assertFalse(character['selection_disabled'])
            akane = next(c for c in catalog['characters'] if c['id'] == 'char_akane')
            self.assertEqual(akane['selection_disabled'], token != tester)
            kongmu = self.client.get('/api/kongmu/catalog', headers=headers).get_json()
            self.assertIn('char_heiyu', [c['id'] for c in kongmu['characters']])
            plan = self.client.post('/api/kongmu/plan', json={'character_id': 'char_heiyu', 'cartridge_id': 'attack'}, headers=headers)
            self.assertEqual(plan.status_code, 200)
        axis = {'team': [{'slot': 0, 'character_id': 'char_heiyu', 'arc_id': '', 'cartridge_id': ''}], 'steps': []}
        payload = {'title': '黑羽公开权限', 'axis': axis, 'result': self._shaft_client_result(axis)}
        for owner in (regular, invited, tester):
            saved = self._post('/api/shaft/axes', payload, token=owner)
            if owner == regular:
                published = self._post(f"/api/shaft/axes/{saved['id']}/publish", {}, token=owner)
        for viewer in (None, regular, invited, tester):
            market = self._get('/api/shaft/market', token=viewer)
            self.assertIn(published['id'], [a['id'] for a in market['items']])
            self._get(f"/api/shaft/axes/{published['id']}", token=viewer)
        for collector in (invited, tester):
            self._post(f"/api/shaft/axes/{published['id']}/favorite", {}, token=collector)
            favorites = self._get('/api/shaft/me/favorites', token=collector)
            self.assertIn(published['id'], [a['id'] for a in favorites['items']])

    def test_akane_kongmu_is_public_with_source_backed_grid(self):
        models = importlib.import_module('app.models')
        regular = self._issue_login_and_get_token('akane-kongmu-regular')
        tester = self._issue_login_and_get_token('akane-kongmu-tester')
        models.Player.update(shaft_test_whitelisted=True).where(
            models.Player.player_uid == 'akane-kongmu-tester'
        ).execute()
        regular_catalog = self._get('/api/kongmu/catalog', token=regular)
        self.assertIn('char_akane', [c['id'] for c in regular_catalog['characters']])
        catalog = self.client.get('/api/kongmu/catalog').get_json()
        akane = next(c for c in catalog['characters'] if c['id'] == 'char_akane')
        self.assertEqual(akane['avatar'], '/static/images/characters/avatar/明音凛.webp')
        self.assertIn('攻击力百分比', akane['kongmu_passive']['text'])
        self.assertIn('10%', akane['kongmu_passive']['text'])
        self.assertEqual(akane['owner_grid_count'], 3)
        plan = self.client.post('/api/kongmu/plan', json={
            'character_id': 'char_akane', 'cartridge_id': 'attack'
        }).get_json()
        self.assertEqual(plan['character']['id'], 'char_akane')
        slots = plan['character']['equip_slots']['slots']
        self.assertEqual(sum(cell >= 0 for row in slots for cell in row), 20)
        self.assertGreater(plan['result']['total_solution_count'], 0)
        regular_plan = self.client.post('/api/kongmu/plan', json={
            'character_id': 'char_akane', 'cartridge_id': 'attack'
        }, headers={'Authorization': f'Bearer {regular}'})
        self.assertEqual(regular_plan.status_code, 200)
        tester_plan = self.client.post('/api/kongmu/plan', json={
            'character_id': 'char_akane', 'cartridge_id': 'attack'
        }, headers={'Authorization': f'Bearer {tester}'})
        self.assertEqual(tester_plan.status_code, 200)

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
