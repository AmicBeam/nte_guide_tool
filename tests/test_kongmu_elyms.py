import importlib

from tests.test_solo_room_flow import RoomFlowTestCase


class ElymsKongmuTest(RoomFlowTestCase):
    def accounts(self):
        models = importlib.import_module('app.models')
        tokens = {role: self._issue_login_and_get_token('kongmu-elyms-' + role)
                  for role in ('regular', 'invited', 'tester')}
        models.Player.update(shaft_invited=True).where(models.Player.player_uid == 'kongmu-elyms-invited').execute()
        models.Player.update(shaft_test_whitelisted=True).where(models.Player.player_uid == 'kongmu-elyms-tester').execute()
        return tokens

    def test_all_users_can_see_and_calculate_elyms(self):
        tokens = self.accounts()
        # Warm the tester cache first, then verify every public access path.
        tester_catalog = self._get('/api/kongmu/catalog', token=tokens['tester'])
        tester_ids = {c['id'] for c in tester_catalog['characters']}
        self.assertIn('1090', tester_ids)
        for token in (None, tokens['regular'], tokens['invited'], 'invalid-token'):
            public = self._get('/api/kongmu/catalog', token=token)
            ids = {c['id'] for c in public['characters']}
            self.assertIn('1090', ids)
            self.assertEqual(tester_ids, ids)
            self.assertEqual(public['meta']['character_count'], len(ids))
            headers = {'Authorization': f'Bearer {token}'} if token else {}
            for identifier in ('1090', 'char_elyms', '曳尔沐', 'Elyms'):
                response = self.client.post('/api/kongmu/plan', json={
                    'character_id': identifier, 'cartridge_id': 'attack',
                    'access_level': 'test', 'shaft_test_whitelisted': True}, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json['character']['id'], '1090')
            self.assertEqual(self.client.post('/api/kongmu/plan', json={
                'character_id': '1057', 'cartridge_id': 'attack'}, headers=headers).status_code, 200)
        models = importlib.import_module('app.models')
        models.Player.update(shaft_test_whitelisted=False).where(models.Player.player_uid == 'kongmu-elyms-tester').execute()
        self.assertIn('1090', {c['id'] for c in self._get('/api/kongmu/catalog', token=tokens['tester'])['characters']})

    def test_public_plan_preserves_the_twenty_cell_grid(self):
        token = None
        catalog = self._get('/api/kongmu/catalog', token=token)
        character = next(c for c in catalog['characters'] if c['id'] == '1090')
        self.assertEqual(character['name'], '曳尔沐')
        self.assertEqual(character['element'], '光')
        self.assertEqual(character['owner_grid_count'], 3)
        self.assertEqual(character['avatar'], '/static/images/characters/avatar/曳尔沐-square-v2.webp')
        self.assertIn('8%', character['kongmu_passive']['text'])
        self.assertIn('暴击率', character['kongmu_passive']['text'])
        self.assertEqual(self.client.get(character['avatar']).status_code, 200)
        expected = ['11110', '11110', '11011', '01111', '01111']
        for identifier in ('1090', 'char_elyms', '曳尔沐'):
            response = self.client.post('/api/kongmu/plan', json={
                'character_id': identifier, 'cartridge_id': 'attack'},
                headers={})
            self.assertEqual(response.status_code, 200)
            plan = response.get_json()
            slots = plan['character']['equip_slots']['slots']
            self.assertEqual([''.join('1' if c >= 0 else '0' for c in row) for row in slots], expected)
            self.assertEqual(sum(c >= 0 for row in slots for c in row), 20)
            self.assertGreater(plan['result']['total_solution_count'], 0)
            for solution in plan['result']['solutions']:
                self.assertEqual(solution['owner_drive_count'], plan['result']['best_owner_drive_count'])
