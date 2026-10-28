from tests.test_solo_room_flow import RoomFlowTestCase


class ElymsKongmuTest(RoomFlowTestCase):
    def test_public_catalog_and_plan_preserve_the_twenty_cell_grid(self):
        catalog = self.client.get('/api/kongmu/catalog').get_json()
        character = next(c for c in catalog['characters'] if c['id'] == 'char_elyms')
        self.assertEqual(character['name'], '曳尔沐')
        self.assertEqual(character['element'], '光')
        self.assertEqual(character['owner_grid_count'], 3)
        self.assertEqual(character['avatar'], '/static/guide/character-previews/elyms-avatar.webp')
        self.assertIn('8%', character['kongmu_passive']['text'])
        self.assertIn('暴击率', character['kongmu_passive']['text'])
        self.assertEqual(self.client.get(character['avatar']).status_code, 200)
        regular = self._issue_login_and_get_token('kongmu-elyms-regular')
        expected = ['11110', '11110', '11011', '01111', '01111']
        for identifier, headers in [('char_elyms', {}), ('曳尔沐', {'Authorization': f'Bearer {regular}'})]:
            response = self.client.post('/api/kongmu/plan', json={
                'character_id': identifier, 'cartridge_id': 'attack'}, headers=headers)
            self.assertEqual(response.status_code, 200)
            plan = response.get_json()
            slots = plan['character']['equip_slots']['slots']
            self.assertEqual([''.join('1' if c >= 0 else '0' for c in row) for row in slots], expected)
            self.assertEqual(sum(c >= 0 for row in slots for c in row), 20)
            self.assertGreater(plan['result']['total_solution_count'], 0)
            for solution in plan['result']['solutions']:
                # The solver's own placement validation is also covered by the shared module tests.
                self.assertEqual(solution.get('owner_drive_count'), plan['result']['best_owner_drive_count'])
