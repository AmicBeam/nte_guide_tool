import unittest

from app.errors import AppError
from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout


class LingkeModulesTest(unittest.TestCase):
    def test_kongmu_catalog_keeps_lingke_test_only_and_canhong_public(self) -> None:
        public_catalog = get_kongmu_catalog_payload()
        test_catalog = get_kongmu_catalog_payload(include_test_characters=True)

        self.assertIn('残虹', [character['name'] for character in public_catalog['characters']])
        self.assertNotIn('灵可', [character['name'] for character in public_catalog['characters']])
        lingke = next(character for character in test_catalog['characters'] if character['name'] == '灵可')
        self.assertEqual(lingke['id'], '1072')
        self.assertEqual(lingke['element'], '灵')
        self.assertEqual(lingke['owner_grid_count'], 3)
        self.assertEqual(lingke['avatar'], '/static/images/characters/avatar/灵可.png')
        self.assertIn('8%', lingke['kongmu_passive']['text'])
        self.assertIn('暴击率', lingke['kongmu_passive']['text'])

    def test_kongmu_plan_requires_test_access_for_lingke(self) -> None:
        with self.assertRaises(AppError):
            plan_kongmu_layout('1072', 'attack')

        plan = plan_kongmu_layout('1072', 'attack', include_test_characters=True)
        self.assertEqual(plan['character']['name'], '灵可')
        self.assertEqual(plan['character']['equip_slots']['owner_grid_count'], 3)


if __name__ == '__main__':
    unittest.main()
