import unittest

from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout


class LingkeModulesTest(unittest.TestCase):
    def test_kongmu_catalog_exposes_lingke_and_canhong(self) -> None:
        catalog = get_kongmu_catalog_payload()
        self.assertIn('残虹', [character['name'] for character in catalog['characters']])
        lingke = next(character for character in catalog['characters'] if character['name'] == '灵可')
        self.assertEqual(lingke['id'], '1072')
        self.assertEqual(lingke['element'], '灵')
        self.assertEqual(lingke['owner_grid_count'], 3)
        self.assertEqual(lingke['avatar'], '/static/images/characters/avatar/灵可.png')
        self.assertIn('8%', lingke['kongmu_passive']['text'])
        self.assertIn('暴击率', lingke['kongmu_passive']['text'])

    def test_kongmu_plan_allows_lingke_without_account(self) -> None:
        plan = plan_kongmu_layout('1072', 'attack')
        self.assertEqual(plan['character']['name'], '灵可')
        self.assertEqual(plan['character']['equip_slots']['owner_grid_count'], 3)


if __name__ == '__main__':
    unittest.main()
