"""Skill costs follow explicit bindings and survive refreshes and public reads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from flask import Flask
from tests.guide_release_fixtures import published_character_data
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guide_energy_importer', ROOT / 'scripts/guide_skill_details.py')
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


class EnergyCostImportTest(unittest.TestCase):
    def fixture(self, value):
        ability = {'id': 'GA_Test_Ultra', 'phases': [{'name': '终结', 'desc': '终结说明'}],
                   'param_rows': [{'key': 'damage', 'label': '伤害', 'values': ['10%', '20%']}]}
        tables = {'DT_GameplayAbilityTipsData': {'GA_Test_Ultra': {'GALevelUpArray': [{
            'LevelUpData': {'SkillCostName': 'independent_cost'},
            'Desc': {'LocalizedString': '消耗终结能量'},
            'DescValue': {'Key': 'shared_cost_text', 'LocalizedString': '{0}'}}]}},
            'DT_SkillDamageData': {}, 'DT_GameplayAbilityCostData': {'independent_cost': {'GACost': value}}}
        return ability, tables

    def test_reference_not_ability_name_and_repeated_refresh(self):
        ability, tables = self.fixture(100)
        importer.enrich_ability(ability, tables)
        importer.enrich_ability(ability, tables)
        rows = [row for row in ability['param_rows'] if row['key'] == 'shared_cost_text']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['values'], ['100', '100'])
        self.assertIn(rows[0], ability['groups'][0]['param_rows'])
        tables['DT_GameplayAbilityCostData']['independent_cost']['GACost'] = 0
        importer.enrich_ability(ability, tables)
        self.assertEqual(rows[0]['values'], ['0', '0'])

    def test_missing_or_invalid_cost_never_becomes_zero(self):
        for value in (-1, True, float('nan'), float('inf'), '100'):
            ability, tables = self.fixture(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                importer.enrich_ability(ability, tables)
        ability, tables = self.fixture(100)
        tables['DT_GameplayAbilityCostData'] = {}
        with self.assertRaises(KeyError): importer.enrich_ability(ability, tables)

    def test_unknown_city_description_survives_enrichment(self):
        ability = {'id': 'City_Unknown', 'phases': [], 'param_rows': [],
                   'groups': [{'name': '都市技能', 'description': '都市技能效果暂未提供。', 'param_rows': []}]}
        expected = deepcopy(ability)
        importer.enrich_ability(ability, {'DT_GameplayAbilityTipsData': {}, 'DT_SkillDamageData': {}})
        self.assertEqual(ability, expected)


@published_character_data
class PublishedEnergyCostTest(unittest.TestCase):
    def test_elyms_energy_cost_in_page_json_markdown_and_level_view(self):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        app.add_url_rule('/login', endpoint='main.login_page', view_func=lambda: 'login')
        client = app.test_client()
        ability = next(a for a in service.detail('characters', '1090')['abilities'] if a['id'] == 'GA_Elyms_UltraSkill')
        cost = next(row for row in ability['param_rows'] if row['label'] == '消耗终结能量')
        self.assertEqual(cost['values'], ['100'] * 13)
        for path in ('/guide/characters/1090', '/guide/characters/1090.md',
                     '/api/guide/characters/1090?ability_id=GA_Elyms_UltraSkill&level=10'):
            response = client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn('消耗终结能量', json.dumps(response.json, ensure_ascii=False) if response.is_json else response.text)
        self.assertEqual(service.detail('characters', 'preview-exe')['id'], '1074')
        self.assertEqual(service.detail('characters', 'preview-nitsa')['status'], 'preview')
