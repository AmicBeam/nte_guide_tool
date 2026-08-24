import unittest
from pathlib import Path

from app.modules.shaft.service import normalize_axis_payload
from app.errors import RuleValidationError
from app.modules.shaft.domain.catalog import get_record_map, load_shaft_catalog


ROOT = Path(__file__).resolve().parents[1]
SHAFT_JS = ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft.js'
ADAPTATIONS = {'固态', '液态', '气态', '等离子', '聚合'}
EXPECTED_CHARACTER_ADAPTATIONS = {
    '主角': '固态',
    '真红': '聚合',
    '浔': '固态',
    '小吱': '气态',
    '埃德嘉': '液态',
    '残虹': '气态',
    '灵可': '等离子',
    '伊洛伊': '液态',
    '娜娜莉': '等离子',
    '九原': '固态',
    '薄荷': '液态',
    '白藏': '聚合',
    '早雾': '气态',
    '阿德勒': '聚合',
    '安魂曲': '液态',
    '达芙蒂尔': '液态',
    '法帝娅': '聚合',
    '海月': '等离子',
    '哈尼娅': '固态',
    '卡厄斯': '聚合',
    '哈索尔': '等离子',
    '翳': '气态',
}


class ShaftArcCompatibilityTestCase(unittest.TestCase):
    def test_all_characters_and_arcs_have_known_adaptation(self) -> None:
        catalog = load_shaft_catalog()

        self.assertTrue(catalog['characters'])
        self.assertTrue(catalog['arcs'])
        self.assertTrue(all(character.get('adaptation') in ADAPTATIONS for character in catalog['characters']))
        self.assertTrue(all(arc.get('adaptation') in ADAPTATIONS for arc in catalog['arcs']))
        self.assertEqual(
            {character['name']: character['adaptation'] for character in catalog['characters']},
            EXPECTED_CHARACTER_ADAPTATIONS,
        )
        refinements = catalog['arc_refinements']
        self.assertEqual(refinements['source'], 'nanoka.cc')
        self.assertEqual(set(refinements['arcs']), {arc['id'] for arc in catalog['arcs']})
        self.assertTrue(all(
            set(record['levels']) == {'1', '2', '3', '4', '5'}
            for record in refinements['arcs'].values()
        ))

    def test_starter_build_arcs_match_character_adaptation(self) -> None:
        catalog = load_shaft_catalog()
        characters = get_record_map(catalog['characters'])
        arcs = get_record_map(catalog['arcs'])

        for character_id, raw_build in catalog['starter_axis']['character_builds'].items():
            build = raw_build[0] if isinstance(raw_build, list) else raw_build
            arc = arcs.get(str(build.get('arc_id') or ''))
            if arc:
                self.assertEqual(characters[character_id]['adaptation'], arc['adaptation'])

    def test_cartridge_element_restrictions_and_starter_builds_match(self) -> None:
        catalog = load_shaft_catalog()
        characters = get_record_map(catalog['characters'])
        cartridges = get_record_map(catalog['cartridges'])
        expected_elements = {
            '失落光芒': '光',
            '森林萤火之心': '灵',
            '真红：双生蝶': '咒',
            '迪亚波罗斯': '暗',
            '恶魔之血：诅咒': '魂',
            '街头拳王': '相',
        }

        self.assertFalse(any(cartridge['name'].startswith('【借】') for cartridge in catalog['cartridges']))
        self.assertEqual(
            {
                cartridge['name']: cartridge.get('required_element')
                for cartridge in catalog['cartridges']
                if cartridge.get('required_element')
            },
            expected_elements,
        )
        for character_id, raw_build in catalog['starter_axis']['character_builds'].items():
            build = raw_build[0] if isinstance(raw_build, list) else raw_build
            cartridge = cartridges.get(str(build.get('cartridge_id') or ''))
            if cartridge and cartridge.get('required_element'):
                self.assertEqual(
                    characters[character_id]['element'],
                    cartridge['required_element'],
                    msg=f"{characters[character_id]['name']} 默认卡带属性不匹配",
                )

    def test_backend_keeps_active_cross_element_cartridge(self) -> None:
        normalized = normalize_axis_payload({
            'team': [{
                'slot': 0,
                'character_id': 'char_b52cc8f160',
                'arc_id': '',
                'cartridge_id': 'cartridge_29793225a0',
            }],
            'steps': [],
        })

        self.assertEqual(normalized['team'][0]['cartridge_id'], 'cartridge_29793225a0')

    def test_backend_keeps_inactive_cross_element_cartridge(self) -> None:
        normalized = normalize_axis_payload({
            'team': [{
                'slot': 0,
                'character_id': 'char_b52cc8f160',
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [],
            'character_builds': {
                'char_b2e3b2bf7a': {
                    'character_id': 'char_b2e3b2bf7a',
                    'arc_id': '',
                    'cartridge_id': 'cartridge_f4282fad3f',
                },
            },
        })

        self.assertEqual(
            normalized['character_builds']['char_b2e3b2bf7a']['cartridge_id'],
            'cartridge_f4282fad3f',
        )

    def test_frontend_shows_all_cartridges_without_element_restrictions(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function cartridgesForCharacter()', source)
        self.assertIn('return state.catalog?.cartridges || [];', source)
        self.assertIn('function ensureMemberCompatibleCartridge(member)', source)
        self.assertIn('optionHtml(compatibleCartridges, member.cartridge_id)', source)
        self.assertNotIn('function cartridgeMatchesCharacter(', source)
        self.assertNotIn('（需${cartridge.required_element}属性）', source)
        self.assertGreaterEqual(source.count('ensureMemberCompatibleCartridge(member);'), 2)

    def test_backend_rejects_active_incompatible_arc(self) -> None:
        catalog = load_shaft_catalog()
        axis = dict(catalog['starter_axis'])
        axis.pop('character_builds', None)
        axis['team'] = [dict(member) for member in catalog['starter_axis']['team']]
        member = axis['team'][0]
        character = get_record_map(catalog['characters'])[member['character_id']]
        incompatible_arc = next(
            arc for arc in catalog['arcs']
            if arc['adaptation'] != character['adaptation']
        )
        member['arc_id'] = incompatible_arc['id']

        with self.assertRaisesRegex(RuleValidationError, '适配类型不一致'):
            normalize_axis_payload(axis)

    def test_frontend_filters_arc_options_and_repairs_character_switch(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function arcsForCharacter(characterId)', source)
        self.assertIn("String(arc.adaptation || '') === adaptation", source)
        self.assertIn('optionHtml(compatibleArcs, member.arc_id)', source)
        self.assertIn('ensureMemberCompatibleArc(member);', source)
        self.assertIn('data-field="arc_refinement"', source)
        self.assertNotIn('>当前默认</option>', source)
        self.assertIn('defaultArcRefinement(member.arc_id)', source)

    def test_backend_normalizes_arc_refinement_without_affecting_hash_fields(self) -> None:
        catalog = load_shaft_catalog()
        axis = dict(catalog['starter_axis'])
        axis.pop('character_builds', None)
        axis['team'] = [dict(member) for member in catalog['starter_axis']['team']]
        axis['team'][0]['arc_refinement'] = 9

        normalized = normalize_axis_payload(axis)

        arc_id = normalized['team'][0]['arc_id']
        expected = catalog['arc_refinements']['arcs'][arc_id]['default_level']
        self.assertEqual(normalized['team'][0]['arc_refinement'], expected)
        character_id = normalized['team'][0]['character_id']
        self.assertEqual(normalized['character_builds'][character_id]['arc_refinement'], expected)

    def test_existing_arc_values_define_refinement_defaults(self) -> None:
        catalog = load_shaft_catalog()
        refinements = catalog['arc_refinements']['arcs']

        self.assertEqual(refinements['arc_1a476075cd']['default_level'], 1)
        self.assertEqual(refinements['arc_74578e2ec4']['default_level'], 5)
        self.assertFalse(any(
            arc['name'].endswith(('满', '满精', '白板'))
            for arc in catalog['arcs']
        ))

    def test_legacy_arc_options_migrate_to_refinement_selection(self) -> None:
        catalog = load_shaft_catalog()
        axis = dict(catalog['starter_axis'])
        axis.pop('character_builds', None)
        axis['team'] = [dict(member) for member in catalog['starter_axis']['team']]
        axis['team'][0]['arc_id'] = 'arc_2b6d5881ef'
        axis['team'][0].pop('arc_refinement', None)

        normalized = normalize_axis_payload(axis)

        self.assertEqual(normalized['team'][0]['arc_id'], 'arc_27dc4a7281')
        self.assertEqual(normalized['team'][0]['arc_name'], '穿过胭红蜃景')
        self.assertEqual(normalized['team'][0]['arc_refinement'], 5)


if __name__ == '__main__':
    unittest.main()
