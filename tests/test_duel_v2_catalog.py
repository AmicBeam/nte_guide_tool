import copy
import json
import unittest
from pathlib import Path

from app.modules.card_game.content.duel_v2 import (
    CARDS,
    CHARACTERS,
    STARTER_DECK,
    get_catalog,
    validate_deck,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / 'app' / 'static'
APPROVED_ROWS_PATH = Path('/private/tmp/nte-duel-v2-impl-20260909/approved-card-rows.json')
CORE_DIR = Path('/private/tmp/nte-duel-manual-20260909')

CHARACTER_ORDER = ('nanali', 'zero', 'jiuyuan', 'xun')
CARD_PREFIXES = {'nanali': 'N', 'zero': 'Z', 'jiuyuan': 'J', 'xun': 'X'}
EXPECTED_STARTER_COPIES = 2
STATIC_FILES = {
    'nanali': ('images/characters/avatar/娜娜莉.webp', 'images/characters/portrait/娜娜莉.webp'),
    'zero': ('images/characters/avatar/鉴定师.webp', 'images/characters/portrait/鉴定师.webp'),
    'jiuyuan': ('images/characters/avatar/九原.webp', 'images/characters/portrait/九原.webp'),
    'xun': ('images/characters/avatar/浔.webp', 'images/characters/portrait/浔.webp'),
}


def _approved_rows() -> list[dict]:
    return json.loads(APPROVED_ROWS_PATH.read_text(encoding='utf-8'))


def _core(name: str) -> dict:
    return json.loads((CORE_DIR / name).read_text(encoding='utf-8'))['character']


class DuelV2CatalogTest(unittest.TestCase):
    def test_exports_four_characters_in_fixed_order(self) -> None:
        self.assertEqual(list(CHARACTERS), list(CHARACTER_ORDER))
        catalog = get_catalog()
        self.assertEqual([character['id'] for character in catalog['characters']], list(CHARACTER_ORDER))
        self.assertEqual(CHARACTERS['nanali']['name'], '娜娜莉')
        self.assertEqual(CHARACTERS['zero']['name'], '主角「零」（鉴定师）')
        self.assertEqual(CHARACTERS['jiuyuan']['name'], '九原')
        self.assertEqual(CHARACTERS['xun']['name'], '浔')
        self.assertEqual(CHARACTERS['nanali']['attribute'], '灵')
        self.assertEqual(CHARACTERS['zero']['attribute'], '光')
        self.assertEqual(CHARACTERS['jiuyuan']['attribute'], '灵')
        self.assertEqual(CHARACTERS['xun']['attribute'], '光')
        self.assertEqual((CHARACTERS['nanali']['attack'], CHARACTERS['nanali']['max_hp']), (3, 7))
        self.assertEqual((CHARACTERS['zero']['attack'], CHARACTERS['zero']['max_hp']), (2, 9))
        self.assertEqual((CHARACTERS['jiuyuan']['attack'], CHARACTERS['jiuyuan']['max_hp']), (2, 8))
        self.assertEqual((CHARACTERS['xun']['attack'], CHARACTERS['xun']['max_hp']), (2, 8))
        self.assertIn('获得 1 点成长', CHARACTERS['nanali']['passive'])
        self.assertIn('成长上限提升至 4 点', CHARACTERS['nanali']['awakened_passive'])
        self.assertIn('额外获得 1 点自己的环合值', CHARACTERS['zero']['passive'])
        self.assertIn('若零在备战区且存活', CHARACTERS['zero']['awakened_passive'])
        self.assertIn('创生花', CHARACTERS['jiuyuan']['passive'])
        self.assertIn('额外产生 2 次创生花', CHARACTERS['jiuyuan']['awakened_passive'])
        self.assertIn('记录首张其他己方角色结算的 1 费战斗/战术原牌', CHARACTERS['xun']['passive'])
        self.assertIn('记录容量提升至 2', CHARACTERS['xun']['awakened_passive'])

    def test_cards_keep_thirty_two_official_names_and_four_by_eight_groups(self) -> None:
        approved = {row['id']: row for row in _approved_rows()}
        self.assertEqual(list(CARDS), [row['id'] for row in _approved_rows()])
        self.assertEqual(len(CARDS), 32)
        type_counts = {'battle': 0, 'tactic': 0, 'form': 0}
        for character_id, prefix in CARD_PREFIXES.items():
            owned = [card_id for card_id, card in CARDS.items() if card['character_id'] == character_id]
            self.assertEqual(owned, [f'{prefix}{index:02d}' for index in range(1, 9)])
        for card_id, card in CARDS.items():
            row = approved[card_id]
            self.assertEqual(card['name'], row['name'])
            self.assertEqual(card['type'], row['type'])
            self.assertEqual(card['cost'], row['cost'])
            self.assertEqual(card['terminal'], row['terminal'])
            self.assertEqual(card['description'], row['description'])
            self.assertEqual(card['effect_id'], card_id)
            self.assertIn(card['type'], type_counts)
            type_counts[card['type']] += 1
            self.assertNotIn('starter_copies', card)
        self.assertEqual(sum(type_counts.values()), 32)
        self.assertGreater(type_counts['battle'], 0)
        self.assertGreater(type_counts['tactic'], 0)
        self.assertGreater(type_counts['form'], 0)

    def test_source_keeps_everness_awaken_and_resonance_indices(self) -> None:
        cores = {
            'nanali': _core('nanali-core.json'),
            'zero': _core('zero-core.json'),
            'jiuyuan': _core('jiuyuan-core.json'),
            'xun': _core('xun-core.json'),
        }
        expected_ids = {'nanali': 1010, 'zero': 1051, 'jiuyuan': 1055, 'xun': 1052}
        for character_id, prefix in CARD_PREFIXES.items():
            core = cores[character_id]
            self.assertEqual(len(core['awaken']), 6)
            self.assertEqual(len(core['resonance']), 2)
            for index in range(1, 7):
                card = CARDS[f'{prefix}{index:02d}']
                self.assertEqual(card['source']['character_id'], expected_ids[character_id])
                self.assertEqual(card['source']['type'], 'awaken')
                self.assertEqual(card['source']['index'], index)
                self.assertEqual(card['name'], core['awaken'][index - 1]['name'])
            for index in range(1, 3):
                card = CARDS[f'{prefix}{index + 6:02d}']
                self.assertEqual(card['source']['character_id'], expected_ids[character_id])
                self.assertEqual(card['source']['type'], 'resonance')
                self.assertEqual(card['source']['index'], index)
                self.assertEqual(card['name'], core['resonance'][index - 1]['name'])

    def test_starter_deck_uses_two_copies_of_each_selected_card(self) -> None:
        approved = {row['id']: row for row in _approved_rows()}
        self.assertEqual(STARTER_DECK['id'], 'starter')
        self.assertEqual(STARTER_DECK['name'], '基础预组')
        self.assertEqual(STARTER_DECK['character_ids'], list(CHARACTER_ORDER))
        self.assertEqual(len(STARTER_DECK['card_ids']), 32)
        counts: dict[str, int] = {}
        for card_id in STARTER_DECK['card_ids']:
            counts[card_id] = counts.get(card_id, 0) + 1
        selected = 0
        for card_id, row in approved.items():
            expected = int(row['starter_copies'])
            self.assertEqual(counts.get(card_id, 0), expected)
            if expected:
                self.assertEqual(expected, EXPECTED_STARTER_COPIES)
                selected += 1
        self.assertEqual(selected, 16)
        self.assertEqual(sum(1 for total in counts.values() if total == EXPECTED_STARTER_COPIES), 16)
        normalized = validate_deck(STARTER_DECK)
        self.assertEqual(normalized, STARTER_DECK)
        self.assertIsNot(normalized, STARTER_DECK)
        self.assertIsNot(normalized['card_ids'], STARTER_DECK['card_ids'])
        self.assertIsNot(normalized['character_ids'], STARTER_DECK['character_ids'])

    def test_catalog_returns_independent_copies(self) -> None:
        first = get_catalog()
        second = get_catalog()
        self.assertEqual(first['rules_version'], 'duel_v2')
        self.assertEqual(first['schema_version'], 1)
        self.assertEqual(
            first['deck_rules'],
            {'size': 32, 'per_character': 8, 'max_copies': 2, 'character_count': 4},
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first['characters'][0], CHARACTERS['nanali'])
        self.assertIsNot(first['cards'][0], CARDS['N01'])
        self.assertIsNot(first['starter_deck']['card_ids'], STARTER_DECK['card_ids'])
        first['characters'][0]['name'] = 'changed'
        first['cards'][0]['name'] = 'changed'
        first['starter_deck']['card_ids'].append('N01')
        first['characters'][0]['source'] = {'changed': True}
        restored = get_catalog()
        self.assertEqual(restored['characters'][0]['name'], '娜娜莉')
        self.assertEqual(restored['cards'][0]['name'], '帮派初建成')
        self.assertEqual(len(restored['starter_deck']['card_ids']), 32)
        self.assertEqual(CHARACTERS['nanali']['name'], '娜娜莉')
        self.assertEqual(CARDS['N01']['name'], '帮派初建成')
        self.assertEqual(len(STARTER_DECK['card_ids']), 32)
        self.assertEqual(second['characters'][0]['name'], '娜娜莉')

    def test_existing_character_images_are_present(self) -> None:
        for character_id, (avatar, portrait) in STATIC_FILES.items():
            character = CHARACTERS[character_id]
            self.assertEqual(character['avatar'], f'/static/{avatar}')
            self.assertEqual(character['portrait'], f'/static/{portrait}')
            self.assertTrue((STATIC_ROOT / avatar).is_file(), avatar)
            self.assertTrue((STATIC_ROOT / portrait).is_file(), portrait)

    def test_validate_deck_rejects_unknown_duplicate_and_out_of_range_builds(self) -> None:
        unknown = copy.deepcopy(STARTER_DECK)
        unknown['card_ids'][-1] = 'N99'
        with self.assertRaises(ValueError):
            validate_deck(unknown)
        self.assertEqual(unknown['card_ids'][-1], 'N99')

        duplicate_characters = copy.deepcopy(STARTER_DECK)
        duplicate_characters['character_ids'][-1] = 'nanali'
        with self.assertRaises(ValueError):
            validate_deck(duplicate_characters)

        unknown_character = copy.deepcopy(STARTER_DECK)
        unknown_character['character_ids'][-1] = 'bohe'
        with self.assertRaises(ValueError):
            validate_deck(unknown_character)

        too_many_copies = copy.deepcopy(STARTER_DECK)
        too_many_copies['card_ids'] = ['N01'] * 3 + STARTER_DECK['card_ids'][3:]
        with self.assertRaises(ValueError):
            validate_deck(too_many_copies)
        self.assertEqual(too_many_copies['card_ids'][:3], ['N01', 'N01', 'N01'])

        truncated = copy.deepcopy(STARTER_DECK)
        truncated['card_ids'] = truncated['card_ids'][:-1]
        with self.assertRaises(ValueError):
            validate_deck(truncated)
        self.assertEqual(len(truncated['card_ids']), 31)

        extra = copy.deepcopy(STARTER_DECK)
        extra['card_ids'] = extra['card_ids'] + ['N04']
        with self.assertRaises(ValueError):
            validate_deck(extra)
        self.assertEqual(len(extra['card_ids']), 33)

        unbalanced = copy.deepcopy(STARTER_DECK)
        unbalanced['card_ids'] = STARTER_DECK['card_ids'][:-2] + ['N04', 'N04']
        self.assertEqual(len(unbalanced['card_ids']), 32)
        with self.assertRaises(ValueError):
            validate_deck(unbalanced)

        missing_character = copy.deepcopy(STARTER_DECK)
        missing_character['character_ids'] = ['nanali', 'zero', 'jiuyuan']
        with self.assertRaises(ValueError):
            validate_deck(missing_character)

        extra_character = copy.deepcopy(STARTER_DECK)
        extra_character['character_ids'] = list(CHARACTER_ORDER) + ['nanali']
        with self.assertRaises(ValueError):
            validate_deck(extra_character)

    def test_validate_deck_does_not_mutate_input_or_silently_fix_it(self) -> None:
        payload = copy.deepcopy(STARTER_DECK)
        original_invalid = copy.deepcopy(payload)
        original_invalid['card_ids'][-1] = 'N99'
        payload['card_ids'][-1] = 'N99'
        with self.assertRaises(ValueError):
            validate_deck(payload)
        self.assertEqual(payload, original_invalid)

        valid = copy.deepcopy(STARTER_DECK)
        snapshot = copy.deepcopy(valid)
        normalized = validate_deck(valid)
        valid['card_ids'].append('N04')
        valid['character_ids'].append('nanali')
        valid['name'] = 'changed'
        self.assertEqual(normalized['name'], snapshot['name'])
        self.assertEqual(len(normalized['card_ids']), 32)
        self.assertEqual(normalized['character_ids'], list(CHARACTER_ORDER))
        self.assertIsNot(normalized['card_ids'], valid['card_ids'])
        self.assertNotEqual(valid, snapshot)
        self.assertEqual(validate_deck(snapshot), snapshot)


if __name__ == '__main__':
    unittest.main()
