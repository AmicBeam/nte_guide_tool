import copy
import json
import re
import unittest
from pathlib import Path

from app.modules.card_game.content.duel_v2 import (
    CARDS,
    CHARACTERS,
    STARTER_DECK,
    STARTER_DECKS,
    get_catalog,
    validate_deck,
)
from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER as CATALOG_ORDER


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / 'app' / 'static'
APPROVED_ROWS_PATH = Path('/private/tmp/nte-duel-v2-impl-20260909/approved-card-rows.json')
CORE_DIR = Path('/private/tmp/nte-duel-manual-20260909')

STARTER_CHARACTERS = ('nanali', 'iloy', 'zero', 'jiuyuan')
ATTRIBUTE_ORDER = ('光', '灵', '相', '暗', '咒', '魂')
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
        self.assertEqual(list(CHARACTERS), list(CATALOG_ORDER))
        self.assertGreaterEqual(len(CATALOG_ORDER), 11)
        catalog = get_catalog()
        self.assertEqual([character['id'] for character in catalog['characters']], list(CATALOG_ORDER))
        self.assertEqual(
            [CHARACTERS[character_id]['attribute'] for character_id in CATALOG_ORDER],
            sorted((CHARACTERS[character_id]['attribute'] for character_id in CATALOG_ORDER),
                   key=lambda item: ATTRIBUTE_ORDER.index(item) if item in ATTRIBUTE_ORDER else 99),
        )
        self.assertEqual(CHARACTERS['nanali']['name'], '娜娜莉')
        self.assertEqual(CHARACTERS['zero']['name'], '零')
        self.assertEqual(CHARACTERS['jiuyuan']['name'], '九原')
        self.assertEqual(CHARACTERS['xun']['name'], '浔')
        self.assertEqual(CHARACTERS['nanali']['attribute'], '灵')
        self.assertEqual(CHARACTERS['zero']['attribute'], '光')
        self.assertEqual(CHARACTERS['jiuyuan']['attribute'], '灵')
        self.assertEqual(CHARACTERS['xun']['attribute'], '光')
        self.assertEqual(CHARACTERS['nanali']['energy_max'], 4)
        self.assertEqual(CHARACTERS['zero']['energy_max'], 4)
        self.assertEqual(CHARACTERS['hathor']['energy_max'], 4)
        self.assertEqual(CHARACTERS['anhunqu']['energy_max'], 5)
        self.assertEqual(CHARACTERS['canhong']['energy_max'], 5)
        self.assertEqual(CHARACTERS['zaowu']['energy_max'], 5)
        self.assertEqual((CHARACTERS['nanali']['attack'], CHARACTERS['nanali']['max_hp']), (2, 5))
        self.assertEqual((CHARACTERS['zero']['attack'], CHARACTERS['zero']['max_hp']), (2, 5))
        self.assertEqual((CHARACTERS['jiuyuan']['attack'], CHARACTERS['jiuyuan']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['canhong']['attack'], CHARACTERS['canhong']['max_hp']), (2, 5))
        self.assertEqual((CHARACTERS['zaowu']['attack'], CHARACTERS['zaowu']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['lingke']['attack'], CHARACTERS['lingke']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['xun']['attack'], CHARACTERS['xun']['max_hp']), (2, 5))
        self.assertIn('家族壮大', CHARACTERS['nanali']['passive'])
        self.assertIn('追击', CHARACTERS['nanali']['awakened_passive'])
        self.assertIn('充满自己的环合值', CHARACTERS['zero']['passive'])
        self.assertIn('己方武备牌', CHARACTERS['zero']['awakened_passive'])
        self.assertEqual(CHARACTERS['nanali']['ultimate']['kind'], 'status')
        self.assertEqual(CHARACTERS['nanali']['ultimate']['turns'], 2)
        self.assertIn('创生强化', CHARACTERS['jiuyuan']['passive'])
        self.assertIn('枚约', CHARACTERS['jiuyuan']['awakened_passive'])
        self.assertIn('前排优先', CHARACTERS['lingke']['passive'])
        self.assertIn('延滞', CHARACTERS['hathor']['passive'])
        self.assertIn('记录', CHARACTERS['xun']['passive'])
        self.assertIn('记录容量', CHARACTERS['xun']['awakened_passive'])

    def test_cards_keep_thirty_two_official_names_and_four_by_eight_groups(self) -> None:
        named = {card_id: card for card_id, card in CARDS.items() if not card.get('derived')}
        self.assertEqual(len(named), 88)
        self.assertTrue(CARDS['NF01']['derived'])
        self.assertEqual(CARDS['NF01']['name'], '家族壮大')
        type_counts = {'battle': 0, 'tactic': 0, 'form': 0}
        for character_id in CATALOG_ORDER:
            owned = [card_id for card_id, card in named.items() if card['character_id'] == character_id]
            self.assertEqual(len(owned), 8)
        for character_id, prefix in CARD_PREFIXES.items():
            owned = [card_id for card_id, card in named.items() if card['character_id'] == character_id]
            self.assertEqual(owned, [f'{prefix}{index:02d}' for index in range(1, 9)])
        for card_id, card in CARDS.items():
            self.assertEqual(card['effect_id'], card_id)
            self.assertIn(card['type'], type_counts)
            type_counts[card['type']] += 1
        self.assertEqual(sum(type_counts.values()), len(CARDS))
        self.assertGreater(type_counts['battle'], 0)
        self.assertGreater(type_counts['tactic'], 0)
        self.assertGreater(type_counts['form'], 0)
        for card in CARDS.values():
            self.assertEqual(card['cost'], 1)
            if card['type'] == 'form':
                self.assertIsInstance(card.get('attack'), int)
                self.assertIsInstance(card.get('hp'), int)
                self.assertGreaterEqual(card['attack'], 0)
                self.assertGreaterEqual(card['hp'], 0)
            elif card['type'] == 'battle':
                self.assertIsInstance(card.get('attack'), int)
                self.assertIsInstance(card.get('shield'), int)
                self.assertGreaterEqual(card['attack'], 0)
                self.assertGreaterEqual(card['shield'], 0)
        self.assertEqual(CARDS['N08']['attack'], 2)
        self.assertEqual(CARDS['N08']['hp'], 6)
        self.assertEqual(CARDS['N04']['shield'], 2)
        self.assertEqual(CARDS['N03']['attack'], 1)
        self.assertTrue(CARDS['N06'].get('playable_downed'))
        self.assertIn('倒地时也能打出', CARDS['N06']['description'])
        self.assertEqual(CARDS['N02']['response'], 'ally_attacked')
        self.assertEqual(CARDS['N02']['shield'], 2)
        self.assertEqual(CARDS['N02']['attack'], 0)
        self.assertTrue(CARDS['N01']['instant'])
        self.assertTrue(CARDS['N01']['description'].startswith('瞬发。'))
        self.assertTrue(CARDS['N02']['description'].startswith('响应。'))
        self.assertEqual(CARDS['N03']['description'], '额外造成 1 点追击伤害。')
        self.assertEqual(CARDS['N04']['description'], '')
        names = '|'.join(re.escape(character['name']) for character in CHARACTERS.values())
        sortie = re.compile(rf'(?:^|(?<=。))(?:{names})出击。')
        for card in CARDS.values():
            if card['type'] == 'battle':
                self.assertIsNone(sortie.search(card['description'] or ''), card['id'])
        self.assertEqual(CARDS['N07']['description'], '本局每使用过一次「家族壮大」，娜娜莉的战斗牌额外获得攻击 +1、护盾 +1。')
        self.assertEqual(CARDS['Z08']['name'], '倾世之雨')
        self.assertEqual(CARDS['Z07']['name'], '休息日')
        self.assertEqual(CARDS['Z06']['type'], 'form')
        self.assertEqual(CARDS['Z03']['shield'], 2)
        self.assertTrue(CARDS['X05']['redeem'])
        self.assertTrue(CARDS['X06']['redeem'])
        self.assertNotIn('redeem', CARDS['X03'])
        self.assertEqual(CARDS['N07']['type'], 'form')
        self.assertEqual(CARDS['N08']['type'], 'form')
        self.assertEqual(CARDS['N07']['name'], '心有猛虎')
        self.assertEqual(CARDS['N08']['name'], '预备备')
        for character_id, prefix in CARD_PREFIXES.items():
            self.assertEqual(CARDS[f'{prefix}08']['type'], 'form', character_id)
        for character_id in CATALOG_ORDER:
            owned = [card_id for card_id, card in named.items() if card['character_id'] == character_id]
            card_07, card_08 = CARDS[owned[6]], CARDS[owned[7]]
            self.assertEqual(card_08['source']['type'], 'arc', character_id)
            if character_id == 'zero':
                self.assertEqual(card_07['source']['type'], 'arc', character_id)
            else:
                self.assertEqual(card_07['source']['type'], 'ability', character_id)
        if APPROVED_ROWS_PATH.is_file():
            approved = {row['id']: row for row in _approved_rows()}
            for card_id, row in approved.items():
                card = CARDS[card_id]
                self.assertEqual(card['name'], row['name'])

    def test_source_keeps_everness_awaken_and_arc_ability_slots(self) -> None:
        if not CORE_DIR.is_dir():
            self.skipTest('everness core snapshot not present')
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
            card_07 = CARDS[f'{prefix}07']
            card_08 = CARDS[f'{prefix}08']
            self.assertEqual(card_07['source']['character_id'], expected_ids[character_id])
            self.assertEqual(card_08['source']['character_id'], expected_ids[character_id])
            self.assertEqual(card_08['source']['type'], 'arc')
            if character_id == 'zero':
                self.assertEqual(card_07['source']['type'], 'arc')
            else:
                self.assertEqual(card_07['source']['type'], 'ability')
            self.assertTrue(str(card_08['name']))
            if character_id not in ('zero', 'nanali', 'jiuyuan'):
                self.assertTrue(str(card_07['name']).endswith('之绵'))

    def test_starter_deck_uses_two_copies_of_each_selected_card(self) -> None:
        self.assertEqual(STARTER_DECK['id'], 'starter')
        self.assertEqual(STARTER_DECK['name'], '创生预组')
        self.assertEqual(STARTER_DECK['character_ids'], list(STARTER_CHARACTERS))
        self.assertEqual(len(STARTER_DECK['card_ids']), 32)
        counts: dict[str, int] = {}
        for card_id in STARTER_DECK['card_ids']:
            counts[card_id] = counts.get(card_id, 0) + 1
        selected = 0
        for card_id, total in counts.items():
            self.assertIn(total, (1, EXPECTED_STARTER_COPIES), card_id)
            selected += 1
        self.assertEqual(sum(counts.values()), 32)
        self.assertGreaterEqual(selected, 16)
        for deck in STARTER_DECKS:
            self.assertEqual(validate_deck(deck)['card_ids'], deck['card_ids'])
        self.assertEqual(sum(1 for total in counts.values() if total == 1), 2)
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
        self.assertEqual(restored['characters'][0]['name'], CHARACTERS[CATALOG_ORDER[0]]['name'])
        self.assertEqual(restored['cards'][0]['name'], '帮派初建成')
        self.assertEqual(len(restored['starter_deck']['card_ids']), 32)
        self.assertEqual(CHARACTERS['nanali']['name'], '娜娜莉')
        self.assertEqual(CARDS['N01']['name'], '帮派初建成')
        self.assertEqual(len(STARTER_DECK['card_ids']), 32)
        self.assertEqual(second['characters'][0]['name'], CHARACTERS[CATALOG_ORDER[0]]['name'])

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
        extra_character['character_ids'] = list(STARTER_CHARACTERS) + ['nanali']
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
        self.assertEqual(normalized['character_ids'], list(STARTER_CHARACTERS))
        self.assertIsNot(normalized['card_ids'], valid['card_ids'])
        self.assertNotEqual(valid, snapshot)
        self.assertEqual(validate_deck(snapshot), snapshot)


if __name__ == '__main__':
    unittest.main()
