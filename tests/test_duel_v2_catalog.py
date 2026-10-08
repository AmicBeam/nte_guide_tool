import copy
import re
import unittest
from pathlib import Path

from app.modules.card_game.content.duel_v2 import (
    CARDS,
    CHARACTERS,
    PUBLIC_CHARACTER_IDS,
    STARTER_DECK,
    STARTER_DECKS,
    deck_uses_test_characters,
    get_catalog,
    validate_deck,
)
from app.modules.card_game.content.duel_v2.catalog import (
    CHARACTER_ORDER as CATALOG_ORDER,
    DARK_ENERGY_MAX,
    LIGHT_ENERGY_MAX,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / 'app' / 'static'

STARTER_CHARACTERS = ('nanali', 'iloy', 'zero', 'jiuyuan')
ATTRIBUTE_ORDER = ('光', '灵', '相', '暗', '咒', '魂')
CARD_PREFIXES = {'nanali': 'N', 'zero': 'Z', 'jiuyuan': 'J', 'xun': 'X'}
EXPECTED_STARTER_COPIES = 2
STATIC_FILES = {
    'nanali': ('images/characters/avatar/娜娜莉.webp', 'images/characters/portrait/娜娜莉.webp'),
    'zero': ('images/characters/avatar/鉴定师.webp', 'images/characters/portrait/鉴定师.webp'),
    'jiuyuan': ('images/characters/avatar/九原.webp', 'images/characters/portrait/九原.webp'),
    'xun': ('images/characters/avatar/浔.webp', 'images/characters/portrait/浔.webp'),
    'bohe': ('images/characters/avatar/薄荷.webp', 'images/characters/portrait/薄荷.webp'),
    'baicang': ('images/characters/avatar/白藏.webp', 'images/characters/portrait/白藏.webp'),
}


class DuelV2CatalogTest(unittest.TestCase):
    def test_character_mechanisms_are_reference_cards_outside_the_build_pool(self):
        catalog = get_catalog(include_test_characters=False)
        characters = {c['id']: c for c in catalog['characters']}
        expected = {'zhenhong': '独行', 'anhunqu': '噩梦', 'canhong': '蚀心与鸩火', 'adler': '诛恶护持'}
        for owner, name in expected.items():
            card, = characters[owner]['mechanisms']
            self.assertEqual((card['name'], card['type'], card['character_id']), (name, 'mechanism', owner))
            self.assertNotIn(card['id'], CARDS)
            self.assertNotIn(card['id'], {c['id'] for c in catalog['cards']})
            bad = copy.deepcopy(STARTER_DECKS[-1])
            bad['card_ids'][0] = card['id']
            with self.assertRaises(ValueError):
                validate_deck(bad)
        self.assertEqual(characters['canhong']['mechanisms'][0]['reference_names'], ['蚀心', '鸩火'])
        self.assertEqual(characters['adler']['awakened_passive'],
                         '施加「诛恶护持」，持续 2 个己方回合。移除所有对方角色的护盾。')
        catalog['characters'][0]['passive'] = 'mutated'
        self.assertNotEqual(get_catalog()['characters'][0]['passive'], 'mutated')

    def test_canhong_passive_describes_initial_harmony_without_dot_attack(self):
        passive = next(c for c in get_catalog()['characters'] if c['id'] == 'canhong')['passive']
        self.assertTrue(passive.startswith('初始具有 2 点环合值。'))
        self.assertIn('重复触发时可叠层，上限 3 层。', passive)
        self.assertIn('幻境', passive)
        self.assertNotIn('每回合 1 次', passive)
        self.assertNotIn('攻击 +1', passive)

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
        self.assertEqual(LIGHT_ENERGY_MAX, 5)
        self.assertEqual(DARK_ENERGY_MAX, 6)
        self.assertEqual(CHARACTERS['nanali']['energy_max'], LIGHT_ENERGY_MAX)
        self.assertEqual(CHARACTERS['zero']['energy_max'], LIGHT_ENERGY_MAX)
        self.assertEqual(CHARACTERS['hathor']['energy_max'], LIGHT_ENERGY_MAX)
        self.assertEqual(CHARACTERS['anhunqu']['energy_max'], DARK_ENERGY_MAX)
        self.assertEqual(CHARACTERS['canhong']['energy_max'], DARK_ENERGY_MAX)
        self.assertEqual(CHARACTERS['zaowu']['energy_max'], DARK_ENERGY_MAX)
        self.assertEqual((CHARACTERS['nanali']['attack'], CHARACTERS['nanali']['max_hp']), (2, 5))
        self.assertEqual((CHARACTERS['zero']['attack'], CHARACTERS['zero']['max_hp']), (2, 5))
        self.assertEqual((CHARACTERS['jiuyuan']['attack'], CHARACTERS['jiuyuan']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['canhong']['attack'], CHARACTERS['canhong']['max_hp']), (2, 6))
        self.assertEqual((CHARACTERS['zaowu']['attack'], CHARACTERS['zaowu']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['lingke']['attack'], CHARACTERS['lingke']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['xun']['attack'], CHARACTERS['xun']['max_hp']), (3, 6))
        self.assertEqual((CHARACTERS['bohe']['attack'], CHARACTERS['bohe']['max_hp']), (3, 4))
        self.assertEqual((CHARACTERS['baicang']['attack'], CHARACTERS['baicang']['max_hp']), (2, 6))
        self.assertEqual(CHARACTERS['bohe']['energy_max'], LIGHT_ENERGY_MAX)
        self.assertEqual(CHARACTERS['baicang']['energy_max'], DARK_ENERGY_MAX)
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
        self.assertIn('溢出伤害', CHARACTERS['bohe']['passive'])
        self.assertIn('穿透', CHARACTERS['bohe']['passive'])
        self.assertEqual(CHARACTERS['bohe']['passive'], '薄荷战斗造成的溢出伤害穿透至玩家。')
        self.assertIn('免疫对方的非战斗伤害', CHARACTERS['bohe']['awakened_passive'])
        self.assertIn('受到伤害', CHARACTERS['baicang']['passive'])
        self.assertIn('消灭', CHARACTERS['baicang']['awakened_passive'])
        self.assertIn('荒时', CHARACTERS['xun']['passive'])
        self.assertIn('时计', CHARACTERS['xun']['awakened_passive'])
        self.assertEqual(PUBLIC_CHARACTER_IDS, frozenset(('nanali', 'zero', 'jiuyuan', 'iloy', 'bohe', 'baicang', 'xiaozhi', 'haiyue', 'zhenhong', 'yi', 'anhunqu', 'canhong', 'zaowu', 'adler')))
        access = {item['id']: item['access_level'] for item in catalog['characters']}
        self.assertEqual(access['nanali'], 'public')
        self.assertEqual(access['zero'], 'public')
        self.assertEqual(access['jiuyuan'], 'public')
        self.assertEqual(access['iloy'], 'public')
        self.assertEqual(access['xun'], 'test')
        self.assertEqual(access['canhong'], 'public')
        self.assertEqual(access['bohe'], 'public')
        self.assertEqual(access['baicang'], 'public')

    def test_cards_keep_thirty_two_official_names_and_four_by_eight_groups(self) -> None:
        named = {card_id: card for card_id, card in CARDS.items() if not card.get('derived')}
        self.assertEqual(len(named), 152)
        self.assertTrue(CARDS['NF01']['derived'])
        self.assertEqual(CARDS['NF01']['name'], '家族壮大')
        self.assertIn('第 5 回合及以后具有瞬发', CARDS['NF01']['description'])
        self.assertFalse(CARDS['NF01'].get('instant'))
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
            self.assertEqual(card['cost'], 0 if card['id'] in ('I04', 'A03') or (card['character_id'] == 'xun' and card['id'] not in ('X08', 'XF01')) else 1)
            if card['type'] == 'form':
                self.assertIsInstance(card.get('attack'), int)
                self.assertIsInstance(card.get('hp'), int)
                self.assertGreaterEqual(card['attack'], 0)
                self.assertGreaterEqual(card['hp'], 0)
            elif card['type'] == 'battle':
                self.assertIsInstance(card.get('attack'), int)
                self.assertIsInstance(card.get('shield'), int)
                self.assertGreaterEqual(card['attack'], {'A06': -2, 'R02': -1}.get(card['id'], 0))
                self.assertGreaterEqual(card['shield'], 0)
        self.assertEqual(CARDS['N08']['attack'], 2)
        self.assertEqual(CARDS['N08']['hp'], 6)
        self.assertEqual(CARDS['N04']['shield'], 2)
        self.assertEqual(CARDS['N03']['attack'], 1)
        self.assertTrue(CARDS['N06'].get('instant'))
        self.assertTrue(CARDS['N06']['description'].startswith('瞬发。'))
        self.assertTrue(CARDS['N06'].get('playable_downed'))
        self.assertIn('倒地时也能打出', CARDS['N06']['description'])
        self.assertEqual(CARDS['N02']['response'], 'ally_attacked')
        self.assertEqual(CARDS['N02']['shield'], 2)
        self.assertEqual(CARDS['N02']['attack'], 0)
        self.assertFalse(CARDS['N01']['instant'])
        self.assertEqual(CARDS['N01']['description'], '回满娜娜莉的生命。')
        self.assertIn('响应：', CARDS['N02']['description'])
        self.assertTrue(CARDS['B01']['instant'])
        self.assertTrue(CARDS['B01']['description'].startswith('瞬发。'))
        self.assertTrue(CARDS['B07']['instant'])
        self.assertTrue(CARDS['B07']['description'].startswith('瞬发。'))
        self.assertEqual(CARDS['B05']['response'], 'self_lethal')
        self.assertIn('响应：', CARDS['B05']['description'])
        self.assertEqual(CARDS['M04']['response'], 'self_attacked')
        self.assertIn('响应：', CARDS['M04']['description'])
        self.assertEqual(CARDS['B03']['require'], 'own_burn')
        self.assertEqual(CARDS['M07']['name'], '美好的一天从睡醒开始')
        self.assertEqual(CARDS['M08']['name'], '开始净空')
        self.assertEqual(CARDS['B07']['name'], '判予秋')
        self.assertEqual(CARDS['B08']['name'], '茶花会')
        self.assertEqual(CARDS['B07']['attack'], 2)
        # 2026-09-20 用户指定桌游面板，不是 Everness 原作数值。
        self.assertEqual(CARDS['B08']['attack'], 2)
        self.assertEqual(CARDS['B06']['description'], '若本回合白藏受到过己方伤害，额外 +1 攻击。')
        self.assertEqual(CARDS['M06']['description'], '对对方前排造成 2 点伤害。薄荷的永久攻击每比 3 高 1 点，此伤害 +1。')
        self.assertEqual(CARDS['Y06']['description'], '复活 1 名己方异能者。')
        rush = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        self.assertEqual(
            sorted(set(rush['card_ids'])),
            ['B01', 'B02', 'B06', 'B07', 'B08', 'M01', 'M03', 'M04', 'M07', 'Y02', 'Y03', 'Y04', 'Y06', 'Y07', 'Z03', 'Z04', 'Z05', 'Z08'],
        )
        self.assertTrue(all(1 <= rush['card_ids'].count(card_id) <= 2 for card_id in set(rush['card_ids'])))
        self.assertEqual(CARDS['N02']['description'], '响应：己方其他异能者被攻击时，自动使用。')
        self.assertEqual(CARDS['N03']['description'], '额外造成 1 点追击伤害。')
        self.assertEqual(CARDS['N04']['description'], '本回合己方创生伤害 +1。')
        names = '|'.join(re.escape(character['name']) for character in CHARACTERS.values())
        sortie = re.compile(rf'(?:^|(?<=。))(?:{names})出击。')
        for card in CARDS.values():
            if card['type'] == 'battle':
                self.assertIsNone(sortie.search(card['description'] or ''), card['id'])
        self.assertEqual(CARDS['N07']['description'], '本局每使用过一次「家族壮大」，娜娜莉的战斗牌额外获得攻击 +1、护盾 +1。')
        self.assertEqual(CARDS['Z08']['name'], '倾世之雨')
        self.assertEqual(CARDS['Z08']['description'], '回合结束时，己方前排攻击 +1。')
        self.assertEqual(CARDS['Z02']['description'], '从牌库将 2 张武备牌加入手牌。')
        self.assertEqual(CHARACTERS['zero']['passive'], '零使用战斗牌后，立刻充满自己的环合值。零已装备武备时，零的武备牌获得瞬发，使用时抽 1 张牌。')
        self.assertEqual(CHARACTERS['zero']['awakened_passive'], '己方武备牌变为瞬发，使用时抽 1 张牌，持续 2 个己方回合。')
        self.assertEqual(CARDS['Z07']['name'], '休息日')
        self.assertEqual(CARDS['Z06']['type'], 'form')
        self.assertEqual(CARDS['Z03']['shield'], 2)
        self.assertNotIn('redeem', CARDS['X05'])
        self.assertNotIn('redeem', CARDS['X06'])
        self.assertNotIn('redeem', CARDS['X03'])
        self.assertEqual(CARDS['N07']['type'], 'form')
        self.assertEqual(CARDS['N08']['type'], 'form')
        self.assertEqual(CARDS['N07']['name'], '心有猛虎')
        self.assertEqual(CARDS['N08']['name'], '预备备')
        self.assertEqual(CARDS['S08']['name'], '好狗狗走四方')
        self.assertEqual(CARDS['S08']['source']['type'], 'arc')
        self.assertEqual(CARDS['X07']['name'], '壶中往昔')
        self.assertEqual(CARDS['A07']['name'], '广告拍摄中！')
        self.assertEqual(CARDS['C07']['name'], '惑心谲影')
        self.assertEqual(CARDS['K07']['name'], '宇宙的频段')
        self.assertEqual(CARDS['R07']['name'], '梦的边缘')
        self.assertEqual(CARDS['Y07']['name'], '想做什么梦？')
        self.assertEqual(CARDS['Y07']['attack'], 1)
        self.assertEqual(CARDS['M08']['hp'], 4)
        self.assertEqual(CARDS['B08']['hp'], 9)
        self.assertEqual(CARDS['J07']['description'], '九原的终结成功清算枚约时，每个目标使九原获得 1 点能量。')
        self.assertEqual(CARDS['J07']['hp'], 5)
        self.assertIn('伊洛伊永久获得攻击 +1、生命 +1', CARDS['Y05']['description'])
        self.assertIn('第 5 回合', CARDS['NF01']['description'])
        self.assertEqual(CARDS['S07']['name'], '小鬼头大胃口')
        self.assertEqual(CARDS['H07']['name'], '铁骑速递')
        for character_id, prefix in CARD_PREFIXES.items():
            self.assertEqual(CARDS[f'{prefix}08']['type'], 'form', character_id)
        for character_id in CATALOG_ORDER:
            owned = [card_id for card_id, card in named.items() if card['character_id'] == character_id]
            card_07, card_08 = CARDS[owned[6]], CARDS[owned[7]]
            self.assertEqual(card_08['source']['type'], 'ability' if character_id in ('yi', 'adler') else 'arc', character_id)
            if character_id == 'zero':
                self.assertEqual(card_07['source']['type'], 'arc', character_id)
            elif character_id in ('zaowu', 'hathor', 'jiuyuan', 'baicang', 'xiaozhi', 'haiyue', 'edgar', 'haniya', 'yi', 'adler'):
                self.assertEqual(card_07['source']['type'], 'ability', character_id)
            else:
                self.assertEqual(card_07['source']['type'], 'pv', character_id)

    def test_source_metadata_keeps_character_ids_and_naming_slots(self) -> None:
        # Offline metadata contract; official-name verification belongs to the
        # source review documented in docs/everness-item-chain-card-design.md.
        expected_ids = {'nanali': 1010, 'zero': 1051, 'jiuyuan': 1055, 'xun': 1052}
        for character_id, prefix in CARD_PREFIXES.items():
            for index in range(1, 7):
                card = CARDS[f'{prefix}{index:02d}']
                self.assertEqual(card['source']['character_id'], expected_ids[character_id])
                self.assertEqual(card['source']['type'], 'awaken')
                self.assertEqual(card['source']['index'], index)
            card_07 = CARDS[f'{prefix}07']
            card_08 = CARDS[f'{prefix}08']
            self.assertEqual(card_07['source']['character_id'], expected_ids[character_id])
            self.assertEqual(card_08['source']['character_id'], expected_ids[character_id])
            self.assertEqual(card_08['source']['type'], 'arc')
            if character_id == 'zero':
                self.assertEqual(card_07['source']['type'], 'arc')
            elif character_id in ('jiuyuan',):
                self.assertEqual(card_07['source']['type'], 'ability')
            else:
                self.assertEqual(card_07['source']['type'], 'pv')
            self.assertTrue(str(card_08['name']))
            if character_id == 'xun':
                self.assertEqual(card_07['name'], '壶中往昔')

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
        self.assertEqual(sum(1 for total in counts.values() if total == 1), 8)
        normalized = validate_deck(STARTER_DECK)
        self.assertEqual(normalized, STARTER_DECK)
        self.assertIsNot(normalized, STARTER_DECK)
        self.assertIsNot(normalized['card_ids'], STARTER_DECK['card_ids'])
        self.assertIsNot(normalized['character_ids'], STARTER_DECK['character_ids'])

    def test_zhenhong_preset_is_separate_legal_test_deck(self):
        from collections import Counter
        deck = next(d for d in STARTER_DECKS if d['id'] == 'zhenhong')
        self.assertEqual(deck['name'], '真红预组')
        self.assertEqual(deck['character_ids'], ['zhenhong', 'zero', 'iloy', 'yi'])
        self.assertEqual(len(deck['card_ids']), 32)
        self.assertEqual(Counter(CARDS[c]['character_id'] for c in deck['card_ids']),
                         Counter(dict.fromkeys(deck['character_ids'], 8)))
        self.assertLessEqual(max(Counter(deck['card_ids']).values()), 2)
        self.assertFalse(any(CARDS[c].get('derived') for c in deck['card_ids']))
        self.assertEqual(Counter(c for c in deck['card_ids'] if CARDS[c]['character_id'] == 'yi'),
                         Counter({'I01': 2, 'I03': 2, 'I04': 2, 'I08': 2}))
        self.assertNotIn('crimson', {d['id'] for d in STARTER_DECKS})

    def test_murk_preset_is_public(self):
        from collections import Counter
        deck = next(d for d in STARTER_DECKS if d['id'] == 'murk')
        self.assertEqual(deck['name'], '浊燃预组')
        self.assertEqual(deck['character_ids'], ['anhunqu', 'canhong', 'zaowu', 'adler'])
        self.assertEqual(len(deck['card_ids']), 32)
        self.assertEqual(Counter(CARDS[c]['character_id'] for c in deck['card_ids']),
                         Counter(dict.fromkeys(deck['character_ids'], 8)))
        self.assertTrue(all(count == 2 for count in Counter(deck['card_ids']).values()))
        self.assertEqual(len(Counter(deck['card_ids'])), 16)
        self.assertFalse(any(CARDS[c].get('derived') for c in deck['card_ids']))
        self.assertFalse(deck_uses_test_characters(deck))

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

    def test_public_catalog_hides_test_characters_and_their_presets(self) -> None:
        public = get_catalog(include_test_characters=False)
        self.assertEqual({item['id'] for item in public['characters']}, set(PUBLIC_CHARACTER_IDS))
        self.assertTrue(all(item['character_id'] in PUBLIC_CHARACTER_IDS for item in public['cards']))
        self.assertEqual([item['name'] for item in public['starter_decks']], ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])
        self.assertFalse(deck_uses_test_characters(public['starter_deck']))
        self.assertFalse(any(deck_uses_test_characters(item) for item in public['starter_decks']))
        full = get_catalog()
        self.assertEqual(len(full['characters']), 19)
        self.assertEqual(len(full['starter_decks']), 5)
        self.assertEqual(
            [item['name'] for item in full['starter_decks']],
            ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'],
        )
        self.assertFalse(deck_uses_test_characters(next(item for item in STARTER_DECKS if item['id'] == 'murk')))
        self.assertFalse(deck_uses_test_characters(next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')))

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
        unknown_character['character_ids'][-1] = 'not-a-hero'
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
