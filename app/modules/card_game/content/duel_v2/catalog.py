from copy import deepcopy
from pathlib import Path
from typing import Any
import json


CATALOG_PATH = Path(__file__).with_name('catalog.json')

CHARACTER_ORDER = ('nanali', 'zero', 'jiuyuan', 'xun')
CARD_ID_ORDER = tuple(
    f'{prefix}{index:02d}'
    for prefix in ('N', 'Z', 'J', 'X')
    for index in range(1, 9)
)
REQUIRED_CHARACTER_FIELDS = (
    'id',
    'name',
    'attribute',
    'attack',
    'max_hp',
    'avatar',
    'portrait',
    'passive',
    'awakened_passive',
)
REQUIRED_CARD_FIELDS = (
    'id',
    'character_id',
    'name',
    'type',
    'cost',
    'terminal',
    'description',
    'effect_id',
    'source',
)
STARTER_DECK_ID = 'starter'
STARTER_DECK_NAME = '基础预组'


def _load_raw_catalog() -> dict[str, Any]:
    payload = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('duel v2 catalog.json 必须是对象')
    return payload


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{label} 必须是对象')
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f'{label} 必须是列表')
    return value


def _character_public(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CHARACTER_FIELDS if field not in raw]
    if missing:
        raise ValueError(f'角色缺少字段: {", ".join(missing)}')
    return {field: deepcopy(raw[field]) for field in REQUIRED_CHARACTER_FIELDS}


def _card_public(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CARD_FIELDS if field not in raw]
    if missing:
        raise ValueError(f'卡牌缺少字段: {", ".join(missing)}')
    source = _require_mapping(raw['source'], f'卡牌 {raw.get("id")} 的 source')
    for field in ('character_id', 'type', 'index'):
        if field not in source:
            raise ValueError(f'卡牌 {raw.get("id")} 的 source 缺少 {field}')
    return {field: deepcopy(raw[field]) for field in REQUIRED_CARD_FIELDS}


def _index_by_id(items: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get('id')
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f'{label} 缺少有效 id')
        if item_id in indexed:
            raise ValueError(f'{label} 出现重复 id: {item_id}')
        indexed[item_id] = item
    return indexed


_RAW_CATALOG = _load_raw_catalog()
_RAW_CHARACTERS = [_require_mapping(item, '角色') for item in _require_list(_RAW_CATALOG.get('characters'), 'characters')]
_RAW_CARDS = [_require_mapping(item, '卡牌') for item in _require_list(_RAW_CATALOG.get('cards'), 'cards')]
_RAW_CHARACTER_INDEX = _index_by_id(_RAW_CHARACTERS, '角色')
_RAW_CARD_INDEX = _index_by_id(_RAW_CARDS, '卡牌')

if tuple(character['id'] for character in _RAW_CHARACTERS) != CHARACTER_ORDER:
    raise ValueError('角色顺序必须为 nanali, zero, jiuyuan, xun')
if tuple(card['id'] for card in _RAW_CARDS) != CARD_ID_ORDER:
    raise ValueError('卡牌顺序必须为 N01-N08、Z01-Z08、J01-J08、X01-X08')

_DECK_RULES = _require_mapping(_RAW_CATALOG.get('deck_rules'), 'deck_rules')
DECK_SIZE = int(_DECK_RULES['size'])
PER_CHARACTER = int(_DECK_RULES['per_character'])
MAX_COPIES = int(_DECK_RULES['max_copies'])
CHARACTER_COUNT = int(_DECK_RULES['character_count'])
RULES_VERSION = str(_RAW_CATALOG.get('rules_version') or 'duel_v2')
SCHEMA_VERSION = int(_RAW_CATALOG.get('schema_version') or 1)

_CHARACTERS: dict[str, dict[str, Any]] = {
    character_id: _character_public(_RAW_CHARACTER_INDEX[character_id])
    for character_id in CHARACTER_ORDER
}
_CARDS: dict[str, dict[str, Any]] = {
    card_id: _card_public(_RAW_CARD_INDEX[card_id])
    for card_id in CARD_ID_ORDER
}


def _starter_card_ids() -> list[str]:
    card_ids: list[str] = []
    for card_id in CARD_ID_ORDER:
        copies = int(_RAW_CARD_INDEX[card_id].get('starter_copies') or 0)
        if copies < 0 or copies > MAX_COPIES:
            raise ValueError(f'卡牌 {card_id} 的预组数量非法')
        card_ids.extend([card_id] * copies)
    if len(card_ids) != DECK_SIZE:
        raise ValueError('基础预组必须恰好 32 张')
    return card_ids


_STARTER_DECK: dict[str, Any] = {
    'id': STARTER_DECK_ID,
    'name': STARTER_DECK_NAME,
    'character_ids': list(CHARACTER_ORDER),
    'card_ids': _starter_card_ids(),
}

CHARACTERS: dict[str, dict[str, Any]] = deepcopy(_CHARACTERS)
CARDS: dict[str, dict[str, Any]] = deepcopy(_CARDS)
STARTER_DECK: dict[str, Any] = deepcopy(_STARTER_DECK)


def get_catalog() -> dict[str, Any]:
    return {
        'rules_version': RULES_VERSION,
        'schema_version': SCHEMA_VERSION,
        'characters': [deepcopy(_CHARACTERS[character_id]) for character_id in CHARACTER_ORDER],
        'cards': [deepcopy(_CARDS[card_id]) for card_id in CARD_ID_ORDER],
        'starter_deck': deepcopy(_STARTER_DECK),
        'deck_rules': {
            'size': DECK_SIZE,
            'per_character': PER_CHARACTER,
            'max_copies': MAX_COPIES,
            'character_count': CHARACTER_COUNT,
        },
    }


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == '':
        raise ValueError(f'{label} 必须是非空字符串')
    return value


def _require_id_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{label} 必须是列表')
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == '':
            raise ValueError(f'{label} 含有无效 id')
        normalized.append(item)
    return normalized


def validate_deck(deck: dict) -> dict:
    if not isinstance(deck, dict):
        raise ValueError('构筑必须是对象')

    deck_id = _require_text(deck.get('id'), '构筑 id')
    deck_name = _require_text(deck.get('name'), '构筑名称')
    character_ids = _require_id_list(deck.get('character_ids'), 'character_ids')
    card_ids = _require_id_list(deck.get('card_ids'), 'card_ids')

    if len(character_ids) != CHARACTER_COUNT:
        raise ValueError(f'构筑必须恰好包含 {CHARACTER_COUNT} 名角色')
    if len(set(character_ids)) != CHARACTER_COUNT:
        raise ValueError('构筑不能包含重复角色')
    unknown_characters = [character_id for character_id in character_ids if character_id not in _CHARACTERS]
    if unknown_characters:
        raise ValueError(f'构筑包含未知角色: {", ".join(unknown_characters)}')

    if len(card_ids) != DECK_SIZE:
        raise ValueError(f'构筑必须恰好包含 {DECK_SIZE} 张卡牌')

    counts: dict[str, int] = {}
    per_character_counts: dict[str, int] = {character_id: 0 for character_id in character_ids}
    for card_id in card_ids:
        card = _CARDS.get(card_id)
        if card is None:
            raise ValueError(f'构筑包含未知卡牌: {card_id}')
        owner = card['character_id']
        if owner not in per_character_counts:
            raise ValueError(f'构筑包含其他角色的卡牌: {card_id}')
        counts[card_id] = counts.get(card_id, 0) + 1
        if counts[card_id] > MAX_COPIES:
            raise ValueError(f'同名卡牌最多 {MAX_COPIES} 张: {card_id}')
        per_character_counts[owner] += 1

    mismatched = [
        character_id
        for character_id, total in per_character_counts.items()
        if total != PER_CHARACTER
    ]
    if mismatched:
        raise ValueError(f'每名角色必须恰好携带 {PER_CHARACTER} 张卡牌: {", ".join(mismatched)}')

    return {
        'id': deck_id,
        'name': deck_name,
        'character_ids': list(character_ids),
        'card_ids': list(card_ids),
    }
