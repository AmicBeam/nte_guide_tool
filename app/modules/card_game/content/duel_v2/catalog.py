from copy import deepcopy
from pathlib import Path
from typing import Any
import json


CATALOG_PATH = Path(__file__).with_name('catalog.json')
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
STARTER_DECK_NAME = '创生预组'
DARK_ENERGY_ATTRIBUTES = frozenset(('暗', '魂', '咒'))
LIGHT_ENERGY_MAX = 5
DARK_ENERGY_MAX = 6


def energy_max_for_attribute(attribute: str) -> int:
    return DARK_ENERGY_MAX if attribute in DARK_ENERGY_ATTRIBUTES else LIGHT_ENERGY_MAX


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
    public = {field: deepcopy(raw[field]) for field in REQUIRED_CHARACTER_FIELDS}
    public['energy_max'] = energy_max_for_attribute(public['attribute'])
    for extra in ('everness_id', 'ultimate', 'faction', 'resource_name', 'turn_snapshots'):
        if extra in raw:
            public[extra] = deepcopy(raw[extra])
    mechanisms = []
    for item in _require_list(raw.get('mechanisms', []), '角色机制说明'):
        item = _require_mapping(item, '机制说明卡片')
        if any(not isinstance(item.get(key), str) or not item[key] for key in ('id', 'name', 'description')):
            raise ValueError('机制说明卡片必须有 id、name 和 description')
        names = _require_list(item.get('reference_names', [item['name']]), '机制引用名')
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError('机制引用名必须为非空文本')
        mechanisms.append({**deepcopy(item), 'character_id': raw['id'], 'type': 'mechanism',
                           'reference_names': deepcopy(names)})
    if mechanisms:
        public['mechanisms'] = mechanisms
    return public


def _card_public(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CARD_FIELDS if field not in raw]
    if missing:
        raise ValueError(f'卡牌缺少字段: {", ".join(missing)}')
    source = _require_mapping(raw['source'], f'卡牌 {raw.get("id")} 的 source')
    for field in ('character_id', 'type', 'index'):
        if field not in source:
            raise ValueError(f'卡牌 {raw.get("id")} 的 source 缺少 {field}')
    public = {field: deepcopy(raw[field]) for field in REQUIRED_CARD_FIELDS}
    public['starter_copies'] = int(raw.get('starter_copies') or 0)
    card_id = raw.get('id')
    kind = raw.get('type')
    if type(public['cost']) is not int or public['cost'] not in (0, 1):
        raise ValueError(f'卡牌 {card_id} 费用必须为 0 或 1')
    if kind == 'form':
        attack = int(raw.get('attack') or 0)
        hp = int(raw.get('hp') or 0)
        if 'stat' in raw and 'attack' not in raw and 'hp' not in raw:
            if raw['stat'] == 'attack':
                attack = int(raw.get('stat_value') or 0)
            elif raw['stat'] == 'hp':
                hp = int(raw.get('stat_value') or 0)
        if attack < 0 or hp < 0:
            raise ValueError(f'武备 {card_id} 的攻击与生命不能为负')
        public['attack'] = attack
        public['hp'] = hp
        if 'enemy_turn_attack' in raw:
            aura = raw['enemy_turn_attack']
            if type(aura) is not int or aura < 0:
                raise ValueError(f'武备 {card_id} 的对方回合攻击力必须为非负整数')
            public['enemy_turn_attack'] = aura
    elif kind == 'battle':
        attack = int(raw.get('attack') or 0)
        shield = int(raw.get('shield') or 0)
        if shield < 0:
            raise ValueError(f'战斗牌 {card_id} 的护盾不能为负')
        public['attack'] = attack
        public['shield'] = shield
    for extra in ('tooltip', 'instant', 'response', 'redeem', 'require', 'derived', 'playable_downed', 'attack_mode', 'play_options', 'spend_all_ap', 'training_excluded', 'response_faction', 'free_if_surplus', 'retain'):
        if extra in raw:
            public[extra] = deepcopy(raw[extra])
    if public.get('derived'):
        public['starter_copies'] = 0
    return public


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
CHARACTER_ORDER = tuple(character['id'] for character in _RAW_CHARACTERS)
CARD_ID_ORDER = tuple(card['id'] for card in _RAW_CARDS)

_raw_public_ids = _RAW_CATALOG.get('public_character_ids')
if _raw_public_ids is None:
    PUBLIC_CHARACTER_IDS = frozenset(CHARACTER_ORDER)
else:
    if not isinstance(_raw_public_ids, list) or not _raw_public_ids:
        raise ValueError('public_character_ids 必须是非空列表')
    _public_ids: list[str] = []
    for item in _raw_public_ids:
        if not isinstance(item, str) or not item:
            raise ValueError('public_character_ids 含有无效 id')
        if item not in _RAW_CHARACTER_INDEX:
            raise ValueError(f'public_character_ids 含有未知角色: {item}')
        _public_ids.append(item)
    PUBLIC_CHARACTER_IDS = frozenset(_public_ids)


def is_public_character(character_id: str) -> bool:
    return str(character_id) in PUBLIC_CHARACTER_IDS


def character_access_level(character_id: str) -> str:
    return 'public' if is_public_character(character_id) else 'test'


def deck_uses_test_characters(deck: dict[str, Any] | None) -> bool:
    if not isinstance(deck, dict):
        return False
    return any(not is_public_character(item) for item in (deck.get('character_ids') or []))


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


def _starter_card_ids(character_ids: list[str]) -> list[str]:
    allowed = set(character_ids)
    card_ids: list[str] = []
    for card_id in CARD_ID_ORDER:
        raw = _RAW_CARD_INDEX[card_id]
        if raw.get('character_id') not in allowed:
            continue
        if raw.get('derived'):
            continue
        copies = int(raw.get('starter_copies') or 0)
        if copies < 0 or copies > MAX_COPIES:
            raise ValueError(f'卡牌 {card_id} 的预组数量非法')
        card_ids.extend([card_id] * copies)
    if len(card_ids) != DECK_SIZE:
        raise ValueError(f'预组必须恰好 {DECK_SIZE} 张，实际 {len(card_ids)}')
    return card_ids


def _preset_decks() -> list[dict[str, Any]]:
    raw_presets = _RAW_CATALOG.get('starter_decks')
    if not raw_presets:
        return [{
            'id': STARTER_DECK_ID,
            'name': STARTER_DECK_NAME,
            'official': True,
            'character_ids': ['nanali', 'zero', 'jiuyuan', 'xun'],
            'card_ids': _starter_card_ids(['nanali', 'zero', 'jiuyuan', 'xun']),
        }]
    decks = []
    for item in _require_list(raw_presets, 'starter_decks'):
        preset = _require_mapping(item, '预组')
        character_ids = list(preset['character_ids'])
        if 'card_ids' in preset:
            card_ids = [str(card_id) for card_id in list(preset['card_ids'])]
            if len(card_ids) != DECK_SIZE:
                raise ValueError(f'预组 {preset.get("id")} 必须恰好 {DECK_SIZE} 张，实际 {len(card_ids)}')
        else:
            card_ids = _starter_card_ids(character_ids)
        decks.append({
            'id': str(preset['id']),
            'name': str(preset['name']),
            'official': bool(preset.get('official', True)),
            'character_ids': character_ids,
            'card_ids': card_ids,
        })
    return decks


_PRESET_DECKS = _preset_decks()
_STARTER_DECK: dict[str, Any] = {
    key: deepcopy(_PRESET_DECKS[0][key])
    for key in ('id', 'name', 'character_ids', 'card_ids')
}

CHARACTERS: dict[str, dict[str, Any]] = deepcopy(_CHARACTERS)
CARDS: dict[str, dict[str, Any]] = deepcopy(_CARDS)
STARTER_DECK: dict[str, Any] = deepcopy(_STARTER_DECK)
STARTER_DECKS: list[dict[str, Any]] = deepcopy(_PRESET_DECKS)


def _filter_catalog(payload: dict[str, Any], *, include_test_characters: bool) -> dict[str, Any]:
    if include_test_characters:
        return payload
    visible = set(PUBLIC_CHARACTER_IDS)
    payload['characters'] = [item for item in payload['characters'] if item.get('id') in visible]
    payload['cards'] = [item for item in payload['cards'] if item.get('character_id') in visible]
    payload['starter_decks'] = [
        item for item in payload['starter_decks'] if not deck_uses_test_characters(item)
    ]
    starter = payload.get('starter_deck')
    if deck_uses_test_characters(starter):
        payload['starter_deck'] = deepcopy(payload['starter_decks'][0]) if payload['starter_decks'] else starter
    return payload


def get_catalog(*, include_test_characters: bool = True) -> dict[str, Any]:
    payload = {
        'rules_version': RULES_VERSION,
        'schema_version': SCHEMA_VERSION,
        'characters': [
            {**deepcopy(_CHARACTERS[character_id]), 'access_level': character_access_level(character_id)}
            for character_id in CHARACTER_ORDER
        ],
        'cards': [deepcopy(_CARDS[card_id]) for card_id in CARD_ID_ORDER],
        'starter_deck': deepcopy(_STARTER_DECK),
        'starter_decks': deepcopy(_PRESET_DECKS),
        'deck_rules': {
            'size': DECK_SIZE,
            'per_character': PER_CHARACTER,
            'max_copies': MAX_COPIES,
            'character_count': CHARACTER_COUNT,
        },
    }
    return _filter_catalog(payload, include_test_characters=include_test_characters)


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
        if card.get('derived'):
            raise ValueError(f'衍生牌不可构筑: {card_id}')
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
