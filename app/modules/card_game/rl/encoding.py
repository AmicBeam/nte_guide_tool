"""Fixed-shape V2 observation/action encoder. Encodes observe only."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECKS
from app.modules.card_game.content.duel_v2.catalog import CARD_ID_ORDER

from .contracts import ACTION_TYPES, PHASES, RULES_VERSION, SIDES

FEATURE_VERSION = 'duel_v2_obs_v7'
ACTION_FEATURE_VERSION = 'duel_v2_act_v4'
ENCODER_VERSION = f'{FEATURE_VERSION}+{ACTION_FEATURE_VERSION}'

MIRROR_DECK_IDS = ('starter', 'weave-rush')


def _mirror_character_ids() -> tuple[str, ...]:
    ordered: list[str] = []
    for deck_id in MIRROR_DECK_IDS:
        preset = next((item for item in STARTER_DECKS if item.get('id') == deck_id), None)
        if preset is None:
            raise RuntimeError(f'RL encoder requires official preset {deck_id}')
        for character_id in preset['character_ids']:
            if character_id not in CHARACTERS:
                raise RuntimeError(f'RL encoder seat {character_id} is not in the catalog')
            if character_id not in ordered:
                ordered.append(character_id)
    if len(ordered) < 4:
        raise RuntimeError('RL encoder requires the official starter and weave-rush seats')
    return tuple(ordered)


# Seats are the union of 创生预组 and 覆纹快攻. Mirror training still uses one preset
# on both sides; missing seats are padded. Mixed matchups stay rejected in transfer.
CHARACTER_IDS = _mirror_character_ids()
CARD_IDS = tuple(
    card_id for card_id in CARD_ID_ORDER
    if CARDS[card_id].get('character_id') in CHARACTER_IDS
)
ATTRIBUTES = tuple(sorted({CHARACTERS[cid]['attribute'] for cid in CHARACTER_IDS}))
CARD_TYPES = ('battle', 'tactic', 'form')
FORM_CARD_IDS = tuple(card_id for card_id in CARD_IDS if CARDS[card_id]['type'] == 'form')
FORM_NAME_TO_ID = {CARDS[card_id]['name']: card_id for card_id in FORM_CARD_IDS}
HAND_SLOTS = 10
RECORD_SLOTS = 2
CHOICE_SLOTS = 10
HISTORY_SLOTS = 8
MULLIGAN_LIMIT = 3
MAX_HP_PLAYER = 30.0
MAX_SHIELD_PLAYER = 10.0
MAX_AP = 4.0
MAX_ATTACK = 10.0
MAX_HP_CHARACTER = 10.0
MAX_SHIELD_CHARACTER = 10.0
MAX_HARMONY = 2.0
MAX_ENERGY = 6.0
MAX_GROWTH = 4.0
MAX_DOWN = 3.0
MAX_HAND = float(HAND_SLOTS)
MAX_DECK = 32.0
MAX_TURN = 50.0
MAX_COST = 2.0
MAX_EXPIRES = 8.0

PENDING_KINDS = (
    'inspect_top', 'enemy_hand',
    'discard',
    'inspect',
    'library',
    'hand',
    'recover',
    'target',
    'reveal',
    'mulligan',
    'generic',
    'choose',
)
EVENT_TYPES = (
    'turn', 'enter', 'move', 'play', 'shape', 'resource', 'redeem', 'revive',
    'finish', 'discard', 'gain', 'remove', 'reveal', 'effect', 'combat', 'flower',
    'followup', 'fatigue', 'burn', 'star', 'overlay', 'penetration', 'unknown',
    'mulligan',
    'attack',
    'play_card',
    'ultimate',
    'cycle',
    'choose',
    'end_turn',
    'concede',
    'damage',
    'draw',
    'harmony',
    'record',
    'copy',
    'down',
    'recover',
    'heal',
    'shield',
    'energy',
    'growth',
    'log',
)
FORBIDDEN_OBSERVATION_KEYS = {
    'seed', 'rng', 'random', 'rng_state', 'random_state',
    'deck', 'library', 'library_order', 'hidden_hand', 'internal', 'private_state',
}
REQUIRED_OBSERVATION_KEYS = {
    'rules_version', 'version', 'phase', 'active_side', 'viewer_side', 'is_my_turn',
    'turn', 'winner', 'sides', 'pending_choice', 'legal_actions', 'events', 'logs',
}
REQUIRED_SIDE_KEYS = {
    'name', 'hp', 'shield', 'ap', 'normal_attack_available', 'front', 'characters',
    'hand', 'hand_count', 'deck_count', 'discard', 'records',
}
REQUIRED_CHARACTER_KEYS = {
    'id', 'name', 'attribute', 'attack', 'hp', 'max_hp', 'shield',
    'harmony', 'energy', 'awakened', 'growth', 'shape', 'down_turns',
}
ACTION_ALLOWED_FIELDS = {
    'mulligan': {'type', 'card_ids'},
    'attack': {'type', 'character_id'},
    'play_card': {'type', 'card_id', 'target_id'},
    'ultimate': {'type', 'character_id'},
    'cycle': {'type', 'card_id'},
    'choose': {'type', 'choice_id'},
    'end_turn': {'type'},
    'concede': {'type'},
}

USED_KEYS = (
    'cycle', 'growth', 'record', 'redeem', 'instant', 'sync', 'gaze',
    'N05', 'Z03', 'J02', 'J05', 'X02',
    'A07', 'C07', 'S07', 'S08', 'K08', 'H08', 'R07',
)
PUBLIC_CARD_WIDTH = 2 + len(CARD_IDS) + len(CHARACTER_IDS) * 2 + len(CARD_TYPES) + 4
CHARACTER_WIDTH = 1 + len(ATTRIBUTES) + 8 + len(FORM_CARD_IDS) + 1 + 2 + 7
RECORD_WIDTH = 1 + len(CARD_IDS) + len(CHARACTER_IDS) + len(CARD_TYPES)
SIDE_WIDTH = 9 + 3 + len(CHARACTER_IDS) * 2 + len(USED_KEYS) + len(CHARACTER_IDS) * CHARACTER_WIDTH + HAND_SLOTS * PUBLIC_CARD_WIDTH + len(CARD_IDS) + RECORD_SLOTS * RECORD_WIDTH
PENDING_WIDTH = 3 + len(PENDING_KINDS) + CHOICE_SLOTS * PUBLIC_CARD_WIDTH
EVENT_WIDTH = 1 + len(EVENT_TYPES) + len(SIDES) + 1
HISTORY_WIDTH = 2 + HISTORY_SLOTS * EVENT_WIDTH
GLOBAL_WIDTH = 2 + len(PHASES) + len(SIDES) + 1 + 4
OBSERVATION_DIM = GLOBAL_WIDTH + SIDE_WIDTH * 2 + PENDING_WIDTH + HISTORY_WIDTH + PUBLIC_CARD_WIDTH
ACTION_DIM = (
    len(ACTION_TYPES) + len(CHARACTER_IDS) + len(CARD_IDS) + len(CARD_TYPES) + 3
    + len(SIDES) + len(CHARACTER_IDS) + 1 + len(CARD_IDS) + 1 + HAND_SLOTS + 1
    + PUBLIC_CARD_WIDTH * 2
)
SHARED_ACTION_TYPES = ('mulligan', 'attack', 'ultimate', 'end_turn')
PUBLIC_CARD_ID_OFFSET = 2


def _take_slice(cursor: int, width: int) -> tuple[slice, int]:
    return slice(cursor, cursor + width), cursor + width


def action_feature_layout() -> dict[str, slice]:
    cursor = 0
    layout: dict[str, slice] = {}
    layout['action_type'], cursor = _take_slice(cursor, len(ACTION_TYPES))
    layout['actor_character'], cursor = _take_slice(cursor, len(CHARACTER_IDS))
    layout['action_card_id'], cursor = _take_slice(cursor, len(CARD_IDS))
    layout['action_card_type'], cursor = _take_slice(cursor, len(CARD_TYPES))
    layout['action_card_meta'], cursor = _take_slice(cursor, 3)
    layout['target_side'], cursor = _take_slice(cursor, len(SIDES))
    layout['target_character'], cursor = _take_slice(cursor, len(CHARACTER_IDS))
    layout['target_is_card'], cursor = _take_slice(cursor, 1)
    layout['target_card_id'], cursor = _take_slice(cursor, len(CARD_IDS))
    layout['mulligan_count'], cursor = _take_slice(cursor, 1)
    layout['mulligan_hand'], cursor = _take_slice(cursor, HAND_SLOTS)
    layout['choose_index'], cursor = _take_slice(cursor, 1)
    layout['played_card'], cursor = _take_slice(cursor, PUBLIC_CARD_WIDTH)
    layout['choice_card'], cursor = _take_slice(cursor, PUBLIC_CARD_WIDTH)
    if cursor != ACTION_DIM:
        raise RuntimeError(f'Action feature layout width {cursor} != {ACTION_DIM}')
    return layout


ACTION_LAYOUT = action_feature_layout()
CARD_IDENTITY_SLICES = (
    ACTION_LAYOUT['action_card_id'],
    ACTION_LAYOUT['target_card_id'],
    slice(
        ACTION_LAYOUT['played_card'].start + PUBLIC_CARD_ID_OFFSET,
        ACTION_LAYOUT['played_card'].start + PUBLIC_CARD_ID_OFFSET + len(CARD_IDS),
    ),
    slice(
        ACTION_LAYOUT['choice_card'].start + PUBLIC_CARD_ID_OFFSET,
        ACTION_LAYOUT['choice_card'].start + PUBLIC_CARD_ID_OFFSET + len(CARD_IDS),
    ),
)


def _one_hot_index(values: Sequence[float], width: int) -> int:
    if width <= 0 or len(values) < width:
        return 0
    best = 0
    score = -1.0
    for index in range(width):
        item = float(values[index])
        if item > score:
            score = item
            best = index
    return best + 1 if score > 0.0 else 0


def card_identity_index(vector: Sequence[float]) -> int:
    """Map a candidate to CARD_IDS index+1; 0 means no catalog card."""
    if len(vector) != ACTION_DIM:
        return 0
    played = ACTION_LAYOUT['action_card_id']
    index = _one_hot_index(vector[played], len(CARD_IDS))
    if index:
        return index
    for key in ('played_card', 'choice_card'):
        block = ACTION_LAYOUT[key]
        visible = float(vector[block.start])
        if visible <= 0.0:
            continue
        start = block.start + PUBLIC_CARD_ID_OFFSET
        index = _one_hot_index(vector[start:start + len(CARD_IDS)], len(CARD_IDS))
        if index:
            return index
    target = ACTION_LAYOUT['target_card_id']
    return _one_hot_index(vector[target], len(CARD_IDS))


def zero_card_identity(vector: Sequence[float]) -> list[float]:
    """Keep type/cost/target seats; drop catalog card one-hots."""
    generic = list(vector)
    if len(generic) != ACTION_DIM:
        return generic
    for region in CARD_IDENTITY_SLICES:
        for index in range(region.start, region.stop):
            generic[index] = 0.0
    return generic


def is_shared_action_vector(vector: Sequence[float], overlap_card_indexes: Sequence[int] = ()) -> bool:
    if len(vector) != ACTION_DIM:
        return False
    types = ACTION_LAYOUT['action_type']
    for offset, action_type in enumerate(ACTION_TYPES):
        if action_type in SHARED_ACTION_TYPES and float(vector[types.start + offset]) > 0.5:
            return True
    card_index = card_identity_index(vector)
    return bool(card_index) and card_index in set(overlap_card_indexes)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f'{label} must be a list')
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == '':
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _bool(value: Any, label: str) -> float:
    if not isinstance(value, bool):
        raise ValueError(f'{label} must be a boolean')
    return 1.0 if value else 0.0


def _number(value: Any, label: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a finite number')
    result = float(value)
    if result != result or result in (float('inf'), float('-inf')):
        raise ValueError(f'{label} must be a finite number')
    return result


def _one_hot(value: str | None, options: Sequence[str], label: str, *, required: bool) -> list[float]:
    if value is None:
        if required:
            raise ValueError(f'{label} is required')
        return [0.0] * len(options)
    if value not in options:
        raise ValueError(f'Unsupported {label}: {value}')
    return [1.0 if option == value else 0.0 for option in options]


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _norm(value: Any, scale: float, label: str, *, default: float = 0.0) -> float:
    return _clip01(_number(value, label, default=default) / scale)


def _require_keys(payload: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f'{label} missing fields: {", ".join(missing)}')


def _reject_forbidden(payload: Mapping[str, Any], label: str) -> None:
    leaked = sorted(key for key in payload if key in FORBIDDEN_OBSERVATION_KEYS)
    if leaked:
        raise ValueError(f'{label} includes hidden engine fields: {", ".join(leaked)}')


def _card_id(value: Any, label: str) -> str:
    card_id = _text(value, label)
    if card_id not in CARDS:
        raise ValueError(f'Unknown card_id: {card_id}')
    return card_id


def _character_id(value: Any, label: str) -> str:
    character_id = _text(value, label)
    if character_id not in CHARACTERS:
        raise ValueError(f'Unknown character_id: {character_id}')
    return character_id


def _shape_card_id(shape: Any) -> str | None:
    if shape in (None, '', False):
        return None
    if not isinstance(shape, str):
        raise ValueError('shape must be a form id, form name, or null')
    if shape in FORM_CARD_IDS:
        return shape
    if shape in FORM_NAME_TO_ID:
        return FORM_NAME_TO_ID[shape]
    raise ValueError(f'Unknown shape: {shape}')


def _copy_flag(value: Any) -> float:
    if value in (None, 0):
        return 0.0
    if not isinstance(value, bool):
        raise ValueError('copy must be a boolean')
    return 1.0 if value else 0.0


def _pad_rows(rows: list[list[float]], slots: int, width: int) -> list[float]:
    if len(rows) > slots:
        raise ValueError('Encoder slot capacity exceeded; no rows were truncated')
    packed: list[float] = []
    for row in rows:
        if len(row) != width:
            raise ValueError('Encoder row width mismatch')
        packed.extend(row)
    packed.extend([0.0] * width * (slots - len(rows)))
    return packed


def _public_card_vector(card: Mapping[str, Any] | None, *, hidden: bool, turn_count: float = 0) -> list[float]:
    vector = [0.0 if hidden or card is None else 1.0, 1.0 if hidden else 0.0]
    if hidden or card is None:
        return vector + [0.0] * (PUBLIC_CARD_WIDTH - 2)
    _require_keys(card, ('card_id', 'character_id', 'type', 'cost', 'terminal'), 'public card')
    card_id = _card_id(card['card_id'], 'card_id')
    character_id = _character_id(card['character_id'], 'character_id')
    catalog = CARDS[card_id]
    card_type = _text(card['type'], 'card type')
    copied = bool(_copy_flag(card.get('copy', False)))
    original = card.get('original_character_id', catalog['character_id'])
    if original != catalog['character_id']:
        raise ValueError('Original card author does not match catalog')
    if card_type != catalog['type']:
        raise ValueError(f'card {card_id} does not match catalog identity')
    if copied:
        if character_id not in CHARACTER_IDS:
            raise ValueError(f'copy holder {character_id} is not an encoder seat')
    elif character_id != catalog['character_id']:
        raise ValueError(f'card {card_id} does not match catalog identity')
    vector.extend(_one_hot(card_id, CARD_IDS, 'card_id', required=True))
    vector.extend(_one_hot(character_id, CHARACTER_IDS, 'character_id', required=True))
    vector.extend(_one_hot(original, CHARACTER_IDS, 'original_character_id', required=True))
    vector.extend(_one_hot(card_type, CARD_TYPES, 'card type', required=True))
    vector.append(_norm(card['cost'], MAX_COST, 'cost'))
    vector.append(_bool(card['terminal'], 'terminal'))
    vector.append(_copy_flag(card.get('copy', False)))
    expires = card.get('expires_turn')
    vector.append(0.0 if expires is None else _norm(_number(expires, 'expires_turn') - turn_count, MAX_EXPIRES, 'remaining expiry'))
    if len(vector) != PUBLIC_CARD_WIDTH:
        raise ValueError('public card vector width mismatch')
    return vector


def _empty_character_vector() -> list[float]:
    return [0.0] * CHARACTER_WIDTH


def _character_vector(character: Mapping[str, Any], front: str | None, turn_count: float = 0) -> list[float]:
    _require_keys(character, REQUIRED_CHARACTER_KEYS, 'character')
    character_id = _character_id(character['id'], 'character id')
    catalog = CHARACTERS[character_id]
    if character['name'] != catalog['name']:
        raise ValueError(f'character {character_id} name mismatch')
    attribute = _text(character['attribute'], 'attribute')
    if attribute != catalog['attribute']:
        raise ValueError(f'character {character_id} attribute mismatch')
    shape_id = _shape_card_id(character.get('shape_id', character.get('shape')))
    vector: list[float] = [1.0]
    vector.extend(_one_hot(attribute, ATTRIBUTES, 'attribute', required=True))
    vector.append(_norm(character['attack'], MAX_ATTACK, 'attack'))
    vector.append(_norm(character['hp'], MAX_HP_CHARACTER, 'hp'))
    vector.append(_norm(character['max_hp'], MAX_HP_CHARACTER, 'max_hp'))
    vector.append(_norm(character['shield'], MAX_SHIELD_CHARACTER, 'character shield'))
    vector.append(_norm(character['harmony'], MAX_HARMONY, 'harmony'))
    vector.append(_norm(character['energy'], MAX_ENERGY, 'energy'))
    vector.append(_bool(character['awakened'], 'awakened'))
    vector.append(_norm(character.get('family', character.get('growth', 0)), MAX_GROWTH, 'growth'))
    vector.extend(_one_hot(shape_id, FORM_CARD_IDS, 'shape', required=False))
    vector.append(1.0 if shape_id is None else 0.0)
    vector.append(_norm(character['down_turns'], MAX_DOWN, 'down_turns'))
    vector.append(1.0 if front == character_id else 0.0)
    slow = character.get('slow') or {}
    star = character.get('star')
    vector.extend([
        _norm(character.get('next_attack_bonus', 0), 4, 'next_attack_bonus'),
        _bool(character.get('return_after_sortie', False), 'return_after_sortie'),
        _norm(slow.get('amount', 0), 4, 'slow.amount'),
        _norm(_number(slow.get('end', turn_count), 'slow.end') - turn_count, 3, 'slow.remaining'),
        _norm(character.get('burn', 0), 2, 'burn'),
        float(star is not None),
        0.0 if star is None else _norm(_number(star, 'star') - turn_count, 3, 'star.remaining'),
    ])
    if len(vector) != CHARACTER_WIDTH:
        raise ValueError('character vector width mismatch')
    return vector


def _record_vector(record: Any) -> list[float]:
    payload = _mapping(record, 'record')
    card_id = payload.get('card_id') or payload.get('id')
    name = payload.get('name')
    if card_id is None:
        if not isinstance(name, str) or name == '':
            raise ValueError('Public record needs a public card_id or name')
        matches = [item_id for item_id, card in CARDS.items() if card['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'Unknown public record name: {name}')
        card_id = matches[0]
    card_id = _card_id(card_id, 'record card_id')
    character_id = _character_id(payload.get('character_id') or CARDS[card_id]['character_id'], 'record character_id')
    card_type = _text(payload.get('type') or CARDS[card_id]['type'], 'record type')
    if character_id != CARDS[card_id]['character_id'] or card_type != CARDS[card_id]['type']:
        raise ValueError(f'Public record does not match catalog card {card_id}')
    vector = [1.0]
    vector.extend(_one_hot(card_id, CARD_IDS, 'record card_id', required=True))
    vector.extend(_one_hot(character_id, CHARACTER_IDS, 'record character_id', required=True))
    vector.extend(_one_hot(card_type, CARD_TYPES, 'record type', required=True))
    if len(vector) != RECORD_WIDTH:
        raise ValueError('record vector width mismatch')
    return vector


def _discard_bag(cards: list[Any], label: str) -> list[float]:
    counts = {card_id: 0.0 for card_id in CARD_IDS}
    for item in cards:
        payload = _mapping(item, label)
        if payload.get('hidden'):
            raise ValueError(f'{label} cannot hide discard identities')
        counts[_card_id(payload.get('card_id'), f'{label} card_id')] += 1.0
    return [_clip01(counts[card_id] / 2.0) for card_id in CARD_IDS]


def _ordered_characters(characters: list[Any], front: str | None, side_id: str, turn_count: float = 0) -> list[list[float]]:
    by_id: dict[str, list[float]] = {}
    for character in characters:
        payload = _mapping(character, 'character')
        character_id = _character_id(payload.get('id'), 'character id')
        if character_id not in CHARACTER_IDS:
            raise ValueError(f'character {character_id} is not an encoder seat')
        if character_id in by_id:
            raise ValueError(f'Duplicate character in side {side_id}')
        by_id[character_id] = _character_vector(payload, front, turn_count)
    if len(by_id) != 4:
        raise ValueError(f'side {side_id} must expose 4 duel characters')
    return [by_id.get(character_id) or _empty_character_vector() for character_id in CHARACTER_IDS]


def _hand_rows(hand: list[Any], *, viewer: bool, turn_count: float = 0) -> list[list[float]]:
    rows: list[list[float]] = []
    for card in hand:
        payload = _mapping(card, 'hand card')
        hidden = bool(payload.get('hidden'))
        if viewer and hidden:
            raise ValueError('Viewer hand cannot be hidden')
        if hidden:
            rows.append(_public_card_vector(None, hidden=True))
            continue
        rows.append(_public_card_vector(payload, hidden=False, turn_count=turn_count))
    return rows


def _side_vector(side: Mapping[str, Any], side_id: str, viewer_side: str, active_side: str | None) -> list[float]:
    _require_keys(side, REQUIRED_SIDE_KEYS, f'side {side_id}')
    front = side['front']
    if front not in (None, *CHARACTER_IDS):
        raise ValueError(f'Unsupported front character: {front}')
    vector: list[float] = [
        _norm(side['hp'], MAX_HP_PLAYER, f'{side_id} hp'),
        _norm(side['shield'], MAX_SHIELD_PLAYER, f'{side_id} shield'),
        _norm(side['ap'], MAX_AP, f'{side_id} ap'),
        _bool(side['normal_attack_available'], f'{side_id} normal_attack_available'),
        _norm(side['hand_count'], MAX_HAND, f'{side_id} hand_count'),
        _norm(side['deck_count'], MAX_DECK, f'{side_id} deck_count'),
        _norm(len(_list(side['discard'], f'{side_id} discard')), MAX_DECK, f'{side_id} discard_count'),
        1.0 if viewer_side == side_id else 0.0,
        1.0 if active_side == side_id else 0.0,
    ]
    turn_count = _number(side.get('turn_count', 0), 'turn_count')
    used = _mapping(side.get('used', {}), 'used')
    harmonized = _mapping(side.get('harmonized', {}), 'harmonized')
    if set(used) - set(USED_KEYS) or set(harmonized) - set(CHARACTER_IDS):
        raise ValueError('Unencoded public counter key; update encoder version')
    vector.extend([_norm(turn_count, MAX_TURN, 'turn_count'),
                   _norm(side.get('fatigue', 0), MAX_HP_PLAYER, 'fatigue'),
                   _bool(side.get('extra_flower', False), 'extra_flower')])
    vector.extend(_one_hot(side.get('last_front'), CHARACTER_IDS, 'last_front', required=False))
    vector.extend(_bool(used.get(key, False), key) for key in USED_KEYS)
    vector.extend(_bool(harmonized.get(key, False), key) for key in CHARACTER_IDS)
    for row in _ordered_characters(_list(side['characters'], f'side {side_id} characters'), front, side_id, turn_count):
        vector.extend(row)
    hand = _list(side['hand'], f'{side_id} hand')
    vector.extend(_pad_rows(_hand_rows(hand, viewer=viewer_side == side_id, turn_count=turn_count), HAND_SLOTS, PUBLIC_CARD_WIDTH))
    vector.extend(_discard_bag(_list(side['discard'], f'{side_id} discard'), f'{side_id} discard'))
    records = _list(side['records'], f'{side_id} records')
    vector.extend(_pad_rows([_record_vector(record) for record in records], RECORD_SLOTS, RECORD_WIDTH))
    if len(vector) != SIDE_WIDTH:
        raise ValueError(f'side {side_id} vector width mismatch')
    return vector


def _pending_vector(observation: Mapping[str, Any], viewer_side: str) -> list[float]:
    pending = observation.get('pending_choice')
    if pending is None:
        vector = [0.0, 0.0, 0.0] + [0.0] * len(PENDING_KINDS)
        vector.extend(_pad_rows([], CHOICE_SLOTS, PUBLIC_CARD_WIDTH))
        return vector
    payload = _mapping(pending, 'pending_choice')
    side = _text(payload.get('side'), 'pending_choice.side')
    if side not in SIDES:
        raise ValueError(f'Unsupported pending_choice.side: {side}')
    kind = _text(payload.get('kind'), 'pending_choice.kind')
    if kind not in PENDING_KINDS:
        raise ValueError(f'Unsupported pending_choice.kind: {kind}')
    choices = _list(payload.get('choices'), 'pending_choice.choices')
    mine = side == viewer_side
    vector = [1.0, 1.0 if mine else 0.0, _clip01(len(choices) / float(CHOICE_SLOTS))]
    vector.extend(_one_hot(kind, PENDING_KINDS, 'pending_choice.kind', required=True))
    rows: list[list[float]] = []
    if mine:
        for choice in choices:
            item = _mapping(choice, 'pending choice')
            if 'id' not in item:
                raise ValueError('pending choice requires id')
            card = item.get('card')
            if card is None:
                empty = _public_card_vector(None, hidden=False)
                empty[0] = 0.0
                rows.append(empty)
            else:
                # enemy_hand candidates belong to the opponent's local turn clock.
                owner_side = next(s for s in SIDES if s != viewer_side) if kind == 'enemy_hand' else side
                clock = observation['sides'][owner_side].get('turn_count', 0)
                rows.append(_public_card_vector(_mapping(card, 'pending choice card'), hidden=False, turn_count=clock))
    vector.extend(_pad_rows(rows, CHOICE_SLOTS, PUBLIC_CARD_WIDTH))
    if len(vector) != PENDING_WIDTH:
        raise ValueError('pending_choice vector width mismatch')
    return vector


def _history_vector(observation: Mapping[str, Any]) -> list[float]:
    events = _list(observation.get('events'), 'events')
    logs = _list(observation.get('logs'), 'logs')
    vector = [_clip01(len(events) / 32.0), _clip01(len(logs) / 32.0)]
    rows: list[list[float]] = []
    for event in events[-HISTORY_SLOTS:]:
        payload = _mapping(event, 'event')
        event_type = _text(payload.get('type'), 'event.type')
        if event_type not in EVENT_TYPES:
            event_type = 'unknown'  # Public cosmetic events must not break decisions.
        side = payload.get('side')
        if side not in (None, *SIDES):
            raise ValueError(f'Unsupported event.side: {side}')
        row = [1.0]
        row.extend(_one_hot(event_type, EVENT_TYPES, 'event.type', required=True))
        row.extend(_one_hot(side, SIDES, 'event.side', required=False))
        row.append(_norm(payload.get('amount'), 10.0, 'event.amount', default=0.0))
        rows.append(row)
    vector.extend(_pad_rows(rows, HISTORY_SLOTS, EVENT_WIDTH))
    if len(vector) != HISTORY_WIDTH:
        raise ValueError('history vector width mismatch')
    return vector


def _visible_cards(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    viewer = _text(observation['viewer_side'], 'viewer_side')
    sides = _mapping(observation['sides'], 'sides')
    for side_id in SIDES:
        side = _mapping(sides.get(side_id), f'side {side_id}')
        for label in ('hand', 'discard'):
            for item in _list(side.get(label), f'{side_id} {label}'):
                payload = _mapping(item, label)
                if payload.get('hidden') and side_id != viewer:
                    continue
                instance_id = payload.get('instance_id')
                if isinstance(instance_id, str) and instance_id:
                    found[instance_id] = payload
        for record in _list(side.get('records'), f'{side_id} records'):
            payload = _mapping(record, 'record')
            instance_id = payload.get('instance_id')
            if isinstance(instance_id, str) and instance_id:
                found[instance_id] = payload
    pending = observation.get('pending_choice')
    if isinstance(pending, dict) and pending.get('side') == viewer:
        for choice in _list(pending.get('choices'), 'pending_choice.choices'):
            item = _mapping(choice, 'choice')
            choice_id = item.get('id')
            card = item.get('card')
            if isinstance(choice_id, str) and isinstance(card, dict):
                found[choice_id] = card
                instance_id = card.get('instance_id')
                if isinstance(instance_id, str) and instance_id:
                    found[instance_id] = card
    return found


def encoder_metadata() -> dict[str, Any]:
    return {
        'rules_version': RULES_VERSION,
        'encoder_version': ENCODER_VERSION,
        'feature_version': FEATURE_VERSION,
        'action_feature_version': ACTION_FEATURE_VERSION,
        'observation_dim': OBSERVATION_DIM,
        'action_dim': ACTION_DIM,
        'action_capacity': 256,
        'character_ids': list(CHARACTER_IDS),
        'card_ids': list(CARD_IDS),
        'action_types': list(ACTION_TYPES),
        'form_card_ids': list(FORM_CARD_IDS),
    }


class V2Encoder:
    version = ENCODER_VERSION
    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_DIM
    rules_version = RULES_VERSION
    feature_version = FEATURE_VERSION
    action_feature_version = ACTION_FEATURE_VERSION

    def metadata(self) -> dict[str, Any]:
        return encoder_metadata()

    def encode_observation(self, observation: dict[str, Any]) -> list[float]:
        payload = _mapping(observation, 'observation')
        _reject_forbidden(payload, 'observation')
        _require_keys(payload, REQUIRED_OBSERVATION_KEYS, 'observation')
        if payload.get('rules_version') != RULES_VERSION:
            raise ValueError('Encoder requires rules_version=duel_v2')
        viewer_side = _text(payload['viewer_side'], 'viewer_side')
        if viewer_side not in SIDES:
            raise ValueError(f'Unsupported viewer_side: {viewer_side}')
        active_side = payload.get('active_side')
        if active_side not in (None, *SIDES):
            raise ValueError(f'Unsupported active_side: {active_side}')
        phase = _text(payload['phase'], 'phase')
        if phase not in PHASES:
            raise ValueError(f'Unsupported phase: {phase}')
        winner = payload.get('winner')
        if winner not in (None, 'a', 'b', 'draw'):
            raise ValueError(f'Unsupported winner: {winner}')
        sides = _mapping(payload['sides'], 'sides')
        if set(sides) != set(SIDES):
            raise ValueError('observation.sides must contain a and b')
        vector: list[float] = [
            _norm(payload['version'], 1000.0, 'version'),
            _norm(payload['turn'], MAX_TURN, 'turn'),
        ]
        vector.extend(_one_hot(phase, PHASES, 'phase', required=True))
        vector.extend(_one_hot(active_side, SIDES, 'active_side', required=False))
        vector.append(_bool(payload['is_my_turn'], 'is_my_turn'))
        vector.extend([
            1.0 if winner is None else 0.0,
            1.0 if winner == 'a' else 0.0,
            1.0 if winner == 'b' else 0.0,
            1.0 if winner == 'draw' else 0.0,
        ])
        for side_id in SIDES:
            vector.extend(_side_vector(_mapping(sides[side_id], f'side {side_id}'), side_id, viewer_side, active_side))
        vector.extend(_pending_vector(payload, viewer_side))
        vector.extend(_history_vector(payload))
        resolving = payload.get('resolving_card')
        clock_side = active_side or viewer_side
        vector.extend(_public_card_vector(resolving, hidden=False, turn_count=sides[clock_side].get('turn_count', 0)))
        if len(vector) != self.observation_dim:
            raise ValueError(f'Observation encoding size {len(vector)} != {self.observation_dim}')
        if any(item != item or item in (float('inf'), float('-inf')) for item in vector):
            raise ValueError('Observation encoding produced nonfinite features')
        return vector

    def encode_action(self, observation: dict[str, Any], action: dict[str, Any]) -> list[float]:
        view = _mapping(observation, 'observation')
        payload = _mapping(action, 'action')
        action_type = _text(payload.get('type'), 'action.type')
        allowed = ACTION_ALLOWED_FIELDS.get(action_type)
        if allowed is None:
            raise ValueError(f'Unsupported action.type: {action_type}')
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f'Unsupported action fields for {action_type}: {sorted(extra)}')
        visible = _visible_cards(view) if 'sides' in view else {}
        viewer_side = view.get('viewer_side')
        viewer_hand: list[Any] = []
        if viewer_side in SIDES and isinstance(view.get('sides'), dict):
            viewer_hand = _list(_mapping(view['sides'].get(viewer_side), 'viewer side').get('hand'), 'viewer hand')
        vector = _one_hot(action_type, ACTION_TYPES, 'action.type', required=True)
        character_id = None
        if action_type in ('attack', 'ultimate'):
            character_id = _character_id(payload.get('character_id'), 'action.character_id')
        vector.extend(_one_hot(character_id, CHARACTER_IDS, 'action.character_id', required=False))
        resolved_card_id = None
        card_payload: dict[str, Any] | None = None
        if action_type in ('play_card', 'cycle'):
            raw_card_id = _text(payload.get('card_id'), 'action.card_id')
            if raw_card_id in visible:
                card_payload = visible[raw_card_id]
                resolved_card_id = _card_id(card_payload.get('card_id'), 'visible card_id')
            elif raw_card_id in CARD_IDS:
                raise ValueError('Action card_id must be a visible instance, not a hidden catalog id')
            else:
                raise ValueError('Action card_id is not visible in this observation')
        if card_payload is None:
            vector.extend([0.0] * (len(CARD_IDS) + len(CARD_TYPES) + 3))
        else:
            catalog = CARDS[resolved_card_id]
            vector.extend(_one_hot(resolved_card_id, CARD_IDS, 'action card_id', required=True))
            vector.extend(_one_hot(card_payload.get('type') or catalog['type'], CARD_TYPES, 'action card type', required=True))
            vector.append(_norm(card_payload.get('cost', catalog['cost']), MAX_COST, 'action cost'))
            terminal = card_payload.get('terminal', catalog['terminal'])
            if not isinstance(terminal, bool):
                raise ValueError('terminal must be a boolean')
            vector.append(_bool(terminal, 'action terminal'))
            vector.append(_copy_flag(card_payload.get('copy', False)))
        target_side = None
        target_character = None
        target_card = None
        target_is_card = 0.0
        if action_type == 'play_card' and 'target_id' in payload and payload.get('target_id') is not None:
            target_text = _text(payload.get('target_id'), 'action.target_id')
            if ':' in target_text and target_text.split(':', 1)[0] in SIDES:
                target_side, raw_character = target_text.split(':', 1)
                if raw_character != 'player':
                    target_character = _character_id(raw_character, 'target character')
            elif target_text in visible:
                target_is_card = 1.0
                target_card = _card_id(visible[target_text].get('card_id'), 'target card_id')
            else:
                raise ValueError('Action target_id is not a visible character or card')
        vector.extend(_one_hot(target_side, SIDES, 'target side', required=False))
        vector.extend(_one_hot(target_character, CHARACTER_IDS, 'target character', required=False))
        vector.append(target_is_card)
        vector.extend(_one_hot(target_card, CARD_IDS, 'target card', required=False))
        mulligan_ids: list[str] = []
        if action_type == 'mulligan':
            raw_ids = payload.get('card_ids')
            if not isinstance(raw_ids, list):
                raise ValueError('mulligan.card_ids must be a list')
            if len(raw_ids) > MULLIGAN_LIMIT:
                raise ValueError('mulligan cannot select more than 3 cards')
            if any(not isinstance(item, str) or item == '' for item in raw_ids):
                raise ValueError('mulligan.card_ids must be instance ids')
            if len(set(raw_ids)) != len(raw_ids):
                raise ValueError('mulligan.card_ids must be unique')
            mulligan_ids = list(raw_ids)
        vector.append(_clip01(len(mulligan_ids) / float(MULLIGAN_LIMIT)))
        hand_mask = [0.0] * HAND_SLOTS
        visible_hand_ids: list[str | None] = []
        for item in viewer_hand:
            card = _mapping(item, 'hand card')
            instance_id = card.get('instance_id')
            visible_hand_ids.append(instance_id if isinstance(instance_id, str) else None)
        for selected in mulligan_ids:
            if selected not in visible_hand_ids:
                raise ValueError('mulligan card is not in the visible hand')
            hand_mask[visible_hand_ids.index(selected)] = 1.0
        vector.extend(hand_mask)
        choose_index = 0.0
        if action_type == 'choose':
            choice_id = _text(payload.get('choice_id'), 'action.choice_id')
            pending = view.get('pending_choice')
            if not isinstance(pending, dict) or pending.get('side') != viewer_side:
                raise ValueError('choose action requires a private pending_choice for the viewer')
            ids = [_mapping(choice, 'choice').get('id') for choice in _list(pending.get('choices'), 'pending_choice.choices')]
            if choice_id not in ids:
                raise ValueError('choice_id is not a visible private candidate')
            choose_index = (ids.index(choice_id) + 1) / max(len(ids), 1)
        vector.append(choose_index)
        # Explicit card features distinguish copied instances with different expiry
        # and make choice identity available without learning an index lookup.
        selected_card = None
        if action_type == 'choose':
            selected_card = visible.get(payload['choice_id'])
        own_clock = view.get('sides', {}).get(viewer_side, {}).get('turn_count', 0)
        choice_clock = own_clock
        if action_type == 'choose' and view['pending_choice']['kind'] == 'enemy_hand':
            enemy = next(s for s in SIDES if s != viewer_side)
            choice_clock = view['sides'][enemy].get('turn_count', 0)
        vector.extend(_public_card_vector(card_payload, hidden=False, turn_count=own_clock))
        vector.extend(_public_card_vector(selected_card, hidden=False, turn_count=choice_clock))
        if len(vector) != self.action_dim:
            raise ValueError(f'Action encoding size {len(vector)} != {self.action_dim}')
        if any(item != item or item in (float('inf'), float('-inf')) for item in vector):
            raise ValueError('Action encoding produced nonfinite features')
        return vector
