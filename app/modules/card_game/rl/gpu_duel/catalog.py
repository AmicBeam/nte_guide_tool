"""GPU duel identity: 创生预组 + 覆纹快攻 seats. Other presets are rejected."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, STARTER_DECKS, validate_deck
from app.modules.card_game.content.duel_v2.catalog import energy_max_for_attribute
from app.modules.card_game.rl.encoding import CARD_IDS as ENCODER_CARD_IDS, CHARACTER_IDS

SEATS = tuple(CHARACTER_IDS)
SEAT_INDEX = {cid: i for i, cid in enumerate(SEATS)}
NANALI, ILOY, ZERO, JIUYUAN, BOHE, BAICANG = (SEAT_INDEX[cid] for cid in (
    'nanali', 'iloy', 'zero', 'jiuyuan', 'bohe', 'baicang'))
CARD_IDS = tuple(ENCODER_CARD_IDS)
CARD_INDEX = {card_id: i for i, card_id in enumerate(CARD_IDS)}
ATTR_IDS = ('光', '灵', '相', '暗', '魂', '咒')
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_IDS)}
TYPE_INDEX = {'battle': 0, 'tactic': 1, 'form': 2}
HARMONY_NONE, HARMONY_GENESIS, HARMONY_DELAY, HARMONY_OVERLAY = 0, 1, 2, 3
PHASE_MULLIGAN, PHASE_PLAYING, PHASE_CHOICE, PHASE_FINISHED = 0, 1, 2, 3
ACT_END, ACT_ATTACK, ACT_PLAY, ACT_ULTIMATE, ACT_MULLIGAN, ACT_CHOOSE, ACT_CONCEDE = range(7)
TARGET_NONE, TARGET_SEAT, TARGET_PLAYER = 0, 1, 2
HAND_SLOTS, DECK_SLOTS, DISCARD_SLOTS, REMOVED_SLOTS = 10, 32, 32, 16
CHOICE_SLOTS = 10
MAX_LEGAL = 256
SIDES = 2
N_SEATS = len(SEATS)
POLICY_NONE, POLICY_ALLY, POLICY_INJURED, POLICY_DOWN_ALLY, POLICY_ENEMY, POLICY_ENEMY_FRONT, POLICY_DOWN_ENEMY, POLICY_ALLY_FRONT = range(8)
RESPONSE_NONE, RESPONSE_ALLY, RESPONSE_SELF, RESPONSE_SELF_LETHAL = 0, 1, 2, 3
REQUIRE_NONE, REQUIRE_OWN_BURN, REQUIRE_SURPLUS, REQUIRE_HARMONY_DAMAGE = 0, 1, 2, 3
_POLICY = {
    'living_ally': POLICY_ALLY,
    'injured_role': POLICY_INJURED,
    'downed_ally': POLICY_DOWN_ALLY,
    'living_enemy': POLICY_ENEMY,
    'enemy_front': POLICY_ENEMY_FRONT,
    'downed_enemy': POLICY_DOWN_ENEMY,
    'ally_front': POLICY_ALLY_FRONT,
}
_REQUIRE = {
    'own_burn': REQUIRE_OWN_BURN,
    'surplus': REQUIRE_SURPLUS,
    'harmony_damage': REQUIRE_HARMONY_DAMAGE,
}
LEGAL_END = 0
LEGAL_ATK = 1
LEGAL_ULT = LEGAL_ATK + N_SEATS
LEGAL_PLAY = LEGAL_ULT + N_SEATS
PLAY_TARGETS = 1 + N_SEATS + N_SEATS + 1
LEGAL_WIDTH = LEGAL_PLAY + HAND_SLOTS * PLAY_TARGETS
if LEGAL_WIDTH > MAX_LEGAL:
    raise RuntimeError(f'GPU legal layout {LEGAL_WIDTH} exceeds {MAX_LEGAL}')

ALLOWED_PRESETS = ('starter', 'weave-rush')
STARTER_CARD_IDS = tuple(STARTER_DECK['card_ids'])
GPU_LOCK = sha256(
    ('|'.join(SEATS) + '#' + ','.join(CARD_IDS)).encode('utf-8')
).hexdigest()
STARTER_LOCK = GPU_LOCK


def preset_by_id(deck_id: str) -> dict[str, Any]:
    preset = next((item for item in STARTER_DECKS if item.get('id') == deck_id), None)
    if preset is None:
        raise ValueError(f'unknown GPU preset {deck_id}')
    return assert_gpu_deck(preset)


def assert_gpu_deck(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = validate_deck(dict(deck or STARTER_DECK))
    extra = [cid for cid in payload['character_ids'] if cid not in SEAT_INDEX]
    if extra:
        raise ValueError('GPU duel seats are 创生预组 plus 覆纹快攻; got ' + ','.join(payload['character_ids']))
    if len(payload['character_ids']) != 4:
        raise ValueError('GPU duel still requires exactly 4 characters per side')
    unknown = [card_id for card_id in payload['card_ids'] if card_id not in CARD_INDEX]
    if unknown:
        raise ValueError('GPU duel cannot encode ' + ','.join(sorted(set(unknown))))
    return payload


def assert_starter_deck(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    return assert_gpu_deck(deck)


PUBLIC_CHARACTER_IDS = tuple(cid for cid in SEATS)


def public_card_ids() -> tuple[str, ...]:
    from app.modules.card_game.content.duel_v2 import CARDS
    return tuple(
        card_id for card_id in CARD_IDS
        if CARDS[card_id].get('character_id') in SEAT_INDEX
        and (not CARDS[card_id].get('derived') or card_id == 'NF01')
    )


def encode_public_deck_row(deck: dict[str, Any]) -> list[int]:
    """Pack one public four-person deck as [4 seat indices in team order] + [32 card kinds]."""
    payload = assert_gpu_deck(deck)
    seats = [SEAT_INDEX[cid] for cid in payload['character_ids']]
    if len(seats) != 4:
        raise ValueError('public compiled decks require exactly 4 characters')
    kinds = [CARD_INDEX[card_id] for card_id in payload['card_ids']]
    if len(kinds) != 32:
        raise ValueError('public compiled decks require exactly 32 cards')
    return seats + kinds


def decode_public_deck_row(row) -> dict[str, Any]:
    values = [int(v) for v in row]
    if len(values) != 36:
        raise ValueError('public deck row must be 36 ints')
    seats = values[:4]
    kinds = values[4:]
    if any(seat < 0 or seat >= N_SEATS for seat in seats) or len(set(seats)) != 4:
        raise ValueError('public deck seats must be four distinct catalog seats')
    unknown = [kind for kind in kinds if kind < 0 or kind >= len(CARD_IDS)]
    if unknown:
        raise ValueError('public deck contains unknown card kinds')
    character_ids = [SEATS[seat] for seat in seats]
    card_ids = [CARD_IDS[kind] for kind in kinds]
    return assert_gpu_deck({'character_ids': character_ids, 'card_ids': card_ids})


def present_mask(deck: dict[str, Any]) -> list[int]:
    owned = set(assert_gpu_deck(deck)['character_ids'])
    return [1 if cid in owned else 0 for cid in SEATS]


def deck_kind_ids(deck: dict[str, Any]) -> list[int]:
    payload = assert_gpu_deck(deck)
    return [CARD_INDEX[card_id] for card_id in payload['card_ids']]


def card_table() -> dict[str, list[int]]:
    from app.modules.card_game.content.duel_v2.characters import TARGET_POLICIES
    cost, atk, shield, hp, typ, owner, response, instant, derived, policy, require = [], [], [], [], [], [], [], [], [], [], []
    for card_id in CARD_IDS:
        card = CARDS[card_id]
        cost.append(int(card.get('cost') or 0))
        atk.append(int(card.get('attack') or 0))
        shield.append(int(card.get('shield') or 0))
        hp.append(int(card.get('hp') or 0))
        typ.append(TYPE_INDEX[card['type']])
        owner.append(SEAT_INDEX[card['character_id']])
        response.append(
            RESPONSE_ALLY if card.get('response') == 'ally_attacked'
            else RESPONSE_SELF if card.get('response') == 'self_attacked'
            else RESPONSE_SELF_LETHAL if card.get('response') == 'self_lethal'
            else RESPONSE_NONE
        )
        instant.append(1 if card.get('instant') else 0)
        derived.append(1 if card.get('derived') else 0)
        policy.append(_POLICY.get(TARGET_POLICIES.get(card_id), POLICY_NONE))
        require.append(_REQUIRE.get(card.get('require'), REQUIRE_NONE))
    attr = [ATTR_INDEX[CHARACTERS[cid]['attribute']] for cid in SEATS]
    base_atk = [int(CHARACTERS[cid]['attack']) for cid in SEATS]
    base_hp = [int(CHARACTERS[cid]['max_hp']) for cid in SEATS]
    energy_max = [energy_max_for_attribute(CHARACTERS[cid]['attribute']) for cid in SEATS]
    ult_kind = [0 if (CHARACTERS[cid].get('ultimate') or {}).get('kind') == 'instant' else 1 for cid in SEATS]
    ult_turns = [int((CHARACTERS[cid].get('ultimate') or {}).get('turns') or 0) for cid in SEATS]
    return {
        'cost': cost, 'atk': atk, 'shield': shield, 'hp': hp, 'type': typ, 'owner': owner,
        'response': response, 'instant': instant, 'derived': derived, 'policy': policy,
        'require': require,
        'playable_downed':[int(bool(CARDS[c].get('playable_downed'))) for c in CARD_IDS],
        'attr': attr, 'base_atk': base_atk, 'base_hp': base_hp, 'energy_max': energy_max,
        'ult_kind': ult_kind, 'ult_turns': ult_turns,
    }


def card_id_of(index: int) -> str:
    if index < 0 or index >= len(CARD_IDS):
        return ''
    return CARD_IDS[index]


def seat_id_of(index: int) -> str:
    return SEATS[index]
