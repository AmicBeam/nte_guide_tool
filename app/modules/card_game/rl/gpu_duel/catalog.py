"""Frozen 创生预组 identity for the GPU duel. Unsupported decks are rejected."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, validate_deck
from app.modules.card_game.content.duel_v2.catalog import CARD_ID_ORDER, energy_max_for_attribute

SEATS = tuple(STARTER_DECK['character_ids'])
if len(SEATS) != 4:
    raise RuntimeError('GPU duel requires the four-character official starter')
SEAT_INDEX = {cid: i for i, cid in enumerate(SEATS)}
CARD_IDS = tuple(
    card_id for card_id in CARD_ID_ORDER
    if CARDS[card_id].get('character_id') in SEAT_INDEX
)
CARD_INDEX = {card_id: i for i, card_id in enumerate(CARD_IDS)}
ATTR_IDS = ('光', '灵', '相', '暗', '魂', '咒')
ATTR_INDEX = {name: i for i, name in enumerate(ATTR_IDS)}
TYPE_INDEX = {'battle': 0, 'tactic': 1, 'form': 2}
HARMONY_NONE, HARMONY_GENESIS, HARMONY_DELAY = 0, 1, 2
PHASE_MULLIGAN, PHASE_PLAYING, PHASE_CHOICE, PHASE_FINISHED = 0, 1, 2, 3
ACT_END, ACT_ATTACK, ACT_PLAY, ACT_ULTIMATE, ACT_MULLIGAN, ACT_CHOOSE, ACT_CONCEDE = range(7)
TARGET_NONE, TARGET_SEAT, TARGET_PLAYER = 0, 1, 2
HAND_SLOTS, DECK_SLOTS, DISCARD_SLOTS, REMOVED_SLOTS = 10, 32, 32, 16
CHOICE_SLOTS = 10
MAX_LEGAL = 256
SIDES = 2
N_SEATS = 4
POLICY_NONE, POLICY_ALLY, POLICY_INJURED, POLICY_DOWN_ALLY, POLICY_ENEMY, POLICY_ENEMY_FRONT = range(6)
_POLICY = {
    'living_ally': POLICY_ALLY,
    'injured_role': POLICY_INJURED,
    'downed_ally': POLICY_DOWN_ALLY,
    'living_enemy': POLICY_ENEMY,
    'enemy_front': POLICY_ENEMY_FRONT,
}
LEGAL_END = 0
LEGAL_ATK = 1
LEGAL_ULT = 5
LEGAL_PLAY = 9
PLAY_TARGETS = 10
LEGAL_WIDTH = LEGAL_PLAY + HAND_SLOTS * PLAY_TARGETS

STARTER_CARD_IDS = tuple(STARTER_DECK['card_ids'])
STARTER_LOCK = sha256(
    ('|'.join(SEATS) + '#' + ','.join(STARTER_CARD_IDS)).encode('utf-8')
).hexdigest()


def assert_starter_deck(deck: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = validate_deck(dict(deck or STARTER_DECK))
    if tuple(payload['character_ids']) != SEATS:
        raise ValueError('GPU duel only accepts official starter seats: ' + ','.join(SEATS))
    unknown = [card_id for card_id in payload['card_ids'] if card_id not in CARD_INDEX]
    if unknown:
        raise ValueError('GPU duel cannot encode ' + ','.join(sorted(set(unknown))))
    return payload


def card_table() -> dict[str, list[int]]:
    from app.modules.card_game.content.duel_v2.characters import TARGET_POLICIES
    cost, atk, shield, hp, typ, owner, response, instant, derived, policy = [], [], [], [], [], [], [], [], [], []
    for card_id in CARD_IDS:
        card = CARDS[card_id]
        cost.append(int(card.get('cost') or 0))
        atk.append(int(card.get('attack') or 0))
        shield.append(int(card.get('shield') or 0))
        hp.append(int(card.get('hp') or 0))
        typ.append(TYPE_INDEX[card['type']])
        owner.append(SEAT_INDEX[card['character_id']])
        response.append(1 if card.get('response') == 'ally_attacked' else 0)
        instant.append(1 if card.get('instant') else 0)
        derived.append(1 if card.get('derived') else 0)
        policy.append(_POLICY.get(TARGET_POLICIES.get(card_id), POLICY_NONE))
    attr = [ATTR_INDEX[CHARACTERS[cid]['attribute']] for cid in SEATS]
    base_atk = [int(CHARACTERS[cid]['attack']) for cid in SEATS]
    base_hp = [int(CHARACTERS[cid]['max_hp']) for cid in SEATS]
    energy_max = [energy_max_for_attribute(CHARACTERS[cid]['attribute']) for cid in SEATS]
    ult_kind = [0 if (CHARACTERS[cid].get('ultimate') or {}).get('kind') == 'instant' else 1 for cid in SEATS]
    ult_turns = [int((CHARACTERS[cid].get('ultimate') or {}).get('turns') or 0) for cid in SEATS]
    return {
        'cost': cost, 'atk': atk, 'shield': shield, 'hp': hp, 'type': typ, 'owner': owner,
        'response': response, 'instant': instant, 'derived': derived, 'policy': policy,
        'attr': attr, 'base_atk': base_atk, 'base_hp': base_hp, 'energy_max': energy_max,
        'ult_kind': ult_kind, 'ult_turns': ult_turns,
    }


def card_id_of(index: int) -> str:
    if index < 0 or index >= len(CARD_IDS):
        return ''
    return CARD_IDS[index]


def seat_id_of(index: int) -> str:
    return SEATS[index]
