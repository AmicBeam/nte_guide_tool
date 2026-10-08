"""Public-character training/serving capability. No training is started here.

Encoder seats already cover the six public characters and their 48 kit cards
plus derived NF01. Shared compiled rules support public custom decks.
"""
from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any, Mapping

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS, validate_deck
from app.modules.card_game.content.duel_v2.catalog import (
    CARD_ID_ORDER, MAX_COPIES, PER_CHARACTER, PUBLIC_CHARACTER_IDS,
)
from app.modules.card_game.rl.encoding import CARD_IDS, CHARACTER_IDS
from app.modules.card_game.rl.rule_ir.starter import STARTER_CARDS

LEGACY_MIRROR_KIND = 'legacy_exact_mirror'
PUBLIC_ASYMMETRIC_KIND = 'public_asymmetric_v1'
PUBLIC_CANDIDATE_KIND = 'public_candidate_v1'
COMPILED_PRESETS = ('starter', 'weave-rush')
PUBLIC_CHARACTERS = tuple(cid for cid in CHARACTER_IDS if cid in PUBLIC_CHARACTER_IDS)
PUBLIC_KIT_CARDS = tuple(
    card_id for card_id in CARD_ID_ORDER
    if CARDS[card_id].get('character_id') in PUBLIC_CHARACTER_IDS and not CARDS[card_id].get('derived')
)
GPU_IR_CARDS = frozenset(program.card_id for program in STARTER_CARDS)
PUBLIC_GPU_IR_CARDS = tuple(card_id for card_id in PUBLIC_KIT_CARDS if card_id in GPU_IR_CARDS)
PUBLIC_GPU_IR_MISSING = tuple(card_id for card_id in PUBLIC_KIT_CARDS if card_id not in GPU_IR_CARDS)


def public_kit_cards(character_id: str) -> tuple[str, ...]:
    if character_id not in PUBLIC_CHARACTER_IDS:
        raise ValueError(f'{character_id} is not a public character')
    return tuple(
        card_id for card_id in CARD_ID_ORDER
        if CARDS[card_id].get('character_id') == character_id and not CARDS[card_id].get('derived')
    )


def encoder_public_support() -> dict[str, Any]:
    missing_seats = sorted(PUBLIC_CHARACTER_IDS - set(CHARACTER_IDS))
    missing_cards = [card_id for card_id in PUBLIC_KIT_CARDS if card_id not in CARD_IDS]
    return {
        'ok': not missing_seats and not missing_cards and len(PUBLIC_KIT_CARDS) == 48,
        'characters': list(PUBLIC_CHARACTERS),
        'character_count': len(PUBLIC_CHARACTERS),
        'kit_cards': list(PUBLIC_KIT_CARDS),
        'kit_card_count': len(PUBLIC_KIT_CARDS),
        'encoder_card_count': len(CARD_IDS),
        'missing_seats': missing_seats,
        'missing_cards': missing_cards,
        'note': 'Encoder covers six public seats and 48 kit cards; derived NF01 is extra.',
    }


def compiled_public_support() -> dict[str, Any]:
    from .rule_ir.completeness import validate_compiled_deck
    for preset_id in COMPILED_PRESETS:
        validate_compiled_deck(preset_id)
    return {
        'ok': not PUBLIC_GPU_IR_MISSING,
        'presets': list(COMPILED_PRESETS),
        'cuda_full_public': not PUBLIC_GPU_IR_MISSING,
        'cuda_runtime_verified': False,
        'missing_ir_cards': list(PUBLIC_GPU_IR_MISSING),
        'ir_public_cards': list(PUBLIC_GPU_IR_CARDS),
        'note': 'Shared compiled rules support public custom reset; actual CUDA hardware must pass preflight.',
    }


def is_exact_preset(deck: Mapping[str, Any], preset_id: str) -> bool:
    preset = next((item for item in STARTER_DECKS if item.get('id') == preset_id), None)
    if preset is None:
        return False
    payload = validate_deck(dict(deck))
    return (
        list(payload['character_ids']) == list(preset['character_ids'])
        and list(payload['card_ids']) == list(preset['card_ids'])
    )


def preset_opponent_deck(preset_id: str) -> dict[str, Any]:
    if preset_id not in COMPILED_PRESETS:
        raise ValueError(f'Advanced/training opponent preset must be {COMPILED_PRESETS}, got {preset_id}')
    preset = next(item for item in STARTER_DECKS if item.get('id') == preset_id)
    return assert_public_encoder_deck(preset)


def assert_public_encoder_deck(deck: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_deck(dict(deck))
    extra = [cid for cid in payload['character_ids'] if cid not in PUBLIC_CHARACTER_IDS]
    if extra:
        raise ValueError('Public training/serving decks cannot use unpublished characters: ' + ','.join(extra))
    unseen = [cid for cid in payload['character_ids'] if cid not in CHARACTER_IDS]
    if unseen:
        raise ValueError('Encoder cannot represent seats: ' + ','.join(unseen))
    unknown = [card_id for card_id in payload['card_ids'] if card_id not in CARD_IDS]
    if unknown:
        raise ValueError('Encoder cannot represent cards: ' + ','.join(sorted(set(unknown))))
    return payload


def sample_public_deck(seed: int, *, name: str = '公开自组', character_ids=None) -> dict[str, Any]:
    rng = Random(int(seed))
    characters = list(PUBLIC_CHARACTERS if character_ids is None else character_ids)
    if character_ids is not None and (len(characters)!=4 or len(set(characters))!=4
                                      or not set(characters)<=set(PUBLIC_CHARACTERS)):
        raise ValueError('Sampling team must contain four distinct public characters')
    rng.shuffle(characters)
    character_ids = characters[:4]
    card_ids: list[str] = []
    for character_id in character_ids:
        kit = list(public_kit_cards(character_id))
        if not kit:
            raise ValueError(f'No public kit cards for {character_id}')
        counts = {card_id: 0 for card_id in kit}
        remaining = PER_CHARACTER
        while remaining > 0:
            candidates = [card_id for card_id, count in counts.items() if count < MAX_COPIES]
            if not candidates:
                raise ValueError(f'Cannot sample {PER_CHARACTER} cards for {character_id}')
            chosen = rng.choice(candidates)
            counts[chosen] += 1
            remaining -= 1
        for card_id in kit:
            card_ids.extend([card_id] * counts[card_id])
    payload = validate_deck({
        'id': f'public-{int(seed) & 0xFFFFFFFF:08x}',
        'name': name,
        'character_ids': character_ids,
        'card_ids': card_ids,
    })
    return assert_public_encoder_deck(payload)


def matchup_decks(*, learning_side: str, learner_deck: Mapping[str, Any], opponent_deck: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if learning_side not in ('a', 'b'):
        raise ValueError('learning_side must be a/b')
    learner = assert_public_encoder_deck(learner_deck)
    opponent = assert_public_encoder_deck(opponent_deck)
    other = 'b' if learning_side == 'a' else 'a'
    return {learning_side: deepcopy(learner), other: deepcopy(opponent)}


def training_matchup(seed, *, learner_deck, learning_side='a', opponent_deck=None, sample_public=False):
    """Fixed AI preset versus a reproducible varied player deck."""
    if sample_public and not any(is_exact_preset(learner_deck, key) for key in COMPILED_PRESETS):
        raise ValueError('Public sampling requires a fixed starter/weave-rush learner')
    opponent = sample_public_deck(seed) if sample_public else (opponent_deck or learner_deck)
    return matchup_decks(learning_side=learning_side, learner_deck=learner_deck, opponent_deck=opponent)


def default_legacy_capability(deck_id: str) -> dict[str, Any]:
    if deck_id not in COMPILED_PRESETS:
        raise ValueError(f'legacy serving models only exist for {COMPILED_PRESETS}')
    return {
        'kind': LEGACY_MIRROR_KIND,
        'bot_preset': deck_id,
        'human_public_custom': False,
        'trained_distribution': 'exact_mirror',
        'validated_distribution': 'exact_mirror',
    }


def capability_ready_manifest(deck_id: str) -> dict[str, Any]:
    if deck_id not in COMPILED_PRESETS:
        raise ValueError(f'capability-ready bot preset must be {COMPILED_PRESETS}')
    support = encoder_public_support()
    if not support['ok']:
        raise ValueError('encoder is not ready for public asymmetric serving')
    return {
        'kind': PUBLIC_ASYMMETRIC_KIND,
        'bot_preset': deck_id,
        'human_public_custom': True,
        'trained_distribution': 'public_asymmetric_v1',
        'validated_distribution': 'public_asymmetric_v1',
        'encoder_characters': support['characters'],
        'encoder_kit_cards': support['kit_cards'],
    }


def resolve_model_capability(manifest: Mapping[str, Any], deck_id: str) -> dict[str, Any]:
    raw = manifest.get('capability')
    if raw is None:
        return default_legacy_capability(deck_id)
    if not isinstance(raw, dict):
        raise ValueError('Model capability must be an object')
    kind = raw.get('kind')
    if kind == PUBLIC_CANDIDATE_KIND:
        if raw.get('human_public_custom') is not False or raw.get('bot_preset') != deck_id:
            raise ValueError('Invalid unvalidated public candidate capability')
        return deepcopy(raw)
    if raw.get('human_public_custom') and kind not in (PUBLIC_ASYMMETRIC_KIND,):
        raise ValueError('Existing exact-mirror weights are not trained or validated on public custom decks')
    if kind in (None, LEGACY_MIRROR_KIND):
        if raw.get('human_public_custom'):
            raise ValueError('legacy exact-mirror models cannot claim public custom training')
        if raw.get('bot_preset') not in (None, deck_id):
            raise ValueError('legacy model bot preset mismatch')
        capability = default_legacy_capability(deck_id)
        capability.update({key: raw[key] for key in raw if key not in capability})
        capability['kind'] = LEGACY_MIRROR_KIND
        capability['bot_preset'] = deck_id
        capability['human_public_custom'] = False
        capability['trained_distribution'] = 'exact_mirror'
        capability['validated_distribution'] = 'exact_mirror'
        return capability
    if kind == PUBLIC_ASYMMETRIC_KIND:
        expected = capability_ready_manifest(deck_id)
        for key, value in expected.items():
            if raw.get(key) != value or (key == 'human_public_custom' and raw.get(key) is not True):
                raise ValueError(f'public asymmetric capability mismatch: {key}')
        return deepcopy(raw)
    raise ValueError(f'Unsupported model capability: {kind}')


def fixed_team_serving_deck(deck, preset_id):
    """Normalize a serving allocation without allowing character choice/reordering."""
    payload = assert_public_encoder_deck(deck)
    if payload['character_ids'] != preset_opponent_deck(preset_id)['character_ids']:
        raise ValueError('Serving allocation must preserve preset characters and order')
    return payload


def serving_deck_hash(deck):
    import hashlib
    import json
    payload = assert_public_encoder_deck(deck)
    identity = {key: payload[key] for key in ('character_ids', 'card_ids')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
