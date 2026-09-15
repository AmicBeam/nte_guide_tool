"""Single-deck transfer helpers. One deck per run; freeze generic ops when adapting."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECK, STARTER_DECKS, validate_deck
from app.modules.card_game.content.duel_v2.catalog import PUBLIC_CHARACTER_IDS

from .capability import assert_public_encoder_deck, preset_opponent_deck, sample_public_deck
from .encoding import CARD_IDS, CHARACTER_IDS


def default_training_deck() -> dict[str, Any]:
    """Official 创生预组; both learner and rule-AI opponent use this unless --deck is set."""
    return assert_encoder_compatible(STARTER_DECK)


def encoder_seat_deck() -> dict[str, Any]:
    return default_training_deck()


def assert_encoder_compatible(deck: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_deck(dict(deck))
    extra = [character_id for character_id in payload['character_ids'] if character_id not in CHARACTER_IDS]
    if extra:
        raise ValueError(
            'RL encoder seats are the six public characters ('
            + ','.join(CHARACTER_IDS)
            + '); this deck uses '
            + ','.join(payload['character_ids'])
        )
    unpublished = [character_id for character_id in payload['character_ids'] if character_id not in PUBLIC_CHARACTER_IDS]
    if unpublished:
        raise ValueError('Training decks must use public characters only: ' + ','.join(unpublished))
    if len(payload['character_ids']) != 4:
        raise ValueError('Training decks must still use exactly 4 characters')
    unknown = [card_id for card_id in payload['card_ids'] if card_id not in CARD_IDS]
    if unknown:
        raise ValueError('Deck contains cards the RL encoder cannot represent: ' + ','.join(sorted(set(unknown))))
    return assert_public_encoder_deck(payload)


def _preset_by_id(deck_id: str) -> dict[str, Any] | None:
    for preset in STARTER_DECKS:
        if preset.get('id') == deck_id:
            return deepcopy(preset)
    return None


def resolve_training_deck(spec: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    if spec in (None, '', 'default', 'starter', 'encoder-seats'):
        return default_training_deck()
    if spec in ('public-sample', 'public'):
        return sample_public_deck(0)
    if isinstance(spec, Mapping):
        return assert_encoder_compatible(spec)
    text = str(spec)
    preset = _preset_by_id(text)
    if preset is not None:
        return assert_encoder_compatible(preset)
    path = Path(text)
    if path.is_file():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('Deck JSON must be an object')
        return assert_encoder_compatible(payload)
    raise ValueError(f'Unknown training deck: {text}')


def parse_card_swaps(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for item in text.split(','):
        raw = item.strip()
        if not raw:
            continue
        if ':' not in raw and '=' not in raw:
            raise ValueError(f'Card swap must be OLD:NEW, got {raw!r}')
        left, right = raw.replace('=', ':', 1).split(':', 1)
        old, new = left.strip(), right.strip()
        if old not in CARD_IDS or new not in CARD_IDS:
            raise ValueError(f'Card swap {old}->{new} is outside the encoder catalog')
        if CARDS[old]['character_id'] != CARDS[new]['character_id']:
            raise ValueError(f'Card swap {old}->{new} must stay on the same character')
        if CARDS[new]['character_id'] not in PUBLIC_CHARACTER_IDS:
            raise ValueError(f'Card swap {old}->{new} must stay on public characters')
        pairs.append((old, new))
    return pairs


def apply_card_swaps(deck: Mapping[str, Any], swaps: list[tuple[str, str]]) -> dict[str, Any]:
    payload = assert_encoder_compatible(deck)
    if not swaps:
        return payload
    mapping = dict(swaps)
    card_ids = [mapping.get(card_id, card_id) for card_id in payload['card_ids']]
    return validate_deck({
        'id': payload.get('id') or 'custom',
        'name': (payload.get('name') or 'custom') + '+swap',
        'character_ids': list(payload['character_ids']),
        'card_ids': card_ids,
    })


def similar_card_sources(previous_card_ids: list[str] | tuple[str, ...],
                         next_card_ids: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    previous = [card_id for card_id in previous_card_ids if card_id in CARD_IDS]
    nxt = [card_id for card_id in next_card_ids if card_id in CARD_IDS]
    mapping: dict[str, list[str]] = {}
    previous_set = set(previous)
    for card_id in nxt:
        if card_id in previous_set:
            continue
        catalog = CARDS[card_id]
        similar = [
            item for item in previous
            if CARDS[item]['character_id'] == catalog['character_id']
            and CARDS[item]['type'] == catalog['type']
        ]
        if not similar:
            similar = [item for item in previous if CARDS[item]['type'] == catalog['type']]
        mapping[card_id] = similar
    return mapping


def unique_card_ids(deck: Mapping[str, Any]) -> list[str]:
    seen: list[str] = []
    for card_id in deck.get('card_ids') or []:
        if card_id not in seen:
            seen.append(card_id)
    return seen


def apply_transfer(
    model,
    *,
    freeze_generic: bool,
    generic_lr_mult: float,
    learning_rate: float,
    kl_coef: float,
    teacher=None,
    previous_card_ids: list[str] | None = None,
    next_card_ids: list[str] | None = None,
) -> dict[str, Any]:
    policy = model.policy
    init_info: dict[str, Any] = {'initialized': [], 'new_cards': []}
    if previous_card_ids is not None and next_card_ids is not None:
        init_info = policy.init_new_card_embeddings(previous_card_ids, next_card_ids)
    if freeze_generic:
        policy.freeze_generic()
        generic_lr_mult = 0.0
    card_params = list(policy.card_parameters())
    generic_params = [param for param in policy.generic_parameters() if param.requires_grad]
    trainable = card_params + generic_params
    if not trainable:
        raise ValueError('Transfer left no trainable parameters')
    if generic_params and generic_lr_mult != 1.0:
        param_groups = [
            {'params': generic_params, 'lr': learning_rate * generic_lr_mult},
            {'params': card_params, 'lr': learning_rate},
        ]
    else:
        param_groups = [{'params': trainable, 'lr': learning_rate}]
    policy.optimizer = policy.optimizer_class(param_groups, **policy.optimizer_kwargs)
    overlap = sorted(set(previous_card_ids or []) & set(next_card_ids or []))
    if kl_coef > 0:
        if teacher is None:
            raise ValueError('--kl-coef requires --resume so the previous policy can stay frozen')
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        ent_coef = float(getattr(model, 'ent_coef', 0.01) or 0.01)
        policy.attach_teacher(teacher, kl_coef=kl_coef, ent_coef=ent_coef, overlap_card_ids=overlap)
    return {
        'freeze_generic': freeze_generic,
        'generic_lr_mult': generic_lr_mult,
        'kl_coef': kl_coef,
        'overlap_cards': overlap,
        **init_info,
    }


def resolve_opponent_deck(spec: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Independent opponent policy deck. Default is the fixed 创生预组 mirror."""
    if spec in (None, '', 'default', 'rule', 'starter'):
        return preset_opponent_deck('starter')
    if isinstance(spec, Mapping):
        return assert_encoder_compatible(spec)
    text = str(spec)
    if text in ('starter', 'weave-rush'):
        return preset_opponent_deck(text)
    return resolve_training_deck(text)
