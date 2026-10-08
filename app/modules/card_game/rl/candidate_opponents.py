"""Deterministic candidate-opponent pair schedule.

Compiled runtime later plays each pair as two opposite-seat games. It uses the
returned even seed together with seed+1 so the two shuffles differ. Every
candidate reuses this same returned schedule.
"""
from __future__ import annotations

import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from app.modules.card_game.content.duel_v2.catalog import MAX_COPIES, PER_CHARACTER
from app.modules.card_game.rl.capability import (
    assert_public_encoder_deck,
    fixed_team_serving_deck,
    preset_opponent_deck,
    public_kit_cards,
    sample_public_deck,
    serving_deck_hash,
)

PUBLIC_PRESETS = ('starter', 'weave-rush')
_SEED31 = 1 << 31
_SERVING_DIR = Path(__file__).resolve().parent.parent / 'engine' / 'ai' / 'models'
_SERVING_BUILDS: dict[str, dict[str, Any]] = {}


def build_opponent_schedule(
    pairs_per_preset,
    *,
    seed,
    random_fraction=0.2,
    rule_fraction=0.2,
    model_ids=('learner', 'peer'),
    historical_builds=None,
):
    pairs_per_preset = _require_pairs(pairs_per_preset)
    seed = _require_int(seed, 'seed')
    random_fraction = _require_fraction(random_fraction, 'random_fraction')
    rule_fraction = _require_fraction(rule_fraction, 'rule_fraction')
    if len(PUBLIC_PRESETS)*pairs_per_preset+_generalization_count(pairs_per_preset,random_fraction)>8192:
        raise ValueError('Opponent schedule exceeds 8192 pairs; reduce pairs or random_fraction')
    models = _require_model_ids(model_ids)
    history = _validate_historical_builds(historical_builds)
    rng = Random(seed)
    include_random = True  # Fixed-lineup card variation is independent of random-team fraction.
    schedule: list[dict[str, Any]] = []
    for preset_id in PUBLIC_PRESETS:
        official = preset_opponent_deck(preset_id)
        history_decks = history.get(preset_id, ())
        allocations = _allocation_cycle(
            pairs_per_preset,
            include_random=include_random,
            has_history=bool(history_decks),
        )
        history_index = 0
        for allocation in allocations:
            pair_seed = _even_seed(rng)
            if allocation == 'official':
                deck = deepcopy(official)
            elif allocation == 'serving':
                deck = deepcopy(_bundled_serving_build(preset_id))
            elif allocation == 'history':
                deck = deepcopy(history_decks[history_index % len(history_decks)])
                history_index += 1
            else:
                deck = _sample_fixed_lineup(
                    pair_seed,
                    official['character_ids'],
                    deck_id=f'{preset_id}-random-{len(schedule):04d}',
                    name=f"{official['name']}·随机",
                )
            schedule.append(_pair_record(
                pair=len(schedule),
                lineup=preset_id,
                allocation=allocation,
                deck=deck,
                seed=pair_seed,
            ))
    for _ in range(_generalization_count(pairs_per_preset, random_fraction)):
        pair_seed = _even_seed(rng)
        schedule.append(_pair_record(
            pair=len(schedule),
            lineup='public-random',
            allocation='random',
            deck=sample_public_deck(pair_seed),
            seed=pair_seed,
        ))
    for record, opponent_model in zip(schedule, _opponent_models(len(schedule), rule_fraction, models)):
        record['opponent_model'] = opponent_model
    return schedule


def _pair_record(*, pair, lineup, allocation, deck, seed):
    payload = deepcopy(assert_public_encoder_deck(deck))
    return {
        'pair': pair,
        'lineup': lineup,
        'allocation': allocation,
        'deck': payload,
        'build_sha256': serving_deck_hash(payload),
        'seed': seed,
    }


def _allocation_cycle(count, *, include_random, has_history):
    cycle = ['official', 'serving']
    if include_random:
        cycle.append('random')
    if has_history:
        cycle.append('history')
    return [cycle[index % len(cycle)] for index in range(count)]


def _generalization_count(pairs_per_preset, random_fraction):
    if random_fraction <= 0:
        return 0
    return max(1, int(round(len(PUBLIC_PRESETS) * pairs_per_preset * random_fraction / (1 - random_fraction))))


def _opponent_models(count, rule_fraction, model_ids):
    labels = []
    model_index = 0
    for index in range(count):
        if round((index + 1) * rule_fraction) > round(index * rule_fraction):
            labels.append('rule')
            continue
        labels.append(model_ids[model_index % len(model_ids)])
        model_index += 1
    return labels


def _even_seed(rng: Random) -> int:
    return rng.randrange(0, _SEED31 >> 1) << 1


def _sample_fixed_lineup(seed, character_ids, *, deck_id, name):
    rng = Random(int(seed))
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
    return assert_public_encoder_deck({
        'id': deck_id,
        'name': name,
        'character_ids': list(character_ids),
        'card_ids': card_ids,
    })


def _bundled_serving_build(preset_id: str) -> dict[str, Any]:
    cached = _SERVING_BUILDS.get(preset_id)
    if cached is not None:
        return cached
    path = _SERVING_DIR / f'{preset_id}.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path.name} must be a JSON object')
    raw = payload.get('serving_build')
    if not isinstance(raw, dict):
        raise ValueError(f'{path.name} is missing a serving_build object')
    build = fixed_team_serving_deck(raw, preset_id)
    _SERVING_BUILDS[preset_id] = build
    return build


def _validate_historical_builds(historical_builds):
    if historical_builds is None:
        return {}
    if not isinstance(historical_builds, Mapping):
        raise ValueError('historical_builds must be a mapping of preset to decks')
    validated: dict[str, list[dict[str, Any]]] = {}
    for preset_id, decks in historical_builds.items():
        if preset_id not in PUBLIC_PRESETS:
            raise ValueError(f'historical_builds preset must be one of {PUBLIC_PRESETS}')
        if not isinstance(decks, Sequence) or isinstance(decks, (str, bytes)):
            raise ValueError(f'historical_builds[{preset_id!r}] must be a list of decks')
        official = preset_opponent_deck(preset_id)
        validated_decks = []
        for deck in decks:
            payload = assert_public_encoder_deck(deck)
            if payload['character_ids'] != official['character_ids']:
                raise ValueError('historical builds must preserve preset characters and order')
            validated_decks.append(deepcopy(payload))
        validated[preset_id] = validated_decks
    return validated


def _require_pairs(value):
    if type(value) is not int or value < 1:
        raise ValueError('pairs_per_preset must be an integer >= 1')
    return value


def _require_int(value, name):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    return value


def _require_fraction(value, name):
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number in [0, 1)')
    number = float(value)
    if not isfinite(number) or number < 0 or number >= 1:
        raise ValueError(f'{name} must be in [0, 1)')
    return number


def _require_model_ids(model_ids):
    if isinstance(model_ids, (str, bytes)) or not isinstance(model_ids, Sequence):
        raise ValueError('model_ids must be a non-empty sequence of model ids')
    labels = list(model_ids)
    if not labels:
        raise ValueError('model_ids must not be empty')
    normalized = []
    for label in labels:
        if not isinstance(label, str) or not label or label == 'rule':
            raise ValueError('model_ids must be explicit non-empty ids, not rule')
        normalized.append(label)
    return tuple(normalized)
