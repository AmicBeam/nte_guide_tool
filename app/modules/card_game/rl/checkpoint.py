"""Checkpoint metadata guards. Incompatible weights must fail closed."""
from __future__ import annotations

from typing import Any, Mapping
import hashlib
import json
from pathlib import Path

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS

from .contracts import ACTION_CAPACITY, ACTION_TYPES, DESIGN_VERSION, RULES_VERSION
from .encoding import ACTION_DIM, ENCODER_VERSION, OBSERVATION_DIM, encoder_metadata

REQUIRED_METADATA_KEYS = (
    'schema_version', 'policy_type', 'catalog_sha256', 'form_card_ids',
    'rules_version',
    'design_version',
    'encoder_version',
    'observation_dim',
    'action_dim',
    'action_capacity',
    'action_types',
    'character_ids',
    'card_ids',
)


def current_checkpoint_config() -> dict[str, Any]:
    meta = encoder_metadata()
    return {
        'schema_version': 1,
        'policy_type': 'MaskablePPO/CandidateScoringPolicy/generic_plus_card_v1',
        'catalog_sha256': hashlib.sha256(json.dumps({
            'cards': {cid: CARDS[cid] for cid in meta['card_ids']},
            'characters': {cid: CHARACTERS[cid] for cid in meta['character_ids']},
        }, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'form_card_ids': meta['form_card_ids'],
        'rules_version': RULES_VERSION,
        'design_version': DESIGN_VERSION,
        'encoder_version': ENCODER_VERSION,
        'feature_version': meta['feature_version'],
        'action_feature_version': meta['action_feature_version'],
        'observation_dim': OBSERVATION_DIM,
        'action_dim': ACTION_DIM,
        'action_capacity': ACTION_CAPACITY,
        'action_types': list(ACTION_TYPES),
        'character_ids': list(meta['character_ids']),
        'card_ids': list(meta['card_ids']),
        'weights': None,
        'trainable': False,
    }


def _same_sequence(actual: Any, expected: Any, label: str) -> None:
    if not isinstance(actual, list) or actual != expected:
        raise ValueError(f'Checkpoint {label} is incompatible')


def validate_checkpoint(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError('Checkpoint metadata must be a JSON object')
    payload = current_checkpoint_config() if metadata is None else dict(metadata)
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError('Checkpoint metadata missing fields: ' + ', '.join(missing))
    expected = current_checkpoint_config()
    for key in ('rules_version', 'design_version', 'encoder_version', 'policy_type', 'catalog_sha256'):
        if payload[key] != expected[key]:
            raise ValueError(f'Checkpoint {key} is incompatible: {payload[key]!r}')
    for key in ('schema_version', 'observation_dim', 'action_dim', 'action_capacity'):
        if type(payload[key]) is not int or payload[key] != expected[key]:
            raise ValueError(f'Checkpoint {key} is incompatible: {payload[key]!r}')
    _same_sequence(payload['action_types'], expected['action_types'], 'action_types')
    _same_sequence(payload['character_ids'], expected['character_ids'], 'character_ids')
    _same_sequence(payload['card_ids'], expected['card_ids'], 'card_ids')
    _same_sequence(payload['form_card_ids'], expected['form_card_ids'], 'form_card_ids')
    if payload.get('trainable', False) is not False:
        raise ValueError('Training is disabled for this framework release')
    if payload.get('weights') not in (None, False, '', {}):
        raise ValueError('This framework round has no compatible trained weights')
    return expected


def load_checkpoint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a metadata-only checkpoint. Does not deserialize model weights."""
    return validate_checkpoint(metadata)


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.stat().st_size > 1024 * 1024:
        raise ValueError('Checkpoint metadata exceeds 1 MiB')
    return validate_checkpoint(json.loads(source.read_text(encoding='utf-8')))


def write_checkpoint_metadata(path: str | Path, metadata=None) -> None:
    """Write JSON metadata only; no weights, replay buffers, or game states."""
    payload = validate_checkpoint(metadata)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def assert_compatible_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Compare catalog/encoder identity without requiring inspect-only flags."""
    if not isinstance(metadata, Mapping):
        raise ValueError('Checkpoint metadata must be a JSON object')
    payload = dict(metadata)
    expected = current_checkpoint_config()
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError('Checkpoint metadata missing fields: ' + ', '.join(missing))
    for key in ('rules_version', 'design_version', 'encoder_version', 'policy_type', 'catalog_sha256'):
        if payload[key] != expected[key]:
            raise ValueError(f'Checkpoint {key} is incompatible: {payload[key]!r}')
    for key in ('schema_version', 'observation_dim', 'action_dim', 'action_capacity'):
        if type(payload[key]) is not int or payload[key] != expected[key]:
            raise ValueError(f'Checkpoint {key} is incompatible: {payload[key]!r}')
    _same_sequence(payload['action_types'], expected['action_types'], 'action_types')
    _same_sequence(payload['character_ids'], expected['character_ids'], 'character_ids')
    _same_sequence(payload['card_ids'], expected['card_ids'], 'card_ids')
    _same_sequence(payload['form_card_ids'], expected['form_card_ids'], 'form_card_ids')
    return expected


def write_training_checkpoint(directory: str | Path, model, stats: Mapping[str, Any] | None = None) -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / 'model.zip'
    model.save(str(weights))
    payload = current_checkpoint_config()
    payload['trainable'] = True
    payload['weights'] = 'model.zip'
    payload['num_timesteps'] = int(getattr(model, 'num_timesteps', 0) or 0)
    payload['n_updates'] = int(getattr(model, '_n_updates', 0) or 0)
    if stats:
        payload.update(dict(stats))
    (folder / 'checkpoint.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return weights


def load_training_checkpoint(directory: str | Path, *, env=None, device: str = 'auto'):
    folder = Path(directory)
    meta_path = folder / 'checkpoint.json'
    payload = json.loads(meta_path.read_text(encoding='utf-8'))
    assert_compatible_identity(payload)
    weights_name = payload.get('weights') or 'model.zip'
    weights = folder / str(weights_name)
    if not weights.is_file():
        raise ValueError(f'Training checkpoint is missing weights: {weights}')
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(weights), env=env, device=device)
    return model, payload
