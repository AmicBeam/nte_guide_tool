"""Explicit recovery vs seven GPU observation/candidate contracts.

Historical five-team buffers stay 2420/254. Seven uses a separate versioned
width and named-feature order from seven_lineup; this module never resizes
recovery tensors in place. Preset keys come from next_training.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.modules.card_game.rl import cross_lineup, league_schema, seven_grounded, seven_lineup, seven_policy
from app.modules.card_game.rl.league_schema import OPTION_IDS

RECOVERY_RUNTIME = 'recovery'
SEVEN_RUNTIME = 'seven'
GPU_RUNTIMES = (RECOVERY_RUNTIME, SEVEN_RUNTIME)
DEFAULT_RUNTIME = RECOVERY_RUNTIME

_NEXT_TRAINING = (
    Path(__file__).resolve().parents[2] / 'content' / 'duel_v2' / 'next_training.json'
)


class GpuSchemaContractError(ValueError):
    """Fail-closed GPU runtime/schema mismatch."""


def normalize_runtime(runtime: str | None = None) -> str:
    if runtime is None:
        return DEFAULT_RUNTIME
    if runtime not in GPU_RUNTIMES:
        raise GpuSchemaContractError(
            "Unknown GPU runtime %r; expected 'recovery' or 'seven'" % (runtime,)
        )
    return runtime


def next_training_keys() -> tuple[str, ...]:
    payload = json.loads(_NEXT_TRAINING.read_text())
    roster = payload.get('roster')
    learning = payload.get('learning_keys')
    if not isinstance(roster, list) or not roster or any(not isinstance(item, str) or not item for item in roster):
        raise GpuSchemaContractError('next_training.json roster keys are required')
    keys = tuple(roster)
    if isinstance(learning, list) and tuple(learning) != keys:
        raise GpuSchemaContractError('next_training.json learning_keys must match roster order')
    return keys


def _named_sha(schema: str, width: int, names: list[str]) -> str:
    return hashlib.sha256(('%s:%s:%s' % (schema, width, ','.join(names))).encode('utf-8')).hexdigest()


def _recovery_contract() -> dict[str, Any]:
    features = list(cross_lineup.all_features())
    candidates = list(cross_lineup.candidate_names())
    obs_dim = 2420
    cand_dim = int(cross_lineup.CAND_DIM)
    if len(features) != obs_dim:
        raise GpuSchemaContractError('Recovery observation width drifted from frozen 2420')
    if len(candidates) != cand_dim or cand_dim != 254:
        raise GpuSchemaContractError('Recovery candidate width drifted from frozen 254')
    return {
        'runtime': RECOVERY_RUNTIME,
        'schema': cross_lineup.SCHEMA,
        'observation_schema': cross_lineup.SCHEMA,
        'policy_schema': None,
        'grounded_schema': None,
        'keys': tuple(cross_lineup.KEYS),
        'seats': tuple(cross_lineup.SEATS),
        'card_ids': tuple(cross_lineup.CARD_IDS),
        'option_ids': tuple(OPTION_IDS),
        'action_types': ('end_turn', 'attack', 'play_card', 'ultimate', 'mulligan', 'choose'),
        'feature_names': features,
        'candidate_names': candidates,
        'obs_dim': obs_dim,
        'cand_dim': cand_dim,
        'prefix_obs_dim': obs_dim,
        'old_cand_dim': cand_dim,
        'num_old_cards': len(league_schema.OLD_CARDS),
        'sampler': 'app.modules.card_game.rl.recovery_runtime',
        'sampler_contract': 'existing recovery runtime; additional catalog rosters are not implicitly supported',
        'schema_sha': _named_sha(cross_lineup.SCHEMA, obs_dim, features),
        'candidate_schema_sha': _named_sha(cross_lineup.SCHEMA, cand_dim, candidates),
    }


def _seven_contract() -> dict[str, Any]:
    keys = next_training_keys()
    if keys != tuple(seven_lineup.KEYS):
        raise GpuSchemaContractError(
            'seven_lineup.KEYS must match next_training.json roster: %s vs %s' % (seven_lineup.KEYS, keys)
        )
    features = list(seven_lineup.all_features())
    candidates = list(seven_lineup.candidate_names())
    prefix = list(seven_lineup._old_all_features())
    obs_dim = len(features)
    cand_dim = int(seven_lineup.CAND_DIM)
    old_cand_dim = int(seven_lineup.OLD_CAND_DIM)
    if len(candidates) != cand_dim:
        raise GpuSchemaContractError('Seven candidate names/width mismatch')
    if len(prefix) != 2912:
        raise GpuSchemaContractError('Seven prefix observation width drifted from 2912')
    return {
        'runtime': SEVEN_RUNTIME,
        'schema': seven_lineup.SCHEMA,
        'observation_schema': seven_lineup.SCHEMA,
        'policy_schema': seven_policy.SCHEMA,
        'grounded_schema': seven_grounded.SCHEMA,
        'grounded_dim': int(seven_grounded.DIM),
        'keys': keys,
        'seats': tuple(seven_lineup.SEATS),
        'card_ids': tuple(seven_lineup.CARD_IDS),
        'option_ids': tuple(OPTION_IDS),
        'action_types': tuple(seven_lineup.ACTION_TYPES),
        'feature_names': features,
        'candidate_names': candidates,
        'obs_dim': obs_dim,
        'cand_dim': cand_dim,
        'prefix_obs_dim': len(prefix),
        'old_cand_dim': old_cand_dim,
        'new_side': tuple(seven_lineup.NEW_SIDE),
        'new_char': tuple(seven_lineup.NEW_CHAR),
        'attributes': tuple(seven_lineup.ATTRIBUTES),
        'choices': tuple(seven_lineup.CHOICES),
        'order_slots': int(seven_lineup.ORDER_SLOTS),
        'num_old_cards': len(league_schema.OLD_CARDS),
        'sampler': 'app.modules.card_game.rl.seven_runtime',
        'sampler_contract': 'explicit seven runtime; existing recovery sampler is not replaced',
        'schema_sha': _named_sha(seven_lineup.SCHEMA, obs_dim, features),
        'candidate_schema_sha': _named_sha(seven_lineup.SCHEMA, cand_dim, candidates),
    }


GPU_RUNTIME_CONTRACTS: dict[str, dict[str, Any]] = {
    RECOVERY_RUNTIME: _recovery_contract(),
    SEVEN_RUNTIME: _seven_contract(),
}


def runtime_contract(runtime: str | None = None) -> dict[str, Any]:
    name = normalize_runtime(runtime)
    return dict(GPU_RUNTIME_CONTRACTS[name])


def runtime_metadata(runtime: str | None = None) -> dict[str, Any]:
    contract = runtime_contract(runtime)
    return {
        'runtime': contract['runtime'],
        'schema': contract['schema'],
        'observation_schema': contract['observation_schema'],
        'policy_schema': contract['policy_schema'],
        'grounded_schema': contract.get('grounded_schema'),
        'keys': list(contract['keys']),
        'seats': list(contract['seats']),
        'card_ids': list(contract['card_ids']),
        'action_types': list(contract['action_types']),
        'obs_dim': contract['obs_dim'],
        'cand_dim': contract['cand_dim'],
        'prefix_obs_dim': contract['prefix_obs_dim'],
        'old_cand_dim': contract['old_cand_dim'],
        'feature_names': list(contract['feature_names']),
        'candidate_names': list(contract['candidate_names']),
        'schema_sha': contract['schema_sha'],
        'candidate_schema_sha': contract['candidate_schema_sha'],
        'sampler_contract': contract['sampler_contract'],
        'sampler': contract['sampler'],
    }
