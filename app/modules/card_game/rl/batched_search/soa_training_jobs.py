"""Pure metadata binding for real seven_round jobs; never a device driver.

Folder/key sources bind independently, including historical opponents with the
same team key. Initializers preserve mulligans and source job order. The actual
network/horizon-three contract is retained; this adapter grants no execution,
capacity, learning or serving qualification and performs no model loading.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from ..seven_schedule import ROSTER


_SIDES = ('a', 'b')
_PROFILE = dict(search_algorithm='gumbel', teacher_mode='network', q_scale='natural_wdl',
                simulations=32, gumbel_candidates=16, terminal_horizon=3,
                max_depth=10, max_actions=800)
_FIELDS = frozenset(('id', 'seed', 'first', 'left', 'right', 'policies', 'decks',
    'pure', 'training', 'train_sides', 'deadline', 'value_precheck', 'collection_replica',
    'covered_value_source', 'runtime', 'first_side', 'skip_mulligan', 'escalation',
    'exploration', 'guided_policy', 'zhenhong_passive_reward', 'setup_rewards', *_PROFILE))


def _source(folder, key):
    if not isinstance(folder, str) or not folder.strip() or '\0' in folder or key not in ROSTER:
        raise ValueError('Explicit known seven-team folder/key source required')
    return str(Path(folder).resolve())


def policy_source_alias(folder, key):
    """Match training_rollout.source_alias without importing its execution graph."""
    return key + '@' + sha256(_source(folder, key).encode()).hexdigest()


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Exact lowercase SHA256 required: ' + label)
    return value


def _build(value):
    from ..build_acceptance import canonical_build
    result = canonical_build(value)
    return {key: result[key] for key in ('character_ids', 'card_ids')}


def _models(models):
    from ..seven_policy import SCHEMA
    if not isinstance(models, Mapping) or not models:
        raise ValueError('Explicit loaded model source bank required')
    result = {}
    teams = set()
    for alias, model in models.items():
        manifest = getattr(model, 'manifest', None)
        if not isinstance(manifest, Mapping) or manifest.get('deck') not in ROSTER:
            raise ValueError('Known seven-team model manifest required')
        if manifest.get('schema') != SCHEMA:
            raise ValueError('Exact seven-policy schema required; models are never migrated')
        key = manifest['deck']
        folder = _source(getattr(model, 'directory', None), key)
        if alias != policy_source_alias(folder, key):
            raise ValueError('Model bank alias differs from its exact folder/key source')
        version = _hash(getattr(model, 'version', None), 'model version')
        if manifest.get('sha256') != version:
            raise ValueError('Model version differs from loaded manifest')
        serving = getattr(model, 'serving_deck', None)
        build = _build(serving)
        if build != _build(manifest.get('build')):
            raise ValueError('Loaded model build differs from manifest')
        from ..league_schema import build_hash
        if _hash(manifest.get('build_sha256'), 'model build') != build_hash(serving):
            raise ValueError('Loaded model build hash differs from manifest')
        result[alias] = dict(key=key, folder=folder, version=version, build=build,
                             build_sha256=manifest['build_sha256'])
        teams.add(key)
    if teams != set(ROSTER):
        raise ValueError('The fixed seven-team model bank must include all seven teams')
    return result


@dataclass(frozen=True)
class SoATrainingJobs:
    jobs: tuple
    initializers: tuple
    inference_keys: tuple
    model_indices: tuple
    model_versions: tuple
    keys: tuple
    train_masks: tuple
    identities: tuple
    search: dict
    formal_training_qualified: bool = False
    gpu_qualified: bool = False


def adapt_training_jobs(jobs, models):
    """Bind formal job metadata without changing jobs, builds or execution mode.

    ``models`` is the alias-keyed bank returned by training_rollout.models_for_jobs
    (with every fixed team represented). Inference uses its sorted aliases;
    collection row keys remain the original team names. No missing source,
    unsupported search configuration or duplicate physical game is repaired.
    """
    if not isinstance(jobs, (tuple, list)) or not 1 <= len(jobs) <= 1600:
        raise ValueError('One to 1600 explicit seven-round jobs required')
    bank = _models(models)
    inference_keys = tuple(sorted(bank))
    indices = {alias: index for index, alias in enumerate(inference_keys)}
    seen_ids, seen_games = set(), set()
    copied, initializers, model_indices, versions, keys, masks, identities = ([] for _ in range(7))
    required = {'id', 'seed', 'first', 'left', 'right', 'policies', 'decks',
                'pure', 'training', 'train_sides', 'deadline'}
    for job in jobs:
        if not isinstance(job, dict) or not required <= set(job) or set(job) - _FIELDS:
            raise ValueError('Exact supported seven_round job fields required')
        gid = job['id']
        if not isinstance(gid, str) or not gid.strip() or '\0' in gid or gid in seen_ids:
            raise ValueError('Unique nonempty training job ID required')
        seen_ids.add(gid)
        seed = job['seed']
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Nonnegative int64 training seed required')
        if job['first'] not in _SIDES or job.get('first_side', job['first']) != job['first']:
            raise ValueError('Exact first/first_side identity required')
        if type(job['pure']) is not bool or job['pure'] is not False or type(job['training']) is not bool:
            raise ValueError('Explicit boolean non-pure search collection required')
        for field, expected in (('skip_mulligan', False), ('escalation', True)):
            if field in job and (type(job[field]) is not bool or job[field] is not expected):
                raise ValueError('Unsupported initializer flag: ' + field)
        if 'exploration' in job and type(job['exploration']) is not bool:
            raise ValueError('Boolean exploration control required')
        deadline = job['deadline']
        if type(deadline) not in (int, float) or not 0 < deadline < 2**63 or not math.isfinite(deadline):
            raise ValueError('Finite positive original wall deadline required')
        sides = job['train_sides']
        if (not isinstance(sides, (tuple, list)) or not sides or
                any(side not in _SIDES for side in sides) or len(set(sides)) != len(sides)):
            raise ValueError('Distinct known training sides required')
        for field, expected in _PROFILE.items():
            actual = job.get(field, expected)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError('Unsupported formal search setting: ' + field)
        if job.get('runtime', 'seven') != 'seven':
            raise ValueError('Only the seven runtime is supported')
        if any(job.get(field) is not None for field in ('guided_policy', 'covered_value_source')):
            raise ValueError('Unsupported guide or covered teacher source')
        if job.get('zhenhong_passive_reward') is not None and (
                type(job['zhenhong_passive_reward']) not in (int, float) or job['zhenhong_passive_reward'] != 0):
            raise ValueError('Unsupported shaped reward')
        if job.get('setup_rewards') is not None and job.get('setup_rewards') is not False:
            raise ValueError('Unsupported setup rewards')
        if 'collection_replica' in job and (type(job['collection_replica']) is not int or job['collection_replica'] < 0):
            raise ValueError('Nonnegative integer collection replica required')
        for name in ('policies', 'decks'):
            if not isinstance(job[name], Mapping) or set(job[name]) != set(_SIDES):
                raise ValueError('Both exact side ' + name + ' required')
        row_keys, row_versions, row_models, bindings = {}, {}, [], {}
        for side, key in (('a', job['left']), ('b', job['right'])):
            if key not in ROSTER:
                raise ValueError('Unknown seven-team job identity')
            spec = job['policies'][side]
            if not isinstance(spec, tuple) or len(spec) != 2 or spec[1] != key:
                raise ValueError('Side policy key differs from left/right team')
            alias = policy_source_alias(spec[0], key)
            if alias not in bank:
                raise ValueError('Missing exact side model source')
            binding = bank[alias]
            build = _build(job['decks'][side])
            if list(job['decks'][side]['card_ids']) != build['card_ids']:
                raise ValueError('Canonical job card order required; initializer decks are never reordered')
            if build != binding['build']:
                raise ValueError('Job build differs from the fixed model build')
            row_keys[side], row_versions[side] = key, binding['version']
            row_models.append(indices[alias])
            bindings[side] = dict(alias=alias, **deepcopy(binding))
        physical = json.dumps([seed, job['first'], [bindings[s] for s in _SIDES]],
                              sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        if physical in seen_games:
            raise ValueError('Duplicate physical job under another ID')
        seen_games.add(physical)
        copied.append(deepcopy(job))
        initializers.append(dict(seed=seed, decks=deepcopy(job['decks']), first_side=job['first'],
                                 skip_mulligan=False, escalation=True))
        model_indices.append(tuple(row_models)); versions.append(row_versions); keys.append(row_keys)
        masks.append(tuple(side in sides for side in _SIDES))
        identities.append(dict(id=gid, seed=seed, first=job['first'], left=job['left'], right=job['right'],
            model_versions=deepcopy(row_versions), policy_bindings=bindings,
            train_sides=deepcopy(sides), training=job['training'], pure=False,
            exploration=job.get('exploration', job['training']), deadline=deadline,
            collection_replica=job.get('collection_replica'), search=deepcopy(_PROFILE)))
    return SoATrainingJobs(tuple(copied), tuple(initializers), inference_keys, tuple(model_indices),
        tuple(versions), tuple(keys), tuple(masks), tuple(identities), deepcopy(_PROFILE))
