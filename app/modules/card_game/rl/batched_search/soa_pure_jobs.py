"""Pure metadata for real value-calibration and paired argmax jobs.

Search settings on the original jobs remain inactive historical metadata.
No rules, models, scope, driver or calibration evidence are executed/created.
Policy and reference banks retain independent exact source/version indices.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math

from ..seven_schedule import ROSTER
from .soa_training_jobs import _FIELDS, _build, _models, policy_source_alias


_SIDES = ('a', 'b')
_COMPARISON = frozenset(('key', 'replica', 'label', 'learner', 'references'))
_EXECUTION = dict(algorithm='argmax', counts_as_search=False, simulations=0,
                  teacher_used=False, world_sampling=False)


@dataclass(frozen=True)
class SoAPureJobs:
    jobs: tuple
    initializers: tuple
    inference_keys: tuple
    model_indices: tuple
    model_versions: tuple
    reference_inference_keys: tuple
    reference_indices: tuple
    reference_versions: tuple
    keys: tuple
    train_masks: tuple
    identities: tuple
    execution: dict
    learning_eligible: bool = False
    formal_training_qualified: bool = False
    gpu_qualified: bool = False


def adapt_pure_jobs(jobs, models, *, reference_models=None):
    """Bind jobs_for(pure=True) or compare_jobs(search=False), without migration.

    Both model arguments are alias-keyed loaded seven-team banks. Comparison
    jobs require the separate reference bank explicitly, even if exported
    sources happen to coincide with policy sources. No precheck is required or
    invented for pure trajectories used to establish their first calibration.
    """
    if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= 1600:
        raise ValueError('One to 1600 explicit pure seven-team jobs required')
    bank = _models(models)
    references = _models(reference_models) if reference_models is not None else {}
    for alias in set(bank) & set(references):
        if bank[alias] != references[alias]:
            raise ValueError('Same source differs between policy/reference snapshots')
    inference_keys, reference_keys = tuple(sorted(bank)), tuple(sorted(references))
    indices = {key: i for i, key in enumerate(inference_keys)}
    ref_indices = {key: i for i, key in enumerate(reference_keys)}
    required = frozenset(('id', 'seed', 'first', 'left', 'right', 'policies', 'decks',
                          'pure', 'training', 'train_sides', 'deadline'))
    ids, physical = set(), set()
    copied, initializers, policy_ids, versions, ref_ids, ref_versions, row_keys, masks, identities = ([] for _ in range(9))
    for job in jobs:
        if not isinstance(job, dict) or not required <= set(job) or set(job) - (_FIELDS | _COMPARISON):
            raise ValueError('Exact supported pure job fields required')
        gid = job['id']
        if not isinstance(gid, str) or not gid.strip() or '\0' in gid or gid in ids:
            raise ValueError('Distinct nonempty pure job ID required')
        ids.add(gid)
        if job['pure'] is not True or job['training'] is not False:
            raise ValueError('Pure argmax jobs must explicitly disable search training')
        seed = job['seed']
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Nonnegative int64 seed required')
        if job['first'] not in _SIDES or job.get('first_side', job['first']) != job['first']:
            raise ValueError('Exact first-side identity required')
        deadline = job['deadline']
        if type(deadline) not in (int, float) or not 0 < deadline < 2**63 or not math.isfinite(deadline):
            raise ValueError('Finite original positive wall deadline required')
        sides = job['train_sides']
        if (not isinstance(sides, (list, tuple)) or not sides or
                any(side not in _SIDES for side in sides) or len(set(sides)) != len(sides)):
            raise ValueError('Distinct original training-viewer sides required')
        for field, default in (('skip_mulligan', False), ('escalation', True)):
            if field in job and (type(job[field]) is not bool or job[field] is not default):
                raise ValueError('Unsupported pure initializer flag: '+field)
        if 'exploration' in job and (type(job['exploration']) is not bool or job['exploration']):
            raise ValueError('Pure argmax cannot request exploration')
        if job.get('runtime', 'seven') != 'seven' or job.get('guided_policy') is not None:
            raise ValueError('Only unguided seven-team pure policies are supported')
        if job.get('setup_rewards') is not None and job.get('setup_rewards') is not False:
            raise ValueError('Pure evaluation cannot request setup rewards')
        reward = job.get('zhenhong_passive_reward')
        if reward is not None and (type(reward) not in (int, float) or reward != 0):
            raise ValueError('Pure evaluation cannot request shaped rewards')
        for field in ('collection_replica',):
            if field in job and (type(job[field]) is not int or job[field] < 0):
                raise ValueError('Nonnegative collection replica required')
        # These fields are validated and retained, never treated as an actual
        # Gumbel budget/teacher execution on this pure path.
        for field, expected in (('simulations', 32), ('gumbel_candidates', 16),
                                ('terminal_horizon', 3), ('max_depth', 10), ('max_actions', 800)):
            if field in job and (type(job[field]) is not int or job[field] != expected):
                raise ValueError('Unsupported residual pure job setting: '+field)
        if job.get('teacher_mode', 'network') not in ('network', 'covered_value'):
            raise ValueError('Unsupported residual teacher metadata')
        if job.get('q_scale', 'natural_wdl') != 'natural_wdl' or job.get('search_algorithm', 'gumbel') != 'gumbel':
            raise ValueError('Unsupported residual search metadata')
        source = job.get('covered_value_source')
        if source is not None and (not isinstance(source, str) or not source.strip() or '\0' in source):
            raise ValueError('Explicit residual covered-source string required')
        if job.get('world_sampling', 'per_simulation') != 'per_simulation':
            raise ValueError('Unsupported residual world-sampling metadata')
        for field in ('policies', 'decks'):
            if not isinstance(job[field], Mapping) or set(job[field]) != set(_SIDES):
                raise ValueError('Both exact side '+field+' required')
        bindings, keys, policy_row, version_row = {}, {}, [], {}
        for side, key in (('a', job['left']), ('b', job['right'])):
            if key not in ROSTER:
                raise ValueError('Known fixed seven-team side identity required')
            spec = job['policies'][side]
            if not isinstance(spec, tuple) or len(spec) != 2 or spec[1] != key:
                raise ValueError('Pure policy source differs from side team identity')
            alias = policy_source_alias(*spec)
            if alias not in bank or _build(job['decks'][side]) != bank[alias]['build']:
                raise ValueError('Missing exact pure model or changed fixed build')
            bindings[side] = dict(alias=alias, **deepcopy(bank[alias]))
            keys[side], version_row[side] = key, bank[alias]['version']
            policy_row.append(indices[alias])
        comparison = bool(set(job) & _COMPARISON)
        reference_bindings, reference_row, reference_version_row = {}, [-1, -1], {}
        role = None
        if comparison:
            if not _COMPARISON <= set(job):
                raise ValueError('All paired comparison identity fields required')
            key, learner, label, replica = (job[k] for k in ('key', 'learner', 'label', 'replica'))
            if (key not in ROSTER or learner not in _SIDES or keys[learner] != key
                    or label not in ('candidate', 'reference') or type(replica) is not int or replica < 0
                    or not isinstance(job['references'], Mapping) or set(job['references']) != {learner}):
                raise ValueError('Exact original paired-comparison role/reference required')
            if reference_models is None:
                raise ValueError('Explicit independent reference source bank required')
            spec = job['references'][learner]
            if not isinstance(spec, tuple) or len(spec) != 2 or spec[1] != key:
                raise ValueError('Comparison reference key differs from learner key')
            alias = policy_source_alias(*spec)
            if alias not in references or references[alias]['build'] != bindings[learner]['build']:
                raise ValueError('Missing exact comparison reference source/build')
            reference_bindings[learner] = dict(alias=alias, **deepcopy(references[alias]))
            reference_row[int(learner == 'b')] = ref_indices[alias]
            reference_version_row[learner] = references[alias]['version']
            enemy = 'b' if learner == 'a' else 'a'
            if bindings[enemy]['folder'] != references[alias]['folder']:
                raise ValueError('Paired opponent differs from the fixed reference folder')
            if label == 'reference' and bindings[learner] != reference_bindings[learner]:
                raise ValueError('Reference-labelled policy differs from the actual reference snapshot')
            role = dict(key=key, label=label, learner=learner)
        fingerprint = json.dumps([seed, job['first'], bindings,
            {s: [job['decks'][s]['character_ids'], job['decks'][s]['card_ids']] for s in _SIDES}, role],
            sort_keys=True, separators=(',', ':'))
        if fingerprint in physical:
            raise ValueError('Duplicate physical pure job under another identity')
        physical.add(fingerprint)
        copied.append(deepcopy(job))
        initializers.append(dict(seed=seed, first_side=job['first'], decks=deepcopy(job['decks']),
                                 skip_mulligan=False, escalation=True))
        policy_ids.append(tuple(policy_row)); versions.append(version_row)
        ref_ids.append(tuple(reference_row)); ref_versions.append(reference_version_row)
        row_keys.append(keys); masks.append(tuple(s in sides for s in _SIDES))
        identity = {name: deepcopy(job[name]) for name in ('id', 'seed', 'first', 'left', 'right',
            'pure', 'training', 'train_sides', 'deadline')}
        identity.update(policy_bindings=bindings, reference_bindings=reference_bindings,
                        execution=deepcopy(_EXECUTION))
        if comparison:
            identity.update({name: deepcopy(job[name]) for name in _COMPARISON})
        identities.append(identity)
    return SoAPureJobs(tuple(copied), tuple(initializers), inference_keys, tuple(policy_ids), tuple(versions),
        reference_keys, tuple(ref_ids), tuple(ref_versions), tuple(row_keys), tuple(masks),
        tuple(identities), deepcopy(_EXECUTION))
