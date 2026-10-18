"""Pure broad rule-value plans and SoA metadata; no execution or qualification.

Explicit seven-team decks are retained as the starting builds. The prescribed
variations preserve the original planner's card order and paired seed schedule.
No model, auxiliary scope, rules rollout, or readiness evidence is created.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math

import numpy as np

from .build_acceptance import canonical_build
from .seven_schedule import ROSTER, learning_schedule
from ..content.duel_v2 import CARDS, validate_deck


AUXILIARY_DECISION_LIMIT = 512


_FIELDS = frozenset(('seed', 'first', 'keys', 'decks', 'deadline'))
_EXECUTION = dict(policy='visible_rule', counts_as_search=False, simulations=0,
                  exploration_steps=6, exploration_phase='playing',
                  exploration_rng='numpy_pcg64_bounded_integers', value_start_step=6,
                  max_decisions=AUXILIARY_DECISION_LIMIT)


def _deadline(value):
    if type(value) not in (int, float) or not 0 < value < 2**63 or not math.isfinite(value):
        raise ValueError('Original finite positive absolute deadline required')
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('Nonnegative int64 auxiliary seed required')
    return value


def _deck(value):
    if not isinstance(value, dict) or set(value) != {'id', 'name', 'character_ids', 'card_ids'}:
        raise ValueError('Exact validated auxiliary build schema required')
    return validate_deck(deepcopy(value))


def auxiliary_jobs(base, replicas, *, decks, deadline):
    """Return the unchanged 56-pair broad-build plan for each replica.

    ``deadline`` is already the desired absolute job deadline (the CLI passes
    its original end minus30). This function neither reads clocks nor reserves
    seeds. Explicit starting decks replace no caller data and are never mutated.
    """
    _seed(base); _deadline(deadline)
    if type(replicas) is not int or replicas not in (1, 2, 3, 6, 9, 12):
        raise ValueError('Supported auxiliary replica count required')
    if base + (replicas - 1) * 100 + 27 >= 2**63:
        raise ValueError('Auxiliary paired seed range exceeds int64')
    if not isinstance(decks, Mapping) or set(decks) != set(ROSTER):
        raise ValueError('All seven explicit starting builds required')
    decks = {key: _deck(decks[key]) for key in ROSTER}
    plan = learning_schedule() + [dict(left=k, right=k, first=f) for k in ROSTER for f in ('a', 'b')]
    result = []
    for replica in range(replicas):
        for i, item in enumerate(plan):
            seed = base + replica * 100 + i // 2; rng = np.random.default_rng(seed); builds = {}
            for side, key in (('a', item['left']), ('b', item['right'])):
                cards = []
                for hero in decks[key]['character_ids']:
                    kit = [c for c, v in CARDS.items() if v['character_id'] == hero and not v.get('derived')]
                    if (replica + i // 2) % 3 == 0:
                        cards += kit
                    else:
                        bag = [c for c in kit for _ in range(2)]
                        cards += [bag[int(j)] for j in rng.choice(len(bag), 8, replace=False)]
                builds[side] = canonical_build(dict(decks[key], card_ids=cards))
            result.append(dict(seed=seed, first=item['first'], keys={'a': item['left'], 'b': item['right']},
                               decks=builds, deadline=deadline))
    return result


@dataclass(frozen=True)
class SoAAuxiliaryJobs:
    jobs: tuple
    initializers: tuple
    keys: tuple
    identities: tuple
    execution: dict
    starting_decks: dict
    starting_lineups: dict
    formal_training_qualified: bool = False
    gpu_qualified: bool = False


def adapt_auxiliary_jobs(jobs, *, starting_decks):
    """Bind real broad-build jobs without requiring neural policies or scope.

    Subsets are allowed for bounded collection. The original seed, initiative,
    team keys, ordered builds and wall deadline are retained. Each team must
    match the explicit frozen starting lineup, including character order.
    Legal card variations and original build names/IDs remain untouched.
    """
    if not isinstance(jobs, (list, tuple)) or not 1 <= len(jobs) <= 672:
        raise ValueError('One to672 explicit auxiliary jobs required')
    if not isinstance(starting_decks, Mapping) or set(starting_decks) != set(ROSTER):
        raise ValueError('All seven actual frozen starting builds required')
    starts = {key: _deck(starting_decks[key]) for key in ROSTER}
    lineups = {key: tuple(starts[key]['character_ids']) for key in ROSTER}
    copied, initializers, keys, identities, seen = [], [], [], [], set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != _FIELDS:
            raise ValueError('Exact original auxiliary job fields required')
        _seed(job['seed']); _deadline(job['deadline'])
        if job['first'] not in ('a', 'b'):
            raise ValueError('Explicit original first side required')
        if (not isinstance(job['keys'], Mapping) or set(job['keys']) != {'a', 'b'}
                or any(key not in ROSTER for key in job['keys'].values())):
            raise ValueError('Both actual seven-team keys required')
        if not isinstance(job['decks'], Mapping) or set(job['decks']) != {'a', 'b'}:
            raise ValueError('Both actual ordered auxiliary builds required')
        for side in ('a', 'b'):
            deck = _deck(job['decks'][side])
            if tuple(deck['character_ids']) != lineups[job['keys'][side]]:
                raise ValueError('Auxiliary team differs from its frozen ordered starting lineup')
        fingerprint = json.dumps([job['seed'], job['first'], job['keys'],
            {s: [job['decks'][s]['character_ids'], job['decks'][s]['card_ids']] for s in ('a', 'b')}],
            sort_keys=True, separators=(',', ':'))
        if fingerprint in seen:
            raise ValueError('Duplicate physical auxiliary job')
        seen.add(fingerprint)
        copied.append(deepcopy(job)); identities.append(deepcopy(job)); keys.append(deepcopy(job['keys']))
        initializers.append(dict(seed=job['seed'], first_side=job['first'], decks=deepcopy(job['decks']),
                                 skip_mulligan=False, escalation=True))
    return SoAAuxiliaryJobs(tuple(copied), tuple(initializers), tuple(keys), tuple(identities), deepcopy(_EXECUTION), deepcopy(starts), deepcopy(lineups))
