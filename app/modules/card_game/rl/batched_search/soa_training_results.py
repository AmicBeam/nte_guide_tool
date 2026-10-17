"""Verify completed numerical GPU labels after the batch, without writing files.

The CPU oracle checks supplied trajectories only. Missing GPU rows, policies,
search-profile receipts or viewer records are rejected, never regenerated.
This adapter supplies the existing seven-team game schema; it is not a driver
and cannot grant formal training or deployment qualification.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from .soa_training_jobs import SoATrainingJobs, adapt_training_jobs, policy_source_alias


def _oracle():
    from ...engine.duel_v2 import observe, acting_side, apply_action
    from ..league_rollout import clean
    from ..seven_lineup import encode
    return observe, acting_side, apply_action, clean, encode


def _check_deadline(deadline):
    if deadline is not None:
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError('Finite positive monotonic proof deadline required')
        if time.monotonic() >= deadline:
            raise TimeoutError('Postbatch training proof deadline exhausted')


def _coordinate(row, names, counts):
    if not isinstance(row, dict) or any(type(row.get(name)) is not int for name in names):
        raise ValueError('Exact integer GPU row coordinates required')
    key = tuple(row[name] for name in names)
    if not 0 <= key[0] < counts or key[1] < 0 or (len(key) == 3 and key[2] not in (0, 1)):
        raise ValueError('GPU row coordinates outside the submitted job batch')
    return key


def _indexed(rows, names, count):
    if not isinstance(rows, (tuple, list)):
        raise ValueError('Complete postbatch GPU row lists required')
    result = {}
    for row in rows:
        key = _coordinate(row, names, count)
        if key in result:
            raise ValueError('Duplicate GPU row coordinate')
        result[key] = row
    return result


def _same_graph(actual, expected):
    """Compare field order, entity/container types, values and alias bijection."""
    forward, backward = {}, {}
    def walk(left, right):
        if type(left) is not type(right):
            raise ValueError('Replay state scalar/container/entity type mismatch')
        if isinstance(left, (dict, list, tuple)):
            lid, rid = id(left), id(right)
            if forward.get(lid, rid) != rid or backward.get(rid, lid) != lid:
                raise ValueError('Replay state alias mismatch')
            if lid in forward:
                return
            forward[lid], backward[rid] = rid, lid
        if isinstance(left, dict):
            if list(left) != list(right):
                raise ValueError('Replay state field order mismatch')
            for key in left:
                walk(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise ValueError('Replay state sequence length mismatch')
            for a, b in zip(left, right):
                walk(a, b)
        elif left != right:
            raise ValueError('Replay state value/RNG mismatch')
    walk(actual, expected)


def _array(value, *, kind, ndim, shape=None):
    if (not isinstance(value, np.ndarray) or value.ndim != ndim
            or value.dtype.kind not in kind or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all()):
        raise ValueError('Invalid or missing GPU numerical array')
    return value


def _equal_array(actual, expected):
    if np.shape(actual) != np.shape(expected) or not np.array_equal(actual, expected):
        raise ValueError('GPU numerical array differs from independent postbatch proof')


def verified_training_games(dataset, *, initial, final, raw_records, metadata,
                            models, policy_reference, deadline=None):
    """Return all verified games, or raise without partial output or side effects.

    Dataset receipts must contain ``search`` and side-wise ``model_versions``.
    Every policy reference additionally contains the complete raw ``actions``
    in original order, preventing candidate-encoding collisions from hiding an
    action-order mismatch. Initializer correctness remains a driver prerequisite;
    this postbatch adapter starts from its supplied independent GPU snapshots.
    """
    _check_deadline(deadline)
    if not isinstance(metadata, SoATrainingJobs):
        raise TypeError('Real SoATrainingJobs metadata required')
    rebound = adapt_training_jobs(metadata.jobs, models)
    for name in ('initializers', 'inference_keys', 'model_indices', 'model_versions',
                 'keys', 'train_masks', 'identities', 'search'):
        if getattr(metadata, name) != getattr(rebound, name):
            raise ValueError('Training metadata or frozen model binding changed')
    count = len(metadata.jobs)
    if (not isinstance(dataset, dict) or dataset.get('complete') is not True
            or type(dataset.get('optimizer_steps', 0)) is not int or dataset.get('optimizer_steps', 0) != 0
            or type(dataset.get('error_rows', 0)) is not int or dataset.get('error_rows', 0) != 0
            or dataset.get('truncated', False) is not False
            or dataset.get('search') != metadata.search
            or not isinstance(dataset.get('model_versions'), (list, tuple))
            or tuple(dataset['model_versions']) != metadata.model_versions):
        raise ValueError('Complete version-bound GPU dataset and exact search receipt required')
    if (not isinstance(initial, (list, tuple)) or not isinstance(final, (list, tuple))
            or len(initial) != count or len(final) != count or len({id(s) for s in initial}) != count):
        raise ValueError('Independent initial/final state for every original job required')
    policies = _indexed(dataset.get('policy'), ('owner', 'step'), count)
    values = _indexed(dataset.get('values'), ('owner', 'step', 'actor'), count)
    raw = _indexed(raw_records, ('game', 'step'), count)
    if not policies or set(policies) != set(raw):
        raise ValueError('Missing policy/raw trajectory coordinates')
    if set(values) != {(*key, viewer) for key in raw for viewer in (0, 1)}:
        raise ValueError('Both original terminal-WDL viewers required at every decision')
    if not isinstance(policy_reference, Mapping) or set(policy_reference) != set(raw):
        raise ValueError('Full independent policy-reference coordinates required')
    for key in policy_reference:
        if not isinstance(key, tuple) or len(key) != 2 or any(type(x) is not int for x in key):
            raise ValueError('Exact independent policy-reference coordinates required')
    per_owner = [[r for r in raw_records if r['game'] == owner] for owner in range(count)]
    for owner, records in enumerate(per_owner):
        if not records or [r['step'] for r in records] != list(range(len(records))):
            raise ValueError('Every job requires a unique complete contiguous trajectory')
        if len(records) > metadata.search['max_actions']:
            raise ValueError('Trajectory exceeds its original action budget')
        state, terminal = initial[owner], final[owner]
        job = metadata.jobs[owner]
        if (not isinstance(state, dict) or state.get('phase') == 'finished'
                or state.get('first_side') != job['first']
                or not isinstance(terminal, dict) or terminal.get('phase') != 'finished'
                or terminal.get('winner') not in ('a', 'b', 'draw')
                or terminal.get('reason') in ('action_limit', 'timeout', 'deadline', 'truncated', 'incomplete')):
            raise ValueError('Matching active initial state and natural terminal required')
        for side in ('a', 'b'):
            if state['sides'][side]['order'] != job['decks'][side]['character_ids']:
                raise ValueError('Initial state differs from fixed job character order')
    from ..seven_runtime import value_precheck_ok
    eligibility = []
    for job in metadata.jobs:
        side_models = {s: models[policy_source_alias(*job['policies'][s])] for s in ('a', 'b')}
        qualified = value_precheck_ok(job.get('value_precheck'), side_models)
        if job['training'] and not qualified:
            raise ValueError('Training requires actual value-precheck evidence for current model versions')
        eligibility.append(job['training'] and qualified)
    observe, acting_side, apply_action, clean, encode = _oracle()
    states = deepcopy(list(initial))
    games = []
    for owner, records in enumerate(per_owner):
        _check_deadline(deadline)
        job = metadata.jobs[owner]
        game = dict(rows=[], value_rows=[], episode_actions=[], roots=[])
        winner = final[owner]['winner']
        for record in records:
            _check_deadline(deadline)
            step, state = record['step'], states[owner]
            actor = record.get('actor')
            if actor not in ('a', 'b') or actor != acting_side(state) or state.get('phase') == 'finished':
                raise ValueError('Trajectory actor mismatch or action after terminal')
            actor_id = int(actor == 'b')
            row, reference = policies[owner, step], policy_reference[owner, step]
            view = observe(state, actor, include_previews=False)
            legal = [item['action'] for item in view['legal_actions'] if item['action']['type'] != 'concede']
            if not legal or not isinstance(reference, dict) or reference.get('actions') != legal:
                raise ValueError('Full original legal-action ordering proof required')
            chosen = record.get('chosen')
            if type(chosen) is not int or not 0 <= chosen < len(legal) or record.get('raw_action') != legal[chosen]:
                raise ValueError('Chosen raw action differs from its original legal coordinate')
            budget = 1 if len(legal) == 1 else metadata.search['simulations']
            expected = dict(actor=actor_id, selected=chosen, version=state['version'],
                            job_seed=job['seed'], requested=budget, completed=budget)
            if any(type(row.get(k)) is not int or row[k] != v for k, v in expected.items()):
                raise ValueError('GPU policy actor/seed/version/action/budget changed')
            if any(type(record.get(k)) is not int or record[k] != budget for k in ('requested', 'completed')):
                raise ValueError('Raw search budget differs from the original job')
            x, c = encode(view, legal)
            gx = _array(row.get('x'), kind='f', ndim=1)
            gc = _array(row.get('c'), kind='f', ndim=2)
            _equal_array(gx, x); _equal_array(gc, c)
            pi = _array(row.get('pi'), kind='f', ndim=1, shape=(len(legal),))
            visit_pi = _array(row.get('visit_pi'), kind='f', ndim=1, shape=(len(legal),))
            visits = _array(row.get('visits'), kind='iu', ndim=1, shape=(len(legal),))
            if (np.any(pi < 0) or not np.isclose(pi.sum(), 1, rtol=0, atol=1e-5)
                    or np.any(visits < 0) or int(visits.sum()) != budget
                    or not np.allclose(visit_pi, visits.astype(np.float32)/budget, rtol=0, atol=1e-5)):
                raise ValueError('Invalid complete GPU policy or visits')
            _equal_array(pi, reference.get('pi')); _equal_array(visits, reference.get('visits'))
            z = 0 if winner == 'draw' else (1 if winner == actor else -1)
            eligible = bool(eligibility[owner] and actor in job['train_sides'])
            if (type(row.get('encode_error')) is not int or row['encode_error'] != 0
                    or type(row.get('learning_eligible')) is not bool or row['learning_eligible'] != eligible
                    or type(row.get('z')) not in (int, float) or row['z'] != z):
                raise ValueError('GPU policy encoding/eligibility/terminal target mismatch')
            game['rows'].append(dict(x=gx.copy(), c=gc.copy(), pi=pi.copy(), visit_pi=visit_pi.copy(),
                visits=visits.copy(), selected=chosen, z=row['z'], side=actor, key=metadata.keys[owner][actor],
                step=step, phase=view['phase'], value_actor=1, search_budget=budget,
                search_algorithm=metadata.search['search_algorithm'], teacher_mode=metadata.search['teacher_mode'],
                terminal_horizon=metadata.search['terminal_horizon'], q_scale=metadata.search['q_scale'],
                policy_source='gumbel_q_natural_wdl'))
            for side in ('a', 'b'):
                value = values[owner, step, int(side == 'b')]
                vx, _ = encode(view if side == actor else observe(state, side, include_previews=False), [])
                actual = _array(value.get('x'), kind='f', ndim=1)
                _equal_array(actual, vx)
                vz = 0 if winner == 'draw' else (1 if winner == side else -1)
                valid = bool(eligibility[owner] and side in job['train_sides'])
                if (value.get('value_actor') != int(side == actor) or value.get('z') != vz
                        or type(value.get('value_actor')) not in (int, float) or type(value.get('z')) not in (int, float)
                        or type(value.get('learning_eligible')) is not bool or value['learning_eligible'] != valid
                        or any(type(value.get(k)) is not int or value[k] != v
                               for k, v in dict(version=state['version'], job_seed=job['seed']).items())):
                    raise ValueError('GPU fixed-viewer context/version/seed/terminal target mismatch')
                game['value_rows'].append(dict(x=actual.copy(), z=value['z'], side=side,
                    key=metadata.keys[owner][side], step=step, value_actor=value['value_actor']))
            action = deepcopy(record['raw_action'])
            game['episode_actions'].append(dict(side=actor, action=action))
            game['roots'].append(dict(step=step, side=actor, chosen=deepcopy(action), legal=len(legal),
                                     diagnostics=deepcopy(reference.get('diagnostics', {}))))
            states[owner] = clean(apply_action(state, actor, deepcopy(action)))
        _check_deadline(deadline)
        _same_graph(states[owner], final[owner])
        game.update({key: deepcopy(job[key]) for key in ('id', 'seed', 'first', 'left', 'right')})
        game.update(job=deepcopy(job), model_versions=deepcopy(metadata.model_versions[owner]),
            complete=True, winner=winner, reason=None, decisions=len(records), searched=len(records),
            turn=final[owner]['turn'], behavior={}, learning_eligible=eligibility[owner],
            rule_backend='cuda_soa', formal_training_qualified=False,
            final_state_sha256=hashlib.sha256(json.dumps(final[owner], sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
        games.append(game)
    _check_deadline(deadline)
    return games
