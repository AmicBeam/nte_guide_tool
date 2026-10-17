"""Postbatch pure value-calibration proof; no rollout, search labels or qualification.

``values`` must be the full two-viewer PureValueTape export. Policy references
are independent CUDA postbatch receipts, not policy targets or driver approval.
Raw coordinates follow RawActionTape (integer actor, owner, seed and version).
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import numpy as np

from .soa_pure_jobs import SoAPureJobs, adapt_pure_jobs
from .soa_training_jobs import policy_source_alias
from .soa_training_results import (
    _oracle, _check_deadline, _indexed, _same_graph, _array, _equal_array)


def _initial_state(parameters):
    from ...engine.duel_v2 import new_game
    return new_game(**deepcopy(parameters))


def _integers(row, expected, label):
    if any(type(row.get(key)) is not int or row[key] != value for key, value in expected.items()):
        raise ValueError(label)


def verified_pure_calibration_games(dataset, *, initial, final, raw_records,
                                    metadata, models, policy_reference, deadline=None):
    """Verify one complete pure-calibration batch, or return no games.

    Dataset requires complete/error_rows/truncated/optimizer_steps, execution,
    model_versions and all values; policy targets must be absent/empty. Each
    (owner,step) reference requires actions, x, c, logits, selected, info_hash
    (eight uint32 words), source_alias, model_version, owner/step/actor/job_seed/
    version, encode_error=0, inference_error=0, requested=completed=0 and
    inference_backend='cuda'. Its independent CUDA logits must agree with the
    loaded model's NumPy scores and exact argmax. Close but different argmaxes
    fail closed; the GPU choice is never corrected from CPU scores. Initial
    state includes original setup events/cache; only replayed actions are
    followed by clean. This receipt never grants driver/training qualification.
    """
    _check_deadline(deadline)
    if not isinstance(metadata, SoAPureJobs):
        raise TypeError('Real SoAPureJobs metadata required')
    comparison = {'key', 'replica', 'label', 'learner', 'references'}
    if metadata.reference_inference_keys or any(comparison & set(job) for job in metadata.jobs):
        raise ValueError('Pure calibration cannot admit comparison/reference-role jobs')
    rebound = adapt_pure_jobs(metadata.jobs, models)
    for field in metadata.__dataclass_fields__:
        if getattr(metadata, field) != getattr(rebound, field):
            raise ValueError('Pure metadata or loaded numerical model bindings changed')
    count = len(metadata.jobs)
    if (not isinstance(dataset, dict) or dataset.get('complete') is not True
            or type(dataset.get('error_rows')) is not int or dataset['error_rows'] != 0
            or type(dataset.get('optimizer_steps')) is not int or dataset['optimizer_steps'] != 0
            or dataset.get('truncated') is not False or dataset.get('execution') != metadata.execution
            or not isinstance(dataset.get('model_versions'), (list, tuple))
            or tuple(dataset['model_versions']) != metadata.model_versions
            or not isinstance(dataset.get('policy', []), (list, tuple)) or dataset.get('policy', [])):
        raise ValueError('Complete error-free version-bound pure value dataset required')
    if (not isinstance(initial, (list, tuple)) or not isinstance(final, (list, tuple))
            or len(initial) != count or len(final) != count
            or len({id(s) for s in initial}) != count or len({id(s) for s in final}) != count):
        raise ValueError('Independent initial/final state for every pure job required')
    raw = _indexed(raw_records, ('game', 'step'), count)
    values = _indexed(dataset.get('values'), ('owner', 'step', 'actor'), count)
    if not raw or set(values) != {(*key, viewer) for key in raw for viewer in (0, 1)}:
        raise ValueError('Every pure decision requires both actual GPU viewer rows')
    if not isinstance(policy_reference, Mapping) or set(policy_reference) != set(raw):
        raise ValueError('Every raw action requires an independent CUDA policy reference')
    if any(not isinstance(key, tuple) or len(key) != 2 or any(type(v) is not int for v in key)
           for key in policy_reference):
        raise ValueError('Exact policy-reference coordinates required')
    observe, acting_side, apply_action, clean, encode = _oracle()
    states = deepcopy(list(initial))
    games = []
    for owner, job in enumerate(metadata.jobs):
        _check_deadline(deadline)
        records = [row for row in raw_records if row['game'] == owner]
        if (not records or [row['step'] for row in records] != list(range(len(records)))
                or len(records) > job.get('max_actions', 800)):
            raise ValueError('Complete unique contiguous pure trajectory required')
        terminal = final[owner]
        if (not isinstance(initial[owner], dict) or initial[owner].get('phase') == 'finished'
                or not isinstance(terminal, dict) or terminal.get('phase') != 'finished'
                or terminal.get('winner') not in ('a', 'b', 'draw')
                or terminal.get('reason') in (None, '', 'action_limit', 'timeout', 'deadline', 'truncated', 'incomplete')):
            raise ValueError('Only complete natural pure terminals are admissible')
        # This also binds actual initial card order, RNG and entity aliases;
        # character names alone cannot establish the normal-mulligan initializer.
        _same_graph(initial[owner], _initial_state(metadata.initializers[owner]))
        game = dict(rows=[], value_rows=[], audit_value_rows=[], episode_actions=[], roots=[])
        for record in records:
            _check_deadline(deadline)
            step, state = record['step'], states[owner]
            side = acting_side(state)
            if side not in ('a', 'b') or state.get('phase') == 'finished':
                raise ValueError('Pure action after terminal or invalid physical actor')
            actor = int(side == 'b')
            _integers(record, dict(owner=owner, seed=job['seed'], version=state['version'], actor=actor),
                      'Raw pure actor/owner/seed/version mismatch')
            if any(field in record and (type(record[field]) is not int or record[field] != 0)
                   for field in ('requested', 'completed', 'searched')):
                raise ValueError('Raw pure trajectory contains search execution')
            reference = policy_reference[owner, step]
            if not isinstance(reference, dict) or reference.get('inference_backend') != 'cuda':
                raise ValueError('Independent CUDA policy inference receipt required')
            _integers(reference, dict(owner=owner, step=step, actor=actor, job_seed=job['seed'],
                version=state['version'], encode_error=0, inference_error=0, requested=0, completed=0),
                'Pure policy coordinates/errors/search counters mismatch')
            view = observe(state, side, include_previews=False)
            actions = [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']
            if not actions or reference.get('actions') != actions:
                raise ValueError('Original complete legal-action ordering mismatch')
            for supplied_action, original_action in zip(reference['actions'], actions):
                _same_graph(supplied_action, original_action)
            chosen = reference.get('selected')
            if (type(chosen) is not int or not 0 <= chosen < len(actions)
                    or record.get('raw_action') != actions[chosen]
                    or ('chosen' in record and (type(record['chosen']) is not int or record['chosen'] != chosen))):
                raise ValueError('Pure chosen raw action mismatch')
            _same_graph(record['raw_action'], actions[chosen])
            x, c = encode(view, actions)
            gx = _array(reference.get('x'), kind='f', ndim=1)
            gc = _array(reference.get('c'), kind='f', ndim=2)
            _equal_array(gx, x); _equal_array(gc, c)
            words = np.frombuffer(hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()
                + np.asarray(c, dtype=np.float32).tobytes()).digest(), dtype='>u4').astype(np.int64)
            supplied = np.asarray(reference.get('info_hash'))
            if supplied.shape != (8,) or supplied.dtype.kind not in 'iu' or not np.array_equal(supplied, words):
                raise ValueError('Pure information hash mismatch')
            alias = policy_source_alias(*job['policies'][side])
            if reference.get('source_alias') != alias or reference.get('model_version') != metadata.model_versions[owner][side]:
                raise ValueError('Pure CUDA source/model version mismatch')
            model = models[alias]
            if not callable(getattr(model, 'scores', None)):
                raise ValueError('Loaded NumPy policy scores required for independent proof')
            logits = _array(reference.get('logits'), kind='f', ndim=1, shape=(len(actions),))
            expected = _array(model.scores(gx.copy(), gc.copy()), kind='f', ndim=1, shape=logits.shape)
            if (not np.allclose(logits, expected, rtol=2e-5, atol=2e-5)
                    or int(np.argmax(logits)) != chosen or int(np.argmax(expected)) != chosen):
                raise ValueError('CUDA/NumPy pure scores or exact argmax mismatch')
            for viewer, viewer_side in enumerate(('a', 'b')):
                value = values[owner, step, viewer]
                _integers(value, dict(viewer=viewer, physical_actor=actor, version=state['version'], job_seed=job['seed']),
                          'GPU value viewer/actor/seed/version mismatch')
                if value.get('pure_value') is not True:
                    raise ValueError('Actual pure value tape rows required')
                for field in ('slot', 'generation'):
                    if type(value.get(field)) is not int or value[field] < 0:
                        raise ValueError('Invalid value lease metadata')
                    if value[field] != values[owner, step, 0].get(field):
                        raise ValueError('Both GPU viewers must belong to the same physical root')
                context = int(viewer == actor)
                z = 0 if terminal['winner'] == 'draw' else (1 if terminal['winner'] == viewer_side else -1)
                eligible = metadata.train_masks[owner][viewer]
                if (type(value.get('value_actor')) not in (int, float) or value['value_actor'] != context
                        or type(value.get('z')) not in (int, float) or value['z'] != z
                        or type(value.get('learning_eligible')) is not bool or value['learning_eligible'] != eligible):
                    raise ValueError('GPU pure context/terminal/eligibility mismatch')
                vx, _ = encode(view if viewer == actor else observe(state, viewer_side, include_previews=False), [])
                actual_x = _array(value.get('x'), kind='f', ndim=1)
                _equal_array(actual_x, vx)
                output = deepcopy(value)
                output.update(side=viewer_side, key=metadata.keys[owner][viewer_side])
                game['audit_value_rows'].append(deepcopy(output))
                if eligible:
                    game['value_rows'].append(output)
            action = deepcopy(record['raw_action'])
            game['episode_actions'].append(dict(side=side, action=action))
            game['roots'].append(dict(step=step, side=side, chosen=deepcopy(action), legal=len(actions),
                                     source_alias=alias, model_version=reference['model_version']))
            states[owner] = clean(apply_action(state, side, deepcopy(action)))
        _check_deadline(deadline)
        _same_graph(states[owner], terminal)
        game.update({field: deepcopy(job[field]) for field in ('id', 'seed', 'first', 'left', 'right')})
        game.update(job=deepcopy(job), model_versions=deepcopy(metadata.model_versions[owner]),
            metadata=deepcopy(metadata.identities[owner]), execution=deepcopy(metadata.execution),
            complete=True, winner=terminal['winner'], reason=terminal['reason'], turn=terminal['turn'],
            decisions=len(records), searched=0, pure_value=True, learning_eligible=False,
            rule_backend='cuda_soa', formal_training_qualified=False, gpu_qualified=False,
            final_state_sha256=hashlib.sha256(json.dumps(terminal, sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
        games.append(game)
    _check_deadline(deadline)
    return games
