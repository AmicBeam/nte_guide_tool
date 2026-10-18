"""Immutable numerical pure-collection checkpoints; never pickle or qualification.

JSON describes a whitelisted typed object graph; NPZ contains only numerical
arrays. Hashes detect incomplete/changed files. An explicit continuation receipt
can change deadlines alone while retaining the original binding in the archive.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = 'pure_cuda_archive_v1'


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value): return sha256(_json(value).encode()).hexdigest()


def binding_for(metadata, executor, max_decisions):
    hydration = getattr(executor, 'hydration_probe_binding', None)
    return json.loads(_json(dict(metadata=asdict(metadata), max_decisions=max_decisions,
        source=dict(rule=executor.rule_source_sha256, image=executor.image.identity,
                    schema=executor.schema.identity,
                    hydration=None if hydration is None else hydration.to_dict()))))


def continuation_receipt(old_binding, new_binding, *, authorization):
    """Explicit caller assertion of newly authorized deadline-only continuation."""
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError('Explicit new-budget authorization receipt required')
    old, new = json.loads(_json(old_binding)), json.loads(_json(new_binding))
    old_deadlines = [j['deadline'] for j in old['metadata']['jobs']]
    new_deadlines = [j['deadline'] for j in new['metadata']['jobs']]
    for job in old['metadata']['jobs']: job.pop('deadline')
    for job in new['metadata']['jobs']: job.pop('deadline')
    if old != new or len(old_deadlines) != len(new_deadlines) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= before
            for before, v in zip(old_deadlines, new_deadlines)):
        raise ValueError('Continuation may extend only the original explicit deadlines')
    return dict(schema='pure_deadline_continuation_v1', old_binding=_digest(old_binding),
        new_binding=_digest(new_binding), old_deadlines=old_deadlines, new_deadlines=new_deadlines,
        authorization=authorization, original_binding=json.loads(_json(old_binding)))


def _check_binding(stored, expected, continuation):
    if stored == expected: return
    if not isinstance(continuation, dict):
        raise ValueError('Pure checkpoint identity/deadline changed without explicit continuation')
    actual = continuation_receipt(stored, expected, authorization=continuation.get('authorization'))
    if actual != continuation:
        raise ValueError('Pure continuation receipt differs from original/new bound metadata')


def _encode(value):
    import torch
    from ...engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
    kinds = {dict:'dict', list:'list', tuple:'tuple', PlayerEntity:'player', CharacterEntity:'character', CardEntity:'card'}
    entries, arrays, memo, visiting = [], {}, {}, set()
    def encode(item):
        kind = type(item)
        if kind in (type(None), bool, int, str): return ['scalar', item]
        if kind is float:
            if not math.isfinite(item): raise ValueError('Finite checkpoint scalars required')
            return ['scalar', item]
        if isinstance(item, np.generic): return encode(item.item())
        ident = id(item)
        if ident in visiting: raise ValueError('Checkpoint cycles are unsupported')
        if ident in memo: return ['ref', memo[ident]]
        index = len(entries); memo[ident] = index; entries.append(None)
        if isinstance(item, (np.ndarray, torch.Tensor)):
            tensor = isinstance(item, torch.Tensor)
            if tensor and item.device.type != 'cpu': raise ValueError('Postbatch CPU tensor archives required')
            array = item.detach().numpy() if tensor else item
            if array.dtype.kind not in 'biuf' or array.dtype.hasobject:
                raise ValueError('Numeric checkpoint arrays only')
            name = 'array_' + str(len(arrays))
            arrays[name] = np.array(array, copy=True, order='C')
            entries[index] = dict(kind='tensor' if tensor else 'array', name=name,
                dtype=array.dtype.str, shape=list(array.shape))
        elif kind in kinds:
            visiting.add(ident)
            body = [[encode(k), encode(v)] for k, v in item.items()] if isinstance(item, dict) else [encode(v) for v in item]
            visiting.remove(ident)
            entries[index] = dict(kind=kinds[kind], items=body)
        else: raise TypeError('Unsupported checkpoint type: ' + kind.__name__)
        return ['ref', index]
    root = encode(value)
    return dict(root=root, entries=entries), arrays


def _decode(graph, arrays):
    import torch
    from ...engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
    classes = dict(player=PlayerEntity, character=CharacterEntity, card=CardEntity, dict=dict)
    entries, memo, active = graph['entries'], {}, set()
    used = set()
    def decode(ref):
        if not isinstance(ref, list) or len(ref) != 2: raise ValueError('Invalid typed checkpoint reference')
        if ref[0] == 'scalar':
            value = ref[1]
            if type(value) not in (type(None), bool, int, float, str) or (type(value) is float and not math.isfinite(value)):
                raise ValueError('Invalid checkpoint scalar')
            return value
        if ref[0] != 'ref' or type(ref[1]) is not int or not 0 <= ref[1] < len(entries):
            raise ValueError('Checkpoint reference outside object table')
        i = ref[1]
        if i in active: raise ValueError('Checkpoint cycle rejected')
        if i in memo: return memo[i]
        entry = entries[i]; kind = entry['kind']; active.add(i)
        if kind in ('array', 'tensor'):
            name = entry['name']; used.add(name)
            array = arrays[name]
            if array.dtype.str != entry['dtype'] or list(array.shape) != entry['shape'] or array.dtype.kind not in 'biuf':
                raise ValueError('Checkpoint numerical shape/dtype changed')
            value = np.array(array, copy=True, order='C')
            if kind == 'tensor': value = torch.from_numpy(value)
        elif kind in classes:
            value = classes[kind].__new__(classes[kind])
            for k, v in entry['items']:
                key = decode(k)
                if key in value: raise ValueError('Duplicate checkpoint map key')
                value[key] = decode(v)
        elif kind in ('list', 'tuple'):
            value = [decode(v) for v in entry['items']]
            if kind == 'tuple': value = tuple(value)
        else: raise ValueError('Unregistered checkpoint object type')
        active.remove(i); memo[i] = value; return value
    value = decode(graph['root'])
    if used != set(arrays) or len(memo) != len(entries): raise ValueError('Unreferenced checkpoint payload')
    return value


def write_payload(path, payload, binding, *, kind, continuation=None):
    """Create a new immutable directory. Partial writes cannot be loaded."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    graph, arrays = _encode(payload)
    with (path/'arrays.npz').open('xb') as output: np.savez_compressed(output, **arrays)
    manifest = dict(schema=SCHEMA, kind=kind, binding=binding, graph=graph,
        arrays_sha256=sha256((path/'arrays.npz').read_bytes()).hexdigest(), continuation=continuation)
    manifest['manifest_sha256'] = _digest(manifest)
    with (path/'manifest.json').open('x', encoding='utf-8') as output: output.write(_json(manifest))
    return str(path.resolve())


def read_payload(path, expected_binding, *, kind, continuation=None):
    path = Path(path)
    data = json.loads((path/'manifest.json').read_text())
    digest = data.pop('manifest_sha256')
    if digest != _digest(data) or data['schema'] != SCHEMA or data['kind'] != kind:
        raise ValueError('Pure checkpoint manifest integrity/type mismatch')
    _check_binding(data['binding'], expected_binding, continuation)
    if sha256((path/'arrays.npz').read_bytes()).hexdigest() != data['arrays_sha256']:
        raise ValueError('Pure checkpoint numerical integrity mismatch')
    with np.load(path/'arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return _decode(data['graph'], arrays)


def save_history(path, history, binding, *, continuation=None):
    from .device_pure_segments import _validate_history
    _validate_history(history, len(binding['metadata']['jobs']), binding['max_decisions'])
    return write_payload(path, dict(initial=history.initial, final=history.final, steps=history.steps,
        complete=history.complete, raw=history.raw, value=history.value, audit=history.audit), binding,
        kind='history', continuation=continuation)


def load_history(path, binding, *, continuation=None):
    from .device_pure_segments import _History, _validate_history
    payload = read_payload(path, binding, kind='history', continuation=continuation)
    history = _History(**payload)
    _validate_history(history, len(binding['metadata']['jobs']), binding['max_decisions'])
    return history


def latest_checkpoint(directory):
    paths = sorted(Path(directory).glob('segment-*')) if Path(directory).exists() else []
    return str(paths[-1]) if paths else None


def batch_cache_path(directory, binding):
    """Stable address only; deadline changes still need load-time receipt checks."""
    identity = json.loads(_json(binding))
    for job in identity['metadata']['jobs']: job.pop('deadline')
    return Path(directory)/('batch-' + _digest(identity))
