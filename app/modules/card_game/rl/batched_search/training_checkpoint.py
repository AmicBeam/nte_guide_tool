"""Detached, lossless cooperative collection prefixes at completed decisions.

These are collector continuation records, never learner examples until a natural
terminal state supplies WDL. No scheduler, worker, generator or CUDA object is
part of the closed checkpoint graph.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

SCHEMA = 'hybrid_training_resume_v1'


def digest(value):
    from ..device_auxiliary_continuation import encode_auxiliary_payload
    payload = encode_auxiliary_payload(value)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _contract_value(value):
    if isinstance(value, dict):
        return {str(k): _contract_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_contract_value(v) for v in value]
    if is_dataclass(value):
        return _contract_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    return value


def collection_contract(jobs, models, value_models, **profile):
    from ..seven_lineup import identity
    code_dir = Path(__file__).parent
    code = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
            for name in ('training_checkpoint.py', 'training_rollout.py', 'cooperative.py',
                         'traversal.py', 'inference.py', 'evaluator.py')}
    for name,path in (
            ('seven_sampler.py',code_dir.parent/'seven_sampler.py'),
            ('seven_covered_value.py',code_dir.parent/'seven_covered_value.py'),
            ('process_backend.py',code_dir.parent/'batched_duel/process_backend.py')):
        code[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    def model_identity(bank):
        return {key: dict(version=str(model.version), weights=_contract_value(getattr(model, 'weights', {})))
                for key, model in sorted((bank or {}).items())}
    return digest(_contract_value(dict(jobs=jobs, models=model_identity(models),
        value_models=model_identity(value_models), rule_hash=identity(), code=code, profile=profile)))


def validate_checkpoint(checkpoint, contract, game_ids):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
            'schema', 'contract', 'records', 'finished', 'uncompleted', 'action_counts',
            'states', 'root_generations', 'ready', 'buffer_bytes', 'payload_sha256'}:
        raise ValueError('Exact hybrid training checkpoint schema required')
    payload = {k: v for k, v in checkpoint.items() if k != 'payload_sha256'}
    if checkpoint['schema'] != SCHEMA or checkpoint['contract'] != contract:
        raise ValueError('Hybrid checkpoint jobs, frozen models or search profile changed')
    if checkpoint['payload_sha256'] != digest(payload):
        raise ValueError('Hybrid checkpoint payload integrity mismatch')
    ids = set(game_ids)
    if set(checkpoint['records']) != ids or set(checkpoint['action_counts']) != ids:
        raise ValueError('Hybrid checkpoint must retain every original game')
    active, finished, incomplete = (set(checkpoint[k]) for k in ('states', 'finished', 'uncompleted'))
    if active | finished | incomplete != ids or active & finished or active & incomplete or finished & incomplete:
        raise ValueError('Hybrid checkpoint game partition mismatch')
    if set(checkpoint['root_generations']) != active or set(checkpoint['ready']) != active or len(checkpoint['ready']) != len(active):
        raise ValueError('Hybrid checkpoint ready state mismatch')
    if type(checkpoint['buffer_bytes']) is not int or checkpoint['buffer_bytes'] < 0:
        raise ValueError('Hybrid checkpoint buffer accounting invalid')
    for gid in ids:
        rec = checkpoint['records'][gid]
        if not isinstance(rec, dict) or set(rec) != {'rows', 'values', 'trace', 'roots', 'step', 'searched'}:
            raise ValueError('Hybrid checkpoint trajectory schema invalid')
        step = rec['step']
        if type(step) is not int or step < 0 or checkpoint['action_counts'][gid] != step:
            raise ValueError('Hybrid checkpoint decision counters invalid')
        if gid in active:
            if rec['searched'] != step or len(rec['rows']) != step or len(rec['trace']) != step or len(rec['roots']) != step or len(rec['values']) != 2 * step:
                raise ValueError('Hybrid checkpoint contains a partial search')
            if checkpoint['root_generations'][gid] != step + 1:
                raise ValueError('Hybrid checkpoint root generation invalid')
            if any('z' in row for row in rec['rows'] + rec['values']):
                raise ValueError('Unfinished hybrid game cannot have terminal labels')
            if not isinstance(checkpoint['states'][gid], dict):
                raise ValueError('Hybrid checkpoint private state required')
        if gid in finished and not checkpoint['finished'][gid].get('complete'):
            raise ValueError('Hybrid checkpoint completed game invalid')
    return checkpoint
