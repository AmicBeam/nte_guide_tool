"""Safe numeric/optimizer restoration checks, never an automatic training resume."""
from hashlib import sha256
from pathlib import Path
import json

SCHEMA = 'recovery_restore_check_v1'


def algorithm_identity():
    root = Path(__file__).parent
    files = ('recovery_search.py', 'covered_rollout.py', 'covered_value.py', 'recovery_sampler.py', 'policies.py', 'recovery_runtime.py', 'outcome_runtime.py',
             'residual_policy.py', 'preserved_policy.py', 'information_search.py', 'search_policy.py',
             'recovery_capacity.py', 'recovery_policy.py', 'rule_foundation.py', 'residual_runtime.py', 'cross_teacher_distillation.py', 'league_learning.py', 'guided_recovery.py', '../engine/ai/zhenhong_guide.py',
             'batched_search/inference.py', 'batched_search/value_metrics.py')
    digest = sha256()
    for name in files:
        digest.update(name.encode()); digest.update((root / name).read_bytes())
    return digest.hexdigest()


def save(path, net, optimizer, *, directory, key, rng_state, metadata):
    import torch
    from .recovery_runtime import model
    policy = model(directory, key)
    weights = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    for name, tensor in weights.items():
        if not torch.equal(tensor, torch.as_tensor(policy.weights[name].copy())):
            raise ValueError('Checkpoint must match exported numeric arrays')
    payload = dict(schema=SCHEMA, algorithm_sha256=algorithm_identity(), key=key,
                   model_sha256=policy.version, rule_hash=policy.manifest['rule_hash'],
                   network=weights, optimizer=optimizer.state_dict(),
                   rng_state=json.loads(json.dumps(rng_state)), torch_rng=torch.get_rng_state(),
                   metadata=json.loads(json.dumps(metadata)), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   diagnostic_restore_only=True,
                   quality_approved=False, automatic_serving_approval=False)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp); tmp.replace(path)


def restore(path, *, directory, key, device='cpu'):
    import torch
    from .recovery_runtime import model, restore as restore_network
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    policy = model(directory, key)
    if (data.get('schema') != SCHEMA or data.get('algorithm_sha256') != algorithm_identity()
            or data.get('key') != key or data.get('model_sha256') != policy.version
            or data.get('rule_hash') != policy.manifest['rule_hash']
            or data.get('diagnostic_restore_only') is not True
            or data.get('quality_approved') is not False
            or data.get('automatic_serving_approval') is not False):
        raise ValueError('Recovery checkpoint identity mismatch')
    if set(data['network']) != set(policy.weights):
        raise ValueError('Checkpoint tensor set mismatch')
    for name, tensor in data['network'].items():
        if not torch.equal(tensor, torch.as_tensor(policy.weights[name].copy())):
            raise ValueError('Checkpoint arrays differ from numeric model')
    net, _, _ = restore_network(directory, key, device=device)
    net.load_state_dict(data['network'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(data['optimizer'])
    return net, optimizer, data
