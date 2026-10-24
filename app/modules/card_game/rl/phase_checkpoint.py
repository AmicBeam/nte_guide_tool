"""Safe numeric/optimizer restoration for phase-isolated seven-team training.

Does not resume training. Opening, playing, and value Adam groups stay disjoint.
"""
from hashlib import sha256
from pathlib import Path
import json

SCHEMA = 'seven_phase_restore_check_v1'
BRANCHES = ('opening', 'playing', 'value')


def algorithm_identity():
    from . import seven_policy
    root = Path(__file__).parent
    digest = sha256()
    for name in ('phase_policy.py', 'phase_runtime.py', 'phase_checkpoint.py', 'phase_update.py',
                 'phase_replay.py', 'phase_hybrid.py', 'policy_runtime.py', 'phase_rollout.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    from .seven_checkpoint import algorithm_identity as shared_identity
    digest.update(shared_identity().encode())
    digest.update(seven_policy.fingerprint().encode())
    return digest.hexdigest()


def create_optimizers(net, learning_rate=.0003):
    import torch
    foreach = next(net.parameters()).device.type == 'cuda'
    return dict(
        opening=torch.optim.Adam(list(net.policy_parameters('opening')), lr=learning_rate, foreach=foreach),
        playing=torch.optim.Adam(list(net.policy_parameters('playing')), lr=learning_rate, foreach=foreach),
        value=torch.optim.Adam(list(net.value_parameters()), lr=learning_rate, foreach=foreach),
    )


def _group_names(net):
    ids = {id(param): name for name, param in net.named_parameters()}
    groups = dict(
        opening=[ids[id(param)] for param in net.policy_parameters('opening')],
        playing=[ids[id(param)] for param in net.policy_parameters('playing')],
        value=[ids[id(param)] for param in net.value_parameters()],
    )
    owned = [name for names in groups.values() for name in names]
    if len(owned) != len(set(owned)):
        raise ValueError('Optimizer groups overlap')
    if set(owned) != {name for name, parameter in net.named_parameters() if parameter.requires_grad}:
        raise ValueError('Optimizer groups miss trainable parameters')
    return groups


def _optimizer_names(net, optimizer):
    ids = {id(param): name for name, param in net.named_parameters()}
    names = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if id(param) not in ids:
                raise ValueError('Optimizer owns an unknown tensor')
            names.append(ids[id(param)])
    return names


def _assert_groups(net, optimizers, expected=None):
    if set(optimizers) != set(BRANCHES):
        raise ValueError('Missing optimizer branch')
    expected = _group_names(net) if expected is None else expected
    if set(expected) != set(BRANCHES):
        raise ValueError('Missing optimizer branch')
    for branch in BRANCHES:
        if _optimizer_names(net, optimizers[branch]) != list(expected[branch]):
            raise ValueError('Altered optimizer group ownership')
    return expected


def _validate_adam(optimizer):
    import math
    import torch
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError('Phase checkpoints require Adam')
    for group in optimizer.param_groups:
        if not math.isfinite(group['lr']) or group['lr'] <= 0:
            raise ValueError('Invalid Adam learning rate')
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            required = {'step', 'exp_avg', 'exp_avg_sq'}
            if group.get('amsgrad'):
                required.add('max_exp_avg_sq')
            if set(state) != required:
                raise ValueError('Invalid Adam state fields')
            for name, value in state.items():
                if not torch.is_tensor(value) or not torch.isfinite(value).all():
                    raise ValueError('Nonfinite Adam state')
                if name == 'step':
                    if value.numel() != 1 or value.item() < 0 or value.item() != int(value.item()):
                        raise ValueError('Invalid Adam step')
                elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError('Adam moment shape/dtype mismatch')


def save(path, net, optimizers, *, directory, key, rng_state, metadata):
    import torch
    from .phase_runtime import model
    from . import phase_policy
    policy = model(directory, key)
    if policy.schema != phase_policy.SCHEMA:
        raise ValueError('Checkpoint requires a phase-isolated export')
    groups = _assert_groups(net, optimizers)
    for optimizer in optimizers.values():
        _validate_adam(optimizer)
    weights = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    if set(weights) != set(policy.weights):
        raise ValueError('Checkpoint tensor set mismatch')
    for name, tensor in weights.items():
        if not torch.isfinite(tensor).all():
            raise ValueError('Checkpoint contains non-finite tensors')
        if not torch.equal(tensor, torch.as_tensor(policy.weights[name].copy())):
            raise ValueError('Checkpoint must match exported numeric arrays')
    payload = dict(
        schema=SCHEMA, algorithm_sha256=algorithm_identity(), key=key,
        policy_schema=phase_policy.SCHEMA, model_sha256=policy.version,
        rule_hash=policy.manifest['rule_hash'], network=weights,
        optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        optimizer_groups=groups, rng_state=json.loads(json.dumps(rng_state)),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        metadata=json.loads(json.dumps(metadata)), diagnostic_restore_only=True,
        quality_approved=False, automatic_serving_approval=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def restore(path, *, directory, key, device='cpu'):
    import torch
    from .phase_runtime import model, restore as restore_network
    from . import phase_policy
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    policy = model(directory, key)
    if (data.get('schema') != SCHEMA or data.get('algorithm_sha256') != algorithm_identity()
            or data.get('key') != key or data.get('policy_schema') != phase_policy.SCHEMA
            or data.get('model_sha256') != policy.version
            or data.get('rule_hash') != policy.manifest['rule_hash']
            or data.get('diagnostic_restore_only') is not True
            or data.get('quality_approved') is not False
            or data.get('automatic_serving_approval') is not False):
        raise ValueError('Phase checkpoint identity mismatch')
    if set(data.get('network') or ()) != set(policy.weights):
        raise ValueError('Checkpoint tensor set mismatch')
    if set(data.get('optimizers') or ()) != set(BRANCHES) or set(data.get('optimizer_groups') or ()) != set(BRANCHES):
        raise ValueError('Missing optimizer branch')
    for name, tensor in data['network'].items():
        if not torch.isfinite(tensor).all():
            raise ValueError('Checkpoint contains non-finite tensors')
        if not torch.equal(tensor, torch.as_tensor(policy.weights[name].copy())):
            raise ValueError('Checkpoint arrays differ from numeric model')
    net, _, _ = restore_network(directory, key, device=device)
    net.load_state_dict(data['network'])
    optimizers = create_optimizers(net)
    _assert_groups(net, optimizers, data['optimizer_groups'])
    for branch, optimizer in optimizers.items():
        payload = data['optimizers'][branch]
        ids = [index for group in payload['param_groups'] for index in group['params']]
        if ids != list(range(len(data['optimizer_groups'][branch]))) or not set(payload['state']) <= set(ids):
            raise ValueError('Altered serialized Adam parameter order')
        optimizer.load_state_dict(payload)
        _validate_adam(optimizer)
    _assert_groups(net, optimizers, data['optimizer_groups'])
    return net, optimizers, data
