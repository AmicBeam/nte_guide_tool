"""Safe numeric/optimizer restoration checks, never an automatic training resume."""
from hashlib import sha256
from pathlib import Path
import json

SCHEMA = 'seven_restore_check_v1'


def algorithm_identity():
    from .recovery_checkpoint import algorithm_identity as shared
    root=Path(__file__).parent;digest=sha256(shared().encode())
    for name in ('seven_lineup.py','seven_grounded.py','seven_policy.py','seven_runtime.py','seven_round.py','seven_checkpoint.py','seven_schedule.py','seven_sampler.py','seven_source.py','seven_covered_value.py','seven_value_metrics.py','seven_rule_foundation.py','full_cycle_experiment.py','new_model_report.py','seven_prepared.py','training_config.py',
                 'recovery_capacity.py','next_training.py','batched_search/training_rollout.py',
                 'batched_search/cooperative.py','batched_search/traversal.py','batched_search/inference.py',
                 'batched_search/statistics.py','batched_search/nodes.py','batched_search/budget.py',
                 'batched_search/evaluator.py','batched_duel/gpu_rule_backend.py',
                 'batched_duel/process_backend.py','batched_duel/cuda_vm.py',
                 'batched_duel/vm_deepcopy.py','rule_ir/portable_program.py',
                 'batched_duel/gpu_rule_entries.py','batched_duel/gpu_rule_program.py',
                 'batched_duel/gpu_rule_capacity.py','batched_duel/gpu_rule_arguments.py',
                 'batched_duel/gpu_seven_projection.py','batched_duel/gpu_seven_observation.py',
                 'batched_duel/gpu_seven_candidates.py','batched_duel/gpu_schema_contract.py'):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    digest.update((root.parents[3]/'scripts/train_duel_v2_current_round.py').read_bytes())
    return digest.hexdigest()


def save(path, net, optimizer, *, directory, key, rng_state, metadata):
    import torch
    from .seven_runtime import model
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
    from .seven_runtime import model, restore as restore_network
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
