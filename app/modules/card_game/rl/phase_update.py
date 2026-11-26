"""Single-phase batched CE plus independent terminal WDL, with disjoint Adam state."""
import numpy as np
from .phase_replay import row_phase


def update(net, optimizers, rows, *, value_only=False):
    import torch
    from .league_learning import tensors
    if not rows:
        raise ValueError('Empty phase update')
    if not isinstance(optimizers, dict) or set(optimizers) != {'opening', 'playing', 'value'}:
        raise ValueError('Phase-isolated training requires three owned optimizers')
    expected = {p: list(net.policy_parameters(p)) for p in ('opening', 'playing')}
    expected['value'] = list(net.value_parameters())
    for name, optimizer in optimizers.items():
        actual = [p for group in optimizer.param_groups for p in group['params']]
        if len(actual) != len(expected[name]) or {id(p) for p in actual} != {id(p) for p in expected[name]}:
            raise ValueError('Optimizer branch ownership mismatch: ' + name)
    if any(r.get('z') not in (-1, 0, 1) or r.get('reward_to_go') is not None
           or r.get('setup_reward', 0) for r in rows):
        raise ValueError('Natural terminal WDL labels required')
    policy_rows = [r for r in rows if 'pi' in r] if not value_only else []
    phases = {row_phase(r) for r in policy_rows}
    if len(phases) > 1:
        raise ValueError('Group policy rows by phase before the device update')
    phase = next(iter(phases), None)
    device = next(net.parameters()).device
    if policy_rows:
        if len(policy_rows) != len(rows) or any(r.get('value_actor', 1) != 1 for r in rows):
            raise ValueError('Policy batches must contain only actor policy rows')
        x, candidates, mask = tensors([(r['x'], r['c']) for r in rows], device)
    else:
        x = torch.as_tensor(np.stack([r['x'] for r in rows]), device=device, dtype=torch.float32)
    actor_values = [r.get('value_actor', 1) for r in rows]
    if any(v not in (0, 1) for v in actor_values):
        raise ValueError('Exact value observer context required')
    weights = np.asarray([r.get('sample_weight', 1.) for r in rows], dtype=np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive finite sample weights required')
    w = torch.as_tensor(weights, device=device)
    context = torch.tensor(actor_values, dtype=x.dtype, device=device).unsqueeze(1)
    value_logits = net.wdl(torch.cat((net.value_net(x), context), -1))
    z = torch.tensor([int(r['z']) + 1 for r in rows], device=device)
    value_losses = torch.nn.functional.cross_entropy(value_logits, z, reduction='none')
    value_loss = (w * value_losses).mean()
    policy_loss = value_loss.new_zeros(())
    if policy_rows:
        from .phase_policy import host_policy_metadata
        aliases, supported = host_policy_metadata([r['c'] for r in rows], getattr(net, 'source_identity', None))
        aliases = torch.as_tensor(aliases, device=device)
        supported = torch.as_tensor(supported, device=device) if supported is not None else None
        logits = net.policy_logits(x, candidates, mask, phase, aliases=aliases, legacy_rows=supported)
        target_host = np.zeros(tuple(logits.shape), dtype=np.float32)
        for i, row in enumerate(rows):
            pi = np.asarray(row['pi'], dtype=np.float32)
            if (len(pi) != len(row['c']) or not np.isfinite(pi).all()
                    or (pi < 0).any() or not np.isclose(pi.sum(), 1.)):
                raise ValueError('Invalid full legal-axis search policy')
            target_host[i, :len(pi)] = pi
        target = torch.as_tensor(target_host, device=device)
        policy_loss = (w * -(target * logits.log_softmax(-1).masked_fill(~mask, 0)).sum(-1)).mean()
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite phase loss')
    net.zero_grad(set_to_none=True)
    loss.backward()
    groups = {'value': list(net.value_parameters())}
    if phase:
        groups[phase] = list(net.policy_parameters(phase))
    def gradient_norm(parameters):
        squares = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
        return torch.stack(squares).sum().sqrt() if squares else loss.new_zeros(())
    value_parts = [gradient_norm(net.value_net.parameters()), gradient_norm(net.wdl.parameters())]
    policy_parts = [loss.new_zeros(())] * 3
    if phase:
        branch = net.opening if phase == 'opening' else net
        policy_parts = [gradient_norm(getattr(branch, n).parameters()) for n in ('state_net', 'cand_net', 'score')]
    norms = {}
    for name, parameters in groups.items():
        norms[name] = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True,
                                                    foreach=device.type == 'cuda')
    # No step on the inactive optimizer: even retained Adam moments cannot move it.
    for name in groups:
        optimizers[name].step()
    metrics = torch.stack([policy_loss.detach(), value_loss.detach(),
                           norms.get(phase, loss.new_zeros(())), norms['value'],
                           *policy_parts, *value_parts]).cpu().tolist()
    return dict(policy_phase=phase, policy_loss=metrics[0], value_loss=metrics[1],
                policy_gradient_norm=metrics[2], value_gradient_norm=metrics[3],
                policy_gradient_norms=dict(zip(('state_net', 'cand_net', 'score'), metrics[4:7])),
                value_gradient_norms=dict(zip(('value_net', 'wdl'), metrics[7:9])),
                optimizer_steps={name: int(name in groups) for name in optimizers},
                samples=len(rows), policy_samples=len(policy_rows), value_only=value_only)
