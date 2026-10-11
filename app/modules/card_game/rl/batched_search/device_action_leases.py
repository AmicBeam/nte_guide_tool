"""Checked action graph reclamation; no rules or qualification are executed."""
from ..batched_duel.device_rule_protocol import DeviceStateBatch

HANDLE_ERROR, OWNER_ERROR = 1, 2


def action_states(decision):
    return DeviceStateBatch(decision.action_pool, decision.action_slots,
        decision.action_generations, decision.states.owners,
        decision.action_slots >= 0, decision.errors)


def release_actions_device(decision, active, expected_owners, *, protected=()):
    """Release numeric snapshots once, preserving borrowed/protected state refs.

    The action arena validates ownership separately from prior execution errors.
    Root/current aliases do not acquire additional graph leases when cloned.
    """
    import torch as t
    pool = decision.action_pool
    live = getattr(pool, 'live', None)
    if not isinstance(live, t.Tensor) or not callable(getattr(pool, 'release_device', None)):
        raise TypeError('Checked action arena with live lease registry required')
    slots, generations = decision.action_slots, decision.action_generations
    owners = decision.states.owners
    errors = t.zeros_like(decision.errors)
    allocated = active & (slots >= 0)
    fit = (slots < live.numel()) & (generations >= 0)
    errors |= (allocated & ~fit).to(t.int32) * HANDLE_ERROR
    errors |= (allocated & (owners != expected_owners)).to(t.int32) * OWNER_ERROR
    safe = slots.clamp(0, live.numel() - 1)
    take = allocated & fit & (errors == 0) & live[safe].bool()
    registry_owner = getattr(pool, 'owner', None)
    registry_gen = getattr(pool, 'generation', None)
    if isinstance(registry_owner, t.Tensor):
        errors |= (take & (registry_owner[safe] != owners)).to(t.int32) * OWNER_ERROR
    if isinstance(registry_gen, t.Tensor):
        errors |= (take & (registry_gen[safe] != generations)).to(t.int32) * HANDLE_ERROR
    take &= errors == 0
    for refs in (decision.states,) + tuple(protected):
        if refs.pool is not pool:
            continue
        ordered, order = refs.slots.sort()
        at = t.searchsorted(ordered, slots).clamp(0, refs.slots.numel() - 1)
        row = order[at]
        match = ((slots == refs.slots[row]) & (generations == refs.generations[row])
                 & (refs.slots[row] >= 0) & (refs.generations[row] >= 0))
        if refs.slots.shape == slots.shape:
            match |= (slots == refs.slots) & (generations == refs.generations) & (slots >= 0)
        take &= ~match
    request = DeviceStateBatch(pool, slots, generations, owners, take, t.zeros_like(errors))
    pool.release_device(request)
    errors |= request.errors
    freed = take & (request.errors == 0)
    slots.masked_fill_(freed, -1)
    generations.masked_fill_(freed, 0)
    return errors
