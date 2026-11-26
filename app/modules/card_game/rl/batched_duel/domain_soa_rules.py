"""Resident, masked arithmetic stages for the domain SoA transition pipeline.

These stages take already-resolved eligibility/modifier masks. They do not
replace event dispatch, responses, damage redirection or card continuations.
All loops have a layout bound, never a per-game host decision. No packing,
decoding, .item(), CPU rule call, or hidden fallback occurs here.
"""
from dataclasses import dataclass

INVALID_DOMAIN = 1
INTEGER_OVERFLOW = 2
UNKNOWN_OPERATION = 4
supports_full_duel_rules = False
INT_MIN, INT_MAX = -(2 ** 31), 2 ** 31 - 1


@dataclass(frozen=True)
class FieldResult:
    value: object
    delta: object
    errors: object


@dataclass(frozen=True)
class DamageFields:
    hp: object
    shield: object
    absorbed: object
    loss: object
    hp_lost: object
    lethal: object
    errors: object


@dataclass(frozen=True)
class ModifierFields:
    value: object
    boolean: object
    applied: object
    errors: object


def _field(value, name, *, like=None, boolean=False, ndim=None):
    import torch
    if not isinstance(value, torch.Tensor):
        raise TypeError(name + ' must be a resident tensor')
    if value.dtype != (torch.bool if boolean else torch.int32):
        raise ValueError(name + ' has an incompatible exact dtype')
    if ndim is not None and value.ndim != ndim:
        raise ValueError(name + ' has an incompatible rank')
    if like is not None and (value.device != like.device or value.shape != like.shape):
        raise ValueError(name + ' must match the field shape and device')
    return value


def capped_add(value, amount, cap, eligible, active):
    """Canonical min(cap, value + amount), including signed amounts.

    ``eligible`` is the resolved rule decision (alive, resource exceptions,
    source hooks, etc.), NOT a guessed hp>0 substitute. ``active`` can have
    any identical field shape, e.g. [B,2,C]. Failed lanes retain their input.
    """
    import torch
    _field(value, 'value')
    for name, item in (('amount', amount), ('cap', cap)):
        _field(item, name, like=value)
    for name, item in (('eligible', eligible), ('active', active)):
        _field(item, name, like=value, boolean=True)
    work = active & eligible
    total = value.long() + amount.long()
    result = torch.minimum(cap.long(), total)
    errors = (work & (cap < 0)).int() * INVALID_DOMAIN
    errors |= (work & ((result < INT_MIN) | (result > INT_MAX))).int() * INTEGER_OVERFLOW
    accepted = work & (errors == 0)
    out = torch.where(accepted, result, value.long()).int()
    # delta may itself exceed int32 even when both endpoints fit.
    delta = torch.where(accepted, out.long() - value.long(), 0)
    return FieldResult(out, delta, errors)


def shield_add(shield, amount, eligible, active):
    """Shield accumulation after canonical target/finished checks."""
    import torch
    _field(shield, 'shield')
    _field(amount, 'amount', like=shield)
    _field(eligible, 'eligible', like=shield, boolean=True)
    _field(active, 'active', like=shield, boolean=True)
    work = eligible & active
    total = shield.long() + amount.long().clamp_min(0)
    errors = (work & (shield < 0)).int() * INVALID_DOMAIN
    errors |= (work & (total > INT_MAX)).int() * INTEGER_OVERFLOW
    good = work & (errors == 0)
    out = torch.where(good, total, shield.long()).int()
    return FieldResult(out, torch.where(good, out.long() - shield.long(), 0), errors)


def damage_fields(hp, shield, amount, bypass, immune, life_locked, floor, limit, active):
    """Damage arithmetic AFTER source/resistance/response/redirect decisions.

    limit=-1 means no cap. Immutable source fields allow simultaneous combat
    to prepare both sides before settlement. ``loss`` preserves the formal
    un-clamped loss returned to callbacks; ``hp_lost`` is actual HP change.
    Callers must route shield-overflow redirects before using this stage.
    """
    import torch
    _field(hp, 'hp')
    for name, item in (('shield', shield), ('amount', amount), ('floor', floor), ('limit', limit)):
        _field(item, name, like=hp)
    for name, item in (('bypass', bypass), ('immune', immune),
                       ('life_locked', life_locked), ('active', active)):
        _field(item, name, like=hp, boolean=True)
    invalid = (hp < 0) | (shield < 0) | (floor < 0) | (limit < -1)
    errors = (active & invalid).int() * INVALID_DOMAIN
    good = active & (errors == 0)
    damage = torch.where(immune, 0, amount.clamp_min(0))
    damage = torch.where(limit >= 0, torch.minimum(damage, limit), damage)
    absorbed = torch.where(bypass, 0, torch.minimum(shield, damage))
    loss = torch.where(life_locked, 0, damage - absorbed)
    out_hp = torch.maximum(floor.long(), hp.long() - loss.long()).int()
    out_shield = shield - absorbed
    out_hp = torch.where(good, out_hp, hp)
    out_shield = torch.where(good, out_shield, shield)
    loss = torch.where(good, loss, 0)
    absorbed = torch.where(good, absorbed, 0)
    return DamageFields(out_hp, out_shield, absorbed, loss,
        torch.where(good, hp.long() - out_hp.long(), 0),
        good & (hp > 0) & (out_hp <= 0), errors)


def resource_caps(turn, escalation_enabled, base_energy_cap):
    """Current global-turn 13 cap reduction, including downed entities."""
    import torch
    _field(turn, 'turn', ndim=1)
    _field(escalation_enabled, 'escalation_enabled', like=turn, boolean=True)
    _field(base_energy_cap, 'base_energy_cap')
    if base_energy_cap.ndim < 2 or base_energy_cap.shape[0] != turn.shape[0] or base_energy_cap.device != turn.device:
        raise ValueError('Energy caps need a batch dimension on the same device')
    reduced = escalation_enabled & (turn >= 13)
    extended = reduced.reshape((-1,) + (1,) * (base_energy_cap.ndim - 1)).int()
    return base_energy_cap - extended, 2 - reduced.int()


def reduce_integer_modifiers(base, layers, operations, amounts, present, active, *,
                             base_boolean=None, amount_boolean=None):
    """Stable canonical layer / set-add-multiply-allow-forbid reduction.

    Input [B,E] effects MUST already satisfy property and condition checks.
    Integer subset is exact and overflow-checked at every intermediate step;
    floating multipliers require a separately qualified path. No unknown
    effect is dropped. op codes: set=0, add=1, multiply=2, allow=3, forbid=4.
    ``boolean`` preserves a boolean result's type for following rule logic.
    """
    import torch
    _field(base, 'base', ndim=1)
    _field(layers, 'layers', ndim=2)
    if layers.shape[0] != base.shape[0] or layers.device != base.device:
        raise ValueError('Modifier batch dimension/device mismatch')
    for name, item in (('operations', operations), ('amounts', amounts)):
        _field(item, name, like=layers)
    _field(present, 'present', like=layers, boolean=True)
    _field(active, 'active', like=base, boolean=True)
    if base_boolean is None:
        base_boolean = torch.zeros_like(active)
    _field(base_boolean, 'base_boolean', like=base, boolean=True)
    if amount_boolean is None:
        amount_boolean = torch.zeros_like(present)
    _field(amount_boolean, 'amount_boolean', like=layers, boolean=True)
    unknown = present & ((operations < 0) | (operations > 4))
    errors = (active & unknown.any(-1)).int() * UNKNOWN_OPERATION
    errors |= (active & base_boolean & (base != 0) & (base != 1)).int() * INVALID_DOMAIN
    errors |= (active & (present & amount_boolean & (amounts != 0) & (amounts != 1)).any(-1)).int() * INVALID_DOMAIN
    # int64 key keeps extreme int32 layers ordered without overflow.
    keys = layers.long() * 5 + operations.long().clamp(0, 4)
    permutation = keys.argsort(dim=1, stable=True)
    ops = operations.gather(1, permutation)
    values = amounts.gather(1, permutation).long()
    booleans = amount_boolean.gather(1, permutation)
    enabled = present.gather(1, permutation)
    current, is_bool = base.long(), base_boolean.clone()
    for slot in range(layers.shape[1]):
        apply = active & (errors == 0) & enabled[:, slot]
        op, value = ops[:, slot], values[:, slot]
        candidate = current
        candidate = torch.where(op == 0, value, candidate)
        candidate = torch.where(op == 1, current + value, candidate)
        # Inputs and previous accepted outputs are int32, so this int64 product is safe.
        candidate = torch.where(op == 2, current * value, candidate)
        candidate = torch.where(op == 3, ((current != 0) | (value != 0)).long(), candidate)
        candidate = torch.where((op == 4) & (value != 0), 0, candidate)
        overflow = apply & ((candidate < INT_MIN) | (candidate > INT_MAX))
        errors |= overflow.int() * INTEGER_OVERFLOW
        accept = apply & ~overflow
        next_bool = torch.where(op == 0, booleans[:, slot],
            torch.where((op == 1) | (op == 2), False,
                torch.where((op == 3) | ((op == 4) & (value != 0)), True, is_bool)))
        current = torch.where(accept, candidate, current)
        is_bool = torch.where(accept, next_bool, is_bool)
    good = active & (errors == 0)
    return ModifierFields(torch.where(good, current.int(), base),
        torch.where(good, is_bool, base_boolean),
        present & good[:, None], errors)


def expire_effects(expiry_side, expiry_phase, expiry_turn, present, owner_side,
                   current_side, current_phase, turn_count, active):
    """Select canonical timed expiries, preserving source order.

    expiry_side=-1 uses holder side. expiry_phase=-1 means no expiry.
    Returns (kept_mask, expired_mask, per-row errors); event dispatch and
    stable compaction are separate pipeline stages.
    """
    import torch
    _field(expiry_side, 'expiry_side', ndim=2)
    for name, item in (('expiry_phase', expiry_phase), ('expiry_turn', expiry_turn), ('owner_side', owner_side)):
        _field(item, name, like=expiry_side)
    _field(present, 'present', like=expiry_side, boolean=True)
    rows = expiry_side.shape[0]
    for name, item in (('current_side', current_side), ('current_phase', current_phase)):
        _field(item, name, ndim=1)
        if item.shape != (rows,) or item.device != expiry_side.device:
            raise ValueError('Clock row/device mismatch')
    _field(active, 'active', like=current_side, boolean=True)
    _field(turn_count, 'turn_count', ndim=2)
    if turn_count.shape != (rows, 2) or turn_count.device != expiry_side.device:
        raise ValueError('Two player turn counters required')
    invalid = present & ((expiry_side < -1) | (expiry_side > 1)
        | (owner_side < 0) | (owner_side > 1) | (expiry_phase < -1) | (expiry_phase > 1))
    errors = (active & (invalid.any(-1) | (current_side < 0) | (current_side > 1)
        | (current_phase < 0) | (current_phase > 1))).int() * INVALID_DOMAIN
    clock = turn_count.gather(1, current_side.clamp(0, 1).long()[:, None])
    side = torch.where(expiry_side < 0, owner_side, expiry_side)
    expired = present & active[:, None] & (errors == 0)[:, None] & (expiry_phase >= 0)
    expired &= (side == current_side[:, None]) & (expiry_phase == current_phase[:, None]) & (expiry_turn <= clock)
    return present & ~expired, expired, errors


def compact_slots(values, present, active):
    """Stable bounded compaction of ordered effect/card slots on device.

    values [B,M,...] may contain integer identities or numeric fields.
    Returned permutation lets every field use the same ordering. Inactive
    rows keep values/masks. No capacity-dependent slot is silently dropped.
    """
    import torch
    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise ValueError('Resident slot fields required')
    _field(present, 'present', boolean=True, ndim=2)
    _field(active, 'active', boolean=True, ndim=1)
    if present.shape != values.shape[:2] or active.shape != (values.shape[0],):
        raise ValueError('Slot/mask shape mismatch')
    if present.device != values.device or active.device != values.device:
        raise ValueError('Slot/mask device mismatch')
    slots = torch.arange(values.shape[1], device=values.device).expand_as(present)
    permutation = (~present).long().argsort(dim=1, stable=True)
    permutation = torch.where(active[:, None], permutation, slots)
    indices = permutation.reshape(permutation.shape + (1,) * (values.ndim - 2)).expand_as(values)
    return values.gather(1, indices), present.gather(1, permutation), permutation
