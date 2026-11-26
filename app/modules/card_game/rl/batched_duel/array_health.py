"""Bounded per-unit max-HP cuts as resident integer arrays.

Matches formal reduce_max_hp on a single unit's hp/max_hp: one integer cut at
a time, then proportional ROUND_HALF_UP. Life-lock, finished, knockdowns,
events and victory stay caller mask/settlement. Not a complete duel rule, and
not a VM, Decimal, host per-row, or CPU-fallback path.
"""
from dataclasses import dataclass


INVALID_INPUT, AMOUNT_EXCEEDS_MAX_CUTS, UNSAFE_OVERFLOW = 1, 2, 4
MAX_SUPPORTED_CUTS = 256
supports_full_duel_rules = False


@dataclass(frozen=True)
class ArrayHealthResult:
    hp: object
    max_hp: object
    cuts: object
    errors: object


def reduce_max_hp_arrays(hp, max_hp, amount, active, *, max_cuts=64):
    """Masked batch of canonical per-unit max-HP decrements.

    Supported active domain: resident 1-D int32 hp/max_hp/amount and bool
    active on one device; 0 <= hp,max_hp <= 2**31-1; amount_eff = max(0,
    amount) <= max_cuts. Exact ROUND_HALF_UP of hp*(max_hp-1)/max_hp uses the
    int64 remainder (hp*(max_hp-1) + max_hp//2) // max_hp, which cannot
    overflow that nonnegative int32 domain. Negative amount is a canonical
    no-op. Nonnegative hp == 0 is a canonical no-op. Negative hp/max_hp,
    amount > max_cuts, or an int64 product that could wrap are per-row errors
    and leave that row unchanged. Inactive rows and error rows keep their
    input hp/max_hp. Inputs are not mutated. max_cuts is a static Python
    bound in [1, 256], default 64.
    """
    import torch
    if type(max_cuts) is not int or not 1 <= max_cuts <= MAX_SUPPORTED_CUTS:
        raise ValueError('Bounded positive max_cuts required')
    if not isinstance(hp, torch.Tensor) or hp.dtype != torch.int32 or hp.ndim != 1:
        raise ValueError('Resident int32 hp vector required')
    device, shape = hp.device, hp.shape

    def vec(name, value, dtype):
        if (not isinstance(value, torch.Tensor) or value.device != device
                or value.dtype != dtype or tuple(value.shape) != tuple(shape)):
            raise ValueError(f'Resident {dtype} vector {name} required')

    vec('max_hp', max_hp, torch.int32)
    vec('amount', amount, torch.int32)
    vec('active', active, torch.bool)
    hp_out, max_out = hp.clone(), max_hp.clone()
    cuts = torch.zeros_like(hp)
    errors = torch.zeros_like(hp)
    hp64, max64 = hp.to(torch.int64), max_hp.to(torch.int64)
    new0 = max64 - 1
    limit = (hp.new_ones((), dtype=torch.int64) << 62) // torch.clamp(new0, min=1)
    invalid = active & ((hp < 0) | (max_hp < 0))
    too_many = active & (amount > max_cuts)
    unsafe = active & (hp > 0) & (max_hp > 1) & (hp64 > limit)
    errors = errors | invalid.to(torch.int32) * INVALID_INPUT
    errors = errors | too_many.to(torch.int32) * AMOUNT_EXCEEDS_MAX_CUTS
    errors = errors | unsafe.to(torch.int32) * UNSAFE_OVERFLOW
    work = active & (errors == 0) & (hp > 0)
    amount_n = torch.clamp(amount.to(torch.int64), min=0)
    for _ in range(max_cuts):
        old_max = max_out.to(torch.int64)
        apply = work & (cuts.to(torch.int64) < amount_n) & (old_max > 0)
        old_used = torch.clamp(old_max, min=1)
        new_used = old_used - 1
        rounded = (hp_out.to(torch.int64) * new_used + old_used // 2) // old_used
        next_hp = torch.minimum(rounded, new_used)
        hp_out = torch.where(apply, next_hp.to(torch.int32), hp_out)
        max_out = torch.where(apply, new_used.to(torch.int32), max_out)
        cuts = cuts + apply.to(torch.int32)
        work = work & (hp_out > 0) & (max_out > 0)
    return ArrayHealthResult(hp_out, max_out, cuts, errors)
