"""Batched Torch integer lowering of the same Program IR.

Compile expressions once into a prepared tensor kernel. Mixed 0/1 sides live
in one batch; inactive and failed rows keep the original snapshot. This is not
a seven-rule engine or a GPU bytecode VM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ast import Bin, ClampMin, Cmp, Const, Name, Not, Program, Select, validate
from .schema import FIELDS, SIDE_ALIASES

I32_MIN, I32_MAX = -2147483648, 2147483647
ERR_ACTOR, ERR_OVERFLOW = 1, 2
_CMP = {
    'eq': torch.eq, 'ne': torch.ne, 'lt': torch.lt, 'le': torch.le,
    'gt': torch.gt, 'ge': torch.ge,
}


@dataclass(frozen=True)
class ArrayRuleResult:
    states: torch.Tensor
    errors: torch.Tensor


def _in_i32(value: int) -> bool:
    return I32_MIN <= int(value) <= I32_MAX


def _false(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros(ref.shape, dtype=torch.bool, device=ref.device)


def _compile(expr):
    if isinstance(expr, Const):
        if type(expr.value) is not int:
            raise ValueError('Exact integer IR constant required')
        value = int(expr.value)
        if not _in_i32(value):
            raise ValueError('IR constant exceeds int32 domain')

        def fn(env, value=value):
            ref = env['side']
            return torch.full(ref.shape, value, dtype=torch.int64, device=ref.device), _false(ref)

        return fn
    if isinstance(expr, Name):
        ident = expr.ident

        def fn(env, ident=ident):
            value = env[ident]
            return value, _false(value)

        return fn
    if isinstance(expr, Not):
        inner = _compile(expr.inner)

        def fn(env, inner=inner):
            value, overflow = inner(env)
            return (~value.bool()).to(torch.int64), overflow

        return fn
    if isinstance(expr, ClampMin):
        if type(expr.lo) is not int:
            raise ValueError('Exact integer IR clamp required')
        if not _in_i32(expr.lo):
            raise ValueError('IR constant exceeds int32 domain')
        inner = _compile(expr.inner)
        lo = int(expr.lo)

        def fn(env, inner=inner, lo=lo):
            value, overflow = inner(env)
            return value.clamp(min=lo), overflow

        return fn
    if isinstance(expr, Select):
        cond, then, orelse = _compile(expr.cond), _compile(expr.then), _compile(expr.orelse)

        def fn(env, cond=cond, then=then, orelse=orelse):
            flag, o_flag = cond(env)
            left, o_left = then(env)
            right, o_right = orelse(env)
            take = flag.bool()
            return torch.where(take, left, right), o_flag | torch.where(take, o_left, o_right)

        return fn
    if isinstance(expr, Cmp):
        left, right, op = _compile(expr.left), _compile(expr.right), _CMP[expr.op]

        def fn(env, left=left, right=right, op=op):
            a, oa = left(env)
            b, ob = right(env)
            return op(a, b).to(torch.int64), oa | ob

        return fn
    if isinstance(expr, Bin):
        left, right, op = _compile(expr.left), _compile(expr.right), expr.op
        if op in ('and', 'or'):
            combine = (lambda a, b: a.bool() & b.bool()) if op == 'and' else (lambda a, b: a.bool() | b.bool())

            def fn(env, left=left, right=right, combine=combine):
                a, oa = left(env)
                b, ob = right(env)
                return combine(a, b).to(torch.int64), oa | ob

            return fn
        if op in ('min', 'max'):
            combine = torch.minimum if op == 'min' else torch.maximum

            def fn(env, left=left, right=right, combine=combine):
                a, oa = left(env)
                b, ob = right(env)
                return combine(a, b), oa | ob

            return fn
        if op in ('add', 'sub'):
            def fn(env, left=left, right=right, op=op):
                a, oa = left(env)
                b, ob = right(env)
                value = a + b if op == 'add' else a - b
                return value, oa | ob | (value < I32_MIN) | (value > I32_MAX)

            return fn
    raise ValueError(f'cannot lower {type(expr).__name__}')


def _target(field, field_index, aliases):
    if field in field_index:
        idx = field_index[field]

        def fn(sides, idx=idx):
            return torch.full_like(sides, idx)

        return fn
    a_name, b_name = next((a, b) for alias, a, b in aliases if alias == field)
    ia, ib = field_index[a_name], field_index[b_name]

    def fn(sides, ia=ia, ib=ib):
        return torch.where(sides == 0, torch.full_like(sides, ia), torch.full_like(sides, ib))

    return fn


def lower_array(program: Program):
    validate(program)
    fields = program.fields or FIELDS
    aliases = program.aliases if program.aliases is not None else SIDE_ALIASES
    field_index = {name: i for i, name in enumerate(fields)}
    width = len(fields)
    skip_fn = _compile(program.skip_when)
    bind_fns = tuple((bind.name, _compile(bind.expr)) for bind in program.binds)
    store_fns = tuple((_target(store.field, field_index, aliases), _compile(store.expr))
                      for store in program.stores)

    def step(x: torch.Tensor, sides: torch.Tensor, active: torch.Tensor) -> ArrayRuleResult:
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != width
                or x.dtype != torch.int32 or not isinstance(sides, torch.Tensor)
                or sides.ndim != 1 or sides.dtype != torch.int64
                or not isinstance(active, torch.Tensor) or active.ndim != 1
                or active.dtype != torch.bool or sides.shape[0] != x.shape[0]
                or active.shape[0] != x.shape[0] or sides.device != x.device
                or active.device != x.device):
            raise ValueError(f'expected int32[B,{width}], int64[B], bool[B] on one device')
        env = {name: x[:, idx].to(torch.int64) for name, idx in field_index.items()}
        is_a = sides == 0
        for alias, a_name, b_name in aliases:
            env[alias] = torch.where(is_a, env[a_name], env[b_name])
        env['side'] = sides
        skip_v, overflow = skip_fn(env)
        skip = skip_v.bool()
        for name, fn in bind_fns:
            value, more = fn(env)
            env[name] = value
            overflow = overflow | ((~skip) & more)
        y = x.clone()
        store_overflow = _false(sides)
        for target_fn, value_fn in store_fns:
            value, more = value_fn(env)
            store_overflow = store_overflow | more | (value < I32_MIN) | (value > I32_MAX)
            y = y.scatter(1, target_fn(sides).unsqueeze(1), value.to(torch.int32).unsqueeze(1))
        overflow = overflow | ((~skip) & store_overflow)
        invalid = (sides != 0) & (sides != 1)
        errors = (active & invalid).to(torch.int32) * ERR_ACTOR
        errors = errors + (active & ~invalid & overflow).to(torch.int32) * ERR_OVERFLOW
        live = active & (errors == 0) & ~skip
        return ArrayRuleResult(torch.where(live.unsqueeze(1), y, x), errors)

    step.program_name = program.name
    step.program_version = program.version
    return step
