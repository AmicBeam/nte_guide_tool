"""Lower IR to a batched PyTorch integer kernel."""
from __future__ import annotations

import torch

from .ast import Bind, Bin, ClampMin, Cmp, Const, Name, Not, Program, Select, Store, validate
from .schema import FIELD_INDEX, FIELDS, N_FIELDS, SIDE_ALIASES


def _eval(expr, env: dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(expr, Const):
        ref = next(iter(env.values()))
        return torch.full_like(ref, expr.value)
    if isinstance(expr, Name):
        return env[expr.ident]
    if isinstance(expr, Not):
        return (~_eval(expr.inner, env).bool()).to(torch.int32)
    if isinstance(expr, ClampMin):
        return _eval(expr.inner, env).clamp(min=expr.lo)
    if isinstance(expr, Select):
        cond = _eval(expr.cond, env).bool()
        return torch.where(cond, _eval(expr.then, env), _eval(expr.orelse, env))
    if isinstance(expr, Cmp):
        left, right = _eval(expr.left, env), _eval(expr.right, env)
        ops = {
            'eq': torch.eq, 'ne': torch.ne, 'lt': torch.lt, 'le': torch.le,
            'gt': torch.gt, 'ge': torch.ge,
        }
        return ops[expr.op](left, right).to(torch.int32)
    if isinstance(expr, Bin):
        left, right = _eval(expr.left, env), _eval(expr.right, env)
        if expr.op == 'add':
            return left + right
        if expr.op == 'sub':
            return left - right
        if expr.op == 'min':
            return torch.minimum(left, right)
        if expr.op == 'max':
            return torch.maximum(left, right)
        if expr.op == 'and':
            return (left.bool() & right.bool()).to(torch.int32)
        if expr.op == 'or':
            return (left.bool() | right.bool()).to(torch.int32)
    raise ValueError(f'cannot lower {type(expr).__name__}')


def _aliases(x: torch.Tensor, side: int, fields=FIELDS, aliases=SIDE_ALIASES) -> dict[str, torch.Tensor]:
    field_index = {name: i for i, name in enumerate(fields)}
    env = {name: x[:, field_index[name]] for name in fields}
    pick = 2 if side else 1
    for alias, a_name, b_name in aliases:
        env[alias] = x[:, field_index[a_name if pick == 1 else b_name]]
    env['side'] = torch.full((x.shape[0],), side, dtype=torch.int32, device=x.device)
    return env


def lower_cpu(program: Program):
    validate(program)
    fields = program.fields or FIELDS
    field_index = {name: i for i, name in enumerate(fields)}
    aliases = program.aliases if program.aliases is not None else SIDE_ALIASES

    def step(x: torch.Tensor, side: int) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != len(fields):
            raise ValueError(f'expected [N,{len(fields)}] state')
        env = _aliases(x, side, fields, aliases)
        skip = _eval(program.skip_when, env).bool()
        live = ~skip
        for bind in program.binds:
            env[bind.name] = _eval(bind.expr, env)
        y = x.clone()
        for store in program.stores:
            value = _eval(store.expr, env)
            if store.field in field_index:
                idx = field_index[store.field]
            else:
                a_name, b_name = next((a, b) for alias, a, b in aliases if alias == store.field)
                idx = field_index[a_name if side == 0 else b_name]
            y[:, idx] = torch.where(live, value, x[:, idx])
        return y

    step.program_name = program.name
    step.program_version = program.version
    return step
