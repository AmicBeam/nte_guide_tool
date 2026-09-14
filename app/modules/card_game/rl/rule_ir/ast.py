"""Constrained IR. Arbitrary Python callbacks are rejected, not silently lowered."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: int


@dataclass(frozen=True)
class Name(Expr):
    ident: str


@dataclass(frozen=True)
class Bin(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Cmp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Not(Expr):
    inner: Expr


@dataclass(frozen=True)
class Select(Expr):
    cond: Expr
    then: Expr
    orelse: Expr


@dataclass(frozen=True)
class ClampMin(Expr):
    inner: Expr
    lo: int


@dataclass(frozen=True)
class Bind:
    name: str
    expr: Expr


@dataclass(frozen=True)
class Store:
    field: str
    expr: Expr


@dataclass(frozen=True)
class Program:
    name: str
    version: str
    skip_when: Expr
    binds: tuple[Bind, ...]
    stores: tuple[Store, ...]
    fields: tuple[str, ...] | None = None
    aliases: tuple[tuple[str, str, str], ...] | None = None


ALLOWED_BIN = frozenset({'add', 'sub', 'min', 'max', 'and', 'or'})
ALLOWED_CMP = frozenset({'eq', 'ne', 'lt', 'le', 'gt', 'ge'})


def validate(program: Program) -> None:
    from .schema import FIELDS, SIDE_ALIASES
    if program.skip_when is None or not program.binds or not program.stores:
        raise ValueError('IR program is empty')
    FIELDS = program.fields or FIELDS
    SIDE_ALIASES = program.aliases if program.aliases is not None else SIDE_ALIASES
    import re
    identifiers = [program.name, *FIELDS, *(a for a, _, _ in SIDE_ALIASES), *(b.name for b in program.binds)]
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in identifiers):
        raise ValueError('IR identifiers must be valid code-generation symbols')
    if len(set(FIELDS)) != len(FIELDS) or 'side' in FIELDS:
        raise ValueError('IR layout has duplicate or reserved fields')
    if any(a not in FIELDS or b not in FIELDS for _, a, b in SIDE_ALIASES):
        raise ValueError('IR alias targets an unknown field')
    alias_names = [alias for alias, _, _ in SIDE_ALIASES]
    if len(set(alias_names)) != len(alias_names) or set(alias_names) & (set(FIELDS) | {'side'}):
        raise ValueError('IR alias conflicts with layout')
    aliases = set(alias_names)
    seen = {'side'} | set(FIELDS) | aliases
    _check_expr(program.skip_when, seen)
    for bind in program.binds:
        _check_expr(bind.expr, seen)
        if bind.name in seen:
            raise ValueError(f'rebound name {bind.name!r}')
        seen.add(bind.name)
    for store in program.stores:
        if store.field not in FIELDS and store.field not in aliases:
            raise ValueError(f'unknown store field {store.field!r}')
        _check_expr(store.expr, seen)


def _check_expr(expr: Expr, seen: set[str]) -> None:
    if isinstance(expr, Const):
        return
    if isinstance(expr, Name):
        if expr.ident not in seen:
            raise ValueError(f'unbound {expr.ident!r}')
        return
    if isinstance(expr, Bin):
        if expr.op not in ALLOWED_BIN:
            raise ValueError(f'unsupported binop {expr.op!r}')
        _check_expr(expr.left, seen)
        _check_expr(expr.right, seen)
        return
    if isinstance(expr, Cmp):
        if expr.op not in ALLOWED_CMP:
            raise ValueError(f'unsupported cmp {expr.op!r}')
        _check_expr(expr.left, seen)
        _check_expr(expr.right, seen)
        return
    if isinstance(expr, Not):
        _check_expr(expr.inner, seen)
        return
    if isinstance(expr, Select):
        _check_expr(expr.cond, seen)
        _check_expr(expr.then, seen)
        _check_expr(expr.orelse, seen)
        return
    if isinstance(expr, ClampMin):
        _check_expr(expr.inner, seen)
        return
    raise ValueError(f'unsupported IR node {type(expr).__name__}')
