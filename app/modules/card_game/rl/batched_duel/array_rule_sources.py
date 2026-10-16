"""Canonical-source array specializations for complete native compilation.

reduce_max_hp is derived from formal AST. Only the nonnegative Decimal ratio
round_half_up site becomes the exact int64 formula
(hp * new_max + old_max // 2) // old_max. Per-unit looping, min, preconditions,
logs, knockdowns, victory, events and in-place lifetime stay canonical.
Unsupported AST or non-integer domains fail closed. Production rules are not
patched. Substitution helpers rebind AST and global names for a frozen
ProgramCapturer without mutating original modules. This is not a GPU VM fix
or a readiness claim.
"""
from __future__ import annotations

import ast
import copy
import inspect
import textwrap
import types

from app.modules.card_game.engine.duel_v2 import state as duel_state
from app.modules.card_game.rl.batched_duel.gpu_owned_transitions import (
    scratch_apply_action,
    scratch_simulate_action,
)
from app.modules.card_game.rl.batched_duel.gpu_policy_projection import (
    ProjectionSpecializationError,
    _compile,
)


class ArrayRuleSpecializationError(RuntimeError):
    """Fail-closed error when canonical array-rule AST cannot be specialized."""


_REQUIRED_NAMES = frozenset({
    'PLAYER_ID', 'amount', 'cid', 'cut', 'esper', 'finished', 'group_id', 'h',
    'hero', 'is_esper', 'knockdowns', 'log', 'max', 'min', 'new_max', 'old_max',
    'range', 'settle', 'side', 'state', 'victory', 'int',
})
_FORBIDDEN_NAMES = frozenset({'Decimal', 'round_half_up', 'ROUND_HALF_UP', 'float'})


def _name(node, ident):
    return isinstance(node, ast.Name) and node.id == ident


def _const(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def _call_name(node, ident, n_args=None):
    if not isinstance(node, ast.Call) or not _name(node.func, ident):
        return False
    if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
        return False
    if n_args is not None and len(node.args) != n_args:
        return False
    return True


def _decimal_of(node, inner_pred):
    return _call_name(node, 'Decimal', 1) and inner_pred(node.args[0])


def _hp_load(node):
    return (isinstance(node, ast.Subscript) and _name(node.value, 'h')
            and _const(node.slice, 'hp') and isinstance(node.ctx, ast.Load))


def canonical_ratio_operands(node):
    """Return (hp, old_max, new_max) for the canonical Decimal ratio, else None.

    Matches only round_half_up(Decimal(h['hp']) / Decimal(old_max) * Decimal(new_max)).
    """
    if not _call_name(node, 'round_half_up', 1):
        return None
    expr = node.args[0]
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult)):
        return None
    left, right = expr.left, expr.right
    if not _decimal_of(right, lambda inner: _name(inner, 'new_max')):
        return None
    if not (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div)):
        return None
    hp_dec, old_dec = left.left, left.right
    if not _decimal_of(hp_dec, _hp_load):
        return None
    if not _decimal_of(old_dec, lambda inner: _name(inner, 'old_max')):
        return None
    return hp_dec.args[0], old_dec.args[0], right.args[0]


def integer_ratio_expression(hp, old_max, new_max):
    """Exact nonnegative int64 ROUND_HALF_UP of hp * new_max / old_max."""
    hp = copy.deepcopy(hp)
    old_used = copy.deepcopy(old_max)
    new_max = copy.deepcopy(new_max)
    checked_hp = ast.Call(func=ast.Name('_array_ratio_hp', ast.Load()),
                          args=[hp, copy.deepcopy(old_max), copy.deepcopy(new_max)], keywords=[])
    product = ast.BinOp(left=checked_hp, op=ast.Mult(), right=new_max)
    half = ast.BinOp(left=old_used, op=ast.FloorDiv(), right=ast.Constant(2))
    return ast.BinOp(
        left=ast.BinOp(left=product, op=ast.Add(), right=half),
        op=ast.FloorDiv(),
        right=copy.deepcopy(old_max),
    )


def _array_ratio_hp(hp, old_max, new_max):
    """Guard the exact integer specialization before any HP store occurs."""
    if (not isinstance(hp, int) or isinstance(hp, bool)
            or not isinstance(old_max, int) or isinstance(old_max, bool)
            or not isinstance(new_max, int) or isinstance(new_max, bool)):
        raise ValueError('Native max-HP ratio requires integer fields')
    if (hp < 0 or hp > 2147483647 or old_max < 1 or old_max > 2147483647
            or new_max != old_max-1):
        raise ValueError('Native max-HP ratio exceeds its exact integer domain')
    return hp


class ReduceMaxHpRatioTransformer(ast.NodeTransformer):
    """Replace the single canonical Decimal ratio; leave every other node."""

    def __init__(self):
        self.sites = 0
        self.min_sites = 0
        self.hp_assign = 0
        self.loops = 0
        self.old_max = 0
        self.new_max = 0

    def visit_FunctionDef(self, node):
        args = [item.arg for item in node.args.args]
        kwonly = [item.arg for item in node.args.kwonlyargs]
        if (args != ['state', 'side', 'cid', 'amount']
                or kwonly != ['source', 'settle', 'group_id']
                or node.args.posonlyargs or node.args.vararg or node.args.kwarg
                or node.args.defaults or node.decorator_list):
            raise ArrayRuleSpecializationError('Canonical reduce_max_hp signature changed')
        defaults = node.args.kw_defaults
        if (len(defaults) != 3 or not _const(defaults[0], None)
                or not _const(defaults[1], True) or not _const(defaults[2], None)):
            raise ArrayRuleSpecializationError('Canonical reduce_max_hp signature changed')
        return self.generic_visit(node)

    def visit_For(self, node):
        if (_name(node.target, '_') and isinstance(node.iter, ast.Call)
                and _name(node.iter.func, 'range') and len(node.iter.args) == 1
                and _name(node.iter.args[0], 'amount') and not node.iter.keywords):
            self.loops += 1
        return self.generic_visit(node)

    def visit_Assign(self, node):
        target = node.targets[0] if len(node.targets) == 1 else None
        if (_name(target, 'old_max') and isinstance(node.value, ast.Subscript)
                and _name(node.value.value, 'h') and _const(node.value.slice, 'max_hp')):
            self.old_max += 1
        if (_name(target, 'new_max') and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Sub)
                and _name(node.value.left, 'old_max') and _const(node.value.right, 1)):
            self.new_max += 1
        if (isinstance(target, ast.Subscript) and _name(target.value, 'h')
                and _const(target.slice, 'hp')
                and _call_name(node.value, 'min', 2)
                and _name(node.value.args[1], 'new_max')
                and canonical_ratio_operands(node.value.args[0]) is not None):
            self.hp_assign += 1
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (_call_name(node, 'min', 2) and _name(node.args[1], 'new_max')
                and canonical_ratio_operands(node.args[0]) is not None):
            self.min_sites += 1
        node = self.generic_visit(node)
        operands = canonical_ratio_operands(node)
        if operands is None:
            return node
        self.sites += 1
        replacement = integer_ratio_expression(*operands)
        return ast.copy_location(replacement, node)

    def visit_Module(self, node):
        node = self.generic_visit(node)
        if (self.sites != 1 or self.min_sites != 1 or self.hp_assign != 1
                or self.loops != 1 or self.old_max != 1 or self.new_max != 1):
            raise ArrayRuleSpecializationError(
                'Canonical reduce_max_hp ratio structure changed')
        leftover = []
        names = set()
        constants = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                names.add(item.id)
                if item.id in _FORBIDDEN_NAMES:
                    leftover.append(item.id)
            elif isinstance(item, ast.BinOp) and isinstance(item.op, ast.Div):
                leftover.append('true_div')
            elif isinstance(item, ast.Constant):
                constants.add(item.value)
        missing = _REQUIRED_NAMES - names
        if leftover:
            raise ArrayRuleSpecializationError(
                'Specialized max-HP rule kept non-integer domain: ' + ','.join(leftover))
        if missing:
            raise ArrayRuleSpecializationError(
                'Canonical reduce_max_hp dropped ' + ','.join(sorted(missing)))
        if 'life_locked' not in constants or 'max_hp' not in constants:
            raise ArrayRuleSpecializationError('Canonical reduce_max_hp preconditions changed')
        return node


def _assert_canonical_round_half_up():
    tree = ast.parse(textwrap.dedent(inspect.getsource(duel_state.round_half_up)))
    expected = ast.parse(
        "def round_half_up(value):\n"
        "    return int(Decimal(value).quantize(Decimal('1'), rounding=ROUND_HALF_UP))\n"
    )
    if ast.dump(tree) != ast.dump(expected):
        raise ArrayRuleSpecializationError('Canonical round_half_up changed')


def specialize_canonical_source(original, name, transform, extra=None, extra_arg=None):
    """Compile a specialized sibling of original. Does not mutate original globals."""
    extra = dict(extra or {})
    if isinstance(transform, ReduceMaxHpRatioTransformer):
        extra['_array_ratio_hp'] = _array_ratio_hp
    try:
        result, _namespace = _compile(
            original, name, transform, extra=extra, extra_arg=extra_arg)
    except ProjectionSpecializationError as exc:
        raise ArrayRuleSpecializationError(str(exc)) from exc
    if result.__globals__ is original.__globals__:
        raise ArrayRuleSpecializationError('Specialization cloned production globals')
    result.__module__ = __name__
    return result


def rebind_function_globals(fn, replacements, *, name=None):
    """Return a sibling function with selected globals replaced.

    replacements maps global names to new values. The original function object,
    its module dict and its code object are left unchanged. Missing names fail
    closed rather than inserting a silent binding.
    """
    if not isinstance(fn, types.FunctionType):
        raise ArrayRuleSpecializationError('Canonical function required')
    if fn.__closure__:
        raise ArrayRuleSpecializationError('Closed functions cannot be rebound')
    if any(type(key) is not str for key in replacements):
        raise ArrayRuleSpecializationError('Global substitution names required')
    namespace = dict(fn.__globals__)
    for key, value in replacements.items():
        if key not in namespace:
            raise ArrayRuleSpecializationError(f'Canonical global {key!r} is missing')
        namespace[key] = value
    new = types.FunctionType(
        fn.__code__, namespace, name or fn.__name__, fn.__defaults__, None)
    new.__kwdefaults__ = None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
    new.__module__ = __name__
    new.__qualname__ = name or fn.__qualname__
    new.__doc__ = fn.__doc__
    new.__annotations__ = dict(fn.__annotations__)
    new.__dict__.update(fn.__dict__)
    return new


def substitute_ast_names(tree, name_map):
    """Rewrite Load Name nodes. Store names and production modules stay intact."""
    if not isinstance(name_map, dict) or any(
            type(key) is not str or type(value) is not str for key, value in name_map.items()):
        raise ArrayRuleSpecializationError('AST name substitutions must be strings')

    class _Names(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in name_map:
                return ast.copy_location(ast.Name(name_map[node.id], ast.Load()), node)
            return node

    return ast.fix_missing_locations(_Names().visit(tree))


def _specialize_reduce_max_hp():
    original = duel_state.reduce_max_hp
    _assert_canonical_round_half_up()
    result = specialize_canonical_source(
        original, 'array_reduce_max_hp', ReduceMaxHpRatioTransformer())
    names = result.__code__.co_names
    if 'Decimal' in names or 'round_half_up' in names:
        raise ArrayRuleSpecializationError('Specialized max-HP rule still loads Decimal')
    if result is original:
        raise ArrayRuleSpecializationError('Specialization reused production reduce_max_hp')
    if duel_state.reduce_max_hp is not original:
        raise ArrayRuleSpecializationError('Production reduce_max_hp was replaced')
    return result


_CANONICAL_REDUCE_MAX_HP = duel_state.reduce_max_hp
array_reduce_max_hp = _specialize_reduce_max_hp()


def canonical_array_rule_identities():
    """Map canonical function objects to specialized siblings for identity capture."""
    current = duel_state.reduce_max_hp
    if current is not _CANONICAL_REDUCE_MAX_HP:
        raise ArrayRuleSpecializationError('Production reduce_max_hp was replaced')
    if current is array_reduce_max_hp:
        raise ArrayRuleSpecializationError('Production reduce_max_hp was replaced')
    return {current: array_reduce_max_hp}


def array_rule_bindings_for(fn):
    """Name-to-replacement globals present on fn, keyed by canonical identity."""
    if not isinstance(fn, types.FunctionType):
        raise ArrayRuleSpecializationError('Canonical function required')
    identities = canonical_array_rule_identities()
    bound = {}
    used = set()
    stack = [fn.__code__]
    while stack:
        code = stack.pop()
        used.update(code.co_names)
        stack.extend(item for item in code.co_consts if inspect.iscode(item))
    for original, replacement in identities.items():
        for name, value in fn.__globals__.items():
            if value is original and name in used:
                bound[name] = replacement
    return bound


def array_scratch_apply_action(state, side, action):
    """Owned-scratch apply; identity capture substitutes array_reduce_max_hp."""
    return scratch_apply_action(state, side, action)


def array_scratch_simulate_action(state, side, action):
    """Owned-scratch simulate; identity capture substitutes array_reduce_max_hp."""
    return scratch_simulate_action(state, side, action)
