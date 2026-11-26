"""No-cache fast path derived from canonical legacy-effect conversion.

An ordinary flags dictionary with none of the canonical legacy keys produces
no effects. All other inputs execute the unchanged canonical conversion body.
No mutable flags/effects/provider result is cached, so legal-action trials,
mimic, equipment and in-place state edits need no invalidation protocol.
"""
import ast
import inspect
import textwrap
from hashlib import sha256
from copy import deepcopy

from ...engine.duel_v2 import modifiers
from ...engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity


class EffectQuerySpecializationError(RuntimeError):
    pass


def _same(left, text):
    return ast.dump(left) == ast.dump(ast.parse(text).body[0])


def derive_legacy_effects(original=modifiers.legacy_effects):
    """Insert one absence guard; retain the complete original fallback body.

    Dependencies come from literal ``flags.get`` and ``flags[...]`` reads.
    Any other flags access or changed initialization/empty branch fails closed.
    The exact-dict check preserves custom mapping/subclass method semantics.
    """
    lines, start = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise EffectQuerySpecializationError('Canonical legacy function required')
    function = tree.body[0]
    shape = deepcopy(function)
    # Python 3.12 added an empty type_params field to ordinary function ASTs.
    # Normalize that nonsemantic addition when loading a 3.10 freeze elsewhere.
    for node in ast.walk(shape):
        if hasattr(node,'type_params') and node.type_params == []:
            delattr(node,'type_params')
    # Functional safety guard: an unrecognized formula/default or additional
    # write must be reviewed before claiming all-absent flags imply no effect.
    if sha256(ast.dump(shape,include_attributes=False).encode()).hexdigest() != 'e265dc18da3cc0fe0b17e0459bda542b4c4ebe32d4671959297cabab5d34396d':
        raise EffectQuerySpecializationError('Canonical legacy conversion shape changed')
    if (function.name != 'legacy_effects' or function.decorator_list
            or [a.arg for a in function.args.args] != ['holder']
            or function.args.posonlyargs or function.args.kwonlyargs
            or function.args.vararg or function.args.kwarg or function.args.defaults):
        raise EffectQuerySpecializationError('Canonical legacy signature changed')
    if (len(function.body) < 3
            or not _same(function.body[0], "flags = holder.get('flags') or {}")
            or not _same(function.body[1], 'if not flags:\n    return []')):
        raise EffectQuerySpecializationError('Canonical legacy initialization changed')
    dependencies = set()
    reads = set()
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'flags'):
            if (node.func.attr != 'get' or not 1 <= len(node.args) <= 2 or node.keywords
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)):
                raise EffectQuerySpecializationError('Unsupported canonical flags call')
            dependencies.add(node.args[0].value)
            reads.add(id(node.func.value))
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'flags'):
            if (not isinstance(node.ctx, ast.Load) or not isinstance(node.slice, ast.Constant)
                    or not isinstance(node.slice.value, str)):
                raise EffectQuerySpecializationError('Unsupported canonical flags subscript')
            dependencies.add(node.slice.value)
            reads.add(id(node.value))
    # Reject aliases, iteration or passing flags to another function before
    # returning early. The only other accesses are assignment and truth test.
    reads.add(id(function.body[0].targets[0]))
    reads.add(id(function.body[1].test.operand))
    if any(id(node) not in reads for node in ast.walk(function)
            if isinstance(node, ast.Name) and node.id == 'flags'):
        raise EffectQuerySpecializationError('Untracked canonical flags dependency')
    # A newly added key may introduce a nonzero absent default. Fail closed
    # until its missing-key behavior has been checked against the real rules.
    expected = {'atk_buff', 'atk_buff_expiring', 'atk_debuff', 'form_attack',
                'timed_attack', 'next_attack', 'damage_immunity_until'}
    if dependencies != expected:
        raise EffectQuerySpecializationError('Canonical legacy dependency set changed')
    for entity_type in (PlayerEntity, CharacterEntity, CardEntity):
        if (entity_type.get is not dict.get
                or entity_type.__getattribute__ is not dict.__getattribute__):
            raise EffectQuerySpecializationError('Canonical entity holder lookup changed')
    guard = ast.parse('if type(flags) is dict and '+
        ' and '.join(repr(key)+' not in flags' for key in sorted(dependencies))+':\n'
        '    _query_holder_type = type(holder)\n'
        '    if (_query_holder_type is dict or '
        '_query_holder_type is _QueryPlayerEntity or '
        '_query_holder_type is _QueryCharacterEntity or '
        '_query_holder_type is _QueryCardEntity):\n'
        '        return []').body[0]
    function.body.insert(2, ast.copy_location(guard, function.body[1]))
    function.name = 'query_legacy_effects'
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start-1)
    namespace = dict(original.__globals__)
    namespace.update(_QueryPlayerEntity=PlayerEntity,
        _QueryCharacterEntity=CharacterEntity, _QueryCardEntity=CardEntity)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    result = namespace[function.name]
    result.__module__ = __name__
    return result


query_legacy_effects = derive_legacy_effects()


def derive_all_effects(original=modifiers.all_effects):
    """Return fresh empty effects only for exact, demonstrably empty holders.

    Unknown holders never incur extra get calls. Unknown stored containers
    reach canonical list conversion before any flags read, preserving iterator
    errors and side effects. Custom flag truth values also keep the original
    path. The complete canonical body remains the fallback, with no wrapper
    call that could recursively substitute itself during isolated capture.
    """
    lines, start = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(lines)))
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or not _same(tree.body[0],
                "def all_effects(holder):\n    return list(holder.get('effects', [])) + legacy_effects(holder)")):
        raise EffectQuerySpecializationError('Canonical all-effects conversion shape changed')
    for entity_type in (PlayerEntity, CharacterEntity, CardEntity):
        if (entity_type.get is not dict.get
                or entity_type.__getattribute__ is not dict.__getattribute__):
            raise EffectQuerySpecializationError('Canonical entity holder lookup changed')
    prefix = ast.parse(
        '_query_holder_type = type(holder)\n'
        'if (_query_holder_type is dict or _query_holder_type is _QueryPlayerEntity or '
        '_query_holder_type is _QueryCharacterEntity or _query_holder_type is _QueryCardEntity):\n'
        "    _query_stored = holder.get('effects', ())\n"
        '    _query_stored_type = type(_query_stored)\n'
        '    if (_query_stored_type is list or _query_stored_type is tuple) and not _query_stored:\n'
        "        _query_flags = holder.get('flags')\n"
        '        if _query_flags is None or (type(_query_flags) is dict and not _query_flags):\n'
        '            return []').body
    function = tree.body[0]
    function.body[:0] = prefix
    function.name = 'query_all_effects'
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start-1)
    namespace = dict(original.__globals__)
    namespace.update(_QueryPlayerEntity=PlayerEntity,
        _QueryCharacterEntity=CharacterEntity, _QueryCardEntity=CardEntity)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    result = namespace[function.name]
    result.__module__ = __name__
    return result


query_all_effects = derive_all_effects()
