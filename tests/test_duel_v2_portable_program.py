"""Tests for duel v2 portable bytecode program export and validation."""

import copy
import contextvars
import json
import math
import sys
import types
from typing import Any
from unittest import mock
import unittest

from app.modules.card_game.rl.rule_ir.portable_program import (
    ALL_8_AND_DERIVED,
    DEFAULT_ALLOWED_MODULE_PREFIXES,
    DEFAULT_EXPLICIT_INTRINSICS,
    ExportError,
    FIVE_UNION,
    FORMAT_VERSION,
    FrozenRuleProgram,
    ValidationError,
    capture_duel_program,
    capture_program,
    compute_identity,
)
from app.modules.card_game.content.duel_v2.registry import KITS, _ensure_kits


# --- Test fixtures for pure small functions ---

def _pure_loop_and_branch(n: int) -> int:
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total += i
        else:
            total -= 1
    return total


def _pure_nested_closure(factor: int):
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier


def _pure_generator(limit: int):
    val = 0
    while val < limit:
        yield val
        val += 1


def _pure_args_and_kwargs(a: int = 10, *, b: int = 20, **kwargs: Any) -> int:
    return a + b + len(kwargs)


def _pure_math_symbols(x: float) -> bool:
    import math as local_math
    from math import isfinite as local_isfinite
    return local_math.isfinite(x) and local_isfinite(x)


def _pure_add(x: int, y: int) -> int:
    return x + y


def _pure_sub(x: int, y: int) -> int:
    return x - y


def _closure_bias(amount):
    def shifted(value):
        return value+amount
    return shifted


CLOSURE_CALLBACKS=(_closure_bias(1),_closure_bias(2))


def _closure_pair(value):
    return CLOSURE_CALLBACKS[0](value),CLOSURE_CALLBACKS[1](value)


def test_distinct_static_closures_keep_separate_function_identities():
    prog=capture_program({'pair':_closure_pair},allowed_module_prefixes=('tests',),
                         scope_character_ids=None,scope_card_ids=None)
    prog.validate()
    shifted=[fn for fn in prog.functions if fn['name']=='shifted']
    assert len(shifted)==2
    assert shifted[0]['id']!=shifted[1]['id']
    assert shifted[0]['closure']!=shifted[1]['closure']


def _same_line_comprehensions(n):
    return [x for x in range(n)]+[y for y in range(n)]


def test_same_line_comprehensions_do_not_collide():
    prog=capture_program({'entry':_same_line_comprehensions},allowed_module_prefixes=('tests',),
                         scope_character_ids=None,scope_card_ids=None)
    prog.validate()
    assert len(prog.codes)==2
    assert len({code['id'] for code in prog.codes})==2


PORTABLE_CONTEXT=contextvars.ContextVar('portable-test-context',default=True)


def _read_context():
    return PORTABLE_CONTEXT.get()


def _missing_relative_import():
    from .portable_missing_module_xyz import callback
    return callback()


def test_missing_import_never_becomes_an_invented_intrinsic():
    prog=capture_program({'entry':_missing_relative_import},allowed_module_prefixes=('tests',),
                         scope_character_ids=None,scope_card_ids=None)
    assert prog.unresolved
    assert not any('portable_missing_module_xyz' in name for name in prog.required_intrinsics)
    with unittest.TestCase().assertRaisesRegex(ValidationError,'unresolved references'):
        prog.validate()


def test_contextvar_declared_default_is_independent_of_active_context():
    token=PORTABLE_CONTEXT.set(False)
    try:
        prog=capture_program({'entry':_read_context},allowed_module_prefixes=('tests',),
                             scope_character_ids=None,scope_card_ids=None)
        prog.validate()
        item=next(c for c in prog.constants if c['kind']=='ContextVar')
        assert item['default']=={'kind':'singleton','name':'True'}
        assert PORTABLE_CONTEXT.get() is False
    finally:
        PORTABLE_CONTEXT.reset(token)


PURE_CALLBACK_MAP = {
    'add': _pure_add,
    'sub': _pure_sub,
}


class PureBaseClass:
    def base_method(self) -> str:
        return "base"


class PureSampleClass(PureBaseClass):
    class_attr = 42

    def __init__(self, value: int = 0) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @classmethod
    def create_default(cls):
        return cls(100)

    @staticmethod
    def static_helper(x: int) -> int:
        return x * 2

    def normal_method(self, x: int) -> int:
        return self._value + x


def _entry_using_pure_structures(x: int) -> int:
    calc = _pure_loop_and_branch(x)
    closure_fn = _pure_nested_closure(2)
    step1 = closure_fn(calc)
    obj = PureSampleClass.create_default()
    static_val = PureSampleClass.static_helper(step1)
    callback_val = PURE_CALLBACK_MAP['add'](static_val, obj.value)
    return callback_val


# --- Tests ---

def test_pure_small_program_export_and_validation():
    """Verify real instructions, targets, refs, and identity stability on small functions."""
    entries = {
        'main': _entry_using_pure_structures,
        'generator': _pure_generator,
        'args_kwargs': _pure_args_and_kwargs,
        'math_symbols': _pure_math_symbols,
    }

    prog = capture_program(
        entries,
        allowed_module_prefixes=('tests', 'app.modules.card_game'),
        scope_character_ids=None,
        scope_card_ids=None,
    )

    assert prog.format == FORMAT_VERSION
    assert prog.compiler_python.startswith("3.10")
    assert len(prog.identity) == 64
    assert prog.unresolved == []
    assert 'main' in prog.entries
    assert 'generator' in prog.entries
    assert 'args_kwargs' in prog.entries

    func_ids = {f['id'] for f in prog.functions}
    assert prog.entries['main'] in func_ids
    assert prog.entries['generator'] in func_ids

    # Find the loop_and_branch function
    loop_fn = next((f for f in prog.functions if f['name'] == '_pure_loop_and_branch'), None)
    assert loop_fn is not None
    assert len(loop_fn['instructions']) > 0

    # Verify real instructions and normalized jump targets
    has_jump = False
    for inst in loop_fn['instructions']:
        assert isinstance(inst['op'], str)
        assert inst['op'] != ''
        if inst['target'] is not None:
            has_jump = True
            assert isinstance(inst['target'], int)
            assert 0 <= inst['target'] < len(loop_fn['instructions'])
    assert has_jump, "Loop and branch must contain jump instructions"

    # Verify closure code captured in codes
    closure_code = next((c for c in prog.codes if c['name'] == 'multiplier'), None)
    assert closure_code is not None
    assert 'factor' in closure_code['freevars']

    # Verify generator function flags
    gen_fn = next((f for f in prog.functions if f['name'] == '_pure_generator'), None)
    assert gen_fn is not None
    # Generator flag: 0x20 in co_flags
    assert bool(gen_fn['flags'] & 0x20)

    # Verify class properties, methods, staticmethods, classmethods
    cls_desc = next((c for c in prog.classes if c['name'] == 'PureSampleClass'), None)
    assert cls_desc is not None
    assert 'value' in cls_desc['properties']
    assert cls_desc['properties']['value']['fget'] is not None
    assert 'create_default' in cls_desc['classmethods']
    assert 'static_helper' in cls_desc['staticmethods']
    assert 'normal_method' in cls_desc['methods']

    # Identity stability test: re-export produces the exact same identity
    prog2 = capture_program(
        entries,
        allowed_module_prefixes=('tests', 'app.modules.card_game'),
        scope_character_ids=None,
        scope_card_ids=None,
    )
    assert prog.identity == prog2.identity

    # Full validation
    prog.validate()


def test_validation_rejects_corrupted_data():
    """Verify fail-closed rejection on corrupted refs, illegal jumps, and bad identities."""
    entries = {'loop': _pure_loop_and_branch}
    prog = capture_program(
        entries,
        allowed_module_prefixes=('tests',),
        scope_character_ids=None,
        scope_card_ids=None,
    )
    prog.validate()

    # 1. Corrupted constant ref
    bad_data = prog.to_dict()
    bad_data['functions'][0]['constants'].append({"kind": "constant", "id": 999999})
    bad_data['identity'] = compute_identity(bad_data)
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "out of range"):
        bad_prog.validate()

    # 2. Corrupted function ref
    bad_data = prog.to_dict()
    bad_data['entries']['loop'] = "func:nonexistent:function"
    bad_data['identity'] = compute_identity(bad_data)
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "unknown function id"):
        bad_prog.validate()

    # 3. Corrupted jump target
    bad_data = prog.to_dict()
    for inst in bad_data['functions'][0]['instructions']:
        if inst['target'] is not None:
            inst['target'] = 88888
            break
    bad_data['identity'] = compute_identity(bad_data)
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "Illegal jump target"):
        bad_prog.validate()

    # 4. Corrupted identity
    bad_data = prog.to_dict()
    bad_data['identity'] = "0" * 64
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "Identity mismatch"):
        bad_prog.validate()

    # 5. Invalid format
    bad_data = prog.to_dict()
    bad_data['format'] = "invalid_format_v9"
    bad_data['identity'] = compute_identity(bad_data)
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "Program format mismatch"):
        bad_prog.validate()

    # 6. Rejection of ready=true
    bad_data = prog.to_dict()
    bad_data['ready'] = True
    bad_data['identity'] = compute_identity(bad_data)
    bad_prog = FrozenRuleProgram(bad_data)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "ready=true"):
        bad_prog.validate()


def test_rejection_of_unauthorized_module():
    """Verify fail-closed rejection for modules outside allowed prefixes."""
    entries = {'loop': _pure_loop_and_branch}
    with unittest.TestCase().assertRaisesRegex(ExportError, "not within allowed module prefixes"):
        capture_program(
            entries,
            allowed_module_prefixes=('app.modules.card_game.engine.duel_v2',),
            scope_character_ids=None,
            scope_card_ids=None,
        )


def test_rejection_of_external_source():
    """Verify fail-closed rejection for functions whose source is outside the repository."""
    # Create a code object with a simulated outside file path
    orig_code = _pure_loop_and_branch.__code__
    fake_code = orig_code.replace(co_filename="/tmp/external_unauthorized_script.py")
    fake_fn = types.FunctionType(fake_code, _pure_loop_and_branch.__globals__, "fake_fn")
    fake_fn.__module__ = _pure_loop_and_branch.__module__

    with unittest.TestCase().assertRaisesRegex(ExportError, "outside repository root"):
        capture_program(
            {'fake': fake_fn},
            allowed_module_prefixes=('tests',),
            scope_character_ids=None,
            scope_card_ids=None,
        )


def test_rejection_of_forbidden_builtin_io():
    """Verify builtins like open, eval, exec are rejected or marked unresolved."""
    def _io_func(path: str) -> str:
        f = open(path)
        return f.read()

    prog = capture_program(
        {'io_fn': _io_func},
        allowed_module_prefixes=('tests',),
        scope_character_ids=None,
        scope_card_ids=None,
    )
    # Must be added to unresolved and rejected by validate
    assert len(prog.unresolved) > 0
    assert any(u.get('name') == 'open' for u in prog.unresolved)
    with unittest.TestCase().assertRaisesRegex(ValidationError, "unresolved references"):
        prog.validate()


def test_rejection_of_unknown_global_object():
    """Verify unknown global objects are captured as unresolved rather than faked."""
    opaque_instance = object()

    def _func_with_opaque():
        return opaque_instance

    prog = capture_program(
        {'opaque_fn': _func_with_opaque},
        allowed_module_prefixes=('tests',),
        scope_character_ids=None,
        scope_card_ids=None,
    )
    assert len(prog.unresolved) > 0
    with unittest.TestCase().assertRaisesRegex(ValidationError, "unresolved references"):
        prog.validate()


def test_host_python_version_check():
    """Verify that host bytecode export explicitly refuses non-3.10 host Python."""
    entries = {'loop': _pure_loop_and_branch}
    with mock.patch.object(sys, 'version_info', (3, 13, 0, 'final', 0)):
        with unittest.TestCase().assertRaisesRegex(ExportError, "Host bytecode export must be run under CPython 3.10"):
            capture_program(
                entries,
                allowed_module_prefixes=('tests',),
                scope_character_ids=None,
                scope_card_ids=None,
            )


def test_capture_duel_program_real():
    """Verify real duel rules capture without game execution.

    Asserts all 6 required entries, source hashes, scoped character kits,
    all 107 card effects, and required intrinsics are fully captured.
    """
    _ensure_kits()
    prog = capture_duel_program()

    assert prog.format == FORMAT_VERSION
    assert prog.compiler_python.startswith("3.10")
    assert len(prog.identity) == 64

    # 1. Assert all 6 required entries are present (plus step alias)
    expected_entries = {
        'apply_action',
        'step',
        'legal_actions',
        'observe',
        'decision',
        'encode',
        'determinize',
    }
    for e in expected_entries:
        assert e in prog.entries, f"Missing required entry: {e}"

    # 2. Assert source hashes contain flow.py, projection.py, and catalog.json
    source_hashes = prog.source_hashes
    assert 'app/modules/card_game/content/duel_v2/catalog.json' in source_hashes
    assert any('flow.py' in p for p in source_hashes)
    assert any('projection.py' in p for p in source_hashes)
    assert any('cross_runtime.py' in p for p in source_hashes)
    assert any('cross_lineup.py' in p for p in source_hashes)
    assert any('recovery_sampler.py' in p for p in source_hashes)

    # 3. Assert all 13 characters in FIVE_UNION are captured
    class_names = {c['name'] for c in prog.classes}
    for cid in FIVE_UNION:
        kit = KITS[cid]
        kit_name = kit.__name__ if isinstance(kit, type) else type(kit).__name__
        assert kit_name in class_names, f"Missing kit class for character '{cid}' ({kit_name})"

    # 4. Assert all 107 cards in ALL_8_AND_DERIVED have effect function records
    all_func_names = {f['name'] for f in prog.functions} | {f['qualname'] for f in prog.functions}
    for card_id in ALL_8_AND_DERIVED:
        # Find effect function from kit
        found = False
        for cid in FIVE_UNION:
            kit = KITS[cid]
            effects = getattr(kit, 'effects', {})
            if card_id in effects:
                fn = effects[card_id]
                fn_name = fn.__name__
                fn_qual = fn.__qualname__
                if fn_name in all_func_names or fn_qual in all_func_names:
                    found = True
                    break
        assert found, f"Card effect for '{card_id}' not found in captured functions"

    # 5. Assert required intrinsics are listed
    intrinsics = set(prog.required_intrinsics)
    assert 'copy.deepcopy' in intrinsics or 'deepcopy' in intrinsics
    assert 'random.Random' in intrinsics or 'Random' in intrinsics
    assert 'contextvars.ContextVar' in intrinsics or 'ContextVar' in intrinsics

    # 6. Assert real bytecode instructions (no empty bodies or placeholders)
    for fn in prog.functions:
        assert len(fn['instructions']) > 0, f"Function {fn['id']} has empty instructions"
        for inst in fn['instructions']:
            assert isinstance(inst['op'], str)
            assert inst['op'] != ""

    # 7. Assert ready=true is not set
    assert prog.data.get('ready') is not True

    # 8. Assert unresolved is empty and validate passes
    assert prog.unresolved == [], f"Unresolved references found: {prog.unresolved}"
    prog.validate()

    # 9. JSON serialization roundtrip produces identical identity
    json_str = prog.to_json()
    reconstructed = FrozenRuleProgram.from_json(json_str)
    assert reconstructed.identity == prog.identity
    reconstructed.validate()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name,fn in sorted(globals().items())
                              if name.startswith('test_') and callable(fn))

if __name__ == '__main__':
    unittest.main()
