"""Portable bytecode program exporter for duel v2 rules.

Exports formal Python rule functions, kits, and card effects into a frozen,
verifiable, fail-closed IR (format: duel_rule_bytecode_310_v1) for consumption
by native and CUDA device interpreters.
"""

from __future__ import annotations

import builtins
import contextvars
import copy
import dis
from enum import Enum
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Mapping, Sequence
import numpy as np

FORMAT_VERSION: str = "duel_rule_bytecode_310_v1"

# 13 characters forming the union across the five preset lineups
# (starter, weave-rush, quick-rush, zhenhong, murk)
FIVE_UNION: tuple[str, ...] = (
    'nanali',
    'iloy',
    'zero',
    'jiuyuan',
    'bohe',
    'baicang',
    'xiaozhi',
    'zhenhong',
    'yi',
    'anhunqu',
    'canhong',
    'zaowu',
    'adler',
)

# 107 cards: 8 standard cards for each of the 13 characters (104) + 3 derived cards (NF01, RF01, AF01)
ALL_8_AND_DERIVED: tuple[str, ...] = (
    # nanali (8 standard + NF01 derived)
    'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'NF01',
    # zero (8 standard)
    'Z01', 'Z02', 'Z03', 'Z04', 'Z05', 'Z06', 'Z07', 'Z08',
    # jiuyuan (8 standard)
    'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J07', 'J08',
    # iloy (8 standard)
    'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08',
    # bohe (8 standard)
    'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
    # baicang (8 standard)
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
    # xiaozhi (8 standard)
    'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08',
    # zhenhong (8 standard + RF01 derived)
    'R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'RF01',
    # yi (8 standard)
    'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08',
    # anhunqu (8 standard + AF01 derived)
    'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'AF01',
    # canhong (8 standard)
    'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08',
    # zaowu (8 standard)
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
    # adler (8 standard)
    'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08',
)

def _build_card_to_character_map() -> dict[str, str]:
    mapping: dict[str, str] = {
        'N01': 'nanali', 'N02': 'nanali', 'N03': 'nanali', 'N04': 'nanali',
        'N05': 'nanali', 'N06': 'nanali', 'N07': 'nanali', 'N08': 'nanali', 'NF01': 'nanali',
        'Z01': 'zero', 'Z02': 'zero', 'Z03': 'zero', 'Z04': 'zero',
        'Z05': 'zero', 'Z06': 'zero', 'Z07': 'zero', 'Z08': 'zero',
        'J01': 'jiuyuan', 'J02': 'jiuyuan', 'J03': 'jiuyuan', 'J04': 'jiuyuan',
        'J05': 'jiuyuan', 'J06': 'jiuyuan', 'J07': 'jiuyuan', 'J08': 'jiuyuan',
        'Y01': 'iloy', 'Y02': 'iloy', 'Y03': 'iloy', 'Y04': 'iloy',
        'Y05': 'iloy', 'Y06': 'iloy', 'Y07': 'iloy', 'Y08': 'iloy',
        'M01': 'bohe', 'M02': 'bohe', 'M03': 'bohe', 'M04': 'bohe',
        'M05': 'bohe', 'M06': 'bohe', 'M07': 'bohe', 'M08': 'bohe',
        'B01': 'baicang', 'B02': 'baicang', 'B03': 'baicang', 'B04': 'baicang',
        'B05': 'baicang', 'B06': 'baicang', 'B07': 'baicang', 'B08': 'baicang',
        'Q01': 'xiaozhi', 'Q02': 'xiaozhi', 'Q03': 'xiaozhi', 'Q04': 'xiaozhi',
        'Q05': 'xiaozhi', 'Q06': 'xiaozhi', 'Q07': 'xiaozhi', 'Q08': 'xiaozhi',
        'R01': 'zhenhong', 'R02': 'zhenhong', 'R03': 'zhenhong', 'R04': 'zhenhong',
        'R05': 'zhenhong', 'R06': 'zhenhong', 'R07': 'zhenhong', 'R08': 'zhenhong', 'RF01': 'zhenhong',
        'I01': 'yi', 'I02': 'yi', 'I03': 'yi', 'I04': 'yi',
        'I05': 'yi', 'I06': 'yi', 'I07': 'yi', 'I08': 'yi',
        'A01': 'anhunqu', 'A02': 'anhunqu', 'A03': 'anhunqu', 'A04': 'anhunqu',
        'A05': 'anhunqu', 'A06': 'anhunqu', 'A07': 'anhunqu', 'A08': 'anhunqu', 'AF01': 'anhunqu',
        'C01': 'canhong', 'C02': 'canhong', 'C03': 'canhong', 'C04': 'canhong',
        'C05': 'canhong', 'C06': 'canhong', 'C07': 'canhong', 'C08': 'canhong',
        'S01': 'zaowu', 'S02': 'zaowu', 'S03': 'zaowu', 'S04': 'zaowu',
        'S05': 'zaowu', 'S06': 'zaowu', 'S07': 'zaowu', 'S08': 'zaowu',
        'D01': 'adler', 'D02': 'adler', 'D03': 'adler', 'D04': 'adler',
        'D05': 'adler', 'D06': 'adler', 'D07': 'adler', 'D08': 'adler',
    }
    try:
        from app.modules.card_game.content.duel_v2.catalog import CARDS
        for cid, cdef in CARDS.items():
            ch = cdef.get('character_id')
            if ch:
                mapping[cid] = ch
    except Exception:
        pass
    return mapping


CARD_TO_CHARACTER: dict[str, str] = _build_card_to_character_map()

DEFAULT_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    'app.modules.card_game.engine.duel_v2',
    'app.modules.card_game.content.duel_v2',
    'app.modules.card_game.rl',
    'app.modules.card_game.engine.events',
)

FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    'open', 'eval', 'exec', '__import__', 'input', 'compile', 'breakpoint',
    'globals', 'locals', 'vars', 'exit', 'quit'
})

ALLOWED_BUILTIN_INTRINSICS: frozenset[str] = frozenset({
    'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr', 'classmethod',
    'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
    'oct', 'ord', 'pow', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type',
    'zip',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'StopIteration', 'RuntimeError', 'AssertionError',
    'ZeroDivisionError', 'OverflowError', 'LookupError', 'Warning',
})

DEFAULT_EXPLICIT_INTRINSICS: frozenset[str] = frozenset({
    'app.modules.card_game.engine.duel_v2.entities.clone_state',
    'copy.deepcopy',
    'random.Random',
    'contextvars.ContextVar',
    'enum.Enum',
    'uuid.uuid4',
    'json.dumps',
    'json.loads',
    'numpy.array_equal',
    'numpy.asarray',
    'numpy.zeros',
    'numpy.concatenate',
    'decimal.Decimal',
    'app.modules.card_game.engine.duel_v2.simulation._without_patches',
    'app.modules.card_game.content.duel_v2.registry._ensure_kits',
})

SAFE_STDLIB_MODULES: frozenset[str] = frozenset({
    'math', 'itertools', 'functools', 'collections', 'copy', 'random',
    'contextvars', 'hashlib', 'json', 'pathlib', 'sys', 'time', 'uuid',
    'dataclasses', 'types', 'typing', 'numpy', 'enum'
})


class ExportError(Exception):
    """Raised when rule program capture encounters fail-closed conditions."""
    pass


class ValidationError(Exception):
    """Raised when program IR validation fails."""
    pass


def find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / 'AGENTS.md').exists() or (p / '.git').exists():
            return p.resolve()
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[5]


def compute_canonical_json(data: dict[str, Any]) -> str:
    clean_data = {k: v for k, v in data.items() if k != 'identity'}
    return json.dumps(clean_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_identity(data: dict[str, Any]) -> str:
    payload = compute_canonical_json(data).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def serialize_argval(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else str(val)
    if isinstance(val, types.CodeType):
        return f"<code {val.co_name}>"
    if isinstance(val, (tuple, list)):
        return [serialize_argval(x) for x in val]
    if isinstance(val, dict):
        return {str(k): serialize_argval(v) for k, v in val.items()}
    return str(val)


class FrozenRuleProgram:
    """Frozen rule program bytecode and metadata descriptor container."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data: dict[str, Any] = data

    @property
    def format(self) -> str:
        return str(self.data.get('format', ''))

    @property
    def compiler_python(self) -> str:
        return str(self.data.get('compiler_python', ''))

    @property
    def identity(self) -> str:
        return str(self.data.get('identity', ''))

    @property
    def entries(self) -> dict[str, str]:
        return dict(self.data.get('entries', {}))

    @property
    def functions(self) -> list[dict[str, Any]]:
        return list(self.data.get('functions', []))

    @property
    def classes(self) -> list[dict[str, Any]]:
        return list(self.data.get('classes', []))

    @property
    def codes(self) -> list[dict[str, Any]]:
        return list(self.data.get('codes', []))

    @property
    def constants(self) -> list[dict[str, Any]]:
        return list(self.data.get('constants', []))

    @property
    def modules(self) -> dict[str, dict[str, dict[str, Any]]]:
        return dict(self.data.get('modules', {}))

    @property
    def source_hashes(self) -> dict[str, str]:
        return dict(self.data.get('source_hashes', {}))

    @property
    def unresolved(self) -> list[dict[str, Any]]:
        return list(self.data.get('unresolved', []))

    @property
    def required_intrinsics(self) -> list[str]:
        return list(self.data.get('required_intrinsics', []))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.data, indent=indent, sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenRuleProgram:
        return cls(copy.deepcopy(data))

    @classmethod
    def from_json(cls, json_str: str) -> FrozenRuleProgram:
        return cls(json.loads(json_str))

    def validate(self) -> None:
        if self.format != FORMAT_VERSION:
            raise ValidationError(
                f"Program format mismatch: expected {FORMAT_VERSION}, got {self.format}"
            )

        expected_identity = compute_identity(self.data)
        if self.identity != expected_identity:
            raise ValidationError(
                f"Identity mismatch: payload identity '{self.identity}' does not match computed '{expected_identity}'"
            )

        if self.data.get('ready') is True:
            raise ValidationError("Program specifies ready=true; device runtime is not yet implemented")

        if self.unresolved:
            raise ValidationError(
                f"Program contains {len(self.unresolved)} unresolved references/objects: {self.unresolved[:3]}"
            )

        forbidden_intrinsics = set(self.required_intrinsics) & FORBIDDEN_BUILTINS
        if forbidden_intrinsics:
            raise ValidationError(
                f"Program requires forbidden builtins in intrinsics: {forbidden_intrinsics}"
            )

        # Check for duplicate IDs
        func_ids: set[str] = set()
        for fn in self.functions:
            fid = fn.get('id')
            if not fid or fid in func_ids:
                raise ValidationError(f"Duplicate or empty function id: '{fid}'")
            func_ids.add(fid)

        class_ids: set[str] = set()
        for cls in self.classes:
            cid = cls.get('id')
            if not cid or cid in class_ids:
                raise ValidationError(f"Duplicate or empty class id: '{cid}'")
            class_ids.add(cid)

        code_ids: set[str] = set()
        for code in self.codes:
            coid = code.get('id')
            if not coid or coid in code_ids:
                raise ValidationError(f"Duplicate or empty code id: '{coid}'")
            code_ids.add(coid)

        n_constants = len(self.constants)
        const_ids: set[int] = set()
        valid_constant_kinds = frozenset({
            'literal', 'sequence', 'mapping', 'ndarray', 'enum', 'ContextVar', 'object', 'unresolved'
        })
        for const in self.constants:
            cid = const.get('id')
            if type(cid) is not int or isinstance(cid, bool) or cid < 0:
                raise ValidationError(f"Constant id {cid} is invalid (must be non-negative integer, not bool)")
            if cid in const_ids:
                raise ValidationError(f"Duplicate constant id: {cid}")
            const_ids.add(cid)
            kind = const.get('kind')
            if kind not in valid_constant_kinds:
                raise ValidationError(f"Unknown constant kind '{kind}' in constant {cid}")
            if kind == 'unresolved':
                raise ValidationError(f"Unresolved constant encountered: {const}")

        req_intrinsics_set = set(self.required_intrinsics)

        for entry_name, fid in self.entries.items():
            if fid not in func_ids:
                raise ValidationError(f"Entry '{entry_name}' references unknown function id: '{fid}'")

        def validate_ref(ref: Any, context: str) -> None:
            if not isinstance(ref, dict):
                raise ValidationError(f"Invalid reference in {context}: expected dict, got {type(ref)}")
            kind = ref.get('kind')
            if kind == 'function':
                fid = ref.get('id')
                if fid not in func_ids:
                    raise ValidationError(f"Function ref '{fid}' not found in functions ({context})")
            elif kind == 'code':
                coid = ref.get('id')
                if coid not in code_ids:
                    raise ValidationError(f"Code ref '{coid}' not found in codes ({context})")
            elif kind == 'class':
                cid = ref.get('id')
                if cid not in class_ids:
                    raise ValidationError(f"Class ref '{cid}' not found in classes ({context})")
            elif kind == 'constant':
                cid = ref.get('id')
                if type(cid) is not int or isinstance(cid, bool) or not (0 <= cid < n_constants):
                    raise ValidationError(f"Constant ref id {cid} invalid or out of range [0, {n_constants}) ({context})")
            elif kind == 'singleton':
                name = ref.get('name')
                if name not in ('None', 'True', 'False', 'Ellipsis', 'NotImplemented'):
                    raise ValidationError(f"Unknown singleton name '{name}' ({context})")
            elif kind == 'intrinsic':
                name = ref.get('name')
                if not name or name in FORBIDDEN_BUILTINS:
                    raise ValidationError(f"Invalid intrinsic '{name}' ({context})")
                if name not in req_intrinsics_set:
                    raise ValidationError(f"Intrinsic '{name}' referenced in {context} is not in required_intrinsics")
            elif kind == 'module':
                name = ref.get('name')
                if not name or not isinstance(name, str):
                    raise ValidationError(f"Module ref missing or invalid name ({context})")
                if name not in self.modules and name not in req_intrinsics_set:
                    raise ValidationError(f"Module ref '{name}' not found in modules or required_intrinsics ({context})")
            elif kind == 'unresolved':
                raise ValidationError(f"Unresolved reference encountered ({context}): {ref}")
            else:
                raise ValidationError(f"Unknown reference kind '{kind}' ({context})")

        def validate_instructions(instructions: list[dict[str, Any]], constants_list: list[Any], varnames_count: int, deref_count: int, ctx: str) -> None:
            n_inst = len(instructions)
            n_consts = len(constants_list)
            for idx, inst in enumerate(instructions):
                target = inst.get('target')
                if target is not None:
                    if type(target) is not int or isinstance(target, bool) or not (0 <= target < n_inst):
                        raise ValidationError(
                            f"Illegal jump target {target} at instruction {idx} in {ctx} (len={n_inst})"
                        )
                op = inst.get('op')
                arg = inst.get('arg')
                if arg is not None:
                    if type(arg) is not int or isinstance(arg, bool):
                        raise ValidationError(f"Instruction {idx} in {ctx} has non-integer arg: {arg}")
                    if op == 'LOAD_CONST':
                        if not (0 <= arg < n_consts):
                            raise ValidationError(
                                f"LOAD_CONST argument {arg} out of range [0, {n_consts}) at instruction {idx} in {ctx}"
                            )
                    elif op in ('LOAD_FAST', 'STORE_FAST', 'DELETE_FAST'):
                        if not (0 <= arg < varnames_count):
                            raise ValidationError(
                                f"{op} argument {arg} out of range [0, {varnames_count}) at instruction {idx} in {ctx}"
                            )
                    elif op in ('LOAD_DEREF', 'STORE_DEREF', 'DELETE_DEREF', 'LOAD_CLASSDEREF'):
                        if not (0 <= arg < deref_count):
                            raise ValidationError(
                                f"{op} argument {arg} out of range [0, {deref_count}) at instruction {idx} in {ctx}"
                            )

        for fn in self.functions:
            ctx = f"function {fn.get('id')}"
            instructions = fn.get('instructions', [])
            constants_list = fn.get('constants', [])
            n_locals = max(fn.get('nlocals', 0), len(fn.get('varnames', [])))
            n_deref = len(fn.get('cellvars', [])) + len(fn.get('freevars', []))
            validate_instructions(instructions, constants_list, n_locals, n_deref, ctx)

            for const_ref in constants_list:
                validate_ref(const_ref, f"{ctx} constants")
            for gname, gref in fn.get('globals', {}).items():
                validate_ref(gref, f"{ctx} global {gname}")
            for dref in fn.get('defaults', []):
                validate_ref(dref, f"{ctx} defaults")
            for kwname, kwref in fn.get('kwdefaults', {}).items():
                validate_ref(kwref, f"{ctx} kwdefaults {kwname}")
            for idx, cl_ref in enumerate(fn.get('closure', [])):
                validate_ref(cl_ref, f"{ctx} closure[{idx}]")

        for code in self.codes:
            ctx = f"code {code.get('id')}"
            instructions = code.get('instructions', [])
            constants_list = code.get('constants', [])
            n_locals = max(code.get('nlocals', 0), len(code.get('varnames', [])))
            n_deref = len(code.get('cellvars', [])) + len(code.get('freevars', []))
            validate_instructions(instructions, constants_list, n_locals, n_deref, ctx)

            for const_ref in constants_list:
                validate_ref(const_ref, f"{ctx} constants")

        for cls in self.classes:
            ctx = f"class {cls.get('id')}"
            for base_ref in cls.get('bases', []):
                validate_ref(base_ref, f"{ctx} bases")
            for mname, mref in cls.get('methods', {}).items():
                validate_ref(mref, f"{ctx} method {mname}")
            for cmname, cmref in cls.get('classmethods', {}).items():
                validate_ref(cmref, f"{ctx} classmethod {cmname}")
            for smname, smref in cls.get('staticmethods', {}).items():
                validate_ref(smref, f"{ctx} staticmethod {smname}")
            for pname, pref in cls.get('properties', {}).items():
                for accessor in ('fget', 'fset', 'fdel'):
                    if pref.get(accessor):
                        validate_ref(pref[accessor], f"{ctx} property {pname}.{accessor}")
            for aname, aref in cls.get('attributes', {}).items():
                validate_ref(aref, f"{ctx} attribute {aname}")

        for const in self.constants:
            ctx = f"constant {const.get('id')}"
            kind = const.get('kind')
            if kind == 'sequence':
                for item_ref in const.get('items', []):
                    validate_ref(item_ref, f"{ctx} sequence item")
            elif kind == 'mapping':
                for entry in const.get('entries', []):
                    validate_ref(entry['key'], f"{ctx} mapping key")
                    validate_ref(entry['value'], f"{ctx} mapping value")
            elif kind == 'enum':
                if const.get('class'):
                    validate_ref(const['class'], f"{ctx} enum class")
                validate_ref(const['value'], f"{ctx} enum value")
            elif kind == 'object':
                if const.get('class_id') not in class_ids:
                    raise ValidationError(f"Object constant class_id '{const.get('class_id')}' not found")
                for aname, aref in const.get('attributes', {}).items():
                    validate_ref(aref, f"{ctx} object attr {aname}")
            elif kind == 'ContextVar':
                if const.get('default') is not None:
                    validate_ref(const['default'], f"{ctx} contextvar default")

        for mod_name, sym_map in self.modules.items():
            for sym_name, sym_ref in sym_map.items():
                validate_ref(sym_ref, f"module {mod_name} symbol {sym_name}")


class ProgramExporter:
    """Internal recursive bytecode program collector and exporter."""

    def __init__(
        self,
        allowed_module_prefixes: Sequence[str] = DEFAULT_ALLOWED_MODULE_PREFIXES,
        explicit_intrinsics: Iterable[str] = DEFAULT_EXPLICIT_INTRINSICS,
        scope_character_ids: Iterable[str] | None = None,
        scope_card_ids: Iterable[str] | None = None,
    ) -> None:
        self.repo_root = find_repo_root()
        self.allowed_prefixes = tuple(allowed_module_prefixes)
        self.explicit_intrinsics = set(explicit_intrinsics)
        self.scope_character_ids = set(scope_character_ids) if scope_character_ids is not None else None
        self.scope_card_ids = set(scope_card_ids) if scope_card_ids is not None else None

        self.functions: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.codes: list[dict[str, Any]] = []
        self.constants: list[dict[str, Any]] = []
        self.modules: dict[str, dict[str, dict[str, Any]]] = {}
        self.source_hashes: dict[str, str] = {}
        self.unresolved: list[dict[str, Any]] = []
        self.required_intrinsics: set[str] = set()

        self._func_id_map: dict[int, str] = {}
        self._func_queue: list[types.FunctionType] = []
        self._processed_func_ids: set[str] = set()

        self._class_id_map: dict[type, str] = {}
        self._class_queue: list[type] = []
        self._processed_class_ids: set[str] = set()

        self._code_id_map: dict[int, str] = {}
        self._constant_memo: dict[int, dict[str, Any]] = {}
        self._constant_objects: dict[int, Any] = {}
        self._literal_memo: dict[tuple[type, Any], dict[str, Any]] = {}

    def check_source_validity(self, obj: Any) -> None:
        mod_name = getattr(obj, '__module__', None) or ''
        if not any(mod_name == p or mod_name.startswith(p + '.') for p in self.allowed_prefixes):
            raise ExportError(
                f"Module '{mod_name}' is not within allowed module prefixes {self.allowed_prefixes}"
            )

        src = None
        try:
            src = inspect.getsourcefile(obj)
        except Exception:
            pass
        if not src:
            mod = inspect.getmodule(obj)
            if mod and getattr(mod, '__file__', None):
                src = mod.__file__
        if not src and hasattr(obj, '__code__'):
            src = obj.__code__.co_filename

        if not src:
            raise ExportError(f"No source file found for {obj}")

        if os.path.islink(src) or Path(src).is_symlink():
            raise ExportError(f"External symlink rejected: {src}")

        src_path = Path(src).resolve()
        try:
            rel_path = src_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            raise ExportError(f"Source file {src} is outside repository root {self.repo_root}")

        if rel_path not in self.source_hashes:
            try:
                content = src_path.read_bytes()
                self.source_hashes[rel_path] = hashlib.sha256(content).hexdigest()
            except Exception as exc:
                raise ExportError(f"Cannot read source file {src_path}: {exc}") from exc

    def get_func_id(self, fn: types.FunctionType) -> str:
        existing=self._func_id_map.get(id(fn))
        if existing is not None:return existing
        mod = fn.__module__ or ''
        qual = fn.__qualname__ or fn.__name__
        return f"func:{mod}:{qual}:{fn.__code__.co_firstlineno}:{len(self._func_id_map)}"

    def queue_function(self, fn: Any) -> dict[str, Any] | None:
        if isinstance(fn, (staticmethod, classmethod)):
            fn = fn.__func__

        if isinstance(fn,np.ufunc):
            name=fn.__name__
            if getattr(np,name,None) is not fn:
                self.unresolved.append({'name':name,'reason':'Unregistered NumPy ufunc'})
                return {'kind':'unresolved','name':name,'reason':'Unregistered NumPy ufunc'}
            qual='numpy.'+name
            self.required_intrinsics.add(qual)
            return {'kind':'intrinsic','name':qual}

        qual = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', ''))}"
        name = getattr(fn, '__name__', '')
        if qual in self.explicit_intrinsics or name in self.explicit_intrinsics:
            int_name = qual if qual in self.explicit_intrinsics else name
            self.required_intrinsics.add(int_name)
            return {"kind": "intrinsic", "name": int_name}

        if hasattr(fn, '__wrapped__'):
            raise ExportError(f"Wrapped callable '{qual}' needs an explicit intrinsic; unwrapping changes semantics")
        if not isinstance(fn, (types.FunctionType, types.BuiltinFunctionType)):
            return None

        if getattr(fn, '__module__', '') == 'enum':
            self.required_intrinsics.add(qual)
            return {"kind": "intrinsic", "name": qual}

        if isinstance(fn, types.BuiltinFunctionType):
            mod = getattr(fn, '__module__', '')
            if mod == 'builtins' and name in ALLOWED_BUILTIN_INTRINSICS:
                self.required_intrinsics.add(name)
                return {"kind": "intrinsic", "name": name}
            elif mod in SAFE_STDLIB_MODULES:
                int_name = f"{mod}.{name}"
                self.required_intrinsics.add(int_name)
                return {"kind": "intrinsic", "name": int_name}
            else:
                self.unresolved.append({"name": name, "module": mod, "reason": "Disallowed builtin function"})
                return {"kind": "unresolved", "name": name, "reason": "Disallowed builtin function"}

        self.check_source_validity(fn)

        fid = self.get_func_id(fn)
        if id(fn) not in self._func_id_map:
            self._func_id_map[id(fn)] = fid
            self._func_queue.append(fn)
        return {"kind": "function", "id": fid}

    def get_class_id(self, cls: type) -> str:
        if cls in self._class_id_map:return self._class_id_map[cls]
        mod = cls.__module__ or ''
        qual = cls.__qualname__ or cls.__name__
        return f"class:{mod}:{qual}:{len(self._class_id_map)}"

    def queue_class(self, cls: type) -> dict[str, Any]:
        mod = getattr(cls, '__module__', '')
        name = getattr(cls, '__name__', '')
        qual = f"{mod}.{name}"
        if qual in self.explicit_intrinsics or name in self.explicit_intrinsics:
            int_name = qual if qual in self.explicit_intrinsics else name
            self.required_intrinsics.add(int_name)
            return {"kind": "intrinsic", "name": int_name}

        if mod == 'builtins' and name in ALLOWED_BUILTIN_INTRINSICS:
            self.required_intrinsics.add(name)
            return {"kind": "intrinsic", "name": name}
        elif mod in SAFE_STDLIB_MODULES:
            int_name = f"{mod}.{name}"
            self.required_intrinsics.add(int_name)
            return {"kind": "intrinsic", "name": int_name}

        self.check_source_validity(cls)
        cid = self.get_class_id(cls)
        if cls not in self._class_id_map:
            self._class_id_map[cls] = cid
            self._class_queue.append(cls)
        return {"kind": "class", "id": cid}

    def capture_constant_or_ref(self, val: Any) -> dict[str, Any]:
        if val is None:
            return {"kind": "singleton", "name": "None"}
        if val is True:
            return {"kind": "singleton", "name": "True"}
        if val is False:
            return {"kind": "singleton", "name": "False"}
        if val is Ellipsis:
            return {"kind": "singleton", "name": "Ellipsis"}
        if val is NotImplemented:
            return {"kind": "singleton", "name": "NotImplemented"}

        if callable(val) and not isinstance(val,type):
            res = self.queue_function(val)
            if res:
                return res

        if isinstance(val, type):
            return self.queue_class(val)

        if isinstance(val, types.ModuleType):
            self.required_intrinsics.add(val.__name__)
            return {"kind": "module", "name": val.__name__}

        # Deduplicate literals by value
        if type(val) in (int, str, bytes):
            lit_key = (type(val), val)
            if lit_key in self._literal_memo:
                return self._literal_memo[lit_key]
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            self._literal_memo[lit_key] = ref
            desc = {
                "id": cid,
                "kind": "literal",
                "type": type(val).__name__,
                "value": val if not isinstance(val, bytes) else list(val)
            }
            self.constants.append(desc)
            return ref

        if isinstance(val, float):
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            if math.isfinite(val):
                desc = {"id": cid, "kind": "literal", "type": "float", "value": val}
                self.constants.append(desc)
                return ref
            else:
                self.unresolved.append({"type": "float", "value": str(val), "reason": "Non-finite float"})
                desc = {"id": cid, "kind": "unresolved", "type": "float", "reason": "Non-finite float"}
                self.constants.append(desc)
                return ref

        # Memoize compound objects by id to handle cycles
        val_id = id(val)
        if val_id in self._constant_memo:
            return self._constant_memo[val_id]
        # Scoped registry dictionaries can be temporary. Retain memoized objects
        # so Python cannot recycle their addresses for a different constant.
        self._constant_objects[val_id]=val

        if isinstance(val, (tuple, list, set, frozenset)):
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            self._constant_memo[val_id] = ref
            desc = {
                "id": cid,
                "kind": "sequence",
                "type": type(val).__name__,
                "items": []
            }
            self.constants.append(desc)
            desc["items"] = [self.capture_constant_or_ref(x) for x in val]
            return ref

        if isinstance(val, dict):
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            self._constant_memo[val_id] = ref
            desc = {
                "id": cid,
                "kind": "mapping",
                "type": "dict",
                "entries": []
            }
            self.constants.append(desc)
            entries = []
            for k, v in val.items():
                k_ref = self.capture_constant_or_ref(k)
                v_ref = self.capture_constant_or_ref(v)
                entries.append({"key": k_ref, "value": v_ref})
            desc["entries"] = entries
            return ref

        if isinstance(val, np.ndarray):
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            self._constant_memo[val_id] = ref
            desc = {
                "id": cid,
                "kind": "ndarray",
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "data": val.tolist()
            }
            self.constants.append(desc)
            return ref

        if isinstance(val, Enum):
            cid=len(self.constants);ref={'kind':'constant','id':cid}
            self._constant_memo[val_id]=ref
            desc=dict(id=cid,kind='enum',name=val.name,str=str(val))
            self.constants.append(desc)
            desc['class']=self.queue_class(type(val))
            desc['value']=self.capture_constant_or_ref(val.value)
            return ref

        if isinstance(val, contextvars.ContextVar):
            self.required_intrinsics.add('contextvars.ContextVar')
            cid=len(self.constants);ref={'kind':'constant','id':cid}
            self._constant_memo[val_id]=ref
            desc=dict(id=cid,kind='ContextVar',name=val.name,default=None)
            self.constants.append(desc)
            try:desc['default']=self.capture_constant_or_ref(contextvars.Context().run(val.get))
            except LookupError:pass
            return ref

        if hasattr(val, '__dict__') and isinstance(type(val), type) and getattr(type(val), '__module__', None) not in ('builtins', None):
            cls = type(val)
            cls_ref = self.queue_class(cls)
            cid = len(self.constants)
            ref = {"kind": "constant", "id": cid}
            self._constant_memo[val_id] = ref
            desc={'id':cid,'kind':'object','class_id':cls_ref.get('id'),'attributes':{}}
            self.constants.append(desc)
            for attr_k,attr_v in val.__dict__.items():
                if not attr_k.startswith('__'):
                    desc['attributes'][attr_k]=self.capture_constant_or_ref(attr_v)
            return ref

        cid = len(self.constants)
        ref = {"kind": "constant", "id": cid}
        self._constant_memo[val_id] = ref
        self.unresolved.append({"type": type(val).__name__, "reason": f"Unknown object of type {type(val)}"})
        desc = {
            "id": cid,
            "kind": "unresolved",
            "type": type(val).__name__,
            "reason": f"Unknown object of type {type(val)}"
        }
        self.constants.append(desc)
        return ref

    def capture_code(self, co: types.CodeType, parent_qualname: str, fn_module: str, fn_globals: dict[str, Any]) -> str:
        code_key = id(co)
        if code_key in self._code_id_map:
            return self._code_id_map[code_key]

        code_id = f"code:{fn_module}:{parent_qualname}.{co.co_name}:{co.co_firstlineno}:{len(self._code_id_map)}"
        self._code_id_map[code_key] = code_id

        instructions = list(dis.get_instructions(co))
        offset_to_index = {inst.offset: idx for idx, inst in enumerate(instructions)}

        inst_list = []
        for inst in instructions:
            target_idx = None
            if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
                target_offset = inst.argval
                if isinstance(target_offset, int) and target_offset in offset_to_index:
                    target_idx = offset_to_index[target_offset]

            argval = serialize_argval(inst.argval)
            inst_list.append({
                'op': inst.opname,
                'arg': inst.arg,
                'argval': argval,
                'target': target_idx,
            })

        consts_refs = []
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                nested_id = self.capture_code(c, f"{parent_qualname}.{co.co_name}", fn_module, fn_globals)
                consts_refs.append({"kind": "code", "id": nested_id})
            else:
                consts_refs.append(self.capture_constant_or_ref(c))

        code_desc = {
            "id": code_id,
            "module": fn_module,
            "name": co.co_name,
            "qualname": f"{parent_qualname}.{co.co_name}",
            "argcount": co.co_argcount,
            "posonlyargcount": co.co_posonlyargcount,
            "kwonlyargcount": co.co_kwonlyargcount,
            "nlocals": co.co_nlocals,
            "stacksize": co.co_stacksize,
            "varnames": list(co.co_varnames),
            "freevars": list(co.co_freevars),
            "cellvars": list(co.co_cellvars),
            "flags": co.co_flags,
            "firstlineno": co.co_firstlineno,
            "instructions": inst_list,
            "constants": consts_refs,
        }
        self.codes.append(code_desc)
        return code_id

    def resolve_function_imports(self, fn: types.FunctionType, instructions: list[dis.Instruction]) -> None:
        """Resolve real Python relative imports; missing modules never become intrinsics."""
        import importlib
        package=fn.__globals__.get('__package__') or fn.__module__.rpartition('.')[0]
        bindings={name:value for name,value in fn.__globals__.items()
                  if isinstance(value,types.ModuleType)}

        def missing(name, reason):
            self.unresolved.append(dict(name=name,reason=reason))
            return {'kind':'unresolved','name':name,'reason':reason}

        def load_module(name):
            root=name.partition('.')[0]
            allowed=any(name==p or name.startswith(p+'.') for p in self.allowed_prefixes)
            if not allowed and root not in SAFE_STDLIB_MODULES:
                missing(name,'Import outside pure-module scope');return None
            try:
                return sys.modules.get(name) or importlib.import_module(name)
            except (ImportError,ValueError) as exc:
                missing(name,'Unresolved import: '+str(exc));return None

        def symbol(module,name):
            table=self.modules.setdefault(module.__name__,{})
            if not hasattr(module,name):
                table[name]=missing(module.__name__+'.'+name,'Missing imported symbol');return None
            value=getattr(module,name)
            table[name]=self.capture_constant_or_ref(value)
            return value

        def scan(code):
            insts=[i for i in dis.get_instructions(code) if i.opname!='EXTENDED_ARG']
            imported=None;pending=None
            for pos,inst in enumerate(insts):
                if inst.opname=='IMPORT_NAME':
                    if pos<2 or any(i.opname!='LOAD_CONST' for i in insts[pos-2:pos]):
                        missing(str(inst.argval),'Import level/fromlist not statically known');continue
                    level=insts[pos-2].argval
                    if type(level) is not int or level<0:
                        missing(str(inst.argval),'Invalid static import level');continue
                    name=importlib.util.resolve_name('.'*level+(inst.argval or ''),package) if level else inst.argval
                    imported=load_module(name);pending=imported
                    if imported is not None:self.modules.setdefault(imported.__name__,{})
                elif inst.opname=='IMPORT_FROM' and imported is not None:
                    pending=symbol(imported,inst.argval)
                elif inst.opname in ('STORE_FAST','STORE_NAME','STORE_DEREF'):
                    if isinstance(pending,types.ModuleType):bindings[inst.argval]=pending
                    else:bindings.pop(inst.argval,None)
                    pending=None
                elif inst.opname in ('LOAD_FAST','LOAD_GLOBAL','LOAD_NAME','LOAD_DEREF') and inst.argval in bindings:
                    value=bindings[inst.argval];index=pos+1
                    while index<len(insts) and insts[index].opname in ('LOAD_ATTR','LOAD_METHOD'):
                        if isinstance(value,types.ModuleType):value=symbol(value,insts[index].argval)
                        else:break
                        index+=1
            for const in code.co_consts:
                if isinstance(const,types.CodeType):scan(const)
        scan(fn.__code__)

    def process_function(self, fn: types.FunctionType) -> None:
        fid = self.get_func_id(fn)
        if fid in self._processed_func_ids:
            return
        self._processed_func_ids.add(fid)

        co = fn.__code__
        instructions = list(dis.get_instructions(co))
        offset_to_index = {inst.offset: idx for idx, inst in enumerate(instructions)}

        inst_list = []
        for inst in instructions:
            target_idx = None
            if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
                target_offset = inst.argval
                if isinstance(target_offset, int) and target_offset in offset_to_index:
                    target_idx = offset_to_index[target_offset]

            argval = serialize_argval(inst.argval)
            inst_list.append({
                'op': inst.opname,
                'arg': inst.arg,
                'argval': argval,
                'target': target_idx,
            })

        consts_refs = []
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                nested_id = self.capture_code(c, fn.__qualname__, fn.__module__, fn.__globals__)
                consts_refs.append({"kind": "code", "id": nested_id})
            else:
                consts_refs.append(self.capture_constant_or_ref(c))

        def collect_global_names(code_obj: types.CodeType) -> set[str]:
            names = set()
            for i in dis.get_instructions(code_obj):
                if i.opname == 'LOAD_GLOBAL':
                    names.add(i.argval)
                elif isinstance(i.argval, types.CodeType):
                    names.update(collect_global_names(i.argval))
            for c in code_obj.co_consts:
                if isinstance(c, types.CodeType):
                    names.update(collect_global_names(c))
            return names

        global_names = collect_global_names(co)
        globals_dict = {}

        try:
            from app.modules.card_game.content.duel_v2.registry import KITS
            from app.modules.card_game.content.duel_v2 import characters
        except ImportError:
            KITS = {}
            characters = None

        for gname in sorted(global_names):
            if gname in FORBIDDEN_BUILTINS:
                self.unresolved.append({"name": gname, "reason": "forbidden_io_or_dynamic_exec"})
                globals_dict[gname] = {"kind": "unresolved", "name": gname, "reason": "forbidden_io_or_dynamic_exec"}
            elif gname in fn.__globals__:
                val = fn.__globals__[gname]
                if val is KITS and self.scope_character_ids is not None:
                    scoped_kits = {cid: kit for cid, kit in KITS.items() if cid in self.scope_character_ids}
                    globals_dict[gname] = self.capture_constant_or_ref(scoped_kits)
                elif (gname == 'EFFECTS' or (characters and val is getattr(characters, 'EFFECTS', None))) and self.scope_card_ids is not None:
                    src_effects = getattr(characters, 'EFFECTS', {}) if characters else {}
                    scoped_eff = {cid: ef for cid, ef in src_effects.items() if cid in self.scope_card_ids}
                    globals_dict[gname] = self.capture_constant_or_ref(scoped_eff)
                else:
                    globals_dict[gname] = self.capture_constant_or_ref(val)
            elif hasattr(builtins, gname):
                if gname in ALLOWED_BUILTIN_INTRINSICS:
                    self.required_intrinsics.add(gname)
                    globals_dict[gname] = {"kind": "intrinsic", "name": gname}
                else:
                    self.unresolved.append({"name": gname, "reason": "Unapproved builtin"})
                    globals_dict[gname] = {"kind": "unresolved", "name": gname, "reason": "Unapproved builtin"}
            else:
                self.unresolved.append({"name": gname, "reason": "Symbol not found in globals or builtins"})
                globals_dict[gname] = {"kind": "unresolved", "name": gname, "reason": "Symbol not found"}

        defaults_refs = [self.capture_constant_or_ref(d) for d in fn.__defaults__] if fn.__defaults__ else []
        kwdefaults_refs = {k: self.capture_constant_or_ref(v) for k, v in fn.__kwdefaults__.items()} if fn.__kwdefaults__ else {}

        closure_refs = []
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    closure_refs.append(self.capture_constant_or_ref(cell.cell_contents))
                except ValueError:
                    closure_refs.append({"kind": "singleton", "name": "None"})

        self.resolve_function_imports(fn, instructions)

        func_desc = {
            "id": fid,
            "module": fn.__module__,
            "qualname": fn.__qualname__,
            "name": fn.__name__,
            "argcount": co.co_argcount,
            "posonlyargcount": co.co_posonlyargcount,
            "kwonlyargcount": co.co_kwonlyargcount,
            "nlocals": co.co_nlocals,
            "stacksize": co.co_stacksize,
            "varnames": list(co.co_varnames),
            "freevars": list(co.co_freevars),
            "cellvars": list(co.co_cellvars),
            "flags": co.co_flags,
            "firstlineno": co.co_firstlineno,
            "instructions": inst_list,
            "constants": consts_refs,
            "globals": globals_dict,
            "defaults": defaults_refs,
            "kwdefaults": kwdefaults_refs,
            "closure": closure_refs,
        }
        self.functions.append(func_desc)

    def process_class(self, cls: type) -> None:
        cid = self.get_class_id(cls)
        if cid in self._processed_class_ids:
            return
        self._processed_class_ids.add(cid)

        bases_refs = [self.queue_class(b) for b in cls.__bases__]

        methods = {}
        classmethods = {}
        staticmethods = {}
        properties = {}
        attributes = {}

        for name, member in cls.__dict__.items():
            if name.startswith('__') and name.endswith('__'):
                if name not in ('__init__', '__call__', '__getitem__', '__setitem__', '__len__', '__iter__', '__contains__', '__enter__', '__exit__', '__repr__', '__str__', '__eq__', '__hash__'):
                    continue
            if isinstance(member, staticmethod):
                ref = self.queue_function(member.__func__)
                if ref:
                    staticmethods[name] = ref
            elif isinstance(member, classmethod):
                ref = self.queue_function(member.__func__)
                if ref:
                    classmethods[name] = ref
            elif isinstance(member, property):
                fget_ref = self.queue_function(member.fget) if member.fget else None
                fset_ref = self.queue_function(member.fset) if member.fset else None
                fdel_ref = self.queue_function(member.fdel) if member.fdel else None
                properties[name] = {"fget": fget_ref, "fset": fset_ref, "fdel": fdel_ref}
            elif isinstance(member, (types.FunctionType, types.BuiltinFunctionType)):
                ref = self.queue_function(member)
                if ref:
                    methods[name] = ref
            else:
                attributes[name] = self.capture_constant_or_ref(member)

        class_desc = {
            "id": cid,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "name": cls.__name__,
            "bases": bases_refs,
            "methods": methods,
            "classmethods": classmethods,
            "staticmethods": staticmethods,
            "properties": properties,
            "attributes": attributes,
        }
        self.classes.append(class_desc)

    def run_capture(self, entries: Mapping[str, Any]) -> FrozenRuleProgram:
        if sys.version_info[:2] != (3, 10):
            raise ExportError(
                f"Host bytecode export must be run under CPython 3.10, got {sys.version}. "
                f"Windows 3.13 and subsequent runtimes only consume exported portable programs."
            )

        if self.scope_character_ids:
            from app.modules.card_game.content.duel_v2.registry import KITS, _ensure_kits
            _ensure_kits()
            for cid in sorted(self.scope_character_ids):
                kit = KITS.get(cid)
                if kit is None:
                    raise ExportError(f"Scope character '{cid}' has no registered kit in KITS")
                if isinstance(kit, type):
                    self.queue_class(kit)
                else:
                    self.capture_constant_or_ref(kit)

                effects = getattr(kit, 'effects', {})
                for card_id in sorted(self.scope_card_ids or ()):
                    owner = CARD_TO_CHARACTER.get(card_id)
                    if owner == cid:
                        if card_id not in effects:
                            raise ExportError(f"Missing in-scope callback for card '{card_id}' in kit '{cid}'")
                        eff_fn = effects[card_id]
                        self.queue_function(eff_fn)

        entry_refs = {}
        for entry_name, fn in entries.items():
            ref = self.queue_function(fn)
            if not ref or ref.get("kind") != "function":
                raise ExportError(f"Entry '{entry_name}' must resolve to a captured function, got {ref}")
            entry_refs[entry_name] = ref["id"]

        while self._func_queue or self._class_queue:
            while self._func_queue:
                fn = self._func_queue.pop(0)
                self.process_function(fn)
            while self._class_queue:
                cls = self._class_queue.pop(0)
                self.process_class(cls)

        catalog_path = self.repo_root / 'app/modules/card_game/content/duel_v2/catalog.json'
        if catalog_path.exists():
            rel_catalog = catalog_path.relative_to(self.repo_root).as_posix()
            self.source_hashes[rel_catalog] = hashlib.sha256(catalog_path.read_bytes()).hexdigest()

        program_data = {
            "format": FORMAT_VERSION,
            "compiler_python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "entries": dict(sorted(entry_refs.items())),
            "functions": sorted(self.functions, key=lambda f: f["id"]),
            "classes": sorted(self.classes, key=lambda c: c["id"]),
            "codes": sorted(self.codes, key=lambda c: c["id"]),
            "constants": self.constants,
            "modules": {k: dict(sorted(v.items())) for k, v in sorted(self.modules.items())},
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "unresolved": sorted(self.unresolved, key=lambda x: json.dumps(x, sort_keys=True)),
            "required_intrinsics": sorted(list(self.required_intrinsics)),
        }
        identity = compute_identity(program_data)
        program_data["identity"] = identity
        return FrozenRuleProgram(program_data)


def capture_program(
    entries: Mapping[str, Any],
    *,
    allowed_module_prefixes: Sequence[str] = DEFAULT_ALLOWED_MODULE_PREFIXES,
    scope_character_ids: Iterable[str] | None = FIVE_UNION,
    scope_card_ids: Iterable[str] | None = ALL_8_AND_DERIVED,
    explicit_intrinsics: Iterable[str] = DEFAULT_EXPLICIT_INTRINSICS,
) -> FrozenRuleProgram:
    exporter = ProgramExporter(
        allowed_module_prefixes=allowed_module_prefixes,
        explicit_intrinsics=explicit_intrinsics,
        scope_character_ids=scope_character_ids,
        scope_card_ids=scope_card_ids,
    )
    return exporter.run_capture(entries)


def capture_duel_program(
    entries: Mapping[str, Any] | None = None,
    *,
    allowed_module_prefixes: Sequence[str] = DEFAULT_ALLOWED_MODULE_PREFIXES,
    scope_character_ids: Iterable[str] = FIVE_UNION,
    scope_card_ids: Iterable[str] = ALL_8_AND_DERIVED,
    explicit_intrinsics: Iterable[str] = DEFAULT_EXPLICIT_INTRINSICS,
) -> FrozenRuleProgram:
    from app.modules.card_game.engine.duel_v2 import flow, projection
    from app.modules.card_game.rl import cross_lineup, cross_runtime, recovery_sampler

    duel_entries: dict[str, Any] = {
        'apply_action': flow.apply_action,
        'step': flow.apply_action,
        'legal_actions': flow.legal_actions,
        'observe': projection.observe,
        'decision': cross_runtime.decision,
        'encode': cross_lineup.encode,
        'determinize': recovery_sampler.determinize,
    }
    if entries:
        duel_entries.update(entries)

    return capture_program(
        duel_entries,
        allowed_module_prefixes=allowed_module_prefixes,
        scope_character_ids=scope_character_ids,
        scope_card_ids=scope_card_ids,
        explicit_intrinsics=explicit_intrinsics,
    )
