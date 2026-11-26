"""Typed, alias-preserving domain record pools in field-major SoA storage.

Record fields have dedicated numeric columns. Composite values reference a
declared record/list/map pool; there is no node/edge/payload graph or serialized
state blob. The schema is closed, including each union. Host encode/decode is
an explicit validation/ingress boundary, never a rule execution fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from hashlib import sha256
import json
import math
from types import MappingProxyType

import numpy as np

ABSENT, NULL, BOOL, INT, FLOAT, TEXT, REF = range(7)
TAG_LIST, TAG_TUPLE, TAG_DICT, TAG_PLAYER, TAG_CHARACTER, TAG_CARD = 5, 6, 7, 8, 9, 10
MISSING = -2147483648
_SCALARS = {'bool': (BOOL,), 'int': (INT,), 'float': (FLOAT,),
            'str': (TEXT,), 'number': (INT, FLOAT), 'scalar': (BOOL, INT, FLOAT, TEXT)}
_DTYPES = {'i64': np.int64, 'f64': np.float64, 'i32': np.int32, 'u8': np.uint8}


class RecordSoAError(ValueError):
    pass


class RecordSoACapacityError(RecordSoAError):
    pass


class RecordSoATypeError(RecordSoAError, TypeError):
    pass


def _positive(value, name, maximum=2147483647):
    if type(value) is not int or not 1 <= value <= maximum:
        raise RecordSoAError(f'{name}: bounded positive integer required')
    return value


@dataclass(frozen=True)
class ValueColumn:
    name: str
    scalar_mask: int
    references: tuple
    tag: int
    integer: int
    real: int
    word: int
    stride: int
    extent: int


@dataclass(frozen=True)
class RecordPool:
    index: int
    name: str
    kind: str
    capacity: int
    base: int
    tags: tuple
    used: int
    tag: int
    count: int
    order: int
    max_items: int
    fields: tuple
    keys: ValueColumn | None
    values: tuple

    def global_ref(self, slot):
        if not 0 <= slot < self.capacity:
            raise RecordSoAError('Record slot outside pool')
        return self.base + slot + 1


class CompiledRecordSchema:
    """Frozen column offsets shared by host codec and native view codegen.

    Type expressions: scalar names; ``@Pool``; or ``{'one_of': [...]}``.
    Composite definitions have kind record/list/map/tuple. Every field also
    distinguishes ABSENT and None. A list definition can preserve list/tuple
    tags; a map has explicitly typed keys and values.
    """
    def __init__(self, definitions, *, root='Game', capacities=None,
                 max_strings=4096, max_string_bytes=1048576):
        if not isinstance(definitions, Mapping) or not definitions:
            raise RecordSoAError('Nonempty closed domain schema required')
        # JSON here freezes schema metadata only; no game values are serialized.
        definitions = json.loads(json.dumps(dict(definitions), ensure_ascii=False, allow_nan=False))
        capacities = dict(capacities or {})
        allowed_capacities = set(definitions) | {
            name+'.items' for name, spec in definitions.items()
            if spec.get('kind') in ('list', 'map')}
        if set(capacities) - allowed_capacities:
            raise RecordSoAError('Unknown capacity overrides: '+repr(sorted(set(capacities)-allowed_capacities)))
        if any(type(name) is not str or not name.isidentifier() for name in definitions):
            raise RecordSoAError('Pool names must be identifiers')
        self.definitions = MappingProxyType(definitions)
        self.names = tuple(definitions)
        self.type_ids = MappingProxyType({name: i for i, name in enumerate(self.names)})
        if root not in self.type_ids:
            raise RecordSoAError('Root record type missing')
        self.root = self.type_ids[root]
        self.max_strings = _positive(max_strings, 'max_strings')
        self.max_string_bytes = _positive(max_string_bytes, 'max_string_bytes')
        self.lengths = {name: 0 for name in _DTYPES}

        def allocate(dtype, count):
            start = self.lengths[dtype]
            if type(count) is not int or count < 0 or start + count > 2147483647:
                raise RecordSoACapacityError('Column address exceeds int32 domain')
            self.lengths[dtype] += count
            return start

        def value_column(name, expression, capacity, stride=1):
            scalar, references = self.value_types(expression)
            extent = capacity * stride
            return ValueColumn(name, scalar, references,
                allocate('u8', extent),
                allocate('i64', extent) if scalar & ((1 << BOOL) | (1 << INT)) else -1,
                allocate('f64', extent) if scalar & (1 << FLOAT) else -1,
                allocate('i32', extent) if scalar & (1 << TEXT) or references else -1,
                stride, extent)

        pools, base = [], 0
        for index, (name, definition) in enumerate(definitions.items()):
            kind = definition.get('kind')
            if kind not in ('record', 'list', 'map', 'tuple'):
                raise RecordSoAError(f'{name}: unsupported record kind {kind}')
            capacity = _positive(capacities.get(name, definition.get('capacity', 128)), name)
            tags = tuple(definition.get('tags') or {
                'record': (TAG_DICT,), 'map': (TAG_DICT,),
                'list': (TAG_LIST,), 'tuple': (TAG_TUPLE,)}[kind])
            allowed_tags = (TAG_DICT, TAG_PLAYER, TAG_CHARACTER, TAG_CARD) if kind in ('record', 'map') else (TAG_LIST, TAG_TUPLE)
            if not tags or any(type(t) is not int or t not in allowed_tags for t in tags):
                raise RecordSoAError(f'{name}: invalid object tags')
            used, tag, count = allocate('i32', 1), allocate('u8', capacity), allocate('i32', capacity)
            fields, keys, values, order, max_items = (), None, (), -1, 0
            if kind == 'record':
                raw_fields = definition.get('fields')
                if not isinstance(raw_fields, dict) or any(type(k) is not str for k in raw_fields):
                    raise RecordSoAError(f'{name}: named fields required')
                fields = tuple(value_column(f'{name}.{key}', expr, capacity) for key, expr in raw_fields.items())
                max_items = len(fields)
                order = allocate('i32', capacity * max_items)
            elif kind == 'tuple':
                items = definition.get('items')
                if not isinstance(items, list):
                    raise RecordSoAError(f'{name}: tuple item types required')
                values = tuple(value_column(f'{name}[{i}]', expr, capacity) for i, expr in enumerate(items))
                max_items = len(items)
            else:
                max_items = _positive(capacities.get(name+'.items', definition.get('max_items')), name+'.items')
                if kind == 'map':
                    keys = value_column(f'{name}.keys', definition.get('key', 'str'), capacity, max_items)
                    if keys.references:
                        raise RecordSoAError('Domain map keys must be declared scalars')
                values = (value_column(f'{name}.items', definition.get('value'), capacity, max_items),)
            pools.append(RecordPool(index, name, kind, capacity, base, tags, used,
                                    tag, count, order, max_items, fields, keys, values))
            base += capacity
            if base >= 2147483647:
                raise RecordSoACapacityError('Object handle exceeds int32 domain')
        self.pools = tuple(pools)
        self.total_objects = base
        self.strings_count = allocate('i32', 1)
        self.bytes_used = allocate('i32', 1)
        self.string_offsets = allocate('i32', self.max_strings)
        self.string_lengths = allocate('i32', self.max_strings)
        self.string_static_refs = allocate('i32', self.max_strings)
        self.string_hashes = allocate('i64', self.max_strings)
        self.string_bytes = allocate('u8', self.max_string_bytes)
        # A declared-but-unused channel retains one inert cell so both native
        # ABIs have non-null typed pointers even for small conformance schemas.
        self.lengths = MappingProxyType({key: max(1, value) for key, value in self.lengths.items()})
        self.identity = sha256(json.dumps(dict(abi='typed_record_soa_v1', definitions=definitions,
            capacities=capacities, root=root, max_strings=max_strings,
            max_string_bytes=max_string_bytes), sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

    def value_types(self, expression):
        if isinstance(expression, str):
            if expression in _SCALARS:
                return sum(1 << k for k in _SCALARS[expression]), ()
            if expression.startswith('@') and expression[1:] in self.type_ids:
                return 0, (self.type_ids[expression[1:]],)
        elif isinstance(expression, dict) and set(expression) == {'one_of'}:
            choices = expression['one_of']
            if not isinstance(choices, (list, tuple)) or not choices:
                raise RecordSoAError('Nonempty declared union required')
            scalar, references = 0, []
            for choice in choices:
                mask, refs = self.value_types(choice)
                scalar |= mask
                references.extend(refs)
            return scalar, tuple(dict.fromkeys(references))
        raise RecordSoAError(f'Unknown/opaque type expression {expression!r}')

    def locate(self, reference):
        if type(reference) is not int or reference < 1:
            raise RecordSoAError('Positive typed object reference required')
        for pool in self.pools:
            if pool.base < reference <= pool.base + pool.capacity:
                return pool, reference - pool.base - 1
        raise RecordSoAError('Object reference outside schema pools')

    def allocate(self, rows):
        return RecordSoABatch(self, rows)


class RecordSoABatch:
    """Four homogeneous field-major buffers, with named typed column views.

    Buffer indexing is [column,row], so the same field across GPU roots is
    contiguous. ``column`` exposes [B,...] strided views of authoritative data.
    """
    def __init__(self, schema, rows, *, buffers=None, roots=None):
        if not isinstance(schema, CompiledRecordSchema) or type(rows) is not int or rows < 0:
            raise RecordSoAError('Compiled schema and nonnegative row count required')
        self.schema, self.rows = schema, rows
        self.buffers = {} if buffers is None else dict(buffers)
        for key, dtype in _DTYPES.items():
            shape = (schema.lengths[key], rows)
            if buffers is None:
                self.buffers[key] = np.zeros(shape, dtype=dtype)
            array = self.buffers.get(key)
            if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
                raise RecordSoAError(f'Exact contiguous {key} buffer required')
        if buffers is None:
            self.buffers['i32'][schema.string_static_refs:schema.string_static_refs+schema.max_strings] = MISSING
        self.roots = np.zeros(rows, dtype=np.int32) if roots is None else roots
        if self.roots.dtype != np.int32 or self.roots.shape != (rows,) or not self.roots.flags.c_contiguous:
            raise RecordSoAError('Exact int32 root references required')

    @property
    def allocated_bytes(self):
        return sum(x.nbytes for x in self.buffers.values()) + self.roots.nbytes

    def column(self, pool_name, field_name, channel='integer'):
        pool = self.schema.pools[self.schema.type_ids[pool_name]]
        field = next((f for f in pool.fields if f.name == pool_name+'.'+field_name), None)
        if field is None:
            raise KeyError((pool_name, field_name))
        dtype, attribute = {'integer': ('i64', 'integer'), 'real': ('f64', 'real'),
                            'word': ('i32', 'word'), 'tag': ('u8', 'tag')}[channel]
        offset = getattr(field, attribute)
        if offset < 0:
            raise RecordSoATypeError('This field has no requested value channel')
        values = self.buffers[dtype][offset:offset+field.extent]
        return np.moveaxis(values.reshape(pool.capacity, field.stride, self.rows), -1, 0).reshape(self.rows, pool.capacity)

    def clone_rows(self, indices):
        values = np.asarray(indices, dtype=np.intp) if type(indices) in (list, tuple) and not indices else np.asarray(indices)
        if values.ndim != 1 or values.dtype.kind not in ('i', 'u'):
            raise RecordSoATypeError('Exact integer row indices required')
        if np.any(values < 0) or np.any(values >= self.rows):
            raise RecordSoAError('Row index outside batch')
        values = values.astype(np.intp)
        return RecordSoABatch(self.schema, len(values),
            buffers={k: np.ascontiguousarray(v[:, values]) for k, v in self.buffers.items()},
            roots=np.ascontiguousarray(self.roots[values]))

    def to_torch(self, device='cpu'):
        import torch
        return TorchRecordSoABatch(self.schema,
            {k: torch.tensor(v, device=device) for k, v in self.buffers.items()},
            torch.tensor(self.roots, device=device))


@dataclass(frozen=True)
class TorchRecordSoABatch:
    schema: CompiledRecordSchema
    buffers: dict
    roots: object

    @property
    def rows(self):
        return self.roots.shape[0]

    def to_host(self):
        return RecordSoABatch(self.schema, self.rows,
            buffers={k: v.detach().cpu().numpy().copy() for k, v in self.buffers.items()},
            roots=self.roots.detach().cpu().numpy().copy())


def compile_schema(definitions, **options):
    return CompiledRecordSchema(definitions, **options)


def object_tag(value):
    from app.modules.card_game.engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
    return {dict: TAG_DICT, list: TAG_LIST, tuple: TAG_TUPLE, PlayerEntity: TAG_PLAYER,
            CharacterEntity: TAG_CHARACTER, CardEntity: TAG_CARD}.get(type(value), -1)


def string_hash(text):
    value = 14695981039346656037
    for byte in text.encode('utf-8'):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64)-1)
    return value if value < (1 << 63) else value - (1 << 64)


class _EncodeRow:
    def __init__(self, batch, row):
        self.batch, self.schema, self.row = batch, batch.schema, row
        self.buffers = batch.buffers
        self.memo, self.active, self.strings = {}, set(), {}

    def text(self, value):
        old = self.strings.get(value)
        if old is not None:
            return old
        s, b, row = self.schema, self.buffers, self.row
        count, used = int(b['i32'][s.strings_count, row]), int(b['i32'][s.bytes_used, row])
        raw = value.encode('utf-8')
        if count >= s.max_strings or used + len(raw) > s.max_string_bytes:
            raise RecordSoACapacityError('String columns exhausted')
        b['i32'][s.string_offsets+count, row] = used
        b['i32'][s.string_lengths+count, row] = len(raw)
        b['i64'][s.string_hashes+count, row] = string_hash(value)
        b['u8'][s.string_bytes+used:s.string_bytes+used+len(raw), row] = np.frombuffer(raw, dtype=np.uint8)
        b['i32'][s.strings_count, row] = count + 1
        b['i32'][s.bytes_used, row] = used + len(raw)
        self.strings[value] = count + 1
        return count + 1

    def value(self, column, index, value, path):
        b, row = self.buffers, self.row
        if value is None:
            b['u8'][column.tag+index, row] = NULL
            return
        kind = {bool: BOOL, int: INT, float: FLOAT, str: TEXT}.get(type(value))
        if kind is not None:
            if not column.scalar_mask & (1 << kind):
                raise RecordSoATypeError(f'{path}: unexpected scalar {type(value).__name__}')
            if kind in (INT, BOOL):
                if kind == INT and not -(1 << 63) <= value < (1 << 63):
                    raise RecordSoATypeError(f'{path}: integer exceeds exact int64 domain')
                b['i64'][column.integer+index, row] = value
            elif kind == FLOAT:
                if not math.isfinite(value):
                    raise RecordSoATypeError(f'{path}: non-finite value')
                b['f64'][column.real+index, row] = value
            else:
                b['i32'][column.word+index, row] = self.text(value)
        else:
            if not column.references:
                raise RecordSoATypeError(f'{path}: container has no declared domain type')
            reference = self.reference(value, column.references, path)
            b['i32'][column.word+index, row] = reference
            kind = REF
        b['u8'][column.tag+index, row] = kind

    def reference(self, value, candidates, path):
        tag = object_tag(value)
        if tag < 0:
            raise RecordSoATypeError(f'{path}: unsupported object type {type(value).__name__}')
        previous = self.memo.get(id(value))
        if previous is not None:
            if id(value) in self.active:
                raise RecordSoATypeError(f'{path}: circular game state is not JSON serializable')
            pool, slot = self.schema.locate(previous)
            if pool.index not in candidates:
                raise RecordSoATypeError(f'{path}: aliased object has incompatible declared pool {pool.name}')
            return previous
        possible = []
        for tid in candidates:
            pool = self.schema.pools[tid]
            if tag not in pool.tags:
                continue
            if pool.kind == 'record':
                names = {f.name.split('.', 1)[1] for f in pool.fields}
                if not set(value).issubset(names):
                    continue
            elif pool.kind == 'tuple' and len(value) != pool.max_items:
                continue
            possible.append(pool)
        if not possible:
            names = [self.schema.pools[i].name for i in candidates]
            detail = list(value)[:30] if isinstance(value, dict) else type(value).__name__
            if isinstance(value, dict):
                declared = {f.name.split('.', 1)[1] for i in candidates for f in self.schema.pools[i].fields}
                detail = {'unknown_fields': sorted(set(value)-declared), 'class': type(value).__name__}
            raise RecordSoATypeError(f'{path}: no matching domain pool in {names}; fields={detail!r}')
        # Prefer the narrowest declared record when a union shares dict tags.
        pool = min(possible, key=lambda p: len(p.fields) if p.kind == 'record' else 0)
        b, row = self.buffers, self.row
        slot = int(b['i32'][pool.used, row])
        if slot >= pool.capacity:
            raise RecordSoACapacityError(f'{path}: {pool.name} record capacity exhausted')
        b['i32'][pool.used, row] = slot + 1
        reference = pool.global_ref(slot)
        self.memo[id(value)] = reference
        self.active.add(id(value))
        b['u8'][pool.tag+slot, row] = tag
        b['i32'][pool.count+slot, row] = len(value)
        try:
            if pool.kind == 'record':
                fields = {f.name.split('.', 1)[1]: (i, f) for i, f in enumerate(pool.fields)}
                for ordinal, (key, item) in enumerate(value.items()):
                    field_index, column = fields[key]
                    b['i32'][pool.order+slot*pool.max_items+ordinal, row] = field_index
                    self.value(column, slot, item, path+'.'+key)
            elif pool.kind == 'map':
                if len(value) > pool.max_items:
                    raise RecordSoACapacityError(f'{path}: map slot capacity exhausted')
                for ordinal, (key, item) in enumerate(value.items()):
                    index = slot*pool.max_items+ordinal
                    self.value(pool.keys, index, key, path+'.key')
                    self.value(pool.values[0], index, item, path+f'[{key!r}]')
            else:
                if len(value) > pool.max_items:
                    raise RecordSoACapacityError(f'{path}: sequence slot capacity exhausted')
                for ordinal, item in enumerate(value):
                    column = pool.values[ordinal] if pool.kind == 'tuple' else pool.values[0]
                    index = slot if pool.kind == 'tuple' else slot*pool.max_items+ordinal
                    self.value(column, index, item, path+f'[{ordinal}]')
        finally:
            self.active.remove(id(value))
        return reference


class _DecodeRow:
    def __init__(self, batch, row):
        self.batch, self.schema, self.row = batch, batch.schema, row
        self.buffers, self.memo, self.active = batch.buffers, {}, set()

    def text(self, reference):
        s, b, row = self.schema, self.buffers, self.row
        count, used = int(b['i32'][s.strings_count, row]), int(b['i32'][s.bytes_used, row])
        if not 0 <= count <= s.max_strings or not 0 <= used <= s.max_string_bytes or not 1 <= reference <= count:
            raise RecordSoAError('String reference outside checked columns')
        start = int(b['i32'][s.string_offsets+reference-1, row])
        size = int(b['i32'][s.string_lengths+reference-1, row])
        if start < 0 or size < 0 or start+size > used:
            raise RecordSoAError('Invalid string column extent')
        text = b['u8'][s.string_bytes+start:s.string_bytes+start+size, row].tobytes().decode('utf-8')
        if string_hash(text) != int(b['i64'][s.string_hashes+reference-1, row]):
            raise RecordSoAError('String column hash mismatch')
        return text

    def value(self, column, index):
        b, row = self.buffers, self.row
        kind = int(b['u8'][column.tag+index, row])
        if kind == ABSENT:
            raise RecordSoAError('Absent value read as an active field/element')
        if kind == NULL:
            return None
        if kind in (BOOL, INT, FLOAT, TEXT):
            if not column.scalar_mask & (1 << kind):
                raise RecordSoATypeError('Scalar channel incompatible with domain schema')
            if kind in (BOOL, INT):
                value = int(b['i64'][column.integer+index, row])
                if kind == BOOL and value not in (0, 1):
                    raise RecordSoAError('Invalid boolean column')
                return bool(value) if kind == BOOL else value
            if kind == FLOAT:
                value = float(b['f64'][column.real+index, row])
                if not math.isfinite(value):
                    raise RecordSoAError('Non-finite float column')
                return value
            return self.text(int(b['i32'][column.word+index, row]))
        if kind == REF:
            reference = int(b['i32'][column.word+index, row])
            pool, _ = self.schema.locate(reference)
            if pool.index not in column.references:
                raise RecordSoATypeError('Reference channel incompatible with domain schema')
            return self.reference(reference)
        raise RecordSoAError('Unknown value channel')

    def reference(self, reference):
        if reference in self.active:
            raise RecordSoATypeError('Circular state columns are not JSON serializable')
        if reference in self.memo:
            return self.memo[reference]
        pool, slot = self.schema.locate(reference)
        b, row = self.buffers, self.row
        used = int(b['i32'][pool.used, row])
        tag, count = int(b['u8'][pool.tag+slot, row]), int(b['i32'][pool.count+slot, row])
        if not 0 <= used <= pool.capacity or slot >= used or tag not in pool.tags or not 0 <= count <= pool.max_items:
            raise RecordSoAError(f'{pool.name}: invalid record occupancy/tag/count')
        if pool.kind == 'tuple' and count != pool.max_items:
            raise RecordSoAError('Fixed tuple length changed')
        self.active.add(reference)
        try:
            if tag in (TAG_DICT, TAG_PLAYER, TAG_CHARACTER, TAG_CARD):
                from app.modules.card_game.engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
                cls = {TAG_DICT: dict, TAG_PLAYER: PlayerEntity, TAG_CHARACTER: CharacterEntity, TAG_CARD: CardEntity}[tag]
                value = dict.__new__(cls)
                self.memo[reference] = value
                if pool.kind == 'record':
                    visited = set()
                    for ordinal in range(count):
                        field_id = int(b['i32'][pool.order+slot*pool.max_items+ordinal, row])
                        if not 0 <= field_id < len(pool.fields) or field_id in visited:
                            raise RecordSoAError('Invalid field insertion order')
                        visited.add(field_id)
                        column = pool.fields[field_id]
                        value[column.name.split('.', 1)[1]] = self.value(column, slot)
                    present = {i for i, f in enumerate(pool.fields) if b['u8'][f.tag+slot, row] != ABSENT}
                    if visited != present:
                        raise RecordSoAError('Field presence disagrees with record order')
                else:
                    for ordinal in range(count):
                        index = slot*pool.max_items+ordinal
                        key = self.value(pool.keys, index)
                        if key in value:
                            raise RecordSoAError('Duplicate map key in columns')
                        value[key] = self.value(pool.values[0], index)
            else:
                value = []
                if tag == TAG_LIST:
                    self.memo[reference] = value
                for ordinal in range(count):
                    column = pool.values[ordinal] if pool.kind == 'tuple' else pool.values[0]
                    index = slot if pool.kind == 'tuple' else slot*pool.max_items+ordinal
                    value.append(self.value(column, index))
                if tag == TAG_TUPLE:
                    value = tuple(value)
                self.memo[reference] = value
        finally:
            self.active.remove(reference)
        return value


def pack_game_states(states, schema=None, **schema_options):
    """One explicit host ingress of complete declared states, preserving aliases."""
    if schema is None:
        from .domain_soa_records import domain_types
        schema = compile_schema(domain_types(), **schema_options)
    if type(states) not in (list, tuple):
        raise RecordSoATypeError('Explicit sequence of states required')
    result = RecordSoABatch(schema, len(states))
    for row, state in enumerate(states):
        result.roots[row] = _EncodeRow(result, row).reference(state, (schema.root,), '$')
    return result


def unpack_game_states(batch):
    """Explicit host egress; never called by the native/device rule hot path."""
    if not isinstance(batch, RecordSoABatch):
        raise RecordSoATypeError('Typed domain record batch required')
    return [_DecodeRow(batch, row).reference(int(reference)) for row, reference in enumerate(batch.roots)]
