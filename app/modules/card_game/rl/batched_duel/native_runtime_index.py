"""Index frozen class lookup only in the shared runtime compilation unit.

Uses the existing exact C3/descriptor index emitter. Live instance attributes,
getters, method binding, error/budget behavior and all rule objects are retained.
The rule/value compilation units and their cached binaries are untouched.
"""
from dataclasses import replace
import re

from .native_array_codegen import _matching_brace_end
from .vm_objects import emit_object_source


def _definition(source, name):
    pattern = re.compile(r'(?m)^__device__ inline int '+name+r'\([^;{}]*\)\s*\{')
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError('Unique frozen attribute definition required: '+name)
    match = matches[0]
    return source[match.start():_matching_brace_end(source, match.end()-1)]


def index_runtime_attributes(unit, image):
    if unit.name != 'runtime':
        raise ValueError('Attribute indexing requires the shared runtime unit')
    source = unit.source
    if 'frozen_attr_lookup' in source:
        raise ValueError('Frozen attributes already indexed')
    original = emit_object_source(image)
    indexed = emit_object_source(image, frozen_lookup=True)
    anchor = '__device__ inline int class_constructible('
    if original.count(anchor) != 1 or source.count(anchor) != 1:
        raise ValueError('Frozen object runtime boundary changed')
    prefix = indexed[:indexed.index(anchor)]
    replacements = {'lookup_own': 'native_indexed_lookup_own',
                    'lookup_class': 'native_indexed_lookup_class'}
    for name in replacements:
        before = _definition(original, name)
        if source.count(before) != 1:
            raise ValueError('Canonical frozen attribute body changed: '+name)
        # The legacy inline definitions remain byte-identical across units.
        # New private names avoid ODR conflicts with cached rule/value objects.
        body = _definition(indexed, name)
        for old, new in replacements.items():
            body = re.sub(r'\b'+old+r'\b', new, body)
        prefix += body+'\n'
    changed = set()
    # Only these shared out-of-line owners can use the private index. Never
    # rewrite a common-header inline body in just one compilation unit.
    owners = ('object_attribute', 'call', 'contains', 'stdlib_call',
              'native_builtin_call', 'new_frame')
    for owner in owners:
        pattern = re.compile(r'(?m)^__device__ __noinline__ (?:int|void) '+owner+r'\([^;{}]*\)\s*\{')
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError('Unique shared attribute owner required: '+owner)
        match = matches[0]
        end = _matching_brace_end(source, match.end()-1)
        body = source[match.start():end]
        rewritten = body
        for old, new in replacements.items():
            rewritten = re.sub(r'\b'+old+r'(?=\s*\()', new, rewritten)
        if rewritten != body:
            changed.add(owner)
            source = source[:match.start()]+rewritten+source[end:]
    if not {'object_attribute', 'call'} <= changed:
        raise ValueError('Missing shared method/constructor lookup sites')
    position = source.index(anchor)
    source = source[:position]+prefix+source[position:]
    return replace(unit,source=source)
