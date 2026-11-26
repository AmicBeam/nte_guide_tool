"""Resolve frozen container query names in the shared runtime only.

Effects and kit dispatch repeatedly use mapping/list methods. Their frozen
attribute names need no UTF-8 payload scans. Dynamic names retain the canonical
lookup, and bound-method allocation and invocation remain unchanged. No mutable
effect, character, weapon or hook result is cached.
"""
from dataclasses import replace
import re

from .native_array_codegen import _matching_brace_end
from .vm_image import VmImage


_METHODS = {
    'dict': ('get', 'keys', 'values', 'items', 'setdefault', 'pop', 'update', 'copy'),
    'list': ('append', 'extend', 'pop', 'insert', 'remove', 'reverse', 'copy', 'index', 'count'),
    'tuple': ('index', 'count'),
}


def specialize_effect_queries(unit, image):
    """Return a runtime-only specialization; cached rule units stay identical.

    A verified immutable image identifies names by negative reference. Owner
    validation, method metadata creation, frame/step accounting, and all live
    reads/mutations follow the original code. Fail closed if its definition or
    recognized method inventory changes.
    """
    if unit.name != 'runtime':
        raise ValueError('Effect query specialization requires the shared runtime unit')
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    image.verify()
    if 'native_effect_query_names' in unit.source:
        raise ValueError('Effect query names already specialized')
    pattern = re.compile(r'(?m)^__device__ __noinline__ int builtin_attribute\(Context\* c, int owner, int name\)\s*\{')
    matches = list(pattern.finditer(unit.source))
    if len(matches) != 1:
        raise ValueError('Unique shared builtin attribute definition required')
    match = matches[0]
    end = _matching_brace_end(unit.source, match.end()-1)
    body = unit.source[match.start():end]
    anchor = '    int otag = kind(c, owner);'
    if body.count(anchor) != 1:
        raise ValueError('Canonical container owner validation changed')
    # Extract the canonical method codes, rather than duplicate their ABI.
    canonical = re.findall(r'vm_str_equals\(c, name, "(\w+)", \d+\)\)\s*\{\s*return meta\(c, 34, owner, 0, (METHOD_\w+)\);', body)
    expected = [name for names in _METHODS.values() for name in names]
    if [name for name, _ in canonical] != expected:
        raise ValueError('Canonical container method inventory changed')
    references = {}
    for index, (tag, offset, count, _) in enumerate(image.nodes):
        if int(tag) != 4:
            continue
        begin, size = int(offset), int(count)
        if begin < 0 or size < 0 or begin+size > len(image.payload):
            raise ValueError('Frozen method name outside image')
        raw = image.payload[begin:begin+size].tobytes()
        references.setdefault(raw, []).append(-1-index)
    dispatch = '\n    // native_effect_query_names: immutable name references only.\n'
    cursor = 0
    guards = ('otag == TAG_DICT || (otag >= 8 && otag <= 10)',
              'otag == TAG_LIST', 'otag == TAG_TUPLE')
    for guard, names in zip(guards, _METHODS.values()):
        dispatch += '    if ('+guard+') {\n        switch(name) {\n'
        for name, code in canonical[cursor:cursor+len(names)]:
            refs = references.get(name.encode('ascii'), ())
            if refs:
                dispatch += ''.join('        case '+str(ref)+':\n' for ref in refs)
                dispatch += '            return meta(c, 34, owner, 0, '+code+');\n'
        dispatch += '        default: break;\n        }\n    }\n'
        cursor += len(names)
    body = body.replace(anchor, anchor+dispatch, 1)
    return replace(unit, source=unit.source[:match.start()]+body+unit.source[end:])
