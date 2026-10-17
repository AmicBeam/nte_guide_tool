"""Separate native rule bodies from the shared runtime without rewriting them.

This accepts only the generated native-control-flow source. Common runtime ABI
types and rule inline helpers remain identical in every unit; out-of-line
implementations and device tables have one owner. Projection-only helpers stay
with the runtime kernels. No function/candidate is pruned.
"""
from dataclasses import dataclass
from hashlib import sha256
import re


@dataclass(frozen=True)
class NativeCompileUnit:
    name: str
    source: str

    @property
    def sha256(self):
        return sha256(self.source.encode('utf8')).hexdigest()


# Mask literals/comments before matching delimiters, including braces in text.
_LEXICAL = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_FUNCTION = re.compile(
    r'(?m)^(?P<template>template\s*<int\s+(?:OP|MODE)>\s*\n)?'
    r'(?P<signature>__device__ __noinline__ (?:int|void) (?P<name>\w+)\([^;{}]*\))\s*\{')
_GLOBAL = re.compile(r'(?m)^extern "C" (?:__global__ )?void \w+\([^;{}]*\)\s*\{')
_TABLE = re.compile(r'(?m)^__device__ (?P<decl>(?:const )?[\w ]+\s+\w+\[[^;{}]*)(?P<init>\s*=\s*\{|;)')
_TRAILER_HELPERS = frozenset(('soa_export_index', 'soa_export_enqueue', 'soa_export_into',
                              'literal_equal', 'field_ref', 'integer64', 'side_code'))
_TRAILER_KERNELS = frozenset(('soa_export_projection', 'soa_select_result', 'soa_decode_metadata',
                              'soa_decode_pair', 'soa_decode_float'))
_TRAILER_HOST = frozenset('native_' + name for name in ('soa_execute', *_TRAILER_KERNELS))
_INLINE = re.compile(r'(?m)^__device__ (?:inline|__noinline__) [\w *]+? (\w+)\([^;{}]*\)\s*\{')


def _runtime_trailer(source):
    """Move only the known, unreferenced projection trailer to its owner.

    This runs after the common ABI prefix was generated. Any new helper,
    declaration, table or backwards reference fails closed instead of silently
    changing cached rule units or omitting a dependency they need.
    """
    mask = _LEXICAL.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
    kernels = [m for m in _GLOBAL.finditer(source) if mask[m.end()-1] == '{'
               and re.search(r'\bvm_soa_execute\(', m[0])]
    if len(kernels) != 1:
        raise ValueError('Unique native rule kernel boundary required')
    cut = _end(mask, kernels[0].end()-1)
    trailer, tail_mask = source[cut:], mask[cut:]
    if not tail_mask.strip():
        return source, ''
    helpers = list(_INLINE.finditer(tail_mask))
    globals_ = [m for m in _GLOBAL.finditer(trailer) if tail_mask[m.end()-1] == '{']
    helper_names = [m[1] for m in helpers]
    global_names = [re.search(r'void (\w+)\(', m[0])[1] for m in globals_]
    if (len(helper_names) != len(set(helper_names)) or set(helper_names) != _TRAILER_HELPERS
            or len(global_names) != len(set(global_names))
            or not _TRAILER_KERNELS <= set(global_names)
            or not set(global_names) <= _TRAILER_KERNELS | _TRAILER_HOST):
        raise ValueError('Unknown full-module projection trailer definitions')
    for name in (*helper_names, *global_names):
        if re.search(r'\b' + name + r'\b', mask[:cut]):
            raise ValueError('Rule prefix references a projection-only symbol: ' + name)
    spans = sorted((m.start(), _end(tail_mask, m.end()-1)) for m in (*helpers, *globals_))
    cursor, residual = 0, []
    for begin, end in spans:
        if begin < cursor:
            raise ValueError('Overlapping full-module trailer definitions')
        residual.append(tail_mask[cursor:begin])
        cursor = end
    residual.append(tail_mask[cursor:])
    residue = ''.join(residual)
    if (residue.count('using namespace duel_vm_heap;') != 1
            or residue.count('const int RESULT_INVALID=150;') != 1):
        raise ValueError('Canonical projection metadata declarations changed')
    residue = residue.replace('using namespace duel_vm_heap;', '').replace('const int RESULT_INVALID=150;', '')
    namespaces = len(re.findall(r'namespace\s+(?:duel_vm_heap|soa_metadata_result)\s*\{', residue))
    residue = re.sub(r'namespace\s+(?:duel_vm_heap|soa_metadata_result)\s*\{', '', residue)
    residue = re.sub(r'(?m)^\s*#(?:ifndef __CUDACC__|endif)\s*$', '', residue)
    if residue.count('}') != namespaces or residue.replace('}', '').strip():
        raise ValueError('Unrecognized declarations in full-module projection trailer')
    return source[:cut], trailer


def _end(mask, opening):
    depth = 1
    cursor = opening + 1
    while depth and cursor < len(mask):
        depth += (mask[cursor] == '{') - (mask[cursor] == '}')
        cursor += 1
    if depth:
        raise ValueError('Unbalanced generated native source')
    return cursor


def split_native_source(source, *, functions_per_unit=64):
    """Return runtime, value-region chunks and complete rule-function chunks.

    CUDA callers must compile all units with relocatable device code and link
    every object. Keeping the exact original source separately identifies the
    rule program independently of the compilation/link strategy.
    """
    if type(functions_per_unit) is not int or not 1 <= functions_per_unit <= 256:
        raise ValueError('functions_per_unit must be an integer from 1 to 256')
    if not isinstance(source, str) or 'void vm_soa_execute(' not in source:
        raise ValueError('Generated native SoA source required')
    source, trailer = _runtime_trailer(source)
    mask = _LEXICAL.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
    functions = []
    for match in _FUNCTION.finditer(mask):
        end = _end(mask, match.end()-1)
        functions.append((match, end))
    rules = [m['name'] for m, _ in functions if re.fullmatch(r'native_function_\d+', m['name'])]
    values = [m['name'] for m, _ in functions if re.fullmatch(r'native_values_\d+', m['name'])]
    if not rules or len(set(rules)) != len(rules) or len(set(values)) != len(values):
        raise ValueError('Unique complete native rule functions required')
    if 'native_pc_shard_' in source:
        raise ValueError('Complete native functions required, not PC shards')
    owners = {name: f'{kind}_{i//functions_per_unit:03d}'
              for kind, names in (('rules', rules), ('values', values))
              for i, name in enumerate(names)}
    # Each replacement carries the declaration present in every other unit.
    edits = []
    instantiations = []
    for match, end in functions:
        name = match['name']
        signature = source[match.start('signature'):match.end('signature')]
        template = source[match.start('template'):match.end('template')] if match['template'] else ''
        declaration = template + signature + ';'
        owner = owners.get(name, 'runtime')
        edits.append((match.start(), end, declaration, owner, source[match.start():end]))
        if template:
            arguments = sorted(set(re.findall(r'\b' + name + r'<(OP_\w+|\d+)>', mask)))
            for arg in arguments:
                instantiations.append('template ' + signature.replace(name+'(', name+'<'+arg+'>(', 1) + ';')
    # Kernels and host diagnostic launchers belong exclusively to runtime.
    for match in _GLOBAL.finditer(source):
        # The regex is anchored outside function bodies in generated sources.
        if mask[match.end()-1] != '{':
            continue
        end = _end(mask, match.end()-1)
        edits.append((match.start(), end, '', 'runtime', source[match.start():end]))
    # Device constants need external linkage to be shared across CUDA units.
    for match in _TABLE.finditer(mask):
        begin = match.start()
        if any(a <= begin < b for a, b, *_ in edits):
            continue
        end = mask.index(';', match.end()-1) + 1
        text = source[begin:end]
        equal = text.find('=')
        decl = text[:equal].rstrip() if equal >= 0 else text[:-1]
        declaration = 'extern ' + decl + ';'
        definition = 'extern ' + text if equal >= 0 else text
        edits.append((begin, end, declaration, 'runtime', definition))
    edits.sort(key=lambda item: item[0])
    for left, right in zip(edits, edits[1:]):
        if left[1] > right[0]:
            raise ValueError('Overlapping generated source boundaries')
    # Avoid reconstructing the very large monolithic source for each chunk.
    segments, cursor = [], 0
    for begin, end, declaration, owner, body in edits:
        segments.append((None, source[cursor:begin], ''))
        segments.append((owner, body, declaration))
        cursor = end
    segments.append((None, source[cursor:], ''))
    names = ('runtime', *sorted(set(owners.values())))
    result = []
    for name in names:
        text = ''.join(body if owner is None or owner == name else declaration
                       for owner, body, declaration in segments)
        if name == 'runtime':
            text += trailer
            # All template declarations live in this namespace in the emitter.
            text += '\nnamespace duel_rule_vm {\n' + '\n'.join(instantiations) + '\n}\n'
        result.append(NativeCompileUnit(name, text))
    return tuple(result)
