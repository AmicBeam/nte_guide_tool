"""Separate native rule bodies from the shared runtime without rewriting them.

This accepts only the generated native-control-flow source. Runtime ABI types
and inline helpers remain identical in every unit; out-of-line implementations
and device tables have one owner. No function/candidate is pruned.
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
            # All template declarations live in this namespace in the emitter.
            text += '\nnamespace duel_rule_vm {\n' + '\n'.join(instantiations) + '\n}\n'
        result.append(NativeCompileUnit(name, text))
    return tuple(result)
