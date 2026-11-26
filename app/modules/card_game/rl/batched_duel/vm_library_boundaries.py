"""Optional large-library CUDA call boundaries; function bodies stay identical."""
import re

LIBRARY_FUNCTIONS = ('stdlib_invoke','stdlib_resume','stdlib_minmax_next',
    'lookup_super','stdlib_lookup','stdlib_sorted_keys','object_attribute',
    'object_store','stdlib_call','stdlib_handle_opcode')
_NONCODE = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|//[^\n]*|/\*[\s\S]*?\*/''')


def separate_library_calls_source(source,functions=LIBRARY_FUNCTIONS):
    if type(source) is not str:
        raise TypeError('CUDA source text required')
    if type(functions) not in (tuple,list) or not functions or any(type(n) is not str for n in functions):
        raise TypeError('Nonempty explicit function names required')
    if len(set(functions)) != len(functions) or any(n not in LIBRARY_FUNCTIONS for n in functions):
        raise ValueError('Unique supported device functions required')
    # Keep indices/newlines intact while excluding comments and string literals.
    masked = _NONCODE.sub(lambda m: ''.join('\n' if c=='\n' else ' ' for c in m.group()),source)
    replacements=[]
    for name in functions:
        pattern=re.compile(r'(?m)^([ \t]*__device__[ \t]+)(inline)([ \t]+(?:void|int)[ \t]+'+name+r'[ \t]*\()')
        matches=list(pattern.finditer(masked))
        if not matches:
            raise ValueError('Missing device function boundary: '+name)
        replacements.extend((m.start(2),m.end(2)) for m in matches)
    for start,end in sorted(replacements,reverse=True):
        source=source[:start]+'__noinline__'+source[end:]
    return source
