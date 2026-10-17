"""Runtime-only call routing, retaining cached rule-unit inline definitions.

The canonical function binding and selected scalar builtin bodies are copied
verbatim into their uniquely owned out-of-line caller. No object representation,
allocation, logical instruction, descriptor or generator behavior changes.
"""
from dataclasses import replace
import re

from .native_array_codegen import _matching_brace_end
from .vm_builtins import emit_builtin_source
from .vm_stdlib import emit_stdlib_source


_SCALARS = ('ID', 'LEN', 'BOOL', 'INT', 'FLOAT')
_HOOK = ' if(native_fast_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;'


def _body(source, name):
    pattern = re.compile(r'(?m)^__device__ __noinline__ (?:int|void) ' + name +
                         r'\([^;{}]*\)\s*\{')
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError('Unique shared call owner required: ' + name)
    match = matches[0]
    end = _matching_brace_end(source, match.end() - 1)
    return match.start(), end, source[match.start():end]


def transform_runtime_calls(unit, image, *, function_calls=True, scalar_builtins=True,
                            type_builtins=False, mapping_get=False, deepcopy_calls=False):
    """Return the modified runtime unit; optional routes allow GPU ablations.

    The optional routes default off so the original scalar/function source
    and cached binary remain unchanged until their own timing is qualified.
    """
    options = (function_calls, scalar_builtins, type_builtins, mapping_get, deepcopy_calls)
    if any(type(option) is not bool for option in options):
        raise TypeError('Call route options must be booleans')
    if not any(options):
        raise ValueError('At least one call route must be selected')
    if unit.name != 'runtime':
        raise ValueError('Call routing requires the shared runtime unit')
    source = unit.source
    start, end, caller = _body(source, 'call')
    _, _, fast = _body(source, 'native_fast_call')
    function_marker = ' if(n[0]==T_FUNCTION||n[0]==T_DYNFN){'
    if any(marker in caller for marker in (
            'native_runtime_scalar_done', 'native_runtime_get_done', 'native_runtime_copy_done', function_marker,
            'if(NATIVE_STDLIB_TYPE>=0&&id==NATIVE_STDLIB_TYPE)')):
        raise ValueError('Shared call routing already specialized')
    if fast.count(function_marker) != 1 or caller.count(_HOOK) != 1:
        raise ValueError('Canonical function call boundary changed')
    begin = fast.index(function_marker)
    stop = _matching_brace_end(fast, fast.index('{', begin))
    functions = fast[begin:stop]
    # native_fast_call reports consumption whereas its caller returns void.
    if 'return 0;' in functions or 'return;' in functions:
        raise ValueError('Canonical function consumption contract changed')
    functions = functions.replace('return 1;', 'return;')
    if not functions.endswith('return;\n }'):
        raise ValueError('Canonical function return boundary changed')
    builtin = emit_builtin_source(image.intrinsic_names)
    # Refuse a drifted runtime even if the generator still has familiar cases.
    _, _, runtime_builtin = _body(source, 'native_builtin_call')
    cases = []
    for name in _SCALARS:
        marker = '            case NATIVE_' + name + ': {'
        if builtin.count(marker) != 1:
            raise ValueError('Canonical scalar builtin boundary changed: ' + name)
        begin = builtin.index(marker)
        stop = _matching_brace_end(builtin, builtin.index('{', begin))
        case = builtin[begin:stop]
        if runtime_builtin.count(case) != 1:
            raise ValueError('Shared scalar builtin body changed: ' + name)
        absent = 'if (NATIVE_' + name + ' < 0) return 0;'
        if case.count(absent) != 1 or case.count('return 0;') != 1:
            raise ValueError('Canonical scalar fallback changed: ' + name)
        case = case.replace(absent, '')
        case = case.replace('return 1;', 'goto native_runtime_scalar_done;')
        cases.append(case)
    guard = ' || '.join('(NATIVE_' + n + '>=0&&n[3]==NATIVE_' + n + ')' for n in _SCALARS)
    scalars = '''
 if(n[0]==T_NATIVE && (''' + guard + ''')){
  if(nkw){c->error=BAD_CALL;return;}
  int native_runtime_result=MISSING;int* out=&native_runtime_result;
  switch(n[3]){
''' + '\n'.join(cases) + '''
  }
  native_runtime_scalar_done:
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_runtime_result);
  return;
 }
'''
    stdlib = emit_stdlib_source(image)
    _, _, runtime_stdlib = _body(source, 'stdlib_call')
    exact_types = []
    for name in ('TYPE', 'ISINSTANCE'):
        marker = ' if(NATIVE_STDLIB_' + name + '>=0&&id==NATIVE_STDLIB_' + name + '){'
        if stdlib.count(marker) != 1:
            raise ValueError('Canonical type builtin boundary changed: ' + name)
        begin = stdlib.index(marker)
        stop = _matching_brace_end(stdlib, stdlib.index('{', begin))
        branch = stdlib[begin:stop]
        if runtime_stdlib.count(branch) != 1 or 'return 0;' in branch:
            raise ValueError('Shared type builtin body changed: ' + name)
        exact_types.append(branch.replace('return 1;', 'return;'))
    types = '\n if(n[0]==T_NATIVE){int id=n[3];\n' + '\n'.join(exact_types) + '\n }\n'
    marker = '            case METHOD_DICT_GET: {'
    if builtin.count(marker) != 1:
        raise ValueError('Canonical mapping get boundary changed')
    begin = builtin.index(marker)
    stop = _matching_brace_end(builtin, builtin.index('{', begin))
    case = builtin[begin:stop]
    if runtime_builtin.count(case) != 1 or 'return 0;' in case:
        raise ValueError('Shared mapping get body changed')
    # Only already-resolved original builtin bound methods use this route.
    # User overrides have function/bound-function tags and take the old route.
    get_body = case[case.index('{') + 1:-1].replace('return 1;', 'goto native_runtime_get_done;')
    get = '''
 if(n[0]==34&&n[3]==METHOD_DICT_GET){
  if(nkw){c->error=BAD_CALL;return;}
  int owner=n[1],native_runtime_result=MISSING;int* out=&native_runtime_result;
  {
''' + get_body + '''
  }
  native_runtime_get_done:
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_runtime_result);
  return;
 }
'''
    copies = ''
    if deepcopy_calls:
        # Keep the canonical argument/memo validation and deepcopy_tick. This
        # changes only routing, including the old blanket keyword rejection.
        from .vm_deepcopy import emit_deepcopy_source
        canonical_copy = emit_deepcopy_source(image.intrinsic_names)
        marker = '__device__ inline int deepcopy_call('
        begin = canonical_copy.index(marker)
        stop = _matching_brace_end(canonical_copy, canonical_copy.index('{', begin))
        if source.count(canonical_copy[begin:stop]) != 1:
            raise ValueError('Canonical deepcopy call body changed')
        if caller.count('if(deepcopy_call(c,callable,args,nargs,&native_result)||') != 1:
            raise ValueError('Canonical deepcopy routing boundary changed')
        copies = '''
 if(n[0]==T_NATIVE&&((NATIVE_DEEPCOPY>=0&&n[3]==NATIVE_DEEPCOPY)||
                    (NATIVE_GRAPH_COPY>=0&&n[3]==NATIVE_GRAPH_COPY))){
  if(nkw){c->error=BAD_CALL;return;}
  int native_runtime_copy_done=MISSING;
  if(!deepcopy_call(c,callable,args,nargs,&native_runtime_copy_done)){c->error=BAD_CALL;return;}
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_runtime_copy_done);
  return;
 }
'''
    caller = caller.replace(_HOOK, (functions if function_calls else '') +
                            (scalars if scalar_builtins else '') +
                            (types if type_builtins else '') +
                            (get if mapping_get else '') + copies + _HOOK)
    return replace(unit, source=source[:start] + caller + source[end:])
