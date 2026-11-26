"""Optional straight-line stack continuations in the static native PC cases.

Every original entry PC remains available. Only canonical LOAD_CONST,
LOAD_FAST, STORE_FAST and POP_TOP bodies are grouped; calls, branches, heap
operations and exception handlers remain barriers. Logical step/deadline and
error checks still execute between instructions. No rule formula is changed.
"""
import re
from .native_array_codegen import _DIRECT_STACK_OPS, _EXECUTE_PREAMBLE, _emit_pc_case, _stack_body, _c_int


def fuse_stack_runs(source, plan, *, max_run=8):
    if type(max_run) is not int or not 2<=max_run<=16:
        raise ValueError('Static stack run bound must be 2..16')
    plan.verify()
    rows=plan.coverage['instructions']
    marker='  if(++c->steps>c->step_limit)'
    if _EXECUTE_PREAMBLE.count(marker)!=1:
        raise ValueError('Canonical logical-step guard changed')
    guards=_EXECUTE_PREAMBLE[_EXECUTE_PREAMBLE.index(marker):]
    if 'continue;' in guards or guards.count('break;')!=4:
        raise ValueError('Canonical stack-continuation guards changed')
    guards=guards.replace('break;','return;')
    replacements={}
    for index,row in enumerate(rows):
        if row['op'] not in _DIRECT_STACK_OPS:
            continue
        follow=[]
        for candidate in rows[index+1:index+max_run]:
            if (candidate['op'] not in _DIRECT_STACK_OPS
                    or candidate['function_index']!=row['function_index']):
                break
            follow.append(candidate)
        if not follow:
            continue
        original=''.join(_emit_pc_case(row,direct_stack_dispatch=True))
        if not original.endswith('return;}\n') or '\n' in original[:-1]:
            raise ValueError('Exact canonical static stack case required')
        body=original[:-len('return;}\n')]
        for candidate in follow:
            # Error conversion/unwinding stays in the outer canonical loop.
            # Do not consume another logical step before that loop sees it.
            body+=('if(c->error||c->h.error||c->raised!=MISSING||f[1]!='+str(candidate['pc'])+')return;\n'
                   '{ /* native stack continuation */\n'+guards+'f[1]++;\n'+
                   _stack_body(candidate['op'],_c_int(candidate['arg']),_c_int(candidate['operand'])))+'\n}\n'
        body+='return;}\n'
        replacements[original]=body
    # Compute all matches against the unmodified source. Retaining each case
    # independently permits jumps into any middle PC without changing targets.
    seen=set()
    def replace(match):
        original=match.group(0)
        if original not in replacements:
            return original
        if original in seen:
            raise ValueError('Duplicate canonical static stack case')
        seen.add(original)
        return replacements[original]
    result=re.sub(r'(?m)^  case [0-9]+:\{ /\* stack [A-Z_]+ \*/ [^\n]*return;\}\n',replace,source)
    if seen!=set(replacements):
        raise ValueError('Missing canonical static stack continuation case')
    return result
