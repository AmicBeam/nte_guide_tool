"""Incremental whole-function control-flow lowering of frozen rule functions.

Selected leaf functions execute native branches instead of returning to a PC
switch after every operation. Objects, frames and primitive helpers retain the
canonical ABI; this is the first migration stage, not a fully typed rule engine.
Unsupported functions remain on the existing device path. Selection is static,
auditable and independent of game state. No runtime Python callback is used.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from .native_array_codegen import _EXECUTE_PREAMBLE, _c_int
from .vm_compiled_blocks import SAFE_OPS, _emit_op


_UNCONDITIONAL = frozenset(('JUMP_ABSOLUTE', 'JUMP_FORWARD'))
_CONDITIONAL = frozenset(('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
                          'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP'))
_BRANCHES = _UNCONDITIONAL | _CONDITIONAL
SUPPORTED = SAFE_OPS | _BRANCHES | frozenset((
    'LOAD_GLOBAL', 'STORE_SUBSCR', 'RETURN_VALUE',
))


@dataclass(frozen=True)
class NativeFunctions:
    source: str
    function_indexes: tuple
    function_ids: tuple
    instruction_count: int
    rejected: tuple
    identity: str

    def verify(self):
        if self.identity != _identity(self.source, self.function_indexes,
                self.function_ids, self.instruction_count, self.rejected):
            raise ValueError('Native function lowering identity changed')
        return True


def _identity(*values):
    return sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def _admission_guards():
    marker = '  if(++c->steps>c->step_limit)'
    if _EXECUTE_PREAMBLE.count(marker) != 1:
        raise ValueError('Canonical native function step guard changed')
    guards = _EXECUTE_PREAMBLE[_EXECUTE_PREAMBLE.index(marker):]
    before = 'int* f=frame(c,c->current);const int* fn=function(c,f[0]);'
    if guards.count('break;') != 4 or guards.count(before) != 1:
        raise ValueError('Canonical native function admission guards changed')
    return (guards.replace('break;', 'return;')
            .replace(before, 'f=frame(c,c->current);fn=function(c,f[0]);')
            .replace('int pc=f[1];', 'pc=f[1];'))


def compile_native_functions(plan, *, max_function_ops=256):
    """Compile complete supported leaf functions, never a truncated prefix.

    Calls, properties, closures, generators and exception handlers are explicit
    barriers for this stage. A function containing one is wholly unselected.
    """
    plan.verify()
    if type(max_function_ops) is not int or not 1 <= max_function_ops <= 1024:
        raise ValueError('Native function size must be 1..1024 operations')
    rows = plan.coverage['instructions']
    selected, rejected = [], []
    for fn in plan.coverage['functions']:
        instructions = rows[fn['start']:fn['start'] + fn['count']]
        unsupported = sorted({r['op'] for r in instructions} - SUPPORTED)
        if not instructions or len(instructions) > max_function_ops:
            rejected.append((fn['index'], ('size',)))
        elif unsupported:
            rejected.append((fn['index'], tuple(unsupported)))
        elif not _valid_edges(instructions):
            rejected.append((fn['index'], ('control_flow',)))
        else:
            selected.append((fn, instructions))
    guards = _admission_guards()
    source = ''.join(_emit_function(fn, instructions, guards)
                     for fn, instructions in selected)
    source += '__device__ inline int native_function_dispatch(Context* c,int* f,const int* fn,int pc){\n switch(f[0]){\n'
    for fn, _ in selected:
        source += (' case ' + str(fn['index']) + ':native_function_' +
                   str(fn['index']) + '(c,f,fn,pc);return 1;\n')
    source += ' default:return 0;\n }\n}\n'
    indexes = tuple(fn['index'] for fn, _ in selected)
    ids = tuple(fn['function_id'] for fn, _ in selected)
    count = sum(len(inst) for _, inst in selected)
    rejected = tuple(rejected)
    result = NativeFunctions(source, indexes, ids, count, rejected,
                             _identity(source, indexes, ids, count, rejected))
    result.verify()
    return result


def _valid_edges(rows):
    pcs = {r['pc'] for r in rows}
    for row in rows:
        op = row['op']
        if op in _BRANCHES:
            if row['target'] not in pcs:
                return False
        elif row['target'] != -1:
            return False
        if op not in _UNCONDITIONAL and op != 'RETURN_VALUE' and row['pc'] + 1 not in pcs:
            return False
    return True


def _body(row):
    op, arg, value = row['op'], _c_int(row['arg']), _c_int(row['operand'])
    if op in SAFE_OPS:
        # Reuse checked canonical primitive bodies, including overflow and
        # mutation ordering. Only the enclosing helper's return type differs.
        return ''.join(_emit_op(row, 'fn[5]')).replace('return 1;', 'return;')
    if op == 'LOAD_GLOBAL':
        return ('int v=dict_get(&c->h,f[8],' + value + ',MISSING);'
                'if(v==MISSING)c->error=UNBOUND_LOCAL;else push(c,f,v);\n')
    if op == 'STORE_SUBSCR':
        # This opcode is entirely owned by the preceding canonical handlers.
        return 'c->error=UNSUPPORTED_OP;\n'
    if op == 'RETURN_VALUE':
        return 'return_value(c,pop(c,f),0);\n'
    if op in _UNCONDITIONAL:
        return 'f[1]=' + str(row['target']) + ';\n'
    if op in _CONDITIONAL:
        peek = op.startswith('JUMP_IF_')
        wanted = '1' if op in ('POP_JUMP_IF_TRUE', 'JUMP_IF_TRUE_OR_POP') else '0'
        return ('int yes=truth_value(c,' + ('peek' if peek else 'pop') + '(c,f));'
                'if(yes==' + wanted + ')f[1]=' + str(row['target']) + ';' +
                ('else pop(c,f);' if peek else '') + '\n')
    raise ValueError('Unsupported whole-function operation: ' + op)


def _emit_function(fn, rows, guards):
    fid = fn['index']
    parts = ['__device__ __noinline__ void native_function_' + str(fid) +
             '(Context* c,int* f,const int* fn,int pc){\n',
             ' const int owner=c->current;\n switch(pc){\n']
    # Only entry/resume uses a PC switch. Every internal edge is native goto.
    for row in rows:
        parts.append(' case ' + str(row['pc']) + ':goto op_' + str(row['pc']) + ';\n')
    parts.append(' default:c->error=BAD_PROGRAM;return;\n }\n')
    for row in rows:
        pc, op = row['pc'], row['op']
        parts.append('op_' + str(pc) + ':{\n')
        # execute() already admitted and advanced the first operation. The
        # admission label is used only by internal edges, exactly once each.
        parts.append(' if(!native_handlers<OP_' + op + '>(c,f,' +
                     ','.join(_c_int(row[k]) for k in ('arg', 'operand', 'target')) + ')){\n')
        parts.append(_body(row))
        parts.append(' }\n if(c->error||c->h.error||c->raised!=MISSING)return;\n')
        parts.append(' if(c->current!=owner||f[0]!=' + str(fid) + ')return;\n')
        if op == 'RETURN_VALUE':
            parts.append(' return;\n')
        else:
            if op in _BRANCHES:
                parts.append(' if(f[1]==' + str(row['target']) + ')goto admit_' + str(row['target']) + ';\n')
            if op not in _UNCONDITIONAL:
                parts.append(' if(f[1]==' + str(pc + 1) + ')goto admit_' + str(pc + 1) + ';\n')
            parts.append(' c->error=BAD_PROGRAM;return;\n')
        parts.append('}\n')
        parts.append('admit_' + str(pc) + ':{\n' + guards + ' f[1]++;\n goto op_' + str(pc) + ';\n}\n')
    parts.append('}\n')
    return ''.join(parts)


def integrate_native_functions(source, plan):
    """Intercept admitted function entries before the existing PC dispatcher."""
    compiled = compile_native_functions(plan)
    marker = '__device__ inline void native_dispatch_pc(Context* c,int* f,const int* fn,int pc){\n'
    if source.count(marker) != 1:
        raise ValueError('Canonical native function dispatch boundary changed')
    hook = ' if(native_function_dispatch(c,f,fn,pc))return;\n'
    return source.replace(marker, compiled.source + marker + hook, 1), compiled
