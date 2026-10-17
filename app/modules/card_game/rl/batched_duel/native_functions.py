"""Incremental whole-function control-flow lowering of frozen rule functions.

Complete mode lowers every frozen function, including calls, suspension and
exception recovery, and removes shard PC dispatch. Partial leaf planning remains
available for diagnostics. Objects, frames and primitive helpers retain their
canonical ABI; this is not a fully unboxed domain engine. No CPU callback runs.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from .native_array_codegen import _EXECUTE_PREAMBLE, _c_int, _matching_brace_end
from .cuda_vm import SUPPORTED_OPS
from .vm_compiled_blocks import SAFE_OPS, _emit_op


_UNCONDITIONAL = frozenset(('JUMP_ABSOLUTE', 'JUMP_FORWARD'))
_CONDITIONAL = frozenset(('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
                          'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP'))
_BRANCHES = _UNCONDITIONAL | _CONDITIONAL
_EXTRA_BRANCHES = frozenset(('FOR_ITER', 'JUMP_IF_NOT_EXC_MATCH'))
_METADATA_TARGETS = frozenset(('SETUP_FINALLY', 'SETUP_WITH'))
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


def compile_native_functions(plan, *, max_function_ops=256, complete=False, resume_pcs=None):
    """Compile complete functions, never a truncated prefix.

    Calls, properties, closures, generators and exception handlers are explicit
    barriers in leaf planning. Complete mode supports canonical operations and
    refuses any rejected function instead of silently retaining shard fallback.
    """
    plan.verify()
    if type(complete) is not bool:
        raise TypeError('Explicit complete native control-flow option required')
    if type(max_function_ops) is not int or not 1 <= max_function_ops <= 16384:
        raise ValueError('Native function size must be 1..16384 operations')
    entry_pcs = None
    if resume_pcs is not None:
        if not complete or type(resume_pcs) is not tuple or any(type(pc) is not int for pc in resume_pcs):
            raise ValueError('Complete native resume PCs require an explicit integer tuple')
        entry_pcs = _resume_points(plan, resume_pcs)
    rows = plan.coverage['instructions']
    selected, rejected = [], []
    for fn in plan.coverage['functions']:
        instructions = rows[fn['start']:fn['start'] + fn['count']]
        unsupported = sorted({r['op'] for r in instructions} - (SUPPORTED_OPS if complete else SUPPORTED))
        if not instructions or len(instructions) > max_function_ops:
            rejected.append((fn['index'], ('size',)))
        elif unsupported:
            rejected.append((fn['index'], tuple(unsupported)))
        elif not _valid_edges(instructions, complete=complete):
            rejected.append((fn['index'], ('control_flow',)))
        else:
            selected.append((fn, instructions))
    if complete and rejected:
        raise ValueError('Incomplete native function coverage: ' + repr(rejected[:8]))
    guards = _admission_guards()
    prefix = ''
    if complete:
        boundary = 'f=frame(c,c->current);fn=function(c,f[0]);'
        if guards.count(boundary) != 1:
            raise ValueError('Canonical complete native admission changed')
        admission = guards[:guards.index(boundary)].replace('return;', 'return 0;')
        prefix = ('__device__ inline int native_function_admit(Context* c,int* f,int pc){\n' +
                  admission + ' f[1]=pc+1;return 1;\n}\n'
                  '__device__ inline int native_function_active(Context* c,int* f,int owner,int fid){\n'
                  ' return !c->error&&!c->h.error&&c->raised==MISSING&&c->current==owner&&f[0]==fid;\n}\n')
    source = prefix + ''.join(_emit_function(fn, instructions, guards, complete=complete, entry_pcs=entry_pcs)
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


def _resume_points(plan, extra):
    rows = plan.coverage['instructions']
    if any(not 0 <= pc < len(rows) for pc in extra):
        raise ValueError('Native resume PC outside frozen program')
    result = set(extra)
    # These canonical operations cannot suspend into another VM frame. For
    # every other operation conservatively retain its following PC. Exception
    # block targets and all branch targets are retained independently.
    synchronous = SUPPORTED | frozenset(('LOAD_DEREF', 'STORE_DEREF', 'LOAD_CLOSURE',
        'MAKE_FUNCTION', 'SETUP_FINALLY', 'POP_BLOCK', 'POP_EXCEPT', 'GEN_START',
        'RERAISE', 'RAISE_VARARGS', 'LOAD_ASSERTION_ERROR'))
    for fn in plan.coverage['functions']:
        result.add(fn['start'])
        end = fn['start'] + fn['count']
        for row in rows[fn['start']:end]:
            if row['target'] >= 0:
                result.add(row['target'])
            if row['op'] not in synchronous and row['pc'] + 1 < end:
                result.add(row['pc'] + 1)
    return result


def _valid_edges(rows, *, complete=False):
    pcs = {r['pc'] for r in rows}
    for row in rows:
        op = row['op']
        if op in _BRANCHES or complete and op in _EXTRA_BRANCHES | _METADATA_TARGETS:
            if row['target'] not in pcs:
                return False
        elif row['target'] != -1:
            return False
        terminal = ('RETURN_VALUE', 'RERAISE', 'RAISE_VARARGS') if complete else ('RETURN_VALUE',)
        if op not in _UNCONDITIONAL and op not in terminal and row['pc'] + 1 not in pcs:
            return False
    return True


def _body(row, *, complete=False):
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
    if complete and op in SUPPORTED_OPS:
        return ('native_step<OP_' + op + '>(c,f,fn,' +
                ','.join(_c_int(row[k]) for k in ('arg', 'operand', 'target')) + ');\n')
    raise ValueError('Unsupported whole-function operation: ' + op)


def _emit_function(fn, rows, guards, *, complete=False, entry_pcs=None):
    fid = fn['index']
    parts = ['__device__ __noinline__ void native_function_' + str(fid) +
             '(Context* c,int* f,const int* fn,int pc){\n',
             ' const int owner=c->current;\n switch(pc){\n']
    # Only entry/resume uses a PC switch. Every internal edge is native goto.
    for row in rows:
        if entry_pcs is None or row['pc'] in entry_pcs:
            parts.append(' case ' + str(row['pc']) + ':goto op_' + str(row['pc']) + ';\n')
    parts.append(' default:c->error=BAD_PROGRAM;return;\n }\n')
    for row in rows:
        pc, op = row['pc'], row['op']
        if complete:
            # Entry/resume skips admission (execute already consumed it).
            # Native fall-through reaches admission for the next operation.
            parts.append('admit_' + str(pc) + ':if(!native_function_admit(c,f,' + str(pc) + '))return;\n')
        parts.append('op_' + str(pc) + ':{\n')
        # execute() already admitted and advanced the first operation. The
        # admission label is used only by internal edges, exactly once each.
        if complete and op not in ('LOAD_FAST', 'STORE_FAST', 'LOAD_CONST', 'POP_TOP'):
            # Keep large primitive/library implementations shared. Inlining
            # them into every rule function makes optimization scale with
            # every PC instead of with the small opcode specialization set.
            parts.append('native_step<OP_' + op + '>(c,f,fn,' +
                         ','.join(_c_int(row[k]) for k in ('arg', 'operand', 'target')) + ');\n')
        else:
            parts.append(' if(!native_handlers<OP_' + op + '>(c,f,' +
                         ','.join(_c_int(row[k]) for k in ('arg', 'operand', 'target')) + ')){\n')
            parts.append(_body(row))
            parts.append(' }\n')
        if complete:
            parts.append(' if(!native_function_active(c,f,owner,' + str(fid) + '))return;\n')
        else:
            parts.append(' if(c->error||c->h.error||c->raised!=MISSING)return;\n')
            parts.append(' if(c->current!=owner||f[0]!=' + str(fid) + ')return;\n')
        if op in ('RETURN_VALUE', 'YIELD_VALUE', 'RERAISE', 'RAISE_VARARGS'):
            parts.append(' return;\n')
        else:
            if op in _BRANCHES or complete and op in _EXTRA_BRANCHES:
                parts.append(' if(f[1]==' + str(row['target']) + ')goto admit_' + str(row['target']) + ';\n')
            if op not in _UNCONDITIONAL:
                if complete and op not in _BRANCHES | _EXTRA_BRANCHES:
                    # Canonical non-branch operations either suspend/change
                    # frame (already returned above), fail, or fall through.
                    pass
                elif complete:
                    parts.append(' if(f[1]!=' + str(pc + 1) + '){c->error=BAD_PROGRAM;return;}\n')
                else:
                    parts.append(' if(f[1]==' + str(pc + 1) + ')goto admit_' + str(pc + 1) + ';\n')
            if not complete or op in _UNCONDITIONAL:
                parts.append(' c->error=BAD_PROGRAM;return;\n')
        parts.append('}\n')
        if not complete:
            parts.append('admit_' + str(pc) + ':{\n' + guards + ' f[1]++;\n goto op_' + str(pc) + ';\n}\n')
    parts.append('}\n')
    return ''.join(parts)


def integrate_native_functions(source, plan, *, complete=False, resume_pcs=None):
    """Intercept admitted function entries before the existing PC dispatcher."""
    compiled = compile_native_functions(plan, complete=complete,
        max_function_ops=16384 if complete else 256, resume_pcs=resume_pcs)
    marker = '__device__ inline void native_dispatch_pc(Context* c,int* f,const int* fn,int pc){\n'
    if source.count(marker) != 1:
        raise ValueError('Canonical native function dispatch boundary changed')
    if not complete:
        hook = ' if(native_function_dispatch(c,f,fn,pc))return;\n'
        return source.replace(marker, compiled.source + marker + hook, 1), compiled
    first = source.index('__device__ inline void native_pc_shard_0(')
    end = _matching_brace_end(source, source.index('{', source.index(marker)))
    dispatch = marker + ' if(!native_function_dispatch(c,f,fn,pc))c->error=BAD_PROGRAM;\n}\n'
    # Complete mode has one native body per function, without retaining a
    # second copy of all PCs in shard dispatchers. Unknown functions fail.
    return source[:first] + compiled.source + dispatch + source[end:], compiled
