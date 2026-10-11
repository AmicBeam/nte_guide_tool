"""CPU compiler of frozen VM-image basic blocks into a CUDA device helper.

Generated CUDA inlines selected straight-line prefixes so the existing
execute(Context*) opcode loop can skip per-instruction table lookup. Official
frozen Python rules remain the sole semantic truth. This module does not invent
game logic, run a CPU VM, or claim a measured speedup.

Primary integrates the snippet; this file does not edit cuda_vm.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType

from .vm_image import VmImage
from .vm_numeric import NUMERIC_OPS

HELPER_NAME = 'run_compiled_block'
BEGIN_OP_NAME = 'compiled_begin_op'

SAFE_STACK_OPS = frozenset((
    'NOP', 'EXTENDED_ARG',
    'LOAD_CONST', 'LOAD_FAST', 'STORE_FAST', 'DELETE_FAST',
    'POP_TOP', 'DUP_TOP', 'DUP_TOP_TWO',
    'ROT_TWO', 'ROT_THREE', 'ROT_FOUR', 'ROT_N',
))
SAFE_PRIMITIVE_OPS = frozenset((
    'BUILD_TUPLE', 'BUILD_LIST',
    'UNARY_NOT', 'UNARY_NEGATIVE',
    'IS_OP', 'COMPARE_OP',
    'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY',
)) | NUMERIC_OPS
SAFE_OPS = SAFE_STACK_OPS | SAFE_PRIMITIVE_OPS

_CALL_OPS = frozenset(('CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX', 'CALL_METHOD'))
_JUMP_OPS = frozenset((
    'JUMP_ABSOLUTE', 'JUMP_FORWARD',
    'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
    'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP',
    'FOR_ITER', 'JUMP_IF_NOT_EXC_MATCH',
))
_RETURN_OPS = frozenset(('RETURN_VALUE',))
_GENERATOR_OPS = frozenset(('YIELD_VALUE', 'GEN_START'))
_CONTEXT_OPS = frozenset(('SETUP_WITH', 'WITH_EXCEPT_START'))
_EXCEPTION_OPS = frozenset((
    'SETUP_FINALLY', 'POP_BLOCK', 'POP_EXCEPT', 'RERAISE', 'RAISE_VARARGS',
))
_GETTER_OPS = frozenset(('LOAD_ATTR', 'LOAD_METHOD', 'STORE_ATTR', 'IMPORT_FROM'))
_FRAME_OPS = frozenset(('MAKE_FUNCTION', 'GET_ITER', 'IMPORT_NAME'))
FENCE_OPS = (
    _CALL_OPS | _JUMP_OPS | _RETURN_OPS | _GENERATOR_OPS | _CONTEXT_OPS
    | _EXCEPTION_OPS | _GETTER_OPS | _FRAME_OPS
)

_INTEGRATION = MappingProxyType({
    'hook_function': 'duel_rule_vm::execute',
    'insert_source_before': '__device__ inline void execute(Context* c)',
    'call': 'if(' + HELPER_NAME + '(c)) continue;',
    'site': 'after STOP_ITERATION/raised/h.error handling, before step increment and instruction fetch',
    'return_0': 'no compiled block at current PC; Context is not mutated',
    'return_1': 'block ran; steps, next PC, stack/locals, c->error and c->h.error match per-instruction execute()',
    'uncompiled_path': 'existing GPU VM opcode loop; never CPU fallback',
    'diagnostics': 'STEP_LIMIT/DEADLINE leave PC unadvanced; op body failures leave f[1] at next PC; statuses still use current function/pc/steps',
})


class CompiledBlockError(ValueError):
    """Malformed frozen IR or illegal compiler budget."""


@dataclass(frozen=True)
class CompiledBlocks:
    source: str
    coverage: object
    identity: str
    helper_name: str = HELPER_NAME

    def verify(self):
        if self.helper_name != HELPER_NAME:
            raise CompiledBlockError('Compiled helper name changed')
        if self.identity != _identity(self.source, self.coverage):
            raise CompiledBlockError('Compiled block identity changed')
        if HELPER_NAME not in self.source:
            raise CompiledBlockError('Compiled helper missing from CUDA source')
        return True


def emit_compiled_blocks(
    image,
    *,
    max_blocks=64,
    max_ops=1024,
    max_source_chars=65536,
    max_functions=8,
    max_prefix_ops=256,
    max_block_ops=32,
    entries=None,
):
    """Compile selected frozen basic blocks to a __device__ helper.

    Budgets are deterministic fail-closed caps, not throughput claims.
    """
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    budget = dict(
        max_blocks=_bound('max_blocks', max_blocks, 4096),
        max_ops=_bound('max_ops', max_ops, 8192),
        max_source_chars=_bound('max_source_chars', max_source_chars, 200000),
        max_functions=_bound('max_functions', max_functions, 1024),
        max_prefix_ops=_bound('max_prefix_ops', max_prefix_ops, 8192),
        max_block_ops=_bound('max_block_ops', max_block_ops, 256),
    )
    _validate_image(image)
    selected = _hot_functions(image, entries, budget['max_functions'])
    names = image.opcode_names
    blocks = []
    fenced = []
    compiled_ops = 0
    for fid in selected:
        start = int(image.functions[fid, 0])
        n = int(image.functions[fid, 1])
        nlocals = int(image.functions[fid, 5])
        prefix_n = n if n < budget['max_prefix_ops'] else budget['max_prefix_ops']
        prefix_end = start + prefix_n
        targets = _jump_targets(image, start, n)
        pc = start
        while pc < prefix_end:
            if len(blocks) >= budget['max_blocks'] or compiled_ops >= budget['max_ops']:
                break
            op = names[int(image.instructions[pc, 0])]
            reason = _fence_reason(op, int(image.instructions[pc, 3]))
            if reason is not None:
                if len(fenced) < budget['max_ops']:
                    fenced.append(dict(pc=pc, op=op, reason=reason, function_index=fid))
                pc += 1
                continue
            ops = []
            block_pc = pc
            while pc < prefix_end and len(ops) < budget['max_block_ops']:
                if compiled_ops + len(ops) >= budget['max_ops']:
                    break
                if ops and pc in targets:
                    break
                inst_op = names[int(image.instructions[pc, 0])]
                inst_reason = _fence_reason(inst_op, int(image.instructions[pc, 3]))
                if inst_reason is not None:
                    break
                row = image.instructions[pc]
                ops.append(dict(
                    pc=pc,
                    op=inst_op,
                    arg=int(row[1]),
                    operand=int(row[2]),
                    target=int(row[3]),
                ))
                pc += 1
            if not ops:
                break
            blocks.append(dict(
                pc=block_pc,
                end_pc=ops[-1]['pc'] + 1,
                function_index=fid,
                function_id=image.function_ids[fid],
                nlocals=nlocals,
                ops=tuple(item['op'] for item in ops),
                args=tuple(item['arg'] for item in ops),
                operands=tuple(item['operand'] for item in ops),
                instructions=tuple(ops),
            ))
            compiled_ops += len(ops)
        if len(blocks) >= budget['max_blocks'] or compiled_ops >= budget['max_ops']:
            break
    source = _emit_source(blocks, budget['max_source_chars'])
    kept = len(blocks)
    if len(source) > budget['max_source_chars']:
        lo, hi = 0, len(blocks)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = _emit_source(blocks[:mid], budget['max_source_chars'])
            if len(candidate) <= budget['max_source_chars']:
                lo = mid
                source = candidate
            else:
                hi = mid - 1
        kept = lo
        source = _emit_source(blocks[:kept], budget['max_source_chars'])
        if len(source) > budget['max_source_chars']:
            raise CompiledBlockError('Compiled CUDA source exceeds code budget')
    blocks = blocks[:kept]
    compiled_ops = sum(len(block['ops']) for block in blocks)
    coverage = MappingProxyType({
        'helper': HELPER_NAME,
        'compiled_blocks': tuple(MappingProxyType({
            'pc': block['pc'],
            'end_pc': block['end_pc'],
            'function_index': block['function_index'],
            'function_id': block['function_id'],
            'ops': block['ops'],
            'args': block['args'],
            'operands': block['operands'],
        }) for block in blocks),
        'compiled_pc': tuple(block['pc'] for block in blocks),
        'compiled_op_count': compiled_ops,
        'selected_functions': tuple(image.function_ids[i] for i in selected),
        'fenced_ops': tuple(MappingProxyType(item) for item in fenced),
        'budget': MappingProxyType(budget),
        'safe_ops': tuple(sorted(SAFE_OPS)),
        'integration': _INTEGRATION,
        'cpu_fallback': False,
        'measured_speedup': False,
    })
    result = CompiledBlocks(source, coverage, _identity(source, coverage), HELPER_NAME)
    result.verify()
    return result


def _bound(name, value, hi):
    if type(value) is not int or isinstance(value, bool) or not 1 <= value <= hi:
        raise ValueError('Positive bounded ' + name + ' required')
    return value


def _validate_image(image):
    image.verify()
    instructions = image.instructions
    functions = image.functions
    if instructions.ndim != 2 or instructions.shape[1] != 4:
        raise CompiledBlockError('Instruction table must be int32[N,4]')
    if functions.ndim != 2 or functions.shape[1] != 18:
        raise CompiledBlockError('Function table must be int32[M,18]')
    ninst = int(instructions.shape[0])
    nops = len(image.opcode_names)
    if nops < 1 and ninst:
        raise CompiledBlockError('Opcode names missing for instruction stream')
    for fid in range(int(functions.shape[0])):
        start = int(functions[fid, 0])
        n = int(functions[fid, 1])
        nlocals = int(functions[fid, 5])
        stacksize = int(functions[fid, 6])
        if start < 0 or n < 0 or start > ninst or start + n > ninst:
            raise CompiledBlockError('Function instruction range outside image')
        if nlocals < 0 or stacksize < 0:
            raise CompiledBlockError('Negative function capacity')
        for pc in range(start, start + n):
            opi = int(instructions[pc, 0])
            target = int(instructions[pc, 3])
            if not 0 <= opi < nops:
                raise CompiledBlockError('Opcode index outside opcode_names')
            if target != -1 and not start <= target < start + n:
                raise CompiledBlockError('Jump outside function')


def _hot_functions(image, entries, max_functions):
    if entries is None:
        names = tuple(sorted(image.entries))
    else:
        if isinstance(entries, (str, bytes)) or not hasattr(entries, '__iter__'):
            raise TypeError('entries must be an iterable of entry names')
        names = tuple(entries)
        for name in names:
            if type(name) is not str:
                raise TypeError('VM entry name required')
            if name not in image.entries:
                raise CompiledBlockError('Unknown VM entry')
    selected = []
    seen = set()
    for name in names:
        idx = int(image.entries[name])
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= max_functions:
            return tuple(selected)
    for idx in range(len(image.functions)):
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= max_functions:
            break
    return tuple(selected)


def _jump_targets(image, start, n):
    targets = set()
    for pc in range(start, start + n):
        target = int(image.instructions[pc, 3])
        if target == -1:
            continue
        if not start <= target < start + n:
            raise CompiledBlockError('Jump outside function')
        targets.add(target)
    return targets


def _fence_reason(op, target):
    if target != -1 or op in _JUMP_OPS:
        return 'jump'
    if op in _CALL_OPS:
        return 'call'
    if op in _RETURN_OPS:
        return 'return'
    if op in _GENERATOR_OPS:
        return 'generator'
    if op in _CONTEXT_OPS:
        return 'context'
    if op in _EXCEPTION_OPS:
        return 'exception'
    if op in _GETTER_OPS:
        return 'getter'
    if op in _FRAME_OPS:
        return 'frame'
    if op in SAFE_OPS:
        return None
    return 'unsupported'


def _c_int(value):
    value = int(value)
    if value <= -2147483648:
        return '(-2147483647-1)'
    if value >= 2147483647:
        return '2147483647'
    return str(value)


def _c_comment(text):
    out = []
    for ch in str(text)[:96]:
        out.append(ch if ch.isalnum() or ch in '._:+-' else '_')
    return ''.join(out) or 'block'


def _identity(source, coverage):
    payload = dict(helper=HELPER_NAME, coverage=_jsonable(coverage), source=source)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return sha256(encoded).hexdigest()


def _jsonable(value):
    if isinstance(value, MappingProxyType):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _emit_source(blocks, max_source_chars):
    parts = [
        '/* Frozen IR basic-block helper. Official frozen rules stay sole truth.\n',
        '   Integration: in duel_rule_vm::execute, after raised/error handling and\n',
        '   before step increment / instruction fetch:\n',
        '     if(', HELPER_NAME, '(c)) continue;\n',
        '   return 0 leaves Context unchanged; return 1 consumed a compiled block.\n',
        '   Uncompiled PCs remain on the existing GPU VM path. No CPU fallback. */\n',
    ]
    if not blocks:
        parts.extend((
            '__device__ inline int ', HELPER_NAME, '(Context* c){ (void)c; return 0; }\n',
        ))
        source = ''.join(parts)
        if len(source) > max_source_chars:
            raise CompiledBlockError('Compiled CUDA source exceeds code budget')
        return source
    parts.extend((
        '__device__ inline int ', BEGIN_OP_NAME, '(Context* c,int* f,int next_pc){\n',
        ' if(++c->steps>c->step_limit){c->error=STEP_LIMIT;return 0;}\n',
        ' if(c->deadline && (c->steps&63)==0 && clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;return 0;}\n',
        ' c->error_value=MISSING;c->error_kind=0;f[1]=next_pc;return 1;\n',
        '}\n',
        '__device__ inline int ', HELPER_NAME, '(Context* c){\n',
        ' if(!c||c->error||c->h.error)return 0;\n',
        ' int id=c->current;if(id<0||id>=VM_FRAMES)return 0;\n',
        ' int* f=frame(c,id);int fid=f[0];\n',
        ' if(fid<0||fid>=c->function_count)return 0;\n',
        ' const int* fn=c->functions+fid*18;int pc=f[1];\n',
        ' if(pc<fn[0]||pc>=fn[0]+fn[1])return 0;\n',
        ' switch(pc){\n',
    ))
    for block in blocks:
        parts.extend(_emit_block(block))
    parts.extend((
        '  default:return 0;\n',
        ' }\n',
        '}\n',
    ))
    source = ''.join(parts)
    if len(source) > max_source_chars * 4:
        raise CompiledBlockError('Compiled CUDA source exceeds code budget')
    return source


def _emit_block(block):
    nlocals = _c_int(block['nlocals'])
    lines = [
        '  case ', _c_int(block['pc']), ':{ /* ', _c_comment(block['function_id']),
        ' pc ', _c_int(block['pc']), '..', _c_int(block['end_pc']), ' */\n',
    ]
    for inst in block['instructions']:
        next_pc = inst['pc'] + 1
        lines.extend((
            '   if(!', BEGIN_OP_NAME, '(c,f,', _c_int(next_pc), '))return 1;\n',
        ))
        lines.extend(_emit_op(inst, nlocals))
        lines.append('   if(c->error||c->h.error)return 1;\n')
    lines.append('   return 1;}\n')
    return lines


def _emit_op(inst, nlocals):
    op = inst['op']
    arg = _c_int(inst['arg'])
    operand = _c_int(inst['operand'])
    if op in ('NOP', 'EXTENDED_ARG'):
        return ('   /* ', op, ' */\n',)
    if op == 'LOAD_CONST':
        return ('   push(c,f,', operand, '); /* LOAD_CONST */\n',)
    if op == 'LOAD_FAST':
        return (
            '   if(', arg, '<0||', arg, '>=', nlocals, '){c->error=BAD_PROGRAM;return 1;}\n',
            '   else if(f[LBASE+', arg, ']==MISSING){c->error=UNBOUND_LOCAL;return 1;}\n',
            '   else push(c,f,f[LBASE+', arg, ']);\n',
        )
    if op == 'STORE_FAST':
        return (
            '   if(', arg, '<0||', arg, '>=', nlocals, ')c->error=BAD_PROGRAM;\n',
            '   else f[LBASE+', arg, ']=pop(c,f);\n',
        )
    if op == 'DELETE_FAST':
        return (
            '   if(', arg, '<0||', arg, '>=', nlocals, ')c->error=BAD_PROGRAM;\n',
            '   else f[LBASE+', arg, ']=MISSING;\n',
        )
    if op == 'POP_TOP':
        return ('   pop(c,f);\n',)
    if op == 'DUP_TOP':
        return ('   push(c,f,peek(c,f));\n',)
    if op == 'DUP_TOP_TWO':
        return (
            '   if(f[2]<2){c->error=BAD_PROGRAM;return 1;}\n',
            '   if(f[2]+2>VM_STACK){c->error=STACK_OVERFLOW;return 1;}\n',
            '   {int a=f[SBASE+f[2]-2],b=f[SBASE+f[2]-1];push(c,f,a);push(c,f,b);}\n',
        )
    if op == 'ROT_TWO':
        return ('   {int a=pop(c,f),b=pop(c,f);push(c,f,a);push(c,f,b);}\n',)
    if op == 'ROT_THREE':
        return (
            '   if(f[2]<3){c->error=BAD_PROGRAM;return 1;}\n',
            '   {int top=f[SBASE+f[2]-1],second=f[SBASE+f[2]-2],third=f[SBASE+f[2]-3];\n',
            '    f[SBASE+f[2]-1]=second;f[SBASE+f[2]-2]=third;f[SBASE+f[2]-3]=top;}\n',
        )
    if op in ('ROT_FOUR', 'ROT_N'):
        count = '4' if op == 'ROT_FOUR' else arg
        return (
            '   {int count=', count, ';\n',
            '    if(count<1||count>f[2]){c->error=BAD_PROGRAM;return 1;}\n',
            '    int top=f[SBASE+f[2]-1];\n',
            '    for(int i=f[2]-1;i>f[2]-count;i--)f[SBASE+i]=f[SBASE+i-1];\n',
            '    f[SBASE+f[2]-count]=top;}\n',
        )
    if op in ('BUILD_LIST', 'BUILD_TUPLE'):
        if op == 'BUILD_TUPLE':
            build = 'int ref=make_tuple(&c->h,f+SBASE+f[2]-' + arg + ',' + arg + ');'
        else:
            build = (
                'int ref=make_list(&c->h);'
                'for(int i=f[2]-' + arg + ';i<f[2];i++)list_append(&c->h,ref,f[SBASE+i]);'
            )
        return (
            '   if(', arg, '<0||', arg, '>f[2]){c->error=BAD_PROGRAM;return 1;}\n',
            '   {', build, 'f[2]-=', arg, ';push(c,f,ref);}\n',
        )
    if op == 'UNARY_NOT':
        return ('   {int yes=truth_value(c,pop(c,f));push(c,f,yes?FALSE_REF:TRUE_REF);}\n',)
    if op == 'UNARY_NEGATIVE':
        return (
            '   {int ref=pop(c,f);if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);push(c,f,make_float64(&c->h,-a));}\n',
            '    else{long long a=0;numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else push(c,f,make_int64(&c->h,-a));}}\n',
        )
    if op == 'IS_OP':
        return (
            '   {int b=pop(c,f),a=pop(c,f);int ta=kind(c,a),tb=kind(c,b);int yes=a==b;\n',
            '    if(ta==TAG_NONE&&tb==TAG_NONE)yes=1;else if(ta==TAG_BOOL&&tb==TAG_BOOL)yes=node(c,a)[3]==node(c,b)[3];\n',
            '    push(c,f,(', arg, '?!yes:yes)?TRUE_REF:FALSE_REF);}\n',
        )
    if op == 'COMPARE_OP':
        return (
            '   {int b=pop(c,f),a=pop(c,f);push(c,f,compare(c,a,b,', arg, ')?TRUE_REF:FALSE_REF);}\n',
        )
    if op in SAFE_PRIMITIVE_OPS:
        return (
            '   {int b=pop(c,f),a=pop(c,f);int stdlib_out=MISSING;\n',
            '    if(stdlib_binary(c,a,b,OP_', op, ',&stdlib_out))push(c,f,stdlib_out);\n',
            '    else push(c,f,numeric_binary(c,a,b,OP_', op, '));}\n',
        )
    raise CompiledBlockError('Refusing to emit unknown opcode ' + op)

