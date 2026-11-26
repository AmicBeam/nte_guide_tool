"""Static native-array codegen for a frozen VM image.

Every frozen instruction becomes a compile-time PC case. Official frozen
rules remain the sole semantic truth. This module does not invent game
logic, run a CPU VM, claim a measured speedup, or introduce a domain
SoA layout. Existing C++ heap/frame/call/exception/container/intrinsic
helpers keep their original semantics; they are not CPU engine callbacks.

Primary owns CUDA/CPU compilation of the SAME generated source. This file
only emits it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import re
from types import MappingProxyType

from .cuda_vm import VmLimits, _VM_SOURCE, emit_vm_source, SUPPORTED_OPS
from .vm_image import VmImage

HELPER_NAME = 'native_dispatch_pc'
OPCODE_HELPER = 'native_opcode'
STEP_HELPER = 'native_step'
HANDLER_HELPER = 'native_handlers'
SHARD_SIZE = 128
_FETCH_LINE = (
    'const int* ins=c->instructions+pc*4;'
    'int op=ins[0],arg=ins[1],value=ins[2],target=ins[3];f[1]++;'
)
_EXECUTE_DECL = '__device__ inline void execute(Context* c){'
_EXECUTE_FIND = re.compile(
    r'__device__\s+(?:inline|__noinline__)\s+void execute\(Context\*\s*c\)\{'
)
_OP_TOKEN = re.compile(r'^[A-Z_][A-Z0-9_]*$')
_CASE_OP = re.compile(r'case OP_([A-Z0-9_]+)')
_STACK_OPS = frozenset((
    'NOP', 'EXTENDED_ARG',
    'LOAD_CONST', 'LOAD_FAST', 'STORE_FAST', 'DELETE_FAST',
    'POP_TOP', 'DUP_TOP', 'ROT_TWO',
))
_DIRECT_STACK_OPS = ('LOAD_CONST', 'LOAD_FAST', 'STORE_FAST', 'POP_TOP')


class NativeArrayCodegenError(ValueError):
    """Malformed frozen IR or a canonical execute body that can no longer be reused."""


@dataclass(frozen=True)
class NativeArrayPlan:
    coverage: object
    identity: str
    interpreter_fallback: bool = False

    def verify(self):
        if self.interpreter_fallback or self.coverage['interpreter_fallback']:
            raise NativeArrayCodegenError('Interpreter fallback is not allowed')
        if self.identity != _identity(self.coverage):
            raise NativeArrayCodegenError('Native array plan identity changed')
        if self.coverage['compiled_op_count'] != self.coverage['instruction_count']:
            raise NativeArrayCodegenError('Native array instruction coverage is incomplete')
        if self.coverage['compiled_function_count'] != self.coverage['function_count']:
            raise NativeArrayCodegenError('Native array function coverage is incomplete')
        return True


def native_array_plan(image):
    """Metadata for a complete static native lowering of `image`."""
    instructions, functions, names = _validate_image(image)
    ninst = int(instructions.shape[0])
    nfunc = int(functions.shape[0])
    owners = _function_owners(functions, ninst)
    rows = []
    ops_used = []
    for pc in range(ninst):
        opi = int(instructions[pc, 0])
        op = names[opi]
        if not _OP_TOKEN.match(op):
            raise NativeArrayCodegenError('Opcode name is not a C identifier')
        ops_used.append(op)
        rows.append(MappingProxyType({
            'pc': pc,
            'op': op,
            'arg': int(instructions[pc, 1]),
            'operand': int(instructions[pc, 2]),
            'target': int(instructions[pc, 3]),
            'function_index': owners[pc],
        }))
    func_rows = []
    for fid in range(nfunc):
        start = int(functions[fid, 0])
        count = int(functions[fid, 1])
        func_rows.append(MappingProxyType({
            'index': fid,
            'function_id': image.function_ids[fid],
            'start': start,
            'count': count,
            'pcs': tuple(range(start, start + count)),
        }))
    switch_ops = tuple(sorted(_SWITCH_OPCODES))
    unknown = tuple(sorted(set(ops_used) - _SWITCH_OPCODES))
    shard_count = (ninst + SHARD_SIZE - 1) // SHARD_SIZE if ninst else 0
    coverage = MappingProxyType({
        'helper': HELPER_NAME,
        'opcode_helper': OPCODE_HELPER,
        'step_helper': STEP_HELPER,
        'handler_helper': HANDLER_HELPER,
        'instruction_count': ninst,
        'function_count': nfunc,
        'compiled_op_count': ninst,
        'compiled_function_count': nfunc,
        'compiled_pc': tuple(range(ninst)),
        'compiled_function_indexes': tuple(range(nfunc)),
        'compiled_function_ids': tuple(image.function_ids),
        'instructions': tuple(rows),
        'functions': tuple(func_rows),
        'opcodes': tuple(sorted(set(ops_used))),
        'switch_opcodes': switch_ops,
        'helper_opcodes': unknown,
        'unknown_opcodes': (),
        'interpreter_fallback': False,
        'cpu_fallback': False,
        'measured_speedup': False,
        'soa_layout': False,
        'shard_size': SHARD_SIZE,
        'shard_count': shard_count,
        'stack_ops': tuple(sorted(_STACK_OPS)),
    })
    plan = NativeArrayPlan(coverage, _identity(coverage), False)
    plan.verify()
    return plan


def emit_native_array_source(image, limits=VmLimits(), **options):
    """Emit CUDA source with the vm_execute ABI and a fully static execute()."""
    bound = inspect.signature(emit_vm_source).bind(image, limits, **options)
    bound.apply_defaults()
    arguments = bound.arguments
    compiled_blocks = arguments['compiled_blocks']
    if compiled_blocks is not None:
        if not isinstance(compiled_blocks, dict):
            raise TypeError('Explicit compiled-block budget mapping required')
        from .vm_compiled_blocks import emit_compiled_blocks
        emit_compiled_blocks(image, **compiled_blocks)
    plan = native_array_plan(image)
    vm_options = {
        key: arguments[key]
        for key in arguments
        if key not in ('image', 'limits', 'compiled_blocks')
    }
    # Full static coverage; never keep the opcode-interpreter execute().
    source = emit_vm_source(image, arguments['limits'], compiled_blocks=None, **vm_options)
    helpers = _emit_helpers(plan, direct_stack_dispatch=arguments['direct_stack_dispatch'])
    return _replace_execute(source, helpers)


def _canonical_execute():
    start = _VM_SOURCE.find(_EXECUTE_DECL)
    if start < 0:
        raise NativeArrayCodegenError('Canonical execute declaration changed')
    end = _matching_brace_end(_VM_SOURCE, _VM_SOURCE.index('{', start))
    return _VM_SOURCE[start:end]


def _canonical_switch(execute):
    if execute.count(_FETCH_LINE) != 1:
        raise NativeArrayCodegenError('Canonical instruction fetch changed')
    if execute.count('switch(op){') != 1:
        raise NativeArrayCodegenError('Canonical opcode switch changed')
    idx = execute.index('switch(op){')
    return execute[idx:_matching_brace_end(execute, execute.index('{', idx))]


def _validate_image(image):
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    image.verify()
    instructions = image.instructions
    functions = image.functions
    if instructions.ndim != 2 or instructions.shape[1] != 4:
        raise NativeArrayCodegenError('Instruction table must be int32[N,4]')
    if functions.ndim != 2 or functions.shape[1] != 18:
        raise NativeArrayCodegenError('Function table must be int32[M,18]')
    ninst = int(instructions.shape[0])
    nops = len(image.opcode_names)
    if nops < 1 and ninst:
        raise NativeArrayCodegenError('Opcode names missing for instruction stream')
    if len(image.function_ids) != int(functions.shape[0]):
        raise NativeArrayCodegenError('Function id count does not match function table')
    for fid in range(int(functions.shape[0])):
        start = int(functions[fid, 0])
        n = int(functions[fid, 1])
        nlocals = int(functions[fid, 5])
        stacksize = int(functions[fid, 6])
        if start < 0 or n < 0 or start > ninst or start + n > ninst:
            raise NativeArrayCodegenError('Function instruction range outside image')
        if nlocals < 0 or stacksize < 0:
            raise NativeArrayCodegenError('Negative function capacity')
        for pc in range(start, start + n):
            opi = int(instructions[pc, 0])
            target = int(instructions[pc, 3])
            if not 0 <= opi < nops:
                raise NativeArrayCodegenError('Opcode index outside opcode_names')
            if image.opcode_names[opi] not in SUPPORTED_OPS:
                raise NativeArrayCodegenError('Unsupported frozen opcode: '+image.opcode_names[opi])
            if target != -1 and not start <= target < start + n:
                raise NativeArrayCodegenError('Jump outside function')
    return instructions, functions, image.opcode_names


def _function_owners(functions, ninst):
    owners = [-1] * ninst
    for fid in range(int(functions.shape[0])):
        start = int(functions[fid, 0])
        count = int(functions[fid, 1])
        for pc in range(start, start + count):
            if owners[pc] != -1:
                raise NativeArrayCodegenError('Overlapping function instruction range')
            owners[pc] = fid
    if any(owner < 0 for owner in owners):
        raise NativeArrayCodegenError('Instruction not owned by a function')
    return owners


def _identity(coverage):
    encoded = json.dumps(_jsonable(coverage), sort_keys=True, separators=(',', ':')).encode()
    return sha256(encoded).hexdigest()


def _jsonable(value):
    if isinstance(value, MappingProxyType):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _c_int(value):
    value = int(value)
    if value <= -2147483648:
        return '(-2147483647-1)'
    if value >= 2147483647:
        return '2147483647'
    return str(value)


def _matching_brace_end(source, open_index):
    if open_index < 0 or open_index >= len(source) or source[open_index] != '{':
        raise NativeArrayCodegenError('Expected a C++ block')
    depth = 0
    in_str = None
    escape = False
    i = open_index
    n = len(source)
    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == in_str:
                in_str = None
        elif ch == '"' or ch == "'":
            in_str = ch
        elif ch == '/' and i + 1 < n and source[i + 1] == '/':
            i = source.find('\n', i)
            if i < 0:
                break
        elif ch == '/' and i + 1 < n and source[i + 1] == '*':
            i = source.find('*/', i + 2)
            if i < 0:
                raise NativeArrayCodegenError('Unclosed C++ comment')
            i += 1
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise NativeArrayCodegenError('Unclosed C++ block')



_CANONICAL_EXECUTE = _canonical_execute()
_OPCODE_SWITCH = _canonical_switch(_CANONICAL_EXECUTE)
_SWITCH_OPCODES = frozenset(_CASE_OP.findall(_OPCODE_SWITCH))
_EXECUTE_PREAMBLE = _CANONICAL_EXECUTE.split(_FETCH_LINE, 1)[0]


def _emit_helpers(plan, *, direct_stack_dispatch):
    parts = [
        '/* Static native-array execute. Official frozen rules stay sole truth.\n',
        '   Every frozen PC is a compile-time case. execute() never fetches\n',
        '   c->instructions and never dispatches switch(op) on a runtime opcode.\n',
        '   Opcode bodies are the original execute() switch, with op a compile-time\n',
        '   constant. Stack ops are __device__ inline; complex ops call the same\n',
        '   specialization. No interpreter fallback, CPU engine callback, measured\n',
        '   speedup, or domain SoA claim. identity=',
        plan.identity,
        ' */\n',
        'template<int OP>\n',
        '__device__ inline void ', OPCODE_HELPER, '(Context* c,int* f,const int* fn,int arg,int value,int target){\n',
        ' (void)fn;const int op=OP;\n',
        _OPCODE_SWITCH,
        '\n}\n',
        'template<int OP>\n',
        '__device__ inline int ', HANDLER_HELPER, '(Context* c,int* f,int arg,int value,int target){\n',
    ]
    handler = (
        ' if(handle_strings(c,f,OP,arg))return 1;\n'
        ' if(handle_exceptions(c,f,OP,arg,value,target))return 1;\n'
        ' if(stdlib_handle_opcode(c,f,OP,arg,value,target))return 1;\n'
        ' if(handle_collections(c,f,OP,arg,value,target))return 1;\n'
        ' return 0;\n'
    )
    parts.append(handler)
    parts.extend((
        '}\n',
        'template<int OP>\n',
        '__device__ inline void ', STEP_HELPER, '(Context* c,int* f,const int* fn,int arg,int value,int target){\n',
        ' if(', HANDLER_HELPER, '<OP>(c,f,arg,value,target))return;\n',
        ' ', OPCODE_HELPER, '<OP>(c,f,fn,arg,value,target);\n',
        '}\n',
    ))
    ninst = plan.coverage['instruction_count']
    rows = plan.coverage['instructions']
    shard_count = plan.coverage['shard_count']
    for shard in range(shard_count):
        begin = shard * SHARD_SIZE
        end = ninst if begin + SHARD_SIZE > ninst else begin + SHARD_SIZE
        parts.extend((
            '__device__ inline void native_pc_shard_', str(shard),
            '(Context* c,int* f,const int* fn,int pc){\n',
            ' switch(pc){\n',
        ))
        for row in rows[begin:end]:
            parts.extend(_emit_pc_case(row, direct_stack_dispatch=direct_stack_dispatch))
        parts.extend((
            '  default:c->error=BAD_PROGRAM;return;\n',
            ' }\n',
            '}\n',
        ))
    parts.extend((
        '__device__ inline void ', HELPER_NAME, '(Context* c,int* f,const int* fn,int pc){\n',
    ))
    if shard_count == 0:
        parts.append(' (void)f;(void)fn;(void)pc;c->error=BAD_PROGRAM;\n}\n')
    else:
        shift = SHARD_SIZE.bit_length() - 1
        if 1 << shift != SHARD_SIZE:
            raise NativeArrayCodegenError('Shard size must be a power of two')
        parts.append(' switch((unsigned)pc>>' + str(shift) + '){\n')
        for shard in range(shard_count):
            parts.extend((
                '  case ', str(shard), ':native_pc_shard_', str(shard), '(c,f,fn,pc);return;\n',
            ))
        parts.extend((
            '  default:c->error=BAD_PROGRAM;return;\n',
            ' }\n',
            '}\n',
        ))
    return ''.join(parts)


def _emit_pc_case(row, *, direct_stack_dispatch):
    op = row['op']
    arg = _c_int(row['arg'])
    value = _c_int(row['operand'])
    target = _c_int(row['target'])
    pc = _c_int(row['pc'])
    if op not in _STACK_OPS:
        call = STEP_HELPER + '<OP_' + op + '>(c,f,fn,' + arg + ',' + value + ',' + target + ');return;'
        return ('  case ', pc, ':', call, '\n')
    body = _stack_body(op, arg, value)
    if body is None:
        raise NativeArrayCodegenError('Missing stack op inlining for ' + op)
    parts = ['  case ', pc, ':{ /* stack ', op, ' */ ']
    skip_handlers = direct_stack_dispatch and op in _DIRECT_STACK_OPS
    if not skip_handlers:
        parts.extend((
            'if(', HANDLER_HELPER, '<OP_', op, '>(c,f,', arg, ',', value, ',', target, '))return;',
        ))
    if body:
        parts.append(body)
    parts.append('return;}\n')
    return tuple(parts)


def _stack_body(op, arg, value):
    if op in ('NOP', 'EXTENDED_ARG'):
        return ''
    if op == 'LOAD_CONST':
        return 'push(c,f,' + value + ');'
    if op == 'LOAD_FAST':
        return (
            'if(' + arg + '<0||' + arg + '>=fn[5])c->error=BAD_PROGRAM;'
            'else if(f[LBASE+' + arg + ']==MISSING)c->error=UNBOUND_LOCAL;'
            'else push(c,f,f[LBASE+' + arg + ']);'
        )
    if op == 'STORE_FAST':
        return (
            'if(' + arg + '<0||' + arg + '>=fn[5])c->error=BAD_PROGRAM;'
            'else f[LBASE+' + arg + ']=pop(c,f);'
        )
    if op == 'DELETE_FAST':
        return (
            'if(' + arg + '<0||' + arg + '>=fn[5])c->error=BAD_PROGRAM;'
            'else f[LBASE+' + arg + ']=MISSING;'
        )
    if op == 'POP_TOP':
        return 'pop(c,f);'
    if op == 'DUP_TOP':
        return 'push(c,f,peek(c,f));'
    if op == 'ROT_TWO':
        return '{int a=pop(c,f),b=pop(c,f);push(c,f,a);push(c,f,b);}'
    return None


def _render_execute(qual):
    preamble = _EXECUTE_PREAMBLE.replace(
        _EXECUTE_DECL,
        '__device__ ' + qual + ' void execute(Context* c){',
        1,
    )
    return (
        preamble
        + 'f[1]++;\n'
        + '  ' + HELPER_NAME + '(c,f,fn,pc);\n'
        + ' }\n'
        + '}'
    )


def _replace_execute(source, helpers):
    matches = list(_EXECUTE_FIND.finditer(source))
    if len(matches) != 1:
        raise NativeArrayCodegenError('Canonical execute declaration changed')
    match = matches[0]
    qual = '__noinline__' if '__noinline__' in match.group(0) else 'inline'
    end = _matching_brace_end(source, source.index('{', match.start()))
    execute = _render_execute(qual)
    if _FETCH_LINE in execute or 'c->instructions' in execute:
        raise NativeArrayCodegenError('Static execute must not fetch instructions')
    if 'switch(op)' in execute:
        raise NativeArrayCodegenError('Static execute must not dispatch a runtime opcode')
    if execute.count('f[1]++') != 1:
        raise NativeArrayCodegenError('PC must advance exactly once per instruction')
    return source[:match.start()] + helpers + execute + source[end:]
