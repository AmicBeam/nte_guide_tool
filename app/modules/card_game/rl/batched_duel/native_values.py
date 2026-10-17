"""Lift bounded straight-line rule regions into native reference variables.

Calls/control flow remain barriers. Every logical interruption restores the
original frame stack/locals before returning to canonical error handling. Value
references retain their original boxing/identity; this pass does not speculate
about scalar types, skip allocations or replace rule formulas.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from .native_array_codegen import _c_int, native_array_plan
from .native_functions import _admission_guards
from .vm_numeric import NUMERIC_OPS


SUPPORTED = frozenset(('LOAD_FAST', 'LOAD_CONST', 'STORE_FAST', 'POP_TOP',
                       'DUP_TOP', 'DUP_TOP_TWO', 'ROT_TWO', 'BINARY_SUBSCR',
                       'STORE_SUBSCR', 'COMPARE_OP', 'CONTAINS_OP', 'IS_OP',
                       'UNARY_NOT', 'UNARY_NEGATIVE', 'BINARY_ADD',
                       'BINARY_SUBTRACT', 'BINARY_MULTIPLY')) | NUMERIC_OPS
_BINARY = NUMERIC_OPS | frozenset(('BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY'))
_HOT_FUNCTIONS = ('legal_actions', 'attack', 'damage', 'attack_value',
                  'apply_owned_action', 'scratch_apply_action', 'scratch_simulate_action',
                  'policy_public_legal_decision', 'cached_policy_public_legal_decision')


@dataclass(frozen=True)
class NativeValues:
    source: str
    regions: tuple
    constant_reads: int
    identity: str

    def verify(self):
        if self.identity != _identity(self.source, self.regions, self.constant_reads):
            raise ValueError('Native value lowering identity changed')
        return True


def _identity(*values):
    return sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def _effect(op):
    if op in _BINARY or op in ('COMPARE_OP', 'CONTAINS_OP', 'IS_OP'):
        return 2, -1
    if op in ('UNARY_NOT', 'UNARY_NEGATIVE'):
        return 1, 0
    if op == 'STORE_SUBSCR':
        return 3, -3
    return {'LOAD_FAST': (0, 1), 'LOAD_CONST': (0, 1), 'STORE_FAST': (1, -1),
            'POP_TOP': (1, -1), 'DUP_TOP': (1, 1), 'DUP_TOP_TWO': (2, 2),
            'ROT_TWO': (2, 0), 'BINARY_SUBSCR': (2, -1)}[op]


def _shape(rows):
    depth, need, peak = 0, 0, 0
    for row in rows:
        required, delta = _effect(row['op'])
        need = max(need, required - depth)
        depth += delta
        peak = max(peak, depth)
    return need, peak


def _constant_keys(image, schema):
    fields = {f.name.split('.', 1)[1] for p in schema.pools for f in p.fields}
    result = {}
    for index, (tag, offset, count, _) in enumerate(image.nodes):
        if int(tag) == 4:
            name = image.payload[int(offset):int(offset + count)].tobytes().decode('utf8')
            if name in fields:
                result[-1 - index] = name
    return result


def _read_keys(rows, keys):
    need, _ = _shape(rows)
    stack, locals_, result = [None] * need, {}, []
    for row in rows:
        op = row['op']
        if op == 'LOAD_CONST':
            stack.append(row['operand'] if row['operand'] in keys else None)
        elif op == 'LOAD_FAST':
            stack.append(locals_.get(row['arg']))
        elif op == 'STORE_FAST':
            locals_[row['arg']] = stack.pop()
        elif op == 'POP_TOP':
            stack.pop()
        elif op == 'DUP_TOP':
            stack.append(stack[-1])
        elif op == 'DUP_TOP_TWO':
            stack.extend(stack[-2:])
        elif op == 'ROT_TWO':
            stack[-2:] = stack[-2:][::-1]
        elif op in ('UNARY_NOT', 'UNARY_NEGATIVE'):
            stack[-1] = None
        elif op == 'STORE_SUBSCR':
            del stack[-3:]
        else:
            key = stack.pop()
            stack.pop()
            if row['op'] == 'BINARY_SUBSCR' and key is not None:
                result.append(key)
            stack.append(None)
    return result


def compile_native_values(image, schema, plan, *, max_regions=256, max_region_ops=16):
    """Select complete regions with bounded code growth, favoring rule hot paths.

    A region never crosses a function, branch target, call or unsupported op.
    All ordinary PC cases remain available; this is not action pruning.
    """
    image.verify()
    plan.verify()
    if (type(max_regions) is not int or not 1 <= max_regions <= 1024 or
            type(max_region_ops) is not int or not 2 <= max_region_ops <= 32):
        raise ValueError('Explicit bounded native region budgets required')
    if native_array_plan(image).identity != plan.identity:
        raise ValueError('Native value image/plan mismatch')
    keys = _constant_keys(image, schema)
    rows = plan.coverage['instructions']
    groups = []
    for fn in plan.coverage['functions']:
        body = rows[fn['start']:fn['start'] + fn['count']]
        targets = {r['target'] for r in body if r['target'] >= 0}
        candidates, run = [], []
        def finish():
            if len(run) >= 2 and (_read_keys(run, keys) or
                    len(run) >= 4 and any(r['op'] == 'STORE_FAST' for r in run)):
                candidates.append(tuple(run))
            run.clear()
        for row in body:
            if row['pc'] in targets or len(run) == max_region_ops:
                finish()
            invalid_local = (row['op'] in ('LOAD_FAST', 'STORE_FAST') and
                             not 0 <= row['arg'] < int(image.functions[fn['index'], 5]))
            if row['op'] not in SUPPORTED or row['target'] != -1 or invalid_local:
                finish()
            else:
                run.append(row)
        finish()
        if candidates:
            # Round-robin across functions so one large projection cannot
            # exhaust the code budget before legal actions/state transitions.
            candidates.sort(key=lambda run: (-len(_read_keys(run, keys)), run[0]['pc']))
            name = fn['function_id']
            priority = (0 if any(':' + hot + ':' in name for hot in _HOT_FUNCTIONS) else
                        1 if any('.engine.duel_v2.' + module + ':' in name
                                 for module in ('flow', 'state', 'combat')) else 2)
            groups.append((priority, fn, candidates))
    groups.sort(key=lambda item: (item[0], item[1]['index']))
    selected = []
    for priority in sorted({g[0] for g in groups}):
        bucket = [g for g in groups if g[0] == priority]
        while len(selected) < max_regions and any(g[2] for g in bucket):
            for _, fn, candidates in bucket:
                if candidates and len(selected) < max_regions:
                    selected.append((fn, candidates.pop(0)))
    selected.sort(key=lambda item: item[1][0]['pc'])
    used_keys = sorted({key for _, run in selected for key in _read_keys(run, keys)})
    from .native_scalar_ops import emit_native_scalar_source
    source = emit_native_scalar_source()
    source += '''__device__ inline int native_value_subscript(Context* c,int owner,int key){
 int out=MISSING;if(stdlib_subscript(c,owner,key,&out))return out;
 return collection_subscript(c,owner,key);
}
__device__ inline void native_value_store(Context* c,int owner,int key,int value){
 if(stdlib_store(c,owner,key,value))return;
 collection_store(c,owner,key,value);
}
'''
    source += ''.join(_field_helper(key, keys[key], schema) for key in used_keys)
    source += ''.join(_emit_region(fn, run, keys) for fn, run in selected)
    regions = tuple((fn['index'], fn['function_id'], run[0]['pc'], run[-1]['pc'] + 1)
                    for fn, run in selected)
    reads = sum(len(_read_keys(run, keys)) for _, run in selected)
    result = NativeValues(source, regions, reads, _identity(source, regions, reads))
    result.verify()
    return result


def _field_helper(key, name, schema):
    """Bind the constant key to exact pool/column offsets, retaining boxing."""
    text = ('__device__ inline int native_field_' + str(-key) + '(Context* c,int owner){\n'
            ' if(soa_is_view(&c->h,owner)){\n'
            '  int pool=0,slot=0;\n'
            '  c->error_value=' + _c_int(key) + ';c->error_kind=1;\n'
            '  if(!soa_refresh_header(&c->h,owner))return MISSING;\n'
            '  if(!soa_locate(c->h.nodes[owner*4+3],&pool,&slot))return soa_fail(&c->h,ERROR_INVALID_REF);\n'
            '  switch(pool){\n')
    col = 0
    for pool in schema.pools:
        for field in pool.fields:
            if field.name.split('.', 1)[1] == name:
                text += ('   case ' + str(pool.index) + ':\n'
                         '    if(soa_load_u8(c->h.soa,' + str(field.tag) + '+slot)!=SOA_ABSENT)'
                         'return soa_box_cell(&c->h,' + str(col) + ',slot);\n    break;\n')
            col += 1
        # SOA column tables also include tuple slots and list/map columns.
        col += len(pool.values) + (1 if pool.keys is not None else 0)
    return text + ('   default:break;\n  }\n }\n'
                   ' return native_value_subscript(c,owner,' + _c_int(key) + ');\n}\n')


def _emit_region(fn, rows, keys):
    need, peak = _shape(rows)
    first = rows[0]['pc']
    locals_ = sorted({r['arg'] for r in rows if r['op'] in ('LOAD_FAST', 'STORE_FAST')})
    text = ('__device__ __noinline__ int native_values_' + str(first) +
            '(Context* c,int* f,const int* fn,int pc){\n'
            ' if(f[0]!=' + str(fn['index']) + '||f[2]<' + str(need) +
            '||f[2]>VM_STACK-' + str(peak) + ')return 0;\n')
    for index in locals_:
        text += ' if(' + str(index) + '<0||' + str(index) + '>=fn[5])return 0;\n'
    text += ' const int owner=c->current,base=f[2]-' + str(need) + ';\n'
    for index in locals_:
        text += ' int local_' + str(index) + '=f[LBASE+' + str(index) + '];\n'
    stack = [('input_' + str(i), None) for i in range(need)]
    for i, (value, _) in enumerate(stack):
        text += ' int ' + value + '=f[SBASE+base+' + str(i) + '];\n'
    known_locals, written_locals = {}, set()
    def flush():
        lines = ['f[2]=base+' + str(len(stack)) + ';\n']
        for i, (value, _) in enumerate(stack):
            lines.append('f[SBASE+base+' + str(i) + ']=' + value + ';\n')
        for index in sorted(written_locals):
            lines.append('f[LBASE+' + str(index) + ']=local_' + str(index) + ';\n')
        return ''.join(lines)
    guards = _admission_guards()
    boundary = 'f=frame(c,c->current);fn=function(c,f[0]);'
    if guards.count(boundary) != 1:
        raise ValueError('Canonical native value admission boundary changed')
    # Proven straight-line operations cannot change the frame, function or PC.
    # Keep exact logical budget/deadline/error guards; do not reload dynamic
    # frame/PC metadata between these compile-time edges.
    guards = guards[:guards.index(boundary)].replace(
        'return;', '{NATIVE_VALUE_FLUSHreturn 1;}')
    for offset, row in enumerate(rows):
        if offset:
            text += guards.replace('NATIVE_VALUE_FLUSH', flush()) + ' f[1]=' + str(row['pc'] + 1) + ';\n'
        op, index = row['op'], row['arg']
        value = 'value_' + str(offset)
        if op == 'LOAD_FAST':
            text += (' if(local_' + str(index) + '==MISSING){c->error=UNBOUND_LOCAL;' + flush() + 'return 1;}\n'
                     ' int ' + value + '=local_' + str(index) + ';\n')
            stack.append((value, known_locals.get(index)))
        elif op == 'LOAD_CONST':
            text += ' int ' + value + '=' + _c_int(row['operand']) + ';\n'
            stack.append((value, row['operand'] if row['operand'] in keys else None))
        elif op == 'STORE_FAST':
            ref, key = stack.pop()
            text += ' local_' + str(index) + '=' + ref + ';\n'
            known_locals[index] = key
            written_locals.add(index)
        elif op == 'POP_TOP':
            stack.pop()
        elif op == 'DUP_TOP':
            stack.append(stack[-1])
        elif op == 'DUP_TOP_TWO':
            stack.extend(stack[-2:])
        elif op == 'ROT_TWO':
            stack[-2:] = stack[-2:][::-1]
        elif op == 'STORE_SUBSCR':
            key_ref, _ = stack.pop()
            owner_ref, _ = stack.pop()
            stored_ref, _ = stack.pop()
            text += ' native_value_store(c,' + ','.join((owner_ref, key_ref, stored_ref)) + ');\n'
        elif op in _BINARY:
            right, _ = stack.pop()
            left, _ = stack.pop()
            text += ' int ' + value + '=native_scalar_binary<OP_' + op + '>(c,' + left + ',' + right + ');\n'
            stack.append((value, None))
        elif op == 'COMPARE_OP':
            right, _ = stack.pop()
            left, _ = stack.pop()
            text += (' int ' + value + '=native_scalar_compare<' + str(row['arg']) +
                     '>(c,' + left + ',' + right + ')?TRUE_REF:FALSE_REF;\n')
            stack.append((value, None))
        elif op == 'CONTAINS_OP':
            owner_ref, _ = stack.pop()
            needle, _ = stack.pop()
            text += ' int ' + value + '=contains(c,' + needle + ',' + owner_ref + ');\n'
            text += value + '=' + ('!' if row['arg'] else '') + value + '?TRUE_REF:FALSE_REF;\n'
            stack.append((value, None))
        elif op == 'IS_OP':
            right, _ = stack.pop()
            left, _ = stack.pop()
            text += (' int tag_a_' + str(offset) + '=kind(c,' + left + '),tag_b_' + str(offset) + '=kind(c,' + right + ');\n'
                     ' int yes_' + str(offset) + '=' + left + '==' + right + ';\n'
                     ' if(tag_a_' + str(offset) + '==TAG_NONE&&tag_b_' + str(offset) + '==TAG_NONE)yes_' + str(offset) + '=1;\n'
                     ' else if(tag_a_' + str(offset) + '==TAG_BOOL&&tag_b_' + str(offset) + '==TAG_BOOL)yes_' + str(offset) + '=node(c,' + left + ')[3]==node(c,' + right + ')[3];\n'
                     ' int ' + value + '=' + ('!' if row['arg'] else '') + 'yes_' + str(offset) + '?TRUE_REF:FALSE_REF;\n')
            stack.append((value, None))
        elif op == 'UNARY_NOT':
            operand, _ = stack.pop()
            text += ' int ' + value + '=native_scalar_not(c,' + operand + ')?TRUE_REF:FALSE_REF;\n'
            stack.append((value, None))
        elif op == 'UNARY_NEGATIVE':
            operand, _ = stack.pop()
            text += ' int ' + value + '=native_scalar_negative(c,' + operand + ');\n'
            text += ' if(c->h.error==ERROR_INT_OVERFLOW){' + flush() + 'return 1;}\n'
            stack.append((value, None))
        elif op == 'BINARY_SUBSCR':
            key_ref, key = stack.pop()
            owner_ref, _ = stack.pop()
            read = ('native_field_' + str(-key) + '(c,' + owner_ref + ')' if key is not None else
                    'native_value_subscript(c,' + owner_ref + ',' + key_ref + ')')
            text += ' int ' + value + '=' + read + ';\n'
            stack.append((value, None))
        if op not in ('LOAD_CONST', 'LOAD_FAST', 'STORE_FAST', 'POP_TOP', 'DUP_TOP', 'DUP_TOP_TWO', 'ROT_TWO'):
            text += (' if(c->error||c->h.error||c->raised!=MISSING||c->current!=owner){' + flush() + 'return 1;}\n')
    return text + flush() + ' return 1;\n}\n'


def integrate_native_values(source, image, schema, plan):
    from .native_array_codegen import _emit_pc_case
    from .native_shard_runs import _continued_case
    compiled = compile_native_values(image, schema, plan)
    complete = '__device__ inline void native_pc_shard_0(' not in source
    marker = ('__device__ __noinline__ void native_function_' if complete else
              '__device__ inline void native_pc_shard_0(')
    if not source.count(marker) or not complete and source.count(marker) != 1:
        raise ValueError('Canonical native value definition boundary changed')
    source = source.replace(marker, compiled.source + '\n' + marker, 1)
    for _, _, begin, _ in compiled.regions:
        if complete:
            label = 'op_' + str(begin) + ':{\n'
            if source.count(label) != 1:
                raise ValueError('Canonical native function value boundary changed: ' + str(begin))
            hook = ' if(native_values_' + str(begin) + '(c,f,fn,pc))return;\n'
            source = source.replace(label, label + hook, 1)
            continue
        row = plan.coverage['instructions'][begin]
        case = _continued_case(''.join(_emit_pc_case(row, direct_stack_dispatch=True)))
        if source.count(case) != 1:
            raise ValueError('Canonical native value PC boundary changed: ' + str(begin))
        label = '  case ' + str(begin) + ':'
        hook = 'if(native_values_' + str(begin) + '(c,f,fn,pc))return;'
        source = source.replace(case, case.replace(label, label + hook, 1), 1)
    return source, compiled
