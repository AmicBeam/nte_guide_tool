"""Keep canonical PC operations in a shard until a semantic boundary.

The first instruction is admitted by execute(). Every subsequent instruction
uses the original step/deadline, error-context and frame/PC guards. Errors,
exceptions, frame changes and out-of-shard jumps return to execute() before
another logical instruction is admitted. No opcode body or formula is changed.
"""
import re
from .native_array_codegen import SHARD_SIZE, _EXECUTE_PREAMBLE, _emit_pc_case


def _continued_case(case):
    if case.endswith('return;}\n'):
        result = case[:-len('return;}\n')] + 'break;}\n'
    elif case.endswith('return;\n'):
        result = case[:-len('return;\n')] + 'break;\n'
    else:
        raise ValueError('Canonical PC case exit changed')
    # A primitive handler's early exit has exactly the same switch scope as
    # the final case exit. Never rewrite a return in an unknown nested body.
    result = re.sub(r'if\(native_handlers<OP_[A-Z0-9_]+>\(c,f,[^;\n]*\)\)return;',
                    lambda match: match.group().replace('return;', 'break;'), result)
    if 'return;' in result:
        raise ValueError('Unknown return inside a native PC case')
    return result


def run_native_shards(source, plan):
    plan.verify()
    marker = '  if(++c->steps>c->step_limit)'
    if _EXECUTE_PREAMBLE.count(marker) != 1:
        raise ValueError('Canonical shard step guard changed')
    guards = _EXECUTE_PREAMBLE[_EXECUTE_PREAMBLE.index(marker):]
    if 'continue;' in guards or guards.count('break;') != 4:
        raise ValueError('Canonical shard admission guards changed')
    guards = guards.replace('break;', 'return;')
    before = 'int* f=frame(c,c->current);const int* fn=function(c,f[0]);'
    if guards.count(before) != 1 or guards.count('int pc=f[1];') != 1:
        raise ValueError('Canonical shard frame/PC guards changed')
    guards = guards.replace(before, 'f=frame(c,c->current);fn=function(c,f[0]);')
    guards = guards.replace('int pc=f[1];', 'pc=f[1];')
    rows = plan.coverage['instructions']
    for shard in range(plan.coverage['shard_count']):
        begin = shard * SHARD_SIZE
        end = min(len(rows), begin + SHARD_SIZE)
        pc_cases = [''.join(_emit_pc_case(row, direct_stack_dispatch=True)) for row in rows[begin:end]]
        cases = ''.join(pc_cases)
        header = ('__device__ inline void native_pc_shard_' + str(shard) +
                  '(Context* c,int* f,const int* fn,int pc){\n')
        old = (header + ' switch(pc){\n' + cases +
               '  default:c->error=BAD_PROGRAM;return;\n }\n}\n')
        if source.count(old) != 1:
            raise ValueError('Canonical native shard body changed: ' + str(shard))
        continued = ''.join(_continued_case(case) for case in pc_cases)
        new = (header +
               ' const int owner=c->current, fid=f[0];\n'
               ' for(;;){ /* native shard continuation */\n'
               '  switch(pc){\n' + continued +
               '   default:c->error=BAD_PROGRAM;return;\n  }\n'
               '  if(c->error||c->h.error||c->raised!=MISSING)return;\n'
               '  if(c->current!=owner||f[0]!=fid)return;\n'
               '  int next_pc=f[1];\n'
               '  if(next_pc<' + str(begin) + '||next_pc>=' + str(end) + ')return;\n' +
               guards + '  f[1]++;\n }\n}\n')
        source = source.replace(old, new, 1)
    return source
