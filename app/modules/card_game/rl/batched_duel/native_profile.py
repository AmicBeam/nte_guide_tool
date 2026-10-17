"""Explicit diagnostic counters in the runtime unit only; rule ABI unchanged.

Times are execution-slice totals by frozen function and inclusive primitive
costs. Nested categories overlap and must not be summed as independent work.
This instrumentation is never enabled by the production source emitter.
Reset, execute and copy must be ordered on one stream; concurrent queries are
not supported by these diagnostic counters.
"""
from dataclasses import dataclass, replace
import ctypes
import re



@dataclass(frozen=True)
class NativeProfileLayout:
    functions: tuple
    opcodes: tuple
    helpers: tuple
    rows: int

    @property
    def stride(self):
        return len(self.functions)+len(self.opcodes)+len(self.helpers)

    @property
    def slots(self):
        return self.rows*self.stride


DEFAULT_HELPERS = ('call', 'new_frame', 'return_value', 'native_builtin_call',
                   'native_fast_call', 'builtin_attribute', 'object_attribute',
                   'numeric_binary', 'compare', 'contains', 'stdlib_call',
                   'stdlib_invoke', 'stdlib_resume', 'soa_clone', 'soa_encode',
                   'soa_write_cell', 'soa_write_encoded')


def instrument_runtime(unit, image, *, rows=2, helpers=DEFAULT_HELPERS):
    """Return one profiled runtime unit and its bounded counter layout.

    Call this after splitting: all rule/value object source identities remain
    unchanged. The profile source and linked binary have their own cache keys.
    """
    if unit.name != 'runtime' or type(rows) is not int or not 1 <= rows <= 16:
        raise ValueError('Runtime unit and one to sixteen rows required')
    if (type(helpers) is not tuple or len(set(helpers)) != len(helpers)
            or any(name not in DEFAULT_HELPERS for name in helpers)):
        raise ValueError('Unique supported profiling helpers required')
    layout = NativeProfileLayout(tuple(image.function_ids), tuple(image.opcode_names), helpers, rows)
    if not layout.functions or not layout.opcodes:
        raise ValueError('Frozen function and opcode identities required')
    source = unit.source
    scope = f'''
namespace native_rule_profile {{
__device__ unsigned long long elapsed[{layout.slots}];
__device__ unsigned long long calls[{layout.slots}];
__device__ int enabled;
struct Scope {{
 int slot;unsigned long long begin;
 __device__ inline Scope(int category,int index){{
  int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  slot=(enabled&category)&&index>=0&&index<{layout.stride}&&row<{rows}?row*{layout.stride}+index:-1;
  begin=0;if(slot>=0)asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
 }}
 __device__ inline ~Scope(){{if(slot>=0){{unsigned long long end;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
  elapsed[slot]+=end-begin;calls[slot]++;
 }}}}
}};
}}
'''
    # Define the diagnostic scope before every namespace/body it instruments.
    source = scope + source
    insertions = []
    def add(name, expression):
        pattern = re.compile(r'(?m)^__device__ (?:inline|__noinline__) (?:int|void) '+name+r'\([^;{}]*\)\s*\{')
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError('Unique profiling definition required: '+name)
        insertions.append((matches[0].end(), '\n native_rule_profile::Scope native_profile_scope('+expression+');\n'))
    add('native_dispatch_pc', '1,f[0]')
    # Some declared supported opcodes have negative sentinel IDs and cannot
    # occur in this frozen image. Do not index counters for those specializations.
    add('native_step', '2,(OP>=0&&OP<'+str(len(layout.opcodes))+'?'+str(len(layout.functions))+'+OP:-1)')
    for index, name in enumerate(helpers):
        add(name, '4,'+str(len(layout.functions)+len(layout.opcodes)+index))
    for position, text in sorted(insertions, reverse=True):
        source = source[:position]+text+source[position:]
    source += f'''
extern "C" __global__ void native_profile_reset(int mask){{
 int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<{layout.slots}){{native_rule_profile::elapsed[i]=0;native_rule_profile::calls[i]=0;}}
 if(i==0)native_rule_profile::enabled=mask;
}}
extern "C" __global__ void native_profile_copy(unsigned long long* times,unsigned long long* counts){{
 int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<{layout.slots}){{times[i]=native_rule_profile::elapsed[i];counts[i]=native_rule_profile::calls[i];}}
}}
'''
    return replace(unit, source=source), layout


def bind_profile_kernels(module):
    """Bind diagnostic exports after linking, keeping cached rule specs stable."""
    from .cuda_runtime import KernelSpec, CudaDriverError
    bound = {}
    for name, spec in (('native_profile_reset', ['int32']),
                       ('native_profile_copy', ['int64*', 'int64*'])):
        if name in module._kernels:
            raise ValueError('Profiling kernels already bound')
        function = ctypes.c_void_p()
        status = module._drv.cuModuleGetFunction(ctypes.byref(function), module._module, name.encode())
        if status:
            raise CudaDriverError('Missing profiling export '+name+': '+str(status))
        bound[name] = (function, KernelSpec.from_spec(spec))
    for name, (function, spec) in bound.items():
        module._kernels[name] = function
        module._kernel_specs[name] = spec


def decode_profile(times, counts, layout):
    """Keep per-row counters; function, opcode and helper totals overlap."""
    import numpy as np
    times, counts = np.asarray(times), np.asarray(counts)
    if times.size != layout.slots or counts.size != layout.slots:
        raise ValueError('Matching bounded profiling buffers required')
    times, counts = times.reshape(layout.rows,layout.stride), counts.reshape(layout.rows,layout.stride)
    names = layout.functions+layout.opcodes+layout.helpers
    result = []
    for i,name in enumerate(names):
        if not counts[:,i].any():
            continue
        category = 'function' if i<len(layout.functions) else 'opcode' if i<len(layout.functions)+len(layout.opcodes) else 'helper'
        result.append(dict(category=category,name=name,nanoseconds=[int(v) for v in times[:,i]],
                           calls=[int(v) for v in counts[:,i]]))
    return result
