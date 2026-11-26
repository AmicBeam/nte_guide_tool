"""Bind persistent domain columns to the canonical statically compiled rules.

The CPU reference and CUDA executor compile this same source. State arguments
are typed record references, never encoded into the transient language heap.
"""
from hashlib import sha256
import re

from .native_array_cpu import HOST_CUDA_SHIMS, _require_native_array_source


SOA_ARGUMENTS = '''long long* soa_i64,double* soa_f64,int* soa_i32,unsigned char* soa_u8,
 int* soa_views,int* soa_value_cache,int* soa_clone_map,int* soa_clone_queue,
 int* soa_constant_adopt,const int* soa_arguments,int* soa_output_roots,const unsigned char* soa_active'''
SOA_NAMES = ('soa_i64', 'soa_f64', 'soa_i32', 'soa_u8', 'soa_views',
             'soa_value_cache', 'soa_clone_map', 'soa_clone_queue',
             'soa_constant_adopt', 'soa_arguments', 'soa_output_roots', 'soa_active')

_DOMAIN_CALLS = ('soa_write_cell', 'soa_write_encoded', 'soa_promote', 'soa_encode',
                 'soa_dict_set', 'soa_list_set', 'soa_list_append',
                 'soa_list_concat', 'soa_list_repeat', 'soa_clone',
                 'soa_clone_gid', 'soa_clone_cell_refs')
_RUNTIME_CALLS = ('call', 'new_frame', 'return_value', 'native_builtin_call',
                  'native_fast_call', 'builtin_attribute', 'compare', 'contains',
                  'numeric_binary')


def separate_domain_calls(source):
    """Share large domain mutation/graph bodies without changing their code."""
    for name in _DOMAIN_CALLS:
        pattern = re.compile(r'(?m)^(__device__ )inline( int ' + name + r'\()')
        if not pattern.search(source):
            raise ValueError('Missing native domain call boundary: ' + name)
        source = pattern.sub(r'\1__noinline__\2', source)
    return source


def separate_runtime_calls(source):
    """Keep one copy of large dynamic-call/type dispatch implementations."""
    for name in _RUNTIME_CALLS:
        pattern = re.compile(r'(?m)^(__device__ )inline( (?:int|void) ' + name + r'\()')
        if not pattern.search(source):
            raise ValueError('Missing native runtime call boundary: ' + name)
        source = pattern.sub(r'\1__noinline__\2', source)
    return source


def separate_soa_calls(source):
    """Bound PC shards while retaining inlined opcode/library specialization.

    One shared step specialization handles each opcode. Its arguments remain
    unchanged; native_handlers and native_opcode can still inline within that
    specialization. Avoid a deep noinline chain through language libraries.
    """
    names=r'native_pc_shard_\d+|native_dispatch_pc|native_step|execute'
    pattern=re.compile(r'(?m)^(__device__ )inline( void ('+names+r')\()')
    changed=set()
    def replace(match):
        changed.add(match[3])
        return match[1]+'__noinline__'+match[2]
    result=pattern.sub(replace,source)
    required={'native_dispatch_pc','native_step','execute'}
    if not required<=changed:
        raise ValueError('Native SoA call boundary changed: '+str(sorted(required-changed)))
    from .vm_library_boundaries import separate_library_calls_source
    return separate_library_calls_source(result,functions=(
        'stdlib_invoke','stdlib_resume','stdlib_call','object_attribute'))


def _kernel(source, name):
    start = source.index('extern "C" __global__ void '+name+'(')
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def emit_native_soa_source(image, schema, limits, *, stack_runs=False, native_functions=False,
                           native_values=False):
    if any(type(value) is not bool for value in (stack_runs, native_functions, native_values)):
        raise TypeError('Explicit native compilation options required')
    if stack_runs and (native_functions or native_values):
        raise ValueError('Stack fusion and native lowering require separate validation')
    from .vm_soa_views import integrate_soa_source
    source, plan = _require_native_array_source(image, limits,
        direct_stack_dispatch=True, compact_frame_init=True, constant_lookup=True)
    if native_functions:
        from .native_functions import integrate_native_functions
        resume_pcs = ()
        if native_values:
            from .native_values import compile_native_values
            value_plan = compile_native_values(image, schema, plan)
            resume_pcs = tuple(end for _, _, _, end in value_plan.regions)
        source, _ = integrate_native_functions(source, plan, complete=True, resume_pcs=resume_pcs)
    elif stack_runs:
        from .native_stack_runs import fuse_stack_runs
        source=fuse_stack_runs(source,plan)
    else:
        from .native_shard_runs import run_native_shards
        source=run_native_shards(source,plan)
    source = integrate_soa_source(source, schema, image=image)
    from .vm_soa_constant_fields import integrate_constant_fields
    source = integrate_constant_fields(source,schema,image)
    if native_values or native_functions:
        from .native_calls import integrate_native_calls
        source = integrate_native_calls(source)
    if native_values:
        from .native_values import integrate_native_values
        source, _ = integrate_native_values(source, image, schema, plan)
    kernel = _kernel(source, 'vm_execute')
    marker = 'DOMAIN_SOA_ATTACH_POINT'
    if kernel.count(marker) != 1:
        raise ValueError('Native SoA attachment boundary changed')
    kernel = kernel.replace('void vm_execute(', 'void vm_soa_execute(', 1)
    opening = kernel.index('{')
    kernel = kernel[:opening].rstrip()[:-1]+','+SOA_ARGUMENTS+')'+kernel[opening:]
    kernel = kernel.replace('if(tid>=rows)return;',
        'if(tid>=rows)return;if(!soa_active[tid]){outputs[tid]=-2147483648;'
        'soa_output_roots[tid]=0;for(int k=0;k<4;k++)statuses[tid*4+k]=0;return;}', 1)
    attach = '''
 DomainSoAContext domain;
 domain.i64=soa_i64;domain.f64=soa_f64;domain.i32=soa_i32;domain.u8=soa_u8;
 domain.rows=rows;domain.row=tid;domain.views=soa_views;domain.value_cache=soa_value_cache;
 domain.clone_map=soa_clone_map;domain.clone_queue=soa_clone_queue;
 domain.constant_adopt=soa_constant_adopt;
 domain.constant_node_count=constant_node_count;
 domain.failed_pool=-1;
 soa_attach(&c.h,&domain);
 int domain_args[256];
 if(nargs>256){statuses[tid*4]=6;return;}
 for(int i=0;i<nargs;i++){
   int typed=soa_arguments[(long long)tid*nargs+i];
   domain_args[i]=typed?soa_view_ref(&c.h,typed):args[(long long)tid*nargs+i];
 }
'''
    line_start = kernel.rfind('\n', 0, kernel.index(marker)) + 1
    line_end = kernel.index('\n', kernel.index(marker))
    kernel = kernel[:line_start]+attach+kernel[line_end:]
    kernel = kernel.replace('new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3)',
                            'new_frame(&c,entry,domain_args,nargs,-1,3)')
    kernel = kernel.replace('outputs[tid]=c.output;',
                            'soa_output_roots[tid]=soa_is_view(&c.h,c.output)?soa_global_ref(&c.h,c.output):0;'
                            'if(c.h.error&&domain.failed_pool>=0)soa_output_roots[tid]=-domain.failed_pool-1;outputs[tid]=c.output;')
    from .native_soa_projection import emit_soa_projection_source
    # The generic graph entry points do not have a domain context and are
    # never launched by this backend. Do not compile extra copies of execute.
    for name in ('vm_execute', 'vm_execute_device'):
        source = source.replace(_kernel(source, name), '', 1)
    source += '\n'+kernel+'\n'+emit_soa_projection_source(schema)
    source = separate_soa_calls(source)
    if native_functions or native_values:
        source = separate_domain_calls(source)
        source = separate_runtime_calls(source)
    return source, plan


def emit_host_soa_source(source):
    launchers = []
    for kernel_name, host_name in (('vm_soa_execute', 'native_soa_execute'),
                                   ('soa_export_projection', 'native_soa_export_projection'),
                                   ('soa_select_result','native_soa_select_result'),
                                   ('soa_decode_metadata','native_soa_decode_metadata'),
                                   ('soa_decode_pair','native_soa_decode_pair'),
                                   ('soa_decode_float','native_soa_decode_float')):
        kernel = _kernel(source, kernel_name)
        signature = kernel[:kernel.index('{')].replace('__global__ ', '')
        parameters = signature[signature.index('(')+1:signature.rfind(')')]
        names = [re.search(r'(\w+)\s*$', parameter).group(1) for parameter in parameters.split(',')]
        signature = signature.replace('void '+kernel_name+'(', 'void '+host_name+'(', 1)
        qualified=('soa_metadata_result::' if kernel_name.startswith('soa_decode_') else '')+kernel_name
        launchers.append(signature+'{for(int row=0;row<rows;row++){native_array_configure_row(row,rows);'+qualified+'('+','.join(names)+');}}')
    source = source.replace('asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value))',
                            'value=native_array_clock_ns()')
    return HOST_CUDA_SHIMS+source+'\n#ifndef __CUDACC__\n'+'\n'.join(launchers)+'\n#endif\n'


def execute_only_source(source):
    """Keep the identical rule kernel/libraries for explicit host diagnostics.

    Projection/export/decode kernels have no callers inside vm_soa_execute.
    Closed device search still compiles the full module and must qualify it.
    """
    kernel = _kernel(source, 'vm_soa_execute')
    if source.count(kernel) != 1:
        raise ValueError('Unique native rule kernel required')
    result = source[:source.index(kernel) + len(kernel)]
    compact = _kernel(result, 'vm_compact')
    result = result.replace(compact, '', 1)
    if _kernel(result, 'vm_soa_execute') != kernel:
        raise ValueError('Diagnostic rule kernel changed')
    return result


def source_identity(source):
    return sha256(source.encode()).hexdigest()
