"""CPU host compilation of the same native-array CUDA semantic source.

Compiles emit_native_array_source with clang++/g++ and CUDA keyword/intrinsic
shims, then runs vm_execute / vm_compact over NumPy arrays via ctypes.
There is no bytecode interpreter, Python rule callback, or CUDA/GPU claim.
Compilation happens only when NativeArrayCPU is constructed.
"""
from __future__ import annotations

import ctypes
from hashlib import sha256
import os
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
from pathlib import Path

import numpy as np

from .codec import pack_states, unpack_states
from .cuda_vm import VmLimits, SUPPORTED_OPS, SUPPORTED_INTRINSICS, DENIED_STANDALONE_INTRINSICS
from .schema import GraphLayout, PackedStateBatch, TAG_LIST, TAG_TUPLE
from .vm_image import VmImage

COMPILE_TIMEOUT_SECONDS = 180
INSTRUCTION_PLACEHOLDER = np.int32(-2147483648)
_MISSING = -2147483647 - 1

_EXECUTE_ARGTYPES = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
)
_COMPACT_ARGTYPES = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
)

HOST_CUDA_SHIMS = r'''
#ifndef __CUDACC__
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __device__
#define __global__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __restrict__
#define __syncthreads() ((void)0)
struct uint3 { unsigned int x, y, z; };
struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1): x(x), y(y), z(z) {}
};
static thread_local uint3 threadIdx{0,0,0};
static thread_local uint3 blockIdx{0,0,0};
static thread_local uint3 blockDim{1,1,1};
static thread_local uint3 gridDim{1,1,1};
static inline unsigned long long native_array_clock_ns() {
  using clock = std::chrono::steady_clock;
  return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
      clock::now().time_since_epoch()).count();
}
static inline void native_array_configure_row(int row, int rows) {
  threadIdx.x = threadIdx.y = threadIdx.z = 0;
  blockIdx.x = (unsigned)row; blockIdx.y = 0; blockIdx.z = 0;
  blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
  gridDim.x = rows > 0 ? (unsigned)rows : 1u; gridDim.y = 1; gridDim.z = 1;
}
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrt(double x) { return 1.0 / std::sqrt(x); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline float __expf(float x) { return std::exp(x); }
inline float __exp2f(float x) { return std::exp2(x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __log10f(float x) { return std::log10(x); }
inline float __powf(float a, float b) { return std::pow(a, b); }
inline float __fsqrt_rn(float x) { return std::sqrt(x); }
inline double __dsqrt_rn(double x) { return std::sqrt(x); }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline double __ddiv_rn(double a, double b) { return a / b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline double __dadd_rn(double a, double b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline double __dsub_rn(double a, double b) { return a - b; }
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
inline int __float2int_rn(float x) { return (int)std::lrintf(x); }
inline int __double2int_rn(double x) { return (int)std::lrint(x); }
inline long long __float2ll_rn(float x) { return (long long)std::llrintf(x); }
inline long long __double2ll_rn(double x) { return (long long)std::llrint(x); }
inline int __clzll(unsigned long long n) { return n ? __builtin_clzll(n) : 64; }
#endif
'''

HOST_CUDA_LAUNCHERS = r'''
#ifndef __CUDACC__
extern "C" void native_array_vm_execute(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* args,int nargs,
 const int* constants_nodes,const int* constants_edges,const unsigned char* constants_payload,
 const int* instructions,const int* functions,int function_count,
 int constant_node_count,int constant_edge_count,int constant_payload_count,
 int* frames,int* outputs,int* statuses,const long long* deadlines,
 int rows,int max_nodes,int max_edges,int max_payload,int entry,int step_limit,
 const int* classes,int class_count,const int* modules,int module_count,long long* gc_marks,int* gc_queue){
  for(int row=0; row<rows; ++row){
    native_array_configure_row(row, rows);
    vm_execute(nodes,edges,payload,counts,args,nargs,constants_nodes,constants_edges,constants_payload,
      instructions,functions,function_count,constant_node_count,constant_edge_count,constant_payload_count,
      frames,outputs,statuses,deadlines,rows,max_nodes,max_edges,max_payload,entry,step_limit,
      classes,class_count,modules,module_count,gc_marks,gc_queue);
  }
}
extern "C" void native_array_vm_compact(
 int* sn,int* se,unsigned char* sp,int* sc,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* roots,int* dn,int* de,unsigned char* dp,int* dc,int* dr,
 int* map,int* queue,int* status,int rows,int maxn,int maxe,int maxp,int capacity){
  for(int row=0; row<rows; ++row){
    native_array_configure_row(row, rows);
    vm_compact(sn,se,sp,sc,cn,ce,cp,cnc,cec,cpc,roots,dn,de,dp,dc,dr,map,queue,status,
      rows,maxn,maxe,maxp,capacity);
  }
}
#endif
'''


class NativeArrayExecutionError(RuntimeError):
    """Native kernel status or compaction failure. Diagnostics include code/fid/PC."""


class NativeArrayCapabilityError(RuntimeError):
    """Missing native-array generator, entry, or host compiler."""


def prepare_host_cuda_source(source):
    """Rewrite CUDA semantic source for host C++. Does not interpret bytecode."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Non-empty CUDA C++ source required')
    if 'extern "C" __global__ void vm_execute' not in source:
        raise ValueError('vm_execute ABI missing from native array source')
    if 'extern "C" __global__ void vm_compact' not in source:
        raise ValueError('vm_compact ABI missing from native array source')
    patched = source.replace(
        'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value))',
        'value=native_array_clock_ns()',
    )
    return HOST_CUDA_SHIMS + patched + HOST_CUDA_LAUNCHERS


def _cxx_compiler():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    raise NativeArrayCapabilityError('Native array CPU executor requires clang++ or g++')


def _library_path(directory, suffix=''):
    directory = Path(directory)
    if os.name == 'nt':
        return directory / ('native_array_cpu'+suffix+'.dll')
    if sys.platform == 'darwin':
        return directory / ('libnative_array_cpu'+suffix+'.dylib')
    return directory / ('libnative_array_cpu'+suffix+'.so')


def compile_host_native_library(source, directory, *, timeout=COMPILE_TIMEOUT_SECONDS):
    """Compile already-shimmed host C++ to a shared library. No CUDA driver."""
    if type(timeout) is not int or timeout < 1:
        raise ValueError('Positive bounded compile timeout required')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # A loaded library must never be overwritten or reused for another image
    # by the platform's dlopen cache. Keep each construction immutable.
    suffix = '-'+sha256(source.encode()).hexdigest()[:12]+'-'+uuid4().hex[:12]
    cpp = directory / ('native_array_cpu'+suffix+'.cpp')
    library = _library_path(directory, suffix)
    cpp.write_text(source, encoding='utf-8')
    compiler = _cxx_compiler()
    cmd = [
        compiler, '-std=c++17', '-shared', '-fPIC', '-O1',
        '-fno-fast-math', '-ffp-contract=off', '-fno-finite-math-only',
        str(cpp), '-o', str(library),
    ]
    try:
        result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise NativeArrayExecutionError(
            'Native array CPU compile exceeded %ss' % timeout
        ) from exc
    if result.returncode:
        log = (result.stdout or '') + (result.stderr or '')
        raise NativeArrayExecutionError('Native array CPU compile failed:\n' + log)
    loaded = ctypes.CDLL(str(library))
    loaded._native_array_source_path = str(cpp)
    return loaded


def _bind_kernels(library):
    execute = library.native_array_vm_execute
    execute.argtypes = _EXECUTE_ARGTYPES
    execute.restype = None
    compact = library.native_array_vm_compact
    compact.argtypes = _COMPACT_ARGTYPES
    compact.restype = None
    return execute, compact


def _positive_capacity(name, value):
    if type(value) is not int or value < 1 or value > 2 ** 30:
        raise ValueError('Positive bounded %s required' % name)
    return value


def _cptr(array):
    if not isinstance(array, np.ndarray):
        raise TypeError('Contiguous NumPy kernel buffer required')
    if not array.flags.c_contiguous:
        raise ValueError('C-contiguous kernel buffer required')
    if array.size == 0:
        return ctypes.c_void_p(0)
    return ctypes.c_void_p(int(array.ctypes.data))


def pack_call_arguments(arguments, layout):
    """Pack list[list[object]] with the existing codec and extract argument refs."""
    if type(arguments) is not list:
        raise TypeError('call() requires a list of per-row argument lists')
    if not arguments:
        raise ValueError('Nonempty native array batch required')
    if any(type(row) is not list for row in arguments):
        raise TypeError('Each call() row must be a list of argument objects')
    nargs = len(arguments[0])
    if any(len(row) != nargs for row in arguments):
        raise ValueError('Uniform argument arity required')
    packed = pack_states(arguments, layout)
    rows = packed.n_rows
    refs = np.empty((rows, nargs), dtype=np.int32)
    if nargs == 0:
        return packed, refs
    for index in range(rows):
        root = int(packed.roots[index])
        tag, offset, count, extra = map(int, packed.nodes[index, root])
        if tag not in (TAG_LIST, TAG_TUPLE) or count != nargs:
            raise ValueError('Packed call arguments must retain list children')
        refs[index] = packed.edges[index, offset:offset + nargs, 0]
    return packed, np.ascontiguousarray(refs, dtype=np.int32)


def _require_native_array_source(image, limits, **source_options):
    try:
        from .native_array_codegen import emit_native_array_source, native_array_plan
    except ImportError as exc:
        raise NativeArrayCapabilityError(
            'NativeArrayCPU requires emit_native_array_source; interpreter fallback is rejected'
        ) from exc
    if not isinstance(limits, VmLimits):
        raise TypeError('VmLimits required')
    plan = native_array_plan(image)
    plan.verify()
    if plan.interpreter_fallback:
        raise NativeArrayCapabilityError('Interpreter fallback is rejected')
    source = emit_native_array_source(image, limits, **source_options)
    if not isinstance(source, str) or 'native_dispatch_pc' not in source:
        raise NativeArrayCapabilityError('Native array source must be statically compiled')
    if 'c->instructions+pc*4' in source:
        raise NativeArrayCapabilityError('Native array source must not fetch a runtime opcode stream')
    return source, plan


class NativeArrayCall:
    """Typed native-call receipt. Decode happens only after the bounded batch."""

    def __init__(self, image, statuses, packed, outputs, compact_status):
        self.image = image
        self.statuses = statuses
        self.packed = packed
        self.outputs = outputs
        self.compact_status = compact_status

    def check(self):
        statuses = np.asarray(self.statuses)
        compact = np.asarray(self.compact_status)
        bad = np.flatnonzero(statuses[:, 0])
        source = 'status'
        if len(bad) == 0:
            bad = np.flatnonzero(compact)
            source = 'compact status'
            if len(bad) == 0:
                return statuses
        row = int(bad[0])
        code, fid, pc, steps = map(int, statuses[row])
        if source == 'compact status':
            code = int(compact[row])
        name = (
            self.image.function_ids[fid]
            if 0 <= fid < len(self.image.function_ids) else 'unknown'
        )
        raise NativeArrayExecutionError(
            'Native array CPU row %s: status=%s, fid=%s, function=%s, pc=%s, steps=%s'
            % (row, code, fid, name, pc, steps)
        )

    def decode(self):
        self.check()
        return unpack_states(self.packed)


class NativeArrayCPU:
    """Host CPU reference backend for the same compiled native-array CUDA source."""

    execution_mode = 'native_array_aot'
    runtime_instruction_fetch = False
    interpreter_fallback = False
    supports_full_duel_rules = False

    def __init__(
        self,
        image,
        *,
        limits=VmLimits(),
        directory=None,
        max_nodes=262144,
        max_edges=524288,
        max_payload_bytes=33554432,
    ):
        if not isinstance(image, VmImage):
            raise TypeError('Numeric VM image required')
        image.verify()
        if not isinstance(limits, VmLimits):
            raise TypeError('VmLimits required')
        self.image = image
        self.limits = limits
        self.max_nodes = _positive_capacity('max_nodes', max_nodes)
        self.max_edges = _positive_capacity('max_edges', max_edges)
        self.max_payload_bytes = _positive_capacity('max_payload_bytes', max_payload_bytes)
        source, self.native_plan = _require_native_array_source(image, limits)
        host_source = prepare_host_cuda_source(source)
        self.source_sha256 = sha256(source.encode()).hexdigest()
        self.host_source_sha256 = sha256(host_source.encode()).hexdigest()
        self._temp = None
        self._library = None
        self.closed = False
        if directory is None:
            self._temp = tempfile.TemporaryDirectory(prefix='native_array_cpu_')
            workdir = Path(self._temp.name)
        else:
            workdir = Path(directory).resolve()
        self.directory = workdir
        try:
            self._library = compile_host_native_library(host_source, workdir)
            self.source_path = Path(self._library._native_array_source_path)
            self._execute, self._compact = _bind_kernels(self._library)
        except BaseException:
            self.close()
            raise
        self._constants = {
            'nodes': np.ascontiguousarray(image.nodes, dtype=np.int32),
            'edges': np.ascontiguousarray(image.edges, dtype=np.int32),
            'payload': np.ascontiguousarray(image.payload, dtype=np.uint8),
            'functions': np.ascontiguousarray(image.functions, dtype=np.int32),
            'classes': np.ascontiguousarray(image.classes, dtype=np.int32),
            'modules': np.ascontiguousarray(image.modules, dtype=np.int32),
            'instructions': np.full((1, 4), INSTRUCTION_PLACEHOLDER, dtype=np.int32),
        }

    def capability(self, entry):
        if entry not in self.image.entries:
            raise NativeArrayCapabilityError('Unknown native array entry: %s' % entry)
        missing = self.image.missing_capabilities(entry, opcodes=SUPPORTED_OPS,
            intrinsics=SUPPORTED_INTRINSICS | DENIED_STANDALONE_INTRINSICS)
        return dict(
            entry=entry,
            ready=not any(missing.values()),
            missing=missing,
            full_duel_rules=False,
            actual_backend='cpu_native_array',
            execution_mode=self.execution_mode,
            runtime_instruction_fetch=False,
            interpreter_fallback=False,
            gpu_qualified=False,
            native_plan=dict(identity=self.native_plan.identity,
                instructions=self.native_plan.coverage['compiled_op_count'],
                functions=self.native_plan.coverage['compiled_function_count']),
            source_sha256=self.source_sha256,
        )

    def call(self, entry, arguments, *, steps=None):
        layout = GraphLayout(
            max(len(arguments) if type(arguments) is list else 1, 1),
            self.max_nodes,
            self.max_edges,
            self.max_payload_bytes,
        )
        packed, refs = pack_call_arguments(arguments, layout)
        return self.run_packed(entry, packed, refs, steps=steps).decode()

    def run_packed(self, entry, packed, argument_refs, steps=None):
        if self.closed:
            raise RuntimeError('Native array CPU executor is closed')
        cap = self.capability(entry)
        if not cap['ready']:
            raise NativeArrayCapabilityError('Unsupported native array entry')
        if not isinstance(packed, PackedStateBatch):
            raise TypeError('Packed graph ingress required')
        rows = packed.n_rows
        if rows < 1:
            raise ValueError('Nonempty native array batch required')
        if (not isinstance(argument_refs, np.ndarray) or argument_refs.dtype != np.int32
                or not argument_refs.flags.c_contiguous
                or argument_refs.ndim != 2 or argument_refs.shape[0] != rows):
            raise ValueError('argument_refs must be int32[rows, nargs]')
        refs = argument_refs
        if np.any(packed.counts < 0) or np.any(packed.counts[:, 0] < 1):
            raise ValueError('Invalid native ingress graph counts')
        if np.any((refs >= packed.counts[:, 0, None]) | (refs < -len(self.image.nodes))):
            raise ValueError('Argument reference outside its checked array graph')
        step_limit = self.limits.steps if steps is None else steps
        if type(step_limit) is not int or step_limit < 1 or step_limit > 2 ** 30:
            raise ValueError('Positive bounded step limit required')
        if packed.layout.max_nodes > self.max_nodes:
            raise ValueError('Ingress node capacity exceeds native array CPU heap')
        if packed.layout.max_edges > self.max_edges:
            raise ValueError('Ingress edge capacity exceeds native array CPU heap')
        if packed.layout.max_payload_bytes > self.max_payload_bytes:
            raise ValueError('Ingress payload capacity exceeds native array CPU heap')
        if np.any(packed.counts[:, 0] > self.max_nodes) or np.any(packed.counts[:, 1] > self.max_edges):
            raise ValueError('Ingress graph exceeds native array CPU heap')
        if np.any(packed.counts[:, 2] > self.max_payload_bytes):
            raise ValueError('Ingress payload exceeds native array CPU heap')

        nodes = np.zeros((rows, self.max_nodes, 4), dtype=np.int32)
        edges = np.zeros((rows, self.max_edges, 2), dtype=np.int32)
        payload = np.zeros((rows, self.max_payload_bytes), dtype=np.uint8)
        counts = np.zeros((rows, 3), dtype=np.int32)
        nodes[:, :packed.layout.max_nodes] = packed.nodes
        edges[:, :packed.layout.max_edges] = packed.edges
        payload[:, :packed.layout.max_payload_bytes] = packed.payload
        counts[:] = packed.counts

        stride = 14 + self.limits.locals + self.limits.stack + self.limits.cells
        frames = np.zeros((rows, self.limits.frames, stride), dtype=np.int32)
        outputs = np.full(rows, _MISSING, dtype=np.int32)
        statuses = np.zeros((rows, 4), dtype=np.int32)
        deadlines = np.zeros(rows, dtype=np.int64)
        constant_nodes = int(self._constants['nodes'].shape[0])
        gc_marks = np.zeros((rows, self.max_nodes + constant_nodes), dtype=np.int64)
        gc_queue = np.zeros((rows, self.max_nodes), dtype=np.int32)
        constants = self._constants
        entry_id = int(self.image.entries[entry])
        ref_storage = refs if refs.size else np.zeros(1, dtype=np.int32)
        self._execute(
            _cptr(nodes), _cptr(edges), _cptr(payload), _cptr(counts), _cptr(ref_storage), ctypes.c_int(int(refs.shape[1])),
            _cptr(constants['nodes']), _cptr(constants['edges']), _cptr(constants['payload']),
            _cptr(constants['instructions']), _cptr(constants['functions']),
            ctypes.c_int(len(self.image.functions)),
            ctypes.c_int(constant_nodes), ctypes.c_int(len(self.image.edges)),
            ctypes.c_int(len(self.image.payload)),
            _cptr(frames), _cptr(outputs), _cptr(statuses), _cptr(deadlines),
            ctypes.c_int(rows), ctypes.c_int(self.max_nodes), ctypes.c_int(self.max_edges),
            ctypes.c_int(self.max_payload_bytes), ctypes.c_int(entry_id), ctypes.c_int(step_limit),
            _cptr(constants['classes']), ctypes.c_int(len(self.image.classes)),
            _cptr(constants['modules']), ctypes.c_int(len(self.image.modules)),
            _cptr(gc_marks), _cptr(gc_queue),
        )

        capacity = self.max_nodes + constant_nodes
        dest_nodes = np.zeros_like(nodes)
        dest_edges = np.zeros_like(edges)
        dest_payload = np.zeros_like(payload)
        dest_counts = np.zeros_like(counts)
        dest_roots = np.zeros(rows, dtype=np.int32)
        mapping = np.zeros((rows, capacity), dtype=np.int32)
        queue = np.zeros((rows, capacity), dtype=np.int32)
        compact_status = np.zeros(rows, dtype=np.int32)
        self._compact(
            _cptr(nodes), _cptr(edges), _cptr(payload), _cptr(counts),
            _cptr(constants['nodes']), _cptr(constants['edges']), _cptr(constants['payload']),
            ctypes.c_int(constant_nodes), ctypes.c_int(len(self.image.edges)),
            ctypes.c_int(len(self.image.payload)), _cptr(outputs),
            _cptr(dest_nodes), _cptr(dest_edges), _cptr(dest_payload), _cptr(dest_counts),
            _cptr(dest_roots), _cptr(mapping), _cptr(queue), _cptr(compact_status),
            ctypes.c_int(rows), ctypes.c_int(self.max_nodes), ctypes.c_int(self.max_edges),
            ctypes.c_int(self.max_payload_bytes), ctypes.c_int(capacity),
        )
        layout = GraphLayout(rows, self.max_nodes, self.max_edges, self.max_payload_bytes)
        packed_out = PackedStateBatch(layout, dest_nodes, dest_edges, dest_payload, dest_counts, dest_roots)
        return NativeArrayCall(self.image, statuses, packed_out, outputs, compact_status)

    def close(self):
        if getattr(self, 'closed', False):
            return
        library = getattr(self, '_library', None)
        if library is not None:
            try:
                import _ctypes
                closer = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
                closer(library._handle)
            except Exception:
                pass
            self._library = None
        temp = getattr(self, '_temp', None)
        if temp is not None:
            temp.cleanup()
            self._temp = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
