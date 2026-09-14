"""Lower IR to an NVRTC CUDA kernel. Generated code, not a second rule source."""
from __future__ import annotations

from .ast import Bin, ClampMin, Cmp, Const, Name, Not, Program, Select, validate
from .schema import FIELD_INDEX, FIELDS, N_FIELDS, SIDE_ALIASES, SLICE_VERSION


def _c_expr(expr) -> str:
    if isinstance(expr, Const):
        return str(int(expr.value))
    if isinstance(expr, Name):
        return expr.ident
    if isinstance(expr, Not):
        return f'(!({_c_expr(expr.inner)}))'
    if isinstance(expr, ClampMin):
        return f'max({expr.lo},{_c_expr(expr.inner)})'
    if isinstance(expr, Select):
        return f'(({_c_expr(expr.cond)})?({_c_expr(expr.then)}):({_c_expr(expr.orelse)}))'
    if isinstance(expr, Cmp):
        ops = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}
        return f'(({_c_expr(expr.left)}){ops[expr.op]}({_c_expr(expr.right)}))'
    if isinstance(expr, Bin):
        if expr.op == 'add':
            return f'(({_c_expr(expr.left)})+({_c_expr(expr.right)}))'
        if expr.op == 'sub':
            return f'(({_c_expr(expr.left)})-({_c_expr(expr.right)}))'
        if expr.op == 'min':
            return f'min({_c_expr(expr.left)},{_c_expr(expr.right)})'
        if expr.op == 'max':
            return f'max({_c_expr(expr.left)},{_c_expr(expr.right)})'
        if expr.op == 'and':
            return f'(({_c_expr(expr.left)})&&({_c_expr(expr.right)}))'
        if expr.op == 'or':
            return f'(({_c_expr(expr.left)})||({_c_expr(expr.right)}))'
    raise ValueError(f'cannot emit {type(expr).__name__}')


def emit_cuda(program: Program) -> str:
    """Emit a multi-step kernel with the same launch shape as the experimental handwritten slice."""
    validate(program)
    fields = program.fields or FIELDS
    field_index = {name: i for i, name in enumerate(fields)}
    aliases = program.aliases if program.aliases is not None else SIDE_ALIASES
    n_fields = len(fields)
    reload = [f'   int {name}=x[{idx}];' for name, idx in field_index.items()]
    for alias, a_name, b_name in aliases:
        ia, ib = field_index[a_name], field_index[b_name]
        reload.append(f'   int {alias}=side?x[{ib}]:x[{ia}];')
    binds = [f'   int {bind.name}={_c_expr(bind.expr)};' for bind in program.binds]
    stores = []
    for store in program.stores:
        if store.field in field_index:
            idx = str(field_index[store.field])
        else:
            a_name, b_name = next((a, b) for alias, a, b in aliases if alias == store.field)
            idx = f'(side?{field_index[b_name]}:{field_index[a_name]})'
        stores.append(f'   x[{idx}]={_c_expr(store.expr)};')
    skip = _c_expr(program.skip_when)
    return '\n'.join([
        f'extern "C" __global__ void {program.name}(const int* input, int* output, int n, int steps) {{',
        ' int row=blockIdx.x*blockDim.x+threadIdx.x;',
        ' if(row>=n)return;',
        f' int x[{n_fields}];',
        ' #pragma unroll',
        f' for(int j=0;j<{n_fields};j++)x[j]=input[row*{n_fields}+j];',
        ' for(int k=0;k<steps;k++){',
        '  int side=k%2;',
        *reload,
        f'  if({skip})continue;',
        *binds,
        *stores,
        ' }',
        ' #pragma unroll',
        f' for(int j=0;j<{n_fields};j++)output[row*{n_fields}+j]=x[j];',
        '}',
    ])


def compile_cuda_source(source: str, fn_name: str):
    """NVRTC helper. Experimental handwritten kernels also go through here."""
    import ctypes as C
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to compile the IR kernel')
    torch.cuda.init()
    anchor = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    libdir = Path(torch.__file__).parent / 'lib'
    nv = C.WinDLL(str(next(path for path in libdir.glob('nvrtc64_*.dll') if '.alt.' not in path.name)))

    def check(rc):
        if rc:
            raise RuntimeError('CUDA/NVRTC result ' + str(rc))

    prog = C.c_void_p()
    src = source.encode()
    check(nv.nvrtcCreateProgram(C.byref(prog), C.c_char_p(src), C.c_char_p(b'ir.cu'), 0, None, None))
    arch = (' --gpu-architecture=compute_%d%d' % torch.cuda.get_device_capability()).strip().encode()
    options = (C.c_char_p * 2)(arch, b'--std=c++17')
    rc = nv.nvrtcCompileProgram(prog, 2, options)
    if rc:
        size = C.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, C.byref(size))
        buf = C.create_string_buffer(size.value)
        nv.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError(buf.value.decode())
    size = C.c_size_t()
    check(nv.nvrtcGetPTXSize(prog, C.byref(size)))
    ptx = C.create_string_buffer(size.value)
    check(nv.nvrtcGetPTX(prog, ptx))
    check(nv.nvrtcDestroyProgram(C.byref(prog)))
    drv = C.WinDLL('nvcuda.dll')
    check(drv.cuInit(0))
    mod = C.c_void_p()
    check(drv.cuModuleLoadData(C.byref(mod), ptx))
    fn = C.c_void_p()
    check(drv.cuModuleGetFunction(C.byref(fn), mod, fn_name.encode()))

    def launch(inp, out, steps: int = 4):
        p = C.c_void_p(inp.data_ptr())
        q = C.c_void_p(out.data_ptr())
        n = C.c_int(inp.shape[0])
        s = C.c_int(int(steps))
        params = (C.c_void_p * 4)(
            C.cast(C.byref(p), C.c_void_p),
            C.cast(C.byref(q), C.c_void_p),
            C.cast(C.byref(n), C.c_void_p),
            C.cast(C.byref(s), C.c_void_p),
        )
        check(drv.cuLaunchKernel(
            fn, C.c_uint((n.value + 127) // 128), C.c_uint(1), C.c_uint(1),
            C.c_uint(128), C.c_uint(1), C.c_uint(1), C.c_uint(0),
            C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None,
        ))

    launch.source = source
    launch.keep = (drv, mod, fn, anchor)
    launch.program_version = SLICE_VERSION
    return launch


def compile_cuda(program: Program):
    launch = compile_cuda_source(emit_cuda(program), program.name)
    launch.program_version = program.version
    return launch
