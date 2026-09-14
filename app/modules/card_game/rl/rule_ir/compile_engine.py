"""Explicit compilation of the shared starter engine to CPU native or CUDA."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

from .emit_engine import emit_engine_source
from .layout import ENGINE_NAME, ENGINE_VERSION, ROW_WIDTH


def write_engine_source(directory: str | Path, *, cuda: bool = False, deck_id="starter") -> Path:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (ENGINE_NAME + ('.cu' if cuda else '.c'))
    source = emit_engine_source(deck_id)
    if not cuda:
        source = '#ifndef __CUDACC__\n#define __CUDACC_CPU__ 1\n#endif\n' + source
    path.write_text(source, encoding='utf-8')
    return path


def compile_native_engine(directory: str | Path, *, deck_id="starter"):
    directory = Path(directory).resolve()
    source = write_engine_source(directory, cuda=False, deck_id=deck_id)
    if os.name == 'nt':
        candidates = list(Path('C:/Program Files/Microsoft Visual Studio/2022').glob(
            '*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        compiler = shutil.which('cl') or (str(sorted(candidates)[-1]) if candidates else None)
        if compiler is None:
            raise RuntimeError('Native compiled engine requires an installed C compiler')
        library = directory / (ENGINE_NAME + '.dll')
        cmd = [compiler, '/nologo', '/LD', '/O2', '/TC', str(source),
               '/link', '/OUT:' + str(library)]
    else:
        compiler = shutil.which('clang') or shutil.which('gcc')
        if compiler is None:
            raise RuntimeError('Native compiled engine requires a C compiler')
        library = directory / (ENGINE_NAME + '.so')
        cmd = [compiler, '-O3', '-shared', '-fPIC', str(source), '-o', str(library)]
    result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    dll = ctypes.CDLL(str(library))
    fn = getattr(dll, ENGINE_NAME)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    reset_fn = getattr(dll, 'starter_reset', None)
    if reset_fn is not None:
        reset_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        reset_fn.restype = None
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        import _ctypes
        (_ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose)(dll._handle)
        closed = True

    def launch_ptr(row_ptr, action_ptr, n):
        if closed:
            raise RuntimeError('Native engine is closed')
        fn(row_ptr, action_ptr, int(n))

    def launch_lists(rows, actions):
        if closed:
            raise RuntimeError('Native engine is closed')
        n = len(rows)
        if n != len(actions):
            raise ValueError('row/action count mismatch')
        flat = []
        for row in rows:
            if len(row) != ROW_WIDTH:
                raise ValueError(f'Native engine requires width {ROW_WIDTH}')
            flat.extend(int(v) for v in row)
        row_arr = (ctypes.c_int * (n * ROW_WIDTH))(*flat)
        act_arr = (ctypes.c_int * n)(*[int(v) for v in actions])
        fn(ctypes.addressof(row_arr), ctypes.addressof(act_arr), n)
        out = []
        for i in range(n):
            out.append([row_arr[i * ROW_WIDTH + j] for j in range(ROW_WIDTH)])
        return out

    def reset_lists(rows, seeds):
        if closed:
            raise RuntimeError('Native engine is closed')
        if reset_fn is None:
            raise RuntimeError('Native engine has no compiled reset')
        n = len(rows)
        if n != len(seeds):
            raise ValueError('row/seed count mismatch')
        flat = []
        for row in rows:
            if len(row) != ROW_WIDTH:
                raise ValueError(f'Native engine requires width {ROW_WIDTH}')
            flat.extend(int(v) for v in row)
        row_arr = (ctypes.c_int * (n * ROW_WIDTH))(*flat)
        seed_arr = (ctypes.c_int * n)(*[int(v) for v in seeds])
        reset_fn(ctypes.addressof(row_arr), ctypes.addressof(seed_arr), n)
        out = []
        for i in range(n):
            out.append([row_arr[i * ROW_WIDTH + j] for j in range(ROW_WIDTH)])
        return out

    def launch(rows, actions):
        if closed:
            raise RuntimeError('Native engine is closed')
        import torch
        if not torch.is_tensor(rows):
            packed = launch_lists(rows, actions)
            return packed
        if rows.device.type != 'cpu' or actions.device.type != 'cpu':
            raise ValueError('Native engine requires CPU tensors')
        if rows.dtype != torch.int32 or actions.dtype != torch.int32:
            raise ValueError('Native engine requires int32 tensors')
        if not rows.is_contiguous() or not actions.is_contiguous():
            raise ValueError('Native engine requires contiguous tensors')
        if rows.ndim != 2 or rows.shape[1] != ROW_WIDTH:
            raise ValueError(f'Native engine requires [N,{ROW_WIDTH}] rows')
        if actions.ndim != 1 or actions.shape[0] != rows.shape[0]:
            raise ValueError('Native engine actions must be [N]')
        fn(rows.data_ptr(), actions.data_ptr(), int(rows.shape[0]))

    launch.close = close
    launch.keep = dll
    launch.source = source.read_text(encoding='utf-8')
    launch.backend = 'native'
    launch.engine_name = ENGINE_NAME
    launch.engine_version = ENGINE_VERSION
    launch.row_width = ROW_WIDTH
    launch.launch_lists = launch_lists
    launch.reset_lists = reset_lists
    launch.reset = lambda rows, seeds: reset_fn(rows.data_ptr(), seeds.data_ptr(), int(rows.shape[0]))
    return launch


def compile_cuda_engine(*, deck_id="starter"):
    # CUDA annotations only; both targets execute the identical rule body.
    source = emit_engine_source(deck_id).replace('#include <stdint.h>', '')
    source = source.replace('static ', 'static __device__ ')
    launch = _compile_cuda_engine_nvrtc(source)
    launch.backend = 'cuda'
    launch.engine_name = ENGINE_NAME
    launch.engine_version = ENGINE_VERSION
    launch.row_width = ROW_WIDTH
    launch.source = source
    return launch


def _compile_cuda_engine_nvrtc(source: str):
    import ctypes as C
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to compile the starter engine')
    torch.cuda.init()
    anchor = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    libdir = Path(torch.__file__).parent / 'lib'
    nvrtc_paths = list(libdir.glob('nvrtc64_*.dll')) + list(libdir.glob('libnvrtc.so*'))
    if not nvrtc_paths:
        raise RuntimeError('NVRTC library not found')
    nvrtc_path = str(next(path for path in nvrtc_paths if '.alt.' not in path.name))
    nv = C.WinDLL(nvrtc_path) if os.name == 'nt' else C.CDLL(nvrtc_path)

    def check(rc, what='CUDA/NVRTC'):
        if rc:
            raise RuntimeError(f'{what} result {rc}')

    prog = C.c_void_p()
    src = source.encode()
    check(nv.nvrtcCreateProgram(C.byref(prog), C.c_char_p(src), C.c_char_p(b'starter_engine.cu'), 0, None, None))
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
    if os.name == 'nt':
        drv = C.WinDLL('nvcuda.dll')
    else:
        drv = C.CDLL('libcuda.so.1')
    check(drv.cuInit(0), 'cuInit')
    mod = C.c_void_p()
    check(drv.cuModuleLoadData(C.byref(mod), ptx), 'cuModuleLoadData')
    fn = C.c_void_p()
    check(drv.cuModuleGetFunction(C.byref(fn), mod, ENGINE_NAME.encode()), 'cuModuleGetFunction')

    reset_fn = C.c_void_p()
    check(drv.cuModuleGetFunction(C.byref(reset_fn), mod, b'starter_reset'), 'reset function')
    closed = False

    def invoke(kernel, rows, actions):
        if closed:
            raise RuntimeError('CUDA engine is closed')
        import torch
        if rows.device.type != 'cuda' or actions.device.type != 'cuda':
            raise ValueError('CUDA engine requires CUDA tensors')
        if rows.dtype != torch.int32 or actions.dtype != torch.int32:
            raise ValueError('CUDA engine requires int32 tensors')
        if rows.ndim != 2 or rows.shape[1] != ROW_WIDTH:
            raise ValueError(f'CUDA engine requires [N,{ROW_WIDTH}] rows')
        if not rows.is_contiguous() or not actions.is_contiguous():
            raise ValueError('CUDA engine requires contiguous tensors')
        if actions.ndim != 1 or actions.shape[0] != rows.shape[0]:
            raise ValueError('actions/seeds must be [N]')
        p = C.c_void_p(rows.data_ptr())
        q = C.c_void_p(actions.data_ptr())
        n = C.c_int(rows.shape[0])
        params = (C.c_void_p * 3)(
            C.cast(C.byref(p), C.c_void_p),
            C.cast(C.byref(q), C.c_void_p),
            C.cast(C.byref(n), C.c_void_p),
        )
        check(drv.cuLaunchKernel(
            kernel, C.c_uint((n.value + 127) // 128), C.c_uint(1), C.c_uint(1),
            C.c_uint(128), C.c_uint(1), C.c_uint(1), C.c_uint(0),
            C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None,
        ), 'cuLaunchKernel')

    def launch(rows, actions):
        invoke(fn, rows, actions)

    def reset(rows, seeds):
        invoke(reset_fn, rows, seeds)

    def lists_call(rows, values, kernel):
        rows_t = torch.tensor(rows, dtype=torch.int32, device='cuda')
        values_t = torch.tensor(values, dtype=torch.int32, device='cuda')
        invoke(kernel, rows_t, values_t)
        return rows_t.cpu().tolist()

    def close():
        nonlocal closed
        if not closed:
            torch.cuda.synchronize()
            check(drv.cuModuleUnload(mod), 'cuModuleUnload')
            closed = True

    launch.reset = reset
    launch.launch_lists = lambda rows, actions: lists_call(rows, actions, fn)
    launch.reset_lists = lambda rows, seeds: lists_call(rows, seeds, reset_fn)
    launch.close = close
    launch.keep = (drv, mod, fn, reset_fn, anchor)
    return launch
