"""Explicit native CPU compilation of the same scalar source emitted for CUDA.

No compilation on import. Caller owns the cache directory and compiler lifetime.
"""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
from .lower_cuda import emit_cuda


def emit_native(program):
    source = emit_cuda(program)
    export = '__declspec(dllexport)' if os.name == 'nt' else '__attribute__((visibility("default")))'
    source = source.replace('__global__', export, 1)
    source = source.replace(' int row=blockIdx.x*blockDim.x+threadIdx.x;\n if(row>=n)return;',
                            ' for(int row=0;row<n;row++){', 1)
    source += '\n}\n'
    return '#define min(a,b) ((a)<(b)?(a):(b))\n#define max(a,b) ((a)>(b)?(a):(b))\n' + source


def compile_native(program, directory):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    source = directory / (program.name + '.cpp')
    source.write_text(emit_native(program), encoding='utf-8')
    if os.name == 'nt':
        candidates = list(Path('C:/Program Files/Microsoft Visual Studio/2022').glob('*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        compiler = shutil.which('cl') or (str(sorted(candidates)[-1]) if candidates else None)
        if compiler is None: raise RuntimeError('Native trial requires an installed C++ compiler')
        library = directory / (program.name + '.dll')
        cmd = [compiler, '/nologo', '/LD', '/O2', '/Oi', '/GS-', str(source), '/link', '/NODEFAULTLIB', '/NOENTRY', '/OUT:'+str(library)]
    else:
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None: raise RuntimeError('Native trial requires a C++ compiler')
        library = directory / (program.name + '.so')
        cmd = [compiler, '-O3', '-shared', '-fPIC', str(source), '-o', str(library)]
    result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True, timeout=60)
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
    dll = ctypes.CDLL(str(library)); fn = getattr(dll, program.name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; fn.restype = None
    closed = False
    def close():
        nonlocal closed
        if not closed:
            import _ctypes
            (_ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose)(dll._handle)
            closed = True
    def launch(x, out, steps=1):
        if closed: raise RuntimeError('Native kernel is closed')
        import torch
        width = len(program.fields) if program.fields else 19
        for t in (x, out):
            if t.device.type != 'cpu' or t.dtype != torch.int32 or not t.is_contiguous() or t.ndim != 2 or t.shape[1] != width:
                raise ValueError('Native kernel requires contiguous CPU int32 [N,fields]')
        if x.shape != out.shape or steps < 1: raise ValueError('Invalid native output/steps')
        fn(x.data_ptr(), out.data_ptr(), x.shape[0], steps)
    launch.close = close
    launch.keep = dll
    launch.source = source.read_text(encoding='utf-8')
    return launch
