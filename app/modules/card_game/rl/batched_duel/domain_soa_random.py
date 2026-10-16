"""Direct SoA shuffle columns with the formal CPython RNG semantics.

One native/CUDA routine reads int64 seed[B], int32 count[B] and ordered
int32 values[M,B]. No object graph, interpreter, or host per-root callback.
The explicit CPU mode is a diagnostic backend, never a CUDA fallback.
"""
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .random_source import emit_random_source

INVALID_INPUT, RNG_EXHAUSTED = 1, 2
supports_full_duel_rules = False


def shuffle_source():
    prefix = r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define SOA_FN __device__
#else
#define SOA_FN
#endif
#define BMT_DEVICE SOA_FN
'''
    body = r'''
SOA_FN void soa_shuffle_row(const int* values,const int* counts,const long long* seed,
 const int* active,int* output,long long* next_seed,int* errors,int rows,int width,int row,int max_draws){
 errors[row]=0;next_seed[row]=seed[row];
 for(int slot=0;slot<width;slot++)output[slot*rows+row]=values[slot*rows+row];
 if(!active[row])return;
 int count=counts[row];
 if(count<0||count>width||seed[row]<0){errors[row]=1;return;}
 unsigned long long raw=(unsigned long long)seed[row];
 uint32_t words[2]={(uint32_t)raw,(uint32_t)(raw>>32)};
 bmt_state_t rng;bmt_seed_words(&rng,words,words[1]?2:1);
 for(int slot=count-1;slot>0;slot--){
   unsigned long long n=(unsigned long long)slot+1,draw=0;
   int bits=bmt_bit_length_u64(n),accepted=0;
   for(int attempt=0;attempt<max_draws;attempt++){
     uint64_t candidate=0;bmt_getbits64(&rng,bits,&candidate);
     if(candidate<n){draw=candidate;accepted=1;break;}
   }
   if(!accepted){
     errors[row]=2;
     for(int k=0;k<width;k++)output[k*rows+row]=values[k*rows+row];
     return;
   }
   int other=(int)draw,temp=output[slot*rows+row];
   output[slot*rows+row]=output[other*rows+row];output[other*rows+row]=temp;
 }
 uint64_t next=0;bmt_getbits64(&rng,63,&next);next_seed[row]=(long long)next;
}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_soa_shuffle(const int* values,const int* counts,const long long* seed,
 const int* active,int* output,long long* next_seed,int* errors,int rows,int width,int max_draws){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);
 if(row<rows)soa_shuffle_row(values,counts,seed,active,output,next_seed,errors,rows,width,row,max_draws);
}
#else
extern "C" void domain_soa_shuffle(const int* values,const int* counts,const long long* seed,
 const int* active,int* output,long long* next_seed,int* errors,int rows,int width,int max_draws){
 for(int row=0;row<rows;row++)soa_shuffle_row(values,counts,seed,active,output,next_seed,errors,rows,width,row,max_draws);
}
#endif
'''
    return prefix + emit_random_source() + body


@dataclass(frozen=True)
class ShuffleFields:
    values: object
    seeds: object
    errors: object


class DomainSoARandom:
    """Explicit prepared backend; CUDA calls never transfer tensor contents."""
    gpu_qualified = False

    def __init__(self, *, device='cuda', allow_cpu_reference=False, max_draws=4096):
        import torch
        if type(max_draws) is not int or not 1 <= max_draws <= 1048576:
            raise ValueError('Bounded RNG rejection budget required')
        self.device = torch.device(device)
        self.max_draws = max_draws
        self._temporary = None
        self._module = None
        self._library = None
        self.closed = False
        if self.device.type == 'cuda':
            from .cuda_runtime import CudaModule
            self.device = torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())
            self._module = CudaModule(shuffle_source(), {
                'domain_soa_shuffle': ['int32*', 'int32*', 'int64*', 'int32*', 'int32*',
                    'int64*', 'int32*', 'int32', 'int32', 'int32']}, device=str(self.device))
        elif self.device.type == 'cpu' and allow_cpu_reference:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None:
                raise RuntimeError('Native CPU SoA RNG requires a C++ compiler')
            self._temporary = tempfile.TemporaryDirectory(prefix='nte-soa-rng-')
            root = Path(self._temporary.name)
            source = root / 'random.cpp'
            import sys
            library = root / ('random.dll' if os.name == 'nt' else
                              'random.dylib' if sys.platform == 'darwin' else 'random.so')
            source.write_text(shuffle_source(), encoding='utf-8')
            try:
                result = subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-O2',
                    str(source), '-o', str(library)], capture_output=True, text=True, timeout=60)
            except BaseException:
                self.close()
                raise
            if result.returncode:
                self.close()
                raise RuntimeError('Native SoA RNG compilation failed: ' + result.stderr)
            self._library = ctypes.CDLL(str(library))
            self._kernel = self._library.domain_soa_shuffle
            self._kernel.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*3
            self._kernel.restype = None
        else:
            raise ValueError('CUDA or explicitly enabled CPU reference required')

    def shuffle(self, values, counts, seeds, active):
        import torch
        if self.closed:
            raise RuntimeError('SoA RNG is closed')
        if (not isinstance(values, torch.Tensor) or values.dtype != torch.int32 or values.ndim != 2
                or values.device != self.device or not values.is_contiguous()):
            raise ValueError('Contiguous resident int32[slots,batch] required')
        width, rows = values.shape
        if rows < 1 or width < 1:
            raise ValueError('Nonempty bounded batch and slots required')
        if rows * width > 2147483647:
            raise ValueError('Shuffle slot addressing exceeds exact int32 range')
        for name, item, dtype in (('counts', counts, torch.int32), ('seeds', seeds, torch.int64), ('active', active, torch.int32)):
            if (not isinstance(item, torch.Tensor) or item.dtype != dtype or item.device != self.device
                    or item.shape != (rows,) or not item.is_contiguous()):
                raise ValueError('Exact resident RNG column required: ' + name)
        output, next_seed = torch.empty_like(values), torch.empty_like(seeds)
        errors = torch.empty_like(counts)
        if self._module is not None:
            self._module.launch('domain_soa_shuffle', [values, counts, seeds, active, output,
                next_seed, errors, rows, width, self.max_draws], grid=((rows+63)//64,), block=(64,))
        else:
            # CPU-only native ABI: tensors already reside on CPU. No GPU fallback.
            self._kernel(*(ctypes.c_void_p(item.data_ptr()) for item in
                (values, counts, seeds, active, output, next_seed, errors)), rows, width, self.max_draws)
        return ShuffleFields(output, next_seed, errors)

    def close(self):
        if self.closed:
            return
        if self._module is not None:
            self._module.close()
        if self._library is not None:
            import _ctypes
            (_ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose)(self._library._handle)
            self._library = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
