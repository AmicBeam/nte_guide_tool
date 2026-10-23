"""Resident monotonic deadlines, shared by all three H3 steps and simulations."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

_SOURCE=r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define DDEV __device__
#define DKERNEL extern "C" __global__
DDEV unsigned long long d_now_ns(){unsigned long long result;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(result));return result;}
#else
#include <chrono>
#define DDEV inline
#define DKERNEL extern "C"
DDEV unsigned long long d_now_ns(){return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#endif
DKERNEL void domain_deadline_start(long long*deadline,const double*milliseconds,int B){
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=B)return;
#else
 for(int row=0;row<B;row++){
#endif
 double ms=milliseconds[row];deadline[row]=ms==ms&&ms>0&&ms<=86400000.?(long long)(d_now_ns()+(unsigned long long)(ms*1000000.)):0;
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
 }
#endif
}
DKERNEL void domain_deadline_live(const long long*deadline,const unsigned char*active,unsigned char*live,int B){
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=B)return;
#else
 for(int row=0;row<B;row++){
#endif
 live[row]=active[row]&&deadline[row]>0&&d_now_ns()<(unsigned long long)deadline[row];
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
 }
#endif
}
'''


class DomainDeviceDeadline:
    """Explicit start; a leaf evaluation never resets the parent search budget."""
    def __init__(self,rows,*,device='cuda',allow_cpu_oracle=False):
        import torch
        self.t=torch;self.device=torch.device(device);self.rows=rows;self.closed=False;self.module=None;self.library=None;self.temp=None
        if type(rows)is not int or not 1<=rows<=400:raise ValueError('Bounded deadline rows required')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            self.module=CudaModule(_SOURCE,{'domain_deadline_start':['int64*','float64*','int32'],'domain_deadline_live':['int64*','uint8*','uint8*','int32']},device=str(self.device))
        elif self.device.type=='cpu'and allow_cpu_oracle is True:
            compiler=shutil.which('clang++')or shutil.which('g++')
            if not compiler:raise RuntimeError('C++ compiler required')
            self.temp=tempfile.TemporaryDirectory(prefix='nte-domain-deadline-');root=Path(self.temp.name)
            (root/'deadline.cpp').write_text(_SOURCE)
            r=subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++17',str(root/'deadline.cpp'),'-o',str(root/'deadline.so')],capture_output=True,text=True)
            if r.returncode:self.temp.cleanup();raise RuntimeError(r.stderr)
            self.library=ctypes.CDLL(str(root/'deadline.so'));self.start_kernel=self.library.domain_deadline_start;self.live_kernel=self.library.domain_deadline_live
            self.start_kernel.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];self.live_kernel.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]
        else:raise RuntimeError('CUDA or explicitly selected C++ deadline oracle required')
        self.deadline=torch.zeros(rows,dtype=torch.int64,device=self.device);self.live=torch.zeros(rows,dtype=torch.bool,device=self.device)
    def start(self,milliseconds):
        t=self.t
        if self.closed or milliseconds.device!=self.device or milliseconds.dtype!=t.float64 or milliseconds.shape!=(self.rows,):raise ValueError('Resident float64 budget required')
        if self.module:self.module.launch('domain_deadline_start',[self.deadline,milliseconds,self.rows],grid=((self.rows+127)//128,),block=(128,),capture_safe=True)
        else:self.start_kernel(ctypes.c_void_p(self.deadline.data_ptr()),ctypes.c_void_p(milliseconds.data_ptr()),self.rows)
    def refresh(self,active):
        if self.closed or active.device!=self.device or active.dtype!=self.t.bool or active.shape!=(self.rows,):raise ValueError('Resident deadline active mask required')
        if self.module:self.module.launch('domain_deadline_live',[self.deadline,active.view(self.t.uint8),self.live.view(self.t.uint8),self.rows],grid=((self.rows+127)//128,),block=(128,),capture_safe=True)
        else:self.live_kernel(ctypes.c_void_p(self.deadline.data_ptr()),ctypes.c_void_p(active.data_ptr()),ctypes.c_void_p(self.live.data_ptr()),self.rows)
        return self.live
    def close(self):
        if self.closed:return
        if self.module:self.module.close()
        if self.library:
            import _ctypes
            _ctypes.dlclose(self.library._handle)
        if self.temp:self.temp.cleanup()
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
