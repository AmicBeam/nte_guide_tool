"""Finite/equality row reductions without full float temporaries.

All candidate columns remain present and all requested checks run on device.
No semantic sampling, candidate truncation, or readiness claim.
"""
from hashlib import sha256
_CACHE={}
_SOURCE=r'''
extern "C" __global__ void candidate_nonfinite(const unsigned int* bits,
 unsigned char* output,int rows,int width){
 int row=blockIdx.x;if(row>=rows)return;
 __shared__ unsigned int flags[128];unsigned int bad=0;
 for(int col=threadIdx.x;col<width;col+=blockDim.x)
  bad|=(bits[(long long)row*width+col]&0x7f800000U)==0x7f800000U;
 flags[threadIdx.x]=bad;__syncthreads();
 for(int step=64;step;step>>=1){if(threadIdx.x<step)flags[threadIdx.x]|=flags[threadIdx.x+step];__syncthreads();}
 if(threadIdx.x==0)output[row]=flags[0]!=0;
}
extern "C" __global__ void candidate_mismatch(const float* left,const float* right,
 const unsigned char* lm,const unsigned char* rm,unsigned char* output,
 int rows,int actions,int width){
 int row=blockIdx.x;if(row>=rows)return;
 __shared__ unsigned int flags[128];unsigned int bad=0;
 long long stride=(long long)actions*width;
 for(long long col=threadIdx.x;col<stride;col+=blockDim.x){
  long long mask=(long long)row*actions+col/width;
  if(lm[mask]||rm[mask]){
   long long at=(long long)row*stride+col;float a=left[at],b=right[at];
   unsigned int ab=((const unsigned int*)left)[at],bb=((const unsigned int*)right)[at];
   bad|=(a!=b)||((ab&0x7f800000U)==0x7f800000U)||((bb&0x7f800000U)==0x7f800000U);
  }
 }
 flags[threadIdx.x]=bad;__syncthreads();
 for(int step=64;step;step>>=1){if(threadIdx.x<step)flags[threadIdx.x]|=flags[threadIdx.x+step];__syncthreads();}
 if(threadIdx.x==0)output[row]=flags[0]!=0;
}
'''
ROW_CHECK_SOURCE_ID=sha256(_SOURCE.encode()).hexdigest()


def prepare_candidate_row_checks(device):
    import torch as t
    d=t.device(device)
    if d.type!='cuda':raise ValueError('Actual CUDA module required')
    if d.index is None:d=t.device('cuda',t.cuda.current_device())
    if t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare row-check module before capture')
    key=str(d)
    if key not in _CACHE:
        from ..batched_duel.cuda_runtime import CudaModule
        _CACHE[key]=CudaModule(_SOURCE,{
            'candidate_nonfinite':['uint32*','uint8*','int32','int32'],
            'candidate_mismatch':['float32*','float32*','uint8*','uint8*','uint8*','int32','int32','int32']},device=key)
    return _CACHE[key]


def _module(value):
    module=_CACHE.get(str(value.device))
    if module is None:raise RuntimeError('Row-check module not prepared')
    return module


def candidate_nonfinite_rows(value):
    import torch as t
    if value.dtype!=t.float32 or value.ndim!=3 or not value.is_contiguous():
        raise ValueError('Exact contiguous float32 candidate matrix required')
    if value.device.type=='cpu':return ~t.isfinite(value).all(-1).all(-1)
    b,m,w=value.shape;result=t.empty(b,dtype=t.bool,device=value.device)
    event=_module(value).launch('candidate_nonfinite',[value.view(t.int32),result.view(t.uint8),b,m*w],grid=b,block=128,capture_safe=t.cuda.is_current_stream_capturing())
    if not t.cuda.is_current_stream_capturing():t.cuda.current_stream(value.device).wait_event(event)
    return result


def candidate_mismatch_rows(left,right,lmask,rmask):
    import torch as t
    if (left.dtype!=t.float32 or left.ndim!=3 or right.dtype!=t.float32 or right.shape!=left.shape
        or left.device!=right.device or not left.is_contiguous() or not right.is_contiguous()):
        raise ValueError('Exact paired contiguous float32 candidate matrices required')
    b,m,w=left.shape
    for mask in (lmask,rmask):
        if mask.dtype!=t.bool or mask.device!=left.device or mask.shape!=(b,m) or not mask.is_contiguous():
            raise ValueError('Exact contiguous candidate masks required')
    if left.device.type=='cpu':
        relevant=(lmask|rmask)[:,:,None]
        return (relevant&((left!=right)|~t.isfinite(left)|~t.isfinite(right))).any(-1).any(-1)
    result=t.empty(b,dtype=t.bool,device=left.device)
    event=_module(left).launch('candidate_mismatch',[left,right,lmask.view(t.uint8),rmask.view(t.uint8),result.view(t.uint8),b,m,w],grid=b,block=128,capture_safe=t.cuda.is_current_stream_capturing())
    if not t.cuda.is_current_stream_capturing():t.cuda.current_stream(left.device).wait_event(event)
    return result
