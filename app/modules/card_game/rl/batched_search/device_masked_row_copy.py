"""Byte-exact masked copy into independent contiguous CUDA row storage."""
_COPY_CACHE={}
_SOURCE=r'''
extern "C" __global__ void masked_row_copy(const unsigned char* source,unsigned char* target,
 const unsigned char* active,long long rows,long long stride){
 long long count=rows*stride;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(long long)gridDim.x*blockDim.x)
  if(active[i/stride])target[i]=source[i];
}
'''


def prepare_masked_row_copy(device):
    import torch as t
    dev=t.device(device)
    if dev.type!='cuda':raise ValueError('Actual CUDA copy required')
    if dev.index is None:dev=t.device('cuda',t.cuda.current_device())
    key=str(dev)
    if key not in _COPY_CACHE:
        if t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare row copy before capture')
        from ..batched_duel.cuda_runtime import CudaModule
        _COPY_CACHE[key]=CudaModule(_SOURCE,{'masked_row_copy':['uint8*','uint8*','uint8*','int64','int64']},device=key)
    return _COPY_CACHE[key]


def copy_masked_rows_device(source,target,active,module):
    import torch as t
    if (source.device.type!='cuda' or source.device!=target.device or source.device!=active.device
            or source.dtype!=target.dtype or source.shape!=target.shape or source.ndim<1
            or not source.is_contiguous() or not target.is_contiguous()
            or active.dtype!=t.bool or active.shape!=(source.shape[0],) or not active.is_contiguous()):
        raise ValueError('Matching contiguous CUDA rows and boolean mask required')
    if source.numel()==0:return
    if source.untyped_storage().data_ptr()==target.untyped_storage().data_ptr():
        if source.data_ptr()==target.data_ptr():return
        raise ValueError('Overlapping row views need snapshot copy')
    rows=source.shape[0];stride=source.numel()//rows*source.element_size();count=rows*stride
    module.launch('masked_row_copy',[source.reshape(-1).view(t.uint8),target.reshape(-1).view(t.uint8),
        active.view(t.uint8),rows,stride],grid=min(4096,(count+255)//256),block=256,capture_safe=True)
