"""Direct byte-preserving scatter for validated, unique SoA destinations.

Source rows must be snapshots gathered before writes, including overlapping
source/destination copies. This is a storage kernel, not a rule interpreter.
"""

SOURCE=r'''
extern "C" __global__ void domain_scatter_bytes(
 unsigned char* destination,const long long* slots,
 const unsigned char* source,const unsigned char* active,
 long long width,long long destination_stride,int capacity,int rows) {
 int row=blockIdx.y;
 if(row>=rows || !active[row])return;
 long long slot=slots[row];
 if(slot<0 || slot>=capacity)return;
 for(long long col=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     col<width;col+=(long long)gridDim.x*blockDim.x)
  destination[slot*destination_stride+col]=source[(long long)row*width+col];
}
extern "C" __global__ void domain_gather_bytes(
 unsigned char* destination,const long long* slots,
 const unsigned char* source,const unsigned char* active,
 long long width,long long source_stride,int capacity,int rows) {
 int row=blockIdx.y;
 if(row>=rows)return;
 long long slot=slots[row];
 int valid=active[row] && slot>=0 && slot<capacity;
 for(long long col=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     col<width;col+=(long long)gridDim.x*blockDim.x)
  destination[(long long)row*width+col]=valid?source[slot*source_stride+col]:0;
}
'''


class DomainCudaCopy:
    def __init__(self,device='cuda:0'):
        from .cuda_runtime import CudaModule
        abi=(
            'uint8*','const int64*','const uint8*','const uint8*',
            'int64','int64','int32','int32')
        self.module=CudaModule(SOURCE,{name:abi for name in ('domain_scatter_bytes','domain_gather_bytes')},device)

    def scatter(self,destination,slots,source,active):
        import torch as t
        if (destination.device!=self.module.device or source.device!=self.module.device
                or slots.device!=self.module.device or active.device!=self.module.device
                or destination.dtype!=t.uint8 or source.dtype!=t.uint8
                or slots.dtype!=t.int64 or active.dtype!=t.bool
                or destination.ndim!=2 or source.ndim!=2
                or destination.shape[1]!=source.shape[1]
                or slots.shape!=(source.shape[0],)or active.shape!=slots.shape
                or destination.stride(1)!=1 or destination.stride(0)<destination.shape[1]
                or not all(v.is_contiguous()for v in(slots,source,active))):
            raise ValueError('Same-device byte rows with contiguous inner columns and validated selectors required')
        rows,width=source.shape
        if not 1<=rows<=65535 or not 1<=destination.shape[0]<2**31 or width<1:
            raise ValueError('Bounded nonempty byte-copy work required')
        # Use the caller stream in eager mode as well as graph capture, so
        # subsequent metadata writes remain ordered without host synchronization.
        # The driver wrapper accepts contiguous pointer arguments. Expose the
        # bounded byte span including row gaps; the kernel skips those gaps.
        span=(destination.shape[0]-1)*destination.stride(0)+width
        raw_destination=destination.as_strided((span,),(1,))
        self.module.launch('domain_scatter_bytes',(
            raw_destination,slots,source,active.view(t.uint8),width,destination.stride(0),destination.shape[0],rows),
            grid=(min((width+255)//256,32),rows,1),block=256,capture_safe=True)

    def gather(self,destination,slots,source,active):
        import torch as t
        if (destination.device!=self.module.device or source.device!=self.module.device
                or slots.device!=self.module.device or active.device!=self.module.device
                or destination.dtype!=t.uint8 or source.dtype!=t.uint8
                or slots.dtype!=t.int64 or active.dtype!=t.bool
                or destination.ndim!=2 or source.ndim!=2
                or destination.shape[1]!=source.shape[1]
                or slots.shape!=(destination.shape[0],)or active.shape!=slots.shape
                or source.stride(1)!=1 or source.stride(0)<source.shape[1]
                or not all(v.is_contiguous()for v in(destination,slots,active))):
            raise ValueError('Independent contiguous output and strided source byte rows required')
        rows,width=destination.shape
        if not 1<=rows<=65535 or not 1<=source.shape[0]<2**31 or width<1:
            raise ValueError('Bounded nonempty byte-copy work required')
        if destination.untyped_storage().data_ptr()==source.untyped_storage().data_ptr():
            raise ValueError('Gather output must not alias source storage')
        span=(source.shape[0]-1)*source.stride(0)+width
        raw_source=source.as_strided((span,),(1,))
        self.module.launch('domain_gather_bytes',(
            destination,slots,raw_source,active.view(t.uint8),width,source.stride(0),source.shape[0],rows),
            grid=(min((width+255)//256,32),rows,1),block=256,capture_safe=True)

    def close(self):self.module.close()
