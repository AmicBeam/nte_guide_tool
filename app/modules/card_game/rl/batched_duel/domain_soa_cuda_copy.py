"""Direct byte-preserving scatter for validated, unique SoA destinations.

Source rows must be snapshots gathered before writes, including overlapping
source/destination copies. This is a storage kernel, not a rule interpreter.
"""

SOURCE=r'''
extern "C" __global__ void domain_scatter_bytes(
 unsigned char* destination,const long long* slots,
 const unsigned char* source,const unsigned char* active,
 long long width,int capacity,int rows) {
 int row=blockIdx.y;
 if(row>=rows || !active[row])return;
 long long slot=slots[row];
 if(slot<0 || slot>=capacity)return;
 for(long long col=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     col<width;col+=(long long)gridDim.x*blockDim.x)
  destination[slot*width+col]=source[(long long)row*width+col];
}
'''


class DomainCudaCopy:
    def __init__(self,device='cuda:0'):
        from .cuda_runtime import CudaModule
        self.module=CudaModule(SOURCE,{'domain_scatter_bytes':(
            'uint8*','const int64*','const uint8*','const uint8*',
            'int64','int32','int32')},device)

    def scatter(self,destination,slots,source,active):
        import torch as t
        if (destination.device!=self.module.device or source.device!=self.module.device
                or slots.device!=self.module.device or active.device!=self.module.device
                or destination.dtype!=t.uint8 or source.dtype!=t.uint8
                or slots.dtype!=t.int64 or active.dtype!=t.bool
                or destination.ndim!=2 or source.ndim!=2
                or destination.shape[1]!=source.shape[1]
                or slots.shape!=(source.shape[0],)or active.shape!=slots.shape
                or not all(v.is_contiguous()for v in(destination,slots,source,active))):
            raise ValueError('Contiguous same-device packed byte rows and validated selectors required')
        rows,width=source.shape
        if not 1<=rows<=65535 or not 1<=destination.shape[0]<2**31 or width<1:
            raise ValueError('Bounded nonempty byte-copy work required')
        # Use the caller stream in eager mode as well as graph capture, so
        # subsequent metadata writes remain ordered without host synchronization.
        self.module.launch('domain_scatter_bytes',(
            destination,slots,source,active.view(t.uint8),width,destination.shape[0],rows),
            grid=(min((width+255)//256,32),rows,1),block=256,capture_safe=True)

    def close(self):self.module.close()
