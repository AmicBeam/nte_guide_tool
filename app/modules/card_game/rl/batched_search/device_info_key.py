"""Device SHA-256 of the exact public observation/candidate byte stream.

Matches information_search.info_key without storing a full candidate matrix
for every tree node. This numerical primitive does not execute game rules.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceKeyResult:
    keys: Any          # int64[B,8], unsigned big-endian digest words
    errors: Any        # int32[B]: 1=invalid count, 2=nonfinite input


_CUDA_SOURCE = r'''
__device__ unsigned int rotate(unsigned int x, int n) { return (x >> n) | (x << (32-n)); }
__constant__ unsigned int round_k[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
extern "C" __global__ void information_keys(
 const float* x, const float* c, const long long* counts, const unsigned char* active,
 long long* output, int* errors, int rows, int d, int m, int width) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows) return;
 errors[row]=0;
 for(int j=0;j<8;j++) output[row*8+j]=0;
 if(!active[row]) return;
 long long count=counts[row];
 if(count<1||count>m){ errors[row]=1; return; }
 const unsigned int* xb=(const unsigned int*)(x+(long long)row*d);
 const unsigned int* cb=(const unsigned int*)(c+(long long)row*m*width);
 for(int j=0;j<d;j++) if((xb[j]&0x7f800000)==0x7f800000){ errors[row]=2; return; }
 for(long long j=0;j<count*width;j++) if((cb[j]&0x7f800000)==0x7f800000){ errors[row]=2; return; }
 const unsigned char* xx=(const unsigned char*)xb;
 const unsigned char* cc=(const unsigned char*)cb;
 unsigned long long split=(unsigned long long)d*4;
 unsigned long long length=split+(unsigned long long)count*width*4;
 unsigned long long padded=((length+9+63)/64)*64;
 unsigned long long bits=length*8;
 unsigned int h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 for(unsigned long long base=0;base<padded;base+=64){
   unsigned int w[64];
   for(int j=0;j<16;j++){
     unsigned int word=0;
     for(int k=0;k<4;k++){
       unsigned long long at=base+j*4+k;
       unsigned int byte=0;
       if(at<length) byte=at<split?xx[at]:cc[at-split];
       else if(at==length) byte=128;
       else if(at>=padded-8) byte=(unsigned int)((bits>>((padded-1-at)*8))&255);
       word=(word<<8)|byte;
     }
     w[j]=word;
   }
   for(int j=16;j<64;j++){
     unsigned int a=w[j-15],b=w[j-2];
     w[j]=w[j-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[j-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
   }
   unsigned int a=h[0],b=h[1],c0=h[2],e=h[4],f=h[5],g=h[6],d0=h[3],hh=h[7];
   for(int j=0;j<64;j++){
     unsigned int t1=hh+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+round_k[j]+w[j];
     unsigned int t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c0)^(b&c0));
     hh=g;g=f;f=e;e=d0+t1;d0=c0;c0=b;b=a;a=t1+t2;
   }
   h[0]+=a;h[1]+=b;h[2]+=c0;h[3]+=d0;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
 }
 for(int j=0;j<8;j++) output[row*8+j]=(long long)h[j];
}
'''


class DeviceInformationKey:
    def __init__(self, *, device='cuda', max_rows=1600, max_input_bytes=16*1024*1024,
                 allow_cpu_reference=False):
        import torch
        if any(type(v) is not int or v < 1 for v in (max_rows, max_input_bytes)):
            raise ValueError('Positive bounded hash dimensions required')
        self.t = torch
        self.device = torch.device(device)
        self.max_rows, self.max_input_bytes = max_rows, max_input_bytes
        self.module = None
        if self.device.type == 'cpu' and allow_cpu_reference is True:
            return
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU is an explicit hash oracle only')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        from ..batched_duel.cuda_runtime import CudaModule
        self.module = CudaModule(_CUDA_SOURCE, {'information_keys': [
            'float32*', 'float32*', 'int64*', 'uint8*', 'int64*', 'int32*',
            'int32', 'int32', 'int32', 'int32']}, device=str(self.device))

    def hash(self, observations, candidates, counts, active):
        t = self.t
        for tensor, dtype, ndim in ((observations, t.float32, 2), (candidates, t.float32, 3),
                                    (counts, t.int64, 1), (active, t.bool, 1)):
            if (not isinstance(tensor, t.Tensor) or tensor.device != self.device
                    or tensor.dtype != dtype or tensor.ndim != ndim or not tensor.is_contiguous()):
                raise ValueError('Exact contiguous bound-device hash inputs required')
        b, d = observations.shape
        if (not 0 < b <= self.max_rows or candidates.shape[0] != b
                or counts.shape != (b,) or active.shape != (b,)
                or d < 1 or candidates.shape[1] < 1 or candidates.shape[2] < 1
                or (d + candidates.shape[1] * candidates.shape[2]) * 4 > self.max_input_bytes):
            raise ValueError('Bounded observation/full-candidate dimensions required')
        if self.device.type == 'cpu':
            return self._reference(observations, candidates, counts, active)
        keys = t.empty((b, 8), dtype=t.int64, device=self.device)
        errors = t.empty(b, dtype=t.int32, device=self.device)
        capture_safe = t.cuda.is_current_stream_capturing()
        event = self.module.launch('information_keys', [observations, candidates, counts,
            active.view(t.uint8), keys, errors, b, d, candidates.shape[1], candidates.shape[2]],
            grid=(b+127)//128, block=128, capture_safe=capture_safe)
        if not capture_safe:
            t.cuda.current_stream(self.device).wait_event(event)
        return DeviceKeyResult(keys, errors)

    def _reference(self, observations, candidates, counts, active):
        import hashlib
        import numpy as np
        x, c, n, on = (v.numpy() for v in (observations, candidates, counts, active))
        keys = np.zeros((len(x), 8), dtype=np.int64)
        errors = np.zeros(len(x), dtype=np.int32)
        for i in range(len(x)):
            if not on[i]:
                continue
            if not 1 <= n[i] <= c.shape[1]:
                errors[i] = 1
            elif not np.isfinite(x[i]).all() or not np.isfinite(c[i, :n[i]]).all():
                errors[i] = 2
            else:
                digest = hashlib.sha256(x[i].tobytes()+c[i, :n[i]].tobytes()).digest()
                keys[i] = np.frombuffer(digest, dtype='>u4').astype(np.int64)
        return DeviceKeyResult(self.t.from_numpy(keys), self.t.from_numpy(errors))

    def close(self):
        if self.module is not None:
            self.module.close()
