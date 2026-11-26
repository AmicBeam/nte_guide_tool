"""Device SHA-256 of recovery:{job_seed}:{step}:{actor} per-root Gumbel seeds.

Matches training_rollout.root_search_seed without host hashing, UUID, torch RNG
or truncated uint64 seeds. This numerical primitive does not execute game rules.
"""
from dataclasses import dataclass
from typing import Any


SEED_OK = 0
SEED_ERR_NEGATIVE_STEP = 1
SEED_ERR_INVALID_ACTOR = 2


@dataclass(frozen=True)
class DeviceSeedResult:
    limbs: Any   # int64[B,5], unsigned little-endian 64-bit limbs, bit 256 preserved
    errors: Any  # int32[B]: 1=negative step, 2=invalid actor


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
__device__ int write_u64(unsigned char* dst, unsigned long long v){
 unsigned char tmp[20];
 int n=0;
 do{ tmp[n++]=(unsigned char)('0'+(v%10ULL)); v/=10ULL; }while(v);
 for(int i=0;i<n;i++) dst[i]=tmp[n-1-i];
 return n;
}
__device__ int write_i64(unsigned char* dst, long long v){
 int n=0;
 unsigned long long u;
 if(v<0){ dst[n++]='-'; u=0ULL-(unsigned long long)v; }
 else u=(unsigned long long)v;
 return n+write_u64(dst+n,u);
}
extern "C" __global__ void root_search_seeds(
 const long long* job_seeds, const long long* steps, const long long* actors,
 const unsigned char* active, long long* output, int* errors, int rows) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows) return;
 errors[row]=0;
 for(int j=0;j<5;j++) output[(long long)row*5+j]=0;
 if(!active[row]) return;
 if(steps[row]<0){ errors[row]=1; return; }
 long long actor=actors[row];
 if(actor!=0 && actor!=1){ errors[row]=2; return; }
 unsigned char msg[64];
 int length=0;
 msg[length++]='r'; msg[length++]='e'; msg[length++]='c'; msg[length++]='o';
 msg[length++]='v'; msg[length++]='e'; msg[length++]='r'; msg[length++]='y';
 msg[length++]=':';
 length+=write_i64(msg+length, job_seeds[row]);
 msg[length++]=':';
 length+=write_u64(msg+length, (unsigned long long)steps[row]);
 msg[length++]=':';
 msg[length++]=(unsigned char)(actor?'b':'a');
 unsigned long long bits=(unsigned long long)length*8;
 unsigned long long padded=((length+9+63)/64)*64;
 unsigned int h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 for(unsigned long long base=0;base<padded;base+=64){
   unsigned int w[64];
   for(int j=0;j<16;j++){
     unsigned int word=0;
     for(int k=0;k<4;k++){
       unsigned long long at=base+j*4+k;
       unsigned int byte=0;
       if(at< (unsigned long long)length) byte=msg[at];
       else if(at==(unsigned long long)length) byte=128;
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
 unsigned long long limb0=((unsigned long long)h[6]<<32)|(unsigned long long)h[7];
 unsigned long long limb1=((unsigned long long)h[4]<<32)|(unsigned long long)h[5];
 unsigned long long limb2=((unsigned long long)h[2]<<32)|(unsigned long long)h[3];
 unsigned long long limb3=((unsigned long long)h[0]<<32)|(unsigned long long)h[1];
 output[(long long)row*5+0]=(long long)limb0;
 output[(long long)row*5+1]=(long long)limb1;
 output[(long long)row*5+2]=(long long)limb2;
 output[(long long)row*5+3]=(long long)limb3;
 output[(long long)row*5+4]=1;
}
'''


class DeviceRootSeeds:
    def __init__(self, *, device='cuda', max_rows=1600, allow_cpu_reference=False):
        import torch
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError('Positive bounded seed rows required')
        self.t = torch
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.module = None
        if self.device.type == 'cpu' and allow_cpu_reference is True:
            return
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU is an explicit seed oracle only')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        from ..batched_duel.cuda_runtime import CudaModule
        self.module = CudaModule(_CUDA_SOURCE, {'root_search_seeds': [
            'int64*', 'int64*', 'int64*', 'uint8*', 'int64*', 'int32*', 'int32']},
            device=str(self.device))

    def derive(self, job_seeds, steps, actors, active):
        t = self.t
        for tensor, dtype in ((job_seeds, t.int64), (steps, t.int64),
                              (actors, t.int64), (active, t.bool)):
            if (not isinstance(tensor, t.Tensor) or tensor.device != self.device
                    or tensor.dtype != dtype or tensor.ndim != 1
                    or not tensor.is_contiguous()):
                raise ValueError('Exact contiguous bound-device seed inputs required')
        b = job_seeds.shape[0]
        if (not 0 < b <= self.max_rows or steps.shape != (b,)
                or actors.shape != (b,) or active.shape != (b,)):
            raise ValueError('Bounded seed row count required')
        if self.device.type == 'cpu':
            return self._reference(job_seeds, steps, actors, active)
        limbs = t.empty((b, 5), dtype=t.int64, device=self.device)
        errors = t.empty(b, dtype=t.int32, device=self.device)
        capture_safe = t.cuda.is_current_stream_capturing()
        event = self.module.launch('root_search_seeds', [
            job_seeds, steps, actors, active.view(t.uint8), limbs, errors, b],
            grid=(b + 127) // 128, block=128, capture_safe=capture_safe)
        if not capture_safe:
            t.cuda.current_stream(self.device).wait_event(event)
        return DeviceSeedResult(limbs, errors)

    def _reference(self, job_seeds, steps, actors, active):
        import hashlib
        import numpy as np
        js, st, ac, on = (v.numpy() for v in (job_seeds, steps, actors, active))
        limbs = np.zeros((len(js), 5), dtype=np.int64)
        errors = np.zeros(len(js), dtype=np.int32)
        mask = (1 << 64) - 1
        for i in range(len(js)):
            if not on[i]:
                continue
            if st[i] < 0:
                errors[i] = SEED_ERR_NEGATIVE_STEP
                continue
            if ac[i] != 0 and ac[i] != 1:
                errors[i] = SEED_ERR_INVALID_ACTOR
                continue
            payload = f"recovery:{int(js[i])}:{int(st[i])}:{'b' if ac[i] else 'a'}".encode('utf-8')
            seed = (1 << 256) | int(hashlib.sha256(payload).hexdigest(), 16)
            words = np.empty(5, dtype=np.uint64)
            for w in range(5):
                words[w] = (seed >> (64 * w)) & mask
            limbs[i] = words.view(np.int64)
        return DeviceSeedResult(self.t.from_numpy(limbs), self.t.from_numpy(errors))

    def close(self):
        if self.module is not None:
            self.module.close()
