"""One direct CUDA kernel for already-resolved domain damage fields.

No VM, interpreter, boxed object graph or Python rule callback is used. This
implements only domain_soa_rules.damage_fields, AFTER resistance, redirects and
responses were resolved. It does not implement combat, card effects or a game.
"""
from dataclasses import dataclass

supports_full_duel_rules = False

SOURCE = r'''
extern "C" __global__ void domain_damage(
 const int* hp, const int* shield, const int* amount,
 const unsigned char* bypass, const unsigned char* immune,
 const unsigned char* locked, const int* floor, const int* limit,
 const unsigned char* active, int* ohp, int* oshield, int* absorbed,
 int* loss, long long* hp_lost, unsigned char* lethal, int* errors, int n) {
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= n) return;
 int h=hp[i], s=shield[i];
 int error=active[i] && (h<0 || s<0 || floor[i]<0 || limit[i]<-1);
 errors[i]=error;
 ohp[i]=h; oshield[i]=s; absorbed[i]=0; loss[i]=0;
 hp_lost[i]=0; lethal[i]=0;
 if (!active[i] || error) return;
 int d=immune[i] ? 0 : (amount[i]>0 ? amount[i] : 0);
 if (limit[i]>=0 && d>limit[i]) d=limit[i];
 int a=bypass[i] ? 0 : (s<d ? s : d);
 int l=locked[i] ? 0 : d-a;
 long long next=(long long)h-l;
 if (next<floor[i]) next=floor[i];
 ohp[i]=(int)next; oshield[i]=s-a; absorbed[i]=a; loss[i]=l;
 hp_lost[i]=(long long)h-next; lethal[i]=(h>0 && next<=0);
}
'''


@dataclass(frozen=True)
class DamageBatch:
    inputs: tuple
    result: object
    elements: int


class DomainCudaDamage:
    """Preallocate once, then launch/capture one fused numeric operation."""
    def __init__(self, device='cuda:0'):
        from .cuda_runtime import CudaModule
        self.module = CudaModule(SOURCE, {'domain_damage': (
            'const int32*', 'const int32*', 'const int32*',
            'const uint8*', 'const uint8*', 'const uint8*',
            'const int32*', 'const int32*', 'const uint8*',
            'int32*', 'int32*', 'int32*', 'int32*', 'int64*',
            'uint8*', 'int32*', 'int32')}, device)

    def prepare(self, hp, shield, amount, bypass, immune, life_locked,
                floor, limit, active):
        import torch
        from .domain_soa_rules import DamageFields
        values=(hp, shield, amount, bypass, immune, life_locked, floor, limit, active)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare damage buffers before capture')
        if not isinstance(hp, torch.Tensor) or hp.numel()<1 or hp.numel()>2**31-1:
            raise ValueError('Nonempty bounded resident fields required')
        boolean={3,4,5,8}
        for i,value in enumerate(values):
            if (not isinstance(value,torch.Tensor) or value.shape!=hp.shape
                    or value.device!=self.module.device or not value.is_contiguous()
                    or value.dtype!=(torch.bool if i in boolean else torch.int32)):
                raise ValueError('Exact contiguous same-device damage fields required')
        result=DamageFields(*(torch.empty_like(hp) for _ in range(4)),
            torch.empty_like(hp,dtype=torch.int64),torch.empty_like(hp,dtype=torch.bool),
            torch.empty_like(hp))
        raw=tuple(value.view(torch.uint8) if i in boolean else value
                  for i,value in enumerate(values))
        return DamageBatch(raw,result,hp.numel())

    def launch(self, batch):
        import torch
        from dataclasses import fields
        if not isinstance(batch,DamageBatch):
            raise TypeError('Prepared damage batch required')
        result=batch.result
        outputs=tuple(getattr(result,f.name).view(torch.uint8)
                      if f.name=='lethal' else getattr(result,f.name)
                      for f in fields(result))
        return self.module.launch('domain_damage',(*batch.inputs,*outputs,batch.elements),
            grid=(batch.elements+127)//128,block=128,
            capture_safe=torch.cuda.is_current_stream_capturing())

    def close(self):
        self.module.close()
