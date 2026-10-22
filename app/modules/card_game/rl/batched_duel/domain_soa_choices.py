"""Direct legal-choice enumeration over resolved domain fields.

This stage produces exactly ordered choice payloads. Deck-order constraints
must first pass the formal kit decision stage; an unresolved spec is rejected.
It does not execute card choices or provide complete playing-phase legality.
"""
from dataclasses import dataclass
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

INVALID = 1
OVERFLOW = 2
UNRESOLVED_SPEC = 4
supports_full_duel_rules = False

# Inputs: kind, count, max_select, min_bottom, max_bottom, has_max_bottom,
# multi_bottom, hand_placement, deck_order, spec_resolved, actor, finished, active.
# kind follows domain_soa_schema.CHOICE_KINDS, bound at prepare time.
BODY = r'''
CHOICE_FN int choose_count(int n,int k,int cap){
 if(k<0||k>n)return 0;
 if(k>n-k)k=n-k;
 long long v=1;
 for(int i=1;i<=k;i++){
  v=v*(n-k+i)/i;
  if(v>cap)return cap+1;
 }
 return (int)v;
}
CHOICE_FN int factorial_count(int n,int cap){
 int v=1;
 for(int i=2;i<=n;i++){if(v>cap/i)return cap+1;v*=i;}
 return v;
}
CHOICE_FN void choice_metadata(const int* input,const int* refs,
 int* counts,int* errors,int refs_width,int actions,int payload,int kinds,
 int redraw_kind,int multi_kind,int row){
 const int* p=input+row*13;
 int error=0,count=0;
 if(!p[12]||p[11]){counts[row]=0;errors[row]=0;return;}
 int n=p[1],k=p[2],deck=p[8],min_bottom=0,max_bottom=0,perms=0;
 // A viewer who is not currently choosing has only concede, regardless of
 // private choice payload. Hidden choice data cannot affect their result.
 if(!p[10])count=1;
 else if(p[0]<0||p[0]>=kinds||n<0||n>refs_width)error=1;
 else {
  // Choices are physical references, not template IDs. Duplicate IDs would
  // make formal apply_action payloads ambiguous and are invalid at ingress.
  for(int i=0;i<n;i++){
   if(refs[row*refs_width+i]<0)error=1;
   for(int j=0;j<i;j++)if(refs[row*refs_width+j]==refs[row*refs_width+i])error=1;
  }
  if(!error&&deck){
   if(!p[9])error=4;
   else if(p[7])count=1+2*n;
   else {
    min_bottom=p[3]>0?p[3]:0;
    max_bottom=p[5]?p[4]:(p[6]?n:1);
    if(max_bottom>n)max_bottom=n;
    perms=factorial_count(n,actions);
    int groups=max_bottom-min_bottom+1;
    if(groups<=0)count=1;
    else if(n>payload||perms>actions||groups>(actions-1)/perms)error=2;
    else count=1+groups*perms;
   }
  }else if(!error&&p[0]==redraw_kind){
   count=1;
   for(int size=0;size<=3&&size<=n;size++){
    int ways=choose_count(n,size,actions);
    if(ways>actions-count){error=2;break;}
    count+=ways;
   }
   if(payload<3&&n>payload)error=2;
  }else if(!error&&p[0]==multi_kind){
   int ways=choose_count(n,k,actions);
   if(k<0||k>payload)error=1;
   else if(ways>actions-1)error=2;
   else count=1+ways;
  }else if(!error)count=1+n;
 }
 if(!error&&count>actions)error=2;
 counts[row]=error?0:count;errors[row]=error;

}
CHOICE_FN void enumerate_choice(const int* input,const int* refs,int* tokens,
 int* selections,const int* counts,const int* errors,int rows,int refs_width,int actions,
 int payload,int kinds,int redraw_kind,int multi_kind,int row,int action){
 const int* p=input+row*13;
 long long out=((long long)row*actions+action);
 for(int j=0;j<7;j++)tokens[out*7+j]=-1;
 for(int j=0;j<payload;j++)selections[out*payload+j]=-1;
 if(errors[row]||action>=counts[row])return;
 int n=p[1],k=p[2],deck=p[8];
 int min_bottom=p[3]>0?p[3]:0;
 int perms=deck&&!p[7]?factorial_count(n,actions):0;
 if(action==0){tokens[out*7]=9;return;}
 int rank=action-1;
 if(deck){
  tokens[out*7]=6;
  if(p[7]){
   int which=rank/2,bottom=rank%2;
   selections[out*payload]=which;
   tokens[out*7+2]=bottom?0:1;
   tokens[out*7+3]=bottom?1:0;
   tokens[out*7+4]=1; // hand_placement always uses bottom_card_ids
  }else {
   int bottom=min_bottom+rank/perms;
   rank%=perms;
   int available[64];for(int i=0;i<n;i++)available[i]=i;
   for(int i=0;i<n;i++){
    int block=factorial_count(n-i-1,actions);
    int index=rank/block;rank%=block;
    selections[out*payload+i]=available[index];
    for(int j=index;j<n-i-1;j++)available[j]=available[j+1];
   }
   tokens[out*7+2]=n-bottom;tokens[out*7+3]=bottom;tokens[out*7+4]=p[6]?1:0;
  }
 }else if(p[0]==redraw_kind||p[0]==multi_kind){
  tokens[out*7]=8;
  if(p[0]==redraw_kind){
   for(k=0;k<=3&&k<=n;k++){
    int ways=choose_count(n,k,actions);
    if(rank<ways)break;
    rank-=ways;
   }
  }
  tokens[out*7+2]=k;
  int next=0;
  for(int i=0;i<k;i++){
   for(int j=next;j<n;j++){
    int block=choose_count(n-j-1,k-i-1,actions);
    if(rank<block){selections[out*payload+i]=j;next=j+1;break;}
    rank-=block;
   }
  }
 }else {
  tokens[out*7]=4;tokens[out*7+1]=refs[row*refs_width+rank];
  tokens[out*7+2]=1;selections[out*payload]=rank;
 }
}
'''


def source():
    return r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CHOICE_FN __device__
#else
#define CHOICE_FN
#endif
''' + BODY + r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_choice_metadata(const int* input,const int* refs,
 int* counts,int* errors,int rows,int width,int actions,int payload,int kinds,
 int redraw_kind,int multi_kind){
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row<rows)choice_metadata(input,refs,counts,errors,width,actions,payload,kinds,redraw_kind,multi_kind,row);
}
extern "C" __global__ void domain_choices(const int* input,const int* refs,
 int* tokens,int* selections,int* counts,int* errors,int rows,int width,
 int actions,int payload,int kinds,int redraw_kind,int multi_kind){
 int row=blockIdx.y,action=blockIdx.x*blockDim.x+threadIdx.x;
 if(row<rows&&action<actions)enumerate_choice(input,refs,tokens,selections,
  counts,errors,rows,width,actions,payload,kinds,redraw_kind,multi_kind,row,action);
}
#else
extern "C" void domain_choices(const int* input,const int* refs,
 int* tokens,int* selections,int* counts,int* errors,int rows,int width,
 int actions,int payload,int kinds,int redraw_kind,int multi_kind){
 for(int row=0;row<rows;row++)choice_metadata(input,refs,counts,errors,width,actions,
   payload,kinds,redraw_kind,multi_kind,row);
 for(int row=0;row<rows;row++)for(int a=0;a<actions;a++)
  enumerate_choice(input,refs,tokens,selections,counts,errors,rows,width,
   actions,payload,kinds,redraw_kind,multi_kind,row,a);
}
#endif
'''


@dataclass(frozen=True)
class ChoiceBatch:
    owner: object
    inputs: object
    references: object
    tokens: object
    selections: object
    counts: object
    errors: object
    binding: object = None


class DomainChoiceActions:
    """Preallocate and capture enumeration; no data leaves the chosen device.

    tokens[...,0] is the action type (9 concede, 4 choose, 8 choose_cards,
    6 choose_deck_order). Column1 is the instance reference for choose;
    columns2/3 are selected-top/bottom counts; column4 selects plural bottom
    payloads. selections stores original choice-list ordinals in exact order.
    Columns5/6 remain unused. This is a domain payload, not model features.
    """
    supports_full_duel_rules = False
    resolves_deck_order_spec = False
    def __init__(self, *, device='cuda', allow_cpu_reference=False,
                 max_actions=800, selection_width=64):
        import torch
        from .domain_soa_schema import CHOICE_KINDS
        if type(max_actions) is not int or not 1 <= max_actions <= 800:
            raise ValueError('Full candidate capacity must be 1..800')
        if type(selection_width) is not int or not 1 <= selection_width <= 64:
            raise ValueError('Bounded selection payload must be 1..64')
        self.device=torch.device(device)
        self.max_actions,self.selection_width=max_actions,selection_width
        self.kinds,self.redraw,self.multi=len(CHOICE_KINDS),CHOICE_KINDS.index('hand_redraw'),CHOICE_KINDS.index('kit_multi')
        self.closed=False;self.module=None;self.library=None;self.temporary=None
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
            self.device=torch.device('cuda',self.device.index if self.device.index is not None else torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            self.module=CudaModule(source(),{'domain_choices':('const int32*','const int32*',
                'int32*','int32*','int32*','int32*',*('int32',)*7),
                'domain_choice_metadata':('const int32*','const int32*','int32*','int32*',*('int32',)*7)},str(self.device))
        elif self.device.type=='cpu' and allow_cpu_reference:
            compiler=shutil.which('clang++')or shutil.which('g++')
            if not compiler:raise RuntimeError('Explicit CPU oracle requires a C++ compiler')
            self.temporary=tempfile.TemporaryDirectory(prefix='nte-domain-choice-')
            root=Path(self.temporary.name);cpp=root/'choices.cpp'
            lib=root/('choices.dylib'if sys.platform=='darwin'else 'choices.so')
            cpp.write_text(source())
            try:
                subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(lib)],
                               capture_output=True,text=True,check=True,timeout=60)
                self.library=ctypes.CDLL(str(lib));self.kernel=self.library.domain_choices
                self.kernel.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*7;self.kernel.restype=None
            except BaseException:
                self.close();raise
        else:raise ValueError('CUDA or explicitly enabled CPU oracle required')

    def prepare(self, inputs, references):
        import torch
        if self.closed:raise RuntimeError('Choice backend is closed')
        if self.device.type=='cuda'and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare buffers before capture')
        if (not isinstance(inputs,torch.Tensor)or inputs.dtype!=torch.int32
                or inputs.device!=self.device or inputs.ndim!=2 or inputs.shape[1]!=13
                or not 1<=inputs.shape[0]<=65535 or not inputs.is_contiguous()):
            raise ValueError('Resident contiguous int32[B,13] resolved choice inputs required')
        b=inputs.shape[0]
        if (not isinstance(references,torch.Tensor)or references.dtype!=torch.int32
                or references.device!=self.device or references.ndim!=2
                or references.shape[0]!=b or not 1<=references.shape[1]<=64
                or not references.is_contiguous()):
            raise ValueError('Resident contiguous int32[B,K<=64] instance references required')
        return ChoiceBatch(self,inputs,references,
            torch.empty((b,self.max_actions,7),dtype=torch.int32,device=self.device),
            torch.empty((b,self.max_actions,self.selection_width),dtype=torch.int32,device=self.device),
            torch.empty(b,dtype=torch.int32,device=self.device),torch.empty(b,dtype=torch.int32,device=self.device))

    def _fields_inputs(self, fields, viewers, active, *, spec_resolved):
        """Bind named domain columns; kit-resolved deck specs stay explicit.

        spec_resolved is a device bool[B] supplied by the kit decision stage.
        Passing raw pending fields for an unresolved deck spec fails closed.
        """
        import torch as t
        from .domain_soa_schema import CHOICE_INT_KEYS, CHOICE_BOOL_KEYS, CHOICE_KINDS, PRESENCE_VALUE
        fields = fields.arrays if hasattr(fields, 'arrays') else fields
        if not isinstance(fields, dict):
            fields = dict(fields)
        if (not isinstance(viewers,t.Tensor) or viewers.device!=self.device
                or viewers.dtype!=t.int64 or viewers.ndim!=1):
            raise ValueError('Resident int64 viewer rows required')
        b=viewers.shape[0]
        for value in (active,spec_resolved):
            if (not isinstance(value,t.Tensor) or value.device!=self.device
                    or value.dtype!=t.bool or value.shape!=(b,)):
                raise ValueError('Resident active/resolved bool rows required')
        specs={
            'row_valid':(t.bool,(b,)), 'row_error':(t.int32,(b,)),
            'phase':(t.int16,(b,)), 'pending_choice_present':(t.bool,(b,)),
            'choice_kind':(t.int16,(b,)), 'choice_side':(t.int8,(b,)),
            'choice_card_count':(t.int16,(b,)),
            'choice_int_presence':(t.uint8,(b,len(CHOICE_INT_KEYS))),
            'choice_int_value':(t.int32,(b,len(CHOICE_INT_KEYS))),
            'choice_bool_presence':(t.uint8,(b,len(CHOICE_BOOL_KEYS))),
            'choice_bool_value':(t.bool,(b,len(CHOICE_BOOL_KEYS))),
        }
        for name,(dtype,shape) in specs.items():
            value=fields[name]
            if (not isinstance(value,t.Tensor)or value.dtype!=dtype
                    or value.device!=self.device or value.shape!=shape):
                raise ValueError('Exact resident domain column required: '+name)
        refs=fields['choice_card_index']
        if (not isinstance(refs,t.Tensor)or refs.dtype!=t.int32 or refs.ndim!=2
                or refs.device!=self.device or refs.shape[0]!=b):
            raise ValueError('Resident choice instance indices required')
        valid_cards=fields['card_valid']
        if (not isinstance(valid_cards,t.Tensor)or valid_cards.dtype!=t.bool
                or valid_cards.device!=self.device or valid_cards.ndim!=2
                or valid_cards.shape[0]!=b or valid_cards.shape[1]<1):
            raise ValueError('Resident card instance validity table required')
        ip,iv=fields['choice_int_presence'],fields['choice_int_value']
        bp,bv=fields['choice_bool_presence'],fields['choice_bool_value']
        def integer(name):
            i=CHOICE_INT_KEYS.index(name)
            return t.where(ip[:,i]==PRESENCE_VALUE,iv[:,i],0)
        def boolean(name):
            if name not in CHOICE_BOOL_KEYS:
                return t.zeros(b,dtype=t.bool,device=self.device)
            i=CHOICE_BOOL_KEYS.index(name)
            return (bp[:,i]==PRESENCE_VALUE)&bv[:,i]
        kind=fields['choice_kind'].int()
        finished=fields['phase']==3
        deck=boolean('deck_order')|(kind==(CHOICE_KINDS.index('heiyu_divination') if 'heiyu_divination' in CHOICE_KINDS else -1))
        actor=viewers==fields['choice_side'].long()
        multi=kind==CHOICE_KINDS.index('kit_multi')
        hasmax=ip[:,CHOICE_INT_KEYS.index('max_bottom')]!=0
        missing_k=ip[:,CHOICE_INT_KEYS.index('max_select')]!=PRESENCE_VALUE
        null_max=hasmax&(ip[:,CHOICE_INT_KEYS.index('max_bottom')]!=PRESENCE_VALUE)
        invalid=(~fields['row_valid']|(fields['row_error']!=0)|(viewers<0)|(viewers>1)
                 |(~finished&((fields['phase']!=2)|~fields['pending_choice_present']))
                 |(actor&multi&missing_k)|(actor&deck&null_max))
        positions=t.arange(refs.shape[1],device=self.device)[None,:]
        present=positions<fields['choice_card_count'].int()[:,None]
        in_range=(refs>=0)&(refs<valid_cards.shape[1])
        valid=valid_cards.gather(1,refs.long().clamp(0,valid_cards.shape[1]-1))
        invalid=invalid|(actor&(present&(~in_range|~valid)).any(1))
        count=t.where(invalid,-1,fields['choice_card_count'].int())
        # Force invalid rows through validation even if the malformed actor
        # metadata would otherwise select the non-actor concede-only branch.
        actor=actor|invalid
        inputs=t.stack((kind,count,integer('max_select'),integer('min_bottom'),
            integer('max_bottom'),hasmax.int(),boolean('multi_bottom').int(),
            boolean('hand_placement').int(),deck.int(),spec_resolved.int(),
            actor.int(),finished.int(),active.int()),dim=1).contiguous()
        return inputs,refs.contiguous()

    def prepare_from_fields(self,fields,viewers,active,*,spec_resolved):
        import torch
        from dataclasses import replace
        if self.closed:raise RuntimeError('Choice backend is closed')
        if self.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare buffers before capture')
        inputs,refs=self._fields_inputs(fields,viewers,active,spec_resolved=spec_resolved)
        return replace(self.prepare(inputs,refs),binding=(fields,viewers,active,spec_resolved))

    def launch(self,batch):
        if self.closed:raise RuntimeError('Choice backend is closed')
        if not isinstance(batch,ChoiceBatch)or batch.owner is not self:
            raise ValueError('Prepared choice buffers from this backend required')
        if batch.binding is not None:
            fields,viewers,active,resolved=batch.binding
            inputs,refs=self._fields_inputs(fields,viewers,active,spec_resolved=resolved)
            batch.inputs.copy_(inputs);batch.references.copy_(refs)
        b,k=batch.references.shape
        tensors=(batch.inputs,batch.references,batch.tokens,batch.selections,batch.counts,batch.errors)
        numbers=(b,k,self.max_actions,self.selection_width,self.kinds,self.redraw,self.multi)
        if self.module is not None:
            self.module.launch('domain_choice_metadata',(batch.inputs,batch.references,batch.counts,batch.errors,*numbers),
                grid=((b+127)//128,),block=128,capture_safe=True)
            self.module.launch('domain_choices',(*tensors,*numbers),
                grid=((self.max_actions+127)//128,b,1),block=128,capture_safe=True)
        else:self.kernel(*(ctypes.c_void_p(v.data_ptr())for v in tensors),*numbers)
        return batch

    def close(self):
        if self.closed:return
        if self.module:self.module.close()
        if self.library:
            import _ctypes
            (_ctypes.FreeLibrary if sys.platform=='win32'else _ctypes.dlclose)(self.library._handle)
            self.library=None
        if self.temporary:self.temporary.cleanup();self.temporary=None
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
