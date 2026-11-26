"""Direct physical state primitives shared by the unfinished no-VM executor.

Draw uses the formal public-kit hand-added hook, overflow discard and empty-deck
terminal rules. C++ CPU compilation is an explicit oracle of the same CUDA
source, never a runtime rules fallback. This is not a complete action executor.
Presentation payloads are omitted; authoritative event counters are retained.
"""
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .domain_soa_schema import (DomainSoALayout,array_specs,catalog_fingerprint,
    CARD_VOCAB,CARD_BOOL_FIELDS,CARD_STR_FIELDS,CHARACTER_VOCAB,
    CHARACTER_BOOL_FIELDS,PLAYER_BOOL_FIELDS)
from .random_source import emit_random_source

supports_full_duel_rules=False
INVALID=1
CAPACITY=2
RNG_EXHAUSTED=4
UNSUPPORTED=8
TYPES=('tactic','battle','form')
_FIELDS=('row_valid','catalog_digest','row_error','phase','rng','event_seq',
    'winner_presence','winner','reason_presence','reason_slot',
    'zone_count','zone_card_index','card_valid','card_is_view','card_template',
    'card_bool_presence','card_bool_value','card_str_presence','card_str_slot',
    'card_jingu_mark_presence','card_jingu_delta','card_jingu_source_slot',
    'player_character_count','character_valid','character_template','character_hp',
    'character_entity_slot','character_bool_presence','character_bool_value',
    'flag_mimic_presence','flag_mimic_id_presence','flag_mimic_template',
    'player_bool_presence','player_bool_value','string_count','string_bytes_used',
    'string_offset','string_length','string_bytes')
_C_TYPES={'bool':'unsigned char','uint8':'unsigned char','int8':'signed char',
    'int16':'short','int32':'int','int64':'long long','float64':'double'}


def physical_source(layout,max_draws,max_rng_draws):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    import numpy as np
    specs={k:(np.dtype(dtype),shape)for k,(dtype,shape)in array_specs(layout).items()}
    n=max(CARD_VOCAB.values())+1
    types=[-1]*n
    for cid,index in CARD_VOCAB.items():
        name=lookup_card(cid).get('type')
        if name not in TYPES:raise ValueError('Unknown physical card type')
        types[index]=TYPES.index(name)
    dims=dict(C=layout.max_characters,K=layout.max_cards,Z=layout.max_zone,
        SS=layout.max_string_slots,SB=layout.max_string_bytes,NT=n,
        NB=len(CHARACTER_BOOL_FIELDS),KB=len(CARD_BOOL_FIELDS),KS=len(CARD_STR_FIELDS),
        PB=len(PLAYER_BOOL_FIELDS),AW=CHARACTER_BOOL_FIELDS.index('awakened'),
        COPY=CARD_BOOL_FIELDS.index('copy'),TYPE=CARD_STR_FIELDS.index('type'),
        EMPTY_WIN=PLAYER_BOOL_FIELDS.index('empty_draw_wins'),XIAOZHI=CHARACTER_VOCAB['xiaozhi'],
        MAX_DRAW=max_draws,MAX_RNG=max_rng_draws)
    prefix='''#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define PFN __device__
#define BMT_DEVICE __device__
#else
#define PFN
#define BMT_DEVICE
#endif
'''+emit_random_source()
    prefix+='\n'+''.join('#define '+k+' '+str(v)+'\n'for k,v in dims.items())
    prefix+='typedef struct {\n'+''.join(_C_TYPES[specs[k][0].name]+'* '+k+';\n'for k in _FIELDS)+'} PRow;\n'
    prefix+='PFN int physical_type(int t){switch(t){'+''.join('case '+str(i)+':return '+str(v)+';'for i,v in enumerate(types))+'default:return -1;}}\n'
    body=r'''
PFN int eq(PRow*f,int slot,const char*text,int*err){
 if(slot<0||slot>=f->string_count[0]||slot>=SS){*err=INVALID;return 0;}
 int at=f->string_offset[slot],n=f->string_length[slot];
 if(at<0||n<0||at>SB||n>SB||at+n>f->string_bytes_used[0]||at+n>SB){*err=INVALID;return 0;}
 int j=0;for(;text[j];j++)if(j>=n||f->string_bytes[at+j]!=(unsigned char)text[j])return 0;return j==n;
}
PFN int intern(PRow*f,const char*text,int*err){
 int n=0;while(text[n])n++;
 if(f->string_count[0]<0||f->string_count[0]>SS||f->string_bytes_used[0]<0||f->string_bytes_used[0]>SB){*err=INVALID;return 0;}
 for(int j=0;j<f->string_count[0];j++)if(eq(f,j,text,err))return j;
 if(*err)return 0;
 int slot=f->string_count[0],at=f->string_bytes_used[0];
 if(slot>=SS||at+n>SB){*err=CAPACITY;return 0;}
 f->string_count[0]++;f->string_bytes_used[0]+=n;f->string_offset[slot]=at;f->string_length[slot]=n;
 for(int j=0;j<n;j++)f->string_bytes[at+j]=(unsigned char)text[j];return slot;
}
PFN int card_type(PRow*f,int ref,int*err){
 if(ref<0||ref>=K||!f->card_valid[ref]||f->card_is_view[ref]){*err=INVALID;return -1;}
 int p=f->card_str_presence[ref*KS+TYPE],v=f->card_template[ref];
 if(v<=0||v>=NT){*err=INVALID;return -1;}
 if(p==3)return physical_type(v);if(p==0||p==1)return -1;
 if(p!=2){*err=INVALID;return -1;}int slot=f->card_str_slot[ref*KS+TYPE];
 if(eq(f,slot,"tactic",err))return 0;if(eq(f,slot,"battle",err))return 1;if(eq(f,slot,"form",err))return 2;return -1;
}
PFN int log_event(PRow*f){if(f->event_seq[0]==2147483647)return CAPACITY;f->event_seq[0]++;return 0;}
PFN int add_hand_hook(PRow*f,int side,int card){
 for(int c=0;c<f->player_character_count[side];c++){
  int h=side*C+c;
  if(!f->character_valid[h]||f->character_template[h]!=XIAOZHI||f->character_hp[h]<=0)continue;
  if(f->character_bool_presence[h*NB+AW]!=2||!f->character_bool_value[h*NB+AW])continue;
  int holder=side*(C+1)+1+c;
  /* Every registered mimic wrapper temporarily forces awakened=False. A valid
     public mimic id therefore suppresses this actual Xiaozhi hand-added hook. */
  if(f->flag_mimic_presence[holder]==2&&f->flag_mimic_id_presence[holder]==2&&f->flag_mimic_template[holder]>0)continue;
  long long seed=f->rng[0];if(seed<0)return INVALID;
  bmt_state_t random;uint32_t words[2]={(uint32_t)seed,(uint32_t)((unsigned long long)seed>>32)};
  bmt_seed_words(&random,words,words[1]?2:1);uint64_t value=0;int found=0;
  for(int j=0;j<MAX_RNG;j++){bmt_getbits64(&random,2,&value);if(value<3){found=1;break;}}
  if(!found)return RNG_EXHAUSTED;
  f->card_jingu_mark_presence[card]=2;f->card_jingu_delta[card]=(int)value-1;
  f->card_jingu_source_slot[card]=f->character_entity_slot[h];
  bmt_getbits64(&random,63,&value);f->rng[0]=(long long)value;
  int err=log_event(f);if(err)return err;
 }return 0;
}
PFN int physical_draw(PRow*f,int side,int count,int filter,int*drew){
 if(side<0||side>1||count<0||count>MAX_DRAW||filter< -1||filter>2)return INVALID;
 if(f->phase[0]<0||f->phase[0]>3)return INVALID;
 if(f->phase[0]==3||count==0)return 0;
 int seen[K];for(int k=0;k<K;k++)seen[k]=0;
 for(int s=0;s<2;s++){
  if(f->player_character_count[s]<4||f->player_character_count[s]>C)return INVALID;
  for(int c=0;c<f->player_character_count[s];c++)if(!f->character_valid[s*C+c])return INVALID;
  for(int z=0;z<4;z++){
   int count=f->zone_count[s*4+z];if(count<0||count>Z)return INVALID;
   for(int p=0;p<count;p++){
    int ref=f->zone_card_index[(s*4+z)*Z+p];
    if(ref<0||ref>=K||!f->card_valid[ref]||f->card_is_view[ref]||seen[ref])return INVALID;
    seen[ref]=1;
   }
  }
 }
 for(int step=0;step<count;step++){
  int dn=f->zone_count[side*4+1],at=0,err=0;
  if(filter>=0){
   at=-1;for(int p=0;p<dn;p++){int ref=f->zone_card_index[(side*4+1)*Z+p];int type=card_type(f,ref,&err);if(err)return err;if(type==filter){at=p;break;}}
   if(at<0)break;
  }
  if(dn==0){
   int wins=f->player_bool_presence[side*PB+EMPTY_WIN]==2&&f->player_bool_value[side*PB+EMPTY_WIN];
   int reason=intern(f,wins?"empty_draw_win":"empty_draw",&err);if(err)return err;
   f->phase[0]=3;f->winner_presence[0]=2;f->winner[0]=(signed char)(wins?side:1-side);
   f->reason_presence[0]=2;f->reason_slot[0]=reason;return log_event(f);
  }
  int ref=f->zone_card_index[(side*4+1)*Z+at];card_type(f,ref,&err);if(err)return err;
  for(int p=at;p<dn-1;p++)f->zone_card_index[(side*4+1)*Z+p]=f->zone_card_index[(side*4+1)*Z+p+1];
  f->zone_card_index[(side*4+1)*Z+dn-1]=0;f->zone_count[side*4+1]--;
  int hn=f->zone_count[side*4];
  if(hn>=10){
   int copy=f->card_bool_presence[ref*KB+COPY]==2&&f->card_bool_value[ref*KB+COPY];
   int zone=copy?3:2,n=f->zone_count[side*4+zone];if(n>=Z)return CAPACITY;
   f->card_jingu_mark_presence[ref]=0;f->card_jingu_delta[ref]=0;f->card_jingu_source_slot[ref]=0;
   f->zone_card_index[(side*4+zone)*Z+n]=ref;f->zone_count[side*4+zone]++;
   err=log_event(f);if(err)return err;
  }else{
   if(hn>=Z)return CAPACITY;
   f->zone_card_index[(side*4)*Z+hn]=ref;f->zone_count[side*4]++;
   err=add_hand_hook(f,side,ref);if(err)return err;
  }
  err=log_event(f);if(err)return err;(*drew)++;
 }return 0;
}
'''
    prefix+='\n#define INVALID 1\n#define CAPACITY 2\n#define RNG_EXHAUSTED 4\n'+body
    params=[];rows=[];abi=[]
    for k in _FIELDS:
        dtype,shape=specs[k];name=dtype.name;size=1
        for x in shape:size*=x
        ctype=_C_TYPES[name];wire=ctype if name in('int32','int64','float64')else'unsigned char'
        params.append(wire+'* a_'+k);abi.append({'int32':'int32*','int64':'int64*','float64':'float64*'}.get(name,'uint8*'))
        rows.append('f.'+k+'=('+ctype+'*)a_'+k+'+row*'+str(size)+';')
    digest=','.join(str(x)for x in catalog_fingerprint())
    code='\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)\nextern "C" __global__\n#else\nextern "C"\n#endif\nvoid domain_draw('+','.join(params)+',const int*sides,const int*counts,const int*filters,const unsigned char*active,int*drawn,int B){\n'
    code+='''#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=B)return;
#else
 for(int row=0;row<B;row++){
#endif
 PRow f;'''+''.join(rows)+'''drawn[row]=0;
 if(active[row]&&f.row_error[0]==0){
 int bad=!f.row_valid[0];const unsigned char digest[32]={'''+digest+'''};
 for(int j=0;j<32;j++)if(f.catalog_digest[j]!=digest[j])bad=1;
 f.row_error[0]=bad?INVALID:physical_draw(&f,sides[row],counts[row],filters[row],drawn+row);
 }
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
 }
#endif
}
'''
    return prefix+code,abi+['int32*','int32*','int32*','uint8*','int32*','int32']


@dataclass(frozen=True)
class PreparedPhysicalDraw:
    owner:object
    sources:dict
    fields:dict
    sources_query:tuple
    query:tuple
    pointers:tuple
    drawn:object


class DomainPhysicalCore:
    """Physical draw primitive; not yet the complete apply_action interface."""
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False,max_draws=32,max_rng_draws=4096):
        import torch as t
        from app.modules.card_game.content.duel_v2.registry import KITS,_ensure_kits
        import numpy as np
        self.t=t;self.layout=layout or DomainSoALayout();self.device=t.device(device)
        if not isinstance(self.layout,DomainSoALayout)or self.layout.max_cards>512:raise ValueError('Bounded domain layout with max_cards<=512 required')
        if type(max_draws)is not int or not 1<=max_draws<=128:raise ValueError('Bounded physical draw count required')
        if type(max_rng_draws)is not int or not 1<=max_rng_draws<=1048576:raise ValueError('Bounded RNG attempts required')
        _ensure_kits()
        for cid in CHARACTER_VOCAB:
            kit=KITS[cid]
            if cid!='xiaozhi'and getattr(kit,'on_hand_added',None)is not None:raise RuntimeError('Unported hand-added hook')
            if any('on_hand_added'in hooks for hooks in getattr(kit,'weapon_hooks',{}).values()):raise RuntimeError('Unported weapon hand-added hook')
        self.specs={k:(np.dtype(dtype),shape)for k,(dtype,shape)in array_specs(self.layout).items()};self.closed=False;self.module=None;self.library=None;self.temp=None
        source,abi=physical_source(self.layout,max_draws,max_rng_draws)
        if self.device.type=='cuda':
            if not t.cuda.is_available():raise RuntimeError('Actual CUDA required; no physical CPU fallback')
            if self.device.index is None:self.device=t.device('cuda',t.cuda.current_device())
            from .cuda_runtime import CudaModule
            self.module=CudaModule(source,{'domain_draw':abi},device=str(self.device))
        elif self.device.type=='cpu'and allow_cpu_oracle is True:
            compiler=shutil.which('clang++')or shutil.which('g++')
            if not compiler:raise RuntimeError('Explicit physical oracle requires a C++ compiler')
            self.temp=tempfile.TemporaryDirectory(prefix='nte-physical-draw-');path=Path(self.temp.name)
            (path/'draw.cpp').write_text(source)
            lib=path/('draw.dll'if os.name=='nt'else'draw.so')
            r=subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++17',str(path/'draw.cpp'),'-o',str(lib)],capture_output=True,text=True)
            if r.returncode:self.close();raise RuntimeError('Physical compilation failed: '+r.stderr)
            self.library=ctypes.CDLL(str(lib));self.kernel=self.library.domain_draw
            self.kernel.argtypes=[ctypes.c_void_p]*(len(abi)-1)+[ctypes.c_int];self.kernel.restype=None
        else:raise RuntimeError('CUDA or explicitly selected CPU oracle required')

    def prepare_draw(self,fields,sides,counts,filters,active):
        t=self.t
        if self.closed:raise RuntimeError('Physical core closed')
        if self.device.type=='cuda'and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare outside capture')
        if set(fields)!=set(self.specs):raise ValueError('Complete named state columns required')
        B=sides.shape[0]
        if not 1<=B<=min(400,self.layout.game_capacity):raise ValueError('Batch capacity exceeded')
        for value in(sides,counts,filters):
            if value.dtype!=t.int32 or value.device!=self.device or value.shape!=(B,):raise ValueError('Resident int32 draw selectors required')
        if active.dtype!=t.bool or active.device!=self.device or active.shape!=(B,):raise ValueError('Resident bool active required')
        output={}
        for name,(dtype,shape)in self.specs.items():
            v=fields[name]
            if v.dtype!=getattr(t,dtype.name)or v.device!=self.device or v.shape!=(B,)+shape:raise ValueError('Invalid named column '+name)
            output[name]=v.clone(memory_format=t.contiguous_format)
        query=tuple(v.clone(memory_format=t.contiguous_format)for v in(sides,counts,filters,active));drawn=t.zeros(B,dtype=t.int32,device=self.device)
        pointers=tuple(output[k]if self.specs[k][0].name in('int32','int64','float64')else output[k].view(t.uint8)for k in _FIELDS)+query[:3]+(query[3].view(t.uint8),drawn)
        return PreparedPhysicalDraw(self,dict(fields),output,(sides,counts,filters,active),query,pointers,drawn)

    def launch_draw(self,p):
        if self.closed or not isinstance(p,PreparedPhysicalDraw)or p.owner is not self:raise ValueError('Closed or foreign physical buffers')
        t=self.t;B=p.drawn.shape[0]
        for k,v in p.fields.items():v.copy_(p.sources[k])
        for dst,src in zip(p.query,p.sources_query):dst.copy_(src)
        if self.module:self.module.launch('domain_draw',list(p.pointers)+[B],grid=((B+63)//64,),block=(64,),capture_safe=True)
        else:self.kernel(*[ctypes.c_void_p(v.data_ptr())for v in p.pointers],B)
        failed=p.fields['row_error']!=0
        for k,v in p.fields.items():
            if k!='row_error':v.copy_(t.where(failed.reshape((B,)+(1,)*(v.ndim-1)),p.sources[k],v))
        p.drawn.masked_fill_(failed,0)
        return dict(fields=p.fields,drawn=p.drawn,errors=p.fields['row_error'],full_duel_qualified=False,full_canonical_codec_qualified=False)

    def close(self):
        if self.closed:return
        if self.module:self.module.close()
        if self.library:
            import _ctypes
            (_ctypes.FreeLibrary if os.name=='nt'else _ctypes.dlclose)(self.library._handle);self.library=None
        if self.temp:self.temp.cleanup();self.temp=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
