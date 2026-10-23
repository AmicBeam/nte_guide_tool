"""Bounded immutable modifier queries for device legality and public metadata.

Static named properties and declarative effects only. These are not VM opcodes,
physical rule transitions or complete legal-action generation.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import ctypes
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from .domain_soa_character_queries import _source
from .domain_soa_schema import DomainSoALayout,array_specs,catalog_fingerprint

PROPERTIES=('attack.panel','attack.override','rule.damage_immune','damage.limit',
            'card.instant','card.cost','rule.normal_attack_instant','marker.lingke_taunt')
supports_full_duel_rules=False


def source(layout):
    text,names=_source(layout);specs=array_specs(layout)
    types={'bool':'unsigned char','uint8':'unsigned char','int8':'signed char','int16':'short','int32':'int','int64':'long long','float64':'double'}
    params=','.join('const %s*p_%s'%(types[str(np.dtype(specs[n][0]))],n)for n in names)
    initialize=''.join('f.%s=p_%s+row*%d;'%(n,n,prod(specs[n][1]))for n in names)
    args=','.join('p_'+n for n in names)
    extra=',const long long*subjects,const int*properties,const double*bases,const unsigned char*base_present,const unsigned char*active,double*values,unsigned char*present,int*errors,int rows,int entries'
    entry='''
QFN void run_rule_query('''+params+extra+''',int index){
 int row=index/entries;values[index]=0;present[index]=0;errors[index]=0;if(!active[index])return;QRow f;'''+initialize+'''
 if(f.row_error[0])return;
 if(!f.row_valid[0]||f.string_count[0]<0||f.string_count[0]>SS||f.string_bytes_used[0]<0||f.string_bytes_used[0]>SB){errors[index]=1;return;}
 for(int s=0;s<2;s++)if(f.player_character_count[s]<1||f.player_character_count[s]>C){errors[index]=1;return;}
 long long subject=subjects[index];int property=properties[index],err=0,has=base_present[index]!=0;
 if(subject<0||subject>=H||property<0||property>=8){errors[index]=1;return;}
 int h=(int)subject;
 if(!exists(&f,h)||(holder_char(h)>=0&&!f.character_valid[holder_side(h)*C+holder_char(h)])){errors[index]=1;return;}
 double base=bases[index];if(!qfinite(base)||qabs(base)>2147483647.){errors[index]=3;return;}
 double value=query(&f,h,property,base,&has,&err);
 if(!qfinite(value)||qabs(value)>2147483647.)err=3;
 if(err){errors[index]=err;return;}values[index]=has?value:0;present[index]=has!=0;
}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_rule_queries('''+params+extra+'''){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<rows*entries)run_rule_query('''+args+''',subjects,properties,bases,base_present,active,values,present,errors,rows,entries,index);
}
#else
extern "C" void domain_rule_queries('''+params+extra+'''){
 for(int index=0;index<rows*entries;index++)run_rule_query('''+args+''',subjects,properties,bases,base_present,active,values,present,errors,rows,entries,index);
}
#endif
'''
    return text+entry,names


@dataclass(frozen=True)
class PreparedRuleQueries:
    owner: object
    sources: object
    fields: object
    inputs: object
    buffers: object
    values: object
    present: object
    errors: object
    valid: object


class DomainRuleQueries:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False):
        import torch
        self.t=torch;self.layout=layout or DomainSoALayout();self.device=torch.device(device)
        if not (4<=self.layout.max_characters<=16 and self.layout.max_effects<=32 and self.layout.max_timed_attack<=16 and self.layout.max_conditions<=8):raise ValueError('Bounded rule query layout required')
        code,self.names=source(self.layout);self.specs=array_specs(self.layout);self.module=None;self.library=None;self.temporary=None;self.closed=False
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no query CPU fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            self.module=CudaModule(code,{'domain_rule_queries':['uint8*']*len(self.names)+['int64*','int32*','float64*','uint8*','uint8*','float64*','uint8*','int32*','int32','int32']},device=str(self.device))
        elif self.device.type=='cpu'and allow_cpu_oracle is True:
            compiler=shutil.which('clang++')or shutil.which('g++')
            if not compiler:raise RuntimeError('Explicit compiler oracle unavailable')
            self.temporary=tempfile.TemporaryDirectory(prefix='nte-rule-query-');root=Path(self.temporary.name);path=root/'query.cpp';path.write_text(code)
            lib=root/('query.dylib'if sys.platform=='darwin'else'query.so')
            result=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',str(path),'-o',str(lib)],capture_output=True,text=True,timeout=60)
            if result.returncode:self.close();raise RuntimeError(result.stderr)
            self.library=ctypes.CDLL(str(lib));self.kernel=self.library.domain_rule_queries
            self.kernel.argtypes=[ctypes.c_void_p]*(len(self.names)+8)+[ctypes.c_int32,ctypes.c_int32];self.kernel.restype=None
        else:raise ValueError('CUDA or explicit CPU compiler oracle required')
        self.digest=torch.tensor(list(catalog_fingerprint()),dtype=torch.uint8,device=self.device)

    def prepare(self,fields,subjects,properties,bases,base_present,active):
        t=self.t
        if self.closed:raise RuntimeError('Rule query backend closed')
        if self.device.type=='cuda'and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare rule buffers outside capture')
        if not isinstance(subjects,t.Tensor)or subjects.ndim!=2:raise ValueError('Rank-two resident subjects required')
        B,N=subjects.shape
        if not 1<=B<=self.layout.game_capacity or not 1<=N<=1024:raise ValueError('Rule query capacity exceeded')
        inputs={'subjects':subjects,'properties':properties,'bases':bases,'base_present':base_present,'active':active}
        dtypes=(t.int64,t.int32,t.float64,t.bool,t.bool)
        for (name,v),dtype in zip(inputs.items(),dtypes):
            if not isinstance(v,t.Tensor)or v.device!=self.device or v.dtype!=dtype or tuple(v.shape)!=(B,N):raise ValueError('Resident '+name+' contract mismatch')
        cols=fields if isinstance(fields,dict)else getattr(fields,'arrays',None)
        if cols is None:raise TypeError('Named resident rule fields required')
        sources={};dtype={'bool':t.bool,'uint8':t.uint8,'int8':t.int8,'int16':t.int16,'int32':t.int32,'int64':t.int64,'float64':t.float64}
        for name in self.names:
            v=cols.get(name);kind,tail=self.specs[name]
            if not isinstance(v,t.Tensor)or v.device!=self.device or v.dtype!=dtype[str(np.dtype(kind))]or tuple(v.shape)!=(B,)+tuple(tail):raise ValueError('Resident typed rule field required: '+name)
            sources[name]=v
        return PreparedRuleQueries(self,sources,{k:v.clone(memory_format=t.contiguous_format)for k,v in sources.items()},inputs,
            {k:v.clone(memory_format=t.contiguous_format)for k,v in inputs.items()},
            t.empty((B,N),dtype=t.float64,device=self.device),t.empty((B,N),dtype=t.bool,device=self.device),
            t.empty((B,N),dtype=t.int32,device=self.device),t.empty((B,N),dtype=t.bool,device=self.device))

    def launch(self,p):
        t=self.t
        if self.closed or not isinstance(p,PreparedRuleQueries)or p.owner is not self:raise ValueError('Closed or foreign query buffers')
        for k,v in p.fields.items():v.copy_(p.sources[k])
        for k,v in p.buffers.items():v.copy_(p.inputs[k])
        B,N=p.values.shape
        args=[p.fields[n].view(t.uint8)for n in self.names]+[p.buffers['subjects'],p.buffers['properties'],p.buffers['bases'],p.buffers['base_present'].view(t.uint8),p.buffers['active'].view(t.uint8),p.values,p.present.view(t.uint8),p.errors]
        if self.module is not None:self.module.launch('domain_rule_queries',args+[B,N],grid=((B*N+63)//64,),block=(64,),capture_safe=True)
        else:self.kernel(*[ctypes.c_void_p(x.data_ptr())for x in args],ctypes.c_int32(B),ctypes.c_int32(N))
        bad=(p.fields['catalog_digest']!=self.digest[None,:]).any(-1)
        p.errors.copy_(t.where(bad[:,None]&p.buffers['active'],t.ones_like(p.errors),p.errors))
        inherited=p.fields['row_error'][:,None];p.errors.copy_(t.where(inherited!=0,inherited,p.errors))
        p.valid.copy_(p.buffers['active']&(p.errors==0));p.values.masked_fill_(~p.valid,0);p.present.masked_fill_(~p.valid,False)
        return {'values':p.values,'present':p.present,'valid':p.valid,'errors':p.errors,'properties':PROPERTIES,'full_duel_qualified':False}

    def query(self,fields,subjects,properties,bases,base_present,active):return self.launch(self.prepare(fields,subjects,properties,bases,base_present,active))
    def close(self):
        if self.closed:return
        if self.module is not None:self.module.close()
        self.library=None
        if self.temporary is not None:self.temporary.cleanup()
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
