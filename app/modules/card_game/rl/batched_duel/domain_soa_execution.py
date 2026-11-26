"""Integration of named domain C++ handlers, without a rule VM or CPU fallback.

The implementation remains unqualified until complete formal-state comparisons
and actual CUDA natural-game validation pass. Source presence is not acceptance.
"""
import importlib
import inspect
import re
from . import domain_soa_schema as schema
from . import domain_soa_execution_base as base
from . import domain_soa_effect_source as effects
from . import domain_soa_runtime_source as runtime
from . import domain_soa_mutation_source as mutations
from . import domain_soa_execution_gc as collection
from . import domain_soa_action_source as actions
from .domain_soa_execution_workspace import prepare_workspace

supports_full_duel_rules=False
gpu_qualified=False
_REQUIRED=('domain_soa_combat_source','domain_soa_combat_helpers','domain_soa_cards_source',
           'domain_soa_card_helpers','domain_soa_lifecycle_source','domain_soa_lifecycle_helpers')


def _emit(module,layout):
    signature=inspect.signature(module.emit_source)
    return module.emit_source(layout)if len(signature.parameters)else module.emit_source()


def handler_sources(layout):
    """Build actual handler sources; absent dependencies fail rather than stub."""
    pieces=[base.emit_accessors(),base.emit_player_deletions(),base.emit_strings_and_cards(),runtime.emit_zone_source(),
            runtime.emit_operation_source(layout),effects.emit_source(layout),
            effects.emit_evaluate_source(layout),mutations.emit_source(layout),mutations.emit_equipment_source(),mutations.emit_random_helpers(),mutations.emit_context_mutations()]
    missing=[]
    for name in _REQUIRED:
        try:module=importlib.import_module('.'+name,__package__)
        except ModuleNotFoundError as error:
            if error.name==__package__+'.'+name:missing.append(name);continue
            raise
        pieces.append(_emit(module,layout))
    if missing:raise RuntimeError('Physical execution dependencies missing: '+', '.join(missing))
    pieces.extend((effects.emit_clock_source(),collection.emit_source(layout),actions.emit_source()))
    return pieces


def emit_source(layout=None):
    """Static compilation composition, never dynamic dispatch over instructions."""
    layout=layout or schema.DomainSoALayout()
    header=base.emit_header(layout);pieces=handler_sources(layout)
    text='\n'.join(pieces)
    if not text.isascii():raise ValueError('Direct CUDA source must use UTF8 byte literals')
    # Global domain structs are declared before cross-file function prototypes.
    # Each actual definition remains in its owning source module.
    structs=sorted(set(re.findall(r'\bstruct\s+([A-Za-z_]\w*)\s*\{',text)))
    forwards=''.join('struct '+name+';\n'for name in structs)
    headers=re.findall(r'(?m)^[ \t]*DFN[ \t]+([^{};#]+?)\s*\{',text)
    for signature in headers:
        signature=re.sub(r'=\s*[^,)]+','',signature)
        forwards+='DFN '+signature.strip()+';\n'
    return header+'\n'+forwards+'\n'+text


def coverage():
    return dict(supports_full_duel_rules=False,gpu_qualified=False,
                physical_handlers_qualified=False,natural_complete_games_qualified=False,
                h3_qualified=False,search400_qualified=False,
                no_vm=True,no_host_rules_fallback=True)


def kernel_source(layout,action_capacity,selection_width):
    """One device thread per physical root, only resident legal table selection."""
    from math import prod
    import numpy as np
    specs=schema.array_specs(layout);params=[];bind=[];abi=[]
    for name,(dtype,shape)in specs.items():
        dtype=np.dtype(dtype);ctype=base._TYPES[dtype.name];wire=ctype if dtype.name in ('int32','int64','float64')else'unsigned char'
        params.append(wire+'* a_'+name);bind.append(f'f.{name}=({ctype}*)a_{name}+row*{prod(shape)};')
        abi.append({'int32':'int32*','int64':'int64*','float64':'float64*'}.get(dtype.name,'uint8*'))
    source=emit_source(layout)
    source+='''\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__
#else
extern "C"
#endif
void apply_domain_action('''+','.join(params)+''',const int*sides,const long long*selectors,const unsigned char*active,
 const long long*counts,const int*prior_errors,const long long*payloads,const long long*selected,
 const long long*top,const long long*bottom,const unsigned char*bottom_plural,int*error_sites,double*query_workspace,int*damage_workspace,int B){
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=blockIdx.x;if(threadIdx.x!=0||row>=B)return;
#else
 for(int row=0;row<B;row++){
#endif
 DS f;'''+''.join(bind)+'''f.query_scratch=(unsigned char*)(query_workspace+row*2*MAXQ);f.damage_scratch=damage_workspace+row*MAXQ;f.error_site=0;error_sites[row]=0;
 if(active[row]&&f.row_error[0]==0){
  if(prior_errors[row])f.row_error[0]=prior_errors[row];
  else if(selectors[row]<0||selectors[row]>=counts[row]||counts[row]>'''+str(action_capacity)+''')f.row_error[0]=1;
  else{DX c={&f,sides[row],-1,-1,DR{-1,-2},0};int at=row*'''+str(action_capacity)+'''+(int)selectors[row];
   d_apply_root(&c,payloads+at*7,selected+at*'''+str(selection_width)+','+str(selection_width)+''',top+at*4,bottom+at*4,bottom_plural[at]);error_sites[row]=f.error_site;}
 }
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
 }
#endif
}
'''
    return source,abi+['int32*','int64*','uint8*','int64*','int32*','int64*','int64*','int64*','int64*','uint8*','int32*','float64*','int32*','int32']


class DomainSoAExecution:
    """Unqualified integration sandbox; not selectable by training controllers."""
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False,allow_unqualified=False,max_actions=800):
        import ctypes,shutil,tempfile,subprocess
        from pathlib import Path
        import torch
        from .domain_soa_actions import DomainSoAActions
        if allow_unqualified is not True:raise RuntimeError('Physical handlers have not completed integration acceptance')
        self.t=torch;self.layout=layout or schema.DomainSoALayout();self.device=torch.device(device);self.closed=False
        self.module=None;self.library=None;self.temp=None;self.actions=None
        if self.device.type=='cpu'and allow_cpu_oracle is not True:raise RuntimeError('Explicit C++ compiler oracle required')
        if self.device.type not in ('cpu','cuda'):raise RuntimeError('Actual CUDA or explicit C++ oracle required')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU rules fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.max_actions=max_actions;self.selection_width=max(4,self.layout.max_choice_cards)
        self.specs=schema.array_specs(self.layout)
        source,abi=kernel_source(self.layout,max_actions,self.selection_width)
        self.source=source
        try:
            if self.device.type=='cuda':
                from .cuda_runtime import CudaModule
                self.module=CudaModule(source,{'apply_domain_action':abi},device=str(self.device))
                self.module.ensure_stack(32768)
            else:
                compiler=shutil.which('clang++')or shutil.which('g++')
                if not compiler:raise RuntimeError('C++ compiler required')
                self.temp=tempfile.TemporaryDirectory(prefix='nte-domain-execution-');root=Path(self.temp.name)
                (root/'execution.cpp').write_text(source)
                result=subprocess.run([compiler,'-shared','-fPIC','-O1','-std=c++17',str(root/'execution.cpp'),'-o',str(root/'execution.so')],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('Physical compilation failed: '+result.stderr)
                self.library=ctypes.CDLL(str(root/'execution.so'));self.kernel=self.library.apply_domain_action
                self.kernel.argtypes=[ctypes.c_void_p]*(len(abi)-1)+[ctypes.c_int];self.kernel.restype=None
            self.actions=DomainSoAActions(self.layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle,max_actions=max_actions)
        except BaseException:self.close();raise

    def prepare(self,fields,sides,selectors,active):
        import numpy as np
        from types import SimpleNamespace
        t=self.t
        if self.closed:raise RuntimeError('Closed physical executor')
        B=sides.shape[0]
        if not 1<=B<=min(400,self.layout.game_capacity):raise ValueError('Physical row capacity exceeded')
        for v,dtype in ((sides,t.int64),(selectors,t.int64),(active,t.bool)):
            if v.device!=self.device or v.dtype!=dtype or v.shape!=(B,):raise ValueError('Resident exact selector columns required')
        if set(fields)!=set(self.specs):raise ValueError('Complete named physical fields required')
        outputs={}
        for name,(dtype,shape)in self.specs.items():
            v=fields[name]
            if v.device!=self.device or v.dtype!=getattr(t,np.dtype(dtype).name)or v.shape!=(B,)+shape:raise ValueError('Invalid physical column '+name)
            outputs[name]=v.clone(memory_format=t.contiguous_format)
        si=sides.clone();pick=selectors.clone();on=active.clone()
        return SimpleNamespace(owner=self,sources=dict(fields),fields=outputs,query_sources=(sides,selectors,active),
                               sides=si,selectors=pick,active=on,side_wire=si.to(t.int32),workspace=prepare_workspace(t,self.layout,B,self.device),
                               action_buffers=self.actions.prepare(outputs,si,on),error_sites=t.zeros(B,dtype=t.int32,device=self.device))

    def launch(self,p):
        import ctypes,numpy as np
        t=self.t
        if self.closed or p.owner is not self:raise ValueError('Closed or foreign execution buffers')
        for name,v in p.fields.items():v.copy_(p.sources[name])
        for dst,src in zip((p.sides,p.selectors,p.active),p.query_sources):dst.copy_(src)
        p.side_wire.copy_(p.sides)
        out=self.actions.launch(p.action_buffers)
        pointers=[p.fields[name]if np.dtype(self.specs[name][0]).name in ('int32','int64','float64')else p.fields[name].view(t.uint8)for name in self.specs]
        pointers.extend((p.side_wire,p.selectors,p.active.view(t.uint8),out['counts'],out['errors'],out['payloads'],out['selected'],out['top'],out['bottom'],out['bottom_plural'].view(t.uint8),p.error_sites))
        pointers.extend(p.workspace)
        B=p.active.shape[0]
        if self.module:self.module.launch('apply_domain_action',pointers+[B],grid=(B,),block=(32,),capture_safe=True)
        else:self.kernel(*[ctypes.c_void_p(v.data_ptr())for v in pointers],B)
        failed=p.fields['row_error']!=0
        for name,v in p.fields.items():
            if name!='row_error':t.where(failed.reshape((B,)+(1,)*(v.ndim-1)),p.sources[name],v,out=v)
        return dict(fields=p.fields,errors=p.fields['row_error'],error_sites=p.error_sites,legal_counts=out['counts'],full_duel_qualified=False,gpu_qualified=False)

    def close(self):
        if self.closed:return
        if self.actions:self.actions.close()
        if self.module:self.module.close()
        if self.library:
            import _ctypes
            _ctypes.dlclose(self.library._handle);self.library=None
        if self.temp:self.temp.cleanup();self.temp=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
