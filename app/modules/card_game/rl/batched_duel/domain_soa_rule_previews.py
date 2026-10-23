"""Actual anonymous rule execution for canonical H3 preview score fields.

Two private state batches are reused across all candidate previews; no per-action
full state batch is allocated. Unqualified until full physical acceptance passes.
"""
from math import prod
from pathlib import Path
import ctypes
import shutil
import subprocess
import tempfile
import numpy as np
from . import domain_soa_schema as schema
from .domain_soa_preview_source import kernel_source


class DomainRulePreviews:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',max_actions=800,allow_cpu_oracle=False,allow_unqualified=False):
        import torch
        if allow_unqualified is not True:raise RuntimeError('Physical preview integration has not passed acceptance')
        self.t=torch;self.layout=layout or schema.DomainSoALayout();self.device=torch.device(device);self.closed=False
        self.module=None;self.library=None;self.temp=None;self.max_actions=max_actions;self.width=max(4,self.layout.max_choice_cards)
        if self.device.type=='cpu'and allow_cpu_oracle is not True:raise RuntimeError('Explicit C++ oracle required')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU preview fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        elif self.device.type!='cpu':raise RuntimeError('CUDA or explicit C++ oracle required')
        self.specs=schema.array_specs(self.layout)
        from app.modules.card_game.rl import seven_lineup as seven
        from app.modules.card_game.content.duel_v2 import CARDS
        self.battle_catalog=torch.tensor([CARDS[cid]['type']=='battle'for cid in seven.CARD_IDS],dtype=torch.bool,device=self.device)
        try:
            source,abi=kernel_source(self.layout,max_actions,self.width)
            if self.device.type=='cuda':
                from .cuda_runtime import CudaModule
                self.module=CudaModule(source,{'domain_rule_previews':abi},device=str(self.device))
            else:
                compiler=shutil.which('clang++')or shutil.which('g++')
                if not compiler:raise RuntimeError('C++ compiler required')
                self.temp=tempfile.TemporaryDirectory(prefix='nte-domain-preview-');root=Path(self.temp.name)
                (root/'preview.cpp').write_text(source)
                result=subprocess.run([compiler,'-shared','-fPIC','-O1','-std=c++17',str(root/'preview.cpp'),'-o',str(root/'preview.so')],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('Physical preview compilation failed: '+result.stderr)
                self.library=ctypes.CDLL(str(root/'preview.so'));self.kernel=self.library.domain_rule_previews
                self.kernel.argtypes=[ctypes.c_void_p]*(len(abi)-1)+[ctypes.c_int];self.kernel.restype=None
        except BaseException:self.close();raise

    def prepare(self,fields,sides,active,actions):
        from types import SimpleNamespace
        t=self.t;B=sides.shape[0];A=self.max_actions
        if self.closed:raise RuntimeError('Closed previews')
        if sides.device!=self.device or sides.dtype!=t.int64 or sides.shape!=(B,)or not 1<=B<=min(400,self.layout.game_capacity):raise ValueError('Resident side selectors required')
        if active.device!=self.device or active.dtype!=t.bool or active.shape!=(B,):raise ValueError('Resident active mask required')
        if set(fields)!=set(self.specs):raise ValueError('Complete named fields required')
        for name,(dtype,shape)in self.specs.items():
            v=fields[name]
            if v.device!=self.device or v.dtype!=getattr(t,np.dtype(dtype).name)or v.shape!=(B,)+shape or not v.is_contiguous():raise ValueError('Invalid preview column '+name)
        for name,shape,dtype in [('counts',(B,),t.int64),('errors',(B,),t.int32),('payloads',(B,A,7),t.int64),('selected',(B,A,self.width),t.int64),('top',(B,A,4),t.int64),('bottom',(B,A,4),t.int64),('bottom_plural',(B,A),t.bool)]:
            v=actions[name]
            if v.device!=self.device or v.dtype!=dtype or v.shape!=shape or not v.is_contiguous():raise ValueError('Invalid resident preview action table '+name)
        private=tuple({name:v.clone(memory_format=t.contiguous_format)for name,v in fields.items()}for _ in range(2))
        tables=tuple(t.tensor([batch[name].data_ptr()for name in self.specs],dtype=t.int64,device=self.device)for batch in (fields,)+private)
        return SimpleNamespace(owner=self,fields=dict(fields),sides=sides,source_active=active,actions=actions,private=private,tables=tables,
            side_wire=sides.to(t.int32),active=active.clone(),battle=t.empty((B,A),dtype=t.bool,device=self.device),
            attack=t.empty((B,A),dtype=t.float64,device=self.device),attack_count=t.empty((B,A),dtype=t.float64,device=self.device),
            genesis=t.empty((B,A),dtype=t.float64,device=self.device),counter=t.empty((B,A),dtype=t.float64,device=self.device),
            errors=t.empty(B,dtype=t.int32,device=self.device))

    def launch(self,p):
        if self.closed or p.owner is not self:raise ValueError('Closed or foreign preview buffers')
        t=self.t;B=p.sides.shape[0];K=self.layout.max_cards;A=self.max_actions;out=p.actions
        p.side_wire.copy_(p.sides);p.active.copy_(p.source_active&(p.fields['phase']==1))
        kinds=out['public_cards'][:,:K,0].long();card=out['payloads'][...,2].clamp(0,K-1)
        public_kind=kinds.gather(1,card)
        legal=t.arange(A,device=self.device)[None,:]<out['counts'][:,None]
        p.battle.copy_(legal&(out['payloads'][...,0]==3)&(public_kind>=0)&self.battle_catalog[public_kind.clamp(0,self.battle_catalog.shape[0]-1)])
        pointers=list(p.tables)+[p.side_wire,p.active.view(t.uint8),out['counts'],out['errors'],out['payloads'],p.battle.view(t.uint8),out['selected'],out['top'],out['bottom'],out['bottom_plural'].view(t.uint8),p.attack,p.attack_count,p.genesis,p.counter,p.errors]
        if self.module:self.module.launch('domain_rule_previews',pointers+[B],grid=(B,),block=(256,),capture_safe=True)
        else:self.kernel(*[ctypes.c_void_p(v.data_ptr())for v in pointers],B)
        p.errors.copy_(t.where(p.source_active,out['errors']|p.errors,t.zeros_like(p.errors)))
        return dict(attack=p.attack,attack_count=p.attack_count,genesis=p.genesis,counter=p.counter,battle=p.battle,errors=p.errors,
                    full_duel_qualified=False,h3_qualified=False,gpu_qualified=False)

    def close(self):
        if self.closed:return
        if self.module:self.module.close()
        if self.library:
            import _ctypes
            _ctypes.dlclose(self.library._handle);self.library=None
        if self.temp:self.temp.cleanup();self.temp=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
