"""Explicit CPU compiler oracle for unqualified physical executor primitives.

This utility never selects a backend automatically and is not a search backend.
It exercises the same emitted C++ functions used for CUDA integration.
"""
import ctypes
from math import prod
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from . import domain_soa_schema as schema
from .domain_soa_execution_base import emit_header,emit_accessors,emit_strings_and_cards,_TYPES
from .domain_soa_runtime_source import emit_zone_source,emit_operation_source
from .domain_soa_effect_source import emit_source,emit_evaluate_source
from .domain_soa_execution_gc import emit_source as emit_gc_source


def primitive_source(layout):
    return (emit_header(layout)+emit_accessors()+emit_strings_and_cards()+
            emit_zone_source()+emit_operation_source(layout)+emit_source(layout)+emit_evaluate_source(layout)+emit_gc_source(layout))


class DomainPrimitiveOracle:
    """Diagnostic only. Explicit consent is required; never a GPU fallback."""
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,allow_cpu_oracle=False):
        if allow_cpu_oracle is not True:raise ValueError('Explicit CPU compiler oracle required')
        self.layout=layout or schema.DomainSoALayout()
        self.specs=schema.array_specs(self.layout)
        compiler=shutil.which('clang++')or shutil.which('g++')
        if not compiler:raise RuntimeError('C++ compiler required')
        self.temp=tempfile.TemporaryDirectory(prefix='nte-domain-primitives-')
        root=Path(self.temp.name);self.closed=False
        params=[];rows=[]
        for name,(dtype,shape)in self.specs.items():
            typ=_TYPES[np.dtype(dtype).name];params.append(typ+'* a_'+name)
            rows.append(f'f.{name}=a_{name};')
        source=primitive_source(self.layout)+'\nextern "C" int primitive_probe('+','.join(params)+',int mode,int side,int actor,int amount){DS f;'+''.join(rows)+'''double query_workspace[2*MAXQ];int damage_workspace[MAXQ];f.query_scratch=(unsigned char*)query_workspace;f.damage_scratch=damage_workspace;DX x={&f,side,actor,-1,DR{side,actor},0};
 if(mode==0){d_zone_set(&x,side,"delay","slow_turn",amount);d_zone_set(&x,side,"delay","by",1-side);d_zone_set(&x,side,"delay","slow",3);d_zone_set(&x,side,"delay","left",2);}
 else if(mode==1){d_reset_operation(&x);d_set_op_target(&x,"target_id",DR{1-side,actor});d_set_op_int(&x,"card_bonus",amount);d_set_op_int(&x,"normal_attack",1,1);}
 else if(mode==2){DEffect e={"probe.attack","attack.panel",(double)amount,"add",0,"self","independent",nullptr,-1,0,nullptr,nullptr,"down",nullptr,nullptr,nullptr,0};d_effect(&x,DR{side,actor},&e);}
 else if(mode==3)return (int)d_evaluate(&x,DR{side,actor},"attack.panel",amount);
 else if(mode==4){d_set_ci(&x,side,actor,"hp",amount);d_set_ci(&x,side,actor,"energy",amount);d_set_ci(&x,side,actor,"shield",amount);}
 else if(mode==5){d_clear_effect_id(&x,DR{side,actor},"probe.attack");}
 else if(mode==6)return d_attack_value(&x,side,actor,amount);
 else if(mode==7)return d_zone_int(&x,side,"delay","slow_turn",-1);
 else if(mode==9){d_collect_cards(&x);d_collect_strings(&x);}
 else if(mode==10){d_intern(&x,"unreachable diagnostic string");}
 else if(mode==11){DEffect e={"probe.maximum","attack.panel",(double)amount,"add",0,"self","max","maximum",-1,0,nullptr,nullptr,"down",nullptr,nullptr,nullptr,0};e.label=amount>2?"larger":"smaller";e.source_key=amount>2?"first":"second";d_effect(&x,DR{side,actor},&e);}
 else if(mode==12){DEffect e={"probe.resistance","damage.resistance",-2.,"add",0,"self","independent",nullptr,-1,0,nullptr,"damage","down","damage_attribute","attribute","\\xe5\\x85\\x89",0};d_effect(&x,DR{side,actor},&e);}
 else if(mode==13){DEvt event={};event.attribute=amount;return (int)d_evaluate(&x,DR{side,actor},"damage.resistance",0,&event);}
 else if(mode==14){DEffect e={"probe.bounded","attack.panel",(double)amount,"add",0,"self","independent",nullptr,-1,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0};e.max_stacks=2;e.clear_count=0;d_effect(&x,DR{side,actor},&e);}
 else if(mode==16){Candidate entries[3]={{7.,-100,2},{3.,0,0},{-5.,1000,0}};return (int)d_evaluate_entries(&x,DR{side,actor},"attack.hit",0,nullptr,entries,3);}
 else {d_fail(&x,4);}return f.row_error[0];}'''
        (root/'probe.cpp').write_text(source)
        result=subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++17',str(root/'probe.cpp'),'-o',str(root/'probe.so')],capture_output=True,text=True)
        if result.returncode:self.temp.cleanup();raise RuntimeError(result.stderr)
        self.library=ctypes.CDLL(str(root/'probe.so'));self.kernel=self.library.primitive_probe
        self.kernel.argtypes=[ctypes.c_void_p]*len(self.specs)+[ctypes.c_int]*4;self.kernel.restype=ctypes.c_int

    def run(self,batch,mode,side,actor,amount,row=0):
        if self.closed:raise RuntimeError('Closed oracle')
        if batch.layout!=self.layout:raise ValueError('Layout mismatch')
        ptr=[]
        for name,(dtype,shape)in self.specs.items():
            a=getattr(batch,name)
            if a.dtype!=np.dtype(dtype)or not a.flags.c_contiguous:raise ValueError('Contiguous exact named columns required')
            ptr.append(ctypes.c_void_p(a.ctypes.data+row*prod(shape)*a.itemsize))
        # Successful probes mutate only the selected row. Errors roll it back.
        saved={name:getattr(batch,name)[row].copy()for name in self.specs}
        value=self.kernel(*ptr,mode,side,actor,amount)
        error=int(batch.row_error[row])
        if error:
            for name in self.specs:
                a=getattr(batch,name)
                if name!='row_error':a[row]=saved[name]
        return value

    def close(self):
        if self.closed:return
        import _ctypes
        _ctypes.dlclose(self.library._handle);self.temp.cleanup();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
