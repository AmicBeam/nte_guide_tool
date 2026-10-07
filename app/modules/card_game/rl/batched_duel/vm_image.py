"""Lossless numeric lowering of frozen rule IR for the CUDA executor.

This is a program image, not an executor or GPU-rule completeness claim.
Imports neither Torch nor the formal rules. State ingress uses the existing
codec ABI; negative object references address this immutable constant graph.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import struct
import importlib.util
from types import MappingProxyType
import numpy as np

from ..rule_ir.portable_program import FrozenRuleProgram
from . import schema

IMAGE_VERSION='duel_cuda_vm_image_v1'
# 0..10 retain schema.PackedStateBatch compatibility.
FUNCTION=11;CODE=12;CLASS=13;MODULE=14;INTRINSIC=15
CELL=16;ITERATOR=17;BOUND_METHOD=18;GENERATOR=19;SLICE=20
SET=21;FROZENSET=22;EXCEPTION=23;CONTEXT_VAR=24;TOKEN=25;RANGE=26
ENUM=27;DECIMAL=28;OBJECT=29;UNBOUND=30;ELLIPSIS=31;NOT_IMPLEMENTED=32


class ImageError(ValueError):pass


@dataclass(frozen=True)
class VmImage:
    program_identity:str
    identity:str
    instructions:np.ndarray
    functions:np.ndarray
    classes:np.ndarray
    modules:np.ndarray
    nodes:np.ndarray
    edges:np.ndarray
    payload:np.ndarray
    opcode_names:tuple
    intrinsic_names:tuple
    function_ids:tuple
    class_ids:tuple
    module_names:tuple
    entries:dict
    required:dict

    def verify(self):
        if self.identity!=_image_identity(self):raise ImageError('Numeric VM image changed')
        for name in ('instructions','functions','classes','modules','nodes','edges'):
            value=getattr(self,name)
            if value.dtype!=np.dtype('<i4') or not value.flags.c_contiguous:raise ImageError('Invalid VM int32 table')
        if self.payload.dtype!=np.uint8 or not self.payload.flags.c_contiguous:raise ImageError('Invalid VM byte table')
        return True

    @property
    def allocated_bytes(self):
        return sum(getattr(self,k).nbytes for k in ('instructions','functions','classes','modules','nodes','edges','payload'))

    def missing_capabilities(self,entry,*,opcodes=(),intrinsics=()):
        if entry not in self.required:raise ImageError('Unknown VM entry')
        need=self.required[entry]
        return dict(opcodes=sorted(set(need['opcodes'])-set(opcodes)),
                    intrinsics=sorted(set(need['intrinsics'])-set(intrinsics)))


def _image_identity(image):
    header=dict(version=IMAGE_VERSION,program=image.program_identity,opcodes=image.opcode_names,
        intrinsics=image.intrinsic_names,functions=image.function_ids,classes=image.class_ids,
        modules=image.module_names,entries=dict(image.entries),required={k:dict(v) for k,v in image.required.items()})
    h=sha256(json.dumps(header,sort_keys=True,separators=(',',':')).encode())
    for name in ('instructions','functions','classes','modules','nodes','edges','payload'):
        a=getattr(image,name);h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


class _Lower:
    def __init__(self,program):
        program.validate();self.data=json.loads(program.to_json())
        if self.data.get('unresolved'):raise ImageError('Unresolved source dependencies')
        self.fns=list(self.data['functions'])+list(self.data['codes'])
        self.fidx={f['id']:i for i,f in enumerate(self.fns)}
        self.cidx={c['id']:i for i,c in enumerate(self.data['classes'])}
        self.cdesc={c['id']:c for c in self.data['constants']}
        self.nodes=[];self.edges=[];self.payload=bytearray();self.const_refs={};self.ref_memo={};self.literals={}
        self.native_names=set();self.module_names=set(self.data['modules'])
        def walk(v):
            if isinstance(v,dict):
                if v.get('kind')=='intrinsic':self.native_names.add(v['name'])
                elif v.get('kind')=='module':self.module_names.add(v['name'])
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(self.data)
        self.native_names=tuple(sorted(self.native_names));self.nidx={s:i for i,s in enumerate(self.native_names)}
        self.module_names=tuple(sorted(self.module_names));self.midx={s:i for i,s in enumerate(self.module_names)}
        self.opnames=tuple(sorted({i['op'] for f in self.fns for i in f['instructions']}));self.opidx={s:i for i,s in enumerate(self.opnames)}
        self.none=self.node(schema.TAG_NONE);self.false=self.node(schema.TAG_BOOL,extra=0);self.true=self.node(schema.TAG_BOOL,extra=1)

    def node(self,tag,offset=0,count=0,extra=0):
        if len(self.nodes)>=2**31-1:raise ImageError('Constant graph exceeds int32')
        ref=-1-len(self.nodes);self.nodes.append([tag,offset,count,extra]);return ref

    def set_node(self,ref,tag,offset=0,count=0,extra=0):self.nodes[-1-ref]=[tag,offset,count,extra]

    def literal(self,value):
        if value is None:return self.none
        if value is False:return self.false
        if value is True:return self.true
        if type(value) not in (int,float,str,bytes):raise ImageError('Not a numeric image literal')
        if isinstance(value,float) and not np.isfinite(value):raise ImageError('Nonfinite image literal')
        key=(type(value).__name__,struct.pack('<d',value) if type(value) is float else value)
        if key in self.literals:return self.literals[key]
        if type(value) is int:
            size=max(1,(value.bit_length()+8)//8);raw=value.to_bytes(size,'little',signed=True);tag=schema.TAG_INT
        elif type(value) is float:raw=struct.pack('<d',value);tag=schema.TAG_FLOAT
        elif type(value) is str:raw=value.encode('utf-8');tag=schema.TAG_STR
        else:raise ImageError('Python bytes need a distinct VM tag; not a Unicode string')
        offset=len(self.payload);self.payload.extend(raw);ref=self.node(tag,offset,len(raw));self.literals[key]=ref;return ref

    def sequence(self,values,tag=schema.TAG_TUPLE,ref=None):
        refs=[self.value(v) for v in values];offset=len(self.edges);self.edges.extend([r,-1] for r in refs)
        if ref is None:return self.node(tag,offset,len(refs))
        self.set_node(ref,tag,offset,len(refs));return ref

    def mapping(self,pairs,ref=None):
        refs=[[self.value(k),self.value(v)] for k,v in pairs];offset=len(self.edges);self.edges.extend(refs)
        if ref is None:return self.node(schema.TAG_DICT,offset,len(refs))
        self.set_node(ref,schema.TAG_DICT,offset,len(refs));return ref

    def reference(self,desc):
        kind=desc['kind']
        if kind=='constant':return self.constant(desc['id'])
        if kind=='singleton':
            name=desc['name']
            if name in ('None','True','False'):return {'None':self.none,'True':self.true,'False':self.false}[name]
            if name in ('Ellipsis','NotImplemented'):
                key=(kind,name)
                if key not in self.ref_memo:self.ref_memo[key]=self.node(ELLIPSIS if name=='Ellipsis' else NOT_IMPLEMENTED)
                return self.ref_memo[key]
            raise ImageError('Unknown singleton')
        table={'function':(FUNCTION,self.fidx),'code':(CODE,self.fidx),'class':(CLASS,self.cidx),
               'module':(MODULE,self.midx),'intrinsic':(INTRINSIC,self.nidx)}
        if kind not in table:raise ImageError('Unknown source reference '+kind)
        tag,index=table[kind];name=desc.get('id',desc.get('name'));key=(kind,name)
        if name not in index:raise ImageError('Dangling image reference')
        if key not in self.ref_memo:self.ref_memo[key]=self.node(tag,extra=index[name])
        return self.ref_memo[key]

    def value(self,value):
        if isinstance(value,dict):
            if isinstance(value.get('kind'),str) and value['kind'] in ('constant','singleton','function','code','class','module','intrinsic'):
                return self.reference(value)
            return self.mapping(value.items())
        if isinstance(value,(list,tuple)):return self.sequence(value)
        return self.literal(value)

    def constant(self,cid):
        if cid in self.const_refs:return self.const_refs[cid]
        d=self.cdesc[cid];kind=d['kind']
        if kind=='literal':
            typ=d['type'];value=d['value']
            if typ not in ('str','int','float','bool'):raise ImageError('Unsupported literal type '+typ)
            ref=self.literal(value);self.const_refs[cid]=ref;return ref
        ref=self.node(UNBOUND);self.const_refs[cid]=ref
        if kind=='sequence':
            tags={'list':schema.TAG_LIST,'tuple':schema.TAG_TUPLE,'set':SET,'frozenset':FROZENSET}
            if d['type'] not in tags:raise ImageError('Unknown sequence type')
            self.sequence(d['items'],tags[d['type']],ref)
        elif kind=='mapping':self.mapping([(e['key'],e['value']) for e in d['entries']],ref)
        elif kind=='enum':
            # Metadata children: class, scalar value, member name, exact str(enum).
            self.sequence([d['class'],d['value'],d['name'],d['str']],ENUM,ref)
        elif kind=='ContextVar':self.sequence([d['name'],d.get('default')],CONTEXT_VAR,ref)
        elif kind=='object':
            self.sequence([{'kind':'class','id':d['class_id']},d['attributes']],OBJECT,ref)
        else:raise ImageError('Unsupported source constant kind '+kind)
        return ref

    def reachable(self,entry):
        visited=set();ops=set();native=set()
        fs={f['id']:f for f in self.fns};cs={c['id']:c for c in self.data['classes']}
        def walk(v):
            if isinstance(v,dict):
                kind=v.get('kind');identity=(kind,v.get('id',v.get('name')))
                if kind in ('function','code','class','constant','module'):
                    if identity in visited:return
                    visited.add(identity)
                    if kind in ('function','code'):
                        f=fs[v['id']];ops.update(i['op'] for i in f['instructions'])
                        for k in ('globals','constants','defaults','kwdefaults','closure'):walk(f.get(k,{}))
                    elif kind=='class':walk(cs[v['id']])
                    elif kind=='constant':walk(self.cdesc[v['id']])
                    else:walk(self.data['modules'].get(v['name'],{}))
                    return
                if kind=='intrinsic':native.add(v['name'])
                for x in v.values():walk(x)
            elif isinstance(v,(list,tuple)):
                for x in v:walk(x)
        walk({'kind':'function','id':entry})
        return dict(opcodes=sorted(ops),intrinsics=sorted(native))

    def lower(self):
        instructions=[];functions=[]
        for f in self.fns:
            start=len(instructions);n=len(f['instructions'])
            consts=f['constants'];names=f['cellvars']+f['freevars']
            for position,i in enumerate(f['instructions']):
                arg=-1 if i['arg'] is None else i['arg']
                if type(arg) is not int or not -1<=arg<2**31:raise ImageError('Opcode arg overflow')
                if i['op']=='LOAD_CONST':operand=self.reference(consts[arg])
                elif i['op']=='IMPORT_NAME':
                    prior=f['instructions'][max(0,position-2):position]
                    if len(prior)!=2 or any(x['op']!='LOAD_CONST' for x in prior):raise ImageError('Static import operands required')
                    level=prior[0]['argval'];fromlist=prior[1]['argval']
                    module=f.get('module','');package=module if module.replace('.','/')+'/__init__.py' in self.data['source_hashes'] else module.rpartition('.')[0]
                    absolute=importlib.util.resolve_name('.'*level+(i['argval'] or ''),package) if level else i['argval']
                    if not fromlist:absolute=absolute.split('.')[0]
                    if absolute not in self.midx:raise ImageError('Frozen import module missing: '+absolute)
                    operand=self.reference({'kind':'module','name':absolute})
                elif i['op'] in ('LOAD_DEREF' ,'STORE_DEREF','LOAD_CLOSURE'):
                    if not 0<=arg<len(names) or names[arg]!=i['argval']:raise ImageError('Cell slot mismatch')
                    operand=self.literal(i['argval'])
                else:
                    # Code-object argval is handled by its canonical LOAD_CONST ref.
                    operand=self.value(i['argval'])
                target=-1 if i['target'] is None else start+i['target']
                if target!=-1 and not start<=target<start+n:raise ImageError('Jump outside function')
                instructions.append([self.opidx[i['op']],arg,operand,target])
            module=f.get('module','')
            package=module if module.replace('.','/')+'/__init__.py' in self.data['source_hashes'] else module.rpartition('.')[0]
            refs=[self.value(f.get(k,{} if k in ('globals','kwdefaults') else [])) for k in
                ('globals','constants','varnames','cellvars','freevars','defaults','kwdefaults','closure')]
            functions.append([start,n,f['argcount'],f['posonlyargcount'],f['kwonlyargcount'],f['nlocals'],
                              f['stacksize'],f['flags'],*refs,self.literal(package),int(f['id'].startswith('code:'))])
        classes=[]
        for c in self.data['classes']:
            classes.append([self.value(c.get(k,[] if k=='bases' else {})) for k in
                ('bases','attributes','methods','staticmethods','classmethods','properties')])
        modules=[self.value(self.data['modules'].get(n,{})) for n in self.module_names]
        # All captured constants are retained, including shared or unknown state metadata descriptors.
        for cid in self.cdesc:self.constant(cid)
        def ints(value,width):
            raw=np.asarray(value,dtype='<i4').reshape(-1,width).tobytes()
            return np.frombuffer(raw,dtype='<i4').reshape(-1,width)
        payload=np.frombuffer(bytes(self.payload),dtype=np.uint8)
        entries=MappingProxyType({k:self.fidx[v] for k,v in self.data['entries'].items()})
        required=MappingProxyType({k:MappingProxyType({n:tuple(items) for n,items in self.reachable(v).items()})
                                   for k,v in self.data['entries'].items()})
        image=VmImage(self.data['identity'],'',ints(instructions,4),ints(functions,18),ints(classes,6),
            ints(modules,1),ints(self.nodes,4),ints(self.edges,2),payload,self.opnames,self.native_names,
            tuple(f['id'] for f in self.fns),tuple(c['id'] for c in self.data['classes']),self.module_names,entries,required)
        object.__setattr__(image,'identity',_image_identity(image));image.verify();return image


def lower_program(program):
    if not isinstance(program,FrozenRuleProgram):raise TypeError('Validated frozen source program required')
    return _Lower(program).lower()
