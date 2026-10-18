"""Explicit new wall budget for an immutable verified auxiliary prefix.

A receipt extends scheduling only. Old guards, 512-decision limits, archived
observations/actions/RNG and prior attempts remain binding. Safe checkpoints
use a closed type vocabulary and lossless ndarray bits; never pickle/code.
"""
from copy import deepcopy
from dataclasses import dataclass,fields,is_dataclass
import base64
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

MAX_CONTINUATIONS=8


@dataclass(frozen=True)
class AuxiliaryContinuationBudget:
    old_segment: object
    old_segment_sha256: str
    old_metadata_sha256: str
    new_metadata: object
    new_metadata_sha256: str
    old_deadline: float
    new_deadline: float
    source_binding: dict
    original_steps: tuple
    authorization: str
    parent_report_sha256: str
    parent_checkpoint_sha256: str
    segment_seconds: int=90
    soft_seconds: int=150
    outer_seconds: int=180
    receipt_sha256: str=''


def _types():
    from .device_auxiliary_collection import AuxiliarySegment
    from .seven_auxiliary_jobs import SoAAuxiliaryJobs
    from ..engine.duel_v2.entities import Entity,PlayerEntity,CharacterEntity,CardEntity
    return {t.__name__:t for t in (AuxiliarySegment,SoAAuxiliaryJobs,AuxiliaryContinuationBudget,
                                     Entity,PlayerEntity,CharacterEntity,CardEntity)}


def encode_auxiliary_payload(value,*,max_nodes=1000000):
    """Closed graph JSON vocabulary; aliases and numeric bit patterns survive."""
    nodes=[];seen={};allowed=_types()
    def encode(obj):
        if obj is None or type(obj)in(bool,int,str):return obj
        if type(obj)is float:return {'float64':struct.pack('!d',obj).hex()}
        if id(obj)in seen:return {'ref':seen[id(obj)]}
        if len(nodes)>=max_nodes:raise ValueError('Auxiliary payload node bound exceeded')
        index=len(nodes);seen[id(obj)]=index;nodes.append(None)
        if isinstance(obj,np.ndarray)or isinstance(obj,np.generic):
            array=np.asarray(obj)
            if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in 'biuf':
                raise ValueError('Only plain numeric/bool auxiliary arrays allowed')
            node=dict(kind='scalar'if isinstance(obj,np.generic)else'ndarray',dtype=array.dtype.str,
                      shape=list(array.shape),data=base64.b64encode(array.tobytes(order='C')).decode('ascii'))
        elif is_dataclass(obj)and type(obj).__name__ in allowed and type(obj)is allowed[type(obj).__name__]:
            node=dict(kind='dataclass',type=type(obj).__name__,fields={f.name:encode(getattr(obj,f.name))for f in fields(obj)})
        elif isinstance(obj,dict)and(type(obj)is dict or type(obj)in allowed.values()):
            node=dict(kind='dict',type=type(obj).__name__,items=[[encode(k),encode(v)]for k,v in obj.items()])
        elif type(obj)in(tuple,list):node=dict(kind=type(obj).__name__,items=[encode(v)for v in obj])
        else:raise TypeError('Unsupported auxiliary checkpoint type: '+type(obj).__name__)
        nodes[index]=node;return {'ref':index}
    root=encode(value)
    return dict(schema='auxiliary_lossless_graph_v1',root=root,nodes=nodes)


def decode_auxiliary_payload(payload,*,max_nodes=1000000,max_array_bytes=512<<20):
    if not isinstance(payload,dict)or set(payload)!={'schema','root','nodes'}or payload['schema']!='auxiliary_lossless_graph_v1':
        raise ValueError('Exact auxiliary lossless checkpoint schema required')
    nodes=payload['nodes'];allowed=_types();cache={};active=set();used=[0]
    if not isinstance(nodes,list)or len(nodes)>max_nodes:raise ValueError('Auxiliary payload node bound exceeded')
    def decode(value):
        if value is None or type(value)in(bool,int,str):return value
        if not isinstance(value,dict):raise ValueError('Invalid typed auxiliary value')
        if set(value)=={'float64'}:
            raw=bytes.fromhex(value['float64'])
            if len(raw)!=8:raise ValueError('Exact float64 bits required')
            return struct.unpack('!d',raw)[0]
        if set(value)!={'ref'}or type(value['ref'])is not int or not 0<=value['ref']<len(nodes):
            raise ValueError('Valid auxiliary node reference required')
        i=value['ref']
        if i in cache:return cache[i]
        if i in active:raise ValueError('Cyclic immutable auxiliary payload rejected')
        active.add(i);node=nodes[i]
        if not isinstance(node,dict):raise ValueError('Typed auxiliary node required')
        kind=node.get('kind')
        if kind in ('ndarray','scalar'):
            if set(node)!={'kind','dtype','shape','data'}:raise ValueError('Exact numerical checkpoint node required')
            dtype=np.dtype(node['dtype']);shape=node['shape']
            if (dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biuf'
                    or not isinstance(shape,list)or len(shape)>16 or any(type(v)is not int or v<0 for v in shape)):
                raise ValueError('Plain bounded numeric shape required')
            size=math.prod(shape)*dtype.itemsize;used[0]+=size
            if used[0]>max_array_bytes:raise ValueError('Auxiliary ndarray byte bound exceeded')
            raw=base64.b64decode(node['data'],validate=True)
            if len(raw)!=size:raise ValueError('Auxiliary array bit length differs')
            array=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
            out=array[()]if kind=='scalar'else array
        elif kind=='dict':
            if set(node)!={'kind','type','items'}:raise ValueError('Exact dictionary node required')
            cls=dict if node['type']=='dict'else allowed.get(node['type'])
            if cls is None or not issubclass(cls,dict):raise ValueError('Unapproved auxiliary dictionary type')
            out=cls.__new__(cls);dict.__init__(out);cache[i]=out
            for k,v in node['items']:
                key=decode(k)
                if type(key)not in(str,int)or key in out:raise ValueError('Unique ordinary checkpoint keys required')
                out[key]=decode(v)
        elif kind in ('tuple','list'):
            if set(node)!={'kind','items'}:raise ValueError('Exact sequence node required')
            if kind=='list':
                out=[];cache[i]=out;out.extend(decode(v)for v in node['items'])
            else:out=tuple(decode(v)for v in node['items'])
        elif kind=='dataclass':
            if set(node)!={'kind','type','fields'}:raise ValueError('Exact dataclass node required')
            cls=allowed.get(node['type'])
            if cls is None or not is_dataclass(cls)or set(node['fields'])!={f.name for f in fields(cls)}:
                raise ValueError('Unapproved typed auxiliary checkpoint fields')
            out=cls(**{k:decode(v)for k,v in node['fields'].items()})
        else:raise ValueError('Unapproved auxiliary checkpoint node kind')
        cache[i]=out;active.remove(i);return out
    result=decode(payload['root'])
    if len(cache)!=len(nodes):raise ValueError('Unreachable auxiliary checkpoint nodes rejected')
    return result


def auxiliary_content_sha256(value):
    return sha256(json.dumps(encode_auxiliary_payload(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)


def _receipt_hash(receipt):
    return auxiliary_content_sha256({f.name:getattr(receipt,f.name)for f in fields(receipt)if f.name!='receipt_sha256'})


def _same(a,b):return auxiliary_content_sha256(a)==auxiliary_content_sha256(b)


def _metadata_transition(old,new):
    from .device_auxiliary_collection import _metadata
    _metadata(old);_metadata(new)
    deadlines={job['deadline']for job in old.jobs}
    new_deadlines={job['deadline']for job in new.jobs}
    if len(deadlines)!=1 or len(new_deadlines)!=1:raise ValueError('Uniform exact old/new auxiliary wall deadline required')
    previous,next_deadline=next(iter(deadlines)),next(iter(new_deadlines))
    if not next_deadline>previous:raise ValueError('Explicit continuation deadline must increase')
    expected=deepcopy(old)
    for group in (expected.jobs,expected.identities):
        for job in group:job['deadline']=next_deadline
    if not _same(expected,new):raise ValueError('Continuation may only change scheduling deadlines')
    return previous,next_deadline


def create_auxiliary_continuation(segment,*,deadline,authorization,parent_report_sha256,parent_checkpoint_sha256):
    from dataclasses import replace
    from .device_auxiliary_collection import AuxiliarySegment,_metadata
    from .seven_auxiliary_jobs import adapt_auxiliary_jobs
    if (not isinstance(segment,AuxiliarySegment)or segment.replay_verified is not True
            or segment.decision_limit!=512 or segment.export_errors or all(segment.complete)
            or any(segment.errors) or any(segment.truncated)):
        raise ValueError('Verified unfinished512-decision clean planned prefix required')
    if any(n>=512 and not done for n,done in zip(segment.steps,segment.complete)):
        raise ValueError('Global512 ceiling cannot be extended')
    if set(segment.source_binding)!={'rule_source','image','schema'}or not all(_sha(v)for v in segment.source_binding.values()):
        raise ValueError('Exact frozen rule/image/schema source binding required')
    _metadata(segment.metadata);verify_auxiliary_budget_history(segment)
    if len(segment.budget_continuations)>=MAX_CONTINUATIONS:raise ValueError('Auxiliary continuation chain bound exceeded')
    if not isinstance(authorization,str)or not authorization.strip():raise ValueError('Explicit nonempty new user authorization required')
    if not _sha(parent_report_sha256)or not _sha(parent_checkpoint_sha256):raise ValueError('Exact parent report/checkpoint SHA256 required')
    if type(deadline)not in(int,float)or not math.isfinite(deadline)or deadline<=time.time():
        raise ValueError('Live explicit new absolute continuation deadline required')
    jobs=deepcopy(list(segment.metadata.jobs))
    for job in jobs:job['deadline']=deadline
    new=adapt_auxiliary_jobs(jobs,starting_decks=segment.metadata.starting_decks)
    old_deadline,new_deadline=_metadata_transition(segment.metadata,new)
    old=deepcopy(segment)
    receipt=AuxiliaryContinuationBudget(old,auxiliary_content_sha256(old),auxiliary_content_sha256(old.metadata),
        new,auxiliary_content_sha256(new),old_deadline,new_deadline,deepcopy(old.source_binding),old.steps,
        authorization.strip(),parent_report_sha256,parent_checkpoint_sha256)
    receipt=replace(receipt,receipt_sha256=_receipt_hash(receipt))
    validate_auxiliary_continuation(receipt,resume=segment,metadata=segment.metadata,
        deadline=deadline,source_binding=segment.source_binding)
    return receipt


def validate_auxiliary_continuation(receipt,*,resume,metadata,deadline,source_binding,seconds=90,soft_seconds=150):
    if not isinstance(receipt,AuxiliaryContinuationBudget):raise TypeError('Typed auxiliary continuation receipt required')
    from .device_auxiliary_collection import AuxiliarySegment
    if (not isinstance(resume,AuxiliarySegment) or not isinstance(receipt.old_segment,AuxiliarySegment)
            or resume.replay_verified is not True or resume.export_errors or any(resume.errors)
            or any(resume.truncated) or resume.decision_limit!=512
            or any(any(a.get(k,()))for a in resume.attempts for k in
                ('errors','raw_errors','value_errors','audit_append_errors'))
            or any(a.get('memory_limit_exceeded',False)for a in resume.attempts)):
        raise ValueError('Only verified historically clean planned prefixes may receive new budget')
    if (not _sha(receipt.parent_report_sha256)or not _sha(receipt.parent_checkpoint_sha256)
            or not isinstance(receipt.authorization,str)or not receipt.authorization.strip()
            or receipt.segment_seconds!=90 or receipt.soft_seconds!=150 or receipt.outer_seconds!=180
            or type(seconds)not in(int,float)or not 0<seconds<=90
            or type(soft_seconds)not in(int,float)or not 0<soft_seconds<=150):
        raise ValueError('Explicit parent authorization and original bounded segment limits required')
    if receipt.receipt_sha256!=_receipt_hash(receipt):raise ValueError('Continuation receipt changed')
    if (not _same(resume,receipt.old_segment)or auxiliary_content_sha256(resume)!=receipt.old_segment_sha256
            or auxiliary_content_sha256(resume.metadata)!=receipt.old_metadata_sha256
            or auxiliary_content_sha256(receipt.new_metadata)!=receipt.new_metadata_sha256
            or resume.steps!=receipt.original_steps or not _same(source_binding,receipt.source_binding)
            or not _same(resume.source_binding,receipt.source_binding)):
        raise ValueError('Continuation old prefix/source binding changed')
    old,new=_metadata_transition(resume.metadata,receipt.new_metadata)
    if old!=receipt.old_deadline or new!=receipt.new_deadline or deadline!=new:
        raise ValueError('Continuation absolute budget differs')
    if not(_same(metadata,resume.metadata)or _same(metadata,receipt.new_metadata)):
        raise ValueError('Continuation caller metadata differs beyond deadline')
    return deepcopy(receipt.new_metadata)


def verify_auxiliary_budget_history(segment):
    history=segment.budget_continuations
    if type(history)is not tuple or len(history)>MAX_CONTINUATIONS:raise ValueError('Bounded typed continuation ancestry required')
    for i,receipt in enumerate(history):
        if not isinstance(receipt,AuxiliaryContinuationBudget):raise ValueError('Typed continuation ancestry required')
        old=receipt.old_segment
        if not _same(old.budget_continuations,history[:i]):raise ValueError('Continuation predecessor chain changed')
        validate_auxiliary_continuation(receipt,resume=old,metadata=receipt.new_metadata,
            deadline=receipt.new_deadline,source_binding=segment.source_binding)
        if i+1<len(history):
            if not _same(history[i+1].old_segment.metadata,receipt.new_metadata):raise ValueError('Continuation metadata chain disconnected')
        elif not _same(segment.metadata,receipt.new_metadata):raise ValueError('Output metadata differs from last continuation')
        if not _same(segment.initial,old.initial):raise ValueError('Continuation initial state changed')
        if segment.decision_limit!=old.decision_limit or segment.source_binding!=old.source_binding:
            raise ValueError('Continuation source/global decision limit changed')
        if not _same(segment.attempts[:len(old.attempts)],old.attempts):raise ValueError('Historical attempt statuses changed')
        for name in ('raw_records','audit_records'):
            prefix=getattr(segment,name)[:len(getattr(old,name))]
            if not _same(prefix,getattr(old,name)):raise ValueError('Historical auxiliary '+name+' changed')
        old_values={(r['owner'],r['step']):r for r in old.value_records}
        current={(r['owner'],r['step']):r for r in segment.value_records}
        for coordinate,previous in old_values.items():
            if coordinate not in current:raise ValueError('Historical auxiliary value prefix missing')
            actual=current[coordinate]
            # Flat value-record field insertion order is not a rule property:
            # legacy checkpoint identities were emitted in sorted order,
            # whereas GPU exports use their fixed tensor field order. Keep
            # entity graph/dictionary ordering strict everywhere else.
            before={k:previous[k]for k in sorted(previous)if k not in('z','eligible')}
            after={k:actual[k]for k in sorted(actual)if k not in('z','eligible')}
            if not _same(before,after):raise ValueError('Historical x/context/value metadata changed')
            owner=coordinate[0]
            if not segment.complete[owner] and (not _same(previous['z'],actual['z'])or not _same(previous['eligible'],actual['eligible'])):
                raise ValueError('Only actual natural terminal may update prior finish arrays')
        if len(segment.steps)!=len(old.steps)or any(a<b for a,b in zip(segment.steps,old.steps)):
            raise ValueError('Continuation physical counters moved backwards')
        for owner,n in enumerate(old.steps):
            if segment.steps[owner]==n and not _same(segment.pcg_states[owner],old.pcg_states[owner]):
                raise ValueError('Unadvanced continuation PCG state changed')
    return True


def reverify_auxiliary_evidence(segment,*,proof_deadline,original_checkpoint_sha256,authorization):
    """New independent proof receipt; never alter the original failed archive."""
    from dataclasses import replace
    from .device_auxiliary_collection import verify_auxiliary_segment
    if not _sha(original_checkpoint_sha256)or not isinstance(authorization,str)or not authorization.strip():
        raise ValueError('Explicit original checkpoint and reproof authorization required')
    if (any(segment.errors)or any(segment.truncated)or segment.export_errors
            or any(any(a.get(k,()))for a in segment.attempts
                   for k in('errors','raw_errors','value_errors','audit_append_errors'))
            or any(a.get('memory_limit_exceeded',False)for a in segment.attempts)):
        raise ValueError('Actual historical rule/guard/archive errors cannot be requalified')
    before=auxiliary_content_sha256(segment)
    verify_auxiliary_segment(segment,proof_deadline=proof_deadline)
    checked=replace(segment,replay_verified=True)
    receipt=dict(schema='auxiliary_independent_reproof_v1',authorization=authorization.strip(),
        original_checkpoint_sha256=original_checkpoint_sha256,original_segment_sha256=before,
        verified_segment_sha256=auxiliary_content_sha256(checked),original_replay_verified=segment.replay_verified,
        original_reason=segment.reason,verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_rule_errors=list(segment.errors),steps=list(segment.steps),complete=list(segment.complete),passed=True)
    return checked,receipt


def dump_auxiliary_checkpoint(segment,path):
    """Write a NEW bounded gzip JSON checkpoint; return exact file SHA256."""
    path=Path(path)
    if path.exists():raise ValueError('New auxiliary checkpoint path required')
    payload=encode_auxiliary_payload(segment)
    data=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(data)>512<<20:raise ValueError('Auxiliary checkpoint byte bound exceeded')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as stream:stream.write(data)
    return sha256(path.read_bytes()).hexdigest()


def load_auxiliary_checkpoint(path,*,expected_sha256,max_bytes=512<<20):
    """Load trusted-integrity bytes through the closed safe type decoder only."""
    from .device_auxiliary_collection import AuxiliarySegment
    path=Path(path)
    if type(max_bytes)is not int or not 1<=max_bytes<=512<<20:
        raise ValueError('Positive bounded checkpoint byte limit required')
    if path.stat().st_size>max_bytes:raise ValueError('Auxiliary checkpoint compressed byte bound exceeded')
    if not _sha(expected_sha256)or sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Auxiliary checkpoint file SHA256 differs')
    with gzip.open(path,'rb')as stream:data=stream.read(max_bytes+1)
    if len(data)>max_bytes:raise ValueError('Auxiliary checkpoint decompressed byte bound exceeded')
    result=decode_auxiliary_payload(json.loads(data),max_array_bytes=max_bytes)
    if not isinstance(result,AuxiliarySegment):raise ValueError('Typed auxiliary segment checkpoint required')
    verify_auxiliary_budget_history(result)
    return result
