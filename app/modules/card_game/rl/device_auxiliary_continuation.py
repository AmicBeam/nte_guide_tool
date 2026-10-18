"""Explicit wall budgets and audited backend correction for actual prefixes.

A receipt extends scheduling only. Old guards, 512-decision limits, archived
observations/actions/RNG and prior attempts remain binding. Safe checkpoints
use a closed type vocabulary and lossless ndarray bits; never pickle/code.
Backend correction preserves historical producer evidence and never represents
old GPU trajectories as having been replayed by the corrected kernel.
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


_ORIGINAL_PREVIEW_SOURCE='5fff2041d03b53104f0cb2c61538de3adb6a9b3c3cb94c886a63b020e6da99a2'
_KERNEL_EVENT_KIND='kernel_revalidation'


def _preview_derived_binding(binding,identity):
    from .batched_duel import native_preview_arena as arena
    fixed=dict(backend=arena.BACKEND_ABI,program=arena.PROGRAM_IDENTITY,image=arena.IMAGE_IDENTITY,
        schema=arena.SCHEMA_IDENTITY,reclaim='private typed slots only; heap and strings retained',cuda_qualified=False)
    if (not isinstance(binding,dict)or set(binding)!=set(fixed)|{'canonical_units','derived_units'}
            or any(binding.get(k)!=v for k,v in fixed.items())
            or binding['canonical_units']!=dict(arena.CANONICAL_UNIT_SHA256)):
        raise ValueError('Exact opt-in preview backend derivation required')
    actual=binding['derived_units'];expected=dict(arena.CANONICAL_UNIT_SHA256)
    if (not isinstance(actual,dict)or set(actual)!=set(expected)or not all(_sha(v)for v in actual.values())
            or actual[arena.TARGET_UNIT]==expected[arena.TARGET_UNIT]
            or any(actual[k]!=v for k,v in expected.items()if k!=arena.TARGET_UNIT)
            or not _sha(identity)or arena._digest(binding)!=identity):
        raise ValueError('Exactly21 unchanged units and independently bound derived unit required')
    return dict(rule_source=identity,image=arena.IMAGE_IDENTITY,schema=arena.SCHEMA_IDENTITY)


def _read_kernel_file(path,expected_sha256,*,maximum=256<<20):
    path=Path(path)
    if not _sha(expected_sha256)or not path.is_file()or path.stat().st_size>maximum:
        raise ValueError('Bounded actual kernel evidence file/SHA required')
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Actual kernel evidence file SHA differs')
    return raw


def _kernel_implementation_hashes():
    from .batched_duel import native_preview_arena
    directory=Path(native_preview_arena.__file__).resolve().parent
    return {name:sha256((directory/(name+'.py')).read_bytes()).hexdigest()
            for name in ('native_preview_arena','native_soa_cuda','native_soa_cuda_query')}


def _diagnostic_raw_files(value):
    """Enumerate every output-local numeric/graph archive in the raw report.

    Source module paths and the external historical parent checkpoint retain
    their separate source/ledger bindings; they are not output-local archives.
    """
    result={};active=set();seen=set()
    def walk(obj):
        compound=isinstance(obj,(dict,list,tuple))
        if compound:
            if id(obj)in active:raise ValueError('Cyclic diagnostic manifest rejected')
            seen.add(id(obj))
            if len(seen)>1000000:raise ValueError('Diagnostic manifest node bound exceeded')
            active.add(id(obj))
        if isinstance(obj,dict):
            if set(obj)=={'path','sha256'}and isinstance(obj['path'],str):
                name=obj['path'];path=Path(name)
                if ('\\'not in name and not path.is_absolute()and '..'not in path.parts
                        and name.endswith(('.npz','.json.gz'))):
                    if name in result and result[name]!=obj['sha256']:raise ValueError('Conflicting original raw archive identity')
                    result[name]=obj['sha256']
            for child in obj.values():walk(child)
        elif isinstance(obj,(list,tuple)):
            for child in obj:walk(child)
        if compound:active.remove(id(obj))
    walk(value)
    return result


def verify_auxiliary_kernel_fix_report(path,*,expected_sha256,proof_deadline):
    """Read actual bounded GPU archives, not a ready/approval JSON flag.

    All descriptor paths must stay within the adapter's directory. The returned
    lossless numeric snapshot supports later CPU revalidation after repatriation;
    no Windows paths or executable objects are required for that revalidation.
    """
    import io
    import zipfile
    path=Path(path).resolve()
    report=json.loads(_read_kernel_file(path,expected_sha256,maximum=32<<20))
    if report.get('schema')!='auxiliary_preview_kernel_fix_v1':raise ValueError('Explicit kernel-fix adapter required')
    roles={'frozen_program','v85_raw','v85_decoded','cpu_input','cpu_clean','cpu_expected_rule','original_parent_after'}
    files=report.get('files');implementations=report.get('implementation_files')
    if not isinstance(files,dict)or set(files)!=roles or not isinstance(implementations,dict)or set(implementations)!=set(_kernel_implementation_hashes()):
        raise ValueError('Full raw/status/state/source evidence roles required')
    extra=report.get('additional_raw_files')
    if not isinstance(extra,list):raise ValueError('Full original raw-file manifest required')
    checked={};data={}
    def read(desc):
        if time.monotonic()>=proof_deadline:raise TimeoutError('Kernel evidence proof deadline exhausted')
        if not isinstance(desc,dict)or set(desc)!={'path','sha256'}or not isinstance(desc['path'],str):
            raise ValueError('Exact raw evidence descriptor required')
        name=desc['path'];actual=(path.parent/name).resolve()
        if Path(name).is_absolute()or not actual.is_relative_to(path.parent)or actual==path:
            raise ValueError('Contained immutable kernel evidence path required')
        raw=_read_kernel_file(actual,desc['sha256'])
        if name in checked and checked[name]!=desc['sha256']:raise ValueError('Conflicting raw file identity')
        checked[name]=desc['sha256'];return raw
    for desc in extra:read(desc)
    for name,desc in implementations.items():
        raw=read(desc)
        if sha256(raw).hexdigest()!=_kernel_implementation_hashes()[name]:
            raise ValueError('GPU implementation source differs from current exact package')
    for role,desc in files.items():
        raw=read(desc)
        if role=='frozen_program':data[role]=json.loads(raw)
        elif role=='v85_raw':
            with zipfile.ZipFile(io.BytesIO(raw))as archive:
                if sum(i.file_size for i in archive.infolist())>256<<20:raise ValueError('Kernel NPZ expansion bound exceeded')
            with np.load(io.BytesIO(raw),allow_pickle=False)as archive:
                data[role]={k:archive[k].copy()for k in archive.files}
        else:
            with gzip.GzipFile(fileobj=io.BytesIO(raw))as stream:payload=stream.read((256<<20)+1)
            if len(payload)>256<<20:raise ValueError('Kernel graph expansion bound exceeded')
            data[role]=decode_auxiliary_payload(json.loads(payload),max_array_bytes=256<<20)
    original_files=_diagnostic_raw_files(report.get('diagnostic',{}))
    if any(checked.get(k)!=v for k,v in original_files.items()):
        raise ValueError('Original diagnostic raw-file manifest is incomplete')
    evidence=dict(schema='auxiliary_preview_kernel_fix_snapshot_v1',adapter_sha256=expected_sha256,
        report=report,file_sha256=checked,implementation_sha256=_kernel_implementation_hashes(),numeric=data)
    verify_auxiliary_kernel_fix_snapshot(evidence,proof_deadline=proof_deadline)
    return evidence


def verify_auxiliary_kernel_fix_snapshot(evidence,*,proof_deadline=None):
    """Recheck a self-contained actual archive snapshot; never execute GPU rules."""
    from types import SimpleNamespace
    from .batched_duel import native_preview_arena as arena
    from .batched_duel.domain_soa_records import compile_execution_schema
    from .batched_duel.vm_image import lower_program
    from .rule_ir.portable_program import FrozenRuleProgram
    from .batched_duel.record_soa import RecordSoABatch,unpack_game_states
    from .batched_duel.native_soa_cpu import NativeSoACall
    from .device_auxiliary_collection import _same_graph,_oracle
    if proof_deadline is not None and time.monotonic()>=proof_deadline:raise TimeoutError('Kernel snapshot proof deadline exhausted')
    if (not isinstance(evidence,dict)or set(evidence)!={'schema','adapter_sha256','report','file_sha256','implementation_sha256','numeric'}
            or evidence['schema']!='auxiliary_preview_kernel_fix_snapshot_v1'or not _sha(evidence['adapter_sha256'])
            or evidence['implementation_sha256']!=_kernel_implementation_hashes()):
        raise ValueError('Exact immutable kernel evidence snapshot required')
    report=evidence['report'];numeric=evidence['numeric']
    source=_preview_derived_binding(report.get('derived_binding'),report.get('derived_identity'))
    original=dict(source,rule_source=_ORIGINAL_PREVIEW_SOURCE)
    if report.get('original_source_binding')!=original:raise ValueError('Only the exact original preview backend may migrate')
    descriptors=[*report['files'].values(),*report['implementation_files'].values(),*report['additional_raw_files']]
    if any(evidence['file_sha256'].get(d.get('path'))!=d.get('sha256')or not _sha(d.get('sha256'))for d in descriptors):
        raise ValueError('Actual full raw-file SHA receipts disconnected')
    if any(evidence['file_sha256'].get(k)!=v for k,v in _diagnostic_raw_files(report.get('diagnostic',{})).items()):
        raise ValueError('Original diagnostic raw-file receipt omitted')
    diagnostic=report.get('diagnostic',{});query=diagnostic.get('v85_rule_query',{})
    profile=diagnostic.get('rule_compilation',{});units=profile.get('compilation_units',[])
    if (len(units)!=22 or len({u.get('name')for u in units})!=22
            or {u.get('name'):u.get('source_sha256')for u in units}!=report['derived_binding']['derived_units']
            or any(u.get('cache_status')!='hit'for u in units if u.get('name')!=arena.TARGET_UNIT)
            or any(u.get('cache_status')not in('hit','miss')for u in units)
            or profile.get('device_stack_bytes')!=131072):
        raise ValueError('Actual22-unit profile/21 cache hits required')
    if (diagnostic.get('cpu_preflight')is not False or diagnostic.get('diagnostic_complete')is not True
            or diagnostic.get('parity_passed')is not True or diagnostic.get('cleanup_errors')
            or diagnostic.get('training_games_counted')!=0 or diagnostic.get('learning_updates')!=0
            or diagnostic.get('query_parent_purity',{}).get('passed')is not True
            or any(query.get(k,{}).get('passed')is not True for k in('legal_parity','choice_parity','query_parent_purity'))):
        raise ValueError('Actual successful diagnostic-only GPU query/purity evidence required')
    program=FrozenRuleProgram(deepcopy(numeric['frozen_program']));image=lower_program(program);schema=compile_execution_schema()
    arena.verify_frozen_preview(program,image,schema)
    raw=numeric['v85_raw'];required={'outputs','statuses','roots','views','heap_counts','heap_nodes','heap_edges','heap_payload',
        'typed_i64','typed_f64','typed_i32','typed_u8','typed_input_roots'}
    if set(raw)!=required or any(not isinstance(a,np.ndarray)or a.dtype.hasobject or a.dtype.kind not in'biuf'for a in raw.values()):
        raise ValueError('Actual closed numeric GPU call archive required')
    statuses=raw['statuses']
    if (statuses.shape!=(1,4)or statuses.dtype!=np.int32 or statuses[:,0].any()
            or statuses[0,1]!=image.entries['rule_decision']or statuses[0,2]!=-1
            or not 0<int(statuses[0,3])<=48000000):
        raise ValueError('Actual v85 native call must have zero statuses')
    if query.get('statuses')!=raw['statuses'].tolist():raise ValueError('GPU report/status archive differs')
    state=RecordSoABatch(schema,1,buffers={k:raw['typed_'+k]for k in('i64','f64','i32','u8')},roots=raw['typed_input_roots'])
    backend=SimpleNamespace(image=image,_constants={k:getattr(image,k)for k in('nodes','edges','payload')})
    call=NativeSoACall(backend,state,{k:raw['heap_'+k]for k in('counts','nodes','edges','payload')},
        raw['outputs'],raw['statuses'],raw['roots'],raw['views'],())
    decoded=call.decode();_same_graph(decoded,numeric['v85_decoded'])
    new,observe,actor,apply,clean,encode,rule=_oracle()
    initial=numeric['cpu_input'];parent=numeric['original_parent_after']
    if isinstance(parent,(list,tuple))and len(parent)==1:parent=parent[0]
    _same_graph(initial,parent)
    cohort_steps=diagnostic.get('original_cohort_steps')
    if (initial.get('version')!=84 or diagnostic.get('row')!=3
            or not isinstance(cohort_steps,list)or len(cohort_steps)!=8 or cohort_steps[3]!=84
            or not _sha(diagnostic.get('parent_checkpoint',{}).get('sha256'))
            or report['action']!=diagnostic.get('expected_action')
            or report['action'].get('type')!='play_card'or report['action'].get('card_id')!='b-37'):
        raise ValueError('Exact original row3/v84/playb-37 diagnostic parent required')
    expected=clean(apply(deepcopy(initial),actor(initial),deepcopy(report['action'])))
    if expected.get('version')!=85:raise ValueError('Exact v85 post-action diagnostic required')
    _same_graph(expected,numeric['cpu_clean']);_same_graph(unpack_game_states(state),[expected])
    before=deepcopy(expected);legal,index=rule(expected,actor(expected));_same_graph(expected,before)
    _same_graph((legal,int(index)),numeric['cpu_expected_rule']);_same_graph(decoded,[(legal,int(index))])
    if proof_deadline is not None and time.monotonic()>=proof_deadline:raise TimeoutError('Kernel snapshot proof deadline exhausted')
    return source


def _kernel_clean(segment):
    from .device_auxiliary_collection import AuxiliarySegment
    return (isinstance(segment,AuxiliarySegment)and segment.decision_limit==512
        and not segment.export_errors and not any(segment.errors)and not any(segment.truncated)
        and all(not any(a.get(k,()))for a in segment.attempts for k in
                ('errors','raw_errors','value_errors','audit_append_errors'))
        and not any(a.get('memory_limit_exceeded',False)for a in segment.attempts))


def _kernel_event_hash(event):
    return auxiliary_content_sha256({k:v for k,v in event.items()if k!='event_sha256'})


def _verify_kernel_origin(event):
    """Old rejected ledger stays rejected; only its exact clean checkpoint moves."""
    old=event['old_segment'];ledger=event['old_ledger'];report=event['old_failed_report']
    if (not _kernel_clean(old)or old.replay_verified is not True
            or any(a.get('kind')==_KERNEL_EVENT_KIND for a in old.attempts)
            or auxiliary_content_sha256(old)!=event['old_segment_sha256']
            or not _same(old.budget_continuations,event['old_budget_continuations'])
            or old.source_binding!=event['old_source_binding']):
        raise ValueError('Only unchanged historically clean source snapshots may migrate')
    verify_auxiliary_budget_history(old)
    from .device_auxiliary_collection import _metadata,_coordinates
    _metadata(old.metadata)
    if any(len(getattr(old,k))!=len(old.steps)for k in('initial','final','complete','truncated','errors','pcg_states')):
        raise ValueError('Full historical kernel cohort required')
    if (type(old.decision_limit)is not int or old.formal_training_qualified is not False or old.gpu_qualified is not False
            or any(type(n)is not int or not 0<=n<=512 for n in old.steps)
            or any(type(v)is not bool for v in old.complete)
            or any(v is not False for v in old.truncated)
            or any(type(v)is not int or v!=0 for v in old.errors)):
        raise ValueError('Exact clean bounded historical outcome types required')
    expected={(r,s)for r,n in enumerate(old.steps)for s in range(n)}
    if any(set(_coordinates(getattr(old,name),len(old.steps)))!=expected for name in('raw_records','audit_records','value_records')):
        raise ValueError('Historical kernel archive coverage changed')
    for value in old.value_records:
        owner=value['owner'];z=value.get('z');eligible=value.get('eligible')
        if (not isinstance(z,np.ndarray)or z.dtype!=np.float32 or z.shape!=(2,)
                or not isinstance(eligible,np.ndarray)or eligible.dtype!=np.bool_ or eligible.shape!=(2,)):
            raise ValueError('Exact original finish arrays required')
        if not old.complete[owner]:
            if z.any()or eligible.any():raise ValueError('Old unfinished prefix cannot have finish labels')
        else:
            winner=old.final[owner].get('winner')
            wanted=np.asarray([0 if winner=='draw'else 1 if winner==s else -1 for s in('a','b')],np.float32)
            if winner not in('a','b','draw')or not eligible.all()or not np.array_equal(z,wanted):
                raise ValueError('Old natural finish labels changed')
    if (ledger.get('contract',{}).get('source_binding')!=old.source_binding
            or sha256(json.dumps(ledger.get('contract'),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=ledger.get('contract_sha256')
            or report.get('ledger_contract_sha256')!=ledger.get('contract_sha256')
            or not any(e.get('status')=='REJECTED'for e in ledger.get('cohorts',[]))
            or report.get('ledger_manifest_sha256')!=event['old_ledger_sha256']
            or report.get('source_binding')!=old.source_binding
            or not any(w.get('status')=='rejected'and w.get('manifest_sha256')==event['old_ledger_sha256']for w in report.get('windows',[]))):
        raise ValueError('Actual preserved rejected-ledger/failure-report backreference required')
    physical=[{k:deepcopy(v)for k,v in j.items()if k!='deadline'}for j in old.metadata.jobs]
    matched=[]
    for entry in ledger.get('cohorts',[]):
        jobs=ledger.get('contract',{}).get('physical_plans',{}).get(entry.get('phase'),[])
        offset,count=entry.get('offset'),entry.get('count')
        if type(offset)is not int or type(count)is not int:continue
        if jobs[offset:offset+count]!=physical:continue
        for checkpoint in entry.get('checkpoints',[]):
            if (checkpoint.get('sha256')==event['old_checkpoint_sha256']
                    and checkpoint.get('content_sha256')==event['old_segment_sha256']):matched.append(entry)
    if len(matched)!=1:raise ValueError('Exact clean parent checkpoint must exist in preserved ledger history')
    diagnostic=event['fix_evidence']['report']['diagnostic']
    checkpoint_sha=diagnostic.get('parent_checkpoint',{}).get('sha256');found=[]
    for entry in ledger.get('cohorts',[]):
        for checkpoint in entry.get('checkpoints',[]):
            if checkpoint.get('sha256')==checkpoint_sha:found.append(entry)
    if len(found)!=1:raise ValueError('Fix diagnostic must bind an actual clean parent from the same preserved ledger')
    entry=found[0];jobs=ledger['contract']['physical_plans'][entry['phase']][entry['offset']:entry['offset']+entry['count']]
    if (len(jobs)!=8 or {k:v for k,v in diagnostic.get('original_job',{}).items()if k!='deadline'}!=jobs[3]):
        raise ValueError('Fix diagnostic row/physical-job lineage disconnected')
    if physical==jobs and old.steps[3]==84:
        from .device_auxiliary_collection import _same_graph
        _same_graph(old.final[3],event['fix_evidence']['numeric']['cpu_input'])
    return old


def auxiliary_kernel_lineage(segment):
    """Small portable lineage for game/ledger reports; full proof stays private."""
    out=[]
    for event in segment.attempts:
        if event.get('kind')==_KERNEL_EVENT_KIND:
            item={k:deepcopy(event[k])for k in('kind','event_sha256','gpu_replay_executed',
                'old_segment_sha256','old_checkpoint_sha256','old_source_binding','new_source_binding',
                'old_failed_report_sha256','old_ledger_sha256','fix_report_sha256')}
            item['original_steps']=list(event['original_steps'])
            out.append(item)
    return out


def verify_auxiliary_kernel_history(segment,*,proof_deadline=None):
    """Verify immutable lineage, returning boundaries for the ONE CPU replay.

    The caller must compare these old final states/PCGs at their exact original
    physical step inside its normal current-prefix replay. No old GPU trajectory
    is relabelled as having executed under the new backend.
    """
    if any(a.get('kind')!=_KERNEL_EVENT_KIND and (a.get('schema')=='auxiliary_kernel_revalidation_v1'
            or 'fix_evidence'in a or 'old_source_binding'in a)for a in segment.attempts):
        raise ValueError('Malformed/hidden kernel revalidation event')
    events=[a for a in segment.attempts if a.get('kind')==_KERNEL_EVENT_KIND]
    if not events:return ()
    if len(events)!=1:raise ValueError('Only one exact original-to-preview kernel transition supported')
    event=events[0]
    required={'kind','schema','gpu_replay_executed','authorization','old_segment','old_segment_sha256',
        'old_checkpoint_sha256','old_source_binding','old_budget_continuations','new_source_binding',
        'old_failed_report','old_failed_report_sha256','old_ledger','old_ledger_sha256','fix_report_sha256',
        'fix_evidence','original_steps','errors','raw_errors','value_errors','audit_append_errors',
        'memory_limit_exceeded','event_sha256'}
    if (set(event)!=required or event['schema']!='auxiliary_kernel_revalidation_v1'
            or event['gpu_replay_executed']is not False or not isinstance(event['authorization'],str)
            or not event['authorization'].strip()or not all(_sha(event[k])for k in(
                'old_segment_sha256','old_checkpoint_sha256','old_failed_report_sha256','old_ledger_sha256','fix_report_sha256','event_sha256'))
            or event['event_sha256']!=_kernel_event_hash(event)):
        raise ValueError('Exact explicit immutable kernel-revalidation event required')
    old=_verify_kernel_origin(event)
    source=verify_auxiliary_kernel_fix_snapshot(event['fix_evidence'],proof_deadline=proof_deadline)
    if (event['fix_report_sha256']!=event['fix_evidence']['adapter_sha256']
            or event['new_source_binding']!=source or segment.source_binding!=source
            or old.source_binding!=event['fix_evidence']['report']['original_source_binding']
            or event['original_steps']!=old.steps):
        raise ValueError('Source migration/fix evidence disconnected')
    rows=len(old.steps)
    if (any(event[k]!=(0,)*rows for k in('errors','raw_errors','value_errors','audit_append_errors'))
            or event['memory_limit_exceeded']is not False):raise ValueError('Kernel revalidation is not a GPU replay attempt')
    n=len(old.attempts)
    if len(segment.attempts)<=n or not _same(segment.attempts[:n],old.attempts)or segment.attempts[n]is not event:
        raise ValueError('Original attempts/order cannot change during migration')
    base=segment.budget_continuations[0].old_segment if segment.budget_continuations else segment
    if (not _same(base.metadata,old.metadata)or not _same(segment.initial,old.initial)
            or segment.decision_limit!=old.decision_limit or len(segment.steps)!=rows
            or any(a<b for a,b in zip(segment.steps,old.steps))):
        raise ValueError('Kernel migration changed metadata/initial/global counters')
    for name in('raw_records','audit_records'):
        if not _same(getattr(segment,name)[:len(getattr(old,name))],getattr(old,name)):
            raise ValueError('Kernel migration changed historical raw/audit prefix')
    values={(r['owner'],r['step']):r for r in segment.value_records}
    for previous in old.value_records:
        actual=values.get((previous['owner'],previous['step']))
        if actual is None:raise ValueError('Kernel migration lost historical value coordinate')
        before={k:previous[k]for k in sorted(previous)if k not in('z','eligible')}
        after={k:actual[k]for k in sorted(actual)if k not in('z','eligible')}
        if not _same(before,after):raise ValueError('Kernel migration changed historical x/context/value metadata')
        if (old.complete[previous['owner']]or not segment.complete[previous['owner']])and any(
                not _same(previous[k],actual[k])for k in('z','eligible')):
            raise ValueError('Only later actual natural terminal may update old finish labels')
    for owner,steps in enumerate(old.steps):
        if old.complete[owner]and segment.steps[owner]!=steps:raise ValueError('Already complete historical game cannot advance')
        if segment.steps[owner]==steps and (not _same(segment.final[owner],old.final[owner])
                or not _same(segment.pcg_states[owner],old.pcg_states[owner])or segment.complete[owner]!=old.complete[owner]):
            raise ValueError('Unadvanced historical state/RNG/outcome changed')
    return (old,)


def create_auxiliary_kernel_revalidation(segment,*,proof_deadline,original_checkpoint_sha256,
        old_ledger_path,old_ledger_sha256,old_failed_report_path,old_failed_report_sha256,
        fix_report_path,fix_report_sha256,authorization):
    """Explicit corrective-ledger ingress; never modify old ledger/checkpoints.

    This performs complete CPU proof and actual GPU-archive validation, NOT a
    new GPU replay. Active scheduling receipts restart at this source boundary;
    every old receipt and its full snapshot remain in the immutable event.
    """
    from dataclasses import replace
    from .device_auxiliary_collection import verify_auxiliary_segment
    if (not _kernel_clean(segment)or segment.replay_verified is not True
            or not isinstance(authorization,str)or not authorization.strip()
            or not _sha(original_checkpoint_sha256)
            or any(a.get('kind')==_KERNEL_EVENT_KIND for a in segment.attempts)):
        raise ValueError('Explicit authorization and historically clean actual prefix required')
    ledger_raw=_read_kernel_file(old_ledger_path,old_ledger_sha256,maximum=32<<20)
    report_raw=_read_kernel_file(old_failed_report_path,old_failed_report_sha256,maximum=32<<20)
    fix=verify_auxiliary_kernel_fix_report(fix_report_path,expected_sha256=fix_report_sha256,proof_deadline=proof_deadline)
    new=_preview_derived_binding(fix['report']['derived_binding'],fix['report']['derived_identity'])
    old=deepcopy(segment);rows=len(old.steps)
    event=dict(kind=_KERNEL_EVENT_KIND,schema='auxiliary_kernel_revalidation_v1',gpu_replay_executed=False,
        authorization=authorization.strip(),old_segment=old,old_segment_sha256=auxiliary_content_sha256(old),
        old_checkpoint_sha256=original_checkpoint_sha256,old_source_binding=deepcopy(old.source_binding),
        old_budget_continuations=old.budget_continuations,new_source_binding=new,
        old_failed_report=json.loads(report_raw),old_failed_report_sha256=old_failed_report_sha256,
        old_ledger=json.loads(ledger_raw),old_ledger_sha256=old_ledger_sha256,fix_report_sha256=fix_report_sha256,
        fix_evidence=fix,original_steps=old.steps,errors=(0,)*rows,raw_errors=(0,)*rows,
        value_errors=(0,)*rows,audit_append_errors=(0,)*rows,memory_limit_exceeded=False)
    event['event_sha256']=_kernel_event_hash(event)
    _verify_kernel_origin(event)
    verify_auxiliary_segment(old,proof_deadline=proof_deadline)
    migrated=replace(deepcopy(segment),source_binding=deepcopy(new),budget_continuations=(),
                     attempts=deepcopy(segment.attempts)+(event,),replay_verified=True)
    verify_auxiliary_kernel_history(migrated,proof_deadline=proof_deadline)
    return migrated,auxiliary_kernel_lineage(migrated)[0]


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
