"""Explicit, externally anchored auxiliary-prefix proof reuse.

No JSON flag grants a proof. A trusted supervisor pins the final witness SHA,
worker SHA and immutable committed checkpoint/revision after actual CPU exit.
Only this loader can mint process-local capabilities; checkpoint encoding and
legacy full verification are unchanged. No CUDA or model arrays are loaded.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import time
from weakref import WeakKeyDictionary


@dataclass(frozen=True)
class AuxiliaryProofAnchor:
    directory: Path
    witness: str
    witness_sha256: str
    worker_sha256: str


_SEAL=object()
_CAPABILITIES=WeakKeyDictionary()


class AuxiliaryParentProof:
    __slots__=('__weakref__',)
    def __init__(self,seal=None):
        if seal is not _SEAL:raise TypeError('Load an externally anchored actual CPU proof witness')


def _sha(value):
    return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def implementation_binding():
    """Bind the oracle, encoders, sampler, receipt checks and ledger semantics."""
    from .seven_lineup import identity
    from . import seven_covered_value
    import os,sys,numpy as np
    root=Path(__file__).resolve().parent
    sources=[];links={}
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[name for name in dirs if name!='__pycache__']
        for name in dirs:
            path=Path(directory)/name
            if path.is_symlink():
                target=path.resolve()
                if not target.is_relative_to(root):
                    raise ValueError('RL proof implementation tree contains an external directory symlink')
                links[path.relative_to(root).as_posix()]=target.relative_to(root).as_posix()
        for name in names:
            if not name.endswith('.py'):continue
            path=Path(directory)/name
            target=path.resolve()
            if not target.is_relative_to(root):
                raise ValueError('RL proof implementation tree contains an external Python symlink')
            if path.is_symlink():links[path.relative_to(root).as_posix()]=target.relative_to(root).as_posix()
            sources.append(path)
    # The rule identity covers all engine/duel_v2 and content/duel_v2 Python,
    # catalog.json, league_schema and league_observation. Bind the remaining
    # RL transitive helper closure conservatively, including newly added files.
    sources.sort(key=lambda path:path.relative_to(root).as_posix())
    # This directly imported shared V2 event vocabulary is outside the two
    # trees covered by the historical rule hash; do not change that old hash.
    events=root.parent/'engine/events.py'
    if not events.resolve().is_relative_to(root.parent):
        raise ValueError('Shared V2 rule helper resolves outside the card-game module')
    return dict(rule_helpers={'engine/events.py':sha256(events.read_bytes()).hexdigest()},python=list(sys.version_info[:3]),numpy=np.__version__,
        symlinks=dict(sorted(links.items())),files={path.relative_to(root).as_posix():sha256(path.read_bytes()).hexdigest()for path in sources},
        rule_hash=identity(),auxiliary_algorithm_sha256=seven_covered_value.identity())


def _file(root,descriptor):
    if not isinstance(descriptor,dict)or set(descriptor)!={'path','sha256'}or not _sha(descriptor['sha256']):
        raise ValueError('Exact immutable proof artifact path/SHA required')
    name=descriptor['path']
    if (not isinstance(name,str)or not name or '\0'in name or PureWindowsPath(name).drive
            or Path(name.replace('\\','/')).is_absolute()):raise ValueError('Contained relative proof artifact required')
    path=(root/name.replace('\\','/')).resolve()
    if not path.is_relative_to(root)or not path.is_file()or path.is_symlink():raise ValueError('Contained actual proof file required')
    if sha256(path.read_bytes()).hexdigest()!=descriptor['sha256']:raise ValueError('Immutable proof artifact SHA changed')
    return path


def _read(path):
    if path.stat().st_size>16<<20:raise ValueError('Bounded proof JSON required')
    return json.loads(path.read_text(encoding='utf8'))


def _committed(root,checkpoint,revision,*,phase,offset):
    from .device_auxiliary_calibration import _manifest_sha,_physical_job,DeviceAuxiliaryCalibrationLedger,PHASE_COUNTS
    from .seven_schedule import ROSTER
    from .device_auxiliary_continuation import load_auxiliary_checkpoint,auxiliary_content_sha256
    if phase not in('training','selection','heldout')or type(offset)is not int or offset<0:
        raise ValueError('Exact physical phase/integer offset required')
    cp=_file(root,checkpoint);rev=_file(root,revision);manifest=_read(rev)
    if rev.name!=f"manifest-revision-{manifest.get('revision',-1):08d}.json":raise ValueError('Actual immutable committed revision required')
    contract=manifest.get('contract',{})
    impl=implementation_binding()
    if (manifest.get('contract_sha256')!=_manifest_sha(contract)or contract.get('schema')!='device_auxiliary_ledger_v1'
            or contract.get('phase_counts')!=PHASE_COUNTS or contract.get('decision_limit')!=512
            or contract.get('rule_hash')!=impl['rule_hash']or contract.get('algorithm_sha256')!=impl['auxiliary_algorithm_sha256']
            or set(contract.get('source_model_versions',{}))!=set(ROSTER)
            or any(not _sha(v)for v in contract['source_model_versions'].values())):
        raise ValueError('Committed plan/model/rules/algorithm contract differs')
    if any(e.get('status')in('RUNNING','REJECTED')for e in manifest.get('cohorts',[])):
        raise ValueError('Uncertain/rejected ledger cannot attest a clean CPU prefix')
    entries=[e for e in manifest['cohorts']if e['phase']==phase and e['offset']==offset]
    if len(entries)!=1 or entries[0]['status']not in('PAUSED','COMPLETE'):raise ValueError('Actual clean committed cohort required')
    entry=entries[0];latest=entry['checkpoints'][-1]
    if (latest['sha256']!=checkpoint['sha256']or(rev.parent/latest['path']).resolve()!=cp
            or not _sha(latest.get('content_sha256'))):raise ValueError('Checkpoint is not this immutable revision/cohort head')
    segment=load_auxiliary_checkpoint(cp,expected_sha256=checkpoint['sha256'])
    if (not DeviceAuxiliaryCalibrationLedger._clean(segment)or segment.decision_limit!=512
            or segment.source_binding!=contract.get('source_binding')
            or segment.metadata.starting_decks!=contract.get('starting_decks')
            or [_physical_job(j)for j in segment.metadata.jobs]!=contract['physical_plans'][phase][offset:offset+entry['count']]
            or auxiliary_content_sha256(segment)!=latest['content_sha256']
            or list(segment.steps)!=latest['steps']or list(segment.complete)!=latest['complete']
            or list(segment.errors)!=latest['errors']or list(segment.truncated)!=latest['truncated']):
        raise ValueError('Exact clean committed prefix/content/physical plan required')
    return segment,manifest,entry


def _context(proof):
    if type(proof)is not AuxiliaryParentProof or proof not in _CAPABILITIES:
        raise TypeError('Actual externally anchored parent-proof capability required')
    value=_CAPABILITIES[proof]
    if value['implementation']!=implementation_binding():raise ValueError('Prefix proof verifier/rules/encoding implementation changed')
    for path,digest in value['files'].items():
        if not path.is_file()or sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Persisted trusted proof/checkpoint/process evidence changed after loading')
    return value


def require_auxiliary_parent_proof(proof,*,checkpoint_sha256=None,contract_sha256=None,phase=None,offset=None):
    value=_context(proof);claim=value['claim']
    checks=((checkpoint_sha256,claim['checkpoint']['sha256']),(contract_sha256,claim['contract_sha256']),
            (phase,claim['phase']),(offset,claim['offset']))
    if any(expected is not None and expected!=actual for expected,actual in checks):
        raise ValueError('Parent proof belongs to a different checkpoint/contract/cohort')
    return dict(witness_sha256=value['witness_sha256'],claim_sha256=value['claim_sha256'],
                checkpoint_sha256=claim['checkpoint']['sha256'],proof_mode=claim['proof_mode'])


def verify_auxiliary_parent_checkpoint(segment,parent_proof):
    from .device_auxiliary_continuation import auxiliary_content_sha256
    value=_context(parent_proof)
    if auxiliary_content_sha256(segment)!=value['claim']['content_sha256']:
        raise ValueError('GPU resume input differs from exact independently proved parent checkpoint')
    return True


def _verified_parent_for_segment(proof,segment):
    from .device_auxiliary_continuation import _same,auxiliary_content_sha256
    from .device_auxiliary_collection import _coordinates
    from .device_auxiliary_calibration import DeviceAuxiliaryCalibrationLedger
    from dataclasses import replace
    value=_context(proof);old=value['parent']
    if not DeviceAuxiliaryCalibrationLedger._clean(replace(segment,replay_verified=True)):
        raise ValueError('Actual rule/guard/archive/memory failure cannot use inductive proof')
    if (segment.source_binding!=old.source_binding or segment.decision_limit!=old.decision_limit
            or len(segment.steps)!=len(old.steps)or not _same(segment.initial,old.initial)
            or segment.metadata.starting_decks!=old.metadata.starting_decks):
        raise ValueError('Inductive parent source/initial/build/cohort changed')
    # Budget-history verification remains mandatory in the caller. A changed
    # deadline additionally needs an exact typed extension of THIS parent.
    if not _same(segment.metadata,old.metadata):
        digest=auxiliary_content_sha256(old)
        if not any(r.old_segment_sha256==digest and _same(r.new_metadata,segment.metadata)
                   for r in segment.budget_continuations[len(old.budget_continuations):]):
            raise ValueError('Metadata differs without exact typed parent budget continuation')
    if not _same(segment.budget_continuations[:len(old.budget_continuations)],old.budget_continuations):
        raise ValueError('Previously proved budget history changed')
    if not _same(segment.attempts[:len(old.attempts)],old.attempts):raise ValueError('Previously proved attempts/kernel evidence changed')
    if any(a.get('kind')=='kernel_revalidation'or'fix_evidence'in a or'old_source_binding'in a
            or a.get('schema')=='auxiliary_kernel_revalidation_v1'for a in segment.attempts[len(old.attempts):]):
        raise ValueError('New kernel transitions require full bootstrap proof')
    for name in('raw_records','audit_records'):
        if not _same(getattr(segment,name)[:len(getattr(old,name))],getattr(old,name)):
            raise ValueError('Previously proved '+name+' prefix changed')
    values=_coordinates(segment.value_records,len(old.steps))
    for previous in old.value_records:
        row=values.get((previous['owner'],previous['step']))
        if row is None:raise ValueError('Previously proved value row missing')
        before={k:previous[k]for k in sorted(previous)if k not in('z','eligible')}
        after={k:row[k]for k in sorted(row)if k not in('z','eligible')}
        if not _same(before,after):raise ValueError('Previously proved value input/context/identity changed')
        owner=previous['owner']
        if old.complete[owner]or not segment.complete[owner]:
            if any(not _same(previous[k],row[k])for k in('z','eligible')):
                raise ValueError('Only new actual natural terminal may update finish targets')
    for owner,start in enumerate(old.steps):
        if segment.steps[owner]<start:raise ValueError('Inductive physical counters regressed')
        if old.complete[owner]and(segment.steps[owner]!=start or not segment.complete[owner]):
            raise ValueError('Completed physical game cannot advance/reopen')
        if segment.steps[owner]==start and(not _same(segment.final[owner],old.final[owner])
                or not _same(segment.pcg_states[owner],old.pcg_states[owner])
                or segment.complete[owner]!=old.complete[owner]):
            raise ValueError('Unadvanced parent final/PCG/outcome changed')
    return old


def write_auxiliary_proof_claim(directory,path,*,checkpoint,revision,phase,offset,proof_deadline,parent_proof=None):
    """Run real full/inductive proof and write a NEW claim for an external CPU supervisor.

    The claim alone is deliberately not loadable as a capability. The supervisor
    must observe normal worker exit and pin a witness containing its receipts.
    """
    from .device_auxiliary_collection import verify_auxiliary_segment
    from .device_auxiliary_continuation import auxiliary_content_sha256
    if type(proof_deadline)not in(int,float)or not time.monotonic()<proof_deadline:
        raise ValueError('Live explicit bounded monotonic proof deadline required')
    implementation=implementation_binding()
    root=Path(directory).resolve();target=(root/path).resolve()
    if not target.is_relative_to(root):raise ValueError('Contained new proof claim required')
    segment,manifest,entry=_committed(root,checkpoint,revision,phase=phase,offset=offset)
    if parent_proof is not None:
        previous=_context(parent_proof)
        if ((previous['claim']['phase'],previous['claim']['offset'],previous['claim']['count'])!=(phase,offset,len(segment.steps))
                or previous['claim']['contract_sha256']!=manifest['contract_sha256']):
            raise ValueError('Incremental proof cannot change plan/model/source contract')
        if len(entry['checkpoints'])<2 or entry['checkpoints'][-2]['sha256']!=previous['claim']['checkpoint']['sha256']:
            raise ValueError('Inductive parent must belong to the immutable committed checkpoint chain')
    verify_auxiliary_segment(segment,proof_deadline=proof_deadline,parent_proof=parent_proof)
    if time.monotonic()>=proof_deadline:raise TimeoutError('Full/inductive proof claim deadline exhausted')
    if implementation!=implementation_binding():raise ValueError('Verifier/source changed during proof')
    claim=dict(schema='auxiliary_prefix_proof_claim_v1',proof_mode='full'if parent_proof is None else'inductive',
        checkpoint=deepcopy(checkpoint),revision=deepcopy(revision),phase=phase,offset=offset,count=len(segment.steps),
        contract_sha256=manifest['contract_sha256'],content_sha256=auxiliary_content_sha256(segment),
        implementation=implementation,steps=list(segment.steps),complete=list(segment.complete),
        metadata_sha256=auxiliary_content_sha256(segment.metadata),source_binding=deepcopy(segment.source_binding),
        parent_witness=None if parent_proof is None else deepcopy(_context(parent_proof)['witness_descriptor']))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8')as stream:json.dump(claim,stream,sort_keys=True,indent=2,allow_nan=False)
    return dict(path=target.relative_to(root).as_posix(),sha256=sha256(target.read_bytes()).hexdigest())


def load_auxiliary_parent_proof(anchor):
    """Load an explicitly pinned supervisor witness; reject legacy/self-asserted flags.

    ``anchor`` is a trust-root input from the actual supervisor, never obtained
    by taking expected hashes from the same untrusted witness being checked.
    Every ancestor must carry the same actual worker/verifier identity and a
    normal CPU process receipt; only the latest numeric checkpoint is decoded.
    """
    if type(anchor)is not AuxiliaryProofAnchor or not _sha(anchor.witness_sha256)or not _sha(anchor.worker_sha256):
        raise TypeError('Explicit externally pinned witness and approved worker SHA required')
    root=Path(anchor.directory).resolve();implementation=implementation_binding();seen=set();files={}
    def checked(descriptor):
        path=_file(root,descriptor);files[path]=descriptor['sha256'];return path
    def witness(descriptor,depth=0):
        if depth>512 or descriptor['sha256']in seen:raise ValueError('Cyclic/excessive proof witness ancestry')
        seen.add(descriptor['sha256']);actual=checked(descriptor);data=_read(actual)
        required={'schema','claim','cpu_report','process_receipt','worker'}
        if set(data)!=required or data['schema']!='auxiliary_prefix_supervisor_witness_v1':raise ValueError('Actual supervisor witness schema required')
        worker=checked(data['worker'])
        if data['worker']['sha256']!=anchor.worker_sha256:raise ValueError('Witness worker is not the externally approved implementation')
        claim_path=checked(data['claim']);claim=_read(claim_path)
        claim_fields={'schema','proof_mode','checkpoint','revision','phase','offset','count','contract_sha256',
            'content_sha256','implementation','steps','complete','metadata_sha256','source_binding','parent_witness'}
        if (type(claim.get('count'))is not int or not 1<=claim['count']<=8
                or not isinstance(claim.get('steps'),list)or len(claim['steps'])!=claim['count']
                or any(type(n)is not int or not 0<=n<=512 for n in claim['steps'])
                or not isinstance(claim.get('complete'),list)or len(claim['complete'])!=claim['count']
                or any(type(done)is not bool for done in claim['complete'])):
            raise ValueError('Exact physical proof counters/outcome masks required')
        if set(claim)!=claim_fields or claim.get('schema')!='auxiliary_prefix_proof_claim_v1'or claim.get('implementation')!=implementation:
            raise ValueError('Prefix claim verifier/source binding differs')
        report_path=checked(data['cpu_report']);report=_read(report_path)
        process=_read(checked(data['process_receipt']))
        if (report.get('schema')!='auxiliary_prefix_cpu_proof_v1'or report.get('passed')is not True
                or report.get('cuda_initialized')is not False or report.get('cleanup_complete')is not True
                or report.get('producer_inputs_unchanged')is not True
                or report.get('claim_sha256')!=data['claim']['sha256']
                or report.get('checkpoint_sha256')!=claim['checkpoint']['sha256']
                or report.get('committed_revision_sha256')!=claim['revision']['sha256']
                or report.get('worker_sha256')!=anchor.worker_sha256
                or report.get('contract_sha256')!=claim['contract_sha256']
                or report.get('source_binding')!=claim['source_binding']
                or report.get('proof_mode')!=claim['proof_mode']
                or report.get('learning_updates')!=0 or report.get('qualified_games')!=0
                or process.get('schema')!='auxiliary_prefix_cpu_process_v1'
                or type(process.get('returncode'))is not int or process['returncode']!=0 or process.get('reason')is not None
                or process.get('cpu_only')is not True or process.get('outer_seconds')!=180
                or type(process.get('child_pid'))is not int or process['child_pid']<1
                or type(process.get('seconds'))not in(int,float)or not 0<=process['seconds']<=180
                or process.get('worker_sha256')!=anchor.worker_sha256
                or process.get('cpu_report_sha256')!=data['cpu_report']['sha256']):
            raise ValueError('Actual normal CPU exit/report/worker/committed proof chain required')
        checked(claim['checkpoint']);revision=_read(checked(claim['revision']))
        if (revision.get('contract_sha256')!=claim.get('contract_sha256')
                or report.get('source_model_versions')!=revision.get('contract',{}).get('source_model_versions')):
            raise ValueError('Proof claim differs from immutable committed plan/model contract')
        parent=claim.get('parent_witness')
        if claim.get('proof_mode')=='full':
            if parent is not None:raise ValueError('Full bootstrap cannot claim an unverified parent')
        elif claim.get('proof_mode')=='inductive':
            if not isinstance(parent,dict):raise ValueError('Inductive claim requires a real normal-exit bootstrap chain')
            old=witness(parent,depth+1)
            if (old['contract_sha256']!=claim['contract_sha256']or old['implementation']!=claim['implementation']
                    or(old['phase'],old['offset'],old['count'])!=(claim['phase'],claim['offset'],claim['count'])
                    or len(old['steps'])!=len(claim['steps'])or any(a>b for a,b in zip(old['steps'],claim['steps']))
                    or any(done and(claim['steps'][i]!=old['steps'][i]or not claim['complete'][i])for i,done in enumerate(old['complete']))):
                raise ValueError('Inductive witness chain changed contract/cohort/counters')
            entries=[e for e in revision['cohorts']if e['phase']==claim['phase']and e['offset']==claim['offset']]
            if (len(entries)!=1 or len(entries[0]['checkpoints'])<2
                    or entries[0]['checkpoints'][-2]['sha256']!=old['checkpoint']['sha256']):
                raise ValueError('Parent witness is not in this immutable committed checkpoint chain')
        else:raise ValueError('Unknown proof mode cannot grant reuse')
        return claim
    descriptor=dict(path=anchor.witness,sha256=anchor.witness_sha256)
    claim=witness(descriptor)
    parent,manifest,_=_committed(root,claim['checkpoint'],claim['revision'],phase=claim['phase'],offset=claim['offset'])
    from .device_auxiliary_continuation import auxiliary_content_sha256
    if (claim['content_sha256']!=auxiliary_content_sha256(parent)or claim['metadata_sha256']!=auxiliary_content_sha256(parent.metadata)
            or claim['steps']!=list(parent.steps)or claim['complete']!=list(parent.complete)
            or claim['count']!=len(parent.steps)or claim['source_binding']!=parent.source_binding):
        raise ValueError('Actual parent bytes/content/metadata differ from certified proof')
    proof=AuxiliaryParentProof(_SEAL)
    _CAPABILITIES[proof]=dict(parent=parent,claim=claim,implementation=implementation,files=files,
        witness_sha256=anchor.witness_sha256,witness_descriptor=descriptor,
        claim_sha256=sha256(checked(_read(checked(descriptor))['claim']).read_bytes()).hexdigest())
    return proof
