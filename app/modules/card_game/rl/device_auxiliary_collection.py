"""Real zero-search CUDA auxiliary collection, not a closed training driver.

One collector owns one bounded segment and the three supplied fresh arenas;
its executor is borrowed. Resume records are actual previous batch outputs.
CPU rules are used solely to verify archived batches outside graph execution.
Only an entirely natural, verified trajectory exports auxiliary value rows.
"""
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import time

import numpy as np

from .seven_auxiliary_jobs import AUXILIARY_DECISION_LIMIT, SoAAuxiliaryJobs, adapt_auxiliary_jobs
from .batched_search.soa_training_results import _same_graph, _equal_array


@dataclass(frozen=True)
class AuxiliarySegment:
    metadata: SoAAuxiliaryJobs
    initial: tuple
    final: tuple
    steps: tuple
    complete: tuple
    truncated: tuple
    errors: tuple
    pcg_states: tuple
    raw_records: tuple
    audit_records: tuple
    value_records: tuple
    attempts: tuple
    reason: str
    decision_limit: int = 800
    source_binding: dict = field(default_factory=dict)
    export_errors: tuple = ()
    replay_verified: bool = False
    formal_training_qualified: bool = False
    gpu_qualified: bool = False
    budget_continuations: tuple = ()


def _metadata(metadata):
    if not isinstance(metadata, SoAAuxiliaryJobs):
        raise TypeError('Actual frozen auxiliary job metadata required')
    rebound = adapt_auxiliary_jobs(metadata.jobs, starting_decks=metadata.starting_decks)
    for name in ('jobs', 'initializers', 'keys', 'identities', 'execution', 'starting_decks', 'starting_lineups'):
        if getattr(metadata, name) != getattr(rebound, name):
            raise ValueError('Auxiliary metadata or frozen lineup binding changed')
    if metadata.gpu_qualified is not False or metadata.formal_training_qualified is not False:
        raise ValueError('Metadata cannot grant auxiliary driver qualification')
    return rebound


def _deadline(wall, metadata, started, soft_seconds):
    if (type(wall) not in (int, float) or not 0 < wall < 2**63 or not math.isfinite(wall)
            or type(soft_seconds) not in (int, float) or not 0 < soft_seconds <= 150):
        raise ValueError('Explicit finite absolute deadline and at most150s soft budget required')
    wall = min(wall, *(job['deadline'] for job in metadata.jobs))
    remaining = min(wall-time.time(), soft_seconds-(time.monotonic()-started))
    if remaining <= 0:
        raise TimeoutError('Auxiliary absolute/soft deadline exhausted')
    return remaining


def _oracle():
    from ..engine.duel_v2 import new_game, observe, acting_side, apply_action
    from .league_rollout import clean
    from .seven_lineup import encode
    from .covered_rollout import rule_decision
    return new_game, observe, acting_side, apply_action, clean, encode, rule_decision


def _coordinates(records, rows):
    out = {}
    for record in records:
        if not isinstance(record, dict):raise ValueError('Actual numerical record dictionaries required')
        owner, step = record.get('owner'), record.get('step')
        if type(owner) is not int or type(step) is not int or not 0 <= owner < rows or step < 0:
            raise ValueError('Exact auxiliary owner/step coordinates required')
        if (owner, step) in out:raise ValueError('Duplicate auxiliary trajectory coordinate')
        out[owner, step] = record
    return out


def verify_auxiliary_segment(segment, *, proof_deadline=None):
    """Verify supplied GPU evidence; never fill missing x/context/z or actions."""
    if not isinstance(segment, AuxiliarySegment):raise TypeError('Actual auxiliary segment required')
    if segment.export_errors:raise ValueError('Incomplete/corrupt GPU archive export remains audit-only')
    metadata = _metadata(segment.metadata);rows = len(metadata.jobs)
    from .device_auxiliary_continuation import verify_auxiliary_budget_history
    verify_auxiliary_budget_history(segment)
    if set(segment.source_binding)!={'rule_source','image','schema'} or any(
            not isinstance(v,str)or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)
            for v in segment.source_binding.values()):
        raise ValueError('Actual frozen rule/image/schema source binding required')
    if segment.formal_training_qualified is not False or segment.gpu_qualified is not False:
        raise ValueError('Auxiliary component cannot grant qualification')
    for name in ('initial','final','steps','complete','truncated','errors','pcg_states'):
        if len(getattr(segment,name)) != rows:raise ValueError('Complete cohort evidence required')
    if type(segment.decision_limit)is not int or not 1<=segment.decision_limit<=800:
        raise ValueError('Explicit bounded physical decision ceiling required')
    if any(type(v) is not int or not 0 <= v <= segment.decision_limit for v in segment.steps):
        raise ValueError('Actual bounded global step counters required')
    for name in ('complete','truncated'):
        if any(type(v) is not bool for v in getattr(segment,name)):raise ValueError('Exact device outcome masks required')
    if any(type(v) is not int or v not in (0,256) for v in segment.errors):
        raise ValueError('Rule/archive errors prohibit verification and recovery')
    raw = _coordinates(segment.raw_records, rows);audit = _coordinates(segment.audit_records, rows)
    values = _coordinates(segment.value_records, rows)
    expected = {(r,s) for r,n in enumerate(segment.steps) for s in range(n)}
    if set(raw) != expected or set(audit) != expected or set(values) != expected:
        raise ValueError('Full continuous raw/audit/both-viewer numerical prefix required')
    for attempt in segment.attempts:
        if (any(type(v)is not int or v not in (0,256)for v in attempt['errors'])
                or any(attempt['raw_errors']) or any(attempt['value_errors']) or any(attempt.get('audit_append_errors',())) or attempt.get('memory_limit_exceeded',False)):
            raise ValueError('Prior semantic/archive/memory failure cannot be washed by resume')
    new_game, observe, acting_side, apply_action, clean, encode, rule_decision = _oracle()
    from .batched_search.device_random import verify_numpy_pcg64_state
    for owner, job in enumerate(metadata.jobs):
        state = new_game(**deepcopy(metadata.initializers[owner]))
        _same_graph(segment.initial[owner], state)  # Normal init retains events/cache.
        rng = np.random.default_rng(job['seed'])
        for step in range(segment.steps[owner]):
            if proof_deadline is not None and time.monotonic() >= proof_deadline:
                raise TimeoutError('Auxiliary postbatch proof deadline exhausted')
            key=owner,step;r,a,v=raw[key],audit[key],values[key]
            if state['phase']=='finished':raise ValueError('Action after natural terminal state')
            before=deepcopy(state);actor=acting_side(state);legal,rule_choice=rule_decision(state,actor)
            _same_graph(state,before)
            explore=step<6 and state['phase']=='playing'
            choice=int(rng.integers(len(legal))) if explore else rule_choice
            for record in (r,a,v):
                for field in ('owner','step','seed','version','physical_actor'):
                    if type(record.get(field)) is not int:raise ValueError('Exact GPU integer identities required')
                if (record['seed']!=job['seed'] or record['version']!=state['version']
                        or record['physical_actor']!=int(actor=='b')):
                    raise ValueError('GPU raw/value job, version or actor mismatch')
            if type(r.get('game'))is not int or r['game']!=owner:raise ValueError('Raw physical game identity differs')
            for field in ('requested','completed','visit_sum'):
                if type(a.get(field))is not int or a[field]!=0:raise ValueError('Auxiliary policy must perform zero simulations')
            for field, actual in (('count',len(legal)),('rule_chosen',rule_choice),('chosen',choice),('explored',int(explore))):
                if type(a.get(field)) is not int or a[field]!=actual:raise ValueError('GPU rule/exploration choice mismatch')
            _same_graph(a['actions'],legal);_same_graph(r['raw_action'],legal[choice])
            verify_numpy_pcg64_state(a['pcg_state'])
            if a['pcg_state']!=rng.bit_generator.state:raise ValueError('All six exploration RNG fields must match')
            if (type(v.get('slot')) is not int or v['slot']<0 or type(v.get('generation')) is not int or v['generation']<1
                    or type(v.get('errors')) is not int or v['errors']!=0):raise ValueError('Actual clean GPU value lease required')
            if not isinstance(v.get('x'),np.ndarray) or v['x'].dtype!=np.float32 or v['x'].ndim!=2 or v['x'].shape[0]!=2:
                raise ValueError('Both actual GPU float32 viewer inputs required')
            if (not isinstance(v.get('context'),np.ndarray) or v['context'].dtype!=np.float32 or v['context'].shape!=(2,)
                    or not np.isfinite(v['x']).all() or not np.isfinite(v['context']).all()):
                raise ValueError('Both finite GPU value contexts required')
            for viewer,side in enumerate(('a','b')):
                x,_=encode(observe(state,side,include_previews=False),[])
                _equal_array(v['x'][viewer],x)
                if v['context'][viewer]!=int(actor==side):raise ValueError('GPU value actor context differs')
            _same_graph(state,before)
            state=clean(apply_action(state,actor,deepcopy(r['raw_action'])))
        if segment.final[owner] is None:raise ValueError('Missing registry-verified final state')
        _same_graph(segment.final[owner],state)
        if segment.pcg_states[owner]!=rng.bit_generator.state:raise ValueError('Final six-field PCG state differs')
        natural=state['phase']=='finished'
        if segment.complete[owner] != natural or (natural and (segment.errors[owner] or segment.truncated[owner])):
            raise ValueError('Actual natural terminal outcome differs from device result')
        for step in range(segment.steps[owner]):
            value=values[owner,step];z=value.get('z');eligible=value.get('eligible')
            if (not isinstance(z,np.ndarray)or z.dtype!=np.float32 or z.shape!=(2,)
                    or not isinstance(eligible,np.ndarray)or eligible.dtype!=np.bool_ or eligible.shape!=(2,)):
                raise ValueError('Actual dual-viewer GPU finish arrays required')
            if not natural and (z.any()or eligible.any()):raise ValueError('Pending/truncated trajectories cannot carry learning labels')
        if natural:
            winner=state.get('winner')
            if winner not in ('a','b','draw') or state.get('reason') in (None,'','timeout','deadline','truncated','action_limit'):
                raise ValueError('Natural winner/reason required')
            for step in range(segment.steps[owner]):
                value=values[owner,step]
                z=value.get('z');eligible=value.get('eligible')
                if (not isinstance(z,np.ndarray) or z.dtype!=np.float32 or z.shape!=(2,)
                        or not isinstance(eligible,np.ndarray) or eligible.dtype!=np.bool_ or eligible.shape!=(2,) or not eligible.all()):
                    raise ValueError('Real GPU finish must label the entire historical dual-viewer prefix')
                _equal_array(z,np.asarray([0 if winner=='draw' else 1 if winner==s else -1 for s in ('a','b')],np.float32))
    return True


def verified_auxiliary_games(segment, *, proof_deadline=None):
    """Return only covered-value auxiliary dictionaries, never policy learning rows."""
    if not isinstance(segment,AuxiliarySegment):raise TypeError('Actual auxiliary segment required')
    metadata=_metadata(segment.metadata)
    if (type(segment.decision_limit)is not int or segment.decision_limit!=AUXILIARY_DECISION_LIMIT
            or metadata.execution.get('max_decisions')!=AUXILIARY_DECISION_LIMIT):
        raise ValueError('Only the exact512-decision auxiliary collection contract may export')
    if segment.replay_verified is not True:raise ValueError('Successful in-budget postbatch verification required')
    if not all(segment.complete) or any(segment.errors) or any(segment.truncated):
        raise ValueError('Entire cohort must end naturally without errors/truncation')
    if any(any(attempt['errors'])for attempt in segment.attempts):
        raise ValueError('A prior guard failure remains audit-only, even after recovery')
    verify_auxiliary_segment(segment,proof_deadline=proof_deadline)
    values=_coordinates(segment.value_records,len(segment.steps));raw=_coordinates(segment.raw_records,len(segment.steps))
    games=[]
    for owner,job in enumerate(segment.metadata.jobs):
        rows=[];actions=[]
        for step in range(segment.steps[owner]):
            record=raw[owner,step];actions.append(dict(side=('a','b')[record['physical_actor']],action=deepcopy(record['raw_action'])))
            if step<6:continue
            value=values[owner,step]
            for viewer,side in enumerate(('a','b')):
                rows.append(dict(x=value['x'][viewer].copy(),value_actor=float(value['context'][viewer]),
                    z=float(value['z'][viewer]),side=side,key=job['keys'][side],step=step))
        if not rows:raise ValueError('No post-exploration auxiliary value observations')
        games.append(dict(complete=True,rows=[],value_rows=rows,actions=actions,winner=segment.final[owner]['winner'],
            seed=job['seed'],first=job['first'],job=deepcopy(job),rule_backend='cuda_soa',
            auxiliary_value_only=True,formal_training_qualified=False,decision_limit=segment.decision_limit,
            source_binding=deepcopy(segment.source_binding),execution=deepcopy(metadata.execution)))
    return games


def _resume_counters(resume, ingress):
    """Derive archive counters from actual prefix records, never guessed offsets."""
    if resume is None:
        if any(type(s.get('version'))is not int or s['version']!=0 for s in ingress):
            raise ValueError('Fresh normal initialization must retain version0')
        return [0]*len(ingress),[-1]*len(ingress)
    rows=len(ingress);raw=_coordinates(resume.raw_records,rows)
    offsets=[];previous=[]
    for owner,state in enumerate(ingress):
        count=resume.steps[owner]
        if type(count)is not int or count<0:raise ValueError('Actual integer resume step required')
        prior=-1 if count==0 else raw[owner,count-1]['version']
        if type(prior)is not int or type(state.get('version'))is not int or state['version']!=prior+1:
            raise ValueError('Saved physical version differs from last archived version')
        offsets.append(count);previous.append(prior)
    return offsets,previous


def _planned_pause(loop, offsets, decisions):
    """Device boundary only: pause without terminal, truncation, errors or labels."""
    remaining=(loop._steps-offsets)<decisions
    loop._unfinished.bitwise_and_(remaining.to(loop._unfinished.dtype))


def _restore_counters(loop, raw, tape, offsets, previous_versions):
    """Called only by the static capture initializer; all inputs stay on device."""
    loop._steps.copy_(offsets)
    raw._expected_step.copy_(offsets);raw._last_version.copy_(previous_versions)
    tape._expected_step.copy_(offsets)
    tape._initial_version.sub_(offsets)
    loop._update_predicate()


def _history_tensors(t, device, resume, rows, observation_dim):
    """Validate CPU archived evidence before constructing exact upload tensors."""
    if resume is None:return {}
    records=sorted(resume.value_records,key=lambda r:(r['owner'],r['step']))
    fields={'_rec_game':'owner','_rec_owner':'owner','_rec_seed':'seed','_rec_actor':'physical_actor',
        '_rec_step':'step','_rec_version':'version','_rec_slot':'slot','_rec_generation':'generation','_rec_errors':'errors'}
    out={name:t.tensor([r[field]for r in records],dtype=t.int32 if name=='_rec_errors' else t.int64,device=device)
         for name,field in fields.items()}
    if records:
        x=np.stack([r['x']for r in records]);ctx=np.stack([r['context']for r in records])
        if x.dtype!=np.float32 or x.shape!=(len(records),2,observation_dim) or ctx.dtype!=np.float32 or ctx.shape!=(len(records),2):
            raise ValueError('Exact historical GPU dual-viewer numerical shapes required')
        out['_x']=t.tensor(x,device=device);out['_ctx']=t.tensor(ctx,device=device)
    else:
        out['_x']=t.empty((0,2,observation_dim),dtype=t.float32,device=device);out['_ctx']=t.empty((0,2),dtype=t.float32,device=device)
    if records:
        for field,name,dtype,kind in (('z','_z',np.float32,t.float32),('eligible','_eligible',np.bool_,t.bool)):
            array=np.stack([r[field]for r in records])
            if array.dtype!=dtype or array.shape!=(len(records),2):raise ValueError('Actual historical finish arrays required')
            out[name]=t.tensor(array,dtype=kind,device=device)
    out['_complete']=t.tensor(resume.complete,dtype=t.bool,device=device)
    out['_finished']=out['_complete'].clone()
    out['_game_winners']=t.tensor([(-1 if s.get('winner')=='draw' else ('a','b').index(s['winner']))if done else -2
        for s,done in zip(resume.final,resume.complete)],dtype=t.int64,device=device)
    return out


def _restore_history(tape, tensors):
    if not tensors:return
    count=tensors['_x'].shape[0]
    if count>tape.max_records:raise ValueError('Historical GPU tape exceeds record budget')
    for name,value in tensors.items():
        destination=getattr(tape,name)
        if name in ('_complete','_finished','_game_winners'):
            destination.copy_(value)
        else:
            destination[:count].copy_(value)
    tape._count.fill_(count)
    # Existing GPU terminal labels are copied exactly. Pending prefix labels stay
    # zero/ineligible until a subsequent actual GPU finish labels them.


class _AuxiliarySink:
    """Actual pure-value tape plus immutable full-legal/PCG query audit."""
    def __init__(self,tape,controller,audit_store):
        from .batched_duel.device_rule_calls import _empty_graph
        self.tape,self.controller,self.store=tape,controller,audit_store
        self.t,self.rows,self.device=tape.t,tape.rows,tape.device
        t=self.t;n=tape.max_records+1;z=dict(device=self.device)
        self.graph=_empty_graph(t,controller.rules.action_layout,self.device)
        self.count=t.zeros((),dtype=t.int64,**z);self.sentinel=t.tensor(tape.max_records,dtype=t.int64,**z)
        self.info=t.zeros((n,9),dtype=t.int64,**z);self.pcg=t.zeros((n,6),dtype=t.int64,**z)
        self.slots=t.full((n,),-1,dtype=t.int64,**z);self.gens=t.zeros(n,dtype=t.int64,**z)
        self.history={}
        self.append_errors=t.zeros(self.rows,dtype=t.int32,**z)
    def begin_device(self,*args):
        errors=self.tape.begin_device(*args);self.count.zero_();self.append_errors.zero_();_restore_history(self.tape,self.history);return errors
    def append_device(self,roots,actors,steps,result,active):
        from .batched_duel.device_rule_protocol import DeviceStateBatch
        t=self.t;errors=self.tape.append_device(roots,actors,steps,result,active)
        take=active&(errors==0)
        leases=DeviceStateBatch(result.decision.action_pool,result.decision.action_slots,result.decision.action_generations,
                               roots.owners,take,errors.clone())
        copied=result.decision.action_pool.gather_into_device(leases,self.graph,shared_reads=True)
        errors=errors|copied.errors;take&=errors==0
        saved=self.store.put_device(self.graph,roots.owners,take)
        errors=errors|saved.errors;take&=errors==0
        rank=take.long().cumsum(0)-take.long();dest=t.where(take,self.count+rank,self.sentinel)
        self.info.index_copy_(0,dest,t.stack((roots.owners,steps,result.decision.counts,result.actions,
            self.controller.rule_choices_device,self.controller.explored_device.long(),result.requested,result.completed,
            result.visits.long().sum(-1)),dim=1))
        state=self.controller.exploration_state_device
        self.pcg.index_copy_(0,dest,t.stack((state.state_hi.view(t.int64),state.state_lo.view(t.int64),
            state.inc_hi.view(t.int64),state.inc_lo.view(t.int64),state.has_uint32.long(),state.uinteger.long()),dim=1))
        self.slots.index_copy_(0,dest,saved.slots);self.gens.index_copy_(0,dest,saved.generations)
        self.append_errors.bitwise_or_(t.where(active,errors,0))
        self.tape._game_errors.bitwise_or_(self.append_errors)
        self.count.add_(take.long().sum());return errors
    def finish_device(self,*args):return self.tape.finish_device(*args)


class DeviceAuxiliaryCollector:
    """Single-segment collector owning supplied fresh arenas, borrowing executor.

    The caller provides the absolute stop time (including13:00 when required),
    measured arenas and an outer process watchdog. Device guards bound loops;
    this synchronous component cannot forcibly preempt a stuck CUDA driver.
    max_decisions is an explicit ceiling (the legacy auxiliary path uses512);
    reaching it produces truncation, never a natural game or labels. Clean
    planned pauses may resume; any guard-failed history remains audit-only.
    No methods or receipts satisfy the formal closed-Gumbel driver contract.
    """
    formal_training_qualified=False
    gpu_qualified=False
    supports_full_duel_rules=False
    def __init__(self,executor,state_store,action_store,audit_store,*,rows,max_decisions,max_actions=801,
                 segment_decisions=32,max_device_bytes=14<<30,value_max_bytes=256<<20,
                 raw_capacities=(65536,131072,1048576)):
        from .batched_duel.native_soa_cuda import CudaSoARules
        from .batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
        from .batched_duel.device_graph_storage import CompactDeviceGraphStore
        if (not isinstance(executor,CudaSoARules) or executor.execute_only
                or not isinstance(state_store,DeviceSparseSoAStore)
                or any(not isinstance(s,CompactDeviceGraphStore)for s in (action_store,audit_store))):
            raise TypeError('Real full native CUDA executor and checked arenas required')
        self._closed=False;self._objects=[]
        self._arenas=(audit_store,action_store,state_store)
        try:
            for name,v,upper in (('rows',rows,64),('max_actions',max_actions,16384),('max_decisions',max_decisions,800),
                                 ('segment_decisions',segment_decisions,512),('max_device_bytes',max_device_bytes,14<<30),
                                 ('value_max_bytes',value_max_bytes,14<<30)):
                if type(v)is not int or not 1<=v<=upper:raise ValueError('Bounded '+name+' required')
            if (type(raw_capacities)is not tuple or len(raw_capacities)!=3
                    or any(type(v)is not int or not 1<=v<2**31 for v in raw_capacities)):
                raise ValueError('Three bounded raw byte/element capacities required')
            self.executor,self.state_store,self.action_store,self.audit_store=executor,state_store,action_store,audit_store
            self.device,self.t=executor.device,executor.torch
            if self.device.type!='cuda' or any(s.device!=self.device or s.closed for s in (action_store,audit_store)) or state_store.device!=self.device or state_store.closed:
                raise ValueError('Live actual CUDA arenas must share device')
            if state_store.schema.identity!=executor.schema.identity or audit_store.slot_capacity<rows*max_decisions:
                raise ValueError('Exact schema and full history audit slots required')
            if any(bool(s.live.any().item())for s in (state_store,action_store,audit_store)):
                raise ValueError('Fresh arenas required; segment takes their ownership')
            if len({id(s)for s in (state_store,action_store,audit_store)})!=3:raise ValueError('Independent arenas required')
            self.rows,self.max_actions,self.max_decisions=rows,max_actions,max_decisions
            self.segment_decisions,self.max_device_bytes=segment_decisions,max_device_bytes
            self.value_max_bytes,self.raw_capacities=value_max_bytes,raw_capacities
            self._source_binding=dict(rule_source=executor.rule_source_sha256,image=executor.image.identity,schema=executor.schema.identity)
            self._used=False
        except BaseException:
            try:self.close()
            except BaseException:pass
            raise

    def _memory(self, *, enforce=True):
        free,total=self.t.cuda.mem_get_info(self.device)
        used=total-free
        if enforce and used>self.max_device_bytes:raise MemoryError('Auxiliary whole-device memory ceiling exceeded')
        return used
    def close(self):
        if self._closed:return
        self._closed=True;errors=[]
        seen=set()
        for obj in (*reversed(self._objects),*self._arenas):
            if id(obj)in seen:continue
            seen.add(id(obj))
            try:obj.close()
            except BaseException as error:errors.append(repr(error))
        # Captured instance closures and controller/rules backreferences form
        # cycles otherwise retaining the entire workspace after a segment.
        for obj in self._objects:
            owned=vars(obj)
            if 'physical_loop'in owned:owned['physical_loop']=None
            if '_loop'in owned:owned['_loop']=None
            if '_teacher'in owned:owned['_teacher']=None
            if '_teacher_pipeline'in owned:owned['_teacher_pipeline']=None
            if 'initialize'in owned:owned.pop('initialize')
            if '_update_predicate'in owned:owned.pop('_update_predicate')
        self._objects=[];self._arenas=()
        self.state_store=self.action_store=self.audit_store=None
        self.cleanup_errors=tuple(errors)
        if errors:raise RuntimeError('Auxiliary cleanup failed: '+repr(errors))
    def __enter__(self):
        if self._closed:raise RuntimeError('Auxiliary collector closed')
        return self
    def __exit__(self,exc_type,exc,tb):
        try:self.close()
        except BaseException:
            if exc is None:raise
        return False
    def run_segment(self,metadata,*,deadline,resume=None,seconds=90,soft_seconds=150,continuation_budget=None):
        if self._closed:raise RuntimeError('Auxiliary collector closed')
        try:
            result=self._run_segment(metadata,deadline=deadline,resume=resume,seconds=seconds,soft_seconds=soft_seconds,continuation_budget=continuation_budget)
        except BaseException:
            try:self.close()
            except BaseException:pass
            raise
        self.close()
        return result
    def _run_segment(self,metadata,*,deadline,resume=None,seconds=90,soft_seconds=150,continuation_budget=None):
        metadata=_metadata(metadata);started=time.monotonic()
        if continuation_budget is not None:
            from .device_auxiliary_continuation import validate_auxiliary_continuation
            if resume is None:raise ValueError('Continuation budget requires actual previous segment')
            metadata=validate_auxiliary_continuation(continuation_budget,resume=resume,metadata=metadata,
                deadline=deadline,source_binding=self._source_binding,seconds=seconds,soft_seconds=soft_seconds)
        if self._used:raise RuntimeError('One actual graph replay per collector; use fresh arenas for next segment')
        if len(metadata.jobs)!=self.rows:raise ValueError('Actual job cohort must match arenas')
        if type(seconds)not in (int,float) or not 0<seconds<=90:raise ValueError('At most90s device loop required')
        _deadline(deadline,metadata,started,soft_seconds);self._memory()
        if resume is not None:
            if not isinstance(resume,AuxiliarySegment) or ((continuation_budget is None and resume.metadata!=metadata) or resume.decision_limit!=self.max_decisions or resume.source_binding!=self._source_binding):raise ValueError('Same actual job cohort required for resume')
            if not resume.replay_verified or any(code not in (0,256)for code in resume.errors):raise ValueError('Only verified clean/guard segment may resume')
            verify_auxiliary_segment(resume,proof_deadline=time.monotonic()+_deadline(deadline,metadata,started,soft_seconds))
            if all(resume.complete) or any(n>=self.max_decisions and not done for n,done in zip(resume.steps,resume.complete)):
                raise ValueError('Finished or global-decision-limit cohort cannot resume')
        self._used=True;objects=self._objects
        from .batched_search.device_raw_action_tape import DeviceRawActionTape
        from .batched_search.device_pure_value_tape import DevicePureValueTape
        from .batched_search.device_rule_policy import DeviceRulePolicy
        from .batched_search.device_soa_rules import DeviceSoARules
        from .batched_search.device_games import DeviceGameLoop
        from .batched_search.device_seeds import DeviceRootSeeds
        from .batched_search.device_physical_action_capture import RawActionPolicySink
        from .batched_duel.record_soa import pack_game_states,unpack_game_states
        from .seven_lineup import all_features
        t,z=self.t,dict(device=self.device);n=self.rows;capacity=n*self.max_decisions
        raw=DeviceRawActionTape(n,capacity,*self.raw_capacities,device=self.device);objects.append(raw);self._memory()
        rules=DeviceSoARules(self.executor,self.state_store,self.action_store,rows=n,max_actions=self.max_actions,require_teacher=True,raw_tape=raw)
        objects.append(rules);self._memory()
        controller=DeviceRulePolicy(rules,rows=n,max_actions=self.max_actions,device=self.device,
            explore_first_six=True,exploration_seeds=tuple(j['seed']for j in metadata.jobs));objects.append(controller)
        tape=DevicePureValueTape(rules,rows=n,max_records=capacity,max_bytes=self.value_max_bytes,
            observation_dim=len(all_features()),max_actions=self.max_actions,device=self.device)
        sink=_AuxiliarySink(tape,controller,self.audit_store)
        sink.history=_history_tensors(t,self.device,resume,n,tape.observation_dim)
        seeds=DeviceRootSeeds(device=self.device,max_rows=n);objects.append(seeds);self._memory()
        remaining=_deadline(deadline,metadata,started,soft_seconds)
        loop=DeviceGameLoop(rules,controller,RawActionPolicySink(sink,raw),seeds,max_decisions=self.max_decisions,
            outer_max_ticks=self.segment_decisions+1,inner_max_ticks=2,seconds=min(seconds,remaining),
            device=self.device,defer_physical_presentation=True);objects.append(loop)
        rules.bind_game_loop(loop);controller.bind_game_loop(loop)
        offsets=t.tensor(resume.steps if resume else [0]*n,dtype=t.int64,**z)
        if resume:
            # Only the actual archived final is uploaded; oracle worlds are discarded.
            parents=self.state_store.put_device(pack_game_states(deepcopy(resume.final),self.executor.schema).to_torch(self.device),
                t.arange(n,dtype=t.int64,**z),t.ones(n,dtype=t.bool,**z))
            controller._exploration_rng.load_states(resume.pcg_states)
        else:parents=rules.initialize_device(list(deepcopy(metadata.initializers)))
        t.cuda.synchronize()
        if parents.errors.cpu().tolist()!=[0]*n:raise RuntimeError('Actual GPU initializer/restore ingress failed')
        ingress=unpack_game_states(self.state_store.gather_device(parents,shared_reads=True).to_host())
        initial=deepcopy(resume.initial)if resume else tuple(deepcopy(ingress))
        if resume:
            for actual,expected in zip(ingress,resume.final):_same_graph(actual,expected)
        active=t.tensor([not v for v in resume.complete]if resume else [True]*n,dtype=t.bool,**z)
        parents=replace(parents,active=active)
        restored_offsets,restored_versions=_resume_counters(resume,ingress)
        if tuple(restored_offsets)!=(resume.steps if resume else (0,)*n):raise ValueError('Physical offset binding changed')
        previous=t.tensor(restored_versions,dtype=t.int64,**z)
        original_predicate=loop._update_predicate
        def update_predicate():
            original_predicate();_planned_pause(loop,offsets,self.segment_decisions)
        loop._update_predicate=update_predicate
        original=loop.initialize
        def initialize(*args):
            original(*args);_restore_counters(loop,raw,tape,offsets,previous)
        loop.initialize=initialize
        ids=t.zeros((n,2),dtype=t.int64,**z);train=t.ones((n,2),dtype=t.bool,**z)
        remaining=_deadline(deadline,metadata,started,soft_seconds)
        device_seconds=min(seconds,remaining-20)  # Reserve capture and postbatch proof time.
        if device_seconds<=0:raise TimeoutError('Insufficient bounded capture/proof window')
        loop._outer.budget_ns=loop._inner.budget_ns=int(device_seconds*1e9)
        loop.capture(parents,t.tensor([j['seed']for j in metadata.jobs],dtype=t.int64,**z),ids,ids,train)
        if _deadline(deadline,metadata,started,soft_seconds)<device_seconds+10:
            raise TimeoutError('Capture consumed deadline margin; replay was not launched')
        self._memory()
        loop.replay();t.cuda.synchronize()  # No host rule/selector/metadata callback during replay.
        result=loop.result_device();used=self._memory(enforce=False)
        outcome=self._export(metadata,initial,ingress,resume,raw,tape,sink,controller,result,used,continuation_budget=continuation_budget)
        export_memory=self._memory(enforce=False)
        outcome.attempts[-1]['device_used_bytes']=max(used,export_memory)
        outcome.attempts[-1]['memory_limit_exceeded']=max(used,export_memory)>self.max_device_bytes
        try:
            verify_auxiliary_segment(outcome,proof_deadline=time.monotonic()+_deadline(deadline,metadata,started,soft_seconds))
            outcome=replace(outcome,replay_verified=True)
            if outcome.reason=='incomplete':outcome=replace(outcome,reason='planned_segment_pause')
        except (ValueError,KeyError,TypeError,TimeoutError,AssertionError)as error:
            outcome=replace(outcome,reason=outcome.reason+'; proof_failed:'+str(error))
        return outcome

    def _export(self,metadata,initial,ingress,resume,raw,tape,sink,controller,result,used,continuation_budget=None):
        """Postbatch only. Preserve uncommitted raw/value records and all errors."""
        t=self.t;n=self.rows;export_errors=[]
        from .batched_duel.device_rule_protocol import DeviceStateBatch
        from .batched_search.device_random import numpy_pcg64_state_from_limbs
        steps=tuple(result.steps.cpu().tolist());errors=tuple(result.errors.cpu().tolist())
        old_complete=resume.complete if resume else (False,)*n
        complete=tuple(a or b for a,b in zip(old_complete,result.complete.cpu().tolist()))
        truncated=tuple(result.truncated.cpu().tolist())
        count=int(tape._count.cpu());values=[]
        names=('_rec_owner','_rec_seed','_rec_actor','_rec_step','_rec_version','_rec_slot','_rec_generation','_rec_errors')
        arrays={name:getattr(tape,name)[:count].cpu().numpy().copy()for name in (*names,'_x','_ctx','_z','_eligible')}
        for i in range(count):
            values.append(dict(zip(('owner','seed','physical_actor','step','version','slot','generation','errors'),
                [int(arrays[k][i])for k in names]),x=arrays['_x'][i],context=arrays['_ctx'][i],
                z=arrays['_z'][i],eligible=arrays['_eligible'][i]))
        raw_count=int(raw._record_count.cpu())
        raw_ids=list(range(raw_count))
        try:actions=_decode_raw(raw,raw_ids)
        except (ValueError,RuntimeError,AssertionError)as error:
            actions=[None]*raw_count;export_errors.append('raw:'+repr(error))
        raw_fields=[getattr(raw,k)[:raw_count].cpu().tolist()for k in ('_rec_game','_rec_step','_rec_owner','_rec_seed','_rec_actor','_rec_version')]
        records=[] if resume is None else list(deepcopy(resume.raw_records))
        for i,action in enumerate(actions):
            record=dict(zip(('game','step','owner','seed','physical_actor','version'),[v[i]for v in raw_fields]));record['raw_action']=action;records.append(record)
        audit_count=int(sink.count.cpu());infos=sink.info[:audit_count].cpu().tolist();pcgs=sink.pcg[:audit_count].cpu().tolist()
        audit=[] if resume is None else list(deepcopy(resume.audit_records))
        for begin in range(0,audit_count,8):
            end=min(begin+8,audit_count);ids=t.arange(begin,end,dtype=t.int64,device=self.device)
            owners=t.tensor([v[0]for v in infos[begin:end]],dtype=t.int64,device=self.device)
            leases=DeviceStateBatch(self.audit_store,sink.slots.index_select(0,ids),sink.gens.index_select(0,ids),owners,
                t.ones(end-begin,dtype=t.bool,device=self.device),t.zeros(end-begin,dtype=t.int32,device=self.device))
            try:actions_batch=_decode_leases(self.audit_store,leases,end-begin)
            except (ValueError,RuntimeError,AssertionError)as error:
                actions_batch=[None]*(end-begin);export_errors.append('legal:'+repr(error))
            for i,legal in zip(range(begin,end),actions_batch):
                owner,step,count,chosen,rule,explored,requested,completed,visit_sum=infos[i]
                value=next(v for v in values if (v['owner'],v['step'])==(owner,step))
                audit.append(dict(owner=owner,step=step,count=count,chosen=chosen,rule_chosen=rule,explored=explored,
                    requested=requested,completed=completed,visit_sum=visit_sum,
                    seed=value['seed'],version=value['version'],physical_actor=value['physical_actor'],actions=legal,
                    pcg_state=numpy_pcg64_state_from_limbs(*pcgs[i])))
        try:final,registry=_final_states(self.state_store,result.states,old_complete)
        except (ValueError,RuntimeError,AssertionError)as error:
            final=[None]*n;registry=[];export_errors.append('final:'+repr(error))
        pcg=controller._exploration_rng.export_numpy_states()
        raw_errors=tuple(raw.errors_device.cpu().tolist());value_errors=tuple(tape._game_errors.cpu().tolist())
        combined=tuple(a|b|c for a,b,c in zip(errors,raw_errors,value_errors))
        attempt=dict(errors=errors,raw_errors=raw_errors,value_errors=value_errors,audit_append_errors=tuple(sink.append_errors.cpu().tolist()),steps=steps,truncated=truncated,
            complete=complete,registry=registry,device_used_bytes=used,memory_limit_exceeded=used>self.max_device_bytes,
            counter_restore=dict(offsets=list(resume.steps)if resume else [0]*n,
                previous_versions=_resume_counters(resume,ingress)[1],source='previous_verified_raw_prefix'if resume else 'normal_initialization'))
        attempts=(() if resume is None else deepcopy(resume.attempts))+(attempt,)
        reason='natural'if all(complete)and not any(combined)else('device_error'if any(v not in (0,256)for v in combined)
            else 'device_guard_stop'if any(combined)else 'global_decision_limit'if any(truncated)else 'incomplete')
        return AuxiliarySegment(metadata,tuple(initial),tuple(final),steps,complete,truncated,combined,tuple(pcg),
            tuple(records),tuple(audit),tuple(values),attempts,reason,decision_limit=self.max_decisions,source_binding=deepcopy(self._source_binding),export_errors=tuple(export_errors),
            budget_continuations=(() if resume is None else deepcopy(resume.budget_continuations))
                +(() if continuation_budget is None else (deepcopy(continuation_budget),)))


def _graph_values(graph):
    from .batched_duel.schema import PackedStateBatch
    from .batched_duel.codec import unpack_states
    arrays={k:np.ascontiguousarray(getattr(graph,k).cpu().numpy())for k in ('nodes','edges','payload','counts','roots')}
    return unpack_states(PackedStateBatch(graph.layout,**arrays))


def _decode_raw(raw,indices):
    from .batched_duel.device_rule_calls import _empty_graph
    from .batched_duel.schema import GraphLayout
    t=raw.t;out=[]
    for begin in range(0,len(indices),8):
        selected=indices[begin:begin+8];count=len(selected)
        graph=_empty_graph(t,GraphLayout(count,16384,32768,131072),raw.device)
        errors=raw.gather_into_device(t.tensor(selected,dtype=t.int64,device=raw.device),graph,
            t.ones(count,dtype=t.bool,device=raw.device))
        if errors.cpu().tolist()!=[0]*count:raise RuntimeError('Raw prefix gather failed')
        out.extend(_graph_values(graph))
    return out


def _decode_leases(store,leases,count):
    from .batched_duel.device_rule_calls import _empty_graph
    from .batched_duel.schema import GraphLayout
    graph=_empty_graph(store.t,GraphLayout(count,16384,32768,131072),store.device)
    gathered=store.gather_into_device(leases,graph,shared_reads=True)
    if gathered.errors.cpu().tolist()!=[0]*count:raise RuntimeError('Full legal audit gather failed')
    return _graph_values(graph)


def _final_states(store,states,prior_complete):
    """Only clean/guard statuses may create a separate read-only gather request."""
    t=store.t;old={k:getattr(states,k).cpu().tolist()for k in ('slots','generations','owners','active','errors')}
    safe=states.slots.clamp(0,store.capacity-1)
    registry={k:getattr(store,k).index_select(0,safe).cpu().tolist()for k in ('live','generation','owner','roots')}
    root=store.schema.pools[store.schema.root];checks=[];final=[None]*len(prior_complete)
    for row,done in enumerate(prior_complete):
        good=((old['active'][row]or done)and old['errors'][row]in (0,256)and 0<=old['slots'][row]<store.capacity
            and old['generations'][row]>=1 and registry['live'][row]
            and registry['generation'][row]==old['generations'][row]
            and registry['owner'][row]==old['owners'][row]==row
            and root.base<registry['roots'][row]<=root.base+root.capacity)
        checks.append(dict(row=row,verified=bool(good),original={k:v[row]for k,v in old.items()}))
    good=[c['row']for c in checks if c['verified']]
    if good:
        ids=t.tensor(good,dtype=t.int64,device=states.slots.device)
        selected=replace(states,slots=states.slots[ids],generations=states.generations[ids],owners=states.owners[ids],
            active=t.ones(len(good),dtype=t.bool,device=states.slots.device),errors=t.zeros(len(good),dtype=t.int32,device=states.slots.device))
        from .batched_duel.record_soa import unpack_game_states
        decoded=unpack_game_states(store.gather_device(selected,shared_reads=True).to_host())
        for row,state in zip(good,decoded):final[row]=state
    if states.errors.cpu().tolist()!=old['errors']:raise AssertionError('Diagnostic gather altered original errors')
    return final,checks
