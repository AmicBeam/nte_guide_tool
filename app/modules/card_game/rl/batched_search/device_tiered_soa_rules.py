"""Opt-in wide small-scratch SoA calls with checked narrow OOM fallback.

Both tiers use the same CUDA executor and persistent arenas. Only proven VM
node-capacity failures may retry; numerical error code 1 alone is ambiguous.
This component does not qualify full rules, search or GPU throughput.
"""
from dataclasses import fields,replace

from .device_soa_rules import DeviceSoARules
from .device_tiled_soa_rules import DeviceTiledSoARules,_candidate_indices
from ..batched_duel.device_rule_protocol import DeviceStateBatch,DeviceDecision


OUTPUT_LEASE_ERROR=4096
_OPERATIONS=('metadata','decision','value','sample','clean','teacher','rule_decision','step','step_clean')


class _ProbeQueue:
    """Static capture wrapper; the existing queue remains the device scheduler."""
    def __init__(self,queue,owner):self.queue=queue;self.owner=owner
    def __getattr__(self,name):return getattr(self.queue,name)
    def capture_dispatch(self,states,operation):
        p=self.owner;q=self.queue
        p._proven_capacity.zero_()
        def body(local,ids,valid):
            p._lane_capacity.zero_()
            operation(local,ids,valid)
            q.scatter_tensor_device(p._lane_capacity,p._proven_capacity)
        q.capture_dispatch(states,body)
        p._fast_guard.copy_(q.loop.errors)


class DeviceTieredSoARules(DeviceSoARules):
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,executor,state_store,action_store,*,rows,work_rows=32,fallback_rows=2,
                 inference_rows=2,max_actions=800,raw_tape=None,seconds=1800,
                 candidate_int32=False,require_teacher=False,staged_teacher=False):
        if (type(work_rows) is not int or not 1<=work_rows<=64 or type(fallback_rows) is not int
                or not 1<=fallback_rows<=work_rows):
            raise ValueError('Bounded fast lanes 1..64 and fallback lanes no wider than fast required')
        if getattr(executor,'execute_only',True):
            raise ValueError('Tiered search requires the full SoA executor and its 128KiB stack contract')
        self.closed=False;self.fast=None;self.slow=None
        try:
            common=dict(rows=rows,inference_rows=inference_rows,max_actions=max_actions,seconds=seconds,
                        candidate_int32=candidate_int32,require_teacher=require_teacher,staged_teacher=staged_teacher)
            self.fast=DeviceTiledSoARules(executor,state_store,action_store,work_rows=work_rows,raw_tape=None,
                scratch_nodes=262144,scratch_edges=524288,scratch_payload=4194304,**common)
            self.slow=DeviceTiledSoARules(executor,state_store,action_store,work_rows=fallback_rows,raw_tape=raw_tape,
                scratch_nodes=1048576,scratch_edges=2097152,scratch_payload=33554432,**common)
            self._prepare(self.fast,self.slow)
        except BaseException:
            self.close();raise

    def _prepare(self,fast,slow):
        """Bind self-owned adapters; CPU tests inject fake device queues here."""
        if (fast.executor is not slow.executor or fast.store is not slow.store
                or fast.action_store is not slow.action_store or fast.rows!=slow.rows
                or fast.device!=slow.device or fast.max_actions!=slow.max_actions):
            raise ValueError('Both SoA tiers must share executor, cohorts and independent arenas')
        self.fast,self.slow=fast,slow
        self.executor,self.store,self.action_store=fast.executor,fast.store,fast.action_store
        self.t,self.device,self.rows,self.max_actions=fast.t,fast.device,fast.rows,fast.max_actions
        self.work_rows,self.fallback_rows=fast.work_rows,slow.work_rows
        self.inference_rows=fast.inference_rows;self.candidate_integer_dtype=fast.candidate_integer_dtype
        self.templates,self.layout,self.action_layout=fast.templates,fast.layout,fast.action_layout
        self.require_teacher=fast.require_teacher;self.staged_teacher=fast.staged_teacher
        self.teacher_horizon=fast.teacher_horizon;self.raw_tape=slow.raw_tape;self.physical_loop=None
        self.closed=False;self._keepalive=[]
        kw=dict(device=self.device);t=self.t
        self._lane_capacity=t.zeros(self.work_rows,dtype=t.int32,**kw)
        self._proven_capacity=t.zeros(self.rows,dtype=t.int32,**kw)
        self._fast_guard=t.zeros(1,dtype=t.int32,**kw)
        self.fallback_counts=t.zeros((len(_OPERATIONS),self.rows),dtype=t.int64,**kw)
        self._row_ids=t.arange(self.rows,dtype=t.int64,**kw)

    @property
    def _queues(self):return self.fast._queues+self.slow._queues

    def _original(self,states):
        self.fast._check_states(states)
        if self.closed:raise RuntimeError('Tiered SoA rules closed')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Tiered SoA operations require CUDA capture')
        return replace(states,errors=states.errors.clone())

    def _valid_lease(self,states,pool):
        slots=states.slots;fit=(slots>=0)&(slots<pool.live.numel())&(states.generations>=0)
        safe=slots.clamp(0,pool.live.numel()-1)
        return (fit&pool.live[safe].bool()&(pool.generation[safe]==states.generations)
                &(pool.owner[safe]==states.owners))

    def _invoke_fast(self,name,states,*args,**kwargs):
        # Hooks exist only while this owned adapter builds one static site.
        # Replay uses captured tensor nodes; no Python dispatch/hook survives.
        inner=self.fast.inner;invoke=inner._invoke;site=self.fast._site
        def probed(*a,**kw):
            call=invoke(*a,**kw)
            self._lane_capacity.bitwise_or_((call.statuses[:,0]==1).to(self.t.int32))
            return call
        inner._invoke=probed
        self.fast._site=lambda kind:_ProbeQueue(site(kind),self)
        self._proven_capacity.zero_();self._fast_guard.zero_()
        try:return getattr(self.fast,name)(states,*args,**kwargs)
        finally:inner._invoke=invoke;self.fast._site=site

    def _retry_mask(self,source,errors,valid_input):
        return (source.active&(source.errors==0)&valid_input&(errors==1)
                &(self._proven_capacity!=0)&(self._fast_guard.reshape(())==0))

    def _clear_failed_output(self,output,source,retry,before,*,actions=False):
        """Retire only newly issued, checked leases before overwriting a row."""
        t=self.t;p=self.action_store if actions else self.store
        slots=output.action_slots if actions else output.slots
        gens=output.action_generations if actions else output.generations
        owners=output.states.owners if actions else output.owners
        allocated=slots>=0
        fit=allocated&(slots<p.live.numel())&(gens>=0)
        safe=slots.clamp(0,p.live.numel()-1)
        live=fit&p.live[safe].bool()&(p.generation[safe]==gens)
        preexisting=before[0][safe].bool()&(before[1][safe]==gens)
        owned=live&(p.owner[safe]==owners)&(owners==source.owners)&~preexisting
        # A parent may be inactive, but its live lease remains protected.
        if not actions:
            alias=((slots[:,None]==source.slots[None,:])&(gens[:,None]==source.generations[None,:])
                   &(slots[:,None]>=0)).any(-1)
            owned&=~alias
        duplicate=((slots[:,None]==slots[None,:])&(gens[:,None]==gens[None,:])
                   &retry[None,:]&retry[:,None]&(self._row_ids[:,None]!=self._row_ids[None,:])).any(-1)
        unsafe=retry&allocated&(~owned|duplicate)
        output.errors.bitwise_or_(unsafe.to(t.int32)*OUTPUT_LEASE_ERROR)
        take=retry&allocated&owned&~duplicate
        request=DeviceStateBatch(p,slots.clone(),gens.clone(),owners.clone(),take.contiguous(),
                                t.zeros_like(output.errors))
        p.release_device(request)
        failed=take&(request.errors!=0)
        output.errors.bitwise_or_(failed.to(t.int32)*OUTPUT_LEASE_ERROR)
        freed=take&(request.errors==0)
        slots.masked_fill_(freed,-1);gens.masked_fill_(freed,0)
        if not actions:output.active.masked_fill_(freed,False)
        self._keepalive.extend((before,request))
        return retry&~unsafe&~failed

    def _scatter(self,queue,local,operation):
        # Queue.valid normally includes inactive roots. Limit writeback to
        # requested retries so a slow inactive row cannot erase fast success.
        before=queue.valid.clone()
        queue.valid.copy_((before.bool()&local.active).to(queue.valid.dtype))
        try:operation()
        finally:queue.valid.copy_(before)
        self._keepalive.append(before)

    def _fallback(self,kind,source,retry,output,operation,*,choices=None):
        t=self.t
        self.fallback_counts[_OPERATIONS.index(kind)].add_(retry.long())
        requested=replace(source,active=retry,errors=source.errors.clone())
        queue=self.slow._site(kind)
        def body(local,ids,valid):
            result=operation(local,ids,valid)
            if choices is not None:
                decision,chosen,errors=result
                result=replace(decision,errors=decision.errors|errors)
            def write():
                self._write(queue,result,output,local)
                if choices is not None:queue.scatter_tensor_device(chosen,choices)
            self._scatter(queue,local,write)
            self._keepalive.append(result)
        queue.capture_dispatch(requested,body)
        output.errors.bitwise_or_(retry.to(t.int32)*queue.loop.errors.reshape(()))
        self._keepalive.extend((requested,queue,retry,output))
        return output

    def _write(self,queue,result,output,local):
        if isinstance(output,DeviceStateBatch):
            queue.scatter_states_device(result,output)
            # State scatter ORs errors. This queue is restricted to authorized
            # retries, so replace their recovered fast capacity status with
            # the slow result, including any slow failure.
            queue.scatter_tensor_device(result.errors,output.errors)
            return
        if isinstance(output,DeviceDecision):
            for name in ('action_slots','action_generations','action_refs','action_types','counts','valid','observations'):
                if name=='observations' and not output.observations.numel():continue
                queue.scatter_tensor_device(getattr(result,name),getattr(output,name))
            extra=self.t.zeros_like(result.errors)
            if not isinstance(output.candidates,self.t.Tensor):
                for field in fields(output.candidates.tokens):
                    if field.name=='request_ids':continue
                    src=getattr(result.candidates.tokens,field.name);dst=getattr(output.candidates.tokens,field.name)
                    if src.dtype!=dst.dtype:
                        src,errors=_candidate_indices(self.t,src,dst.dtype,local.active)
                        extra.bitwise_or_(errors)
                    queue.scatter_tensor_device(src,dst)
            queue.scatter_tensor_device(result.errors|extra,output.errors)
            return
        for field in fields(output):queue.scatter_tensor_device(getattr(result,field.name),getattr(output,field.name))

    def _simple(self,kind,method,states,*args,lease=False,**kwargs):
        source=self._original(states);valid=self._valid_lease(source,self.store)
        before=(self.store.live.clone(),self.store.generation.clone()) if lease else None
        output=self._invoke_fast(method,source,*args,**kwargs)
        retry=self._retry_mask(source,output.errors,valid)
        if lease:retry=self._clear_failed_output(output,source,retry,before)
        return self._fallback(kind,source,retry,output,lambda local,ids,valid:
            getattr(self.slow.inner,method)(local,*[arg.index_select(0,ids) for arg in args],**kwargs))

    def metadata_device(self,states):return self._simple('metadata','metadata_device',states)
    def value_observation_device(self,states,viewers):return self._simple('value','value_observation_device',states,viewers)
    def sample_world_device(self,states,viewers,seeds):return self._simple('sample','sample_world_device',states,viewers,seeds,lease=True)
    def clean_device(self,states):return self._simple('clean','clean_device',states,lease=True)
    def teacher_continue_device(self,states,viewers,*,horizon=3):
        if not self.require_teacher:raise RuntimeError('Tiered SoA teacher is absent')
        if type(horizon) is not int or horizon!=3:raise ValueError('Frozen teacher requires horizon=3')
        return self._simple('teacher','teacher_continue_device',states,viewers,lease=True,horizon=horizon)

    def _decision(self,states,viewers,*,raw=False):
        source=self._original(states);valid=self._valid_lease(source,self.store)
        before=(self.action_store.live.clone(),self.action_store.generation.clone())
        name='rule_decision_device' if raw else 'decision_device'
        result=self._invoke_fast(name,source,viewers)
        output,choices,extra=result if raw else (result,None,None)
        if raw:output.errors.bitwise_or_(extra)
        # Preserve the caller's original decision-state identity without copying
        # or allocating a second full-B candidate/token matrix.
        output=replace(output,states=states)
        retry=self._retry_mask(source,output.errors,valid)
        retry=self._clear_failed_output(output,source,retry,before,actions=True)
        def operation(local,ids,valid):
            return getattr(self.slow.inner,name)(local,viewers.index_select(0,ids))
        if raw:
            self._fallback('rule_decision',source,retry,output,operation,choices=choices)
            return output,choices,output.errors
        return self._fallback('decision',source,retry,output,operation)

    def decision_device(self,states,viewers):return self._decision(states,viewers)
    def rule_decision_device(self,states,viewers):
        if not self.require_teacher:raise RuntimeError('Tiered SoA rule teacher is absent')
        return self._decision(states,viewers,raw=True)

    def _selected_decision(self,decision,ids):
        from .compact_device_candidates import CompactDeviceCandidates
        if isinstance(decision.candidates,self.t.Tensor):
            if decision.candidates.shape!=(self.rows,self.max_actions,0):raise ValueError('Internal raw teacher candidates required')
            candidates=decision.candidates.index_select(0,ids)
        else:
            tokens=type(decision.candidates.tokens)(**{f.name:getattr(decision.candidates.tokens,f.name).index_select(0,ids)
                for f in fields(decision.candidates.tokens) if f.name!='request_ids'})
            candidates=CompactDeviceCandidates(tokens,scratch_rows=self.slow.inference_rows,
                                               device=self.device,template_table=self.templates)
        saved=decision.states
        states=DeviceStateBatch(self.store,*[getattr(saved,name).index_select(0,ids)
            for name in ('slots','generations','owners','active','errors')])
        return DeviceDecision(states,self.action_store,*[getattr(decision,name).index_select(0,ids)
            for name in ('action_slots','action_generations','action_refs','action_types','counts','valid','observations')],
            candidates,decision.errors.index_select(0,ids))

    def _step(self,states,actors,decision,indices,*,physical=False,fused=False):
        if type(physical) is not bool:raise TypeError('Explicit physical step flag required')
        if physical:
            if fused:raise ValueError('Fused steps are hypothetical only')
            return self.slow.step_device(states,actors,decision,indices,physical=True)
        if not isinstance(decision,DeviceDecision) or decision.states.pool is not self.store or decision.action_pool is not self.action_store:
            raise ValueError('Original checked SoA decision arenas required')
        source=self._original(states)
        same=((decision.states.slots==source.slots)&(decision.states.generations==source.generations)
              &(decision.states.owners==source.owners))
        handles=DeviceStateBatch(self.action_store,decision.action_slots,decision.action_generations,
                                 source.owners,source.active,decision.errors)
        valid=self._valid_lease(source,self.store)&self._valid_lease(handles,self.action_store)&same&(decision.errors==0)
        before=(self.store.live.clone(),self.store.generation.clone())
        method='step_clean_device' if fused else 'step_device';kind='step_clean' if fused else 'step'
        output=self._invoke_fast(method,source,actors,decision,indices,physical=False)
        retry=self._clear_failed_output(output,source,self._retry_mask(source,output.errors,valid),before)
        return self._fallback(kind,source,retry,output,lambda local,ids,valid:
            getattr(self.slow.inner,method)(local,actors.index_select(0,ids),self._selected_decision(decision,ids),
                                          indices.index_select(0,ids),physical=False))

    def step_device(self,states,actors,decision,indices,*,physical=False):
        return self._step(states,actors,decision,indices,physical=physical)
    def step_clean_device(self,states,actors,decision,indices,*,physical=False):
        return self._step(states,actors,decision,indices,physical=physical,fused=True)
    def prepare_fused_steps(self):self.fast.prepare_fused_steps();self.slow.prepare_fused_steps()
    def initialize_device(self,parameters,*,owners=None):
        if self.closed:raise RuntimeError('Tiered SoA rules closed')
        return self.slow.initialize_device(parameters,owners=owners)
    def bind_game_loop(self,loop):
        if self.physical_loop is not None:raise ValueError('Bind tiered SoA loop once')
        self.slow.bind_game_loop(loop);self.physical_loop=loop
    def release_device(self,states):self.store.release_device(states)
    def fallback_receipt_device(self):
        return dict(fast_lanes=self.work_rows,slow_lanes=self.fallback_rows,operations=_OPERATIONS,
                    per_root_counts=self.fallback_counts,gpu_qualified=False)
    @property
    def workspace_bytes(self):
        own=sum(value.numel()*value.element_size() for value in
                (self._lane_capacity,self._proven_capacity,self._fast_guard,self.fallback_counts,self._row_ids))
        return self.fast.workspace_bytes+self.slow.workspace_bytes-getattr(self.store,'workspace_bytes',0)+own
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.fast is not None:self.fast.close()
        finally:
            try:
                if self.slow is not None:self.slow.close()
            finally:getattr(self,'_keepalive',[]).clear()
