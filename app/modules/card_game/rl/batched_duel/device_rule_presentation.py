"""GPU metadata and public-decision plans over checked shared rule lanes.

These are concrete VM/projection calls, not a qualified full search adapter.
Captured outputs borrow buffers and leases until the caller consumes/releases
or copies them. No host states or action dictionaries are reconstructed.
"""
from dataclasses import fields,replace

from .device_rule_calls import DeviceRuleGraphCalls,_empty_graph
from .device_rule_lanes import DeviceRuleLanes
from .device_rule_protocol import DeviceDecision,DeviceStateBatch,DeviceValueObservation
from .device_rule_results import DeviceRuleResultReader
from .gpu_rule_projection import GraphProjectionKernel
from .gpu_seven_candidates import RawCandidateTokens,MAX_MULLIGAN_SELECTIONS,ORDER_SLOTS
from ..batched_search.compact_device_candidates import CompactDeviceCandidates


class DeviceRulePresentation:
    supports_full_duel_rules=False

    def __init__(self,calls,*,max_actions,max_output_bytes,seconds=60):
        if not isinstance(calls,DeviceRuleGraphCalls) or calls.closed:
            raise TypeError('Live shared device rule graph calls required')
        if any(type(v) is not int or v<1 for v in (max_actions,max_output_bytes)):
            raise ValueError('Explicit positive action and output-byte bounds required')
        self.calls=calls;self.vm=calls.vm;self.t=calls.t;self.device=calls.device
        self.rows=calls.rows;self.scratch_rows=calls.lanes;self.max_actions=max_actions
        self.max_output_bytes=max_output_bytes;self.seconds=seconds;self.closed=False;self.plans=[]
        self.reader=DeviceRuleResultReader(self.vm)
        try:self.projection=GraphProjectionKernel(device=self.device,max_candidates=max_actions,runtime='seven')
        except BaseException:self.reader.close();raise

    def close(self):
        if not self.closed:
            for plan in self.plans:plan.close()
            self.reader.close();self.projection.close();self.closed=True

    def prepare_metadata(self):
        return self._prepare('metadata')

    def prepare_decision(self,action_store):
        return self._prepare('decision',action_store)

    def prepare_value(self):
        return self._prepare('value')

    def _prepare(self,kind,action_store=None):
        if self.closed or self.calls.closed:raise RuntimeError('Rule presentation closed')
        if self.t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare presentation plans before capture')
        required=(('acting_side','finished','public_legal_decision','terminal_value') if kind=='value'
                  else ('acting_side','finished') if kind=='metadata' else ('public_legal_decision',))
        for entry in required:
            if entry not in self.vm.image.entries or not self.vm.capability(entry)['ready']:
                raise RuntimeError('Missing frozen presentation entry: '+entry)
        plan=DevicePresentationPlan(self,kind,action_store)
        self.plans.append(plan)
        return plan

    def release_decision_device(self,decision,*,active=None):
        """Release only a caller-owned decision lease once, including errors.

        Cloning numeric decision buffers does not acquire another graph lease;
        callers must deduplicate root/current aliases before releasing them.
        """
        if self.closed:raise RuntimeError('Rule presentation closed')
        if not isinstance(decision,DeviceDecision):raise TypeError('DeviceDecision required')
        allocated=decision.action_slots>=0
        if active is not None:
            self.calls._exact(active,self.t.bool,(self.rows,),'decision release mask')
            allocated=allocated&active
        decision.action_pool.release_device(DeviceStateBatch(decision.action_pool,
            decision.action_slots,decision.action_generations,decision.states.owners,allocated,decision.errors))


class DevicePresentationPlan:
    def __init__(self,presentation,kind,action_store):
        p=presentation;c=p.calls;t=p.t
        self.p=p;self.kind=kind;self.captured=False;self.closed=False
        self.lanes=DeviceRuleLanes(p.rows,p.scratch_rows,device=p.device,seconds=p.seconds)
        self.refs=t.empty((p.scratch_rows,1 if kind=='metadata' else 2),dtype=t.int32,device=p.device)
        self.load_errors=t.empty(p.scratch_rows,dtype=t.int32,device=p.device)
        self.bind_errors=t.empty(p.scratch_rows,dtype=t.int32,device=p.device)
        self.actor_rows=t.empty(p.scratch_rows,dtype=t.int64,device=p.device)
        self.actor_refs=t.empty(p.scratch_rows,dtype=t.int32,device=p.device)
        self.row_indices=t.arange(p.scratch_rows,device=p.device)
        if kind=='metadata':
            self.local=p.reader.prepare_metadata_workspace(p.scratch_rows)
            self.output=p.reader.prepare_metadata_workspace(p.rows)
            self.action_store=None
            return
        if kind=='value':
            self.local=p.reader.prepare_metadata_workspace(p.scratch_rows)
            self.pair=p.reader.prepare_pair_workspace(p.scratch_rows)
            self.scalar=p.reader.prepare_float_workspace(p.scratch_rows)
            self.root_refs=t.empty((p.scratch_rows,1),dtype=t.int32,device=p.device)
            self.output=DeviceValueObservation(
                t.zeros((p.rows,p.projection.obs_dim),dtype=t.float32,device=p.device),
                t.zeros((p.rows,1),dtype=t.float32,device=p.device),
                t.zeros(p.rows,dtype=t.bool,device=p.device),
                t.zeros(p.rows,dtype=t.float64,device=p.device),
                t.zeros(p.rows,dtype=t.int32,device=p.device))
            self.action_store=None
            return
        from .device_graph_storage import CompactDeviceGraphStore
        if not isinstance(action_store,CompactDeviceGraphStore) or action_store.closed or action_store.device!=p.device:
            self.close();raise TypeError('Live same-device compact action graph store required')
        self.action_store=action_store
        b,m=p.rows,p.max_actions
        # Explicit bound includes raw buffers, public x, refs/types, and leases.
        per_candidate=(7*8+8*4+2*4+MAX_MULLIGAN_SELECTIONS*16+2*ORDER_SLOTS*8+4+1+4+8)
        required=b*m*per_candidate+b*(p.projection.obs_dim*4+8+4+8+8+8+1)
        if required>p.max_output_bytes:
            self.close();raise MemoryError('Full legal public-decision output exceeds declared byte budget')
        z=dict(device=p.device)
        tokens=RawCandidateTokens(
            int_tokens=t.full((b,m,7),-1,dtype=t.int64,**z),
            float_tokens=t.zeros((b,m,8),dtype=t.float32,**z),
            item_flags=t.zeros((b,m,2),dtype=t.float32,**z),
            turn_counts=t.zeros(b,dtype=t.float32,**z),
            mulligan_cards=t.full((b,m,MAX_MULLIGAN_SELECTIONS),-1,dtype=t.int64,**z),
            counts=t.zeros(b,dtype=t.int64,**z),mask=t.zeros((b,m),dtype=t.bool,**z),
            order_cards=t.full((b,m,2,ORDER_SLOTS),-1,dtype=t.int64,**z),
            select_cards=t.full((b,m,MAX_MULLIGAN_SELECTIONS),-1,dtype=t.int64,**z),
            extra_ap=t.zeros((b,m),dtype=t.float32,**z),request_ids=())
        self.candidates=CompactDeviceCandidates(tokens,scratch_rows=p.scratch_rows,device=p.device)
        self.observations=t.zeros((b,p.projection.obs_dim),dtype=t.float32,**z)
        self.action_refs=t.full((b,m),-1,dtype=t.int32,**z)
        self.action_types=t.full((b,m),-1,dtype=t.int64,**z)
        self.action_leases=DeviceStateBatch(action_store,t.full((b,),-1,dtype=t.int64,**z),
            t.zeros(b,dtype=t.int64,**z),t.zeros(b,dtype=t.int64,**z),
            t.zeros(b,dtype=t.bool,**z),t.zeros(b,dtype=t.int32,**z))
        self.compact_graph=_empty_graph(t,c.input_graph.layout,p.device)
        self.compact=p.vm.prepare_compaction_workspace(c.large,self.compact_graph)

    def close(self):
        if not self.closed:self.lanes.close();self.closed=True

    def capture_device(self,states,observers=None):
        p=self.p;c=p.calls;t=p.t
        if self.closed or p.closed or c.closed:raise RuntimeError('Rule presentation plan closed')
        if self.captured:raise RuntimeError('Presentation plan already captured')
        if not t.cuda.is_current_stream_capturing():raise RuntimeError('Embed presentation in CUDA capture')
        c._validate_states(states,p.rows,pool=c.state_store)
        if states.active.dtype!=t.bool:raise ValueError('Device lease active mask must be bool')
        if self.kind in ('decision','value'):
            c._exact(observers,t.int64,(p.rows,),'decision observers')
        if self.kind=='value':
            self.output.observations.zero_();self.output.viewer_is_actor.zero_()
            self.output.terminal.zero_();self.output.terminal_value.zero_();self.output.errors.copy_(states.errors)
        elif self.kind=='decision':
            self.action_leases.slots.fill_(-1);self.action_leases.generations.zero_()
            self.action_leases.owners.copy_(states.owners);self.action_leases.active.zero_()
            self.action_leases.errors.copy_(states.errors)
            self.observations.zero_();self.candidates.tokens.counts.zero_();self.candidates.tokens.mask.zero_()
            self.action_refs.fill_(-1);self.action_types.fill_(-1)
        elif observers is not None:raise ValueError('Metadata does not take an observer')
        else:
            self.output.meta.actors.fill_(-1);self.output.meta.phases.fill_(-1)
            self.output.meta.winners.fill_(-2);self.output.meta.finished.zero_()
            self.output.meta.versions.zero_();self.output.meta.errors.copy_(states.errors)
        def operation(local,ids,valid):
            gathered=c.state_store.gather_into_device(local,c.input_graph)
            errors=local.errors|gathered.errors
            active=(local.active&(errors==0)).view(t.uint8)
            p.vm.load_graph_into_device(c.workspace,c.input_graph,active=active,errors=self.load_errors)
            errors=errors|self.load_errors;active=(local.active&(errors==0)).view(t.uint8)
            if self.kind in ('metadata','value'):
                root_refs=self.refs if self.kind=='metadata' else self.root_refs
                root_refs[:,0].copy_(c.large.roots)
                actor=p.vm.launch_into_device(c.workspace,'acting_side',argument_refs=root_refs,
                    active=active,gpu_deadlines=c.deadlines)
                self.actor_refs.copy_(actor.outputs);errors=errors|actor.statuses[:,0]
                active=(local.active&(errors==0)).view(t.uint8)
                ended=p.vm.launch_into_device(c.workspace,'finished',argument_refs=root_refs,
                    active=active,gpu_deadlines=c.deadlines)
                errors=errors|ended.statuses[:,0]
                meta=p.reader.metadata_into_device(c.large,actor_refs=self.actor_refs,finished_refs=ended.outputs,
                    input_errors=errors.contiguous(),active=local.active.view(t.uint8),output=self.local)
                meta=replace(meta,errors=meta.errors|errors)
                if self.kind=='metadata':
                    for field in fields(meta):
                        self.lanes.scatter_tensor_device(getattr(meta,field.name),getattr(self.output.meta,field.name))
                    return
                errors=meta.errors
                active=(local.active&(errors==0)).view(t.uint8)
            self.actor_rows.copy_(t.where(valid.bool(),observers[ids],0))
            c.binder.bind_into_device(c.large,self.actor_rows,active,argument_refs=self.refs,errors=self.bind_errors)
            errors=errors|self.bind_errors;active=(local.active&(errors==0)).view(t.uint8)
            call=p.vm.launch_into_device(c.workspace,'public_legal_decision',argument_refs=self.refs,
                active=active,gpu_deadlines=c.deadlines)
            errors=errors|call.statuses[:,0];active=(local.active&(errors==0)).view(t.uint8)
            if self.kind=='value':
                pair=p.reader.pair_into_device(call,active=active,output=self.pair)
                errors=errors|pair.errors;active=(local.active&(errors==0)).view(t.uint8)
                observed,view_errors=p.projection.project_observation(call,pair.second,active=active)
                errors=errors|view_errors;active=(local.active&(errors==0)).view(t.uint8)
                terminal=p.vm.launch_into_device(c.workspace,'terminal_value',argument_refs=self.refs,
                    active=active,gpu_deadlines=c.deadlines)
                errors=errors|terminal.statuses[:,0]
                scalar=p.reader.float_into_device(c.large,terminal.outputs,input_errors=errors.contiguous(),
                    active=active,output=self.scalar)
                errors=errors|scalar.errors
                bad=meta.finished&((scalar.values!=0.)&(scalar.values!=1.)&(scalar.values!=-1.))
                errors=errors|bad.to(t.int32)*153
                value=DeviceValueObservation(observed.observations,
                    ((meta.actors==self.actor_rows)&local.active).to(t.float32)[:,None],meta.finished,
                    t.where(meta.finished,scalar.values,0.),errors)
                for field in fields(value):
                    self.lanes.scatter_tensor_device(getattr(value,field.name),getattr(self.output,field.name))
                return
            raw=p.projection.project_tokens_device(call,active=active)
            errors=errors|raw.errors;active=(local.active&(errors==0)).view(t.uint8)
            p.vm.compact_into_device(call,self.compact,active=active)
            errors=errors|self.compact.errors
            old=raw.action_refs.long()
            keys=t.where(old>=0,old,c.large.counts[:,0].long()[:,None]-1-old)
            live=raw.candidates.mask&(local.active&(errors==0))[:,None]
            out_of_range=(keys<0)|(keys>=self.compact.mapping.shape[1])
            mapped=self.compact.mapping[self.row_indices[:,None],keys.clamp(0,self.compact.mapping.shape[1]-1)]
            bad=(live&(out_of_range|(mapped<0))).any(-1)
            errors=errors|bad.to(t.int32)*151
            active=(local.active&(errors==0)).view(t.uint8)
            leased=self.action_store.put_device(self.compact_graph,local.owners,active)
            errors=errors|leased.errors
            lease=replace(leased,errors=errors)
            self.lanes.scatter_states_device(lease,self.action_leases)
            self.lanes.scatter_tensor_device(raw.observations.observations,self.observations)
            for field in fields(raw.candidates):
                if field.name!='request_ids':
                    self.lanes.scatter_tensor_device(getattr(raw.candidates,field.name),getattr(self.candidates.tokens,field.name))
            self.lanes.scatter_tensor_device(t.where(live&~out_of_range,mapped,-1).to(t.int32),self.action_refs)
            self.lanes.scatter_tensor_device(raw.action_types,self.action_types)
        self.lanes.capture_dispatch(states,operation)
        guard_error=(self.lanes.loop.errors!=0).to(t.int32)[0]*152
        destination_errors=(self.output.meta.errors if self.kind=='metadata' else self.output.errors
                            if self.kind=='value' else self.action_leases.errors)
        destination_errors.bitwise_or_(states.active.to(t.int32)*guard_error)
        self.captured=True
        if self.kind=='metadata':return self.output.meta
        if self.kind=='value':return self.output
        return DeviceDecision(states,self.action_store,self.action_leases.slots,self.action_leases.generations,
            self.action_refs,self.action_types,self.candidates.tokens.counts,self.candidates.tokens.mask,
            self.observations,self.candidates,self.action_leases.errors)
