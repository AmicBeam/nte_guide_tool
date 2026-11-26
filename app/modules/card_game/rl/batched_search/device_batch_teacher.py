"""Exact GPU prepare -> independent preview tasks -> checked batch ranking.

One instance owns one capture site. It preserves complete canonical legal
order and raw action graphs; no public model encoding or host rule callback.
"""
from dataclasses import replace
from ..batched_duel.cuda_vm import DeviceCall, DeviceGraph
from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from .cuda_conditional_graph import CudaConditionalWhile
from .device_frozen_rules import DeviceFrozenRules, _ActiveRuleLanes
from .device_teacher_decode import DeviceTeacherDecoder
from .device_teacher_preview_jobs import DeviceTeacherPreviewJobs
from .device_teacher_ranking import DeviceTeacherRanking

PREVIEW_ERROR=4096


class _PreviewQueue:
    def __init__(self, owner):
        t=owner.t;d=owner.device
        self.owner=owner
        self.positions=t.arange(owner.calls.lanes,dtype=t.int64,device=d)
        self.cursor=t.zeros(1,dtype=t.int64,device=d)
        self.pending=t.zeros(1,dtype=t.uint8,device=d)
        self.errors=t.zeros(1,dtype=t.int32,device=d)
        self.loop=CudaConditionalWhile(device=str(d),
            max_ticks=(owner.capacity+owner.calls.lanes-1)//owner.calls.lanes+1,
            seconds=owner.calls.seconds)

    def capture(self, operation):
        o=self.owner;t=o.t
        def initialize():
            self.cursor.zero_()
            count=o.jobs.job_count
            bad=(count<0)|(count>o.capacity)
            self.errors.copy_(bad.to(t.int32))
            self.pending.copy_(((count>0)&~bad).to(t.uint8))
        def body():
            raw=self.cursor[0]+self.positions
            inside=raw<o.capacity;ids=raw.clamp(0,o.capacity-1)
            on=inside&(raw<o.jobs.job_count[0])
            operation(ids,on,t.where(inside,raw,o.capacity))
            self.cursor.add_(o.calls.lanes)
            self.pending.copy_((self.cursor<o.jobs.job_count).to(t.uint8))
        self.loop.capture_loop(self.pending,body,initialize=initialize)
        return self.errors|self.loop.errors.to(t.int32)

    def close(self): self.loop.close()


class DeviceBatchTeacherActions:
    supports_full_duel_rules=False
    host_rule_callbacks=False
    root_policy_compatible=False

    def __init__(self,rules,*,max_actions=800,shared_template=False):
        if not isinstance(rules,DeviceFrozenRules) or rules.closed:
            raise TypeError('Live same-source DeviceFrozenRules required')
        if type(max_actions)is not int or not 1<=max_actions<=800:
            raise ValueError('Complete bounded teacher action capacity required')
        if type(shared_template)is not bool:
            raise TypeError('Static shared-template declaration required')
        c=rules.calls;t=c.t;d=c.device
        if t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare teacher outside capture')
        self.shared_template=shared_template
        self.prepare_entry='teacher_prepare_with_template' if shared_template else 'teacher_prepare'
        self.preview_entry='teacher_preview_with_template' if shared_template else 'teacher_preview'
        for entry in (self.prepare_entry,self.preview_entry):
            if entry not in c.vm.image.entries or not c.vm.capability(entry)['ready']:
                raise RuntimeError('Missing exact frozen teacher entry: '+entry)
        self.rules,self.calls,self.t,self.device=rules,c,t,d
        self.rows,self.max_actions,self.capacity=c.rows,max_actions,c.rows*max_actions
        self.closed=False;self.captured=False
        self.prepare_lanes=self.decoder=self.jobs=self.ranking=self.queue=None
        try:
            self.decoder=DeviceTeacherDecoder(c.vm)
            self.prepare_lanes=_ActiveRuleLanes(c.rows,c.lanes,device=d,seconds=c.seconds)
            self.jobs=DeviceTeacherPreviewJobs(rows=c.rows,max_actions=max_actions,
                capacity=self.capacity,device=d)
            self.ranking=DeviceTeacherRanking(rows=c.rows,max_actions=max_actions,device=d)
            self.queue=_PreviewQueue(self)
            kw=dict(device=d);b,m,l=c.rows,max_actions,c.lanes
            self.kinds=t.full((b,m),-1,dtype=t.int64,**kw)
            self.battle=t.zeros((b,m),dtype=t.bool,**kw)
            self.valid=t.zeros((b,m),dtype=t.bool,**kw)
            self.counts=t.zeros(b,dtype=t.int64,**kw)
            self.errors=t.zeros(b,dtype=t.int32,**kw)
            self.action_refs=t.full((b,m),-1,dtype=t.int32,**kw)
            self.preview_refs=t.full((b,m),-1,dtype=t.int32,**kw)
            self.leases=DeviceStateBatch(rules.action_store,
                t.full((b,),-1,dtype=t.int64,**kw),t.zeros(b,dtype=t.int64,**kw),
                t.zeros(b,dtype=t.int64,**kw),t.zeros(b,dtype=t.bool,**kw),self.errors)
            self.preview_values=[t.zeros((b,m),dtype=t.float64,**kw) for _ in range(4)]
            self.preview_statuses=t.zeros(self.capacity+1,dtype=t.int32,**kw)
            self.error_hits=t.zeros(b+1,dtype=t.int64,**kw)
            self.prepare_refs=t.empty((l,2),dtype=t.int32,**kw)
            self.legal_refs=t.empty(l,dtype=t.int32,**kw)
            self.local_kinds=t.empty((l,m),dtype=t.int64,**kw)
            self.local_battle=t.empty((l,m),dtype=t.bool,**kw)
            self.local_counts=t.empty(l,dtype=t.int64,**kw)
            self.local_refs=t.empty((l,m),dtype=t.int32,**kw)
            self.local_preview_refs=t.empty((l,m),dtype=t.int32,**kw)
            self.local_valid=t.empty((l,m),dtype=t.bool,**kw)
            self.local_errors=t.empty(l,dtype=t.int32,**kw)
            self.decode_errors=t.empty_like(self.local_errors)
            self.local_values=[t.empty(l,dtype=t.float64,**kw) for _ in range(4)]
            self.actor_rows=t.empty(l,dtype=t.int64,**kw)
            self.observations=t.empty((b,0),dtype=t.float32,**kw)
            self.candidates=t.empty((b,m,0),dtype=t.float32,**kw)
            small=c.input_graph
            self.graph=DeviceGraph(small.layout,t.zeros_like(small.nodes),t.zeros_like(small.edges),
                t.zeros_like(small.payload),t.zeros_like(small.counts),t.zeros_like(small.roots))
            self.compact=c.vm.prepare_compaction_workspace(c.large,self.graph)
        except BaseException:
            self.close();raise

    def close(self):
        if not self.closed:
            for obj in (self.queue,self.ranking,self.jobs,self.prepare_lanes,self.decoder):
                if obj is not None:obj.close()
            self.closed=True

    def _prepare(self,states,actors):
        c=self.calls;t=self.t
        def operation(local,ids,valid):
            loaded=c.state_store.gather_into_device(local,c.input_graph)
            errors=local.errors|loaded.errors
            active=(local.active&(errors==0)).view(t.uint8)
            c.vm.load_graph_into_device(c.workspace,c.input_graph,active=active,errors=c.load_errors)
            errors=errors|c.load_errors;active=(local.active&(errors==0)).view(t.uint8)
            self.actor_rows.copy_(t.where(valid.bool(),actors[ids],0))
            c.binder.bind_into_device(c.large,self.actor_rows,active,
                argument_refs=self.prepare_refs,errors=c.bind_errors)
            errors=errors|c.bind_errors;active=(local.active&(errors==0)).view(t.uint8)
            call=c.vm.launch_into_device(c.workspace,self.prepare_entry,
                argument_refs=self.prepare_refs,active=active,gpu_deadlines=c.deadlines)
            errors=errors|call.statuses[:,0]
            self.decoder.decode('decode_prepare',call,errors,local.active,
                [self.legal_refs,self.local_kinds,self.local_battle.view(t.uint8),
                 self.local_counts,self.local_errors],width=self.max_actions,
                 shared_template=self.shared_template)
            errors=errors|self.local_errors;active=(local.active&(errors==0)).view(t.uint8)
            legal_call=DeviceCall(c.vm,call.arena,self.legal_refs,call.statuses,call.frames,None)
            c.vm.compact_into_device(legal_call,self.compact,active=active)
            errors=errors|self.compact.errors
            self.decoder.legal(self.graph,errors,local.active,
                [self.local_refs,self.local_preview_refs,self.local_valid.view(t.uint8),self.local_counts,self.local_errors],
                self.max_actions,shared_template=self.shared_template)
            errors=errors|self.local_errors
            saved=self.rules.action_store.put_device(self.graph,local.owners,
                (local.active&(errors==0)).contiguous())
            saved.errors.bitwise_or_(errors)
            self.prepare_lanes.scatter_states_device(saved,self.leases)
            for scratch,output in ((self.local_kinds,self.kinds),(self.local_battle,self.battle),
                    (self.local_counts,self.counts),(self.local_refs,self.action_refs),
                    (self.local_preview_refs,self.preview_refs),(self.local_valid,self.valid)):
                self.prepare_lanes.scatter_tensor_device(scratch,output)
        self.prepare_lanes.capture_dispatch(states,operation)
        self.errors.bitwise_or_(states.active.to(t.int32)*(self.prepare_lanes.loop.errors!=0).to(t.int32)[0]*PREVIEW_ERROR)

    def _previews(self,states):
        c=self.calls;t=self.t
        def operation(ids,on,destinations):
            parent=self.jobs.parent_row[ids].clamp(0,self.rows-1)
            ordinal=self.jobs.ordinal[ids].clamp(0,self.max_actions-1)
            local=DeviceStateBatch(c.state_store,self.jobs.slots[ids],self.jobs.generations[ids],
                self.jobs.owners[ids],on,self.errors[parent])
            loaded=c.state_store.gather_into_device(local,c.input_graph,shared_reads=True)
            errors=local.errors|loaded.errors
            active=(on&(errors==0)).view(t.uint8)
            c.vm.load_graph_into_device(c.workspace,c.input_graph,active=active,errors=c.load_errors)
            errors=errors|c.load_errors
            action_states=DeviceStateBatch(self.rules.action_store,self.leases.slots[parent],
                self.leases.generations[parent],self.leases.owners[parent],
                (on&(errors==0)).contiguous(),errors)
            gathered=self.rules.action_store.gather_into_device(action_states,c.action_graph,shared_reads=True)
            errors=errors|gathered.errors;active=(on&(errors==0)).view(t.uint8)
            c.selected_refs.copy_(t.where(on,self.preview_refs[parent,ordinal],0))
            c.action_statuses.zero_()
            c.vm.compact_into_device(c.action_call,c.action_compact,active=active)
            errors=errors|c.action_compact.errors;active=(on&(errors==0)).view(t.uint8)
            self.actor_rows.copy_(t.where(on,self.jobs.actors[ids],0))
            c.binder.bind_action_into_device(c.large,self.actor_rows,c.selected_action,active,
                argument_refs=c.refs_bound,errors=c.bind_errors)
            errors=errors|c.bind_errors;active=(on&(errors==0)).view(t.uint8)
            call=c.vm.launch_into_device(c.workspace,self.preview_entry,argument_refs=c.refs_bound,
                active=active,gpu_deadlines=c.deadlines)
            errors=errors|call.statuses[:,0]
            self.decoder.decode('decode_preview',call,errors,on,[*self.local_values,self.decode_errors])
            errors=errors|self.decode_errors
            self.preview_statuses.index_copy_(0,destinations,t.where(on,errors,0))
            self.error_hits.zero_()
            self.error_hits.index_add_(0,t.where(on,parent,self.rows),(on&(errors!=0)).to(t.int64))
            self.errors.bitwise_or_((self.error_hits[:self.rows]!=0).to(t.int32)*PREVIEW_ERROR)
            self.jobs.scatter_previews_device(ids,self.jobs.parent_row[ids],self.jobs.ordinal[ids],
                self.jobs.owners[ids],self.jobs.generations[ids],self.jobs.slots[ids],on,
                *self.local_values,*self.preview_values,self.errors,
                epochs=self.jobs.job_epoch[ids],actors=self.jobs.actors[ids])
        flags=self.queue.capture(operation)
        self.errors.bitwise_or_(states.active.to(self.t.int32)*(flags!=0).to(self.t.int32)[0]*PREVIEW_ERROR)

    def capture_device(self,states,actors):
        c=self.calls;t=self.t
        if self.closed or self.rules.closed or c.closed:raise RuntimeError('Batch teacher closed')
        if self.captured:raise RuntimeError('Batch teacher capture site already frozen')
        if not t.cuda.is_current_stream_capturing():raise RuntimeError('Embed batch teacher in CUDA graph')
        c._validate_states(states,c.rows,pool=c.state_store)
        c._exact(actors,t.int64,(c.rows,),'teacher actors')
        self.errors.copy_(states.errors);self.counts.zero_();self.valid.zero_();self.battle.zero_()
        self.action_refs.fill_(-1);self.kinds.fill_(-1);self.preview_statuses.zero_()
        self.preview_refs.fill_(-1)
        self.leases.slots.fill_(-1);self.leases.generations.zero_()
        self.leases.owners.copy_(states.owners);self.leases.active.zero_()
        for value in self.preview_values:value.zero_()
        self._prepare(states,actors)
        self.jobs.compact_device(self.valid,self.battle,self.counts,states.active,self.errors,
            states.slots,states.generations,states.owners,actors)
        self.errors.bitwise_or_(self.jobs.errors)
        self._previews(states)
        self.jobs.finalize_device(self.errors)
        ranked=self.ranking.rank_device(self.kinds,self.battle,*self.preview_values,
            self.valid,self.counts,states.active,self.errors)
        self.errors.bitwise_or_(ranked.errors)
        decision=DeviceDecision(states,self.rules.action_store,self.leases.slots,self.leases.generations,
            self.action_refs,self.kinds,self.counts,self.valid,self.observations,self.candidates,self.errors)
        self.captured=True
        return decision,ranked.choices,self.errors
