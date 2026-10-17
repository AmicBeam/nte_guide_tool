"""Closed-device root cohort over bounded native typed-SoA work lanes.

The host builds each static operation once. Nested CUDA WHILE nodes select
leases and scatter results during replay; no host root/decision loop runs.
"""
from dataclasses import fields

from .device_soa_rules import DeviceSoARules
from ..batched_duel.device_rule_protocol import DeviceStateBatch, DeviceMeta, DeviceDecision, DeviceValueObservation
from ..batched_duel.device_rule_lanes import DeviceRuleLanes


class _ActiveSoALanes(DeviceRuleLanes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_rows=self.t.zeros(1,dtype=self.t.int64,device=self.device)
        self.total_dispatches=self.t.zeros(1,dtype=self.t.int64,device=self.device)
        self.peak_selected=self.t.zeros(1,dtype=self.t.int64,device=self.device)
        self.root_visits=self.t.zeros(self.rows,dtype=self.t.int64,device=self.device)
        self.tail_dispatches=self.t.zeros(1,dtype=self.t.int64,device=self.device)
        self.errored_rows=self.t.zeros(1,dtype=self.t.int64,device=self.device)

    def capture_dispatch(self, states, operation):
        if self.closed:
            raise RuntimeError('SoA lane dispatcher closed')
        self._validate_states(states, self.rows)
        if not callable(operation) or not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Static SoA operation must be embedded in CUDA capture')

        def initialize():
            self.cursor.zero_()
            self.pending.copy_(states.active.any().reshape(1).to(self.t.uint8))
            self.processed.zero_(); self.peak.zero_()

        def body():
            local = self.select_states_device(states)
            selected=self.selected.long()
            self.total_rows.add_(selected.sum().reshape(1))
            self.total_dispatches.add_(1)
            self.tail_dispatches.add_((self.valid.long().sum()<self.lanes).long().reshape(1))
            self.errored_rows.add_((local.active & (local.errors!=0)).long().sum().reshape(1))
            self.t.maximum(self.peak_selected,selected.sum().reshape(1),out=self.peak_selected)
            self.root_visits.index_add_(0,self.ids,selected)
            operation(local, self.ids, self.valid)
            self.module.launch('advance_rule_lanes', [self.cursor, self.pending, self.rows, self.lanes],
                               grid=1, block=1, capture_safe=True)
        self.loop.capture_loop(self.pending, body, initialize=initialize)


def _empty_tokens(t, rows, actions, device, *, integer_dtype=None):
    from ..batched_duel.gpu_seven_candidates import RawCandidateTokens, MAX_MULLIGAN_SELECTIONS, ORDER_SLOTS
    integer_dtype=t.int64 if integer_dtype is None else integer_dtype
    if integer_dtype not in (t.int32,t.int64):raise ValueError('Exact int32 or int64 candidate indices required')
    ints = lambda shape: t.full(shape, -1, dtype=integer_dtype, device=device)
    floats = lambda shape: t.zeros(shape, dtype=t.float32, device=device)
    return RawCandidateTokens(
        int_tokens=ints((rows, actions, 7)), float_tokens=floats((rows, actions, 8)),
        item_flags=floats((rows, actions, 2)), turn_counts=floats((rows,)),
        mulligan_cards=ints((rows, actions, MAX_MULLIGAN_SELECTIONS)),
        counts=t.zeros(rows, dtype=t.int64, device=device),
        mask=t.zeros((rows, actions), dtype=t.bool, device=device),
        order_cards=ints((rows, actions, 2, ORDER_SLOTS)),
        select_cards=ints((rows, actions, MAX_MULLIGAN_SELECTIONS)),
        extra_ap=floats((rows, actions)))


def _candidate_indices(t,source,target_dtype,active):
    if source.dtype!=t.int64 or target_dtype!=t.int32 or source.shape[0]!=active.shape[0]:
        raise ValueError('Checked int64 to int32 candidate transport required')
    overflow=((source<-(2**31))|(source>=2**31)).reshape(source.shape[0],-1).any(-1)
    return source.to(t.int32),(overflow&active).to(t.int32)*156


class DeviceTiledSoARules(DeviceSoARules):
    """The same canonical native rules; persistent roots and scratch are separate."""

    def __init__(self, executor, state_store, action_store, *, rows, work_rows=8,
                 inference_rows=2, max_actions=800, raw_tape=None, seconds=1800,candidate_int32=False,
                 require_teacher=False,staged_teacher=False, **scratch):
        if type(require_teacher) is not bool:raise TypeError('Explicit frozen teacher selection required')
        if type(staged_teacher) is not bool or (staged_teacher and not require_teacher):
            raise ValueError('Staged teacher requires explicit frozen teacher entries')
        if type(candidate_int32) is not bool:raise ValueError('Explicit candidate storage width required')
        if type(rows) is not int or not 1 <= rows <= 1600:
            raise ValueError('Independent root count must be 1..1600')
        if type(work_rows) is not int or not 1 <= work_rows <= 64:
            raise ValueError('Native rule work lanes must be 1..64')
        if type(inference_rows) is not int or not 1 <= inference_rows <= 32:
            raise ValueError('Independent inference tile must be 1..32')
        self.rows, self.work_rows, self.inference_rows = rows, min(rows, work_rows), min(rows, inference_rows)
        self.inner = DeviceSoARules(executor, state_store, action_store, rows=self.work_rows,
                                   max_actions=max_actions, require_teacher=require_teacher,
                                   staged_teacher=staged_teacher, **scratch)
        self.require_teacher=require_teacher
        self.staged_teacher=staged_teacher
        self.teacher_horizon=3 if require_teacher else 0
        self.executor, self.store, self.action_store = executor, state_store, action_store
        self.t, self.device, self.max_actions = self.inner.t, self.inner.device, max_actions
        self.candidate_integer_dtype=self.t.int32 if candidate_int32 else self.t.int64
        self.templates = self.inner.templates
        self.layout = self.inner.layout
        from ..batched_duel.schema import GraphLayout
        from ..batched_duel.device_graph_storage import MAX_NODES, MAX_EDGES, MAX_PAYLOAD
        self.action_layout = GraphLayout(rows, MAX_NODES, MAX_EDGES, MAX_PAYLOAD)
        self.raw_tape, self.physical_loop = raw_tape, None
        if raw_tape is not None and (raw_tape.rows != rows or raw_tape.device != self.device):
            self.inner.close()
            raise ValueError('Raw tape must match the independent root cohort')
        self.seconds, self.closed = seconds, False
        self._queues, self._sites, self._keepalive, self._init_resources = [], {}, [], []
        self._init_graph = None
        # Dispatchers/streams must exist before parent graph capture. A site is
        # consumed only while building, never by a replay-time Python branch.
        for name, count in (('metadata', 16), ('decision', 6), ('value', 6),
                            ('sample', 2), ('step', 4), ('clean', 4)):
            self._sites[name] = [self._new_queue() for _ in range(count)]
        if require_teacher:
            for name in ('teacher','rule_decision'):
                self._sites[name]=[self._new_queue()]
        self._site_index = {name: 0 for name in self._sites}

    def _new_queue(self):
        queue = _ActiveSoALanes(self.rows, self.work_rows, device=str(self.device), seconds=self.seconds)
        self._queues.append(queue)
        return queue

    def _site(self, name):
        if self.closed or not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Bounded SoA sites require an active parent capture')
        if name not in self._sites:
            raise RuntimeError('Prepare the SoA '+name+' capture site before capture')
        index = self._site_index[name]
        if index >= len(self._sites[name]):
            raise RuntimeError('No prepared SoA '+name+' capture site remains')
        self._site_index[name] += 1
        return self._sites[name][index]

    def _state_output(self, states):
        t = self.t
        return DeviceStateBatch(self.store,
            t.full((self.rows,), -1, dtype=t.int64, device=self.device),
            t.zeros(self.rows, dtype=t.int64, device=self.device), states.owners.clone(),
            t.zeros(self.rows, dtype=t.bool, device=self.device), states.errors.clone())

    def _check_states(self, states):
        if not isinstance(states, DeviceStateBatch) or states.pool is not self.store or len(states.slots) != self.rows:
            raise ValueError('Exact typed independent root cohort required')

    def _finish(self, queue, states, output):
        # Queue guards are sticky batch errors; they are never interpreted as
        # finished games or a successful partially processed cohort.
        output.errors.bitwise_or_(states.active.to(self.t.int32)*queue.loop.errors.reshape(()))
        self._keepalive.extend((queue, output))
        return output

    def metadata_device(self, states):
        self._check_states(states); t = self.t; queue = self._site('metadata')
        make = lambda dtype: t.zeros(self.rows, dtype=dtype, device=self.device)
        output = DeviceMeta(make(t.int64), make(t.int32), make(t.int64), make(t.bool),
                            make(t.int64), states.errors.clone())

        def operation(local, ids, valid):
            result = self.inner.metadata_device(local)
            for field in fields(DeviceMeta):
                queue.scatter_tensor_device(getattr(result, field.name), getattr(output, field.name))
        queue.capture_dispatch(states, operation)
        return self._finish(queue, states, output)

    def decision_device(self, states, observers):
        self._check_states(states); t = self.t; queue = self._site('decision')
        from ..batched_duel.gpu_seven_observation import OBS_DIM
        from .compact_device_candidates import CompactDeviceCandidates
        tokens = _empty_tokens(t, self.rows, self.max_actions, self.device,integer_dtype=self.candidate_integer_dtype)
        candidates = CompactDeviceCandidates(tokens, scratch_rows=self.inference_rows,
                                              device=self.device, template_table=self.templates)
        output = DeviceDecision(states, self.action_store,
            t.full((self.rows,), -1, dtype=t.int64, device=self.device),
            t.zeros(self.rows, dtype=t.int64, device=self.device),
            t.full((self.rows, self.max_actions), -1, dtype=t.int32, device=self.device),
            t.zeros((self.rows, self.max_actions), dtype=t.int64, device=self.device),
            tokens.counts, tokens.mask,
            t.zeros((self.rows, OBS_DIM), dtype=t.float32, device=self.device), candidates, states.errors.clone())

        def operation(local, ids, valid):
            result = self.inner.decision_device(local, observers.index_select(0, ids))
            for name in ('action_slots', 'action_generations', 'action_refs', 'action_types','observations'):
                queue.scatter_tensor_device(getattr(result, name), getattr(output, name))
            extra=t.zeros_like(result.errors)
            for field in fields(tokens):
                if field.name != 'request_ids':
                    source,target=getattr(result.candidates.tokens,field.name),getattr(tokens,field.name)
                    if source.dtype!=target.dtype:
                        source,errors=_candidate_indices(t,source,target.dtype,local.active)
                        extra.bitwise_or_(errors)
                    queue.scatter_tensor_device(source,target)
            queue.scatter_tensor_device(result.errors|extra,output.errors)
        queue.capture_dispatch(states, operation)
        return self._finish(queue, states, output)

    def value_observation_device(self, states, viewers):
        self._check_states(states); t = self.t; queue = self._site('value')
        from ..batched_duel.gpu_seven_observation import OBS_DIM
        output = DeviceValueObservation(
            t.zeros((self.rows, OBS_DIM), dtype=t.float32, device=self.device),
            t.zeros((self.rows, 1), dtype=t.float32, device=self.device),
            t.zeros(self.rows, dtype=t.bool, device=self.device),
            t.zeros(self.rows, dtype=t.float64, device=self.device), states.errors.clone())

        def operation(local, ids, valid):
            result = self.inner.value_observation_device(local, viewers.index_select(0, ids))
            for field in fields(DeviceValueObservation):
                queue.scatter_tensor_device(getattr(result, field.name), getattr(output, field.name))
        queue.capture_dispatch(states, operation)
        return self._finish(queue, states, output)

    def _state_call(self, kind, states, operation):
        self._check_states(states); queue = self._site(kind); output = self._state_output(states)

        def body(local, ids, valid):
            result = operation(local, ids, valid)
            queue.scatter_states_device(result, output)
        queue.capture_dispatch(states, body)
        return self._finish(queue, states, output)

    def sample_world_device(self, states, viewers, seeds):
        return self._state_call('sample', states,
            lambda local, ids, valid: self.inner.sample_world_device(
                local, viewers.index_select(0, ids), seeds.index_select(0, ids)))

    def clean_device(self, states):
        return self._state_call('clean', states, lambda local, ids, valid: self.inner.clean_device(local))

    def teacher_continue_device(self,states,viewers,*,horizon=3):
        if not self.require_teacher:
            raise RuntimeError('Three-step teacher is absent from the SoA rule path')
        if type(horizon) is not int or horizon!=3:
            raise ValueError('Frozen SoA teacher_continue requires horizon=3')
        return self._state_call('teacher',states,lambda local,ids,valid:
            self.inner.teacher_continue_device(local,viewers.index_select(0,ids),horizon=horizon))

    def rule_decision_device(self,states,observers):
        if not self.require_teacher:
            raise RuntimeError('Rule opponent is absent from the SoA rule path')
        self._check_states(states);t=self.t;queue=self._site('rule_decision')
        output=DeviceDecision(states,self.action_store,
            t.full((self.rows,),-1,dtype=t.int64,device=self.device),
            t.zeros(self.rows,dtype=t.int64,device=self.device),
            t.full((self.rows,self.max_actions),-1,dtype=t.int32,device=self.device),
            t.full((self.rows,self.max_actions),-1,dtype=t.int64,device=self.device),
            t.zeros(self.rows,dtype=t.int64,device=self.device),
            t.zeros((self.rows,self.max_actions),dtype=t.bool,device=self.device),
            t.empty((self.rows,0),dtype=t.float32,device=self.device),
            t.empty((self.rows,self.max_actions,0),dtype=t.float32,device=self.device),states.errors.clone())
        choices=t.full((self.rows,),-1,dtype=t.int64,device=self.device)
        def operation(local,ids,valid):
            result,chosen,errors=self.inner.rule_decision_device(local,observers.index_select(0,ids))
            for name in ('action_slots','action_generations','action_refs','action_types','counts','valid'):
                queue.scatter_tensor_device(getattr(result,name),getattr(output,name))
            queue.scatter_tensor_device(errors,output.errors)
            queue.scatter_tensor_device(chosen,choices)
        queue.capture_dispatch(states,operation)
        self._finish(queue,states,output)
        self._keepalive.append(choices)
        return output,choices,output.errors

    def step_device(self, states, actors, decision, action_indices, *, physical=False):
        return self._step_device(states,actors,decision,action_indices,physical=physical)

    def step_deferred_physical_device(self,states,actors,decision,action_indices):
        """Use the physical raw hook and shared step queue, then loop-owned clean."""
        return self._step_device(states,actors,decision,action_indices,physical=True,_defer=True)

    def step_clean_device(self,states,actors,decision,action_indices,*,physical=False):
        if physical is not False:raise ValueError('Fused clean currently accepts hypothetical steps only')
        return self._step_device(states,actors,decision,action_indices,fused=True)

    def prepare_fused_steps(self):
        if self.closed or self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare fused steps before CUDA capture')
        self.inner.prepare_fused_steps()
        if 'step_clean' not in self._sites:
            self._sites['step_clean']=[self._new_queue()]
            self._site_index['step_clean']=0

    def _step_device(self, states, actors, decision, action_indices, *, physical=False,fused=False,_defer=False):
        if type(_defer) is not bool:
            raise TypeError('Explicit private deferred step control required')
        if _defer:
            if physical is not True or fused:
                raise ValueError('Deferred presentation requires an unfused physical step')
            from .deferred_physical_presentation import require_deferred_step
            require_deferred_step(self)
        if type(physical) is not bool or not isinstance(decision, DeviceDecision):
            raise ValueError('Explicit physical mode and full legal decision required')
        if decision.states.pool is not self.store or decision.action_pool is not self.action_store:
            raise ValueError('Decision leases must use the exact SoA arenas')
        from .compact_device_candidates import CompactDeviceCandidates

        def operation(local, ids, valid):
            if isinstance(decision.candidates,self.t.Tensor):
                if decision.candidates.shape!=(self.rows,self.max_actions,0):
                    raise ValueError('Only internal raw teacher tensors may bypass candidate transport')
                candidates=decision.candidates.index_select(0,ids)
            else:
                tokens = type(decision.candidates.tokens)(**{
                    field.name: getattr(decision.candidates.tokens, field.name).index_select(0, ids)
                    for field in fields(decision.candidates.tokens) if field.name != 'request_ids'})
                candidates = CompactDeviceCandidates(tokens, scratch_rows=self.inference_rows,
                                                      device=self.device, template_table=self.templates)
            saved = decision.states
            selected_states = DeviceStateBatch(self.store, *[
                getattr(saved, name).index_select(0, ids)
                for name in ('slots', 'generations', 'owners', 'active', 'errors')])
            selected = DeviceDecision(selected_states, self.action_store, *[
                getattr(decision, name).index_select(0, ids) for name in
                ('action_slots', 'action_generations', 'action_refs', 'action_types', 'counts', 'valid', 'observations')],
                candidates, decision.errors.index_select(0, ids))
            original = self.inner._raw_capture_hook
            if physical and self.raw_tape is not None:
                if self.physical_loop is None:
                    raise RuntimeError('Bind physical counters before capture')
                self.inner._raw_capture_hook = lambda graph, owners, who, active: self.raw_tape.append_device(
                    graph, ids, owners, self.physical_loop.steps_device.index_select(0, ids),
                    self.physical_loop.source_versions_device.index_select(0, ids), who, active)
            try:
                if _defer:
                    return self.inner._step_device(local,actors.index_select(0,ids),selected,
                        action_indices.index_select(0,ids),physical=True,_defer=True,_deferred_owner=self)
                step=self.inner.step_clean_device if fused else self.inner.step_device
                return step(local, actors.index_select(0, ids), selected,
                            action_indices.index_select(0, ids), physical=physical)
            finally:
                self.inner._raw_capture_hook = original
        return self._state_call('step_clean' if fused else 'step', states, operation)

    def initialize_device(self, parameters, *, owners=None):
        """Static option ingress; all actual games are created by native CUDA."""
        if self.closed or self._init_graph is not None or self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Initialize this SoA cohort once outside parent capture')
        if (type(parameters) is not list or len(parameters) != self.rows
                or any(type(p) is not dict or {'sides', 'rng', 'next_entity'} <= set(p) for p in parameters)):
            raise ValueError('Exact static initializer metadata, never host games, required')
        import numpy as np
        from ..batched_duel.native_array_cpu import pack_call_arguments
        from ..batched_duel.schema import GraphLayout
        from ..batched_duel.cuda_vm import DeviceGraph
        from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
        from ..batched_duel.device_rule_calls import _empty_graph
        t = self.t; layout = GraphLayout(self.rows, 4096, 8192, 131072)
        packed, _ = pack_call_arguments([[p] for p in parameters], layout)
        sizes = packed.counts.astype(np.int64)*np.array([16, 8, 1], dtype=np.int64)
        pages = np.maximum(1, (sizes+np.array([4095,4095,4095]))//4096)
        pool = CompactDeviceGraphStore(slots=self.rows, node_pages=int(pages[:,0].sum()),
            edge_pages=int(pages[:,1].sum()), payload_pages=int(pages[:,2].sum()),
            node_page_bytes=4096, edge_page_bytes=4096, payload_page_bytes=4096,
            max_node_pages_per_slot=int(pages[:,0].max()), max_edge_pages_per_slot=int(pages[:,1].max()),
            max_payload_pages_per_slot=int(pages[:,2].max()), device=str(self.device))
        dense = DeviceGraph(layout, *[t.tensor(getattr(packed, n), device=self.device)
                                     for n in ('nodes','edges','payload','counts','roots')])
        if owners is None:
            owners = t.arange(self.rows, dtype=t.int64, device=self.device)
        roots = pool.put_device(dense, owners, t.ones(self.rows, dtype=t.bool, device=self.device))
        t.cuda.synchronize()
        if bool((roots.errors != 0).any().cpu()):
            pool.close()
            raise RuntimeError('Initializer option arena capacity/identity failed')
        del dense, packed
        queue = self._new_queue()
        output = DeviceStateBatch(self.store, t.full_like(roots.slots, -1), t.zeros_like(roots.generations),
            owners.clone(), t.zeros_like(roots.active), roots.errors.clone())
        options = _empty_graph(t, GraphLayout(self.work_rows,4096,8192,131072), self.device)
        graph = t.cuda.CUDAGraph()

        def operation(local, ids, valid):
            gathered = pool.gather_into_device(local, options, shared_reads=True)
            batch, scratch = self.inner._shared_batch, self.inner._shared_scratch
            for value in batch.buffers.values():
                value.zero_()
            schema = batch.schema
            batch.buffers['i32'][schema.string_static_refs:schema.string_static_refs+schema.max_strings] = -2147483648
            batch.roots.zero_(); self.executor.reset_workspace(batch, scratch)
            for name in ('nodes','edges','payload'):
                source = getattr(options,name); getattr(scratch,name)[:,:source.shape[1]].copy_(source)
            scratch.counts.copy_(options.counts); scratch.roots.copy_(options.roots)
            lane = t.arange(self.work_rows, dtype=t.int64, device=self.device)
            root = options.roots.long().clamp(0, options.layout.max_nodes-1)
            edge = options.nodes[lane,root,1].long().clamp(0, options.layout.max_edges-1)
            refs = options.edges[lane,edge,0][:,None].contiguous()
            errors = local.errors | gathered.errors
            active = (local.active & (errors==0)).view(t.uint8)
            call = self.inner._invoke('initialize',batch,scratch,refs,active)
            initialized = self.inner._commit(call,local.owners,active,errors)
            queue.scatter_states_device(initialized,output)
            self._keepalive.extend((gathered,refs,active,call,initialized))
        with t.cuda.graph(graph):
            queue.capture_dispatch(roots, operation)
            output.errors.bitwise_or_(roots.active.to(t.int32)*queue.loop.errors.reshape(()))
        graph.replay(); t.cuda.synchronize()
        self._init_graph = graph
        self._init_resources.extend((pool, roots, options, output))
        return output

    @property
    def workspace_bytes(self):
        return self.inner.workspace_bytes+sum(v.numel()*v.element_size() for q in self._queues for v in
            (q.cursor,q.pending,q.ids,q.selected,q.valid,q.processed,q.peak,q.total_rows,
             q.total_dispatches,q.peak_selected,q.root_visits,q.tail_dispatches,q.errored_rows))

    def close(self):
        if not self.closed:
            self._init_graph = None
            for queue in self._queues:
                queue.close()
            self.inner.close()
            for obj in self._init_resources:
                close = getattr(obj, 'close', None)
                if callable(close):
                    close()
            self._keepalive.clear(); self._init_resources.clear(); self.closed = True
