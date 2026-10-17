"""Closed-device rule/search interface over persistent typed domain columns.

Canonical rule control flow is compiled once. Host Python configures static
calls; roots, owners, masks, chance transitions and action selection remain
on CUDA. Full software parity and hardware qualification are separate gates.
"""
from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceValueObservation, DeviceStateBatch
from ..batched_duel.device_soa_storage import DeviceSoAStore
from ..batched_duel.native_soa_cuda import CudaSoARules


def _validate_fused_clean(image):
    """Bind the one-list allocation proof to the exact frozen clean source."""
    import ast
    import inspect
    import textwrap
    from ..league_rollout import clean
    body=ast.parse(textwrap.dedent(inspect.getsource(clean))).body[0].body
    expected=ast.parse("s['events']=[]\ns.pop('_public_board',None)\nreturn s").body
    if [ast.dump(v) for v in body]!=[ast.dump(v) for v in expected]:
        raise ValueError('Canonical clean allocation behavior changed')
    functions=image.function_ids
    entry=image.entries.get('clean')
    if (type(entry) is not int or not 0<=entry<len(functions)
            or not functions[entry].startswith(('func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:clean:',
                'func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:seven_clean:'))
            or not any(name.startswith('func:app.modules.card_game.rl.league_rollout:clean:') for name in functions)):
        raise ValueError('Exact frozen canonical clean entry required')


class DeviceSoARules:
    supports_full_duel_rules = False
    require_teacher = False
    teacher_horizon = 0
    execution_kind = 'device_domain_record_soa_aot'

    def __init__(self, executor, state_store, action_store, *, rows, max_actions=800,
                 scratch_nodes=1048576, scratch_edges=2097152, scratch_payload=33554432,
                 raw_tape=None, require_teacher=False,staged_teacher=False):
        if type(require_teacher) is not bool:
            raise TypeError('Explicit frozen teacher selection required')
        if type(staged_teacher) is not bool or (staged_teacher and not require_teacher):
            raise ValueError('Staged teacher requires explicit frozen teacher entries')
        if not isinstance(executor,CudaSoARules) or not isinstance(state_store,DeviceSoAStore):
            raise TypeError('Native typed SoA CUDA executor and state arena required')
        if require_teacher and not {'teacher_continue','rule_decision'} <= set(executor.image.entries):
            raise ValueError('Both frozen SoA teacher entries required')
        from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
        if not isinstance(action_store,CompactDeviceGraphStore):
            raise TypeError('Independent transient action arena required')
        if executor.device != state_store.device or action_store.device != executor.device:
            raise ValueError('Rule, state and action arenas must share CUDA device')
        if any(type(value) is not int or not 0 < value < 2**31
               for value in (rows,max_actions,scratch_nodes,scratch_edges,scratch_payload)):
            raise ValueError('Explicit bounded rule dimensions required')
        self.executor, self.store, self.action_store = executor,state_store,action_store
        self.t,self.device,self.rows,self.max_actions=executor.torch,executor.device,rows,max_actions
        from ..batched_duel.schema import GraphLayout
        self.layout=GraphLayout(rows,scratch_nodes,scratch_edges,scratch_payload)
        from ..batched_duel.device_graph_storage import MAX_NODES, MAX_EDGES, MAX_PAYLOAD
        self.action_layout=GraphLayout(rows,MAX_NODES,MAX_EDGES,MAX_PAYLOAD)
        from ..batched_duel.device_rule_arguments import DeviceRuleArgumentBinder
        from ..batched_duel.gpu_rule_projection import GraphProjectionKernel
        from ..batched_duel.record_soa_compact import CudaRecordSoACompact
        from ..batched_duel.gpu_seven_candidates import get_card_template_table
        from .device_inplace_guard_encoding import prepare_inplace_guard_encoding
        self.binder=DeviceRuleArgumentBinder(device=self.device)
        self.projection=GraphProjectionKernel(device=self.device,max_candidates=max_actions,runtime='seven')
        self.compactor=CudaRecordSoACompact(executor.schema,device=str(self.device))
        self.compactor._ensure()  # Compile before CUDA graph capture.
        self.templates=get_card_template_table(self.device)
        prepare_inplace_guard_encoding(self.device)
        from .device_masked_row_copy import prepare_masked_row_copy
        prepare_masked_row_copy(self.device)
        from ..batched_duel.device_rule_calls import _empty_graph
        self._shared_batch=executor.schema.allocate(rows).to_torch(self.device)
        self._shared_scratch=_empty_graph(self.t,self.layout,self.device)
        self.executor.prepare_workspace(self._shared_batch,self._shared_scratch)
        prepare_store=getattr(self.store,'prepare_workspace',None)
        if callable(prepare_store):
            prepare_store(rows)
        self._compact_workspace=self.compactor.prepare_workspace(rows)
        self._keepalive=[]
        self.raw_tape=raw_tape
        self.require_teacher=require_teacher
        self.teacher_horizon=3 if require_teacher else 0
        self._teacher=None
        self._teacher_pipeline=None
        self.staged_teacher=staged_teacher
        self._raw_capture_hook=None  # Optional static graph-building sink, never a replay-time Python call.
        self.physical_loop=None
        if raw_tape is not None and (raw_tape.rows!=rows or raw_tape.device!=self.device):
            raise ValueError('Raw action tape must match the exact SoA cohort')
        if require_teacher:
            from .device_soa_teacher import DeviceSoATeacher
            try:
                self._teacher=DeviceSoATeacher(self,max_actions=max_actions)
                if staged_teacher:
                    from .device_soa_teacher_pipeline import DeviceSoATeacherPipeline
                    self._teacher_pipeline=DeviceSoATeacherPipeline(self,self._teacher)
            except BaseException:
                self.close()
                raise

    def bind_game_loop(self,loop):
        if self.physical_loop is not None:
            raise ValueError('Bind the SoA physical loop once')
        if loop.rows!=self.rows or loop.device!=self.device:
            raise ValueError('Matching SoA physical loop required')
        self.physical_loop=loop

    def initialize_device(self, parameters, *, owners=None):
        """One setup ingress of seeds/deck metadata, followed by native setup.

        No Python Game or formal-rule callback is accepted. This setup method
        runs before graph capture; subsequent decisions retain typed leases.
        """
        t=self.t
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare initial deck metadata before CUDA capture')
        if type(parameters) is not list or len(parameters)!=self.rows:
            raise ValueError('Matching initializer parameter rows required')
        if any(type(row) is not dict or {'sides','rng','next_entity'} <= set(row) for row in parameters):
            raise ValueError('Only initializer metadata is accepted')
        from ..batched_duel.native_array_cpu import pack_call_arguments
        from ..batched_duel.schema import GraphLayout
        packed,refs=pack_call_arguments([[row] for row in parameters],GraphLayout(self.rows,4096,8192,131072))
        blank=self._shared_batch
        for value in blank.buffers.values():
            value.zero_()
        schema=blank.schema
        blank.buffers['i32'][schema.string_static_refs:schema.string_static_refs+schema.max_strings]=-2147483648
        blank.roots.zero_()
        scratch=self._shared_scratch
        self.executor.reset_workspace(blank,scratch)
        for name in ('nodes','edges','payload'):
            src=getattr(packed,name)
            getattr(scratch,name)[:,:src.shape[1]].copy_(t.tensor(src,device=self.device))
        scratch.counts.copy_(t.tensor(packed.counts,device=self.device))
        scratch.roots.copy_(t.tensor(packed.roots,device=self.device))
        refs=t.tensor(refs,device=self.device)
        typed=t.zeros_like(refs)
        active=t.ones(self.rows,dtype=t.uint8,device=self.device)
        call=self.executor.launch('initialize',blank,scratch,refs,typed,active=active)
        if owners is None:
            owners=t.arange(self.rows,dtype=t.int64,device=self.device)
        errors=t.zeros(self.rows,dtype=t.int32,device=self.device)
        return self._commit(call,owners,active,errors)

    def _begin(self, states):
        t=self.t
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('SoA device rule sites must be captured before replay')
        if states.pool is not self.store or len(states.slots)!=self.rows:
            raise ValueError('Matching checked typed state leases required')
        batch=self.store.gather_into_device(states,self._shared_batch,shared_reads=True)
        errors=states.errors.clone()
        active=(states.active & (errors==0)).view(t.uint8)
        scratch=self._shared_scratch
        self.executor.reset_workspace(batch,scratch)
        return batch,scratch,errors,active

    def _invoke(self, entry, batch, scratch, refs, active):
        typed=self.t.zeros_like(refs)
        typed[:,0]=self.t.where(active.bool(),batch.roots,0)
        call=self.executor.launch(entry,batch,scratch,refs,typed,active=active)
        self._keepalive.append(call)
        return call

    def _bind(self, scratch, actors, active, *, seeds=None):
        t=self.t
        refs=t.empty((self.rows,3 if seeds is not None else 2),dtype=t.int32,device=self.device)
        errors=t.zeros(self.rows,dtype=t.int32,device=self.device)
        self.binder.bind_into_device(scratch,actors,active,argument_refs=refs,errors=errors,seeds=seeds)
        return refs,errors

    def _project(self, call, active):
        exported=self.executor.export_projection(call,active=active)
        self._keepalive.append(exported)
        return self.projection.project_tokens_device(exported,active=active)

    def metadata_device(self, states):
        from dataclasses import replace
        batch,scratch,errors,active=self._begin(states)
        refs=scratch.roots[:,None].contiguous()
        actor=self._invoke('acting_side',batch,scratch,refs,active)
        ended=self._invoke('finished',batch,scratch,refs,active)
        meta=self.executor.metadata(actor,ended,active)
        self._keepalive.append(meta)
        return replace(meta,errors=meta.errors|errors)

    def decision_device(self, states, observers):
        batch,scratch,errors,active=self._begin(states)
        refs,extra=self._bind(scratch,observers,active)
        errors.bitwise_or_(extra)
        active=(states.active & (errors==0)).view(self.t.uint8)
        call=self._invoke('public_legal_decision',batch,scratch,refs,active)
        exported=self.executor.export_projection(call,active=active)
        raw=self.projection.project_tokens_device(exported,active=active)
        errors=errors|raw.errors
        active=(states.active & (errors==0)).view(self.t.uint8)
        leased=self.action_store.put_device(exported.arena,states.owners,active)
        errors.bitwise_or_(leased.errors)
        from .compact_device_candidates import CompactDeviceCandidates
        candidates=CompactDeviceCandidates(raw.candidates,scratch_rows=self.rows,
            device=self.device,template_table=self.templates)
        self._keepalive.extend((exported,raw,candidates,leased))
        return DeviceDecision(states,self.action_store,leased.slots,leased.generations,
            raw.action_refs,raw.action_types,raw.candidates.counts.long(),raw.candidates.mask,
            raw.observations.observations,candidates,errors)

    def value_observation_device(self, states, viewers):
        batch,scratch,errors,active=self._begin(states)
        unary=scratch.roots[:,None].contiguous()
        actor=self._invoke('acting_side',batch,scratch,unary,active)
        ended=self._invoke('finished',batch,scratch,unary,active)
        meta=self.executor.metadata(actor,ended,active)
        self._keepalive.append(meta)
        refs,extra=self._bind(scratch,viewers,active)
        errors=errors|extra|meta.errors
        active=(states.active & (errors==0)).view(self.t.uint8)
        # A fixed viewer need not be the actor at the depth limit or terminal.
        # Its public observation is valid even when it has no legal actions.
        call=self._invoke('public_legal_decision',batch,scratch,refs,active)
        view=self.executor.select_result(call,(1,),active=active)
        exported=self.executor.export_projection(view,active=active)
        observed,observation_errors=self.projection.project_observation(
            exported,exported.outputs,active=active)
        self._keepalive.extend((view,exported,observed,observation_errors))
        terminal=self._invoke('terminal_value',batch,scratch,refs,active)
        value=self.executor.read_result(terminal,active)
        self._keepalive.append(value)
        errors=errors|observation_errors|value.errors
        invalid=meta.finished & (value.values!=-1.) & (value.values!=0.) & (value.values!=1.)
        errors.bitwise_or_(invalid.to(self.t.int32)*153)
        return DeviceValueObservation(observed.observations,
            ((meta.actors==viewers)&states.active).float()[:,None],meta.finished,
            self.t.where(meta.finished,value.values,0.),errors)

    def _result_errors(self,call,active,errors):
        t=self.t
        errors=errors|call.statuses[:,0]
        root=call.state.schema.pools[call.state.schema.root]
        invalid=(call.roots<=root.base)|(call.roots>root.base+root.capacity)
        errors.bitwise_or_((active.bool() & (errors==0) & invalid).to(t.int32)*150)
        return errors

    def _commit(self, call, owners, active, errors):
        t=self.t
        errors=self._result_errors(call,active,errors)
        eligible=active.bool() & (errors==0)
        from ..batched_duel.record_soa import TorchRecordSoABatch
        # Compaction has no active argument. Do not traverse failed/inactive
        # roots, including the inactive half of a fused-clean capacity split.
        batch=TorchRecordSoABatch(call.state.schema,call.state.buffers,
            t.where(eligible,call.roots,0))
        compact,status,event=self.compactor.compact_into_device(batch,self._compact_workspace)
        errors.bitwise_or_(t.where(eligible,status,0))
        put=self.store.put_device(compact,owners,active.bool() & (errors==0),errors=errors)
        self._keepalive.append((call,compact,status,event,put))
        return put

    def clean_device(self, states):
        batch,scratch,errors,active=self._begin(states)
        call=self._invoke('clean',batch,scratch,scratch.roots[:,None].contiguous(),active)
        return self._commit(call,states.owners,active,errors)

    def _compare_sampled_guard(self, call, guard, active, errors):
        from .device_view_sampler import compare_projected_views_device
        pair=[]
        for member in (0,1):
            selected=self.executor.select_result(call,(1,guard,member),active=active)
            pair.append(self._project(selected,active))
        errors.bitwise_or_(compare_projected_views_device(pair[0],pair[1],active.bool(),self.templates))
        self._keepalive.append(pair)
        return errors

    def sample_world_device(self, states, viewers, seeds):
        batch,scratch,errors,active=self._begin(states)
        unary=scratch.roots[:,None].contiguous()
        actor=self._invoke('acting_side',batch,scratch,unary,active)
        ended=self._invoke('finished',batch,scratch,unary,active)
        meta=self.executor.metadata(actor,ended,active)
        errors=errors|meta.errors
        active=(states.active & (errors==0)).view(self.t.uint8)
        refs,extra=self._bind(scratch,viewers,active,seeds=seeds)
        errors.bitwise_or_(extra)
        active=(states.active & (errors==0)).view(self.t.uint8)
        call=self._invoke('sample_world_guard_views_packet',batch,scratch,refs,active)
        from ..batched_duel.device_rule_results import DeviceRuleResultReader
        mulligan=DeviceRuleResultReader.phase_codes.index('mulligan')
        # Playing/choice inner and outer pairs alias exactly. Mulligan needs a
        # second encoded pair. Device masks skip native projection work on
        # inactive outer rows without host reads or capture-unsafe branches.
        outer_active=(active.bool() & (meta.phases==mulligan)).to(dtype=self.t.uint8).contiguous()
        self._keepalive.append((meta,outer_active,unary,actor,ended))
        errors=self._compare_sampled_guard(call,0,active,errors)
        errors=self._compare_sampled_guard(call,1,outer_active,errors)
        world=self.executor.select_result(call,(0,),active=active)
        return self._commit(world,states.owners,active,errors)

    def step_device(self, states, actors, decision, action_indices, *, physical=False):
        return self._step_device(states,actors,decision,action_indices,physical=physical)

    def _step_device(self, states, actors, decision, action_indices, *, physical=False,commit=True):
        if type(physical) is not bool or not isinstance(decision,DeviceDecision):
            raise ValueError('Explicit physical mode and full legal decision required')
        t=self.t
        batch,scratch,errors,active=self._begin(states)
        mismatch=((states.slots!=decision.states.slots)|(states.generations!=decision.states.generations)
                  |(states.owners!=decision.states.owners))
        if decision.states.pool is not states.pool:
            raise ValueError('Decision state belongs to another arena')
        errors=errors|decision.errors|(mismatch & states.active).to(t.int32)*151
        width=decision.action_refs.shape[1]
        safe=action_indices.clamp(0,width-1)
        selected_valid=decision.valid.gather(1,safe[:,None]).squeeze(1)
        bad=(action_indices<0)|(action_indices>=decision.counts)|(action_indices>=width)|~selected_valid
        errors.bitwise_or_((states.active & bad).to(t.int32)*9)
        active=(states.active & (errors==0)).view(t.uint8)
        leases=DeviceStateBatch(self.action_store,decision.action_slots,decision.action_generations,
            states.owners,active.bool(),errors.clone())
        from ..batched_duel.device_rule_calls import _empty_graph
        action_graph=_empty_graph(t,self.action_layout,self.device)
        gathered=self.action_store.gather_into_device(leases,action_graph,shared_reads=True)
        errors.bitwise_or_(gathered.errors)
        selected=decision.action_refs.gather(1,safe[:,None]).squeeze(1).contiguous()
        compact=self.executor.compact_transient(action_graph,selected,active)
        errors.bitwise_or_(compact.errors)
        if physical and self.raw_tape is not None:
            if self.physical_loop is None:
                raise RuntimeError('Bind physical counters before raw action capture')
            active=(states.active & (errors==0)).view(t.uint8)
            errors.bitwise_or_(self.raw_tape.append_device(compact.arena,
                t.arange(self.rows,dtype=t.int64,device=self.device),states.owners,
                self.physical_loop.steps_device,self.physical_loop.source_versions_device,
                actors,active.bool()))
        if physical and self._raw_capture_hook is not None:
            active=(states.active & (errors==0)).view(t.uint8)
            errors.bitwise_or_(self._raw_capture_hook(compact.arena,states.owners,actors,active.bool()))
        refs=t.empty((self.rows,3),dtype=t.int32,device=self.device)
        extra=t.zeros_like(errors)
        active=(states.active & (errors==0)).view(t.uint8)
        self.binder.bind_action_into_device(scratch,actors,compact.arena,active,
            argument_refs=refs,errors=extra)
        errors.bitwise_or_(extra)
        active=(states.active & (errors==0)).view(t.uint8)
        call=self._invoke('apply_action' if physical else 'simulate_action',batch,scratch,refs,active)
        self._keepalive.extend((action_graph,gathered,compact,refs,extra))
        if not commit:return call,errors,active
        return self._commit(call,states.owners,active,errors)

    def step_clean_device(self,states,actors,decision,action_indices,*,physical=False):
        """Keep disposable typed columns between exact step and exact clean.

        Reset the transient heap and clone memo before clean. Rows with no
        spare EventList object use the original compact/store/gather path;
        clean must still allocate a fresh list without changing capacity errors.
        No persistent input lease or random search stream is modified here.
        """
        if physical is not False:
            raise ValueError('Fused clean currently accepts hypothetical steps only')
        if not getattr(self,'_fused_clean_checked',False):
            _validate_fused_clean(self.executor.image)
            self._fused_clean_checked=True
        t=self.t
        call,errors,active=self._step_device(states,actors,decision,action_indices,
            physical=physical,commit=False)
        errors=self._result_errors(call,active,errors)
        batch,scratch=call.state,call.scratch
        pool=next((p for p in batch.schema.pools if p.name=='EventList'),None)
        if pool is None:raise ValueError('Exact typed EventList capacity required for fused clean')
        retry=states.active & (errors==0) & (batch.buffers['i32'][pool.used]>=pool.capacity)
        fast=(states.active & (errors==0) & ~retry).view(t.uint8)
        batch.roots.copy_(t.where(fast.bool(),call.roots,batch.roots))
        self.executor.reset_workspace(batch,scratch)
        cleaned=self._invoke('clean',batch,scratch,scratch.roots[:,None].contiguous(),fast)
        result=self._commit(cleaned,states.owners,fast,errors)
        # Static graph sites remain present; the device mask makes this slow
        # path inert when clean has its one required fresh typed list available.
        # Inactive clean rows retain their original typed step result. Compact
        # that result directly instead of executing the canonical action twice.
        advanced=self._commit(call,states.owners,retry,errors)
        fallback=self.clean_device(advanced)
        self.release_device(advanced)
        self._keepalive.extend((call,errors,retry,fast,result,advanced,fallback))
        return DeviceStateBatch(self.store,
            t.where(retry,fallback.slots,result.slots),
            t.where(retry,fallback.generations,result.generations),states.owners,
            result.active|fallback.active,t.where(retry,fallback.errors,result.errors))

    def prepare_fused_steps(self):
        if self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare fused steps before CUDA capture')
        _validate_fused_clean(self.executor.image)
        self._fused_clean_checked=True

    def release_device(self, states):
        self.store.release_device(states)

    def teacher_continue_device(self,states,viewers,*,horizon=3):
        if self._teacher is None:
            raise RuntimeError('Three-step teacher is absent from the SoA rule path')
        if self._teacher_pipeline is not None:
            return self._teacher_pipeline.teacher_continue_device(states,viewers,horizon=horizon)
        return self._teacher.teacher_continue_device(states,viewers,horizon=horizon)

    def rule_decision_device(self,states,observers):
        if self._teacher is None:
            raise RuntimeError('Rule opponent is absent from the SoA rule path')
        return self._teacher.rule_decision_device(states,observers)

    def close(self):
        if self._teacher_pipeline is not None:
            self._teacher_pipeline.close()
        if self._teacher is not None:
            self._teacher.close()
        self._keepalive.clear()
        self.binder.close();self.projection.close();self.compactor.close()

    @property
    def workspace_bytes(self):
        scratch=self._shared_scratch
        workspace=scratch._soa_workspace
        tensors=[*self._shared_batch.buffers.values(),self._shared_batch.roots,
            scratch.nodes,scratch.edges,scratch.payload,scratch.counts,scratch.roots,
            workspace.frames,workspace.marks,workspace.queue,*workspace.context,
            *self._compact_workspace.destination.buffers.values(),self._compact_workspace.destination.roots,
            self._compact_workspace.obj_map,self._compact_workspace.stack,self._compact_workspace.color,
            self._compact_workspace.strings,self._compact_workspace.status]
        return sum(value.numel()*value.element_size() for value in tensors)+getattr(self.store,'workspace_bytes',0)


def create_soa_search(rules,inference,auxiliary,statistics,keys,*,rows,random,
                      max_actions=800,zhenhong_prior=True,prior_weights=None,**options):
    if not isinstance(rules,DeviceSoARules):
        raise TypeError('Typed domain SoA rules required')
    reserved={'teacher_mode','terminal_horizon','opponent_mode','prior_guide',
              'simulations','gumbel_candidates','device','reuse_sampled_world','reuse_root_decision'}
    if reserved & set(options):
        raise ValueError('SoA search fixes Gumbel32/cap16/network-WDL/policy opponents')
    from .array_zhenhong_prior import ArrayZhenhongPrior
    from .device_search import DeviceGumbelSearch
    prior=ArrayZhenhongPrior(device=rules.device,weights=prior_weights) if zhenhong_prior else None
    # Both paths use public_legal_decision. Borrow the matching nonterminal
    # own-node observation; terminal/opponent inspections retain their queries.
    options.setdefault('reuse_interior_observation', True)
    options.setdefault('reuse_root_observation', True)
    return DeviceGumbelSearch(rules,inference,auxiliary,statistics,keys,rows=rows,
        max_actions=max_actions,simulations=32,gumbel_candidates=16,terminal_horizon=0,
        teacher_mode='network_value',opponent_mode='policy',prior_guide=prior,random=random,
        device=str(rules.device),reuse_sampled_world=True,reuse_root_decision=True,**options)
