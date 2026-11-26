"""Collector sampler packet + mandatory exact GPU information validation.

No qualification is inherited. The canonical sampler entry remains untouched;
this adapter requires a separately frozen sample_world_transition_packet entry.
Raw packet capture is internal; public capture without the validator fails.
"""
from ..batched_duel.cuda_vm import DeviceCall, VmCapabilityError
from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls, DeviceRuleCallPlan
from ..batched_duel.device_rule_results import DeviceRuleResultReader
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .device_action_leases import release_actions_device

TRANSITION_ENTRY = 'sample_world_transition_packet'
VALIDATION_ERROR = 158


class _PacketPlan(DeviceRuleCallPlan):
    def capture_device(self, *args, **kwargs):
        raise RuntimeError('Sampler packet requires exact GPU information validator')

    def _capture_packet_device(self, *args, **kwargs):
        return super().capture_device(*args, **kwargs)


class DeviceTransitionSampleCalls(DeviceRuleGraphCalls):
    """Same leased graph-call machinery; only sampling consumes a packet."""
    supports_full_duel_rules = False

    def __init__(self, vm, *args, **kwargs):
        self._packet_reader = None
        if TRANSITION_ENTRY not in vm.image.entries or not vm.capability(TRANSITION_ENTRY)['ready']:
            raise VmCapabilityError('Explicit frozen sampler packet entry required')
        super().__init__(vm, *args, **kwargs)
        try:
            self._packet_reader = DeviceRuleResultReader(vm)
            self._packet_pair = self._packet_reader.prepare_pair_workspace(self.lanes)
            self._packet_flag = self._packet_reader.prepare_float_workspace(self.lanes)
        except BaseException:
            self.close()
            raise

    def close(self):
        reader = getattr(self, '_packet_reader', None)
        if reader is not None:
            reader.close()
            self._packet_reader = None
        super().close()

    def prepare_call(self, entry):
        base = super().prepare_call(entry)
        if entry != 'sample_world':
            return base
        plan = _PacketPlan(self, entry, base.lanes, base.output)
        plan.validation_required = self.t.zeros(self.rows+1, dtype=self.t.int64, device=self.device)
        self._plans[-1] = plan
        return plan

    def _reset_output(self, plan):
        super()._reset_output(plan)
        if isinstance(plan, _PacketPlan):
            plan.validation_required.zero_()

    def _execute(self, plan, local, ids, valid):
        if plan.entry != 'sample_world':
            return super()._execute(plan, local, ids, valid)
        if not isinstance(plan, _PacketPlan):
            raise RuntimeError('Guarded sampler packet plan required')
        t = self.t
        self._lanes = plan.lanes
        self.live_active.copy_(local.active.bool())
        self.lane_owners.copy_(local.owners)
        self.lane_errors.copy_(local.errors)
        self._refresh_active()
        gathered = self.state_store.gather_into_device(local, self.input_graph)
        self._or_errors(gathered.errors)
        self.vm.load_graph_into_device(self.workspace, self.input_graph,
                                      active=self.lane_active, errors=self.load_errors)
        self._or_errors(self.load_errors)
        self.lane_actors.copy_(self._pick(plan.actors))
        self.lane_seeds.copy_(self._pick(plan.seeds))
        self.binder.bind_into_device(self.large, self.lane_actors, self.lane_active,
                                    argument_refs=self.refs_bound, errors=self.bind_errors,
                                    seeds=self.lane_seeds)
        self._or_errors(self.bind_errors)
        call = self.vm.launch_into_device(self.workspace, TRANSITION_ENTRY,
            argument_refs=self.refs_bound, active=self.lane_active, gpu_deadlines=self.deadlines)
        self._or_errors(call.statuses[:,0])
        pair = self._packet_reader.pair_into_device(call, active=self.lane_active, output=self._packet_pair)
        self._or_errors(pair.errors)
        flag = self._packet_reader.float_into_device(call.arena, pair.second,
            input_errors=pair.errors, active=self.lane_active, output=self._packet_flag)
        self._or_errors(flag.errors)
        invalid_flag = (flag.values != 0) & (flag.values != 1)
        self._or_errors((self.lane_active.bool() & invalid_flag).to(t.int32)*VALIDATION_ERROR)
        target = t.where(valid.bool(), ids, self.rows).long()
        plan.validation_required.index_copy_(0,target,
            t.where(self.lane_active.bool(),flag.values.to(t.int64),0))
        state_call = DeviceCall(self.vm,call.arena,pair.first,call.statuses,call.frames,None)
        self.vm.compact_into_device(state_call,self.output_compact,active=self.lane_active)
        self._or_errors(self.output_compact.errors)
        put = self.state_store.put_device(self.output_graph,self.lane_owners,self.lane_active)
        self._or_errors(put.errors)
        self.out_slots.copy_(put.slots)
        self.out_gens.copy_(put.generations)
        self.out_active.copy_((put.slots>=0)&(self.lane_errors==0))
        plan.lanes.scatter_states_device(DeviceStateBatch(self.state_store,self.out_slots,
            self.out_gens,self.lane_owners,self.out_active,self.lane_errors),plan.output)


class DeviceValidatedSampler:
    """Internal packet becomes usable only after whole-row exact x/c checks."""
    supports_full_duel_rules = False

    def __init__(self, rules):
        if not isinstance(rules.calls,DeviceTransitionSampleCalls):
            raise TypeError('Explicit packet graph-call helper required')
        from .device_decision_equality import compare_decisions_device
        self.compare = compare_decisions_device
        self.rules = rules
        self.before = self.after = None
        self.closed = False
        try:
            self.before = rules.presentation.prepare_decision(rules.action_store)
            self.after = rules.presentation.prepare_decision(rules.action_store)
        except BaseException:
            self.close()
            raise
        self.captured = False

    def capture_device(self, states, viewers, seeds):
        if self.closed or self.captured:
            raise RuntimeError('Sampler site closed or already captured')
        r = self.rules
        r.calls._exact(seeds,r.t.int64,(r.rows,5),'sampler seeds')
        plan = r._consume('sample_world')
        if not isinstance(plan,_PacketPlan):
            raise TypeError('Guarded sampler packet plan required')
        output = plan._capture_packet_device(states,actors=viewers,seeds=seeds)
        required = states.active & output.active & (plan.validation_required[:r.rows]==1)
        def masked(value):
            return DeviceStateBatch(value.pool,value.slots,value.generations,value.owners,
                                    value.active & required,value.errors)
        before = self.before.capture_device(masked(states),viewers)
        after = self.after.capture_device(masked(output),viewers)
        errors = output.errors.clone() | states.errors
        errors |= self.compare(before,after,required)
        for decision in (before,after):
            errors |= release_actions_device(decision,decision.action_slots>=0,
                states.owners,protected=(states,output))
        # Downstream search reclaims active states. Rejected new worlds must be
        # reclaimed here before their activity is cleared, including failures.
        rejected = (output.slots>=0) & (errors!=0)
        release = DeviceStateBatch(output.pool,output.slots,output.generations,
            output.owners,rejected,r.t.zeros_like(errors))
        r.calls.release_device(release)
        errors |= release.errors
        freed = rejected & (release.errors==0)
        output.slots.masked_fill_(freed,-1)
        output.generations.masked_fill_(freed,0)
        self.captured = True
        return DeviceStateBatch(output.pool,output.slots,output.generations,output.owners,
                                output.active & (errors==0),errors)

    def close(self):
        if not self.closed:
            for plan in (self.before,self.after):
                if plan is not None:plan.close()
            self.closed = True
