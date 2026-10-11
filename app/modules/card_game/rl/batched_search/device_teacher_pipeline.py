"""Three canonical teacher steps with compact GPU states between VM calls.

This collector composition bounds transient allocations per call. It is not a
full-rules validation receipt; exact real-state oracle evidence is still needed.
"""
from ..batched_duel.cuda_vm import DeviceGraph
from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls
from ..batched_duel.device_rule_presentation import DeviceRulePresentation
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from ..batched_duel.schema import GraphLayout
from .device_action_leases import release_actions_device
from .device_frozen_rules import DeviceFrozenRules, _ActiveRuleLanes

LEASE_ERROR = 2048


class DeviceTeacherPipeline:
    """Capture exactly three ordinary teacher decisions and hypothetical steps."""
    supports_full_duel_rules = False
    host_rule_callbacks = False
    teacher_horizon = 3

    def __init__(self, calls, presentation, *, action_store):
        if not isinstance(calls, DeviceRuleGraphCalls) or calls.closed:
            raise TypeError('Live device rule calls required')
        if not isinstance(presentation, DeviceRulePresentation) or presentation.calls is not calls:
            raise TypeError('Presentation must use the same calls')
        t = calls.t
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare teacher pipeline before capture')
        self.calls, self.t, self.device = calls, t, calls.device
        self.rows, self.pool = calls.rows, calls.state_store
        self.closed, self.captured = False, False
        self.rules = None
        self.copy_lanes = None
        try:
            # These queues belong to this pipeline, not the surrounding search.
            self.rules = DeviceFrozenRules(calls, presentation, action_store=action_store,
                sites=dict(metadata=3, decision=1, rule_decision=3, value=1,
                           simulate_action=3, clean=3))
            self.copy_lanes = _ActiveRuleLanes(self.rows, calls.lanes,
                device=self.device, seconds=calls.seconds)
            small = calls.input_graph.layout
            shape = GraphLayout(calls.lanes, small.max_nodes, small.max_edges,
                                small.max_payload_bytes)
            kw = dict(device=self.device)
            self.copy_graph = DeviceGraph(shape,
                t.zeros((calls.lanes, shape.max_nodes, 4), dtype=t.int32, **kw),
                t.zeros((calls.lanes, shape.max_edges, 2), dtype=t.int32, **kw),
                t.zeros((calls.lanes, shape.max_payload_bytes), dtype=t.uint8, **kw),
                t.zeros((calls.lanes, 3), dtype=t.int32, **kw),
                t.zeros(calls.lanes, dtype=t.int32, **kw))
            def batch():
                return DeviceStateBatch(self.pool,
                    t.full((self.rows,), -1, dtype=t.int64, **kw),
                    t.zeros(self.rows, dtype=t.int64, **kw),
                    t.zeros(self.rows, dtype=t.int64, **kw),
                    t.zeros(self.rows, dtype=t.bool, **kw),
                    t.zeros(self.rows, dtype=t.int32, **kw))
            self.current, self.copied, self.work = batch(), batch(), batch()
            self.stage_errors = t.zeros((3, 6, self.rows), dtype=t.int32, **kw)
            self._owned = [self.current, self.copied]
            for name in ('simulate_action', 'clean'):
                self._owned.extend(p.output for p in self.rules._queues[name].plans)
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            if self.copy_lanes is not None:
                self.copy_lanes.close()
            if self.rules is not None:
                self.rules.close()
            self.closed = True

    def _release(self, states, mask, protected):
        """Release live owned leases once, nulling every internal alias."""
        t, pool = self.t, self.pool
        slots, gens, owners = states.slots.clone(), states.generations.clone(), states.owners.clone()
        in_range = (slots >= 0) & (slots < pool.slot_capacity)
        idx = slots.clamp(0, pool.slot_capacity - 1)
        live = in_range & (pool.live[idx] != 0) & (pool.generation[idx] == gens)
        owner_ok = pool.owner[idx] == owners
        protected_alias = (slots == protected.slots) & (gens == protected.generations) & (slots >= 0)
        bad = mask & (((slots >= 0) & ~in_range) | (live & (~owner_ok | (protected_alias & (owners != protected.owners)))))
        self.current.errors.bitwise_or_(bad.to(t.int32) * LEASE_ERROR)
        take = mask & live & owner_ok & ~protected_alias
        request = DeviceStateBatch(pool, slots, gens, owners, take.contiguous(), t.zeros_like(states.errors))
        pool.release_device(request)
        self.current.errors.bitwise_or_(request.errors.to(t.int32) * LEASE_ERROR)
        clear = mask & (~live | (take & (request.errors == 0)))
        for other in self._owned:
            alias = clear & (other.slots == slots) & (other.generations == gens)
            self.current.errors.bitwise_or_((alias & (other.owners != owners)).to(t.int32) * LEASE_ERROR)
            other.slots.masked_fill_(alias, -1)
            other.generations.masked_fill_(alias, 0)
            other.active.masked_fill_(alias, False)

    def _assign(self, target, source, mask=None):
        t = self.t
        for name in ('slots', 'generations', 'owners', 'active'):
            before, after = getattr(target, name), getattr(source, name)
            before.copy_(after if mask is None else t.where(mask, after, before))

    def capture_device(self, states, viewers, *, horizon=3):
        if self.closed:
            raise RuntimeError('Teacher pipeline closed')
        if self.captured:
            raise RuntimeError('Teacher pipeline capture site already frozen')
        if type(horizon) is not int or horizon != 3:
            raise ValueError('Teacher pipeline requires exactly horizon 3')
        self.calls._validate_states(states, self.rows, pool=self.pool)
        self.calls._exact(viewers, self.t.int64, (self.rows,), 'viewers')
        t = self.t
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Embed teacher pipeline in CUDA capture')
        self.current.errors.zero_()
        for previous in self._owned:
            self._release(previous, previous.slots >= 0, states)
        self.current.errors.bitwise_or_(states.errors)
        cleanup_errors = self.current.errors.clone()
        self.copied.slots.fill_(-1);self.copied.generations.zero_()
        self.copied.owners.copy_(states.owners);self.copied.active.zero_()
        self.copied.errors.copy_(states.errors | cleanup_errors)
        def copy(local, ids, valid):
            loaded = self.pool.gather_into_device(local, self.copy_graph)
            active = local.active & (local.errors == 0) & (loaded.errors == 0)
            saved = self.pool.put_device(self.copy_graph, local.owners, active.contiguous())
            saved.errors.bitwise_or_(local.errors | loaded.errors)
            self.copy_lanes.scatter_states_device(saved, self.copied)
        self.copy_lanes.capture_dispatch(states, copy)
        self._assign(self.current, self.copied)
        self.current.errors.copy_(self.copied.errors | cleanup_errors)
        for stage in range(3):
            meta = self.rules.metadata_device(self.current)
            self.stage_errors[stage, 0].copy_(meta.errors)
            self.current.errors.bitwise_or_(meta.errors)
            advance = self.current.active & (self.current.errors == 0) & ~meta.finished.bool()
            self._assign(self.work, self.current)
            self.work.active.copy_(advance)
            self.work.errors.copy_(self.current.errors)
            actor = meta.actors.clamp(0, 1).contiguous()
            decision, chosen, decision_errors = self.rules.rule_decision_device(self.work, actor)
            self.stage_errors[stage, 1].copy_(decision_errors)
            self.current.errors.bitwise_or_(decision_errors)
            self.work.errors.copy_(self.current.errors)
            stepped = self.rules.step_device(self.work, actor, decision, chosen, physical=False)
            self.stage_errors[stage, 2].copy_(stepped.errors)
            self.current.errors.bitwise_or_(stepped.errors)
            cleaned = self.rules.clean_device(stepped)
            self.stage_errors[stage, 3].copy_(cleaned.errors)
            self.current.errors.bitwise_or_(cleaned.errors)
            action_errors = release_actions_device(decision, self.work.active,
                self.work.owners, protected=(states, self.current, stepped, cleaned))
            self.stage_errors[stage, 4].copy_(action_errors)
            self.current.errors.bitwise_or_(action_errors.to(t.int32) * LEASE_ERROR)
            keep = advance & (self.current.errors == 0) & cleaned.active
            self._release(self.current, keep, states)
            self._assign(self.current, cleaned, keep)
            self._release(stepped, stepped.slots >= 0, states)
            # Failed outputs are not retained as apparent successes.
            self._release(cleaned, (cleaned.slots >= 0) & ~keep, states)
            self.stage_errors[stage, 5].copy_(self.current.errors)
        self.current.active.copy_(states.active)
        self.captured = True
        return self.current
