"""Bounded CUDA graph-call pipeline for frozen rule entries.

This helper routes compact leased graphs through reusable VM scratch lanes.
It is not a RuleAdapter, search driver, or full-duel qualification receipt.
`supports_full_duel_rules` stays False until a real adapter and preflight exist.
"""
from .cuda_vm import (
    CudaVm,
    DeviceCall,
    DeviceGraph,
    VmCapabilityError,
)
from .device_graph_pool import INVALID_HANDLE
from .device_graph_storage import (
    MAX_EDGES,
    MAX_NODES,
    MAX_PAYLOAD,
    CompactDeviceGraphStore,
)
from .device_rule_arguments import DeviceRuleArgumentBinder
from .device_rule_lanes import DeviceRuleLanes
from .device_rule_protocol import DeviceDecision, DeviceStateBatch
from .schema import GraphLayout
from .vm_heap import ERROR_INDEX_OUT_OF_BOUNDS


ALLOWED_ENTRIES = frozenset(('clean', 'sample_world', 'simulate_action', 'apply_action'))
LARGE_NODES = 262144
LARGE_EDGES = 524288
LARGE_PAYLOAD_BYTES = 32 * 1024 * 1024


def _empty_graph(torch, layout, device):
    rows = layout.game_capacity
    return DeviceGraph(
        layout,
        torch.zeros((rows, layout.max_nodes, 4), dtype=torch.int32, device=device),
        torch.zeros((rows, layout.max_edges, 2), dtype=torch.int32, device=device),
        torch.zeros((rows, layout.max_payload_bytes), dtype=torch.uint8, device=device),
        torch.zeros((rows, 3), dtype=torch.int32, device=device),
        torch.zeros((rows,), dtype=torch.int32, device=device),
    )


class DeviceRuleCallPlan:
    """One frozen entry's captured WHILE. CUDA buffers may replay many times."""

    def __init__(self, helper, entry, lanes, output):
        self.helper = helper
        self.entry = entry
        self.lanes = lanes
        self.output = output
        self.captured = False
        self.inputs = None
        self.actors = None
        self.seeds = None
        self.decision = None
        self.action_indices = None

    def close(self):
        if self.lanes is not None:
            self.lanes.close()
            self.lanes = None

    def capture_device(self, inputs, *, actors=None, seeds=None, decision=None,
                       action_indices=None):
        """Embed one GPU cohort WHILE. Returns borrowed full-B output leases.

        Requires an active CUDA capture. Replay overwrites these output buffers;
        consume, copy, or release them before the next replay. Input leases stay
        immutable; the caller releases consumed inputs, including failed rows.
        """
        helper = self.helper
        if helper.closed or helper.vm.closed:
            raise RuntimeError('Rule graph-call helper closed')
        if self.lanes is None or self.lanes.closed:
            raise RuntimeError('Rule call plan closed')
        if self.captured:
            raise RuntimeError('Operation plan already captured')
        t = helper.t
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Rule graph calls must be embedded in CUDA capture')
        helper._validate_inputs(self.entry, inputs, actors=actors, seeds=seeds,
                                decision=decision, action_indices=action_indices)
        self.inputs = inputs
        self.actors = actors
        self.seeds = seeds
        self.decision = decision
        self.action_indices = action_indices
        helper._reset_output(self)

        def operation(local, ids, valid):
            helper._execute(self, local, ids, valid)

        self.lanes.capture_dispatch(inputs, operation)
        self.captured = True
        return self.output


class DeviceRuleGraphCalls:
    """CUDA-only frozen clean/sample/step graph calls. Not a full adapter."""

    supports_full_duel_rules = False

    def __init__(self, vm, state_store, *, rows, lanes=32, seconds=60, compact_layout):
        if not isinstance(vm, CudaVm) or vm.closed:
            raise TypeError('Live frozen CUDA VM required')
        if not isinstance(state_store, CompactDeviceGraphStore) or state_store.closed:
            raise TypeError('Live compact device graph store required')
        if vm.device.type != 'cuda' or state_store.device.type != 'cuda':
            raise RuntimeError('Actual CUDA required; no CPU rule-call fallback')
        if vm.device.index != state_store.device.index:
            raise ValueError('VM and graph store must share the CUDA device')
        if not isinstance(compact_layout, GraphLayout):
            raise TypeError('compact_layout GraphLayout required')
        if (compact_layout.max_nodes > MAX_NODES or compact_layout.max_edges > MAX_EDGES
                or compact_layout.max_payload_bytes > MAX_PAYLOAD):
            raise ValueError('compact_layout exceeds compact store source maxima')
        if any(type(value) is not int or not 0 < value < 2 ** 31 for value in (rows, lanes)):
            raise ValueError('Positive bounded root and scratch lane counts required')
        t = vm.torch
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Allocate rule graph-call scratch before CUDA capture')
        self.vm = vm
        self.state_store = state_store
        self.t = t
        self.device = vm.device
        self.rows = rows
        self.lanes = min(rows, lanes)
        self.seconds = seconds
        self.closed = False
        self._plans = []
        self._lanes = None
        lane_layout = GraphLayout(
            self.lanes, compact_layout.max_nodes, compact_layout.max_edges,
            compact_layout.max_payload_bytes)
        large_layout = GraphLayout(self.lanes, LARGE_NODES, LARGE_EDGES, LARGE_PAYLOAD_BYTES)
        self.binder = DeviceRuleArgumentBinder(device=self.device)
        try:
            self.large = _empty_graph(t, large_layout, self.device)
            self.input_graph = _empty_graph(t, lane_layout, self.device)
            self.output_graph = _empty_graph(t, lane_layout, self.device)
            self.action_graph = _empty_graph(t, lane_layout, self.device)
            self.selected_action = _empty_graph(t, lane_layout, self.device)
            self.workspace = vm.prepare_device_workspace(self.large)
            self.output_compact = vm.prepare_compaction_workspace(self.large, self.output_graph)
            self.action_compact = vm.prepare_compaction_workspace(
                self.action_graph, self.selected_action)
        except BaseException:
            self.binder.close()
            raise
        self.live_active = t.zeros(self.lanes, dtype=t.bool, device=self.device)
        self.lane_active = t.zeros(self.lanes, dtype=t.uint8, device=self.device)
        self.lane_errors = t.zeros(self.lanes, dtype=t.int32, device=self.device)
        self.lane_owners = t.zeros(self.lanes, dtype=t.int64, device=self.device)
        self.lane_actors = t.zeros(self.lanes, dtype=t.int64, device=self.device)
        self.lane_seeds = t.zeros((self.lanes, 5), dtype=t.int64, device=self.device)
        self.refs_clean = t.empty((self.lanes, 1), dtype=t.int32, device=self.device)
        self.refs_bound = t.empty((self.lanes, 3), dtype=t.int32, device=self.device)
        self.load_errors = t.empty(self.lanes, dtype=t.int32, device=self.device)
        self.bind_errors = t.empty(self.lanes, dtype=t.int32, device=self.device)
        self.vm_status = t.empty(self.lanes, dtype=t.int32, device=self.device)
        self.deadlines = t.zeros(self.lanes, dtype=t.int64, device=self.device)
        self.selected_refs = t.empty(self.lanes, dtype=t.int32, device=self.device)
        self.action_statuses = t.zeros((self.lanes, 4), dtype=t.int32, device=self.device)
        self.out_slots = t.empty(self.lanes, dtype=t.int64, device=self.device)
        self.out_gens = t.empty(self.lanes, dtype=t.int64, device=self.device)
        self.out_active = t.empty(self.lanes, dtype=t.bool, device=self.device)
        self.action_call = DeviceCall(
            vm, self.action_graph, self.selected_refs, self.action_statuses,
            self.workspace.frames, None)

    def close(self):
        if self.closed:
            return
        for plan in self._plans:
            plan.close()
        self._plans.clear()
        self.binder.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def release_device(self, states):
        """Checked store release, including failed rows with valid input leases."""
        if self.closed:
            raise RuntimeError('Rule graph-call helper closed')
        self.state_store.release_device(states)

    def prepare_call(self, entry):
        """Allocate one operation's dispatcher and full-B output leases."""
        if self.closed or self.vm.closed or self.state_store.closed:
            raise RuntimeError('Rule graph-call helper closed')
        if self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare rule call plans before CUDA capture')
        if type(entry) is not str or entry not in ALLOWED_ENTRIES:
            raise ValueError('Unsupported rule graph-call entry')
        if entry not in self.vm.image.entries:
            raise VmCapabilityError('Unsupported frozen device entry')
        if not self.vm.capability(entry)['ready']:
            raise VmCapabilityError('Unsupported frozen device entry')
        lanes = DeviceRuleLanes(
            self.rows, self.lanes, device=self.device, seconds=self.seconds)
        t = self.t
        try:
            output = DeviceStateBatch(
                self.state_store,
                t.full((self.rows,), -1, dtype=t.int64, device=self.device),
                t.zeros(self.rows, dtype=t.int64, device=self.device),
                t.zeros(self.rows, dtype=t.int64, device=self.device),
                t.zeros(self.rows, dtype=t.bool, device=self.device),
                t.zeros(self.rows, dtype=t.int32, device=self.device))
        except BaseException:
            lanes.close()
            raise
        plan = DeviceRuleCallPlan(self, entry, lanes, output)
        self._plans.append(plan)
        return plan

    def _exact(self, value, dtype, shape, name):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact %s tensor required' % name)

    def _validate_states(self, states, rows, *, pool=None, name='states'):
        if not isinstance(states, DeviceStateBatch):
            raise TypeError('DeviceStateBatch required')
        if pool is not None and states.pool is not pool:
            raise ValueError('DeviceStateBatch does not belong to this store')
        t = self.t
        self._exact(states.slots, t.int64, (rows,), name + ' slots')
        self._exact(states.generations, t.int64, (rows,), name + ' generations')
        self._exact(states.owners, t.int64, (rows,), name + ' owners')
        self._exact(states.errors, t.int32, (rows,), name + ' errors')
        if (not isinstance(states.active, t.Tensor) or states.active.device != self.device
                or states.active.dtype not in (t.bool, t.uint8)
                or tuple(states.active.shape) != (rows,) or not states.active.is_contiguous()):
            raise ValueError('Exact %s active mask required' % name)

    def _validate_inputs(self, entry, inputs, *, actors, seeds, decision, action_indices):
        self._validate_states(inputs, self.rows, pool=self.state_store, name='input')
        t = self.t
        if entry == 'clean':
            if actors is not None or seeds is not None or decision is not None or action_indices is not None:
                raise ValueError('clean takes no actor, seed, or action arguments')
            return
        self._exact(actors, t.int64, (self.rows,), 'actors')
        if entry == 'sample_world':
            self._exact(seeds, t.int64, (self.rows, 5), 'seeds')
            if decision is not None or action_indices is not None:
                raise ValueError('sample_world does not take a decision or action index')
            return
        if seeds is not None:
            raise ValueError('step calls do not take sampler seeds')
        if not isinstance(decision, DeviceDecision):
            raise TypeError('DeviceDecision required')
        self._validate_states(decision.states, self.rows, pool=self.state_store, name='decision')
        if not isinstance(decision.action_pool, CompactDeviceGraphStore) or decision.action_pool.closed:
            raise TypeError('Live compact action graph store required')
        if (decision.action_pool.device.type != 'cuda'
                or decision.action_pool.device.index != self.device.index):
            raise ValueError('Action pool must share the CUDA device')
        self._exact(decision.action_slots, t.int64, (self.rows,), 'action slots')
        self._exact(decision.action_generations, t.int64, (self.rows,), 'action generations')
        self._exact(decision.counts, t.int64, (self.rows,), 'action counts')
        self._exact(decision.errors, t.int32, (self.rows,), 'decision errors')
        refs = decision.action_refs
        if (not isinstance(refs, t.Tensor) or refs.device != self.device or refs.dtype != t.int32
                or refs.ndim != 2 or refs.shape[0] != self.rows or refs.shape[1] < 1
                or not refs.is_contiguous()):
            raise ValueError('Exact action ref matrix required')
        self._exact(decision.valid, t.bool, tuple(refs.shape), 'action valid mask')
        self._exact(action_indices, t.int64, (self.rows,), 'action indices')

    def _reset_output(self, plan):
        plan.output.slots.fill_(-1)
        plan.output.generations.zero_()
        plan.output.owners.zero_()
        plan.output.active.zero_()
        plan.output.errors.zero_()

    def _refresh_active(self):
        self.lane_active.copy_((self.live_active & (self.lane_errors == 0)).view(self.t.uint8))

    def _or_errors(self, extra):
        self.lane_errors.bitwise_or_(extra)
        self._refresh_active()

    def _pick(self, source, fill=0):
        ids = self._lanes.ids
        valid = self._lanes.valid.bool()
        if source.ndim == 1:
            return self.t.where(valid, source[ids], fill)
        return self.t.where(valid[:, None], source[ids], fill)

    def _execute(self, plan, local, ids, valid):
        self._lanes = plan.lanes
        self.live_active.copy_(local.active.bool())
        self.lane_owners.copy_(local.owners)
        self.lane_errors.copy_(local.errors)
        self._refresh_active()
        gathered = self.state_store.gather_into_device(local, self.input_graph)
        self._or_errors(gathered.errors)
        self.vm.load_graph_into_device(
            self.workspace, self.input_graph, active=self.lane_active, errors=self.load_errors)
        self._or_errors(self.load_errors)
        if plan.entry == 'clean':
            self.refs_clean[:, 0].copy_(self.large.roots)
            argument_refs = self.refs_clean
        elif plan.entry == 'sample_world':
            self.lane_actors.copy_(self._pick(plan.actors))
            self.lane_seeds.copy_(self._pick(plan.seeds))
            self.binder.bind_into_device(
                self.large, self.lane_actors, self.lane_active,
                argument_refs=self.refs_bound, errors=self.bind_errors, seeds=self.lane_seeds)
            self._or_errors(self.bind_errors)
            argument_refs = self.refs_bound
        else:
            argument_refs = self._bind_step(plan, local)
        call = self.vm.launch_into_device(
            self.workspace, plan.entry, argument_refs=argument_refs,
            active=self.lane_active, gpu_deadlines=self.deadlines)
        self.vm_status.copy_(call.statuses[:, 0])
        self._or_errors(self.vm_status)
        self.vm.compact_into_device(call, self.output_compact, active=self.lane_active)
        self._or_errors(self.output_compact.errors)
        put = self.state_store.put_device(self.output_graph, self.lane_owners, self.lane_active)
        self._or_errors(put.errors)
        self.out_slots.copy_(put.slots)
        self.out_gens.copy_(put.generations)
        self.out_active.copy_((put.slots >= 0) & (self.lane_errors == 0))
        plan.lanes.scatter_states_device(
            DeviceStateBatch(
                self.state_store, self.out_slots, self.out_gens, self.lane_owners,
                self.out_active, self.lane_errors),
            plan.output)

    def _bind_step(self, plan, local):
        t = self.t
        decision = plan.decision
        self._or_errors(self._pick(decision.errors))
        mismatch = self.lane_active.bool() & (
            (self._pick(decision.states.slots, -1) != local.slots)
            | (self._pick(decision.states.generations) != local.generations)
            | (self._pick(decision.states.owners) != local.owners))
        self._or_errors(mismatch.to(t.int32) * INVALID_HANDLE)
        width = int(decision.action_refs.shape[1])
        indices = self._pick(plan.action_indices, -1)
        counts = self._pick(decision.counts)
        in_range = (indices >= 0) & (indices < counts) & (indices < width)
        count_fit = (counts >= 0) & (counts <= width)
        safe = indices.clamp(0, width - 1)
        picked = decision.valid[self._lanes.ids].gather(1, safe.unsqueeze(1)).squeeze(1)
        bad = self.lane_active.bool() & (~in_range | ~picked | ~count_fit)
        self._or_errors(bad.to(t.int32) * ERROR_INDEX_OUT_OF_BOUNDS)
        action_states = DeviceStateBatch(
            decision.action_pool,
            self._pick(decision.action_slots, -1),
            self._pick(decision.action_generations),
            self._pick(decision.states.owners),
            self.lane_active.bool(),
            self.lane_errors)
        action_gathered = decision.action_pool.gather_into_device(action_states, self.action_graph)
        self._or_errors(action_gathered.errors)
        selected = decision.action_refs[self._lanes.ids].gather(1, safe.unsqueeze(1).to(t.int64)).squeeze(1)
        self.selected_refs.copy_(t.where(self._lanes.valid.bool(), selected, 0))
        self.action_statuses.zero_()
        self.vm.compact_into_device(self.action_call, self.action_compact, active=self.lane_active)
        self._or_errors(self.action_compact.errors)
        self.lane_actors.copy_(self._pick(plan.actors))
        self.binder.bind_action_into_device(
            self.large, self.lane_actors, self.selected_action, self.lane_active,
            argument_refs=self.refs_bound, errors=self.bind_errors)
        self._or_errors(self.bind_errors)
        return self.refs_bound

