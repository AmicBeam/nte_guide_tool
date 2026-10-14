"""Bounded GPU RAW physical replay. Not Gumbel, learning, or qualification.

One outer CUDA WHILE consumes recorded raw-action graphs through bounded
lanes, then the existing apply_action and clean graph-call plans. CPU action
decode and Python rules stay out of the hot path. Completion is
replay_sequence_complete, never a natural game_complete claim.
"""
from dataclasses import dataclass

from ..batched_duel.cuda_vm import DeviceGraph
from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
from ..batched_duel.device_rule_calls import DeviceRuleGraphCalls
from ..batched_duel.device_rule_lanes import DeviceRuleLanes
from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from ..batched_duel.schema import GraphLayout
from .cuda_conditional_graph import CudaConditionalWhile
from .device_action_leases import release_actions_device
from .device_initial_archive import DeviceInitialArchive
from .device_raw_action_tape import DeviceRawActionTape


INVALID_RECORD, OWNER_ERROR, ACTOR_ERROR, STEP_ERROR = 1, 2, 4, 8
SEED_ERROR, VERSION_ERROR, RULE_ERROR, LEASE_ERROR = 16, 32, 64, 128
GUARD_ERROR = 256
CAVEAT = (
    'Bounded RAW physical replay only. Completion is replay_sequence_complete '
    'after the exact declared counts are consumed with zero errors. This is '
    'not game_complete, Gumbel search, learning, or full-duel qualification.'
)


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


@dataclass(frozen=True)
class DeviceRawReplayResult:
    """Borrowed per-row physical replay outputs. complete is sequence-complete."""
    states: DeviceStateBatch
    steps: object
    errors: object
    complete: object
    metadata_checked: object


class DeviceRawReplay:
    """Replay recorded raw actions onto cloned physical states inside one WHILE."""

    supports_full_duel_rules = False

    def __init__(self, calls, raw_tape, initial_states, action_store, *,
                 max_steps, compact_layout, lanes, seconds=60,
                 metadata_provider=None, require_metadata=False):
        import torch
        if not isinstance(calls, DeviceRuleGraphCalls) or calls.closed:
            raise TypeError('Live DeviceRuleGraphCalls required')
        if not isinstance(raw_tape, DeviceRawActionTape) or raw_tape.closed:
            raise TypeError('Live DeviceRawActionTape required')
        if not isinstance(initial_states, DeviceStateBatch):
            raise TypeError('DeviceStateBatch initial originals required')
        if not isinstance(action_store, CompactDeviceGraphStore) or action_store.closed:
            raise TypeError('Live compact action graph store required')
        if not isinstance(compact_layout, GraphLayout):
            raise TypeError('Bounded compact GraphLayout required')
        if type(require_metadata) is not bool:
            raise TypeError('Explicit metadata qualification control required')
        if metadata_provider is not None and not callable(metadata_provider):
            raise TypeError('Captured GPU metadata provider required')
        if require_metadata and metadata_provider is None:
            raise ValueError('Actual metadata required for strict physical replay')
        if type(max_steps) is not int or not 0 < max_steps < 2 ** 31:
            raise ValueError('Positive bounded max_steps required')
        if type(lanes) is not int or not 0 < lanes < 2 ** 31:
            raise ValueError('Positive bounded lane count required')
        if isinstance(seconds, bool) or not 0 < seconds <= 86400:
            raise ValueError('Positive bounded time budget required')
        if not torch.cuda.is_available() or calls.device.type != 'cuda':
            raise RuntimeError('Actual CUDA required; no host raw-replay fallback')
        if (raw_tape.device != calls.device or action_store.device != calls.device
                or initial_states.pool.device != calls.device):
            raise ValueError('Tape, originals, action store, and calls must share CUDA')
        if raw_tape.rows != calls.rows:
            raise ValueError('Raw action tape rows must match the rule-call cohort')
        if initial_states.pool is calls.state_store or initial_states.pool is action_store:
            raise ValueError('Immutable originals must stay on a separate source store')
        if action_store is calls.state_store:
            raise ValueError('Action store must be a separate checked compact store')
        t = calls.t
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Allocate raw-replay scratch and plans before CUDA capture')
        self.t = t
        self.device = calls.device
        self.calls = calls
        self.raw_tape = raw_tape
        self.initial_states = initial_states
        self.action_store = action_store
        self.max_steps = max_steps
        self.seconds = seconds
        self.rows = calls.rows
        self.lanes = min(self.rows, lanes)
        self.compact_layout = compact_layout
        self.closed = False
        self.metadata_provider = metadata_provider
        self._captured = False
        self._capture_inputs = None
        calls._validate_states(initial_states, self.rows, name='initial')
        lane_layout = GraphLayout(
            self.lanes, compact_layout.max_nodes, compact_layout.max_edges,
            compact_layout.max_payload_bytes)
        kw = dict(device=self.device)
        self._archive = DeviceInitialArchive(
            initial_states.pool, calls.state_store, rows=self.rows,
            lanes=self.lanes, layout=compact_layout, seconds=seconds)
        self._restore = None
        self._outer = None
        try:
            self._restore = DeviceRuleLanes(
                self.rows, self.lanes, device=self.device, seconds=seconds)
            self._outer = CudaConditionalWhile(
                device=str(self.device), max_ticks=max_steps + 1, seconds=seconds)
            self._apply_plan = calls.prepare_call('apply_action')
            self._clean_plan = calls.prepare_call('clean')
            self._action_scratch = _empty_graph(t, lane_layout, self.device)
        except BaseException:
            if self._outer is not None:
                self._outer.close()
            if self._restore is not None:
                self._restore.close()
            self._archive.close()
            raise
        self._record_index_plan = t.zeros((max_steps, self.rows), dtype=t.int64, **kw)
        self._job_seeds = t.zeros(self.rows, dtype=t.int64, **kw)
        self._decision_counts = t.zeros(self.rows, dtype=t.int64, **kw)
        self._expected_versions = t.full((self.rows,), -1, dtype=t.int64, **kw)
        self._expected_actors = t.full((self.rows,), -1, dtype=t.int64, **kw)
        self._checked_steps = t.zeros(self.rows, dtype=t.int64, **kw)
        self._step = t.zeros((), dtype=t.int64, **kw)
        self._steps = t.zeros(self.rows, dtype=t.int64, **kw)
        self._errors = t.zeros(self.rows, dtype=t.int32, **kw)
        self._present = t.zeros(self.rows, dtype=t.bool, **kw)
        self._step_active = t.zeros(self.rows, dtype=t.bool, **kw)
        self._unfinished = t.zeros(self.rows, dtype=t.uint8, **kw)
        self._actors = t.zeros(self.rows, dtype=t.int64, **kw)
        self._action_slots = t.full((self.rows,), -1, dtype=t.int64, **kw)
        self._action_generations = t.zeros(self.rows, dtype=t.int64, **kw)
        self._action_refs = t.zeros((self.rows, 1), dtype=t.int32, **kw)
        self._action_types = t.zeros((self.rows, 1), dtype=t.int64, **kw)
        self._action_counts = t.zeros(self.rows, dtype=t.int64, **kw)
        self._action_valid = t.zeros((self.rows, 1), dtype=t.bool, **kw)
        self._action_indices = t.zeros(self.rows, dtype=t.int64, **kw)
        self._decision_errors = t.zeros(self.rows, dtype=t.int32, **kw)
        self._observations = t.zeros((self.rows, 1), dtype=t.float32, **kw)
        self._candidates = t.zeros((self.rows, 1, 1), dtype=t.float32, **kw)
        self.current = DeviceStateBatch(
            calls.state_store,
            t.full((self.rows,), -1, dtype=t.int64, **kw),
            t.zeros(self.rows, dtype=t.int64, **kw),
            t.zeros(self.rows, dtype=t.int64, **kw),
            t.zeros(self.rows, dtype=t.bool, **kw),
            self._errors)
        self._apply_inputs = DeviceStateBatch(
            calls.state_store, self.current.slots, self.current.generations,
            self.current.owners, self._step_active, self._errors)
        self._action_leases = DeviceStateBatch(
            action_store, self._action_slots, self._action_generations,
            self.current.owners, t.zeros(self.rows, dtype=t.bool, **kw),
            self._decision_errors)
        self.decision = DeviceDecision(
            self._apply_inputs, action_store, self._action_slots, self._action_generations,
            self._action_refs, self._action_types, self._action_counts, self._action_valid,
            self._observations, self._candidates, self._decision_errors)

    def close(self):
        if self.closed:
            return
        if self._outer is not None:
            self._outer.close()
        if self._restore is not None:
            self._restore.close()
        self._archive.close()
        self._capture_inputs = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _exact(self, value, dtype, shape, name):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact %s tensor required' % name)

    def _release_replay(self, states, mask):
        t = self.t
        if states.pool is self.initial_states.pool:
            return
        if states.pool is not self.calls.state_store:
            return
        request = DeviceStateBatch(
            states.pool, states.slots, states.generations, states.owners,
            mask & states.active.bool() & (states.slots >= 0), t.zeros_like(self._errors))
        self.calls.release_device(request)
        self._errors.bitwise_or_(request.errors)
        freed = request.active & (request.errors == 0)
        states.slots.masked_fill_(freed, -1)
        states.generations.masked_fill_(freed, 0)

    def _initialize(self, record_index_plan, job_seeds, decision_counts):
        t = self.t
        self._errors.zero_()
        # Borrowed output vectors can name the same lease as current. Null every
        # alias before one checked release, rather than releasing the lease twice.
        outputs = (self.current, self._apply_plan.output, self._clean_plan.output)
        for index, states in enumerate(outputs):
            slots, generations = states.slots.clone(), states.generations.clone()
            mask = (slots >= 0) & states.active.bool()
            for other in outputs[index + 1:]:
                alias = mask & (other.slots == slots) & (other.generations == generations)
                self._errors.bitwise_or_((alias & (other.owners != states.owners)).to(t.int32) * OWNER_ERROR)
                other.slots.masked_fill_(alias, -1)
                other.generations.masked_fill_(alias, 0)
            self._release_replay(states, mask)
        action_errors = release_actions_device(
            self.decision, self._action_slots >= 0, self.current.owners,
            protected=(self.current, self._apply_plan.output, self._clean_plan.output,
                       self.initial_states))
        self._errors.bitwise_or_(action_errors.to(t.int32) * LEASE_ERROR)
        cleanup_errors = self._errors.clone()
        self._record_index_plan.copy_(record_index_plan)
        self._job_seeds.copy_(job_seeds)
        self._decision_counts.copy_(decision_counts)
        self._expected_versions.fill_(-1)
        self._expected_actors.fill_(-1)
        self._checked_steps.zero_()
        self._step.zero_()
        self._steps.zero_()
        self._errors.zero_()
        self._actors.zero_()
        self._action_slots.fill_(-1)
        self._action_generations.zero_()
        self._action_refs.zero_()
        self._action_types.zero_()
        self._action_counts.zero_()
        self._action_valid.zero_()
        self._action_indices.zero_()
        self._decision_errors.zero_()
        self._observations.zero_()
        self._candidates.zero_()
        cloned = self._archive.capture_device(self.initial_states)
        self.current.slots.copy_(cloned.slots)
        self.current.generations.copy_(cloned.generations)
        self.current.owners.copy_(cloned.owners)
        self.current.active.copy_(cloned.active)
        self._present.copy_(self.initial_states.active.bool())
        self._errors.copy_(cloned.errors | self.initial_states.errors | cleanup_errors
                           | self.raw_tape.errors_device)
        self._errors.bitwise_or_((job_seeds != self.raw_tape._job_seeds).to(t.int32) * SEED_ERROR)
        self._errors.bitwise_or_(((decision_counts < 0) | (decision_counts > self.max_steps)).to(t.int32) * STEP_ERROR)
        self._errors.bitwise_or_((self._present & (decision_counts != self.raw_tape.steps_device)).to(t.int32) * STEP_ERROR)
        self._unfinished.copy_(
            (self._present & (self._decision_counts > 0) & (self._errors == 0)
             & (self.current.slots >= 0)).to(t.uint8))

    def _begin_step(self):
        t = self.t
        self._step_active.copy_(
            self._present & (self._steps < self._decision_counts) & (self._errors == 0)
            & (self.current.slots >= 0))
        self._action_slots.fill_(-1)
        self._action_generations.zero_()
        self._action_leases.active.zero_()
        self._decision_errors.zero_()
        self._action_refs.zero_()
        self._action_types.zero_()
        self._action_counts.zero_()
        self._action_valid.zero_()
        self._actors.zero_()
        self._observations.zero_()
        self._candidates.zero_()
        if self.metadata_provider is not None:
            meta = self.metadata_provider(self._apply_inputs)
            self._exact(meta.actors, t.int64, (self.rows,), 'metadata actors')
            self._exact(meta.versions, t.int64, (self.rows,), 'metadata versions')
            self._exact(meta.errors, t.int32, (self.rows,), 'metadata errors')
            bad = self._step_active & ((meta.errors != 0) | (meta.versions < 0)
                                      | (meta.actors < 0) | (meta.actors > 1))
            self._errors.bitwise_or_(bad.to(t.int32) * VERSION_ERROR)
            self._expected_versions.copy_(meta.versions)
            self._expected_actors.copy_(meta.actors)
            self._step_active.logical_and_(self._errors == 0)
            self._checked_steps.add_(self._step_active.long())

    def _restore_actions(self, local, ids, valid):
        t = self.t
        records = self._record_index_plan.index_select(0, self._step.reshape(1)).reshape(self.rows)
        lane_records = records.index_select(0, ids)
        take = local.active.bool() & (local.errors == 0)
        idx = lane_records.clamp(0, self.raw_tape.max_records)
        in_count = (lane_records >= 0) & (lane_records < self.raw_tape._record_count)
        rec_game = self.raw_tape._rec_game[idx]
        rec_owner = self.raw_tape._rec_owner[idx]
        rec_seed = self.raw_tape._rec_seed[idx]
        rec_actor = self.raw_tape._rec_actor[idx]
        rec_step = self.raw_tape._rec_step[idx]
        rec_version = self.raw_tape._rec_version[idx]
        job_seeds = self._job_seeds.index_select(0, ids)
        expected = self._expected_versions.index_select(0, ids)
        expected_actor = self._expected_actors.index_select(0, ids)
        flags = t.zeros_like(local.errors)
        flags = flags | ((take & ~in_count).to(t.int32) * INVALID_RECORD)
        flags = flags | ((take & in_count & (rec_game != ids)).to(t.int32) * INVALID_RECORD)
        flags = flags | ((take & in_count & (rec_owner != local.owners)).to(t.int32) * OWNER_ERROR)
        flags = flags | ((take & in_count & (rec_seed != job_seeds)).to(t.int32) * SEED_ERROR)
        flags = flags | ((take & in_count & (rec_step != self._step)).to(t.int32) * STEP_ERROR)
        flags = flags | ((take & in_count & (rec_actor != 0) & (rec_actor != 1)).to(t.int32) * ACTOR_ERROR)
        flags = flags | ((take & in_count & (rec_version < 0)).to(t.int32) * VERSION_ERROR)
        flags = flags | ((take & in_count & (expected >= 0) & (rec_version != expected)).to(t.int32) * VERSION_ERROR)
        flags = flags | ((take & in_count & (expected_actor >= 0) & (rec_actor != expected_actor)).to(t.int32) * ACTOR_ERROR)
        gather_take = take & (flags == 0)
        gather_errors = self.raw_tape.gather_into_device(
            lane_records, self._action_scratch, gather_take)
        flags = flags | gather_errors
        put_take = gather_take & (gather_errors == 0)
        put = self.action_store.put_device(self._action_scratch, local.owners, put_take)
        flags = flags | put.errors
        stored = put_take & (put.errors == 0)
        self._restore.scatter_states_device(
            DeviceStateBatch(
                self.action_store, put.slots, put.generations, local.owners, stored, flags),
            self._action_leases)
        self._restore.scatter_tensor_device(rec_actor * stored.long(), self._actors)
        self._restore.scatter_tensor_device(
            (self._action_scratch.roots * stored.to(t.int32)).unsqueeze(1), self._action_refs)

    def _finish_restore(self):
        t = self.t
        self._errors.bitwise_or_(self._decision_errors)
        self._step_active.logical_and_(
            self._action_leases.active & (self._decision_errors == 0))
        self._action_counts.copy_(self._action_leases.active.to(t.int64))
        self._action_valid.zero_()
        self._action_valid[:, 0].copy_(self._action_leases.active)

    def _commit(self, applied, cleaned):
        t = self.t
        old = self.current
        self._errors.bitwise_or_(applied.errors)
        self._errors.bitwise_or_(cleaned.errors)
        missing = self._step_active & (
            (applied.slots < 0) | ~applied.active | (cleaned.slots < 0) | ~cleaned.active)
        owner_bad = self._step_active & (
            (applied.owners != old.owners) | (cleaned.owners != old.owners))
        self._errors.bitwise_or_(missing.to(t.int32) * RULE_ERROR)
        self._errors.bitwise_or_(owner_bad.to(t.int32) * OWNER_ERROR)
        success = (self._step_active & (self._errors == 0) & cleaned.active
                   & (cleaned.slots >= 0) & (cleaned.generations >= 0)
                   & (applied.slots >= 0) & applied.active)
        distinct_old = (cleaned.slots != old.slots) | (cleaned.generations != old.generations)
        distinct_mid = ((cleaned.slots != applied.slots)
                        | (cleaned.generations != applied.generations))
        applied_distinct = ((applied.slots != old.slots)
                            | (applied.generations != old.generations))
        self._release_replay(
            cleaned, (~success) & distinct_old & distinct_mid & cleaned.active
            & (cleaned.owners == old.owners))
        self._release_replay(
            applied, applied_distinct & (~success | distinct_mid) & applied.active
            & (applied.owners == old.owners))
        self._release_replay(old, success & distinct_old)
        old.slots.copy_(t.where(success, cleaned.slots, old.slots))
        old.generations.copy_(t.where(success, cleaned.generations, old.generations))
        old.active.copy_(t.where(success, cleaned.active, old.active))
        self._steps.add_(success.long())
        action_errors = release_actions_device(
            self.decision, self._action_slots >= 0, old.owners,
            protected=(old, applied, cleaned, self.initial_states))
        self._errors.bitwise_or_(action_errors.to(t.int32) * LEASE_ERROR)
        self._step.add_(1)
        self._unfinished.copy_(
            (self._present & (self._steps < self._decision_counts) & (self._errors == 0)
             & (self.current.slots >= 0)).to(t.uint8))

    def _tick(self):
        self._begin_step()
        self._restore.capture_dispatch(self._apply_inputs, self._restore_actions)
        self._finish_restore()
        applied = self._apply_plan.capture_device(
            self._apply_inputs, actors=self._actors, decision=self.decision,
            action_indices=self._action_indices)
        cleaned = self._clean_plan.capture_device(applied)
        self._commit(applied, cleaned)

    def capture(self, record_index_plan, job_seeds, decision_counts):
        if self.closed:
            raise RuntimeError('Raw replay closed')
        if self._captured or self._outer.graph is not None or self._outer._loop_built:
            raise RuntimeError('Device raw replay is already frozen')
        t = self.t
        self._exact(record_index_plan, t.int64, (self.max_steps, self.rows), 'record_index_plan')
        self._exact(job_seeds, t.int64, (self.rows,), 'job_seeds')
        self._exact(decision_counts, t.int64, (self.rows,), 'decision_counts')
        # The graph's prologue copies from these external device allocations.
        # Keep them alive even when the caller passed temporary tensors; CUDA
        # graph capture does not retain their Python owners for us.
        self._capture_inputs = (record_index_plan, job_seeds, decision_counts)
        self._outer.capture(
            self._unfinished, self._tick,
            initialize=lambda: self._initialize(record_index_plan, job_seeds, decision_counts))
        self._captured = True

    def replay(self):
        if self.closed or self._outer.graph is None:
            raise RuntimeError('No frozen device RAW replay graph')
        self._outer.replay()

    def result_device(self):
        if not self._captured:
            raise RuntimeError('No device RAW replay cohort captured')
        t = self.t
        errors = self._errors
        failed = (self._outer.errors != 0).reshape(())
        errors = errors.clone()
        leftover = self._unfinished != 0
        errors.bitwise_or_((leftover & failed).to(t.int32) * GUARD_ERROR)
        complete = self._present & (self._steps == self._decision_counts) & (errors == 0)
        states = DeviceStateBatch(
            self.current.pool, self.current.slots, self.current.generations,
            self.current.owners, self.current.active, errors)
        metadata_checked = complete & (self._checked_steps == self._steps)
        if self.metadata_provider is None:
            metadata_checked.zero_()
        return DeviceRawReplayResult(states, self._steps, errors, complete, metadata_checked)

