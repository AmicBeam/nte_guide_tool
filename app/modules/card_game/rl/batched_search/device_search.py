"""Tensor state machine for captured, device-controlled Gumbel search.

This driver has no Python per-root coroutines or scalar device reads. It needs
a graph-capable formal-rule adapter, exact random stream, device information
keys and qualified auxiliary value inference. It is not a rule executor and
does not claim complete game support merely by constructing the FSM.

The production execution wrapper must put initialize/tick inside a device
conditional CUDA graph. CPU and synthetic execution are explicit test modes.
Covered-value retains the historical teacher contract. network_value performs
no leaf continuation and uses the explicitly bound fixed-observer WDL model.
Serving public-proof budgeting is a separate layer.
"""
from dataclasses import dataclass, fields
from typing import Any

from ..batched_duel.device_rule_protocol import DeviceStateBatch, DeviceDecision, DeviceValueObservation


SAMPLE, INSPECT, EVALUATE_NODE, LEAF, OPPONENT, STEP, BACKUP, FINAL, CLEANUP, DONE = range(10)
RULE_ERROR, INFERENCE_ERROR, STATISTICS_ERROR, KEY_ERROR = 1, 2, 4, 8
CAPACITY_ERROR, RANDOM_ERROR, METADATA_ERROR = 16, 32, 64
PRIOR_ERROR = 128


@dataclass(frozen=True)
class DeviceGumbelResult:
    decision: DeviceDecision
    actions: Any
    policy: Any
    visit_policy: Any
    visits: Any
    requested: Any
    completed: Any
    complete: Any
    errors: Any
    node_count: Any
    phase_work: Any
    peak_device_roots: Any
    information_keys: Any = None
    prior_shifts: Any = None
    prior_version: str | None = None


def add_world_seed_stride(limbs, simulation):
    """Unsigned 320-bit little-endian addition, preserving signed int64 ABI."""
    import torch
    if (not isinstance(limbs, torch.Tensor) or limbs.dtype != torch.int64
            or limbs.ndim != 2 or limbs.shape[1] != 5 or type(simulation) is not int
            or simulation < 0 or simulation * 104729 >= 2**32):
        raise ValueError('int64[B,5] seed limbs and bounded static simulation required')
    carry = torch.full_like(limbs[:, 0], simulation * 104729)
    result = []
    for word in range(5):
        low = (limbs[:, word] & 0xffffffff) + carry
        high = ((limbs[:, word] >> 32) & 0xffffffff) + (low >> 32)
        result.append((low & 0xffffffff) | ((high & 0xffffffff) << 32))
        carry = high >> 32
    return torch.stack(result, -1)


class DeviceGumbelSearch:
    """One reusable search cohort; tick counts and phases are device tensors.

    Call ``initialize`` in the graph prologue, ``tick`` in its device while
    body, and use ``unfinished_device`` as the device loop predicate. Calling
    ticks eagerly on CUDA is refused except explicitly synthetic tests.
    """
    supports_full_duel_rules = False

    def __init__(self, rules, inference, auxiliary_inference, statistics, information_keys,
                 *, rows, max_actions, simulations=32, gumbel_candidates=16,
                 max_depth=10, terminal_horizon=3, teacher_mode='covered_value',
                 random=None, noise=False, device='cuda', allow_cpu_reference=False,
                 allow_synthetic=False, value_batch_rows=32,queued_row_work=False,row_work_seconds=120,borrow_root_candidates=False,
                 reuse_root_policy_value=False, verify_root_value_reuse=False,
                 group_ready_leaves=False,reuse_interior_observation=False,reuse_root_observation=False,opponent_mode='rule',
                 prior_guide=None, reuse_sampled_world=False, reuse_root_decision=False):
        import torch
        from ..search_policy import visit_schedule
        for value in (rows, max_actions, simulations, gumbel_candidates, max_depth):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer bounded device search dimensions required')
        if type(terminal_horizon) is not int or terminal_horizon < 0:
            raise ValueError('Bounded terminal horizon required')
        if type(value_batch_rows) is not int or value_batch_rows < 1:
            raise ValueError('Positive bounded value inference row count required')
        if teacher_mode not in ('covered_value', 'network_value'):
            raise NotImplementedError('Unknown device leaf evaluation mode')
        if teacher_mode == 'network_value' and terminal_horizon != 0:
            raise ValueError('network_value requires zero teacher horizon')
        if teacher_mode != 'network_value' and not getattr(rules, 'require_teacher', True):
            raise ValueError('Teacher-free rule adapter requires network_value')
        self.teacher_mode = teacher_mode
        if type(reuse_sampled_world) is not bool:
            raise TypeError('Explicit decision-world reuse option required')
        self.reuse_sampled_world = reuse_sampled_world
        if type(reuse_root_decision) is not bool or (reuse_root_decision and not reuse_sampled_world):
            raise ValueError('Root decision reuse requires a decision-scoped checked particle')
        self.reuse_root_decision = reuse_root_decision
        if opponent_mode not in ('rule', 'policy'):
            raise ValueError('Unknown device opponent mode')
        if opponent_mode == 'rule' and not getattr(rules, 'require_teacher', True):
            raise ValueError('Teacher-free rule adapter requires policy opponents')
        if opponent_mode == 'policy' and not callable(getattr(random, 'categorical', None)):
            raise ValueError('Exact device categorical RNG required for policy opponents')
        self.opponent_mode = opponent_mode
        if prior_guide is not None and not callable(getattr(prior_guide, 'score_device', None)):
            raise TypeError('Array prior must provide score_device')
        if prior_guide is not None and teacher_mode != 'network_value':
            raise ValueError('Array-prior branch requires network_value with zero teacher horizon')
        if type(noise) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit boolean noise and synthetic controls required')
        if type(borrow_root_candidates) is not bool:raise TypeError('Explicit root candidate borrow option required')
        self.borrow_root_candidates=borrow_root_candidates
        if borrow_root_candidates and not callable(getattr(rules,'borrow_root_candidates_device',None)):
            raise ValueError('Dedicated immutable root candidate binding required')
        if type(queued_row_work) is not bool:raise TypeError('Explicit queued row-work option required')
        if type(reuse_root_policy_value) is not bool or type(verify_root_value_reuse) is not bool:
            raise TypeError('Explicit root value reuse and verification options required')
        if verify_root_value_reuse and not reuse_root_policy_value:
            raise ValueError('Root value verification requires opt-in reuse')
        self.verify_root_value_reuse = verify_root_value_reuse
        if type(group_ready_leaves) is not bool:
            raise TypeError('Explicit ready-leaf grouping option required')
        self.group_ready_leaves=group_ready_leaves
        if type(reuse_interior_observation) is not bool:
            raise TypeError('Explicit interior observation reuse option required')
        self.reuse_interior_observation=reuse_interior_observation
        if type(reuse_root_observation) is not bool:
            raise TypeError('Explicit root observation reuse option required')
        self.reuse_root_observation=reuse_root_observation
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit FSM oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU search fallback')
        if self.device.type == 'cpu' and not allow_synthetic:
            raise ValueError('CPU FSM oracle requires explicit synthetic fixture binding')
        if prior_guide is not None and getattr(prior_guide, 'device', None) != self.device:
            raise ValueError('Array prior and search must share a device')
        self.prior_guide = prior_guide
        if noise and (random is None or not callable(getattr(random, 'gumbel_device', None))):
            raise ValueError('Exact device Gumbel random stream required for exploration')
        if (statistics.device != self.device or statistics.q_scale != 'natural_wdl'
                or statistics.capacity < rows * (simulations + 1)
                or statistics.max_actions != max_actions or statistics.max_batch < rows
                or statistics.max_path_len < max_depth):
            raise ValueError('Exact bounded device statistics layout required')
        for name in ('metadata_device', 'decision_device', 'sample_world_device', 'step_device',
                     'clean_device', 'value_observation_device', 'teacher_continue_device',
                     'rule_decision_device', 'release_device'):
            if not callable(getattr(rules, name, None)):
                raise ValueError('Missing device rule interface: ' + name)
        if auxiliary_inference is None:
            raise ValueError('Qualified independent auxiliary WDL inference required')
        self.rules, self.inference, self.auxiliary = rules, inference, auxiliary_inference
        self.root_value_reuse = None
        if reuse_root_policy_value:
            from .device_root_value_reuse import DeviceRootValueReuse
            self.root_value_reuse = DeviceRootValueReuse(rules, inference, auxiliary_inference)
        self.statistics, self.information_keys = statistics, information_keys
        self.random, self.noise, self.allow_synthetic = random, noise, allow_synthetic
        self.rows, self.max_actions = rows, max_actions
        self.value_batch_rows = min(rows, value_batch_rows)
        self.simulations, self.max_depth, self.terminal_horizon = simulations, max_depth, terminal_horizon
        self.nodes_per_root = simulations + 1
        z = dict(device=self.device)
        self._rows = torch.arange(rows, dtype=torch.int64, **z)
        self._base_slots = self._rows * self.nodes_per_root
        self._columns = torch.arange(self.nodes_per_root, dtype=torch.int64, **z)
        self._schedule = torch.as_tensor([visit_schedule(k, simulations)
            for k in range(1, min(gumbel_candidates, simulations) + 1)], dtype=torch.int64, **z)
        self._candidate_cap = min(gumbel_candidates, simulations)
        self.phase = torch.full((rows,), DONE, dtype=torch.int64, **z)
        self.errors = torch.zeros(rows, dtype=torch.int32, **z)
        self.requested = torch.zeros(rows, dtype=torch.int64, **z)
        self.completed = torch.zeros(rows, dtype=torch.int64, **z)
        self.depth = torch.zeros(rows, dtype=torch.int64, **z)
        self.node_count = torch.zeros(rows, dtype=torch.int64, **z)
        self.keys = torch.zeros((rows, self.nodes_per_root, 8), dtype=torch.int64, **z)
        self.node_generations = torch.ones((rows, self.nodes_per_root), dtype=torch.int64, **z)
        self.node_slots = self._base_slots[:, None] + self._columns[None, :]
        self.path_slots = torch.zeros((rows, max_depth), dtype=torch.int64, **z)
        self.path_generations = torch.ones_like(self.path_slots)
        self.path_actions = torch.zeros_like(self.path_slots)
        self.path_valid = torch.zeros((rows, max_depth), dtype=torch.bool, **z)
        self.actions = torch.zeros(rows, dtype=torch.int64, **z)
        self.leaf_values = torch.zeros(rows, dtype=torch.float64, **z)
        self.policy = torch.zeros((rows, max_actions), dtype=torch.float32, **z)
        self.visit_policy = torch.zeros_like(self.policy)
        self.visits = torch.zeros((rows, max_actions), dtype=torch.int32, **z)
        self.root_prior_shifts = None if prior_guide is None else torch.zeros_like(self.policy)
        self.phase_work = torch.zeros((DONE, rows), dtype=torch.int64, **z)
        self.peak_device_roots = torch.zeros((), dtype=torch.int64, **z)
        self.root_value_reuse_rows = torch.zeros((), dtype=torch.int64, **z)
        self.root_value_reuse_validation_errors = torch.zeros(rows, dtype=torch.int32, **z)
        self._unfinished = torch.zeros(rows, dtype=torch.uint8, **z)
        self._world_live = torch.zeros(rows, dtype=torch.bool, **z)
        self._sampled_live = torch.zeros_like(self._world_live)
        self.world_samples = torch.zeros(rows, dtype=torch.int64, **z)
        self._pool = None
        self._root_decision = self._decision = None
        self.row_work = None
        if queued_row_work:
            from .device_search_row_work import DeviceSearchRowWork
            self.row_work=DeviceSearchRowWork(inference,auxiliary_inference,information_keys,
                rows=rows,max_actions=max_actions,tile=self.value_batch_rows,seconds=row_work_seconds,
                policy_sites=3 if opponent_mode=='policy' else 2)

    def close(self):
        if self.row_work is not None:self.row_work.close()

    def _build_context(self):
        if (self.device.type == 'cuda' and not self.allow_synthetic
                and not self.t.cuda.is_current_stream_capturing()):
            raise RuntimeError('Formal device FSM must be inside a captured/device-controlled graph')

    def _vector(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.shape != shape):
            raise ValueError('Exact bound-device search tensor metadata required')

    def _masked(self, states, mask):
        return DeviceStateBatch(states.pool, states.slots, states.generations, states.owners,
                                mask & states.active & (states.errors == 0), states.errors)

    def _world(self, mask):
        return DeviceStateBatch(self._pool, self._world_slots, self._world_generations,
                                self._world_owners, mask, self.errors)

    def _error(self, raw, mask, code):
        self.errors.bitwise_or_(((raw != 0) & mask).to(self.t.int32) * code)

    def _replace_world(self, states, mask):
        if states.pool is not self._pool:
            raise ValueError('Device worlds must share a stable checked state arena')
        physical = self._is_physical_lease(states)
        self._error(physical | (states.owners != self._roots.owners), mask, METADATA_ERROR)
        self._world_live.copy_(self.t.where(mask, states.active & ~physical, self._world_live))
        for target, value in ((self._world_slots, states.slots),
                              (self._world_generations, states.generations),
                              (self._world_owners, states.owners)):
            target.copy_(self.t.where(mask, value, target))
        self._error(states.errors, mask, RULE_ERROR)

    def _release_replaced(self, old, new, mask):
        changed = (old.slots != new.slots) | (old.generations != new.generations)
        self._release_lease(old, mask & changed)

    def _release_lease(self, states, mask):
        # A search execution error is not a failed ownership check. Keep it in
        # self.errors while validating/reclaiming the separately issued lease.
        physical = self._is_physical_lease(states)
        self._error(physical, mask & states.active, METADATA_ERROR)
        shared = (self._is_sampled_lease(states) if self.reuse_sampled_world
                  else self.t.zeros_like(states.active))
        request = DeviceStateBatch(states.pool, states.slots, states.generations,
            states.owners, mask & states.active & ~physical & ~shared,
            self.t.zeros_like(self.errors))
        self.rules.release_device(request)
        self._error(request.errors, mask, RULE_ERROR)

    def _sampled_world(self, mask):
        return DeviceStateBatch(self._pool, self._sampled_slots, self._sampled_generations,
            self._sampled_owners, mask & self._sampled_live, self.t.zeros_like(self.errors))

    def _is_sampled_lease(self, states):
        return (self._sampled_live & (states.slots == self._sampled_slots)
                & (states.generations == self._sampled_generations)
                & (states.owners == self._sampled_owners))

    def _is_physical_lease(self, states):
        t = self.t
        at = t.searchsorted(self._sorted_root_slots, states.slots).clamp(0, self.rows-1)
        root = self._root_order[at]
        return ((states.slots == self._roots.slots[root])
                & (states.generations == self._roots.generations[root]))

    def _root_observation(self, states, viewers, decision, meta, mask):
        """Borrow only this initialization's checked own-view decision encoding.

        This saves a rule query, never auxiliary inference or random draws.
        Terminal and other-viewer rows retain the canonical observation path.
        Stale/foreign decisions are execution errors, not cache misses.
        """
        t=self.t
        saved=decision.states
        if saved.pool is not states.pool:
            raise ValueError('Root decision observation must use the same state arena')
        stale=((saved.slots!=states.slots)|(saved.generations!=states.generations)
               |(saved.owners!=states.owners))
        self._error(stale,mask,METADATA_ERROR)
        self._error(saved.errors|decision.errors|meta.errors,mask,RULE_ERROR)
        eligible=(mask & states.active & saved.active & ~stale & (self.errors==0)
                  & ~meta.finished & (meta.actors==viewers) & (viewers>=0) & (viewers<=1)
                  & (states.slots>=0) & (states.generations>0) & (states.owners>=0)
                  & (decision.counts>0)
                  & t.isfinite(decision.observations).all(-1))
        fallback=self.rules.value_observation_device(
            self._masked(states,mask & ~eligible & (self.errors==0)),viewers)
        observed=DeviceValueObservation(
            t.where(eligible[:,None],decision.observations,fallback.observations),
            t.where(eligible[:,None],t.ones_like(fallback.viewer_is_actor),fallback.viewer_is_actor),
            t.where(eligible,t.zeros_like(fallback.terminal),fallback.terminal),
            t.where(eligible,t.zeros_like(fallback.terminal_value),fallback.terminal_value),
            t.where(eligible,decision.errors,fallback.errors))
        return observed,eligible

    def _fixed_value(self, states, viewers, model_ids, mask, *, observed=None,reuse_mask=None):
        t = self.t
        reused=observed is not None
        if not reused:
            observed = self.rules.value_observation_device(self._masked(states, mask), viewers)
        self._last_value_observation = observed
        self._error(observed.errors, mask, RULE_ERROR)
        meta = self.rules.metadata_device(self._masked(states, mask))
        self._error(meta.errors, mask, RULE_ERROR)
        if reused:
            self._error(meta.finished|(meta.actors!=viewers),
                mask if reuse_mask is None else mask & reuse_mask,METADATA_ERROR)
        invalid_terminal_meta = meta.finished & ((meta.winners < -1) | (meta.winners > 1))
        self._error(invalid_terminal_meta | (meta.finished != observed.terminal), mask, METADATA_ERROR)
        expected_terminal = t.where(meta.winners == viewers, 1.,
                                     t.where(meta.winners >= 0, -1., 0.)).to(t.float64)
        self._error(meta.finished & (observed.terminal_value != expected_terminal), mask, METADATA_ERROR)
        bad_terminal = observed.terminal & ((observed.terminal_value != -1.)
            & (observed.terminal_value != 0.) & (observed.terminal_value != 1.))
        self._error(bad_terminal, mask, METADATA_ERROR)
        if self.row_work is not None:
            value,invalid=self.row_work.value(observed,model_ids,mask)
            self._error(invalid,mask & ~observed.terminal,INFERENCE_ERROR)
            return t.where(observed.terminal,observed.terminal_value,value)
        # Static capture-time slices bound replicated network weights and
        # intermediates. All roots remain resident; GPU replay runs every slice.
        values, invalid = [], []
        for start in range(0, self.rows, self.value_batch_rows):
            stop = min(start + self.value_batch_rows, self.rows)
            count = stop - start
            c = t.empty((count, 0, self.auxiliary.candidate_dim), dtype=t.float32, device=self.device)
            valid = t.empty((count, 0), dtype=t.bool, device=self.device)
            value_only = t.ones(count, dtype=t.bool, device=self.device)
            prediction = self.auxiliary.predict_device(observed.observations[start:stop], c, valid,
                model_ids[start:stop], observed.viewer_is_actor[start:stop], value_only=value_only)
            values.append((prediction.wdl[:, 2] - prediction.wdl[:, 0]).to(t.float64))
            invalid.append(prediction.invalid_rows)
        self._error(t.cat(invalid), mask & ~observed.terminal, INFERENCE_ERROR)
        value = t.cat(values)
        return t.where(observed.terminal, observed.terminal_value, value)

    def _policy_prediction(self, decision, actors, mask, *, root=False):
        ids = self._policy_ids.gather(1, actors.clamp(0, 1)[:, None])[:, 0]
        context = self.t.ones((self.rows, 1), dtype=self.t.float32, device=self.device)
        if self.row_work is not None:
            prediction=self.row_work.policy(decision,ids,context,mask)
        elif isinstance(decision.candidates, self.t.Tensor):
            prediction = self.inference.predict_device(decision.observations, decision.candidates,
                decision.valid, ids, context)
        else:
            prediction = decision.candidates.predict_device(self.inference, decision.observations,
                decision.valid, ids, context, mask)
        self._error(prediction.invalid_rows, mask, INFERENCE_ERROR)
        self._last_policy_prediction = prediction
        logits = prediction.logits.to(self.t.float64).masked_fill(~decision.valid, -float('inf'))
        if root and self.prior_guide is not None:
            if self.root_prior_shifts is None:
                raise RuntimeError('Bind the array prior before constructing the search')
            hint = self.prior_guide.score_device(decision.observations, decision.candidates,
                                                 decision.valid, mask)
            self._vector(hint.shifts, self.t.float32, (self.rows, self.max_actions))
            self._vector(hint.errors, self.t.int32, (self.rows,))
            bad = ~self.t.isfinite(hint.shifts).all(-1)
            self._error(hint.errors | bad.to(self.t.int32), mask, PRIOR_ERROR)
            shifts = self.t.where((mask & (self.errors == 0))[:, None], hint.shifts, 0.)
            self.root_prior_shifts.copy_(shifts)
            logits = logits + shifts.to(self.t.float64)
        prior = logits.softmax(-1)
        return self.t.where(decision.valid, prior, 0.)

    def _hash_decision(self, decision, mask):
        # A failed rule/inference row has no valid candidate byte stream. Keep
        # its original error instead of hashing empty output and adding KEY_ERROR.
        mask = mask & (self.errors == 0) & (decision.errors == 0)
        if self.row_work is not None:return self.row_work.hash(decision,mask)
        if isinstance(decision.candidates, self.t.Tensor):
            return self.information_keys.hash(decision.observations, decision.candidates,
                                              decision.counts, mask)
        return decision.candidates.hash_device(self.information_keys, decision.observations,
                                               decision.counts, mask)

    def _save_decision(self, decision, mask, *, actions_only=False):
        t = self.t
        if type(actions_only) is not bool:
            raise TypeError('Static action-only decision declaration required')
        if self._decision is None:
            if actions_only:
                raise ValueError('An encoded root decision must precede raw teacher actions')
            self._decision = self._clone_decision(decision)
            return
        if decision.action_pool is not self._decision.action_pool:
            raise ValueError('Decision action arena identity must remain stable')
        self._release_decision_actions(mask, incoming=decision)
        for name in ('slots', 'generations', 'owners', 'active', 'errors'):
            target, source = getattr(self._decision.states, name), getattr(decision.states, name)
            target.copy_(t.where(mask, source, target))
        for field in fields(decision):
            name = field.name
            if name in ('states', 'action_pool'):
                continue
            # Opponent rule teachers provide raw actions for STEP. They do not
            # feed policy/value/hash inference; those buffers are replaced at
            # the next own INSPECT before EVALUATE_NODE may consume them.
            if actions_only and name in ('observations', 'candidates'):
                continue
            target, source = getattr(self._decision, name), getattr(decision, name)
            if source is None and target is None:
                continue
            if name == 'candidates' and not isinstance(target, t.Tensor):
                errors = target.copy_masked_device(source, mask)
                self._error(errors, mask, RULE_ERROR)
                continue
            expand = mask.reshape((self.rows,) + (1,) * (target.ndim - 1))
            target.copy_(t.where(expand, source, target))

    def _release_decision_actions(self, mask, *, incoming=None):
        from .device_action_leases import action_states, release_actions_device
        if self._decision is None:
            return
        protected = [self._roots, self._world(self._world_live), action_states(self._root_decision)]
        if self.reuse_sampled_world:
            protected.append(self._sampled_world(self._sampled_live))
        if incoming is not None:
            protected.append(action_states(incoming))
        errors = release_actions_device(self._decision, mask, self._roots.owners,
                                        protected=protected)
        self._error(errors, mask, RULE_ERROR)

    def _clone_decision(self, decision, *, root=False):
        copied = {}
        for field in fields(decision):
            value = getattr(decision, field.name)
            if field.name == 'candidates' and root and self.borrow_root_candidates:
                copied[field.name]=self.rules.borrow_root_candidates_device(decision)
            elif field.name == 'states':
                copied[field.name] = DeviceStateBatch(value.pool, value.slots.clone(),
                    value.generations.clone(), value.owners.clone(), value.active.clone(), value.errors.clone())
            elif field.name == 'action_pool' or value is None:
                copied[field.name] = value
            else:
                copied[field.name] = value.clone()
        return DeviceDecision(**copied)

    def initialize(self, roots, viewers, world_seed_limbs, policy_index_by_side,
                   auxiliary_index_by_side):
        self._build_context()
        t = self.t
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        for value in (roots.slots, roots.generations, roots.owners, viewers):
            self._vector(value, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(world_seed_limbs, t.int64, (self.rows, 5))
        self._vector(policy_index_by_side, t.int64, (self.rows, 2))
        self._vector(auxiliary_index_by_side, t.int64, (self.rows, 2))
        self._roots, self._viewers, self._seeds = roots, viewers, world_seed_limbs
        self._sorted_root_slots, self._root_order = roots.slots.sort()
        self._simulation_seeds = t.stack([add_world_seed_stride(world_seed_limbs, i)
                                          for i in range(self.simulations)], 1)
        self._pool = roots.pool
        self._policy_ids = policy_index_by_side
        self._aux_ids = auxiliary_index_by_side.gather(1, viewers.clamp(0, 1)[:, None])[:, 0]
        self._world_slots = roots.slots.clone()
        self._world_generations = roots.generations.clone()
        self._world_owners = roots.owners.clone()
        self._sampled_slots = roots.slots.clone()
        self._sampled_generations = roots.generations.clone()
        self._sampled_owners = roots.owners.clone()
        self.errors.copy_(roots.errors.to(t.int32))
        for value in (self.completed, self.depth, self.node_count, self.keys, self.path_slots,
                      self.path_generations, self.path_actions, self.path_valid, self.actions,
                      self.leaf_values, self.policy, self.visit_policy, self.visits, self.phase_work):
            value.zero_()
        self.peak_device_roots.zero_()
        self.root_value_reuse_rows.zero_()
        self.root_value_reuse_validation_errors.zero_()
        self._world_live.zero_()
        self._sampled_live.zero_()
        self.world_samples.zero_()
        if self.root_prior_shifts is not None:
            self.root_prior_shifts.zero_()
        mask = roots.active & (self.errors == 0)
        meta = self.rules.metadata_device(self._masked(roots, mask))
        self._error(meta.errors, mask, RULE_ERROR)
        self._error((meta.actors != viewers) | meta.finished | (viewers < 0) | (viewers > 1)
                    | (roots.owners < 0), mask, METADATA_ERROR)
        mask = mask & (self.errors == 0)
        decision = self.rules.decision_device(self._masked(roots, mask), viewers)
        self._root_decision = self._clone_decision(decision,root=True)
        self._decision = None
        self._save_decision(decision, mask)
        self._error(decision.errors, mask, RULE_ERROR)
        self._error((decision.counts < 1) | (decision.counts > self.max_actions), mask, CAPACITY_ERROR)
        self.requested.copy_(t.where(decision.counts == 1, 1, self.simulations))
        self._limits = decision.counts.clamp(1, self._candidate_cap) - 1
        prior = self._policy_prediction(decision, viewers, mask, root=True)
        observed,observation_mask=None,None
        # Verification deliberately keeps its independent rule query.
        if self.reuse_root_observation and not self.verify_root_value_reuse:
            observed,observation_mask=self._root_observation(roots,viewers,decision,meta,mask)
        def fixed_root_value(value_mask):
            return self._fixed_value(roots,viewers,self._aux_ids,value_mask,
                observed=observed,reuse_mask=observation_mask)
        reuse = self.root_value_reuse
        if reuse is not None and reuse.enabled:
            ids = self._policy_ids.gather(1, viewers.clamp(0, 1)[:, None])[:, 0]
            cached, eligible = reuse.value_device(roots, decision, meta,
                self._last_policy_prediction, viewers, ids, self._aux_ids, mask)
            self.root_value_reuse_rows.copy_(eligible.to(t.int64).sum())
            fallback_mask = mask if self.verify_root_value_reuse else mask & ~eligible
            original = fixed_root_value(fallback_mask)
            if self.verify_root_value_reuse:
                observed = self._last_value_observation
                mismatch = eligible & ((cached != original)
                    | (decision.observations != observed.observations).any(-1)
                    | observed.terminal | (observed.viewer_is_actor[:, 0] != 1.))
                self.root_value_reuse_validation_errors.copy_(mismatch.to(t.int32))
                self._error(mismatch, mask, INFERENCE_ERROR)
                value = original
            else:
                value = t.where(eligible, cached, original)
        else:
            value = fixed_root_value(mask)
        hashed = self._hash_decision(decision, mask)
        self._error(hashed.errors, mask, KEY_ERROR)
        active = mask & (self.errors == 0)
        written = self.statistics.write_nodes(self._base_slots, roots.owners, prior, value,
            decision.valid, active=active, expected_generations=self.node_generations[:, 0])
        self._error(written.error, active, STATISTICS_ERROR)
        claimed = active & ~written.error
        self.node_generations[:, 0].copy_(t.where(claimed, written.generations,
                                                 self.node_generations[:, 0]))
        self.keys[:, 0].copy_(t.where(claimed[:, None], hashed.keys, 0))
        self.node_count.copy_(claimed.long())
        self._noise = t.zeros_like(prior)
        if self.noise:
            draw = self.random.gumbel_device(decision.valid, active)
            self._error(draw.errors, active, RANDOM_ERROR)
            self._noise.copy_(draw.values)
        if self.reuse_sampled_world:
            sample = active & (self.errors == 0)
            sampled = self.rules.sample_world_device(self._masked(roots, sample),
                viewers, world_seed_limbs)
            self.world_samples.add_(sample.long())
            self._replace_world(sampled, sample)
            issued = sample & sampled.active & ~self._is_physical_lease(sampled)
            for target, source in ((self._sampled_slots, sampled.slots),
                                   (self._sampled_generations, sampled.generations),
                                   (self._sampled_owners, sampled.owners)):
                target.copy_(t.where(issued, source, target))
            self._sampled_live.logical_or_(issued)
        self.phase.copy_(t.where(roots.active, SAMPLE, DONE))
        self.phase.copy_(t.where(roots.active & (self.errors != 0), CLEANUP, self.phase))
        self._update_predicate()

    def _update_predicate(self):
        self._unfinished.copy_((self.phase != DONE).to(self.t.uint8))

    @property
    def unfinished_device(self):
        return self._unfinished

    def tick(self):
        self._build_context()
        if self._pool is None:
            raise RuntimeError('Initialize the fixed search cohort first')
        t = self.t
        before = self.phase.clone()
        alive = self._roots.active & (before != DONE)
        self.peak_device_roots.copy_(t.maximum(self.peak_device_roots, alive.sum()))
        masks = tuple((before == phase) & (self.errors == 0) for phase in range(DONE))
        if self.group_ready_leaves:
            # A per-simulation device barrier fills independent teacher lanes.
            # Every root keeps its own selection, RNG and backup order. Roots
            # with an error or in FINAL/CLEANUP/DONE cannot hold the barrier.
            earlier=t.stack([masks[p] for p in (SAMPLE,INSPECT,STEP)]).any()
            opposing_waiting=masks[OPPONENT].any()
            evaluating_waiting=masks[EVALUATE_NODE].any()
            progressing=earlier|opposing_waiting|evaluating_waiting
            leaf_waiting=masks[LEAF].any()
            grouped=list(masks)
            grouped[OPPONENT]=masks[OPPONENT]&~earlier
            grouped[EVALUATE_NODE]=masks[EVALUATE_NODE]&~earlier&~opposing_waiting
            grouped[LEAF]=masks[LEAF]&~progressing
            grouped[BACKUP]=masks[BACKUP]&~progressing&~leaf_waiting
            masks=tuple(grouped)
        for phase in range(DONE):
            self.phase_work[phase].add_((masks[phase] if self.group_ready_leaves else before==phase).long())

        sample = masks[SAMPLE]
        if self.reuse_sampled_world:
            # No sampler kernels in the simulation loop, even with empty masks.
            # All traversals borrow the immutable particle issued in initialize.
            sampled = self._sampled_world(sample)
        else:
            selected_seeds = self._simulation_seeds[self._rows,
                                                    self.completed.clamp(0, self.simulations - 1)]
            sampled = self.rules.sample_world_device(self._masked(self._roots, sample),
                                                     self._viewers, selected_seeds)
            self.world_samples.add_(sample.long())
        # recovery_search starts traversal directly from sample_world. Applying
        # an extra clean here would silently change that source contract.
        self._replace_world(sampled, sample)
        leased = ((sampled.slots != self._roots.slots)
                  | (sampled.generations != self._roots.generations))
        self._error(~leased, sample & (self.errors == 0), RULE_ERROR)
        self.depth.copy_(t.where(sample, 0, self.depth))
        self.path_valid.masked_fill_(sample[:, None], False)
        self.phase.copy_(t.where(sample, INSPECT, self.phase))

        inspect = masks[INSPECT]
        world = self._world(inspect)
        meta = self.rules.metadata_device(world)
        self._error(meta.errors, inspect, RULE_ERROR)
        terminal = inspect & meta.finished
        self._error((meta.winners < -1) | (meta.winners > 1), terminal, METADATA_ERROR)
        terminal_value = t.where(meta.winners == self._viewers, 1.,
                                 t.where(meta.winners >= 0, -1., 0.)).to(t.float64)
        terminal_observation = self.rules.value_observation_device(self._world(terminal), self._viewers)
        self._error(terminal_observation.errors, terminal, RULE_ERROR)
        self._error(~terminal_observation.terminal
                    | (terminal_observation.terminal_value != terminal_value), terminal, METADATA_ERROR)
        self.leaf_values.copy_(t.where(terminal, terminal_value, self.leaf_values))
        exhausted = inspect & ~meta.finished & (self.depth >= self.max_depth)
        traverse = inspect & ~meta.finished & ~exhausted
        self._error((meta.actors < 0) | (meta.actors > 1), traverse, METADATA_ERROR)
        own = traverse & (meta.actors == self._viewers) & (self.errors == 0)
        opposing = traverse & (meta.actors != self._viewers) & (self.errors == 0)
        query = own
        if self.reuse_root_decision:
            from dataclasses import replace
            root_rows = own & (self.depth == 0)
            self._save_decision(replace(self._root_decision,
                states=self._world(root_rows)), root_rows)
            query = own & ~root_rows
        decision = self.rules.decision_device(self._world(query), self._viewers)
        self._error(decision.errors, query, RULE_ERROR)
        own = own & (self.errors == 0)
        self._save_decision(decision, query & (self.errors == 0))
        if self.reuse_root_decision:
            decision = self._decision
        hashed = self._hash_decision(decision, own)
        self._error(hashed.errors, own, KEY_ERROR)
        matches = (self.keys == hashed.keys[:, None, :]).all(-1)
        matches = matches & (self._columns[None, :] < self.node_count[:, None])
        found = matches.any(-1)
        node = matches.long().argmax(-1)
        self._error((self.depth == 0) & (~found | (node != 0)), own, KEY_ERROR)
        existing = own & found & (self.errors == 0)
        fresh = own & ~found & (self.errors == 0)
        self._error(self.node_count >= self.nodes_per_root, fresh, CAPACITY_ERROR)
        slots = self._base_slots + node
        generations = self.node_generations[self._rows, node]
        root_choice = self.statistics.select_root(slots, generations, self._noise,
            self._schedule[self._limits, self.completed.clamp(0, self.simulations - 1)],
            active=existing & (self.depth == 0))
        inner_choice = self.statistics.select_interior(slots, generations,
                                                       active=existing & (self.depth > 0))
        self._error(root_choice.error | inner_choice.error, existing, STATISTICS_ERROR)
        chosen = t.where(self.depth == 0, root_choice.actions, inner_choice.actions)
        depth = self.depth.clamp(0, self.max_depth - 1)
        path_mask = t.nn.functional.one_hot(depth, self.max_depth).bool() & existing[:, None]
        self.path_slots.copy_(t.where(path_mask, slots[:, None], self.path_slots))
        self.path_generations.copy_(t.where(path_mask, generations[:, None], self.path_generations))
        self.path_actions.copy_(t.where(path_mask, chosen[:, None], self.path_actions))
        self.path_valid.logical_or_(path_mask)
        self.actions.copy_(t.where(existing, chosen, self.actions))
        self.phase.copy_(t.where(terminal, BACKUP, self.phase))
        self.phase.copy_(t.where(exhausted, LEAF, self.phase))
        self.phase.copy_(t.where(opposing, OPPONENT, self.phase))
        self.phase.copy_(t.where(existing, STEP, self.phase))
        self.phase.copy_(t.where(fresh, EVALUATE_NODE, self.phase))

        fresh = masks[EVALUATE_NODE]
        prior = self._policy_prediction(self._decision, self._viewers, fresh)
        observed=None
        if self.reuse_interior_observation:
            world=self._world(fresh);saved=self._decision.states
            if saved.pool is not world.pool:
                raise ValueError('Interior decision observation must use the same state arena')
            mismatch=(saved.slots!=world.slots)|(saved.generations!=world.generations)|(saved.owners!=world.owners)
            self._error(mismatch,fresh,METADATA_ERROR)
            observed=DeviceValueObservation(self._decision.observations,
                t.ones((self.rows,1),dtype=t.float32,device=self.device),
                t.zeros(self.rows,dtype=t.bool,device=self.device),
                t.zeros(self.rows,dtype=t.float64,device=self.device),self._decision.errors)
        value = self._fixed_value(self._world(fresh), self._viewers, self._aux_ids, fresh,observed=observed)
        hashed = self._hash_decision(self._decision, fresh)
        self._error(hashed.errors, fresh, KEY_ERROR)
        index = self.node_count.clamp(0, self.nodes_per_root - 1)
        written = self.statistics.write_nodes(self._base_slots + index, self._roots.owners,
            prior, value, self._decision.valid, active=fresh & (self.errors == 0),
            expected_generations=self.node_generations[self._rows, index])
        self._error(written.error, fresh, STATISTICS_ERROR)
        success = fresh & (self.errors == 0)
        self.keys[self._rows, index] = t.where(success[:, None], hashed.keys, self.keys[self._rows, index])
        self.node_generations[self._rows, index] = t.where(success, written.generations,
                                                          self.node_generations[self._rows, index])
        self.node_count.add_(success.long())
        if self.teacher_mode == 'network_value':
            self.leaf_values.copy_(t.where(success, value, self.leaf_values))
            self.phase.copy_(t.where(success, BACKUP, self.phase))
        else:
            self.phase.copy_(t.where(success, LEAF, self.phase))

        leaf = masks[LEAF]
        old = self._world(leaf)
        if self.teacher_mode == 'covered_value':
            continuation = self.rules.teacher_continue_device(old, self._viewers, horizon=self.terminal_horizon)
            self._release_replaced(old, continuation, leaf)
            self._replace_world(continuation, leaf)
        value = self._fixed_value(self._world(leaf), self._viewers, self._aux_ids, leaf)
        self.leaf_values.copy_(t.where(leaf, value, self.leaf_values))
        self.phase.copy_(t.where(leaf, BACKUP, self.phase))

        opponent = masks[OPPONENT]
        meta = self.rules.metadata_device(self._world(opponent))
        if self.opponent_mode == 'policy':
            opponent_decision = self.rules.decision_device(self._world(opponent), meta.actors)
            self._error(opponent_decision.errors | meta.errors, opponent, RULE_ERROR)
            probabilities = self._policy_prediction(opponent_decision, meta.actors, opponent)
            draw = self.random.categorical(probabilities.contiguous(),
                opponent_decision.counts.contiguous(),
                (opponent & (self.errors == 0)).contiguous())
            self._error(draw.errors, opponent, RANDOM_ERROR)
            self._save_decision(opponent_decision, opponent)
            self.actions.copy_(t.where(opponent, draw.indices, self.actions))
        else:
            teacher_decision, teacher_choice, teacher_errors = self.rules.rule_decision_device(
                self._world(opponent), meta.actors)
            self._error(meta.errors | teacher_errors, opponent, RULE_ERROR)
            self._save_decision(teacher_decision, opponent, actions_only=True)
            self.actions.copy_(t.where(opponent, teacher_choice, self.actions))
        self.phase.copy_(t.where(opponent, STEP, self.phase))

        step = masks[STEP]
        old = self._world(step)
        meta = self.rules.metadata_device(old)
        self._error(meta.errors, step, RULE_ERROR)
        advanced = self.rules.step_device(old, meta.actors, self._decision, self.actions, physical=False)
        cleaned = self.rules.clean_device(self._masked(advanced, step))
        self._release_replaced(advanced, cleaned, step)
        distinct_advanced = ((old.slots != advanced.slots)
                             | (old.generations != advanced.generations))
        self._release_replaced(old, cleaned, step & distinct_advanced)
        self._replace_world(cleaned, step)
        self.depth.add_(step.long())
        self.phase.copy_(t.where(step, INSPECT, self.phase))

        backup = masks[BACKUP]
        backed = self.statistics.backup(self.path_slots, self.path_generations, self.path_actions,
            self.path_valid, self.leaf_values, self._roots.owners, active=backup)
        self._error(backed.error, backup, STATISTICS_ERROR)
        accepted = backup & (self.errors == 0)
        self.completed.add_(accepted.long())
        self._release_lease(self._world(backup), backup & self._world_live)
        self._world_live.logical_and_(~backup)
        self.phase.copy_(t.where(accepted & (self.completed < self.requested), SAMPLE, self.phase))
        self.phase.copy_(t.where(accepted & (self.completed == self.requested), FINAL, self.phase))

        final = masks[FINAL]
        final_choice = self.statistics.select_final(self._base_slots, self.node_generations[:, 0],
                                                    self._noise, active=final)
        policy = self.statistics.improved_policy(self._base_slots, self.node_generations[:, 0], active=final)
        snapshot = self.statistics.snapshot(self._base_slots, self.node_generations[:, 0], active=final)
        self._error(final_choice.error | policy.error | snapshot.error, final, STATISTICS_ERROR)
        self.actions.copy_(t.where(final, final_choice.actions, self.actions))
        self.policy.copy_(t.where(final[:, None], policy.policy.to(t.float32), self.policy))
        self.visits.copy_(t.where(final[:, None], snapshot.visits, self.visits))
        denominator = snapshot.visits.sum(-1, keepdim=True)
        visit_policy = t.where(denominator > 0, snapshot.visits / denominator.clamp_min(1), snapshot.prior)
        self.visit_policy.copy_(t.where(final[:, None], visit_policy.to(t.float32), self.visit_policy))
        self.phase.copy_(t.where(final, CLEANUP, self.phase))

        cleanup = (before == CLEANUP)
        self._drain_device(cleanup)
        self.phase.copy_(t.where((self.errors != 0) & (self.phase != DONE), CLEANUP, self.phase))
        self._update_predicate()

    def _drain_device(self, cleanup):
        """Retire issued transient leases/statistics, retaining root actions.

        Runs in the captured graph both on normal CLEANUP and after an inner
        WHILE guard stops before the search can reach its next cleanup tick.
        """
        t=self.t
        self._release_decision_actions(cleanup)
        # Execution errors must not strand already-issued hypothetical leases.
        # Release is a checked ownership operation, not further rule execution;
        # the adapter validates slot/generation/owner even after search failure.
        cleanup_world = DeviceStateBatch(self._pool, self._world_slots,
            self._world_generations, self._world_owners, cleanup & self._world_live,
            t.zeros_like(self.errors))
        self._release_lease(cleanup_world, cleanup & self._world_live)
        self._world_live.logical_and_(~cleanup)
        if self.reuse_sampled_world:
            cached = self._sampled_world(cleanup)
            # Remove protection only after transient worlds/actions are drained.
            self._sampled_live.logical_and_(~cleanup)
            self._release_lease(cached, cleanup)
        for column in range(self.nodes_per_root):
            active = cleanup & (column < self.node_count)
            released = self.statistics.release_nodes(self.node_slots[:, column],
                self.node_generations[:, column], active=active)
            self._error(released.error, active, STATISTICS_ERROR)
            self.node_generations[:, column].add_((active & ~released.error).long())
        self.phase.copy_(t.where(cleanup, DONE, self.phase))

    def abort_device(self, active, *, error_code):
        """Fail unfinished rows and drain them without a simulation or commit.

        Completed counts, requested budgets, keys and random streams remain
        unchanged. The physical driver still consumes/releases root actions.
        """
        self._build_context()
        self._vector(active,self.t.bool,(self.rows,))
        if type(error_code) is not int or error_code<1:
            raise ValueError('Explicit nonzero device abort error required')
        stopped=active & self._roots.active & (self.phase!=DONE)
        self._error(stopped,stopped,error_code)
        self._drain_device(stopped)
        self._update_predicate()

    def result_device(self):
        if self._root_decision is None:
            raise RuntimeError('No device search cohort initialized')
        complete = (self.phase == DONE) & (self.errors == 0) & self._roots.active
        complete = complete & (self.completed == self.requested)
        return DeviceGumbelResult(self._root_decision, self.actions, self.policy, self.visit_policy,
            self.visits, self.requested, self.completed, complete, self.errors, self.node_count,
            self.phase_work, self.peak_device_roots, self.keys[:, 0].contiguous(),
            self.root_prior_shifts, getattr(self.prior_guide, 'version', None))
