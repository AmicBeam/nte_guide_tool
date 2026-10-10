"""Tensor state machine for captured, device-controlled Gumbel search.

This driver has no Python per-root coroutines or scalar device reads. It needs
a graph-capable formal-rule adapter, exact random stream, device information
keys and qualified auxiliary value inference. It is not a rule executor and
does not claim complete game support merely by constructing the FSM.

The production execution wrapper must put initialize/tick inside a device
conditional CUDA graph. CPU and synthetic execution are explicit test modes.
Covered-value is the current implemented teacher contract; other modes are
rejected, not substituted. Serving public-proof budgeting is a separate layer.
"""
from dataclasses import dataclass, fields
from typing import Any

from ..batched_duel.device_rule_protocol import DeviceStateBatch, DeviceDecision


SAMPLE, INSPECT, EVALUATE_NODE, LEAF, OPPONENT, STEP, BACKUP, FINAL, CLEANUP, DONE = range(10)
RULE_ERROR, INFERENCE_ERROR, STATISTICS_ERROR, KEY_ERROR = 1, 2, 4, 8
CAPACITY_ERROR, RANDOM_ERROR, METADATA_ERROR = 16, 32, 64


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
                 allow_synthetic=False):
        import torch
        from ..search_policy import visit_schedule
        for value in (rows, max_actions, simulations, gumbel_candidates, max_depth):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer bounded device search dimensions required')
        if type(terminal_horizon) is not int or terminal_horizon < 0:
            raise ValueError('Bounded terminal horizon required')
        if teacher_mode != 'covered_value':
            raise NotImplementedError('Device FSM currently requires the covered_value teacher')
        if type(noise) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit boolean noise and synthetic controls required')
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
        self.statistics, self.information_keys = statistics, information_keys
        self.random, self.noise, self.allow_synthetic = random, noise, allow_synthetic
        self.rows, self.max_actions = rows, max_actions
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
        self.phase_work = torch.zeros((DONE, rows), dtype=torch.int64, **z)
        self.peak_device_roots = torch.zeros((), dtype=torch.int64, **z)
        self._unfinished = torch.zeros(rows, dtype=torch.uint8, **z)
        self._pool = None
        self._root_decision = self._decision = None

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
        for target, value in ((self._world_slots, states.slots),
                              (self._world_generations, states.generations),
                              (self._world_owners, states.owners)):
            target.copy_(self.t.where(mask, value, target))
        self._error(states.errors, mask, RULE_ERROR)

    def _release_replaced(self, old, new, mask):
        changed = (old.slots != new.slots) | (old.generations != new.generations)
        self.rules.release_device(self._masked(old, mask & changed))

    def _fixed_value(self, states, viewers, model_ids, mask):
        t = self.t
        observed = self.rules.value_observation_device(self._masked(states, mask), viewers)
        self._error(observed.errors, mask, RULE_ERROR)
        c = t.zeros((self.rows, 1, self.auxiliary.candidate_dim), dtype=t.float32, device=self.device)
        valid = t.zeros((self.rows, 1), dtype=t.bool, device=self.device)
        value_only = t.ones(self.rows, dtype=t.bool, device=self.device)
        prediction = self.auxiliary.predict_device(observed.observations, c, valid, model_ids,
                                                   observed.viewer_is_actor, value_only=value_only)
        self._error(prediction.invalid_rows, mask & ~observed.terminal, INFERENCE_ERROR)
        value = (prediction.wdl[:, 2] - prediction.wdl[:, 0]).to(t.float64)
        return t.where(observed.terminal, observed.terminal_value, value)

    def _policy_prediction(self, decision, actors, mask):
        ids = self._policy_ids.gather(1, actors.clamp(0, 1)[:, None])[:, 0]
        prediction = self.inference.predict_device(decision.observations, decision.candidates,
            decision.valid, ids, self.t.ones((self.rows, 1), dtype=self.t.float32, device=self.device))
        self._error(prediction.invalid_rows, mask, INFERENCE_ERROR)
        logits = prediction.logits.to(self.t.float64).masked_fill(~decision.valid, -float('inf'))
        prior = logits.softmax(-1)
        return self.t.where(decision.valid, prior, 0.)

    def _save_decision(self, decision, mask):
        t = self.t
        if self._decision is None:
            self._decision = self._clone_decision(decision)
            return
        if decision.action_pool is not self._decision.action_pool:
            raise ValueError('Decision action arena identity must remain stable')
        for name in ('slots', 'generations', 'owners', 'active', 'errors'):
            target, source = getattr(self._decision.states, name), getattr(decision.states, name)
            target.copy_(t.where(mask, source, target))
        for field in fields(decision):
            name = field.name
            if name in ('states', 'action_pool'):
                continue
            target, source = getattr(self._decision, name), getattr(decision, name)
            if source is None and target is None:
                continue
            expand = mask.reshape((self.rows,) + (1,) * (target.ndim - 1))
            target.copy_(t.where(expand, source, target))

    def _clone_decision(self, decision):
        copied = {}
        for field in fields(decision):
            value = getattr(decision, field.name)
            if field.name == 'states':
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
        self._simulation_seeds = t.stack([add_world_seed_stride(world_seed_limbs, i)
                                          for i in range(self.simulations)], 1)
        self._pool = roots.pool
        self._policy_ids = policy_index_by_side
        self._aux_ids = auxiliary_index_by_side.gather(1, viewers.clamp(0, 1)[:, None])[:, 0]
        self._world_slots = roots.slots.clone()
        self._world_generations = roots.generations.clone()
        self._world_owners = roots.owners.clone()
        self.errors.copy_(roots.errors.to(t.int32))
        for value in (self.completed, self.depth, self.node_count, self.keys, self.path_slots,
                      self.path_generations, self.path_actions, self.path_valid, self.actions,
                      self.leaf_values, self.policy, self.visit_policy, self.visits, self.phase_work):
            value.zero_()
        self.peak_device_roots.zero_()
        mask = roots.active & (self.errors == 0)
        meta = self.rules.metadata_device(self._masked(roots, mask))
        self._error(meta.errors, mask, RULE_ERROR)
        self._error((meta.actors != viewers) | meta.finished | (viewers < 0) | (viewers > 1)
                    | (roots.owners < 0), mask, METADATA_ERROR)
        mask = mask & (self.errors == 0)
        decision = self.rules.decision_device(self._masked(roots, mask), viewers)
        self._root_decision = self._clone_decision(decision)
        self._decision = None
        self._save_decision(decision, mask)
        self._error(decision.errors, mask, RULE_ERROR)
        self._error((decision.counts < 1) | (decision.counts > self.max_actions), mask, CAPACITY_ERROR)
        self.requested.copy_(t.where(decision.counts == 1, 1, self.simulations))
        self._limits = decision.counts.clamp(1, self._candidate_cap) - 1
        prior = self._policy_prediction(decision, viewers, mask)
        value = self._fixed_value(roots, viewers, self._aux_ids, mask)
        hashed = self.information_keys.hash(decision.observations, decision.candidates,
                                            decision.counts, mask)
        self._error(hashed.errors, mask, KEY_ERROR)
        active = mask & (self.errors == 0)
        written = self.statistics.write_nodes(self._base_slots, roots.owners, prior, value,
            decision.valid, active=active, expected_generations=self.node_generations[:, 0])
        self._error(written.error, active, STATISTICS_ERROR)
        self.node_generations[:, 0].copy_(written.generations)
        self.keys[:, 0].copy_(hashed.keys)
        self.node_count.copy_(active.long())
        self._noise = t.zeros_like(prior)
        if self.noise:
            draw = self.random.gumbel_device(decision.valid, active)
            self._error(draw.errors, active, RANDOM_ERROR)
            self._noise.copy_(draw.values)
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
        for phase in range(DONE):
            self.phase_work[phase].add_((before == phase).long())

        sample = masks[SAMPLE]
        selected_seeds = self._simulation_seeds[self._rows,
                                                self.completed.clamp(0, self.simulations - 1)]
        sampled = self.rules.sample_world_device(self._masked(self._roots, sample),
                                                 self._viewers, selected_seeds)
        cleaned = self.rules.clean_device(self._masked(sampled, sample))
        self._release_replaced(sampled, cleaned, sample)
        self._replace_world(cleaned, sample)
        self.depth.copy_(t.where(sample, 0, self.depth))
        self.path_valid.masked_fill_(sample[:, None], False)
        self.phase.copy_(t.where(sample, INSPECT, self.phase))

        inspect = masks[INSPECT]
        world = self._world(inspect)
        meta = self.rules.metadata_device(world)
        self._error(meta.errors, inspect, RULE_ERROR)
        terminal = inspect & meta.finished
        terminal_value = t.where(meta.winners == self._viewers, 1.,
                                 t.where(meta.winners >= 0, -1., 0.)).to(t.float64)
        self.leaf_values.copy_(t.where(terminal, terminal_value, self.leaf_values))
        exhausted = inspect & ~meta.finished & (self.depth >= self.max_depth)
        traverse = inspect & ~meta.finished & ~exhausted
        self._error((meta.actors < 0) | (meta.actors > 1), traverse, METADATA_ERROR)
        own = traverse & (meta.actors == self._viewers) & (self.errors == 0)
        opposing = traverse & (meta.actors != self._viewers) & (self.errors == 0)
        decision = self.rules.decision_device(self._world(own), self._viewers)
        self._error(decision.errors, own, RULE_ERROR)
        self._save_decision(decision, own)
        hashed = self.information_keys.hash(decision.observations, decision.candidates, decision.counts, own)
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
        value = self._fixed_value(self._world(fresh), self._viewers, self._aux_ids, fresh)
        hashed = self.information_keys.hash(self._decision.observations, self._decision.candidates,
                                            self._decision.counts, fresh)
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
        self.phase.copy_(t.where(success, LEAF, self.phase))

        leaf = masks[LEAF]
        old = self._world(leaf)
        continuation = self.rules.teacher_continue_device(old, self._viewers, horizon=self.terminal_horizon)
        self._release_replaced(old, continuation, leaf)
        self._replace_world(continuation, leaf)
        value = self._fixed_value(self._world(leaf), self._viewers, self._aux_ids, leaf)
        self.leaf_values.copy_(t.where(leaf, value, self.leaf_values))
        self.phase.copy_(t.where(leaf, BACKUP, self.phase))

        opponent = masks[OPPONENT]
        meta = self.rules.metadata_device(self._world(opponent))
        teacher_decision, teacher_choice, teacher_errors = self.rules.rule_decision_device(
            self._world(opponent), meta.actors)
        self._error(meta.errors | teacher_errors, opponent, RULE_ERROR)
        self._save_decision(teacher_decision, opponent)
        self.actions.copy_(t.where(opponent, teacher_choice, self.actions))
        self.phase.copy_(t.where(opponent, STEP, self.phase))

        step = masks[STEP]
        old = self._world(step)
        meta = self.rules.metadata_device(old)
        self._error(meta.errors, step, RULE_ERROR)
        advanced = self.rules.step_device(old, meta.actors, self._decision, self.actions, physical=False)
        cleaned = self.rules.clean_device(self._masked(advanced, step))
        self._release_replaced(advanced, cleaned, step)
        self._release_replaced(old, cleaned, step)
        self._replace_world(cleaned, step)
        self.depth.add_(step.long())
        self.phase.copy_(t.where(step, INSPECT, self.phase))

        backup = masks[BACKUP]
        backed = self.statistics.backup(self.path_slots, self.path_generations, self.path_actions,
            self.path_valid, self.leaf_values, self._roots.owners, active=backup)
        self._error(backed.error, backup, STATISTICS_ERROR)
        accepted = backup & (self.errors == 0)
        self.completed.add_(accepted.long())
        self.rules.release_device(self._world(backup))
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
        for column in range(self.nodes_per_root):
            active = cleanup & (column < self.node_count)
            released = self.statistics.release_nodes(self.node_slots[:, column],
                self.node_generations[:, column], active=active)
            self._error(released.error, active, STATISTICS_ERROR)
            self.node_generations[:, column].add_((active & ~released.error).long())
        self.phase.copy_(t.where(cleanup, DONE, self.phase))
        self.phase.copy_(t.where((self.errors != 0) & (self.phase != DONE), CLEANUP, self.phase))
        self._update_predicate()

    def result_device(self):
        if self._root_decision is None:
            raise RuntimeError('No device search cohort initialized')
        complete = (self.phase == DONE) & (self.errors == 0) & self._roots.active
        complete = complete & (self.completed == self.requested)
        return DeviceGumbelResult(self._root_decision, self.actions, self.policy, self.visit_policy,
            self.visits, self.requested, self.completed, complete, self.errors, self.node_count,
            self.phase_work, self.peak_device_roots)
