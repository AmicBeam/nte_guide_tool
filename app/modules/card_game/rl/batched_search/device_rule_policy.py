"""Canonical device rule choices with actual public actor value inputs.

This is a zero-simulation controller, not Gumbel or a qualified auxiliary
collector. Raw legal ordering (including concede) and the rule's exact integer
choice survive unchanged by default. Optional first-six exploration uses exact
bounded PCG64 integer draws and retains the original rule choice for audit.
"""
from dataclasses import replace

from ..batched_duel.device_rule_protocol import DeviceDecision, DeviceStateBatch
from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
from .device_search import DeviceGumbelResult, METADATA_ERROR


class DeviceRulePolicy:
    supports_full_duel_rules = False
    gpu_qualified = False
    algorithm = 'canonical_rule'
    counts_as_search = False
    noise = False
    opponent_mode = 'rule'

    def __init__(self, rules, *, rows, max_actions, device='cuda',
                 allow_cpu_reference=False, allow_synthetic=False,
                 explore_first_six=False, exploration_seeds=None, exploration_max_retries=128):
        import torch
        if any(type(v) is not bool for v in (allow_cpu_reference, allow_synthetic, explore_first_six)):
            raise TypeError('Explicit boolean reference, synthetic and exploration controls required')
        if type(rows) is not int or not 1 <= rows <= 4096:
            raise ValueError('Bounded positive device rule rows required')
        if type(max_actions) is not int or not 1 <= max_actions <= 16384:
            raise ValueError('Bounded complete raw legal width required')
        self.t, self.device = torch, torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_reference and allow_synthetic):
            raise ValueError('CPU requires explicit synthetic and reference controls')
        for name in ('metadata_device', 'rule_decision_device', 'value_observation_device'):
            if not callable(getattr(rules, name, None)):
                raise TypeError('Missing canonical device rule interface: ' + name)
        if not allow_synthetic:
            from .device_soa_rules import DeviceSoARules
            from .device_soa_teacher import inspect_soa_teacher_dependencies
            if not isinstance(rules, DeviceSoARules):
                raise TypeError('Formal rule policy requires actual DeviceSoARules')
            if (rules.require_teacher is not True or type(rules.rows) is not int
                    or type(rules.max_actions) is not int or rules.rows != rows
                    or rules.max_actions != max_actions or rules.device != self.device):
                raise ValueError('Exact teacher-enabled SoA device, rows and raw legal width required')
            gaps = inspect_soa_teacher_dependencies(rules)
            if gaps:
                raise ValueError('; '.join(gaps))
        self.rules, self.rows, self.max_actions = rules, rows, max_actions
        self.allow_synthetic = allow_synthetic
        if (type(exploration_max_retries) is not int
                or not 1 <= exploration_max_retries <= 1024):
            raise ValueError('Bounded exact PCG64 integer rejection budget required')
        self.explore_first_six = explore_first_six
        if explore_first_six:
            self.algorithm = 'canonical_rule_first6_pcg64_integers'
        self.exploration_max_retries = exploration_max_retries
        self._loop = self._steps = self._loop_seeds = self._loop_owners = None
        self._seed_metadata = self._exploration_rng = None
        self._closed = False
        if explore_first_six:
            if (type(exploration_seeds) is not tuple or len(exploration_seeds) != rows
                    or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in exploration_seeds)):
                raise ValueError('One exact nonnegative int64 job seed per rule row required')
            if self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('PCG64 seed expansion belongs only to setup before capture')
            from .device_random import DevicePCG64
            self._exploration_rng = DevicePCG64(exploration_seeds, device=str(self.device),
                max_rows=rows, max_draws=1, max_choices=1,
                allow_cpu_reference=self.device.type == 'cpu')
            self._seed_metadata = torch.tensor(exploration_seeds, dtype=torch.int64, device=self.device)
        elif exploration_seeds is not None:
            raise ValueError('Exploration seeds require explicit first-six mode')
        self._unfinished = torch.zeros(rows, dtype=torch.uint8, device=self.device)
        self._action_ids = torch.arange(max_actions, dtype=torch.int64, device=self.device)
        self._rule_choices = torch.full((rows,), -1, dtype=torch.int64, device=self.device)
        self._explored = torch.zeros(rows, dtype=torch.bool, device=self.device)
        self._exploration_errors = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self._result = None

    def bind_game_loop(self, loop):
        """Bind one cohort once; initialize never reseeds the exploration bank."""
        if self._closed or not self.explore_first_six or self._loop is not None:
            raise RuntimeError('Enabled exploration may bind one game cohort exactly once')
        if not self.allow_synthetic:
            from .device_games import DeviceGameLoop
            if not isinstance(loop, DeviceGameLoop):
                raise TypeError('Actual DeviceGameLoop required for formal exploration')
        if (loop.search is not self or loop.rules is not self.rules
                or type(loop.rows) is not int or loop.rows != self.rows or loop.device != self.device):
            raise ValueError('Exploration loop must share the exact controller, rules, rows and device')
        steps = loop.steps_device
        self._exact(steps, self.t.int64, (self.rows,))
        self._exact(loop._job_seeds, self.t.int64, (self.rows,))
        self._exact(loop._owners, self.t.int64, (self.rows,))
        if steps is not loop._steps:
            raise ValueError('Bind the physical loop step counter itself')
        self._loop, self._steps = loop, steps
        self._loop_seeds, self._loop_owners = loop._job_seeds, loop._owners

    def _exact(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape or not value.is_contiguous()):
            raise ValueError('Exact bound-device rule-policy tensor metadata required')

    def _states(self, states):
        if not isinstance(states, DeviceStateBatch):
            raise TypeError('Checked device state lease required')
        t = self.t
        for name, dtype in (('slots', t.int64), ('generations', t.int64), ('owners', t.int64),
                            ('active', t.bool), ('errors', t.int32)):
            self._exact(getattr(states, name), dtype, (self.rows,))

    def _meta(self, roots):
        meta = self.rules.metadata_device(roots)
        t = self.t
        for name, dtype in (('actors', t.int64), ('versions', t.int64), ('phases', t.int32),
                            ('finished', t.bool), ('winners', t.int64), ('errors', t.int32)):
            self._exact(getattr(meta, name), dtype, (self.rows,))
        return meta

    def initialize(self, roots, actors, world_seeds, policy_ids, auxiliary_ids):
        t = self.t
        if self._closed:
            raise RuntimeError('Rule controller closed')
        if self.explore_first_six and self._loop is None:
            raise RuntimeError('Bind the real physical game loop before exploring')
        if self.device.type == 'cuda' and not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Device rule policy requires CUDA graph capture')
        self._states(roots)
        self._exact(actors, t.int64, (self.rows,))
        self._exact(policy_ids, t.int64, (self.rows, 2))
        self._exact(auxiliary_ids, t.int64, (self.rows, 2))
        # The deterministic rule controller does not consume world seeds or
        # model banks. Formal GameLoop still supplies its real seed arguments.
        if world_seeds is not None or not self.allow_synthetic:
            self._exact(world_seeds, t.int64, (self.rows, 5))
        if getattr(self.rules, 'store', roots.pool) is not roots.pool:
            raise ValueError('Root arena differs from the rule executor')
        initial_slots = roots.slots.clone()
        initial_generations = roots.generations.clone()
        initial_owners = roots.owners.clone()
        validate = getattr(roots.pool, 'validate', None)
        if not callable(validate):
            raise TypeError('Actual checked root lease registry required')
        _, valid = validate(roots, shared_reads=True)
        self._exact(valid, t.bool, (self.rows,))
        meta = self._meta(roots)
        source_versions = meta.versions.clone()
        source_actors = meta.actors.clone()
        source_phases = meta.phases.clone()
        errors = roots.errors.clone() | t.where(roots.active, meta.errors, 0)
        bad = (~valid | (roots.slots < 0) | (roots.generations < 1) | (roots.owners < 0)
               | (actors < 0) | (actors > 1) | (actors != meta.actors) | meta.finished | (meta.versions < 0))
        errors.bitwise_or_((roots.active & bad).to(t.int32) * METADATA_ERROR)
        if self.explore_first_six:
            if (self._loop.steps_device is not self._steps or self._loop._steps is not self._steps
                    or self._loop._job_seeds is not self._loop_seeds or self._loop._owners is not self._loop_owners):
                raise ValueError('Bound physical counter/seed/owner tensors were replaced')
            seed_bad = ((self._loop_seeds != self._seed_metadata) | (self._steps < 0)
                        | (roots.owners != self._loop_owners))
            errors.bitwise_or_((roots.active & seed_bad).to(t.int32) * METADATA_ERROR)
        admitted = roots.active & (errors == 0)
        selected = replace(roots, active=admitted, errors=errors.clone())
        decision, chosen, rule_errors = self.rules.rule_decision_device(selected, actors)
        if not isinstance(decision, DeviceDecision):
            raise TypeError('Actual raw DeviceDecision required')
        for value, dtype, shape in (
                (chosen, t.int64, (self.rows,)), (rule_errors, t.int32, (self.rows,)),
                (decision.errors, t.int32, (self.rows,)), (decision.counts, t.int64, (self.rows,)),
                (decision.valid, t.bool, (self.rows, self.max_actions)),
                (decision.action_refs, t.int32, (self.rows, self.max_actions)),
                (decision.action_slots, t.int64, (self.rows,)),
                (decision.action_generations, t.int64, (self.rows,))):
            self._exact(value, dtype, shape)
        self._states(decision.states)
        if decision.states.pool is not roots.pool:
            raise ValueError('Rule decision must retain the actual root arena')
        pool = decision.action_pool
        if (not isinstance(pool, CompactDeviceGraphStore) or pool.closed or pool.device != self.device
                or getattr(self.rules, 'action_store', pool) is not pool):
            raise ValueError('Actual live rule action arena required')
        checking = (admitted & (rule_errors == 0) & (decision.errors == 0)).contiguous()
        slots, action_good, action_errors, _ = pool._leases(decision.action_slots,
            decision.action_generations, decision.states.owners, checking, live=True, shared_reads=True)
        saved = decision.states
        same = (saved.active & (saved.errors == 0) & (saved.slots == roots.slots)
                & (saved.generations == roots.generations) & (saved.owners == roots.owners))
        counts = decision.counts
        prefix = self._action_ids[None, :] < counts[:, None]
        legal = ((counts >= 1) & (counts <= self.max_actions) & (decision.valid == prefix).all(-1)
                 & (chosen >= 0) & (chosen < counts))
        safe_refs = (decision.action_refs >= 0) & (decision.action_refs.long() < pool.counts[slots, 0, None])
        legal &= (~prefix | safe_refs).all(-1)
        errors.bitwise_or_(t.where(admitted, rule_errors | decision.errors | saved.errors | action_errors, 0))
        errors.bitwise_or_((admitted & (~same | ~action_good | ~legal)).to(t.int32) * METADATA_ERROR)
        live = admitted & (errors == 0)
        observed = self.rules.value_observation_device(replace(selected, active=live, errors=errors.clone()), actors)
        x = observed.observations
        if (not isinstance(x, t.Tensor) or x.device != self.device or x.dtype != t.float32
                or x.ndim != 2 or x.shape[0] != self.rows or x.shape[1] < 1 or not x.is_contiguous()):
            raise ValueError('Actual nonempty public actor observation matrix required')
        for name, dtype, shape in (('viewer_is_actor', t.float32, (self.rows, 1)),
                ('terminal', t.bool, (self.rows,)), ('terminal_value', t.float64, (self.rows,)),
                ('errors', t.int32, (self.rows,))):
            self._exact(getattr(observed, name), dtype, shape)
        after = self._meta(replace(selected, active=live, errors=errors.clone()))
        checked_after = replace(roots, active=live, errors=t.zeros_like(errors))
        _, valid_after = validate(checked_after, shared_reads=True)
        self._exact(valid_after, t.bool, (self.rows,))
        _, actions_after, action_errors_after, _ = pool._leases(decision.action_slots,
            decision.action_generations, saved.owners, live.contiguous(), live=True, shared_reads=True)
        same_version = ((after.versions == source_versions) & (after.actors == source_actors)
                        & (actors == source_actors) & (after.phases == source_phases) & ~after.finished
                        & (roots.slots == initial_slots) & (roots.generations == initial_generations)
                        & (roots.owners == initial_owners))
        value_ok = (t.isfinite(x).all(-1) & t.isfinite(observed.viewer_is_actor).all(-1)
                    & t.isfinite(observed.terminal_value) & (observed.viewer_is_actor[:, 0] == 1)
                    & ~observed.terminal & (observed.terminal_value == 0) & same_version
                    & valid_after & actions_after)
        errors.bitwise_or_(t.where(live, observed.errors | after.errors | checked_after.errors | action_errors_after, 0))
        errors.bitwise_or_((live & ~value_ok).to(t.int32) * METADATA_ERROR)
        self._rule_choices.copy_(chosen)
        self._explored.zero_()
        self._exploration_errors.zero_()
        selected_choices = chosen
        if self.explore_first_six:
            eligible = (roots.active & (errors == 0) & (self._steps < 6)
                        & (source_phases == 1)).contiguous()
            draw = self._exploration_rng.integers_device(counts, eligible,
                max_retries=self.exploration_max_retries)
            self._exact(draw.indices, t.int64, (self.rows,))
            self._exact(draw.errors, t.int32, (self.rows,))
            self._exploration_errors.copy_(t.where(eligible, draw.errors, 0))
            errors.bitwise_or_(self._exploration_errors)
            invalid_draw = eligible & (draw.errors == 0) & ((draw.indices < 0) | (draw.indices >= counts))
            errors.bitwise_or_(invalid_draw.to(t.int32) * METADATA_ERROR)
            self._explored.copy_(eligible & (errors == 0))
            selected_choices = t.where(self._explored, draw.indices, chosen)
        complete = roots.active & (errors == 0)
        actions = t.where(complete, selected_choices, -1)
        policy = ((self._action_ids[None, :] == actions[:, None]) & complete[:, None]).to(t.float32)
        # Keep raw action refs, ordering, leases, candidates and type codes.
        # Only replace the teacher decoder's deliberately empty observation.
        decision = replace(decision, observations=x, errors=errors)
        zero = t.zeros(self.rows, dtype=t.int64, device=self.device)
        self._unfinished.zero_()
        self._result = DeviceGumbelResult(decision, actions, policy, t.zeros_like(policy),
            t.zeros_like(policy, dtype=t.int32), zero, zero.clone(), complete, errors,
            zero.clone(), t.zeros((self.rows, 10), dtype=t.int64, device=self.device),
            t.zeros((), dtype=t.int64, device=self.device))

    @property
    def unfinished_device(self):
        return self._unfinished

    def tick(self):
        self._unfinished.zero_()

    def result_device(self):
        if self._result is None:
            raise RuntimeError('Rule controller not initialized')
        return self._result

    @property
    def rule_choices_device(self):
        """Borrowed original raw rule indices, including inactive/error rows."""
        return self._rule_choices

    @property
    def explored_device(self):
        return self._explored

    @property
    def exploration_errors_device(self):
        return self._exploration_errors

    @property
    def exploration_state_device(self):
        return None if self._exploration_rng is None else self._exploration_rng.state

    def close(self):
        if not self._closed:
            if self._exploration_rng is not None:
                self._exploration_rng.close()
            self._closed = True
