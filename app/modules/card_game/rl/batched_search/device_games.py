"""Closed nested device WHILE for physical games; not full-rules evidence.

One outer CUDA WHILE drives physical games. Each turn embeds a complete
DeviceGumbelSearch WHILE through a separate CudaConditionalWhile. This
driver is not a qualified training executor and does not mint readiness.
"""
from dataclasses import dataclass
from typing import Any

from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .device_search import METADATA_ERROR, RANDOM_ERROR, RULE_ERROR


SINK_ERROR, GUARD_ERROR = 128, 256


@dataclass(frozen=True)
class DeviceGameResult:
    states: DeviceStateBatch
    winners: Any
    steps: Any
    complete: Any
    truncated: Any
    errors: Any
    searched_count: Any


class DeviceGameLoop:
    """Physical-game driver whose hot path stays inside nested device graphs."""

    supports_full_duel_rules = False

    def __init__(self, rules, search, sink, world_seed_provider, *, max_decisions,
                 random_plan=None, device='cuda', allow_cpu_reference=False,
                 allow_synthetic=False, outer_loop=None, inner_loop=None,
                 outer_max_ticks=None, inner_max_ticks=None, seconds=60, root_guide=None, search_audit_sink=None):
        import torch
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError('Positive bounded physical decision count required')
        if type(allow_synthetic) is not bool or type(allow_cpu_reference) is not bool:
            raise TypeError('Explicit boolean CPU/synthetic controls required')
        if not callable(getattr(world_seed_provider, 'derive', None)):
            raise ValueError('Device world-seed provider with derive() required')
        for name in ('begin_device', 'append_device', 'finish_device'):
            if not callable(getattr(sink, name, None)):
                raise ValueError('Missing device trajectory sink: ' + name)
        for name in ('metadata_device', 'step_device', 'clean_device', 'release_device'):
            if not callable(getattr(rules, name, None)):
                raise ValueError('Missing device rule interface: ' + name)
        if getattr(search, 'supports_full_duel_rules', None) is True:
            raise ValueError('Search core must not claim full duel-rule readiness')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit game-loop oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host loop fallback')
        if self.device.type == 'cpu' and allow_synthetic is not True:
            raise ValueError('CPU game loop requires explicit synthetic fixture binding')
        if search.device != self.device or search.rows < 1:
            raise ValueError('Search cohort must match the game-loop device and rows')
        self._uses_random = search.noise or getattr(search, 'opponent_mode', 'rule') == 'policy'
        if self._uses_random and (random_plan is None or not callable(getattr(random_plan, 'reset_device', None))):
            raise ValueError('Exact RandomPlan.reset_device required for stochastic search')
        if outer_max_ticks is None:
            outer_max_ticks = max_decisions + 1
        if inner_max_ticks is None:
            inner_max_ticks = 10000
        for value, name in ((outer_max_ticks, 'outer_max_ticks'), (inner_max_ticks, 'inner_max_ticks')):
            if type(value) is not int or value < 1:
                raise ValueError('Positive bounded ' + name + ' required')
        if root_guide is not None and not callable(getattr(root_guide, 'capture_result', None)):
            raise TypeError('Device root guide must provide capture_result')
        self.root_guide = root_guide
        if search_audit_sink is not None:
            if search_audit_sink is sink:
                raise ValueError('Original-search audit must use a distinct sink')
            for name in ('begin_device', 'append_device', 'finish_device'):
                if not callable(getattr(search_audit_sink, name, None)):
                    raise TypeError('Original-search audit sink missing ' + name)
        self.search_audit_sink = search_audit_sink
        self.rules, self.search, self.sink = rules, search, sink
        self._counts_as_search = getattr(search, 'counts_as_search', True)
        if type(self._counts_as_search) is not bool:
            raise TypeError('Explicit boolean search accounting required')
        self.world_seed_provider, self.random_plan = world_seed_provider, random_plan
        self.max_decisions = max_decisions
        self.allow_synthetic = allow_synthetic
        self.rows = search.rows
        self._inner_max_ticks = inner_max_ticks
        self._outer = outer_loop
        self._inner = inner_loop
        if self.device.type == 'cuda' and not allow_synthetic:
            from .cuda_conditional_graph import CudaConditionalWhile
            if self._outer is None:
                self._outer = CudaConditionalWhile(device=str(self.device),
                    max_ticks=outer_max_ticks, seconds=seconds)
            if self._inner is None:
                self._inner = CudaConditionalWhile(device=str(self.device),
                    max_ticks=inner_max_ticks, seconds=seconds)
        for runner in (self._outer, self._inner):
            if runner is not None and (runner.graph is not None or runner._loop_built):
                raise RuntimeError('Device loop is already frozen')
        z = dict(device=self.device)
        self._rows = torch.arange(self.rows, dtype=torch.int64, **z)
        self._slots = torch.zeros(self.rows, dtype=torch.int64, **z)
        self._generations = torch.ones_like(self._slots)
        self._owners = torch.zeros_like(self._slots)
        self._present = torch.zeros(self.rows, dtype=torch.bool, **z)
        self._errors = torch.zeros(self.rows, dtype=torch.int32, **z)
        self._steps = torch.zeros(self.rows, dtype=torch.int64, **z)
        self._winners = torch.full((self.rows,), -2, dtype=torch.int64, **z)
        self._finished = torch.zeros(self.rows, dtype=torch.bool, **z)
        self._truncated = torch.zeros_like(self._finished)
        self._searched = torch.zeros(self.rows, dtype=torch.int64, **z)
        self._actors = torch.zeros(self.rows, dtype=torch.int64, **z)
        self._source_versions = torch.zeros_like(self._actors)
        self._search_mask = torch.zeros(self.rows, dtype=torch.bool, **z)
        self._sink_done = torch.zeros(self.rows, dtype=torch.bool, **z)
        self._unfinished = torch.zeros(self.rows, dtype=torch.uint8, **z)
        self._job_seeds = torch.zeros(self.rows, dtype=torch.int64, **z)
        self._policy_ids = torch.zeros((self.rows, 2), dtype=torch.int64, **z)
        self._aux_ids = torch.zeros((self.rows, 2), dtype=torch.int64, **z)
        self._train_sides = torch.zeros((self.rows, 2), dtype=torch.bool, **z)
        self._pool = None
        self._ready = False

    def _vector(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.shape != shape or not value.is_contiguous()):
            raise ValueError('Exact bound-device game tensor metadata required')

    def _physical(self, active):
        return DeviceStateBatch(self._pool, self._slots, self._generations, self._owners,
                                active, self._errors)

    def _error(self, raw, mask, code):
        self._errors.bitwise_or_(((raw != 0) & mask).to(self.t.int32) * code)

    def _release_checked(self, states, mask):
        request = DeviceStateBatch(states.pool, states.slots, states.generations,
            states.owners, mask & states.active, self.t.zeros_like(self._errors))
        self.rules.release_device(request)
        self._error(request.errors, mask, RULE_ERROR)

    def _duplicate_leases(self, slots, generations, active):
        t = self.t
        by_activity = (~active).to(t.int64).argsort(stable=True)
        by_generation = by_activity[generations[by_activity].argsort(stable=True)]
        perm = by_generation[slots[by_generation].argsort(stable=True)]
        sorted_slots, sorted_gens = slots[perm], generations[perm]
        active_sorted = active[perm]
        pair = ((sorted_slots[1:] == sorted_slots[:-1]) & (sorted_gens[1:] == sorted_gens[:-1])
                & active_sorted[1:] & active_sorted[:-1])
        marked = t.zeros(self.rows, dtype=t.bool, device=self.device)
        if pair.numel():
            marked[:-1] |= pair
            marked[1:] |= pair
        out = t.empty(self.rows, dtype=t.bool, device=self.device)
        out[perm] = marked
        return out

    def _cross_alias(self, states, mask):
        t = self.t
        sorted_slots, order = self._slots.sort()
        at = t.searchsorted(sorted_slots, states.slots).clamp(0, self.rows - 1)
        root = order[at]
        other = ((states.slots == self._slots[root]) & (root != self._rows)
                 & (states.generations == self._generations[root]) & mask)
        duplicates = self._duplicate_leases(states.slots, states.generations, mask & states.active)
        return other | duplicates

    def _update_predicate(self):
        self._unfinished.copy_((self._present & ~self._finished & ~self._truncated
                                & (self._errors == 0)).to(self.t.uint8))

    def _build_context(self):
        if (self.device.type == 'cuda' and not self.allow_synthetic
                and not self.t.cuda.is_current_stream_capturing()):
            raise RuntimeError('Formal device game loop must be inside a captured graph')

    @property
    def unfinished_device(self):
        return self._unfinished

    @property
    def steps_device(self):
        """Borrowed counters for a physical trace hook before commit."""
        return self._steps

    @property
    def source_versions_device(self):
        """Physical decision revisions retained across hypothetical search."""
        return self._source_versions

    def initialize(self, roots, job_seeds, policy_index_by_side, auxiliary_index_by_side,
                   train_sides):
        t = self.t
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        for value in (roots.slots, roots.generations, roots.owners, job_seeds):
            self._vector(value, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(roots.errors, t.int32, (self.rows,))
        self._vector(policy_index_by_side, t.int64, (self.rows, 2))
        self._vector(auxiliary_index_by_side, t.int64, (self.rows, 2))
        self._vector(train_sides, t.bool, (self.rows, 2))
        self._pool = roots.pool
        self._slots.copy_(roots.slots)
        self._generations.copy_(roots.generations)
        self._owners.copy_(roots.owners)
        self._present.copy_(roots.active)
        self._errors.copy_(roots.errors)
        self._job_seeds.copy_(job_seeds)
        self._policy_ids.copy_(policy_index_by_side)
        self._aux_ids.copy_(auxiliary_index_by_side)
        self._train_sides.copy_(train_sides)
        self._steps.zero_()
        self._winners.fill_(-2)
        self._finished.zero_()
        self._truncated.zero_()
        self._searched.zero_()
        self._actors.zero_()
        self._source_versions.zero_()
        self._search_mask.zero_()
        self._sink_done.zero_()
        allocated = (roots.slots >= 0) & (roots.generations >= 0) & (roots.owners >= 0)
        duplicate = self._duplicate_leases(roots.slots, roots.generations, allocated)
        self._error(~allocated | duplicate, self._present, METADATA_ERROR)
        admitted = self._present & (self._errors == 0)
        begin = self.sink.begin_device(self._physical(admitted), self._policy_ids,
            self._aux_ids, self._job_seeds, self._train_sides, admitted)
        self._error(begin, admitted, SINK_ERROR)
        if self.search_audit_sink is not None:
            audited = self.search_audit_sink.begin_device(self._physical(admitted), self._policy_ids,
                self._aux_ids, self._job_seeds, self._train_sides, admitted)
            self._error(audited, admitted, SINK_ERROR)
        self._update_predicate()
        self._ready = True

    def _start_search(self):
        t = self.t
        live = self._present & ~self._finished & ~self._truncated & (self._errors == 0)
        meta = self.rules.metadata_device(self._physical(live))
        self._vector(meta.versions.contiguous(), t.int64, (self.rows,))
        self._error(meta.errors, live, RULE_ERROR)
        self._error(meta.versions < 0, live, METADATA_ERROR)
        self._source_versions.copy_(t.where(live, meta.versions, self._source_versions))
        invalid_actor = live & ~meta.finished & ((meta.actors < 0) | (meta.actors > 1))
        self._error(invalid_actor, live, METADATA_ERROR)
        self._error(live & meta.finished & ((meta.winners < -1) | (meta.winners > 1)),
                    live, METADATA_ERROR)
        self._actors.copy_(t.where(live, meta.actors.clamp(0, 1), self._actors))
        search_mask = (live & ~meta.finished & (self._errors == 0)
                       & (self._steps < self.max_decisions))
        derived = self.world_seed_provider.derive(self._job_seeds, self._steps,
                                                  self._actors, search_mask)
        if (not isinstance(derived.limbs, t.Tensor) or derived.limbs.shape != (self.rows, 5)
                or derived.limbs.dtype != t.int64 or derived.limbs.device != self.device
                or not isinstance(derived.errors, t.Tensor)
                or derived.errors.shape != (self.rows,) or derived.errors.dtype != t.int32):
            raise ValueError('World-seed derive must return int64[B,5] limbs and int32[B] errors')
        self._error(derived.errors, search_mask, RANDOM_ERROR)
        search_mask = search_mask & (derived.errors == 0) & (self._errors == 0)
        if self._uses_random:
            random_errors = self.random_plan.reset_device(self._steps, self._actors, search_mask)
            self._error(random_errors, search_mask, RANDOM_ERROR)
            search_mask = search_mask & (random_errors == 0) & (self._errors == 0)
        self._search_mask.copy_(search_mask)
        roots = DeviceStateBatch(self._pool, self._slots, self._generations, self._owners,
                                 search_mask, self._errors)
        self.search.initialize(roots, self._actors, derived.limbs, self._policy_ids, self._aux_ids)

    def _fake_search_while(self):
        for _ in range(self._inner_max_ticks):
            if not bool(self.search.unfinished_device.any()):
                return
            self.search.tick()
        leftover = self.search.unfinished_device != 0
        self._error(leftover, leftover, GUARD_ERROR)

    def _apply_inner_guard(self):
        if self._inner is None:
            return
        failed = (self._inner.errors != 0).reshape(())
        stop = failed & (self._unfinished != 0)
        self._errors.bitwise_or_(stop.to(self.t.int32) * GUARD_ERROR)
        self._unfinished.masked_fill_(failed, 0)

    def _commit_turn(self):
        t = self.t
        result = self.search.result_device()
        searched = self._search_mask
        if self.search_audit_sink is not None:
            audit_active = searched & result.complete & (result.errors == 0) & (self._errors == 0)
            audited = self.search_audit_sink.append_device(self._physical(self._present),
                self._actors, self._steps, result, audit_active)
            self._error(audited, audit_active, SINK_ERROR)
        if self.root_guide is not None:
            result = self.root_guide.capture_result(self._physical(searched),
                self._actors, result, self._policy_ids, searched)
        self._errors.bitwise_or_(t.where(searched, result.errors, 0))
        incomplete = searched & ~result.complete & (self._errors == 0)
        self._error(incomplete, incomplete, GUARD_ERROR)
        ready = (searched & result.complete & (result.completed == result.requested)
                 & (result.errors == 0) & (self._errors == 0))
        if self._counts_as_search:
            self._searched.add_((searched & result.complete).long())
        sink_err = self.sink.append_device(self._physical(self._present), self._actors,
                                           self._steps, result, ready)
        self._error(sink_err, ready, SINK_ERROR)
        commit = ready & (sink_err == 0) & (self._errors == 0)
        old = self._physical(commit)
        advanced = self.rules.step_device(old, self._actors, result.decision, result.actions,
                                          physical=True)
        if advanced.pool is not self._pool:
            raise ValueError('Physical results must share the stable checked state arena')
        self._error(advanced.errors, commit, RULE_ERROR)
        self._error((advanced.owners != self._owners) | ~advanced.active
                    | (advanced.slots < 0) | (advanced.generations < 0), commit, METADATA_ERROR)
        alias = self._cross_alias(advanced, commit)
        self._error(alias, commit, METADATA_ERROR)
        clean_mask = commit & (self._errors == 0)
        clean_input = DeviceStateBatch(advanced.pool, advanced.slots, advanced.generations,
            advanced.owners, clean_mask, advanced.errors)
        cleaned = self.rules.clean_device(clean_input)
        if cleaned.pool is not self._pool:
            raise ValueError('Cleaned physical results must share the checked state arena')
        self._error(cleaned.errors, clean_mask, RULE_ERROR)
        self._error((cleaned.owners != self._owners) | ~cleaned.active
                    | (cleaned.slots < 0) | (cleaned.generations < 0), clean_mask, METADATA_ERROR)
        clean_alias = self._cross_alias(cleaned, clean_mask)
        # Cleaning must not return a different game's still-live apply result.
        sorted_mid, order = advanced.slots.sort()
        at = t.searchsorted(sorted_mid, cleaned.slots).clamp(0, self.rows - 1)
        mid_row = order[at]
        clean_alias |= (clean_mask & clean_mask[mid_row] & (mid_row != self._rows)
                        & (cleaned.slots == advanced.slots[mid_row])
                        & (cleaned.generations == advanced.generations[mid_row]))
        self._error(clean_alias, clean_mask, METADATA_ERROR)
        success = clean_mask & (self._errors == 0)
        distinct_old = (cleaned.slots != self._slots) | (cleaned.generations != self._generations)
        distinct_mid = ((cleaned.slots != advanced.slots)
                        | (cleaned.generations != advanced.generations))
        advanced_distinct = ((advanced.slots != self._slots)
                             | (advanced.generations != self._generations))
        self._release_checked(cleaned, clean_mask & ~success & distinct_old & distinct_mid
                              & ~clean_alias & cleaned.active & (cleaned.owners == self._owners))
        self._release_checked(advanced, commit & advanced_distinct & (~success | distinct_mid)
                              & ~alias & advanced.active & (advanced.owners == self._owners))
        self._release_checked(old, success & distinct_old)
        self._slots.copy_(t.where(success, cleaned.slots, self._slots))
        self._generations.copy_(t.where(success, cleaned.generations, self._generations))
        self._steps.add_(success.long())
        from .device_action_leases import release_actions_device
        action_errors = release_actions_device(result.decision, searched, self._owners,
                                                protected=(old, advanced, cleaned, self._physical(self._present)))
        self._error(action_errors, searched, RULE_ERROR)
        live = self._present & (self._errors == 0)
        meta = self.rules.metadata_device(self._physical(live))
        self._error(meta.errors, live, RULE_ERROR)
        invalid_terminal = meta.finished & ((meta.winners < -1) | (meta.winners > 1))
        self._error(invalid_terminal, live, METADATA_ERROR)
        valid_terminal = meta.finished & (self._errors == 0)
        self._finished.logical_or_(valid_terminal)
        self._winners.copy_(t.where(valid_terminal, meta.winners, self._winners))
        self._truncated.logical_or_(self._present & ~self._finished & (self._errors == 0)
                                    & (self._steps >= self.max_decisions))
        newly = (self._present & ~self._sink_done
                 & (self._finished | self._truncated | (self._errors != 0)))
        complete = self._finished & (self._errors == 0) & ~self._truncated
        finish_err = self.sink.finish_device(self._physical(newly), self._winners, self._steps,
                                             complete, self._errors)
        self._error(finish_err, newly, SINK_ERROR)
        if self.search_audit_sink is not None:
            audited = self.search_audit_sink.finish_device(self._physical(newly),
                self._winners, self._steps, complete, self._errors)
            self._error(audited, newly, SINK_ERROR)
        self._sink_done.logical_or_(newly)
        self._update_predicate()
        self._apply_inner_guard()

    def tick(self):
        self._build_context()
        if not self._ready:
            raise RuntimeError('Initialize the physical game cohort first')
        capturing = (self.device.type == 'cuda' and self.t.cuda.is_current_stream_capturing())
        if self._inner is not None and capturing:
            self._inner.capture_loop(self.search.unfinished_device, self.search.tick,
                                     initialize=self._start_search)
        else:
            if self.device.type == 'cuda' and not self.allow_synthetic:
                raise RuntimeError('Formal nested search requires a captured inner WHILE')
            self._start_search()
            self._fake_search_while()
        self._commit_turn()

    def capture(self, roots, job_seeds, policy_index_by_side, auxiliary_index_by_side,
                train_sides):
        if self._outer is None or self._inner is None:
            raise RuntimeError('CUDA nested loops required for production capture')
        if (self._outer.graph is not None or self._outer._loop_built
                or self._inner.graph is not None or self._inner._loop_built):
            raise RuntimeError('Device loop is already frozen')
        if self._counts_as_search:
            self.search.statistics.bind_capture_streams(self._outer.body_stream, self._inner.body_stream)
        if self.random_plan is not None and callable(getattr(self.random_plan, 'bind_capture_streams', None)):
            self.random_plan.bind_capture_streams(self._outer.body_stream, self._inner.body_stream)
        from contextlib import nullcontext
        scope = self.search.statistics.capture_scope() if self._counts_as_search else nullcontext()
        with scope:
            self._outer.capture(self._unfinished, self.tick, initialize=lambda:
                self.initialize(roots, job_seeds, policy_index_by_side,
                                auxiliary_index_by_side, train_sides))

    def replay(self):
        if self._outer is None or self._outer.graph is None:
            raise RuntimeError('No frozen device WHILE graph')
        self._outer.replay()

    def result_device(self):
        if not self._ready:
            raise RuntimeError('No device game cohort initialized')
        errors = self._errors
        if self._outer is not None:
            failed = (self._outer.errors != 0).reshape(())
            errors = errors.clone()
            leftover = self._unfinished != 0
            errors.bitwise_or_((leftover & failed).to(self.t.int32) * GUARD_ERROR)
        complete = self._finished & (errors == 0) & ~self._truncated & self._present
        if self._outer is not None:
            failed = (self._outer.errors != 0).reshape(())
            complete = complete & ~(failed & ~self._finished)
        states = DeviceStateBatch(self._pool, self._slots, self._generations, self._owners,
                                  self._present, errors)
        return DeviceGameResult(states, self._winners, self._steps, complete, self._truncated,
                                errors, self._searched)

    def close(self):
        for runner in (self._outer, self._inner):
            if runner is not None:
                runner.close()
