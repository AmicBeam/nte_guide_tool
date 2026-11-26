"""Bounded CUDA archive of full-legal search pi/visit prefixes.

This is a light numerical sink for later exact GPU policy replay. It stores
ragged pi/visit_pi/visits and decision metadata only. Observations, compact
candidate tokens, raw action graphs, and immutable initial states stay outside.
A DeviceRawActionTape plus an immutable initial-state collector remain
mandatory; this module does not prove exact physical raw replay.
"""
from dataclasses import dataclass
from typing import Any

from ..batched_duel.device_rule_protocol import DeviceStateBatch


MASK_ERROR, OWNER_ERROR, GEN_ERROR, STEP_ERROR = 1, 2, 4, 8
NONFINITE_ERROR, PI_ERROR, BUDGET_ERROR, CAPACITY_ERROR = 16, 32, 64, 128
FINISH_ERROR = 256
_INT32_MAX = 2 ** 31
_MAX_ACTIONS = 16384
CAVEAT = (
    'Policy and visit prefixes are a bounded numerical archive only. '
    'They do not prove exact physical raw replay; a DeviceRawActionTape and '
    'an immutable initial-state collector are mandatory.'
)


@dataclass(frozen=True)
class DeviceReplayPolicyTapeResult:
    record_count: Any
    legal_count: Any
    errors: Any
    normal_complete: Any
    eligible: Any
    unfinished: Any
    z: Any
    winners: Any
    truncated: Any
    caveat: str = CAVEAT


def _layout_bytes(rows, record_n, legal_n, max_actions):
    b, r, l, a = rows, record_n, legal_n, max_actions
    return (
        4 * 8
        + b * (8 * 8 + 2 * 4 + 4)
        + b * 2 * (8 + 8 + 1)
        + r * (13 * 8 + 4 + 8 * 8 + 3 + 4 + 8)
        + l * (4 + 4 + 4 + 8)
        + a * 8
    )


class DeviceReplayPolicyTape:
    """Fixed-capacity device archive for begin_device/append_device/finish_device."""

    supports_full_duel_rules = False

    def __init__(self, rows, max_records, max_legal_entries, max_actions, max_bytes,
                 device='cuda', allow_cpu_reference=False, allow_synthetic=False):
        import torch
        if type(allow_cpu_reference) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit boolean CPU/synthetic controls required')
        for name, value in (('rows', rows), ('max_records', max_records),
                            ('max_legal_entries', max_legal_entries),
                            ('max_bytes', max_bytes)):
            if type(value) is not int or not 1 <= value < _INT32_MAX:
                raise ValueError('Positive bounded int32 ' + name + ' required')
        if type(max_actions) is not int or not 1 <= max_actions <= _MAX_ACTIONS:
            raise ValueError('Positive max_actions at most 16384 required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit replay-policy-tape oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host loop fallback')
        if self.device.type == 'cpu' and allow_synthetic is not True:
            raise ValueError('CPU replay policy tape requires explicit synthetic fixture binding')
        self.allow_cpu_reference = allow_cpu_reference is True
        self.allow_synthetic = allow_synthetic is True
        self.rows = rows
        self.max_records = max_records
        self.max_legal_entries = max_legal_entries
        self.max_actions = max_actions
        self.max_bytes = max_bytes
        record_n, legal_n = max_records + 1, max_legal_entries + 1
        required = _layout_bytes(rows, record_n, legal_n, max_actions)
        if required > max_bytes:
            raise MemoryError('Declared device replay policy tape exceeds byte budget')
        z = dict(device=self.device)
        self._record_count = torch.zeros((), dtype=torch.int64, **z)
        self._legal_count = torch.zeros((), dtype=torch.int64, **z)
        self._record_sink = torch.full((), max_records, dtype=torch.int64, **z)
        self._legal_sink = torch.full((), max_legal_entries, dtype=torch.int64, **z)
        self._row_ids = torch.arange(rows, dtype=torch.int64, **z)
        self._job_seeds = torch.zeros(rows, dtype=torch.int64, **z)
        self._owners = torch.full((rows,), -1, dtype=torch.int64, **z)
        self._generations = torch.zeros(rows, dtype=torch.int64, **z)
        self._expected_step = torch.zeros(rows, dtype=torch.int64, **z)
        self._last_version = torch.full((rows,), -1, dtype=torch.int64, **z)
        self._begun = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_errors = torch.zeros(rows, dtype=torch.int32, **z)
        self._game_winners = torch.full((rows,), -2, dtype=torch.int64, **z)
        self._game_steps = torch.zeros(rows, dtype=torch.int64, **z)
        self._game_complete = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_truncated = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_finished = torch.zeros(rows, dtype=torch.bool, **z)
        self._append_errors = torch.zeros(rows, dtype=torch.int32, **z)
        self._policy_ids = torch.zeros((rows, 2), dtype=torch.int64, **z)
        self._aux_ids = torch.zeros((rows, 2), dtype=torch.int64, **z)
        self._train_sides = torch.zeros((rows, 2), dtype=torch.bool, **z)
        self._record_ids = torch.arange(record_n, dtype=torch.int64, **z)
        self._legal_ids = torch.arange(legal_n, dtype=torch.int64, **z)
        self._action_ids = torch.arange(max_actions, dtype=torch.int64, **z)
        self._rec_game = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_owner = torch.full((record_n,), -1, dtype=torch.int64, **z)
        self._rec_seed = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_actor = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_step = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_version = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_chosen = torch.full((record_n,), -1, dtype=torch.int64, **z)
        self._rec_count = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_offset = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_requested = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_completed = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_policy = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_aux = torch.zeros(record_n, dtype=torch.int64, **z)
        self._rec_errors = torch.zeros(record_n, dtype=torch.int32, **z)
        self._rec_info = torch.zeros((record_n, 8), dtype=torch.int64, **z)
        self._rec_scope = torch.zeros(record_n, dtype=torch.bool, **z)
        self._z = torch.zeros(record_n, dtype=torch.float32, **z)
        self._eligible = torch.zeros(record_n, dtype=torch.bool, **z)
        self._unfinished = torch.zeros(record_n, dtype=torch.bool, **z)
        self._packed_pi = torch.zeros(legal_n, dtype=torch.float32, **z)
        self._packed_visit_pi = torch.zeros(legal_n, dtype=torch.float32, **z)
        self._packed_visits = torch.zeros(legal_n, dtype=torch.int32, **z)
        self._ready = False
        self.allocated_bytes = self._allocated_bytes()
        if self.allocated_bytes > max_bytes:
            raise MemoryError('Declared device replay policy tape exceeds byte budget')
        if self.allocated_bytes != required:
            raise RuntimeError('Replay policy tape byte inventory is inconsistent')

    def _persistent_tensors(self):
        return (
            self._record_count, self._legal_count, self._record_sink, self._legal_sink,
            self._row_ids, self._job_seeds, self._owners, self._generations, self._expected_step,
            self._last_version,
            self._begun, self._game_errors, self._game_winners, self._game_steps,
            self._game_complete, self._game_truncated, self._game_finished, self._append_errors,
            self._policy_ids, self._aux_ids, self._train_sides, self._record_ids, self._legal_ids,
            self._action_ids, self._rec_game, self._rec_owner, self._rec_seed, self._rec_actor,
            self._rec_step, self._rec_version, self._rec_chosen, self._rec_count, self._rec_offset,
            self._rec_requested, self._rec_completed, self._rec_policy, self._rec_aux,
            self._rec_errors, self._rec_info, self._rec_scope, self._z, self._eligible,
            self._unfinished, self._packed_pi, self._packed_visit_pi, self._packed_visits,
        )

    def _allocated_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self._persistent_tensors())

    def _vector(self, value, dtype, shape):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact bound-device replay-policy-tape tensor metadata required')

    def _require_ready(self):
        if not self._ready:
            raise RuntimeError('begin_device must allocate the cohort first')

    def begin_device(self, roots, policy_ids, aux_ids, job_seeds, train_sides, active):
        t = self.t
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        self._vector(roots.slots, t.int64, (self.rows,))
        self._vector(roots.generations, t.int64, (self.rows,))
        self._vector(roots.owners, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(roots.errors, t.int32, (self.rows,))
        self._vector(job_seeds, t.int64, (self.rows,))
        self._vector(policy_ids, t.int64, (self.rows, 2))
        self._vector(aux_ids, t.int64, (self.rows, 2))
        self._vector(train_sides, t.bool, (self.rows, 2))
        self._vector(active, t.bool, (self.rows,))
        self._record_count.zero_()
        self._legal_count.zero_()
        self._expected_step.zero_()
        self._last_version.fill_(-1)
        self._pool = roots.pool
        self._game_errors.copy_(roots.errors)
        self._append_errors.zero_()
        self._game_winners.fill_(-2)
        self._game_steps.zero_()
        self._game_complete.zero_()
        self._game_truncated.zero_()
        self._game_finished.zero_()
        self._eligible.zero_()
        self._unfinished.zero_()
        self._z.zero_()
        self._rec_chosen.fill_(-1)
        self._rec_scope.zero_()
        invalid = active & ((roots.owners < 0) | (roots.generations < 0) | (roots.slots < 0)
                            | ~roots.active | (roots.errors != 0))
        self._game_errors.bitwise_or_(invalid.to(t.int32) * OWNER_ERROR)
        begun = active & ~invalid
        self._begun.copy_(begun)
        self._job_seeds.copy_(t.where(begun, job_seeds, t.zeros_like(job_seeds)))
        self._owners.copy_(t.where(begun, roots.owners, t.full_like(roots.owners, -1)))
        self._generations.copy_(t.where(begun, roots.generations, 0))
        self._policy_ids.copy_(t.where(begun[:, None], policy_ids, 0))
        self._aux_ids.copy_(t.where(begun[:, None], aux_ids, 0))
        self._train_sides.copy_(t.where(begun[:, None], train_sides, False))
        self._ready = True
        return invalid.to(t.int32) * OWNER_ERROR

    def append_device(self, roots, actors, steps, versions, search_result, information_keys,
                      active):
        t = self.t
        self._require_ready()
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        if roots.pool is not self._pool:
            raise ValueError('Physical state arena changed during one replay cohort')
        self._vector(roots.slots, t.int64, (self.rows,))
        self._vector(roots.generations, t.int64, (self.rows,))
        self._vector(roots.owners, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(roots.errors, t.int32, (self.rows,))
        self._vector(actors, t.int64, (self.rows,))
        self._vector(steps, t.int64, (self.rows,))
        self._vector(versions, t.int64, (self.rows,))
        self._vector(information_keys, t.int64, (self.rows, 8))
        self._vector(active, t.bool, (self.rows,))
        decision = search_result.decision
        self._vector(search_result.actions, t.int64, (self.rows,))
        self._vector(search_result.policy, t.float32, (self.rows, self.max_actions))
        self._vector(search_result.visit_policy, t.float32, (self.rows, self.max_actions))
        self._vector(search_result.visits, t.int32, (self.rows, self.max_actions))
        self._vector(search_result.requested, t.int64, (self.rows,))
        self._vector(search_result.completed, t.int64, (self.rows,))
        self._vector(search_result.complete, t.bool, (self.rows,))
        self._vector(search_result.errors, t.int32, (self.rows,))
        self._vector(decision.counts, t.int64, (self.rows,))
        self._vector(decision.valid, t.bool, (self.rows, self.max_actions))
        self._vector(decision.errors, t.int32, (self.rows,))
        counts = decision.counts
        prefix = self._action_ids[None, :] < counts[:, None]
        mask_ok = ((counts >= 1) & (counts <= self.max_actions)
                   & (decision.valid == prefix).all(-1))
        owner_ok = (self._begun & roots.active & (roots.errors == 0)
                    & (roots.owners == self._owners) & (self._owners >= 0)
                    & (roots.slots >= 0))
        gen_ok = ((roots.generations >= 0) & (decision.errors == 0)
                  & (search_result.errors == 0) & search_result.complete
                  & (actors >= 0) & (actors <= 1) & (versions >= 0))
        states = getattr(decision, 'states', None)
        if isinstance(states, DeviceStateBatch):
            if states.pool is not roots.pool:
                raise ValueError('Decision and physical state arenas must match')
            self._vector(states.slots, t.int64, (self.rows,))
            self._vector(states.generations, t.int64, (self.rows,))
            self._vector(states.owners, t.int64, (self.rows,))
            self._vector(states.active, t.bool, (self.rows,))
            self._vector(states.errors, t.int32, (self.rows,))
            gen_ok = (gen_ok & states.active & (states.errors == 0)
                      & (states.slots == roots.slots)
                      & (states.generations == roots.generations)
                      & (states.owners == roots.owners))
        elif not self.allow_synthetic:
            raise ValueError('Formal tape requires the decision state lease')
        step_ok = ((steps == self._expected_step) & (steps >= 0)
                   & (versions > self._last_version))
        chosen = search_result.actions
        chosen_ok = (chosen >= 0) & (chosen < counts)
        pi, visit_pi, visits = search_result.policy, search_result.visit_policy, search_result.visits
        illegal = ~prefix
        pi_mass = t.where(prefix, pi, 0.).sum(-1)
        visit_mass = t.where(prefix, visit_pi, 0.).sum(-1)
        visit_count = t.where(prefix, visits, 0).sum(-1)
        expected_visit_pi = visits.to(t.float32) / visit_count.clamp_min(1)[:, None]
        finite_ok = t.isfinite(pi).all(-1) & t.isfinite(visit_pi).all(-1)
        pi_ok = (finite_ok & (pi >= 0).all(-1) & (visit_pi >= 0).all(-1) & (visits >= 0).all(-1)
                 & (t.where(illegal, pi, 0.).abs().sum(-1) == 0)
                 & (t.where(illegal, visit_pi, 0.).abs().sum(-1) == 0)
                 & (t.where(illegal, visits, 0).sum(-1) == 0)
                 & ((pi_mass - 1).abs() <= 1e-5)
                 & ((visit_mass - 1).abs() <= 1e-5)
                 & ((visit_pi - expected_visit_pi).abs() <= 1e-5).all(-1)
                 & chosen_ok)
        budget_ok = ((search_result.requested >= 1)
                     & (search_result.completed == search_result.requested)
                     & (visit_count == search_result.completed)
                     & (search_result.completed >= 1))
        errors = t.zeros(self.rows, dtype=t.int32, device=self.device)
        errors = errors | ((active & ~mask_ok).to(t.int32) * MASK_ERROR)
        errors = errors | ((active & ~owner_ok).to(t.int32) * OWNER_ERROR)
        errors = errors | ((active & ~gen_ok).to(t.int32) * GEN_ERROR)
        errors = errors | ((active & ~step_ok).to(t.int32) * STEP_ERROR)
        errors = errors | ((active & ~finite_ok).to(t.int32) * NONFINITE_ERROR)
        errors = errors | ((active & ~pi_ok).to(t.int32) * PI_ERROR)
        errors = errors | ((active & ~budget_ok).to(t.int32) * BUDGET_ERROR)
        errors = errors | ((active & self._game_finished).to(t.int32) * FINISH_ERROR)
        errors.bitwise_or_(t.where(active, self._game_errors, 0))
        admitted = active & (errors == 0)
        need_r = admitted.long().sum()
        need_l = (admitted.long() * counts.clamp(min=0, max=self.max_actions)).sum()
        fits = ((self._record_count + need_r) <= self.max_records) & (
            (self._legal_count + need_l) <= self.max_legal_entries)
        overflow = (~fits) & (need_r > 0)
        errors = errors | ((admitted & overflow).to(t.int32) * CAPACITY_ERROR)
        commit = admitted & ~overflow
        safe_count = t.where(commit, counts.clamp(0, self.max_actions), 0)
        rec_rank = commit.long().cumsum(0) - commit.long()
        rec_dest = t.where(commit, self._record_count + rec_rank, self._record_sink)
        legal_rank = safe_count.cumsum(0) - safe_count
        legal_dest = t.where(commit, self._legal_count + legal_rank, self._legal_sink)
        take = commit[:, None] & prefix
        packed_dest = t.where(take, legal_dest[:, None] + self._action_ids[None, :], self._legal_sink)
        packed_dest = packed_dest.clamp(0, self.max_legal_entries)
        rec_dest = rec_dest.clamp(0, self.max_records)
        actor_idx = actors.clamp(0, 1)
        self._rec_game.index_copy_(0, rec_dest, self._row_ids)
        self._rec_owner.index_copy_(0, rec_dest, roots.owners)
        self._rec_seed.index_copy_(0, rec_dest, self._job_seeds)
        self._rec_actor.index_copy_(0, rec_dest, actors)
        self._rec_step.index_copy_(0, rec_dest, steps)
        self._rec_version.index_copy_(0, rec_dest, versions)
        self._rec_chosen.index_copy_(0, rec_dest, chosen)
        self._rec_count.index_copy_(0, rec_dest, counts)
        self._rec_offset.index_copy_(0, rec_dest, legal_dest)
        self._rec_requested.index_copy_(0, rec_dest, search_result.requested)
        self._rec_completed.index_copy_(0, rec_dest, search_result.completed)
        self._rec_policy.index_copy_(0, rec_dest, self._policy_ids[self._row_ids, actor_idx])
        self._rec_aux.index_copy_(0, rec_dest, self._aux_ids[self._row_ids, actor_idx])
        self._rec_errors.index_copy_(0, rec_dest, errors)
        self._rec_info.index_copy_(0, rec_dest, information_keys)
        self._rec_scope.index_copy_(0, rec_dest, self._train_sides[self._row_ids, actor_idx])
        flat = packed_dest.reshape(-1)
        self._packed_pi.index_copy_(0, flat, pi.reshape(-1))
        self._packed_visit_pi.index_copy_(0, flat, visit_pi.reshape(-1))
        self._packed_visits.index_copy_(0, flat, visits.reshape(-1))
        self._record_count.add_(commit.long().sum())
        self._legal_count.add_(safe_count.sum())
        self._expected_step.add_(commit.long())
        self._last_version.copy_(t.where(commit, versions, self._last_version))
        self._generations.copy_(t.where(commit, roots.generations, self._generations))
        self._append_errors.bitwise_or_(t.where(active, errors, 0))
        self._game_errors.bitwise_or_(t.where(active, errors, 0))
        self._game_complete.logical_and_(self._game_errors == 0)
        record_failed = self._game_errors[self._rec_game.clamp(0, self.rows - 1)] != 0
        self._eligible.masked_fill_(record_failed, False)
        self._z.masked_fill_(record_failed, 0)
        self._unfinished.copy_(self._record_ids < self._record_count)
        return t.where(active, errors, t.zeros_like(errors))

    def finish_device(self, roots, winners, steps, complete, errors):
        t = self.t
        self._require_ready()
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        if roots.pool is not self._pool:
            raise ValueError('Physical state arena changed during one replay cohort')
        self._vector(roots.owners, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(winners, t.int64, (self.rows,))
        self._vector(steps, t.int64, (self.rows,))
        self._vector(complete, t.bool, (self.rows,))
        self._vector(errors, t.int32, (self.rows,))
        active = roots.active
        owner_ok = self._begun & (roots.owners == self._owners) & (self._owners >= 0)
        winner_ok = (~complete) | ((winners >= -1) & (winners <= 1))
        step_ok = (~complete) | (steps == self._expected_step)
        finish_err = ((active & ~owner_ok).to(t.int32) * OWNER_ERROR
                      | (active & ~winner_ok).to(t.int32) * FINISH_ERROR
                      | (active & ~step_ok).to(t.int32) * STEP_ERROR
                      | (active & self._game_finished).to(t.int32) * FINISH_ERROR)
        qualify = (active & complete & (errors == 0) & (finish_err == 0)
                   & (self._game_errors == 0) & (winners >= -1) & (winners <= 1))
        truncated = active & ~complete & (errors == 0) & (finish_err == 0)
        self._game_finished.logical_or_(active)
        self._game_complete.logical_or_(qualify)
        self._game_truncated.logical_or_(truncated)
        self._game_winners.copy_(t.where(qualify | truncated, winners, self._game_winners))
        self._game_steps.copy_(t.where(active, steps, self._game_steps))
        self._game_errors.bitwise_or_(t.where(active, errors | finish_err, 0))
        self._game_complete.logical_and_(self._game_errors == 0)
        occupied = self._record_ids < self._record_count
        game = self._rec_game.clamp(0, self.rows - 1)
        actor = self._rec_actor.clamp(0, 1)
        winner = self._game_winners[game]
        rec_z = t.where(winner == -1, 0, t.where(winner == actor, 1, -1)).to(t.float32)
        complete_row = occupied & self._game_complete[game] & (self._game_errors[game] == 0)
        complete_row = complete_row & (self._rec_errors == 0)
        self._z.copy_(t.where(complete_row, rec_z, 0))
        self._eligible.copy_(complete_row & self._rec_scope)
        self._unfinished.copy_(occupied & ~self._game_finished[game])
        return t.where(active, finish_err, t.zeros_like(finish_err))

    def result_device(self):
        self._require_ready()
        return DeviceReplayPolicyTapeResult(
            self._record_count, self._legal_count, self._append_errors | self._game_errors,
            self._game_complete, self._eligible, self._unfinished, self._z, self._game_winners,
            self._game_truncated,
        )

    def _export_forbidden(self):
        if self.device.type == 'cuda' and self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Host export is only valid after the bounded batch')

    def records_after_batch(self):
        """Optional CPU metadata+pi export after the whole batch. Not learning-ready."""
        self._require_ready()
        self._export_forbidden()
        t = self.t
        if self.device.type == 'cuda':
            t.cuda.synchronize()
        used = int(self._record_count.item())
        if used < 1:
            return []
        rec_index = t.arange(used, dtype=t.int64, device=self.device)
        game = self._rec_game[:used]
        audit = (self._game_complete[game] & (self._game_errors[game] == 0)
                 & (self._rec_errors[:used] == 0))
        rec_index = rec_index.masked_select(audit)
        if rec_index.numel() == 0:
            return []
        offsets = self._rec_offset.index_select(0, rec_index)
        counts = self._rec_count.index_select(0, rec_index)
        take = self._action_ids[None, :] < counts[:, None]
        idx = t.where(take, offsets[:, None] + self._action_ids[None, :], self._legal_sink)
        idx = idx.clamp(0, self.max_legal_entries).reshape(-1)
        local = rec_index.shape[0]
        pi = self._packed_pi.index_select(0, idx).reshape(local, self.max_actions)
        visit = self._packed_visit_pi.index_select(0, idx).reshape(local, self.max_actions)
        visits = self._packed_visits.index_select(0, idx).reshape(local, self.max_actions)
        pi = t.where(take, pi, 0.)
        visit = t.where(take, visit, 0.)
        visits = t.where(take, visits, 0)
        host = dict(
            pi=pi.detach().cpu(),
            visit_pi=visit.detach().cpu(),
            visits=visits.detach().cpu(),
            selected=self._rec_chosen.index_select(0, rec_index).detach().cpu(),
            actor=self._rec_actor.index_select(0, rec_index).detach().cpu(),
            game=self._rec_game.index_select(0, rec_index).detach().cpu(),
            owner=self._rec_owner.index_select(0, rec_index).detach().cpu(),
            job_seed=self._rec_seed.index_select(0, rec_index).detach().cpu(),
            step=self._rec_step.index_select(0, rec_index).detach().cpu(),
            version=self._rec_version.index_select(0, rec_index).detach().cpu(),
            requested=self._rec_requested.index_select(0, rec_index).detach().cpu(),
            completed=self._rec_completed.index_select(0, rec_index).detach().cpu(),
            counts=counts.detach().cpu(),
            policy_id=self._rec_policy.index_select(0, rec_index).detach().cpu(),
            aux_id=self._rec_aux.index_select(0, rec_index).detach().cpu(),
            information_keys=self._rec_info.index_select(0, rec_index).detach().cpu(),
            z=self._z.index_select(0, rec_index).detach().cpu(),
            learning_scope=self._rec_scope.index_select(0, rec_index).detach().cpu(),
            eligible=self._eligible.index_select(0, rec_index).detach().cpu(),
        )
        rows = []
        for i in range(local):
            count = int(host['counts'][i])
            rows.append(dict(
                pi=host['pi'][i, :count].numpy(),
                visit_pi=host['visit_pi'][i, :count].numpy(),
                visits=host['visits'][i, :count].numpy(),
                selected=int(host['selected'][i]),
                actor=int(host['actor'][i]),
                game=int(host['game'][i]),
                owner=int(host['owner'][i]),
                job_seed=int(host['job_seed'][i]),
                step=int(host['step'][i]),
                version=int(host['version'][i]),
                requested=int(host['requested'][i]),
                completed=int(host['completed'][i]),
                counts=count,
                policy_id=int(host['policy_id'][i]),
                aux_id=int(host['aux_id'][i]),
                information_keys=host['information_keys'][i].numpy(),
                z=float(host['z'][i]),
                learning_scope=bool(host['learning_scope'][i]),
                learning_eligible=bool(host['eligible'][i]),
                learning_ready=False,
                replay_verified=False,
                caveat=CAVEAT,
            ))
        return rows


