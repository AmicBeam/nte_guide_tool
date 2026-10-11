"""Bounded CUDA-resident numerical trajectory sink for DeviceGameLoop.

This is a component, not a collector, closed driver, or readiness receipt.
It stores chosen action indices, compact legal prefixes, and terminal signs.
Initial root states and raw action dictionaries stay outside the tape.
"""
from dataclasses import dataclass
from typing import Any

from ..batched_duel import gpu_seven_candidates as encoder
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from .compact_device_candidates import CompactDeviceCandidates, _TOKEN_TENSORS


MASK_ERROR, OWNER_ERROR, GEN_ERROR, STEP_ERROR = 1, 2, 4, 8
NONFINITE_ERROR, PI_ERROR, BUDGET_ERROR, CAPACITY_ERROR = 16, 32, 64, 128
FINISH_ERROR = 256
CAVEAT = (
    'Chosen action indices, versions and seeds are a numerical sink only. '
    'They do not prove exact private replay or full action archival; '
    'immutable original state and raw actions must be supplied by a qualified driver later.'
)
_PER_ACTION = ('int_tokens', 'float_tokens', 'item_flags', 'mulligan_cards',
               'order_cards', 'select_cards', 'extra_ap')


@dataclass(frozen=True)
class DeviceTapeResult:
    record_count: Any
    legal_count: Any
    learning_eligible: Any
    unfinished: Any
    z: Any
    errors: Any
    winners: Any
    complete: Any
    truncated: Any
    caveat: str = CAVEAT


class DeviceTrainingTape:
    """Fixed-capacity device sink for begin_device/append_device/finish_device."""

    supports_full_duel_rules = False

    def __init__(self, compact, rules, *, rows, max_records, max_legal_entries, max_bytes,
                 observation_dim, information_keys=None, device='cuda',
                 allow_cpu_reference=False, allow_synthetic=False):
        import torch
        if type(compact) is not CompactDeviceCandidates:
            raise TypeError('CompactDeviceCandidates token layout required')
        if type(allow_cpu_reference) is not bool or type(allow_synthetic) is not bool:
            raise TypeError('Explicit boolean CPU/synthetic controls required')
        for name, value in (('rows', rows), ('max_records', max_records),
                            ('max_legal_entries', max_legal_entries),
                            ('max_bytes', max_bytes), ('observation_dim', observation_dim)):
            if type(value) is not int or value < 1:
                raise ValueError('Positive bounded ' + name + ' required')
        for name in ('metadata_device', 'value_observation_device'):
            if not callable(getattr(rules, name, None)):
                raise ValueError('Missing device rule interface: ' + name)
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit training-tape oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host loop fallback')
        if self.device.type == 'cpu' and allow_synthetic is not True:
            raise ValueError('CPU training tape requires explicit synthetic fixture binding')
        if compact.device != self.device or compact.batch_size != rows:
            raise ValueError('Compact token layout must match the tape device and rows')
        if compact.max_candidates < 1:
            raise ValueError('Nonempty compact legal width required')
        self.compact = compact
        self.rules = rules
        self.information_keys = information_keys
        self.allow_cpu_reference = allow_cpu_reference is True
        self.allow_synthetic = allow_synthetic is True
        self.rows = rows
        self.max_records = max_records
        self.max_legal_entries = max_legal_entries
        self.max_bytes = max_bytes
        self.observation_dim = observation_dim
        self.max_actions = compact.max_candidates
        self.cand_dim = compact.cand_dim
        record_n, legal_n = max_records + 1, max_legal_entries + 1
        required = self._estimate_bytes(compact, rows, record_n, legal_n, observation_dim)
        if required > max_bytes:
            raise MemoryError('Declared device training tape exceeds byte budget')
        z = dict(device=self.device)
        self._row_ids = torch.arange(rows, dtype=torch.int64, **z)
        self._record_ids = torch.arange(record_n, dtype=torch.int64, **z)
        self._legal_ids = torch.arange(legal_n, dtype=torch.int64, **z)
        self._action_ids = torch.arange(self.max_actions, dtype=torch.int64, **z)
        self._record_sink = torch.tensor(max_records, dtype=torch.int64, **z)
        self._legal_sink = torch.tensor(max_legal_entries, dtype=torch.int64, **z)
        self._record_count = torch.zeros((), dtype=torch.int64, **z)
        self._legal_count = torch.zeros((), dtype=torch.int64, **z)
        self._job_seeds = torch.zeros(rows, dtype=torch.int64, **z)
        self._owners = torch.full((rows,), -1, dtype=torch.int64, **z)
        self._generations = torch.zeros(rows, dtype=torch.int64, **z)
        self._expected_step = torch.zeros(rows, dtype=torch.int64, **z)
        self._begun = torch.zeros(rows, dtype=torch.bool, **z)
        self._policy_ids = torch.zeros((rows, 2), dtype=torch.int64, **z)
        self._aux_ids = torch.zeros((rows, 2), dtype=torch.int64, **z)
        self._train_sides = torch.zeros((rows, 2), dtype=torch.bool, **z)
        self._game_errors = torch.zeros(rows, dtype=torch.int32, **z)
        self._game_winners = torch.full((rows,), -2, dtype=torch.int64, **z)
        self._game_steps = torch.zeros(rows, dtype=torch.int64, **z)
        self._game_complete = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_truncated = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_finished = torch.zeros(rows, dtype=torch.bool, **z)
        self._append_errors = torch.zeros(rows, dtype=torch.int32, **z)
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
        self._rec_errors = torch.zeros(record_n, dtype=torch.int32, **z)
        self._rec_turn = torch.zeros(record_n, dtype=compact.tokens.turn_counts.dtype, **z)
        self._z = torch.zeros(record_n, dtype=torch.float32, **z)
        self._eligible = torch.zeros(record_n, dtype=torch.bool, **z)
        self._unfinished = torch.zeros(record_n, dtype=torch.bool, **z)
        self._obs = torch.zeros((record_n, observation_dim), dtype=torch.float32, **z)
        self._value_obs = torch.zeros_like(self._obs)
        self._value_ctx = torch.zeros((record_n, 1), dtype=torch.float32, **z)
        self._packed = {}
        self._pad = {}
        for name in _PER_ACTION:
            source = getattr(compact.tokens, name)
            packed = torch.zeros((legal_n,) + source.shape[2:], dtype=source.dtype, **z)
            if name in ('int_tokens', 'mulligan_cards', 'order_cards', 'select_cards'):
                packed.fill_(-1)
                self._pad[name] = torch.tensor(-1, dtype=source.dtype, **z)
            else:
                self._pad[name] = torch.tensor(0, dtype=source.dtype, **z)
            self._packed[name] = packed
        self._packed_pi = torch.zeros(legal_n, dtype=torch.float32, **z)
        self._packed_visit_pi = torch.zeros_like(self._packed_pi)
        self._packed_visits = torch.zeros(legal_n, dtype=torch.int32, **z)
        self._ready = False
        self.allocated_bytes = self._allocated_bytes()
        if self.allocated_bytes > max_bytes:
            raise MemoryError('Declared device training tape exceeds byte budget')

    def _estimate_bytes(self, compact, rows, record_n, legal_n, observation_dim):
        total = rows * 120 + record_n * (120 + observation_dim * 8) + legal_n * 16
        total += (rows + record_n + legal_n + compact.max_candidates) * 8
        for name in _PER_ACTION:
            source = getattr(compact.tokens, name)
            n = 1
            for size in source.shape[2:]:
                n *= int(size)
            total += legal_n * n * int(source.element_size())
        return int(total)

    def _allocated_bytes(self):
        tensors = [
            self._row_ids, self._record_ids, self._legal_ids, self._action_ids,
            self._record_sink, self._legal_sink, self._record_count, self._legal_count,
            self._job_seeds, self._owners, self._generations, self._expected_step, self._begun,
            self._policy_ids, self._aux_ids, self._train_sides, self._game_errors,
            self._game_winners, self._game_steps, self._game_complete, self._game_truncated,
            self._game_finished, self._append_errors, self._rec_game, self._rec_owner,
            self._rec_seed, self._rec_actor, self._rec_step, self._rec_version, self._rec_chosen,
            self._rec_count, self._rec_offset, self._rec_requested, self._rec_completed,
            self._rec_errors, self._rec_turn, self._z, self._eligible, self._unfinished,
            self._obs, self._value_obs, self._value_ctx, self._packed_pi, self._packed_visit_pi,
            self._packed_visits,
        ]
        tensors.extend(self._packed.values())
        tensors.extend(self._pad.values())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _vector(self, value, dtype, shape):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.shape != shape or not value.is_contiguous()):
            raise ValueError('Exact bound-device training-tape tensor metadata required')

    def _tokens(self, result):
        candidates = getattr(getattr(result, 'decision', None), 'candidates', None)
        if type(candidates) is CompactDeviceCandidates:
            if (candidates.device != self.device or candidates.batch_size != self.rows
                    or candidates.max_candidates != self.max_actions):
                raise ValueError('Append compact tokens must match the tape layout')
            return candidates.tokens
        raise ValueError('Append requires the actual compact decision tokens')

    def begin_device(self, roots, policy_ids, aux_ids, job_seeds, train_sides, active):
        t = self.t
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        self._vector(roots.slots, t.int64, (self.rows,))
        self._vector(roots.generations, t.int64, (self.rows,))
        self._vector(roots.owners, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(job_seeds, t.int64, (self.rows,))
        self._vector(policy_ids, t.int64, (self.rows, 2))
        self._vector(aux_ids, t.int64, (self.rows, 2))
        self._vector(train_sides, t.bool, (self.rows, 2))
        self._vector(active, t.bool, (self.rows,))
        self._record_count.zero_()
        self._legal_count.zero_()
        self._expected_step.zero_()
        self._game_errors.zero_()
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
        invalid = active & ((roots.owners < 0) | (roots.generations < 1) | (roots.slots < 0))
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

    def append_device(self, roots, actors, steps, search_result, active):
        t = self.t
        if not self._ready:
            raise RuntimeError('begin_device must allocate the cohort before append_device')
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
        self._vector(roots.slots, t.int64, (self.rows,))
        self._vector(roots.generations, t.int64, (self.rows,))
        self._vector(roots.owners, t.int64, (self.rows,))
        self._vector(roots.active, t.bool, (self.rows,))
        self._vector(actors, t.int64, (self.rows,))
        self._vector(steps, t.int64, (self.rows,))
        self._vector(active, t.bool, (self.rows,))
        decision = search_result.decision
        self._vector(search_result.actions, t.int64, (self.rows,))
        self._vector(search_result.policy, t.float32, (self.rows, self.max_actions))
        self._vector(search_result.visit_policy, t.float32, (self.rows, self.max_actions))
        self._vector(search_result.visits, t.int32, (self.rows, self.max_actions))
        self._vector(search_result.requested, t.int64, (self.rows,))
        self._vector(search_result.completed, t.int64, (self.rows,))
        self._vector(search_result.errors, t.int32, (self.rows,))
        self._vector(decision.counts, t.int64, (self.rows,))
        self._vector(decision.valid, t.bool, (self.rows, self.max_actions))
        self._vector(decision.observations, t.float32, (self.rows, self.observation_dim))
        self._vector(decision.errors, t.int32, (self.rows,))
        tokens = self._tokens(search_result)
        for name in _TOKEN_TENSORS:
            source = getattr(tokens, name)
            layout = getattr(self.compact.tokens, name)
            if source.shape != layout.shape or source.dtype != layout.dtype or source.device != self.device:
                raise ValueError('Stable compact token layout required: ' + name)
        meta = self.rules.metadata_device(roots)
        viewers = 1 - actors.clamp(0, 1)
        value = self.rules.value_observation_device(roots, viewers)
        self._vector(meta.actors, t.int64, (self.rows,))
        self._vector(meta.versions, t.int64, (self.rows,))
        self._vector(meta.errors, t.int32, (self.rows,))
        self._vector(value.observations, t.float32, (self.rows, self.observation_dim))
        self._vector(value.viewer_is_actor, t.float32, (self.rows, 1))
        self._vector(value.errors, t.int32, (self.rows,))
        counts = decision.counts
        prefix = self._action_ids[None, :] < counts[:, None]
        mask_ok = ((counts >= 1) & (counts <= self.max_actions)
                   & (decision.valid == prefix).all(-1)
                   & (tokens.mask == prefix).all(-1)
                   & (tokens.counts == counts))
        owner_ok = (self._begun & roots.active & (roots.errors == 0)
                    & (roots.owners == self._owners) & (self._owners >= 0))
        gen_ok = ((roots.generations > 0) & (meta.errors == 0) & (decision.errors == 0)
                  & (value.errors == 0) & (search_result.errors == 0)
                  & (actors == meta.actors) & (actors >= 0) & (actors <= 1)
                  & (value.viewer_is_actor[:, 0] == 0))
        states = getattr(decision, 'states', None)
        if isinstance(states, DeviceStateBatch):
            if states.pool is not roots.pool:
                raise ValueError('Decision and physical state arenas must match')
            gen_ok = (gen_ok & states.active & (states.errors == 0)
                      & (states.slots == roots.slots)
                      & (states.generations == roots.generations)
                      & (states.owners == roots.owners))
        elif not self.allow_synthetic:
            raise ValueError('Formal tape requires the decision state lease')
        step_ok = (steps == self._expected_step) & (steps >= 0)
        chosen = search_result.actions
        chosen_ok = (chosen >= 0) & (chosen < counts)
        finite_ok = (t.isfinite(decision.observations).all(-1)
                     & t.isfinite(value.observations).all(-1)
                     & t.isfinite(value.viewer_is_actor).all(-1)
                     & t.isfinite(tokens.turn_counts)
                     & t.isfinite(tokens.float_tokens).all(-1).all(-1)
                     & t.isfinite(tokens.item_flags).all(-1).all(-1)
                     & t.isfinite(tokens.extra_ap).all(-1))
        pi, visit_pi, visits = search_result.policy, search_result.visit_policy, search_result.visits
        illegal = ~prefix
        pi_mass = t.where(prefix, pi, 0.).sum(-1)
        visit_mass = t.where(prefix, visit_pi, 0.).sum(-1)
        visit_count = t.where(prefix, visits, 0).sum(-1)
        expected_visit_pi = visits.to(t.float32) / visit_count.clamp_min(1)[:, None]
        pi_ok = (t.isfinite(pi).all(-1) & (pi >= 0).all(-1) & t.isfinite(visit_pi).all(-1)
                 & (visit_pi >= 0).all(-1) & (visits >= 0).all(-1)
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
        admitted = active & (errors == 0)
        need_r = admitted.long().sum()
        need_l = (admitted.long() * counts.clamp(min=0, max=self.max_actions)).sum()
        fits = ((self._record_count + need_r) <= self.max_records) & (
            (self._legal_count + need_l) <= self.max_legal_entries)
        overflow = (~fits) & (need_r > 0)
        errors = errors | (admitted & overflow).to(t.int32) * CAPACITY_ERROR
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
        self._rec_game.index_copy_(0, rec_dest, self._row_ids)
        self._rec_owner.index_copy_(0, rec_dest, roots.owners)
        self._rec_seed.index_copy_(0, rec_dest, self._job_seeds)
        self._rec_actor.index_copy_(0, rec_dest, actors)
        self._rec_step.index_copy_(0, rec_dest, steps)
        self._rec_version.index_copy_(0, rec_dest, meta.versions)
        self._rec_chosen.index_copy_(0, rec_dest, chosen)
        self._rec_count.index_copy_(0, rec_dest, counts)
        self._rec_offset.index_copy_(0, rec_dest, legal_dest)
        self._rec_requested.index_copy_(0, rec_dest, search_result.requested)
        self._rec_completed.index_copy_(0, rec_dest, search_result.completed)
        self._rec_errors.index_copy_(0, rec_dest, errors)
        self._rec_turn.index_copy_(0, rec_dest, tokens.turn_counts)
        self._obs.index_copy_(0, rec_dest, decision.observations)
        self._value_obs.index_copy_(0, rec_dest, value.observations)
        self._value_ctx.index_copy_(0, rec_dest, value.viewer_is_actor)
        flat = packed_dest.reshape(-1)
        for name in _PER_ACTION:
            source = getattr(tokens, name)
            packed = self._packed[name]
            packed.index_copy_(0, flat, source.reshape((-1,) + packed.shape[1:]))
        self._packed_pi.index_copy_(0, flat, pi.reshape(-1))
        self._packed_visit_pi.index_copy_(0, flat, visit_pi.reshape(-1))
        self._packed_visits.index_copy_(0, flat, visits.reshape(-1))
        self._record_count.add_(commit.long().sum())
        self._legal_count.add_(safe_count.sum())
        self._expected_step.add_(commit.long())
        self._generations.copy_(t.where(commit, roots.generations, self._generations))
        self._append_errors.copy_(t.where(active, errors, self._append_errors))
        self._unfinished.copy_(self._record_ids < self._record_count)
        return t.where(active, errors, t.zeros_like(errors))

    def finish_device(self, roots, winners, steps, complete, errors):
        t = self.t
        if not self._ready:
            raise RuntimeError('begin_device must allocate the cohort before finish_device')
        if not isinstance(roots, DeviceStateBatch):
            raise TypeError('Checked device root states required')
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
        self._game_finished.logical_or_(active & (finish_err == 0))
        self._game_complete.logical_or_(qualify)
        self._game_truncated.logical_or_(truncated)
        self._game_winners.copy_(t.where(qualify | truncated, winners, self._game_winners))
        self._game_steps.copy_(t.where(active, steps, self._game_steps))
        self._game_errors.bitwise_or_(t.where(active, errors | finish_err, 0))
        occupied = self._record_ids < self._record_count
        game = self._rec_game.clamp(0, self.rows - 1)
        actor = self._rec_actor.clamp(0, 1)
        side_z = t.where(self._game_winners == 0, 1, t.where(self._game_winners == 1, -1, 0))
        rec_z = t.where(actor == 0, side_z[game],
                        t.where(self._game_winners[game] >= 0, -side_z[game], 0)).to(t.float32)
        eligible = (occupied & self._game_complete[game] & (self._game_errors[game] == 0)
                    & (self._rec_errors == 0) & self._train_sides[game, actor])
        self._eligible.copy_(eligible)
        self._z.copy_(t.where(eligible, rec_z, 0))
        self._unfinished.copy_(occupied & ~self._game_finished[game])
        return t.where(active, finish_err, t.zeros_like(finish_err))

    def result_device(self):
        if not self._ready:
            raise RuntimeError('No device training-tape cohort initialized')
        return DeviceTapeResult(
            self._record_count, self._legal_count, self._eligible, self._unfinished, self._z,
            self._append_errors | self._game_errors, self._game_winners, self._game_complete,
            self._game_truncated,
        )

    def _export_forbidden(self):
        if self.device.type == 'cuda' and self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Host export is only valid after the bounded batch')

    def _gather_tokens(self, rec_index):
        t = self.t
        local = rec_index.shape[0]
        offsets = self._rec_offset.index_select(0, rec_index)
        counts = self._rec_count.index_select(0, rec_index)
        take = self._action_ids[None, :] < counts[:, None]
        idx = t.where(take, offsets[:, None] + self._action_ids[None, :], self._legal_sink)
        idx = idx.clamp(0, self.max_legal_entries).reshape(-1)
        values = {}
        for name in _PER_ACTION:
            packed = self._packed[name]
            gathered = packed.index_select(0, idx).reshape((local, self.max_actions) + packed.shape[1:])
            view = take.reshape((local, self.max_actions) + (1,) * (gathered.ndim - 2))
            values[name] = t.where(view, gathered, self._pad[name])
        values['turn_counts'] = self._rec_turn.index_select(0, rec_index)
        values['counts'] = counts
        values['mask'] = take
        return encoder.RawCandidateTokens(**values, request_ids=()), take, counts, offsets

    def numerical_rows_after_batch(self, *, include_frozen=False):
        self._export_forbidden()
        if type(include_frozen) is not bool:
            raise TypeError('Explicit boolean frozen-opponent audit export required')
        t = self.t
        if self.device.type == 'cuda':
            t.cuda.synchronize()
        used = int(self._record_count.item())
        if used < 1:
            return []
        rec_index = t.arange(used, dtype=t.int64, device=self.device)
        game = self._rec_game[:used]
        completed = (self._game_complete[game] & (self._game_errors[game] == 0)
                     & (self._rec_errors[:used] == 0))
        rec_index = rec_index.masked_select(completed if include_frozen else self._eligible[:used])
        if rec_index.numel() == 0:
            return []
        rows = []
        winner = self._game_winners[self._rec_game.clamp(0, self.rows - 1)]
        actual_z = t.where(winner == -1, 0,
                           t.where(winner == self._rec_actor, 1, -1)).to(t.float32)
        scratch = self.compact.scratch_rows
        for start in range(0, rec_index.numel(), scratch):
            stop = min(start + scratch, rec_index.numel())
            local_ids = rec_index[start:stop]
            tokens, take, counts, offsets = self._gather_tokens(local_ids)
            packed, enc_err = encoder.encode_tokens_device(
                tokens, device=self.device, allow_cpu_reference=self.allow_cpu_reference,
                template_table=self.compact.template_table,
            )
            dest = (offsets[:, None] + self._action_ids[None, :]).clamp(0, self.max_legal_entries).reshape(-1)
            pi = self._packed_pi.index_select(0, dest).reshape(local_ids.shape[0], self.max_actions)
            visit = self._packed_visit_pi.index_select(0, dest).reshape(local_ids.shape[0], self.max_actions)
            visits = self._packed_visits.index_select(0, dest).reshape(local_ids.shape[0], self.max_actions)
            pi = t.where(take, pi, 0.)
            visit = t.where(take, visit, 0.)
            visits = t.where(take, visits, 0)
            host = dict(
                x=self._obs.index_select(0, local_ids).detach().cpu(),
                value_x=self._value_obs.index_select(0, local_ids).detach().cpu(),
                value_ctx=self._value_ctx.index_select(0, local_ids).detach().cpu(),
                c=packed.candidates.detach().cpu(),
                pi=pi.detach().cpu(),
                visit_pi=visit.detach().cpu(),
                visits=visits.detach().cpu(),
                selected=self._rec_chosen.index_select(0, local_ids).detach().cpu(),
                z=actual_z.index_select(0, local_ids).detach().cpu(),
                eligible=self._eligible.index_select(0, local_ids).detach().cpu(),
                actor=self._rec_actor.index_select(0, local_ids).detach().cpu(),
                step=self._rec_step.index_select(0, local_ids).detach().cpu(),
                version=self._rec_version.index_select(0, local_ids).detach().cpu(),
                job_seed=self._rec_seed.index_select(0, local_ids).detach().cpu(),
                owner=self._rec_owner.index_select(0, local_ids).detach().cpu(),
                requested=self._rec_requested.index_select(0, local_ids).detach().cpu(),
                completed=self._rec_completed.index_select(0, local_ids).detach().cpu(),
                counts=counts.detach().cpu(),
                enc_err=enc_err.detach().cpu(),
            )
            for i in range(local_ids.shape[0]):
                count = int(host['counts'][i])
                rows.append(dict(
                    x=host['x'][i].numpy(),
                    value_x=host['value_x'][i].numpy(),
                    value_ctx=float(host['value_ctx'][i, 0]),
                    c=host['c'][i, :count].numpy(),
                    pi=host['pi'][i, :count].numpy(),
                    visit_pi=host['visit_pi'][i, :count].numpy(),
                    visits=host['visits'][i, :count].numpy(),
                    selected=int(host['selected'][i]),
                    z=float(host['z'][i]),
                    actor=int(host['actor'][i]),
                    step=int(host['step'][i]),
                    version=int(host['version'][i]),
                    job_seed=int(host['job_seed'][i]),
                    owner=int(host['owner'][i]),
                    requested=int(host['requested'][i]),
                    completed=int(host['completed'][i]),
                    encode_error=int(host['enc_err'][i]),
                    learning_eligible=bool(host['eligible'][i]),
                    caveat=CAVEAT,
                ))
        return rows

    def value_rows_after_batch(self):
        """Both trained viewers at each completed decision, including opponent turns.

        Policy eligibility depends on the acting side. Value eligibility depends
        on the viewer, so a frozen actor must not hide the learning opponent's
        already recorded public observation. Export happens after the whole batch.
        """
        self._export_forbidden()
        t = self.t
        if self.device.type == 'cuda':
            t.cuda.synchronize()
        used = int(self._record_count.item())
        if used == 0:
            return []
        game = self._rec_game[:used]
        valid = (self._game_complete[game] & (self._game_errors[game] == 0)
                 & (self._rec_errors[:used] == 0))
        ids = t.arange(used, dtype=t.int64, device=self.device).masked_select(valid)
        host = {name: value.index_select(0, ids).detach().cpu() for name, value in (
            ('x', self._obs), ('other_x', self._value_obs), ('other_ctx', self._value_ctx),
            ('actor', self._rec_actor), ('step', self._rec_step), ('version', self._rec_version),
            ('seed', self._rec_seed), ('owner', self._rec_owner), ('game', self._rec_game))}
        train = self._train_sides.detach().cpu()
        winners = self._game_winners.detach().cpu()
        rows = []
        for i in range(ids.numel()):
            actor, game_id = int(host['actor'][i]), int(host['game'][i])
            winner = int(winners[game_id])
            for viewer in (actor, 1 - actor):
                if not bool(train[game_id, viewer]):
                    continue
                own = viewer == actor
                rows.append(dict(
                    x=host['x' if own else 'other_x'][i].numpy(),
                    value_actor=1 if own else float(host['other_ctx'][i, 0]),
                    actor=viewer,
                    step=int(host['step'][i]), version=int(host['version'][i]),
                    job_seed=int(host['seed'][i]), owner=int(host['owner'][i]),
                    z=0.0 if winner == -1 else (1.0 if winner == viewer else -1.0),
                    learning_eligible=True, caveat=CAVEAT,
                ))
        return rows

    def unfinished_records_after_batch(self):
        self._export_forbidden()
        t = self.t
        if self.device.type == 'cuda':
            t.cuda.synchronize()
        used = int(self._record_count.item())
        if used < 1:
            return []
        rec_index = t.arange(used, dtype=t.int64, device=self.device)
        rec_index = rec_index.masked_select(self._unfinished[:used])
        if rec_index.numel() == 0:
            return []
        host = dict(
            selected=self._rec_chosen.index_select(0, rec_index).detach().cpu(),
            actor=self._rec_actor.index_select(0, rec_index).detach().cpu(),
            step=self._rec_step.index_select(0, rec_index).detach().cpu(),
            version=self._rec_version.index_select(0, rec_index).detach().cpu(),
            job_seed=self._rec_seed.index_select(0, rec_index).detach().cpu(),
            owner=self._rec_owner.index_select(0, rec_index).detach().cpu(),
            counts=self._rec_count.index_select(0, rec_index).detach().cpu(),
        )
        rows = []
        for i in range(rec_index.numel()):
            rows.append(dict(
                selected=int(host['selected'][i]),
                actor=int(host['actor'][i]),
                step=int(host['step'][i]),
                version=int(host['version'][i]),
                job_seed=int(host['job_seed'][i]),
                owner=int(host['owner'][i]),
                counts=int(host['counts'][i]),
                learning_eligible=False,
                caveat=CAVEAT,
            ))
        return rows
