"""Graph-capturable device-resident Gumbel statistics.

This is a bounded FP64 numerical primitive for complete-action search, not a
rule engine and not a full-game runtime. CUDA is the default device; CPU is
allowed only with explicit allow_cpu_reference for numerical tests.

Preserved formulas, matching recovery_search / search_stability / nodes.py:
- natural_wdl or opt-in mctx_default completed-Q, including mixed-Q completion
- full improved policy over every legal action (no top-k truncation)
- root sequential-halving by visit-target eligibility
- maximum-visit final selection plus roots_covered
- interior policy-minus-visits selection
- fixed-observer backup of one leaf value along a path
- deterministic first-index argmax ties
- ragged legal widths via boolean padding masks

Hot-path inputs and outputs are device tensors: row slots, generations,
owners, priors, values, visits, totals, legal masks and error flags. Invalid
rows set error flags and do not mutate unrelated or inactive slots. Host
loops over a global simulation or depth budget belong to the caller.

Integration notes for the primary agent:
- Construct one DeviceStatisticsPool(capacity, max_actions, device='cuda',
  max_batch=1600, max_path_len=max_depth). allocated_bytes is fixed.
- write_nodes occupies free slots and returns the generation to store with
  each node identity. Interior nodes of one root must share that root owner.
- visit_schedule(min(n_legal, k, sims), sims) stays on the host. Each
  simulation calls select_root with that visit target over the complete
  legal set; this pool never truncates the action axis.
- CUDA graphs may capture select_*, completed_q, improved_policy, backup,
  clear_visits and set_value. Pad captured batches to a static row count.
  Do not put per-root Python loops, .item/.cpu/.numpy, or stream sync in
  the graph.
- backup is atomic per batch row: stale generation, illegal action, owner
  mismatch, non-finite or out-of-range value, or duplicate root owner
  rejects that simulation only.
- Do not add process rewards. Leaf values are already in the observer
  frame and are added as-is.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from typing import Any


@dataclass(frozen=True)
class DeviceWriteBatch:
    generations: Any
    error: Any


@dataclass(frozen=True)
class DeviceErrorBatch:
    error: Any


@dataclass(frozen=True)
class DeviceQBatch:
    completed_q: Any
    error: Any


@dataclass(frozen=True)
class DevicePolicyBatch:
    policy: Any
    error: Any


@dataclass(frozen=True)
class DeviceChoiceBatch:
    actions: Any
    error: Any


@dataclass(frozen=True)
class DeviceFinalBatch:
    actions: Any
    roots_covered: Any
    error: Any


@dataclass(frozen=True)
class DeviceSnapshotBatch:
    prior: Any
    visits: Any
    total: Any
    value: Any
    legal: Any
    generations: Any
    owners: Any
    error: Any


class DeviceStatisticsPool:
    """Batched device Gumbel statistics with occupancy and generation tokens."""

    def __init__(self, capacity, max_actions, *, device='cuda', q_scale='natural_wdl',
                 allow_cpu_reference=False, max_batch=None, max_path_len=16):
        import torch
        if type(capacity) is not int or type(max_actions) is not int or min(capacity, max_actions) < 1:
            raise ValueError('Positive node and full-action capacities required')
        if max_batch is None:
            max_batch = capacity
        if type(max_batch) is not int or not 1 <= max_batch <= capacity:
            raise ValueError('max_batch must be an integer in 1..capacity')
        if type(max_path_len) is not int or max_path_len < 1:
            raise ValueError('Positive path depth capacity required')
        if q_scale not in ('natural_wdl', 'mctx_default'):
            raise ValueError('Explicit natural_wdl or mctx_default Q profile required')
        self.torch = torch
        self.capacity = capacity
        self.max_actions = max_actions
        self.max_batch = max_batch
        self.max_path_len = max_path_len
        self.q_scale = q_scale
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference):
            raise ValueError('CUDA required; CPU is an explicit numerical oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU fallback')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._stream = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        self._owner_thread = threading.get_ident()
        self._capture_depth = 0
        self._capture_streams = frozenset()
        z = dict(device=self.device)
        self._sink = capacity
        rows = capacity + 1
        self._prior = torch.zeros((rows, max_actions), dtype=torch.float64, **z)
        self._total = torch.zeros_like(self._prior)
        self._visits = torch.zeros((rows, max_actions), dtype=torch.int32, **z)
        self._value = torch.zeros(rows, dtype=torch.float64, **z)
        self._legal = torch.zeros((rows, max_actions), dtype=torch.bool, **z)
        self._generation = torch.ones(rows, dtype=torch.int64, **z)
        self._owner = torch.zeros(rows, dtype=torch.int64, **z)
        self._occupied = torch.zeros(rows, dtype=torch.bool, **z)
        self._all_true = torch.ones(max_batch, dtype=torch.bool, **z)
        buffers = (self._prior, self._total, self._visits, self._value, self._legal,
                   self._generation, self._owner, self._occupied, self._all_true)
        self.allocated_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers)

    def _stream_check(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('Device statistics require their owning driver thread')
        if self._stream is not None and self.torch.cuda.current_stream(self.device) != self._stream:
            current = self.torch.cuda.current_stream(self.device)
            if (self._capture_depth and self.torch.cuda.is_current_stream_capturing()
                    and current.cuda_stream in self._capture_streams):
                return
            raise RuntimeError('Device statistics require the owning CUDA stream')

    def bind_capture_streams(self, parent, body):
        """Register the exact dependency-linked streams before graph capture."""
        t = self.torch
        if self.device.type != 'cuda' or threading.get_ident() != self._owner_thread:
            raise RuntimeError('CUDA capture streams require their owning driver')
        if self._capture_depth or t.cuda.is_current_stream_capturing():
            raise RuntimeError('Bind capture streams before static graph construction')
        for stream in (parent, body):
            if not isinstance(stream, t.cuda.Stream) or stream.device != self.device:
                raise ValueError('Exact same-device CUDA parent/body streams required')
        self._capture_streams = frozenset((parent.cuda_stream, body.cuda_stream))

    @contextmanager
    def capture_scope(self):
        """Allow a single driver's dependency-linked conditional graph streams.

        The scope covers static graph construction only. Eager calls on other
        streams remain rejected; it neither synchronizes nor reads GPU values.
        """
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('Only the owning driver can construct a capture')
        self._capture_depth += 1
        try:
            yield
        finally:
            self._capture_depth -= 1

    def _active(self, active, batch, name='active'):
        t = self.torch
        if active is None:
            return self._all_true[:batch]
        if not isinstance(active, t.Tensor) or active.device != self.device or active.dtype != t.bool or active.shape != (batch,):
            raise ValueError(name + ' must be a device bool mask matching the batch')
        return active

    def _batch_vec(self, tensor, dtype, name, ndim=1, tail=()):
        t = self.torch
        if not isinstance(tensor, t.Tensor) or tensor.device != self.device:
            raise ValueError(name + ' must already be on the bound device')
        if tensor.dtype != dtype:
            raise ValueError(name + ' dtype mismatch')
        if tensor.ndim != ndim or tensor.shape[0] < 1 or tensor.shape[0] > self.max_batch:
            raise ValueError(name + ' exceeds the bounded static batch')
        if ndim > 1 and tensor.shape[1:] != tuple(tail):
            raise ValueError(name + ' shape mismatch; candidates are not truncated')
        return tensor

    def _identity(self, slots, generations, active, *, require_occupied):
        t = self.torch
        slots = self._batch_vec(slots, t.int64, 'slots')
        generations = self._batch_vec(generations, t.int64, 'generations')
        if generations.shape != slots.shape:
            raise ValueError('Slot and generation batches must match')
        active = self._active(active, slots.shape[0])
        self._stream_check()
        in_range = (slots >= 0) & (slots < self.capacity)
        safe = slots.clamp(0, self.capacity - 1)
        occupied = self._occupied.index_select(0, safe)
        stored_gen = self._generation.index_select(0, safe)
        stored_owner = self._owner.index_select(0, safe)
        if require_occupied:
            bad = (~in_range) | (~occupied) | (stored_gen != generations)
        else:
            bad = (~in_range) | occupied
        error = active & bad
        return slots, generations, active, safe, occupied, stored_gen, stored_owner, error

    def _duplicates(self, keys, active):
        t = self.torch
        # Keep active entries adjacent within equal keys even if padded rows
        # carry the same owner/slot. Sorting keys alone can miss T/F/T repeats.
        by_activity = active.long().argsort(descending=True, stable=True)
        by_key = keys[by_activity].argsort(stable=True)
        perm = by_activity[by_key]
        sorted_keys = keys[perm]
        active_sorted = active[perm]
        pair = (sorted_keys[1:] == sorted_keys[:-1]) & active_sorted[1:] & active_sorted[:-1]
        marked = t.zeros(keys.shape[0], dtype=t.bool, device=self.device)
        marked[:-1] |= pair
        marked[1:] |= pair
        out = t.empty(keys.shape[0], dtype=t.bool, device=self.device)
        out[perm] = marked
        return out

    def _scatter_where(self, storage, safe, success, new_rows):
        dest = self.torch.where(success, safe, self.torch.full_like(safe, self._sink))
        storage.index_copy_(0, dest, new_rows)

    def _completed_q(self, prior, visits, total, value, legal):
        t = self.torch
        seen = (visits > 0) & legal
        q = total / visits.clamp_min(1).to(t.float64)
        mass = (prior * seen.to(t.float64)).sum(-1)
        weighted = (prior * seen.to(t.float64) * q).sum(-1)
        mean = t.where(mass > 0, weighted / mass.clamp_min(1e-300), value)
        visit_sum = visits.masked_fill(~legal, 0).to(t.float64).sum(-1)
        mixed = (value + visit_sum * mean) / (1 + visit_sum)
        q = t.where(seen, q, mixed[:, None])
        gain = 0.1 * (50 + visits.masked_fill(~legal, 0).max(-1).values.to(t.float64))
        range_error = (((q.abs() > 1 + 1e-6) | ~t.isfinite(q)) & legal).any(-1)
        if self.q_scale == 'mctx_default':
            low = q.masked_fill(~legal, float('inf')).min(-1).values
            high = q.masked_fill(~legal, -float('inf')).max(-1).values
            # Exclude padding before rescaling; an all-equal legal row is zero.
            q = (q - low[:, None]) / (high - low).clamp_min(1e-8)[:, None]
        return (q * gain[:, None]).masked_fill(~legal, 0.), range_error

    def _gather_stats(self, safe):
        return (self._prior.index_select(0, safe),
                self._visits.index_select(0, safe),
                self._total.index_select(0, safe),
                self._value.index_select(0, safe),
                self._legal.index_select(0, safe))

    def write_nodes(self, slots, owners, priors, values, legal_mask, active=None,
                    *, expected_generations=None):
        t = self.torch
        slots = self._batch_vec(slots, t.int64, 'slots')
        batch = slots.shape[0]
        owners = self._batch_vec(owners, t.int64, 'owners')
        values = self._batch_vec(values, t.float64, 'values')
        priors = self._batch_vec(priors, t.float64, 'priors', 2, (self.max_actions,))
        legal_mask = self._batch_vec(legal_mask, t.bool, 'legal_mask', 2, (self.max_actions,))
        if owners.shape != slots.shape or values.shape != slots.shape:
            raise ValueError('write_nodes row tensors must share the batch axis')
        active = self._active(active, batch)
        self._stream_check()
        in_range = (slots >= 0) & (slots < self.capacity)
        safe = slots.clamp(0, self.capacity - 1)
        occupied = self._occupied.index_select(0, safe)
        stored_gen = self._generation.index_select(0, safe)
        if expected_generations is None:
            # Tokenless creation is only a first-allocation setup operation.
            # Reused slots need an explicit lease to reject delayed writes.
            generation_ok = stored_gen == 1
        else:
            self._batch_vec(expected_generations, t.int64, 'expected_generations')
            if expected_generations.shape != slots.shape:
                raise ValueError('Expected node generations must match the batch')
            generation_ok = stored_gen == expected_generations
        duplicate = self._duplicates(safe, active & in_range)
        mass = priors.masked_fill(~legal_mask, 0.).sum(-1)
        close = (mass - 1.).abs() <= (1e-12 + 1e-10 * mass.abs())
        prior_ok = (t.isfinite(priors).all(-1) & (priors >= 0).all(-1)
                    & (priors.masked_fill(legal_mask, 0.) == 0).all(-1)
                    & close & legal_mask.any(-1))
        value_ok = t.isfinite(values) & (values.abs() <= 1 + 1e-6)
        error = active & ((~in_range) | occupied | duplicate | (~prior_ok) | (~value_ok)
                          | ~generation_ok | (owners < 0))
        success = active & ~error
        zeros_prior = t.zeros((batch, self.max_actions), dtype=t.float64, device=self.device)
        zeros_visits = t.zeros((batch, self.max_actions), dtype=t.int32, device=self.device)
        self._scatter_where(self._prior, safe, success, priors)
        self._scatter_where(self._total, safe, success, zeros_prior)
        self._scatter_where(self._visits, safe, success, zeros_visits)
        self._scatter_where(self._value, safe, success, values)
        self._scatter_where(self._legal, safe, success, legal_mask)
        self._scatter_where(self._owner, safe, success, owners)
        self._scatter_where(self._occupied, safe, success, t.ones(batch, dtype=t.bool, device=self.device))
        generations = self._generation.index_select(0, safe)
        return DeviceWriteBatch(generations, error)

    def release_nodes(self, slots, generations, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        error = error | (active & self._duplicates(safe, active & ~error))
        success = active & ~error
        batch = slots.shape[0]
        zeros_prior = t.zeros((batch, self.max_actions), dtype=t.float64, device=self.device)
        zeros_visits = t.zeros((batch, self.max_actions), dtype=t.int32, device=self.device)
        false_legal = t.zeros((batch, self.max_actions), dtype=t.bool, device=self.device)
        self._scatter_where(self._prior, safe, success, zeros_prior)
        self._scatter_where(self._total, safe, success, zeros_prior)
        self._scatter_where(self._visits, safe, success, zeros_visits)
        self._scatter_where(self._value, safe, success, t.zeros(batch, dtype=t.float64, device=self.device))
        self._scatter_where(self._legal, safe, success, false_legal)
        self._scatter_where(self._owner, safe, success, t.zeros(batch, dtype=t.int64, device=self.device))
        self._scatter_where(self._occupied, safe, success, t.zeros(batch, dtype=t.bool, device=self.device))
        self._scatter_where(self._generation, safe, success, stored_gen + success.to(t.int64))
        return DeviceErrorBatch(error)

    def clear_visits(self, slots, generations, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        error = error | (active & self._duplicates(safe, active & ~error))
        success = active & ~error
        batch = slots.shape[0]
        zeros_prior = t.zeros((batch, self.max_actions), dtype=t.float64, device=self.device)
        zeros_visits = t.zeros((batch, self.max_actions), dtype=t.int32, device=self.device)
        self._scatter_where(self._total, safe, success, zeros_prior)
        self._scatter_where(self._visits, safe, success, zeros_visits)
        return DeviceErrorBatch(error)

    def set_value(self, slots, generations, values, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        values = self._batch_vec(values, t.float64, 'values')
        if values.shape != slots.shape:
            raise ValueError('set_value batch mismatch')
        error = error | (active & self._duplicates(safe, active & ~error))
        error = error | (active & (~t.isfinite(values) | (values.abs() > 1 + 1e-6)))
        success = active & ~error
        self._scatter_where(self._value, safe, success, values)
        return DeviceErrorBatch(error)

    def snapshot(self, slots, generations, active=None):
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        prior, visits, total, value, legal = self._gather_stats(safe)
        invalid = error | ~active
        return DeviceSnapshotBatch(
            prior.masked_fill(invalid[:, None], 0.),
            visits.masked_fill(invalid[:, None], 0),
            total.masked_fill(invalid[:, None], 0.),
            value.masked_fill(invalid, 0.),
            legal & ~invalid[:, None],
            stored_gen.masked_fill(invalid, 0),
            stored_owner.masked_fill(invalid, 0),
            error,
        )

    def completed_q(self, slots, generations, active=None):
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        prior, visits, total, value, legal = self._gather_stats(safe)
        q, range_error = self._completed_q(prior, visits, total, value, legal)
        error = error | (active & range_error)
        invalid = error | ~active
        return DeviceQBatch(q.masked_fill(invalid[:, None], 0.), error)

    def improved_policy(self, slots, generations, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        prior, visits, total, value, legal = self._gather_stats(safe)
        q, range_error = self._completed_q(prior, visits, total, value, legal)
        error = error | (active & range_error)
        logits = prior.clamp_min(1e-300).log() + q
        logits = logits.masked_fill(~legal, -float('inf'))
        policy = logits.softmax(-1)
        bad_policy = ~t.isfinite(policy).all(-1)
        error = error | (active & bad_policy)
        invalid = error | ~active
        policy = t.where(legal, policy, t.zeros_like(policy)).masked_fill(invalid[:, None], 0.)
        return DevicePolicyBatch(policy, error)

    def select_root(self, slots, generations, noise, visit_targets, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        noise = self._batch_vec(noise, t.float64, 'noise', 2, (self.max_actions,))
        visit_targets = self._batch_vec(visit_targets, t.int64, 'visit_targets')
        if noise.shape[0] != slots.shape[0] or visit_targets.shape != slots.shape:
            raise ValueError('Root noise and visit targets must match the node batch')
        prior, visits, total, value, legal = self._gather_stats(safe)
        q, range_error = self._completed_q(prior, visits, total, value, legal)
        error = error | (active & (range_error | ~t.isfinite(noise).all(-1) | (visit_targets < 0)))
        eligible = (visits == visit_targets[:, None]) & legal
        error = error | (active & ~eligible.any(-1))
        score = prior.clamp_min(1e-300).log() + noise + q
        invalid = error | ~active
        score = score.masked_fill((~eligible) | invalid[:, None], -float('inf'))
        actions = score.argmax(-1).masked_fill(invalid, 0)
        return DeviceChoiceBatch(actions, error)

    def select_interior(self, slots, generations, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        prior, visits, total, value, legal = self._gather_stats(safe)
        q, range_error = self._completed_q(prior, visits, total, value, legal)
        error = error | (active & range_error)
        logits = prior.clamp_min(1e-300).log() + q
        logits = logits.masked_fill(~legal, -float('inf'))
        policy = logits.softmax(-1)
        bad_policy = ~t.isfinite(policy).all(-1)
        error = error | (active & bad_policy)
        visit_sum = visits.masked_fill(~legal, 0).to(t.float64).sum(-1, keepdim=True)
        scores = policy - visits.to(t.float64) / (1 + visit_sum)
        invalid = error | ~active
        scores = scores.masked_fill((~legal) | invalid[:, None], -float('inf'))
        actions = scores.argmax(-1).masked_fill(invalid, 0)
        return DeviceChoiceBatch(actions, error)

    def select_final(self, slots, generations, noise, active=None):
        t = self.torch
        slots, generations, active, safe, occupied, stored_gen, stored_owner, error = self._identity(
            slots, generations, active, require_occupied=True)
        noise = self._batch_vec(noise, t.float64, 'noise', 2, (self.max_actions,))
        if noise.shape[0] != slots.shape[0]:
            raise ValueError('Final noise must match the node batch')
        prior, visits, total, value, legal = self._gather_stats(safe)
        q, range_error = self._completed_q(prior, visits, total, value, legal)
        error = error | (active & (range_error | ~t.isfinite(noise).all(-1)))
        max_visits = visits.masked_fill(~legal, 0).max(-1).values
        eligible = (visits == max_visits[:, None]) & legal
        error = error | (active & ~eligible.any(-1))
        score = prior.clamp_min(1e-300).log() + noise + q
        invalid = error | ~active
        score = score.masked_fill((~eligible) | invalid[:, None], -float('inf'))
        actions = score.argmax(-1).masked_fill(invalid, 0)
        covered = ((visits > 0) & legal).sum(-1).to(t.int64).masked_fill(invalid, 0)
        return DeviceFinalBatch(actions, covered, error)

    def backup(self, path_slots, path_generations, path_actions, path_valid, values, owners, active=None):
        t = self.torch
        if not isinstance(path_slots, t.Tensor) or path_slots.device != self.device:
            raise ValueError('path_slots must already be on the bound device')
        if (path_slots.dtype != t.int64 or path_slots.ndim != 2 or path_slots.shape[0] < 1
                or path_slots.shape[0] > self.max_batch
                or path_slots.shape[1] < 1 or path_slots.shape[1] > self.max_path_len):
            raise ValueError('Backup paths must use a static depth in 1..max_path_len')
        depth = path_slots.shape[1]
        batch = path_slots.shape[0]
        path_generations = self._batch_vec(path_generations, t.int64, 'path_generations', 2, (depth,))
        path_actions = self._batch_vec(path_actions, t.int64, 'path_actions', 2, (depth,))
        path_valid = self._batch_vec(path_valid, t.bool, 'path_valid', 2, (depth,))
        values = self._batch_vec(values, t.float64, 'values')
        owners = self._batch_vec(owners, t.int64, 'owners')
        if values.shape != (batch,) or owners.shape != (batch,):
            raise ValueError('One owner and observer value required per backup row')
        active = self._active(active, batch)
        self._stream_check()
        in_range = (path_slots >= 0) & (path_slots < self.capacity)
        safe = path_slots.clamp(0, self.capacity - 1)
        occupied = self._occupied[safe]
        gen_ok = self._generation[safe] == path_generations
        owner_ok = self._owner[safe] == owners[:, None]
        action_in = (path_actions >= 0) & (path_actions < self.max_actions)
        safe_actions = path_actions.clamp(0, self.max_actions - 1)
        legal_ok = self._legal[safe, safe_actions]
        step_ok = in_range & occupied & gen_ok & owner_ok & action_in & legal_ok
        step_fail = path_valid & ~step_ok
        value_ok = t.isfinite(values) & (values.abs() <= 1 + 1e-6)
        duplicate = self._duplicates(owners, active)
        error = active & (step_fail.any(-1) | (~value_ok) | duplicate)
        success = active & ~error
        write = path_valid & success[:, None]
        keys = (safe * self.max_actions + safe_actions).reshape(-1)
        visit_delta = write.reshape(-1).to(t.int32)
        # NaN*0 and Inf*0 are NaN; arithmetic masking would corrupt a live
        # node even when this row's backup was rejected.
        total_delta = t.where(write, values[:, None].expand(batch, depth), 0.).reshape(-1)
        self._visits.view(-1).index_add_(0, keys, visit_delta)
        self._total.view(-1).index_add_(0, keys, total_delta)
        return DeviceErrorBatch(error)

