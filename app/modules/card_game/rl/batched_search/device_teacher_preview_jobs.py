"""Opt-in GPU worklist of canonical battle-preview jobs.

Compacts every relevant combat candidate from a typed legal mask and immutable
parent leases. Jobs only carry state/action identity; they never execute CPU
rules or fabricate preview scalars. Capacity overflow rejects the whole eligible
workload. This component is UNQUALIFIED and not a closed service.
"""
from dataclasses import dataclass
from typing import Any


QUALIFIED = False
CLOSED_SERVICE_READY = False
OPT_IN = True
ERR_MALFORMED, ERR_CAPACITY, ERR_DUPLICATE = 1, 2, 4
ERR_OUTSIDE, ERR_STALE, ERR_FOREIGN, ERR_NONFINITE, ERR_MISSING = 8, 16, 32, 64, 128

@dataclass(frozen=True)
class DeviceTeacherPreviewWorkload:
    parent_row: Any
    ordinal: Any
    owners: Any
    generations: Any
    slots: Any
    actors: Any
    row_workload: Any
    needed: Any
    job_count: Any
    overflow: Any
    errors: Any
    epoch: Any
    qualified: bool = False


class DeviceTeacherPreviewJobs:
    """Preallocated compact battle-preview worklist. UNQUALIFIED / opt-in."""

    qualified = False
    closed_service_ready = False
    opt_in = True

    def __init__(self, *, rows, max_actions, capacity, device='cuda',
                 allow_cpu_reference=False):
        import torch as t
        if type(allow_cpu_reference) is not bool:
            raise TypeError('Explicit allow_cpu_reference boolean required')
        if any(type(value) is not int or value < 1
               for value in (rows, max_actions, capacity)):
            raise ValueError('Positive bounded preview-job dimensions required')
        self.t = t
        self.device = t.device(device)
        self.rows, self.max_actions, self.capacity = rows, max_actions, capacity
        self.allow_cpu_reference = allow_cpu_reference is True
        self.module = None
        self.closed = False
        if self.device.type == 'cpu' and self.allow_cpu_reference:
            self._allocate()
            return
        if self.device.type != 'cuda' or not t.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU is an explicit job oracle only')
        if self.device.index is None:
            self.device = t.device('cuda', t.cuda.current_device())
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare preview jobs before capture')
        self._allocate()

    def _allocate(self):
        t, d = self.t, self.device
        extra = self.capacity + 1
        self.parent_row = t.full((extra,), -1, dtype=t.int64, device=d)
        self.ordinal = t.full((extra,), -1, dtype=t.int64, device=d)
        self.owners = t.full((extra,), -1, dtype=t.int64, device=d)
        self.generations = t.zeros(extra, dtype=t.int64, device=d)
        self.slots = t.full((extra,), -1, dtype=t.int64, device=d)
        self.actors = t.full((extra,), -1, dtype=t.int64, device=d)
        self.job_epoch = t.zeros(extra, dtype=t.int64, device=d)
        self.job_row_workload = t.zeros(extra, dtype=t.int64, device=d)
        self.row_workload = t.zeros(self.rows, dtype=t.int64, device=d)
        self.errors = t.zeros(self.rows, dtype=t.int32, device=d)
        self.needed = t.zeros(1, dtype=t.int64, device=d)
        self.job_count = t.zeros(1, dtype=t.int64, device=d)
        self.overflow = t.zeros(1, dtype=t.uint8, device=d)
        self.epoch = t.zeros(1, dtype=t.int64, device=d)
        self.dest_written = t.zeros(self.rows * self.max_actions + 1, dtype=t.int32, device=d)
        self.expected = t.zeros((self.rows, self.max_actions), dtype=t.bool, device=d)
        self.admitted_rows = t.zeros(self.rows, dtype=t.bool, device=d)
        self.claim_counts = t.zeros(extra, dtype=t.int64, device=d)
        self.error_hits = t.zeros(self.rows + 1, dtype=t.int64, device=d)
        self.preview_values = tuple(t.zeros(self.rows * self.max_actions + 1,
            dtype=t.float64, device=d) for _ in range(4))
        self.job_errors = t.zeros(self.capacity, dtype=t.int32, device=d)
        self._cand_ids = t.arange(self.max_actions, dtype=t.int64, device=d)
        self._row_ids = t.arange(self.rows, dtype=t.int64, device=d)
        self._neg = t.tensor(-1, dtype=t.int64, device=d)
        self._zero64 = t.tensor(0, dtype=t.int64, device=d)
        # Persistent buffers only; excludes caller inputs and graph temporaries.
        owned = [value for value in self.__dict__.values() if isinstance(value, t.Tensor)]
        self.allocated_bytes = sum(value.numel() * value.element_size()
                                   for value in owned + list(self.preview_values))

    def close(self):
        if not self.closed:
            if self.module is not None:
                self.module.close()
            self.closed = True

    def reset_device(self):
        self.parent_row.fill_(-1)
        self.ordinal.fill_(-1)
        self.owners.fill_(-1)
        self.generations.zero_()
        self.slots.fill_(-1)
        self.actors.fill_(-1)
        self.job_epoch.zero_()
        self.job_row_workload.zero_()
        self.row_workload.zero_()
        self.errors.zero_()
        self.needed.zero_()
        self.job_count.zero_()
        self.overflow.zero_()
        self.epoch.add_(1)
        self.dest_written.zero_()
        self.claim_counts.zero_()
        self.expected.zero_()
        self.admitted_rows.zero_()
        for value in self.preview_values:
            value.zero_()

    def _require(self, value, dtype, shape, name):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact contiguous %s required' % name)
        return value

    def compact_device(self, valid, battle, counts, active, input_errors,
                       slots, generations, owners, actors):
        if self.closed:
            raise RuntimeError('Preview jobs closed')
        t = self.t
        b, m = self.rows, self.max_actions
        valid = self._require(valid, t.bool, (b, m), 'legal mask')
        battle = self._require(battle, t.bool, (b, m), 'battle mask')
        counts = self._require(counts, t.int64, (b,), 'legal counts')
        active = self._require(active, t.bool, (b,), 'active rows')
        input_errors = self._require(input_errors, t.int32, (b,), 'input errors')
        slots = self._require(slots, t.int64, (b,), 'state slots')
        generations = self._require(generations, t.int64, (b,), 'state generations')
        owners = self._require(owners, t.int64, (b,), 'state owners')
        actors = self._require(actors, t.int64, (b,), 'actors')
        self.reset_device()
        idx = self._cand_ids
        count_ok = (counts >= 0) & (counts <= m)
        prefix = idx[None, :] < counts.clamp(0, m)[:, None]
        mask_ok = count_ok & (valid == prefix).all(1)
        battle_ok = (~battle | valid).all(1)
        actor_ok = (actors == 0) | (actors == 1)
        lease_ok = (slots >= 0) & (generations >= 0) & (owners >= 0)
        formed = mask_ok & battle_ok & actor_ok & lease_ok
        self.errors.copy_(t.where(active, input_errors, 0))
        malformed = active & (self.errors == 0) & ~formed
        self.errors.copy_(t.where(malformed, ERR_MALFORMED, self.errors))
        eligible_row = active & (self.errors == 0)
        eligible = eligible_row[:, None] & valid & battle
        row_workload = eligible.to(t.int64).sum(1)
        self.row_workload.copy_(row_workload)
        flat = eligible.reshape(-1)
        ranks = flat.to(t.int64).cumsum(0) - 1
        needed = flat.to(t.int64).sum()
        self.needed.copy_(needed.reshape(1))
        overflow = needed > self.capacity
        self.overflow.copy_(overflow.to(t.uint8).reshape(1))
        had_job = row_workload > 0
        self.errors.copy_(t.where(
            had_job & overflow, self.errors | ERR_CAPACITY, self.errors))
        self.expected.copy_(eligible & ~overflow)
        self.admitted_rows.copy_(eligible_row & ~overflow)
        admit = flat & ~overflow
        dest = t.where(admit, ranks.clamp(0, self.capacity - 1), self.capacity)
        parent = self._row_ids[:, None].expand(b, m).reshape(-1)
        ordinal = idx[None, :].expand(b, m).reshape(-1)
        owner_flat = owners[:, None].expand(b, m).reshape(-1)
        gen_flat = generations[:, None].expand(b, m).reshape(-1)
        slot_flat = slots[:, None].expand(b, m).reshape(-1)
        actor_flat = actors[:, None].expand(b, m).reshape(-1)
        load_flat = row_workload[:, None].expand(b, m).reshape(-1)
        sink = self._neg
        zero = self._zero64
        self.parent_row.index_copy_(0, dest, t.where(admit, parent, sink))
        self.ordinal.index_copy_(0, dest, t.where(admit, ordinal, sink))
        self.owners.index_copy_(0, dest, t.where(admit, owner_flat, sink))
        self.generations.index_copy_(0, dest, t.where(admit, gen_flat, zero))
        self.slots.index_copy_(0, dest, t.where(admit, slot_flat, sink))
        self.actors.index_copy_(0, dest, t.where(admit, actor_flat, sink))
        self.job_epoch.index_copy_(0, dest, t.where(admit, self.epoch[0], zero))
        self.job_row_workload.index_copy_(0, dest, t.where(admit, load_flat, zero))
        admitted = t.where(overflow, self._zero64, needed)
        self.job_count.copy_(admitted.reshape(1))
        return DeviceTeacherPreviewWorkload(
            self.parent_row[:self.capacity], self.ordinal[:self.capacity],
            self.owners[:self.capacity], self.generations[:self.capacity],
            self.slots[:self.capacity], self.actors[:self.capacity],
            self.row_workload, self.needed, self.job_count, self.overflow,
            self.errors, self.epoch, False)

    def scatter_previews_device(
            self, job_ids, parents, ordinals, owners, generations, slots, active,
            attack, attack_count, genesis_damage, counter,
            dest_attack, dest_attack_count, dest_genesis, dest_counter, dest_errors,
            *, epochs, actors):
        """Parallel checked receipts; epochs/actors must travel with each job.

        Receipts may arrive in multiple static-size groups. Counts and errors
        persist until the next compact/reset. Call finalize_device before
        ranking; a partially returned workload is never complete.
        """
        if self.closed:
            raise RuntimeError('Preview jobs closed')
        t = self.t
        if (not isinstance(job_ids, t.Tensor) or job_ids.ndim != 1
                or not 0 < job_ids.numel() <= self.capacity):
            raise ValueError('Bounded nonempty job ids required')
        q = job_ids.numel()
        job_ids = self._require(job_ids, t.int64, (q,), 'job ids')
        claims = [self._require(x, t.int64, (q,), name) for x, name in (
            (parents, 'parents'), (ordinals, 'ordinals'), (owners, 'owners'),
            (generations, 'generations'), (slots, 'slots'),
            (epochs, 'epochs'), (actors, 'actors'))]
        parents, ordinals, owners, generations, slots, epochs, actors = claims
        active = self._require(active, t.bool, (q,), 'result active')
        values = [self._require(x, t.float64, (q,), name) for x, name in (
            (attack, 'attack'), (attack_count, 'attack_count'),
            (genesis_damage, 'genesis'), (counter, 'counter'))]
        b, m = self.rows, self.max_actions
        outputs = [self._require(x, t.float64, (b, m), name) for x, name in (
            (dest_attack, 'dest attack'), (dest_attack_count, 'dest count'),
            (dest_genesis, 'dest genesis'), (dest_counter, 'dest counter'))]
        dest_errors = self._require(dest_errors, t.int32, (b,), 'dest errors')
        self.errors.bitwise_or_(dest_errors)
        safe = job_ids.clamp(0, self.capacity - 1)
        known = (job_ids >= 0) & (job_ids < self.job_count[0]) & (job_ids < self.capacity)
        root, ordinal = self.parent_row[safe], self.ordinal[safe]
        aligned = (root >= 0) & (root < b) & (ordinal >= 0) & (ordinal < m)
        known = known & aligned
        outside = active & ~known
        foreign = active & known & ((parents != root) | (ordinals != ordinal)
            | (owners != self.owners[safe]) | (actors != self.actors[safe]))
        stale = active & known & ((generations != self.generations[safe])
            | (slots != self.slots[safe]) | (epochs != self.job_epoch[safe])
            | (epochs != self.epoch[0]))
        finite = t.ones(q, dtype=t.bool, device=self.device)
        for value in values:
            finite = finite & t.isfinite(value)
        nonfinite = active & known & ~finite
        errors = (outside.to(t.int32) * ERR_OUTSIDE
            | foreign.to(t.int32) * ERR_FOREIGN
            | stale.to(t.int32) * ERR_STALE
            | nonfinite.to(t.int32) * ERR_NONFINITE)
        trusted_root = root.clamp(0, b - 1)
        good = active & known & (errors == 0) & (self.errors[trusted_root] == 0)
        sink = self.capacity
        targets = t.where(good, safe, sink)
        self.claim_counts.index_add_(0, targets, good.to(t.int64))
        duplicate = good & (self.claim_counts[safe] != 1)
        errors = errors | duplicate.to(t.int32) * ERR_DUPLICATE
        # Never use a forged claimed parent to poison a different root.
        row_ids = t.where(active & known, trusted_root, b)
        for code in (ERR_FOREIGN, ERR_STALE, ERR_NONFINITE, ERR_DUPLICATE):
            self.error_hits.zero_()
            self.error_hits.index_add_(0, row_ids, ((errors & code) != 0).to(t.int64))
            self.errors.bitwise_or_((self.error_hits[:b] != 0).to(t.int32) * code)
        # An unknown receipt has no trustworthy owner; reject this workload.
        self.errors.bitwise_or_((outside.any() & (self.row_workload > 0)).to(t.int32) * ERR_OUTSIDE)
        good = good & ~duplicate & (self.errors[trusted_root] == 0)
        cells = b * m
        offsets = t.where(good, trusted_root * m + ordinal.clamp(0, m - 1), cells)
        self.dest_written.index_add_(0, offsets, good.to(t.int32))
        for source, internal, output in zip(values, self.preview_values, outputs):
            # All rejected jobs target an isolated identical-zero sink.
            internal.index_copy_(0, offsets, t.where(good, source, 0.))
            written = (self.dest_written[:cells].view(b, m) == 1)
            publish = written & self.expected & (self.errors[:, None] == 0)
            output.copy_(t.where(publish, internal[:cells].view(b, m), output))
        dest_errors.bitwise_or_(self.errors)
        self.job_errors.zero_()
        self.job_errors[:q].copy_(errors)
        return self.job_errors[:q]

    def finalize_device(self, dest_errors):
        """Return a device per-row completion mask after ALL preview receipts."""
        if self.closed:
            raise RuntimeError('Preview jobs closed')
        t = self.t
        dest_errors = self._require(dest_errors, t.int32, (self.rows,), 'dest errors')
        self.errors.bitwise_or_(dest_errors)
        written = self.dest_written[:-1].view(self.rows, self.max_actions)
        missing = (self.expected & (written != 1)).any(1)
        self.errors.bitwise_or_(missing.to(t.int32) * ERR_MISSING)
        dest_errors.bitwise_or_(self.errors)
        return self.admitted_rows & ~missing & (self.errors == 0)
