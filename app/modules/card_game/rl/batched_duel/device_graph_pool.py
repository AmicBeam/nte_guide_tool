"""Bounded device page allocator for future compact rule graphs.

The arena manages byte pages and device handles, not Python graph semantics.
It is not wired into CudaVm and does not establish full-game CUDA support.
Use separate logical arenas for nodes, edges and payload so unused maximum
segment extents do not force allocation of holes. No allocation or CUDA import
occurs merely by importing this module.
"""
from dataclasses import dataclass
from typing import Any


INVALID_HANDLE = 1
DUPLICATE_SLOT = 2
INVALID_SIZE = 4
PAGE_CAPACITY = 8
CORRUPT_PAGES = 16
INVALID_OFFSET = 32


@dataclass(frozen=True)
class DevicePageHandle:
    slots: Any
    generations: Any
    owners: Any


@dataclass(frozen=True)
class DevicePageResult:
    handle: DevicePageHandle
    errors: Any


@dataclass(frozen=True)
class DeviceAddressResult:
    offsets: Any
    errors: Any


class DeviceGraphPagePool:
    """One stream/driver owns the arena; all hot-path identities stay on device.

    Slots are explicitly supplied by the device search driver. Page admission
    is atomic at batch granularity: a batch that exceeds the free page budget
    does not partially mutate otherwise valid requests. This is a capacity
    policy, not truncation. Grow preserves existing page addresses and data.
    Errors are device tensors and must be checked after the device batch.
    """
    supports_full_duel_rules = False

    def __init__(self, *, slots, pages, page_bytes, max_pages_per_slot,
                 device='cuda', allow_cpu_reference=False):
        import torch
        for value in (slots, pages, page_bytes, max_pages_per_slot):
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError('Positive int32 bounded pool dimensions required')
        if page_bytes < 16 or page_bytes & (page_bytes - 1):
            raise ValueError('Power-of-two pages of at least 16 bytes required')
        if max_pages_per_slot > pages:
            raise ValueError('Per-slot page bound exceeds the global arena')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (
                self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit allocator oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU allocator fallback')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.slot_capacity, self.page_capacity = slots, pages
        self.page_bytes, self.max_pages_per_slot = page_bytes, max_pages_per_slot
        self.data = torch.zeros((pages, page_bytes), dtype=torch.uint8, device=self.device)
        self.page_table = torch.full((slots, max_pages_per_slot), -1, dtype=torch.int64, device=self.device)
        self.page_owner = torch.full((pages,), -1, dtype=torch.int64, device=self.device)
        self.generation = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.owner = torch.full((slots,), -1, dtype=torch.int64, device=self.device)
        self.live = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.logical_bytes = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.page_count = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.peak_used_pages = torch.zeros((), dtype=torch.int64, device=self.device)
        self._page_ids = torch.arange(pages, device=self.device)
        self._positions = torch.arange(max_pages_per_slot, device=self.device)

    @property
    def allocated_bytes(self):
        names = ('data', 'page_table', 'page_owner', 'generation', 'owner', 'live',
                 'logical_bytes', 'page_count', 'peak_used_pages', '_page_ids', '_positions')
        return sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in names)

    def metrics_device(self):
        return dict(used_pages=(self.page_owner >= 0).sum(), live_slots=self.live.sum(),
                    logical_bytes=self.logical_bytes.sum(), peak_used_pages=self.peak_used_pages)

    def _vector(self, value, dtype, shape=None):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.ndim != 1
                or (shape is not None and value.shape != shape)):
            raise ValueError('Exact device vector shape and dtype required')

    def _validate(self, handle, active, *, live):
        t = self.t
        if not isinstance(handle, DevicePageHandle):
            raise TypeError('DevicePageHandle required')
        self._vector(handle.slots, t.int64)
        shape = handle.slots.shape
        for value in (handle.generations, handle.owners):
            self._vector(value, t.int64, shape)
        self._vector(active, t.bool, shape)
        if not 0 < handle.slots.numel() <= self.slot_capacity:
            raise ValueError('Nonempty bounded handle batch required')
        indices = handle.slots.clamp(0, self.slot_capacity - 1)
        in_range = (handle.slots >= 0) & (handle.slots < self.slot_capacity)
        valid = (in_range & (handle.generations == self.generation[indices])
                 & (handle.owners >= 0) & (self.live[indices] == int(live)))
        if live:
            valid = valid & (handle.owners == self.owner[indices])
        histogram = t.zeros(self.slot_capacity, dtype=t.int64, device=self.device)
        histogram.index_add_(0, indices, (active & in_range).long())
        duplicate = histogram[indices] > 1
        errors = (active & ~valid).long() * INVALID_HANDLE
        errors = errors | (active & duplicate).long() * DUPLICATE_SLOT
        return indices, active & valid & ~duplicate, errors

    def reserve(self, slots, generations, owners, requested_bytes, active):
        handle = DevicePageHandle(slots, generations, owners)
        return self._admit(handle, requested_bytes, active, grow=False)

    def grow(self, handle, requested_bytes, active):
        return self._admit(handle, requested_bytes, active, grow=True)

    def _admit(self, handle, requested_bytes, active, *, grow):
        t = self.t
        indices, good, errors = self._validate(handle, active, live=grow)
        self._vector(requested_bytes, t.int64, handle.slots.shape)
        old_count = self.page_count[indices]
        old_bytes = self.logical_bytes[indices]
        size_ok = ((requested_bytes > 0)
                   & (requested_bytes <= self.page_bytes * self.max_pages_per_slot))
        if grow:
            size_ok = size_ok & (requested_bytes >= old_bytes)
        errors = errors | (active & ~size_ok).long() * INVALID_SIZE
        good = good & size_ok
        safe_bytes = requested_bytes.clamp(0, self.page_bytes * self.max_pages_per_slot)
        counts = (safe_bytes + self.page_bytes - 1) // self.page_bytes
        needed = t.where(good, counts - old_count, 0)
        free = self.page_owner < 0
        fits = needed.sum() <= free.sum()
        errors = errors | (good & ~fits).long() * PAGE_CAPACITY
        good = good & fits
        needed = t.where(good, needed, 0)
        # Fixed-shape sort avoids nonzero()/data-dependent host allocation.
        free_order = t.argsort((~free).long() * self.page_capacity + self._page_ids)
        positions = self._positions[None, :]
        take = good[:, None] & (positions < needed[:, None])
        rank = needed.cumsum(0)[:, None] - needed[:, None] + positions
        physical = free_order[rank.clamp(0, self.page_capacity - 1)]
        destination = indices[:, None] * self.max_pages_per_slot + (
            old_count[:, None] + positions).clamp(0, self.max_pages_per_slot - 1)
        self.page_table.flatten().index_add_(0, destination.flatten(),
                                            ((physical + 1) * take).flatten())
        self.page_owner.index_add_(0, physical.flatten(),
                                   ((indices[:, None] + 1) * take).flatten())
        new_pages = t.zeros(self.page_capacity, dtype=t.int64, device=self.device)
        new_pages.index_add_(0, physical.flatten(), take.long().flatten())
        # Reset leased pages, including reuse by another owner. Broadcast mask
        # needs no per-root maximum-sized heap or host page-list materialization.
        self.data.masked_fill_(new_pages[:, None] > 0, 0)
        self.page_count.index_add_(0, indices, needed)
        self.logical_bytes.index_add_(0, indices, t.where(good, safe_bytes - old_bytes, 0))
        if not grow:
            self.live.index_add_(0, indices, good.long())
            self.owner.index_add_(0, indices, t.where(good, handle.owners + 1, 0))
        self.peak_used_pages.copy_(t.maximum(self.peak_used_pages, (self.page_owner >= 0).sum()))
        return DevicePageResult(handle, errors)

    def release(self, handle, active):
        t = self.t
        indices, good, errors = self._validate(handle, active, live=True)
        table = self.page_table[indices]
        occupied = self._positions[None, :] < self.page_count[indices, None]
        physical = table.clamp(0, self.page_capacity - 1)
        intact = (~occupied | ((table >= 0) & (table < self.page_capacity)
                              & (self.page_owner[physical] == indices[:, None]))).all(-1)
        errors = errors | (good & ~intact).long() * CORRUPT_PAGES
        good = good & intact
        clear = good[:, None] & occupied
        destination = indices[:, None] * self.max_pages_per_slot + self._positions[None, :]
        self.page_table.flatten().index_add_(0, destination.flatten(),
                                            (-(physical + 1) * clear).flatten())
        self.page_owner.index_add_(0, physical.flatten(),
                                   (-(indices[:, None] + 1) * clear).flatten())
        self.page_count.index_add_(0, indices, -t.where(good, self.page_count[indices], 0))
        self.logical_bytes.index_add_(0, indices, -t.where(good, self.logical_bytes[indices], 0))
        self.owner.index_add_(0, indices, -t.where(good, self.owner[indices] + 1, 0))
        self.live.index_add_(0, indices, -good.long())
        self.generation.index_add_(0, indices, good.long())
        return errors

    def translate(self, handle, byte_offsets, active):
        t = self.t
        indices, good, errors = self._validate(handle, active, live=True)
        if (not isinstance(byte_offsets, t.Tensor) or byte_offsets.device != self.device
                or byte_offsets.dtype != t.int64 or byte_offsets.ndim != 2
                or byte_offsets.shape[0] != indices.numel()):
            raise ValueError('Device int64 [batch, offsets] required')
        bounds = (byte_offsets >= 0) & (byte_offsets < self.logical_bytes[indices, None])
        page = (byte_offsets // self.page_bytes).clamp(0, self.max_pages_per_slot - 1)
        physical = self.page_table[indices[:, None], page]
        intact = ((physical >= 0) & (physical < self.page_capacity)
                  & (self.page_owner[physical.clamp(0, self.page_capacity - 1)] == indices[:, None]))
        errors = errors | (good & ~bounds.all(-1)).long() * INVALID_OFFSET
        errors = errors | (good & bounds.all(-1) & ~intact.all(-1)).long() * CORRUPT_PAGES
        valid = good[:, None] & bounds & intact & (errors[:, None] == 0)
        offsets = physical * self.page_bytes + byte_offsets.remainder(self.page_bytes)
        return DeviceAddressResult(t.where(valid, offsets, -1), errors)
