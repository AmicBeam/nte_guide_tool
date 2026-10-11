"""Collector page admission with exact stable free-page order, without sort."""
from ..batched_duel.device_graph_pool import DeviceGraphPagePool,DevicePageResult,INVALID_SIZE,PAGE_CAPACITY


def stable_free_page_order(free,*,allow_cpu_reference=False):
    import torch as t
    if not isinstance(free,t.Tensor) or free.dtype!=t.bool or free.ndim!=1 or free.numel()<1:
        raise ValueError('Nonempty one-dimensional boolean page mask required')
    if free.device.type!='cuda' and not (free.device.type=='cpu' and allow_cpu_reference is True):
        raise ValueError('CUDA required; CPU explicit page-order oracle only')
    ids=t.arange(free.numel(),dtype=t.int64,device=free.device)
    prefix=free.long().cumsum(0)
    rank=t.where(free,prefix-1,prefix[-1]+ids-prefix)
    return t.empty_like(ids).scatter_(0,rank,ids)


class DeviceStableGraphPagePool(DeviceGraphPagePool):
    """Only admission's free-page ordering differs from the original pool."""
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
        # Fixed-shape prefix keeps the identical free-first ascending order.
        free_order = stable_free_page_order(free,allow_cpu_reference=self.device.type=='cpu')
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

