"""Typed column arena with device ownership/generation leases.

Only declared homogeneous columns move between persistent slots and scratch.
No state nodes/edges, Python rules, or game serialization are involved.
"""
from .device_rule_protocol import DeviceStateBatch
from .record_soa import TorchRecordSoABatch, MISSING

INVALID_LEASE, CAPACITY, DUPLICATE = 1, 2, 4


class DeviceSoAStore:
    supports_full_duel_rules = False

    def __init__(self, schema, capacity, *, device='cuda', allow_cpu_reference=False):
        import torch
        if type(capacity) is not int or not 1 <= capacity < 2**31:
            raise ValueError('Positive bounded SoA slot capacity required')
        self.t, self.device, self.schema, self.capacity = torch, torch.device(device), schema, capacity
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU store fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CPU is an explicit ownership oracle only')
        self.buffers = {key: torch.zeros((schema.lengths[key], capacity), dtype=dtype, device=self.device)
                        for key, dtype in (('i64', torch.int64), ('f64', torch.float64),
                                           ('i32', torch.int32), ('u8', torch.uint8))}
        self.roots = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        self.live = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.generation = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.owner = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)
        self.ids = torch.arange(capacity, device=self.device)

    def _vector(self, value, dtype, rows=None):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device or value.dtype != dtype
                or value.ndim != 1 or (rows is not None and len(value) != rows)):
            raise ValueError('Exact device vector required')

    def validate(self, states, *, shared_reads=False):
        if not isinstance(states, DeviceStateBatch) or states.pool is not self:
            raise ValueError('State lease belongs to another arena')
        t = self.t
        self._vector(states.slots, t.int64)
        rows = len(states.slots)
        for value in (states.generations, states.owners):
            self._vector(value, t.int64, rows)
        self._vector(states.active, t.bool, rows)
        self._vector(states.errors, t.int32, rows)
        indices = states.slots.clamp(0, self.capacity-1)
        valid = ((states.slots >= 0) & (states.slots < self.capacity) & self.live[indices]
                 & (states.generations == self.generation[indices])
                 & (states.owners == self.owner[indices]) & (states.owners >= 0))
        errors = (states.active & ~valid).to(t.int32)*INVALID_LEASE
        if not shared_reads:
            histogram = t.zeros(self.capacity, dtype=t.int64, device=self.device)
            histogram.index_add_(0, indices, (states.active & valid).long())
            duplicates = histogram[indices] > 1
            errors.bitwise_or_((states.active & duplicates).to(t.int32)*DUPLICATE)
            valid = valid & ~duplicates
        states.errors.bitwise_or_(errors)
        return indices, states.active & valid & (states.errors == 0)

    def put_device(self, batch, owners, active, *, errors=None):
        if not isinstance(batch, TorchRecordSoABatch) or batch.schema.identity != self.schema.identity:
            raise ValueError('Matching typed column batch required')
        t, rows = self.t, batch.rows
        if not 0 < rows <= self.capacity:
            raise ValueError('Nonempty bounded device batch required')
        self._vector(owners, t.int64, rows)
        self._vector(active, t.bool, rows)
        errors = t.zeros(rows, dtype=t.int32, device=self.device) if errors is None else errors.clone()
        self._vector(errors, t.int32, rows)
        self._vector(batch.roots, t.int32, rows)
        for key, value in batch.buffers.items():
            expected = self.buffers[key]
            if (value.device != self.device or value.dtype != expected.dtype
                    or tuple(value.shape) != (expected.shape[0], rows) or not value.is_contiguous()):
                raise ValueError('Exact typed device columns required')
        root_pool=self.schema.pools[self.schema.root]
        valid_root=(batch.roots>root_pool.base)&(batch.roots<=root_pool.base+root_pool.capacity)
        valid_owner = owners >= 0
        errors.bitwise_or_((active & (~valid_owner | ~valid_root)).to(t.int32)*INVALID_LEASE)
        wanted = active & valid_owner & valid_root & (errors == 0)
        fits = wanted.sum() <= (~self.live).sum()
        errors.bitwise_or_((wanted & ~fits).to(t.int32)*CAPACITY)
        good = wanted & fits
        free = t.argsort(self.live.long()*self.capacity+self.ids)
        rank = (good.long().cumsum(0)-1).clamp(0, self.capacity-1)
        slots = free[rank]
        # Inactive rows contribute zero to shared destinations. Every admitted
        # destination is unique, including batches containing repeated owners.
        for key, destination in self.buffers.items():
            source = batch.buffers[key]
            if destination.dtype == t.uint8:
                delta = t.where(good[None, :], source.to(t.int16)-destination[:, slots].to(t.int16), 0)
                destination.index_add_(1, slots, delta.to(t.uint8))
            else:
                delta = t.where(good[None, :], source-destination[:, slots], 0)
                destination.index_add_(1, slots, delta)
        self.roots.index_add_(0, slots, t.where(good, batch.roots-self.roots[slots], 0))
        self.generation.index_add_(0, slots, good.long())
        self.owner.index_add_(0, slots, t.where(good, owners-self.owner[slots], 0))
        live_delta = t.zeros(self.capacity, dtype=t.int64, device=self.device)
        live_delta.index_add_(0, slots, good.long())
        self.live.logical_or_(live_delta > 0)
        return DeviceStateBatch(self, t.where(good, slots, -1), t.where(good,self.generation[slots],0),
                                owners.clone(), good, errors)

    def gather_device(self, states, *, shared_reads=False):
        indices, good = self.validate(states, shared_reads=shared_reads)
        buffers = {key: self.t.where(good[None, :], value[:, indices], 0).contiguous()
                   for key, value in self.buffers.items()}
        roots = self.t.where(good, self.roots[indices], 0).contiguous()
        return TorchRecordSoABatch(self.schema, buffers, roots)

    def gather_into_device(self,states,destination,*,shared_reads=False):
        """Fill a reusable column workspace without allocating a state graph."""
        if (not isinstance(destination,TorchRecordSoABatch)
                or destination.schema.identity!=self.schema.identity
                or destination.rows!=len(states.slots)):
            raise ValueError('Matching reusable typed column workspace required')
        indices,good=self.validate(states,shared_reads=shared_reads)
        for key,value in self.buffers.items():
            target=destination.buffers[key]
            if target.device!=self.device or target.dtype!=value.dtype or not target.is_contiguous():
                raise ValueError('Exact typed workspace columns required')
            self.t.index_select(value,1,indices,out=target)
            target.masked_fill_(~good[None,:],0)
        self.t.index_select(self.roots,0,indices,out=destination.roots)
        destination.roots.masked_fill_(~good,0)
        return destination

    def release_device(self, states):
        indices, good = self.validate(states)
        remove = self.t.zeros(self.capacity, dtype=self.t.int64, device=self.device)
        remove.index_add_(0, indices, good.long())
        self.live.logical_and_(remove == 0)
        self.owner.masked_fill_(remove > 0, -1)

    @property
    def allocated_bytes(self):
        return sum(value.numel()*value.element_size() for value in (
            *self.buffers.values(), self.roots, self.live, self.generation, self.owner, self.ids))
