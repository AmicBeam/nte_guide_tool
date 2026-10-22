"""Direct named-field SoA state pool for a future 400-root GPU Gumbel/H3 backend.

This is storage only: slot/owner/generation leases plus a dictionary of
fixed-shape numeric columns prepared once. It is not a VM, graph codec, native
kernel, rule stepper, or search driver. supports_full_duel_rules stays False
and must not be treated as Gumbel/H3 readiness.

Hot gather/clone/copy/release paths keep slot indices, masks and validation as
device tensors. There is no per-root Python loop and no .item / .tolist /
.cpu / .numpy on those paths. Overflow of free slots is atomic for the whole
batch. Duplicate destinations, stale/foreign leases and inactive rows are
rejected or ignored per row as documented on each method.
"""
from types import MappingProxyType
from math import prod

from .device_rule_protocol import DeviceStateBatch
from .domain_soa_transition import FieldSpec, resolve_field_specs, supports_full_duel_rules as _TRANSITION_FULL


INVALID_LEASE, CAPACITY, DUPLICATE = 1, 2, 4
supports_full_duel_rules = False
if _TRANSITION_FULL:
    raise RuntimeError('domain_soa_transition must not claim full duel coverage')

_DTYPE_NAMES = {
    'bool': 'bool', 'bool_': 'bool',
    'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64',
    'uint8': 'uint8',
    'float16': 'float16', 'float32': 'float32', 'float64': 'float64',
}


def _dtype_key(dtype):
    name = getattr(dtype, 'name', None) or str(dtype)
    name = name.replace('torch.', '').replace('numpy.', '').replace('np.', '')
    return _DTYPE_NAMES.get(name)


def _as_spec(item):
    if isinstance(item, FieldSpec):
        name, dtype, tail = item.name, item.dtype, tuple(item.tail)
    elif isinstance(item, dict):
        name = item['name']
        dtype = item.get('dtype', 'int32')
        tail = tuple(item.get('tail') or item.get('shape_tail') or ())
    elif isinstance(item, (tuple, list)):
        name = item[0]
        dtype = item[1]
        tail = tuple(item[2]) if len(item) > 2 else ()
    else:
        name = item.name
        dtype = getattr(item, 'dtype', 'int32')
        tail = tuple(getattr(item, 'tail', getattr(item, 'shape_tail', ())))
    key = _dtype_key(dtype)
    if key is None:
        raise ValueError('Only fixed-shape numeric SoA columns are stored')
    if any(type(dim) is not int or isinstance(dim, bool) or dim < 1 for dim in tail):
        raise ValueError('Positive int field tails required')
    return FieldSpec(name, key, tail)


def prepare_field_specs(specs=None, schema=None):
    """Freeze named numeric column metadata once. Not a rule ABI certificate."""
    raw = resolve_field_specs(specs, schema) if specs is None else specs
    prepared = tuple(_as_spec(item) for item in raw)
    if not prepared:
        raise ValueError('Nonempty named numeric SoA field specs required')
    names = [spec.name for spec in prepared]
    if len(names) != len(set(names)):
        raise ValueError('SoA field names must be unique')
    return prepared


class DomainSoAStatePool:
    """Bounded device leases over direct named SoA columns.

    Slot overflow: whole-batch atomic. If admitted rows exceed free slots, every
    otherwise-valid wanted row receives CAPACITY and no live metadata or column
    bits change.

    Duplicate destinations: per-row. Every duplicated destination row receives
    DUPLICATE and is not written; unique destinations in the same batch still
    copy. Shared source rows may fan out to many distinct destinations.

    Stale or foreign leases: per-row INVALID_LEASE. Inactive rows are no-ops and
    do not consume slots, overwrite slot 0, or raise. Released generations can
    be reused; old leases stay invalid.
    """
    supports_full_duel_rules = False

    def __init__(self, capacity, *, device='cuda', allow_cpu_reference=False,
                 specs=None, schema=None, fused_cuda_copy=True):
        import torch
        if type(capacity) is not int or not 1 <= capacity < 2 ** 31:
            raise ValueError('Positive bounded SoA slot capacity required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU store fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit allocator oracle only')
        self.capacity = capacity
        self.specs = prepare_field_specs(specs, schema)
        dtypes = {
            'bool': torch.bool, 'int8': torch.int8, 'int16': torch.int16,
            'int32': torch.int32, 'int64': torch.int64, 'uint8': torch.uint8,
            'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64,
        }
        # Field views keep a typed domain ABI. Storage is packed by dtype so
        # every fork copies at most one tensor per dtype, not one per field.
        widths = {}
        offsets = {}
        for spec in self.specs:
            offsets[spec.name] = widths.get(spec.dtype, 0)
            widths[spec.dtype] = offsets[spec.name] + prod(spec.tail)
        self._packed = {kind: torch.zeros((capacity, width), dtype=dtypes[kind],
                                         device=self.device)
                        for kind, width in widths.items()}
        self._offsets = offsets
        self.columns = {
            spec.name: self._packed[spec.dtype][
                :, offsets[spec.name]:offsets[spec.name]+prod(spec.tail)
            ].view((capacity,)+spec.tail)
            for spec in self.specs
        }
        self.live = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.generation = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.owner = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)
        self.ids = torch.arange(capacity, dtype=torch.int64, device=self.device)
        if type(fused_cuda_copy)is not bool:
            raise TypeError('Explicit fused-copy selection required')
        self._copy_kernel=None
        if self.device.type=='cuda'and fused_cuda_copy:
            from .domain_soa_cuda_copy import DomainCudaCopy
            self._copy_kernel=DomainCudaCopy(self.device)

    @property
    def allocated_bytes(self):
        tensors = (*self.columns.values(), self.live, self.generation, self.owner, self.ids)
        return sum(value.numel() * value.element_size() for value in tensors)

    def _vector(self, value, dtype, rows=None):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.ndim != 1
                or (rows is not None and value.shape != (rows,))):
            raise ValueError('Exact device vector required')

    def _check_fields(self, fields, rows):
        if not isinstance(fields, dict) or set(fields) != set(self.columns):
            raise ValueError('Exact named SoA field mapping required')
        for name, column in self.columns.items():
            value = fields[name]
            expected = (rows,) + tuple(column.shape[1:])
            if (not isinstance(value, self.t.Tensor) or value.device != self.device
                    or value.dtype != column.dtype or tuple(value.shape) != expected):
                raise ValueError(name + ' has an incompatible SoA layout')

    def _bits(self, column):
        t = self.t
        if column.dtype == t.bool:
            return column.view(t.uint8)
        if column.dtype == t.float64:
            return column.view(t.int64)
        if column.dtype == t.float32:
            return column.view(t.int32)
        if column.dtype == t.float16:
            return column.view(t.int16)
        return column

    def _row_mask(self, good, ndim):
        return good.view((good.shape[0],) + (1,) * (ndim - 1))

    def _scatter(self, destination, dest_idx, source, good):
        """Write source[B, ...] into dest slots without host loops.

        Integer and float payloads are copied as integer bit patterns so NaN,
        signed zero and every integer encoding survive. Inactive rows contribute
        a zero delta even if their ranked dest collides with a live write.
        """
        t = self.t
        dest_bits = self._bits(destination)
        src_bits = self._bits(source)
        if dest_bits.dtype in (t.uint8, t.int8):
            current = dest_bits.to(t.int16).index_select(0, dest_idx)
            delta = t.where(self._row_mask(good, src_bits.ndim),
                            src_bits.to(t.int16) - current, 0)
            dest_bits.index_add_(0, dest_idx, delta.to(dest_bits.dtype))
            return
        current = dest_bits.index_select(0, dest_idx)
        delta = t.where(self._row_mask(good, src_bits.ndim), src_bits - current, 0)
        dest_bits.index_add_(0, dest_idx, delta)

    def _copy_slots(self, src_idx, dest_idx, good):
        src = src_idx.clamp(0, self.capacity - 1)
        dest = dest_idx.clamp(0, self.capacity - 1)
        for storage in self._packed.values():
            snapshot=storage.index_select(0,src)
            if self._copy_kernel is not None:
                self._copy_kernel.scatter(storage.view(self.t.uint8),dest,
                                          snapshot.view(self.t.uint8),good)
            else:
                self._scatter(storage,dest,snapshot,good)

    def _admit(self, owners, active, errors, rows):
        t = self.t
        valid_owner = owners >= 0
        errors.bitwise_or_((active & ~valid_owner).to(t.int32) * INVALID_LEASE)
        wanted = active & valid_owner & (errors == 0)
        fits = wanted.sum() <= (~self.live).sum()
        errors.bitwise_or_((wanted & ~fits).to(t.int32) * CAPACITY)
        good = wanted & fits
        free = t.argsort(self.live.long() * self.capacity + self.ids)
        rank = (good.long().cumsum(0) - 1).clamp(0, self.capacity - 1)
        slots = free[rank]
        return slots, good, errors

    def _bind_lease(self, slots, owners, good, errors):
        t = self.t
        dest = slots.clamp(0, self.capacity - 1)
        self.generation.index_add_(0, dest, good.long())
        self.owner.index_add_(0, dest, t.where(good, owners - self.owner[dest], 0))
        live_delta = t.zeros(self.capacity, dtype=t.int64, device=self.device)
        live_delta.index_add_(0, dest, good.long())
        self.live.logical_or_(live_delta > 0)
        return DeviceStateBatch(
            self, t.where(good, dest, -1), t.where(good, self.generation[dest], 0),
            owners.clone(), good, errors)

    def validate(self, states, *, shared_reads=False):
        if not isinstance(states, DeviceStateBatch) or states.pool is not self:
            raise ValueError('State lease belongs to another arena')
        t = self.t
        self._vector(states.slots, t.int64)
        rows = states.slots.shape[0]
        for value in (states.generations, states.owners):
            self._vector(value, t.int64, rows)
        self._vector(states.active, t.bool, rows)
        self._vector(states.errors, t.int32, rows)
        indices = states.slots.clamp(0, self.capacity - 1)
        valid = ((states.slots >= 0) & (states.slots < self.capacity) & self.live[indices]
                 & (states.generations == self.generation[indices])
                 & (states.owners == self.owner[indices]) & (states.owners >= 0))
        errors = (states.active & ~valid).to(t.int32) * INVALID_LEASE
        if not shared_reads:
            histogram = t.zeros(self.capacity, dtype=t.int64, device=self.device)
            histogram.index_add_(0, indices, (states.active & valid).long())
            duplicates = histogram[indices] > 1
            errors.bitwise_or_((states.active & duplicates).to(t.int32) * DUPLICATE)
            valid = valid & ~duplicates
        states.errors.bitwise_or_(errors)
        return indices, states.active & valid & (states.errors == 0)

    def put_device(self, fields, owners, active, *, errors=None):
        t = self.t
        self._vector(owners, t.int64)
        rows = owners.shape[0]
        if not 0 < rows <= self.capacity:
            raise ValueError('Nonempty bounded device batch required')
        self._vector(active, t.bool, rows)
        self._check_fields(fields, rows)
        errors = t.zeros(rows, dtype=t.int32, device=self.device) if errors is None else errors.clone()
        self._vector(errors, t.int32, rows)
        slots, good, errors = self._admit(owners, active, errors, rows)
        dest = slots.clamp(0, self.capacity - 1)
        for name, column in self.columns.items():
            self._scatter(column, dest, fields[name], good)
        return self._bind_lease(slots, owners, good, errors)

    def clone_device(self, states):
        """Fork active source rows into newly admitted unique destinations.

        Repeated source slots are shared reads. Parent column bits stay put.
        Destination overflow uses the same whole-batch CAPACITY policy as put.
        """
        t = self.t
        src, wanted = self.validate(states, shared_reads=True)
        rows = states.slots.shape[0]
        errors = states.errors.clone()
        self._vector(errors, t.int32, rows)
        slots, good, errors = self._admit(states.owners, wanted, errors, rows)
        self._copy_slots(src, slots, good)
        return self._bind_lease(slots, states.owners, good, errors)

    def copy_device(self, sources, destinations):
        """Snapshot-copy source rows into existing destination leases.

        Shared sources are allowed. Duplicate destinations are rejected per row
        and receive DUPLICATE. Reads use pre-batch bits, so a row that is both
        source and destination in this batch keeps its original pattern unless a
        distinct unique destination overwrites it. Parents that are only sources
        remain untouched.
        """
        if sources.slots.shape != destinations.slots.shape:
            raise ValueError('Aligned source and destination batches required')
        src, src_good = self.validate(sources, shared_reads=True)
        dest, dest_good = self.validate(destinations, shared_reads=False)
        good = src_good & dest_good
        self._copy_slots(src, dest, good)
        return destinations

    def gather_device(self, states, *, shared_reads=False):
        indices, good = self.validate(states, shared_reads=shared_reads)
        packed = {}
        for kind, storage in self._packed.items():
            gathered = storage.index_select(0, indices)
            packed[kind] = self.t.where(self._row_mask(good, gathered.ndim), gathered, 0)
        rows=indices.shape[0]
        fields = {spec.name: packed[spec.dtype][
            :,self._offsets[spec.name]:self._offsets[spec.name]+prod(spec.tail)
        ].view((rows,)+spec.tail) for spec in self.specs}
        return MappingProxyType(fields)

    def release_device(self, states):
        indices, good = self.validate(states)
        remove = self.t.zeros(self.capacity, dtype=self.t.int64, device=self.device)
        remove.index_add_(0, indices, good.long())
        self.live.logical_and_(remove == 0)
        self.owner.masked_fill_(remove > 0, -1)

    def close(self):
        """Stop captured uses before closing the owned direct copy kernel."""
        if self._copy_kernel is not None:
            self._copy_kernel.close()

