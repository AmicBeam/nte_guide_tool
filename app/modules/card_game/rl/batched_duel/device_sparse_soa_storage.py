"""Lossless sparse typed columns with checked device state-slot leases.

Only zero-bit cells are implicit. Logical column offsets, numeric dtypes,
entity references, field presence and all other transported bits are retained.
Dense scratch remains the rule execution ABI; it is never a persistent graph.
"""
from .device_soa_storage import DeviceSoAStore, INVALID_LEASE, CAPACITY
from .device_rule_protocol import DeviceStateBatch
from .record_soa import TorchRecordSoABatch
from .soa_sparse_columns import sparse_column_source, sparse_column_signatures


DEFAULT_CELLS = {'i64': 4096, 'f64': 1024, 'i32': 16384, 'u8': 32768}
WIDTHS = {'i64': 8, 'f64': 8, 'i32': 4, 'u8': 1}
CORRUPT_COLUMN = 8


class DeviceSparseSoAStore(DeviceSoAStore):
    storage_kind = 'sparse_typed_columns'

    def __init__(self, schema, capacity, *, column_cells=None, max_bytes=4 << 30,
                 device='cuda', allow_cpu_reference=False):
        if type(capacity) is not int or not 1 <= capacity < 2**31:
            raise ValueError('Positive bounded SoA slot capacity required')
        if type(max_bytes) is not int or not 1 <= max_bytes < 2**63:
            raise ValueError('Positive finite sparse state byte budget required')
        requested = DEFAULT_CELLS if column_cells is None else column_cells
        if set(requested) != set(WIDTHS) or set(schema.lengths) != set(WIDTHS):
            raise ValueError('Exactly four typed column capacities required')
        if any(type(v) is not int or not 0 <= v < 2**31 for v in requested.values()):
            raise ValueError('Nonnegative integer column capacities required')
        caps = {k: min(requested[k], schema.lengths[k]) for k in WIDTHS}
        estimated = capacity*(45+sum(caps[k]*(4+WIDTHS[k]) for k in WIDTHS))+8
        if estimated > max_bytes:
            raise MemoryError(f'Sparse state arena requires {estimated} bytes; budget {max_bytes}')
        import torch
        self.t, self.device, self.schema, self.capacity = torch, torch.device(device), schema, capacity
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU store fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CPU is an explicit sparse-column ownership oracle only')
        self.column_cells, self.max_bytes = caps, max_bytes
        self.dtypes = {'i64': torch.int64, 'f64': torch.float64, 'i32': torch.int32, 'u8': torch.uint8}
        self.values = {k: torch.empty((caps[k], capacity), dtype=v, device=self.device)
                       for k, v in self.dtypes.items()}
        self.indices = {k: torch.empty((caps[k], capacity), dtype=torch.int32, device=self.device)
                        for k in WIDTHS}
        self.counts = {k: torch.zeros(capacity, dtype=torch.int32, device=self.device) for k in WIDTHS}
        self.roots = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        self.live = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.generation = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.owner = torch.full((capacity,), -1, dtype=torch.int64, device=self.device)
        self.ids = torch.arange(capacity, dtype=torch.int64, device=self.device)
        self.peak_live = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._workspaces, self.module, self.closed = {}, None, False
        if self.device.type == 'cuda':
            from .cuda_runtime import CudaModule
            self.module = CudaModule(sparse_column_source(), sparse_column_signatures(),
                                     device=str(self.device), binary_format='cubin')

    def _open(self):
        if self.closed:
            raise RuntimeError('Sparse state arena is closed')

    def prepare_workspace(self, rows):
        self._open()
        if type(rows) is not int or not 1 <= rows <= min(self.capacity, 64):
            raise ValueError('Sparse pack workspace must use 1..64 bounded work rows')
        if rows not in self._workspaces:
            if self.device.type == 'cuda' and self.t.cuda.is_current_stream_capturing():
                raise RuntimeError('Prepare sparse pack work rows before capture')
            self._workspaces[rows] = {k: dict(
                tiles=max(1, (self.schema.lengths[k]+255)//256),
                counts=self.t.zeros((rows, max(1, (self.schema.lengths[k]+255)//256)),
                                    dtype=self.t.int32, device=self.device),
                prefix=self.t.zeros((rows, max(1, (self.schema.lengths[k]+255)//256)),
                                    dtype=self.t.int32, device=self.device),
                total=self.t.zeros(rows, dtype=self.t.int32, device=self.device)) for k in WIDTHS}
        return self._workspaces[rows]

    def _check_batch(self, batch):
        if not isinstance(batch, TorchRecordSoABatch) or batch.schema.identity != self.schema.identity:
            raise ValueError('Matching typed column batch required')
        rows = batch.rows
        if not 1 <= rows <= min(self.capacity, 64) or set(batch.buffers) != set(WIDTHS):
            raise ValueError('Typed pack uses 1..64 bounded work rows and four columns')
        for k, value in batch.buffers.items():
            if (value.device != self.device or value.dtype != self.dtypes[k]
                    or value.shape != (self.schema.lengths[k], rows) or not value.is_contiguous()):
                raise ValueError('Exact contiguous typed device columns required')
        self._vector(batch.roots, self.t.int32, rows)
        return rows

    def _count(self, batch, wanted, workspace):
        for k, source in batch.buffers.items():
            work = workspace[k]
            if self.device.type == 'cpu':
                bits = source.view(self.t.int64) if k == 'f64' else source
                work['total'].copy_((bits != 0).sum(0, dtype=self.t.int32)*wanted.to(self.t.int32))
            else:
                self.module.launch('count_'+k, [source, wanted.view(self.t.uint8), work['counts'],
                    batch.rows, self.schema.lengths[k], work['tiles']],
                    grid=(work['tiles'], batch.rows), block=256, capture_safe=True)
                self.t.cumsum(work['counts'], 1, dtype=self.t.int32, out=work['prefix'])
                work['total'].copy_(work['prefix'][:, -1])

    def _pack(self, batch, slots, good, workspace, errors):
        for k, source in batch.buffers.items():
            limit = self.column_cells[k]
            if not limit:
                continue
            if self.device.type == 'cpu':
                bits = source.view(self.t.int64) if k == 'f64' else source
                for row in range(batch.rows):
                    if bool(good[row]):
                        chosen = self.t.nonzero(bits[:, row], as_tuple=False)[:, 0]
                        slot = int(slots[row]); n = len(chosen)
                        self.indices[k][:n, slot].copy_(chosen.to(self.t.int32))
                        self.values[k][:n, slot].copy_(source[chosen, row])
            else:
                work = workspace[k]
                self.module.launch('pack_'+k, [source, good.view(self.t.uint8), slots,
                    work['prefix'], self.indices[k], self.values[k], errors, batch.rows,
                    self.schema.lengths[k], work['tiles'], self.capacity, limit],
                    grid=(work['tiles'], batch.rows), block=256, capture_safe=True)

    def put_device(self, batch, owners, active, *, errors=None):
        self._open(); t = self.t; rows = self._check_batch(batch)
        self._vector(owners, t.int64, rows); self._vector(active, t.bool, rows)
        errors = t.zeros(rows, dtype=t.int32, device=self.device) if errors is None else errors.clone()
        self._vector(errors, t.int32, rows)
        root = self.schema.pools[self.schema.root]
        valid = (owners >= 0) & (batch.roots > root.base) & (batch.roots <= root.base+root.capacity)
        errors.bitwise_or_((active & ~valid).to(t.int32)*INVALID_LEASE)
        wanted = active & valid & (errors == 0)
        workspace = self.prepare_workspace(rows)
        self._count(batch, wanted, workspace)
        for k in WIDTHS:
            errors.bitwise_or_((wanted & (workspace[k]['total'] > self.column_cells[k])).to(t.int32)*CAPACITY)
        wanted = wanted & (errors == 0)
        fits = wanted.sum() <= (~self.live).sum()
        errors.bitwise_or_((wanted & ~fits).to(t.int32)*CAPACITY)
        good = wanted & fits
        free = t.argsort(self.live.long()*self.capacity+self.ids)
        rank = (good.long().cumsum(0)-1).clamp(0, self.capacity-1)
        slots = free[rank]
        self._pack(batch, slots, good, workspace, errors)
        good = good & (errors == 0)
        for k in WIDTHS:
            target = self.counts[k]
            target.index_add_(0, slots, t.where(good, workspace[k]['total']-target[slots], 0))
        self.roots.index_add_(0, slots, t.where(good, batch.roots-self.roots[slots], 0))
        self.generation.index_add_(0, slots, good.long())
        self.owner.index_add_(0, slots, t.where(good, owners-self.owner[slots], 0))
        added = t.zeros(self.capacity, dtype=t.int64, device=self.device)
        added.index_add_(0, slots, good.long())
        self.live.logical_or_(added > 0)
        t.maximum(self.peak_live, self.live.sum(dtype=t.int64).reshape(1), out=self.peak_live)
        return DeviceStateBatch(self, t.where(good, slots, -1), t.where(good, self.generation[slots], 0),
                                owners.clone(), good, errors)

    def gather_into_device(self, states, destination, *, shared_reads=False):
        self._open(); t = self.t; rows = len(states.slots)
        if not 1 <= rows <= 64:
            raise ValueError('Dense gather must use 1..64 bounded work rows')
        if (not isinstance(destination, TorchRecordSoABatch) or destination.schema.identity != self.schema.identity
                or destination.rows != rows or set(destination.buffers) != set(WIDTHS)):
            raise ValueError('Matching dense typed work rows required')
        indices, good = self.validate(states, shared_reads=shared_reads)
        for k, target in destination.buffers.items():
            if (target.device != self.device or target.dtype != self.dtypes[k]
                    or target.shape != (self.schema.lengths[k], rows) or not target.is_contiguous()):
                raise ValueError('Exact contiguous typed workspace required')
            target.zero_(); limit = self.column_cells[k]
            if not limit:
                continue
            if self.device.type == 'cpu':
                for row in range(rows):
                    if not bool(good[row]):
                        continue
                    slot = int(indices[row]); count = int(self.counts[k][slot])
                    cols = self.indices[k][:max(0, min(count, limit)), slot].long()
                    corrupt = (not 0 <= count <= limit or bool(((cols < 0) | (cols >= target.shape[0])).any())
                               or bool((cols[1:] <= cols[:-1]).any()))
                    if corrupt:
                        states.errors[row] |= CORRUPT_COLUMN
                    else:
                        target[cols, row] = self.values[k][:count, slot]
            else:
                self.module.launch('gather_'+k, [self.indices[k], self.values[k], self.counts[k], indices,
                    good.view(t.uint8), target, states.errors, rows, self.schema.lengths[k], self.capacity, limit],
                    grid=((limit+255)//256, rows), block=256, capture_safe=True)
        good = good & (states.errors == 0)
        for value in destination.buffers.values():
            value.masked_fill_(~good[None, :], 0)
        t.index_select(self.roots, 0, indices, out=destination.roots)
        destination.roots.masked_fill_(~good, 0)
        return destination

    def gather_device(self, states, *, shared_reads=False):
        rows = len(states.slots)
        if not 1 <= rows <= 64:
            raise ValueError('Dense gather must use 1..64 bounded work rows')
        buffers = {k: self.t.empty((self.schema.lengths[k], rows), dtype=v, device=self.device)
                   for k, v in self.dtypes.items()}
        roots = self.t.empty(rows, dtype=self.t.int32, device=self.device)
        return self.gather_into_device(states, TorchRecordSoABatch(self.schema, buffers, roots),
                                       shared_reads=shared_reads)

    def release_device(self, states):
        self._open(); super().release_device(states)
        for values in self.counts.values():
            values.masked_fill_(~self.live, 0)

    def metrics_device(self):
        return dict(live_slots=self.live.sum(), peak_live_slots=self.peak_live.clone(),
                    cells={k: value.to(self.t.int64).sum() for k, value in self.counts.items()})

    @property
    def allocated_bytes(self):
        return sum(v.numel()*v.element_size() for v in (
            *self.values.values(), *self.indices.values(), *self.counts.values(),
            self.roots, self.live, self.generation, self.owner, self.ids, self.peak_live))

    @property
    def workspace_bytes(self):
        return sum(v.numel()*v.element_size() for work in self._workspaces.values()
                   for field in work.values() for v in (field['counts'], field['prefix'], field['total']))

    def close(self):
        if not self.closed:
            if self.module is not None:
                self.module.close()
            self.closed = True
