"""Compact paged device graph store. Not a rule engine or readiness claim.

Stores only used node/edge/payload bytes behind checked owner/generation leases.
Page arenas stay separate so unused GraphLayout maxima do not allocate holes.
"""
from dataclasses import dataclass

from .cuda_vm import DeviceGraph
from .device_graph_pool import (
    CORRUPT_PAGES, DUPLICATE_SLOT, DeviceGraphPagePool, DevicePageHandle,
    INVALID_HANDLE, INVALID_SIZE, PAGE_CAPACITY,
)
from .device_rule_protocol import DeviceStateBatch


SLOT_CAPACITY = 64
MAX_NODES, MAX_EDGES, MAX_PAYLOAD = 16384, 32768, 131072
_NODE_BYTES, _EDGE_BYTES = 16, 8


@dataclass(frozen=True)
class DeviceGatherResult:
    graph: DeviceGraph
    errors: object


def emit_copy_source():
    source=r'''
extern "C" __global__ void copy_graph_pages(
 unsigned char* dense,long long stride,unsigned char* pages,
 int page_bytes,int page_capacity,const long long* page_table,const long long* page_owner,int max_pages,
 const long long* slots,int slot_capacity,const long long* used,const unsigned char* take,
 int* errors,int rows,int to_pages){
 int row=blockIdx.x;if(row>=rows)return;
 __shared__ int valid;__shared__ long long n,slot;
 if(threadIdx.x==0){
  errors[row]=0;valid=0;n=used[row];slot=slots[row];
  if(take[row]){
   if(slot<0||slot>=slot_capacity)errors[row]=2;
   else if(n<1||n>stride||n>(long long)page_bytes*max_pages)errors[row]=8;
   else{
    valid=1;long long required=(n+page_bytes-1)/page_bytes;
    for(long long page=0;page<required;page++){
     long long phys=page_table[slot*max_pages+page];
     if(phys<0||phys>=page_capacity||page_owner[phys]!=slot){errors[row]=32;valid=0;break;}
    }
   }
  }
 }
 __syncthreads();if(!valid)return;
 for(long long i=threadIdx.x;i<n;i+=blockDim.x){
  long long phys=page_table[slot*max_pages+i/page_bytes];
  long long offset=phys*page_bytes+i%page_bytes;
  if(to_pages)pages[offset]=dense[(long long)row*stride+i];
  else dense[(long long)row*stride+i]=pages[offset];
 }
}
'''
    return source.replace('errors[row]=2;',f'errors[row]={INVALID_HANDLE};').replace('errors[row]=8;',f'errors[row]={INVALID_SIZE};').replace('errors[row]=32;',f'errors[row]={CORRUPT_PAGES};')


_COPY_SPEC = ('uint8*','int64','uint8*','int32','int32','int64*','int64*','int32',
              'int64*','int32','int64*','uint8*','int32*','int32','int32')



class CompactDeviceGraphStore:
    """Device-only compact graph leases. CPU is an explicit storage oracle."""
    supports_full_duel_rules = False

    def __init__(self, *, slots, node_pages, edge_pages, payload_pages,
                 node_page_bytes, edge_page_bytes, payload_page_bytes,
                 max_node_pages_per_slot, max_edge_pages_per_slot,
                 max_payload_pages_per_slot, device='cuda',
                 allow_cpu_reference=False):
        import torch
        self.t = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (
                self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit storage oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU storage fallback')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        kw = dict(slots=slots, device=str(self.device),
                  allow_cpu_reference=allow_cpu_reference)
        self.nodes = DeviceGraphPagePool(
            pages=node_pages, page_bytes=node_page_bytes,
            max_pages_per_slot=max_node_pages_per_slot, **kw)
        self.edges = DeviceGraphPagePool(
            pages=edge_pages, page_bytes=edge_page_bytes,
            max_pages_per_slot=max_edge_pages_per_slot, **kw)
        self.payload = DeviceGraphPagePool(
            pages=payload_pages, page_bytes=payload_page_bytes,
            max_pages_per_slot=max_payload_pages_per_slot, **kw)
        self.slot_capacity = slots
        self.generation = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.owner = torch.full((slots,), -1, dtype=torch.int64, device=self.device)
        self.live = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.counts = torch.zeros((slots, 3), dtype=torch.int64, device=self.device)
        self.roots = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.component_generations = torch.zeros((slots,3),dtype=torch.int64,device=self.device)
        self.peak_used_bytes = torch.zeros((), dtype=torch.int64, device=self.device)
        self._slot_ids = torch.arange(slots, device=self.device)
        self.module = None
        self.closed = False
        if self.device.type == 'cuda':
            from .cuda_runtime import CudaModule
            self.module = CudaModule(
                emit_copy_source(),
                {'copy_graph_pages': _COPY_SPEC},
                device=str(self.device))

    def close(self):
        if self.closed:
            return
        if self.module is not None:
            self.module.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def allocated_bytes(self):
        names = ('generation', 'owner', 'live', 'counts', 'roots',
                 'peak_used_bytes', '_slot_ids', 'component_generations')
        return (sum(getattr(self, n).numel() * getattr(self, n).element_size()
                    for n in names)
                + self.nodes.allocated_bytes + self.edges.allocated_bytes
                + self.payload.allocated_bytes)

    def metrics_device(self):
        used = (self.nodes.logical_bytes.sum() + self.edges.logical_bytes.sum()
                + self.payload.logical_bytes.sum())
        return dict(used_bytes=used, live_slots=self.live.sum(),
                    peak_used_bytes=self.peak_used_bytes)

    def _vector(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact device vector shape and dtype required')

    def _require_graph(self, graph):
        if self.closed:
            raise RuntimeError('Graph store closed')
        if not isinstance(graph, DeviceGraph):
            raise TypeError('DeviceGraph required')
        t, layout, rows = self.t, graph.layout, graph.rows
        if rows < 1:
            raise ValueError('Nonempty device batch required')
        if (layout.max_nodes > MAX_NODES or layout.max_edges > MAX_EDGES
                or layout.max_payload_bytes > MAX_PAYLOAD):
            raise ValueError('GraphLayout exceeds compact store source maxima')
        expected = (
            (graph.nodes, t.int32, (rows, layout.max_nodes, 4)),
            (graph.edges, t.int32, (rows, layout.max_edges, 2)),
            (graph.payload, t.uint8, (rows, layout.max_payload_bytes)),
            (graph.counts, t.int32, (rows, 3)),
            (graph.roots, t.int32, (rows,)),
        )
        for value, dtype, shape in expected:
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or tuple(value.shape) != shape
                    or not value.is_contiguous()):
                raise ValueError('Exact device graph buffers required')
        return rows, layout

    def _row_bytes(self, tensor):
        return tensor.reshape(tensor.shape[0], -1).view(self.t.uint8)

    def _copy(self, *, to_pages, pool, dense, slots, used, take):
        t = self.t
        rows = int(slots.shape[0])
        src = self._row_bytes(dense)
        # Payload count columns are strided views. The native pointer ABI is
        # contiguous even when the logical vector has the correct dtype.
        used=used.contiguous()
        take_u8 = take.view(t.uint8) if take.dtype == t.bool else take
        if self.module is None:
            return self._copy_cpu(to_pages, pool, src, slots, used, take_u8)
        errors=t.zeros(rows,dtype=t.int32,device=self.device)
        self.module.launch(
            'copy_graph_pages',
            [src, int(src.shape[1]), pool.data.view(-1), int(pool.page_bytes),
             int(pool.page_capacity), pool.page_table,pool.page_owner,int(pool.max_pages_per_slot),
             slots, int(self.slot_capacity), used, take_u8,errors,rows,int(to_pages)],
            grid=rows, block=128, capture_safe=True)
        return errors

    def _copy_cpu(self, to_pages, pool, dense, slots, used, take):
        t = self.t
        rows = dense.shape[0]
        stride = dense.shape[1]
        idx = slots.clamp(0, pool.slot_capacity - 1)
        table = pool.page_table[idx]
        enabled=take.bool()
        size_ok=((used>0)&(used<=stride)&(used<=pool.page_bytes*pool.max_pages_per_slot))
        in_range=(slots>=0)&(slots<pool.slot_capacity)
        occupied=pool._positions[None,:]<(used+pool.page_bytes-1)[:,None]//pool.page_bytes
        physical=table.clamp(0,pool.page_capacity-1)
        intact=(~occupied|((table>=0)&(table<pool.page_capacity)
                &(pool.page_owner[physical]==idx[:,None]))).all(-1)
        errors=((enabled&~in_range).to(t.int32)*INVALID_HANDLE
                |(enabled&in_range&~size_ok).to(t.int32)*INVALID_SIZE
                |(enabled&in_range&size_ok&~intact).to(t.int32)*CORRUPT_PAGES)
        take=enabled&(errors==0)
        width = pool.page_bytes
        byte_ids = t.arange(width, device=self.device)
        if to_pages:
            acc = pool.data.view(-1).to(t.int64)
        else:
            dense.zero_()
            acc = dense.view(-1).to(t.int64)
        row_ids = t.arange(rows, device=self.device)
        for page in range(pool.max_pages_per_slot):
            logical = page * width + byte_ids
            take2 = (take[:, None].bool() & (logical[None, :] < used[:, None])
                     & (logical[None, :] < stride))
            phys = table[:, page].clamp(0, pool.page_capacity - 1)
            if to_pages:
                src_idx = logical.clamp(0, stride - 1)
                vals = dense[:, src_idx].long() * take2.long()
                dest = phys[:, None] * width + byte_ids[None, :]
            else:
                vals = pool.data[phys].long() * take2.long()
                dest = row_ids[:, None] * stride + logical[None, :].clamp(0, stride - 1)
            acc.index_add_(0, dest.reshape(-1), vals.reshape(-1))
        if to_pages:
            pool.data.view(-1).copy_(acc.to(t.uint8))
        else:
            dense.view(-1).copy_(acc.to(t.uint8))
        return errors

    def _leases(self, slots, generations, owners, active, *, live):
        t = self.t
        self._vector(slots, t.int64, slots.shape)
        shape = slots.shape
        if slots.ndim != 1 or not 0 < slots.numel() <= self.slot_capacity:
            raise ValueError('Nonempty bounded slot batch required')
        self._vector(generations, t.int64, shape)
        self._vector(owners, t.int64, shape)
        if (not isinstance(active, t.Tensor) or active.device != self.device
                or active.dtype not in (t.bool, t.uint8) or tuple(active.shape) != shape
                or not active.is_contiguous()):
            raise ValueError('Exact device active mask required')
        active_b = active.bool()
        indices = slots.clamp(0, self.slot_capacity - 1)
        in_range = (slots >= 0) & (slots < self.slot_capacity)
        valid = (in_range & (generations == self.generation[indices])
                 & (owners >= 0) & (self.live[indices] == int(live)))
        if live:
            valid = valid & (owners == self.owner[indices])
        histogram = t.zeros(self.slot_capacity, dtype=t.int64, device=self.device)
        histogram.index_add_(0, indices, (active_b & in_range).long())
        duplicate = histogram[indices] > 1
        errors = (active_b & ~valid).to(t.int32) * INVALID_HANDLE
        errors = errors | ((active_b & duplicate).to(t.int32) * DUPLICATE_SLOT)
        return indices, active_b & valid & ~duplicate, errors, active_b

    def _pages_needed(self, pool, byte_count, take):
        t = self.t
        safe = byte_count.clamp(0, pool.page_bytes * pool.max_pages_per_slot)
        pages = (safe + pool.page_bytes - 1) // pool.page_bytes
        return t.where(take & (byte_count > 0), pages, 0)

    def _component_integrity(self,idx,owners,good):
        """Check all arenas before copying or releasing any component."""
        t=self.t;counts=self.counts[idx]
        intact=((counts[:,0]>0)&(counts[:,0]<=MAX_NODES)
                &(counts[:,1]>=0)&(counts[:,1]<=MAX_EDGES)
                &(counts[:,2]>=0)&(counts[:,2]<=MAX_PAYLOAD)
                &(self.roots[idx]>=0)&(self.roots[idx]<counts[:,0]))
        for column,pool,scale,bound in ((0,self.nodes,_NODE_BYTES,MAX_NODES),
                (1,self.edges,_EDGE_BYTES,MAX_EDGES),(2,self.payload,1,MAX_PAYLOAD)):
            expected=counts[:,column].clamp(0,bound)*scale
            present=expected>0
            count=(expected+pool.page_bytes-1)//pool.page_bytes
            table=pool.page_table[idx]
            occupied=pool._positions[None,:]<count[:,None]
            physical=table.clamp(0,pool.page_capacity-1)
            pages_ok=(~occupied|((table>=0)&(table<pool.page_capacity)
                      &(pool.page_owner[physical]==idx[:,None]))).all(-1)
            sorted_pages=t.sort(t.where(occupied,table,pool.page_capacity),dim=-1).values
            duplicate=((sorted_pages[:,1:]==sorted_pages[:,:-1])
                       &(sorted_pages[:,1:]<pool.page_capacity)).any(-1)
            state_ok=((pool.generation[idx]==self.component_generations[idx,column])
                      &(pool.live[idx]==present.long())
                      &(pool.logical_bytes[idx]==expected)&(pool.page_count[idx]==count)
                      &(pool.owner[idx]==t.where(present,owners,-1))
                      &(count<=pool.max_pages_per_slot)&pages_ok&~duplicate)
            intact=intact&state_ok
        return good&intact,(good&~intact).to(t.int32)*CORRUPT_PAGES

    def put_device(self, graph, owners, active):
        rows, layout = self._require_graph(graph)
        t = self.t
        self._vector(owners, t.int64, (rows,))
        if (not isinstance(active, t.Tensor) or active.device != self.device
                or active.dtype not in (t.bool, t.uint8) or tuple(active.shape) != (rows,)
                or not active.is_contiguous()):
            raise ValueError('Exact device active mask required')
        active_b = active.bool()
        counts = graph.counts.long()
        node_b = counts[:, 0] * _NODE_BYTES
        edge_b = counts[:, 1] * _EDGE_BYTES
        pay_b = counts[:, 2]
        size_ok = ((counts[:, 0] > 0) & (counts[:, 0] <= layout.max_nodes)
                   & (counts[:, 1] >= 0) & (counts[:, 1] <= layout.max_edges)
                   & (counts[:, 2] >= 0) & (counts[:, 2] <= layout.max_payload_bytes)
                   & (node_b <= self.nodes.page_bytes * self.nodes.max_pages_per_slot)
                   & (edge_b <= self.edges.page_bytes * self.edges.max_pages_per_slot)
                   & (pay_b <= self.payload.page_bytes * self.payload.max_pages_per_slot)
                   & (owners >= 0)&(graph.roots>=0)&(graph.roots.long()<counts[:,0]))
        errors = (active_b & ~size_ok).to(t.int32) * INVALID_SIZE
        good = active_b & size_ok
        free = self.live == 0
        free_order = t.argsort((~free).long() * self.slot_capacity + self._slot_ids)
        rank = good.long().cumsum(0) - good.long()
        selected = free_order[rank.clamp(0, self.slot_capacity - 1)]
        need = good.long()
        fits_slots = need.sum() <= free.sum()
        n_need = self._pages_needed(self.nodes, node_b, good)
        e_need = self._pages_needed(self.edges, edge_b, good)
        p_need = self._pages_needed(self.payload, pay_b, good)
        fits_pages = ((n_need.sum() <= (self.nodes.page_owner < 0).sum())
                      & (e_need.sum() <= (self.edges.page_owner < 0).sum())
                      & (p_need.sum() <= (self.payload.page_owner < 0).sum()))
        fits = fits_slots & fits_pages
        cap_code = t.where(
            fits_slots,
            t.full_like(errors,PAGE_CAPACITY),t.full_like(errors,SLOT_CAPACITY))
        errors = errors | ((good & ~fits).to(t.int32) * cap_code)
        good = good & fits
        idx = selected.clamp(0, self.slot_capacity - 1)
        n_take = good & (node_b > 0)
        e_take = good & (edge_b > 0)
        p_take = good & (pay_b > 0)
        n_gens, e_gens, p_gens = (
            self.nodes.generation[idx], self.edges.generation[idx],
            self.payload.generation[idx])
        n_res = self.nodes.reserve(selected, n_gens, owners, node_b, n_take)
        e_res = self.edges.reserve(selected, e_gens, owners, edge_b, e_take)
        p_res = self.payload.reserve(selected, p_gens, owners, pay_b, p_take)
        comp_ok = ((n_res.errors == 0) & (e_res.errors == 0) & (p_res.errors == 0))
        final = good & comp_ok
        self.nodes.release(n_res.handle, n_take & (n_res.errors == 0) & ~final)
        self.edges.release(e_res.handle, e_take & (e_res.errors == 0) & ~final)
        self.payload.release(p_res.handle, p_take & (p_res.errors == 0) & ~final)
        errors = errors | ((good & ~final).to(t.int32) * PAGE_CAPACITY)
        errors = errors | n_res.errors.to(t.int32) | e_res.errors.to(t.int32) | p_res.errors.to(t.int32)
        take = final
        n_copy=self._copy(to_pages=True, pool=self.nodes, dense=graph.nodes,
                   slots=selected, used=node_b, take=take & n_take)
        e_copy=self._copy(to_pages=True, pool=self.edges, dense=graph.edges,
                   slots=selected, used=edge_b, take=take & e_take)
        p_copy=self._copy(to_pages=True, pool=self.payload, dense=graph.payload,
                   slots=selected, used=pay_b, take=take & p_take)
        copy_errors=n_copy|e_copy|p_copy
        rollback=take&(copy_errors!=0)
        errors=errors|copy_errors
        errors=errors|self.nodes.release(n_res.handle,rollback&n_take).to(t.int32)
        errors=errors|self.edges.release(e_res.handle,rollback&e_take).to(t.int32)
        errors=errors|self.payload.release(p_res.handle,rollback&p_take).to(t.int32)
        take=take&(copy_errors==0)
        self.live.index_add_(0, idx, take.long())
        self.owner.index_add_(0, idx, t.where(take, owners + 1, 0))
        self.counts[:, 0].index_add_(0, idx, t.where(take, counts[:, 0], 0))
        self.counts[:, 1].index_add_(0, idx, t.where(take, counts[:, 1], 0))
        self.counts[:, 2].index_add_(0, idx, t.where(take, counts[:, 2], 0))
        self.roots.index_add_(0, idx, t.where(take, graph.roots.long(), 0))
        component_gens=t.stack((self.nodes.generation[idx],self.edges.generation[idx],
                                self.payload.generation[idx]),dim=-1)
        self.component_generations.index_add_(0,idx,t.where(take[:,None],
            component_gens-self.component_generations[idx],0))
        used = (self.nodes.logical_bytes.sum() + self.edges.logical_bytes.sum()
                + self.payload.logical_bytes.sum())
        self.peak_used_bytes.copy_(t.maximum(self.peak_used_bytes, used))
        slots_out = t.where(take, selected, t.full_like(selected, -1))
        gens_out = t.where(take, self.generation[idx], 0)
        return DeviceStateBatch(self, slots_out, gens_out, owners, active_b, errors)

    def gather_into_device(self, states, destination):
        if not isinstance(states, DeviceStateBatch):
            raise TypeError('DeviceStateBatch required')
        if states.pool is not self:
            raise ValueError('DeviceStateBatch does not belong to this store')
        rows, layout = self._require_graph(destination)
        t = self.t
        self._vector(states.slots, t.int64, (rows,))
        self._vector(states.errors,t.int32,(rows,))
        idx, good, errors, active = self._leases(
            states.slots, states.generations, states.owners, states.active, live=True)
        good,component_errors=self._component_integrity(idx,states.owners,good)
        errors=errors|component_errors|states.errors
        good=good&(errors==0)
        stored = self.counts[idx]
        overflow = good & ((stored[:, 0] > layout.max_nodes)
                           | (stored[:, 1] > layout.max_edges)
                           | (stored[:, 2] > layout.max_payload_bytes))
        errors = errors | (overflow.to(t.int32) * INVALID_SIZE)
        good = good & ~overflow
        destination.nodes.zero_()
        destination.edges.zero_()
        destination.payload.zero_()
        destination.counts.zero_()
        destination.roots.zero_()
        node_b = stored[:, 0] * _NODE_BYTES
        edge_b = stored[:, 1] * _EDGE_BYTES
        pay_b = stored[:, 2]
        n_copy=self._copy(to_pages=False, pool=self.nodes, dense=destination.nodes,
                   slots=states.slots, used=node_b, take=good & (stored[:, 0] > 0))
        e_copy=self._copy(to_pages=False, pool=self.edges, dense=destination.edges,
                   slots=states.slots, used=edge_b, take=good & (stored[:, 1] > 0))
        p_copy=self._copy(to_pages=False, pool=self.payload, dense=destination.payload,
                   slots=states.slots, used=pay_b, take=good & (stored[:, 2] > 0))
        errors=errors|n_copy|e_copy|p_copy
        good=good&(errors==0)
        destination.counts.copy_(t.where(good[:, None], stored.to(t.int32), 0))
        destination.roots.copy_(t.where(good, self.roots[idx].to(t.int32), 0))
        return DeviceGatherResult(destination, errors)

    def release_device(self, states):
        if not isinstance(states, DeviceStateBatch):
            raise TypeError('DeviceStateBatch required')
        if states.pool is not self:
            raise ValueError('DeviceStateBatch does not belong to this store')
        if self.closed:raise RuntimeError('Graph store closed')
        t = self.t
        self._vector(states.slots, t.int64, states.slots.shape)
        self._vector(states.errors,t.int32,states.slots.shape)
        idx, good, errors, active = self._leases(
            states.slots, states.generations, states.owners, states.active, live=True)
        good,component_errors=self._component_integrity(idx,states.owners,good)
        errors=errors|component_errors
        n_take = good & (self.counts[idx, 0] > 0)
        e_take = good & (self.counts[idx, 1] > 0)
        p_take = good & (self.counts[idx, 2] > 0)
        n_err = self.nodes.release(
            DevicePageHandle(states.slots, self.nodes.generation[idx], states.owners), n_take)
        e_err = self.edges.release(
            DevicePageHandle(states.slots, self.edges.generation[idx], states.owners), e_take)
        p_err = self.payload.release(
            DevicePageHandle(states.slots, self.payload.generation[idx], states.owners), p_take)
        errors = (errors | n_err.to(t.int32) | e_err.to(t.int32) | p_err.to(t.int32)
                  | ((good & ((n_err != 0) | (e_err != 0) | (p_err != 0))).to(t.int32)
                     * CORRUPT_PAGES))
        ok = good & (n_err == 0) & (e_err == 0) & (p_err == 0)
        self.counts[:, 0].index_add_(0, idx, -t.where(ok, self.counts[idx, 0], 0))
        self.counts[:, 1].index_add_(0, idx, -t.where(ok, self.counts[idx, 1], 0))
        self.counts[:, 2].index_add_(0, idx, -t.where(ok, self.counts[idx, 2], 0))
        self.roots.index_add_(0, idx, -t.where(ok, self.roots[idx], 0))
        self.owner.index_add_(0, idx, -t.where(ok, self.owner[idx] + 1, 0))
        self.live.index_add_(0, idx, -ok.long())
        self.generation.index_add_(0, idx, ok.long())
        states.errors.copy_(states.errors | errors)

