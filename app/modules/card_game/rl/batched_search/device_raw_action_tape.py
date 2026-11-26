"""Bounded CUDA archive of physical selected raw-action graphs.

Ordinals are insufficient: each admitted record stores the compacted used
node/edge/payload prefix of one selected raw action, with row-local refs.
This is a reusable primitive, not a collector, driver, or qualification.
Call it only after the physical selected action graph is compacted and
before apply. Hypothetical search graphs are out of scope.
"""
from ..batched_duel.cuda_vm import DeviceGraph


INVALID_ID, OWNER_ERROR, ACTOR_ERROR, STEP_ERROR = 1, 2, 4, 8
STALE_ERROR, DUPLICATE_ERROR, RANGE_ERROR, CAPACITY_ERROR = 16, 32, 64, 128
CAVEAT = (
    'Raw action graphs are a bounded physical archive only. '
    'Ordinals are insufficient; this primitive does not retain immutable '
    'initial physical states, prove terminal or commit success, or qualify '
    'full duel rules.'
)


def emit_copy_source():
    return r'''
extern "C" __global__ void scatter_raw_action(
 const int* sn,const int* se,const unsigned char* sp,const int* sc,
 int* dn,int* de,unsigned char* dp,
 const long long* node_off,const long long* edge_off,const long long* pay_off,
 const unsigned char* take,int rows,int smn,int sme,int smp){
 int row=blockIdx.x;if(row>=rows||!take[row])return;
 int n=sc[row*3],e=sc[row*3+1],p=sc[row*3+2];
 long long no=node_off[row],eo=edge_off[row],po=pay_off[row];
 for(long long i=threadIdx.x;i<(long long)n*4;i+=blockDim.x)
  dn[no*4+i]=sn[(long long)row*smn*4+i];
 for(long long i=threadIdx.x;i<(long long)e*2;i+=blockDim.x)
  de[eo*2+i]=se[(long long)row*sme*2+i];
 for(long long i=threadIdx.x;i<p;i+=blockDim.x)
  dp[po+i]=sp[(long long)row*smp+i];
}
extern "C" __global__ void gather_raw_action(
 const int* sn,const int* se,const unsigned char* sp,
 int* dn,int* de,unsigned char* dp,int* dc,int* dr,
 const long long* node_off,const long long* node_cnt,
 const long long* edge_off,const long long* edge_cnt,
 const long long* pay_off,const long long* pay_cnt,
 const int* roots,const unsigned char* take,
 int rows,int dmn,int dme,int dmp){
 int row=blockIdx.x;if(row>=rows)return;
 __shared__ int valid,n,e,p;
 if(threadIdx.x==0){
  valid=take[row];n=(int)node_cnt[row];e=(int)edge_cnt[row];p=(int)pay_cnt[row];
  if(!valid){dc[row*3]=0;dc[row*3+1]=0;dc[row*3+2]=0;dr[row]=0;}
 }
 __syncthreads();if(!valid)return;
 long long no=node_off[row],eo=edge_off[row],po=pay_off[row];
 for(long long i=threadIdx.x;i<(long long)n*4;i+=blockDim.x)
  dn[(long long)row*dmn*4+i]=sn[no*4+i];
 for(long long i=threadIdx.x;i<(long long)e*2;i+=blockDim.x)
  de[(long long)row*dme*2+i]=se[eo*2+i];
 for(long long i=threadIdx.x;i<p;i+=blockDim.x)
  dp[(long long)row*dmp+i]=sp[po+i];
 __syncthreads();if(threadIdx.x==0){
  dc[row*3]=n;dc[row*3+1]=e;dc[row*3+2]=p;dr[row]=roots[row];
 }
}
'''


_SCATTER_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'int32*', 'uint8*',
    'int64*', 'int64*', 'int64*', 'uint8*', 'int32', 'int32', 'int32', 'int32',
)
_GATHER_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int64*', 'int64*', 'int64*', 'int64*', 'int64*', 'int64*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32',
)


class DeviceRawActionTape:
    """Fixed-capacity device archive for begin_device/append_device/gather_into_device."""

    supports_full_duel_rules = False

    def __init__(self, rows, max_records, node_capacity, edge_capacity, payload_capacity,
                 device='cuda', allow_cpu_reference=False):
        import torch
        if type(allow_cpu_reference) is not bool:
            raise TypeError('Explicit boolean CPU control required')
        for name, value in (('rows', rows), ('max_records', max_records),
                            ('node_capacity', node_capacity),
                            ('edge_capacity', edge_capacity),
                            ('payload_capacity', payload_capacity)):
            if type(value) is not int or not 1 <= value < 2**31:
                raise ValueError('Positive bounded ' + name + ' required')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (
                self.device.type == 'cpu' and allow_cpu_reference is True):
            raise ValueError('CUDA required; CPU is an explicit raw-action-tape oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no host loop fallback')
        self.allow_cpu_reference = allow_cpu_reference is True
        self.rows = rows
        self.max_records = max_records
        self.node_capacity = node_capacity
        self.edge_capacity = edge_capacity
        self.payload_capacity = payload_capacity
        z = dict(device=self.device)
        rec_n = max_records + 1
        self._record_sink = torch.tensor(max_records, dtype=torch.int64, **z)
        self._node_sink = torch.tensor(node_capacity, dtype=torch.int64, **z)
        self._edge_sink = torch.tensor(edge_capacity, dtype=torch.int64, **z)
        self._payload_sink = torch.tensor(payload_capacity, dtype=torch.int64, **z)
        self._record_count = torch.zeros((), dtype=torch.int64, **z)
        self._node_used = torch.zeros((), dtype=torch.int64, **z)
        self._edge_used = torch.zeros((), dtype=torch.int64, **z)
        self._payload_used = torch.zeros((), dtype=torch.int64, **z)
        self._job_seeds = torch.zeros(rows, dtype=torch.int64, **z)
        self._owners = torch.full((rows,), -1, dtype=torch.int64, **z)
        self._expected_step = torch.zeros(rows, dtype=torch.int64, **z)
        self._last_version = torch.full((rows,), -1, dtype=torch.int64, **z)
        self._begun = torch.zeros(rows, dtype=torch.bool, **z)
        self._game_errors = torch.zeros(rows, dtype=torch.int32, **z)
        self._rec_game = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_owner = torch.full((rec_n,), -1, dtype=torch.int64, **z)
        self._rec_seed = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_actor = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_step = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_version = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_root = torch.zeros(rec_n, dtype=torch.int32, **z)
        self._rec_node_off = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_node_count = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_edge_off = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_edge_count = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_pay_off = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._rec_pay_count = torch.zeros(rec_n, dtype=torch.int64, **z)
        self._nodes = torch.zeros((node_capacity + 1, 4), dtype=torch.int32, **z)
        self._edges = torch.zeros((edge_capacity + 1, 2), dtype=torch.int32, **z)
        self._payload = torch.zeros((payload_capacity + 1,), dtype=torch.uint8, **z)
        self._ready = False
        self.closed = False
        self.module = None
        if self.device.type == 'cuda':
            from ..batched_duel.cuda_runtime import CudaModule
            self.module = CudaModule(
                emit_copy_source(),
                {'scatter_raw_action': _SCATTER_SPEC, 'gather_raw_action': _GATHER_SPEC},
                device=str(self.device))

    def close(self):
        if self.closed:
            return
        if self.module is not None:
            self.module.close()
        self.closed = True

    @property
    def allocated_bytes(self):
        """Persistent Torch buffers only; excludes VM/image/process memory."""
        return sum(value.numel() * value.element_size() for value in vars(self).values()
                   if isinstance(value, self.t.Tensor))

    @property
    def errors_device(self):
        """Borrowed permanent per-game archive errors, for terminal qualification."""
        self._require_open()
        if not self._ready:
            raise RuntimeError('Raw action cohort has not begun')
        return self._game_errors

    @property
    def steps_device(self):
        """Borrowed count of successfully archived physical actions per game."""
        self._require_open()
        if not self._ready:
            raise RuntimeError('Raw action cohort has not begun')
        return self._expected_step

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _vector(self, value, dtype, shape):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact bound-device raw-action-tape tensor metadata required')

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Raw action tape closed')

    def _require_graph(self, graph):
        self._require_open()
        if not isinstance(graph, DeviceGraph):
            raise TypeError('DeviceGraph required')
        t, layout, lanes = self.t, graph.layout, graph.rows
        if lanes < 1:
            raise ValueError('Nonempty device batch required')
        expected = (
            (graph.nodes, t.int32, (lanes, layout.max_nodes, 4)),
            (graph.edges, t.int32, (lanes, layout.max_edges, 2)),
            (graph.payload, t.uint8, (lanes, layout.max_payload_bytes)),
            (graph.counts, t.int32, (lanes, 3)),
            (graph.roots, t.int32, (lanes,)),
        )
        for value, dtype, shape in expected:
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or tuple(value.shape) != shape
                    or not value.is_contiguous()):
                raise ValueError('Exact device graph buffers required')
        return lanes, layout

    def _or_into_games(self, game, flags, take):
        t = self.t
        bits = t.arange(16, dtype=t.int64, device=self.device)
        present = take[:, None] & ((flags[:, None].to(t.int64) & (1 << bits)[None, :]) != 0)
        acc = t.zeros((self.rows, 16), dtype=t.int64, device=self.device)
        acc.index_add_(0, game, present.long())
        merged = ((acc > 0).to(t.int32) * (1 << bits).to(t.int32)).sum(-1).to(t.int32)
        self._game_errors.bitwise_or_(merged)

    def begin_device(self, owners, job_seeds):
        t = self.t
        self._require_open()
        self._vector(owners, t.int64, (self.rows,))
        self._vector(job_seeds, t.int64, (self.rows,))
        self._record_count.zero_()
        self._node_used.zero_()
        self._edge_used.zero_()
        self._payload_used.zero_()
        self._expected_step.zero_()
        self._last_version.fill_(-1)
        self._game_errors.zero_()
        invalid = owners < 0
        begun = ~invalid
        self._begun.copy_(begun)
        self._owners.copy_(t.where(begun, owners, t.full_like(owners, -1)))
        self._job_seeds.copy_(t.where(begun, job_seeds, t.zeros_like(job_seeds)))
        self._ready = True
        return invalid.to(t.int32) * OWNER_ERROR

    def append_device(self, graph, game_rows, owners, steps, versions, actors, active):
        t = self.t
        if not self._ready:
            raise RuntimeError('begin_device must allocate the cohort before append_device')
        lanes, layout = self._require_graph(graph)
        self._vector(game_rows, t.int64, (lanes,))
        self._vector(owners, t.int64, (lanes,))
        self._vector(steps, t.int64, (lanes,))
        self._vector(versions, t.int64, (lanes,))
        self._vector(actors, t.int64, (lanes,))
        self._vector(active, t.bool, (lanes,))
        counts = graph.counts.long()
        n, e, p = counts[:, 0], counts[:, 1], counts[:, 2]
        roots = graph.roots.long()
        in_range = (game_rows >= 0) & (game_rows < self.rows)
        game = game_rows.clamp(0, self.rows - 1)
        id_ok = in_range & self._begun[game]
        owner_ok = id_ok & (owners >= 0) & (owners == self._owners[game])
        actor_ok = (actors == 0) | (actors == 1)
        step_ok = id_ok & (steps >= 0) & (steps == self._expected_step[game])
        stale = (versions < 0) | (id_ok & (versions <= self._last_version[game]))
        range_ok = ((n >= 1) & (n <= layout.max_nodes) & (e >= 0) & (e <= layout.max_edges)
                    & (p >= 0) & (p <= layout.max_payload_bytes)
                    & (roots >= 0) & (roots < n))
        histogram = t.zeros(self.rows, dtype=t.int64, device=self.device)
        histogram.index_add_(0, game, (active & in_range).long())
        duplicate = in_range & (histogram[game] > 1)
        sticky = t.where(in_range, self._game_errors[game], 0)
        errors = t.zeros(lanes, dtype=t.int32, device=self.device)
        errors = errors | ((active & ~id_ok).to(t.int32) * INVALID_ID)
        errors = errors | ((active & ~owner_ok).to(t.int32) * OWNER_ERROR)
        errors = errors | ((active & ~actor_ok).to(t.int32) * ACTOR_ERROR)
        errors = errors | ((active & ~step_ok).to(t.int32) * STEP_ERROR)
        errors = errors | ((active & stale).to(t.int32) * STALE_ERROR)
        errors = errors | ((active & duplicate).to(t.int32) * DUPLICATE_ERROR)
        errors = errors | ((active & ~range_ok).to(t.int32) * RANGE_ERROR)
        errors = errors | (active.to(t.int32) * sticky)
        admitted = active & (errors == 0)
        need_r = admitted.long().sum()
        need_n = (admitted.long() * n.clamp(min=0)).sum()
        need_e = (admitted.long() * e.clamp(min=0)).sum()
        need_p = (admitted.long() * p.clamp(min=0)).sum()
        fits = ((self._record_count + need_r) <= self.max_records) & (
            (self._node_used + need_n) <= self.node_capacity) & (
            (self._edge_used + need_e) <= self.edge_capacity) & (
            (self._payload_used + need_p) <= self.payload_capacity)
        overflow = (~fits) & (need_r > 0)
        errors = errors | ((admitted & overflow).to(t.int32) * CAPACITY_ERROR)
        commit = admitted & ~overflow
        safe_n = t.where(commit, n.clamp(0, layout.max_nodes), 0)
        safe_e = t.where(commit, e.clamp(0, layout.max_edges), 0)
        safe_p = t.where(commit, p.clamp(0, layout.max_payload_bytes), 0)
        rec_rank = commit.long().cumsum(0) - commit.long()
        rec_dest = t.where(commit, self._record_count + rec_rank, self._record_sink)
        node_rank = safe_n.cumsum(0) - safe_n
        edge_rank = safe_e.cumsum(0) - safe_e
        pay_rank = safe_p.cumsum(0) - safe_p
        node_off = t.where(commit, self._node_used + node_rank, 0)
        edge_off = t.where(commit, self._edge_used + edge_rank, 0)
        pay_off = t.where(commit, self._payload_used + pay_rank, 0)
        rec_dest = rec_dest.clamp(0, self.max_records)
        self._rec_game.index_copy_(0, rec_dest, t.where(commit, game_rows, self._rec_game[self.max_records]))
        self._rec_owner.index_copy_(0, rec_dest, t.where(commit, owners, self._rec_owner[self.max_records]))
        self._rec_seed.index_copy_(
            0, rec_dest, t.where(commit, self._job_seeds[game], self._rec_seed[self.max_records]))
        self._rec_actor.index_copy_(0, rec_dest, t.where(commit, actors, self._rec_actor[self.max_records]))
        self._rec_step.index_copy_(0, rec_dest, t.where(commit, steps, self._rec_step[self.max_records]))
        self._rec_version.index_copy_(
            0, rec_dest, t.where(commit, versions, self._rec_version[self.max_records]))
        self._rec_root.index_copy_(
            0, rec_dest, t.where(commit, graph.roots, self._rec_root[self.max_records]))
        self._rec_node_off.index_copy_(
            0, rec_dest, t.where(commit, node_off, self._rec_node_off[self.max_records]))
        self._rec_node_count.index_copy_(
            0, rec_dest, t.where(commit, safe_n, self._rec_node_count[self.max_records]))
        self._rec_edge_off.index_copy_(
            0, rec_dest, t.where(commit, edge_off, self._rec_edge_off[self.max_records]))
        self._rec_edge_count.index_copy_(
            0, rec_dest, t.where(commit, safe_e, self._rec_edge_count[self.max_records]))
        self._rec_pay_off.index_copy_(
            0, rec_dest, t.where(commit, pay_off, self._rec_pay_off[self.max_records]))
        self._rec_pay_count.index_copy_(
            0, rec_dest, t.where(commit, safe_p, self._rec_pay_count[self.max_records]))
        self._scatter_used(graph, node_off, edge_off, pay_off, commit)
        self._record_count.add_(commit.long().sum())
        self._node_used.add_(safe_n.sum())
        self._edge_used.add_(safe_e.sum())
        self._payload_used.add_(safe_p.sum())
        self._expected_step.index_add_(0, game, commit.long())
        # Inactive padded lanes may repeat an admitted game's id. A scatter
        # of old values would race with, and overwrite, its newly stored version.
        updates = t.full_like(self._last_version, -1)
        updates.scatter_reduce_(0, game, t.where(commit, versions, -1),
                                reduce='amax', include_self=True)
        self._last_version.copy_(t.maximum(self._last_version, updates))
        self._or_into_games(game, errors, active & in_range)
        return t.where(active, errors, t.zeros_like(errors))

    def _scatter_used(self, graph, node_off, edge_off, pay_off, take):
        t = self.t
        lanes, layout = graph.rows, graph.layout
        take_u8 = take.view(t.uint8)
        if self.module is None:
            self._scatter_cpu(graph, node_off, edge_off, pay_off, take)
            return
        self.module.launch(
            'scatter_raw_action',
            [graph.nodes, graph.edges, graph.payload, graph.counts,
             self._nodes, self._edges, self._payload,
             node_off.contiguous(), edge_off.contiguous(), pay_off.contiguous(),
             take_u8, lanes, int(layout.max_nodes), int(layout.max_edges),
             int(layout.max_payload_bytes)],
            grid=lanes, block=128, capture_safe=True)

    def _scatter_cpu(self, graph, node_off, edge_off, pay_off, take):
        t = self.t
        layout = graph.layout
        n, e, p = graph.counts[:, 0].long(), graph.counts[:, 1].long(), graph.counts[:, 2].long()
        node_ids = t.arange(layout.max_nodes, device=self.device)
        edge_ids = t.arange(layout.max_edges, device=self.device)
        pay_ids = t.arange(layout.max_payload_bytes, device=self.device)
        take_n = take[:, None] & (node_ids[None, :] < n[:, None])
        take_e = take[:, None] & (edge_ids[None, :] < e[:, None])
        take_p = take[:, None] & (pay_ids[None, :] < p[:, None])
        node_dest = t.where(take_n, node_off[:, None] + node_ids[None, :], self._node_sink)
        edge_dest = t.where(take_e, edge_off[:, None] + edge_ids[None, :], self._edge_sink)
        pay_dest = t.where(take_p, pay_off[:, None] + pay_ids[None, :], self._payload_sink)
        self._nodes.index_copy_(
            0, node_dest.reshape(-1).clamp(0, self.node_capacity),
            t.where(take_n[..., None], graph.nodes, self._nodes[self._node_sink]).reshape(-1, 4))
        self._edges.index_copy_(
            0, edge_dest.reshape(-1).clamp(0, self.edge_capacity),
            t.where(take_e[..., None], graph.edges, self._edges[self._edge_sink]).reshape(-1, 2))
        self._payload.index_copy_(
            0, pay_dest.reshape(-1).clamp(0, self.payload_capacity),
            t.where(take_p, graph.payload, self._payload[self._payload_sink]).reshape(-1))

    def gather_into_device(self, record_indices, destination, active):
        t = self.t
        if not self._ready:
            raise RuntimeError('begin_device must allocate the cohort before gather_into_device')
        lanes, layout = self._require_graph(destination)
        self._vector(record_indices, t.int64, (lanes,))
        self._vector(active, t.bool, (lanes,))
        destination.nodes.zero_()
        destination.edges.zero_()
        destination.payload.zero_()
        destination.counts.zero_()
        destination.roots.zero_()
        in_range = (record_indices >= 0) & (record_indices < self._record_count)
        idx = record_indices.clamp(0, self.max_records)
        n = self._rec_node_count[idx]
        e = self._rec_edge_count[idx]
        p = self._rec_pay_count[idx]
        node_off = self._rec_node_off[idx]
        edge_off = self._rec_edge_off[idx]
        pay_off = self._rec_pay_off[idx]
        roots = self._rec_root[idx]
        range_ok = ((n >= 1) & (node_off >= 0) & ((node_off + n) <= self._node_used)
                    & (e >= 0) & (edge_off >= 0) & ((edge_off + e) <= self._edge_used)
                    & (p >= 0) & (pay_off >= 0) & ((pay_off + p) <= self._payload_used)
                    & (roots >= 0) & (roots.long() < n))
        overflow = ((n > layout.max_nodes) | (e > layout.max_edges)
                    | (p > layout.max_payload_bytes))
        errors = t.zeros(lanes, dtype=t.int32, device=self.device)
        errors = errors | ((active & ~in_range).to(t.int32) * INVALID_ID)
        errors = errors | ((active & in_range & ~range_ok).to(t.int32) * RANGE_ERROR)
        errors = errors | ((active & in_range & range_ok & overflow).to(t.int32) * CAPACITY_ERROR)
        good = active & (errors == 0)
        self._gather_used(destination, node_off, n, edge_off, e, pay_off, p, roots, good)
        return t.where(active, errors, t.zeros_like(errors))

    def _gather_used(self, destination, node_off, n, edge_off, e, pay_off, p, roots, take):
        t = self.t
        lanes, layout = destination.rows, destination.layout
        take_u8 = take.view(t.uint8)
        if self.module is None:
            self._gather_cpu(destination, node_off, n, edge_off, e, pay_off, p, roots, take)
            return
        self.module.launch(
            'gather_raw_action',
            [self._nodes, self._edges, self._payload,
             destination.nodes, destination.edges, destination.payload,
             destination.counts, destination.roots,
             node_off.contiguous(), n.contiguous(), edge_off.contiguous(), e.contiguous(),
             pay_off.contiguous(), p.contiguous(), roots.contiguous(), take_u8,
             lanes, int(layout.max_nodes), int(layout.max_edges),
             int(layout.max_payload_bytes)],
            grid=lanes, block=128, capture_safe=True)

    def _gather_cpu(self, destination, node_off, n, edge_off, e, pay_off, p, roots, take):
        t = self.t
        layout = destination.layout
        lanes = destination.rows
        node_ids = t.arange(layout.max_nodes, device=self.device)
        edge_ids = t.arange(layout.max_edges, device=self.device)
        pay_ids = t.arange(layout.max_payload_bytes, device=self.device)
        take_n = take[:, None] & (node_ids[None, :] < n[:, None])
        take_e = take[:, None] & (edge_ids[None, :] < e[:, None])
        take_p = take[:, None] & (pay_ids[None, :] < p[:, None])
        node_src = t.where(take_n, node_off[:, None] + node_ids[None, :], self._node_sink)
        edge_src = t.where(take_e, edge_off[:, None] + edge_ids[None, :], self._edge_sink)
        pay_src = t.where(take_p, pay_off[:, None] + pay_ids[None, :], self._payload_sink)
        copied_n = self._nodes.index_select(
            0, node_src.reshape(-1).clamp(0, self.node_capacity)).reshape(
            lanes, layout.max_nodes, 4)
        copied_e = self._edges.index_select(
            0, edge_src.reshape(-1).clamp(0, self.edge_capacity)).reshape(
            lanes, layout.max_edges, 2)
        copied_p = self._payload.index_select(
            0, pay_src.reshape(-1).clamp(0, self.payload_capacity)).reshape(
            lanes, layout.max_payload_bytes)
        destination.nodes.copy_(t.where(take_n[..., None], copied_n, 0))
        destination.edges.copy_(t.where(take_e[..., None], copied_e, 0))
        destination.payload.copy_(t.where(take_p, copied_p, 0))
        destination.counts.copy_(t.stack((n, e, p), dim=-1).to(t.int32) * take[:, None].to(t.int32))
        destination.roots.copy_(t.where(take, roots, 0))

