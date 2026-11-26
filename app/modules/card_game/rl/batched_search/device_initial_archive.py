"""GPU copy of original physical states for one bounded replay cohort."""
from ..batched_duel.cuda_vm import DeviceGraph
from ..batched_duel.device_rule_protocol import DeviceStateBatch
from ..batched_duel.device_rule_lanes import DeviceRuleLanes
from ..batched_duel.schema import GraphLayout


class DeviceInitialArchive:
    supports_full_duel_rules = False

    def __init__(self, source, archive, *, rows, lanes=16, layout, seconds=60):
        import torch
        if source is archive or source.device != archive.device:
            raise ValueError('Separate source and immutable archive arenas on one device required')
        if source.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU initial-state reconstruction')
        if not isinstance(layout, GraphLayout):
            raise TypeError('Bounded compact GraphLayout required')
        self.t, self.source, self.archive = torch, source, archive
        self.rows, self.device = rows, source.device
        self.dispatch = DeviceRuleLanes(rows, lanes, device=str(self.device), seconds=seconds)
        shape = GraphLayout(self.dispatch.lanes, layout.max_nodes, layout.max_edges, layout.max_payload_bytes)
        kw = dict(device=self.device)
        self.scratch = DeviceGraph(shape,
            torch.zeros((shape.game_capacity, shape.max_nodes, 4), dtype=torch.int32, **kw),
            torch.zeros((shape.game_capacity, shape.max_edges, 2), dtype=torch.int32, **kw),
            torch.zeros((shape.game_capacity, shape.max_payload_bytes), dtype=torch.uint8, **kw),
            torch.zeros((shape.game_capacity, 3), dtype=torch.int32, **kw),
            torch.zeros(shape.game_capacity, dtype=torch.int32, **kw))
        self.output = DeviceStateBatch(archive,
            torch.full((rows,), -1, dtype=torch.int64, **kw),
            torch.zeros(rows, dtype=torch.int64, **kw), torch.zeros(rows, dtype=torch.int64, **kw),
            torch.zeros(rows, dtype=torch.bool, **kw), torch.zeros(rows, dtype=torch.int32, **kw))
        self.captured = False

    def capture_device(self, roots):
        """Capture one cohort copy. Replay once; release only after replay verification."""
        if self.captured or roots.pool is not self.source:
            raise ValueError('Single capture with the exact physical source arena required')
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Initial archive copy must be embedded in CUDA capture')
        self.dispatch._validate_states(roots, self.rows)
        self.output.slots.fill_(-1);self.output.generations.zero_()
        self.output.active.zero_();self.output.owners.copy_(roots.owners)
        self.output.errors.copy_(roots.errors)
        def copy(local, ids, valid):
            loaded = self.source.gather_into_device(local, self.scratch)
            active = local.active & (local.errors == 0) & (loaded.errors == 0)
            stored = self.archive.put_device(self.scratch, local.owners, active.contiguous())
            stored.errors.bitwise_or_(local.errors | loaded.errors)
            self.dispatch.scatter_states_device(stored, self.output)
        self.dispatch.capture_dispatch(roots, copy)
        self.captured = True
        return self.output

    def close(self):
        self.dispatch.close()
