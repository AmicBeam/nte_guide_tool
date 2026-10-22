"""NO-VM GPU grouping of selected domain actions and exact-order scatter.

400-root Gumbel/H3 helper: compact selected root actions by a static registered
handler id, then scatter handler results back into the original root/action
order. Candidate-preview jobs keep multiple slots per root without writing
parents. This is not a rule engine, VM, native SoA graph, boxed heap, or
storage pool. supports_full_duel_rules stays False.
"""
from dataclasses import dataclass

UNKNOWN_KIND = 1
COUNT_OVERFLOW = 2
STALE_RESULT = 4
FOREIGN_RESULT = 8
DUPLICATE_RESULT = 16
INCOMPLETE_GROUP = 32

DEFAULT_MAX_ROOTS = 400
DEFAULT_MAX_CANDIDATES = 800
MAX_KIND = 256
MAX_HANDLER_ID = 2 ** 20 - 1
supports_full_duel_rules = False


@dataclass(frozen=True)
class DomainHandlerSpec:
    kind: int
    handler_id: int


@dataclass(frozen=True)
class DomainActionWorkQueue:
    present: object
    handler_id: object
    root_index: object
    action_index: object
    candidate_ordinal: object
    owners: object
    generations: object
    original_order: object
    group_id: object
    group_size: object
    compact_count: object
    selected: object
    errors: object
    preview: bool
    epoch: object = None
    arena: object = None


@dataclass(frozen=True)
class DomainActionJobResults:
    present: object
    values: object
    root_index: object
    action_index: object
    owners: object
    generations: object
    handler_id: object
    candidate_ordinal: object = None
    epoch: object = None


@dataclass(frozen=True)
class DomainActionScatterResult:
    values: object
    present: object
    errors: object


def _as_spec(spec):
    if isinstance(spec, DomainHandlerSpec):
        kind, handler_id = spec.kind, spec.handler_id
    elif isinstance(spec, tuple) and len(spec) == 2:
        kind, handler_id = spec
    else:
        raise TypeError('Static (kind, handler_id) domain handler registration required')
    if type(kind) is not int or not 0 <= kind < MAX_KIND:
        raise ValueError('Domain action kind is outside the static table')
    if type(handler_id) is not int or not 0 <= handler_id <= MAX_HANDLER_ID:
        raise ValueError('Nonnegative bounded handler id required')
    return kind, handler_id


def cpu_oracle(handlers, **kwargs):
    """Explicit CPU reference; never an implicit CUDA fallback."""
    kwargs.pop('device', None)
    kwargs.pop('allow_cpu_oracle', None)
    return DomainSoAActionBatch(
        handlers, device='cpu', allow_cpu_oracle=True, **kwargs)


class DomainSoAActionBatch:
    supports_full_duel_rules = False

    def __init__(self, handlers, *, max_roots=DEFAULT_MAX_ROOTS,
                 max_candidates=DEFAULT_MAX_CANDIDATES, device='cuda',
                 allow_cpu_oracle=False):
        import torch
        if type(max_roots) is not int or not 1 <= max_roots <= DEFAULT_MAX_ROOTS:
            raise ValueError('Bounded positive root capacity required')
        if type(max_candidates) is not int or not 1 <= max_candidates <= 800:
            raise ValueError('Bounded positive candidate capacity required')
        if handlers is None:
            raise TypeError('Static domain handler registration required')
        try:
            registered = tuple(_as_spec(spec) for spec in handlers)
        except TypeError as exc:
            if 'handler registration' in str(exc):
                raise
            raise TypeError('Static domain handler sequence required') from exc
        kinds = [kind for kind, _handler in registered]
        if len(kinds) != len(set(kinds)):
            raise ValueError('Duplicate domain action kind registration')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU action-batch fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_oracle is True):
            raise ValueError('CPU is an explicit action-batch oracle only')
        self.max_roots = max_roots
        self.max_candidates = max_candidates
        self.work_width = max_roots * max_candidates
        self.handler_table = torch.full(
            (MAX_KIND,), -1, dtype=torch.int32, device=self.device)
        if registered:
            kind_t = torch.tensor([kind for kind, _h in registered],
                                  dtype=torch.int64, device=self.device)
            hid_t = torch.tensor([hid for _kind, hid in registered],
                                 dtype=torch.int32, device=self.device)
            self.handler_table[kind_t] = hid_t
        self._q_present = torch.zeros(self.work_width, dtype=torch.bool, device=self.device)
        self._q_handler = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_root = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_action = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_ordinal = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_owners = torch.zeros(self.work_width, dtype=torch.int64, device=self.device)
        self._q_generations = torch.zeros(self.work_width, dtype=torch.int64, device=self.device)
        self._q_order = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_group_id = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._q_group_size = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._compact_count = torch.zeros((), dtype=torch.int32, device=self.device)
        self._epoch = torch.zeros((), dtype=torch.int64, device=self.device)
        self._grid_i64 = torch.zeros(self.work_width, dtype=torch.int64, device=self.device)
        self._grid_i32 = torch.zeros(self.work_width, dtype=torch.int32, device=self.device)
        self._workspace = (
            self.handler_table, self._q_present, self._q_handler, self._q_root,
            self._q_action, self._q_ordinal, self._q_owners, self._q_generations,
            self._q_order, self._q_group_id, self._q_group_size, self._compact_count,
            self._grid_i64, self._grid_i32, self._epoch,
        )

    @property
    def capacity_bytes(self):
        return sum(value.numel() * value.element_size() for value in self._workspace)

    @property
    def allocated_bytes(self):
        return self.capacity_bytes

    def _matrix(self, value, dtype, rows, name, *, candidates=True):
        shape = (rows, self.max_candidates) if candidates else (rows,)
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape):
            raise ValueError('Resident ' + name + ' required')
        return value

    def _work_vec(self, value, dtype, name):
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != dtype or value.ndim != 1
                or value.shape[0] != self.work_width):
            raise ValueError('Resident work-queue ' + name + ' required')
        return value

    def _clear_queue(self):
        self._q_present.zero_()
        self._q_handler.zero_()
        self._q_root.zero_()
        self._q_action.zero_()
        self._q_ordinal.zero_()
        self._q_owners.zero_()
        self._q_generations.zero_()
        self._q_order.zero_()
        self._q_group_id.zero_()
        self._q_group_size.zero_()
        self._compact_count.zero_()

    def group_selected_actions(
            self, tokens, counts, mask, selected, active, owners, generations,
            *, candidate_ordinals=None, preview=False):
        """Compact selected root actions by handler id, preserving order.

        Inputs are not mutated. Preview may mark several slots on one root.
        Unknown kinds and count/mask overflow stay on their roots and are
        omitted from the compact queue.
        """
        t = self.t
        if (not isinstance(tokens, t.Tensor) or tokens.ndim != 2
                or tokens.dtype != t.int32 or tokens.device != self.device):
            raise ValueError('Resident int32 action tokens [B,M] required')
        rows, width = int(tokens.shape[0]), int(tokens.shape[1])
        if rows < 1 or rows > self.max_roots or width != self.max_candidates:
            raise ValueError('Action-token layout must match the static 400-root capacity')
        counts = self._matrix(counts, t.int32, rows, 'counts', candidates=False)
        mask = self._matrix(mask, t.bool, rows, 'mask')
        selected = self._matrix(selected, t.bool, rows, 'selected')
        active = self._matrix(active, t.bool, rows, 'active', candidates=False)
        owners = self._matrix(owners, t.int64, rows, 'owners', candidates=False)
        generations = self._matrix(generations, t.int64, rows, 'generations',
                                   candidates=False)
        if candidate_ordinals is None:
            ordinals = t.arange(width, dtype=t.int32, device=self.device)
            ordinals = ordinals.unsqueeze(0).expand(rows, width)
        else:
            ordinals = self._matrix(candidate_ordinals, t.int32, rows, 'candidate_ordinals')
        slot = t.arange(width, device=self.device, dtype=t.int32)
        expected = slot.unsqueeze(0) < counts.clamp(min=0, max=width).unsqueeze(1)
        overflow = active & ((counts < 0) | (counts > width) | (mask != expected).any(1))
        overflow = overflow | (active & (selected & ~mask).any(1))
        overflow = overflow | (active & (selected & mask & ((ordinals < 0) | (ordinals >= width))).any(1))
        if not preview:
            overflow |= active & (selected.sum(1)>1)
        errors = overflow.int() * COUNT_OVERFLOW
        errors |= (active & ((owners<0)|(generations<1))).int()*FOREIGN_RESULT
        in_range = (tokens >= 0) & (tokens < MAX_KIND)
        safe_kind = tokens.clamp(0, MAX_KIND - 1).long()
        handler = self.handler_table[safe_kind]
        known = in_range & (handler >= 0)
        unknown = active & (errors == 0) & (selected & mask & ~known).any(1)
        errors = errors | unknown.int() * UNKNOWN_KIND
        admit = (active.unsqueeze(1) & selected & mask & known
                 & (errors == 0).unsqueeze(1))
        root = t.arange(rows, device=self.device, dtype=t.int32).unsqueeze(1).expand(rows, width)
        action = slot.unsqueeze(0).expand(rows, width)
        # cap16 is a search candidate cap, not the full legal-action capacity.
        # Use a mixed radix key so columns beyond 63 cannot collide with roots.
        sort_key = ((handler.to(t.int64) * self.max_roots + root.to(t.int64))
                    * width + ordinals.to(t.int64)) * width + action.to(t.int64)
        sentinel = t.full((), (1 << 62) - 1, dtype=t.int64, device=self.device)
        sort_key = t.where(admit, sort_key, sentinel)
        if preview:
            flat = rows * width
            perm = sort_key.reshape(-1).argsort(stable=True)
        else:
            # Physical commits select at most one action per root. Sort only
            # those B tasks, retaining the full B*M legal-action representation.
            chosen=selected.to(t.int8).argmax(1)
            indices=t.arange(rows,device=self.device)*width+chosen
            keys=sort_key.reshape(-1).index_select(0,indices)
            perm=indices.index_select(0,keys.argsort(stable=True))
            flat=rows
        n = admit.reshape(-1).sum()
        dest = t.arange(flat, device=self.device)
        compact = dest < n
        src = perm
        handler_flat = handler.reshape(-1)
        root_flat = root.reshape(-1)
        action_flat = action.reshape(-1)
        ordinal_flat = ordinals.reshape(-1)
        owner_flat = owners.unsqueeze(1).expand(rows, width).reshape(-1)
        gen_flat = generations.unsqueeze(1).expand(rows, width).reshape(-1)
        self._clear_queue()
        self._q_present[:flat] = compact
        self._q_handler[:flat] = t.where(compact, handler_flat[src], 0)
        self._q_root[:flat] = t.where(compact, root_flat[src], 0)
        self._q_action[:flat] = t.where(compact, action_flat[src], 0)
        self._q_ordinal[:flat] = t.where(compact, ordinal_flat[src], 0)
        self._q_owners[:flat] = t.where(compact, owner_flat[src], 0)
        self._q_generations[:flat] = t.where(compact, gen_flat[src], 0)
        self._q_order[:flat] = t.where(compact, src.to(t.int32), 0)
        present = self._q_present[:flat]
        hid = self._q_handler[:flat]
        idx = t.arange(flat, device=self.device)
        prev_present = t.zeros_like(present)
        prev_present[1:] = present[:-1]
        prev_hid = t.zeros_like(hid)
        prev_hid[1:] = hid[:-1]
        start = present & ((idx == 0) | (~prev_present) | (hid != prev_hid))
        group_id = start.int().cumsum(0) - 1
        self._q_group_id[:flat].copy_(t.where(present, group_id, 0))
        counts_g = t.zeros(flat, dtype=t.int32, device=self.device)
        safe_gid = self._q_group_id[:flat].long()
        counts_g.index_add_(0, safe_gid, present.int())
        self._q_group_size[:flat].copy_(t.where(present, counts_g[safe_gid], 0))
        self._compact_count.copy_(n.to(t.int32))
        self._epoch.add_(1)
        return DomainActionWorkQueue(
            present=self._q_present.clone(),
            handler_id=self._q_handler.clone(),
            root_index=self._q_root.clone(),
            action_index=self._q_action.clone(),
            candidate_ordinal=self._q_ordinal.clone(),
            owners=self._q_owners.clone(),
            generations=self._q_generations.clone(),
            original_order=self._q_order.clone(),
            group_id=self._q_group_id.clone(),
            group_size=self._q_group_size.clone(),
            compact_count=self._compact_count.clone(),
            selected=admit.clone(),
            errors=errors,
            preview=bool(preview),
            epoch=self._epoch.clone(),arena=self,
        )

    def scatter_results(self, queue, results):
        """Scatter handler outputs into original root/action slots.

        Stale, foreign, duplicate, or incomplete groups reject every queued
        root and write no values. Inputs are not mutated.
        """
        t = self.t
        if not isinstance(queue, DomainActionWorkQueue):
            raise TypeError('DomainActionWorkQueue required')
        if not isinstance(results, DomainActionJobResults):
            raise TypeError('DomainActionJobResults required')
        if queue.arena is not self:
            raise ValueError('Work queue belongs to another action-batch arena')
        rows = int(queue.errors.shape[0])
        width = self.max_candidates
        if queue.errors.device != self.device or queue.present.shape[0] != self.work_width:
            raise ValueError('Work queue belongs to another action-batch arena')
        present_q = self._work_vec(queue.present, t.bool, 'present')
        handler_q = self._work_vec(queue.handler_id, t.int32, 'handler_id')
        root_q = self._work_vec(queue.root_index, t.int32, 'root_index')
        action_q = self._work_vec(queue.action_index, t.int32, 'action_index')
        owners_q = self._work_vec(queue.owners, t.int64, 'owners')
        gen_q = self._work_vec(queue.generations, t.int64, 'generations')
        present_r = self._work_vec(results.present, t.bool, 'result present')
        values_r = self._work_vec(results.values, t.int32, 'result values')
        root_r = self._work_vec(results.root_index, t.int32, 'result roots')
        action_r = self._work_vec(results.action_index, t.int32, 'result actions')
        owners_r = self._work_vec(results.owners, t.int64, 'result owners')
        gen_r = self._work_vec(results.generations, t.int64, 'result generations')
        handler_r = self._work_vec(results.handler_id, t.int32, 'result handlers')
        ordinal_r = (self._work_vec(results.candidate_ordinal, t.int32, 'result ordinals')
                     if results.candidate_ordinal is not None else None)
        in_range = (present_r & (root_r >= 0) & (root_r < rows)
                    & (action_r >= 0) & (action_r < width))
        linear_q = (root_q.to(t.int64) * width + action_q.to(t.int64)).clamp(
            0, self.work_width - 1)
        linear_r = (root_r.to(t.int64).clamp(0, self.max_roots - 1) * width
                    + action_r.to(t.int64).clamp(0, width - 1))
        q_hist = self._grid_i64.zero_()
        q_hist.index_add_(0, linear_q, present_q.long())
        r_hist = t.zeros(self.work_width, dtype=t.int64, device=self.device)
        r_hist.index_add_(0, linear_r, in_range.long())
        owner_grid = t.zeros(self.work_width, dtype=t.int64, device=self.device)
        gen_grid = t.zeros(self.work_width, dtype=t.int64, device=self.device)
        handler_grid = self._grid_i32.zero_()
        owner_grid.index_add_(0, linear_q, t.where(present_q, owners_q, 0))
        gen_grid.index_add_(0, linear_q, t.where(present_q, gen_q, 0))
        handler_grid.index_add_(0, linear_q, t.where(present_q, handler_q, 0))
        ordinal_grid = t.zeros(self.work_width, dtype=t.int32, device=self.device)
        ordinal_grid.index_add_(0, linear_q, t.where(present_q, queue.candidate_ordinal, 0))
        matched = in_range & (q_hist[linear_r] > 0)
        stale = matched & (gen_r != gen_grid[linear_r])
        if results.epoch is None:
            stale |= present_r
        else:
            if (not isinstance(results.epoch,t.Tensor) or results.epoch.device!=self.device
                    or results.epoch.dtype!=t.int64 or results.epoch.shape!=()):
                raise ValueError('Resident scalar result issuance epoch required')
            stale |= present_r & (results.epoch!=queue.epoch)
        foreign = present_r & (
            (~in_range) | (q_hist[linear_r] == 0)
            | (matched & ((owners_r != owner_grid[linear_r])
                          | (handler_r != handler_grid[linear_r]))))
        if ordinal_r is not None:
            foreign |= matched & (ordinal_r != ordinal_grid[linear_r])
        elif queue.preview:
            foreign |= present_r
        duplicate = (r_hist > 1).any() | (q_hist > 1).any()
        incomplete = ((q_hist > 0) & (r_hist == 0)).any() | (
            present_q.int().sum() != present_r.int().sum())
        integrity = duplicate | stale.any() | foreign.any() | incomplete
        queued = t.zeros(rows, dtype=t.int64, device=self.device)
        queued.index_add_(0, root_q.long().clamp(0, rows - 1), present_q.long())
        claimed = t.zeros(rows, dtype=t.int64, device=self.device)
        claimed.index_add_(0, root_r.long().clamp(0, rows - 1), present_r.long())
        fail = ((queued > 0) | (claimed > 0)) & integrity
        errors = queue.errors.clone()
        errors = errors | (fail.int() * (
            stale.any().int() * STALE_RESULT
            | foreign.any().int() * FOREIGN_RESULT
            | duplicate.int() * DUPLICATE_RESULT
            | incomplete.int() * INCOMPLETE_GROUP))
        good = in_range & (~integrity)
        value_grid = t.zeros(self.work_width, dtype=t.int32, device=self.device)
        present_grid = t.zeros(self.work_width, dtype=t.int64, device=self.device)
        value_grid.index_add_(0, linear_r, t.where(good, values_r, 0))
        present_grid.index_add_(0, linear_r, good.long())
        laid = present_grid.view(self.max_roots, width)[:rows] > 0
        values = value_grid.view(self.max_roots, width)[:rows]
        values = t.where(laid, values, t.zeros_like(values))
        return DomainActionScatterResult(values=values, present=laid, errors=errors)

