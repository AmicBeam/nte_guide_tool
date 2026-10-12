"""Two independently captured, bounded GPU queues for full ordered view guards."""
from dataclasses import fields
import math
from types import SimpleNamespace
from ..batched_duel.gpu_seven_candidates import RawCandidateTokens
from .device_tensor_row_queue import DeviceTensorRowQueue

VALIDATION_ERROR = 158


class _GuardSite:
    def __init__(self, owner):
        self.owner = owner
        self.queue = None
        self.captured = False
        try:
            self.queue = DeviceTensorRowQueue(rows=owner.rows, scratch_rows=owner.tile,
                seconds=owner.seconds, device=owner.device)
            self.errors = owner.t.zeros(owner.rows + 1, dtype=owner.t.int32, device=owner.device)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.queue is not None:
            self.queue.close()
            self.queue = None

    def _gather(self, view, ids):
        raw = RawCandidateTokens(**{f.name: getattr(view.candidates, f.name).index_select(0, ids)
            for f in fields(RawCandidateTokens) if f.name != 'request_ids'}, request_ids=())
        return SimpleNamespace(candidates=raw, errors=view.errors.index_select(0, ids),
            observations=SimpleNamespace(observations=view.observations.observations.index_select(0, ids)))

    def capture(self, left, right, active, template_table):
        o = self.owner
        t = o.t
        if self.captured or self.queue is None:
            raise RuntimeError('Guard site closed or already frozen')
        self.errors.zero_()
        def operation(ids, on, destinations):
            result = o.comparison(self._gather(left, ids), self._gather(right, ids), on, template_table)
            if (not isinstance(result, t.Tensor) or result.device != o.device or
                    result.dtype != t.int32 or result.shape != (o.tile,)):
                raise ValueError('Exact device int32 scratch guard result required')
            self.errors.index_copy_(0, destinations, t.where(on, result, 0))
        guard = self.queue.capture_device(active, operation)
        self.captured = True
        result = self.errors[:o.rows]
        failed = active & (guard.reshape(-1)[0] != 0) & (result == 0)
        result.copy_(t.where(failed, VALIDATION_ERROR, result))
        return result


class DeviceGuardRowWork:
    def __init__(self, *, rows, tile=8, seconds=120, device='cuda', compare):
        import torch as t
        if any(type(v) is not int or v < 1 for v in (rows, tile)):
            raise ValueError('Positive bounded guard row and tile counts required')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Positive independent guard queue budget required')
        if not callable(compare):
            raise TypeError('Static full-view comparison callback required')
        self.device = t.device(device)
        if self.device.type != 'cuda' or not t.cuda.is_available():
            raise ValueError('Actual CUDA required; no host guard queue fallback')
        if self.device.index is None:
            self.device = t.device('cuda', t.cuda.current_device())
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare guard queues outside capture')
        self.t = t
        self.rows, self.tile, self.seconds = rows, tile, seconds
        self.comparison = compare
        self.sites, self.index, self.closed = [], 0, False
        try:
            for _ in range(2):
                self.sites.append(_GuardSite(self))
        except BaseException:
            self.close()
            raise

    def _validate(self, view):
        t = self.t
        if not isinstance(view.candidates, RawCandidateTokens):
            raise TypeError('Full raw candidate token rows required')
        values = [view.errors, view.observations.observations]
        values.extend(getattr(view.candidates, f.name) for f in fields(RawCandidateTokens)
                      if f.name != 'request_ids')
        if any(not isinstance(v, t.Tensor) or v.device != self.device or v.ndim < 1 or
               v.shape[0] != self.rows for v in values):
            raise ValueError('Exact device cohort for every guard field required')
        if view.errors.dtype != t.int32 or view.errors.shape != (self.rows,):
            raise ValueError('Exact int32 per-row errors required')

    def compare(self, left, right, active, template_table):
        if self.closed or self.index >= len(self.sites):
            raise RuntimeError('Guard queues closed or capture quota exhausted')
        t = self.t
        if (not isinstance(active, t.Tensor) or active.device != self.device or
                active.dtype != t.bool or active.shape != (self.rows,) or not active.is_contiguous()):
            raise ValueError('Exact contiguous device boolean guard cohort required')
        self._validate(left)
        self._validate(right)
        if not t.cuda.is_current_stream_capturing():
            raise RuntimeError('Guard queue requires parent capture')
        site = self.sites[self.index]
        self.index += 1
        return site.capture(left, right, active, template_table)

    def close(self):
        if not self.closed:
            for site in self.sites:
                site.close()
            self.closed = True
