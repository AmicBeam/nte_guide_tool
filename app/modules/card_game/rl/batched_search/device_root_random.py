"""Exact per-decision PCG64 initialization without host search stepping.

The host expands a declared finite set of (game, step, actor) seeds before
capture. The device selects those immutable states using its actual step and
actor, then draws through DevicePCG64. This is an RNG primitive, not a game
executor or a capacity approval. Exceeding the declared decision bound fails.
"""
from hashlib import sha256
import threading


class DeviceRootRandomPlan:
    supports_full_duel_rules = False

    def __init__(self, random, job_seeds, *, max_decisions, max_bytes,
                 allow_cpu_reference=False):
        import numpy as np
        import torch
        if any(type(v) is not int or v < 1 for v in (max_decisions, max_bytes)):
            raise ValueError('Positive decision and allocation bounds required')
        if not isinstance(job_seeds, (list, tuple)) or not job_seeds:
            raise ValueError('Explicit frozen job seed sequence required')
        if any(type(v) is not int or not -(1 << 63) <= v < (1 << 63) for v in job_seeds):
            raise ValueError('Signed int64 game seeds required')
        self.device = random.device
        self.rows = len(job_seeds)
        if random.loaded_rows != self.rows or random.max_rows != self.rows:
            raise ValueError('Random bank must bind exactly the declared game rows')
        if self.device.type != 'cuda' and not (self.device.type == 'cpu'
                                              and allow_cpu_reference is True):
            raise RuntimeError('CUDA required; CPU is an explicit random oracle only')
        required = self.rows * max_decisions * 2 * 6 * 8 + self.rows * 8
        if required > max_bytes:
            raise MemoryError('Declared per-decision random table exceeds byte budget')
        self.t, self.random = torch, random
        self.max_decisions = max_decisions
        self.job_seeds = tuple(job_seeds)
        self._thread = threading.get_ident()
        self._setup_stream = (torch.cuda.current_stream(self.device).cuda_stream
                              if self.device.type == 'cuda' else None)
        self._capture_streams = frozenset()
        table = np.empty((self.rows, max_decisions, 2, 6), dtype=np.uint64)
        mask = (1 << 64) - 1
        for row, seed in enumerate(job_seeds):
            for step in range(max_decisions):
                for actor, name in enumerate(('a', 'b')):
                    root_seed = (1 << 256) | int.from_bytes(
                        sha256(f'recovery:{seed}:{step}:{name}'.encode()).digest(), 'big')
                    state = np.random.PCG64(root_seed).state
                    value, inc = state['state']['state'], state['state']['inc']
                    table[row, step, actor] = (value >> 64, value & mask,
                        inc >> 64, inc & mask, state['has_uint32'], state['uinteger'])
        # Signed storage preserves all uint64 bits and supports device gather.
        self._table = torch.as_tensor(table.view(np.int64), device=self.device)
        self._rows = torch.arange(self.rows, dtype=torch.int64, device=self.device)

    @property
    def allocated_bytes(self):
        return sum(v.numel() * v.element_size() for v in (self._table, self._rows))

    def bind_capture_streams(self, parent, body):
        if threading.get_ident() != self._thread or self.device.type != 'cuda':
            raise RuntimeError('CUDA random plan requires its owning driver')
        if self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Bind capture streams before capture')
        if any(not isinstance(s, self.t.cuda.Stream) or s.device != self.device
               for s in (parent, body)):
            raise ValueError('Exact same-device capture streams required')
        self._capture_streams = frozenset((parent.cuda_stream, body.cuda_stream))

    def reset_device(self, steps, actors, active):
        t = self.t
        if threading.get_ident() != self._thread:
            raise RuntimeError('Random plan requires its owning driver thread')
        if self.device.type == 'cuda':
            stream = t.cuda.current_stream(self.device).cuda_stream
            allowed = (self._capture_streams if t.cuda.is_current_stream_capturing()
                       else frozenset((self._setup_stream,)))
            if stream not in allowed:
                raise RuntimeError('Unbound random-plan stream')
        for value, dtype in ((steps, t.int64), (actors, t.int64), (active, t.bool)):
            if (not isinstance(value, t.Tensor) or value.device != self.device
                    or value.dtype != dtype or value.shape != (self.rows,)
                    or not value.is_contiguous()):
                raise ValueError('Exact device step, actor and active vectors required')
        invalid = active & ((steps < 0) | (steps >= self.max_decisions)
                            | (actors < 0) | (actors > 1))
        selected = self._table[self._rows, steps.clamp(0, self.max_decisions - 1),
                               actors.clamp(0, 1)]
        live = active & ~invalid
        state = self.random.state
        for column, name in enumerate(('state_hi', 'state_lo', 'inc_hi', 'inc_lo',
                                      'has_uint32', 'uinteger')):
            target = getattr(state, name)
            signed = target.view(t.int64 if column < 4 else t.int32)
            signed.copy_(t.where(live, selected[:, column].to(signed.dtype), signed))
        return invalid.to(t.int32)
