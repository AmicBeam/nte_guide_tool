"""Bounded CUDA-native rule queries for post-batch parity, not GPU throughput.

This diagnostic boundary explicitly ingresses/exports whole arrays. Closed
search uses CudaNativeArray through DeviceRuleGraphCalls instead; it must never
call this host query wrapper from a captured game loop.
"""
import numpy as np
from .native_array_cuda import CudaNativeArray
from .cuda_vm import VmLimits
from .codec import pack_states
from .schema import GraphLayout, TAG_TUPLE


class NativeArrayCUDAQuery:
    gpu_throughput_qualified = False
    interpreter_fallback = False

    def __init__(self, image, *, limits=VmLimits(), device='cuda', max_nodes=262144,
                 max_edges=524288, max_payload_bytes=33554432, seconds=120, **options):
        if type(seconds) not in (int, float) or not 0 < seconds <= 3600:
            raise ValueError('Bounded native CUDA diagnostic call budget required')
        self.seconds = seconds
        self.executor = CudaNativeArray(image, limits=limits, device=device,
                                        guard_standalone_entities=True, **options)
        self.bounds = (max_nodes, max_edges, max_payload_bytes)
        self.image = image
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.closed:
            self.executor.close()
            self.closed = True

    def call(self, entry, arguments, *, steps=None):
        if self.closed:
            raise RuntimeError('Native CUDA query closed')
        if steps is not None and steps != self.executor.limits.steps:
            raise ValueError('CUDA-native rule step bound is fixed before capture')
        if not isinstance(arguments, (list, tuple)) or not arguments:
            raise ValueError('Nonempty argument batch required')
        rows = len(arguments)
        arity = len(arguments[0])
        if arity < 1 or any(len(row) != arity for row in arguments):
            raise ValueError('One fixed argument arity per native batch required')
        layout = GraphLayout(rows, *self.bounds)
        packed = pack_states([tuple(row) for row in arguments], layout)
        refs = np.empty((rows, arity), dtype=np.int32)
        for row in range(rows):
            tag, offset, count, _ = packed.nodes[row, packed.roots[row]]
            if tag != TAG_TUPLE or count != arity:
                raise ValueError('Native query argument tuple changed')
            refs[row] = packed.edges[row, offset:offset+count, 0]
        t = self.executor.torch
        arena = self.executor.ingest(packed)
        arguments_device = t.tensor(refs, device=self.executor.device)
        deadlines = t.full((rows,), self.executor.clock_ns()+int(self.seconds*1e9),
                           dtype=t.int64, device=self.executor.device)
        call = self.executor.launch(arena, entry, argument_refs=arguments_device, gpu_deadlines=deadlines)
        return call.decode()
