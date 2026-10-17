"""Explicit host reference for native rules over persistent typed SoA columns.

CPU execution is a software conformance tool. It provides no CUDA timing or
qualification and is never selected as a fallback by a GPU runtime.
"""
from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path
import struct

import numpy as np

from .cuda_vm import VmLimits
from .native_array_cpu import (
    _EXECUTE_ARGTYPES, _cptr, compile_host_native_library,
    pack_call_arguments, NativeArrayCall, NativeArrayExecutionError,
)
from .native_soa_source import emit_native_soa_source, emit_host_soa_source, source_identity
from .record_soa import RecordSoABatch, RecordSoAError, _DecodeRow, MISSING
from .schema import GraphLayout


class NativeSoACall:
    def __init__(self, backend, state, heap, outputs, statuses, roots, views, context):
        self.backend, self.state, self.heap = backend, state, heap
        self.outputs, self.statuses, self.roots, self.views = outputs, statuses, roots, views
        self.context=context

    def check(self):
        try:
            NativeArrayCall(self.backend.image, self.statuses, None, self.outputs,
                            np.zeros(len(self.outputs), dtype=np.int32)).check()
        except NativeArrayExecutionError as exc:
            bad=int(np.flatnonzero(self.statuses[:,0])[0])
            full=[pool.name for pool in self.state.schema.pools
                  if self.state.buffers['i32'][pool.used,bad]>=pool.capacity]
            failed_pool=-int(self.roots[bad])-1
            if self.roots[bad]<0 and failed_pool<len(self.state.schema.pools):
                full.append(self.state.schema.pools[failed_pool].name+' (adoption rolled back)')
            raise NativeArrayExecutionError(str(exc)+'; transient_counts='+repr(self.heap['counts'][bad].tolist())+
                '; full_typed_pools='+repr(full)) from exc
        return self.statuses

    def _transient_ref(self, row, path):
        ref = int(self.outputs[row])
        for position in path:
            if type(position) is not int:
                raise RecordSoAError('Transient packet selectors must be exact indices')
            if ref < 0:
                nodes, edges = (self.backend._constants[k] for k in ('nodes', 'edges'))
                index = -ref-1
            else:
                nodes, edges = (self.heap[k][row] for k in ('nodes', 'edges'))
                index = ref
            if not 0 <= index < len(nodes):
                raise RecordSoAError('Packet reference outside transient storage')
            tag, offset, count, _ = map(int, nodes[index])
            if tag not in (5, 6) or not 0 <= position < count or not 0 <= offset+position < len(edges):
                raise RecordSoAError('Packet selector outside transient sequence')
            ref = int(edges[offset+position, 0])
        return ref

    def decode(self, path=()):
        """Explicit debug egress including transient projections and typed views."""
        self.check()
        result = []
        constants = self.backend._constants
        for row in range(len(self.outputs)):
            output = self._transient_ref(row, path)
            typed = _DecodeRow(self.state, row)
            views = {int(ref): index+1 for index, ref in enumerate(self.views[:, row])
                     if int(ref) != MISSING}
            memo, active = {}, set()

            def decode(ref):
                ref = int(ref)
                if ref == MISSING:
                    raise RecordSoAError('Missing native result')
                if ref in views:
                    return typed.reference(views[ref])
                if ref in memo:
                    return memo[ref]
                if ref in active:
                    raise RecordSoAError('Circular transient result')
                if ref < 0:
                    index = -ref-1
                    nodes, edges, payload = (constants[k] for k in ('nodes', 'edges', 'payload'))
                    if index >= len(nodes):
                        raise RecordSoAError('Native constant reference out of range')
                else:
                    index = ref
                    nodes, edges, payload = (self.heap[k][row] for k in ('nodes', 'edges', 'payload'))
                    if index >= self.heap['counts'][row, 0]:
                        raise RecordSoAError('Native result reference out of range')
                tag, offset, count, extra = map(int, nodes[index])
                if tag == 0:
                    value = None
                elif tag == 1:
                    value = bool(extra)
                elif tag in (2, 3, 4):
                    if offset < 0 or count < 0 or offset+count > len(payload):
                        raise RecordSoAError('Native scalar extent invalid')
                    raw = payload[offset:offset+count].tobytes()
                    value = (int.from_bytes(raw, 'little', signed=True) if tag == 2 else
                             struct.unpack('<d', raw)[0] if tag == 3 else raw.decode('utf-8'))
                elif tag in (5, 6, 7, 8, 9, 10):
                    if offset < 0 or count < 0 or offset+count > len(edges):
                        raise RecordSoAError('Native transient extent invalid')
                    active.add(ref)
                    if tag in (5, 6):
                        value = [decode(edges[offset+i, 0]) for i in range(count)]
                        if tag == 6:
                            value = tuple(value)
                    else:
                        from ...engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity
                        cls = {7: dict, 8: PlayerEntity, 9: CharacterEntity, 10: CardEntity}[tag]
                        value = dict.__new__(cls)
                        for i in range(count):
                            value[decode(edges[offset+i, 0])] = decode(edges[offset+i, 1])
                    active.remove(ref)
                else:
                    raise RecordSoAError('Native result is not a data value: '+str(tag))
                memo[ref] = value
                return value

            result.append(decode(output))
        return result

    def state_result(self):
        """Select returned typed roots without decoding or repacking game state."""
        self.check()
        pool=self.state.schema.pools[self.state.schema.root]
        if np.any(self.roots<=pool.base) or np.any(self.roots>pool.base+pool.capacity):
            raise RecordSoAError('Entry did not return typed state roots')
        self.state.roots[:] = self.roots
        return self.state

    def select_state(self, path=()):
        """Select a typed result/packet member without materializing a Game."""
        self.check()
        roots = np.empty(len(self.outputs), dtype=np.int32)
        schema = self.state.schema
        for row in range(len(self.outputs)):
            ref = self._transient_ref(row, path)
            indices = np.flatnonzero(self.views[:, row] == ref)
            if len(indices) != 1:
                raise RecordSoAError('Result member is not a unique typed state view')
            root = int(indices[0])+1
            pool, _ = schema.locate(root)
            if pool.index != schema.root:
                raise RecordSoAError('Result member is not a declared Game record')
            roots[row] = root
        return RecordSoABatch(schema, len(roots), buffers=self.state.buffers, roots=roots)

    def export_projection(self, path=(), *, max_nodes=16384,max_edges=32768,max_payload_bytes=131072):
        """Exercise the same device exporter explicitly on the host oracle."""
        self.check()
        from .native_soa_cuda import _EXPORT_SIGNATURE
        from .schema import PackedStateBatch
        rows=len(self.outputs)
        function=self.backend._library.native_soa_export_projection
        function.argtypes=tuple(ctypes.c_void_p if name.endswith('*') else ctypes.c_int for name in _EXPORT_SIGNATURE)
        function.restype=None
        layout=GraphLayout(rows,max_nodes,max_edges,max_payload_bytes)
        nodes=np.zeros((rows,max_nodes,4),dtype=np.int32)
        edges=np.zeros((rows,max_edges,2),dtype=np.int32)
        payload=np.zeros((rows,max_payload_bytes),dtype=np.uint8)
        counts=np.zeros((rows,3),dtype=np.int32);roots=np.zeros(rows,dtype=np.int32)
        capacity=self.backend.max_nodes+len(self.backend.image.nodes)
        mapping=np.empty((rows,capacity),dtype=np.int32);queue=np.empty_like(mapping)
        errors=np.zeros(rows,dtype=np.int32);active=np.ones(rows,dtype=np.uint8)
        selected=np.array([self._transient_ref(row,path) for row in range(rows)],dtype=np.int32)
        c=self.backend._constants
        function(*[_cptr(v) for v in (self.heap['nodes'],self.heap['edges'],self.heap['payload'],self.heap['counts'],selected,
            c['nodes'],c['edges'],c['payload'])],len(c['nodes']),len(c['edges']),len(c['payload']),
            *[_cptr(self.state.buffers[key]) for key in ('i64','f64','i32','u8')],*[_cptr(v) for v in self.context],
            *[_cptr(v) for v in (nodes,edges,payload,counts,roots,mapping,queue,errors,active)],
            rows,self.backend.max_nodes,self.backend.max_edges,self.backend.max_payload_bytes,
            max_nodes,max_edges,max_payload_bytes)
        if np.any(errors):
            raise RecordSoAError('Projection export rejected result: '+repr(errors.tolist()))
        return PackedStateBatch(layout,nodes,edges,payload,counts,roots)


class NativeSoACPU:
    execution_mode = 'native_domain_soa_aot_cpu'
    interpreter_fallback = False
    runtime_instruction_fetch = False
    gpu_qualified = False
    supports_full_duel_rules = False

    def __init__(self, image, schema, *, limits=VmLimits(), directory=None,
                 max_nodes=1048576, max_edges=2097152, max_payload_bytes=33554432,stack_runs=False):
        self.image, self.schema, self.limits = image, schema, limits
        for value in (max_nodes, max_edges, max_payload_bytes):
            if type(value) is not int or not 1 <= value <= 2**30:
                raise ValueError('Positive bounded transient capacity required')
        self.max_nodes, self.max_edges, self.max_payload_bytes = max_nodes, max_edges, max_payload_bytes
        self._temp = tempfile.TemporaryDirectory(prefix='native_soa_cpu_') if directory is None else None
        self.directory = Path(self._temp.name if self._temp else directory).resolve()
        self.source, self.native_plan = emit_native_soa_source(image, schema, limits,stack_runs=stack_runs)
        self.source_sha256 = source_identity(self.source)
        self._library = compile_host_native_library(emit_host_soa_source(self.source), self.directory)
        self._execute = self._library.native_soa_execute
        self._execute.argtypes = _EXECUTE_ARGTYPES + (ctypes.c_void_p,)*12
        self._execute.restype = None
        self._constants = {key: np.ascontiguousarray(getattr(image, key), dtype=np.uint8 if key == 'payload' else np.int32)
                           for key in ('nodes', 'edges', 'payload', 'functions', 'classes', 'modules')}
        self._constants['instructions'] = np.full((1, 4), MISSING, dtype=np.int32)

    def call(self, entry, state, arguments, *, state_positions=(0,), steps=None):
        """Run a batch; state slots in arguments must be None placeholders.

        ``arguments`` contains only scalars/actions/metadata. This API cannot
        accept a Python game dict as a replacement for a persistent state row.
        """
        if not isinstance(state, RecordSoABatch) or state.schema.identity != self.schema.identity:
            raise RecordSoAError('Matching persistent domain columns required')
        rows = state.rows
        if rows < 1 or type(arguments) is not list or len(arguments) != rows:
            raise RecordSoAError('Nonempty matching argument batch required')
        if entry not in self.image.entries:
            raise RecordSoAError('Unknown native SoA entry')
        positions = tuple(state_positions)
        for row in arguments:
            for position in positions:
                if type(position) is not int or not 0 <= position < len(row) or row[position] is not None:
                    raise RecordSoAError('Typed state argument must use a None placeholder')
            if any(isinstance(value, dict) and {'sides', 'rng', 'next_entity'} <= set(value) for value in row):
                raise RecordSoAError('Game graph ingress is forbidden in persistent rule calls')
        layout = GraphLayout(rows, self.max_nodes, self.max_edges, self.max_payload_bytes)
        packed, refs = pack_call_arguments(arguments, layout)
        typed_args = np.zeros_like(refs)
        for position in positions:
            typed_args[:, position] = state.roots
        stride = 14+self.limits.locals+self.limits.stack+self.limits.cells
        frames = np.zeros((rows, self.limits.frames, stride), dtype=np.int32)
        outputs = np.full(rows, MISSING, dtype=np.int32)
        roots = np.zeros(rows, dtype=np.int32)
        statuses = np.zeros((rows, 4), dtype=np.int32)
        deadlines = np.zeros(rows, dtype=np.int64)
        constants = self._constants
        cnc = len(constants['nodes'])
        gc_marks = np.zeros((rows, self.max_nodes+cnc), dtype=np.int64)
        gc_queue = np.zeros((rows, self.max_nodes), dtype=np.int32)
        schema = self.schema
        views = np.full((schema.total_objects, rows), MISSING, dtype=np.int32)
        from .vm_soa_views import value_cache_extent
        cache_size = value_cache_extent(schema)
        cache = np.full((cache_size, rows), MISSING, dtype=np.int32)
        clone_map = np.zeros_like(views)
        clone_queue = np.zeros_like(views)
        constant_adopt = np.zeros((cnc, rows), dtype=np.int32)
        active = np.ones(rows, dtype=np.uint8)
        step_limit = self.limits.steps if steps is None else steps
        if type(step_limit) is not int or not 1 <= step_limit <= 2**30:
            raise ValueError('Positive bounded step limit required')
        pointer = lambda key: _cptr(constants[key])
        self._execute(
            _cptr(packed.nodes), _cptr(packed.edges), _cptr(packed.payload), _cptr(packed.counts),
            _cptr(refs), refs.shape[1], pointer('nodes'), pointer('edges'), pointer('payload'),
            pointer('instructions'), pointer('functions'), len(constants['functions']),
            cnc, len(constants['edges']), len(constants['payload']),
            _cptr(frames), _cptr(outputs), _cptr(statuses), _cptr(deadlines),
            rows, self.max_nodes, self.max_edges, self.max_payload_bytes,
            int(self.image.entries[entry]), step_limit, pointer('classes'), len(constants['classes']),
            pointer('modules'), len(constants['modules']), _cptr(gc_marks), _cptr(gc_queue),
            *[_cptr(state.buffers[k]) for k in ('i64', 'f64', 'i32', 'u8')],
            *[_cptr(v) for v in (views, cache, clone_map, clone_queue, constant_adopt, typed_args, roots, active)])
        heap = dict(nodes=packed.nodes, edges=packed.edges, payload=packed.payload, counts=packed.counts)
        return NativeSoACall(self, state, heap, outputs, statuses, roots, views,
                             (views,cache,clone_map,clone_queue,constant_adopt))

    def close(self):
        # dlopen owns the immutable library until this backend is released.
        self._execute = None
        self._library = None
        if self._temp:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
