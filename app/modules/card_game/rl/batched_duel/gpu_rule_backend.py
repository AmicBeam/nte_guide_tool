"""CUDA-only bulk Duel V2 rule backend.

Resident device-graph slots execute frozen GPU-rule entries through CudaVm.
CPU formal apply_action/observe/legal_actions/sampling are not used on the
operation path. Ingress validation, explicit export, recording and oracle
replay stay distinguishable. Full-duel readiness remains false until primary
CUDA acceptance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import math
import queue
import threading
import time
import weakref
import sys
import inspect

import numpy as np

from app.modules.card_game.rl.batched_search.cooperative import (
    OperationQuery,
    OperationResponse,
    RequestExpired,
    _PROTOCOL_METHODS,
)
from app.modules.card_game.rl.batched_duel.process_backend import StateRef
from app.modules.card_game.rl.batched_duel.codec import pack_states, unpack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, DeviceGraph, VmCapabilityError, VmLimits
from app.modules.card_game.rl.batched_duel.gpu_rule_arguments import GraphArgumentKernel
from app.modules.card_game.rl.batched_duel.gpu_rule_program import (
    GpuRuleProgramError,
    inspect_gpu_rule_program,
    load_program,
    lower_gpu_program,
    resolve_entry_names,
    verify_source_hashes,
    detect_frozen_runtime,
)
from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import (
    GpuRuleCapacityError,
    GpuRuleSlotLedger,
    normalize_runtime,
    resolve_gpu_rule_backend_capacity,
    DEFAULT_QUERY_BATCH_ROWS,
    MAX_VM_CONSTANT_NODES,
    graph_state_bytes,
    plan_gpu_rule_capacity,
    split_query_batches,
)
from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch, TAG_TUPLE
from app.modules.card_game.rl.batched_duel.vm_image import ImageError, VmImage
from app.modules.card_game.rl.rule_ir.portable_program import ExportError, FrozenRuleProgram, ValidationError


PROTOCOL_METHODS = _PROTOCOL_METHODS

METHOD_ENTRY = {
    'acting_side': 'acting_side',
    'finished': 'finished',
    'terminal_value': 'terminal_value',
    'decision': 'public_legal_decision',
    'sample_world': 'sample_world',
    'step': 'simulate_action',
    'clean': 'clean',
    'value_observation': 'public_legal_decision',
    'serving_root_metadata': 'serving_root_metadata',
    'public_terminal_probe': 'public_terminal_probe',
    'commit': 'apply_action',
    'rule_decision': 'rule_decision',
}

METHOD_EXTRAS = {
    'acting_side': 0,
    'finished': 0,
    'terminal_value': 1,
    'decision': 1,
    'sample_world': 2,
    'step': 2,
    'clean': 0,
    'value_observation': 1,
    'serving_root_metadata': 1,
    'public_terminal_probe': 2,
    'rule_decision': 1,
}

STATE_METHODS = frozenset({'sample_world', 'step', 'clean'})


class GpuRuleBackendError(RuntimeError):
    """Fail-closed CUDA rule-backend error."""


class GpuRuleCapabilityError(GpuRuleBackendError, VmCapabilityError):
    """Requested GPU-rule entry is missing device capabilities."""


def _enqueue_release(q, key):
    try:
        q.put_nowait(key)
    except Exception:
        pass


def entry_blockers(image, entry_names, *, opcodes, intrinsics):
    """Per-entry capability blockers. Module names in program.required_intrinsics are not the gate."""
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    report = {}
    for name in entry_names:
        missing = image.missing_capabilities(name, opcodes=opcodes, intrinsics=intrinsics)
        report[name] = {
            'entry': name,
            'compile_ready': True,
            'declared_capabilities_ready': not any(missing.values()),
            'device_ready': False,
            'device_verified': False,
            'missing': missing,
            'full_duel_rules': False,
            'actual_backend': 'cuda',
        }
    return report


@dataclass
class _SlotMeta:
    game_id: str
    root_generation: int
    role: str
    is_clean: bool
    actor: object
    phase: str
    winner: object
    kind: str = 'hypothetical'


def _call_with_runtime(fn, /, *args, runtime, **kwargs):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if 'runtime' in params:
        kwargs['runtime'] = runtime
    return fn(*args, **kwargs)


def _program_runtime(program):
    for attr in ('runtime', 'wrapper_runtime'):
        value = getattr(program, attr, None)
        if value in ('recovery', 'seven'):
            return value
    to_dict = getattr(program, 'to_dict', None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            for key in ('runtime', 'wrapper_runtime'):
                if payload.get(key) in ('recovery', 'seven'):
                    return payload[key]
            status = payload.get('status')
            if isinstance(status, dict) and status.get('runtime') in ('recovery', 'seven'):
                return status['runtime']
    return None


class GpuRuleBackend:
    """Resident CUDA bulk resolver. No hidden CPU ResidentRuleBackend fallback."""

    backend_kind = 'cuda_vm'
    supports_gpu = True
    supports_mixed_operations = True
    supports_full_duel_rules = False
    device_verified = False

    def __init__(
        self,
        program=None,
        *,
        device='cuda',
        entries=None,
        layout=None,
        max_games=1600,
        max_states=2048,
        max_nodes=16384,
        max_edges=32768,
        max_payload_bytes=131072,
        operation_timeout=30.0,
        record_output=None,
        verify_sources=True,
        repo_root=None,
        limits=None,
        max_candidates=128,
        guard_standalone_entities=False,
        runtime='recovery',
        capacity_plan=None,
    ):
        self._owner_thread = threading.get_ident()
        if type(guard_standalone_entities) is not bool:
            raise TypeError('Boolean standalone-entity guard required')
        self.guard_standalone_entities = guard_standalone_entities
        if device is None or (isinstance(device, str) and device.split(':', 1)[0] == 'cpu'):
            raise GpuRuleBackendError('CPU device rejected; CUDA-only rule backend')
        if not isinstance(device, str) or not device.startswith('cuda'):
            raise GpuRuleBackendError('CUDA device required; no CPU fallback')
        if (
            type(operation_timeout) not in (int, float)
            or isinstance(operation_timeout, bool)
            or not math.isfinite(operation_timeout)
            or operation_timeout <= 0
        ):
            raise ValueError('Positive finite operation_timeout required')
        try:
            self.runtime = normalize_runtime(runtime)
            self._capacity = resolve_gpu_rule_backend_capacity(
                runtime=self.runtime,
                capacity_plan=capacity_plan,
                max_games=max_games,
                max_states=max_states,
                max_nodes=max_nodes,
                max_edges=max_edges,
                max_payload_bytes=max_payload_bytes,
                max_candidates=max_candidates,
            )
        except GpuRuleCapacityError as exc:
            raise GpuRuleBackendError(str(exc)) from exc
        max_games = self._capacity['max_games']
        max_states = self._capacity['max_states']
        max_nodes = self._capacity['max_nodes']
        max_edges = self._capacity['max_edges']
        max_payload_bytes = self._capacity['max_payload_bytes']
        max_candidates = self._capacity['max_candidates']
        self.max_games = max_games
        self.max_states = max_states
        self.max_roots = self._capacity['max_roots']
        self.query_batch_rows = int(self._capacity.get('query_batch_rows') or DEFAULT_QUERY_BATCH_ROWS)
        self.vm_max_nodes = int(self._capacity.get('vm_max_nodes') or max_nodes)
        self.vm_max_edges = int(self._capacity.get('vm_max_edges') or max_edges)
        self.vm_max_payload_bytes = int(self._capacity.get('vm_max_payload_bytes') or max_payload_bytes)
        if self.vm_max_nodes < max_nodes or self.vm_max_edges < max_edges or self.vm_max_payload_bytes < max_payload_bytes:
            raise GpuRuleBackendError('VM workspace must not shrink persistent graph capacity')
        self.operation_timeout = float(operation_timeout)
        self.backend_id = uuid4().hex
        self.record_output = str(Path(record_output).resolve()) if record_output is not None else None
        self._closed = False
        self._poisoned = False
        self._gpu_released = False
        self.program = self._load_program(
            program, verify_sources=verify_sources, repo_root=repo_root, runtime=self.runtime,
        )
        bound_runtime = _program_runtime(self.program)
        if bound_runtime is not None and bound_runtime != self.runtime:
            raise GpuRuleBackendError(
                'Frozen program runtime %r does not match backend runtime %r' % (bound_runtime, self.runtime)
            )
        self.entry_names = resolve_entry_names(entries if entries is not None else tuple(self.program.entries))
        if 'acting_side' not in self.entry_names:
            raise GpuRuleBackendError('acting_side entry is required for GPU state metadata')
        try:
            self.image = lower_gpu_program(self.program)
        except (ImageError, TypeError, ValueError, RuntimeError) as exc:
            raise GpuRuleBackendError('Failed to lower GPU rule program: %s' % exc) from exc
        if len(self.image.nodes) > MAX_VM_CONSTANT_NODES:
            raise GpuRuleBackendError("Frozen constants exceed the planned VM memo capacity")
        from .cuda_vm import SUPPORTED_INTRINSICS, SUPPORTED_OPS, DENIED_STANDALONE_INTRINSICS
        self._capability_report = entry_blockers(
            self.image, self.entry_names, opcodes=SUPPORTED_OPS, intrinsics=SUPPORTED_INTRINSICS,
        )
        blocked = {
            name: row['missing'] for name, row in self._capability_report.items()
            if row['missing']['opcodes'] or (
                set(row['missing']['intrinsics']) -
                (DENIED_STANDALONE_INTRINSICS if guard_standalone_entities else set()))
        }
        if blocked:
            raise GpuRuleCapabilityError('GPU rule entries missing capabilities: %s' % blocked)
        import torch
        if not torch.cuda.is_available():
            raise GpuRuleBackendError('Actual CUDA required, no CPU VM fallback')
        self.torch = torch
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda':
            raise GpuRuleBackendError('CPU device rejected; CUDA-only rule backend')
        self.vm = CudaVm(self.image, device=str(self.device), limits=limits or VmLimits(),
                         guard_standalone_entities=guard_standalone_entities)
        self.arguments = GraphArgumentKernel(
            device=str(self.device),
            max_nodes=self._capacity['argument_max_nodes'],
            max_edges=self._capacity['argument_max_edges'],
            max_payload_bytes=self._capacity['argument_max_payload_bytes'],
        )
        from .gpu_rule_projection import GraphProjectionKernel
        self.projection = _call_with_runtime(
            GraphProjectionKernel,
            device=str(self.device),
            max_candidates=max_candidates,
            runtime=self.runtime,
        )
        if layout is None:
            layout = GraphLayout(max_states, max_nodes, max_edges, max_payload_bytes)
        if not isinstance(layout, GraphLayout):
            raise TypeError('GraphLayout required')
        if layout.game_capacity < max_states:
            raise ValueError('Layout game_capacity must cover max_states')
        self.layout = layout
        self._init_pool()
        self._ref_cache = weakref.WeakValueDictionary()
        self._release_queue = queue.Queue()
        self._known_clean = set()
        self._game_to_slot = {}
        self._latest_real_generation = {}
        self._method_counts = Counter()
        self._inline_counts = Counter()
        self._released_count = 0
        self._peak_states = 0
        self._decision_cache = {}
        self._physical_records = {}
        self._kernel_ns = 0
        self._peak_query_rows = 0
        self._peak_query_workspace_bytes = 0
        self._observed_query_bytes = 0

    @staticmethod
    def _load_program(program, *, verify_sources, repo_root, runtime='recovery'):
        if program is None:
            raise GpuRuleBackendError('Frozen GPU rule program required')
        if isinstance(program, (str, Path)):
            loaded = _call_with_runtime(
                load_program, program, verify_sources=verify_sources, repo_root=repo_root, runtime=runtime,
            )
        elif isinstance(program, FrozenRuleProgram):
            loaded = program
            loaded.validate()
            detected = detect_frozen_runtime(loaded)
            if detected != runtime:
                raise GpuRuleBackendError(
                    'Frozen program runtime %r does not match backend runtime %r' % (detected, runtime)
                )
            if verify_sources:
                _call_with_runtime(verify_source_hashes, loaded, repo_root=repo_root, runtime=runtime)
        else:
            raise TypeError('FrozenRuleProgram or program path required')
        return loaded

    def _init_pool(self):
        torch = self.torch
        cap = self.max_states
        layout = self.layout
        self._nodes = torch.zeros((cap, layout.max_nodes, 4), dtype=torch.int32, device=self.device)
        self._edges = torch.zeros((cap, layout.max_edges, 2), dtype=torch.int32, device=self.device)
        self._payload = torch.zeros((cap, layout.max_payload_bytes), dtype=torch.uint8, device=self.device)
        self._counts = torch.zeros((cap, 3), dtype=torch.int32, device=self.device)
        self._roots = torch.zeros((cap,), dtype=torch.int32, device=self.device)
        self._ledger = GpuRuleSlotLedger(cap, self.max_games)
        self._slot_meta = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if getattr(self, '_closed', False) and getattr(self, '_gpu_released', False):
            return
        self._closed = True
        try:
            if getattr(self, '_ledger', None) is not None:
                self._ledger.close()
        except Exception:
            pass
        for name in ('projection', 'arguments', 'vm'):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, name, None)
        for name in ('_nodes', '_edges', '_payload', '_counts', '_roots'):
            setattr(self, name, None)
        cache = getattr(self, '_ref_cache', None)
        if cache is not None:
            cache.clear()
        meta = getattr(self, '_slot_meta', None)
        if meta is not None:
            meta.clear()
        known = getattr(self, '_known_clean', None)
        if known is not None:
            known.clear()
        self._gpu_released = True

    def _abort(self):
        self._poisoned = True
        self.close()

    def _check_healthy(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('CUDA rule backend requires its owning scheduler thread')
        if self._poisoned:
            raise RuntimeError('Backend is poisoned from a previous failure and cannot be reused')
        if self._closed:
            raise RuntimeError('Backend is closed')

    def capability_report(self):
        return {
            'supports_full_duel_rules': False,
            'device_verified': False,
            'guard_standalone_entities': self.guard_standalone_entities,
            'actual_backend': 'cuda',
            'runtime': self.runtime,
            'entries': dict(self._capability_report),
            'program_identity': self.program.identity,
            'image_identity': self.image.identity,
        }

    def stats(self):
        self._check_healthy()
        self._flush_releases()
        ledger = self._ledger.metrics()
        pool = 0
        if self._nodes is not None:
            pool = int(
                self._nodes.numel() * self._nodes.element_size()
                + self._edges.numel() * self._edges.element_size()
                + self._payload.numel() * self._payload.element_size()
                + self._counts.numel() * self._counts.element_size()
                + self._roots.numel() * self._roots.element_size()
            )
        return {
            'backend_id': self.backend_id,
            'backend_kind': self.backend_kind,
            'supports_gpu': True,
            'supports_full_duel_rules': False,
            'device_verified': False,
            'device': str(self.device),
            'runtime': self.runtime,
            'max_games': self.max_games,
            'max_states': self.max_states,
            'max_roots': self.max_roots,
            'live_states': ledger['live_states'],
            'peak_states': max(self._peak_states, ledger['peak_states']),
            'released_states': ledger['released_states'],
            'free_slots': ledger['free_slots'],
            'allocated_slots': ledger['allocated_slots'],
            'live_by_kind': ledger['live_by_kind'],
            'candidate_capacity': self._capacity['candidate_capacity'],
            'state_bytes': self._capacity['state_bytes'],
            'query_batch_rows': self.query_batch_rows,
            'vm_max_nodes': self.vm_max_nodes,
            'vm_max_edges': self.vm_max_edges,
            'vm_max_payload_bytes': self.vm_max_payload_bytes,
            'peak_query_rows': self._peak_query_rows,
            'peak_query_workspace_bytes': self._peak_query_workspace_bytes,
            'estimated_peak_bytes': self._capacity['estimated_peak_bytes'],
            'estimated_modeled_bytes': self._capacity.get('estimated_modeled_bytes'),
            'unmodeled_reserve_bytes': self._capacity.get('unmodeled_reserve_bytes'),
            'observed_pool_bytes': pool,
            'observed_query_peak_bytes': self._peak_query_workspace_bytes,
            'observed_peak_bytes': pool + self._peak_query_workspace_bytes,
            'estimated_peak_bytes_are_device_observed': False,
            'derived_from_plan': self._capacity['derived_from_plan'],
            'methods': dict(self._method_counts),
            'inline_operations': dict(self._inline_counts),
            'kernel_ns': self._kernel_ns,
            'pool_bytes': pool,
        }

    def _query_layout(self, rows):
        return GraphLayout(rows, self.vm_max_nodes, self.vm_max_edges, self.vm_max_payload_bytes)

    def _persistent_batch_layout(self, rows):
        return GraphLayout(rows, self.layout.max_nodes, self.layout.max_edges, self.layout.max_payload_bytes)

    def _record_query_workspace(self, rows, layout):
        bytes_ = rows * graph_state_bytes(layout.max_nodes, layout.max_edges, layout.max_payload_bytes)
        if rows > self._peak_query_rows:
            self._peak_query_rows = rows
        if bytes_ > self._peak_query_workspace_bytes:
            self._peak_query_workspace_bytes = bytes_
        self._observed_query_bytes = bytes_

    def _iter_query_batches(self, items):
        items = list(items)
        sizes = split_query_batches(len(items), self.query_batch_rows)
        start = 0
        for size in sizes:
            yield items[start:start + size]
            start += size

    def _runtime_cand_dim(self):
        plan = self._capacity.get('plan')
        if isinstance(plan, dict) and plan.get('cand_dim'):
            return int(plan['cand_dim'])
        if self._capacity.get('cand_dim'):
            return int(self._capacity['cand_dim'])
        from .gpu_schema_contract import GPU_RUNTIME_CONTRACTS
        row = GPU_RUNTIME_CONTRACTS.get(self.runtime)
        if not isinstance(row, dict) or not row.get('cand_dim'):
            raise GpuRuleBackendError(
                'gpu_schema_contract candidate width is unbound for runtime %r' % (self.runtime,)
            )
        return int(row['cand_dim'])

    def _validate_ref(self, ref):
        if not isinstance(ref, StateRef):
            raise TypeError('Expected StateRef instance, got %s' % type(ref))
        if ref.backend_id != self.backend_id:
            raise ValueError('Cross-backend reference rejected')
        if ref.worker != 0:
            raise ValueError('CUDA backend uses worker 0')
        key = (0, ref.slot, ref.generation)
        self._ledger.validate_key(key)
        canonical = self._ref_cache.get(key)
        if canonical is None or canonical != ref:
            raise ValueError('State metadata does not match its issued reference')

    def _require_entry(self, method):
        entry = METHOD_ENTRY[method]
        if entry not in self.image.entries or entry not in self.entry_names:
            raise GpuRuleCapabilityError('GPU rule entry %s is not loaded' % entry)
        row = self._capability_report.get(entry)
        if row is not None:
            from .cuda_vm import DENIED_STANDALONE_INTRINSICS
            missing=row['missing']
            permitted=DENIED_STANDALONE_INTRINSICS if self.guard_standalone_entities else set()
            if missing['opcodes'] or set(missing['intrinsics'])-permitted:
                raise GpuRuleCapabilityError('GPU rule entry %s is unsupported: %s' % (entry, missing))
        return entry

    def _allocate_slots(self, n):
        try:
            return self._ledger.allocate(n)
        except OverflowError:
            self._abort()
            raise

    def _store_graph(self, graph, slots):
        if graph.rows != len(slots):
            raise ValueError('Graph row/slot count mismatch')
        index = self.torch.tensor(slots, dtype=self.torch.int64, device=self.device)
        self._nodes.index_copy_(0, index, graph.nodes)
        self._edges.index_copy_(0, index, graph.edges)
        self._payload.index_copy_(0, index, graph.payload)
        self._counts.index_copy_(0, index, graph.counts)
        self._roots.index_copy_(0, index, graph.roots)

    def gather(self, refs):
        """Device index_select of referenced slots into a compact DeviceGraph."""
        self._check_healthy()
        slots = []
        for ref in refs:
            self._validate_ref(ref)
            slots.append(ref.slot)
        return self._gather_slots(slots), slots

    def _gather_slots(self, slots):
        if not slots:
            raise ValueError('Nonempty gather required')
        index = self.torch.tensor(slots, dtype=self.torch.int64, device=self.device)
        return DeviceGraph(
            self.layout,
            self._nodes.index_select(0, index).contiguous(),
            self._edges.index_select(0, index).contiguous(),
            self._payload.index_select(0, index).contiguous(),
            self._counts.index_select(0, index).contiguous(),
            self._roots.index_select(0, index).contiguous(),
        )

    def _compact_outputs(self, call):
        call.check()
        ex = call.executor
        torch = ex.torch
        a = call.arena
        layout = a.layout
        rows = a.rows
        capacity = layout.max_nodes + len(ex.image.nodes)
        dn = torch.zeros_like(a.nodes)
        de = torch.zeros_like(a.edges)
        dp = torch.zeros_like(a.payload)
        dc = torch.zeros_like(a.counts)
        dr = torch.empty_like(a.roots)
        mapping = torch.empty((rows, capacity), dtype=torch.int32, device=ex.device)
        queue = torch.empty_like(mapping)
        status = torch.empty(rows, dtype=torch.int32, device=ex.device)
        buffers = ex._buffers
        event = ex.module.launch(
            'vm_compact',
            [a.nodes, a.edges, a.payload, a.counts, buffers['nodes'], buffers['edges'], buffers['payload'],
             len(ex.image.nodes), len(ex.image.edges), len(ex.image.payload), call.outputs,
             dn, de, dp, dc, dr, mapping, queue, status,
             rows, layout.max_nodes, layout.max_edges, layout.max_payload_bytes, capacity],
            grid=(rows + 127) // 128, block=128,
        )
        event.synchronize()
        values = status.cpu().numpy()
        bad = np.flatnonzero(values)
        if len(bad):
            raise GpuRuleBackendError(
                'GPU output graph compact failed row=%s status=%s' % (int(bad[0]), int(values[int(bad[0])]))
            )
        compacted = DeviceGraph(layout, dn, de, dp, dc, dr)
        persistent = self._persistent_batch_layout(rows)
        if (
            layout.max_nodes == persistent.max_nodes
            and layout.max_edges == persistent.max_edges
            and layout.max_payload_bytes == persistent.max_payload_bytes
        ):
            return compacted
        try:
            return self.arguments.copy_to_layout(compacted, persistent)
        except Exception as exc:
            raise GpuRuleBackendError(
                'GPU output could not fit persistent graph limits: %s' % exc
            ) from exc

    def _compact_custom_roots(self, arena, roots, executor):
        torch = executor.torch
        layout = arena.layout
        rows = arena.rows
        capacity = layout.max_nodes + len(executor.image.nodes)
        dn = torch.zeros_like(arena.nodes)
        de = torch.zeros_like(arena.edges)
        dp = torch.zeros_like(arena.payload)
        dc = torch.zeros_like(arena.counts)
        dr = torch.empty_like(arena.roots)
        mapping = torch.empty((rows, capacity), dtype=torch.int32, device=executor.device)
        queue = torch.empty_like(mapping)
        status = torch.empty(rows, dtype=torch.int32, device=executor.device)
        buffers = executor._buffers
        event = executor.module.launch(
            'vm_compact',
            [arena.nodes, arena.edges, arena.payload, arena.counts, buffers['nodes'], buffers['edges'], buffers['payload'],
             len(executor.image.nodes), len(executor.image.edges), len(executor.image.payload), roots,
             dn, de, dp, dc, dr, mapping, queue, status,
             rows, layout.max_nodes, layout.max_edges, layout.max_payload_bytes, capacity],
            grid=(rows + 127) // 128, block=128,
        )
        event.synchronize()
        values = status.cpu().numpy()
        bad = np.flatnonzero(values)
        if len(bad):
            raise GpuRuleBackendError(
                'GPU explicit export compact failed row=%s status=%s' % (int(bad[0]), int(values[int(bad[0])]))
            )
        compacted = DeviceGraph(layout, dn, de, dp, dc, dr)
        persistent = self._persistent_batch_layout(rows)
        if (
            layout.max_nodes != persistent.max_nodes
            or layout.max_edges != persistent.max_edges
            or layout.max_payload_bytes != persistent.max_payload_bytes
        ):
            try:
                compacted = self.arguments.copy_to_layout(compacted, persistent)
            except Exception as exc:
                raise GpuRuleBackendError(
                    'GPU explicit export could not fit persistent graph limits: %s' % exc
                ) from exc
        packed = PackedStateBatch(
            compacted.layout, *(value.cpu().numpy() for value in (
                compacted.nodes, compacted.edges, compacted.payload, compacted.counts, compacted.roots,
            )),
        )
        return unpack_states(packed)

    def _register(self, slot, meta):
        kind = meta.kind if getattr(meta, 'kind', None) else ('real' if meta.role == 'real' else 'hypothetical')
        key = self._ledger.register(slot, kind)
        generation = key[2]
        ref = StateRef(
            backend_id=self.backend_id,
            worker=0,
            slot=slot,
            generation=generation,
            game_id=meta.game_id,
            root_generation=meta.root_generation,
            actor=meta.actor,
            phase=meta.phase,
            winner=meta.winner,
            role=meta.role,
        )
        self._ref_cache[key] = ref
        self._slot_meta[slot] = meta
        weakref.finalize(ref, _enqueue_release, self._release_queue, key)
        live = len(self._ledger.live)
        if live > self._peak_states:
            self._peak_states = live
        return ref

    def _release_slot(self, slot, generation):
        key = (0, slot, generation)
        if not self._ledger._release_key(key, require_live=False):
            return False
        self._known_clean.discard(key)
        self._ref_cache.pop(key, None)
        self._slot_meta.pop(slot, None)
        for side in ('a', 'b'):
            self._decision_cache.pop((slot, generation, side), None)
        self._released_count += 1
        return True

    def _get_pending_releases(self):
        released = []
        while True:
            try:
                worker, slot, gen = self._release_queue.get_nowait()
            except queue.Empty:
                break
            if self._release_slot(slot, gen):
                released.append((slot, gen))
        return released

    def _flush_releases(self):
        self._get_pending_releases()

    def _refresh_meta(self, slots, *, game_ids, root_generations, roles, is_clean, kinds=None):
        if kinds is None:
            kinds = ['real' if role == 'real' else 'hypothetical' for role in roles]
        slots = list(slots)
        game_ids = list(game_ids)
        root_generations = list(root_generations)
        roles = list(roles)
        is_clean = list(is_clean)
        kinds = list(kinds)
        refs = []
        try:
            for start in range(0, len(slots), self.query_batch_rows):
                end = start + self.query_batch_rows
                chunk_slots = slots[start:end]
                graph = self._gather_slots(chunk_slots)
                self._record_query_workspace(len(chunk_slots), graph.layout)
                bound = self.arguments.bind(graph, [() for _ in chunk_slots])
                call = self.vm.launch(bound.arena, 'acting_side', argument_refs=bound.argument_refs)
                actors = call.decode()
                phases, winners = self.arguments.read_meta(bound.arena)
                for slot, actor, phase, winner, game_id, root_generation, role, clean, kind in zip(
                    chunk_slots, actors, phases, winners,
                    game_ids[start:end], root_generations[start:end], roles[start:end],
                    is_clean[start:end], kinds[start:end],
                ):
                    if actor not in (None, 'a', 'b'):
                        raise GpuRuleBackendError('GPU acting_side returned an invalid actor')
                    meta = _SlotMeta(game_id, root_generation, role, clean, actor, phase, winner, kind)
                    ref = self._register(slot, meta)
                    if clean:
                        self._known_clean.add((0, slot, ref.generation))
                    refs.append(ref)
        except Exception:
            self._abort()
            raise
        return refs

    def import_states(self, states, game_ids, root_generations):
        self._check_healthy()
        self._flush_releases()
        if len(states) != len(game_ids) or len(states) != len(root_generations):
            raise ValueError('Import state/game/generation lengths must match')
        if not states:
            return []
        for i, (gid, rgen) in enumerate(zip(game_ids, root_generations)):
            if not isinstance(gid, str) or not gid:
                raise ValueError('Nonempty game id required')
            if type(rgen) is not int or isinstance(rgen, bool) or rgen < 0:
                raise TypeError('Non-negative root generation required')
            if self.record_output is not None and any(
                ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in gid
            ):
                raise ValueError('Recorded game id must be a safe filename')
        if len(set(game_ids)) != len(game_ids) or any(gid in self._game_to_slot for gid in game_ids):
            raise ValueError('A game can be imported only once in this backend')
        if len(self._game_to_slot) + len(game_ids) > self.max_games:
            raise OverflowError('Real game admission capacity exceeded')
        try:
            self._require_entry('acting_side')
            slots = self._allocate_slots(len(states))
            refs = []
            for start in range(0, len(states), self.query_batch_rows):
                end = start + self.query_batch_rows
                packed = pack_states(
                    states[start:end],
                    GraphLayout(end - start, self.layout.max_nodes, self.layout.max_edges, self.layout.max_payload_bytes),
                )
                graph = self.vm.ingest(packed)
                chunk_slots = slots[start:end]
                self._store_graph(graph, chunk_slots)
                refs.extend(self._refresh_meta(
                    chunk_slots,
                    game_ids=list(game_ids[start:end]),
                    root_generations=list(root_generations[start:end]),
                    roles=['real'] * (end - start),
                    is_clean=[False] * (end - start),
                    kinds=['real'] * (end - start),
                ))
        except Exception:
            self._abort()
            raise
        self._ledger.admit_real_games(game_ids)
        for ref in refs:
            self._game_to_slot[ref.game_id] = ref.slot
            self._latest_real_generation[ref.game_id] = ref.root_generation
        if self.record_output is not None:
            from .recording import PhysicalGameRecord
            for state, gid in zip(states, game_ids):
                self._physical_records[gid] = PhysicalGameRecord(
                    state, journal_directory=Path(self.record_output) / 'journals' / gid,
                )
        return refs

    def export_states(self, refs):
        """Explicit host snapshot for validation/archival; not a simulation hot path."""
        self._check_healthy()
        self._flush_releases()
        if not refs:
            return []
        exported = []
        for chunk in self._iter_query_batches(refs):
            graph, _ = self.gather(chunk)
            packed = PackedStateBatch(
                self._persistent_batch_layout(len(chunk)),
                graph.nodes.detach().cpu().numpy().copy(),
                graph.edges.detach().cpu().numpy().copy(),
                graph.payload.detach().cpu().numpy().copy(),
                graph.counts.detach().cpu().numpy().copy(),
                graph.roots.detach().cpu().numpy().copy(),
            )
            exported.extend(unpack_states(packed))
        return exported

    def release(self, refs):
        self._check_healthy()
        keys = []
        for ref in refs:
            self._validate_ref(ref)
            keys.append((0, ref.slot, ref.generation))
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate state release')
        for ref in refs:
            self._release_slot(ref.slot, ref.generation)

    def fork(self, refs, *, role='hypothetical'):
        """Clone device rows into new slots without host round-trip."""
        self._check_healthy()
        if role not in ('real', 'hypothetical'):
            raise ValueError('role must be real or hypothetical')
        if not refs:
            return []
        cloned = []
        try:
            for chunk in self._iter_query_batches(refs):
                graph, _ = self.gather(chunk)
                slots = self._allocate_slots(len(chunk))
                self._store_graph(graph, slots)
                cloned.extend(self._refresh_meta(
                    slots,
                    game_ids=[ref.game_id for ref in chunk],
                    root_generations=[ref.root_generation for ref in chunk],
                    roles=[role] * len(chunk),
                    is_clean=[(0, ref.slot, ref.generation) in self._known_clean for ref in chunk],
                    kinds=['real' if role == 'real' else 'hypothetical'] * len(chunk),
                ))
        except Exception:
            self._abort()
            raise
        return cloned

    def _launch(self, refs, entry, extras, deadlines=None):
        if len(refs) > self.query_batch_rows:
            raise GpuRuleBackendError(
                'kernel batch %s exceeds query_batch_rows %s; split the group, do not launch all roots'
                % (len(refs), self.query_batch_rows)
            )
        graph, slots = self.gather(refs)
        try:
            destination_layout = self._query_layout(len(refs))
            self._record_query_workspace(len(refs), destination_layout)
            bound = self.arguments.bind(graph, extras, destination_layout=destination_layout)
            gpu_deadlines = None
            if deadlines is not None:
                gpu_deadlines = self._deadline_tensor(len(refs), deadlines)
            started = time.perf_counter_ns()
            call = self.vm.launch(bound.arena, entry, argument_refs=bound.argument_refs, gpu_deadlines=gpu_deadlines)
            call.check()
            self._kernel_ns += time.perf_counter_ns() - started
        except Exception:
            self._abort()
            raise
        return call, slots

    def _deadline_tensor(self, rows, deadlines):
        torch = self.torch
        out = torch.zeros(rows, dtype=torch.int64, device=self.device)
        clock = self.vm.clock_ns()
        now = time.monotonic()
        for i, deadline in enumerate(deadlines):
            if deadline is None:
                continue
            remaining = deadline - now
            if remaining <= 0:
                out[i] = 1
            else:
                out[i] = int(clock + remaining * 1_000_000_000.0)
        return out

    def _adopt_outputs(self, call, source_refs, *, role, is_clean, root_generations=None, kind=None, kinds=None):
        if kinds is None:
            if kind is None:
                kind = 'real' if role == 'real' else 'hypothetical'
            kinds = [kind] * len(source_refs)
        try:
            graph = self._compact_outputs(call)
            slots = self._allocate_slots(len(source_refs))
            self._store_graph(graph, slots)
            return self._refresh_meta(
                slots,
                game_ids=[ref.game_id for ref in source_refs],
                root_generations=list(root_generations) if root_generations is not None else [ref.root_generation for ref in source_refs],
                roles=[role] * len(source_refs),
                is_clean=[is_clean] * len(source_refs),
                kinds=kinds,
            )
        except Exception:
            self._abort()
            raise

    def _tuple_child_refs(self, call, index):
        """Read a tuple child reference on device. Does not export state or run host extractors."""
        torch = self.torch
        arena = call.arena
        outputs = call.outputs
        rows = arena.rows
        if type(index) is not int or isinstance(index, bool) or index < 0:
            raise ValueError('Non-negative tuple child index required')
        if bool((outputs < 0).any()):
            raise GpuRuleBackendError('GPU entry output is not a device tuple')
        idx = outputs.to(torch.int64)
        node_count = arena.counts[:, 0].to(torch.int64)
        edge_count = arena.counts[:, 1].to(torch.int64)
        if bool((idx >= node_count).any()):
            raise GpuRuleBackendError('GPU tuple output reference is outside the graph')
        node = arena.nodes[torch.arange(rows, device=self.device), idx]
        tag = node[:, 0]
        offset = node[:, 1].to(torch.int64)
        count = node[:, 2]
        if bool((tag != TAG_TUPLE).any()) or bool((count <= index).any()) or bool((offset + index >= edge_count).any()):
            raise GpuRuleBackendError('GPU entry output is not a tuple of legal actions and public view')
        child = arena.edges[torch.arange(rows, device=self.device), offset + index, 0]
        return child.contiguous()

    def _decision_from_call(self, call):
        from .gpu_rule_projection import GraphProjectionError
        try:
            projected = self.projection.project_decision(call)
        except GraphProjectionError as exc:
            if 'empty_policy' in str(exc):
                raise ValueError('No legal non-concede action') from exc
            self._abort()
            raise
        exported = self._compact_custom_roots(projected.arena, projected.action_roots, call.executor)
        observations = projected.observations.observations
        candidates = projected.candidates.candidates
        counts = projected.candidates.counts
        results = []
        for i, actions in enumerate(exported):
            if not actions:
                raise ValueError('No legal non-concede action')
            count = int(counts[i])
            if count > int(candidates.shape[1]):
                raise GpuRuleBackendError(
                    'Candidate count %s exceeds bound %s' % (count, int(candidates.shape[1]))
                )
            x = observations[i].detach().cpu().numpy()
            c = candidates[i, :count].detach().cpu().numpy()
            results.append((list(actions), (x, c)))
        return results

    def resolve_operations(self, queries):
        self._check_healthy()
        self._flush_releases()
        if not queries:
            return []
        seen = set()
        for query in queries:
            if not isinstance(query, OperationQuery):
                raise TypeError('Expected OperationQuery')
            if query.id in seen:
                raise ValueError('Duplicate query identity')
            if query.backend is not self:
                raise ValueError('Operation belongs to another backend')
            seen.add(query.id)
            if query.method not in PROTOCOL_METHODS:
                raise ValueError('Unknown backend operation')
            extras = METHOD_EXTRAS[query.method]
            if len(query.args) != 1 + extras or query.kwargs:
                raise ValueError('Unsupported rule operation signature')
            self._validate_ref(query.args[0])
            if extras >= 1 and query.args[1] not in ('a', 'b'):
                raise ValueError('Rule operation requires side a or b')
            if query.method == 'sample_world' and type(query.args[2]) is not int:
                raise TypeError('Integer world seed required')
            if query.deadline is not None and (
                isinstance(query.deadline, bool)
                or not isinstance(query.deadline, (int, float))
                or not math.isfinite(query.deadline)
            ):
                raise ValueError('Finite monotonic rule operation deadline required')
        responses = {}
        pending = []
        now = time.monotonic()
        for query in queries:
            handled, value = self.inline_operation(query.method, query.args)
            if handled:
                responses[query.id] = OperationResponse(query.id, value)
                continue
            if query.deadline is not None and now >= query.deadline:
                responses[query.id] = OperationResponse(query.id, RequestExpired(query.id))
                continue
            pending.append(query)
        groups = {}
        for query in pending:
            groups.setdefault(query.method, []).append(query)
        for method, group in groups.items():
            entry = self._require_entry(method)
            self._method_counts[method] += len(group)
            for chunk in self._iter_query_batches(group):
                now = time.monotonic()
                live = []
                for query in chunk:
                    if query.deadline is not None and now >= query.deadline:
                        responses[query.id] = OperationResponse(query.id, RequestExpired(query.id))
                    else:
                        live.append(query)
                if not live:
                    continue
                refs = [query.args[0] for query in live]
                extras = [tuple(query.args[1:]) for query in live]
                deadlines = [query.deadline for query in live]
                if method == 'clean':
                    values = []
                    launch_refs = []
                    launch_index = []
                    for i, ref in enumerate(refs):
                        key = (ref.worker, ref.slot, ref.generation)
                        if key in self._known_clean:
                            values.append(ref)
                        else:
                            launch_index.append(i)
                            launch_refs.append(ref)
                            values.append(None)
                    if launch_refs:
                        call, _ = self._launch(launch_refs, entry, [() for _ in launch_refs], [deadlines[i] for i in launch_index])
                        adopted = self._adopt_outputs(
                            call, launch_refs, role=launch_refs[0].role, is_clean=True,
                            kinds=['root_clean' if ref.role == 'real' else 'hypothetical' for ref in launch_refs],
                        )
                        for i, ref in zip(launch_index, adopted):
                            values[i] = ref
                    for query, value in zip(live, values):
                        responses[query.id] = OperationResponse(query.id, value)
                    continue
                call, _ = self._launch(refs, entry, extras, deadlines)
                if method == 'decision':
                    values = self._decision_from_call(call)
                    for query, value in zip(live, values):
                        self._decision_cache[(query.args[0].slot, query.args[0].generation, query.args[1])] = value
                        responses[query.id] = OperationResponse(query.id, value)
                elif method == 'value_observation':
                    view_refs = self._tuple_child_refs(call, 1)
                    projected = self.projection.project_observation(call, view_refs)
                    observations = projected.observations
                    cand_dim = self._runtime_cand_dim()
                    for i, query in enumerate(live):
                        x = observations[i].detach().cpu().numpy()
                        c = np.zeros((0, cand_dim), dtype=np.float32)
                        responses[query.id] = OperationResponse(query.id, (x, c))
                elif method in STATE_METHODS:
                    if method == 'step':
                        mid = self._adopt_outputs(call, refs, role='hypothetical', is_clean=False, kind='simulated')
                        try:
                            call2, _ = self._launch(mid, 'clean', [() for _ in mid], deadlines)
                            values = self._adopt_outputs(call2, mid, role='hypothetical', is_clean=True, kind='temporary')
                            self.release(mid)
                        except Exception:
                            self._abort()
                            raise
                    else:
                        role = 'hypothetical' if method == 'sample_world' else refs[0].role
                        kind = 'sampled' if method == 'sample_world' else ('root_clean' if role == 'real' else 'hypothetical')
                        values = self._adopt_outputs(call, refs, role=role, is_clean=method == 'clean', kind=kind)
                    for query, value in zip(live, values):
                        responses[query.id] = OperationResponse(query.id, value)
                else:
                    decoded = call.decode()
                    for query, value in zip(live, decoded):
                        responses[query.id] = OperationResponse(query.id, value)
        return [responses[query.id] for query in queries]

    def inline_operation(self, method, args):
        if method not in ('acting_side', 'finished', 'terminal_value', 'clean'):
            return False, None
        self._check_healthy()
        expected = 2 if method == 'terminal_value' else 1
        if len(args) != expected:
            raise ValueError('Inline metadata arity mismatch')
        ref = args[0]
        self._validate_ref(ref)
        if method == 'terminal_value' and args[1] not in ('a', 'b'):
            raise ValueError('Invalid inline observer')
        if method == 'clean' and (ref.worker, ref.slot, ref.generation) not in self._known_clean:
            return False, None
        if method == 'acting_side':
            value = ref.actor
        elif method == 'finished':
            value = ref.phase == 'finished'
        elif method == 'terminal_value':
            value = 0. if ref.winner not in ('a', 'b') else (1. if ref.winner == args[1] else -1.)
        else:
            value = ref
        self._method_counts[method] += 1
        self._inline_counts[method] += 1
        return True, value

    def acting_side(self, ref):
        self._validate_ref(ref)
        return ref.actor

    def finished(self, ref):
        self._validate_ref(ref)
        return ref.phase == 'finished'

    def terminal_value(self, ref, viewer):
        self._validate_ref(ref)
        if ref.winner not in ('a', 'b'):
            return 0.0
        return 1.0 if ref.winner == viewer else -1.0

    def rule_decision(self,ref,side):
        if 'rule_decision' not in self.program.entries:
            raise GpuRuleBackendError('Frozen program lacks the canonical rule teacher')
        return self.resolve_operations([OperationQuery(uuid4().hex,self,'rule_decision',(ref,side),())])[0].value

    def decision(self, ref, side):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'decision', (ref, side), ())])[0].value

    def sample_world(self, ref, viewer, seed):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'sample_world', (ref, viewer, seed), ())])[0].value

    def step(self, ref, side, action):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'step', (ref, side, action), ())])[0].value

    def clean(self, ref):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'clean', (ref,), ())])[0].value

    def value_observation(self, ref, viewer):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'value_observation', (ref, viewer), ())])[0].value

    def serving_root_metadata(self, ref, side):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'serving_root_metadata', (ref, side), ())])[0].value

    def public_terminal_probe(self, ref, side, action):
        return self.resolve_operations([OperationQuery(uuid4().hex, self, 'public_terminal_probe', (ref, side, action), ())])[0].value

    def commit_action(self, ref, side, action, next_root_generation, *, search_stats=None):
        return self.commit_actions([(ref, side, action, next_root_generation, search_stats)])[0]

    def commit_actions(self, commits):
        self._check_healthy()
        self._flush_releases()
        commits = list(commits)
        seen = set()
        refs = []
        extras = []
        stats = []
        next_gens = []
        for item in commits:
            if len(item) != 5:
                raise ValueError('Physical commit tuple requires five fields')
            ref, side, action, next_gen, search_stats = item
            self._validate_ref(ref)
            if ref.role != 'real' or ref.game_id in seen:
                raise ValueError('One real commit per game required')
            seen.add(ref.game_id)
            if ref.root_generation != self._latest_real_generation.get(ref.game_id):
                raise ValueError('Physical state has already advanced')
            if type(next_gen) is not int or next_gen != ref.root_generation + 1:
                raise ValueError('Physical generation must advance by one')
            if side not in ('a', 'b') or side != ref.actor or not isinstance(action, dict):
                raise ValueError('Physical action requires its current decision owner')
            refs.append(ref)
            extras.append((side, action))
            stats.append(search_stats)
            next_gens.append(next_gen)
        entry = self._require_entry('commit')
        before_states = None
        if self.record_output is not None:
            before_states = []
            for chunk in self._iter_query_batches(refs):
                before_states.extend(self.export_states(chunk))
        results = []
        for start in range(0, len(refs), self.query_batch_rows):
            end = start + self.query_batch_rows
            call, _ = self._launch(refs[start:end], entry, extras[start:end])
            results.extend(self._adopt_outputs(
                call, refs[start:end], role='real', is_clean=False,
                root_generations=next_gens[start:end], kind='real',
            ))
        for dest in results:
            self._game_to_slot[dest.game_id] = dest.slot
            self._latest_real_generation[dest.game_id] = dest.root_generation
        if self.record_output is not None:
            after_states = []
            for chunk in self._iter_query_batches(results):
                after_states.extend(self.export_states(chunk))
            for source, extra, before, after, search_stats in zip(refs, extras, before_states, after_states, stats):
                side, action = extra
                decision = self._decision_cache.get((source.slot, source.generation, side))
                legal = decision[0] if decision is not None else None
                if legal is None:
                    raise GpuRuleBackendError('Physical recording requires a prior GPU decision for the actor')
                self._physical_records[source.game_id].append(
                    before, after, side, action, legal, search_stats=search_stats,
                )
        return results

    def finish_games(self, refs, jobs):
        self._check_healthy()
        self._flush_releases()
        if self.record_output is None:
            raise ValueError('Physical recording was not configured')
        if len(refs) != len(jobs):
            raise ValueError('Finish job count mismatch')
        states = self.export_states(refs)
        results = []
        seen = set()
        for ref, job, state in zip(refs, jobs, states):
            self._validate_ref(ref)
            if (ref.role != 'real' or job.get('id') != ref.game_id or ref.game_id in seen
                    or self._latest_real_generation.get(ref.game_id) != ref.root_generation):
                raise ValueError('Finish requires each current real game once')
            seen.add(ref.game_id)
            record = self._physical_records[ref.game_id]
            result = record.finish(state, job, self.record_output, public_replay=bool(job.get('replay')))
            del self._physical_records[ref.game_id]
            results.append(result)
        return results


def inspect_backend_configuration(requested=None, *, runtime='recovery'):
    """CPU-only capability/check report. Does not construct a CUDA backend."""
    from .cuda_vm import SUPPORTED_INTRINSICS, SUPPORTED_OPS
    from .gpu_rule_program import capture_gpu_rule_program, source_capture_available
    runtime = normalize_runtime(runtime)
    entry_names = resolve_entry_names(requested)
    program = None
    capture_error = None
    if source_capture_available():
        try:
            program = _call_with_runtime(capture_gpu_rule_program, entry_names, runtime=runtime)
        except (ExportError, GpuRuleProgramError, ValidationError, TypeError, ValueError, RuntimeError) as exc:
            capture_error = str(exc)
    status = _call_with_runtime(
        inspect_gpu_rule_program, entry_names, program=program, lower=program is not None, runtime=runtime,
    )
    if capture_error and not status.get('error'):
        status['error'] = capture_error
    status['mode'] = 'check'
    status['runtime'] = runtime
    status['supports_full_duel_rules'] = False
    status['device_verified'] = False
    status['actual_backend'] = 'cuda'
    status['hidden_cpu_fallback'] = False
    status['source_capture_available'] = source_capture_available()
    status['supported_opcodes'] = sorted(SUPPORTED_OPS)
    status['supported_intrinsics'] = sorted(SUPPORTED_INTRINSICS)
    if runtime == 'recovery':
        status['sampler_contract'] = 'existing recovery runtime; additional catalog rosters are not implicitly supported'
    else:
        status['sampler_contract'] = 'explicit seven-team runtime; recovery five-team identity is not used'
    status['torch_imported'] = 'torch' in sys.modules
    entries = status.get('entries') or {}
    status['capability_ready_entries'] = [
        name for name, row in entries.items()
        if isinstance(row, dict) and row.get('declared_capabilities_ready')
    ]
    status['blocked_entries'] = {
        name: row.get('missing') for name, row in entries.items()
        if isinstance(row, dict) and row.get('declared_capabilities_ready') is False
    }
    return status
