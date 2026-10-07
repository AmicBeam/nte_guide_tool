"""Resident process-based formal rules bulk resolver for Duel V2 batched search.

Hosts official Duel V2 CPU rules and public information-set sampler across a fixed
pool of dedicated worker processes. States are imported once into worker slot storage
and subsequently referenced by immutable StateRef handles.

This is strictly a CPU reference bulk resolver (backend_kind='python_resident_process',
supports_gpu=False). It does NOT execute GPU kernels or load neural network weights,
and does NOT create per-game OS threads.
"""

from __future__ import annotations

import collections
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import math
import multiprocessing
import os
import queue
import sys
import time
import threading
from typing import Any, List, Optional, Sequence, Tuple, Union
from uuid import uuid4
import weakref

import numpy as np

from app.modules.card_game.rl.batched_search.cooperative import (
    OperationQuery,
    OperationResponse,
    RequestExpired,
    _PROTOCOL_METHODS,
)

PROTOCOL_METHODS = _PROTOCOL_METHODS


@dataclass(frozen=True)
class StateRef:
    """Immutable state reference handle referencing resident state in a CPU worker slot.

    Includes identity and metadata for parent-side immediate queries without IPC.
    """

    backend_id: str
    worker: int
    slot: int
    generation: int
    game_id: str
    root_generation: int
    actor: Optional[str]
    phase: str
    winner: Optional[str]
    role: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id:
            raise ValueError("backend_id must be a non-empty string UUID")
        if type(self.worker) is not int or type(self.worker) is bool or self.worker < 0:
            raise TypeError("worker must be a non-negative integer, not boolean or float")
        if type(self.slot) is not int or type(self.slot) is bool or self.slot < 0:
            raise TypeError("slot must be a non-negative integer, not boolean or float")
        if type(self.generation) is not int or type(self.generation) is bool or self.generation < 0:
            raise TypeError("generation must be a non-negative integer, not boolean or float")
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ValueError("game_id must be a non-empty string")
        if (
            type(self.root_generation) is not int
            or type(self.root_generation) is bool
            or self.root_generation < 0
        ):
            raise TypeError("root_generation must be a non-negative integer, not boolean or float")
        if self.actor is not None and self.actor not in ("a", "b"):
            raise ValueError(f"actor must be 'a', 'b', or None, got {self.actor!r}")
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("phase must be a non-empty string")
        if self.winner is not None and self.winner not in ("a", "b", "draw"):
            raise ValueError(f"winner must be 'a', 'b', 'draw', or None, got {self.winner!r}")
        if self.role not in ("real", "hypothetical"):
            raise ValueError(f"role must be 'real' or 'hypothetical', got {self.role!r}")


@dataclass(frozen=True)
class WireStateMeta:
    """Wire metadata returned by worker upon state allocation or recycling."""

    worker: int
    slot: int
    generation: int
    game_id: str
    root_generation: int
    actor: Optional[str]
    phase: str
    winner: Optional[str]
    role: str


@dataclass(frozen=True)
class WireOp:
    """Wire representation of an operation passed to a worker."""

    id: str
    method: str
    slot: int
    generation: int
    args: tuple
    kwargs: tuple
    deadline: float | None = None


@dataclass(frozen=True)
class WorkerBatchResult:
    """Aggregated batch execution result from a worker."""

    worker_id: int
    pid: int
    results: list[tuple[str, Any]]
    released_count: int
    live_states: int
    execution_ns: int
    methods_ns: dict[str, int]


# --- Worker Process State and Handlers ---

_worker_state: Optional[_WorkerState] = None


class _WorkerState:
    """Worker-local resident state storage and execution engine."""

    def __init__(self, worker_id: int, max_states: int, record_output=None) -> None:
        self.worker_id = worker_id
        self.max_states = max_states
        self.states: dict[int, Any] = {}
        self.generations: list[int] = [0] * max_states
        self.is_clean: dict[int, bool] = {}
        self.roles: dict[int, str] = {}
        self.game_ids: dict[int, str] = {}
        self.root_generations: dict[int, int] = {}
        self.free_slots: list[int] = list(range(max_states - 1, -1, -1))
        self.released_count = 0
        self.record_output=record_output
        self.physical_records={}
        self.decision_cache={}

    def allocate_slot(
        self,
        state: Any,
        *,
        role: str,
        game_id: str,
        root_generation: int,
        is_clean: bool = False,
    ) -> WireStateMeta:
        if not self.free_slots:
            raise RuntimeError(
                f"Worker {self.worker_id} capacity overflow: max {self.max_states} states exceeded"
            )
        slot = self.free_slots.pop()
        self.states[slot] = state
        gen = self.generations[slot]
        self.is_clean[slot] = is_clean
        self.roles[slot] = role
        self.game_ids[slot] = game_id
        self.root_generations[slot] = root_generation

        from app.modules.card_game.engine.duel_v2 import acting_side

        actor = acting_side(state)
        phase = state.get("phase", "") if isinstance(state, dict) else getattr(state, "phase", "")
        winner = state.get("winner") if isinstance(state, dict) else getattr(state, "winner", None)
        return WireStateMeta(
            worker=self.worker_id,
            slot=slot,
            generation=gen,
            game_id=game_id,
            root_generation=root_generation,
            actor=actor,
            phase=phase,
            winner=winner,
            role=role,
        )

    def release_slot(self, slot: int, generation: int) -> bool:
        if slot in self.states and self.generations[slot] == generation:
            del self.states[slot]
            self.is_clean.pop(slot, None)
            self.roles.pop(slot, None)
            self.game_ids.pop(slot, None)
            self.root_generations.pop(slot, None)
            for side in ('a','b'):self.decision_cache.pop((slot,generation,side),None)
            self.generations[slot] += 1
            self.free_slots.append(slot)
            self.released_count += 1
            return True
        return False

    def execute_op(self, op: WireOp) -> Any:
        slot = op.slot
        gen = op.generation
        if slot not in self.states or self.generations[slot] != gen:
            raise ValueError(
                f"Worker {self.worker_id}: stale or invalid slot reference (slot={slot}, generation={gen})"
            )
        state = self.states[slot]

        if op.method=='serving_root_metadata':
            from .serving_protocol import root_metadata
            return root_metadata(state,op.args[0])
        if op.method=='public_terminal_probe':
            from .serving_protocol import public_terminal_probe
            return public_terminal_probe(state,op.args[0],op.args[1])

        if op.method == "decision":
            from app.modules.card_game.rl import recovery_runtime

            side = op.args[0]
            key=(slot,gen,side)
            if key not in self.decision_cache:
                self.decision_cache[key]=recovery_runtime.decision(state,side)
            return self.decision_cache[key]

        elif op.method == "sample_world":
            from app.modules.card_game.rl import information_search, recovery_runtime

            viewer = op.args[0]
            seed = op.args[1]
            sampled = information_search.sample_world(state, viewer, seed, runtime=recovery_runtime)
            return self.allocate_slot(
                sampled,
                role="hypothetical",
                game_id=self.game_ids[slot],
                root_generation=self.root_generations[slot],
                is_clean=False,
            )

        elif op.method == "step":
            from app.modules.card_game.engine.duel_v2.simulation import simulate_action
            from app.modules.card_game.rl import league_rollout

            side = op.args[0]
            action = op.args[1]
            # Must not consume input state in place; simulate_action internally clones state
            next_state = league_rollout.clean(simulate_action(state, side, action))
            return self.allocate_slot(
                next_state,
                role="hypothetical",
                game_id=self.game_ids[slot],
                root_generation=self.root_generations[slot],
                is_clean=True,
            )

        elif op.method == "clean":
            from app.modules.card_game.engine.duel_v2 import acting_side

            if self.is_clean.get(slot, False):
                # Return existing handle without allocating new slot or repeating finalizers
                return WireStateMeta(
                    worker=self.worker_id,
                    slot=slot,
                    generation=gen,
                    game_id=self.game_ids[slot],
                    root_generation=self.root_generations[slot],
                    actor=acting_side(state),
                    phase=state.get("phase", "") if isinstance(state, dict) else getattr(state, "phase", ""),
                    winner=state.get("winner") if isinstance(state, dict) else getattr(state, "winner", None),
                    role=self.roles[slot],
                )
            else:
                from app.modules.card_game.rl import league_rollout

                cleaned = league_rollout.clean(deepcopy(state))
                return self.allocate_slot(
                    cleaned,
                    role=self.roles[slot],
                    game_id=self.game_ids[slot],
                    root_generation=self.root_generations[slot],
                    is_clean=True,
                )

        elif op.method == "value_observation":
            from app.modules.card_game.engine.duel_v2 import observe
            from app.modules.card_game.rl import recovery_runtime

            viewer = op.args[0]
            view = observe(state, viewer, include_previews=False)
            return recovery_runtime.encode(view, [])

        elif op.method == "acting_side":
            from app.modules.card_game.engine.duel_v2 import acting_side

            return acting_side(state)

        elif op.method == "finished":
            if isinstance(state, dict):
                return state.get("phase") == "finished"
            return getattr(state, "phase", None) == "finished"

        elif op.method == "terminal_value":
            viewer = op.args[0]
            winner = state.get("winner") if isinstance(state, dict) else getattr(state, "winner", None)
            if winner not in ("a", "b"):
                return 0.0
            return 1.0 if winner == viewer else -1.0

        else:
            raise ValueError(f"Worker {self.worker_id}: unsupported operation method {op.method}")


def _worker_init(worker_id: int, max_states: int, record_output=None) -> None:
    global _worker_state
    _worker_state = _WorkerState(worker_id=worker_id, max_states=max_states,record_output=record_output)


def _worker_warmup(worker_id: int) -> dict[str, Any]:
    torch_imported = ("torch" in sys.modules) or ("torch.cuda" in sys.modules)
    return {
        "worker": worker_id,
        "pid": os.getpid(),
        "torch_imported": torch_imported,
    }


def _worker_import_batch(
    items: list[tuple[int, Any, str, int]],
    releases: list[tuple[int, int]],
) -> tuple[list[tuple[int, WireStateMeta]], int, int]:
    global _worker_state
    assert _worker_state is not None
    for slot, gen in releases:
        _worker_state.release_slot(slot, gen)
    results: list[tuple[int, WireStateMeta]] = []
    for idx, state, game_id, root_gen in items:
        if _worker_state.record_output is not None:
            from .recording import PhysicalGameRecord
            from pathlib import Path
            _worker_state.physical_records[game_id]=PhysicalGameRecord(state,
                journal_directory=Path(_worker_state.record_output)/'journals'/game_id)
        meta = _worker_state.allocate_slot(
            state, role="real", game_id=game_id, root_generation=root_gen, is_clean=False
        )
        results.append((idx, meta))
    return results, _worker_state.released_count, len(_worker_state.states)


def _worker_export_batch(
    slots_and_gens: list[tuple[int, int, int]],
) -> list[tuple[int, Any]]:
    global _worker_state
    assert _worker_state is not None
    results: list[tuple[int, Any]] = []
    for idx, slot, gen in slots_and_gens:
        if slot not in _worker_state.states or _worker_state.generations[slot] != gen:
            raise ValueError(
                f"Worker {_worker_state.worker_id}: stale or invalid slot reference (slot={slot}, generation={gen})"
            )
        results.append((idx, deepcopy(_worker_state.states[slot])))
    return results


def _worker_release_batch(releases: list[tuple[int, int]]) -> tuple[int, int]:
    global _worker_state
    assert _worker_state is not None
    for slot, gen in releases:
        _worker_state.release_slot(slot, gen)
    return _worker_state.released_count, len(_worker_state.states)


def _worker_commit_action(
    slot: int,
    generation: int,
    side: str,
    action: Any,
    next_root_generation: int,
    releases: list[tuple[int, int]],
    search_stats=None,
) -> WireStateMeta:
    global _worker_state
    assert _worker_state is not None
    for r_slot, r_gen in releases:
        _worker_state.release_slot(r_slot, r_gen)
    if slot not in _worker_state.states or _worker_state.generations[slot] != generation:
        raise ValueError(
            f"Worker {_worker_state.worker_id}: stale or invalid slot reference (slot={slot}, generation={generation})"
        )
    if _worker_state.roles[slot] != "real":
        raise ValueError("Can only commit action on state with role='real'")
    if next_root_generation != _worker_state.root_generations[slot] + 1:
        raise ValueError(
            f"next_root_generation ({next_root_generation}) must be root_generation + 1 ({_worker_state.root_generations[slot] + 1})"
        )
    from app.modules.card_game.engine.duel_v2.flow import apply_action

    source = _worker_state.states[slot]
    # Retain old state in source slot, create new real state with official apply_action
    new_state = apply_action(source, side, action)
    if _worker_state.record_output is not None:
        from app.modules.card_game.rl import recovery_runtime,league_rollout
        decision=_worker_state.decision_cache.get((slot,generation,side))
        if decision is None:decision=recovery_runtime.decision(source,side)
        legal=decision[0]
        if action not in legal:
            from app.modules.card_game.engine.duel_v2 import observe
            legal=[entry['action'] for entry in observe(source,side,include_previews=False)['legal_actions']]
        _worker_state.physical_records[_worker_state.game_ids[slot]].append(
            source,new_state,side,action,legal,search_stats=search_stats)
        new_state=league_rollout.clean(new_state)
    return _worker_state.allocate_slot(
        new_state,
        role="real",
        game_id=_worker_state.game_ids[slot],
        root_generation=next_root_generation,
        is_clean=False,
    )


def _worker_commit_batch(items,releases):
    assert _worker_state is not None
    for slot,generation in releases:_worker_state.release_slot(slot,generation)
    return [(index,_worker_commit_action(slot,gen,side,action,next_gen,[],stats))
            for index,slot,gen,side,action,next_gen,stats in items]


def _worker_finish_batch(items):
    assert _worker_state is not None
    results=[]
    for index,slot,gen,job in items:
        if slot not in _worker_state.states or _worker_state.generations[slot]!=gen:
            raise ValueError('Stale physical finish state')
        gid=_worker_state.game_ids[slot]
        if job.get('id')!=gid or _worker_state.roles[slot]!='real':
            raise ValueError('Finish job belongs to another physical game')
        record=_worker_state.physical_records[gid]
        result=record.finish(_worker_state.states[slot],job,_worker_state.record_output,
                             public_replay=bool(job.get('replay')))
        del _worker_state.physical_records[gid]
        results.append((index,result))
    return results


def _worker_ops_batch(
    ops: list[WireOp],
    releases: list[tuple[int, int]],
) -> WorkerBatchResult:
    global _worker_state
    assert _worker_state is not None
    t0 = time.perf_counter_ns()
    for slot, gen in releases:
        _worker_state.release_slot(slot, gen)
    results: list[tuple[str, Any]] = []
    methods_ns: dict[str, int] = {}
    for op in ops:
        if op.deadline is not None and time.monotonic()>=op.deadline:
            results.append((op.id,RequestExpired(op.id)));continue
        op_t0 = time.perf_counter_ns()
        val = _worker_state.execute_op(op)
        op_t1 = time.perf_counter_ns()
        methods_ns[op.method] = methods_ns.get(op.method, 0) + (op_t1 - op_t0)
        if op.deadline is not None and time.monotonic()>=op.deadline:
            if isinstance(val,WireStateMeta) and (val.slot,val.generation)!=(op.slot,op.generation):
                _worker_state.release_slot(val.slot,val.generation)
            val=RequestExpired(op.id,True)
        results.append((op.id, val))
    t1 = time.perf_counter_ns()
    return WorkerBatchResult(
        worker_id=_worker_state.worker_id,
        pid=os.getpid(),
        results=results,
        released_count=_worker_state.released_count,
        live_states=len(_worker_state.states),
        execution_ns=t1 - t0,
        methods_ns=methods_ns,
    )


def _enqueue_release(q: queue.Queue, key: tuple[int, int, int]) -> None:
    try:
        q.put_nowait(key)
    except Exception:
        pass


# --- Resident Rule Backend (Parent Process) ---


class ResidentRuleBackend:
    """Explicit resident CPU worker bulk resolver implementing official Duel V2 rules.

    Maintains bounded worker process capacity, handles garbage-collected StateRef release,
    and resolves operations in batches across sticky worker processes.
    """

    backend_kind = "python_resident_process"
    supports_gpu = False

    def __init__(
        self,
        *,
        workers: int = 16,
        max_games: int = 1600,
        max_states_per_worker: Optional[int] = None,
        operation_timeout: float = 30.0,
        mp_context: Union[str, multiprocessing.context.BaseContext] = "spawn",
        runtime: Optional[Any] = None,
        record_output=None,
    ) -> None:
        if type(workers) is not int or type(workers) is bool or not 1 <= workers <= 64:
            raise ValueError(f"workers must be a positive integer, got {workers!r}")
        if type(max_games) is not int or type(max_games) is bool or max_games < 1:
            raise ValueError(f"max_games must be a positive integer, got {max_games!r}")
        if max_states_per_worker is None:
            max_states_per_worker = max(5000, (max_games // workers + 1) * 64)
        elif (
            type(max_states_per_worker) is not int
            or type(max_states_per_worker) is bool
            or max_states_per_worker < 1
        ):
            raise ValueError(f"max_states_per_worker must be a positive integer, got {max_states_per_worker!r}")

        if (
            type(operation_timeout) not in (int, float)
            or type(operation_timeout) is bool
            or not math.isfinite(operation_timeout)
            or operation_timeout <= 0
        ):
            raise ValueError(f"operation_timeout must be a positive finite number, got {operation_timeout!r}")

        self._owner_thread=threading.get_ident()
        self.workers = workers
        self.max_games = max_games
        self.max_states_per_worker = max_states_per_worker
        self.operation_timeout = float(operation_timeout)
        self.runtime = runtime
        if runtime is not None and getattr(runtime,'__name__',None)!='app.modules.card_game.rl.recovery_runtime':
            raise ValueError('Resident rules bind the explicit recovery runtime')
        from pathlib import Path
        self.record_output=str(Path(record_output).resolve()) if record_output is not None else None
        self.backend_id = uuid4().hex

        self._ctx = multiprocessing.get_context(mp_context) if isinstance(mp_context, str) else mp_context
        self._ref_cache: weakref.WeakValueDictionary[tuple[int, int, int], StateRef] = (
            weakref.WeakValueDictionary()
        )
        self._live_slots: set[tuple[int, int, int]] = set()
        self._release_queue: queue.Queue[tuple[int, int, int]] = queue.Queue()

        self._closed = False
        self._poisoned = False

        self._method_counts: collections.Counter[str] = collections.Counter()
        self._worker_execution_ns = 0
        self._worker_methods_ns: collections.Counter[str] = collections.Counter()
        self._parent_rpc_wall_ns = 0
        self._known_clean: set[tuple[int,int,int]] = set()
        self._inline_counts: collections.Counter[str] = collections.Counter()
        self._parent_queue_ns = 0
        self._released_count = 0
        self._peak_states = 0

        self._game_to_worker: dict[str, int] = {}
        self._latest_real_generation: dict[str,int] = {}
        self._next_worker_idx = 0
        self._owned_processes: list[multiprocessing.Process] = []

        self._executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._ctx,
                initializer=_worker_init,
                initargs=(i, self.max_states_per_worker, self.record_output),
            )
            for i in range(self.workers)
        ]

        # Explicit startup verification
        self.warmup()

    def __enter__(self) -> ResidentRuleBackend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _check_healthy(self) -> None:
        if threading.get_ident()!=self._owner_thread:raise RuntimeError("Resident rules require their owning scheduler thread")
        if self._closed:
            raise RuntimeError("Backend is closed")
        if self._poisoned:
            raise RuntimeError("Backend is poisoned from a previous failure and cannot be reused")

    def _refresh_owned_processes(self) -> None:
        for ex in self._executors:
            processes = getattr(ex, "_processes", None)
            if processes:
                for p in processes.values():
                    if p not in self._owned_processes:
                        self._owned_processes.append(p)

    @property
    def owned_processes(self) -> list[multiprocessing.Process]:
        self._refresh_owned_processes()
        return list(self._owned_processes)

    def close(self) -> None:
        """Stop owned workers while each executor manager retains sole join ownership."""
        self._closed=True;self._refresh_owned_processes();procs=list(self._owned_processes)
        managers=[]
        for executor in self._executors:
            manager=getattr(executor,'_executor_manager_thread',None)
            if manager is not None:managers.append(manager)
            call_queue=getattr(executor,'_call_queue',None)
            if call_queue is not None:call_queue.cancel_join_thread()
            executor.shutdown(wait=False,cancel_futures=True)
        def alive(proc):
            try:return proc.is_alive()
            except ValueError:return False
        for proc in procs:
            if alive(proc):proc.terminate()
        end=time.monotonic()+2.
        for manager in managers:manager.join(timeout=max(0.,end-time.monotonic()))
        for proc in procs:
            if alive(proc):proc.kill()
        end=time.monotonic()+2.
        for manager in managers:manager.join(timeout=max(0.,end-time.monotonic()))
        # ProcessPoolExecutor's manager is the sole waitpid/join owner. Concurrent
        # Process.join here races its reaper and can leave a stale exit status.
        self._live_slots.clear();self._ref_cache.clear()
        if any(alive(proc) for proc in procs) or any(manager.is_alive() for manager in managers):
            raise RuntimeError('An owned rule worker or its manager failed to stop')

    def _validate_ref(self, ref: Any) -> None:
        if not isinstance(ref, StateRef):
            raise TypeError(f"Expected StateRef instance, got {type(ref)}")
        if type(ref.slot) is not int or type(ref.slot) is bool:
            raise TypeError("slot must be an integer, not boolean")
        if type(ref.generation) is not int or type(ref.generation) is bool:
            raise TypeError("generation must be an integer, not boolean")
        if type(ref.worker) is not int or type(ref.worker) is bool:
            raise TypeError("worker must be an integer, not boolean")
        if ref.backend_id != self.backend_id:
            raise ValueError(
                f"Cross-backend reference rejected: ref belongs to backend {ref.backend_id}, this backend is {self.backend_id}"
            )
        if not 0 <= ref.worker < self.workers:
            raise ValueError('State worker is outside this backend')
        if (ref.worker, ref.slot, ref.generation) not in self._live_slots:
            raise ValueError(
                f"Stale or released state reference: (worker={ref.worker}, slot={ref.slot}, generation={ref.generation})"
            )

        canonical=self._ref_cache.get((ref.worker,ref.slot,ref.generation))
        if canonical is None or canonical != ref:
            raise ValueError('State metadata does not match its issued reference')

    def _await_futures(self, futures: dict[concurrent.futures.Future, Any]) -> dict[Any, Any]:
        deadline = time.monotonic() + self.operation_timeout
        self._refresh_owned_processes()
        remaining = max(0.0, deadline - time.monotonic())
        done, not_done = concurrent.futures.wait(futures.keys(), timeout=remaining)
        if not_done:
            self._poisoned = True
            self.close()
            raise TimeoutError(f"Operation timed out after {self.operation_timeout}s")

        results: dict[Any, Any] = {}
        for fut, key in futures.items():
            try:
                results[key] = fut.result()
            except Exception as e:
                self._poisoned = True
                self.close()
                raise RuntimeError(f"Worker failure: {e}") from e
        return results

    def _get_pending_releases(self) -> dict[int, list[tuple[int, int]]]:
        releases_by_worker: dict[int, list[tuple[int, int]]] = {}
        while True:
            try:
                worker, slot, gen = self._release_queue.get_nowait()
            except queue.Empty:
                break
            if (worker, slot, gen) in self._live_slots:
                self._live_slots.discard((worker, slot, gen))
                self._known_clean.discard((worker, slot, gen))
                self._ref_cache.pop((worker, slot, gen), None)
                self._released_count += 1
                releases_by_worker.setdefault(worker, []).append((slot, gen))
        return releases_by_worker

    def _flush_releases(self) -> None:
        if self._closed or self._poisoned:
            return
        releases_by_worker = self._get_pending_releases()
        if not releases_by_worker:
            return
        futures = {}
        for w, rels in releases_by_worker.items():
            fut = self._executors[w].submit(_worker_release_batch, rels)
            futures[fut] = w
        self._await_futures(futures)

    def _register_wire_meta(self, meta: WireStateMeta) -> StateRef:
        key = (meta.worker, meta.slot, meta.generation)
        existing = self._ref_cache.get(key)
        if existing is not None:
            return existing
        ref = StateRef(
            backend_id=self.backend_id,
            worker=meta.worker,
            slot=meta.slot,
            generation=meta.generation,
            game_id=meta.game_id,
            root_generation=meta.root_generation,
            actor=meta.actor,
            phase=meta.phase,
            winner=meta.winner,
            role=meta.role,
        )
        self._live_slots.add(key)
        self._ref_cache[key] = ref
        weakref.finalize(ref, _enqueue_release, self._release_queue, key)
        if len(self._live_slots) > self._peak_states:
            self._peak_states = len(self._live_slots)
        return ref

    def warmup(self) -> list[dict[str, Any]]:
        """Verify worker health, retrieve PIDs, and ensure Torch is never imported in workers."""
        self._check_healthy()
        futures = {}
        for w in range(self.workers):
            fut = self._executors[w].submit(_worker_warmup, w)
            futures[fut] = w
        results = self._await_futures(futures)
        self._refresh_owned_processes()
        info_list = [results[w] for w in range(self.workers)]
        for item in info_list:
            if item.get("torch_imported", True):
                self._poisoned = True
                self.close()
                raise RuntimeError(f"Worker {item.get('worker')} has Torch imported!")
        return info_list

    def stats(self) -> dict[str, Any]:
        """Return runtime statistics on method executions, states, capacity, and latency."""
        self._check_healthy()
        self._flush_releases()
        self._refresh_owned_processes()
        active_processes = sum(1 for p in self._owned_processes if p.is_alive())
        return {
            "backend_id": self.backend_id,
            "backend_kind": self.backend_kind,
            "supports_gpu": self.supports_gpu,
            "workers": self.workers,
            "max_games": self.max_games,
            "max_states_per_worker": self.max_states_per_worker,
            "max_capacity": self.workers * self.max_states_per_worker,
            "active_processes": active_processes,
            "live_states": len(self._live_slots),
            "peak_states": self._peak_states,
            "released_states": self._released_count,
            "methods": dict(self._method_counts),
            "inline_operations":dict(self._inline_counts),
            "worker_execution_ns": self._worker_execution_ns,
            "worker_methods_ns": dict(self._worker_methods_ns),
            "parent_rpc_wall_ns": self._parent_rpc_wall_ns,
            "parent_queue_ns": None,  # not yet measured; never invent zero queue wait
        }

    def import_states(
        self,
        states: Sequence[Any],
        game_ids: Sequence[str],
        root_generations: Sequence[int],
    ) -> list[StateRef]:
        """Import full game states into worker resident slots once and return StateRef handles."""
        self._check_healthy()
        self._flush_releases()
        if len(states) != len(game_ids) or len(states) != len(root_generations):
            raise ValueError(
                f"Lengths mismatch: states ({len(states)}), game_ids ({len(game_ids)}), root_generations ({len(root_generations)})"
            )
        if not states:
            return []
        for i, (gid, rgen) in enumerate(zip(game_ids, root_generations)):
            if not isinstance(gid, str) or not gid:
                raise ValueError(f"game_ids[{i}] must be a non-empty string, got {gid!r}")
            if type(rgen) is not int or type(rgen) is bool or rgen < 0:
                raise TypeError(f"root_generations[{i}] must be a non-negative integer, got {rgen!r}")
            if self.record_output is not None and any(ch not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in gid):
                raise ValueError('Recorded game id must be a safe filename')

        if len(set(game_ids))!=len(game_ids) or any(gid in self._game_to_worker for gid in game_ids):
            raise ValueError('A game can be imported only once in this backend')
        if len(self._game_to_worker)+len(game_ids)>self.max_games:
            raise OverflowError('Real game admission capacity exceeded')
        worker_items: dict[int, list[tuple[int, Any, str, int]]] = {w: [] for w in range(self.workers)}
        for idx, (st, gid, rgen) in enumerate(zip(states, game_ids, root_generations)):
            if gid not in self._game_to_worker:
                self._game_to_worker[gid] = self._next_worker_idx
                self._next_worker_idx = (self._next_worker_idx + 1) % self.workers
            worker_id = self._game_to_worker[gid]
            worker_items[worker_id].append((idx, st, gid, rgen))

        releases_by_worker = self._get_pending_releases()
        futures = {}
        for w, items in worker_items.items():
            if items or releases_by_worker.get(w):
                fut = self._executors[w].submit(_worker_import_batch, items, releases_by_worker.get(w, []))
                futures[fut] = w

        worker_results = self._await_futures(futures)
        refs: list[Optional[StateRef]] = [None] * len(states)
        for w, (results, released_cnt, live_states) in worker_results.items():
            for idx, meta in results:
                ref = self._register_wire_meta(meta)
                self._latest_real_generation[ref.game_id]=ref.root_generation
                refs[idx] = ref
        return [r for r in refs if r is not None]

    def export_states(self, refs: Sequence[StateRef]) -> list[Any]:
        """Export independent deepcopy Python state dictionaries from workers."""
        self._check_healthy()
        self._flush_releases()
        if not refs:
            return []
        worker_items: dict[int, list[tuple[int, int, int]]] = {w: [] for w in range(self.workers)}
        for idx, ref in enumerate(refs):
            self._validate_ref(ref)
            worker_items[ref.worker].append((idx, ref.slot, ref.generation))

        futures = {}
        for w, items in worker_items.items():
            if items:
                fut = self._executors[w].submit(_worker_export_batch, items)
                futures[fut] = w

        worker_results = self._await_futures(futures)
        exported: list[Optional[Any]] = [None] * len(refs)
        for w, results in worker_results.items():
            for idx, st in results:
                exported[idx] = st
        return [s for s in exported if s is not None]

    def release(self, refs: Sequence[StateRef]) -> None:
        """Explicitly release state reference handles and free worker slots."""
        self._check_healthy()
        releases_by_worker: dict[int, list[tuple[int, int]]] = {}
        keys=[]
        for ref in refs:
            self._validate_ref(ref)
            keys.append((ref.worker,ref.slot,ref.generation))
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate state release')
        for ref in refs:
            key = (ref.worker, ref.slot, ref.generation)
            self._live_slots.discard(key)
            self._known_clean.discard(key)
            self._ref_cache.pop(key, None)
            self._released_count += 1
            releases_by_worker.setdefault(ref.worker, []).append((ref.slot, ref.generation))
        if releases_by_worker:
            futures = {}
            for w, rels in releases_by_worker.items():
                fut = self._executors[w].submit(_worker_release_batch, rels)
                futures[fut] = w
            self._await_futures(futures)

    def commit_action(self,ref,side,action,next_root_generation,*,search_stats=None):
        return self.commit_actions([(ref,side,action,next_root_generation,search_stats)])[0]

    def commit_actions(self,commits):
        """Apply one physical operation per game in worker-sized batches."""
        self._check_healthy();self._flush_releases()
        commits=list(commits);seen=set();groups={w:[] for w in range(self.workers)}
        for index,item in enumerate(commits):
            if len(item)!=5:raise ValueError('Physical commit tuple requires five fields')
            ref,side,action,next_gen,stats=item;self._validate_ref(ref)
            if ref.role!='real' or ref.game_id in seen:raise ValueError('One real commit per game required')
            seen.add(ref.game_id)
            if ref.root_generation!=self._latest_real_generation.get(ref.game_id):
                raise ValueError('Physical state has already advanced')
            if type(next_gen) is not int:raise TypeError('Integer physical generation required')
            if next_gen!=ref.root_generation+1:
                raise ValueError('Physical generation must advance by one')
            if side not in ('a','b') or side!=ref.actor or not isinstance(action,dict):
                raise ValueError('Physical action requires its current decision owner')
            groups[ref.worker].append((index,ref.slot,ref.generation,side,action,next_gen,stats))
        releases=self._get_pending_releases();futures={}
        for worker,items in groups.items():
            if items or releases.get(worker):
                futures[self._executors[worker].submit(_worker_commit_batch,items,releases.get(worker,[]))]=worker
        rows=self._await_futures(futures);results=[None]*len(commits)
        for worker,items in rows.items():
            for index,meta in items:
                ref=self._register_wire_meta(meta);results[index]=ref
                self._latest_real_generation[ref.game_id]=ref.root_generation
        if any(ref is None for ref in results):raise RuntimeError('Physical commit result was lost')
        return results

    def finish_games(self,refs,jobs):
        """Write actual private records and optionally verified public replays."""
        self._check_healthy();self._flush_releases()
        if self.record_output is None:raise ValueError('Physical recording was not configured')
        if len(refs)!=len(jobs):raise ValueError('Finish job count mismatch')
        groups={w:[] for w in range(self.workers)};seen=set()
        for index,(ref,job) in enumerate(zip(refs,jobs)):
            self._validate_ref(ref)
            if (ref.role!='real' or job.get('id')!=ref.game_id or ref.game_id in seen or
                self._latest_real_generation.get(ref.game_id)!=ref.root_generation):
                raise ValueError('Finish requires each current real game once')
            seen.add(ref.game_id);groups[ref.worker].append((index,ref.slot,ref.generation,job))
        futures={self._executors[w].submit(_worker_finish_batch,items):w
                 for w,items in groups.items() if items}
        results=[None]*len(refs)
        for worker,items in self._await_futures(futures).items():
            for index,row in items:results[index]=row
        if any(row is None for row in results):raise RuntimeError('Physical finish result was lost')
        return results

    def resolve_operations(
        self, queries: Sequence[OperationQuery]
    ) -> list[OperationResponse]:
        """Resolve a batch of cooperative search operations without inter-process state copying."""
        self._check_healthy()
        self._flush_releases()
        if not queries:
            return []

        query_ids = set()
        for q in queries:
            if not isinstance(q, OperationQuery):
                raise TypeError(f"Expected OperationQuery, got {type(q)}")
            if q.id in query_ids:
                raise ValueError(f"Duplicate query identity: {q.id}")
            if q.backend is not self:raise ValueError('Operation belongs to another backend')
            query_ids.add(q.id)

        methods_by_id={q.id:q.method for q in queries}
        arities={'acting_side':1,'finished':1,'terminal_value':2,'decision':2,
                 'sample_world':3,'step':3,'clean':1,'value_observation':2,
                 'serving_root_metadata':2,'public_terminal_probe':3}
        for q in queries:
            if q.method not in arities or len(q.args)!=arities[q.method] or q.kwargs:
                raise ValueError('Unsupported rule operation signature')
            if q.deadline is not None and (isinstance(q.deadline,bool) or
                not isinstance(q.deadline,(int,float)) or not math.isfinite(q.deadline)):
                raise ValueError('Finite monotonic rule operation deadline required')
            self._validate_ref(q.args[0])
            if len(q.args)>1 and q.args[1] not in ('a','b'):
                raise ValueError('Rule operation requires side a or b')
            if q.method=='sample_world' and type(q.args[2]) is not int:
                raise TypeError('Integer world seed required')

        responses: dict[str, OperationResponse] = {}
        worker_ops: dict[int, list[WireOp]] = {w: [] for w in range(self.workers)}

        for q in queries:
            if q.method not in PROTOCOL_METHODS:
                raise ValueError(f"Unknown operation method: {q.method}")
            if not q.args:
                raise ValueError(f"Operation {q.method} requires at least one argument (StateRef)")
            ref = q.args[0]
            self._validate_ref(ref)

            # Metadata queries are resolved directly in parent without worker roundtrip
            if q.method == "acting_side":
                self._method_counts[q.method] += 1
                responses[q.id] = OperationResponse(q.id, ref.actor)
            elif q.method == "finished":
                self._method_counts[q.method] += 1
                responses[q.id] = OperationResponse(q.id, ref.phase == "finished")
            elif q.method == "terminal_value":
                self._method_counts[q.method] += 1
                if len(q.args) < 2:
                    raise ValueError("terminal_value requires viewer argument")
                viewer = q.args[1]
                winner = ref.winner
                if winner not in ("a", "b"):
                    val = 0.0
                else:
                    val = 1.0 if winner == viewer else -1.0
                responses[q.id] = OperationResponse(q.id, val)
            else:
                self._method_counts[q.method] += 1
                kw = dict(q.kwargs) if isinstance(q.kwargs, (tuple, list)) else (q.kwargs or {})
                wire_op = WireOp(
                    id=q.id,
                    method=q.method,
                    slot=ref.slot,
                    generation=ref.generation,
                    args=tuple(q.args[1:]),
                    kwargs=tuple(kw.items()),
                    deadline=q.deadline,
                )
                worker_ops[ref.worker].append(wire_op)

        releases_by_worker = self._get_pending_releases()
        futures = {}
        t_rpc_start = time.perf_counter_ns()
        for w, ops in worker_ops.items():
            if ops or releases_by_worker.get(w):
                fut = self._executors[w].submit(_worker_ops_batch, ops, releases_by_worker.get(w, []))
                futures[fut] = w

        if futures:
            worker_results = self._await_futures(futures)
            t_rpc_end = time.perf_counter_ns()
            self._parent_rpc_wall_ns += t_rpc_end - t_rpc_start

            for w, batch_res in worker_results.items():
                self._worker_execution_ns += batch_res.execution_ns
                for meth, ns in batch_res.methods_ns.items():
                    self._worker_methods_ns[meth] += ns
                for op_id, res_val in batch_res.results:
                    if isinstance(res_val, WireStateMeta):
                        out_ref = self._register_wire_meta(res_val)
                        method=methods_by_id[op_id]
                        if method in ('step','clean'):
                            self._known_clean.add((out_ref.worker,out_ref.slot,out_ref.generation))
                        responses[op_id] = OperationResponse(op_id, out_ref)
                    else:
                        responses[op_id] = OperationResponse(op_id, res_val)

        return [responses[q.id] for q in queries]

    def inline_operation(self,method,args):
        """Resolve only immutable metadata and proven clean handles, with no RPC.

        No rule execution, encoding, sampling or neural work can use this path.
        Canonical reference validation remains identical to bulk dispatch.
        """
        if method not in ('acting_side','finished','terminal_value','clean'):return False,None
        self._check_healthy()
        expected=2 if method=='terminal_value' else 1
        if len(args)!=expected:raise ValueError('Inline metadata arity mismatch')
        ref=args[0];self._validate_ref(ref)
        if method=='terminal_value' and args[1] not in ('a','b'):raise ValueError('Invalid inline observer')
        if method=='clean' and (ref.worker,ref.slot,ref.generation) not in self._known_clean:return False,None
        if method=='acting_side':value=ref.actor
        elif method=='finished':value=ref.phase=='finished'
        elif method=='terminal_value':value=0. if ref.winner not in ('a','b') else (1. if ref.winner==args[1] else -1.)
        else:value=ref
        self._method_counts[method]+=1;self._inline_counts[method]+=1
        return True,value

    # --- Convenience single-operation methods matching SearchBackend / CPUOracleBackend ---

    def acting_side(self, ref: StateRef) -> Optional[str]:
        self._validate_ref(ref)
        return ref.actor

    def finished(self, ref: StateRef) -> bool:
        self._validate_ref(ref)
        return ref.phase == "finished"

    def terminal_value(self, ref: StateRef, viewer: str) -> float:
        self._validate_ref(ref)
        winner = ref.winner
        if winner not in ("a", "b"):
            return 0.0
        return 1.0 if winner == viewer else -1.0

    def decision(
        self, ref: StateRef, side: str
    ) -> Tuple[List[Any], Tuple[np.ndarray, np.ndarray]]:
        q = OperationQuery(uuid4().hex, self, "decision", (ref, side), ())
        return self.resolve_operations([q])[0].value  # type: ignore

    def sample_world(self, ref: StateRef, viewer: str, seed: int) -> StateRef:
        q = OperationQuery(uuid4().hex, self, "sample_world", (ref, viewer, seed), ())
        return self.resolve_operations([q])[0].value  # type: ignore

    def step(self, ref: StateRef, side: str, action: Any) -> StateRef:
        q = OperationQuery(uuid4().hex, self, "step", (ref, side, action), ())
        return self.resolve_operations([q])[0].value  # type: ignore

    def clean(self, ref: StateRef) -> StateRef:
        q = OperationQuery(uuid4().hex, self, "clean", (ref,), ())
        return self.resolve_operations([q])[0].value  # type: ignore

    def value_observation(
        self, ref: StateRef, viewer: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        q = OperationQuery(uuid4().hex, self, "value_observation", (ref, viewer), ())
        return self.resolve_operations([q])[0].value  # type: ignore
