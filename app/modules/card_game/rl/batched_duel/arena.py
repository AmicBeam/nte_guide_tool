"""Batched duel CPU state arena module.

Provides true contiguous int32 numpy array storage for batched real games
and bounded hypothetical workspaces on CPU.
This module does not execute game rules, generate observations, load neural models,
or initialize CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Tuple
from uuid import uuid4
import numpy as np


class ArenaError(Exception):
    """Base exception for state arena operations."""
    pass


class ArenaCapacityError(ArenaError):
    """Raised when game or workspace capacity has been exhausted."""
    pass


class ArenaInvalidRefError(ArenaError):
    """Raised when a reference is invalid, expired, from another arena, or out of bounds."""
    pass


class ArenaBusyError(ArenaError):
    """Raised when a game slot cannot be freed because child workspaces are still active."""
    pass


class ArenaStateShapeError(ArenaError, ValueError):
    """Raised when state array shape does not match arena layout state_width."""
    pass


class ArenaStateDtypeError(ArenaError, TypeError):
    """Raised when state array dtype is not np.int32."""
    pass


@dataclass(frozen=True)
class ArenaLayout:
    """Frozen layout specifications for a CpuStateArena.

    Attributes:
        game_capacity: Maximum number of simultaneous real game slots (> 0).
        workspace_capacity: Maximum number of simultaneous hypothetical workspace slots (> 0).
        state_width: Width of each 1D int32 state vector (> 0).
    """
    game_capacity: int
    workspace_capacity: int
    state_width: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("game_capacity", self.game_capacity),
            ("workspace_capacity", self.workspace_capacity),
            ("state_width", self.state_width),
        ):
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be an integer, got {type(value).__name__} (bool is rejected)"
                )
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be a positive integer > 0, got {value}"
                )


@dataclass(frozen=True)
class GameRef:
    """Immutable reference to an allocated real game slot in an arena.

    Does not carry numpy arrays, preventing array hash/equality comparison traps.
    """
    arena_id: int
    slot: int
    generation: int


@dataclass(frozen=True)
class WorkspaceRef:
    """Immutable reference to an allocated hypothetical workspace slot in an arena.

    Maintains owner association with the parent real game slot.
    Does not carry numpy arrays.
    """
    arena_id: int
    slot: int
    generation: int
    owner_game_slot: int
    owner_game_generation: int


class CpuStateArena:
    """CPU contiguous int32 state storage for real games and hypothetical workspaces.

    Manages fixed-capacity memory rows using contiguous NumPy int32 arrays.
    Provides slot lifecycle tracking with generational ABA protection,
    cross-arena isolation, independent copy semantics, and explicit memory reporting.
    One scheduler owns each arena; operations are not thread-safe.
    """

    def __init__(self, layout: ArenaLayout) -> None:
        if not isinstance(layout, ArenaLayout):
            raise TypeError(f"Expected ArenaLayout instance, got {type(layout).__name__}")

        self._layout = layout
        # References may cross worker boundaries. A process-local counter can
        # collide with another worker's first arena; identity is never a policy
        # feature or part of the game's random stream.
        self._arena_id: int = uuid4().int

        # Allocate contiguous int32 rows
        self._game_rows = np.zeros(
            (layout.game_capacity, layout.state_width),
            dtype=np.int32,
            order="C",
        )
        self._workspace_rows = np.zeros(
            (layout.workspace_capacity, layout.state_width),
            dtype=np.int32,
            order="C",
        )

        if not self._game_rows.flags.c_contiguous:
            raise RuntimeError("Allocated game_rows is not C-contiguous")
        if not self._workspace_rows.flags.c_contiguous:
            raise RuntimeError("Allocated workspace_rows is not C-contiguous")

        # Game slot tracking
        self._game_allocated = [False] * layout.game_capacity
        self._game_generations = [1] * layout.game_capacity
        self._game_active_workspaces: list[set[int]] = [
            set() for _ in range(layout.game_capacity)
        ]
        self._active_game_count = 0
        self._free_games = list(range(layout.game_capacity))

        # Workspace slot tracking
        self._workspace_allocated = [False] * layout.workspace_capacity
        self._workspace_generations = [1] * layout.workspace_capacity
        self._workspace_owners: list[Tuple[int, int] | None] = [
            None
        ] * layout.workspace_capacity
        self._active_workspace_count = 0
        self._free_workspaces = list(range(layout.workspace_capacity))

    @property
    def layout(self) -> ArenaLayout:
        """Return the layout parameters for this arena."""
        return self._layout

    @property
    def arena_id(self) -> int:
        """Return the unique integer identity of this arena."""
        return self._arena_id

    @property
    def allocated_bytes(self) -> int:
        """Report total memory bytes consumed by arena storage arrays."""
        return int(self._game_rows.nbytes + self._workspace_rows.nbytes)

    @property
    def active_game_count(self) -> int:
        """Return the number of currently active real game slots."""
        return self._active_game_count

    @property
    def active_workspace_count(self) -> int:
        """Return the number of currently active workspace slots."""
        return self._active_workspace_count

    def _validate_state(self, state: np.ndarray) -> None:
        """Strictly validate input state array dtype, ndim, and width.

        Rejects non-numpy objects, non-int32 dtypes (preventing silent truncation
        of int64/float), and incorrect dimensions.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(f"State must be a numpy.ndarray, got {type(state).__name__}")
        if state.dtype != np.int32:
            raise ArenaStateDtypeError(
                f"State dtype must be int32, got {state.dtype} "
                "(silent truncation of int64/float rejected)"
            )
        if state.ndim != 1:
            raise ArenaStateShapeError(
                f"State must be a 1D array of shape ({self._layout.state_width},), "
                f"got ndim={state.ndim} shape={state.shape}"
            )
        if state.shape[0] != self._layout.state_width:
            raise ArenaStateShapeError(
                f"State width mismatch: expected {self._layout.state_width}, "
                f"got {state.shape[0]}"
            )

    def _validate_game_ref(self, ref: GameRef) -> None:
        """Validate game reference identity, bounds, allocation state, and generation."""
        if not isinstance(ref, GameRef):
            raise TypeError(f"Expected GameRef, got {type(ref).__name__}")
        if any(type(value) is not int for value in (ref.arena_id, ref.slot, ref.generation)):
            raise ArenaInvalidRefError('Game reference fields must be integers, not bool/float')
        if ref.arena_id != self._arena_id:
            raise ArenaInvalidRefError(
                f"Cross-arena reference rejected: ref belongs to arena {ref.arena_id}, "
                f"current arena is {self._arena_id}"
            )
        if ref.slot < 0 or ref.slot >= self._layout.game_capacity:
            raise ArenaInvalidRefError(
                f"Game slot {ref.slot} out of range [0, {self._layout.game_capacity})"
            )
        if not self._game_allocated[ref.slot]:
            raise ArenaInvalidRefError(f"Game slot {ref.slot} is currently unallocated")
        if self._game_generations[ref.slot] != ref.generation:
            raise ArenaInvalidRefError(
                f"Expired GameRef: slot {ref.slot} generation is "
                f"{self._game_generations[ref.slot]}, ref generation is {ref.generation}"
            )

    def _validate_workspace_ref(self, ref: WorkspaceRef) -> None:
        """Validate workspace reference identity, bounds, generation, and owner binding."""
        if not isinstance(ref, WorkspaceRef):
            raise TypeError(f"Expected WorkspaceRef, got {type(ref).__name__}")
        if any(type(value) is not int for value in (
            ref.arena_id, ref.slot, ref.generation,
            ref.owner_game_slot, ref.owner_game_generation,
        )):
            raise ArenaInvalidRefError('Workspace reference fields must be integers, not bool/float')
        if ref.arena_id != self._arena_id:
            raise ArenaInvalidRefError(
                f"Cross-arena reference rejected: ref belongs to arena {ref.arena_id}, "
                f"current arena is {self._arena_id}"
            )
        if ref.slot < 0 or ref.slot >= self._layout.workspace_capacity:
            raise ArenaInvalidRefError(
                f"Workspace slot {ref.slot} out of range [0, {self._layout.workspace_capacity})"
            )
        if not self._workspace_allocated[ref.slot]:
            raise ArenaInvalidRefError(
                f"Workspace slot {ref.slot} is currently unallocated"
            )
        if self._workspace_generations[ref.slot] != ref.generation:
            raise ArenaInvalidRefError(
                f"Expired WorkspaceRef: slot {ref.slot} generation is "
                f"{self._workspace_generations[ref.slot]}, ref generation is {ref.generation}"
            )
        expected_owner = (ref.owner_game_slot, ref.owner_game_generation)
        if self._workspace_owners[ref.slot] != expected_owner:
            raise ArenaInvalidRefError(
                f"Workspace slot {ref.slot} owner mismatch: recorded "
                f"{self._workspace_owners[ref.slot]}, ref has {expected_owner}"
            )

    def allocate_game(self, initial_state: np.ndarray) -> GameRef:
        """Allocate an idle real game slot and copy initial_state into storage.

        Input array is defensively copied; modifying initial_state afterwards
        does not alter arena contents.
        Deterministic allocation selects the lowest available slot index.
        """
        self._validate_state(initial_state)

        if not self._free_games:
            raise ArenaCapacityError(
                f"Real game capacity ({self._layout.game_capacity}) exhausted"
            )
        target_slot = heapq.heappop(self._free_games)

        # Copy data into arena array (slice assignment copies values)
        self._game_rows[target_slot, :] = initial_state
        self._game_allocated[target_slot] = True
        self._game_active_workspaces[target_slot].clear()
        self._active_game_count += 1

        gen = self._game_generations[target_slot]
        return GameRef(arena_id=self._arena_id, slot=target_slot, generation=gen)

    def read_game(self, ref: GameRef) -> np.ndarray:
        """Read state from real game slot.

        Returns an independent copy to prevent external code from mutating
        internal storage or holding onto mutable views after slot release.
        """
        self._validate_game_ref(ref)
        return self._game_rows[ref.slot].copy()

    def write_game(self, ref: GameRef, state: np.ndarray) -> None:
        """Write new state into an allocated real game slot.

        Input array is defensively copied into arena storage.
        """
        self._validate_game_ref(ref)
        self._validate_state(state)
        self._game_rows[ref.slot, :] = state

    def free_game(self, ref: GameRef) -> None:
        """Free an allocated real game slot.

        Rejects release if any child workspace is still active.
        Advances slot generation to invalidate the released ref and prevent ABA reuse errors.
        """
        self._validate_game_ref(ref)

        active_ws = self._game_active_workspaces[ref.slot]
        if len(active_ws) > 0:
            raise ArenaBusyError(
                f"Cannot free game slot {ref.slot}: {len(active_ws)} "
                f"workspace(s) still active ({sorted(active_ws)})"
            )

        self._game_allocated[ref.slot] = False
        self._game_generations[ref.slot] += 1
        self._game_rows[ref.slot, :].fill(0)
        self._active_game_count -= 1
        heapq.heappush(self._free_games, ref.slot)

    def fork_workspace(self, game_ref: GameRef) -> WorkspaceRef:
        """Fork an allocated real game slot into an idle workspace slot.

        Uses NumPy array copying (slice assignment), avoiding recursive Python deepcopy.
        Changes made to workspace do not affect the source game slot and vice versa.
        Deterministic allocation selects the lowest available workspace slot.
        """
        self._validate_game_ref(game_ref)

        if not self._free_workspaces:
            raise ArenaCapacityError(
                f"Workspace capacity ({self._layout.workspace_capacity}) exhausted"
            )
        target_slot = heapq.heappop(self._free_workspaces)

        # Fast NumPy row copy
        self._workspace_rows[target_slot, :] = self._game_rows[game_ref.slot]
        self._workspace_allocated[target_slot] = True
        self._workspace_owners[target_slot] = (game_ref.slot, game_ref.generation)
        self._game_active_workspaces[game_ref.slot].add(target_slot)
        self._active_workspace_count += 1

        gen = self._workspace_generations[target_slot]
        return WorkspaceRef(
            arena_id=self._arena_id,
            slot=target_slot,
            generation=gen,
            owner_game_slot=game_ref.slot,
            owner_game_generation=game_ref.generation,
        )

    def read_workspace(self, ref: WorkspaceRef) -> np.ndarray:
        """Read state from workspace slot.

        Returns an independent copy to prevent holding onto mutable views.
        """
        self._validate_workspace_ref(ref)
        return self._workspace_rows[ref.slot].copy()

    def write_workspace(self, ref: WorkspaceRef, state: np.ndarray) -> None:
        """Write new state into an allocated workspace slot.

        Input array is defensively copied into arena storage.
        """
        self._validate_workspace_ref(ref)
        self._validate_state(state)
        self._workspace_rows[ref.slot, :] = state

    def reset_workspace_from_game(self, ref: WorkspaceRef) -> None:
        """Reset workspace state back to the current state of its owner real game slot."""
        self._validate_workspace_ref(ref)
        owner_slot = ref.owner_game_slot
        if (
            not self._game_allocated[owner_slot]
            or self._game_generations[owner_slot] != ref.owner_game_generation
        ):
            raise ArenaInvalidRefError(
                f"Owner game slot {owner_slot} of workspace {ref.slot} is no longer valid"
            )
        self._workspace_rows[ref.slot, :] = self._game_rows[owner_slot]

    def free_workspace(self, ref: WorkspaceRef) -> None:
        """Free an allocated workspace slot.

        Removes ownership link from the parent real game slot.
        Advances generation to invalidate the released ref.
        """
        self._validate_workspace_ref(ref)

        owner_slot = ref.owner_game_slot
        if 0 <= owner_slot < self._layout.game_capacity:
            self._game_active_workspaces[owner_slot].discard(ref.slot)

        self._workspace_allocated[ref.slot] = False
        self._workspace_generations[ref.slot] += 1
        self._workspace_owners[ref.slot] = None
        self._workspace_rows[ref.slot, :].fill(0)
        self._active_workspace_count -= 1
        heapq.heappush(self._free_workspaces, ref.slot)
