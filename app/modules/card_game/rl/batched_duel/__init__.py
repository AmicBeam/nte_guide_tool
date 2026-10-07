"""Batched duel CPU state arena package.

Provides true contiguous int32 numpy array storage for batched game states
and bounded hypothetical workspaces on CPU.
This package does not allocate storage or start runs upon import.
"""

from .arena import (
    ArenaBusyError,
    ArenaCapacityError,
    ArenaError,
    ArenaInvalidRefError,
    ArenaLayout,
    ArenaStateDtypeError,
    ArenaStateShapeError,
    CpuStateArena,
    GameRef,
    WorkspaceRef,
)

__all__ = [
    "ArenaBusyError",
    "ArenaCapacityError",
    "ArenaError",
    "ArenaInvalidRefError",
    "ArenaLayout",
    "ArenaStateDtypeError",
    "ArenaStateShapeError",
    "CpuStateArena",
    "GameRef",
    "WorkspaceRef",
]
