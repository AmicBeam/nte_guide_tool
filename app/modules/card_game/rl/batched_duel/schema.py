"""Batched duel lossless graph codec schema and layout definitions.

Provides frozen GraphLayout specifications, PackedStateBatch array structures,
tag constants, and exception hierarchies for lossless batched duel state encoding.
This module is pure Python and NumPy without PyTorch or CUDA dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


class CodecError(Exception):
    """Base exception for duel state graph codec operations."""
    pass


class CodecCapacityError(CodecError):
    """Raised when game capacity, node count, edge count, or payload bytes exceeds layout limits."""
    pass


class CodecUnsupportedTypeError(CodecError, TypeError):
    """Raised when an unsupported type, un-finite float, circular reference, or unknown entity is encountered."""
    pass


class CodecDecodeError(CodecError, ValueError):
    """Raised when decoded data is corrupt, out of bounds, invalid tag, or malformed."""
    pass


class CodecCycleError(CodecUnsupportedTypeError, ValueError):
    """Raised when circular references are detected in states."""
    pass


# Node tag constants (int32)
TAG_NONE: int = 0
TAG_BOOL: int = 1
TAG_INT: int = 2
TAG_FLOAT: int = 3
TAG_STR: int = 4
TAG_LIST: int = 5
TAG_TUPLE: int = 6
TAG_DICT: int = 7
TAG_PLAYER_ENTITY: int = 8
TAG_CHARACTER_ENTITY: int = 9
TAG_CARD_ENTITY: int = 10

VALID_TAGS: frozenset[int] = frozenset(range(11))


@dataclass(frozen=True)
class GraphLayout:
    """Frozen layout specification for a bounded state graph batch.

    Attributes:
        game_capacity: Maximum number of simultaneous games/states in a batch (> 0).
        max_nodes: Maximum number of nodes per state graph (> 0).
        max_edges: Maximum number of edges per state graph (> 0).
        max_payload_bytes: Maximum number of payload bytes per state graph (> 0).
    """
    game_capacity: int
    max_nodes: int
    max_edges: int
    max_payload_bytes: int
    max_depth: int = 256

    def __post_init__(self) -> None:
        for field_name, value in (
            ("game_capacity", self.game_capacity),
            ("max_nodes", self.max_nodes),
            ("max_edges", self.max_edges),
            ("max_payload_bytes", self.max_payload_bytes),
            ("max_depth", self.max_depth),
        ):
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be an integer, got {type(value).__name__} (bool is rejected)"
                )
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be a positive integer > 0, got {value}"
                )
            if value > 0x7FFFFFFF:
                raise ValueError(f"{field_name} exceeds the signed int32 graph ABI")
        if self.max_depth > 256:
            raise ValueError("max_depth must be at most 256 for the bounded host codec")


@dataclass(frozen=True)
class PackedStateBatch:
    """True contiguous numpy array batch representation of packed state graphs.

    Arrays:
        nodes: np.ndarray of shape (N, max_nodes, 4), dtype int32, C-contiguous.
               Each node is [tag, offset, count, extra].
        edges: np.ndarray of shape (N, max_edges, 2), dtype int32, C-contiguous.
               Each edge is [key_or_child_ref, val_or_sentinel_ref].
        payload: np.ndarray of shape (N, max_payload_bytes), dtype uint8, C-contiguous.
                 Stores UTF-8 string bytes, signed integer bytes, float64 bytes.
        counts: np.ndarray of shape (N, 3), dtype int32, C-contiguous.
                Records [node_count, edge_count, payload_byte_count] per row.
        roots: np.ndarray of shape (N,), dtype int32, C-contiguous.
               Records root node index per row.
    """
    layout: GraphLayout
    nodes: np.ndarray
    edges: np.ndarray
    payload: np.ndarray
    counts: np.ndarray
    roots: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, GraphLayout):
            raise TypeError(f"layout must be a GraphLayout instance, got {type(self.layout).__name__}")

        for name, arr, expected_dtype, expected_ndim, shape_suffix in (
            ("nodes", self.nodes, np.int32, 3, (self.layout.max_nodes, 4)),
            ("edges", self.edges, np.int32, 3, (self.layout.max_edges, 2)),
            ("payload", self.payload, np.uint8, 2, (self.layout.max_payload_bytes,)),
            ("counts", self.counts, np.int32, 2, (3,)),
            ("roots", self.roots, np.int32, 1, ()),
        ):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.dtype != expected_dtype:
                raise TypeError(f"{name} dtype must be {expected_dtype}, got {arr.dtype}")
            if arr.ndim != expected_ndim:
                raise ValueError(f"{name} ndim must be {expected_ndim}, got {arr.ndim}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if arr.shape[1:] != shape_suffix:
                raise ValueError(f"{name} trailing dimensions must be {shape_suffix}, got {arr.shape[1:]}")

        n_rows = self.nodes.shape[0]
        for name, arr in (
            ("edges", self.edges),
            ("payload", self.payload),
            ("counts", self.counts),
            ("roots", self.roots),
        ):
            if arr.shape[0] != n_rows:
                raise ValueError(f"{name} row count {arr.shape[0]} does not match nodes row count {n_rows}")

        if n_rows > self.layout.game_capacity:
            raise CodecCapacityError(
                f"Batch row count {n_rows} exceeds layout game_capacity {self.layout.game_capacity}"
            )

    @property
    def n_rows(self) -> int:
        """Number of packed states currently resident in this batch."""
        return int(self.nodes.shape[0])

    @property
    def allocated_bytes(self) -> int:
        """Total memory bytes consumed by the five contiguous numpy arrays."""
        return int(
            self.nodes.nbytes
            + self.edges.nbytes
            + self.payload.nbytes
            + self.counts.nbytes
            + self.roots.nbytes
        )

    def clone_rows(self, row_indices: Sequence[int]) -> PackedStateBatch:
        """Copy specified rows into a new independent PackedStateBatch."""
        return clone_rows(self, row_indices)


def clone_rows(batch: PackedStateBatch, row_indices: Sequence[int]) -> PackedStateBatch:
    """Copy specified rows from batch into a new independent PackedStateBatch.

    Validates indices strictly: rejects booleans and non-integers, checks bounds [0, n_rows).
    Supports out-of-order and repeated indices. Produces independent C-contiguous arrays.
    """
    if not isinstance(batch, PackedStateBatch):
        raise TypeError(f"batch must be a PackedStateBatch, got {type(batch).__name__}")

    if not isinstance(row_indices, (list, tuple, np.ndarray)):
        indices = list(row_indices)
    else:
        indices = list(row_indices)

    for idx in indices:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Row index must be an integer, got {type(idx).__name__} (bool is rejected)")
        if idx < 0 or idx >= batch.n_rows:
            raise IndexError(f"Row index {idx} out of range [0, {batch.n_rows})")

    new_n = len(indices)
    if new_n > batch.layout.game_capacity:
        raise CodecCapacityError(
            f"Cloned batch size {new_n} exceeds layout game_capacity {batch.layout.game_capacity}"
        )

    if new_n == 0:
        new_nodes = np.zeros((0, batch.layout.max_nodes, 4), dtype=np.int32, order='C')
        new_edges = np.zeros((0, batch.layout.max_edges, 2), dtype=np.int32, order='C')
        new_payload = np.zeros((0, batch.layout.max_payload_bytes), dtype=np.uint8, order='C')
        new_counts = np.zeros((0, 3), dtype=np.int32, order='C')
        new_roots = np.zeros((0,), dtype=np.int32, order='C')
    else:
        idx_arr = np.array(indices, dtype=np.intp)
        new_nodes = np.ascontiguousarray(batch.nodes[idx_arr].copy())
        new_edges = np.ascontiguousarray(batch.edges[idx_arr].copy())
        new_payload = np.ascontiguousarray(batch.payload[idx_arr].copy())
        new_counts = np.ascontiguousarray(batch.counts[idx_arr].copy())
        new_roots = np.ascontiguousarray(batch.roots[idx_arr].copy())

    return PackedStateBatch(
        layout=batch.layout,
        nodes=new_nodes,
        edges=new_edges,
        payload=new_payload,
        counts=new_counts,
        roots=new_roots,
    )
