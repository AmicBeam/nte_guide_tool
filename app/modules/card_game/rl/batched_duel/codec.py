"""Lossless bounded graph codec for batched duel states.

Provides pack_states and unpack_states routines to serialize complete, dynamic,
entity-aware Python duel states into contiguous int32/uint8 numpy arrays without
information loss, rule execution, JSON blob strings, or PyTorch/CUDA dependencies.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Sequence
import numpy as np

from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity,
    CharacterEntity,
    Entity,
    PlayerEntity,
)
from .schema import (
    CodecCapacityError,
    CodecCycleError,
    CodecDecodeError,
    CodecError,
    CodecUnsupportedTypeError,
    GraphLayout,
    PackedStateBatch,
    TAG_BOOL,
    TAG_CARD_ENTITY,
    TAG_CHARACTER_ENTITY,
    TAG_DICT,
    TAG_FLOAT,
    TAG_INT,
    TAG_LIST,
    TAG_NONE,
    TAG_PLAYER_ENTITY,
    TAG_STR,
    TAG_TUPLE,
    VALID_TAGS,
    clone_rows,
)

__all__ = [
    "CodecCapacityError",
    "CodecCycleError",
    "CodecDecodeError",
    "CodecError",
    "CodecUnsupportedTypeError",
    "GraphLayout",
    "PackedStateBatch",
    "clone_rows",
    "pack_states",
    "unpack_states",
]


def _int_to_minimal_signed_bytes(val: int) -> bytes:
    """Encode an arbitrary-size signed integer into minimal signed bytes (little-endian)."""
    if val == 0:
        return b'\x00'
    length = (val.bit_length() + 8) // 8
    if length > 1:
        try:
            return val.to_bytes(length - 1, byteorder='little', signed=True)
        except OverflowError:
            pass
    return val.to_bytes(length, byteorder='little', signed=True)


def _encode_single_state(
    state: Any,
    layout: GraphLayout,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]], bytearray, int]:
    """Encode a single Python state graph into intermediate lists.

    Preserves mutable container aliases, detects cycles, maintains exact dict insertion
    order and list vs tuple distinction, and tags Entity instances.
    """
    nodes: list[tuple[int, int, int, int]] = []
    edges: list[tuple[int, int]] = []
    payload = bytearray()

    container_memo: dict[int, int] = {}
    active_stack: set[int] = set()
    str_memo: dict[str, int] = {}
    none_node_idx: int | None = None
    bool_node_indices: dict[bool, int] = {}

    max_nodes = layout.max_nodes
    max_edges = layout.max_edges
    max_payload = layout.max_payload_bytes

    def encode_value(val: Any) -> int:
        nonlocal none_node_idx

        # 1. None
        if val is None:
            if none_node_idx is not None:
                return none_node_idx
            if len(nodes) >= max_nodes:
                raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")
            none_node_idx = len(nodes)
            nodes.append((TAG_NONE, 0, 0, 0))
            return none_node_idx

        # 2. Boolean (must check before int because bool subclasses int)
        if type(val) is bool:
            if val in bool_node_indices:
                return bool_node_indices[val]
            if len(nodes) >= max_nodes:
                raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")
            idx = len(nodes)
            bool_node_indices[val] = idx
            nodes.append((TAG_BOOL, 0, 0, 1 if val else 0))
            return idx

        # 3. Integer (arbitrary size, signed)
        if type(val) is int:
            raw = _int_to_minimal_signed_bytes(val)
            if len(payload) + len(raw) > max_payload:
                raise CodecCapacityError(
                    f"Payload capacity exceeded (limit {max_payload} bytes)"
                )
            if len(nodes) >= max_nodes:
                raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")
            offset = len(payload)
            payload.extend(raw)
            idx = len(nodes)
            nodes.append((TAG_INT, offset, len(raw), 0))
            return idx

        # 4. Float (finite float64)
        if type(val) is float:
            if not math.isfinite(val):
                raise CodecUnsupportedTypeError(
                    f"Non-finite float {val} is not supported (NaN and infinity are rejected)"
                )
            raw = struct.pack('<d', val)
            if len(payload) + 8 > max_payload:
                raise CodecCapacityError(
                    f"Payload capacity exceeded (limit {max_payload} bytes)"
                )
            if len(nodes) >= max_nodes:
                raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")
            offset = len(payload)
            payload.extend(raw)
            idx = len(nodes)
            nodes.append((TAG_FLOAT, offset, 8, 0))
            return idx

        # 5. String (UTF-8, deduplicated for immutable scalars)
        if type(val) is str:
            if val in str_memo:
                return str_memo[val]
            raw = val.encode('utf-8')
            if len(payload) + len(raw) > max_payload:
                raise CodecCapacityError(
                    f"Payload capacity exceeded (limit {max_payload} bytes)"
                )
            if len(nodes) >= max_nodes:
                raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")
            offset = len(payload)
            payload.extend(raw)
            idx = len(nodes)
            nodes.append((TAG_STR, offset, len(raw), 0))
            str_memo[val] = idx
            return idx

        # 6. Containers and Entities
        if type(val) is PlayerEntity:
            container_tag = TAG_PLAYER_ENTITY
        elif type(val) is CharacterEntity:
            container_tag = TAG_CHARACTER_ENTITY
        elif type(val) is CardEntity:
            container_tag = TAG_CARD_ENTITY
        elif isinstance(val, Entity):
            raise CodecUnsupportedTypeError(
                f"Unsupported Entity subclass: {type(val).__name__}"
            )
        elif type(val) is dict:
            container_tag = TAG_DICT
        elif type(val) is list:
            container_tag = TAG_LIST
        elif type(val) is tuple:
            container_tag = TAG_TUPLE
        else:
            raise CodecUnsupportedTypeError(
                f"Unsupported object type {type(val).__name__}: {val!r}"
            )

        if len(active_stack) >= layout.max_depth:
            raise CodecCapacityError(f'Graph nesting exceeds {layout.max_depth}')
        val_id = id(val)
        if val_id in active_stack:
            raise CodecCycleError(
                f"Circular reference detected in container of type {type(val).__name__}"
            )
        if val_id in container_memo:
            return container_memo[val_id]

        if len(nodes) >= max_nodes:
            raise CodecCapacityError(f"Node capacity exceeded (limit {max_nodes})")

        node_idx = len(nodes)
        nodes.append((0, 0, 0, 0))  # Placeholder to reserve index before children
        container_memo[val_id] = node_idx
        active_stack.add(val_id)

        try:
            if container_tag in (TAG_LIST, TAG_TUPLE):
                child_refs: list[int] = []
                for item in val:
                    child_refs.append(encode_value(item))

                if len(edges) + len(child_refs) > max_edges:
                    raise CodecCapacityError(
                        f"Edge capacity exceeded (limit {max_edges})"
                    )
                start_edge = len(edges)
                for c_ref in child_refs:
                    edges.append((c_ref, -1))
                nodes[node_idx] = (container_tag, start_edge, len(child_refs), 0)

            else:  # TAG_DICT, TAG_PLAYER_ENTITY, TAG_CHARACTER_ENTITY, TAG_CARD_ENTITY
                entry_refs: list[tuple[int, int]] = []
                for k, v in val.items():
                    if type(k) is not str:
                        raise CodecUnsupportedTypeError(
                            f"Dict key must be str, got {type(k).__name__}: {k!r} (JSON contract requires string keys)"
                        )
                    k_ref = encode_value(k)
                    v_ref = encode_value(v)
                    entry_refs.append((k_ref, v_ref))

                if len(edges) + len(entry_refs) > max_edges:
                    raise CodecCapacityError(
                        f"Edge capacity exceeded (limit {max_edges})"
                    )
                start_edge = len(edges)
                for k_ref, v_ref in entry_refs:
                    edges.append((k_ref, v_ref))
                nodes[node_idx] = (container_tag, start_edge, len(entry_refs), 0)
        finally:
            active_stack.remove(val_id)

        return node_idx

    root_idx = encode_value(state)
    return nodes, edges, payload, root_idx


def pack_states(states: Sequence[Any], layout: GraphLayout) -> PackedStateBatch:
    """Pack a sequence of duel states into independent arrays in a PackedStateBatch.

    Inputs remain unmodified. Allocates C-contiguous numpy arrays with first dimension
    matching len(states) (not game_capacity). Strictly fails if any layout capacity
    is exceeded or if unsupported types / cycles are found.
    """
    if not isinstance(layout, GraphLayout):
        raise TypeError(f"layout must be a GraphLayout, got {type(layout).__name__}")

    n_states = len(states)
    if n_states > layout.game_capacity:
        raise CodecCapacityError(
            f"Number of states {n_states} exceeds layout game_capacity {layout.game_capacity}"
        )

    if n_states == 0:
        return PackedStateBatch(
            layout=layout,
            nodes=np.zeros((0, layout.max_nodes, 4), dtype=np.int32, order='C'),
            edges=np.zeros((0, layout.max_edges, 2), dtype=np.int32, order='C'),
            payload=np.zeros((0, layout.max_payload_bytes), dtype=np.uint8, order='C'),
            counts=np.zeros((0, 3), dtype=np.int32, order='C'),
            roots=np.zeros((0,), dtype=np.int32, order='C'),
        )

    batch_nodes = np.zeros((n_states, layout.max_nodes, 4), dtype=np.int32, order='C')
    batch_edges = np.zeros((n_states, layout.max_edges, 2), dtype=np.int32, order='C')
    batch_payload = np.zeros((n_states, layout.max_payload_bytes), dtype=np.uint8, order='C')
    batch_counts = np.zeros((n_states, 3), dtype=np.int32, order='C')
    batch_roots = np.zeros((n_states,), dtype=np.int32, order='C')

    for r, st in enumerate(states):
        row_nodes, row_edges, row_payload, root_idx = _encode_single_state(st, layout)
        n_n = len(row_nodes)
        n_e = len(row_edges)
        n_p = len(row_payload)

        if n_n > 0:
            batch_nodes[r, :n_n, :] = row_nodes
        if n_e > 0:
            batch_edges[r, :n_e, :] = row_edges
        if n_p > 0:
            batch_payload[r, :n_p] = np.frombuffer(row_payload, dtype=np.uint8)
        batch_counts[r, :] = [n_n, n_e, n_p]
        batch_roots[r] = root_idx

    return PackedStateBatch(
        layout=layout,
        nodes=batch_nodes,
        edges=batch_edges,
        payload=batch_payload,
        counts=batch_counts,
        roots=batch_roots,
    )


def unpack_states(batch: PackedStateBatch) -> list[Any]:
    """Unpack a PackedStateBatch into fresh Python state objects.

    Batch arrays remain unmodified. Validates all offsets, counts, tags, and payload
    boundaries against recorded row counts. Preserves container aliases, dict insertion
    order, list vs tuple types, and PlayerEntity / CharacterEntity / CardEntity classes.
    """
    if not isinstance(batch, PackedStateBatch):
        raise TypeError(f"batch must be a PackedStateBatch, got {type(batch).__name__}")

    n_rows = batch.n_rows
    if n_rows == 0:
        return []

    # Validate overall array shapes and dtypes
    if batch.nodes.shape != (n_rows, batch.layout.max_nodes, 4) or batch.nodes.dtype != np.int32:
        raise CodecDecodeError("Invalid nodes array shape or dtype")
    if batch.edges.shape != (n_rows, batch.layout.max_edges, 2) or batch.edges.dtype != np.int32:
        raise CodecDecodeError("Invalid edges array shape or dtype")
    if batch.payload.shape != (n_rows, batch.layout.max_payload_bytes) or batch.payload.dtype != np.uint8:
        raise CodecDecodeError("Invalid payload array shape or dtype")
    if batch.counts.shape != (n_rows, 3) or batch.counts.dtype != np.int32:
        raise CodecDecodeError("Invalid counts array shape or dtype")
    if batch.roots.shape != (n_rows,) or batch.roots.dtype != np.int32:
        raise CodecDecodeError("Invalid roots array shape or dtype")

    results: list[Any] = []
    max_nodes = batch.layout.max_nodes
    max_edges = batch.layout.max_edges
    max_payload = batch.layout.max_payload_bytes

    for r in range(n_rows):
        node_count = int(batch.counts[r, 0])
        edge_count = int(batch.counts[r, 1])
        payload_count = int(batch.counts[r, 2])

        if node_count < 0 or node_count > max_nodes:
            raise CodecDecodeError(f"Row {r}: invalid node_count {node_count} (must be in [0, {max_nodes}])")
        if edge_count < 0 or edge_count > max_edges:
            raise CodecDecodeError(f"Row {r}: invalid edge_count {edge_count} (must be in [0, {max_edges}])")
        if payload_count < 0 or payload_count > max_payload:
            raise CodecDecodeError(f"Row {r}: invalid payload_count {payload_count} (must be in [0, {max_payload}])")

        if node_count == 0:
            raise CodecDecodeError(f"Row {r}: state graph contains 0 nodes")

        root_idx = int(batch.roots[r])
        if root_idx < 0 or root_idx >= node_count:
            raise CodecDecodeError(f"Row {r}: root_idx {root_idx} out of range [0, {node_count})")

        decode_memo: dict[int, Any] = {}
        active_decode_stack: set[int] = set()

        row_nodes = batch.nodes[r]
        row_edges = batch.edges[r]
        row_payload = batch.payload[r]

        def _decode_node(node_idx: int) -> Any:
            if node_idx < 0 or node_idx >= node_count:
                raise CodecDecodeError(f"Row {r}: node index {node_idx} out of range [0, {node_count})")
            if node_idx in active_decode_stack:
                raise CodecDecodeError(f"Row {r}: circular reference detected in node graph at node {node_idx}")
            if node_idx in decode_memo:
                return decode_memo[node_idx]
            if len(active_decode_stack) >= batch.layout.max_depth:
                raise CodecDecodeError('Graph nesting exceeds declared depth')

            active_decode_stack.add(node_idx)

            tag = int(row_nodes[node_idx, 0])
            offset = int(row_nodes[node_idx, 1])
            count = int(row_nodes[node_idx, 2])
            extra = int(row_nodes[node_idx, 3])

            if tag not in VALID_TAGS:
                raise CodecDecodeError(f"Row {r}: invalid tag {tag} at node {node_idx}")

            if tag == TAG_NONE:
                if (offset, count, extra) != (0, 0, 0):
                    raise CodecDecodeError('Invalid null node metadata')
                res = None
                decode_memo[node_idx] = res

            elif tag == TAG_BOOL:
                if offset != 0 or count != 0:
                    raise CodecDecodeError('Invalid bool node metadata')
                if extra not in (0, 1):
                    raise CodecDecodeError(f"Row {r}: invalid bool extra value {extra} at node {node_idx}")
                res = (extra == 1)
                decode_memo[node_idx] = res

            elif tag == TAG_INT:
                if offset < 0 or count < 1 or offset + count > payload_count:
                    raise CodecDecodeError(
                        f"Row {r}: int node {node_idx} payload range [{offset}, {offset + count}) exceeds used payload {payload_count}"
                    )
                raw_bytes = bytes(row_payload[offset : offset + count])
                res = int.from_bytes(raw_bytes, byteorder='little', signed=True)
                decode_memo[node_idx] = res

            elif tag == TAG_FLOAT:
                if count != 8 or offset < 0 or offset + 8 > payload_count:
                    raise CodecDecodeError(
                        f"Row {r}: float node {node_idx} payload range [{offset}, {offset + 8}) exceeds used payload {payload_count}"
                    )
                raw_bytes = bytes(row_payload[offset : offset + 8])
                res = struct.unpack('<d', raw_bytes)[0]
                if not math.isfinite(res):
                    raise CodecDecodeError(f"Row {r}: decoded non-finite float {res} at node {node_idx}")
                decode_memo[node_idx] = res

            elif tag == TAG_STR:
                if offset < 0 or count < 0 or offset + count > payload_count:
                    raise CodecDecodeError(
                        f"Row {r}: str node {node_idx} payload range [{offset}, {offset + count}) exceeds used payload {payload_count}"
                    )
                raw_bytes = bytes(row_payload[offset : offset + count])
                try:
                    res = raw_bytes.decode('utf-8')
                except UnicodeDecodeError as exc:
                    raise CodecDecodeError(f"Row {r}: invalid UTF-8 string at node {node_idx}") from exc
                decode_memo[node_idx] = res

            elif tag == TAG_LIST:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: list node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                res = []
                decode_memo[node_idx] = res
                for i in range(count):
                    child_idx = int(row_edges[offset + i, 0])
                    res.append(_decode_node(child_idx))

            elif tag == TAG_TUPLE:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: tuple node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                items = []
                for i in range(count):
                    child_idx = int(row_edges[offset + i, 0])
                    items.append(_decode_node(child_idx))
                res = tuple(items)
                decode_memo[node_idx] = res

            elif tag == TAG_DICT:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: dict node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                res = {}
                decode_memo[node_idx] = res
                for i in range(count):
                    k_idx = int(row_edges[offset + i, 0])
                    v_idx = int(row_edges[offset + i, 1])
                    k = _decode_node(k_idx)
                    if type(k) is not str:
                        raise CodecDecodeError(f"Row {r}: dict key node {k_idx} decoded to non-str {type(k).__name__}")
                    if k in res:
                        raise CodecDecodeError('Duplicate graph dictionary key')
                    v = _decode_node(v_idx)
                    res[k] = v

            elif tag == TAG_PLAYER_ENTITY:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: PlayerEntity node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                res = PlayerEntity.__new__(PlayerEntity)
                decode_memo[node_idx] = res
                for i in range(count):
                    k_idx = int(row_edges[offset + i, 0])
                    v_idx = int(row_edges[offset + i, 1])
                    k = _decode_node(k_idx)
                    if type(k) is not str:
                        raise CodecDecodeError(f"Row {r}: PlayerEntity key node {k_idx} decoded to non-str {type(k).__name__}")
                    if k in res:
                        raise CodecDecodeError('Duplicate graph dictionary key')
                    v = _decode_node(v_idx)
                    res[k] = v

            elif tag == TAG_CHARACTER_ENTITY:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: CharacterEntity node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                res = CharacterEntity.__new__(CharacterEntity)
                decode_memo[node_idx] = res
                for i in range(count):
                    k_idx = int(row_edges[offset + i, 0])
                    v_idx = int(row_edges[offset + i, 1])
                    k = _decode_node(k_idx)
                    if type(k) is not str:
                        raise CodecDecodeError(f"Row {r}: CharacterEntity key node {k_idx} decoded to non-str {type(k).__name__}")
                    if k in res:
                        raise CodecDecodeError('Duplicate graph dictionary key')
                    v = _decode_node(v_idx)
                    res[k] = v

            elif tag == TAG_CARD_ENTITY:
                if offset < 0 or count < 0 or offset + count > edge_count:
                    raise CodecDecodeError(
                        f"Row {r}: CardEntity node {node_idx} edge range [{offset}, {offset + count}) exceeds used edges {edge_count}"
                    )
                res = CardEntity.__new__(CardEntity)
                decode_memo[node_idx] = res
                for i in range(count):
                    k_idx = int(row_edges[offset + i, 0])
                    v_idx = int(row_edges[offset + i, 1])
                    k = _decode_node(k_idx)
                    if type(k) is not str:
                        raise CodecDecodeError(f"Row {r}: CardEntity key node {k_idx} decoded to non-str {type(k).__name__}")
                    if k in res:
                        raise CodecDecodeError('Duplicate graph dictionary key')
                    v = _decode_node(v_idx)
                    res[k] = v

            else:
                raise CodecDecodeError(f"Row {r}: unhandled tag {tag} at node {node_idx}")

            active_decode_stack.remove(node_idx)
            return res

        state = _decode_node(root_idx)
        if len(decode_memo) != node_count:
            raise CodecDecodeError('Unreachable nodes in used graph storage')
        results.append(state)

    return results
