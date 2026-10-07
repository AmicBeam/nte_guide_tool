"""Container opcode device handlers for the batched duel CUDA VM.

Provides:
- SUPPORTED_COLLECTION_OPS: frozenset of collection opcodes supported by the device handler.
- IMPLEMENTATION_STATUS: detailed inventory of implemented capabilities and explicit scope boundaries.
- emit_collection_source(): returns self-contained CUDA C++ namespace duel_rule_vm source code.

Scope and Contract:
- Operates on Context, Heap, frames, and node/edge primitives without Torch or host fallback.
- Strictly adheres to CPython 3.10 stack effects.
- Nonzero return from handle_collections indicates the opcode was handled; failures set
  c->error or c->h.error and do not forge success.
"""

from __future__ import annotations

from typing import Mapping

# Opcodes handled by the device collections processor (CPython 3.10 stack semantics)
SUPPORTED_COLLECTION_OPS: frozenset[str] = frozenset((
    "BUILD_MAP",
    "BUILD_CONST_KEY_MAP",
    "STORE_SUBSCR",
    "DELETE_SUBSCR",
    "UNPACK_SEQUENCE",
    "BUILD_SLICE",
    "LIST_APPEND",
    "SET_ADD",
    "SET_UPDATE",
    "MAP_ADD",
    "DICT_UPDATE",
    "DICT_MERGE",
    "LIST_EXTEND",
    "LIST_TO_TUPLE",
    "ROT_THREE",
    "DUP_TOP_TWO",
    "BUILD_SET",
))

# Explicit inventory of implemented capabilities and known omissions
IMPLEMENTATION_STATUS: Mapping[str, str] = {
    "BUILD_MAP": "Implemented: pops 2*arg items (k1, v1, ...), inserts into dict with dict_set, pushes dict.",
    "BUILD_CONST_KEY_MAP": "Implemented: pops keys tuple, pops arg values, inserts into dict, pushes dict.",
    "STORE_SUBSCR": "Implemented: pops key, container, value; routes to collection_store (dict/list).",
    "DELETE_SUBSCR": "Implemented: pops key, container; routes to collection_delete (dict/list).",
    "UNPACK_SEQUENCE": "Implemented: pops list/tuple, verifies length == arg, pushes elements right-to-left.",
    "BUILD_SLICE": "Implemented: 2-arg (stop, start) and 3-arg (step, stop, start); rejects step == 0.",
    "LIST_APPEND": "Implemented: pops value, appends to list at relative depth arg, keeps list on stack.",
    "SET_ADD": "Implemented: pops value, adds to set (tag 21) at relative depth arg with key_equal dedup.",
    "SET_UPDATE": "Implemented for materialized set/frozenset/list/tuple; unordered iteration parity is not claimed.",
    "MAP_ADD": "Implemented: pops value, key; sets in map at relative depth arg with dict_set.",
    "DICT_UPDATE": "Implemented: pops other dict, updates target dict at relative depth arg (overwrites keys).",
    "DICT_MERGE": "Implemented: pops other dict, checks for key collisions (fails on duplicate), merges unique keys.",
    "LIST_EXTEND": "Implemented: pops iterable (list/tuple/set), appends each item to list at relative depth arg.",
    "LIST_TO_TUPLE": "Implemented: pops list, creates new immutable tuple referencing identical children.",
    "ROT_THREE": "Implemented: lifts second and third stack items, moves top to 3rd position (CPython 3.10 order).",
    "DUP_TOP_TWO": "Implemented: duplicates top two items on stack preserving order.",
    "BUILD_SET": "Implemented: pops arg items, creates new set (tag 21), adds items with deduplication.",
    "collection_subscript": (
        "Implemented: dict/entity key lookup; list/tuple positive/negative int indexing and slice indexing; "
        "str positive/negative codepoint indexing and slice codepoint extraction (step == 1 and arbitrary step). "
        "Omitted: multidimensional/numpy slicing."
    ),
    "collection_store": (
        "Implemented: dict/entity key assignment; list positive/negative int index assignment. "
        "Constant and immutable writes (tuple/str) rejected with ERROR_CONSTANT_WRITE. "
        "Omitted: list slice assignment (list[a:b] = iterable)."
    ),
    "collection_delete": (
        "Implemented: dict/entity key deletion; list positive/negative int index deletion with shifting. "
        "Omitted: list slice deletion (del list[a:b])."
    ),
    "set_semantics": (
        "Temporary Tag 21 represents sets with key_equal deduplication and membership checking. "
        "CPython 3.10 open-addressing hash table iteration order is NOT implemented; elements "
        "are stored in insertion order."
    ),
    "integer_bounds": (
        "Scalar 64-bit integer indexing is verified; values outside 32-bit signed range fail with "
        "BAD_PROGRAM rather than silent narrowing."
    ),
}


def emit_collection_source() -> str:
    """Return CUDA C++ device source code for container opcodes.

    This source is designed to be placed inside namespace duel_rule_vm after the Context
    and basic helper definitions, before execute().
    """
    return r"""// --- VM Collections and Container Opcode Processor ---
#ifndef T_SLICE
const int T_SLICE = 20;
#endif
#ifndef T_SET
const int T_SET = 21;
#endif

// Create an empty mutable set node (tag 21)
__device__ inline int make_set(Context* c) {
    if (c->h.counts[0] + 1 > c->h.max_nodes) {
        c->h.error = ERROR_NODE_CAPACITY;
        return MISSING;
    }
    int ref = c->h.counts[0]++;
    c->h.nodes[ref * 4 + 0] = T_SET;
    c->h.nodes[ref * 4 + 1] = c->h.counts[1];
    c->h.nodes[ref * 4 + 2] = 0;
    c->h.nodes[ref * 4 + 3] = 0;
    return ref;
}

// Add an element to a set node with key_equal deduplication
__device__ inline int set_add(Context* c, int set_ref, int value) {
    if (c->h.error != ERROR_OK || c->error) return 0;
    if (set_ref < 0) {
        c->h.error = ERROR_CONSTANT_WRITE;
        return 0;
    }
    const int* snode = node(c, set_ref);
    if (!snode) return 0;
    if (snode[0] != T_SET) {
        c->error = UNSUPPORTED_TYPE;
        return 0;
    }
    // Verify hashability: reject lists, dicts, entities, and unhashables
    key_equal(&c->h, value, value);
    if (c->h.error != ERROR_OK) return 0;

    int count = snode[2];
    int offset = snode[1];
    // Check if element already present in set
    for (int i = 0; i < count; i++) {
        int existing = edge(c, set_ref, i);
        if (c->h.error != ERROR_OK || c->error) return 0;
        if (key_equal(&c->h, existing, value)) {
            return 1; // Already present, no-op
        }
        if (c->h.error != ERROR_OK) return 0;
    }

    // Append to set edge storage with relocation if not at tail
    if (offset + count == c->h.counts[1]) {
        if (c->h.counts[1] + 1 > c->h.max_edges) {
            c->h.error = ERROR_EDGE_CAPACITY;
            return 0;
        }
        int edge_idx = c->h.counts[1]++;
        c->h.edges[edge_idx * 2 + 0] = value;
        c->h.edges[edge_idx * 2 + 1] = -1;
        c->h.nodes[set_ref * 4 + 2] = count + 1;
        return 1;
    } else {
        int new_count = count + 1;
        if (c->h.counts[1] + new_count > c->h.max_edges) {
            c->h.error = ERROR_EDGE_CAPACITY;
            return 0;
        }
        int new_offset = c->h.counts[1];
        for (int i = 0; i < count; i++) {
            c->h.edges[(new_offset + i) * 2 + 0] = edge(c, set_ref, i);
            c->h.edges[(new_offset + i) * 2 + 1] = -1;
        }
        c->h.edges[(new_offset + count) * 2 + 0] = value;
        c->h.edges[(new_offset + count) * 2 + 1] = -1;
        c->h.counts[1] += new_count;
        c->h.nodes[set_ref * 4 + 1] = new_offset;
        c->h.nodes[set_ref * 4 + 2] = new_count;
        return 1;
    }
}

// Check membership of needle in set
__device__ inline int set_contains(Context* c, int needle, int set_ref) {
    const int* n = node(c, set_ref);
    if (!n) return 0;
    if (n[0] != T_SET && n[0] != 22) {
        c->error = UNSUPPORTED_TYPE;
        return 0;
    }
    int count = n[2];
    for (int i = 0; i < count; i++) {
        int elem = edge(c, set_ref, i);
        if (key_equal(&c->h, needle, elem)) return 1;
        if (c->h.error != ERROR_OK) return 0;
    }
    return 0;
}

// Resolve start, stop, step for a slice reference against a sequence length.
// Rejects slice step == 0. Clamps indices strictly following CPython 3.10 conventions.
__device__ inline int resolve_slice_indices(
    Context* c, int slice_ref, int length,
    long long* out_start, long long* out_stop, long long* out_step
) {
    const int* sn = node(c, slice_ref);
    if (!sn || sn[0] != T_SLICE || sn[2] != 3) {
        c->error = BAD_PROGRAM;
        return 0;
    }
    int start_ref = edge(c, slice_ref, 0);
    int stop_ref  = edge(c, slice_ref, 1);
    int step_ref  = edge(c, slice_ref, 2);
    if (c->error) return 0;

    long long step = 1;
    if (step_ref != NIL && kind(c, step_ref) != TAG_NONE) {
        if (read_int64(&c->h, step_ref, &step) < 0) return 0;
        if (step == 0) {
            c->error = BAD_PROGRAM; // slice step cannot be zero
            return 0;
        }
    }

    long long start = (step > 0) ? 0 : (length - 1);
    if (start_ref != NIL && kind(c, start_ref) != TAG_NONE) {
        if (read_int64(&c->h, start_ref, &start) < 0) return 0;
        if (start < 0) {
            start += length;
            if (start < 0) start = (step > 0) ? 0 : -1;
        } else if (start >= length) {
            start = (step > 0) ? length : (length - 1);
        }
    }

    long long stop = (step > 0) ? length : -1;
    if (stop_ref != NIL && kind(c, stop_ref) != TAG_NONE) {
        if (read_int64(&c->h, stop_ref, &stop) < 0) return 0;
        if (stop < 0) {
            stop += length;
            if (stop < 0) stop = (step > 0) ? 0 : -1;
        } else if (stop >= length) {
            stop = (step > 0) ? length : (length - 1);
        }
    }

    // Sequence length is int32; larger steps visit at most one element. Clamp
    // without changing the selected indices, preventing signed addition overflow.
    if(step > (long long)length+1)step=(long long)length+1;
    if(step < -(long long)length-1)step=-(long long)length-1;
    *out_start = start;
    *out_stop = stop;
    *out_step = step;
    return 1;
}

// Unified subscript access supporting dict/entity key access, list/tuple indexing,
// and sequence slicing (list, tuple, string).
__device__ inline int collection_subscript(Context* c, int owner, int key) {
    int tag = kind(c, owner);
    if (tag == TAG_DICT || (tag >= 8 && tag <= 10)) {
        if (kind(c, key) == T_SLICE) {
            c->error = UNSUPPORTED_TYPE; // Dict cannot be sliced
            return MISSING;
        }
        int val = dict_get(&c->h, owner, key, MISSING);
        if (val == MISSING && !c->h.error) c->h.error = ERROR_KEY_NOT_FOUND;
        return val;
    }
    if (tag == TAG_LIST || tag == TAG_TUPLE) {
        if (kind(c, key) == T_SLICE) {
            int len = length(c, owner);
            if (c->error) return MISSING;
            long long start = 0, stop = 0, step = 0;
            if (!resolve_slice_indices(c, key, len, &start, &stop, &step)) return MISSING;

            if (tag == TAG_LIST) {
                int out_ref = make_list(&c->h);
                if (out_ref < 0 || c->h.error) return MISSING;
                if ((step > 0 && start < stop) || (step < 0 && start > stop)) {
                    for (long long i = start; (step > 0 ? i < stop : i > stop); i += step) {
                        int elem = edge(c, owner, (int)i);
                        if (c->error) return MISSING;
                        list_append(&c->h, out_ref, elem);
                        if (c->h.error) return MISSING;
                    }
                }
                return out_ref;
            } else {
                int count = 0;
                if ((step > 0 && start < stop) || (step < 0 && start > stop)) {
                    for (long long i = start; (step > 0 ? i < stop : i > stop); i += step) {
                        count++;
                    }
                }
                if (c->h.counts[0] + 1 > c->h.max_nodes) { c->h.error = ERROR_NODE_CAPACITY; return MISSING; }
                if (c->h.counts[1] + count > c->h.max_edges) { c->h.error = ERROR_EDGE_CAPACITY; return MISSING; }
                int start_edge = c->h.counts[1];
                int cur = 0;
                if (count > 0) {
                    for (long long i = start; (step > 0 ? i < stop : i > stop); i += step) {
                        c->h.edges[(start_edge + cur) * 2 + 0] = edge(c, owner, (int)i);
                        c->h.edges[(start_edge + cur) * 2 + 1] = -1;
                        cur++;
                    }
                }
                c->h.counts[1] += count;
                int tup_ref = c->h.counts[0]++;
                c->h.nodes[tup_ref * 4 + 0] = TAG_TUPLE;
                c->h.nodes[tup_ref * 4 + 1] = start_edge;
                c->h.nodes[tup_ref * 4 + 2] = count;
                c->h.nodes[tup_ref * 4 + 3] = 0;
                return tup_ref;
            }
        } else {
            long long index = 0;
            if (read_int64(&c->h, key, &index) < 0) return MISSING;
            if (index < (-2147483647-1LL) || index > 2147483647LL) {
                c->error = BAD_PROGRAM; // integer index overflow without narrowing
                return MISSING;
            }
            return list_get(&c->h, owner, (int)index);
        }
    }
    if (tag == TAG_STR) {
        NodeView snode;
        if (!get_node(&c->h, owner, &snode)) { c->error = BAD_PROGRAM; return MISSING; }
        const unsigned char* sp = get_payload_ptr(&c->h, &snode);
        int err = 0;
        int total_chars = utf8_char_length(sp, snode.count, &err);
        if (err) { c->h.error = ERROR_INVALID_UTF8; return MISSING; }

        if (kind(c, key) == T_SLICE) {
            long long start = 0, stop = 0, step = 0;
            if (!resolve_slice_indices(c, key, total_chars, &start, &stop, &step)) return MISSING;

            if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                return make_utf8(&c->h, "", 0);
            }
            if (step == 1) {
                int byte_pos = 0, cur_c = 0;
                int start_byte = 0, stop_byte = 0;
                while (byte_pos < snode.count && cur_c < stop) {
                    if (cur_c == (int)start) start_byte = byte_pos;
                    unsigned char ch = sp[byte_pos];
                    if (ch < 0x80) byte_pos += 1;
                    else if ((ch & 0xE0) == 0xC0) byte_pos += 2;
                    else if ((ch & 0xF0) == 0xE0) byte_pos += 3;
                    else byte_pos += 4;
                    cur_c++;
                }
                if (cur_c == (int)stop) stop_byte = byte_pos;
                else stop_byte = snode.count;
                return make_utf8(&c->h, (const char*)(sp + start_byte), stop_byte - start_byte);
            } else {
                int init_payload = c->h.counts[2];
                for (long long target_char = start; (step > 0 ? target_char < stop : target_char > stop); target_char += step) {
                    int byte_pos = 0, cur_c = 0;
                    while (byte_pos < snode.count && cur_c < target_char) {
                        unsigned char ch = sp[byte_pos];
                        if (ch < 0x80) byte_pos += 1;
                        else if ((ch & 0xE0) == 0xC0) byte_pos += 2;
                        else if ((ch & 0xF0) == 0xE0) byte_pos += 3;
                        else byte_pos += 4;
                        cur_c++;
                    }
                    if (byte_pos >= snode.count) break;
                    int c_bytes = 1;
                    unsigned char ch = sp[byte_pos];
                    if (ch < 0x80) c_bytes = 1;
                    else if ((ch & 0xE0) == 0xC0) c_bytes = 2;
                    else if ((ch & 0xF0) == 0xE0) c_bytes = 3;
                    else c_bytes = 4;

                    if (c->h.counts[2] + c_bytes > c->h.max_payload) {
                        c->h.counts[2] = init_payload;
                        c->h.error = ERROR_PAYLOAD_CAPACITY;
                        return MISSING;
                    }
                    for (int b = 0; b < c_bytes; b++) {
                        c->h.payload[c->h.counts[2]++] = sp[byte_pos + b];
                    }
                }
                int total_slice_bytes = c->h.counts[2] - init_payload;
                if (c->h.counts[0] + 1 > c->h.max_nodes) {
                    c->h.counts[2] = init_payload;
                    c->h.error = ERROR_NODE_CAPACITY;
                    return MISSING;
                }
                int ref = c->h.counts[0]++;
                c->h.nodes[ref * 4 + 0] = TAG_STR;
                c->h.nodes[ref * 4 + 1] = init_payload;
                c->h.nodes[ref * 4 + 2] = total_slice_bytes;
                c->h.nodes[ref * 4 + 3] = 0;
                return ref;
            }
        } else {
            long long idx = 0;
            if (read_int64(&c->h, key, &idx) < 0) return MISSING;
            if (idx < (-2147483647-1LL) || idx > 2147483647LL) {
                c->error = BAD_PROGRAM;
                return MISSING;
            }
            int char_idx = (int)idx;
            if (char_idx < 0) char_idx += total_chars;
            if (char_idx < 0 || char_idx >= total_chars) {
                c->h.error = ERROR_INDEX_OUT_OF_BOUNDS;
                return MISSING;
            }
            int byte_pos = 0, cur_c = 0;
            while (byte_pos < snode.count && cur_c < char_idx) {
                unsigned char ch = sp[byte_pos];
                if (ch < 0x80) byte_pos += 1;
                else if ((ch & 0xE0) == 0xC0) byte_pos += 2;
                else if ((ch & 0xF0) == 0xE0) byte_pos += 3;
                else byte_pos += 4;
                cur_c++;
            }
            int c_bytes = 1;
            if (byte_pos < snode.count) {
                unsigned char ch = sp[byte_pos];
                if (ch < 0x80) c_bytes = 1;
                else if ((ch & 0xE0) == 0xC0) c_bytes = 2;
                else if ((ch & 0xF0) == 0xE0) c_bytes = 3;
                else c_bytes = 4;
            }
            return make_utf8(&c->h, (const char*)(sp + byte_pos), c_bytes);
        }
    }
    c->error = UNSUPPORTED_TYPE;
    return MISSING;
}

// Subscript assignment supporting dict/entity keys and list element assignment.
// List slice assignment (list[a:b] = iterable) is omitted in this device subset.
__device__ inline int collection_store(Context* c, int owner, int key, int val) {
    if (owner < 0) {
        c->h.error = ERROR_CONSTANT_WRITE;
        return 0;
    }
    int tag = kind(c, owner);
    if (tag == TAG_DICT || (tag >= 8 && tag <= 10)) {
        dict_set(&c->h, owner, key, val);
        return (c->h.error == ERROR_OK) ? 1 : 0;
    }
    if (tag == TAG_LIST) {
        if (kind(c, key) == T_SLICE) {
            c->error = UNSUPPORTED_OP; // slice assignment omitted in device VM
            return 0;
        }
        long long idx = 0;
        if (read_int64(&c->h, key, &idx) < 0) return 0;
        if (idx < (-2147483647-1LL) || idx > 2147483647LL) {
            c->error = BAD_PROGRAM;
            return 0;
        }
        list_set(&c->h, owner, (int)idx, val);
        return (c->h.error == ERROR_OK) ? 1 : 0;
    }
    if (tag == TAG_TUPLE || tag == TAG_STR) {
        c->h.error = ERROR_CONSTANT_WRITE;
        return 0;
    }
    c->error = UNSUPPORTED_TYPE;
    return 0;
}

// Subscript deletion supporting dict/entity keys and list element deletion.
__device__ inline int collection_delete(Context* c, int owner, int key) {
    if (owner < 0) {
        c->h.error = ERROR_CONSTANT_WRITE;
        return 0;
    }
    int tag = kind(c, owner);
    if (tag == TAG_DICT || (tag >= 8 && tag <= 10)) {
        dict_delete(&c->h, owner, key);
        return (c->h.error == ERROR_OK) ? 1 : 0;
    }
    if (tag == TAG_LIST) {
        long long idx = 0;
        if (read_int64(&c->h, key, &idx) < 0) return 0;
        if (idx < (-2147483647-1LL) || idx > 2147483647LL) {
            c->error = BAD_PROGRAM;
            return 0;
        }
        const int* n = node(c, owner);
        if (!n) return 0;
        int count = n[2];
        int i = (int)idx;
        if (i < 0) i = count + i;
        if (i < 0 || i >= count) {
            c->h.error = ERROR_INDEX_OUT_OF_BOUNDS;
            return 0;
        }
        int offset = n[1];
        for (int k = i; k < count - 1; k++) {
            c->h.edges[(offset + k) * 2 + 0] = c->h.edges[(offset + k + 1) * 2 + 0];
            c->h.edges[(offset + k) * 2 + 1] = -1;
        }
        c->h.nodes[owner * 4 + 2] = count - 1;
        if (offset + count == c->h.counts[1]) c->h.counts[1]--;
        return 1;
    }
    c->error = UNSUPPORTED_TYPE;
    return 0;
}

// Master collection opcode device handler.
// Returns 1 if opcode was handled by this module, 0 if it does not belong here.
// Errors set c->error or c->h.error without forging success.
__device__ inline int handle_collections(Context* c, int* f, int op, int arg, int value, int target) {
    switch (op) {
        case OP_BUILD_MAP: {
            if (arg < 0 || 2 * arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int map_ref = make_dict(&c->h);
            if (map_ref < 0 || c->h.error) return 1;
            int base = f[2] - 2 * arg;
            for (int i = 0; i < arg; i++) {
                int k = f[SBASE + base + 2 * i];
                int v = f[SBASE + base + 2 * i + 1];
                dict_set(&c->h, map_ref, k, v);
                if (c->h.error) return 1;
            }
            f[2] -= 2 * arg;
            push(c, f, map_ref);
            return 1;
        }
        case OP_BUILD_CONST_KEY_MAP: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int keys_tuple = pop(c, f);
            if (kind(c, keys_tuple) != TAG_TUPLE) { c->error = UNSUPPORTED_TYPE; return 1; }
            if (length(c, keys_tuple) != arg) { c->error = BAD_PROGRAM; return 1; }
            if (arg < 0 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int map_ref = make_dict(&c->h);
            if (map_ref < 0 || c->h.error) return 1;
            int val_base = f[2] - arg;
            for (int i = 0; i < arg; i++) {
                int k = edge(c, keys_tuple, i);
                if (c->error) return 1;
                int v = f[SBASE + val_base + i];
                dict_set(&c->h, map_ref, k, v);
                if (c->h.error) return 1;
            }
            f[2] -= arg;
            push(c, f, map_ref);
            return 1;
        }
        case OP_STORE_SUBSCR: {
            if (f[2] < 3) { c->error = BAD_PROGRAM; return 1; }
            int key = pop(c, f);
            int container = pop(c, f);
            int val = pop(c, f);
            collection_store(c, container, key, val);
            return 1;
        }
        case OP_DELETE_SUBSCR: {
            if (f[2] < 2) { c->error = BAD_PROGRAM; return 1; }
            int key = pop(c, f);
            int container = pop(c, f);
            collection_delete(c, container, key);
            return 1;
        }
        case OP_UNPACK_SEQUENCE: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int seq = pop(c, f);
            int stag = kind(c, seq);
            if (stag != TAG_LIST && stag != TAG_TUPLE) { c->error = UNSUPPORTED_TYPE; return 1; }
            int seq_len = length(c, seq);
            if (c->error) return 1;
            if (seq_len != arg) { c->error = BAD_PROGRAM; return 1; }
            if (f[2] + arg > VM_STACK) { c->error = STACK_OVERFLOW; return 1; }
            for (int i = arg - 1; i >= 0; i--) {
                int elem = edge(c, seq, i);
                if (c->error) return 1;
                push(c, f, elem);
            }
            return 1;
        }
        case OP_BUILD_SLICE: {
            if (arg != 2 && arg != 3) { c->error = BAD_PROGRAM; return 1; }
            if (f[2] < arg) { c->error = BAD_PROGRAM; return 1; }
            int step = NIL, stop = NIL, start = NIL;
            if (arg == 3) {
                step = pop(c, f);
                stop = pop(c, f);
                start = pop(c, f);
            } else {
                stop = pop(c, f);
                start = pop(c, f);
                step = NIL;
            }
            if (c->h.counts[1] + 3 > c->h.max_edges) {
                c->h.error = ERROR_EDGE_CAPACITY;
                return 1;
            }
            int edge_start = c->h.counts[1];
            c->h.edges[(edge_start + 0) * 2 + 0] = start;
            c->h.edges[(edge_start + 0) * 2 + 1] = -1;
            c->h.edges[(edge_start + 1) * 2 + 0] = stop;
            c->h.edges[(edge_start + 1) * 2 + 1] = -1;
            c->h.edges[(edge_start + 2) * 2 + 0] = step;
            c->h.edges[(edge_start + 2) * 2 + 1] = -1;
            c->h.counts[1] += 3;
            int slice_ref = meta(c, T_SLICE, edge_start, 3, 0);
            push(c, f, slice_ref);
            return 1;
        }
        case OP_LIST_APPEND: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int val = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int list_ref = f[SBASE + f[2] - arg];
            if (kind(c, list_ref) != TAG_LIST) { c->error = UNSUPPORTED_TYPE; return 1; }
            list_append(&c->h, list_ref, val);
            return 1;
        }
        case OP_SET_ADD: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int val = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int set_ref = f[SBASE + f[2] - arg];
            if (kind(c, set_ref) != T_SET) { c->error = UNSUPPORTED_TYPE; return 1; }
            set_add(c, set_ref, val);
            return 1;
        }
        case OP_SET_UPDATE: {
            int other=pop(c,f);
            if(arg<1||arg>f[2]){c->error=BAD_PROGRAM;return 1;}
            int dest=f[SBASE+f[2]-arg],tag=kind(c,other);
            if(tag!=21&&tag!=22&&tag!=TAG_LIST&&tag!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 1;}
            int count=length(c,other);
            for(int i=0;i<count&&!c->error&&!c->h.error;i++)set_add(c,dest,edge(c,other,i));
            return 1;
        }
        case OP_MAP_ADD: {
            if (f[2] < 2) { c->error = BAD_PROGRAM; return 1; }
            int val = pop(c, f);
            int key = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int map_ref = f[SBASE + f[2] - arg];
            int mtag = kind(c, map_ref);
            if (mtag != TAG_DICT && !(mtag >= 8 && mtag <= 10)) { c->error = UNSUPPORTED_TYPE; return 1; }
            dict_set(&c->h, map_ref, key, val);
            return 1;
        }
        case OP_DICT_UPDATE: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int other = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int dict_ref = f[SBASE + f[2] - arg];
            int dtag = kind(c, dict_ref);
            int otag = kind(c, other);
            if ((dtag != TAG_DICT && !(dtag >= 8 && dtag <= 10)) ||
                (otag != TAG_DICT && !(otag >= 8 && otag <= 10))) {
                c->error = UNSUPPORTED_TYPE;
                return 1;
            }
            const int* onode = node(c, other);
            int ocount = onode ? onode[2] : 0;
            for (int i = 0; i < ocount; i++) {
                int k = edge(c, other, i, 0);
                int v = edge(c, other, i, 1);
                if (c->error) return 1;
                dict_set(&c->h, dict_ref, k, v);
                if (c->h.error) return 1;
            }
            return 1;
        }
        case OP_DICT_MERGE: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int other = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int dict_ref = f[SBASE + f[2] - arg];
            int dtag = kind(c, dict_ref);
            int otag = kind(c, other);
            if ((dtag != TAG_DICT && !(dtag >= 8 && dtag <= 10)) ||
                (otag != TAG_DICT && !(otag >= 8 && otag <= 10))) {
                c->error = UNSUPPORTED_TYPE;
                return 1;
            }
            const int* onode = node(c, other);
            int ocount = onode ? onode[2] : 0;
            // DICT_MERGE: fails if any duplicate key exists between dict_ref and other
            for (int i = 0; i < ocount; i++) {
                int k = edge(c, other, i, 0);
                if (c->error) return 1;
                int existing = dict_find(&c->h, dict_ref, k);
                if (c->h.error) return 1;
                if (existing >= 0) {
                    c->error = BAD_CALL; // Duplicate key error
                    return 1;
                }
            }
            for (int i = 0; i < ocount; i++) {
                int k = edge(c, other, i, 0);
                int v = edge(c, other, i, 1);
                if (c->error) return 1;
                dict_set(&c->h, dict_ref, k, v);
                if (c->h.error) return 1;
            }
            return 1;
        }
        case OP_LIST_EXTEND: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int iterable = pop(c, f);
            if (arg < 1 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int list_ref = f[SBASE + f[2] - arg];
            if (kind(c, list_ref) != TAG_LIST) { c->error = UNSUPPORTED_TYPE; return 1; }
            int itag = kind(c, iterable);
            if (itag != TAG_LIST && itag != TAG_TUPLE && itag != T_SET) {
                c->error = UNSUPPORTED_TYPE;
                return 1;
            }
            int n = (itag == T_SET) ? (node(c, iterable) ? node(c, iterable)[2] : 0) : length(c, iterable);
            if (c->error) return 1;
            for (int i = 0; i < n; i++) {
                int elem = edge(c, iterable, i);
                if (c->error) return 1;
                list_append(&c->h, list_ref, elem);
                if (c->h.error) return 1;
            }
            return 1;
        }
        case OP_LIST_TO_TUPLE: {
            if (f[2] < 1) { c->error = BAD_PROGRAM; return 1; }
            int list_ref = pop(c, f);
            if (kind(c, list_ref) != TAG_LIST) { c->error = UNSUPPORTED_TYPE; return 1; }
            int n = length(c, list_ref);
            if (c->error) return 1;
            if (c->h.counts[0] + 1 > c->h.max_nodes) { c->h.error = ERROR_NODE_CAPACITY; return 1; }
            if (c->h.counts[1] + n > c->h.max_edges) { c->h.error = ERROR_EDGE_CAPACITY; return 1; }
            int start_edge = c->h.counts[1];
            for (int i = 0; i < n; i++) {
                c->h.edges[(start_edge + i) * 2 + 0] = edge(c, list_ref, i);
                c->h.edges[(start_edge + i) * 2 + 1] = -1;
            }
            c->h.counts[1] += n;
            int tup_ref = c->h.counts[0]++;
            c->h.nodes[tup_ref * 4 + 0] = TAG_TUPLE;
            c->h.nodes[tup_ref * 4 + 1] = start_edge;
            c->h.nodes[tup_ref * 4 + 2] = n;
            c->h.nodes[tup_ref * 4 + 3] = 0;
            push(c, f, tup_ref);
            return 1;
        }
        case OP_ROT_THREE: {
            if (f[2] < 3) { c->error = BAD_PROGRAM; return 1; }
            int top = f[SBASE + f[2] - 1];
            int second = f[SBASE + f[2] - 2];
            int third = f[SBASE + f[2] - 3];
            f[SBASE + f[2] - 1] = second;
            f[SBASE + f[2] - 2] = third;
            f[SBASE + f[2] - 3] = top;
            return 1;
        }
        case OP_DUP_TOP_TWO: {
            if (f[2] < 2) { c->error = BAD_PROGRAM; return 1; }
            if (f[2] + 2 > VM_STACK) { c->error = STACK_OVERFLOW; return 1; }
            int a = f[SBASE + f[2] - 2];
            int b = f[SBASE + f[2] - 1];
            push(c, f, a);
            push(c, f, b);
            return 1;
        }
        case OP_BUILD_SET: {
            if (arg < 0 || arg > f[2]) { c->error = BAD_PROGRAM; return 1; }
            int set_ref = make_set(c);
            if (set_ref < 0 || c->h.error || c->error) return 1;
            int base = f[2] - arg;
            for (int i = 0; i < arg; i++) {
                int elem = f[SBASE + base + i];
                set_add(c, set_ref, elem);
                if (c->h.error || c->error) return 1;
            }
            f[2] -= arg;
            push(c, f, set_ref);
            return 1;
        }
        default:
            return 0; // Not handled by collections module
    }
}
"""
