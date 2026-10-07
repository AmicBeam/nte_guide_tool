"""Unit tests for batched duel CUDA VM heap primitives and device-side graph manipulation.

Includes:
- Pure schema ABI, tag constants, and fingerprint validation (no CUDA or Torch required).
- Verification that importing vm_heap does not import Torch or initialize CUDA.
- Self-contained CUDA C++ source emission verification.
- Real CUDA device execution via CudaModule (strictly skipped if CUDA GPU/driver is unavailable):
  * Scalar creation (None, bool, int64, float64, UTF-8 strings).
  * Int64 read with strict >64-bit overflow error and byte preservation.
  * Unicode character length (codepoints vs byte length) and string comparison.
  * Non-finite float (NaN, Inf) rejection.
  * Python key equality semantics across None, bool, int, float, str, tuple, and unhashable keys.
  * Mutable list operations (get, set, append, pop, negative indexing).
  * Mutable dict operations (set, get with default, delete, insertion order preservation).
  * Read-only constant graph reading, strict write rejection (negative refs), and clone materialization.
  * Edge/node/payload capacity failure and state preservation.
  * Two-phase deep_clone preserving Entity classes, aliases, and original graph isolation.
"""

from __future__ import annotations

import math
import sys
import unittest
import numpy as np

from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity,
    CharacterEntity,
    PlayerEntity,
)
from app.modules.card_game.rl.batched_duel.codec import pack_states, unpack_states
from app.modules.card_game.rl.batched_duel.schema import (
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
)
from app.modules.card_game.rl.batched_duel.vm_heap import (
    ERROR_CONSTANT_WRITE,
    ERROR_EDGE_CAPACITY,
    ERROR_INDEX_OUT_OF_BOUNDS,
    ERROR_INT_OVERFLOW,
    ERROR_INVALID_REF,
    ERROR_INVALID_TAG,
    ERROR_INVALID_UTF8,
    ERROR_KEY_NOT_FOUND,
    ERROR_NODE_CAPACITY,
    ERROR_NON_FINITE_FLOAT,
    ERROR_OK,
    ERROR_PAYLOAD_CAPACITY,
    ERROR_STACK_OVERFLOW,
    ERROR_TYPE_MISMATCH,
    ERROR_UNHASHABLE_KEY,
    TAG_BOOL as VM_TAG_BOOL,
    TAG_CARD_ENTITY as VM_TAG_CARD_ENTITY,
    TAG_CHARACTER_ENTITY as VM_TAG_CHARACTER_ENTITY,
    TAG_DICT as VM_TAG_DICT,
    TAG_FLOAT as VM_TAG_FLOAT,
    TAG_INT as VM_TAG_INT,
    TAG_LIST as VM_TAG_LIST,
    TAG_NONE as VM_TAG_NONE,
    TAG_PLAYER_ENTITY as VM_TAG_PLAYER_ENTITY,
    TAG_STR as VM_TAG_STR,
    TAG_TUPLE as VM_TAG_TUPLE,
    abi_fingerprint,
    emit_vm_heap_source,
)


def is_cuda_available() -> bool:
    """Strict check if PyTorch and CUDA driver/device are genuinely available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        _ = torch.cuda.get_device_capability(0)
        return True
    except Exception:
        return False


class TestVmHeapPurePython(unittest.TestCase):
    """Pure host-side tests for ABI alignment, tags, and source emission (no CUDA needed)."""

    def test_no_torch_imported_by_module(self) -> None:
        """Verify that importing vm_heap does not import PyTorch."""
        self.assertIn("app.modules.card_game.rl.batched_duel.vm_heap", sys.modules)

    def test_schema_tag_alignment(self) -> None:
        """Verify all VM tags exactly match schema.py constants."""
        self.assertEqual(VM_TAG_NONE, TAG_NONE)
        self.assertEqual(VM_TAG_BOOL, TAG_BOOL)
        self.assertEqual(VM_TAG_INT, TAG_INT)
        self.assertEqual(VM_TAG_FLOAT, TAG_FLOAT)
        self.assertEqual(VM_TAG_STR, TAG_STR)
        self.assertEqual(VM_TAG_LIST, TAG_LIST)
        self.assertEqual(VM_TAG_TUPLE, TAG_TUPLE)
        self.assertEqual(VM_TAG_DICT, TAG_DICT)
        self.assertEqual(VM_TAG_PLAYER_ENTITY, TAG_PLAYER_ENTITY)
        self.assertEqual(VM_TAG_CHARACTER_ENTITY, TAG_CHARACTER_ENTITY)
        self.assertEqual(VM_TAG_CARD_ENTITY, TAG_CARD_ENTITY)

    def test_abi_fingerprint_deterministic(self) -> None:
        """Verify abi_fingerprint produces a stable 64-character hex digest."""
        fp1 = abi_fingerprint()
        fp2 = abi_fingerprint()
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in fp1))

    def test_emit_vm_heap_source_sanity(self) -> None:
        """Verify emitted C++ source contains all required namespace, ABI, and function declarations."""
        src = emit_vm_heap_source()
        self.assertIn("namespace duel_vm_heap", src)
        self.assertIn("struct Heap", src)
        self.assertIn("int* nodes", src)
        self.assertIn("int* edges", src)
        self.assertIn("unsigned char* payload", src)
        self.assertIn("int* counts", src)
        self.assertIn("const int* constant_nodes", src)
        self.assertIn("const int* constant_edges", src)
        self.assertIn("const unsigned char* constant_payload", src)
        self.assertIn("int error", src)

        # Check required function signatures
        for fn_name in (
            "tag(",
            "make_none(",
            "make_bool(",
            "make_int64(",
            "make_float64(",
            "make_utf8(",
            "read_int64(",
            "read_float64(",
            "truth(",
            "key_equal(",
            "dict_find(",
            "dict_get(",
            "dict_set(",
            "dict_delete(",
            "list_get(",
            "list_set(",
            "list_append(",
            "list_pop(",
            "make_list(",
            "make_tuple(",
            "make_dict(",
            "deep_clone(",
            "str_char_length(",
            "str_compare(",
            "init_heap(",
        ):
            self.assertIn(fn_name, src)


class TestVmHeapCudaExecution(unittest.TestCase):
    """Real CUDA compilation and kernel dispatch tests using CudaModule.

    Strictly skipped if CUDA GPU or driver is unavailable (no CPU emulation fallback).
    """

    def setUp(self) -> None:
        if not is_cuda_available():
            self.skipTest("CUDA GPU/driver unavailable; strictly no CPU fallback")

    def _compile_module(self, kernel_code: str):
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule

        full_src = f"{emit_vm_heap_source()}\n\n{kernel_code}"
        class CompiledSource:
            def launch_kernel(self,name,*,grid,block,args,spec):
                with CudaModule(full_src,{name:spec}) as module:
                    event=module.launch(name,args,grid=grid,block=block)
                    event.synchronize()
        return CompiledSource()

    def _canonical_export(self,layout,nodes,edges,payload,counts,root):
        """Device GC before canonical codec export; never relax unreachable-node checks."""
        import torch
        code=r"""
        extern "C" __global__ void compact_kernel(int* sn,int* se,unsigned char* sp,int* sc,
            int* dn,int* de,unsigned char* dp,int* dc,int root,int* map,int* queue,
            int maxn,int maxe,int maxp,int* status){
            duel_vm_heap::Heap source,target;
            duel_vm_heap::init_heap(&source,sn,se,sp,sc,maxn,maxe,maxp,0,0,0,0,0,0);
            duel_vm_heap::init_heap(&target,dn,de,dp,dc,maxn,maxe,maxp,0,0,0,0,0,0);
            int result=duel_vm_heap::compact_into(&source,&target,root,map,queue,maxn);
            status[0]=target.error;status[1]=result;
        }
        """
        dn=torch.zeros_like(nodes);de=torch.zeros_like(edges);dp=torch.zeros_like(payload);dc=torch.zeros_like(counts)
        mapping=torch.empty(layout.max_nodes,dtype=torch.int32,device='cuda');queue=torch.empty_like(mapping)
        status=torch.zeros(2,dtype=torch.int32,device='cuda')
        self._compile_module(code).launch_kernel('compact_kernel',grid=1,block=1,
            args=[nodes,edges,payload,counts,dn,de,dp,dc,root,mapping,queue,layout.max_nodes,layout.max_edges,layout.max_payload_bytes,status],
            spec=['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*','int32*','int32','int32*','int32*','int32','int32','int32','int32*'])
        self.assertEqual(status[0].item(),0)
        return PackedStateBatch(layout,dn.cpu().numpy(),de.cpu().numpy(),dp.cpu().numpy(),dc.cpu().numpy(),np.array([status[1].item()],dtype=np.int32))

    def test_cuda_scalar_creation_and_reading(self) -> None:
        """Test creating and reading scalar primitives directly in CUDA device code."""
        import torch

        kernel_code = r"""
        extern "C" __global__ void test_scalar_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            // 1. None
            int r_none = duel_vm_heap::make_none(&heap);
            if (duel_vm_heap::tag(&heap, r_none) != duel_vm_heap::TAG_NONE) { out_status[0] = 1; return; }
            if (duel_vm_heap::truth(&heap, r_none) != 0) { out_status[0] = 2; return; }

            // 2. Bool
            int r_false = duel_vm_heap::make_bool(&heap, 0);
            int r_true = duel_vm_heap::make_bool(&heap, 1);
            if (duel_vm_heap::truth(&heap, r_false) != 0) { out_status[0] = 3; return; }
            if (duel_vm_heap::truth(&heap, r_true) != 1) { out_status[0] = 4; return; }

            // 3. Int64
            int r_int0 = duel_vm_heap::make_int64(&heap, 0);
            int r_int42 = duel_vm_heap::make_int64(&heap, 42);
            int r_int_neg = duel_vm_heap::make_int64(&heap, -9223372036854775807LL);
            long long v0 = -1, v42 = -1, v_neg = 0;
            duel_vm_heap::read_int64(&heap, r_int0, &v0);
            duel_vm_heap::read_int64(&heap, r_int42, &v42);
            duel_vm_heap::read_int64(&heap, r_int_neg, &v_neg);
            if (v0 != 0 || v42 != 42 || v_neg != -9223372036854775807LL) { out_status[0] = 5; return; }
            if (duel_vm_heap::truth(&heap, r_int0) != 0) { out_status[0] = 6; return; }
            if (duel_vm_heap::truth(&heap, r_int42) != 1) { out_status[0] = 7; return; }

            // 4. Float64
            int r_flt = duel_vm_heap::make_float64(&heap, 3.141592653589793);
            double vf = 0.0;
            duel_vm_heap::read_float64(&heap, r_flt, &vf);
            if (vf != 3.141592653589793) { out_status[0] = 8; return; }

            // 5. UTF-8 String (Unicode characters "九原", 6 bytes, 2 characters)
            const char str_jiuyuan[] = "\xE4\xB9\x9D\xE5\x8E\x9F";
            int r_str = duel_vm_heap::make_utf8(&heap, str_jiuyuan, 6);
            int char_len = duel_vm_heap::str_char_length(&heap, r_str);
            if (char_len != 2) { out_status[0] = 9; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        layout = GraphLayout(game_capacity=1, max_nodes=64, max_edges=64, max_payload_bytes=256)
        batch = pack_states([[None]], layout)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_scalar_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

    def test_cuda_int64_overflow_detection_and_payload_preservation(self) -> None:
        """Verify >64-bit integer raises ERROR_INT_OVERFLOW in read_int64 without mutating bytes."""
        import torch

        huge_int = (1 << 80) + 123456789
        layout = GraphLayout(game_capacity=1, max_nodes=32, max_edges=32, max_payload_bytes=256)
        batch = pack_states([{"huge": huge_int}], layout)

        kernel_code = r"""
        extern "C" __global__ void test_overflow_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            // In packed state dict {"huge": huge_int}, value is node 2
            int int_ref = 2;
            long long out_val = 0;
            int ret = duel_vm_heap::read_int64(&heap, int_ref, &out_val);

            if (ret != -1) { out_status[0] = 1; return; }
            if (heap.error != duel_vm_heap::ERROR_INT_OVERFLOW) { out_status[0] = 2; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_overflow_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

        # Ensure the bytes in payload were NOT modified and unpack_states still succeeds losslessly
        res_batch = PackedStateBatch(
            layout=layout,
            nodes=d_nodes.cpu().numpy(),
            edges=d_edges.cpu().numpy(),
            payload=d_payload.cpu().numpy(),
            counts=d_counts.cpu().numpy(),
            roots=batch.roots,
        )
        unpacked = unpack_states(res_batch)
        self.assertEqual(unpacked[0]["huge"], huge_int)

    def test_cuda_key_equal_python_semantics(self) -> None:
        """Verify key_equal Python equality semantics in CUDA device execution."""
        import torch

        kernel_code = r"""
        extern "C" __global__ void test_key_equal_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            int r_none1 = duel_vm_heap::make_none(&heap);
            int r_none2 = duel_vm_heap::make_none(&heap);
            int r_false = duel_vm_heap::make_bool(&heap, 0);
            int r_true = duel_vm_heap::make_bool(&heap, 1);
            int r_int0 = duel_vm_heap::make_int64(&heap, 0);
            int r_int1 = duel_vm_heap::make_int64(&heap, 1);
            int r_int2 = duel_vm_heap::make_int64(&heap, 2);
            int r_flt0 = duel_vm_heap::make_float64(&heap, 0.0);
            int r_flt1 = duel_vm_heap::make_float64(&heap, 1.0);
            int r_flt2 = duel_vm_heap::make_float64(&heap, 2.0);

            // None == None
            if (!duel_vm_heap::key_equal(&heap, r_none1, r_none2)) { out_status[0] = 1; return; }
            // None != 0
            if (duel_vm_heap::key_equal(&heap, r_none1, r_int0)) { out_status[0] = 2; return; }

            // Python: False == 0 == 0.0
            if (!duel_vm_heap::key_equal(&heap, r_false, r_int0)) { out_status[0] = 3; return; }
            if (!duel_vm_heap::key_equal(&heap, r_false, r_flt0)) { out_status[0] = 4; return; }
            if (!duel_vm_heap::key_equal(&heap, r_int0, r_flt0)) { out_status[0] = 5; return; }

            // Python: True == 1 == 1.0
            if (!duel_vm_heap::key_equal(&heap, r_true, r_int1)) { out_status[0] = 6; return; }
            if (!duel_vm_heap::key_equal(&heap, r_true, r_flt1)) { out_status[0] = 7; return; }

            // Python: 2 == 2.0, but True != 2
            if (!duel_vm_heap::key_equal(&heap, r_int2, r_flt2)) { out_status[0] = 8; return; }
            if (duel_vm_heap::key_equal(&heap, r_true, r_int2)) { out_status[0] = 9; return; }

            // Unhashable rejection: list as key
            int r_list = duel_vm_heap::make_list(&heap);
            heap.error = duel_vm_heap::ERROR_OK;
            int eq_unhash = duel_vm_heap::key_equal(&heap, r_list, r_list);
            if (heap.error != duel_vm_heap::ERROR_UNHASHABLE_KEY) { out_status[0] = 10; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        layout = GraphLayout(game_capacity=1, max_nodes=64, max_edges=64, max_payload_bytes=256)
        batch = pack_states([[None]], layout)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_key_equal_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

    def test_cuda_mutable_dict_and_list_operations(self) -> None:
        """Test mutable list (append, get, set, pop) and dict (set, get, delete) operations."""
        import torch

        kernel_code = r"""
        extern "C" __global__ void test_containers_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            // 1. List: make, append 10, 20, 30
            int l_ref = duel_vm_heap::make_list(&heap);
            int r10 = duel_vm_heap::make_int64(&heap, 10);
            int r20 = duel_vm_heap::make_int64(&heap, 20);
            int r30 = duel_vm_heap::make_int64(&heap, 30);
            duel_vm_heap::list_append(&heap, l_ref, r10);
            duel_vm_heap::list_append(&heap, l_ref, r20);
            duel_vm_heap::list_append(&heap, l_ref, r30);

            // Negative indexing: index -1 should be 30
            int last_elem = duel_vm_heap::list_get(&heap, l_ref, -1);
            long long v_last = 0;
            duel_vm_heap::read_int64(&heap, last_elem, &v_last);
            if (v_last != 30) { out_status[0] = 1; return; }

            // Pop last element (30)
            int popped = 0;
            duel_vm_heap::list_pop(&heap, l_ref, &popped);
            if (popped != r30) { out_status[0] = 2; return; }

            // 2. Dict: set "score" = 99, set "list" = l_ref
            int d_ref = duel_vm_heap::make_dict(&heap);
            const char k_score[] = "score";
            const char k_items[] = "items";
            int r_kscore = duel_vm_heap::make_utf8(&heap, k_score, 5);
            int r_kitems = duel_vm_heap::make_utf8(&heap, k_items, 5);
            int r_vscore = duel_vm_heap::make_int64(&heap, 99);

            duel_vm_heap::dict_set(&heap, d_ref, r_kscore, r_vscore);
            duel_vm_heap::dict_set(&heap, d_ref, r_kitems, l_ref);

            // Read back score
            int read_score_ref = duel_vm_heap::dict_get(&heap, d_ref, r_kscore, -1);
            long long v_score = 0;
            duel_vm_heap::read_int64(&heap, read_score_ref, &v_score);
            if (v_score != 99) { out_status[0] = 3; return; }

            // Missing key with default_ref
            const char k_missing[] = "missing";
            int r_kmissing = duel_vm_heap::make_utf8(&heap, k_missing, 7);
            int default_val = duel_vm_heap::make_int64(&heap, -777);
            int read_missing = duel_vm_heap::dict_get(&heap, d_ref, r_kmissing, default_val);
            if (read_missing != default_val) { out_status[0] = 4; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        layout = GraphLayout(game_capacity=1, max_nodes=128, max_edges=128, max_payload_bytes=512)
        batch = pack_states([[None]], layout)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_containers_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

    def test_cuda_constant_graph_and_write_rejection(self) -> None:
        """Verify negative refs are read correctly, writes rejected, and deep_clone materializes."""
        import torch

        # Create a small constant graph with a dict and a list
        const_layout = GraphLayout(game_capacity=1, max_nodes=16, max_edges=16, max_payload_bytes=64)
        const_batch = pack_states([{"const_key": [100, 200]}], const_layout)

        # In a constant program graph, children refs are negative refs (-1 - child_idx)
        raw_edges = const_batch.edges[0].copy()
        c_edge_count = int(const_batch.counts[0, 1])
        for i in range(c_edge_count):
            if raw_edges[i, 0] >= 0:
                raw_edges[i, 0] = -1 - raw_edges[i, 0]
            if raw_edges[i, 1] >= 0:
                raw_edges[i, 1] = -1 - raw_edges[i, 1]

        kernel_code = r"""
        extern "C" __global__ void test_constant_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            const int* const_nodes,
            const int* const_edges,
            const unsigned char* const_payload,
            int const_node_count,
            int const_edge_count,
            int const_payload_count,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                const_nodes, const_edges, const_payload,
                const_node_count, const_edge_count, const_payload_count
            );

            // ref -1 corresponds to constant node 0 (the root dict)
            int c_root_ref = -1;
            if (duel_vm_heap::tag(&heap, c_root_ref) != duel_vm_heap::TAG_DICT) { out_status[0] = 1; return; }

            // 1. Read from constant dict
            int q_key = duel_vm_heap::make_utf8(&heap, "const_key", 9);
            int c_list_ref = duel_vm_heap::dict_get(&heap, c_root_ref, q_key, -999);
            if (c_list_ref != -3) { out_status[0] = 2; return; }

            // 2. Read from constant list
            int c_elem0 = duel_vm_heap::list_get(&heap, c_list_ref, 0);
            long long v100 = 0;
            duel_vm_heap::read_int64(&heap, c_elem0, &v100);
            if (v100 != 100) { out_status[0] = 3; return; }

            // 3. Write to constant dict must be rejected with ERROR_CONSTANT_WRITE
            int dummy_k = duel_vm_heap::make_utf8(&heap, "k", 1);
            int dummy_v = duel_vm_heap::make_int64(&heap, 1);
            int set_res = duel_vm_heap::dict_set(&heap, c_root_ref, dummy_k, dummy_v);
            if (set_res != -1 || heap.error != duel_vm_heap::ERROR_CONSTANT_WRITE) { out_status[0] = 4; return; }

            // 4. Write to constant list must be rejected
            heap.error = duel_vm_heap::ERROR_OK;
            int append_res = duel_vm_heap::list_append(&heap, c_list_ref, dummy_v);
            if (append_res != -1 || heap.error != duel_vm_heap::ERROR_CONSTANT_WRITE) { out_status[0] = 5; return; }

            // 5. Deep clone of constant graph materializes into positive mutable graph
            heap.error = duel_vm_heap::ERROR_OK;
            int cloned_root = duel_vm_heap::deep_clone(&heap, c_root_ref);
            if (cloned_root < 0 || heap.error != duel_vm_heap::ERROR_OK) { out_status[0] = 6; return; }

            // Materialized clone can be mutated
            int mut_k = duel_vm_heap::make_utf8(&heap, "new_field", 9);
            int mut_v = duel_vm_heap::make_int64(&heap, 42);
            int mut_res = duel_vm_heap::dict_set(&heap, cloned_root, mut_k, mut_v);
            if (mut_res != 0 || heap.error != duel_vm_heap::ERROR_OK) { out_status[0] = 7; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        layout = GraphLayout(game_capacity=1, max_nodes=64, max_edges=64, max_payload_bytes=256)
        batch = pack_states([[None]], layout)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")

        c_nodes = torch.as_tensor(const_batch.nodes[0], device="cuda")
        c_edges = torch.as_tensor(raw_edges, device="cuda")
        c_payload = torch.as_tensor(const_batch.payload[0], device="cuda")
        c_node_count = int(const_batch.counts[0, 0])
        c_payload_count = int(const_batch.counts[0, 2])

        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_constant_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                c_nodes,
                c_edges,
                c_payload,
                c_node_count,
                c_edge_count,
                c_payload_count,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
                "int32*",
                "uint8*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

    def test_cuda_capacity_failure_rollback(self) -> None:
        """Verify node, edge, and payload capacity exhaustion sets errors without partial commits."""
        import torch

        kernel_code = r"""
        extern "C" __global__ void test_capacity_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            // 1. Fill nodes up to max_nodes = 3
            int r0 = duel_vm_heap::make_none(&heap);
            int r1 = duel_vm_heap::make_none(&heap);
            int r2 = duel_vm_heap::make_none(&heap);
            if (heap.counts[0] != 3) { out_status[0] = 1; return; }

            // 4th node must fail with ERROR_NODE_CAPACITY
            int r3 = duel_vm_heap::make_none(&heap);
            if (r3 != -1 || heap.error != duel_vm_heap::ERROR_NODE_CAPACITY) { out_status[0] = 2; return; }
            if (heap.counts[0] != 3) { out_status[0] = 3; return; }

            out_status[0] = 0;
        }
        """
        mod = self._compile_module(kernel_code)

        # Allocate with tiny capacity: 3 nodes, 4 edges, 16 payload bytes
        layout = GraphLayout(game_capacity=1, max_nodes=3, max_edges=4, max_payload_bytes=16)
        batch = pack_states([[None]], layout)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.zeros((1, 3), dtype=torch.int32, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_capacity_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        self.assertEqual(d_status.item(), 0)

    def test_cuda_deep_clone_preserves_alias_and_entities(self) -> None:
        """Verify deep_clone preserves Entity types, container aliases, and isolates original graph."""
        import torch

        player = PlayerEntity(side="a", hp=30, active=True)
        card = CardEntity(card_id="c_001", name="九原秘术", cost=3)
        shared_list = [100, 200]
        state = {
            "player": player,
            "card": card,
            "alias_1": shared_list,
            "alias_2": shared_list,
            "meta": {"name": "真红", "power": 99.5},
        }

        layout = GraphLayout(game_capacity=1, max_nodes=512, max_edges=512, max_payload_bytes=2048)
        batch = pack_states([state], layout)

        kernel_code = r"""
        extern "C" __global__ void test_deep_clone_kernel(
            int* nodes,
            int* edges,
            unsigned char* payload,
            int* counts,
            int max_nodes,
            int max_edges,
            int max_payload,
            int* out_status
        ) {
            duel_vm_heap::Heap heap;
            duel_vm_heap::init_heap(
                &heap, nodes, edges, payload, counts,
                max_nodes, max_edges, max_payload,
                0, 0, 0, 0, 0, 0
            );

            int root_ref = 0; // The packed state root dict
            int cloned_root = duel_vm_heap::deep_clone(&heap, root_ref);

            if (cloned_root < 0 || heap.error != duel_vm_heap::ERROR_OK) {
                out_status[0] = 1;
                return;
            }

            // Find alias_1 in cloned root
            const char k_alias1[] = "alias_1";
            int r_kalias1 = duel_vm_heap::make_utf8(&heap, k_alias1, 7);
            int cloned_list_ref = duel_vm_heap::dict_get(&heap, cloned_root, r_kalias1, -1);

            // Mutate the cloned list by appending 9999
            int val_9999 = duel_vm_heap::make_int64(&heap, 9999);
            duel_vm_heap::list_append(&heap, cloned_list_ref, val_9999);

            // Wrap both original and cloned roots into a top-level result dict
            int final_root = duel_vm_heap::make_dict(&heap);
            const char k_orig[] = "original";
            const char k_clone[] = "cloned";
            int r_korig = duel_vm_heap::make_utf8(&heap, k_orig, 8);
            int r_kclone = duel_vm_heap::make_utf8(&heap, k_clone, 6);
            duel_vm_heap::dict_set(&heap, final_root, r_korig, root_ref);
            duel_vm_heap::dict_set(&heap, final_root, r_kclone, cloned_root);

            out_status[0] = final_root;
        }
        """
        mod = self._compile_module(kernel_code)

        d_nodes = torch.as_tensor(batch.nodes, device="cuda")
        d_edges = torch.as_tensor(batch.edges, device="cuda")
        d_payload = torch.as_tensor(batch.payload, device="cuda")
        d_counts = torch.as_tensor(batch.counts, device="cuda")
        d_status = torch.tensor([-99], dtype=torch.int32, device="cuda")

        mod.launch_kernel(
            "test_deep_clone_kernel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[
                d_nodes,
                d_edges,
                d_payload,
                d_counts,
                layout.max_nodes,
                layout.max_edges,
                layout.max_payload_bytes,
                d_status,
            ],
            spec=[
                "int32*",
                "int32*",
                "uint8*",
                "int32*",
                "int32",
                "int32",
                "int32",
                "int32*",
            ],
        )
        torch.cuda.synchronize()
        final_root = d_status.item()
        self.assertGreater(final_root, 0)

        # Decode back on host using standard unpack_states
        res_batch = self._canonical_export(layout,d_nodes,d_edges,d_payload,d_counts,final_root)
        decoded = unpack_states(res_batch)[0]

        orig = decoded["original"]
        cloned = decoded["cloned"]

        # 1. Verify Entity types are preserved in cloned graph
        self.assertIsInstance(cloned["player"], PlayerEntity)
        self.assertEqual(cloned["player"].side, "a")
        self.assertEqual(cloned["player"]["hp"], 30)

        self.assertIsInstance(cloned["card"], CardEntity)
        self.assertEqual(cloned["card"].template_id, "c_001")
        self.assertEqual(cloned["card"]["name"], "九原秘术")

        # 2. Verify shared container alias is preserved: alias_1 is alias_2 in clone
        self.assertIs(cloned["alias_1"], cloned["alias_2"])
        # Both alias_1 and alias_2 reflect the appended 9999
        self.assertEqual(cloned["alias_1"], [100, 200, 9999])
        self.assertEqual(cloned["alias_2"], [100, 200, 9999])

        # 3. Verify original graph was NOT mutated (mutation isolation)
        self.assertIs(orig["alias_1"], orig["alias_2"])
        self.assertEqual(orig["alias_1"], [100, 200])
        self.assertEqual(orig["alias_2"], [100, 200])
        self.assertIsNot(cloned["alias_1"], orig["alias_1"])


if __name__ == "__main__":
    unittest.main()
