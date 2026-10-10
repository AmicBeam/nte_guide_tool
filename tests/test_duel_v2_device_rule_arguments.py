"""Device actor/seed argument binder tests. Not complete-rule or 1600-root evidence."""
import ast
import inspect
from hashlib import sha256
from unittest.mock import patch
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel import device_rule_arguments as dra
from app.modules.card_game.rl.batched_duel.codec import _int_to_minimal_signed_bytes, pack_states, unpack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
from app.modules.card_game.rl.batched_duel.device_rule_arguments import DeviceRuleArgumentBinder
from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch
from app.modules.card_game.rl.batched_duel.vm_heap import (
    ERROR_NODE_CAPACITY, ERROR_OK, ERROR_PAYLOAD_CAPACITY, ERROR_TYPE_MISMATCH, ERROR_INVALID_REF,
)


ROOT = (1 << 256) | int.from_bytes(sha256(b'recovery:20261011:17:a').digest(), 'big')
WIDE = (
    0, 1, 127, 128, 1 << 63, 1 << 255, 1 << 256, ROOT,
    ROOT + 31 * 104729, (1 << 257) - 1, (1 << 257) - 1 + 31 * 104729,
)


def _limbs(value):
    return np.frombuffer((value).to_bytes(40, 'little', signed=False), dtype='<i8').copy()


def _graph(states, device, max_nodes=64, max_edges=64, max_payload=128):
    import torch
    packed = pack_states(states, GraphLayout(len(states), max_nodes, max_edges, max_payload))
    return DeviceGraph(packed.layout, *(
        torch.tensor(np.array(getattr(packed, name), copy=True), device=device)
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))


def _clone_counts(graph):
    return graph.nodes.clone(), graph.edges.clone(), graph.payload.clone(), graph.counts.clone(), graph.roots.clone()


def _packed_row(arena, row, root):
    layout = GraphLayout(1, arena.layout.max_nodes, arena.layout.max_edges, arena.layout.max_payload_bytes)
    arrays = []
    for name, dtype in (('nodes', np.int32), ('edges', np.int32), ('payload', np.uint8), ('counts', np.int32)):
        arrays.append(np.ascontiguousarray(getattr(arena, name)[row:row + 1].detach().cpu().numpy().astype(dtype, copy=False)))
    roots = np.ascontiguousarray(np.array([int(root)], dtype=np.int32))
    return PackedStateBatch(layout, *arrays, roots)


def _decode_ref(arena, row, ref):
    tag, offset, count, extra = (int(v) for v in arena.nodes[row, int(ref)].detach().cpu().numpy())
    width = max(count, 1)
    nodes = np.zeros((1, 1, 4), dtype=np.int32)
    nodes[0, 0] = (tag, 0, count, extra)
    payload = np.zeros((1, width), dtype=np.uint8)
    if count:
        payload[0, :count] = arena.payload[row, offset:offset + count].detach().cpu().numpy()
    return unpack_states(PackedStateBatch(
        GraphLayout(1, 1, 1, width),
        nodes,
        np.zeros((1, 1, 2), dtype=np.int32),
        payload,
        np.array([[1, 0, count]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    ))[0]


def _unpack_original(arena, row, counts, root):
    packed = _packed_row(arena, row, root)
    packed.counts[0] = np.ascontiguousarray(np.array(counts, dtype=np.int32))
    return unpack_states(packed)[0]


class DeviceRuleArgumentsCpuTest(unittest.TestCase):
    def test_inplace_scratch_binding_uses_preallocated_outputs_without_fork(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
        graph=_graph([{'unknown':7},{'unknown':8}],'cpu')
        counts=graph.counts.clone()
        refs=torch.empty((2,2),dtype=torch.int32)
        errors=torch.empty(2,dtype=torch.int32)
        with self.binder() as binder, patch.object(DeviceGraph,'fork',side_effect=AssertionError('Heap fork')):
            result=binder.bind_into_device(graph,torch.tensor([1,9]),torch.tensor([True,True]),
                                          argument_refs=refs,errors=errors)
        self.assertIs(result.arena,graph)
        self.assertIs(result.argument_refs,refs)
        self.assertEqual(errors.tolist(),[0,ERROR_TYPE_MISMATCH])
        self.assertEqual(_decode_ref(graph,0,int(refs[0,1])),'b')
        torch.testing.assert_close(graph.counts[1],counts[1])

    def test_invalid_live_root_is_rejected_without_binding(self):
        import torch
        graph=_graph([{'v':1},{'v':2}],'cpu')
        graph.roots[1]=999
        with DeviceRuleArgumentBinder(device='cpu',allow_cpu_reference=True) as binder:
            result=binder.bind_actor_device(graph,torch.tensor([0,1]),torch.tensor([True,True]))
        self.assertEqual(result.errors.tolist(),[0,ERROR_INVALID_REF])
        self.assertEqual(int(result.argument_refs[1,1]),-1)
    def binder(self):
        return DeviceRuleArgumentBinder(device='cpu', allow_cpu_reference=True)

    def test_module_import_does_not_import_torch_or_cuda(self):
        tree = ast.parse(inspect.getsource(dra))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split('.')[0]]
            self.assertNotIn('torch', names)
            self.assertNotIn('cuda', names)
        self.assertFalse(DeviceRuleArgumentBinder.supports_full_duel_rules)

    def test_cpu_requires_explicit_oracle_and_skips_actual_cuda(self):
        with self.assertRaises(ValueError):
            DeviceRuleArgumentBinder(device='cpu')
        binder = self.binder()
        self.assertTrue(binder._cpu)
        self.assertIsNone(binder.module)
        graph = _graph([0], 'cpu')
        import torch
        with self.assertRaises(ValueError):
            binder.bind_actor_seed_device(
                graph, torch.tensor([0], dtype=torch.int64),
                torch.zeros((1, 4), dtype=torch.int64), torch.tensor([True]))

    def test_wide_257bit_sim31_carry_zero_sign_byte_and_signed_limbs(self):
        import torch
        binder = self.binder()
        rows = len(WIDE)
        shared = [9]
        states = [{'keep': shared, 'alias': shared} for _ in WIDE]
        graph = _graph(states, 'cpu', max_payload=96)
        before = _clone_counts(graph)
        actors = torch.tensor([i % 2 for i in range(rows)], dtype=torch.int64)
        seeds = torch.tensor(np.stack([_limbs(value) for value in WIDE]), dtype=torch.int64)
        self.assertEqual(int(seeds[4, 0]), -(1 << 63))
        active = torch.ones(rows, dtype=torch.bool)
        bound = binder.bind_actor_seed_device(graph, actors, seeds, active)
        self.assertEqual(bound.errors.tolist(), [ERROR_OK] * rows)
        self.assertEqual(tuple(bound.argument_refs.shape), (rows, 3))
        original = _unpack_original(graph, 0, before[3][0].cpu().numpy(), int(graph.roots[0]))
        copied = _unpack_original(bound.arena, 0, before[3][0].cpu().numpy(), int(bound.arena.roots[0]))
        self.assertEqual(original, copied)
        self.assertIs(copied['keep'], copied['alias'])
        for name, original in zip(('nodes', 'edges', 'payload', 'counts', 'roots'), before):
            torch.testing.assert_close(getattr(graph, name), original)
        for row, value in enumerate(WIDE):
            side = _decode_ref(bound.arena, row, int(bound.argument_refs[row, 1]))
            seed = _decode_ref(bound.arena, row, int(bound.argument_refs[row, 2]))
            raw = _int_to_minimal_signed_bytes(value)
            node = bound.arena.nodes[row, int(bound.argument_refs[row, 2])]
            payload = bytes(bound.arena.payload[row, int(node[1]):int(node[1]) + int(node[2])].numpy())
            self.assertEqual(side, 'b' if row % 2 else 'a')
            self.assertEqual(seed, value)
            self.assertEqual(payload, raw)
            self.assertNotEqual(len(payload), 40)
            if value == 1 << 255:
                self.assertEqual(payload[-1], 0)
                self.assertEqual(payload[-2], 0x80)
            if value == 1 << 63:
                self.assertEqual(payload, b'\x00\x00\x00\x00\x00\x00\x00\x80\x00')

    def test_bind_actor_does_not_append_dummy_seed(self):
        import torch
        binder = self.binder()
        graph = _graph([0, 1], 'cpu')
        before = graph.counts.clone()
        bound = binder.bind_actor_device(
            graph, torch.tensor([0, 1], dtype=torch.int64), torch.tensor([1, 1], dtype=torch.uint8))
        self.assertEqual(tuple(bound.argument_refs.shape), (2, 2))
        self.assertEqual(bound.errors.tolist(), [0, 0])
        self.assertEqual(_decode_ref(bound.arena, 0, int(bound.argument_refs[0, 1])), 'a')
        self.assertEqual(_decode_ref(bound.arena, 1, int(bound.argument_refs[1, 1])), 'b')
        self.assertEqual((bound.arena.counts - before).tolist(), [[1, 0, 1], [1, 0, 1]])

    def test_active_invalid_and_capacity_isolation_leave_source_and_failed_rows(self):
        import torch
        binder = self.binder()
        graph = _graph([0, 1, (1, 2, 3)], 'cpu', max_nodes=4, max_payload=8)
        source = _clone_counts(graph)
        actors = torch.tensor([0, 3, 1], dtype=torch.int64)
        seeds = torch.tensor(np.stack([_limbs(1), _limbs(2), _limbs(1 << 256)]), dtype=torch.int64)
        active = torch.tensor([True, True, False])
        bound = binder.bind_actor_seed_device(graph, actors, seeds, active)
        self.assertEqual(bound.errors.tolist(), [ERROR_OK, ERROR_TYPE_MISMATCH, ERROR_OK])
        self.assertEqual(_decode_ref(bound.arena, 0, int(bound.argument_refs[0, 2])), 1)
        self.assertEqual(bound.argument_refs[1, 1:].tolist(), [-1, -1])
        self.assertEqual(bound.argument_refs[2, 1:].tolist(), [-1, -1])
        torch.testing.assert_close(bound.arena.counts[1], graph.counts[1])
        torch.testing.assert_close(bound.arena.counts[2], graph.counts[2])
        for name, original in zip(('nodes', 'edges', 'payload', 'counts', 'roots'), source):
            torch.testing.assert_close(getattr(graph, name), original)
        tight = _graph([0], 'cpu', max_nodes=1, max_payload=4)
        overflow = binder.bind_actor_seed_device(
            tight, torch.tensor([0], dtype=torch.int64),
            torch.tensor(_limbs(1 << 256), dtype=torch.int64)[None, :],
            torch.tensor([True]))
        self.assertIn(int(overflow.errors[0]), (ERROR_NODE_CAPACITY, ERROR_PAYLOAD_CAPACITY))
        self.assertEqual(overflow.argument_refs[0, 1:].tolist(), [-1, -1])
        torch.testing.assert_close(overflow.arena.counts, tight.counts)


class DeviceRuleArgumentsCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; CPU tests already skip CUDA compilation')
        self.torch = torch
        self.binder = DeviceRuleArgumentBinder(device='cuda')
        self.addCleanup(self.binder.close)

    def _hot_path(self):
        from contextlib import ExitStack
        t = self.torch
        stack = ExitStack()
        for name, label in (
            ('cpu', 'CPU copy'), ('numpy', 'NumPy'), ('item', 'scalar'),
            ('tolist', 'list'), ('__bool__', 'truth'),
        ):
            stack.enter_context(patch.object(
                t.Tensor, name, side_effect=AssertionError('Hot-path ' + label)))
        return stack

    def test_actual_cuda_inplace_capture_reuses_bound_heap_and_outputs(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
        t=self.torch
        graph=_graph([{'unknown':7},{'unknown':8}],'cuda')
        original=graph.fork()
        actors=t.tensor([0,1],dtype=t.int64,device='cuda')
        active=t.tensor([True,False],device='cuda')
        refs=t.empty((2,2),dtype=t.int32,device='cuda')
        errors=t.empty(2,dtype=t.int32,device='cuda')
        args=dict(argument_refs=refs,errors=errors)
        self.binder.bind_into_device(graph,actors,active,**args);t.cuda.synchronize()
        captured=t.cuda.CUDAGraph()
        with self._hot_path(), patch.object(DeviceGraph,'fork',side_effect=AssertionError('Heap fork')):
            with t.cuda.graph(captured):
                result=self.binder.bind_into_device(graph,actors,active,**args)
        for name in ('nodes','edges','payload','counts','roots'):
            getattr(graph,name).copy_(getattr(original,name))
        actors.copy_(t.tensor([1,0],dtype=t.int64,device='cuda'))
        active.fill_(True)
        captured.replay();t.cuda.synchronize()
        self.assertIs(result.arena,graph)
        self.assertEqual(errors.cpu().tolist(),[0,0])
        self.assertEqual([_decode_ref(graph,row,int(refs[row,1].cpu())) for row in range(2)],['b','a'])

    def test_actual_cuda_decodes_bound_args_only_after_completed_batch(self):
        t = self.torch
        values = (0, 1 << 63, 1 << 255, 1 << 256, ROOT, ROOT + 31 * 104729)
        graph = _graph([{'n': i} for i in range(len(values))], 'cuda', max_payload=96)
        source = _clone_counts(graph)
        actors = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.int64, device='cuda')
        seeds = t.tensor(np.stack([_limbs(value) for value in values]), dtype=t.int64, device='cuda')
        active = t.tensor([1, 1, 1, 1, 1, 0], dtype=t.uint8, device='cuda')
        with self._hot_path():
            bound = self.binder.bind_actor_seed_device(graph, actors, seeds, active)
            self.assertEqual(bound.argument_refs.device.type, 'cuda')
            self.assertEqual(bound.errors.device.type, 'cuda')
        t.cuda.synchronize()
        self.assertEqual(bound.errors.cpu().tolist(), [0, 0, 0, 0, 0, 0])
        for row, value in enumerate(values[:-1]):
            self.assertEqual(_decode_ref(bound.arena, row, int(bound.argument_refs[row, 1].cpu())), 'b' if row % 2 else 'a')
            self.assertEqual(_decode_ref(bound.arena, row, int(bound.argument_refs[row, 2].cpu())), value)
            node = bound.arena.nodes[row, int(bound.argument_refs[row, 2].cpu())].cpu()
            payload = bytes(bound.arena.payload[row, int(node[1]):int(node[1]) + int(node[2])].cpu().numpy())
            self.assertEqual(payload, _int_to_minimal_signed_bytes(value))
            self.assertNotEqual(len(payload), 40)
        self.assertEqual(bound.argument_refs[5, 1:].cpu().tolist(), [-1, -1])
        for name, original in zip(('nodes', 'edges', 'payload', 'counts', 'roots'), source):
            t.testing.assert_close(getattr(graph, name), original)

    def test_actual_cuda_capture_has_no_host_scalar_reads(self):
        t = self.torch
        graph = _graph([0, 1], 'cuda', max_payload=64)
        actors = t.tensor([0, 1], dtype=t.int64, device='cuda')
        seeds = t.tensor(np.stack([_limbs(1 << 256), _limbs(ROOT + 31 * 104729)]), dtype=t.int64, device='cuda')
        active = t.tensor([True, True], device='cuda')
        bound = self.binder.bind_actor_seed_device(graph, actors, seeds, active)
        captured = t.cuda.CUDAGraph()
        with self._hot_path():
            with t.cuda.graph(captured):
                bound = self.binder.bind_actor_seed_device(graph, actors, seeds, active)
            captured.replay()
        t.cuda.synchronize()
        self.assertEqual(bound.errors.cpu().tolist(), [0, 0])
        self.assertEqual(_decode_ref(bound.arena, 0, int(bound.argument_refs[0, 2].cpu())), 1 << 256)
        self.assertEqual(_decode_ref(bound.arena, 1, int(bound.argument_refs[1, 2].cpu())), ROOT + 31 * 104729)
        self.assertEqual(_decode_ref(bound.arena, 1, int(bound.argument_refs[1, 1].cpu())), 'b')


if __name__ == '__main__':
    unittest.main()
