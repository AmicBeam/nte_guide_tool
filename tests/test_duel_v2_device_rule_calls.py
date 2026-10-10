"""Device rule graph-call tests. Not a full adapter, game, or CUDA rule gate."""
import os
import sys
from pathlib import Path
import unittest


class DeviceRuleCallsContractTest(unittest.TestCase):
    def test_rejects_cpu_callbacks_missing_entries_and_stays_unqualified(self):
        from app.modules.card_game.rl.batched_duel.device_rule_calls import (
            ALLOWED_ENTRIES, DeviceRuleGraphCalls,
        )
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        self.assertFalse(DeviceRuleGraphCalls.supports_full_duel_rules)
        for name in (
            'metadata_device', 'decision_device', 'rule_decision_device',
            'value_observation_device', 'teacher_continue_device',
        ):
            self.assertFalse(hasattr(DeviceRuleGraphCalls, name))
        self.assertEqual(
            ALLOWED_ENTRIES,
            frozenset(('clean','teacher_continue','sample_world','simulate_action','apply_action')))
        layout = GraphLayout(2, 8, 8, 32)
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            DeviceRuleGraphCalls(object(), object(), rows=2, compact_layout=layout)
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            DeviceRuleGraphCalls(None, None, rows=2, lanes=2, compact_layout=layout)


def _probe_program():
    fixture = os.environ.get('NTE_DEVICE_RULE_CALLS_TEST_PROGRAM')
    if fixture:
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest(
            'CPython 3.10 capture helper or NTE_DEVICE_RULE_CALLS_TEST_PROGRAM fixture required')
    from app.modules.card_game.rl.rule_ir.portable_program import (
        DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program,
    )
    return capture_program(
        {'clean': _clean_probe, 'sample_world': _sample_world_probe},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
        scope_character_ids=None, scope_card_ids=None)


def _clean_probe(state):
    state['cleaned'] = True
    return state


def _sample_world_probe(state, viewer, seed):
    state['viewer'] = viewer
    state['seed'] = seed
    return state


def _limbs(value):
    import numpy as np
    return np.frombuffer((value).to_bytes(40, 'little', signed=False), dtype='<i8').copy()


def _graph(states, device, max_nodes=64, max_edges=64, max_payload=256):
    import torch
    import numpy as np
    from app.modules.card_game.rl.batched_duel.codec import pack_states
    from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
    from app.modules.card_game.rl.batched_duel.schema import GraphLayout
    packed = pack_states(states, GraphLayout(len(states), max_nodes, max_edges, max_payload))
    return DeviceGraph(packed.layout, *(
        torch.tensor(np.array(getattr(packed, name), copy=True), device=device)
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))


def _store(device, slots=8):
    from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
    return CompactDeviceGraphStore(
        slots=slots, node_pages=32, edge_pages=32, payload_pages=32,
        node_page_bytes=128, edge_page_bytes=128, payload_page_bytes=256,
        max_node_pages_per_slot=8, max_edge_pages_per_slot=8, max_payload_pages_per_slot=8,
        device=device)


def _decode_row(store, states, row, layout):
    import torch
    from app.modules.card_game.rl.batched_duel.codec import unpack_states
    from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
    from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
    from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch
    dest = DeviceGraph(
        layout,
        torch.zeros((1, layout.max_nodes, 4), dtype=torch.int32, device=store.device),
        torch.zeros((1, layout.max_edges, 2), dtype=torch.int32, device=store.device),
        torch.zeros((1, layout.max_payload_bytes), dtype=torch.uint8, device=store.device),
        torch.zeros((1, 3), dtype=torch.int32, device=store.device),
        torch.zeros((1,), dtype=torch.int32, device=store.device))
    single = DeviceStateBatch(
        store, states.slots[row:row + 1].contiguous(), states.generations[row:row + 1].contiguous(),
        states.owners[row:row + 1].contiguous(), states.active[row:row + 1].contiguous(),
        states.errors[row:row + 1].contiguous())
    gathered = store.gather_into_device(single, dest)
    if int(gathered.errors[0].cpu()) != 0:
        raise AssertionError('gather failed: %s' % gathered.errors.cpu().tolist())
    packed = PackedStateBatch(GraphLayout(1, layout.max_nodes, layout.max_edges, layout.max_payload_bytes), *(
        getattr(dest, name).detach().cpu().numpy()
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))
    return unpack_states(packed)[0]


class DeviceRuleCallsCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')

    def test_rejects_missing_frozen_entries_and_uncaptured_calls(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, VmCapabilityError
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        program = _probe_program()
        store = _store('cuda')
        self.addCleanup(store.close)
        with CudaVm(lower_program(program)) as vm:
            helper = DeviceRuleGraphCalls(
                vm, store, rows=3, lanes=2, seconds=30,
                compact_layout=GraphLayout(2, 64, 64, 256))
            self.addCleanup(helper.close)
            self.assertFalse(helper.supports_full_duel_rules)
            with self.assertRaises(ValueError):
                helper.prepare_call('public_legal_decision')
            with self.assertRaises(ValueError):
                helper.prepare_call('acting_side')
            with self.assertRaises(VmCapabilityError):
                helper.prepare_call('apply_action')
            with self.assertRaises(VmCapabilityError):
                helper.prepare_call('simulate_action')
            plan = helper.prepare_call('clean')
            owners = torch.arange(3, dtype=torch.int64, device=vm.device)
            active = torch.ones(3, dtype=torch.bool, device=vm.device)
            graph = _graph([{'n': i} for i in range(3)], vm.device)
            inputs = store.put_device(graph, owners, active)
            with self.assertRaises(RuntimeError):
                plan.capture_device(inputs)
            helper.release_device(inputs)

    def test_actual_cuda_masked_clean_and_sample_cohort_routing(self):
        import torch
        from contextlib import ExitStack
        from hashlib import sha256
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, DeviceGraph
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        program = _probe_program()
        device = torch.device('cuda', torch.cuda.current_device())
        store = _store(device, slots=16)
        self.addCleanup(store.close)
        compact = GraphLayout(2, 64, 64, 256)
        root = (1 << 256) | int.from_bytes(sha256(b'recovery:20261011:17:a').digest(), 'big')
        with CudaVm(lower_program(program)) as vm:
            helper = DeviceRuleGraphCalls(
                vm, store, rows=3, lanes=2, seconds=30, compact_layout=compact)
            self.addCleanup(helper.close)
            graph = _graph(
                [{'n': 0, 'keep': True}, {'n': 1, 'keep': True}, {'n': 2, 'keep': True}],
                device)
            owners = torch.arange(11, 14, dtype=torch.int64, device=device)
            active = torch.tensor([True, False, True], device=device)
            inputs = store.put_device(graph, owners, active)
            inputs.errors[2] = 107
            input_slots = inputs.slots.clone()
            clean_plan = helper.prepare_call('clean')
            sample_plan = helper.prepare_call('sample_world')
            actors = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
            seeds = torch.tensor(
                [_limbs(1 << 256), _limbs(0), _limbs(root + 31 * 104729)],
                dtype=torch.int64, device=device)
            captured = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                stack.enter_context(patch.object(
                    DeviceGraph, 'fork', side_effect=AssertionError('Heap fork')))
                for name in ('cpu', 'numpy', 'item', 'tolist', '__bool__'):
                    stack.enter_context(patch.object(
                        torch.Tensor, name, side_effect=AssertionError('Host selector')))
                with torch.cuda.graph(captured, capture_error_mode='thread_local'):
                    cleaned = clean_plan.capture_device(inputs)
                    sampled = sample_plan.capture_device(
                        inputs, actors=actors, seeds=seeds)
            captured.replay()
            torch.cuda.synchronize()
            self.assertGreaterEqual(int(cleaned.slots[0].cpu()), 0)
            self.assertEqual(int(cleaned.slots[1].cpu()), -1)
            self.assertEqual(int(cleaned.slots[2].cpu()), -1)
            self.assertEqual(int(cleaned.errors[2].cpu()), 107)
            self.assertTrue(bool(cleaned.active[0].cpu()))
            self.assertFalse(bool(cleaned.active[1].cpu()))
            self.assertFalse(bool(cleaned.active[2].cpu()))
            decoded = _decode_row(store, cleaned, 0, compact)
            self.assertEqual(decoded['n'], 0)
            self.assertTrue(decoded['cleaned'])
            self.assertTrue(decoded['keep'])
            self.assertTrue(torch.equal(inputs.slots, input_slots))
            self.assertEqual(int(inputs.errors[2].cpu()), 107)
            self.assertGreaterEqual(int(sampled.slots[0].cpu()), 0)
            self.assertEqual(int(sampled.slots[1].cpu()), -1)
            self.assertEqual(int(sampled.slots[2].cpu()), -1)
            sampled_row = _decode_row(store, sampled, 0, compact)
            self.assertEqual(sampled_row['viewer'], 'a')
            self.assertEqual(sampled_row['seed'], 1 << 256)
            self.assertEqual(sampled_row['n'], 0)
            live_before = int(store.live.sum().cpu())
            helper.release_device(cleaned)
            helper.release_device(sampled)
            torch.cuda.synchronize()
            self.assertEqual(int(store.live.sum().cpu()), live_before - 2)
            helper.release_device(inputs)
            torch.cuda.synchronize()
            self.assertEqual(int(store.live.sum().cpu()), 0)


if __name__ == '__main__':
    unittest.main()

