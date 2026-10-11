"""CPU/CUDA tests for bounded GPU RAW physical replay; not full-rules evidence."""
import inspect
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


def apply_action(state, side, action):
    if 'delta' not in action:
        raise ValueError('synthetic apply failure')
    state['n'] = state['n'] + action['delta']
    state['side'] = side
    return state


def clean(state):
    state['cleaned'] = True
    return state


def _probe_program():
    fixture = os.environ.get('NTE_DEVICE_RAW_REPLAY_TEST_PROGRAM')
    if fixture:
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest(
            'CPython 3.10 capture helper or NTE_DEVICE_RAW_REPLAY_TEST_PROGRAM fixture required')
    from app.modules.card_game.rl.rule_ir.portable_program import (
        DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program,
    )
    return capture_program(
        {'apply_action': apply_action, 'clean': clean},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
        scope_character_ids=None, scope_card_ids=None)


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


def _store(device, slots=32):
    from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
    return CompactDeviceGraphStore(
        slots=slots, node_pages=64, edge_pages=64, payload_pages=64,
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
        # Post-batch oracle reads a preserved lease even when execution failed.
        # The store still checks real owner/generation; qualification retains
        # result.errors unchanged.
        torch.zeros_like(states.errors[row:row + 1]))
    gathered = store.gather_into_device(single, dest)
    if int(gathered.errors[0].cpu()) != 0:
        raise AssertionError('gather failed: %s' % gathered.errors.cpu().tolist())
    packed = PackedStateBatch(GraphLayout(1, layout.max_nodes, layout.max_edges, layout.max_payload_bytes), *(
        getattr(dest, name).detach().cpu().numpy()
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))
    return unpack_states(packed)[0]


def _expected(original, moves):
    state = dict(original)
    for side, action in moves:
        apply_action(state, side, action)
        clean(state)
    return state


class DeviceRawReplayContractTest(unittest.TestCase):
    def test_refuses_cpu_readiness_and_host_step_getitem(self):
        from app.modules.card_game.rl.batched_search.device_raw_replay import (
            CAVEAT, DeviceRawReplay, DeviceRawReplayResult,
        )
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        self.assertFalse(DeviceRawReplay.supports_full_duel_rules)
        self.assertFalse(hasattr(DeviceRawReplay, 'ready'))
        self.assertFalse(hasattr(DeviceRawReplay, 'device_validation_receipt'))
        self.assertNotIn('game_complete', DeviceRawReplayResult.__dataclass_fields__)
        self.assertIn('replay_sequence_complete', CAVEAT)
        self.assertNotIn('qualify full', CAVEAT.lower())
        layout = GraphLayout(2, 8, 8, 32)
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            DeviceRawReplay(object(), object(), object(), object(),
                            max_steps=1, compact_layout=layout, lanes=1)
        source = inspect.getsource(DeviceRawReplay)
        self.assertIn('index_select(0, self._step.reshape(1))', source)
        self.assertNotIn('self._record_index_plan[self._step]', source)
        self.assertNotIn('game_complete', source)
        self.assertIn('prepare_call', source)
        self.assertIn("'apply_action'", source)
        self.assertIn("'clean'", source)

    def test_public_api_is_capture_replay_result_device(self):
        from app.modules.card_game.rl.batched_search.device_raw_replay import DeviceRawReplay
        for name in ('capture', 'replay', 'result_device', 'close'):
            self.assertTrue(callable(getattr(DeviceRawReplay, name)))
        self.assertTrue(inspect.isclass(DeviceRawReplay))


class DeviceRawReplayCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')

    def _cohort(self, originals, *, rows=3, lanes=2, max_steps=2, slots=32):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
        from app.modules.card_game.rl.batched_search.device_raw_replay import DeviceRawReplay
        device = torch.device('cuda', torch.cuda.current_device())
        source = _store(device, slots=slots)
        replay_store = _store(device, slots=slots)
        action_store = _store(device, slots=slots)
        self.addCleanup(source.close)
        self.addCleanup(replay_store.close)
        self.addCleanup(action_store.close)
        compact = GraphLayout(lanes, 64, 64, 256)
        program = _probe_program()
        vm = CudaVm(lower_program(program))
        self.addCleanup(vm.close)
        calls = DeviceRuleGraphCalls(
            vm, replay_store, rows=rows, lanes=lanes, seconds=30, compact_layout=compact)
        self.addCleanup(calls.close)
        tape = DeviceRawActionTape(
            rows, 8, 256, 256, 1024, device=device)
        self.addCleanup(tape.close)
        graph = _graph(originals, device)
        owners = torch.arange(10, 10 + rows, dtype=torch.int64, device=device)
        seeds = torch.arange(100, 100 + rows, dtype=torch.int64, device=device)
        put_active = torch.ones(rows, dtype=torch.bool, device=device)
        roots = source.put_device(graph, owners, put_active)
        return {
            'device': device, 'source': source, 'replay_store': replay_store,
            'action_store': action_store, 'compact': compact, 'calls': calls,
            'tape': tape, 'owners': owners, 'seeds': seeds, 'roots': roots,
            'replay_cls': DeviceRawReplay, 'rows': rows, 'lanes': lanes,
            'max_steps': max_steps,
        }

    def test_actual_cuda_three_row_apply_clean_parity(self):
        import torch
        from contextlib import ExitStack
        from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
        originals = [{'n': 10, 'id': 0}, {'n': 20, 'id': 1}, {'n': 30, 'id': 2}]
        env = self._cohort(originals)
        device = env['device']
        tape = env['tape']
        roots = env['roots']
        roots.active.copy_(torch.tensor([True, False, True], device=device))
        tape.begin_device(env['owners'], env['seeds'])
        step0 = _graph([{'delta': 1}, {'delta': 0}, {'delta': 2}], device)
        err0 = tape.append_device(
            step0, torch.arange(3, dtype=torch.int64, device=device), env['owners'],
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.tensor([0, 0, 1], dtype=torch.int64, device=device),
            torch.tensor([True, False, True], device=device))
        self.assertEqual(err0.cpu().tolist(), [0, 0, 0])
        step1 = _graph([{'delta': 5}, {'delta': 0}, {'delta': 0}], device)
        err1 = tape.append_device(
            step1, torch.arange(3, dtype=torch.int64, device=device), env['owners'],
            torch.tensor([1, 0, 0], dtype=torch.int64, device=device),
            torch.ones(3, dtype=torch.int64, device=device),
            torch.zeros(3, dtype=torch.int64, device=device),
            torch.tensor([True, False, False], device=device))
        self.assertEqual(int(err1[0].cpu()), 0)
        self.assertEqual(int(tape._record_count.cpu()), 3)
        source_live = int(env['source'].live.sum().cpu())
        replay = env['replay_cls'](
            env['calls'], tape, roots, env['action_store'],
            max_steps=2, compact_layout=env['compact'], lanes=2, seconds=30)
        self.addCleanup(replay.close)
        plan = torch.tensor([[0, -1, 1], [2, -1, -1]], dtype=torch.int64, device=device)
        counts = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)
        replay.capture(plan, env['seeds'], counts)
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                DeviceGraph, 'fork', side_effect=AssertionError('Heap fork')))
            for name in ('cpu', 'numpy', 'item', 'tolist', '__bool__'):
                stack.enter_context(patch.object(
                    torch.Tensor, name, side_effect=AssertionError('Host selector')))
            replay.replay()
        torch.cuda.synchronize()
        result = replay.result_device()
        self.assertFalse(hasattr(result, 'game_complete'))
        self.assertEqual(result.complete.cpu().tolist(), [True, False, True])
        self.assertFalse(bool(result.metadata_checked.any().cpu()))
        self.assertEqual(result.errors.cpu().tolist(), [0, 0, 0])
        self.assertEqual(result.steps.cpu().tolist(), [2, 0, 1])
        self.assertEqual(int(result.states.slots[1].cpu()), -1)
        decoded0 = _decode_row(env['replay_store'], result.states, 0, env['compact'])
        decoded2 = _decode_row(env['replay_store'], result.states, 2, env['compact'])
        self.assertEqual(decoded0, _expected(originals[0], [('a', {'delta': 1}), ('a', {'delta': 5})]))
        self.assertEqual(decoded2, _expected(originals[2], [('b', {'delta': 2})]))
        self.assertEqual(int(env['source'].live.sum().cpu()), source_live)
        replay.replay()
        torch.cuda.synchronize()
        again = replay.result_device()
        self.assertEqual(again.errors.cpu().tolist(), [0, 0, 0])
        self.assertEqual(again.steps.cpu().tolist(), [2, 0, 1])
        self.assertEqual(int(env['replay_store'].live.sum().cpu()), 2)
        self.assertEqual(int(env['action_store'].live.sum().cpu()), 0)
        preserve = torch.ones(3, dtype=torch.bool, device=device)
        check = env['roots']
        check.active.copy_(preserve)
        still = env['source'].gather_into_device(
            check, _graph(originals, device))
        self.assertEqual(still.errors.cpu().tolist(), [0, 0, 0])
        env['calls'].release_device(result.states)
        env['source'].release_device(roots)

    def test_actual_cuda_partial_lanes_inactive_foreign_owner_and_error(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_raw_replay import OWNER_ERROR
        originals = [{'n': 3, 'id': 0}, {'n': 4, 'id': 1}, {'n': 5, 'id': 2}]
        env = self._cohort(originals, max_steps=1)
        device = env['device']
        tape = env['tape']
        roots = env['roots']
        tape.begin_device(env['owners'], env['seeds'])
        actions = _graph([{'delta': 7}, {'delta': 9}, {'kind': 'fail'}], device)
        games = torch.arange(3, dtype=torch.int64, device=device)
        steps = torch.zeros(3, dtype=torch.int64, device=device)
        versions = torch.zeros(3, dtype=torch.int64, device=device)
        actors = torch.zeros(3, dtype=torch.int64, device=device)
        active = torch.ones(3, dtype=torch.bool, device=device)
        errors = tape.append_device(
            actions, games, env['owners'], steps, versions, actors, active)
        self.assertEqual(errors.cpu().tolist(), [0, 0, 0])
        tape._rec_owner[1] = 999
        source_live = int(env['source'].live.sum().cpu())
        replay = env['replay_cls'](
            env['calls'], tape, roots, env['action_store'],
            max_steps=1, compact_layout=env['compact'], lanes=2, seconds=30)
        self.addCleanup(replay.close)
        plan = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        counts = torch.ones(3, dtype=torch.int64, device=device)
        replay.capture(plan, env['seeds'], counts)
        replay.replay()
        torch.cuda.synchronize()
        result = replay.result_device()
        self.assertTrue(bool(result.complete[0].cpu()))
        self.assertFalse(bool(result.complete[1].cpu()))
        self.assertFalse(bool(result.complete[2].cpu()))
        self.assertEqual(int(result.steps[0].cpu()), 1)
        self.assertEqual(int(result.steps[1].cpu()), 0)
        self.assertEqual(int(result.steps[2].cpu()), 0)
        self.assertEqual(int(result.errors[0].cpu()), 0)
        self.assertTrue(int(result.errors[1].cpu()) & OWNER_ERROR)
        self.assertNotEqual(int(result.errors[2].cpu()), 0)
        decoded0 = _decode_row(env['replay_store'], result.states, 0, env['compact'])
        decoded1 = _decode_row(env['replay_store'], result.states, 1, env['compact'])
        decoded2 = _decode_row(env['replay_store'], result.states, 2, env['compact'])
        self.assertEqual(decoded0, _expected(originals[0], [('a', {'delta': 7})]))
        self.assertEqual(decoded1, originals[1])
        self.assertEqual(decoded2, originals[2])
        self.assertEqual(int(env['source'].live.sum().cpu()), source_live)
        env['calls'].release_device(result.states)
        env['source'].release_device(roots)

    def test_actual_cuda_gpu_metadata_mismatch_stops_before_apply(self):
        import torch
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_search.device_raw_replay import ACTOR_ERROR, VERSION_ERROR
        env = self._cohort([{'n': 1}, {'n': 2}, {'n': 3}], max_steps=1)
        d, tape = env['device'], env['tape']
        zero = torch.zeros(3, dtype=torch.int64, device=d)
        tape.begin_device(env['owners'], env['seeds'])
        tape.append_device(_graph([{'delta': 1}] * 3, d), torch.arange(3, device=d),
                           env['owners'], zero, zero, zero, torch.ones(3, dtype=torch.bool, device=d))
        # This is an explicit synthetic GPU provider; formal metadata stays an
        # injected RuleAdapter responsibility, never inferred from these values.
        meta = SimpleNamespace(actors=torch.tensor([0, 0, 1], dtype=torch.int64, device=d),
                               versions=torch.tensor([0, 999, 0], dtype=torch.int64, device=d),
                               errors=torch.zeros(3, dtype=torch.int32, device=d))
        replay = env['replay_cls'](env['calls'], tape, env['roots'], env['action_store'],
            max_steps=1, compact_layout=env['compact'], lanes=2,
            metadata_provider=lambda states: meta, require_metadata=True)
        self.addCleanup(replay.close)
        replay.capture(torch.arange(3, device=d).reshape(1, 3), env['seeds'], zero + 1)
        replay.replay(); torch.cuda.synchronize()
        result = replay.result_device()
        self.assertEqual(result.complete.cpu().tolist(), [True, False, False])
        self.assertEqual(result.metadata_checked.cpu().tolist(), [True, False, False])
        self.assertTrue(int(result.errors[1].cpu()) & VERSION_ERROR)
        self.assertTrue(int(result.errors[2].cpu()) & ACTOR_ERROR)
        self.assertEqual(result.steps.cpu().tolist(), [1, 0, 0])

    def test_actual_cuda_archive_error_or_wrong_declared_count_cannot_complete(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_raw_replay import STEP_ERROR
        env = self._cohort([{'n': 1}, {'n': 2}, {'n': 3}], max_steps=1)
        d, tape = env['device'], env['tape']
        zero = torch.zeros(3, dtype=torch.int64, device=d)
        all_rows = torch.ones(3, dtype=torch.bool, device=d)
        graph = _graph([{'delta': 1}] * 3, d)
        tape.begin_device(env['owners'], env['seeds'])
        tape.append_device(graph, torch.arange(3, device=d), env['owners'], zero, zero, zero, all_rows)
        tape.append_device(graph, torch.arange(3, device=d), env['owners'], zero, zero, zero,
                           torch.tensor([True, False, False], device=d))
        replay = env['replay_cls'](env['calls'], tape, env['roots'], env['action_store'],
            max_steps=1, compact_layout=env['compact'], lanes=2)
        self.addCleanup(replay.close)
        replay.capture(torch.arange(3, device=d).reshape(1, 3), env['seeds'],
                       torch.tensor([1, 0, 1], dtype=torch.int64, device=d))
        replay.replay(); torch.cuda.synchronize()
        result = replay.result_device()
        self.assertEqual(result.complete.cpu().tolist(), [False, False, True])
        self.assertNotEqual(int(result.errors[0].cpu()), 0)
        self.assertTrue(int(result.errors[1].cpu()) & STEP_ERROR)
        self.assertEqual(result.steps.cpu().tolist(), [0, 0, 1])


if __name__ == '__main__':
    unittest.main()

