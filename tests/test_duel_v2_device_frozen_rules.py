"""DeviceFrozenRules composition contracts. Not full-game or 1600-root evidence."""
from types import SimpleNamespace
from pathlib import Path
import os
import sys
import unittest

from app.modules.card_game.rl.batched_search.device_frozen_rules import (
    DEFAULT_SITES,
    OPERATIONS,
    ROOT_MAX_ACTIONS,
    TEACHER_HORIZON,
    DeviceFrozenRules,
    FrozenRuleGap,
    inspect_device_frozen_rule_dependencies,
    source_teacher_horizon,
)
from app.modules.card_game.rl.batched_search.device_physical_action_capture import (
    PhysicalActionRecordingCalls,
)
from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
from app.modules.card_game.rl.batched_duel.gpu_schema_contract import runtime_metadata


def _schema():
    return runtime_metadata('seven')


class DeviceFrozenRulesContractTest(unittest.TestCase):
    def test_surface_stays_unqualified_and_lists_exact_operations(self):
        self.assertFalse(DeviceFrozenRules.supports_full_duel_rules)
        self.assertIs(DeviceFrozenRules.host_rule_callbacks, False)
        self.assertNotEqual(DeviceFrozenRules.execution_kind, 'frozen_cuda_rules')
        self.assertFalse(hasattr(DeviceFrozenRules, 'device_validation_receipt'))
        self.assertEqual(DeviceFrozenRules.teacher_horizon, 3)
        self.assertEqual(DeviceFrozenRules.root_max_actions, 800)
        self.assertEqual(DeviceFrozenRules.operations, OPERATIONS)
        for name in (
            'metadata_device', 'decision_device', 'rule_decision_device',
            'value_observation_device', 'sample_world_device', 'step_device',
            'clean_device', 'teacher_continue_device', 'release_device',
        ):
            self.assertTrue(callable(getattr(DeviceFrozenRules, name)))
        self.assertEqual(
            set(DEFAULT_SITES),
            {'metadata', 'decision', 'rule_decision', 'value', 'sample_world',
             'apply_action', 'simulate_action', 'clean', 'teacher_continue'})
        self.assertGreaterEqual(DEFAULT_SITES['decision'], 2)
        self.assertGreaterEqual(DEFAULT_SITES['metadata'], 9)
        self.assertTrue(issubclass(PhysicalActionRecordingCalls, DeviceRuleGraphCalls))

    def test_formal_schema_dimensions_and_teacher_horizon_are_current(self):
        meta = _schema()
        self.assertEqual(meta['obs_dim'], 3479)
        self.assertEqual(meta['cand_dim'], 1576)
        self.assertEqual(len(meta['action_types']), 9)
        self.assertEqual(meta['action_types'][0], 'end_turn')
        self.assertEqual(ROOT_MAX_ACTIONS, 800)
        self.assertEqual(source_teacher_horizon(), TEACHER_HORIZON)
        self.assertEqual(TEACHER_HORIZON, 3)

    def test_constructor_rejects_non_helpers_and_cpu_objects(self):
        for calls, presentation, store in (
            (None, None, None),
            ({}, {}, {}),
            (lambda: None, lambda: None, lambda: None),
        ):
            with self.assertRaises(TypeError):
                DeviceFrozenRules(calls, presentation, action_store=store)
        calls = object.__new__(DeviceRuleGraphCalls)
        calls.closed = True
        presentation = object.__new__(DeviceRulePresentation)
        presentation.closed = True
        with self.assertRaises(TypeError):
            DeviceFrozenRules(calls, presentation, action_store=object())

    def test_inspect_reports_exact_helper_and_vm_gaps(self):
        calls = object.__new__(DeviceRuleGraphCalls)
        calls.closed = False
        calls.supports_full_duel_rules = False
        calls.vm = None
        presentation = object.__new__(DeviceRulePresentation)
        presentation.closed = False
        presentation.calls = object()
        presentation.supports_full_duel_rules = True
        gaps = inspect_device_frozen_rule_dependencies(calls, presentation, None)
        self.assertTrue(any('same-source frozen CUDA VM' in item for item in gaps))
        closed = inspect_device_frozen_rule_dependencies(
            object.__new__(DeviceRuleGraphCalls), presentation, None)
        self.assertTrue(any('DeviceRuleGraphCalls' in item for item in closed))
        vm = SimpleNamespace(closed=False, device=SimpleNamespace(type='cpu', index=None),
                             image=SimpleNamespace(entries={}),
                             capability=lambda name: dict(ready=False, missing={'entries': [name]}))
        live = object.__new__(DeviceRuleGraphCalls)
        live.closed = False
        live.supports_full_duel_rules = False
        live.vm = vm
        live.device = SimpleNamespace(type='cpu', index=None)
        live.rows = 2
        live.state_store = object()
        live.prepare_call = True
        live.release_device = lambda states: None
        pres = object.__new__(DeviceRulePresentation)
        pres.closed = False
        pres.supports_full_duel_rules = False
        pres.calls = live
        pres.vm = vm
        pres.device = live.device
        pres.rows = 2
        pres.max_actions = 16
        pres.prepare_metadata = pres.prepare_decision = pres.prepare_value = lambda *a, **k: None
        pres.release_decision_device = lambda *a, **k: None
        pres.reader = object()
        gaps = inspect_device_frozen_rule_dependencies(live, pres, None)
        joined = '; '.join(gaps)
        self.assertIn('actual CUDA required', joined)
        self.assertIn('max_actions is 16, required root maxM 800', joined)
        self.assertIn('missing frozen VM entry rule_decision', joined)
        self.assertIn('missing frozen VM entry teacher_continue', joined)
        self.assertIn('missing frozen VM entry public_legal_decision', joined)
        self.assertIn('live compact action graph store required', joined)

    def test_teacher_continue_refuses_non_horizon_three_without_substitution(self):
        rules = object.__new__(DeviceFrozenRules)
        with self.assertRaises(ValueError) as ctx:
            rules.teacher_continue_device(None, None, horizon=2)
        self.assertIn('horizon 3', str(ctx.exception))
        with self.assertRaises(ValueError):
            rules.teacher_continue_device(None, None, horizon=0)
        with self.assertRaises(TypeError):
            rules.step_device(None, None, None, None, physical='yes')


def _probe_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program,
    )
    fixture = os.environ.get('NTE_DEVICE_FROZEN_RULES_PROGRAM')
    if fixture:
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('Same-source CPython 3.10 exporter or NTE_DEVICE_FROZEN_RULES_PROGRAM required')
    return capture_program({
        'acting_side': _acting_side,
        'finished': _finished,
        'terminal_value': _terminal_value,
        'public_legal_decision': _public_legal_decision,
        'rule_decision': _rule_decision,
        'apply_action': _apply_action,
        'simulate_action': _simulate_action,
        'clean': _clean,
        'sample_world': _sample_world,
        'teacher_continue': _teacher_continue,
    }, allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
        scope_character_ids=None, scope_card_ids=None)


def _acting_side(state):
    return None if state.get('done') else state.get('actor', 'a')


def _finished(state):
    return bool(state.get('done'))


def _terminal_value(state, viewer):
    if not state.get('done'):
        return 0.0
    winner = state.get('winner')
    if winner in (None, 'draw'):
        return 0.0
    return 1.0 if winner == viewer else -1.0


def _public_legal_decision(state, observer):
    legal = [{'type': 'end_turn'}, {'type': 'attack', 'delta': 1}]
    return {'observer': observer, 'legal_actions': [{'action': item} for item in legal]}, legal


def _rule_decision(state, actor):
    legal = [{'type': 'end_turn'}, {'type': 'attack', 'delta': 1}]
    return legal, 0


def _apply_action(state, actor, action):
    out = dict(state)
    out['n'] = out.get('n', 0) + action.get('delta', 0)
    out['actor'] = actor
    out['version'] = out.get('version', 0) + 1
    return out


def _simulate_action(state, actor, action):
    return _apply_action(state, actor, action)


def _clean(state):
    out = dict(state)
    out['cleaned'] = True
    return out


def _sample_world(state, viewer, seed):
    out = dict(state)
    out['viewer'] = viewer
    out['seed'] = seed
    return out


def _teacher_continue(state):
    world = dict(state)
    for _ in range(3):
        if world.get('done'):
            break
        world['n'] = world.get('n', 0) + 1
    world['continued'] = True
    return world


def _partial_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program,
    )
    fixture = os.environ.get('NTE_DEVICE_RULE_CALLS_TEST_PROGRAM')
    if fixture:
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython 3.10 capture helper or NTE_DEVICE_RULE_CALLS_TEST_PROGRAM required')
    return capture_program(
        {'clean': _clean, 'sample_world': _sample_world},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
        scope_character_ids=None, scope_card_ids=None)


def _store(device, slots=16):
    from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
    return CompactDeviceGraphStore(
        slots=slots, node_pages=32, edge_pages=32, payload_pages=32,
        node_page_bytes=256, edge_page_bytes=256, payload_page_bytes=512,
        max_node_pages_per_slot=8, max_edge_pages_per_slot=8, max_payload_pages_per_slot=8,
        device=device)


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


def _limbs(value):
    import numpy as np
    return np.frombuffer((value).to_bytes(40, 'little', signed=False), dtype='<i8').copy()


class DeviceFrozenRulesCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')

    def test_incomplete_frozen_vm_fails_closed_with_named_gaps(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        program = _partial_program()
        store = _store('cuda')
        actions = _store('cuda')
        self.addCleanup(store.close)
        self.addCleanup(actions.close)
        with CudaVm(lower_program(program)) as vm:
            helper = DeviceRuleGraphCalls(
                vm, store, rows=2, lanes=2, seconds=30,
                compact_layout=GraphLayout(2, 64, 64, 256))
            self.addCleanup(helper.close)
            presentation = DeviceRulePresentation(
                helper, max_actions=800, max_output_bytes=1 << 30, seconds=30)
            self.addCleanup(presentation.close)
            with self.assertRaises(FrozenRuleGap) as ctx:
                DeviceFrozenRules(helper, presentation, action_store=actions)
            message = str(ctx.exception)
            self.assertIn('missing frozen VM entry rule_decision', message)
            self.assertIn('missing frozen VM entry teacher_continue', message)
            self.assertIn('missing frozen VM entry public_legal_decision', message)
            self.assertIn('missing frozen VM entry apply_action', message)
            self.assertFalse(DeviceFrozenRules.supports_full_duel_rules)

    def test_prepared_buffers_keep_formal_dtypes_and_captured_toy_rules_run(self):
        import torch
        from contextlib import ExitStack
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceDecision
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
        from tests.test_duel_v2_device_rule_calls import _decode_row
        program = _probe_program()
        device = torch.device('cuda', torch.cuda.current_device())
        states_store = _store(device, slots=32)
        actions_store = _store(device, slots=32)
        self.addCleanup(states_store.close)
        self.addCleanup(actions_store.close)
        compact = GraphLayout(2, 64, 64, 256)
        rows = 3
        meta = _schema()
        with CudaVm(lower_program(program)) as vm:
            calls = DeviceRuleGraphCalls(
                vm, states_store, rows=rows, lanes=2, seconds=30, compact_layout=compact)
            self.addCleanup(calls.close)
            presentation = DeviceRulePresentation(
                calls, max_actions=800, max_output_bytes=1 << 30, seconds=30)
            self.addCleanup(presentation.close)
            rules = DeviceFrozenRules(
                calls, presentation, action_store=actions_store,
                sites={name: 1 for name in DEFAULT_SITES})
            self.addCleanup(rules.close)
            self.assertFalse(rules.supports_full_duel_rules)
            self.assertEqual(rules.prepared['decision'], 1)
            self.assertEqual(rules.prepared['rule_decision'], 1)
            plan = rules._queues['decision'].plans[0]
            self.assertEqual(plan.action_types.dtype, torch.int64)
            self.assertEqual(plan.action_refs.dtype, torch.int32)
            self.assertEqual(plan.action_types.shape, (rows, 800))
            self.assertEqual(plan.observations.shape[1], presentation.projection.obs_dim)
            self.assertEqual(presentation.projection.obs_dim, meta['obs_dim'])
            self.assertIsInstance(plan.candidates, CompactDeviceCandidates)
            self.assertEqual(plan.candidates.shape, (rows, 800, meta['cand_dim']))
            payloads = [
                {'phase': 'playing', 'version': 1, 'winner': None, 'actor': 'a', 'done': False, 'n': 0},
                {'phase': 'playing', 'version': 2, 'winner': None, 'actor': 'b', 'done': False, 'n': 4},
                {'phase': 'finished', 'version': 3, 'winner': 'a', 'actor': None, 'done': True, 'n': 9},
            ]
            owners = torch.arange(11, 14, dtype=torch.int64, device=device)
            active = torch.tensor([True, False, True], device=device)
            roots = states_store.put_device(_graph(payloads, device), owners, active)
            action_graph = _graph([[{'delta': 3}, {'delta': 7}] for _ in range(rows)], device)
            lease = actions_store.put_device(action_graph, owners, active)
            batch_rows = torch.arange(rows, device=device)
            start = action_graph.nodes[batch_rows, action_graph.roots.long(), 1].long()
            refs = torch.stack(
                (action_graph.edges[batch_rows, start, 0], action_graph.edges[batch_rows, start + 1, 0]), 1)
            synthetic = DeviceDecision(
                states=roots, action_pool=actions_store, action_slots=lease.slots,
                action_generations=lease.generations, action_refs=refs.contiguous(),
                counts=torch.full((rows,), 2, dtype=torch.int64, device=device),
                action_types=torch.zeros((rows, 2), dtype=torch.int64, device=device),
                valid=torch.ones((rows, 2), dtype=torch.bool, device=device),
                observations=torch.zeros((rows, 1), device=device),
                candidates=torch.zeros((rows, 2, 1), device=device), errors=roots.errors)
            actors = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
            selected = torch.tensor([1, 0, 0], dtype=torch.int64, device=device)
            viewers = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
            seeds = torch.tensor([_limbs(1 << 8), _limbs(0), _limbs(9)], dtype=torch.int64, device=device)
            captured = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                for name in ('cpu', 'numpy', 'item', 'tolist', '__bool__'):
                    stack.enter_context(patch.object(
                        torch.Tensor, name, side_effect=AssertionError('Host selector')))
                with torch.cuda.graph(captured, capture_error_mode='thread_local'):
                    meta_out = rules.metadata_device(roots)
                    sampled = rules.sample_world_device(roots, viewers, seeds)
                    applied = rules.step_device(
                        roots, actors, synthetic, selected, physical=True)
                    cleaned = rules.clean_device(applied)
                    continued = rules.teacher_continue_device(roots, viewers, horizon=3)
            captured.replay()
            torch.cuda.synchronize()
            self.assertEqual(meta_out.actors.dtype, torch.int64)
            self.assertEqual(meta_out.phases.dtype, torch.int32)
            self.assertEqual(meta_out.winners.dtype, torch.int64)
            self.assertEqual(meta_out.versions.dtype, torch.int64)
            self.assertEqual(meta_out.errors.dtype, torch.int32)
            self.assertEqual(meta_out.actors.tolist(), [0, -1, -1])
            self.assertEqual(meta_out.phases.tolist(), [1, -1, 3])
            self.assertEqual(meta_out.winners.tolist(), [-2, -2, 0])
            self.assertEqual(meta_out.finished.tolist(), [False, False, True])
            self.assertEqual(int(cleaned.slots[1].cpu()), -1)
            decoded = _decode_row(states_store, cleaned, 0, compact)
            self.assertEqual(decoded['n'], 7)
            self.assertTrue(decoded['cleaned'])
            continued_row = _decode_row(states_store, continued, 0, compact)
            self.assertEqual(continued_row['n'], 3)
            self.assertTrue(continued_row['continued'])
            sampled_row = _decode_row(states_store, sampled, 0, compact)
            self.assertEqual(sampled_row['viewer'], 'a')
            self.assertEqual(sampled_row['seed'], 1 << 8)
            rules.release_device(cleaned)
            rules.release_device(applied)
            rules.release_device(sampled)
            rules.release_device(continued)
            rules.release_device(roots)
            actions_store.release_device(lease)


if __name__ == '__main__':
    unittest.main()

