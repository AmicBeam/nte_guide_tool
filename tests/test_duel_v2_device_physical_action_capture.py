"""Physical raw-action transport and label failure isolation; no full-rule evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import os
import sys
from pathlib import Path

from app.modules.card_game.rl.batched_search.device_physical_action_capture import (
    PhysicalActionRecordingCalls, RawActionPolicySink,
)
from app.modules.card_game.rl.batched_search.device_game_policy_sink import DeviceGamePolicySink


def _apply_probe(state, actor, action):
    state['n'] += action['delta']
    state['version'] += 1
    state['actor'] = actor
    return state


def _clean_probe(state):
    state['cleaned'] = True
    return state


def _probe_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program,
    )
    fixture = os.environ.get('NTE_PHYSICAL_ACTION_CAPTURE_PROGRAM')
    if fixture:
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('Same-source CPython3.10 frozen physical probe required')
    return capture_program({'apply_action': _apply_probe, 'clean': _clean_probe},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
        scope_character_ids=None, scope_card_ids=None)


class PhysicalActionCaptureTest(unittest.TestCase):
    def test_only_actual_apply_records_the_selected_raw_graph(self):
        calls = object.__new__(PhysicalActionRecordingCalls)
        seen, failures = [], []
        calls.raw_tape = SimpleNamespace(append_device=lambda *a: seen.append(a) or 'error-vector')
        calls.physical_loop = SimpleNamespace(steps_device='physical-steps',
                                             source_versions_device='physical-versions')
        calls.selected_action = object()
        calls._lanes = SimpleNamespace(ids=object())
        calls.lane_owners, calls.lane_actors = object(), object()
        mask = object()
        calls.lane_active = SimpleNamespace(bool=lambda: mask)
        calls._pick = lambda x: ('gpu-lane-selection', x)
        calls._or_errors = failures.append
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        with patch.object(DeviceRuleGraphCalls, '_bind_step', return_value='gpu-refs'):
            self.assertEqual(calls._bind_step(SimpleNamespace(entry='simulate_action'), None), 'gpu-refs')
            self.assertEqual(seen, [])
            self.assertEqual(calls._bind_step(SimpleNamespace(entry='apply_action'), None), 'gpu-refs')
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], calls.selected_action)
        self.assertEqual(seen[0][3:5], (('gpu-lane-selection', 'physical-steps'),
                                       ('gpu-lane-selection', 'physical-versions')))
        self.assertIs(seen[0][-1], mask)
        self.assertEqual(failures, ['error-vector'])

    def test_missing_raw_archive_cannot_produce_terminal_policy_labels(self):
        import torch
        from tests.test_duel_v2_device_replay_policy_tape import DeviceReplayPolicyTapeTest, _roots, _search
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
        h = DeviceReplayPolicyTapeTest()
        tape = h._tape(rows=1)
        raw = DeviceRawActionTape(1, 4, 128, 128, 512, device='cpu', allow_cpu_reference=True)
        self.addCleanup(raw.close)
        policy_sink = DeviceGamePolicySink(tape)
        sink = RawActionPolicySink(policy_sink, raw)
        zero = torch.zeros(1, dtype=torch.int64)
        loop = SimpleNamespace(sink=sink, source_versions_device=zero)
        policy_sink.bind_game_loop(loop)
        roots = _roots(1, 'cpu')
        seeds, ids, train, keys = h._ids(tape)
        result = _search(roots, torch.tensor([20]), zero)
        result.information_keys = keys
        sink.begin_device(roots, ids, ids, seeds, train, roots.active)
        sink.append_device(roots, zero, zero, result, roots.active)
        flags = sink.finish_device(roots, zero, zero + 1, roots.active, roots.errors)
        self.assertNotEqual(int(flags[0]), 0)
        self.assertFalse(bool(tape.result_device().eligible.any()))
        self.assertEqual(tape.records_after_batch(), [])
        self.assertFalse(sink.supports_full_duel_rules)

    def test_actual_cuda_selected_raw_action_is_archived_before_apply_and_clean(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; synthetic physical action probe only')
        from tests.test_duel_v2_device_rule_calls import _graph, _store, _decode_row
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceDecision
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
        states = _store('cuda', slots=16)
        actions = _store('cuda', slots=16)
        self.addCleanup(states.close); self.addCleanup(actions.close)
        with CudaVm(lower_program(_probe_program())) as vm:
            raw = DeviceRawActionTape(3, 4, 128, 128, 1024, device=str(vm.device))
            self.addCleanup(raw.close)
            layout = GraphLayout(2, 64, 64, 256)
            calls = PhysicalActionRecordingCalls(vm, states, raw_tape=raw,
                rows=3, lanes=2, compact_layout=layout, seconds=30)
            self.addCleanup(calls.close)
            owners = torch.tensor([11, 12, 13], dtype=torch.int64, device=vm.device)
            active = torch.tensor([True, False, True], device=vm.device)
            seeds = torch.tensor([101, 102, 103], dtype=torch.int64, device=vm.device)
            zero = torch.zeros(3, dtype=torch.int64, device=vm.device)
            loop = SimpleNamespace(rows=3, device=vm.device, steps_device=zero,
                                   source_versions_device=zero)
            calls.bind_game_loop(loop)
            initial = states.put_device(_graph([{'n': 0, 'version': 0} for _ in range(3)], vm.device),
                                        owners, active)
            graph = _graph([[{'delta': 3}, {'delta': 7}] for _ in range(3)], vm.device)
            lease = actions.put_device(graph, owners, active)
            rows = torch.arange(3, device=vm.device)
            start = graph.nodes[rows, graph.roots.long(), 1].long()
            refs = torch.stack((graph.edges[rows, start, 1], graph.edges[rows, start + 1, 1]), 1)
            decision = DeviceDecision(states=initial, action_pool=actions,
                action_slots=lease.slots, action_generations=lease.generations,
                action_refs=refs.contiguous(), counts=zero + 2,
                action_types=torch.zeros((3, 2), dtype=torch.int64, device=vm.device),
                valid=torch.ones((3, 2), dtype=torch.bool, device=vm.device),
                observations=torch.zeros((3, 1), device=vm.device),
                candidates=torch.zeros((3, 2, 1), device=vm.device), errors=initial.errors)
            actors = torch.tensor([0, 0, 1], dtype=torch.int64, device=vm.device)
            selected = torch.tensor([1, 0, 0], dtype=torch.int64, device=vm.device)
            apply_plan, clean_plan = calls.prepare_call('apply_action'), calls.prepare_call('clean')
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                raw.begin_device(owners, seeds)
                applied = apply_plan.capture_device(initial, actors=actors, decision=decision,
                                                    action_indices=selected)
                cleaned = clean_plan.capture_device(applied)
            captured.replay(); torch.cuda.synchronize()
            self.assertEqual(cleaned.errors.cpu().tolist(), [0, 0, 0])
            for row, value, actor in ((0, 7, 'a'), (2, 3, 'b')):
                decoded = _decode_row(states, cleaned, row, layout)
                self.assertEqual(decoded, {'n': value, 'version': 1, 'actor': actor, 'cleaned': True})
            self.assertEqual(raw.steps_device.cpu().tolist(), [1, 0, 1])
            self.assertEqual(raw.errors_device.cpu().tolist(), [0, 0, 0])
            self.assertEqual(raw._rec_owner[:2].cpu().tolist(), [11, 13])
            self.assertEqual(raw._rec_actor[:2].cpu().tolist(), [0, 1])
            self.assertEqual(raw._rec_version[:2].cpu().tolist(), [0, 0])
            recovered = _graph([{}, {}], vm.device)
            errors = raw.gather_into_device(torch.arange(2, device=vm.device), recovered,
                torch.ones(2, dtype=torch.bool, device=vm.device))
            torch.cuda.synchronize()
            self.assertEqual(errors.cpu().tolist(), [0, 0])
            from app.modules.card_game.rl.batched_duel.codec import unpack_states
            from app.modules.card_game.rl.batched_duel.schema import PackedStateBatch
            packed = PackedStateBatch(recovered.layout, *(getattr(recovered, name).cpu().numpy()
                for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))
            self.assertEqual(unpack_states(packed), [{'delta': 7}, {'delta': 3}])
            calls.release_device(cleaned); calls.release_device(applied)
            calls.release_device(initial); actions.release_device(lease)
