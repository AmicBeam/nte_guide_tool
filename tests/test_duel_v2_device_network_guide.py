"""Network-leaf search and fixed-plan guide contracts, with real CUDA checks."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class NetworkLeafTest(unittest.TestCase):
    def test_full32_without_any_leaf_teacher_and_grouping_preserves_result(self):
        import torch
        from tests.test_duel_v2_device_search import DeviceSearchTest
        outputs = []
        for grouped in (False, True):
            helper = DeviceSearchTest()
            search, rules = helper.engine(group_ready_leaves=grouped,
                                           reuse_interior_observation=True)
            search.teacher_mode = 'network_value'
            search.terminal_horizon = 0
            with patch.object(rules, 'teacher_continue_device', side_effect=AssertionError('leaf teacher called')):
                roots = helper.initialize(search, rules)
                result = helper.finish_cpu(search)
            self.assertEqual(result.completed.tolist(), [32, 32])
            self.assertEqual(result.errors.tolist(), [0, 0])
            self.assertTrue(bool(result.complete.all()))
            self.assertEqual(int(rules.live.sum()), 2)
            torch.testing.assert_close(rules.data[roots.slots], torch.tensor([[0, -1], [0, -1]]))
            outputs.append([getattr(result, k).clone() for k in ('actions', 'visits', 'policy')])
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b)

    def test_network_mode_requires_zero_horizon(self):
        from tests.test_duel_v2_device_search import DeviceSearchTest
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        search, rules = DeviceSearchTest().engine()
        with self.assertRaisesRegex(ValueError, 'zero teacher horizon'):
            DeviceGumbelSearch(rules, search.inference, search.auxiliary,
                search.statistics, search.information_keys, rows=2, max_actions=3,
                teacher_mode='network_value', terminal_horizon=3,
                device='cpu', allow_cpu_reference=True, allow_synthetic=True)

    def test_guide_mixture_changes_only_guided_rows(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_zhenhong_guide import mix_guided_policy
        p = torch.tensor([[.2, .3, .5], [.1, .6, .3]])
        before = p.clone()
        result = mix_guided_policy(p, torch.tensor([2, 1]), torch.tensor([0, 1]))
        torch.testing.assert_close(result[0], torch.tensor([.6, .15, .25]))
        torch.testing.assert_close(result[1], p[1])
        torch.testing.assert_close(p, before)


class RootGuideCommitTest(unittest.TestCase):
    def test_guidance_is_recorded_and_committed_and_errors_fail_closed(self):
        import torch
        from tests.test_duel_v2_device_games import DeviceGameLoopTest
        helper = DeviceGameLoopTest()
        for fail in (False, True):
            loop, rules, sink, roots, seeds, ids, train = helper.make(max_decisions=1)
            class Guide:
                def capture_result(self, states, actors, result, model_ids, active):
                    errors = result.errors.clone()
                    if fail:
                        errors[1] = 154
                    return replace(result, actions=torch.zeros_like(result.actions), errors=errors)
            loop.root_guide = Guide()
            result = helper.finish_cpu(loop, roots, seeds, ids, train)
            self.assertEqual(result.steps.tolist(), [1, 0] if fail else [1, 1])
            self.assertEqual(sink.appends.tolist(), [1, 0] if fail else [1, 1])
            self.assertEqual(sink.actions[0].item(), 0)
            self.assertEqual(rules.data[result.states.slots[0], 1].item(), 0)
            if fail:
                self.assertNotEqual(result.errors[1].item(), 0)
                self.assertEqual(result.states.slots[1].item(), roots.slots[1].item())


class FixedGuideTest(unittest.TestCase):
    def cases(self):
        from tests.test_duel_v2_zhenhong_guide import ZhenhongGuideTest, ZhenhongOpeningGuideTest
        from app.modules.card_game.engine.duel_v2.state import hero
        helper = ZhenhongGuideTest()
        cases = []
        for cards in (['I04'], ['I04', 'Z04', 'I01'], [], ['R08']):
            s = helper.game(cards)
            cases.append((s, 'a', dict(type='end_turn')))
        s = helper.game(['I04', 'Z04', 'I01'])
        s['sides']['a']['ap'] = 1
        cases.append((s, 'a', dict(type='end_turn')))
        s = helper.game(['I04'])
        s['sides']['b'].update(hp=1, shield=0)
        cases.append((s, 'a', dict(type='attack', character_id='zero')))
        s = helper.game(['I04'])
        hero(s, 'a', 'zhenhong')['hp'] = 0
        cases.append((s, 'a', dict(type='end_turn')))
        for side in ('a', 'b'):
            s = ZhenhongOpeningGuideTest().game(['I01', 'Z04', 'R02', 'Y03', 'Y06'], side)
            out = [c['instance_id'] for c in s['sides'][side]['hand']][:3]
            cases.append((s, side, dict(type='mulligan', card_ids=out)))
        return cases

    def test_fixed_guide_matches_unexpired_shared_v3_and_keeps_state(self):
        from app.modules.card_game.engine.ai.zhenhong_guide import recommend
        from app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide import guide_index
        from app.modules.card_game.rl.batched_duel.gpu_rule_entries import _FrozenSevenRuntime
        from app.modules.card_game.engine.duel_v2 import legal_actions
        for state, side, baseline in self.cases():
            before = deepcopy(state)
            expected = recommend(state, side, baseline,
                seed=1729 + state['version'] * 1000003, seconds=60, max_plans=24,
                runtime=_FrozenSevenRuntime)['selected']
            legal = [a for a in legal_actions(state, side) if a['type'] != 'concede']
            self.assertEqual(legal[guide_index(state, side, baseline)], expected)
            self.assertEqual(state, before)

    def test_hidden_cards_and_order_do_not_change_fixed_guide(self):
        from app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide import guide_index
        s, side, action = self.cases()[0]
        other = deepcopy(s)
        foe = other['sides']['b']
        pool = list(reversed(foe['hand'] + foe['deck']))
        n = len(foe['hand']);foe['hand'], foe['deck'] = pool[:n], pool[n:]
        self.assertEqual(guide_index(s, side, action), guide_index(other, side, action))

    def test_guide_sample_matches_canonical_and_rejects_public_drift(self):
        from app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide import guide_sample_world
        from app.modules.card_game.rl.information_search import sample_world
        from app.modules.card_game.rl.batched_duel.gpu_rule_entries import _FrozenSevenRuntime
        import app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide as guide
        for state, side, _ in self.cases():
            if state['phase'] != 'playing':
                continue
            before = deepcopy(state)
            self.assertEqual(guide_sample_world(state, side, 77881),
                sample_world(state, side, 77881, runtime=_FrozenSevenRuntime))
            self.assertEqual(state, before)
        state, side, _ = self.cases()[0]
        original = guide.observe
        def drift(*args, **kwargs):
            view = original(*args, **kwargs)
            view['turn'] += 1
            return view
        with patch.object(guide, 'observe', side_effect=drift):
            with self.assertRaisesRegex(ValueError, 'full public information'):
                guide_sample_world(state, side, 77881)

    def test_actual_cuda_guide_matches_formal_cpu_and_input_immutable(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        name = os.environ.get('NTE_NETWORK_GUIDE_PROGRAM')
        if not name:
            self.skipTest('Explicit frozen guide program required')
        import numpy as np
        from app.modules.card_game.rl.batched_duel.gpu_rule_program import load_program
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, VmLimits
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide import guide_index
        program = load_program(Path(name), runtime='seven', repo_root=Path.cwd())
        # All formal v3 fixtures, including both mulligan seats and a two-card macro.
        cases = self.cases()
        roots = [dict(state=s, side=side, action=a) for s, side, a in cases]
        layout = GraphLayout(len(roots), 1048576, 2097152, 67108864)
        with CudaVm(lower_program(program), limits=VmLimits(steps=12000000),
                    guard_standalone_entities=True, constant_lookup=True,
                    compact_frame_init=True, device_call_boundaries=True,
                    library_call_boundaries=True, mutable_lookup_cache=True,
                    frozen_class_lookup=True) as vm:
            arena = vm.ingest(pack_states(roots, layout))
            deadlines = torch.full((len(roots),), vm.clock_ns()+int(500e9),
                                   dtype=torch.int64, device=vm.device)
            actual = vm.launch(arena, 'zhenhong_guide_probe', gpu_deadlines=deadlines).decode()
            self.assertEqual(actual, [(guide_index(*c), packet) for c, packet in zip(cases, roots)])
