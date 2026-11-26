"""Real public states for array guidance; controlled fixtures for tree wiring."""
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.modules.card_game.rl.batched_search.array_zhenhong_prior import (
    ArrayZhenhongPrior, ZhenhongPriorWeights, ArrayPriorResult,
)


def public_inputs(state, side='a', device='cpu', compact=False):
    from app.modules.card_game.engine.duel_v2 import observe
    from app.modules.card_game.rl import seven_lineup
    from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
    from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
    view = observe(state, side, include_previews=False)
    actions = [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']
    x, c = seven_lineup.encode(view, actions)
    x = torch.tensor(x[None], device=device)
    valid = torch.ones((1, len(actions)), dtype=torch.bool, device=device)
    if compact:
        tokens = encoder.extract_raw_tokens([view], [actions])
        tokens = encoder.RawCandidateTokens(**{
            f.name: torch.as_tensor(getattr(tokens, f.name), device=device)
            for f in fields(tokens) if f.name != 'request_ids'})
        c = CompactDeviceCandidates(tokens, device=device, allow_cpu_reference=device == 'cpu')
    else:
        c = torch.tensor(c[None], device=device)
    return actions, x, c, valid


class ArrayPriorTest(unittest.TestCase):
    def bind_prior(self, search, rules, guide):
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        return DeviceGumbelSearch(rules, search.inference, search.auxiliary,
            search.statistics, search.information_keys, rows=search.rows,
            max_actions=search.max_actions, simulations=search.simulations, max_depth=4,
            teacher_mode='network_value', terminal_horizon=0, prior_guide=guide,
            device='cpu', allow_cpu_reference=True, allow_synthetic=True)

    def game(self, cards):
        from tests.test_duel_v2_zhenhong_guide import ZhenhongGuideTest
        return ZhenhongGuideTest().game(cards)

    def score(self, state, *, side='a', compact=False, device='cpu'):
        actions, x, c, valid = public_inputs(state, side, device, compact)
        guide = ArrayZhenhongPrior(device=device, allow_cpu_reference=device == 'cpu')
        result = guide.score_device(x, c, valid, torch.ones(1, dtype=torch.bool, device=device))
        self.assertEqual(result.errors.tolist(), [0])
        return actions, result.shifts[0], result

    def test_no_world_sampling_or_rule_simulation_and_inputs_unchanged(self):
        state = self.game(['I04', 'Z04', 'I01'])
        before = deepcopy(state)
        actions, x, c, valid = public_inputs(state)
        old_x, old_c = x.clone(), c.clone()
        guide = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True)
        # Load the module before patching simulate_action; its static import
        # must not retain the mock after this test's context has exited.
        from app.modules.card_game.rl import information_search
        with patch('app.modules.card_game.engine.duel_v2.simulation.simulate_action',
                   side_effect=AssertionError('simulation called')), patch(
                   'app.modules.card_game.rl.information_search.sample_world',
                   side_effect=AssertionError('world sampled')):
            out = guide.score_device(x, c, valid, torch.tensor([True]))
        self.assertTrue(out.applicable.item())
        self.assertEqual(state, before)
        self.assertTrue(torch.equal(x, old_x))
        self.assertTrue(torch.equal(c, old_c))
        self.assertTrue(torch.isfinite(out.shifts).all())
        self.assertLessEqual(out.shifts.abs().max().item(), 2.)
        by_card = {a.get('card_id'): i for i, a in enumerate(actions) if a['type'] == 'play_card'}
        held = {c['card_id']: c['instance_id'] for c in state['sides']['a']['hand']}
        # Visible two-piece readiness favors Zero's first half, but AP1 does
        # not get that speculative combination hint.
        self.assertGreater(out.shifts[0, by_card[held['Z04']]], out.shifts[0, by_card[held['I04']]])
        state['sides']['a']['ap'] = 1
        actions, shifts, _ = self.score(state)
        self.assertEqual(shifts[actions.index(dict(type='play_card', card_id=held['Z04']))].item(), 0.)

    def test_dense_compact_and_hidden_information_invariance(self):
        state = self.game(['I04', 'Z04', 'I01', 'R08'])
        a, dense, _ = self.score(state)
        b, compact, _ = self.score(state, compact=True)
        self.assertEqual(a, b)
        torch.testing.assert_close(dense, compact, rtol=0, atol=0)
        other = deepcopy(state)
        foe = other['sides']['b']
        pool = foe['hand'] + foe['deck']
        pool.reverse()
        n = len(foe['hand'])
        foe['hand'], foe['deck'] = pool[:n], pool[n:]
        c, hidden, _ = self.score(other)
        self.assertEqual(a, c)
        torch.testing.assert_close(dense, hidden, rtol=0, atol=0)

    def test_surplus_and_delay_hints_match_real_entry_opportunities(self):
        from app.modules.card_game.engine.duel_v2 import apply_action
        from app.modules.card_game.engine.duel_v2.state import hero, front_debuff
        state = self.game(['I01'])
        state['sides']['a']['last_front'] = 'zero'
        hero(state, 'a', 'zero')['harmony'] = 2
        front_debuff(state, 'b')['delay'] = dict(left=2, by='a', tick_side='a')
        actions, scores, _ = self.score(state)
        cashout = dict(type='attack', character_id='iloy')
        self.assertGreater(scores[actions.index(cashout)].item(), 0)
        after = apply_action(state, 'a', cashout)
        self.assertGreater(hero(after, 'a', 'zhenhong').get('surplus_passive_triggers', 0), 0)
        front_debuff(state, 'b').pop('delay')
        actions, scores, _ = self.score(state)
        delay = dict(type='attack', character_id='yi')
        self.assertGreater(scores[actions.index(delay)].item(), 0)
        self.assertTrue(front_debuff(apply_action(state, 'a', delay), 'b').get('delay'))

    def test_opening_both_seats_and_duplicate_core(self):
        from tests.test_duel_v2_zhenhong_guide import ZhenhongOpeningGuideTest
        for side in ('a', 'b'):
            state = ZhenhongOpeningGuideTest().game(['I01', 'Z04', 'R02', 'Y03', 'Y06'], side)
            actions, scores, _ = self.score(state, side=side)
            _, compact, _ = self.score(state, side=side, compact=True)
            torch.testing.assert_close(scores, compact, rtol=0, atol=0)
            ids = {c['card_id']: c['instance_id'] for c in state['sides'][side]['hand']}
            keep = dict(type='mulligan', card_ids=[])
            lose = dict(type='mulligan', card_ids=[ids['I01'], ids['Z04']])
            self.assertGreater(scores[actions.index(keep)].item(), scores[actions.index(lose)].item())

    def test_inactive_padding_noneligible_and_nonfinite_fail_closed(self):
        state = self.game(['I04'])
        actions, x, c, valid = public_inputs(state)
        guide = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True)
        x = x.repeat(4, 1)
        c = torch.cat((c.repeat(4, 1, 1), torch.zeros((4, 2, c.shape[-1]))), 1)
        valid = torch.cat((valid.repeat(4, 1), torch.zeros((4, 2), dtype=torch.bool)), 1)
        x[1, guide.names['ch_hp:0:zhenhong']] = 0
        x[2, 0] = float('nan')
        x[3, 0] = float('nan')
        result = guide.score_device(x, c, valid, torch.tensor([True, True, True, False]))
        self.assertEqual(result.errors.tolist(), [0, 0, 1, 0])
        self.assertTrue((result.shifts[:, -2:] == 0).all())
        self.assertTrue((result.shifts[1:] == 0).all())
        with self.assertRaises(ValueError):
            ZhenhongPriorWeights(max_logit_shift=float('inf'))

    def test_search_uses_hint_only_for_root_never_as_value_or_final_override(self):
        from tests.test_duel_v2_device_search import DeviceSearchTest
        helper = DeviceSearchTest()
        results = []
        for guided in (False, True):
            search, rules = helper.engine()
            search.teacher_mode, search.terminal_horizon = 'network_value', 0
            class Hint:
                device = torch.device('cpu')
                version = 'fixture_prior'
                calls = 0
                def score_device(self, x, c, valid, active):
                    self.calls += 1
                    shift = torch.zeros_like(valid, dtype=torch.float32)
                    shift[:, 1] = .7
                    return ArrayPriorResult(shift, torch.zeros(2, dtype=torch.int32), active)
            hint = Hint()
            search = self.bind_prior(search, rules, hint if guided else None)
            with patch.object(rules, 'teacher_continue_device', side_effect=AssertionError('teacher called')):
                helper.initialize(search, rules)
                result = helper.finish_cpu(search)
            self.assertEqual(result.completed.tolist(), [32, 32])
            self.assertEqual(result.errors.tolist(), [0, 0])
            self.assertEqual(hint.calls, int(guided))
            self.assertTrue(torch.isfinite(search.leaf_values).all())
            self.assertEqual(result.prior_version, 'fixture_prior' if guided else None)
            results.append(result)
        torch.testing.assert_close(results[1].prior_shifts[:, 1], torch.full((2,), .7))

    def test_prior_failure_blocks_search_and_old_simulation_guide_conflicts(self):
        from tests.test_duel_v2_device_search import DeviceSearchTest
        from app.modules.card_game.rl.batched_search.device_search import PRIOR_ERROR
        helper = DeviceSearchTest()
        search, rules = helper.engine()
        class BrokenHint:
            device = torch.device('cpu')
            def score_device(self, x, c, valid, active):
                return ArrayPriorResult(torch.zeros_like(valid, dtype=torch.float32),
                                        torch.ones(2, dtype=torch.int32), active)
        search = self.bind_prior(search, rules, BrokenHint())
        helper.initialize(search, rules)
        result = helper.finish_cpu(search)
        self.assertFalse(result.complete.any())
        self.assertTrue(((result.errors & PRIOR_ERROR) != 0).all())
        self.assertTrue((result.completed == 0).all())
        from tests.test_duel_v2_device_games import DeviceGameLoopTest
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        loop, _, _, _, _, _, _ = DeviceGameLoopTest().make(max_decisions=1)
        loop.search.prior_guide = search.prior_guide
        with self.assertRaisesRegex(ValueError, 'replaces'):
            DeviceGameLoop(loop.rules, loop.search, loop.sink, loop.world_seed_provider,
                max_decisions=1, device='cpu', allow_cpu_reference=True, allow_synthetic=True,
                root_guide=SimpleNamespace(capture_result=lambda *args: None))

    def test_real_rule_cpu_full32_root_prior_and_zero_shift_equivalence(self):
        from app.modules.card_game.rl import seven_runtime
        from app.modules.card_game.rl.recovery_search import search
        state = self.game(['I04', 'Z04', 'I01'])
        class Policy:
            def scores(self, x, c):
                return np.linspace(.1, -.1, len(c), dtype=np.float32)
            def wdl(self, x, actor):
                return np.array([.2, .3, .5])
            def scores_value(self, x, c):
                return self.scores(x, c), .3
        policies = {'a': Policy(), 'b': Policy()}
        options = dict(seed=39271, runtime=seven_runtime, simulations=32,
                       terminal_horizon=0, max_depth=4, noise=True)
        original = search(state, 'a', policies, **options)
        guide = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True)
        with patch.object(guide, 'score_numpy', wraps=guide.score_numpy) as calls:
            guided = search(state, 'a', policies, root_prior=guide, **options)
        self.assertEqual(calls.call_count, 1)
        self.assertTrue(guided['complete'])
        self.assertEqual(guided['simulations'], 32)
        self.assertEqual(guided['value'], original['value'])
        self.assertEqual(guided['raw_choice'], original['raw_choice'])
        self.assertTrue(np.array_equal(guided['prior_shifts'], guide.score_numpy(guided['x'], guided['c'])))
        disabled = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True,
                                      weights=ZhenhongPriorWeights(max_logit_shift=0))
        zero = search(state, 'a', policies, root_prior=disabled, **options)
        for name in ('pi', 'visits', 'root_prior', 'root_gumbel'):
            self.assertTrue(np.array_equal(zero[name], original[name]), name)
        self.assertEqual(zero['search_choice'], original['search_choice'])

    def test_tensor_fullgraph_has_no_data_dependent_host_break(self):
        state = self.game(['I04', 'Z04', 'I01'])
        _, x, c, valid = public_inputs(state)
        guide = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True)
        active = torch.tensor([True])
        def score(x, c, valid, active):
            result = guide.score_device(x, c, valid, active)
            return result.shifts, result.errors, result.applicable
        compiled = torch.compile(score, fullgraph=True, backend='eager')
        for left, right in zip(compiled(x, c, valid, active), score(x, c, valid, active)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_actual_cuda_capture_matches_cpu(self):
        state = self.game(['I04', 'Z04', 'I01'])
        _, expected, _ = self.score(state)
        actions, x, c, valid = public_inputs(state, device='cuda', compact=True)
        guide = ArrayZhenhongPrior(device='cuda')
        active = torch.ones(1, dtype=torch.bool, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                guide.score_device(x, c, valid, active)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = guide.score_device(x, c, valid, active)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(result.errors.cpu().tolist(), [0])
        torch.testing.assert_close(result.shifts[0].cpu(), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
