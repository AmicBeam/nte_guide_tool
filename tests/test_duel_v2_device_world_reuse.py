"""Decision-scoped particles: exact budgets, immutable roots and lease cleanup."""
import unittest
from unittest.mock import patch
import torch
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from tests import test_duel_v2_device_search as fixtures


class DecisionWorldReuseTest(unittest.TestCase):
    def make(self, active=None):
        helper = fixtures.DeviceSearchTest()
        engine, rules = helper.engine()
        engine.reuse_sampled_world = True
        engine.reuse_root_decision = True
        counts = torch.zeros(rules.rows, dtype=torch.int64)
        original = rules.sample_world_device
        def sample(states, viewers, seeds):
            counts.add_(states.active.long())
            return original(states, viewers, seeds)
        rules.sample_world_device = sample
        roots = helper.initialize(engine, rules, active=active)
        return helper, engine, rules, roots, counts

    def test_one_particle_per_decision_full32_and_immutable_particle(self):
        torch.set_num_threads(1)
        helper, engine, rules, roots, counts = self.make()
        engine.tick()
        particle = engine._sampled_slots.clone()
        snapshot = rules.data[particle].clone()
        self.assertEqual(counts.tolist(), [1, 1])
        for _ in range(1500):
            if not bool(engine.unfinished_device.any()):
                break
            before = engine._sampled_live.clone()
            engine.tick()
            torch.testing.assert_close(rules.data[particle], snapshot)
            still_live = before & engine._sampled_live
            self.assertTrue(bool(rules.live[particle[still_live]].all()))
        result = engine.result_device()
        self.assertEqual(result.completed.tolist(), [32, 32])
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(counts.tolist(), [1, 1])
        self.assertEqual(engine.world_samples.tolist(), [1, 1])
        self.assertEqual(int(rules.live.sum()), 2)
        self.assertTrue(bool(rules.live[roots.slots].all()))
        # Same deterministic worlds preserve the existing tree statistics.
        baseline, baseline_rules = helper.engine()
        helper.initialize(baseline, baseline_rules)
        expected = helper.finish_cpu(baseline)
        for name in ('actions', 'visits', 'policy', 'information_keys'):
            torch.testing.assert_close(getattr(result, name), getattr(expected, name))
        # The next physical decision must sample afresh, even if its view repeats.
        helper.initialize(engine, rules)
        helper.finish_cpu(engine)
        self.assertEqual(counts.tolist(), [2, 2])
        self.assertEqual(engine.world_samples.tolist(), [1, 1])
        self.assertEqual(int(rules.live.sum()), 2)

    def test_failure_drains_particle_and_transient_without_touching_inactive_root(self):
        helper, engine, rules, roots, counts = self.make(active=[True, False])
        engine.tick()
        engine.errors[0] = 2
        result = helper.finish_cpu(engine)
        self.assertEqual(counts.tolist(), [1, 0])
        self.assertEqual(result.completed.tolist(), [0, 0])
        self.assertFalse(bool(result.complete.any()))
        self.assertEqual(int(rules.live.sum()), 2)
        self.assertTrue(bool(rules.live[roots.slots].all()))
        self.assertFalse(bool(engine._sampled_live.any()))

    def test_ticks_have_no_host_tensor_reads(self):
        helper, engine, rules, roots, counts = self.make()
        for _ in range(8):
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
                 patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar copy')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host branch')):
                engine.tick()
        result = helper.finish_cpu(engine)
        self.assertEqual(result.completed.tolist(), [32, 32])
        self.assertEqual(counts.tolist(), [1, 1])

    def test_root_decision_is_reused_at_all32_starts(self):
        helper = fixtures.DeviceSearchTest()
        engine, rules = helper.engine()
        engine.reuse_sampled_world = True
        engine.reuse_root_decision = True
        calls = torch.zeros(2, dtype=torch.int64)
        original = rules.decision_device
        def decision(states, viewers):
            calls.add_((states.active & (rules.data[states.slots, 0] == 0)).long())
            return original(states, viewers)
        rules.decision_device = decision
        helper.initialize(engine, rules)
        initial = calls.clone()  # Fixture's value encoder also calls decision.
        result = helper.finish_cpu(engine)
        self.assertEqual(result.completed.tolist(), [32, 32])
        self.assertEqual(result.errors.tolist(), [0, 0])
        torch.testing.assert_close(calls, initial)

    def test_cpu_reference_samples_once_and_isolates_in_place_rollouts(self):
        from app.modules.card_game.rl import recovery_search
        state = dict(phase='playing', active_side='a', turn=1, events=[])
        sampled = dict(state, history=[])
        runtime = SimpleNamespace(
            decision=lambda s, side: ([{'type': 'left'}, {'type': 'right'}],
                (np.zeros(1, np.float32), np.eye(2, dtype=np.float32))),
            predict_encoded=lambda policy, x, c: (np.zeros(2), 0.))
        def advance(world, side, action):
            self.assertEqual(world['history'], [])
            world['history'].append(action['type'])
            world.update(phase='finished', winner='a')
            return world
        for mode, expected in (('per_decision', 1), ('per_simulation', 32)):
            with patch.object(recovery_search.core, 'sample_world',
                              side_effect=lambda *a, **kw: deepcopy(sampled)) as sampler, \
                 patch.object(recovery_search.core, 'simulate_action', side_effect=advance):
                result = recovery_search.search(state, 'a', {'a': None, 'b': None}, seed=7,
                    runtime=runtime, simulations=32, terminal_horizon=0,
                    world_sampling=mode)
            self.assertEqual(sampler.call_count, expected)
            self.assertEqual(result['worlds'], expected)
            self.assertEqual(result['simulations'], 32)
            self.assertTrue(result['complete'])
            self.assertEqual(sampled['history'], [])
            self.assertEqual(state['phase'], 'playing')


if __name__ == '__main__':
    unittest.main()
