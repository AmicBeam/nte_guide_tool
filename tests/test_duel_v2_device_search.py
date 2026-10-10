"""Controlled tensor-tree FSM tests, never evidence of formal-rule execution."""
from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_duel.device_rule_protocol import (
    DeviceStateBatch, DeviceMeta, DeviceDecision, DeviceValueObservation,
)
from app.modules.card_game.rl.batched_search.device_search import (
    DeviceGumbelSearch, add_world_seed_stride, METADATA_ERROR, OPPONENT,
)
from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
from app.modules.card_game.rl import search_policy, recovery_search
from app.modules.card_game.rl.information_search import Node


class TensorTreeFixture:
    """Three-ply tree: root action determines terminal fixed-viewer payoff."""
    def __init__(self, rows, device='cpu', actions=3):
        import torch
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.rows, self.capacity = rows, 1024
        self.actions = actions
        self.viewer = torch.arange(rows, device=self.device) % 2
        self.base = torch.arange(rows, device=self.device) * self.capacity
        self.data = torch.zeros((rows * self.capacity, 2), dtype=torch.int64, device=self.device)
        self.data[:, 1] = -1
        self.cursor = torch.zeros(rows, dtype=torch.int64, device=self.device)
        self.live = torch.zeros(rows * self.capacity, dtype=torch.bool, device=self.device)
        self.live[self.base] = True
        self.sample_seeds = torch.zeros((rows, 5), dtype=torch.int64, device=self.device)

    def roots(self):
        t = self.t
        return DeviceStateBatch(self, self.base.clone(), t.ones_like(self.base),
            t.arange(self.rows, device=self.device), t.ones(self.rows, dtype=t.bool, device=self.device),
            t.zeros(self.rows, dtype=t.int32, device=self.device))

    def _new(self, states, values):
        t = self.t
        self.cursor.add_(states.active.long())
        slots = t.where(states.active, self.base + self.cursor, states.slots)
        self.data[slots] = t.where(states.active[:, None], values, self.data[slots])
        self.live[slots] = t.where(states.active, True, self.live[slots])
        return DeviceStateBatch(self, slots, t.ones_like(slots), states.owners,
                                states.active, t.zeros_like(states.errors))

    def metadata_device(self, states):
        t = self.t
        values = self.data[states.slots]
        finished = values[:, 0] >= 3
        actors = self.viewer ^ values[:, 0].remainder(2)
        winners = t.where(values[:, 1] == 1, self.viewer,
                          t.where(values[:, 1] == 0, 1-self.viewer, -1))
        return DeviceMeta(t.where(finished, -1, actors), values[:, 0].to(t.int32),
                           t.where(finished, winners, -2), finished, values[:, 0],
                           t.zeros(self.rows, dtype=t.int32, device=self.device))

    def decision_device(self, states, observers):
        t = self.t
        data = self.data[states.slots]
        observations = t.stack((data[:, 0], data[:, 1], observers), -1).to(t.float32)
        candidates = t.zeros((self.rows, self.actions, 2), device=self.device)
        candidates[:, :, 0] = t.arange(self.actions, device=self.device)[None, :]
        candidates[:, :, 1] = data[:, 0, None]
        parameters = dict(states=states, action_pool=self, action_slots=states.slots.clone(),
            action_generations=states.generations.clone(),
            action_refs=t.arange(self.actions, dtype=t.int32, device=self.device).expand(self.rows, -1).clone(),
            counts=t.full((self.rows,), self.actions, dtype=t.int64, device=self.device),
            valid=t.ones((self.rows, self.actions), dtype=t.bool, device=self.device),
            observations=observations, candidates=candidates,
            errors=t.zeros(self.rows, dtype=t.int32, device=self.device))
        if 'action_types' in {field.name for field in fields(DeviceDecision)}:
            parameters['action_types'] = t.zeros((self.rows, self.actions), dtype=t.int64, device=self.device)
        return DeviceDecision(**parameters)

    def sample_world_device(self, states, viewers, seeds):
        self.sample_seeds.copy_(self.t.where(states.active[:, None], seeds, self.sample_seeds))
        return self._new(states, self.data[states.slots])

    def step_device(self, states, actors, decision, action_indices, *, physical=False):
        t = self.t
        data = self.data[states.slots]
        first = t.where(data[:, 0] == 0, action_indices, data[:, 1])
        return self._new(states, t.stack((data[:, 0]+1, first), -1))

    def clean_device(self, states):
        return states

    def teacher_continue_device(self, states, viewers, *, horizon=3):
        data = self.data[states.slots]
        return self._new(states, self.t.stack(((data[:, 0]+horizon).clamp_max(3), data[:, 1]), -1))

    def rule_decision_device(self, states, actors):
        decision = self.decision_device(states, actors)
        return decision, self.t.zeros(self.rows, dtype=self.t.int64, device=self.device), decision.errors

    def value_observation_device(self, states, viewers):
        t = self.t
        meta = self.metadata_device(states)
        decision = self.decision_device(states, viewers)
        value = t.where(meta.winners == viewers, 1., t.where(meta.winners >= 0, -1., 0.)).double()
        return DeviceValueObservation(decision.observations,
            (meta.actors == viewers).float()[:, None], meta.finished, value, meta.errors)

    def release_device(self, states):
        self.live[states.slots] = self.t.where(states.active, False, self.live[states.slots])


class FixtureInference:
    candidate_dim = 2

    def predict_device(self, x, candidates, valid, ids, value_actor, *, value_only=None):
        import torch
        index = torch.arange(candidates.shape[1], device=x.device)
        base = index.float() * -.1 + .1
        base = torch.where(index == 0, torch.full_like(base, .2), base)
        logits = base.expand(x.shape[0], -1).clone()
        wdl = torch.full((x.shape[0], 3), .4, device=x.device)
        wdl[:, 1] = .2
        return SimpleNamespace(logits=logits, wdl=wdl,
                               invalid_rows=torch.zeros(x.shape[0], dtype=torch.bool, device=x.device))


class InjectiveFixtureKeys:
    """Exact fixture identity; actual SHA-256 has separate CPU/CUDA tests."""
    def hash(self, x, candidates, counts, active):
        import torch
        keys = torch.zeros((len(x), 8), dtype=torch.int64, device=x.device)
        keys[:, :3] = x.contiguous().view(torch.int32).long() & 0xffffffff
        keys[:, 3] = counts
        keys.masked_fill_(~active[:, None], 0)
        return SimpleNamespace(keys=keys, errors=torch.zeros(len(x), dtype=torch.int32, device=x.device))


class DeviceSearchTest(unittest.TestCase):
    def engine(self, rows=2, device='cpu', simulations=32, actions=3):
        fixture = TensorTreeFixture(rows, device, actions=actions)
        stats = DeviceStatisticsPool(rows*(simulations+1), actions, device=str(fixture.device),
            max_batch=rows, max_path_len=4, allow_cpu_reference=device == 'cpu')
        engine = DeviceGumbelSearch(fixture, FixtureInference(), FixtureInference(), stats,
            InjectiveFixtureKeys(), rows=rows, max_actions=actions, simulations=simulations,
            max_depth=4, terminal_horizon=3, device=str(fixture.device),
            allow_cpu_reference=device == 'cpu', allow_synthetic=True)
        return engine, fixture

    def initialize(self, engine, fixture, active=None):
        import torch
        roots = fixture.roots()
        if active is not None:
            roots.active.copy_(torch.tensor(active, device=fixture.device))
        seeds = torch.zeros((fixture.rows, 5), dtype=torch.int64, device=fixture.device)
        seeds[:, 4] = 1
        ids = torch.zeros((fixture.rows, 2), dtype=torch.int64, device=fixture.device)
        engine.initialize(roots, fixture.viewer, seeds, ids, ids)
        return roots

    def finish_cpu(self, engine):
        for _ in range(1500):
            if not bool(engine.unfinished_device.any()):
                return engine.result_device()
            engine.tick()
        self.fail('Bounded fixture FSM did not finish')

    def test_full_budget_matches_numpy_root_semantics_without_value_sign_flip(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = self.initialize(engine, fixture)
        result = self.finish_cpu(engine)
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(result.complete.tolist(), [True, True])
        self.assertEqual(result.completed.tolist(), [32, 32])
        prior = search_policy.normalized_exp(np.asarray([.2, 0., -.1], dtype=np.float32).astype(np.float64))
        node = Node.create(prior, 0.)
        values = [-1., 1., 0.]
        noise = np.zeros(3)
        for visit in search_policy.visit_schedule(3, 32):
            action = recovery_search.choose_root(node, noise, visit, 'natural_wdl')
            node.visits[action] += 1
            node.total[action] += values[action]
        for i in range(2):
            np.testing.assert_array_equal(result.visits[i].numpy(), node.visits)
            np.testing.assert_allclose(result.policy[i].numpy(),
                                      recovery_search.improved_policy(node, 'natural_wdl'), atol=1e-7)
            self.assertEqual(int(result.actions[i]), recovery_search.choose_root(
                node, noise, int(node.visits.max()), 'natural_wdl'))
            self.assertGreater(int(result.node_count[i]), 1)
            self.assertGreater(int(result.phase_work[OPPONENT, i]), 0)
        self.assertEqual(int(fixture.live.sum()), 2)
        torch.testing.assert_close(fixture.data[roots.slots], torch.tensor([[0, -1], [0, -1]]))

    def test_inactive_row_preserved_and_tick_has_no_host_tensor_reads(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        self.initialize(engine, fixture, active=[True, False])
        for _ in range(5):
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
                 patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
                 patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('List copy')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
                engine.tick()
        result = self.finish_cpu(engine)
        self.assertEqual(result.complete.tolist(), [True, False])
        self.assertEqual(result.completed.tolist(), [32, 0])
        self.assertEqual(int(fixture.cursor[1]), 0)

    def test_no_silent_teacher_substitution_and_invalid_root_fails_closed(self):
        engine, fixture = self.engine()
        ids = fixture.viewer.clone()
        fixture.viewer[0] = 1
        roots = fixture.roots()
        import torch
        zeros = torch.zeros((2, 2), dtype=torch.int64)
        engine.initialize(roots, ids, torch.zeros((2, 5), dtype=torch.int64), zeros, zeros)
        result = self.finish_cpu(engine)
        self.assertEqual(int(result.errors[0]) & METADATA_ERROR, METADATA_ERROR)
        self.assertFalse(bool(result.complete[0]))
        with self.assertRaises(NotImplementedError):
            DeviceGumbelSearch(fixture, FixtureInference(), FixtureInference(), engine.statistics,
                InjectiveFixtureKeys(), rows=2, max_actions=3, teacher_mode='network',
                device='cpu', allow_cpu_reference=True, allow_synthetic=True)

    def test_seed_stride_preserves_all_320_bits_and_carry(self):
        import torch
        seeds = [(1 << 256) | (1 << 255) | ((1 << 64)-1),
                 (1 << 256) | ((1 << 128)-1)]
        raw = np.asarray([[(seed >> (64*j)) & ((1 << 64)-1) for j in range(5)]
                          for seed in seeds], dtype=np.uint64).view(np.int64)
        actual = add_world_seed_stride(torch.from_numpy(raw), 31).numpy().view(np.uint64)
        for i, seed in enumerate(seeds):
            value = sum(int(word) << (64*j) for j, word in enumerate(actual[i]))
            self.assertEqual(value, seed + 31*104729)

    def test_actual_cuda_1600_fixture_ticks_can_be_captured(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; controlled tensor tree only')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            engine, fixture = self.engine(rows=1600, device='cuda', simulations=4)
            self.initialize(engine, fixture)
            for _ in range(5):
                engine.tick()
            # Reinitialize only after draining/cleaning the previous synthetic cohort.
            for _ in range(120):
                engine.tick()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.initialize(engine, fixture)
            for _ in range(120):
                engine.tick()
            captured = engine.result_device()
        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(bool((captured.errors != 0).any()))
        self.assertTrue(bool(captured.complete.all()))
        self.assertEqual(int(captured.peak_device_roots), 1600)
        self.assertTrue(bool((captured.completed == 4).all()))

    def test_actual_cuda_1600_full_budget_conditional_fsm_without_host_predicate(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; controlled tensor tree, zero formal games')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        engine, fixture = self.engine(rows=1600, device='cuda', simulations=32, actions=20)
        # All shapes/constants are allocated before the conditional capture.
        roots = fixture.roots()
        seeds = torch.zeros((1600, 5), dtype=torch.int64, device=fixture.device)
        seeds[:, 4] = 1
        ids = torch.zeros((1600, 2), dtype=torch.int64, device=fixture.device)
        runner = CudaConditionalWhile(device=str(fixture.device), max_ticks=3000, seconds=120)
        self.addCleanup(runner.close)
        engine.statistics.bind_capture_streams(runner.capture_stream, runner.body_stream)
        with engine.statistics.capture_scope():
            runner.capture(engine.unfinished_device, engine.tick,
                initialize=lambda: engine.initialize(roots, fixture.viewer, seeds, ids, ids))
        # Only one whole graph replay; no CPU while/any/item controls the search.
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host predicate')):
            runner.replay()
        torch.cuda.synchronize()
        result = engine.result_device()
        self.assertFalse(bool(runner.errors.any()))
        self.assertTrue(bool(result.complete.all()))
        self.assertTrue(bool((result.completed == 32).all()))
        self.assertTrue(bool((result.errors == 0).all()))
        self.assertTrue(bool((result.visits.sum(-1) == 32).all()))
        self.assertEqual(result.policy.shape, (1600, 20))
        self.assertTrue(bool((result.policy.sum(-1)-1).abs().max() < 1e-6))
        self.assertEqual(int(result.peak_device_roots), 1600)
        self.assertGreater(int(runner.counter), 32)


if __name__ == '__main__':
    unittest.main()
