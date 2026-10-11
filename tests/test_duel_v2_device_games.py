"""Controlled 3-ply physical game-loop tests; never full-rules evidence."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch, DeviceMeta
from app.modules.card_game.rl.batched_search.device_games import (
    DeviceGameLoop, SINK_ERROR,
)
from app.modules.card_game.rl.batched_search.device_search import (
    DeviceGumbelSearch, METADATA_ERROR, RULE_ERROR,
)
from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
from tests.test_duel_v2_device_search import (
    FixtureInference, InjectiveFixtureKeys, TensorTreeFixture,
)


class FixtureWorldSeeds:
    def derive(self, job_seeds, steps, actors, active):
        import torch
        limbs = torch.zeros((job_seeds.shape[0], 5), dtype=torch.int64, device=job_seeds.device)
        limbs[:, 0] = job_seeds
        limbs[:, 1] = steps
        limbs[:, 2] = actors
        limbs[:, 4] = 1
        limbs.masked_fill_(~active[:, None], 0)
        return SimpleNamespace(limbs=limbs, errors=torch.zeros(
            job_seeds.shape[0], dtype=torch.int32, device=job_seeds.device))


class RecordingSink:
    def __init__(self, rows, device, capacity=10 ** 9):
        import torch
        self.t = torch
        self.device = device
        self.capacity = capacity
        self.begins = torch.zeros(rows, dtype=torch.int32, device=device)
        self.appends = torch.zeros(rows, dtype=torch.int64, device=device)
        self.finishes = torch.zeros(rows, dtype=torch.int32, device=device)
        self.policy = None
        self.visit_policy = None
        self.actions = None
        self.observations = None
        self.physical = torch.zeros(rows, dtype=torch.bool, device=device)

    def begin_device(self, roots, policy_ids, aux_ids, job_seeds, train_sides, active):
        self.begins.add_(active.to(self.t.int32))
        return self.t.zeros_like(self.begins)

    def append_device(self, roots, actors, steps, search_result, active):
        overflow = active & (self.appends >= self.capacity)
        admitted = active & ~overflow
        self.appends.add_(admitted.long())
        policy = search_result.policy
        if self.policy is None:
            self.policy = policy.clone()
            self.visit_policy = search_result.visit_policy.clone()
            self.actions = search_result.actions.clone()
            self.observations = search_result.decision.observations.clone()
        else:
            expand = admitted.reshape((admitted.shape[0],) + (1,) * (policy.ndim - 1))
            self.policy.copy_(self.t.where(expand, policy, self.policy))
            self.visit_policy.copy_(self.t.where(expand, search_result.visit_policy, self.visit_policy))
            self.actions.copy_(self.t.where(admitted, search_result.actions, self.actions))
            self.observations.copy_(self.t.where(admitted[:, None], search_result.decision.observations,
                                                 self.observations))
        return overflow.to(self.t.int32)

    def finish_device(self, roots, winners, steps, complete, errors):
        self.finishes.add_(roots.active.to(self.t.int32))
        return self.t.zeros_like(errors)


class DeviceGameLoopTest(unittest.TestCase):
    def make(self, rows=2, device='cpu', simulations=2, actions=3, max_decisions=8,
             sink_capacity=10 ** 9, active=None, noise=False):
        import torch
        torch.set_num_threads(1)
        fixture = TensorTreeFixture(rows, device, actions=actions)
        stats = DeviceStatisticsPool(rows * (simulations + 1), actions, device=str(fixture.device),
            max_batch=rows, max_path_len=4, allow_cpu_reference=device == 'cpu')
        search = DeviceGumbelSearch(fixture, FixtureInference(), FixtureInference(), stats,
            InjectiveFixtureKeys(), rows=rows, max_actions=actions, simulations=simulations,
            gumbel_candidates=min(16, actions), max_depth=4, terminal_horizon=3,
            device=str(fixture.device), noise=noise, allow_cpu_reference=device == 'cpu',
            allow_synthetic=True)
        sink = RecordingSink(rows, fixture.device, capacity=sink_capacity)
        loop = DeviceGameLoop(fixture, search, sink, FixtureWorldSeeds(), max_decisions=max_decisions,
            device=str(fixture.device), allow_cpu_reference=device == 'cpu', allow_synthetic=True,
            inner_max_ticks=400)
        roots = fixture.roots()
        if active is not None:
            roots.active.copy_(torch.tensor(active, dtype=torch.bool, device=fixture.device))
        job_seeds = torch.arange(rows, dtype=torch.int64, device=fixture.device)
        ids = torch.zeros((rows, 2), dtype=torch.int64, device=fixture.device)
        train = torch.ones((rows, 2), dtype=torch.bool, device=fixture.device)
        original_step = fixture.step_device
        def tracked(states, actors, decision, action_indices, *, physical=False):
            sink.physical.logical_or_(states.active & physical)
            return original_step(states, actors, decision, action_indices, physical=physical)
        fixture.step_device = tracked
        return loop, fixture, sink, roots, job_seeds, ids, train

    def finish_cpu(self, loop, roots, job_seeds, ids, train):
        loop.initialize(roots, job_seeds, ids, ids, train)
        for _ in range(loop.max_decisions + 2):
            if not bool(loop.unfinished_device.any()):
                return loop.result_device()
            loop.tick()
        return loop.result_device()

    def test_constructor_requires_cuda_without_explicit_fixture_flags(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        with self.assertRaises(ValueError):
            DeviceGameLoop(fixture, loop.search, sink, FixtureWorldSeeds(), max_decisions=3,
                           device='cpu')
        self.assertFalse(loop.supports_full_duel_rules)
        self.assertFalse(hasattr(loop, 'device_validation_receipt'))
        with self.assertRaises(RuntimeError):
            loop.capture(roots, job_seeds, ids, ids, train)

    def test_three_ply_completion_records_sink_before_physical_commit(self):
        import torch
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        initial = roots.slots.clone()
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(result.complete.tolist(), [True, True])
        self.assertEqual(result.truncated.tolist(), [False, False])
        self.assertEqual(result.steps.tolist(), [3, 3])
        self.assertEqual(result.searched_count.tolist(), [3, 3])
        self.assertEqual(loop.steps_device.tolist(), [3, 3])
        self.assertEqual(loop.source_versions_device.tolist(), [2, 2])
        self.assertEqual(result.winners.tolist(), fixture.viewer.tolist())
        self.assertTrue(bool(sink.physical.all()))
        self.assertEqual(sink.begins.tolist(), [1, 1])
        self.assertEqual(sink.appends.tolist(), [3, 3])
        self.assertEqual(sink.finishes.tolist(), [1, 1])
        self.assertIsNotNone(sink.policy)
        self.assertEqual(sink.policy.shape[1], 3)
        self.assertTrue(torch.allclose(sink.policy.sum(-1), torch.ones(2)))
        self.assertFalse(bool(fixture.live[initial].any()))
        self.assertTrue(bool(fixture.live[result.states.slots].all()))
        self.assertTrue(torch.equal(result.states.owners, torch.arange(2)))

    def test_physical_cleanup_replaces_and_releases_intermediate_states(self):
        import torch
        loop, fixture, sink, roots, seeds, ids, train = self.make()
        original_step = fixture.step_device
        physical = torch.zeros_like(fixture.live)
        clean_count = torch.zeros(2, dtype=torch.int64)
        mids = []
        def step(states, actors, decision, indices, *, physical=False):
            out = original_step(states, actors, decision, indices, physical=physical)
            if physical:
                mids.append((out.slots.clone(), out.active.clone()))
                marked[out.slots] |= out.active
            return out
        marked = physical
        def clean(states):
            clean_count.add_((marked[states.slots] & states.active).long())
            return fixture._new(states, fixture.data[states.slots])
        fixture.step_device, fixture.clean_device = step, clean
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertEqual(result.complete.tolist(), [True, True])
        self.assertEqual(clean_count.tolist(), [3, 3])
        for slots, active in mids:
            self.assertFalse(bool(fixture.live[slots[active]].any()))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertTrue(bool(fixture.live[result.states.slots].all()))

    def test_physical_cleanup_failure_releases_new_states_preserves_original(self):
        import torch
        loop, fixture, sink, roots, seeds, ids, train = self.make()
        initial = roots.slots.clone()
        marked = torch.zeros_like(fixture.live)
        original_step = fixture.step_device
        def step(states, actors, decision, indices, *, physical=False):
            out = original_step(states, actors, decision, indices, physical=physical)
            if physical:marked[out.slots] |= out.active
            return out
        def clean(states):
            failed = marked[states.slots] & states.active
            out = fixture._new(states, fixture.data[states.slots])
            return DeviceStateBatch(out.pool, out.slots, out.generations, out.owners,
                                    out.active, failed.to(torch.int32))
        fixture.step_device, fixture.clean_device = step, clean
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & RULE_ERROR).all()))
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertTrue(torch.equal(result.states.slots, initial))
        self.assertEqual(int(fixture.live.sum()), 2)

    def test_physical_cleanup_cannot_steal_other_games_intermediate(self):
        import torch
        loop, fixture, sink, roots, seeds, ids, train = self.make()
        initial = roots.slots.clone()
        marked = torch.zeros_like(fixture.live)
        original_step = fixture.step_device
        def step(states, actors, decision, indices, *, physical=False):
            out = original_step(states, actors, decision, indices, physical=physical)
            if physical:marked[out.slots] |= out.active
            return out
        def clean(states):
            physical = marked[states.slots] & states.active
            slots = torch.where(physical, states.slots.flip(0), states.slots)
            return DeviceStateBatch(states.pool, slots, states.generations, states.owners,
                                    states.active, states.errors)
        fixture.step_device, fixture.clean_device = step, clean
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))
        self.assertTrue(torch.equal(result.states.slots, initial))
        self.assertEqual(int(fixture.live.sum()), 2)

    def test_inactive_row_never_searches_or_steps(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make(active=[True, False])
        initial = roots.slots.clone()
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertEqual(result.complete.tolist(), [True, False])
        self.assertEqual(result.searched_count.tolist(), [3, 0])
        self.assertEqual(result.steps.tolist(), [3, 0])
        self.assertEqual(int(fixture.cursor[1]), 0)
        self.assertTrue(bool(fixture.live[initial[1]]))
        self.assertEqual(sink.appends.tolist(), [3, 0])
        self.assertEqual(sink.physical.tolist(), [True, False])

    def test_inactive_physical_lease_is_protected_from_failed_commit(self):
        import torch
        loop, fixture, sink, roots, seeds, ids, train = self.make(active=[True, False])
        initial = roots.slots.clone()
        step = fixture.step_device
        def foreign(states, actors, decision, indices, *, physical=False):
            if not physical:
                return step(states, actors, decision, indices, physical=False)
            return DeviceStateBatch(fixture, initial[1].expand(2).clone(),
                roots.generations.clone(), roots.owners.flip(0).clone(), states.active,
                torch.zeros_like(states.errors))
        fixture.step_device = foreign
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertTrue(bool(result.errors[0] & METADATA_ERROR))
        self.assertEqual(int(result.errors[1]), 0)

    def test_duplicate_lease_detection_with_inactive_row_between(self):
        import torch
        loop, *_ = self.make(rows=3)
        duplicated = loop._duplicate_leases(torch.tensor([17, 17, 17]),
            torch.ones(3, dtype=torch.int64), torch.tensor([True, False, True]))
        self.assertEqual(duplicated.tolist(), [True, False, True])

    def test_failed_commit_does_not_reclaim_foreign_owner_new_lease(self):
        from dataclasses import replace
        loop, fixture, sink, roots, seeds, ids, train = self.make(active=[True, False])
        initial = roots.slots.clone()
        original = fixture.step_device
        returned = []
        def foreign(states, actors, decision, indices, *, physical=False):
            advanced = original(states, actors, decision, indices, physical=physical)
            if physical:
                returned.append(advanced.slots.clone())
                return replace(advanced, owners=advanced.owners.flip(0).clone())
            return advanced
        fixture.step_device = foreign
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertTrue(bool(result.errors[0] & METADATA_ERROR))
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertTrue(bool(fixture.live[returned[0][0]]))

    def test_active_root_cannot_alias_an_inactive_input_lease(self):
        loop, fixture, sink, roots, seeds, ids, train = self.make(active=[True, False])
        roots.slots[0] = roots.slots[1]
        result = self.finish_cpu(loop, roots, seeds, ids, train)
        self.assertTrue(bool(result.errors[0] & METADATA_ERROR))
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertEqual(sink.begins.tolist(), [0, 0])

    def test_search_failure_does_not_advance_physical_root(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        initial = roots.slots.clone()
        original = fixture.sample_world_device
        def fail_sample(states, viewers, seeds):
            sampled = original(states, viewers, seeds)
            errors = sampled.errors.new_ones(sampled.errors.shape)
            return DeviceStateBatch(sampled.pool, sampled.slots, sampled.generations,
                                    sampled.owners, sampled.active, errors)
        fixture.sample_world_device = fail_sample
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & RULE_ERROR).all()))
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertEqual(sink.physical.tolist(), [False, False])

    def test_sink_overflow_prevents_physical_advance(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make(sink_capacity=0)
        initial = roots.slots.clone()
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & SINK_ERROR).all()))
        self.assertEqual(result.steps.tolist(), [0, 0])
        self.assertEqual(sink.appends.tolist(), [0, 0])
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertEqual(int(fixture.cursor[0]), int(fixture.cursor[1]))
        self.assertEqual(sink.physical.tolist(), [False, False])

    def test_max_decisions_truncates_without_draw_or_wdl(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make(max_decisions=1)
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertEqual(result.complete.tolist(), [False, False])
        self.assertEqual(result.truncated.tolist(), [True, True])
        self.assertEqual(result.winners.tolist(), [-2, -2])
        self.assertEqual(result.steps.tolist(), [1, 1])
        self.assertEqual(result.searched_count.tolist(), [1, 1])
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(sink.finishes.tolist(), [1, 1])

    def test_physical_alias_of_another_root_is_rejected(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        initial = roots.slots.clone()
        original = fixture.step_device
        def steal(states, actors, decision, action_indices, *, physical=False):
            if not physical:
                return original(states, actors, decision, action_indices, physical=physical)
            return DeviceStateBatch(fixture, initial.flip(0), roots.generations.flip(0),
                                    states.owners, states.active, states.errors)
        fixture.step_device = steal
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertEqual(result.states.slots.tolist(), initial.tolist())

    def test_successful_replace_releases_old_lease_and_failure_cleans_new_lease(self):
        import torch
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        initial = roots.slots.clone()
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertTrue(bool(result.complete.all()))
        self.assertFalse(bool(fixture.live[initial].any()))
        self.assertTrue(bool(fixture.live[result.states.slots].all()))

        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        initial = roots.slots.clone()
        original = fixture.step_device
        def bad_owner(states, actors, decision, action_indices, *, physical=False):
            advanced = original(states, actors, decision, action_indices, physical=physical)
            return DeviceStateBatch(fixture, advanced.slots, advanced.generations,
                                    advanced.owners + 1, advanced.active, advanced.errors)
        fixture.step_device = bad_owner
        failed = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertFalse(bool(failed.complete.any()))
        self.assertTrue(bool((failed.errors & METADATA_ERROR).all()))
        self.assertTrue(bool(fixture.live[initial].all()))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertTrue(torch.equal(failed.states.slots, initial))

    def test_terminal_winner_cannot_become_draw_from_unfinished_code(self):
        import torch
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        original = fixture.metadata_device
        def malformed(states):
            meta = original(states)
            winners = torch.where(meta.finished, torch.full_like(meta.winners, -2), meta.winners)
            return DeviceMeta(meta.actors, meta.phases, winners, meta.finished, meta.versions, meta.errors)
        fixture.metadata_device = malformed
        result = self.finish_cpu(loop, roots, job_seeds, ids, train)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))
        self.assertEqual(result.winners.tolist(), [-2, -2])
        self.assertEqual(result.truncated.tolist(), [False, False])

    def test_capture_rejects_already_frozen_runners(self):
        loop, fixture, sink, roots, job_seeds, ids, train = self.make()
        frozen_runner = SimpleNamespace(graph=object(), _loop_built=True, body_stream=None, errors=None)
        with self.assertRaisesRegex(RuntimeError, 'already frozen'):
            DeviceGameLoop(fixture, loop.search, sink, FixtureWorldSeeds(), max_decisions=3,
                device='cpu', allow_cpu_reference=True, allow_synthetic=True,
                outer_loop=frozen_runner, inner_loop=frozen_runner)
        import torch
        if not torch.cuda.is_available():
            return
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        cuda_loop, fixture, sink, roots, job_seeds, ids, train = self.make(device='cuda')
        inner = CudaConditionalWhile(device=str(fixture.device), max_ticks=4, seconds=5)
        outer = CudaConditionalWhile(device=str(fixture.device), max_ticks=4, seconds=5)
        try:
            predicate = torch.zeros(2, dtype=torch.uint8, device=fixture.device)
            inner.capture(predicate, lambda: None)
            with self.assertRaisesRegex(RuntimeError, 'already frozen'):
                DeviceGameLoop(fixture, cuda_loop.search, sink, FixtureWorldSeeds(), max_decisions=3,
                    device=str(fixture.device), allow_cpu_reference=False, allow_synthetic=True,
                    outer_loop=outer, inner_loop=inner)
        finally:
            inner.close()
            outer.close()

    def test_actual_cuda_nested_loop_may_skip_locally(self):
        import torch
        from unittest.mock import patch as mock_patch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; nested WHILE integration, zero formal games')
        loop, fixture, sink, roots, job_seeds, ids, train = self.make(
            rows=1600, device='cuda', simulations=32, actions=20)
        production = DeviceGameLoop(fixture, loop.search, sink, FixtureWorldSeeds(), max_decisions=4,
            device=str(fixture.device), allow_cpu_reference=False, allow_synthetic=False,
            inner_max_ticks=2000, outer_max_ticks=8, seconds=60)
        self.addCleanup(production.close)
        production.capture(roots, job_seeds, ids, ids, train)
        with mock_patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host loop predicate')), \
             mock_patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host root stepping')):
            production.replay()
        torch.cuda.synchronize()
        result = production.result_device()
        self.assertFalse(bool((result.errors != 0).any()))
        self.assertTrue(bool(result.complete.all()))
        self.assertTrue(bool((result.steps == 3).all()))
        self.assertEqual(int(production._outer.errors), 0)
        self.assertEqual(int(production._inner.errors), 0)


if __name__ == '__main__':
    unittest.main()
