"""Distinct action lifetime CPU/CUDA oracles; not actual duel qualification."""
from dataclasses import replace
import unittest

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceValueObservation
from app.modules.card_game.rl.batched_search.device_action_leases import release_actions_device
from tests import test_duel_v2_device_search as search_cases
from tests import test_duel_v2_device_games as game_cases
from tests import test_duel_v2_device_pure_policy as pure_cases
from tests import test_duel_v2_device_graph_storage as graph_cases


def distinct_actions(fixture):
    import torch
    original = fixture.decision_device
    original_release = fixture.release_device
    released = torch.zeros_like(fixture.live, dtype=torch.int64)

    def decision(states, viewers):
        out = original(states, viewers)
        actions = fixture._new(states, fixture.data[states.slots])
        return replace(out, action_slots=torch.where(states.active, actions.slots, -1),
                       action_generations=torch.where(states.active, actions.generations, 0))

    def value(states, viewers):
        meta = fixture.metadata_device(states)
        obs = original(states, viewers).observations
        z = torch.where(meta.winners == viewers, 1., torch.where(meta.winners >= 0, -1., 0.)).double()
        return DeviceValueObservation(obs, (meta.actors == viewers).float()[:, None],
                                      meta.finished, z, meta.errors)

    def release(states):
        safe = states.slots.clamp(0, fixture.live.numel() - 1)
        # Device assertions are checked only after the whole fixture finishes.
        released.index_add_(0, safe, states.active.long())
        return original_release(states)

    fixture.decision_device, fixture.value_observation_device = decision, value
    fixture.release_device = release
    return released


class DistinctActionLeaseTest(unittest.TestCase):
    def test_interrupted_search_drains_independent_action_arena_and_worlds_on_device_path(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_games import GUARD_ERROR
        case=game_cases.DeviceGameLoopTest()
        for cutoff in (1,3,5,9,400):
            for particle in (False,True):
                loop,fixture,sink,roots,seeds,ids,train=case.make(simulations=32,max_decisions=1,
                                                              active=[True,False])
                loop._inner_max_ticks=cutoff
                loop.search.reuse_sampled_world=particle
                loop.search.reuse_root_decision=particle
                arena=graph_cases.DeviceGraphStorageTest().store(slots=16,pages=32)
                self.addCleanup(arena.close)
                graph=graph_cases._graph([{'type':'end_turn'}]*fixture.rows,'cpu')
                original=fixture.decision_device
                def decision(states,viewers):
                    result=original(states,viewers)
                    lease=arena.put_device(graph,states.owners,states.active.view(torch.uint8))
                    return replace(result,action_pool=arena,action_slots=lease.slots,
                                   action_generations=lease.generations,errors=result.errors|lease.errors)
                def value(states,viewers):
                    meta=fixture.metadata_device(states)
                    return DeviceValueObservation(original(states,viewers).observations,
                        (meta.actors==viewers).float()[:,None],meta.finished,
                        torch.where(meta.winners==viewers,1.,torch.where(meta.winners>=0,-1.,0.)).double(),meta.errors)
                fixture.decision_device=decision;fixture.value_observation_device=value
                result=case.finish_cpu(loop,roots,seeds,ids,train)
                self.assertEqual(result.errors.tolist(),[0 if cutoff==400 else GUARD_ERROR,0],(cutoff,particle))
                self.assertEqual(result.steps.tolist(),[1 if cutoff==400 else 0,0])
                self.assertFalse(bool(result.complete.any()))
                self.assertEqual(result.truncated.tolist(),[cutoff==400,False])
                self.assertEqual(int(arena.live.sum()),0,(cutoff,particle,'action arena'))
                self.assertEqual(int(fixture.live.sum()),2,(cutoff,particle,'state arena'))
                self.assertTrue(bool(fixture.live[result.states.slots].all()))
                if cutoff!=400:self.assertTrue(bool(fixture.live[roots.slots].all()))
                self.assertEqual(int(loop.search.statistics._occupied.sum()),0)
                self.assertFalse(bool(loop.search.unfinished_device.any()))
                self.assertEqual(bool((loop.search.completed<loop.search.requested)[0]),cutoff!=400)

    def test_explicit_abort_is_idempotent_and_keeps_budget_keys_and_counts(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_games import GUARD_ERROR
        case=search_cases.DeviceSearchTest();search,fixture=case.engine(simulations=32)
        released=distinct_actions(fixture)
        roots=case.initialize(search,fixture)
        search.tick();search.tick()
        before={name:getattr(search,name).clone() for name in ('completed','requested','keys','world_samples')}
        for _ in range(2):
            search.abort_device(roots.active,error_code=GUARD_ERROR)
            for name,value in before.items():torch.testing.assert_close(getattr(search,name),value)
            self.assertFalse(bool(search.unfinished_device.any()))
            self.assertEqual(int(search.statistics._occupied.sum()),0)
            self.assertEqual(int(fixture.live.sum()),4) # physical roots plus borrowed root actions
        result=search.result_device()
        self.assertFalse(bool(result.complete.any()))
        self.assertEqual(result.errors.tolist(),[GUARD_ERROR,GUARD_ERROR])
        self.assertEqual(release_actions_device(result.decision,roots.active,roots.owners,
                                               protected=(roots,)).tolist(),[0,0])
        self.assertEqual(int(fixture.live.sum()),2)
        self.assertLessEqual(int(released.max()),1)

    def test_first_zero_generation_physical_roots_are_admitted(self):
        import torch
        case = game_cases.DeviceGameLoopTest()
        loop, fixture, sink, roots, seeds, ids, train = case.make()
        roots.generations.zero_()
        result = case.finish_cpu(loop, roots, seeds, ids, train)
        self.assertTrue(bool(result.complete.all()))
        self.assertEqual(result.errors.tolist(), [0, 0])

    def test_checked_store_rejects_foreign_owner_without_reclaiming_it(self):
        import torch
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        graph = graph_cases._graph([{'type': 'end'}, {'type': 'wait'}], 'cpu')
        with graph_cases.DeviceGraphStorageTest().store() as pool:
            owners = torch.arange(2)
            put = pool.put_device(graph, owners, torch.ones(2, dtype=torch.uint8))
            states = DeviceStateBatch('physical', torch.arange(2), torch.ones(2, dtype=torch.int64),
                                     torch.tensor([9, 1]), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.int32))
            decision = SimpleNamespace(action_pool=pool, action_slots=put.slots.clone(),
                action_generations=put.generations.clone(), states=states, errors=states.errors)
            errors = release_actions_device(decision, states.active, owners)
            self.assertNotEqual(int(errors[0]), 0)
            self.assertEqual(int(errors[1]), 0)
            self.assertEqual(int(pool.live.sum()), 1)
            states.owners.copy_(owners)
            errors = release_actions_device(decision, states.active, owners)
            self.assertEqual(errors.tolist(), [0, 0])
            self.assertEqual(int(pool.live.sum()), 0)

    def test_search_drains_current_but_retains_root_action_until_consumed(self):
        import torch
        case = search_cases.DeviceSearchTest()
        search, fixture = case.engine(simulations=4)
        released = distinct_actions(fixture)
        roots = case.initialize(search, fixture)
        root_actions = search._root_decision.action_slots.clone()
        result = case.finish_cpu(search)
        self.assertTrue(bool(result.complete.all()))
        self.assertTrue(bool(fixture.live[root_actions].all()))
        self.assertEqual(int(fixture.live.sum()), 4)
        self.assertLessEqual(int(released.max()), 1)
        errors = release_actions_device(result.decision, roots.active, roots.owners,
                                        protected=(roots,))
        self.assertEqual(errors.tolist(), [0, 0])
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertEqual(result.decision.action_slots.tolist(), [-1, -1])
        again = release_actions_device(result.decision, roots.active, roots.owners,
                                       protected=(roots,))
        self.assertEqual(again.tolist(), [0, 0])
        self.assertLessEqual(int(released.max()), 1)

    def test_search_failure_reclaims_current_without_releasing_root(self):
        import torch
        case = search_cases.DeviceSearchTest()
        search, fixture = case.engine(simulations=4)
        released = distinct_actions(fixture)
        roots = case.initialize(search, fixture)
        root_actions = search._root_decision.action_slots.clone()
        original = fixture.metadata_device
        def fail(states):
            meta = original(states)
            return replace(meta, errors=((fixture.data[states.slots, 0] == 1)
                                         & states.active).to(torch.int32))
        fixture.metadata_device = fail
        result = case.finish_cpu(search)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool(result.errors.any()))
        self.assertTrue(bool(fixture.live[root_actions].all()))
        self.assertEqual(int(fixture.live.sum()), 4)
        self.assertLessEqual(int(released.max()), 1)

    def test_physical_success_and_failure_reclaim_distinct_root_actions(self):
        import torch
        case = game_cases.DeviceGameLoopTest()
        for failing in (False, True):
            loop, fixture, sink, roots, seeds, ids, train = case.make()
            released = distinct_actions(fixture)
            initial = roots.slots.clone()
            if failing:
                original = fixture.step_device
                def fail(states, actors, decision, indices, *, physical=False):
                    out = original(states, actors, decision, indices, physical=physical)
                    return replace(out, errors=out.errors | (out.active & physical).to(torch.int32))
                fixture.step_device = fail
            result = case.finish_cpu(loop, roots, seeds, ids, train)
            self.assertEqual(result.complete.tolist(), [not failing, not failing])
            self.assertEqual(int(fixture.live.sum()), 2)
            self.assertLessEqual(int(released.max()), 1)
            if failing:self.assertTrue(bool(fixture.live[initial].all()))

    def test_pure_policy_consumes_distinct_actions_without_search_counts(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_pure_policy import DevicePurePolicy
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        fixture = search_cases.TensorTreeFixture(2)
        released = distinct_actions(fixture)
        sink = game_cases.RecordingSink(2, fixture.device)
        policy = DevicePurePolicy(fixture, pure_cases.Bank(fixture.device), rows=2, max_actions=3,
                                  device='cpu', allow_cpu_reference=True, allow_synthetic=True)
        loop = DeviceGameLoop(fixture, policy, sink, game_cases.FixtureWorldSeeds(),
            max_decisions=4, device='cpu', allow_cpu_reference=True, allow_synthetic=True)
        roots = fixture.roots(); seeds = torch.arange(2); ids = torch.zeros((2, 2), dtype=torch.int64)
        result = game_cases.DeviceGameLoopTest().finish_cpu(loop, roots, seeds, ids,
                                                         torch.zeros((2, 2), dtype=torch.bool))
        self.assertTrue(bool(result.complete.all()))
        self.assertEqual(result.searched_count.tolist(), [0, 0])
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertLessEqual(int(released.max()), 1)

    def test_actual_cuda_distinct_action_lifetime_in_nested_physical_graph(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; distinct lease fixture, zero formal duel games')
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        case = game_cases.DeviceGameLoopTest()
        loop, fixture, sink, roots, seeds, ids, train = case.make(device='cuda', simulations=4)
        released = distinct_actions(fixture)
        production = DeviceGameLoop(fixture, loop.search, sink, game_cases.FixtureWorldSeeds(),
            max_decisions=4, device=str(fixture.device), allow_synthetic=False,
            inner_max_ticks=500, outer_max_ticks=8, seconds=60)
        self.addCleanup(production.close)
        production.capture(roots, seeds, ids, ids, train)
        production.replay();torch.cuda.synchronize()
        result = production.result_device()
        self.assertTrue(bool(result.complete.all()))
        self.assertFalse(bool(result.errors.any()))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertLessEqual(int(released.max()), 1)


if __name__ == '__main__':
    unittest.main()
