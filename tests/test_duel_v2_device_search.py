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
    def test_raw_opponent_actions_preserve_full_search_and_own_information_keys(self):
        import torch
        from dataclasses import replace
        torch.set_num_threads(1)
        reference,ref_fixture=self.engine()
        self.initialize(reference,ref_fixture)
        expected=self.finish_cpu(reference)
        engine,fixture=self.engine()
        choose=fixture.rule_decision_device
        def raw_teacher(states,actors):
            decision,index,errors=choose(states,actors)
            return replace(decision,observations=decision.observations[:,:0],
                candidates=decision.candidates[:,:,:0]),index,errors
        fixture.rule_decision_device=raw_teacher
        roots=self.initialize(engine,fixture)
        result=self.finish_cpu(engine)
        self.assertEqual(result.errors.tolist(),[0,0])
        self.assertEqual(result.completed.tolist(),[32,32])
        for field in ('policy','visits','actions','information_keys'):
            torch.testing.assert_close(getattr(result,field),getattr(expected,field))
        self.assertGreater(int(engine.phase_work[OPPONENT].sum()),0)
        self.assertEqual(int(fixture.live.sum()),2)
        torch.testing.assert_close(fixture.data[roots.slots],torch.tensor([[0,-1],[0,-1]]))

    def test_value_slices_preserve_models_viewers_terminal_and_invalid_tail(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_search import INFERENCE_ERROR
        engine, fixture = self.engine(rows=65)
        class RowInference:
            candidate_dim = 2
            def __init__(self): self.calls = []
            def predict_device(self, x, c, valid, ids, actor, *, value_only):
                self.calls.append((len(x), c.shape[1], ids.clone(), actor.clone()))
                score = ids.float() * .07 + x[:, 0] * .03 + actor[:, 0] * .11
                wdl = torch.stack((-score, torch.zeros_like(score), score), -1).softmax(-1)
                return SimpleNamespace(wdl=wdl, invalid_rows=(ids == 3))
        inference = RowInference();engine.auxiliary = inference
        roots = fixture.roots()
        fixture.data[roots.slots[0]] = torch.tensor([3, 1])
        fixture.data[roots.slots[-1]] = torch.tensor([3, 0])
        ids = torch.arange(65, dtype=torch.int64) % 7
        mask = torch.ones(65, dtype=torch.bool);mask[10] = False
        observed = fixture.value_observation_device(roots, fixture.viewer)
        whole = RowInference().predict_device(observed.observations, torch.empty(65, 0, 2),
            torch.empty(65, 0, dtype=torch.bool), ids, observed.viewer_is_actor,
            value_only=torch.ones(65, dtype=torch.bool))
        expected = torch.where(observed.terminal, observed.terminal_value,
            (whole.wdl[:, 2] - whole.wdl[:, 0]).double())
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar read')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            value = engine._fixed_value(roots, fixture.viewer, ids, mask)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual([(n, m) for n, m, _, _ in inference.calls], [(32, 0), (32, 0), (1, 0)])
        self.assertTrue(torch.equal(torch.cat([c[2] for c in inference.calls]), ids))
        self.assertTrue(torch.equal(torch.cat([c[3] for c in inference.calls]), observed.viewer_is_actor))
        expected_errors = ((ids == 3) & mask & ~observed.terminal).int() * INFERENCE_ERROR
        self.assertTrue(torch.equal(engine.errors, expected_errors))

    def test_compact_candidates_match_dense_full_budget_and_snapshot_isolation(self):
        import torch
        from dataclasses import replace
        from app.modules.card_game.rl.batched_search.compact_device_candidates import CompactDeviceCandidates
        from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import (
            RawCandidateTokens, ORDER_SLOTS, MAX_MULLIGAN_SELECTIONS, encode_tokens_device,
        )
        results = []
        for compact in (False, True):
            rules = TensorTreeFixture(3, actions=20)
            original = rules.decision_device
            def decision(states, observers, original=original, compact=compact):
                base = original(states, observers)
                integer = torch.full((3, 20, 7), -1, dtype=torch.int64)
                integer[:, :, 0] = 0
                tokens = RawCandidateTokens(integer, torch.zeros(3, 20, 8),
                    torch.zeros(3, 20, 2), torch.zeros(3),
                    torch.full((3, 20, MAX_MULLIGAN_SELECTIONS), -1, dtype=torch.int64),
                    base.counts, base.valid,
                    torch.full((3, 20, 2, ORDER_SLOTS), -1, dtype=torch.int64),
                    torch.full((3, 20, MAX_MULLIGAN_SELECTIONS), -1, dtype=torch.int64),
                    torch.arange(20).float().expand(3, -1).clone())
                if compact:
                    candidates = CompactDeviceCandidates(tokens, device='cpu', scratch_rows=1,
                                                         allow_cpu_reference=True)
                else:
                    candidates = encode_tokens_device(tokens, device='cpu',
                        allow_cpu_reference=True)[0].candidates
                return replace(base, candidates=candidates)
            rules.decision_device = decision
            inference = FixtureInference();inference.device = torch.device('cpu')
            keys = InjectiveFixtureKeys();keys.device = torch.device('cpu')
            statistics = DeviceStatisticsPool(3 * 5, 20, device='cpu', max_batch=3,
                max_path_len=4, allow_cpu_reference=True)
            search = DeviceGumbelSearch(rules, inference, inference, statistics, keys,
                rows=3, max_actions=20, simulations=4, gumbel_candidates=4, max_depth=4,
                device='cpu', allow_cpu_reference=True, allow_synthetic=True)
            roots = rules.roots();ids = torch.zeros(3, 2, dtype=torch.int64)
            search.initialize(roots, rules.viewer, torch.zeros(3, 5, dtype=torch.int64), ids, ids)
            root_snapshot = search._root_decision.candidates
            for _ in range(150):
                if not bool(search.unfinished_device.any()):break
                search.tick()
            result = search.result_device()
            self.assertTrue(bool(result.complete.all()))
            self.assertFalse(bool(result.errors.any()))
            results.append((result.actions.clone(), result.policy.clone(), result.visits.clone()))
            if compact:
                before = root_snapshot.tokens.extra_ap.clone()
                search._decision.candidates.tokens.extra_ap.fill_(900)
                self.assertTrue(torch.equal(root_snapshot.tokens.extra_ap, before))
                self.assertEqual(root_snapshot.shape, (3, 20, 1576))
        for dense, compact in zip(results[0], results[1]):
            self.assertTrue(torch.equal(dense, compact))

    def engine(self, rows=2, device='cpu', simulations=32, actions=3,group_ready_leaves=False,reuse_interior_observation=False):
        fixture = TensorTreeFixture(rows, device, actions=actions)
        stats = DeviceStatisticsPool(rows*(simulations+1), actions, device=str(fixture.device),
            max_batch=rows, max_path_len=4, allow_cpu_reference=device == 'cpu')
        engine = DeviceGumbelSearch(fixture, FixtureInference(), FixtureInference(), stats,
            InjectiveFixtureKeys(), rows=rows, max_actions=actions, simulations=simulations,
            max_depth=4, terminal_horizon=3, device=str(fixture.device),
            allow_cpu_reference=device == 'cpu', allow_synthetic=True,
            group_ready_leaves=group_ready_leaves,reuse_interior_observation=reuse_interior_observation)
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

    def test_ready_leaf_groups_preserve_ragged_root_search_and_random_order(self):
        import torch
        from dataclasses import replace
        torch.set_num_threads(1)
        def run(grouped):
            engine,fixture=self.engine(rows=9,group_ready_leaves=grouped)
            limits=torch.tensor([2,3,6,2,6,3,6,2,3])
            metadata=fixture.metadata_device;teacher=fixture.teacher_continue_device;calls=[]
            def meta(states):
                original=metadata(states);done=fixture.data[states.slots,0]>=limits
                chosen=fixture.data[states.slots,1]
                winners=torch.where(chosen==1,fixture.viewer,
                    torch.where(chosen==0,1-fixture.viewer,-1))
                depth=fixture.data[states.slots,0]
                actors=torch.where(torch.arange(9).remainder(3)==0,fixture.viewer,
                    fixture.viewer^depth.remainder(2))
                return replace(original,finished=done,
                    actors=torch.where(done,-1,actors),
                    winners=torch.where(done,winners,torch.full_like(original.winners,-2)))
            fixture.metadata_device=meta
            def continue_group(states,viewers,*,horizon=3):
                active=int(states.active.sum())
                if active:calls.append(active)
                data=fixture.data[states.slots]
                return fixture._new(states,torch.stack((torch.minimum(data[:,0]+horizon,limits),data[:,1]),-1))
            fixture.teacher_continue_device=continue_group
            self.initialize(engine,fixture,active=[True]*8+[False])
            return self.finish_cpu(engine),fixture,calls
        reference,original,old_calls=run(False)
        result,grouped,new_calls=run(True)
        self.assertEqual(result.errors.tolist(),[0]*9)
        self.assertEqual(result.completed.tolist(),[32]*8+[0])
        for field in ('policy','visits','actions','information_keys','complete'):
            torch.testing.assert_close(getattr(result,field),getattr(reference,field))
        torch.testing.assert_close(grouped.sample_seeds,original.sample_seeds)
        self.assertEqual(int(grouped.live.sum()),int(original.live.sum()))
        self.assertLess(len(new_calls),len(old_calls))
        self.assertGreater(sum(new_calls)/len(new_calls),sum(old_calls)/len(old_calls))

    def test_interior_observation_reuse_preserves_full_search_and_refuses_stale_binding(self):
        import torch
        from dataclasses import replace
        torch.set_num_threads(1)
        baseline,reference=self.engine();self.initialize(baseline,reference);expected=self.finish_cpu(baseline)
        engine,fixture=self.engine(group_ready_leaves=True,reuse_interior_observation=True)
        original=fixture.value_observation_device;calls=[]
        def count(states,viewers):
            if bool(states.active.any()):calls.append(int(states.active.sum()))
            return original(states,viewers)
        fixture.value_observation_device=count
        self.initialize(engine,fixture);actual=self.finish_cpu(engine)
        for field in ('errors','policy','visits','actions','information_keys','complete','completed'):
            torch.testing.assert_close(getattr(actual,field),getattr(expected,field))
        # Both root and H3 still obtain formal value observations; only the
        # matching nonterminal own interior observation may be borrowed.
        self.assertTrue(calls)
        bad,fixture=self.engine(reuse_interior_observation=True)
        self.initialize(bad,fixture)
        from app.modules.card_game.rl.batched_search.device_search import EVALUATE_NODE
        for _ in range(100):
            if bool((bad.phase==EVALUATE_NODE).any()):break
            bad.tick()
        self.assertTrue(bool((bad.phase==EVALUATE_NODE).any()))
        bad._decision=replace(bad._decision,
            states=replace(bad._decision.states,owners=bad._decision.states.owners+1))
        rejected=self.finish_cpu(bad)
        self.assertTrue(bool((rejected.errors&METADATA_ERROR).any()))
        self.assertFalse(bool(rejected.complete.all()))

    def test_full_budget_matches_numpy_root_semantics_without_value_sign_flip(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = self.initialize(engine, fixture)
        result = self.finish_cpu(engine)
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(result.complete.tolist(), [True, True])
        self.assertEqual(result.completed.tolist(), [32, 32])
        expected_keys = engine.information_keys.hash(result.decision.observations,
            result.decision.candidates, result.decision.counts, roots.active).keys
        self.assertTrue(torch.equal(result.information_keys, expected_keys))
        self.assertTrue(result.information_keys.is_contiguous())
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

    def test_failure_cleanup_releases_hypothetical_lease_but_keeps_physical_root(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = self.initialize(engine, fixture)
        engine.tick()  # SAMPLE issues one hypothetical lease per root.
        self.assertEqual(int(fixture.live.sum()), 4)
        engine.errors[0] = 2  # A failed inference must still drain its lease.
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete[0]))
        self.assertTrue(bool(result.complete[1]))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertTrue(bool(fixture.live[roots.slots].all()))

    def test_terminal_winner_cannot_silently_turn_unfinished_code_into_draw(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        self.initialize(engine, fixture)
        original = fixture.metadata_device
        def malformed(states):
            meta = original(states)
            winners = torch.where(meta.finished, torch.full_like(meta.winners, -2), meta.winners)
            return DeviceMeta(meta.actors, meta.phases, winners, meta.finished, meta.versions, meta.errors)
        fixture.metadata_device = malformed
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))

    def test_step_reusing_input_and_clean_replacing_it_releases_old_lease_once(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_search import STEP
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        self.initialize(engine, fixture)
        engine.tick()  # Allocate sampled worlds, retaining physical roots.
        original_release = fixture.release_device
        release_errors = torch.zeros(2, dtype=torch.bool)
        def checked_release(states):
            release_errors.logical_or_(states.active & ~fixture.live[states.slots])
            original_release(states)
        fixture.release_device = checked_release
        fixture.step_device = lambda states, actors, decision, indices, physical=False: states
        fixture.clean_device = lambda states: fixture._new(states, fixture.data[states.slots])
        engine.phase.fill_(STEP)
        engine.tick()
        self.assertFalse(bool(release_errors.any()))
        self.assertEqual(int(fixture.live.sum()), 4)

    def test_valid_but_wrong_viewer_terminal_value_is_rejected(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        self.initialize(engine, fixture)
        original = fixture.value_observation_device
        def wrong_frame(states, viewers):
            observed = original(states, viewers)
            value = torch.where(observed.terminal, -observed.terminal_value, observed.terminal_value)
            return DeviceValueObservation(observed.observations, observed.viewer_is_actor,
                                          observed.terminal, value, observed.errors)
        fixture.value_observation_device = wrong_frame
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))

    def test_backup_failure_does_not_disable_checked_world_release(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_search import BACKUP
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = self.initialize(engine, fixture)
        original_release = fixture.release_device
        def skip_execution_error_rows(states):
            active = states.active & (states.errors == 0)
            original_release(DeviceStateBatch(states.pool, states.slots, states.generations,
                                             states.owners, active, states.errors))
        fixture.release_device = skip_execution_error_rows
        for _ in range(30):
            if bool((engine.phase == BACKUP).all()):
                break
            engine.tick()
        self.assertTrue(bool((engine.phase == BACKUP).all()))
        original_backup = engine.statistics.backup
        fired = [False]
        def reject_first(*args, **kwargs):
            result = original_backup(*args, **kwargs)
            if not fired[0]:
                result.error[0] = True
                fired[0] = True
            return result
        engine.statistics.backup = reject_first
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete[0]))
        self.assertTrue(bool(result.complete[1]))
        self.assertEqual(int(fixture.live.sum()), 2)
        self.assertTrue(bool(fixture.live[roots.slots].all()))

    def test_teacher_cannot_replace_hypothetical_lease_with_physical_root(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = self.initialize(engine, fixture)
        def alias_root(states, viewers, horizon=3):
            return DeviceStateBatch(fixture, roots.slots, roots.generations,
                                    roots.owners, states.active, states.errors)
        fixture.teacher_continue_device = alias_root
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))
        self.assertTrue(bool(fixture.live[roots.slots].all()))
        self.assertEqual(int(fixture.live.sum()), 2)

    def test_failed_root_node_claim_cannot_release_an_existing_foreign_node(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        slots = torch.tensor([0])
        owners = torch.tensor([99])
        prior = torch.tensor([[.4, .3, .3]], dtype=torch.float64)
        first = engine.statistics.write_nodes(slots, owners, prior,
            torch.tensor([0.], dtype=torch.float64), torch.ones((1, 3), dtype=torch.bool))
        self.assertFalse(bool(first.error.any()))
        self.initialize(engine, fixture)
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete[0]))
        self.assertTrue(bool(result.complete[1]))
        self.assertEqual(int(engine.node_count[0]), 0)
        saved = engine.statistics.snapshot(slots, first.generations)
        self.assertFalse(bool(saved.error.any()))
        self.assertEqual(int(saved.owners[0]), 99)

    def test_cross_row_sample_cannot_free_an_inactive_physical_root(self):
        import torch
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        roots = fixture.roots()
        roots.active[1] = False
        def steal_inactive(states, viewers, seeds):
            slots = roots.slots.flip(0)
            generations = roots.generations.flip(0)
            return DeviceStateBatch(fixture, slots, generations, states.owners,
                                    states.active, states.errors)
        fixture.sample_world_device = steal_inactive
        seeds = torch.zeros((2, 5), dtype=torch.int64)
        seeds[:, 4] = 1
        ids = torch.zeros((2, 2), dtype=torch.int64)
        engine.initialize(roots, fixture.viewer, seeds, ids, ids)
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete.any()))
        self.assertEqual(int(result.errors[0]) & METADATA_ERROR, METADATA_ERROR)
        self.assertTrue(bool(fixture.live[roots.slots].all()))

    def test_inspect_terminal_uses_same_fixed_viewer_consistency_check_as_leaf(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_search import INSPECT
        torch.set_num_threads(1)
        engine, fixture = self.engine()
        self.initialize(engine, fixture)
        engine.tick()
        fixture.data[engine._world_slots, 0] = 3
        fixture.data[engine._world_slots, 1] = 1  # viewer wins
        original = fixture.value_observation_device
        def disagree(states, viewers):
            observation = original(states, viewers)
            return DeviceValueObservation(observation.observations, observation.viewer_is_actor,
                observation.terminal, torch.zeros_like(observation.terminal_value), observation.errors)
        fixture.value_observation_device = disagree
        engine.phase.fill_(INSPECT)
        result = self.finish_cpu(engine)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors & METADATA_ERROR).all()))

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

    def test_actual_cuda_ready_leaf_groups_capture_and_preserve_full_budget(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required; controlled fixture only')
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        reference,oracle=self.engine(rows=8)
        self.initialize(reference,oracle);expected=self.finish_cpu(reference)
        engine,fixture=self.engine(rows=8,device='cuda',group_ready_leaves=True,reuse_interior_observation=True)
        roots=fixture.roots();seeds=torch.zeros((8,5),dtype=torch.int64,device=fixture.device);seeds[:,4]=1
        ids=torch.zeros((8,2),dtype=torch.int64,device=fixture.device)
        runner=CudaConditionalWhile(device=str(fixture.device),max_ticks=3000,seconds=120)
        self.addCleanup(runner.close)
        engine.statistics.bind_capture_streams(runner.capture_stream,runner.body_stream)
        with engine.statistics.capture_scope():
            runner.capture(engine.unfinished_device,engine.tick,
                initialize=lambda:engine.initialize(roots,fixture.viewer,seeds,ids,ids))
        runner.replay();torch.cuda.synchronize();actual=engine.result_device()
        self.assertEqual(actual.errors.cpu().tolist(),[0]*8)
        self.assertEqual(actual.completed.cpu().tolist(),[32]*8)
        for field in ('policy','visits','actions','information_keys','complete'):
            torch.testing.assert_close(getattr(actual,field).cpu(),getattr(expected,field))

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
