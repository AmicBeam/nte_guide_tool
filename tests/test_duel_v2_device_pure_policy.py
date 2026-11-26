"""Synthetic pure-policy numerical/control tests, not real game approval."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from app.modules.card_game.rl.batched_search.device_pure_policy import DevicePurePolicy
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch


class Bank:
    def __init__(self, device):
        self.device = device

    def predict_device(self, x, c, valid, ids, context):
        import torch
        logits = c[..., 0] * torch.where(ids == 0, 1., -1.)[:, None]
        return SimpleNamespace(logits=logits.contiguous(), invalid_rows=(ids < 0) | (ids > 1))


class PurePolicyTest(unittest.TestCase):
    def make(self, device='cpu'):
        import torch
        d = torch.device(device)
        if d.type == 'cuda':d = torch.device('cuda', torch.cuda.current_device())
        actors = torch.tensor([0, 1, 0], device=d, dtype=torch.int64)
        roots = DeviceStateBatch(object(), torch.arange(3, device=d), torch.ones(3, device=d, dtype=torch.int64),
            torch.arange(3, device=d), torch.tensor([True, True, False], device=d),
            torch.zeros(3, device=d, dtype=torch.int32))
        logits = torch.tensor([[2., 1., 999., 888.]] * 3, device=d)
        counts = torch.tensor([2, 2, 2], device=d)
        valid = torch.arange(4, device=d)[None, :] < counts[:, None]
        errors = torch.zeros(3, device=d, dtype=torch.int32)
        metadata = SimpleNamespace(actors=actors, finished=torch.zeros(3, device=d, dtype=torch.bool), errors=errors)
        decision = SimpleNamespace(states=roots, counts=counts, valid=valid,
            observations=torch.zeros((3, 2), device=d), candidates=logits[..., None].contiguous(), errors=errors)
        rules = SimpleNamespace(metadata_device=lambda states: metadata,
            decision_device=lambda states, viewers: decision)
        policy = DevicePurePolicy(rules, Bank(d), rows=3, max_actions=4, device=str(d),
            allow_cpu_reference=device == 'cpu', allow_synthetic=True)
        ids = torch.tensor([[0, 1]] * 3, device=d)
        return policy, roots, actors, ids, metadata, decision

    def run_policy(self, policy, roots, actors, ids):
        policy.initialize(roots, actors, None, ids, ids)
        return policy.result_device()

    def test_legal_argmax_routes_actual_actor_and_does_not_claim_search(self):
        policy, roots, actors, ids, meta, decision = self.make()
        result = self.run_policy(policy, roots, actors, ids)
        self.assertEqual(result.actions.tolist(), [0, 1, -1])
        self.assertEqual(result.complete.tolist(), [True, True, False])
        self.assertEqual(result.requested.tolist(), [0, 0, 0])
        self.assertEqual(result.completed.tolist(), [0, 0, 0])
        self.assertEqual(result.visits.sum().item(), 0)
        self.assertEqual(result.visit_policy.sum().item(), 0)
        self.assertFalse(policy.counts_as_search)
        self.assertFalse(policy.supports_full_duel_rules)

    def test_cpu_requires_explicit_oracle_and_bad_model_or_actor_fails_closed(self):
        policy, roots, actors, ids, meta, decision = self.make()
        with self.assertRaises(ValueError):
            DevicePurePolicy(policy.rules, policy.inference, rows=3, max_actions=4, device='cpu')
        ids[0, 0] = 3
        actors[1] = 8
        result = self.run_policy(policy, roots, actors, ids)
        self.assertEqual(result.actions.tolist(), [-1, -1, -1])
        self.assertTrue(bool((result.errors[:2] != 0).all()))

    def test_nonfinite_valid_logits_and_nonprefix_legal_masks_are_rejected(self):
        policy, roots, actors, ids, meta, decision = self.make()
        decision.candidates[0, 0, 0] = float('nan')
        decision.valid[1] = policy.t.tensor([True, False, True, False])
        result = self.run_policy(policy, roots, actors, ids)
        self.assertFalse(bool(result.complete.any()))
        self.assertTrue(bool((result.errors[:2] != 0).all()))

    def test_no_host_reads_in_policy_selection(self):
        import torch
        policy, roots, actors, ids, meta, decision = self.make()
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU')),\
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar')),\
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('truth')):
            self.run_policy(policy, roots, actors, ids)

    def test_physical_loop_argmax_is_zero_searched_decisions(self):
        import torch
        from tests.test_duel_v2_device_games import RecordingSink, FixtureWorldSeeds, DeviceGameLoopTest
        from tests.test_duel_v2_device_search import TensorTreeFixture
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        rules = TensorTreeFixture(2, 'cpu', actions=3)
        sink = RecordingSink(2, rules.device)
        roots = rules.roots(); seeds = torch.arange(2); ids = torch.zeros((2, 2), dtype=torch.int64)
        train = torch.zeros((2, 2), dtype=torch.bool)
        policy = DevicePurePolicy(rules, Bank(rules.device), rows=2, max_actions=3,
            device='cpu', allow_cpu_reference=True, allow_synthetic=True)
        loop = DeviceGameLoop(rules, policy, sink, FixtureWorldSeeds(), max_decisions=8,
            device='cpu', allow_cpu_reference=True, allow_synthetic=True)
        result = DeviceGameLoopTest().finish_cpu(loop, roots, seeds, ids, train)
        self.assertTrue(bool(result.complete.all()))
        self.assertEqual(result.steps.tolist(), [3, 3])
        self.assertEqual(result.searched_count.tolist(), [0, 0])

    def test_actual_cuda_nested_physical_loop_has_zero_search_statistics(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required; synthetic physical rules')
        from tests.test_duel_v2_device_games import RecordingSink, FixtureWorldSeeds
        from tests.test_duel_v2_device_search import TensorTreeFixture
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        rules = TensorTreeFixture(2, 'cuda', actions=3)
        roots = rules.roots(); seeds = torch.arange(2, device=rules.device)
        ids = torch.zeros((2, 2), dtype=torch.int64, device=rules.device)
        train = torch.zeros((2, 2), dtype=torch.bool, device=rules.device)
        policy = DevicePurePolicy(rules, Bank(rules.device), rows=2, max_actions=3,
                                  device=str(rules.device), allow_synthetic=True)
        loop = DeviceGameLoop(rules, policy, RecordingSink(2, rules.device), FixtureWorldSeeds(),
            max_decisions=8, device=str(rules.device), seconds=30)
        self.addCleanup(loop.close)
        loop.capture(roots, seeds, ids, ids, train)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU')),\
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar')),\
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('truth')):
            loop.replay()
        torch.cuda.synchronize()
        result = loop.result_device()
        self.assertEqual(result.errors.cpu().tolist(), [0, 0])
        self.assertEqual(result.steps.cpu().tolist(), [3, 3])
        self.assertEqual(result.searched_count.cpu().tolist(), [0, 0])

    def test_actual_cuda_graph_replays_new_actor_model_routes(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required; numerical fixture only')
        policy, roots, actors, ids, meta, decision = self.make('cuda')
        self.run_policy(policy, roots, actors, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result = self.run_policy(policy, roots, actors, ids)
        ids.copy_(ids.flip(1))
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU')),\
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar')),\
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('truth')):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(result.actions.cpu().tolist(), [1, 0, -1])
        self.assertEqual(result.completed.cpu().tolist(), [0, 0, 0])


if __name__ == '__main__':unittest.main()
