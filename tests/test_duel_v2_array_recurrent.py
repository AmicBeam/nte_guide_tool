"""Fixed-viewer/terminal semantics for the batched recurrent boundary.

Controlled evaluators are not a complete game or neural-value qualification.
"""
from dataclasses import replace
import unittest
import torch

from app.modules.card_game.rl.batched_search.array_recurrent import (
    ArrayState, ArrayEvaluation, ArrayRecurrentFn, ArrayRuleRegistry,
    INVALID_ROW, EVALUATION_ERROR, UNSUPPORTED_ACTION,
)


class ArrayRecurrentTest(unittest.TestCase):
    def inputs(self, device='cpu'):
        return ArrayState(torch.tensor([[5, 7], [5, 7], [5, 7], [5, 7]], dtype=torch.int32, device=device),
            torch.tensor([0, 1, 0, 1], device=device),
            torch.tensor([False, False, True, False], device=device),
            torch.tensor([-2, -2, 0, -2], device=device),
            torch.zeros(4, dtype=torch.int32, device=device))

    def evaluator(self, state, viewers, active):
        # This explicit fixture checks that value view is preserved on a
        # same-player move as well as a change of decision owner.
        wdl = torch.zeros((4, 3), dtype=torch.float32, device=state.values.device)
        wdl[:, 0], wdl[:, 1], wdl[:, 2] = .2, .3, .5
        return ArrayEvaluation(torch.zeros((4, 3), device=state.values.device),
            torch.ones((4, 3), dtype=torch.bool, device=state.values.device), wdl,
            torch.zeros(4, dtype=torch.int32, device=state.values.device))

    def transition(self, state, actions, active):
        values = state.values + active[:, None].to(torch.int32)
        done = state.finished | (active & (actions == 2))
        actors = torch.where(active & (actions == 1), 1-state.actors, state.actors)
        winners = torch.where(active & (actions == 2), state.actors, state.winners)
        return ArrayState(values, actors, done, winners, state.errors.clone())

    def run_case(self, device='cpu'):
        state = self.inputs(device)
        saved = state.values.clone()
        seen = []
        def evaluate(s, viewers, active):
            seen.append((viewers.clone(), active.clone()))
            return self.evaluator(s, viewers, active)
        fn = ArrayRecurrentFn(self.transition, evaluate, max_actions=3, device=device,
                              allow_cpu_reference=device == 'cpu')
        actions = torch.tensor([0, 1, 2, 2], device=device)
        viewers = torch.tensor([1, 1, 0, 0], device=device)
        out = fn(state, actions, viewers, torch.ones(4, dtype=torch.bool, device=device))
        self.assertTrue(torch.equal(state.values, saved))
        self.assertTrue(torch.equal(seen[0][0], viewers))
        self.assertEqual(out.errors.tolist(), [0, 0, 0, 0])
        self.assertEqual(out.state.actors.tolist(), [0, 0, 0, 1])
        self.assertEqual(out.reward.tolist(), [0., 0., 0., -1.])
        self.assertEqual(out.discount.tolist(), [1., 1., 0., 0.])
        torch.testing.assert_close(out.value, torch.tensor([.3, .3, 0., 0.], device=device))
        self.assertEqual(out.valid[2:].tolist(), [[True, False, False], [True, False, False]])
        self.assertTrue(torch.equal(out.state.values[2], saved[2]))
        return out

    def test_same_player_switch_and_absorbing_terminal(self):
        self.run_case()

    def test_errors_roll_back_and_inactive_preserve(self):
        state = self.inputs()
        def transition(s, actions, active):
            s.values.add_(100)  # disposable copy must protect caller state
            return replace(s, errors=torch.tensor([1, 0, 0, 0], dtype=torch.int32))
        def evaluator(s, viewers, active):
            p = self.evaluator(s, viewers, active)
            p.wdl[1, 0] = float('nan')
            return p
        fn = ArrayRecurrentFn(transition, evaluator, max_actions=3, device='cpu', allow_cpu_reference=True)
        out = fn(state, torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64),
                 torch.tensor([True, True, True, False]))
        self.assertEqual(out.errors.tolist(), [1, EVALUATION_ERROR, 0, 0])
        torch.testing.assert_close(out.state.values, state.values)
        self.assertTrue((out.reward == 0).all())
        self.assertTrue((out.discount == 0).all())
        self.assertTrue((state.values[:, 0] == 5).all())

    def test_invalid_metadata_and_bad_wdl_rejected(self):
        state = replace(self.inputs(), actors=torch.tensor([3, 1, 0, 1]))
        fn = ArrayRecurrentFn(self.transition, self.evaluator, max_actions=3,
                              device='cpu', allow_cpu_reference=True)
        out = fn(state, torch.zeros(4, dtype=torch.int64), torch.tensor([0, 2, 0, 0]),
                 torch.tensor([True, True, True, False]))
        self.assertEqual(out.errors.tolist(), [INVALID_ROW, INVALID_ROW, 0, 0])

    def test_shared_ir_registry_rejects_unsupported_action(self):
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.schema import FIELDS, FIELD_INDEX as ix
        registry = ArrayRuleRegistry({0: COMBAT_SLICE})
        values = torch.zeros((3, len(FIELDS)), dtype=torch.int32)
        values[:, ix['a_hp']], values[:, ix['a_front']], values[:, ix['a_atk']] = 4, 1, 3
        values[:, ix['pa_hp']], values[:, ix['pb_hp']] = 30, 30
        result = registry.step(values, torch.tensor([0, 0, 0]), torch.tensor([0, 7, 7]),
                               torch.tensor([True, True, False]))
        self.assertEqual(result.errors.tolist(), [0, UNSUPPORTED_ACTION, 0])
        self.assertEqual(result.states[0, ix['pb_hp']].item(), 27)
        self.assertTrue(torch.equal(result.states[1:], values[1:]))

    def test_shared_ir_transition_through_recurrent_to_terminal_reward(self):
        from app.modules.card_game.rl.batched_search.array_recurrent import SharedIrTransition
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.schema import FIELDS, FIELD_INDEX as ix
        registry = ArrayRuleRegistry({0: COMBAT_SLICE})
        values = torch.zeros((4, len(FIELDS)), dtype=torch.int32)
        values[:, ix['a_hp']], values[:, ix['a_front']], values[:, ix['a_atk']] = 4, 1, 3
        values[:, ix['pa_hp']], values[:, ix['pb_hp']] = 30, 2
        state = ArrayState(values, torch.zeros(4, dtype=torch.int64),
            torch.zeros(4, dtype=torch.bool), torch.full((4,), -2), torch.zeros(4, dtype=torch.int32))
        fn = ArrayRecurrentFn(SharedIrTransition(registry), self.evaluator, max_actions=3,
                             device='cpu', allow_cpu_reference=True)
        result = fn(state, torch.zeros(4, dtype=torch.int64), torch.tensor([0, 1, 0, 1]),
                    torch.ones(4, dtype=torch.bool))
        self.assertEqual(result.errors.tolist(), [0]*4)
        self.assertEqual(result.state.winners.tolist(), [0]*4)
        self.assertEqual(result.reward.tolist(), [1., -1., 1., -1.])
        self.assertEqual(result.discount.tolist(), [0.]*4)
        self.assertEqual(result.value.tolist(), [0.]*4)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_actual_cuda_and_captured_recurrent(self):
        self.run_case('cuda')
        fn = ArrayRecurrentFn(self.transition, self.evaluator, max_actions=3, device='cuda')
        state = self.inputs('cuda')
        a = torch.tensor([0, 1, 2, 2], device='cuda')
        v = torch.tensor([1, 1, 0, 0], device='cuda')
        active = torch.ones(4, dtype=torch.bool, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(state, a, v, active)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(state, a, v, active)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(out.errors.cpu().tolist(), [0, 0, 0, 0])
        self.assertEqual(out.reward.cpu().tolist(), [0., 0., 0., -1.])


if __name__ == '__main__':
    unittest.main()
