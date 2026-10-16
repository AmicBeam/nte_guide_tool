"""Array IR lowering: batched integer kernel vs CPU/ordinary/common combat.

Coverage is the combat slice and common ordinary-sortie program. These tests
are not a seven-rule or whole-game claim.
"""
from __future__ import annotations

import copy
import importlib.util
import random
import unittest

TORCH = importlib.util.find_spec('torch')


def _cpu_expected(program, x, sides, active):
    import torch
    from app.modules.card_game.rl.rule_ir.lower_cpu import lower_cpu
    cpu = lower_cpu(program)
    out = x.clone()
    for side in (0, 1):
        take = active & (sides == side)
        if not bool(take.any()):
            continue
        out = torch.where(take.unsqueeze(1), cpu(x, side), out)
    return out


def _rand_mask(n, seed, tail=3):
    import torch
    rng = random.Random(seed)
    active = [rng.random() > 0.2 for _ in range(max(0, n - tail))] + [False] * min(tail, n)
    sides = [rng.choice((0, 1)) for _ in range(n)]
    return (torch.tensor(sides, dtype=torch.int64), torch.tensor(active, dtype=torch.bool))


@unittest.skipUnless(TORCH, 'PyTorch is required for array rule IR')
class ArrayRuleLoweringTest(unittest.TestCase):
    def test_combat_mixed_batch_matches_lower_cpu_and_engine(self):
        import torch
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.lower_array import lower_array
        from tests.test_duel_v2_rule_ir import _make_cases, _pack, _reference
        step = lower_array(COMBAT_SLICE)
        states, opts = _make_cases(64)
        x = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        sides, active = _rand_mask(len(states), 902, tail=6)
        saved = x.clone()
        result = step(x, sides, active)
        self.assertTrue(torch.equal(x, saved))
        self.assertTrue(torch.equal(result.states, _cpu_expected(COMBAT_SLICE, x, sides, active)))
        self.assertTrue((result.errors == 0).all())
        engine = []
        for i, (state, opt) in enumerate(zip(states, opts)):
            s = copy.deepcopy(state)
            if bool(active[i]):
                _reference(s, opt, int(sides[i]))
            engine.append(_pack(s, opt))
        self.assertTrue(torch.equal(result.states, torch.tensor(engine, dtype=torch.int32)))
        self.assertTrue(torch.equal(result.states[~active], x[~active]))

    def test_ordinary_common_mixed_tail_inactive_matches_cpu(self):
        import torch
        from app.modules.card_game.rl.rule_ir.lower_array import lower_array
        from app.modules.card_game.rl.rule_ir.ordinary_sortie import ordinary_program
        from app.modules.card_game.rl.rule_ir.schema import FIELD_INDEX
        program = ordinary_program(4)
        step = lower_array(program)
        rng = random.Random(441)
        rows = []
        for _ in range(48):
            row = [rng.randint(0, 8) for _ in program.fields]
            row[FIELD_INDEX['a_hp']] = rng.randint(0, 6)
            row[FIELD_INDEX['b_hp']] = rng.randint(0, 6)
            row[FIELD_INDEX['a_atk']] = rng.randint(0, 7)
            row[FIELD_INDEX['b_atk']] = rng.randint(0, 7)
            row[FIELD_INDEX['a_front']] = rng.choice((0, 1))
            row[FIELD_INDEX['b_front']] = rng.choice((0, 1))
            row[FIELD_INDEX['done']] = 1 if rng.random() < 0.12 else 0
            row[program.fields.index('actor')] = rng.randint(0, 3)
            row[program.fields.index('old_front')] = rng.choice((-1, 0, 1, 2, 3))
            row[program.fields.index('last_front')] = rng.choice((-1, 0, 1, 2, 3))
            rows.append(row)
        x = torch.tensor(rows, dtype=torch.int32)
        sides, active = _rand_mask(len(rows), 442, tail=5)
        active[3] = False
        saved = x.clone()
        result = step(x, sides, active)
        self.assertTrue(torch.equal(x, saved))
        self.assertTrue((result.errors == 0).all())
        self.assertTrue(torch.equal(result.states, _cpu_expected(program, x, sides, active)))
        self.assertTrue(torch.equal(result.states[~active], x[~active]))
        for side in (0, 1):
            take = torch.ones(len(rows), dtype=torch.bool)
            one = torch.full((len(rows),), side, dtype=torch.int64)
            got = step(x, one, take)
            self.assertTrue(torch.equal(got.states, _cpu_expected(program, x, one, take)))

    def test_alias_store_uses_per_row_actor_and_keeps_snapshot(self):
        import torch
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.lower_array import lower_array
        from app.modules.card_game.rl.rule_ir.schema import FIELD_INDEX as ix
        step = lower_array(COMBAT_SLICE)
        x = torch.zeros((4, 19), dtype=torch.int32)
        x[:, ix['a_hp']], x[:, ix['b_hp']] = 8, 7
        x[:, ix['a_atk']], x[:, ix['b_atk']] = 2, 4
        x[:, ix['pa_hp']], x[:, ix['pb_hp']] = 30, 20
        x[0, ix['a_front']], x[1, ix['b_front']] = 1, 1
        sides = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        active = torch.tensor([True, True, False, False])
        result = step(x, sides, active)
        self.assertEqual(result.errors.tolist(), [0, 0, 0, 0])
        self.assertEqual(int(result.states[0, ix['pb_hp']]), 18)
        self.assertEqual(int(result.states[0, ix['b_hp']]), 7)
        self.assertEqual(int(result.states[0, ix['a_hp']]), 8)
        self.assertEqual(int(result.states[1, ix['pa_hp']]), 26)
        self.assertEqual(int(result.states[1, ix['pb_hp']]), 20)
        self.assertEqual(int(result.states[1, ix['a_hp']]), 8)
        self.assertTrue(torch.equal(result.states[2:], x[2:]))

    def test_invalid_actors_and_int32_bounds_fail_closed(self):
        import torch
        from app.modules.card_game.rl.rule_ir.ast import Bind, Bin, Const, Name, Program, Store
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.lower_array import (
            ERR_ACTOR, ERR_OVERFLOW, I32_MAX, lower_array,
        )
        step = lower_array(COMBAT_SLICE)
        x = torch.zeros((5, 19), dtype=torch.int32)
        x[:, 0], x[:, 10], x[:, 6], x[:, 8] = 4, 1, 30, 30
        x[:, 2] = 3
        sides = torch.tensor([0, 1, 2, -1, 0], dtype=torch.int64)
        active = torch.tensor([True, True, True, False, False])
        saved = x.clone()
        result = step(x, sides, active)
        self.assertTrue(torch.equal(x, saved))
        self.assertEqual(result.errors.tolist(), [0, 0, ERR_ACTOR, 0, 0])
        self.assertTrue(torch.equal(result.states[2:], x[2:]))
        overflow = Program(
            name='overflow_add', version='t', skip_when=Const(0),
            binds=(Bind('total', Bin('add', Name('a_hp'), Name('b_hp'))),),
            stores=(Store('done', Name('total')),),
        )
        ov = lower_array(overflow)
        y = torch.zeros((3, 19), dtype=torch.int32)
        y[:, 0], y[:, 3] = I32_MAX - 1, 4
        y[1, 0], y[1, 3] = 3, 5
        y[2, 0], y[2, 3] = I32_MAX, 2
        sides = torch.tensor([0, 0, 0], dtype=torch.int64)
        active = torch.tensor([True, True, False])
        saved = y.clone()
        result = ov(y, sides, active)
        self.assertTrue(torch.equal(y, saved))
        self.assertEqual(result.errors.tolist(), [ERR_OVERFLOW, 0, 0])
        self.assertTrue(torch.equal(result.states[0], y[0]))
        self.assertEqual(int(result.states[1, 14]), 8)
        self.assertTrue(torch.equal(result.states[2], y[2]))
        with self.assertRaises(ValueError):
            lower_array(Program(
                name='too_big', version='t', skip_when=Const(I32_MAX + 1),
                binds=(Bind('z', Const(0)),), stores=(Store('done', Name('z')),),
            ))

    def test_cuda_matches_cpu_and_graph_capture_if_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        from app.modules.card_game.rl.rule_ir.lower_array import lower_array
        from tests.test_duel_v2_rule_ir import _make_cases, _pack
        step = lower_array(COMBAT_SLICE)
        states, opts = _make_cases(32, seed=88)
        x_cpu = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        sides_cpu, active_cpu = _rand_mask(len(states), 91, tail=4)
        expected = step(x_cpu, sides_cpu, active_cpu)
        x = x_cpu.cuda()
        sides = sides_cpu.cuda()
        active = active_cpu.cuda()
        got = step(x, sides, active)
        self.assertTrue(torch.equal(got.states.cpu(), expected.states))
        self.assertTrue(torch.equal(got.errors.cpu(), expected.errors))
        self.assertTrue(torch.equal(x.cpu(), x_cpu))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step(x, sides, active)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step(x, sides, active)
        alt_states, alt_opts = _make_cases(32, seed=89)
        x2 = torch.tensor([_pack(s, o) for s, o in zip(alt_states, alt_opts)], dtype=torch.int32, device='cuda')
        sides2, active2 = _rand_mask(32, 92, tail=4)
        sides2, active2 = sides2.cuda(), active2.cuda()
        x.copy_(x2)
        sides.copy_(sides2)
        active.copy_(active2)
        graph.replay()
        torch.cuda.synchronize()
        eager = step(x2, sides2, active2)
        self.assertTrue(torch.equal(captured.states, eager.states))
        self.assertTrue(torch.equal(captured.errors, eager.errors))


if __name__ == '__main__':
    unittest.main()
