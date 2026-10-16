"""Canonical and Decimal oracles for the bounded array max-HP primitive.

Life-lock, finished, knockdowns and events stay outside this helper.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import random
import unittest

import torch

from app.modules.card_game.rl.batched_duel.array_health import (
    AMOUNT_EXCEEDS_MAX_CUTS, INVALID_INPUT, reduce_max_hp_arrays, supports_full_duel_rules,
)


def decimal_cut(hp, max_hp, amount):
    amount = max(0, int(amount))
    if hp <= 0:
        return hp, max_hp, 0
    cut = 0
    for _ in range(amount):
        old = max_hp
        if old <= 0:
            break
        new = old - 1
        hp = min(int((Decimal(hp) / Decimal(old) * Decimal(new)).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP)), new)
        max_hp = new
        cut += 1
        if hp <= 0 or max_hp <= 0:
            hp = 0
            break
    return hp, max_hp, cut


def tensors(rows, device='cpu'):
    hp, mx, amt, active = zip(*rows)
    return (torch.tensor(hp, dtype=torch.int32, device=device),
            torch.tensor(mx, dtype=torch.int32, device=device),
            torch.tensor(amt, dtype=torch.int32, device=device),
            torch.tensor(active, dtype=torch.bool, device=device))


class ArrayHealthTest(unittest.TestCase):
    def test_canonical_players_and_characters(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import hero, reduce_max_hp
        self.assertFalse(supports_full_duel_rules)
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        cases = [
            ('a', 'nanali', 5, 7, 1), ('a', 'nanali', 5, 7, 2), ('a', 'nanali', 5, 5, 2),
            ('a', 'iloy', 3, 6, 1), ('a', 'zero', 1, 5, 4), ('a', 'jiuyuan', 4, 4, 4),
            ('b', 'player', 5, 7, 1), ('b', 'player', 5, 7, 2), ('b', 'player', 20, 30, 1),
            ('a', 'player', 30, 30, 0), ('a', 'player', 8, 30, -4), ('a', 'nanali', 0, 5, 3),
        ]
        expect = []
        for side, cid, hp, mx, amt in cases:
            s = deepcopy(state)
            unit = s['sides'][side] if cid == 'player' else hero(s, side, cid)
            unit.update(hp=hp, max_hp=mx)
            cut = reduce_max_hp(s, side, cid, amt, settle=False)
            expect.append((unit['hp'], unit['max_hp'], cut))
        out = reduce_max_hp_arrays(*tensors([c[2:] + (True,) for c in cases]))
        self.assertEqual(out.errors.tolist(), [0] * len(cases))
        self.assertEqual(list(zip(out.hp.tolist(), out.max_hp.tolist(), out.cuts.tolist())), expect)
        locked = deepcopy(state)
        locked['sides']['a'].update(hp=10, max_hp=30, life_locked=True)
        self.assertEqual(reduce_max_hp(locked, 'a', 'player', 1, settle=False), 0)
        still = reduce_max_hp_arrays(*tensors([(10, 30, 1, True)]))
        self.assertEqual((still.hp.item(), still.max_hp.item(), still.cuts.item()), (10, 29, 1))

    def test_decimal_oracle_repeated_rounding(self):
        rng = random.Random(20261016)
        rows = [(5, 7, 2, True), (3, 6, 1, True), (1, 2, 1, True), (20, 30, 5, True)]
        for mx in range(2, 41, 2):
            rows.append((mx // 2, mx, 1 + (mx % 5), True))
        for _ in range(240):
            mx = rng.randint(1, 48)
            rows.append((rng.randint(0, mx), mx, rng.randint(0, 12), True))
        for _ in range(80):
            mx = rng.randint(1, 400)
            rows.append((rng.randint(0, mx), mx, rng.randint(0, 8), True))
        hp, mx, amt, active = tensors(rows)
        out = reduce_max_hp_arrays(hp, mx, amt, active, max_cuts=16)
        expect = [decimal_cut(h, m, a) for h, m, a, _ in rows]
        self.assertEqual(out.errors.tolist(), [0] * len(rows))
        self.assertEqual(list(zip(out.hp.tolist(), out.max_hp.tolist(), out.cuts.tolist())), expect)

    def test_masks_errors_inactive_immutability(self):
        rows = [
            (5, 7, 1, True), (5, 7, 1, False), (0, 9, 4, True), (-1, 8, 1, True),
            (4, -3, 1, True), (6, 8, 9, True), (6, 8, -5, True), (3, 0, 2, True),
            (8, 8, 9, False),
        ]
        hp, mx, amt, active = tensors(rows)
        saved = (hp.clone(), mx.clone(), amt.clone(), active.clone())
        out = reduce_max_hp_arrays(hp, mx, amt, active, max_cuts=4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((hp, mx, amt, active), saved)))
        want_hp = [4, 5, 0, -1, 4, 6, 6, 3, 8]
        want_max = [6, 7, 9, 8, -3, 8, 8, 0, 8]
        want_cuts = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        want_err = [0, 0, 0, INVALID_INPUT, INVALID_INPUT, AMOUNT_EXCEEDS_MAX_CUTS, 0, 0, 0]
        self.assertEqual(out.hp.tolist(), want_hp)
        self.assertEqual(out.max_hp.tolist(), want_max)
        self.assertEqual(out.cuts.tolist(), want_cuts)
        self.assertEqual(out.errors.tolist(), want_err)
        out.hp[0] = 0
        self.assertEqual(hp[0].item(), 5)
        with self.assertRaises(ValueError):
            reduce_max_hp_arrays(hp.double(), mx, amt, active)
        with self.assertRaises(ValueError):
            reduce_max_hp_arrays(hp, mx, amt, active, max_cuts=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_cuda_capture(self):
        rows = [(5, 7, 2, True), (3, 6, 1, True), (0, 5, 3, True), (4, 8, 1, False)]
        cpu = reduce_max_hp_arrays(*tensors(rows), max_cuts=8)
        args = tensors(rows, 'cuda')
        reduce_max_hp_arrays(*args, max_cuts=8)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            reduce_max_hp_arrays(*args, max_cuts=8)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = reduce_max_hp_arrays(*args, max_cuts=8)
        graph.replay()
        self.assertEqual(out.hp.tolist(), cpu.hp.tolist())
        self.assertEqual(out.max_hp.tolist(), cpu.max_hp.tolist())
        self.assertEqual(out.cuts.tolist(), cpu.cuts.tolist())
        self.assertEqual(out.errors.tolist(), cpu.errors.tolist())


if __name__ == '__main__':
    unittest.main()
