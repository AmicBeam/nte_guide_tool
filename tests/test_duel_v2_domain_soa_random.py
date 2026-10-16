import random
import unittest

import torch

from app.modules.card_game.rl.batched_duel.domain_soa_random import (
    DomainSoARandom, INVALID_INPUT, RNG_EXHAUSTED,
)


class SoARandomTest(unittest.TestCase):
    def test_native_shuffle_matches_formal_full_63bit_stream_and_tail(self):
        from app.modules.card_game.engine.duel_v2.state import shuffle
        seeds = [0, 1, 2**31-1, 2**31, 2**32, 2**62+123, 2**63-1, 700001]
        counts = [0, 1, 2, 3, 5, 10, 32, 17]
        values = torch.arange(32*8, dtype=torch.int32).reshape(32, 8)
        expected = values.clone()
        expected_seeds = []
        for row, (seed, count) in enumerate(zip(seeds, counts)):
            items = values[:count, row].tolist()
            state = {'rng': seed}
            shuffle(state, items)
            expected[:count, row] = torch.tensor(items, dtype=torch.int32)
            expected_seeds.append(state['rng'])
        original = values.clone()
        with DomainSoARandom(device='cpu', allow_cpu_reference=True) as backend:
            result = backend.shuffle(values, torch.tensor(counts, dtype=torch.int32),
                torch.tensor(seeds, dtype=torch.int64), torch.ones(8, dtype=torch.int32))
        self.assertTrue(torch.equal(result.values, expected))
        self.assertEqual(result.seeds.tolist(), expected_seeds)
        self.assertTrue(torch.equal(values, original))
        self.assertEqual(result.errors.tolist(), [0]*8)

    def test_failed_and_inactive_rows_do_not_consume_rng_or_modify_slots(self):
        values = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        seeds = torch.tensor([9, -1, 88, -1], dtype=torch.int64)
        with DomainSoARandom(device='cpu', allow_cpu_reference=True) as backend:
            result = backend.shuffle(values, torch.tensor([4, 2, -1, 999], dtype=torch.int32),
                seeds, torch.tensor([1, 1, 1, 0], dtype=torch.int32))
        self.assertEqual(result.errors.tolist(), [INVALID_INPUT]*3 + [0])
        self.assertTrue(torch.equal(result.values, values))
        self.assertTrue(torch.equal(result.seeds, seeds))

    def test_rejection_budget_fails_closed_instead_of_biasing_shuffle(self):
        seed = next(s for s in range(100) if random.Random(s).getrandbits(2) >= 2)
        values = torch.tensor([[10], [20]], dtype=torch.int32)
        seeds = torch.tensor([seed], dtype=torch.int64)
        with DomainSoARandom(device='cpu', allow_cpu_reference=True, max_draws=1) as backend:
            result = backend.shuffle(values, torch.tensor([2], dtype=torch.int32), seeds,
                torch.ones(1, dtype=torch.int32))
        self.assertEqual(result.errors.tolist(), [RNG_EXHAUSTED])
        self.assertTrue(torch.equal(result.values, values))
        self.assertTrue(torch.equal(result.seeds, seeds))

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA required; native RNG parity is not device qualification')
    def test_actual_cuda_tail_and_repeated_execution(self):
        values = torch.arange(32*8, dtype=torch.int32).reshape(32, 8)
        seeds = torch.tensor([0, 1, 2**31, 2**32, 2**62, 900, 1, -1], dtype=torch.int64)
        counts = torch.tensor([0, 1, 2, 8, 10, 32, 99, 999], dtype=torch.int32)
        active = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.int32)
        with DomainSoARandom(device='cpu', allow_cpu_reference=True) as oracle:
            expected = oracle.shuffle(values, counts, seeds, active)
        with DomainSoARandom(device='cuda') as backend:
            for _ in range(2):
                result = backend.shuffle(values.cuda(), counts.cuda(), seeds.cuda(), active.cuda())
                torch.cuda.synchronize()
                for name in ('values', 'seeds', 'errors'):
                    self.assertTrue(torch.equal(getattr(result, name).cpu(), getattr(expected, name)))


if __name__ == '__main__':
    unittest.main()
