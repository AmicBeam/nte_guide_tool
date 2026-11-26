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

    def test_prepared_shuffle_reuses_outputs_and_rejects_foreign_backend(self):
        values = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        counts = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
        seeds = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
        active = torch.ones(4, dtype=torch.int32)
        with DomainSoARandom(device='cpu', allow_cpu_reference=True) as backend:
            batch = backend.prepare(values, counts, seeds, active)
            first = backend.launch(batch)
            pointer = first.values.data_ptr()
            seeds.add_(100)
            second = backend.launch(batch)
            self.assertEqual(second.values.data_ptr(), pointer)
            for row in range(4):
                rng = random.Random(int(seeds[row]))
                expected = values[:, row].tolist()
                prefix = expected[:int(counts[row])]
                rng.shuffle(prefix)
                expected[:len(prefix)] = prefix
                self.assertEqual(second.values[:, row].tolist(), expected)
                self.assertEqual(int(second.seeds[row]), rng.getrandbits(63))
            with DomainSoARandom(device='cpu', allow_cpu_reference=True) as foreign:
                with self.assertRaises(ValueError):
                    foreign.launch(batch)
        with self.assertRaises(RuntimeError):
            backend.launch(batch)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA Graph required')
    def test_400_prepared_graph_uses_fresh_rng_and_no_native_compiler(self):
        rows, width = 400, 32
        values = torch.arange(width * rows, dtype=torch.int32).reshape(width, rows)
        counts = torch.arange(rows, dtype=torch.int32) % (width + 1)
        seeds = torch.arange(rows, dtype=torch.int64) + 2**40
        active = (torch.arange(rows) % 11 != 0).int()
        with DomainSoARandom(device='cuda') as backend:
            inputs = [v.cuda() for v in (values, counts, seeds, active)]
            batch = backend.prepare(*inputs)
            backend.launch(batch)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                backend.launch(batch)
            for offset in (0, 12345):
                inputs[2].copy_(seeds + offset)
                graph.replay()
                torch.cuda.synchronize()
                got, nxt, errors = (v.cpu() for v in
                    (batch.result.values, batch.result.seeds, batch.result.errors))
                self.assertEqual(errors.tolist(), [0] * rows)
                for row in range(rows):
                    expected = values[:, row].tolist()
                    seed = int(seeds[row]) + offset
                    if active[row]:
                        rng = random.Random(seed)
                        prefix = expected[:int(counts[row])]
                        rng.shuffle(prefix)
                        expected[:len(prefix)] = prefix
                        seed = rng.getrandbits(63)
                    self.assertEqual(got[:, row].tolist(), expected)
                    self.assertEqual(int(nxt[row]), seed)
            del graph
            torch.cuda.synchronize()

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
