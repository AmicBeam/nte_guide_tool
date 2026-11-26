"""Device page allocator tests; numerical fixtures do not execute game rules."""
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel.device_graph_pool import (
    DeviceGraphPagePool, DevicePageHandle, INVALID_HANDLE, DUPLICATE_SLOT,
    INVALID_SIZE, PAGE_CAPACITY, CORRUPT_PAGES, INVALID_OFFSET,
)


class DeviceGraphPoolTest(unittest.TestCase):
    def pool(self, device='cpu', slots=4, pages=8, per_slot=4):
        return DeviceGraphPagePool(slots=slots, pages=pages, page_bytes=16,
                                   max_pages_per_slot=per_slot, device=device,
                                   allow_cpu_reference=device == 'cpu')

    def vectors(self, device='cpu', slots=(0, 1), sizes=(20, 30), generations=None,
                owners=None, active=None):
        import torch
        b = len(slots)
        long = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
        return (long(slots), long([0] * b if generations is None else generations),
                long(list(range(40, 40 + b)) if owners is None else owners), long(sizes),
                torch.tensor([True] * b if active is None else active, device=device))

    def test_reserve_grow_translate_release_reuse_and_zero_pages(self):
        import torch
        p = self.pool()
        args = self.vectors()
        result = p.reserve(*args)
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(p.page_count.tolist(), [2, 2, 0, 0])
        p.data.fill_(93)
        old = p.page_table.clone()
        grown = p.grow(result.handle, torch.tensor([48, 30]), args[-1])
        self.assertEqual(grown.errors.tolist(), [0, 0])
        torch.testing.assert_close(p.page_table[:2, :2], old[:2, :2])
        self.assertTrue(bool((p.data[old[0, 0]] == 93).all()))
        self.assertTrue(bool((p.data[p.page_table[0, 2]] == 0).all()))
        addresses = p.translate(result.handle, torch.tensor([[0, 47], [0, 29]]), args[-1])
        self.assertEqual(addresses.errors.tolist(), [0, 0])
        self.assertEqual(int(addresses.offsets[0, 1]), int(p.page_table[0, 2]) * 16 + 15)
        self.assertEqual(p.release(result.handle, args[-1]).tolist(), [0, 0])
        self.assertEqual(p.generation.tolist(), [1, 1, 0, 0])
        self.assertTrue(bool((p.page_owner == -1).all()))
        self.assertTrue(bool((p.page_table == -1).all()))
        self.assertEqual(p.reserve(*args).errors.tolist(), [INVALID_HANDLE, INVALID_HANDLE])
        again = p.reserve(*self.vectors(generations=[1, 1]))
        self.assertEqual(again.errors.tolist(), [0, 0])
        self.assertTrue(bool((p.data[:4] == 0).all()))

    def test_over_capacity_is_atomic_and_existing_rows_survive(self):
        import torch
        p = self.pool(pages=4)
        first = p.reserve(*self.vectors(slots=[0], sizes=[48]))
        before = [v.clone() for v in (p.page_table, p.page_owner, p.logical_bytes)]
        result = p.reserve(*self.vectors(slots=[1, 2], sizes=[16, 16]))
        self.assertEqual(result.errors.tolist(), [PAGE_CAPACITY, PAGE_CAPACITY])
        for actual, expected in zip((p.page_table, p.page_owner, p.logical_bytes), before):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(p.translate(first.handle, torch.tensor([[47]]), torch.tensor([True])).errors.tolist(), [0])

    def test_duplicate_slots_rejected_without_mutation(self):
        p = self.pool()
        errors = p.reserve(*self.vectors(slots=[0, 0], sizes=[16, 16])).errors
        self.assertEqual(errors.tolist(), [DUPLICATE_SLOT, DUPLICATE_SLOT])
        self.assertEqual(int(p.live.sum()), 0)
        self.assertEqual(int((p.page_owner >= 0).sum()), 0)

    def test_inactive_or_invalid_slots_do_not_overwrite_valid_slot_zero(self):
        import torch
        p = self.pool()
        result = p.reserve(*self.vectors(slots=[0, -1], sizes=[16, 16], active=[True, False]))
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(p.page_table[0].tolist(), [0, -1, -1, -1])
        invalid = p.reserve(*self.vectors(slots=[9], sizes=[16]))
        self.assertEqual(invalid.errors.tolist(), [INVALID_HANDLE])
        self.assertEqual(p.owner.tolist(), [40, -1, -1, -1])
        invalid_release = DevicePageHandle(torch.tensor([-1]), torch.tensor([0]), torch.tensor([40]))
        p.release(invalid_release, torch.tensor([True]))
        self.assertEqual(int(p.live[0]), 1)

    def test_size_owner_bounds_and_corrupt_page_checks(self):
        import torch
        p = self.pool()
        bad = p.reserve(*self.vectors(sizes=[65, 0]))
        self.assertEqual(bad.errors.tolist(), [INVALID_SIZE, INVALID_SIZE])
        args = self.vectors()
        result = p.reserve(*args)
        wrong = DevicePageHandle(result.handle.slots, result.handle.generations,
                                 result.handle.owners + 1)
        self.assertEqual(p.release(wrong, args[-1]).tolist(), [INVALID_HANDLE, INVALID_HANDLE])
        invalid = p.translate(result.handle, torch.tensor([[20], [29]]), args[-1])
        self.assertEqual(invalid.errors.tolist(), [INVALID_OFFSET, 0])
        self.assertEqual(int(invalid.offsets[0, 0]), -1)
        p.page_owner[p.page_table[0, 0]] = 3
        self.assertEqual(p.release(result.handle, args[-1]).tolist(), [CORRUPT_PAGES, 0])
        self.assertEqual(int(p.live[0]), 1)

    def test_cpu_requires_explicit_oracle_and_bad_dimensions_reject(self):
        with self.assertRaises(ValueError):
            DeviceGraphPagePool(slots=4, pages=8, page_bytes=16,
                                max_pages_per_slot=4, device='cpu')
        with self.assertRaises(ValueError):
            DeviceGraphPagePool(slots=True, pages=8, page_bytes=16,
                                max_pages_per_slot=4, device='cpu', allow_cpu_reference=True)
        p = self.pool()
        self.assertEqual(p.allocated_bytes, sum(t.numel() * t.element_size() for t in (
            p.data, p.page_table, p.page_owner, p.generation, p.owner, p.live,
            p.logical_bytes, p.page_count, p.peak_used_pages, p._page_ids, p._positions)))

    def test_hot_path_keeps_all_data_on_device(self):
        import torch
        p = self.pool()
        args = self.vectors()
        offsets = torch.tensor([[0], [0]])
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('List copy')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            result = p.reserve(*args)
            p.grow(result.handle, args[-2], args[-1])
            p.translate(result.handle, offsets, args[-1])
            p.release(result.handle, args[-1])

    def test_actual_cuda_1600_allocator_rows_and_graph_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; fixture does not execute game rules')
        p = self.pool(device='cuda', slots=1600, pages=3200, per_slot=2)
        args = self.vectors(device='cuda', slots=list(range(1600)), sizes=[17] * 1600)
        offsets = torch.tensor([[0, 16]] * 1600, device=p.device)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                args[1].copy_(p.generation)
                result = p.reserve(*args)
                p.release(result.handle, args[-1])
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            args[1].copy_(p.generation)
            result = p.reserve(*args)
            addresses = p.translate(result.handle, offsets, args[-1])
            released = p.release(result.handle, args[-1])
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(bool((result.errors != 0).any()))
        self.assertFalse(bool((addresses.errors != 0).any()))
        self.assertFalse(bool((released != 0).any()))
        self.assertEqual(int(p.peak_used_pages), 3200)
        self.assertEqual(int(p.live.sum()), 0)


if __name__ == '__main__':
    unittest.main()
