"""Sparse typed storage parity; CPU reference never grants CUDA capacity."""
from dataclasses import replace
import unittest
import numpy as np
import torch

from app.modules.card_game.rl.batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore, CORRUPT_COLUMN
from app.modules.card_game.rl.batched_duel.device_soa_storage import CAPACITY, INVALID_LEASE, DUPLICATE
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states, unpack_game_states
from tests.test_duel_v2_record_soa import schema


def assert_bits(test, actual, expected):
    test.assertTrue(torch.equal(actual.roots.cpu(), expected.roots.cpu()))
    for key in expected.buffers:
        left = actual.buffers[key].cpu(); right = expected.buffers[key].cpu()
        if key == 'f64':
            left = left.view(torch.int64); right = right.view(torch.int64)
        test.assertTrue(torch.equal(left, right), key)


class SparseSoAStorageTests(unittest.TestCase):
    def test_64_work_rows_sparse_and_paged_preserve_tail_bits_and_reclaim(self):
        from app.modules.card_game.rl.batched_duel.device_paged_soa_storage import DevicePagedSoAStore
        for cls in (DeviceSparseSoAStore,DevicePagedSoAStore):
            spec=compile_schema({'Game':{'kind':'record','capacity':4,'fields':{'hp':'int','rate':'float'}}},
                                max_strings=4,max_string_bytes=32)
            store=cls(spec,70,device='cpu',allow_cpu_reference=True)
            self.addCleanup(store.close)
            for rows in (33,64):
                source=pack_game_states([{'hp':i+1,'rate':-0.} for i in range(rows)],spec).to_torch('cpu')
                active=torch.ones(rows,dtype=torch.bool);active[1]=False
                owners=torch.arange(rows,dtype=torch.int64)+100
                lease=store.put_device(source,owners,active)
                self.assertEqual(lease.errors.tolist(),[0]*rows)
                actual=store.gather_device(lease)
                for key in source.buffers:
                    a,b=actual.buffers[key][:,active],source.buffers[key][:,active]
                    if key=='f64':a,b=a.view(torch.int64),b.view(torch.int64)
                    self.assertTrue(torch.equal(a,b))
                self.assertEqual(int(store.live.sum()),rows-1)
                store.release_device(lease)
                self.assertEqual(int(store.live.sum()),0)
                stale=replace(lease,errors=torch.zeros_like(lease.errors))
                store.gather_device(stale)
                self.assertTrue(bool((stale.errors[active]&INVALID_LEASE).all()))
            with self.assertRaises(ValueError):store.prepare_workspace(65)

    def setUp(self):
        self.schema = schema()
        self.store = DeviceSparseSoAStore(self.schema, 4, device='cpu', allow_cpu_reference=True)
        self.addCleanup(self.store.close)

    def batch(self, values):
        return pack_game_states(values, self.schema).to_torch('cpu')

    def put(self, values, active=None):
        batch = self.batch(values); n = len(values)
        return self.store.put_device(batch, torch.arange(n, dtype=torch.int64),
            torch.ones(n, dtype=torch.bool) if active is None else torch.tensor(active)), batch

    def test_bits_alias_presence_and_branch_independence(self):
        item = {'n':-(1 << 63)}
        source = {'right':item, 'left':item, 'items':[item,item], 'hp':0,
                  'rate':-0.0, 'ready':False, 'name':'真红', 'mapping':{'b':0,'a':2}, 'pair':(0,'x')}
        lease, packed = self.put([source, {'hp':None}])
        actual = self.store.gather_device(lease)
        assert_bits(self, actual, packed)
        decoded = unpack_game_states(actual.to_host())
        self.assertIs(decoded[0]['right'], decoded[0]['left'])
        self.assertIs(decoded[0]['items'][1], decoded[0]['left'])
        self.assertTrue(np.signbit(decoded[0]['rate']))
        actual.buffers['i64'].zero_()
        assert_bits(self, self.store.gather_device(lease), packed)

    def test_slot_capacity_atomic_masked_tail_and_generation(self):
        first, packed = self.put([{'hp':1}, {'hp':2}, {'hp':3}])
        failed, _ = self.put([{'hp':8}, {'hp':9}])
        self.assertEqual(failed.errors.tolist(), [CAPACITY, CAPACITY])
        self.assertEqual(int(self.store.live.sum()), 3)
        tail, _ = self.put([{'hp':4}, {'hp':99}], [True,False])
        self.assertEqual(tail.active.tolist(), [True,False])
        assert_bits(self, self.store.gather_device(first), packed)
        self.store.release_device(first)
        newer, _ = self.put([{'hp':5}])
        self.store.gather_device(first)
        self.assertTrue(bool((first.errors & INVALID_LEASE).all()))
        self.assertTrue(bool(self.store.live[newer.slots[0]]))
        self.assertEqual(int(self.store.peak_live[0]), 4)

    def test_sparse_cell_limit_rejects_without_truncation(self):
        caps = dict(self.store.column_cells, i64=1)
        store = DeviceSparseSoAStore(self.schema, 4, column_cells=caps, device='cpu', allow_cpu_reference=True)
        self.addCleanup(store.close)
        source = self.batch([{'hp':4}, {'hp':5,'ready':True}])
        lease = store.put_device(source, torch.tensor([0,1]), torch.tensor([True,True]))
        self.assertEqual(lease.errors.tolist(), [0,CAPACITY])
        self.assertEqual(lease.active.tolist(), [True,False])
        result = store.gather_device(lease)
        self.assertEqual(unpack_game_states(result.to_host().clone_rows([0]))[0], {'hp':4})
        self.assertEqual(int(store.live.sum()), 1)

    def test_duplicate_reads_allowed_release_and_foreign_owner_rejected(self):
        lease, _ = self.put([{'hp':3}])
        dupe = DeviceStateBatch(self.store, lease.slots.repeat(2), lease.generations.repeat(2),
            lease.owners.repeat(2), lease.active.repeat(2), lease.errors.repeat(2))
        self.assertEqual(unpack_game_states(self.store.gather_device(dupe, shared_reads=True).to_host()),
                         [{'hp':3}, {'hp':3}])
        self.store.release_device(dupe)
        self.assertEqual(dupe.errors.tolist(), [DUPLICATE,DUPLICATE])
        foreign = replace(lease, owners=lease.owners+1, errors=lease.errors.clone())
        self.store.release_device(foreign)
        self.assertEqual(foreign.errors.tolist(), [INVALID_LEASE])
        self.assertTrue(bool(self.store.live[lease.slots[0]]))

    def test_corrupt_sparse_index_fails_and_clears_result(self):
        lease, _ = self.put([{'hp':3}])
        self.store.indices['i64'][0, int(lease.slots[0])] = self.schema.lengths['i64']
        result = self.store.gather_device(lease)
        self.assertEqual(lease.errors.tolist(), [CORRUPT_COLUMN])
        self.assertEqual(result.roots.tolist(), [0])
        self.assertTrue(all(not bool(value.any()) for value in result.buffers.values()))

    def test_1600_slot_metadata_uses_bounded_pack_workspace(self):
        store = DeviceSparseSoAStore(self.schema, 1600, device='cpu', allow_cpu_reference=True)
        self.addCleanup(store.close)
        store.prepare_workspace(3)
        self.assertEqual(list(store._workspaces), [3])
        with self.assertRaises(ValueError):
            store.prepare_workspace(1600)
        with self.assertRaises(MemoryError):
            DeviceSparseSoAStore(self.schema, 1600, max_bytes=1, device='cpu', allow_cpu_reference=True)
        with self.assertRaises(ValueError):
            DeviceSparseSoAStore(self.schema, True, device='cpu', allow_cpu_reference=True)

    def test_actual_cuda_capture_repeated_bits_and_tail(self):
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        spec = compile_schema({'Game':{'kind':'record','capacity':300,
            'fields':{'hp':'int','rate':'float'}}}, max_strings=32,max_string_bytes=512)
        host = pack_game_states([{'hp':1},{'hp':2},{'hp':3}], spec)
        rng = np.random.default_rng(20261017)
        for key, values in host.buffers.items():
            if key == 'f64':
                bits = rng.integers(-(1 << 63),(1 << 63)-1,values.shape,dtype=np.int64)
                bits[::5] = 0; bits[0,0] = -(1 << 63)
                values[:] = bits.view(np.float64)
            elif key == 'u8':
                values[:] = rng.integers(0,256,values.shape,dtype=np.uint8); values[::4] = 0
            else:
                values[:] = rng.integers(-100,100,values.shape,dtype=values.dtype); values[::4] = 0
        batch = host.to_torch('cuda')
        store = DeviceSparseSoAStore(spec, 8, column_cells=spec.lengths, device='cuda')
        self.addCleanup(store.close); store.prepare_workspace(3)
        active = torch.tensor([True,False,True], device='cuda')
        owners = torch.tensor([8,9,10], device='cuda')
        output = pack_game_states([{}, {}, {}], spec).to_torch('cuda')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lease = store.put_device(batch, owners, active)
            store.gather_into_device(lease, output)
            store.release_device(lease)
        expected = host.to_torch('cpu')
        for value in expected.buffers.values():
            value[:,1] = 0
        expected.roots[1] = 0
        for _ in range(3):
            graph.replay(); torch.cuda.synchronize()
            self.assertEqual(lease.errors.cpu().tolist(), [0,0,0])
            assert_bits(self, output, expected)
            self.assertEqual(int(store.live.sum().cpu()), 0)


if __name__ == '__main__':
    unittest.main()
