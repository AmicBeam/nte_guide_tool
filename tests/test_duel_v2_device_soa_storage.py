import unittest

import torch

from app.modules.card_game.rl.batched_duel.device_soa_storage import DeviceSoAStore, CAPACITY, INVALID_LEASE, DUPLICATE
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states, unpack_game_states


class DeviceSoAStorageTests(unittest.TestCase):
    def setUp(self):
        self.schema = compile_schema({'Game': {'kind': 'record', 'capacity': 2,
            'fields': {'hp': 'int', 'name': 'str'}}}, max_strings=8, max_string_bytes=128)
        self.store = DeviceSoAStore(self.schema, 3, device='cpu', allow_cpu_reference=True)

    def batch(self, values):
        return pack_game_states(values, self.schema).to_torch('cpu')

    def put(self, values, mask=None):
        rows = len(values)
        return self.store.put_device(self.batch(values), torch.arange(rows, dtype=torch.int64),
            torch.ones(rows, dtype=torch.bool) if mask is None else torch.tensor(mask, dtype=torch.bool))

    def test_roundtrip_and_reused_generation(self):
        first = self.put([{'hp': 10, 'name': '真红'}, {'hp': 20}])
        self.assertEqual(unpack_game_states(self.store.gather_device(first).to_host()),
                         [{'hp': 10, 'name': '真红'}, {'hp': 20}])
        self.store.release_device(first)
        second = self.put([{'hp': 50}])
        self.assertTrue(second.active[0])
        self.store.gather_device(first)
        self.assertTrue((first.errors & INVALID_LEASE).all())
        self.assertTrue(self.store.live[second.slots[0]])

    def test_capacity_is_atomic_and_inactive_preserves_live(self):
        first = self.put([{'hp': 1}, {'hp': 2}])
        failed = self.put([{'hp': 9}, {'hp': 8}])
        self.assertTrue((failed.errors == CAPACITY).all())
        self.assertEqual(self.store.live.sum().item(), 2)
        third = self.put([{'hp': 7}, {'hp': 99}], [True, False])
        self.assertEqual(third.active.tolist(), [True, False])
        self.assertEqual(unpack_game_states(self.store.gather_device(first).to_host()), [{'hp': 1}, {'hp': 2}])

    def test_duplicate_release_rejected_shared_read_allowed(self):
        first = self.put([{'hp': 3}])
        duplicate = DeviceStateBatch(self.store, first.slots.repeat(2), first.generations.repeat(2),
            first.owners.repeat(2), first.active.repeat(2), first.errors.repeat(2))
        self.assertEqual(unpack_game_states(self.store.gather_device(duplicate, shared_reads=True).to_host()),
                         [{'hp': 3}, {'hp': 3}])
        self.store.release_device(duplicate)
        self.assertTrue((duplicate.errors == DUPLICATE).all())
        self.assertEqual(self.store.live.sum().item(), 1)

    def test_missing_root_cannot_be_leased(self):
        batch=self.batch([{'hp':3}])
        batch.roots.zero_()
        result=self.store.put_device(batch,torch.tensor([0],dtype=torch.int64),torch.tensor([True]))
        self.assertEqual(result.errors.tolist(),[INVALID_LEASE])
        self.assertFalse(result.active[0])
        self.assertEqual(self.store.live.sum().item(),0)

    def test_reusable_gather_replaces_all_columns_and_preserves_input(self):
        first=self.put([{'hp':3,'name':'old'},{'hp':9}])
        workspace=self.batch([{},{}])
        identities={key:value.data_ptr() for key,value in workspace.buffers.items()}
        result=self.store.gather_into_device(first,workspace)
        self.assertIs(result,workspace)
        self.assertEqual(unpack_game_states(result.to_host()),[{'hp':3,'name':'old'},{'hp':9}])
        self.assertEqual(identities,{key:value.data_ptr() for key,value in workspace.buffers.items()})
        workspace.buffers['i64'].zero_()
        self.assertEqual(unpack_game_states(self.store.gather_device(first).to_host()),
                         [{'hp':3,'name':'old'},{'hp':9}])


if __name__ == '__main__':
    unittest.main()
