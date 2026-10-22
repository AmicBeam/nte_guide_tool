"""CPU-focused leases for the direct named-field domain SoA state pool."""
import unittest
from unittest.mock import patch

import torch

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_duel.domain_soa_state_pool import (
    CAPACITY, DUPLICATE, INVALID_LEASE, DomainSoAStatePool, supports_full_duel_rules,
)
from app.modules.card_game.rl.batched_duel.domain_soa_transition import (
    FieldSpec, empty_fields, supports_full_duel_rules as transition_full,
)


SPECS = (
    FieldSpec('phase', 'int32'),
    FieldSpec('player_hp', 'int32', (2,)),
    FieldSpec('flag', 'bool'),
    FieldSpec('score', 'float64'),
    FieldSpec('stamp', 'uint8'),
)


class DomainSoAStatePoolTests(unittest.TestCase):
    def pool(self, capacity=4, device='cpu', specs=SPECS):
        return DomainSoAStatePool(
            capacity, device=device, allow_cpu_reference=device == 'cpu', specs=specs)

    def fields(self, pool, rows, **updates):
        out = {
            spec.name: torch.zeros(
                (rows,) + spec.tail, dtype=pool.columns[spec.name].dtype, device=pool.device)
            for spec in pool.specs
        }
        for name, value in updates.items():
            out[name].copy_(torch.as_tensor(value, dtype=out[name].dtype, device=pool.device))
        return out

    def owners(self, n, start=0):
        return torch.arange(start, start + n, dtype=torch.int64, device='cpu')

    def active(self, n, mask=None):
        if mask is None:
            return torch.ones(n, dtype=torch.bool)
        return torch.tensor(mask, dtype=torch.bool)

    def put(self, pool, rows, mask=None, start=0, **updates):
        return pool.put_device(
            self.fields(pool, rows, **updates),
            torch.arange(start, start + rows, dtype=torch.int64, device=pool.device),
            torch.ones(rows, dtype=torch.bool, device=pool.device) if mask is None
            else torch.tensor(mask, dtype=torch.bool, device=pool.device))

    def test_does_not_claim_full_duel_or_search_readiness(self):
        self.assertFalse(supports_full_duel_rules)
        self.assertFalse(DomainSoAStatePool.supports_full_duel_rules)
        self.assertFalse(self.pool().supports_full_duel_rules)
        self.assertFalse(transition_full)

    def test_fork_storage_is_packed_by_dtype_not_field(self):
        specs=tuple(FieldSpec('field_'+str(i),'int32',(2,))for i in range(100))
        pool=self.pool(capacity=4,specs=specs)
        self.assertEqual(len(pool._packed),1)
        self.assertEqual(pool._packed['int32'].shape,(4,200))
        parent=self.put(pool,1,field_0=[[7,8]],field_99=[[91,92]])
        child=pool.clone_device(parent)
        gathered=pool.gather_device(child)
        self.assertEqual(gathered['field_0'].tolist(),[[7,8]])
        self.assertEqual(gathered['field_99'].tolist(),[[91,92]])
        self.assertEqual(len({v.untyped_storage().data_ptr()for v in pool.columns.values()}),1)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_fused_cuda_fork_preserves_nan_signed_zero_and_extreme_bits(self):
        from dataclasses import replace
        snapshots=[]
        for fused in(False,True):
            pool=DomainSoAStatePool(8,device='cuda',specs=SPECS,fused_cuda_copy=fused)
            try:
                payload=self.fields(pool,2,phase=[2147483647,-2147483648],
                    player_hp=[[7,8],[-2,4]],score=[float('nan'),-0.0],stamp=[255,0],flag=[True,False])
                lease=pool.put_device(payload,torch.tensor([4,9],dtype=torch.int64,device='cuda'),
                    torch.ones(2,dtype=torch.bool,device='cuda'))
                child=pool.clone_device(lease)
                reverse=replace(child,slots=child.slots.flip(0),generations=child.generations.flip(0),
                    owners=child.owners.flip(0),active=child.active.flip(0),errors=child.errors.clone())
                pool.copy_device(child,reverse)
                fields=pool.gather_device(child)
                snapshots.append({k:v.view(torch.int64).cpu()if v.dtype==torch.float64 else v.cpu()
                                  for k,v in fields.items()})
                original=pool.gather_device(lease)
                for name,value in payload.items():
                    left=original[name];right=value
                    if value.dtype==torch.float64:left=left.view(torch.int64);right=right.view(torch.int64)
                    self.assertTrue(torch.equal(left,right),name)
                self.assertFalse(child.errors.any())
            finally:pool.close()
        for name in snapshots[0]:self.assertTrue(torch.equal(snapshots[0][name],snapshots[1][name]),name)

    def test_cpu_requires_explicit_oracle_and_bytes_match_resident_tensors(self):
        with self.assertRaises(ValueError):
            DomainSoAStatePool(4, device='cpu', specs=SPECS)
        with self.assertRaises(ValueError):
            DomainSoAStatePool(True, device='cpu', allow_cpu_reference=True, specs=SPECS)
        pool = self.pool()
        self.assertEqual(pool.allocated_bytes, sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*pool.columns.values(), pool.live, pool.generation, pool.owner, pool.ids)))

    def test_identity_roundtrip_preserves_integer_and_float_bits(self):
        pool = self.pool()
        score = torch.tensor([float('nan'), -0.0], dtype=torch.float64)
        lease = self.put(
            pool, 2,
            phase=[3, 1],
            player_hp=[[7, 11], [0, -4]],
            flag=[True, False],
            score=score,
            stamp=[255, 1],
        )
        gathered = pool.gather_device(lease)
        self.assertEqual(gathered['phase'].tolist(), [3, 1])
        self.assertEqual(gathered['player_hp'].tolist(), [[7, 11], [0, -4]])
        self.assertEqual(gathered['flag'].tolist(), [True, False])
        self.assertEqual(gathered['stamp'].tolist(), [255, 1])
        torch.testing.assert_close(gathered['score'].view(torch.int64), score.view(torch.int64), rtol=0, atol=0)
        gathered['phase'][0] = 99
        self.assertEqual(pool.gather_device(lease)['phase'].tolist(), [3, 1])

    def test_shared_source_clone_keeps_parent_isolated(self):
        pool = self.pool()
        parent = self.put(pool, 1, phase=[4], player_hp=[[9, 8]], score=[1.5], stamp=[7], flag=[True])
        before = {name: column.clone() for name, column in pool.columns.items()}
        shared = DeviceStateBatch(
            pool, parent.slots.repeat(2), parent.generations.repeat(2),
            parent.owners.repeat(2), parent.active.repeat(2), parent.errors.repeat(2))
        children = pool.clone_device(shared)
        self.assertEqual(children.active.tolist(), [True, True])
        self.assertEqual(len(set(children.slots.tolist())), 2)
        self.assertNotIn(int(parent.slots[0]), children.slots.tolist())
        child_fields = pool.gather_device(children)
        self.assertEqual(child_fields['phase'].tolist(), [4, 4])
        self.assertEqual(child_fields['stamp'].tolist(), [7, 7])
        for name, column in pool.columns.items():
            torch.testing.assert_close(column[parent.slots[0]], before[name][parent.slots[0]], rtol=0, atol=0)
        other = self.put(pool, 1, start=9, phase=[1], player_hp=[[1, 2]], score=[8.0], stamp=[3])
        aligned_parent = DeviceStateBatch(
            pool, parent.slots.repeat(2), parent.generations.repeat(2),
            parent.owners.repeat(2), parent.active.repeat(2), torch.zeros(2, dtype=torch.int32))
        pool.copy_device(DeviceStateBatch(
            pool, other.slots.repeat(2), other.generations.repeat(2),
            other.owners.repeat(2), other.active.repeat(2), torch.zeros(2, dtype=torch.int32)), children)
        self.assertEqual(pool.gather_device(parent)['phase'].tolist(), [4])
        self.assertEqual(pool.gather_device(aligned_parent, shared_reads=True)['phase'].tolist(), [4, 4])
        self.assertEqual(pool.gather_device(children)['phase'].tolist(), [1, 1])

    def test_duplicate_destination_is_rejected_per_row(self):
        pool = self.pool()
        src = self.put(pool, 1, phase=[5], stamp=[9])
        dest = self.put(pool, 2, start=3, phase=[1, 2], stamp=[1, 2])
        duplicate = DeviceStateBatch(
            pool, dest.slots[:1].repeat(2), dest.generations[:1].repeat(2),
            dest.owners[:1].repeat(2), dest.active[:1].repeat(2), torch.zeros(2, dtype=torch.int32))
        sources = DeviceStateBatch(
            pool, src.slots.repeat(2), src.generations.repeat(2),
            src.owners.repeat(2), src.active.repeat(2), torch.zeros(2, dtype=torch.int32))
        pool.copy_device(sources, duplicate)
        self.assertEqual(duplicate.errors.tolist(), [DUPLICATE, DUPLICATE])
        self.assertEqual(pool.gather_device(dest)['phase'].tolist(), [1, 2])
        unique = DeviceStateBatch(
            pool, src.slots.repeat(2), src.generations.repeat(2),
            src.owners.repeat(2), src.active.repeat(2), torch.zeros(2, dtype=torch.int32))
        pool.copy_device(unique, dest)
        self.assertEqual(dest.errors.tolist(), [0, 0])
        self.assertEqual(pool.gather_device(dest)['phase'].tolist(), [5, 5])

    def test_stale_foreign_and_generation_reuse(self):
        pool = self.pool(capacity=2)
        first = self.put(pool, 1, phase=[6], stamp=[4])
        pool.release_device(first)
        pool.gather_device(first)
        self.assertEqual(first.errors.tolist(), [INVALID_LEASE])
        second = self.put(pool, 1, start=8, phase=[9], stamp=[2])
        self.assertTrue(second.active[0])
        stale = DeviceStateBatch(
            pool, first.slots.clone(), first.generations.clone(),
            first.owners.clone(), torch.tensor([True]), torch.zeros(1, dtype=torch.int32))
        pool.gather_device(stale)
        self.assertEqual(stale.errors.tolist(), [INVALID_LEASE])
        foreign = DeviceStateBatch(
            pool, second.slots.clone(), second.generations.clone(),
            second.owners.clone() + 1, second.active.clone(), torch.zeros(1, dtype=torch.int32))
        pool.gather_device(foreign)
        self.assertEqual(foreign.errors.tolist(), [INVALID_LEASE])
        self.assertEqual(pool.gather_device(second)['phase'].tolist(), [9])

    def test_inactive_rows_and_atomic_overflow_leave_parents(self):
        pool = self.pool(capacity=2)
        first = self.put(pool, 2, phase=[1, 2], stamp=[3, 4], mask=[True, False])
        self.assertEqual(first.active.tolist(), [True, False])
        self.assertEqual(int(pool.live.sum()), 1)
        self.assertEqual(pool.gather_device(first)['phase'].tolist(), [1, 0])
        failed = self.put(pool, 2, start=4, phase=[8, 9], stamp=[8, 9])
        self.assertEqual(failed.errors.tolist(), [CAPACITY, CAPACITY])
        self.assertEqual(int(pool.live.sum()), 1)
        self.assertEqual(pool.gather_device(first)['phase'].tolist(), [1, 0])
        shared = DeviceStateBatch(
            pool, first.slots[:1].repeat(2), first.generations[:1].repeat(2),
            first.owners[:1].repeat(2), torch.tensor([True, True]), torch.zeros(2, dtype=torch.int32))
        cloned = pool.clone_device(shared)
        self.assertEqual(cloned.errors.tolist(), [CAPACITY, CAPACITY])
        self.assertEqual(int(pool.live.sum()), 1)

    def test_default_transition_fields_roundtrip(self):
        pool = DomainSoAStatePool(2, device='cpu', allow_cpu_reference=True)
        fields = empty_fields(1, device='cpu')
        fields['phase'].fill_(2)
        fields['player_hp'][0, 1] = 17
        lease = pool.put_device(dict(fields), torch.zeros(1, dtype=torch.int64), torch.tensor([True]))
        gathered = pool.gather_device(lease)
        self.assertEqual(set(gathered), set(fields))
        self.assertEqual(int(gathered['phase'][0]), 2)
        self.assertEqual(int(gathered['player_hp'][0, 1]), 17)

    def test_hot_path_keeps_all_data_on_device(self):
        pool = self.pool()
        fields = self.fields(pool, 1, phase=[2], stamp=[5])
        owners = self.owners(1)
        active = self.active(1)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('List copy')):
            lease = pool.put_device(fields, owners, active)
            cloned = pool.clone_device(lease)
            pool.gather_device(lease)
            pool.copy_device(lease, cloned)
            pool.release_device(cloned)

    def test_cuda_default_refuses_unavailable(self):
        if torch.cuda.is_available():
            pool = DomainSoAStatePool(2, device='cuda', specs=SPECS)
            self.assertEqual(pool.device.type, 'cuda')
            return
        with self.assertRaises(RuntimeError):
            DomainSoAStatePool(2, device='cuda', specs=SPECS)

    def test_actual_cuda_roundtrip_skipped_without_device(self):
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; storage fixture does not execute game rules')
        pool = self.pool(capacity=4, device='cuda')
        lease = pool.put_device(
            self.fields(pool, 1, phase=[4], stamp=[6]),
            torch.zeros(1, dtype=torch.int64, device=pool.device),
            torch.tensor([True], device=pool.device))
        cloned = pool.clone_device(lease)
        gathered = pool.gather_device(cloned, shared_reads=True)
        self.assertEqual(gathered['phase'].device.type, 'cuda')
        self.assertTrue(bool((gathered['phase'] == 4).all()))
        pool.release_device(cloned)
        pool.release_device(lease)


if __name__ == '__main__':
    unittest.main()
