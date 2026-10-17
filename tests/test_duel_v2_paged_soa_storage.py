"""Paged typed transport/lease conformance, separate from rule qualification."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from app.modules.card_game.rl.batched_duel.device_paged_soa_storage import DevicePagedSoAStore,paged_arena_plan
from app.modules.card_game.rl.batched_duel.device_soa_storage import CAPACITY,INVALID_LEASE,DUPLICATE
from app.modules.card_game.rl.batched_duel.device_sparse_soa_storage import CORRUPT_COLUMN,WIDTHS
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema,pack_game_states,TorchRecordSoABatch
from tests.test_duel_v2_record_soa import schema
from tests.test_duel_v2_sparse_soa_storage import assert_bits


def random_columns(spec,device):
    host=pack_game_states([{'hp':1},{'hp':2},{'hp':3}],spec);rng=np.random.default_rng(317)
    for key,value in host.buffers.items():
        if key=='f64':
            bits=rng.integers(-(1<<63),(1<<63)-1,value.shape,dtype=np.int64);bits[::5]=0;bits[0,0]=-(1<<63)
            value[:]=bits.view(np.float64)
        elif key=='u8':value[:]=rng.integers(0,256,value.shape,dtype=np.uint8);value[::4]=0
        else:value[:]=rng.integers(-100,100,value.shape,dtype=value.dtype);value[::4]=0
    return host.to_torch(device)


class PagedTypedStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.schema=schema();self.store=DevicePagedSoAStore(self.schema,4,device='cpu',allow_cpu_reference=True,page_bytes=64)
        self.addCleanup(self.store.close)

    def put(self,states,active=None):
        batch=pack_game_states(states,self.schema).to_torch('cpu');n=len(states)
        lease=self.store.put_device(batch,torch.arange(n,dtype=torch.int64),
            torch.ones(n,dtype=torch.bool) if active is None else torch.tensor(active))
        return lease,batch

    def test_alias_bits_negative_zero_order_and_independent_work_rows(self):
        item={'n':-(1<<63)};value={'right':item,'left':item,'items':[item,item],'hp':0,'rate':-0.,
            'ready':False,'name':'真红','mapping':{'b':0,'a':2},'pair':(0,'x')}
        lease,source=self.put([value,{'hp':None}]);out=self.store.gather_device(lease)
        assert_bits(self,out,source)
        from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
        decoded=unpack_game_states(out.to_host());self.assertIs(decoded[0]['right'],decoded[0]['left'])
        self.assertTrue(np.signbit(decoded[0]['rate']))
        out.buffers['i64'].zero_();assert_bits(self,self.store.gather_device(lease),source)

    def test_atomic_slot_exhaustion_tail_reuse_stale_and_foreign_owner(self):
        first,source=self.put([{'hp':1},{'hp':2},{'hp':3}])
        failed,_=self.put([{'hp':8},{'hp':9}]);self.assertEqual(failed.errors.tolist(),[CAPACITY,CAPACITY])
        tail,_=self.put([{'hp':4},{'hp':99}],[True,False]);self.assertEqual(tail.active.tolist(),[True,False])
        assert_bits(self,self.store.gather_device(first),source)
        self.store.release_device(first)
        newer,_=self.put([{'hp':5}]);self.assertEqual(newer.generations.tolist(),[2])
        self.store.gather_device(first);self.assertEqual(first.errors.tolist(),[INVALID_LEASE]*3)
        foreign=replace(newer,owners=newer.owners+1,errors=newer.errors.clone())
        self.store.release_device(foreign);self.assertEqual(foreign.errors.tolist(),[INVALID_LEASE])
        self.assertTrue(bool(self.store.live[newer.slots[0]]))
        self.assertEqual(int(self.store.peak_live[0]),4)

    def test_shared_reads_but_duplicate_release_is_rejected(self):
        lease,source=self.put([{'hp':3}])
        dupe=DeviceStateBatch(self.store,lease.slots.repeat(2),lease.generations.repeat(2),
            lease.owners.repeat(2),lease.active.repeat(2),lease.errors.repeat(2))
        out=self.store.gather_device(dupe,shared_reads=True)
        assert_bits(self,out,source.to_host().clone_rows([0,0]).to_torch('cpu'))
        self.store.release_device(dupe);self.assertEqual(dupe.errors.tolist(),[DUPLICATE,DUPLICATE])
        self.assertEqual(int(self.store.live.sum()),1)

    def test_shared_byte_budget_failure_is_atomic_for_all_valid_rows(self):
        full=self.schema.allocate(2).to_torch('cpu')
        for tensor in full.buffers.values():tensor.fill_(1)
        full.roots.fill_(1)
        caps=self.schema.lengths;per_state=sum(caps[k]*(4+WIDTHS[k]) for k in WIDTHS)
        max_pages=(per_state+63)//64;fixed=4*(max_pages*8+68)+max_pages*8+16
        target=fixed+(max_pages+1)*80
        store=DevicePagedSoAStore(self.schema,4,column_cells=caps,max_bytes=target,resident_bytes=target,
                                  page_bytes=64,device='cpu',allow_cpu_reference=True)
        self.addCleanup(store.close)
        one=TorchRecordSoABatch(full.schema,{k:v[:,:1].contiguous() for k,v in full.buffers.items()},full.roots[:1].contiguous())
        first=store.put_device(one,torch.tensor([9]),torch.tensor([True]));self.assertEqual(first.errors.tolist(),[0])
        before=store.pages.data.clone()
        failed=store.put_device(full,torch.tensor([1,2]),torch.tensor([True,True]))
        self.assertEqual(failed.errors.tolist(),[CAPACITY,CAPACITY]);self.assertEqual(int(store.live.sum()),1)
        self.assertTrue(torch.equal(before,store.pages.data));assert_bits(self,store.gather_device(first),one)

    def test_page_owner_duplicate_page_counts_and_index_corruption_clear_output(self):
        for corruption in ('owner','duplicate','count','index'):
            store=DevicePagedSoAStore(self.schema,4,device='cpu',allow_cpu_reference=True,page_bytes=64)
            self.addCleanup(store.close);source=pack_game_states([{'hp':3},{'hp':7}],self.schema).to_torch('cpu')
            lease=store.put_device(source,torch.tensor([0,1]),torch.tensor([True,True]));slot=int(lease.slots[0])
            first=int(store.pages.page_table[slot,0])
            if corruption=='owner':store.pages.page_owner[first]+=1
            elif corruption=='duplicate':store.pages.page_table[slot,1]=first
            elif corruption=='count':store.counts['i64'][slot]+=1
            else:store.pages.data[first,:4]=torch.tensor([self.schema.lengths['i64']],dtype=torch.int32).view(torch.uint8)
            out=store.gather_device(lease);self.assertEqual(lease.errors.tolist(),[CORRUPT_COLUMN,0],corruption)
            self.assertEqual(out.roots[0].item(),0)
            self.assertTrue(all(not bool(v[:,0].any()) for v in out.buffers.values()))
            for key in source.buffers:
                a,b=out.buffers[key][:,1],source.buffers[key][:,1]
                if key=='f64':a,b=a.view(torch.int64),b.view(torch.int64)
                self.assertTrue(torch.equal(a,b),corruption)

    def test_random_payload_cross_page_bits_zero_state_and_column_dict_order(self):
        spec=compile_schema({'Game':dict(kind='record',capacity=300,fields={'hp':'int','rate':'float'})},max_strings=32,max_string_bytes=512)
        source=random_columns(spec,'cpu')
        source=TorchRecordSoABatch(spec,dict(reversed(list(source.buffers.items()))),source.roots)
        store=DevicePagedSoAStore(spec,8,column_cells=spec.lengths,page_bytes=64,device='cpu',allow_cpu_reference=True)
        self.addCleanup(store.close)
        for _ in range(3):
            lease=store.put_device(source,torch.tensor([8,9,10]),torch.ones(3,dtype=torch.bool))
            self.assertEqual(lease.errors.tolist(),[0,0,0]);assert_bits(self,store.gather_device(lease),source)
            store.release_device(lease);self.assertEqual(lease.errors.tolist(),[0,0,0]);self.assertEqual(int(store.live.sum()),0)
        for value in source.buffers.values():value.zero_()
        empty=store.put_device(source,torch.tensor([1,2,3]),torch.ones(3,dtype=torch.bool))
        assert_bits(self,store.gather_device(empty),source)

    def test_exact_byte_plan_and_large_metadata_without_dense_root_allocation(self):
        plan=paged_arena_plan(self.schema.lengths,4,page_bytes=64)
        self.assertEqual(plan['allocated_bytes'],self.store.allocated_bytes)
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        large=paged_arena_plan(compile_execution_schema().lengths,9608)
        self.assertLessEqual(large['allocated_bytes'],1536<<20)
        self.assertEqual(large['column_cells']['i32'],16384)
        with self.assertRaises(MemoryError):paged_arena_plan(self.schema.lengths,4,max_bytes=1,page_bytes=64)
        with self.assertRaises(ValueError):DevicePagedSoAStore(self.schema,4,device='cpu')

    def test_actual_cuda_capture_random_bits_tail_reuse_and_release(self):
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        spec=compile_schema({'Game':dict(kind='record',capacity=300,fields={'hp':'int','rate':'float'})},max_strings=32,max_string_bytes=512)
        source=random_columns(spec,'cuda');store=DevicePagedSoAStore(spec,8,column_cells=spec.lengths,page_bytes=64)
        self.addCleanup(store.close);store.prepare_workspace(3)
        owners=torch.tensor([8,9,10],device='cuda');active=torch.tensor([True,False,True],device='cuda')
        output=pack_game_states([{}, {}, {}],spec).to_torch('cuda');graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lease=store.put_device(source,owners,active);store.gather_into_device(lease,output);store.release_device(lease)
        expected=source.to_host().to_torch('cpu')
        for value in expected.buffers.values():value[:,1]=0
        expected.roots[1]=0
        for _ in range(3):
            graph.replay();torch.cuda.synchronize();self.assertEqual(lease.errors.cpu().tolist(),[0,0,0])
            assert_bits(self,output,expected);self.assertEqual(int(store.live.sum().cpu()),0)
        admitted=store.put_device(source,owners,torch.ones(3,dtype=torch.bool,device='cuda'))
        torch.cuda.synchronize();self.assertEqual(admitted.errors.cpu().tolist(),[0,0,0])
        slot=int(admitted.slots[0].cpu());physical=int(store.pages.page_table[slot,0].cpu())
        store.pages.page_owner[physical]+=1
        rejected=store.gather_device(admitted)
        torch.cuda.synchronize();self.assertEqual(admitted.errors.cpu().tolist(),[CORRUPT_COLUMN,0,0])
        expected=source.to_host().to_torch('cpu')
        for value in expected.buffers.values():value[:,0]=0
        expected.roots[0]=0;assert_bits(self,rejected,expected)


if __name__=='__main__':unittest.main()
