"""Explicit CPU protocol oracle; never evidence of a qualified GPU driver."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
from app.modules.card_game.rl.batched_duel.schema import GraphLayout

from app.modules.card_game.rl.batched_duel.device_rule_protocol import (
    DeviceDecision,DeviceMeta,DeviceStateBatch,DeviceValueObservation,
)
from app.modules.card_game.rl.batched_search.device_pure_value_tape import (
    DevicePureValueTape,CAPACITY_ERROR,FINISH_ERROR,GEN_ERROR,MASK_ERROR,
    NONFINITE_ERROR,OWNER_ERROR,PURE_ERROR,STEP_ERROR,
)


class CheckedPool:
    def __init__(self):
        self.live=torch.ones(32,dtype=torch.bool)
        self.generation=torch.ones(32,dtype=torch.int64)
        self.owner=torch.arange(32,dtype=torch.int64)%2+40

    def validate(self,states,*,shared_reads=False):
        slots=states.slots.clamp(0,31)
        valid=(states.slots>=0)&(states.slots<32)&self.live[slots]
        valid&=(states.generations==self.generation[slots])&(states.owners==self.owner[slots])
        states.errors.bitwise_or_((states.active&~valid).to(torch.int32)*1024)
        return slots,states.active&valid&(states.errors==0)


class Rules:
    def __init__(self):
        self.actors=torch.tensor([0,1],dtype=torch.int64)
        self.versions=torch.tensor([7,8],dtype=torch.int64)
        self.finished=torch.zeros(2,dtype=torch.bool)
        self.winners=torch.full((2,),-2,dtype=torch.int64)
        self.errors=torch.zeros(2,dtype=torch.int32)
        self.other_errors=torch.zeros(2,dtype=torch.int32)
        self.bad_context=False;self.bad_obs=False;self.value_calls=0

    def observations(self,viewers):
        return (torch.arange(3,dtype=torch.float32)[None,:]+viewers[:,None]*10
                +self.versions[:,None].to(torch.float32)*100).contiguous()

    def metadata_device(self,roots):
        return DeviceMeta(self.actors,torch.zeros(2,dtype=torch.int32),self.winners,
            self.finished,self.versions,self.errors)

    def value_observation_device(self,roots,viewers):
        self.value_calls+=1
        x=self.observations(viewers)
        if self.bad_obs:x[0,0]=float('nan')
        ctx=(viewers==self.actors).to(torch.float32)[:,None].contiguous()
        if self.bad_context:ctx[0,0]=1
        return DeviceValueObservation(x,ctx,self.finished,torch.zeros(2,dtype=torch.float64),self.other_errors)


class PureValueTapeTests(unittest.TestCase):
    def fixture(self,*,max_records=8,train=None):
        rules=Rules();pool=CheckedPool()
        roots=DeviceStateBatch(pool,torch.tensor([0,1]),torch.ones(2,dtype=torch.int64),
            torch.tensor([40,41]),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.int32))
        tape=DevicePureValueTape(rules,rows=2,max_records=max_records,max_bytes=1<<20,
            observation_dim=3,max_actions=4,device='cpu',allow_cpu_reference=True,allow_synthetic=True)
        mask=torch.tensor([[True,False],[False,True]]) if train is None else train
        zero=torch.zeros((2,2),dtype=torch.int64)
        err=tape.begin_device(roots,zero,zero,torch.tensor([123,456]),mask,roots.active)
        self.assertEqual(err.tolist(),[0,0])
        return tape,rules,roots

    def pure(self,rules,roots):
        if not hasattr(rules,'action_store'):
            rules.action_store=CompactDeviceGraphStore(slots=4,node_pages=16,edge_pages=16,
                payload_pages=16,node_page_bytes=128,edge_page_bytes=128,payload_page_bytes=128,
                max_node_pages_per_slot=8,max_edge_pages_per_slot=8,max_payload_pages_per_slot=8,
                device='cpu',allow_cpu_reference=True)
            packed=pack_states([{'action':0},{'action':1}],GraphLayout(2,16,16,128))
            graph=DeviceGraph(packed.layout,*(torch.tensor(np.array(getattr(packed,name),copy=True))
                for name in ('nodes','edges','payload','counts','roots')))
            rules.action_lease=rules.action_store.put_device(graph,roots.owners,roots.active)
            rules.action_ref=graph.roots[:,None].expand(2,4).contiguous()
        counts=torch.tensor([2,3],dtype=torch.int64)
        valid=torch.arange(4)[None,:]<counts[:,None]
        decision=DeviceDecision(roots,rules.action_store,rules.action_lease.slots.clone(),
            rules.action_lease.generations.clone(),rules.action_ref.clone(),
            torch.zeros((2,4),dtype=torch.int64),counts,valid,
            rules.observations(rules.actors),None,torch.zeros(2,dtype=torch.int32))
        return SimpleNamespace(decision=decision,actions=torch.tensor([0,1]),
            requested=torch.zeros(2,dtype=torch.int64),completed=torch.zeros(2,dtype=torch.int64),
            visits=torch.zeros((2,4),dtype=torch.int32),complete=torch.ones(2,dtype=torch.bool),
            errors=torch.zeros(2,dtype=torch.int32))

    def append(self,tape,rules,roots,result=None,steps=None):
        result=self.pure(rules,roots) if result is None else result
        return tape.append_device(roots,rules.actors,
            torch.zeros(2,dtype=torch.int64) if steps is None else steps,result,roots.active)

    def terminal(self,rules,roots):
        rules.versions+=1;rules.finished.fill_(True);rules.winners.copy_(torch.tensor([0,1]))
        return replace(roots,slots=roots.slots+2,errors=roots.errors.clone())

    def finish(self,tape,rules,roots,*,complete=None,steps=None,errors=None):
        return tape.finish_device(roots,rules.winners,
            torch.ones(2,dtype=torch.int64) if steps is None else steps,
            torch.ones(2,dtype=torch.bool) if complete is None else complete,
            torch.zeros(2,dtype=torch.int32) if errors is None else errors)

    def test_two_viewers_correct_sign_context_and_frozen_learning_mask(self):
        tape,rules,roots=self.fixture(train=torch.tensor([[False,True],[True,False]]))
        pure=self.pure(rules,roots)
        before=pure.visits.clone()
        self.assertEqual(self.append(tape,rules,roots,pure).tolist(),[0,0])
        self.assertEqual(rules.value_calls,1,'Reuse actor observation; only query other viewer')
        self.assertTrue(torch.equal(pure.visits,before))
        x=rules.observations(torch.tensor([1,0])).numpy().copy()
        terminal=self.terminal(rules,roots)
        self.assertEqual(self.finish(tape,rules,terminal).tolist(),[0,0])
        rows=tape.value_rows_after_batch()
        self.assertEqual(len(rows),2)
        for i,row in enumerate(rows):
            self.assertEqual(row['actor'],1-i);self.assertEqual(row['value_actor'],0.)
            self.assertEqual(row['z'],-1.);self.assertTrue(row['learning_eligible'])
            np.testing.assert_array_equal(row['x'],x[i])
            self.assertEqual(row['job_seed'],[123,456][i]);self.assertEqual(row['version'],[7,8][i])
            self.assertEqual(row['owner'],[40,41][i]);self.assertEqual(row['step'],0)
            for forbidden in ('pi','policy','visits','requested','completed','selected','c'):
                self.assertNotIn(forbidden,row)
        audit=tape.value_rows_after_batch(include_frozen=True)
        self.assertEqual(len(audit),4)
        self.assertEqual([r['z'] for r in audit],[1.,-1.,-1.,1.])
        self.assertEqual([r['value_actor'] for r in audit],[1.,0.,0.,1.])
        audit[0]['x'][0]=999
        self.assertNotEqual(tape.value_rows_after_batch(include_frozen=True)[0]['x'][0],999)
        result=tape.result_device()
        self.assertEqual(result.record_count.item(),2)
        self.assertEqual(result.complete.tolist(),[True,True])
        self.assertFalse(tape.supports_full_duel_rules);self.assertFalse(tape.gpu_qualified)

    def test_whole_batch_export_blocks_unfinished_truncated_error_and_duplicate_finish(self):
        tape,rules,roots=self.fixture();self.append(tape,rules,roots)
        with self.assertRaises(RuntimeError):tape.value_rows_after_batch()
        terminal=self.terminal(rules,roots)
        first=replace(terminal,active=torch.tensor([True,False]))
        self.finish(tape,rules,first)
        with self.assertRaises(RuntimeError):tape.value_rows_after_batch(include_frozen=True)
        self.finish(tape,rules,replace(terminal,active=torch.tensor([False,True])))
        self.assertEqual(len(tape.value_rows_after_batch()),2)
        self.assertTrue((self.finish(tape,rules,terminal)&FINISH_ERROR).all())
        with self.assertRaises(RuntimeError):tape.value_rows_after_batch()
        for mode in ('truncated','driver_error','fake_terminal'):
            tape,rules,roots=self.fixture();self.append(tape,rules,roots)
            terminal=self.terminal(rules,roots)
            if mode=='fake_terminal':rules.finished[0]=False
            complete=torch.tensor([mode!='truncated',True])
            incoming=torch.tensor([9 if mode=='driver_error' else 0,0],dtype=torch.int32)
            self.finish(tape,rules,terminal,complete=complete,errors=incoming)
            with self.assertRaises(RuntimeError):tape.value_rows_after_batch(include_frozen=True)

    def test_zero_simulations_required_and_policy_cannot_disguise_search(self):
        for field in ('requested','completed','visits','complete'):
            tape,rules,roots=self.fixture();pure=self.pure(rules,roots)
            if field=='complete':pure.complete[0]=False
            elif field=='visits':pure.visits[0,0]=1
            else:getattr(pure,field)[0]=1
            errors=self.append(tape,rules,roots,pure)
            self.assertTrue(errors[0].item()&PURE_ERROR)
            self.assertEqual(errors[1].item(),0)
            self.assertEqual(tape.result_device().record_count.item(),1)

    def test_capacity_atomicity_and_owner_generation_step_finite_masks(self):
        tape,rules,roots=self.fixture(max_records=1)
        self.assertEqual(self.append(tape,rules,roots).tolist(),[CAPACITY_ERROR]*2)
        self.assertEqual(tape.result_device().record_count.item(),0)
        for mode,code in (('duplicate_step',STEP_ERROR),('version',STEP_ERROR),
                ('stale',OWNER_ERROR),('wrong_decision_generation',GEN_ERROR),
                ('owner',OWNER_ERROR),('nan_actor',NONFINITE_ERROR),('nan_other',NONFINITE_ERROR),
                ('context',GEN_ERROR),('count',MASK_ERROR),('prefix',MASK_ERROR),('choice',MASK_ERROR)):
            tape,rules,roots=self.fixture();pure=self.pure(rules,roots)
            if mode=='duplicate_step':self.append(tape,rules,roots,pure)
            if mode=='version':rules.versions[0]+=1
            if mode=='stale':roots.generations[0]+=1
            if mode=='owner':roots.owners[0]=41
            if mode=='wrong_decision_generation':
                pure.decision=replace(pure.decision,states=replace(roots,generations=roots.generations+1))
            if mode=='nan_actor':pure.decision.observations[0,0]=float('inf')
            if mode=='nan_other':rules.bad_obs=True
            if mode=='context':rules.bad_context=True
            if mode=='count':pure.decision.counts[0]=0
            if mode=='prefix':pure.decision.valid[0,3]=True
            if mode=='choice':pure.actions[0]=3
            errors=self.append(tape,rules,roots,pure)
            self.assertTrue(errors[0].item()&code,(mode,errors.tolist()))
        tape,rules,roots=self.fixture();self.append(tape,rules,roots)
        terminal=self.terminal(rules,roots);self.finish(tape,rules,terminal)
        self.append(tape,rules,terminal)
        self.assertFalse(tape.result_device().learning_eligible.any())

    def test_checked_begin_duplicates_budget_and_tensor_protocol(self):
        tape,rules,roots=self.fixture()
        ids=torch.zeros((2,2),dtype=torch.int64);seeds=torch.tensor([1,2]);train=torch.ones((2,2),dtype=torch.bool)
        duplicates=replace(roots,owners=torch.tensor([40,40]))
        errors=tape.begin_device(duplicates,ids,ids,seeds,train,roots.active)
        self.assertTrue((errors&OWNER_ERROR).all())
        for synthetic,reference in ((False,False),(True,False),(False,True)):
            with self.assertRaises(ValueError):DevicePureValueTape(rules,rows=2,max_records=4,max_bytes=1<<20,
                observation_dim=3,device='cpu',allow_synthetic=synthetic,allow_cpu_reference=reference)
        budget=tape.allocated_bytes
        same=DevicePureValueTape(rules,rows=2,max_records=8,max_bytes=budget,observation_dim=3,
            max_actions=4,device='cpu',allow_synthetic=True,allow_cpu_reference=True)
        self.assertEqual(same.allocated_bytes,budget)
        with self.assertRaises(MemoryError):DevicePureValueTape(rules,rows=2,max_records=8,max_bytes=budget-1,
            observation_dim=3,max_actions=4,device='cpu',allow_synthetic=True,allow_cpu_reference=True)
        with self.assertRaises(RuntimeError):same.result_device()
        with self.assertRaises(RuntimeError):same.value_rows_after_batch()
        tape,rules,roots=self.fixture()
        with self.assertRaises(ValueError):tape.append_device(roots,rules.actors.to(torch.int32),
            torch.zeros(2,dtype=torch.int64),self.pure(rules,roots),roots.active)
        pure=self.pure(rules,roots);pure.visits=pure.visits.to(torch.float32)
        with self.assertRaises(ValueError):self.append(tape,rules,roots,pure)
        with self.assertRaises(ValueError):self.append(tape,rules,replace(roots,pool=CheckedPool()))
        with self.assertRaises(TypeError):tape.value_rows_after_batch(include_frozen=1)

    def test_action_lease_registry_and_selected_ref_reject_entire_export(self):
        for mode in ('stale','foreign_owner','invalid_slot','dead','bad_ref','prior_error'):
            tape,rules,roots=self.fixture();pure=self.pure(rules,roots)
            slot=int(pure.decision.action_slots[0])
            if mode=='stale':pure.decision.action_generations[0]+=1
            if mode=='foreign_owner':rules.action_store.owner[slot]=999
            if mode=='invalid_slot':pure.decision.action_slots[0]=99
            if mode=='dead':rules.action_store.live[slot]=0
            if mode=='bad_ref':pure.decision.action_refs[0,0]=999
            if mode=='prior_error':pure.decision.errors[0]=2048
            errors=self.append(tape,rules,roots,pure)
            self.assertNotEqual(errors[0].item(),0,mode)
            self.assertEqual(errors[1].item(),0,mode)
            self.assertEqual(tape.result_device().record_count.item(),1)
            self.finish(tape,rules,self.terminal(rules,roots))
            with self.assertRaises(RuntimeError):tape.value_rows_after_batch(include_frozen=True)
        tape,rules,roots=self.fixture();pure=self.pure(rules,roots)
        foreign=replace(pure.decision,action_pool=object())
        pure.decision=foreign
        with self.assertRaises(ValueError):self.append(tape,rules,roots,pure)
        _,other_rules,other_roots=self.fixture()
        other=self.pure(other_rules,other_roots)
        pure.decision=replace(foreign,action_pool=other.decision.action_pool)
        with self.assertRaises(ValueError):self.append(tape,rules,roots,pure)

    def test_seed_and_bank_id_bounds(self):
        for field,bad_value in (('seed',-1),('policy',-1),('aux',-2)):
            tape,rules,roots=self.fixture()
            ids=torch.zeros((2,2),dtype=torch.int64);aux=ids.clone();seed=torch.tensor([1,2])
            {'seed':seed,'policy':ids,'aux':aux}[field].view(-1)[0]=bad_value
            errors=tape.begin_device(roots,ids,aux,seed,torch.ones((2,2),dtype=torch.bool),roots.active)
            self.assertTrue(errors[0].item()&GEN_ERROR)
            with self.assertRaises(RuntimeError):tape.value_rows_after_batch()
        aux.fill_(-1)
        self.assertEqual(tape.begin_device(roots,ids*0,aux,torch.tensor([0,2]),
            torch.ones((2,2),dtype=torch.bool),roots.active).tolist(),[0,0])


if __name__=='__main__':unittest.main()
