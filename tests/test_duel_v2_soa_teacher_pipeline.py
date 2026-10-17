"""Stored SoA teacher CPU oracles; no formal game or CUDA qualification."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch,DeviceDecision,DeviceMeta
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema,pack_game_states,unpack_game_states
from app.modules.card_game.rl.batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
from app.modules.card_game.rl.batched_search.device_soa_teacher_pipeline import DeviceSoATeacherPipeline,LEASE_ERROR,METADATA_ERROR
from tests import test_duel_v2_device_graph_storage as graph_cases


class FakeSoARules:
    """Real typed/action stores, fake immutable rule outcomes and injected faults."""
    def __init__(self,test,*,failure=None,alias=None,identity_clean=False):
        torch.set_num_threads(1)
        self.t=torch;self.device=torch.device('cpu');self.rows=4
        self.schema=compile_schema({'Game':{'kind':'record','capacity':4,'fields':{
            'n':'int','actor':'int','finish_at':'int','done':'bool'}}},max_strings=4,max_string_bytes=64)
        self.store=DeviceSparseSoAStore(self.schema,32,device='cpu',allow_cpu_reference=True)
        self.action_store=graph_cases.DeviceGraphStorageTest().store(slots=32,pages=64)
        test.addCleanup(self.store.close);test.addCleanup(self.action_store.close)
        self._shared_batch=self.schema.allocate(self.rows).to_torch('cpu')
        self.initial=[dict(n=0,actor=0,finish_at=20,done=False),dict(n=0,actor=1,finish_at=1,done=False),
                      dict(n=4,actor=0,finish_at=20,done=True),dict(n=0,actor=1,finish_at=20,done=False)]
        self.roots=self.store.put_device(pack_game_states(self.initial,self.schema).to_torch('cpu'),
            torch.tensor([7,11,13,17]),torch.ones(self.rows,dtype=torch.bool))
        self.roots.active[-1]=False
        self.failure=failure;self.alias=alias;self.identity_clean=identity_clean
        self.calls=[];self.counts=dict(metadata=0,decision=0,step=0,clean=0)
        self.foreign=[]
        self.action_graph=graph_cases._graph([[{'type':'end_turn'},{'type':'concede'}]]*self.rows,'cpu')

    def _begin(self,states):
        errors=states.errors.clone()
        request=replace(states,errors=errors)
        self.store.gather_into_device(request,self._shared_batch,shared_reads=True)
        self.calls.append(('begin',states.active.clone(),errors.clone()))
        return self._shared_batch,None,errors,(states.active&(errors==0)).view(torch.uint8)

    def read(self,states):
        batch,_,errors,active=self._begin(states)
        host=batch.to_host()
        decoded=[unpack_game_states(host.clone_rows([row]))[0] if bool(active[row]) else None
                 for row in range(self.rows)]
        return decoded,errors,active.bool()

    def event(self,name,states):
        stage=self.counts[name];self.counts[name]+=1
        self.calls.append((name,states.active.clone(),states.errors.clone()))
        return stage

    def error(self,name,stage,errors):
        out=errors.clone()
        if self.failure==(name,stage):out[0]|={'metadata':107,'decision':108,'step':109,'clean':110}[name]
        return out

    def metadata_device(self,states):
        stage=self.event('metadata',states);data,errors,active=self.read(states)
        errors=self.error('metadata',stage,errors)
        actors=torch.tensor([row.get('actor',0) if row else 0 for row in data],dtype=torch.int64)
        finished=torch.tensor([bool(row.get('done')) if row else False for row in data])
        if self.alias=='bad_actor' and stage==0:actors[0]=-1
        return DeviceMeta(actors,torch.zeros(self.rows,dtype=torch.int32),
            torch.where(finished,0,-2),finished,torch.zeros(self.rows,dtype=torch.int64),errors)

    def rule_decision_device(self,states,actors):
        stage=self.event('decision',states);_,errors,active=self.read(states)
        lease=self.action_store.put_device(self.action_graph,states.owners,active)
        errors=self.error('decision',stage,errors|lease.errors)
        decision=DeviceDecision(states,self.action_store,lease.slots,lease.generations,
            torch.tensor([[1,2]]*self.rows,dtype=torch.int32),torch.full((self.rows,2),-1,dtype=torch.int64),
            torch.where(active,2,0),active[:,None].expand(-1,2).clone(),torch.empty((self.rows,0)),
            torch.empty((self.rows,2,0)),errors)
        return decision,torch.where(active,actors,0),errors

    def save(self,data,states,active,errors):
        packed=pack_game_states([row or {} for row in data],self.schema).to_torch('cpu')
        return self.store.put_device(packed,states.owners,active,errors=errors)

    def step_device(self,states,actors,decision,chosen,*,physical=False):
        if physical:raise AssertionError('Teacher must be hypothetical')
        stage=self.event('step',states);data,errors,active=self.read(states)
        for row,on in enumerate(active.tolist()):
            if on:
                if int(actors[row])!=data[row]['actor']:
                    raise AssertionError('Teacher used the fixed viewer instead of the current actor')
                data[row]=dict(data[row]);data[row]['n']+=1;data[row]['actor']=1-data[row]['actor']
                data[row]['done']=data[row]['n']>=data[row]['finish_at']
        result=self.save(data,states,active,errors)
        result=replace(result,errors=self.error('step',stage,result.errors))
        if stage==0 and self.alias=='parent_object':
            self.store.release_device(replace(result,errors=torch.zeros_like(errors)))
            return self.roots
        if stage==0 and self.alias=='duplicate':
            take=torch.zeros(self.rows,dtype=torch.bool);take[1]=True
            self.store.release_device(replace(result,active=take,errors=torch.zeros_like(errors)))
            slots=result.slots.clone();gens=result.generations.clone();owners=result.owners.clone()
            slots[1]=slots[0];gens[1]=gens[0];owners[1]=owners[0]
            return replace(result,slots=slots,generations=gens,owners=owners)
        if stage==0 and self.alias in ('parent','other_parent'):
            slots=result.slots.clone();gens=result.generations.clone();owners=result.owners.clone()
            # Release the allocated own result before returning an invalid
            # borrow, so only parent protection is under test.
            take=torch.zeros(self.rows,dtype=torch.bool);take[0]=True
            self.store.release_device(replace(result,active=take,errors=torch.zeros_like(errors)))
            index=0 if self.alias=='parent' else 1
            slots[0]=self.roots.slots[index];gens[0]=self.roots.generations[index];owners[0]=self.roots.owners[index]
            return replace(result,slots=slots,generations=gens,owners=owners)
        if stage==0 and self.alias=='stale':
            take=torch.zeros(self.rows,dtype=torch.bool);take[0]=True
            stale=replace(result,active=take,errors=torch.zeros_like(errors))
            self.store.release_device(stale)
            foreign=self.store.put_device(pack_game_states([dict(n=99,actor=1,finish_at=100,done=False)]*self.rows,
                self.schema).to_torch('cpu'),torch.tensor([99,99,99,99]),take)
            self.foreign.append(foreign)
            return result
        return result

    def clean_device(self,states):
        stage=self.event('clean',states);data,errors,active=self.read(states)
        if self.identity_clean:return replace(states,errors=self.error('clean',stage,errors))
        result=self.save(data,states,active,errors)
        return replace(result,errors=self.error('clean',stage,result.errors))


class SoATeacherPipelineTests(unittest.TestCase):
    def make(self,**options):
        rules=FakeSoARules(self,**options)
        teacher=NS(rules=rules,rule_decision_device=rules.rule_decision_device)
        pipeline=object.__new__(DeviceSoATeacherPipeline);pipeline._prepare(rules,teacher)
        return pipeline,rules,torch.tensor([0,0,1,1],dtype=torch.int64)

    def capture(self,pipeline,rules,viewers):
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
            return pipeline.capture_device(rules.roots,viewers)

    def parents_unchanged(self,rules):
        roots=replace(rules.roots,active=torch.ones(4,dtype=torch.bool),errors=torch.zeros(4,dtype=torch.int32))
        self.assertEqual(unpack_game_states(rules.store.gather_device(roots).to_host()),rules.initial)
        self.assertTrue(bool(rules.store.live[rules.roots.slots].all()))

    def test_three_stored_steps_terminal_masks_parent_clone_and_caller_owned_result(self):
        for identity in (False,True):
            pipeline,rules,viewers=self.make(identity_clean=identity)
            output=self.capture(pipeline,rules,viewers)
            self.assertEqual(output.errors.tolist(),[0]*4)
            self.assertEqual(output.active.tolist(),[True,True,True,False])
            host=rules.store.gather_device(output).to_host()
            decoded=[unpack_game_states(host.clone_rows([row]))[0] for row in range(3)]
            self.assertEqual([row['n'] for row in decoded],[3,1,4])
            self.assertEqual(output.owners.tolist(),rules.roots.owners.tolist())
            self.assertFalse(bool((output.slots[:3]==rules.roots.slots[:3]).any()))
            self.parents_unchanged(rules)
            self.assertEqual(int(rules.store.live.sum()),7);self.assertEqual(int(rules.action_store.live.sum()),0)
            decisions=[row[1].tolist() for row in rules.calls if row[0]=='decision']
            self.assertEqual(decisions,[[True,True,False,False],[True,False,False,False],[True,False,False,False]])
            # One initial clone plus distinct metadata/decision/step/clean
            # gather/reset sites at every stage; no accumulated fused call.
            self.assertEqual(sum(row[0]=='begin' for row in rules.calls),13)
            pipeline.close()
            self.assertTrue(bool(rules.store.live[output.slots[:3]].all()))
            rules.store.release_device(output)
            self.assertEqual(int(rules.store.live.sum()),4)

    def test_each_stage_error_preserves_raw_code_and_drains_partial_outputs(self):
        for name in ('metadata','decision','step','clean'):
            for stage in (0,1,2):
                pipeline,rules,viewers=self.make(failure=(name,stage))
                output=self.capture(pipeline,rules,viewers)
                self.assertEqual(int(output.errors[0]),{'metadata':107,'decision':108,'step':109,'clean':110}[name],(name,stage))
                self.assertEqual(output.slots[0].item(),-1);self.assertFalse(bool(output.active[0]))
                self.assertEqual(int(rules.store.live.sum()),6,(name,stage))
                self.assertEqual(int(rules.action_store.live.sum()),0,(name,stage))
                self.parents_unchanged(rules)
                rules.store.release_device(output);self.assertEqual(int(rules.store.live.sum()),4)

    def test_parent_cross_parent_and_stale_generation_aliases_never_reclaim_external_state(self):
        for alias in ('parent','other_parent','stale','bad_actor'):
            pipeline,rules,viewers=self.make(alias=alias)
            output=self.capture(pipeline,rules,viewers)
            self.assertEqual(int(output.errors[0]),METADATA_ERROR if alias=='bad_actor' else LEASE_ERROR,alias)
            self.assertEqual(output.slots[0].item(),-1)
            self.parents_unchanged(rules)
            self.assertEqual(int(rules.action_store.live.sum()),0)
            if alias=='stale':
                foreign=rules.foreign[0]
                self.assertTrue(bool(rules.store.live[foreign.slots[0]]))
                self.assertEqual(int(rules.store.owner[foreign.slots[0]]),99)
                self.assertEqual(int(rules.store.live.sum()),7)
                rules.store.release_device(foreign)
            else:self.assertEqual(int(rules.store.live.sum()),6,alias)
            rules.store.release_device(output);self.assertEqual(int(rules.store.live.sum()),4)

    def test_input_error_capture_horizon_and_default_no_fake_gpu_qualification(self):
        pipeline,rules,viewers=self.make()
        for horizon in (True,0,2,4,3.):
            with self.assertRaises(ValueError):pipeline.capture_device(rules.roots,viewers,horizon=horizon)
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'capture'):pipeline.capture_device(rules.roots,viewers)
        rules.roots.errors[0]=41
        output=self.capture(pipeline,rules,viewers)
        self.assertEqual(output.errors.tolist(),[41,0,0,0]);self.assertEqual(output.slots[0].item(),-1)
        self.assertEqual(int(rules.store.live.sum()),6)
        with self.assertRaisesRegex(RuntimeError,'already frozen'):pipeline.capture_device(rules.roots,viewers)
        pipeline.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):pipeline.capture_device(rules.roots,viewers)
        self.assertFalse(DeviceSoATeacherPipeline.supports_full_duel_rules)
        self.assertFalse(DeviceSoATeacherPipeline.host_rule_callbacks)
        with self.assertRaises(RuntimeError):DeviceSoATeacherPipeline(rules,NS(rules=rules,rule_decision_device=rules.rule_decision_device))

    def test_borrowed_parent_object_and_duplicate_owned_handles_are_protected_and_deduplicated(self):
        for alias in ('parent_object','duplicate'):
            pipeline,rules,viewers=self.make(alias=alias)
            fields_before={name:getattr(rules.roots,name).clone() for name in
                           ('slots','generations','owners','active','errors')}
            output=self.capture(pipeline,rules,viewers)
            self.assertEqual(output.errors.tolist(),[LEASE_ERROR,LEASE_ERROR,0,0],alias)
            self.assertEqual(output.slots[:2].tolist(),[-1,-1])
            for name,value in fields_before.items():torch.testing.assert_close(getattr(rules.roots,name),value)
            self.parents_unchanged(rules)
            self.assertEqual(int(rules.store.live.sum()),5,alias)
            self.assertEqual(int(rules.action_store.live.sum()),0)
            rules.store.release_device(output);self.assertEqual(int(rules.store.live.sum()),4)

    def test_clone_capacity_error_preserves_existing_external_allocations(self):
        from app.modules.card_game.rl.batched_duel.device_soa_storage import CAPACITY
        pipeline,rules,viewers=self.make()
        data=[dict(n=99,actor=0,finish_at=100,done=False)]*28
        extra=rules.store.put_device(pack_game_states(data,rules.schema).to_torch('cpu'),
                                    torch.full((28,),99,dtype=torch.int64),torch.ones(28,dtype=torch.bool))
        self.assertEqual(int(rules.store.live.sum()),32)
        output=self.capture(pipeline,rules,viewers)
        self.assertEqual(output.errors.tolist(),[CAPACITY,CAPACITY,CAPACITY,0])
        self.assertEqual(output.slots.tolist(),[-1]*4);self.assertFalse(bool(output.active.any()))
        self.assertEqual(int(rules.store.live.sum()),32);self.parents_unchanged(rules)
        self.assertEqual(int(rules.action_store.live.sum()),0)
        rules.store.release_device(extra);self.assertEqual(int(rules.store.live.sum()),4)

    def test_stale_or_foreign_parent_handle_is_rejected_without_mutating_parent_metadata(self):
        from app.modules.card_game.rl.batched_duel.device_soa_storage import INVALID_LEASE
        for field in ('generations','owners'):
            pipeline,rules,viewers=self.make()
            wrong=getattr(rules.roots,field).clone();wrong[0]+=1
            supplied=replace(rules.roots,**{field:wrong},errors=torch.zeros(4,dtype=torch.int32))
            with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
                output=pipeline.teacher_continue_device(supplied,viewers)
            self.assertEqual(int(output.errors[0]),INVALID_LEASE)
            self.assertEqual(supplied.errors.tolist(),[0]*4)
            self.assertEqual(output.slots[0].item(),-1)
            self.parents_unchanged(rules)
            self.assertEqual(int(rules.store.live.sum()),6)
            self.assertEqual(int(rules.action_store.live.sum()),0)
            rules.store.release_device(output);self.assertEqual(int(rules.store.live.sum()),4)


if __name__=='__main__':unittest.main()
