"""CPU fake-lane/lease tests; no actual CUDA, game or capacity qualification."""
from dataclasses import fields,replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch as t

from app.modules.card_game.rl.batched_duel.device_rule_protocol import (
    DeviceStateBatch,DeviceDecision,DeviceMeta,DeviceValueObservation)
from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules,_empty_tokens
from app.modules.card_game.rl.batched_search.device_tiered_soa_rules import DeviceTieredSoARules,OUTPUT_LEASE_ERROR
from app.modules.card_game.rl.batched_search import compact_device_candidates as compact
_COMPACT_CLASS=compact.CompactDeviceCandidates


class Registry:
    def __init__(self,capacity=128):
        self.live=t.zeros(capacity,dtype=t.bool);self.generation=t.zeros(capacity,dtype=t.int64)
        self.owner=t.full((capacity,),-1,dtype=t.int64);self.data=t.zeros(capacity,dtype=t.int64)
        self.workspace_bytes=128;self.closed=False;self.released=[]
    def allocate(self,owners,active,values):
        free=(~self.live).nonzero().reshape(-1);slots=t.full_like(owners,-1);gens=t.zeros_like(owners)
        errors=t.zeros(len(owners),dtype=t.int32)
        for row in active.nonzero().reshape(-1).tolist():
            slot=int(free[0]);free=free[1:]
            self.live[slot]=True;self.generation[slot]+=1;self.owner[slot]=owners[row];self.data[slot]=values[row]
            slots[row]=slot;gens[row]=self.generation[slot]
        return DeviceStateBatch(self,slots,gens,owners.clone(),active.clone(),errors)
    def release_device(self,states):
        for row in states.active.nonzero().reshape(-1).tolist():
            slot=int(states.slots[row]);gen=int(states.generations[row]);owner=int(states.owners[row])
            if not 0<=slot<len(self.live) or not self.live[slot] or self.generation[slot]!=gen or self.owner[slot]!=owner:
                states.errors[row]|=1
            else:self.live[slot]=False;self.released.append((slot,gen,owner))
    def close(self):self.closed=True


class FakeQueue:
    def __init__(self,rows,lanes,guard=0):
        self.rows,self.lanes,self.guard=rows,lanes,guard
        self.loop=NS(errors=t.zeros(1,dtype=t.int32));self.valid=None;self.ids=None
    def capture_dispatch(self,states,operation):
        self.loop.errors.zero_()
        if bool(states.active.any()):
            for start in range(0,self.rows,self.lanes):
                raw=t.arange(start,start+self.lanes);self.valid=(raw<self.rows).to(t.uint8)
                valid=self.valid.bool();self.ids=t.where(valid,raw,0)
                local=DeviceStateBatch(states.pool,t.where(valid,states.slots[self.ids],-1),
                    t.where(valid,states.generations[self.ids],0),t.where(valid,states.owners[self.ids],0),
                    states.active[self.ids]&valid,t.where(valid,states.errors[self.ids],0))
                operation(local,self.ids,self.valid)
        self.loop.errors.fill_(self.guard)
    def scatter_tensor_device(self,local,output):
        if not local.numel():raise ValueError('Empty payload cannot scatter')
        valid=self.valid.bool();output[self.ids[valid]]=local[valid]
    def scatter_states_device(self,local,output):
        for name in ('slots','generations','owners','active','errors'):
            self.scatter_tensor_device(getattr(local,name),getattr(output,name))


class FakeInner:
    def __init__(self,outer):self.outer=outer;self._raw_capture_hook=None
    @property
    def workspace_bytes(self):return 1000+self.outer.store.workspace_bytes
    def _invoke(self,kind,states):
        statuses=t.zeros((len(states.slots),4),dtype=t.int32)
        for row in range(len(states.slots)):
            if states.active[row] and states.errors[row]==0:
                statuses[row,0]=self.outer.vm_errors.get((kind,int(states.owners[row])),0)
        return NS(statuses=statuses)
    def result(self,kind,states,*args):
        call=self._invoke(kind,states)
        errors=states.errors|call.statuses[:,0]
        for row in range(len(states.slots)):
            if states.active[row] and states.errors[row]==0:
                errors[row]|=self.outer.other_errors.get((kind,int(states.owners[row])),0)
        self.outer.calls.append((kind,states,*[arg.clone() if isinstance(arg,t.Tensor) else arg for arg in args]))
        return errors,states.active&(errors==0)
    def metadata_device(self,states):
        errors,on=self.result('metadata',states);n=len(states.slots)
        return DeviceMeta(states.owners%2,t.ones(n,dtype=t.int32),t.full((n,),-2),
            t.zeros(n,dtype=t.bool),self.outer.store.data[states.slots.clamp_min(0)].clone(),errors)
    def value_observation_device(self,states,viewers):
        errors,on=self.result('value',states,viewers);n=len(states.slots)
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import OBS_DIM
        obs=t.zeros((n,OBS_DIM));obs[:,0]=viewers;obs[:,1]=states.owners
        return DeviceValueObservation(obs,viewers.float()[:,None],t.zeros(n,dtype=t.bool),t.zeros(n,dtype=t.float64),errors)
    def transform(self,kind,states,*args):
        errors,on=self.result(kind,states,*args)
        partial=t.tensor([self.outer.partial==(kind,int(owner)) for owner in states.owners])&states.active
        allowed=on if self.outer.alias==kind else on|partial
        values=self.outer.store.data[states.slots.clamp_min(0)]+100
        output=self.outer.store.allocate(states.owners,allowed,values)
        if self.outer.alias==kind:
            output=replace(output,slots=t.where(partial,states.slots,output.slots),
                           generations=t.where(partial,states.generations,output.generations))
        return replace(output,errors=errors)
    def clean_device(self,states):return self.transform('clean',states)
    def sample_world_device(self,states,viewers,seeds):return self.transform('sample',states,viewers,seeds)
    def teacher_continue_device(self,states,viewers,*,horizon=3):
        if horizon!=3:raise AssertionError('horizon changed')
        return self.transform('teacher',states,viewers)
    def decision(self,kind,states,viewers):
        errors,on=self.result(kind,states,viewers);n=len(states.slots);m=self.outer.max_actions
        partial=t.tensor([self.outer.partial==(kind,int(owner)) for owner in states.owners])&states.active
        lease=self.outer.action_store.allocate(states.owners,on|partial,states.owners)
        counts=t.where(on,2+states.owners.remainder(2),0);valid=t.arange(m)[None,:]<counts[:,None]
        refs=(states.owners[:,None]*10+t.tensor([[2,0,1]])).to(t.int32)
        if kind=='rule_decision':
            obs=t.empty((n,0));candidates=t.empty((n,m,0))
        else:
            from app.modules.card_game.rl.batched_duel.gpu_seven_observation import OBS_DIM
            obs=states.owners.float()[:,None].expand(-1,OBS_DIM).clone();obs[:,0]=viewers
            tokens=_empty_tokens(t,n,m,'cpu')
            tokens.counts.copy_(counts);tokens.mask.copy_(valid)
            tokens.int_tokens.copy_(states.owners[:,None,None].expand(n,m,7))
            if self.outer.overflow:tokens.int_tokens[:,0,0]=2**35
            candidates=compact.CompactDeviceCandidates(tokens,scratch_rows=2,device='cpu',template_table=self.outer.templates)
        decision=DeviceDecision(states,self.outer.action_store,lease.slots,lease.generations,refs,
            t.full((n,m),-1,dtype=t.int64),counts,valid,obs,candidates,errors)
        return decision
    def decision_device(self,states,viewers):return self.decision('decision',states,viewers)
    def rule_decision_device(self,states,viewers):
        result=self.decision('rule_decision',states,viewers)
        return result,t.where(states.active&(result.errors==0),viewers,-1),result.errors
    def step_device(self,states,actors,decision,chosen,*,physical=False):
        self.outer.steps.append((states,actors.clone(),decision,chosen.clone(),physical))
        if physical and self._raw_capture_hook is not None:
            self._raw_capture_hook(None,states.owners,actors,states.active&(states.errors==0))
        return self.transform('step',states,actors,chosen)
    def step_clean_device(self,states,actors,decision,chosen,*,physical=False):
        self.outer.steps.append((states,actors.clone(),decision,chosen.clone(),physical))
        return self.transform('step_clean',states,actors,chosen)


class FakeTiled(DeviceTiledSoARules):
    def __init__(self,executor,store,actions,*,rows,lanes,templates):
        self.executor,self.store,self.action_store=executor,store,actions
        self.rows,self.work_rows,self.inference_rows,self.max_actions=rows,lanes,2,3
        self.t,self.device=t,t.device('cpu');self.templates=templates
        self.candidate_integer_dtype=t.int32;self.require_teacher=True;self.staged_teacher=True;self.teacher_horizon=3
        self.layout=NS(rows=lanes);self.action_layout=NS(rows=rows)
        self.raw_tape=None;self.physical_loop=None;self.closed=False;self._keepalive=[];self._init_resources=[]
        self.calls=[];self.steps=[];self.raw_writes=0;self.initializations=0;self.closes=0
        self.vm_errors={};self.other_errors={};self.partial=None;self.alias=None;self.overflow=False
        self.inner=FakeInner(self)
        names=('metadata','decision','value','sample','clean','teacher','rule_decision','step','step_clean')
        self._sites={name:[FakeQueue(rows,lanes) for _ in range(3)] for name in names}
        self._site_index={name:0 for name in names};self._queues=[q for queues in self._sites.values() for q in queues]
    @property
    def workspace_bytes(self):return self.inner.workspace_bytes+100
    def close(self):
        if not self.closed:self.closes+=1;self.closed=True
    def initialize_device(self,parameters,*,owners=None):self.initializations+=1;return parameters,owners
    def prepare_fused_steps(self):pass


class TieredSoARuleTests(unittest.TestCase):
    def make(self):
        from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import get_card_template_table
        t.set_num_threads(1);rows=7;store=Registry();actions=Registry();executor=NS(execute_only=False)
        owners=t.arange(11,11+rows,dtype=t.int64)
        roots=store.allocate(owners,t.ones(rows,dtype=t.bool),t.arange(rows,dtype=t.int64)+3)
        roots.active[1]=False;roots.errors[2]=77
        templates=get_card_template_table('cpu')
        fast=FakeTiled(executor,store,actions,rows=rows,lanes=3,templates=templates)
        slow=FakeTiled(executor,store,actions,rows=rows,lanes=2,templates=templates)
        def record(graph,ids,who,steps,versions,actors,active):
            slow.raw_writes+=int(active.sum());return t.zeros_like(active,dtype=t.int32)
        slow.raw_tape=NS(append_device=record)
        rules=object.__new__(DeviceTieredSoARules);rules._prepare(fast,slow)
        actual=_COMPACT_CLASS;allocations=[]
        def cpu_candidates(tokens,**kw):
            allocations.append(tokens.int_tokens.shape[0])
            kw.setdefault('allow_cpu_reference',True)
            return actual(tokens,**kw)
        context=patch.object(compact,'CompactDeviceCandidates',side_effect=cpu_candidates)
        context.start();self.addCleanup(context.stop)
        capture=patch.object(t.cuda,'is_current_stream_capturing',return_value=True)
        capture.start();self.addCleanup(capture.stop)
        return rules,roots,t.arange(rows,dtype=t.int64)%2,allocations

    def test_only_proven_vm_oom_rows_retry_with_tail_and_original_errors_preserved(self):
        rules,roots,viewers,_=self.make()
        rules.fast.vm_errors[('metadata',11)]=1;rules.fast.vm_errors[('metadata',17)]=1
        rules.fast.other_errors[('metadata',14)]=9;rules.fast.other_errors[('metadata',15)]=1
        original_invoke=rules.fast.inner._invoke;original_site=rules.fast._site
        before=roots.errors.clone();data=rules.store.data[roots.slots].clone()
        out=rules.metadata_device(roots)
        self.assertEqual(out.errors.tolist(),[0,0,77,9,1,0,0])
        self.assertEqual(rules.fallback_counts[0].tolist(),[1,0,0,0,0,0,1])
        self.assertEqual(out.actors[[0,3,5,6]].tolist(),[1,0,0,1])
        self.assertEqual(rules.fast.inner._invoke,original_invoke);self.assertEqual(rules.fast._site,original_site)
        t.testing.assert_close(roots.errors,before);t.testing.assert_close(rules.store.data[roots.slots],data)
        self.assertEqual([int(call[1].active.sum()) for call in rules.slow.calls],[1,0,0,1])

    def test_guard_and_invalid_registry_are_not_misclassified_as_capacity(self):
        rules,roots,_,_=self.make();rules.fast.vm_errors[('metadata',11)]=1
        rules.fast._sites['metadata'][0].guard=1
        out=rules.metadata_device(roots)
        self.assertEqual(int(out.errors[0]),1);self.assertEqual(int(rules.fallback_counts.sum()),0)
        self.assertFalse(rules.slow.calls)
        rules,roots,_,_=self.make();rules.fast.vm_errors[('metadata',11)]=1
        roots.generations[0]+=1
        out=rules.metadata_device(roots)
        self.assertEqual(int(out.errors[0]),1);self.assertEqual(int(rules.fallback_counts.sum()),0)

    def test_slow_oom_is_reported_without_another_retry(self):
        rules,roots,_,_=self.make();rules.fast.vm_errors[('metadata',11)]=1;rules.slow.vm_errors[('metadata',11)]=1
        out=rules.metadata_device(roots)
        self.assertEqual(int(out.errors[0]),1);self.assertEqual(int(rules.fallback_counts.sum()),1)

    def test_compact_candidates_reuse_fast_full_batch_and_checked_narrow_transport(self):
        rules,roots,viewers,allocations=self.make();rules.fast.vm_errors[('decision',11)]=1
        out=rules.decision_device(roots,viewers)
        self.assertIs(out.states,roots);self.assertEqual(out.errors.tolist(),[0,0,77,0,0,0,0])
        self.assertEqual(allocations.count(7),1) # No slow full-B candidate allocation.
        self.assertTrue(all(n in (2,3,7) for n in allocations))
        self.assertEqual(out.counts.tolist(),[3,0,0,2,3,2,3])
        self.assertEqual(out.action_refs[0].tolist(),[112,110,111])
        self.assertEqual(out.candidates.tokens.int_tokens.dtype,t.int32)
        self.assertEqual(out.candidates.tokens.int_tokens[0,0].tolist(),[11]*7)
        self.assertEqual(out.observations[:,0].tolist(),viewers.float().tolist())
        rules,roots,viewers,_=self.make();rules.fast.vm_errors[('decision',11)]=1;rules.slow.overflow=True
        out=rules.decision_device(roots,viewers)
        self.assertEqual(int(out.errors[0]),156)

    def test_partial_failed_new_output_is_freed_but_parent_alias_is_refused(self):
        for kind in ('clean','decision'):
            rules,roots,viewers,_=self.make();rules.fast.vm_errors[(kind,11)]=1;rules.fast.partial=(kind,11)
            out=rules.clean_device(roots) if kind=='clean' else rules.decision_device(roots,viewers)
            pool=rules.store if kind=='clean' else rules.action_store
            self.assertEqual(int(out.errors[0]),0);self.assertEqual(len(pool.released),1)
            self.assertTrue(bool(rules.store.live[roots.slots].all()))
        rules,roots,_,_=self.make();rules.fast.vm_errors[('clean',11)]=1
        rules.fast.partial=('clean',11);rules.fast.alias='clean'
        out=rules.clean_device(roots)
        self.assertEqual(int(out.errors[0]),1|OUTPUT_LEASE_ERROR)
        self.assertEqual(int(out.slots[0]),int(roots.slots[0]));self.assertEqual(int(rules.fallback_counts.sum()),0)

    def test_state_value_teacher_and_sample_arguments_retry_from_immutable_parent(self):
        for kind,method in (('value','value_observation_device'),('sample','sample_world_device'),
                            ('teacher','teacher_continue_device')):
            rules,roots,viewers,_=self.make();rules.fast.vm_errors[(kind,17)]=1
            seeds=t.zeros((7,5),dtype=t.int64);seeds[:,0]=t.arange(7)+100;seeds[:,4]=1
            before=rules.store.data[roots.slots].clone()
            args=(viewers,seeds) if kind=='sample' else (viewers,)
            out=getattr(rules,method)(roots,*args)
            self.assertEqual(int(out.errors[-1]),0)
            calls=[row for row in rules.slow.calls if row[1].active.any()]
            self.assertEqual(len(calls),1);self.assertEqual(calls[0][1].owners.tolist(),[17,0])
            self.assertEqual(calls[0][2].tolist(),[int(viewers[-1]),int(viewers[0])])
            if kind=='sample':t.testing.assert_close(calls[0][3],seeds[t.tensor([6,0])])
            t.testing.assert_close(rules.store.data[roots.slots],before)

    def test_raw_order_and_step_retry_transport_original_decision_and_actions(self):
        rules,roots,viewers,_=self.make();rules.fast.vm_errors[('rule_decision',11)]=1
        decision,chosen,errors=rules.rule_decision_device(roots,viewers)
        self.assertEqual(int(errors[0]),0);self.assertEqual(decision.candidates.shape,(7,3,0))
        self.assertEqual(decision.action_refs[0].tolist(),[112,110,111]);self.assertEqual(int(chosen[0]),int(viewers[0]))
        rules.fast.vm_errors[('step',11)]=1
        saved=decision.action_refs.clone();state_before=roots.slots.clone()
        output=rules.step_device(roots,viewers,decision,chosen)
        self.assertEqual(int(output.errors[0]),0);self.assertEqual(int(output.slots[0])>=0,True)
        slowsteps=[row for row in rules.slow.steps if row[0].active.any()]
        self.assertEqual(len(slowsteps),1)
        local,actors,selected,indices,physical=slowsteps[0]
        self.assertEqual(selected.candidates.shape,(2,3,0))
        self.assertEqual(selected.action_refs[0].tolist(),saved[0].tolist())
        self.assertEqual(int(indices[0]),int(chosen[0]));self.assertFalse(physical)
        t.testing.assert_close(decision.action_refs,saved);t.testing.assert_close(roots.slots,state_before)
        # The original checked action lease is read, not consumed by either hypothetical attempt.
        self.assertTrue(bool(rules.action_store.live[decision.action_slots[0]]))

    def test_compact_step_and_fused_retry_keep_original_tokens_and_invalid_action_owner(self):
        for fused in (False,True):
            rules,roots,viewers,_=self.make()
            decision=rules.decision_device(roots,viewers)
            original={f.name:getattr(decision.candidates.tokens,f.name).clone()
                      for f in fields(decision.candidates.tokens) if f.name!='request_ids'}
            kind='step_clean' if fused else 'step'
            rules.fast.vm_errors[(kind,17)]=1
            method=rules.step_clean_device if fused else rules.step_device
            chosen=t.zeros(7,dtype=t.int64)
            result=method(roots,viewers,decision,chosen)
            self.assertEqual(int(result.errors[-1]),0)
            selected=[row[2] for row in rules.slow.steps if row[0].active.any()][0]
            for name,value in original.items():
                t.testing.assert_close(getattr(selected.candidates.tokens,name),value[t.tensor([6,0])])
                t.testing.assert_close(getattr(decision.candidates.tokens,name),value)
            self.assertEqual(selected.candidates.shape[:2],(2,3))
        rules,roots,viewers,_=self.make();decision=rules.decision_device(roots,viewers)
        rules.fast.vm_errors[('step',11)]=1
        decision.action_generations[0]+=1
        result=rules.step_device(roots,viewers,decision,t.zeros(7,dtype=t.int64))
        self.assertEqual(int(result.errors[0]),1);self.assertEqual(int(rules.fallback_counts.sum()),0)

    def test_preexisting_action_lease_is_not_reclaimed_or_overwritten_on_oom(self):
        rules,roots,viewers,_=self.make()
        existing=rules.action_store.allocate(roots.owners,roots.active&(roots.errors==0),roots.owners)
        rules.fast.vm_errors[('decision',11)]=1
        original=rules.fast.inner.decision_device
        def alias(states,observers):
            out=original(states,observers)
            row=(states.owners==11)&states.active
            return replace(out,action_slots=t.where(row,existing.slots[0],out.action_slots),
                           action_generations=t.where(row,existing.generations[0],out.action_generations))
        rules.fast.inner.decision_device=alias
        out=rules.decision_device(roots,viewers)
        self.assertEqual(int(out.errors[0]),1|OUTPUT_LEASE_ERROR)
        self.assertEqual(int(out.action_slots[0]),int(existing.slots[0]))
        self.assertTrue(bool(rules.action_store.live[existing.slots[0]]))
        self.assertEqual(int(rules.fallback_counts.sum()),0)

    def test_physical_once_initialization_bind_close_and_workspace_ownership(self):
        rules,roots,viewers,_=self.make();decision,chosen,_=rules.rule_decision_device(roots,viewers)
        rules.fast.vm_errors[('step',11)]=1
        loop=NS(rows=7,device=t.device('cpu'),steps_device=t.zeros(7,dtype=t.int64),
                source_versions_device=t.zeros(7,dtype=t.int64))
        rules.bind_game_loop(loop)
        rules.step_device(roots,viewers,decision,chosen,physical=True)
        self.assertFalse(rules.fast.steps)
        self.assertEqual(rules.slow.raw_writes,int((roots.active&(roots.errors==0)).sum()))
        parameters=[dict(seed=1)];owner=t.tensor([1])
        self.assertEqual(rules.initialize_device(parameters,owners=owner)[0],parameters)
        self.assertEqual((rules.fast.initializations,rules.slow.initializations),(0,1))
        self.assertIs(rules.slow.physical_loop,loop);self.assertIsNone(rules.fast.physical_loop)
        expected=rules.fast.workspace_bytes+rules.slow.workspace_bytes-rules.store.workspace_bytes
        own=sum(value.numel()*value.element_size() for value in
            (rules._lane_capacity,rules._proven_capacity,rules._fast_guard,rules.fallback_counts,rules._row_ids))
        self.assertEqual(rules.workspace_bytes,expected+own)
        self.assertEqual(rules.fallback_receipt_device()['fast_lanes'],3)
        self.assertEqual(rules.fallback_receipt_device()['slow_lanes'],2)
        self.assertFalse(rules.gpu_qualified)
        rules.close();rules.close()
        self.assertEqual((rules.fast.closes,rules.slow.closes),(1,1))
        self.assertFalse(rules.store.closed);self.assertFalse(rules.action_store.closed)

    def test_constructor_fixed_scratch_tiers_and_hook_restore_on_capture_exception(self):
        rules,roots,viewers,_=self.make()
        with patch('app.modules.card_game.rl.batched_search.device_tiered_soa_rules.DeviceTiledSoARules',
                   side_effect=[rules.fast,rules.slow]) as factory:
            built=DeviceTieredSoARules(rules.executor,rules.store,rules.action_store,rows=7,raw_tape=object())
        fast,slow=[call.kwargs for call in factory.call_args_list]
        self.assertEqual((fast['work_rows'],fast['scratch_nodes'],fast['scratch_edges'],fast['scratch_payload']),
                         (32,262144,524288,4194304))
        self.assertEqual((slow['work_rows'],slow['scratch_nodes'],slow['scratch_edges'],slow['scratch_payload']),
                         (2,1048576,2097152,33554432))
        self.assertIsNone(fast['raw_tape']);self.assertIsNotNone(slow['raw_tape'])
        original=rules.fast.inner._invoke;site=rules.fast._site
        with patch.object(rules.fast,'metadata_device',side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError,'capture failed'):rules.metadata_device(roots)
        self.assertEqual(rules.fast.inner._invoke,original);self.assertEqual(rules.fast._site,site)

    def test_partial_constructor_failure_closes_owned_fast_only(self):
        rules,roots,viewers,_=self.make()
        with patch('app.modules.card_game.rl.batched_search.device_tiered_soa_rules.DeviceTiledSoARules',
                   side_effect=[rules.fast,RuntimeError('slow preparation failed')]):
            with self.assertRaisesRegex(RuntimeError,'slow preparation failed'):
                DeviceTieredSoARules(rules.executor,rules.store,rules.action_store,rows=7)
        self.assertEqual(rules.fast.closes,1)
        self.assertFalse(rules.store.closed);self.assertFalse(rules.action_store.closed)

    def test_diagnostic_executor_and_invalid_lane_widths_fail_before_adapter_allocation(self):
        with patch('app.modules.card_game.rl.batched_search.device_tiered_soa_rules.DeviceTiledSoARules') as factory:
            with self.assertRaisesRegex(ValueError,'128KiB'):
                DeviceTieredSoARules(NS(execute_only=True),object(),object(),rows=7)
            for options in (dict(work_rows=True),dict(work_rows=65),dict(fallback_rows=0),dict(work_rows=2,fallback_rows=3)):
                with self.assertRaisesRegex(ValueError,'lanes'):
                    DeviceTieredSoARules(NS(execute_only=False),object(),object(),rows=7,**options)
            factory.assert_not_called()


if __name__=='__main__':unittest.main()
