"""CPU contract fixtures only; none grants CUDA collection/driver qualification."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import ast
import inspect
import textwrap
import time
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from app.modules.card_game.rl import device_auxiliary_collection as aux
from app.modules.card_game.rl.seven_auxiliary_jobs import auxiliary_jobs,adapt_auxiliary_jobs
from app.modules.card_game.rl.build_acceptance import canonical_build
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.seven_schedule import ROSTER


def metadata():
    decks={d['id']:d for d in STARTER_DECKS}
    starts={key:canonical_build(decks[key])for key in ROSTER}
    jobs=auxiliary_jobs(71,1,decks=starts,deadline=time.time()+600)[:1]
    return adapt_auxiliary_jobs(jobs,starting_decks=starts)


class Oracle:
    """Explicit tiny synthetic oracle, retaining normal initial events/cache."""
    def __init__(self):self.initial_clean_calls=0
    def new(self,**parameters):
        shared={'same':True}
        return dict(version=0,phase='mulligan',active_side=parameters['first_side'],rng=parameters['seed'],
            winner=None,reason=None,events=['normal-init'],_public_board={'original':True},left=shared,right=shared)
    def actor(self,state):return state['active_side']
    def observe(self,state,side,**kwargs):return dict(v=state['version'],rng=state['rng'],viewer=int(side=='b'))
    def encode(self,view,actions):return np.asarray([view['v'],view['rng']%997,view['viewer']],np.float32),np.empty((0,0),np.float32)
    def rule(self,state,actor):return [dict(index=i)for i in range(3)],state['version']%3
    def apply(self,state,side,action):
        out=deepcopy(state);out['version']+=1;out['rng']+=action['index']+1
        out['active_side']='b'if side=='a'else'a'
        out['phase']='finished'if out['version']==8 else 'mulligan'if out['version']<2 else 'playing'
        if out['phase']=='finished':out.update(winner='a',reason='player_hp')
        return out
    def clean(self,state):
        if state['version']==0:self.initial_clean_calls+=1
        out=deepcopy(state);out['events']=[];out.pop('_public_board',None);return out
    def api(self):return self.new,self.observe,self.actor,self.apply,self.clean,self.encode,self.rule


def segment(meta,oracle,steps=8,*,errors=(0,),truncated=(False,)):
    job=meta.jobs[0];initial=oracle.new(**meta.initializers[0]);state=deepcopy(initial)
    rng=np.random.default_rng(job['seed']);raw=[];audit=[];values=[]
    for step in range(steps):
        actor=oracle.actor(state);legal,rule=oracle.rule(state,actor);explored=step<6 and state['phase']=='playing'
        chosen=int(rng.integers(3))if explored else rule
        identity=dict(owner=0,step=step,seed=job['seed'],version=state['version'],physical_actor=int(actor=='b'))
        raw.append(dict(identity,game=0,raw_action=deepcopy(legal[chosen])))
        audit.append(dict(identity,count=3,rule_chosen=rule,chosen=chosen,explored=int(explored),actions=deepcopy(legal),requested=0,completed=0,visit_sum=0,pcg_state=deepcopy(rng.bit_generator.state)))
        values.append(dict(identity,slot=step,generation=1,errors=0,
            x=np.stack([oracle.encode(oracle.observe(state,s),[])[0]for s in ('a','b')]),
            context=np.asarray([actor=='a',actor=='b'],np.float32),
            z=np.asarray([1,-1]if steps==8 else[0,0],np.float32),eligible=np.asarray([steps==8]*2,np.bool_)))
        state=oracle.clean(oracle.apply(state,actor,legal[chosen]))
    return aux.AuxiliarySegment(meta,(initial,),(state,),(steps,),(steps==8,),truncated,errors,
        (deepcopy(rng.bit_generator.state),),tuple(raw),tuple(audit),tuple(values),(),
        'synthetic_contract_fixture',decision_limit=512,source_binding=dict(rule_source='0'*64,image='1'*64,schema='2'*64),replay_verified=True)


class AuxiliaryCollectionContractTests(unittest.TestCase):
    def test_complete_actual_input_contract_and_value_only_export(self):
        oracle=Oracle();evidence=segment(metadata(),oracle);before=deepcopy(evidence)
        with patch.object(aux,'_oracle',oracle.api):
            self.assertTrue(aux.verify_auxiliary_segment(evidence));games=aux.verified_auxiliary_games(evidence)
        self.assertEqual(oracle.initial_clean_calls,0)
        self.assertEqual(games[0]['rows'],[]);self.assertTrue(games[0]['auxiliary_value_only'])
        self.assertFalse(games[0]['formal_training_qualified']);self.assertEqual(len(games[0]['value_rows']),4)
        self.assertEqual([r['step']for r in games[0]['value_rows']],[6,6,7,7])
        self.assertEqual(games[0]['seed'],evidence.metadata.jobs[0]['seed'])
        self.assertEqual(len(games[0]['actions']),8)
        aux._same_graph(evidence.initial,before.initial);aux._same_graph(evidence.final,before.final)
        for a,b in zip(evidence.value_records,before.value_records):np.testing.assert_array_equal(a['x'],b['x'])
        games[0]['value_rows'][0]['x'][0]=-999
        self.assertNotEqual(evidence.value_records[6]['x'][0,0],-999)

    def test_partial_is_auditable_but_never_label_export(self):
        oracle=Oracle();evidence=segment(metadata(),oracle,steps=7,errors=(256,),truncated=(False,))
        with patch.object(aux,'_oracle',oracle.api):
            self.assertTrue(aux.verify_auxiliary_segment(evidence))
            with self.assertRaises(ValueError):aux.verified_auxiliary_games(evidence)
        self.assertEqual(evidence.errors,(256,));self.assertTrue(all(not r['eligible'].any()for r in evidence.value_records))
        with patch.object(aux,'_oracle',oracle.api),self.assertRaises(ValueError):
            aux.verify_auxiliary_segment(replace(evidence,errors=(193,)))

    def test_missing_or_changed_gpu_values_choices_coordinates_rng_rejected(self):
        oracle=Oracle();original=segment(metadata(),oracle)
        changes=[]
        for target,field,value in [('value_records','seed',True),('value_records','physical_actor',1),
            ('value_records','version',1),('value_records','x',None),('value_records','generation',0),
            ('audit_records','count',4),('audit_records','rule_chosen',2),('audit_records','chosen',2),
            ('raw_records','step',1),('raw_records','game',1),('audit_records','requested',1)]:
            item=deepcopy(original);getattr(item,target)[0][field]=value;changes.append(item)
        item=deepcopy(original);item.value_records[0]['context'][0]=0;changes.append(item)
        item=deepcopy(original);item.value_records[0]['z'][0]=-1;changes.append(item)
        item=deepcopy(original);item.value_records[0]['eligible'][1]=False;changes.append(item)
        item=deepcopy(original);item.audit_records[4]['pcg_state']['uinteger']^=1;changes.append(item)
        item=deepcopy(original);item.audit_records[0]['actions'].reverse();changes.append(item)
        changes.append(replace(original,value_records=original.value_records[:-1]))
        for item in changes:
            with self.subTest(record_error=True),patch.object(aux,'_oracle',oracle.api),self.assertRaises((ValueError,KeyError)):
                aux.verify_auxiliary_segment(item)

    def test_normal_initial_events_and_final_rng_aliases_are_required(self):
        oracle=Oracle();original=segment(metadata(),oracle)
        cleaned=oracle.clean(deepcopy(original.initial[0]));oracle.initial_clean_calls=0
        wrong=deepcopy(original);wrong.final[0]['rng']+=1
        alias=deepcopy(original);alias.final[0]['right']=dict(alias.final[0]['left'])
        for bad in (replace(original,initial=(cleaned,)),wrong,alias):
            with patch.object(aux,'_oracle',oracle.api),self.assertRaises(ValueError):aux.verify_auxiliary_segment(bad)
        self.assertEqual(oracle.initial_clean_calls,0)

    def test_counter_sources_and_group_restore_include_raw_previous_version(self):
        oracle=Oracle();prefix=segment(metadata(),oracle,steps=7,errors=(256,))
        self.assertEqual(aux._resume_counters(prefix,prefix.final),([7],[6]))
        wrong=deepcopy(prefix);wrong.raw_records[-1]['version']=5
        with self.assertRaises(ValueError):aux._resume_counters(wrong,wrong.final)
        self.assertEqual(aux._resume_counters(None,[oracle.new(**prefix.metadata.initializers[0])]),([0],[-1]))
        class Loop:
            def __init__(self):self._steps=torch.zeros(1,dtype=torch.int64);self.calls=0
            def _update_predicate(self):self.calls+=1
        loop=Loop();raw=SimpleNamespace(_expected_step=torch.zeros(1,dtype=torch.int64),_last_version=torch.full((1,),-1,dtype=torch.int64))
        tape=SimpleNamespace(_expected_step=torch.zeros(1,dtype=torch.int64),_initial_version=torch.tensor([7],dtype=torch.int64))
        aux._restore_counters(loop,raw,tape,torch.tensor([7]),torch.tensor([6]))
        self.assertEqual(loop._steps.tolist(),[7]);self.assertEqual(raw._expected_step.tolist(),[7]);self.assertEqual(raw._last_version.tolist(),[6])
        self.assertEqual(tape._expected_step.tolist(),[7]);self.assertEqual(tape._initial_version.tolist(),[0]);self.assertEqual(loop.calls,1)

    def test_prior_rule_or_memory_failure_cannot_be_laundered(self):
        oracle=Oracle();original=segment(metadata(),oracle)
        for badattempt in (dict(errors=(193,),raw_errors=(0,),value_errors=(0,)),
                           dict(errors=(0,),raw_errors=(128,),value_errors=(0,)),
                           dict(errors=(0,),raw_errors=(0,),value_errors=(0,),memory_limit_exceeded=True)):
            with patch.object(aux,'_oracle',oracle.api),self.assertRaises(ValueError):
                aux.verify_auxiliary_segment(replace(original,attempts=(badattempt,)))
        with self.assertRaises(TimeoutError):aux._deadline(time.time()-1,original.metadata,time.monotonic(),150)
        with self.assertRaises(ValueError):aux._deadline(True,original.metadata,time.monotonic(),150)
        with self.assertRaises(TypeError):aux.DeviceAuxiliaryCollector(None,None,None,None,rows=1,max_decisions=800)
        from app.modules.card_game.rl.batched_search.device_execution_contract import require_closed_device_driver,DeviceExecutionNotReady
        with self.assertRaises(DeviceExecutionNotReady):require_closed_device_driver(aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector),purpose='training')

    def test_real_value_tape_restored_prefix_gets_z_from_terminal_finish(self):
        from tests.test_duel_v2_device_pure_value_tape import PureValueTapeTests
        fixture=PureValueTapeTests();source,rules,roots=fixture.fixture(train=torch.ones((2,2),dtype=torch.bool))
        roots=replace(roots,owners=torch.tensor([0,1]));roots.pool.owner.sub_(40);source._owners.copy_(roots.owners)
        self.assertEqual(fixture.append(source,rules,roots).tolist(),[0,0])
        records=[]
        for owner in (0,1):
            records.append(dict(owner=owner,step=0,seed=[123,456][owner],physical_actor=owner,version=[7,8][owner],
                slot=owner,generation=1,errors=0,x=source._x[owner].numpy().copy(),context=source._ctx[owner].numpy().copy(),
                z=np.zeros(2,np.float32),eligible=np.zeros(2,np.bool_)))
        resume=SimpleNamespace(value_records=records,complete=(False,False),final=({},{}))
        payload=aux._history_tensors(torch,torch.device('cpu'),resume,2,3)
        target,new_rules,new_roots=fixture.fixture(train=torch.ones((2,2),dtype=torch.bool))
        new_roots=replace(new_roots,owners=torch.tensor([0,1]));new_roots.pool.owner.sub_(40);target._owners.copy_(new_roots.owners)
        aux._restore_history(target,payload)
        self.assertEqual(target._count.item(),2);self.assertFalse(target._eligible.any());self.assertFalse(target._z.any())
        # The resumed root metadata is one physical step beyond the original.
        new_rules.versions+=1
        offsets=torch.ones(2,dtype=torch.int64)
        raw=SimpleNamespace(_expected_step=torch.zeros(2,dtype=torch.int64),_last_version=torch.full((2,),-1,dtype=torch.int64))
        loop=SimpleNamespace(_steps=torch.zeros(2,dtype=torch.int64),_update_predicate=lambda:None)
        target._initial_version.copy_(new_rules.versions)
        aux._restore_counters(loop,raw,target,offsets,torch.tensor([7,8]))
        self.assertEqual(fixture.append(target,new_rules,new_roots,steps=offsets).tolist(),[0,0])
        finished=fixture.terminal(new_rules,new_roots)
        self.assertEqual(fixture.finish(target,new_rules,finished,steps=torch.full((2,),2,dtype=torch.int64)).tolist(),[0,0])
        self.assertTrue(target._eligible[:4].all());self.assertEqual(target._z[:4].tolist(),[[1.,-1.],[-1.,1.],[1.,-1.],[-1.,1.]])
        rows=target.value_rows_after_batch(include_frozen=True);self.assertEqual(len(rows),8)
        self.assertEqual(sorted((r['owner'],r['step'],r['viewer'])for r in rows),[(o,s,v)for o in (0,1)for s in (0,1)for v in (0,1)])
        for policy in (rules,new_rules):policy.action_store.close()

    def test_diagnostic_gather_only_clean_guard_verified_registry_and_original_errors(self):
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        registry=SimpleNamespace(t=torch,capacity=2,schema=SimpleNamespace(root='State',pools={'State':SimpleNamespace(base=100,capacity=20)}),
            live=torch.ones(2,dtype=torch.bool),generation=torch.ones(2,dtype=torch.int64),owner=torch.tensor([0,1]),roots=torch.tensor([101,102]))
        original=DeviceStateBatch(registry,torch.tensor([0,1]),torch.ones(2,dtype=torch.int64),torch.tensor([0,1]),
            torch.ones(2,dtype=torch.bool),torch.tensor([256,193],dtype=torch.int32))
        seen=[]
        def gather(request,**kwargs):
            seen.append(request)
            self.assertEqual(request.slots.tolist(),[0]);self.assertEqual(request.errors.tolist(),[0])
            return SimpleNamespace(to_host=lambda:'only safe row')
        registry.gather_device=gather
        with patch('app.modules.card_game.rl.batched_duel.record_soa.unpack_game_states',return_value=[{'safe':True}]):
            states,checks=aux._final_states(registry,original,(False,False))
        self.assertEqual(states,[{'safe':True},None]);self.assertEqual([c['verified']for c in checks],[True,False])
        self.assertEqual(original.errors.tolist(),[256,193]);self.assertEqual(len(seen),1)
        registry.generation[0]=2
        states,checks=aux._final_states(registry,original,(False,False))
        self.assertEqual(states,[None,None]);self.assertEqual(len(seen),1)

    def test_owned_arenas_close_on_every_preflight_failure_and_context_exit(self):
        class Resource:
            def __init__(self):self.leases=[1,2];self.closes=0
            def close(self):self.closes+=1;self.leases.clear()
        def collector():
            c=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
            c._closed=False;c._used=False;c._objects=[Resource()];c._arenas=tuple(Resource()for _ in range(3));c.rows=1
            c.max_decisions=800;c._source_binding=dict(rule_source='0'*64,image='1'*64,schema='2'*64);c._memory=lambda:0
            return c
        meta=metadata()
        for case in ('metadata','deadline','memory','resume'):
            c=collector()
            if case=='memory':
                def failed_memory():raise MemoryError('whole-card limit')
                c._memory=failed_memory
            resources=(*c._objects,*c._arenas)
            kwargs=dict(deadline=time.time()+300)
            if case=='deadline':kwargs['deadline']=time.time()-1
            if case=='resume':kwargs['resume']=object()
            with self.subTest(case=case),self.assertRaises((ValueError,TypeError,TimeoutError,MemoryError)):
                c.run_segment(None if case=='metadata'else meta,**kwargs)
            self.assertTrue(c._closed);self.assertEqual(c._objects,[]);self.assertEqual(c._arenas,())
            self.assertIsNone(c.state_store)
            for resource in resources:self.assertEqual(resource.closes,1);self.assertFalse(resource.leases)
            c.close()
            for resource in resources:self.assertEqual(resource.closes,1)
        c=collector();resources=(*c._objects,*c._arenas)
        with self.assertRaisesRegex(RuntimeError,'caller interruption'):
            with c:raise RuntimeError('caller interruption')
        for resource in resources:self.assertEqual(resource.closes,1)

    def test_clean_device_boundary_pause_does_not_finish_truncate_or_mark_errors(self):
        from tests.test_duel_v2_device_games import DeviceGameLoopTest
        loop,rules,sink,roots,seeds,ids,train=DeviceGameLoopTest().make()
        original=loop._update_predicate;offsets=torch.zeros(2,dtype=torch.int64)
        def update():original();aux._planned_pause(loop,offsets,2)
        loop._update_predicate=update
        loop.initialize(roots,seeds,ids,ids,train)
        loop.tick();self.assertTrue(loop.unfinished_device.any())
        loop.tick();self.assertFalse(loop.unfinished_device.any())
        result=loop.result_device()
        self.assertEqual(result.steps.tolist(),[2,2]);self.assertEqual(result.errors.tolist(),[0,0])
        self.assertEqual(result.complete.tolist(),[False,False]);self.assertEqual(result.truncated.tolist(),[False,False])
        self.assertEqual(sink.finishes.tolist(),[0,0]);loop.close()

    def test_audit_archive_error_is_sticky_in_actual_value_tape(self):
        from tests.test_duel_v2_device_pure_value_tape import PureValueTapeTests
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        fixture=PureValueTapeTests();tape,rules,roots=fixture.fixture(train=torch.ones((2,2),dtype=torch.bool))
        result=fixture.pure(rules,roots)
        audit=CompactDeviceGraphStore(slots=8,node_pages=16,edge_pages=16,payload_pages=16,
            node_page_bytes=128,edge_page_bytes=128,payload_page_bytes=128,max_node_pages_per_slot=8,
            max_edge_pages_per_slot=8,max_payload_pages_per_slot=8,device='cpu',allow_cpu_reference=True)
        state=SimpleNamespace(state_hi=torch.zeros(2,dtype=torch.uint64),state_lo=torch.ones(2,dtype=torch.uint64),
            inc_hi=torch.zeros(2,dtype=torch.uint64),inc_lo=torch.ones(2,dtype=torch.uint64),
            has_uint32=torch.zeros(2,dtype=torch.uint32),uinteger=torch.zeros(2,dtype=torch.uint32))
        controller=SimpleNamespace(rules=SimpleNamespace(action_layout=GraphLayout(2,16,16,128)),
            rule_choices_device=torch.tensor([0,1]),explored_device=torch.zeros(2,dtype=torch.bool),exploration_state_device=state)
        sink=aux._AuxiliarySink(tape,controller,audit)
        ids=torch.zeros((2,2),dtype=torch.int64)
        sink.begin_device(roots,ids,ids,torch.tensor([123,456]),torch.ones((2,2),dtype=torch.bool),roots.active)
        original=audit.put_device
        def fail(*args,**kwargs):
            value=original(*args,**kwargs)
            return replace(value,errors=value.errors|128)
        audit.put_device=fail
        errors=sink.append_device(roots,rules.actors,torch.zeros(2,dtype=torch.int64),result,roots.active)
        self.assertEqual(errors.tolist(),[128,128]);self.assertEqual(sink.append_errors.tolist(),[128,128])
        self.assertEqual(tape._game_errors.tolist(),[128,128]);self.assertEqual(sink.count.item(),0)
        self.assertEqual(tape._count.item(),2,'Actual numerical prefix remains for error audit')
        terminal=fixture.terminal(rules,roots)
        sink.finish_device(terminal,rules.winners,torch.ones(2,dtype=torch.int64),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.int32))
        self.assertFalse(tape._eligible.any());self.assertFalse(tape._z.any())
        with self.assertRaises(RuntimeError):tape.value_rows_after_batch(include_frozen=True)
        audit.close();rules.action_store.close()

    def test_prior_guard_can_be_audited_but_cannot_become_learning_rows(self):
        oracle=Oracle();original=segment(metadata(),oracle)
        guarded=replace(original,attempts=(dict(errors=(256,),raw_errors=(0,),value_errors=(0,)),))
        with patch.object(aux,'_oracle',oracle.api):
            self.assertTrue(aux.verify_auxiliary_segment(guarded))
            with self.assertRaisesRegex(ValueError,'prior guard'):aux.verified_auxiliary_games(guarded)
        capped=segment(original.metadata,oracle,steps=7,truncated=(True,))
        with patch.object(aux,'_oracle',oracle.api),self.assertRaises(ValueError):aux.verified_auxiliary_games(capped)

    def test_actual_raw_tape_resumed_group_counters_accepts_original_global_step(self):
        from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape,STEP_ERROR
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
        raw=DeviceRawActionTape(2,4,64,64,256,device='cpu',allow_cpu_reference=True)
        owners=torch.tensor([0,1]);raw.begin_device(owners,torch.tensor([71,72]))
        packed=pack_states([{'a':1},{'a':2}],GraphLayout(2,16,16,64))
        graph=DeviceGraph(packed.layout,*(torch.tensor(np.array(getattr(packed,k),copy=True))for k in ('nodes','edges','payload','counts','roots')))
        offsets=torch.full((2,),8,dtype=torch.int64);active=torch.ones(2,dtype=torch.bool)
        self.assertEqual(raw.append_device(graph,owners,owners,offsets,offsets,owners,active).tolist(),[STEP_ERROR]*2)
        raw.begin_device(owners,torch.tensor([71,72]))
        loop=SimpleNamespace(_steps=torch.zeros(2,dtype=torch.int64),_update_predicate=lambda:None)
        tape=SimpleNamespace(_expected_step=torch.zeros(2,dtype=torch.int64),_initial_version=offsets.clone())
        aux._restore_counters(loop,raw,tape,offsets,offsets-1)
        self.assertEqual(raw.append_device(graph,owners,owners,offsets,offsets,owners,active).tolist(),[0,0])
        self.assertEqual(raw.steps_device.tolist(),[9,9]);self.assertEqual(raw._last_version.tolist(),[8,8]);raw.close()

    def test_cleanup_error_does_not_replace_primary_failure(self):
        class FailedResource:
            def __init__(self):self.called=0
            def close(self):self.called+=1;raise RuntimeError('cleanup failure')
        class Resource:
            def __init__(self):self.called=0
            def close(self):self.called+=1
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        collector._closed=False;collector._objects=[FailedResource()];collector._arenas=tuple(Resource()for _ in range(3))
        resources=(*collector._objects,*collector._arenas)
        with self.assertRaises(TypeError):collector.run_segment(None,deadline=time.time()+100)
        self.assertTrue(collector.cleanup_errors)
        for obj in resources:self.assertEqual(obj.called,1)
        collector.close()

    def test_close_releases_teacher_pipeline_and_capture_cycles_but_borrows_executor(self):
        closed=[]
        class Owned:
            def __init__(self,name):self.name=name
            def close(self):closed.append(self.name)
        class Rules(Owned):
            def close(self):
                self._teacher.close();self._teacher_pipeline.close();super().close()
        rules=Rules('rules');teacher=Owned('teacher');pipeline=Owned('pipeline')
        controller=Owned('controller');loop=Owned('loop');executor=Owned('borrowed executor')
        rules._teacher=teacher;rules._teacher_pipeline=pipeline;teacher.rules=rules;pipeline.rules=rules;pipeline.teacher=teacher
        rules.physical_loop=loop;controller._loop=loop;controller.rules=rules;loop.search=controller;loop.rules=rules
        loop.initialize=lambda target=loop:target;loop._update_predicate=lambda target=loop:target
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        collector._closed=False;collector._objects=[rules,controller,loop];collector._arenas=();collector.executor=executor
        refs={obj.name:weakref.ref(obj)for obj in (rules,teacher,pipeline,controller,loop)}
        borrowed=weakref.ref(executor)
        del rules,teacher,pipeline,controller,loop,executor
        collector.close()  # No gc.collect: ownership cycles must be cut directly.
        self.assertTrue(all(ref()is None for ref in refs.values()))
        self.assertIs(borrowed(),collector.executor);self.assertNotIn('borrowed executor',closed)
        self.assertEqual(set(closed),{'rules','teacher','pipeline','controller','loop'})
        collector.close();self.assertEqual(len(closed),5)

    def test_learning_export_requires_exact512_execution_and_returns_bound_identity(self):
        oracle=Oracle();original=segment(metadata(),oracle)
        with patch.object(aux,'_oracle',oracle.api):
            for limit in (800,511):
                diagnostic=replace(original,decision_limit=limit)
                self.assertTrue(aux.verify_auxiliary_segment(diagnostic),'Different ceiling remains a numerical diagnostic')
                with self.assertRaisesRegex(ValueError,'exact512'):aux.verified_auxiliary_games(diagnostic)
            for execution in (dict(original.metadata.execution,max_decisions=800),
                              {k:v for k,v in original.metadata.execution.items()if k!='max_decisions'}):
                wrong=replace(original,metadata=replace(original.metadata,execution=execution))
                with self.assertRaises(ValueError):aux.verified_auxiliary_games(wrong)
            games=aux.verified_auxiliary_games(original)
        self.assertEqual(games[0]['decision_limit'],512)
        self.assertEqual(games[0]['execution'],original.metadata.execution)
        self.assertEqual(games[0]['source_binding'],original.source_binding)
        games[0]['execution']['max_decisions']=800;games[0]['source_binding']['schema']='f'*64
        self.assertEqual(original.metadata.execution['max_decisions'],512)
        self.assertEqual(original.source_binding['schema'],'2'*64)

    def test_runtime_replay_is_single_gpu_call_not_cpu_rule_loop(self):
        source=inspect.getsource(aux.DeviceAuxiliaryCollector._run_segment);tree=ast.parse(textwrap.dedent(source))
        replay=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='replay']
        self.assertEqual(len(replay),1)
        for forbidden in ('simulate_action','apply_action','new_game','rule_decision','observe','encode'):
            self.assertNotIn(forbidden+'(',source)
        self.assertIn('defer_physical_presentation=True',source)
        self.assertIn('RawActionPolicySink(sink,raw)',source)
        self.assertNotIn('allow_cpu_reference=True',source)

class DeferredAuxiliaryProofTests(unittest.TestCase):
    def finish(self,evidence,**options):
        return aux.DeviceAuxiliaryCollector._post_export_replay_proof(evidence,deadline=time.time()+60,
            metadata=evidence.metadata,started=time.monotonic(),soft_seconds=60,**options)

    def test_default_runs_actual_independent_proof_and_keeps_existing_pause_reason(self):
        oracle=Oracle();original=replace(segment(metadata(),oracle,steps=7),replay_verified=False,reason='incomplete')
        real_verify=aux.verify_auxiliary_segment
        with patch.object(aux,'_oracle',oracle.api),patch.object(aux,'verify_auxiliary_segment',wraps=real_verify)as proof:
            result=self.finish(original)
        self.assertEqual(proof.call_count,1);self.assertIs(proof.call_args.args[0],original)
        self.assertTrue(result.replay_verified);self.assertEqual(result.reason,'planned_segment_pause')
        self.assertFalse(original.replay_verified)
        self.assertIs(result.raw_records,original.raw_records);self.assertIs(result.value_records,original.value_records)
        for method in (aux.DeviceAuxiliaryCollector.run_segment,aux.DeviceAuxiliaryCollector._run_segment):
            self.assertIs(inspect.signature(method).parameters['defer_replay_proof'].default,False)

    def test_natural_deferred_export_stays_unverified_and_cannot_enter_learning_or_clean_ledger(self):
        from app.modules.card_game.rl.device_auxiliary_calibration import DeviceAuxiliaryCalibrationLedger
        original=replace(segment(metadata(),Oracle()),replay_verified=False,reason='natural')
        with patch.object(aux,'verify_auxiliary_segment',side_effect=AssertionError('Deferred postproof must not run'))as proof:
            result=self.finish(original,defer_replay_proof=True)
        proof.assert_not_called();self.assertFalse(result.replay_verified)
        self.assertEqual(result.reason,'natural; pending_independent_cpu_proof')
        self.assertEqual(result.complete,(True,));self.assertEqual(result.errors,(0,))
        self.assertIs(result.final,original.final);self.assertIs(result.value_records,original.value_records)
        self.assertFalse(result.formal_training_qualified);self.assertFalse(result.gpu_qualified)
        self.assertFalse(DeviceAuxiliaryCalibrationLedger._clean(result))
        with self.assertRaisesRegex(ValueError,'verification required'):aux.verified_auxiliary_games(result)

    def test_real_export_errors_guards_truncation_and_attempts_are_never_cleared(self):
        from app.modules.card_game.rl.device_auxiliary_calibration import DeviceAuxiliaryCalibrationLedger
        for code in (193,256):
            original=replace(segment(metadata(),Oracle(),steps=7,errors=(code,),truncated=(True,)),
                replay_verified=False,reason='device_error'if code==193 else'device_guard_stop',
                export_errors=('raw decode failed',),attempts=({'errors':(code,),'memory_limit_exceeded':True},))
            result=self.finish(original,defer_replay_proof=True)
            self.assertFalse(result.replay_verified);self.assertTrue(result.reason.startswith(original.reason))
            self.assertIn('pending_independent_cpu_proof',result.reason)
            for name in('errors','truncated','export_errors','attempts','raw_records','audit_records','value_records'):
                self.assertIs(getattr(result,name),getattr(original,name))
            self.assertFalse(DeviceAuxiliaryCalibrationLedger._clean(result))
            with self.assertRaises(ValueError):aux.verify_auxiliary_segment(result)

    def test_default_failed_independent_proof_is_still_failed(self):
        oracle=Oracle();original=replace(segment(metadata(),oracle),replay_verified=False,reason='natural')
        original.audit_records[0]['chosen']=99
        with patch.object(aux,'_oracle',oracle.api):result=self.finish(original)
        self.assertFalse(result.replay_verified);self.assertIn('proof_failed:',result.reason)
        self.assertIs(result.audit_records,original.audit_records)

    def test_deferral_does_not_skip_full_resume_prefix_verification(self):
        class GPUFixtureBoundary(RuntimeError):pass
        oracle=Oracle();meta=metadata();previous=segment(meta,oracle,steps=7)
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        collector._used=False;collector._objects=[];collector.rows=1;collector.max_decisions=512
        collector._source_binding=previous.source_binding;collector._memory=lambda **kw:0
        collector.device='cpu';collector.t=torch;collector.raw_capacities=(1,1,1)
        real_verify=aux.verify_auxiliary_segment
        with patch.object(aux,'_oracle',oracle.api),patch.object(aux,'verify_auxiliary_segment',wraps=real_verify)as proof, \
             patch('app.modules.card_game.rl.batched_search.device_raw_action_tape.DeviceRawActionTape',
                   side_effect=GPUFixtureBoundary('Stop before any CUDA allocation'))as allocation:
            with self.assertRaises(GPUFixtureBoundary):
                collector._run_segment(meta,deadline=time.time()+60,resume=previous,defer_replay_proof=True)
        self.assertEqual(proof.call_count,1);self.assertIs(proof.call_args.args[0],previous)
        allocation.assert_called_once();self.assertEqual(oracle.initial_clean_calls,0)

    def test_public_optin_forwards_flag_and_closes_owned_resources(self):
        class Arena:
            def __init__(self):self.closed=0
            def close(self):self.closed+=1
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        collector._closed=False;collector._objects=[];collector._arenas=tuple(Arena()for _ in range(3))
        arenas=collector._arenas;candidate=replace(segment(metadata(),Oracle()),replay_verified=False)
        calls=[]
        def run(meta,**kwargs):calls.append(kwargs);return candidate
        collector._run_segment=run
        result=collector.run_segment(candidate.metadata,deadline=time.time()+60,defer_replay_proof=True)
        self.assertIs(result,candidate);self.assertIs(calls[0]['defer_replay_proof'],True)
        self.assertTrue(collector._closed);self.assertEqual([a.closed for a in arenas],[1,1,1])

    def test_nonboolean_optin_rejected_before_metadata_or_gpu_work(self):
        collector=aux.DeviceAuxiliaryCollector.__new__(aux.DeviceAuxiliaryCollector)
        with patch.object(aux,'_metadata',side_effect=AssertionError('No work before validating optin')):
            for invalid in(0,1,None,'yes'):
                with self.subTest(invalid=invalid),self.assertRaises(TypeError):
                    collector._run_segment(None,deadline=time.time()+60,defer_replay_proof=invalid)


if __name__=='__main__':unittest.main()
