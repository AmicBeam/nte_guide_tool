"""SoA teacher boundary and actual decoder-source CPU oracles; no GPU proof."""
import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_search.device_soa_teacher import (
    DeviceSoATeacher,inspect_soa_teacher_dependencies,_SOURCE,_INDEX_SPEC,_REFS_SPEC,
    TEACHER_INDEX_ERROR,INVALID_LEGAL)


class TeacherDecoderSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.batched_duel.native_array_cpu import HOST_CUDA_SHIMS,_cxx_compiler
        cls.directory=tempfile.TemporaryDirectory()
        root=Path(cls.directory.name);source=root/'decoder.cpp';library=root/'decoder.so'
        source.write_text(HOST_CUDA_SHIMS+_SOURCE+
            '\nextern "C" void teacher_row(int row,int rows){native_array_configure_row(row,rows);}\n')
        compiled=subprocess.run([_cxx_compiler(),'-std=c++17','-shared','-fPIC','-O1',str(source),'-o',str(library)],
                                capture_output=True,text=True,timeout=20)
        if compiled.returncode:
            cls.directory.cleanup();raise AssertionError(compiled.stderr[-5000:])
        cls.library=ctypes.CDLL(str(library));cls.library.teacher_row.argtypes=[ctypes.c_int,ctypes.c_int]
        for name,spec in (('soa_teacher_index',_INDEX_SPEC),('soa_teacher_refs',_REFS_SPEC)):
            getattr(cls.library,name).argtypes=[ctypes.c_void_p if x.endswith('*') else ctypes.c_int for x in spec]

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def graph(self,values):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        return pack_states(values,GraphLayout(len(values),64,128,1024))

    def launch(self,name,args,rows):
        spec=_INDEX_SPEC if name=='soa_teacher_index' else _REFS_SPEC
        refs=[ctypes.c_void_p(value.ctypes.data) if kind.endswith('*') else value
              for value,kind in zip(args,spec)]
        self.assertEqual(len(args),len(spec))
        for row in range(rows):
            self.library.teacher_row(row,rows);getattr(self.library,name)(*refs)

    def decode_index(self,graph,*,active=None,incoming=None):
        n=graph.nodes.shape[0]
        values=np.full(n,-777,dtype=np.int64);errors=np.zeros(n,dtype=np.int32)
        incoming=np.zeros(n,dtype=np.int32) if incoming is None else incoming
        active=np.ones(n,dtype=np.uint8) if active is None else active
        cn=np.zeros((1,4),dtype=np.int32);ce=np.zeros((1,2),dtype=np.int32);cp=np.zeros(1,dtype=np.uint8)
        l=graph.layout
        self.launch('soa_teacher_index',[graph.nodes,graph.edges,graph.payload,graph.counts,cn,ce,cp,1,1,1,
            graph.roots,incoming,active,values,errors,n,l.max_nodes,l.max_edges,l.max_payload_bytes],n)
        return values,errors

    def test_exact_int64_no_float_bool_truncation_or_precision_loss(self):
        inputs=[0,1,2**53+1,2**63-1,2**63,2**80,-1,True,False,0.,1.25,2.,None]
        graph=self.graph(inputs)
        offset=int(graph.nodes[-2,graph.roots[-2],1])
        graph.payload[-2,offset:offset+8]=np.frombuffer(np.float64(float('inf')).tobytes(),dtype=np.uint8)
        values,errors=self.decode_index(graph)
        self.assertEqual(values[:4].tolist(),inputs[:4]);self.assertEqual(errors[:4].tolist(),[0]*4)
        self.assertEqual(values[4:].tolist(),[-1]*(len(inputs)-4))
        self.assertEqual(errors[4:].tolist(),[TEACHER_INDEX_ERROR]*(len(inputs)-4))
        active=np.ones(len(inputs),dtype=np.uint8);active[-1]=0
        incoming=np.zeros(len(inputs),dtype=np.int32);incoming[0]=108
        _,errors=self.decode_index(graph,active=active,incoming=incoming)
        self.assertEqual(int(errors[0]),108);self.assertEqual(int(errors[-1]),0)

    def decode_legal(self,graph,chosen,*,width=3):
        n=graph.nodes.shape[0];l=graph.layout
        refs=np.full((n,width),77,dtype=np.int32);valid=np.ones((n,width),dtype=np.uint8)
        counts=np.full(n,77,dtype=np.int64);errors=np.zeros(n,dtype=np.int32)
        cn=np.zeros((1,4),dtype=np.int32);ce=np.zeros((1,2),dtype=np.int32);cp=np.zeros(1,dtype=np.uint8)
        incoming=np.zeros(n,dtype=np.int32);active=np.ones(n,dtype=np.uint8)
        self.launch('soa_teacher_refs',[graph.nodes,graph.edges,graph.payload,graph.counts,graph.roots,
            cn,ce,cp,1,1,1,np.asarray(chosen,dtype=np.int64),incoming,active,refs,valid,counts,errors,
            n,l.max_nodes,l.max_edges,l.max_payload_bytes,width],n)
        return refs,valid,counts,errors

    def test_raw_order_concede_and_alias_survive_decoder(self):
        from app.modules.card_game.rl.batched_duel.codec import unpack_states
        shared=[1,2]
        legal=[{'type':'end_turn'},{'type':'concede','left':shared,'right':shared}]
        graph=self.graph([legal])
        refs,valid,counts,errors=self.decode_legal(graph,[1])
        self.assertEqual(errors.tolist(),[0]);self.assertEqual(counts.tolist(),[2])
        self.assertEqual(valid.tolist(),[[1,1,0]])
        list_offset=int(graph.nodes[0,graph.roots[0],1])
        self.assertEqual(refs[0,:2].tolist(),graph.edges[0,list_offset:list_offset+2,0].tolist())
        result=unpack_states(graph)[0]
        self.assertEqual(result,legal)
        self.assertIs(result[1]['left'],result[1]['right'])

    def test_legal_capacity_index_type_and_invalid_reference_fail_closed(self):
        end={'type':'end_turn'}
        graph=self.graph([[end,end,end],[end],[],(end,),[None]])
        _,_,counts,errors=self.decode_legal(graph,[1,1,0,0,0],width=2)
        self.assertEqual(errors.tolist(),[INVALID_LEGAL]*5)
        self.assertEqual(counts.tolist(),[0]*5)
        graph=self.graph([[end]]);graph.roots[0]=999
        self.assertEqual(self.decode_legal(graph,[0])[-1].tolist(),[INVALID_LEGAL])


class SoATeacherBindingTests(unittest.TestCase):
    def test_canonical_source_binding_and_static_decoder_preparation(self):
        import torch as t
        teacher,states,actors,_,_=self.make();rules=teacher.rules
        rules.device=t.device('cuda')
        rules.store=states.pool
        prefix='func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:'
        rules.executor.execute_only=False
        rules.executor.image=NS(entries=dict(teacher_continue=0,rule_decision=1),
            function_ids=(prefix+'teacher_continue:hash',prefix+'rule_decision:hash'))
        paths=[];rules.executor.prepare_result_path=paths.append
        self.assertEqual(inspect_soa_teacher_dependencies(rules),())
        with patch.object(t.cuda,'is_available',return_value=True), \
             patch.object(t.cuda,'is_current_stream_capturing',return_value=False), \
             patch('app.modules.card_game.rl.batched_duel.cuda_runtime.CudaModule') as module:
            prepared=DeviceSoATeacher(rules)
            self.assertEqual(paths,[(0,),(1,)])
            self.assertEqual(set(module.call_args.args[1]),{'soa_teacher_index','soa_teacher_refs'})
            prepared.close();module.return_value.close.assert_called_once()
            for capacity in (True,0,802):
                with self.assertRaises(ValueError):DeviceSoATeacher(rules,max_actions=capacity)
            bounded=DeviceSoATeacher(rules,max_actions=800)
            self.assertEqual(bounded.max_actions,800);bounded.close()
        rules.executor.image.function_ids=(prefix+'teacher_continue:hash','func:unrelated:rule_decision:hash')
        self.assertIn('canonical frozen rule_decision binding required',inspect_soa_teacher_dependencies(rules))
        rules.executor.execute_only=True
        self.assertIn('full SoA projection module required',inspect_soa_teacher_dependencies(rules))

    def make(self):
        import torch as t
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        from tests.test_duel_v2_device_graph_storage import _graph
        rows=2;device=t.device('cpu');events=[]
        states=DeviceStateBatch(object(),t.arange(rows),t.ones(rows,dtype=t.int64),t.arange(rows)+10,
                               t.tensor([True,False]),t.zeros(rows,dtype=t.int32))
        graph=_graph([[{'type':'end_turn'},{'type':'concede'}]]*rows,'cpu')
        call=NS(scratch=graph,outputs=graph.roots,statuses=t.zeros((rows,4),dtype=t.int32))
        continued=replace(states,slots=states.slots+100)
        e=NS(closed=False,constants=dict(nodes=t.zeros((1,4),dtype=t.int32),edges=t.zeros((1,2),dtype=t.int32),
                                       payload=t.zeros(1,dtype=t.uint8)))
        def select_result(value,path,*,active):events.append(('select',path));return value
        def export_projection(value,*,active):events.append(('export',));return NS(arena=graph,statuses=call.statuses)
        e.select_result=select_result;e.export_projection=export_projection
        def invoke(entry,batch,scratch,refs,active):events.append(('invoke',entry,refs.shape));return call
        def commit(value,owners,active,errors):events.append(('commit',owners.clone(),active.clone()));return continued
        def put(value,owners,active):
            events.append(('put',active.clone()))
            return replace(states,pool=arena,slots=t.tensor([4,-1]),active=active,generations=t.tensor([1,0]))
        arena=NS(put_device=put)
        r=NS(t=t,device=device,rows=rows,executor=e,action_store=arena,
             _begin=lambda value:(object(),graph,value.errors.clone(),value.active.view(t.uint8)),
             _invoke=invoke,_commit=commit,_bind=lambda scratch,actors,active:(t.zeros((rows,2),dtype=t.int32),t.zeros(rows,dtype=t.int32)))
        teacher=object.__new__(DeviceSoATeacher)
        teacher.rules=r;teacher.t=t;teacher.device=device;teacher.rows=rows;teacher.max_actions=3
        teacher.closed=False;teacher._keepalive=[]
        def launch(name,args,**kwargs):
            events.append(('kernel',name))
            if name=='soa_teacher_refs':
                args[14][:]=t.tensor([[1,2,-1],[1,2,-1]])
                args[15][:]=t.tensor([[1,1,0],[0,0,0]],dtype=t.uint8)
                args[16][:]=t.tensor([2,0])
        teacher.module=NS(launch=launch)
        return teacher,states,t.tensor([0,1]),continued,events

    def test_unary_frozen_horizon3_and_checked_new_lease_commit(self):
        import torch as t
        teacher,states,viewers,continued,events=self.make()
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            self.assertIs(teacher.teacher_continue_device(states,viewers),continued)
        self.assertEqual(events[0],('invoke','teacher_continue',(2,1)))
        self.assertEqual(events[1][0],'commit')
        for horizon in (True,0,1,2,4):
            with self.assertRaisesRegex(ValueError,'horizon=3'):
                teacher.teacher_continue_device(states,viewers,horizon=horizon)

    def test_raw_calls_bypass_public_observation_and_float_result_reader(self):
        import torch as t
        teacher,states,actors,_,events=self.make()
        teacher.rules.executor.read_result=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Float decoder'))
        def exact_index(call,errors,active):
            events.append(('exact_int64',));return t.tensor([1,-1]),errors
        teacher._decode_index=exact_index
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            decision,chosen,errors=teacher.rule_decision_device(states,actors)
        self.assertEqual(events[0],('invoke','rule_decision',(2,2)))
        self.assertEqual([event[0] for event in events],['invoke','select','exact_int64','select','export','kernel','put'])
        self.assertEqual(chosen.tolist(),[1,-1]);self.assertEqual(errors.tolist(),[0,0])
        self.assertEqual(decision.observations.shape,(2,0));self.assertEqual(decision.candidates.shape,(2,3,0))
        self.assertEqual(decision.counts.tolist(),[2,0])
        self.assertIs(decision.action_pool,teacher.rules.action_store)

    def test_rule_and_index_errors_reach_result_and_prevent_action_leasing(self):
        import torch as t
        teacher,states,actors,_,events=self.make()
        teacher._decode_index=lambda call,errors,active:(t.tensor([-1,-1]),t.tensor([108,0],dtype=t.int32))
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            _,_,errors=teacher.rule_decision_device(states,actors)
        self.assertEqual(errors.tolist(),[108,0])
        self.assertEqual(events[-1][0],'put');self.assertEqual(events[-1][1].tolist(),[False,False])

    def test_continuation_status_capacity_is_not_hidden_or_retried(self):
        import torch as t
        teacher,states,viewers,_,events=self.make()
        invoke=teacher.rules._invoke
        def failing(*args):
            call=invoke(*args);call.statuses[0,0]=1;return call
        teacher.rules._invoke=failing
        def commit(call,owners,active,errors):
            events.append(('failed_commit',))
            return replace(states,errors=errors|call.statuses[:,0],active=t.zeros_like(states.active),
                           slots=t.full_like(states.slots,-1))
        teacher.rules._commit=commit
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            result=teacher.teacher_continue_device(states,viewers)
        self.assertEqual(result.errors.tolist(),[1,0]);self.assertFalse(bool(result.active.any()))
        self.assertEqual([event[0] for event in events],['invoke','failed_commit'])

    def test_fail_closed_binding_capture_and_actor_contract(self):
        import torch as t
        self.assertTrue(inspect_soa_teacher_dependencies(None))
        with self.assertRaises(ValueError):DeviceSoATeacher(None)
        self.assertFalse(DeviceSoATeacher.supports_full_duel_rules)
        self.assertFalse(DeviceSoATeacher.root_policy_compatible)
        teacher,states,viewers,_,_=self.make()
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'capture'):teacher.teacher_continue_device(states,viewers)
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            with self.assertRaises(ValueError):teacher.teacher_continue_device(states,viewers.float())


class SoATeacherTiledBoundaryTests(unittest.TestCase):
    """Execute production tiled gather/scatter with explicit CPU fake lanes."""
    def make(self):
        import torch as t
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch,DeviceDecision
        from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules
        rows,lanes,width=5,2,3
        store,actions=object(),object();calls=[]
        states=DeviceStateBatch(store,t.tensor([17,4,33,9,71]),t.tensor([3,9,5,2,7]),
            t.tensor([31,73,101,49,88]),t.tensor([True,False,True,True,True]),
            t.tensor([0,0,41,0,0],dtype=t.int32))
        observers=t.tensor([1,0,0,1,0],dtype=t.int64)
        class Queue:
            def __init__(self):
                self.loop=NS(errors=t.zeros(1,dtype=t.int32));self.ids=None;self.valid=None
            def capture_dispatch(self,source,operation):
                # Same valid-padding and active/error distinction as production
                # DeviceRuleLanes. This CPU host loop is an explicit test oracle.
                for start in range(0,rows,lanes):
                    actual=t.arange(start,start+lanes);self.valid=actual<rows
                    self.ids=t.where(self.valid,actual,0)
                    ids,valid=self.ids,self.valid
                    local=DeviceStateBatch(source.pool,t.where(valid,source.slots[ids],-1),
                        t.where(valid,source.generations[ids],0),t.where(valid,source.owners[ids],0),
                        valid&source.active[ids],t.where(valid,source.errors[ids],0))
                    operation(local,ids,valid)
            def scatter_tensor_device(self,local,output):
                output[self.ids[self.valid]]=local[self.valid]
            def scatter_states_device(self,local,output):
                for field in ('slots','generations','owners','active','errors'):
                    self.scatter_tensor_device(getattr(local,field),getattr(output,field))
        def continued(local,viewers,*,horizon):
            calls.append(('teacher',local,viewers.clone(),horizon))
            on=local.active&(local.errors==0)
            return replace(local,slots=t.where(on,local.slots+1000,-1),
                generations=t.where(on,local.generations+1,0),active=on)
        def raw(local,viewers):
            calls.append(('rule_decision',local,viewers.clone()))
            on=local.active&(local.errors==0)
            counts=t.where(on,2+local.owners.remainder(2),0)
            refs=local.owners[:,None]*10+t.tensor([[2,1,9]],dtype=t.int64)
            refs=t.where(on[:,None],refs,-1)
            valid=t.arange(width)[None,:]<counts[:,None]
            decision=DeviceDecision(local,actions,t.where(on,local.owners+5000,-1),
                t.where(on,local.generations+7,0),refs.to(t.int32),t.full((lanes,width),-1,dtype=t.int64),
                counts,valid,t.empty((lanes,0)),t.empty((lanes,width,0)),local.errors.clone())
            return decision,t.where(on,viewers,-1),decision.errors
        def step(local,who,decision,chosen,*,physical):
            calls.append(('step',local,who.clone(),decision,chosen.clone(),physical))
            on=local.active&(local.errors==0)
            picked=decision.action_refs.gather(1,chosen.clamp(0,width-1)[:,None])[:,0]
            return replace(local,slots=t.where(on,local.slots+picked,-1),
                generations=t.where(on,local.generations+1,0),active=on)
        inner=NS(teacher_continue_device=continued,rule_decision_device=raw,step_device=step,
                 _raw_capture_hook=None)
        rules=object.__new__(DeviceTiledSoARules)
        rules.rows=rows;rules.work_rows=lanes;rules.max_actions=width;rules.inference_rows=2
        rules.store=store;rules.action_store=actions;rules.t=t;rules.device=t.device('cpu')
        rules.require_teacher=True;rules.closed=False;rules.inner=inner;rules.raw_tape=None
        rules._keepalive=[];rules._sites={name:[Queue()] for name in ('teacher','rule_decision','step')}
        rules._site_index=dict(teacher=0,rule_decision=0,step=0)
        return rules,states,observers,calls

    def test_h3_continuation_tail_preserves_viewer_owner_lease_and_errors(self):
        import torch as t
        rules,states,viewers,calls=self.make()
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            output=rules.teacher_continue_device(states,viewers,horizon=3)
        self.assertIs(output.pool,rules.store)
        self.assertEqual(output.slots.tolist(),[1017,-1,-1,1009,1071])
        self.assertEqual(output.generations.tolist(),[4,0,0,3,8])
        self.assertEqual(output.owners.tolist(),states.owners.tolist())
        self.assertEqual(output.active.tolist(),[True,False,False,True,True])
        self.assertEqual(output.errors.tolist(),[0,0,41,0,0])
        self.assertEqual([row[2].tolist() for row in calls],[[1,0],[0,1],[0,1]])
        self.assertEqual(calls[-1][1].slots.tolist(),[71,-1])
        self.assertEqual(calls[-1][1].owners.tolist(),[88,0])
        self.assertEqual(calls[-1][1].active.tolist(),[True,False])
        self.assertTrue(all(row[3]==3 for row in calls))

    def test_raw_teacher_and_step_keep_refs_order_choices_owners_and_padding(self):
        import torch as t
        rules,states,viewers,calls=self.make()
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            decision,chosen,errors=rules.rule_decision_device(states,viewers)
            result=rules.step_device(states,viewers,decision,chosen,physical=False)
        self.assertIs(decision.states,states);self.assertIs(decision.action_pool,rules.action_store)
        self.assertEqual(decision.action_slots.tolist(),[5031,-1,-1,5049,5088])
        self.assertEqual(decision.action_generations.tolist(),[10,0,0,9,14])
        self.assertEqual(decision.counts.tolist(),[3,0,0,3,2])
        self.assertEqual(chosen.tolist(),[1,-1,-1,1,0])
        self.assertEqual(errors.tolist(),[0,0,41,0,0])
        self.assertEqual(decision.action_refs.tolist(),[[312,311,319],[-1,-1,-1],
            [-1,-1,-1],[492,491,499],[882,881,889]])
        self.assertEqual(decision.valid.tolist(),[[True]*3,[False]*3,[False]*3,[True]*3,[True,True,False]])
        self.assertEqual(decision.observations.shape,(5,0));self.assertEqual(decision.candidates.shape,(5,3,0))
        steps=[row for row in calls if row[0]=='step']
        self.assertEqual(len(steps),3)
        self.assertEqual([row[2].tolist() for row in steps],[[1,0],[0,1],[0,1]])
        self.assertEqual([row[4].tolist() for row in steps],[[1,-1],[-1,1],[0,1]])
        for row in steps:
            local,selected=row[1],row[3]
            self.assertEqual(selected.candidates.shape,(2,3,0))
            self.assertEqual(selected.observations.shape,(2,0))
            self.assertIs(selected.action_pool,rules.action_store)
            on=local.active&(local.errors==0)
            t.testing.assert_close(selected.states.owners[on],local.owners[on])
            t.testing.assert_close(selected.states.slots[on],local.slots[on])
            self.assertFalse(row[5])
        self.assertEqual(result.slots.tolist(),[328,-1,-1,500,953])
        self.assertEqual(result.errors.tolist(),[0,0,41,0,0])
        self.assertEqual(result.owners.tolist(),states.owners.tolist())
        self.assertIsNone(rules.inner._raw_capture_hook)

    def test_disabled_default_horizon_and_raw_transport_fail_closed(self):
        import inspect
        import torch as t
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules
        from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules
        self.assertIs(inspect.signature(DeviceSoARules).parameters['require_teacher'].default,False)
        self.assertIs(inspect.signature(DeviceTiledSoARules).parameters['require_teacher'].default,False)
        rules,states,viewers,calls=self.make()
        for horizon in (True,0,1,2,4):
            with self.assertRaisesRegex(ValueError,'horizon=3'):
                rules.teacher_continue_device(states,viewers,horizon=horizon)
        self.assertFalse(calls)
        rules.require_teacher=False
        for method in (rules.teacher_continue_device,rules.rule_decision_device):
            with self.assertRaisesRegex(RuntimeError,'absent'):method(states,viewers)
        self.assertFalse(calls)
        rules.require_teacher=True
        with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
            decision,chosen,_=rules.rule_decision_device(states,viewers)
            malformed=replace(decision,candidates=t.zeros((5,3,1)))
            with self.assertRaisesRegex(ValueError,'internal raw teacher'):
                rules.step_device(states,viewers,malformed,chosen)
        self.assertFalse(any(row[0]=='step' for row in calls))


if __name__=='__main__':unittest.main()
