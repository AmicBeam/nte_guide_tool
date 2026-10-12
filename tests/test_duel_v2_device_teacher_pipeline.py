"""Teacher pipeline lease regressions; fixture success is not real-rule proof."""
from pathlib import Path
import os,sys,unittest
from dataclasses import replace
from app.modules.card_game.rl.batched_search.device_teacher_pipeline import DeviceTeacherPipeline, LEASE_ERROR
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch


def _simulate(state, actor, action):
    if actor != state['actor']:
        raise ValueError('actor mismatch')
    result = dict(state)
    result['n'] += 1
    result['version'] += 1
    result['actor'] = 'b' if actor == 'a' else 'a'
    if result['n'] >= result['finish_at']:
        result.update(done=True, phase='finished', winner='a', actor=None)
    return result


def _public(state, observer):
    team = {'characters': [], 'front': None, 'last_front': None, 'hand': [],
            'discard': [], 'used': {'instant': False}, 'front_debuff': {}}
    sides = {'a': team, 'b': team}
    legal = [{'type': 'end_turn'}]
    view = {'viewer_side': observer, 'first_side': 'a', 'phase': state['phase'],
            'turn': 1, 'sides': sides, 'policy_state': {'sides': sides},
            'legal_actions': [{'action': legal[0]}]}
    return legal, view


def _choose(state, actor):
    return [{'type': 'end_turn'}], 0


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    from tests import test_duel_v2_device_frozen_rules as old
    path=os.environ.get('NTE_DEVICE_TEACHER_PIPELINE_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    names=('acting_side','finished','terminal_value','public_legal_decision','rule_decision','apply_action','clean','sample_world','teacher_continue')
    entries={name:getattr(old,'_'+name) for name in names};entries['simulate_action']=_simulate;entries['public_legal_decision']=_public;entries['rule_decision']=_choose
    return capture_program(entries,allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,old.__name__),scope_character_ids=None,scope_card_ids=None)


class TeacherPipelineLeaseTest(unittest.TestCase):
    def test_contract_requires_helpers_and_exact_horizon(self):
        with self.assertRaises(TypeError):DeviceTeacherPipeline(None,None,action_store=None)
        pipeline=object.__new__(DeviceTeacherPipeline);pipeline.closed=False;pipeline.captured=False
        for horizon in (True,2,4,3.0):
            with self.assertRaises(ValueError):pipeline.capture_device(None,None,horizon=horizon)
        self.assertFalse(DeviceTeacherPipeline.supports_full_duel_rules)

    def test_fixture_uses_canonical_actions_then_public_view_order(self):
        for actor in ('a','b'):
            actions,view=_public({'phase':'playing'},actor)
            self.assertIsInstance(actions,list)
            self.assertIsInstance(view,dict)
            self.assertEqual(view['viewer_side'],actor)
            self.assertEqual(actions,_choose({},actor)[0])
            self.assertEqual([row['action'] for row in view['legal_actions']],actions)

    def test_generation_zero_aliases_dead_lease_and_root_protection(self):
        import torch
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from tests.test_duel_v2_device_frozen_rules import _graph
        store=CompactDeviceGraphStore(slots=8,node_pages=32,edge_pages=32,payload_pages=32,node_page_bytes=256,edge_page_bytes=256,payload_page_bytes=512,max_node_pages_per_slot=8,max_edge_pages_per_slot=8,max_payload_pages_per_slot=8,device='cpu',allow_cpu_reference=True)
        self.addCleanup(store.close)
        roots=store.put_device(_graph([{'n':0}], 'cpu'),torch.tensor([7]),torch.tensor([True]))
        owned=store.put_device(_graph([{'n':1}], 'cpu'),torch.tensor([7]),torch.tensor([True]))
        def clone(s):return replace(s,**{n:getattr(s,n).clone() for n in ('slots','generations','owners','active','errors')})
        p=object.__new__(DeviceTeacherPipeline);p.t=torch;p.pool=store;p.current=clone(owned);alias=clone(owned);p._owned=[p.current,alias]
        self.assertEqual(owned.generations.tolist(),[0])
        p._release(p.current,torch.tensor([True]),roots)
        self.assertEqual(p.current.slots.tolist(),[-1]);self.assertEqual(alias.slots.tolist(),[-1]);self.assertEqual(int(store.live.sum()),1)
        p._release(owned,torch.tensor([True]),roots);self.assertEqual(p.current.errors.tolist(),[0])
        borrowed=clone(roots);p._owned.append(borrowed);p._release(borrowed,torch.tensor([True]),roots)
        self.assertEqual(int(store.live.sum()),1);self.assertEqual(borrowed.slots.tolist(),roots.slots.tolist())
        foreign=clone(roots);foreign.owners.add_(1);p._owned.append(foreign);p._release(foreign,torch.tensor([True]),roots)
        self.assertNotEqual(p.current.errors.tolist(),[0]);self.assertEqual(int(store.live.sum()),1)


class TeacherPipelineCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_three_steps_terminal_masks_padding_replay_and_foreign_owner(self):
        import torch
        from contextlib import ExitStack
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from tests.test_duel_v2_device_frozen_rules import _graph
        from tests.test_duel_v2_device_rule_calls import _decode_row
        d=torch.device('cuda',torch.cuda.current_device());rows=3
        data=[dict(phase='playing',version=0,winner=None,actor='a',done=False,n=0,finish_at=20),dict(phase='playing',version=0,winner=None,actor='b',done=False,n=0,finish_at=1),dict(phase='finished',version=4,winner='a',actor=None,done=True,n=9,finish_at=20)]
        def store():return CompactDeviceGraphStore(slots=128,node_pages=256,edge_pages=256,payload_pages=256,node_page_bytes=4096,edge_page_bytes=4096,payload_page_bytes=4096,max_node_pages_per_slot=8,max_edge_pages_per_slot=8,max_payload_pages_per_slot=8,device=d)
        physical,actions=store(),store();self.addCleanup(physical.close);self.addCleanup(actions.close)
        compact=GraphLayout(2,512,1024,8192)
        with CudaVm(lower_program(program())) as vm:
            calls=DeviceRuleGraphCalls(vm,physical,rows=rows,lanes=2,seconds=60,compact_layout=compact);self.addCleanup(calls.close)
            pres=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30,seconds=60);self.addCleanup(pres.close)
            pipeline=DeviceTeacherPipeline(calls,pres,action_store=actions,raw_teacher=getattr(self,'raw_teacher',False));self.addCleanup(pipeline.close)
            from app.modules.card_game.rl.batched_search.device_teacher_pipeline import _teacher_capture_sites
            self.assertEqual(pipeline.rules.prepared, _teacher_capture_sites(pipeline.raw_teacher))
            self.assertEqual(len(pipeline.teacher_action_sites), 3 if pipeline.raw_teacher else 0)
            if pipeline.raw_teacher:
                with self.assertRaisesRegex(RuntimeError, 'No remaining prepared rule_decision'):
                    pipeline.rules._queues['rule_decision'].consume()
            owners=torch.arange(rows,dtype=torch.int64,device=d);roots=physical.put_device(_graph(data,d,max_payload=512),owners,torch.ones(rows,dtype=torch.bool,device=d));viewers=torch.zeros(rows,dtype=torch.int64,device=d)
            records={name:torch.zeros((calls.lanes,4),dtype=torch.int32,device=d) for name in vm.image.entries}
            launch=vm.launch_into_device
            def traced(workspace,entry,**kwargs):
                call=launch(workspace,entry,**kwargs)
                bad=(call.statuses[:,0]!=0)&(records[entry][:,0]==0)
                records[entry].copy_(torch.where(bad[:,None],call.statuses,records[entry]))
                return call
            vm.launch_into_device=traced
            graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream(device=d);stream.wait_stream(torch.cuda.current_stream(d))
            with ExitStack() as stack:
                for name in ('cpu','numpy','item','tolist','__bool__'):stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Host selector')))
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):out=pipeline.capture_device(roots,viewers)
            for _ in range(3):
                graph.replay();torch.cuda.synchronize();self.assertEqual(out.errors.cpu().tolist(),[0]*rows, (pipeline.stage_errors.cpu().tolist(), [(site.decision.action_leases.errors.cpu().tolist(), site.choice.errors.cpu().tolist()) for site in pipeline.rules._teacher_sites], {name:value.cpu().tolist() for name,value in records.items()}))
                decoded=[_decode_row(physical,out,i,compact) for i in range(rows)]
                self.assertEqual([s['n'] for s in decoded],[3,1,9]);self.assertEqual([s['version'] for s in decoded],[3,1,4]);self.assertEqual(decoded[0]['actor'],'b')
                self.assertEqual([_decode_row(physical,roots,i,compact) for i in range(rows)],data)
                self.assertEqual(int(physical.live.sum().cpu()),6);self.assertEqual(int(actions.live.sum().cpu()),0)
            roots.owners[0]+=10;graph.replay();torch.cuda.synchronize();self.assertNotEqual(int(out.errors[0].cpu()),0)
            roots.owners[0]-=10;graph.replay();torch.cuda.synchronize();self.assertEqual(out.errors.cpu().tolist(),[0]*rows, (pipeline.stage_errors.cpu().tolist(), [(site.decision.action_leases.errors.cpu().tolist(), site.choice.errors.cpu().tolist()) for site in pipeline.rules._teacher_sites], {name:value.cpu().tolist() for name,value in records.items()}))
            self.assertEqual(int(physical.live.sum().cpu()),6);self.assertEqual(int(actions.live.sum().cpu()),0)


class TeacherPipelineRawCudaTest(TeacherPipelineCudaTest):
    raw_teacher=True


if __name__=='__main__':unittest.main()
