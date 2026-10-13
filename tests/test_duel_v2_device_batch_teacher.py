"""Actual GPU teacher bridge fixtures; not complete-rule/search qualification."""
from pathlib import Path
import os,sys,unittest


def _prepare(state,actor):
    shared=[1,2,3]
    if state.get('early',False):
        actions=[{'type':'mulligan','cards':[]},{'type':'end_turn'}]
        indices=[]
    else:
        actions=[{'type':state.get('kind','attack'),'left':shared,'right':shared},
                 {'type':'concede'},{'type':'end_turn'},{'type':'play_card','card_id':'card'}]
        indices=[0,3]
    view={'legal_actions':[{'action':a} for a in actions]}
    return view,[a for a in actions if a['type']!='concede'],indices


def _preview(state,actor,action):
    return {'attack':8 if action['type']=='play_card' else 4,
            '攻击次数':2,'创生花伤害':0.5,'counter':1}


def _simulate(state,actor,action):
    result=dict(state);result['observed']=action;result['version']+=1
    return result


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES)
    from tests import test_duel_v2_device_frozen_rules as old
    path=os.environ.get('NTE_DEVICE_BATCH_TEACHER_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 export required')
    names=('acting_side','finished','terminal_value','public_legal_decision',
           'rule_decision','apply_action','simulate_action','clean','sample_world','teacher_continue')
    entries={name:getattr(old,'_'+name) for name in names}
    entries.update(teacher_prepare=_prepare,teacher_preview=_preview,simulate_action=_simulate)
    return capture_program(entries,allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+
        (__name__,old.__name__),scope_character_ids=None,scope_card_ids=None)


class BatchTeacherContractTest(unittest.TestCase):
    def test_no_qualification_or_host_fallback(self):
        from app.modules.card_game.rl.batched_search.device_batch_teacher import DeviceBatchTeacherActions
        self.assertFalse(DeviceBatchTeacherActions.supports_full_duel_rules)
        self.assertFalse(DeviceBatchTeacherActions.host_rule_callbacks)
        with self.assertRaises(TypeError):DeviceBatchTeacherActions(None)


class BatchTeacherCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_complete_legal_order_scalar_decode_early_tail_and_replay(self):
        import torch as t
        from contextlib import ExitStack
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_search.device_frozen_rules import DeviceFrozenRules
        from app.modules.card_game.rl.batched_search.device_batch_teacher import DeviceBatchTeacherActions
        from app.modules.card_game.rl.batched_search.device_action_leases import release_actions_device
        from tests.test_duel_v2_device_frozen_rules import _store,_graph
        from tests.test_duel_v2_device_rule_calls import _decode_row
        d=t.device('cuda',t.cuda.current_device());store=_store(d,slots=128);actions=_store(d,slots=128)
        self.addCleanup(store.close);self.addCleanup(actions.close)
        data=[dict(phase='playing',version=0,winner=None,actor='a',done=False),
              dict(phase='playing',version=0,winner=None,actor='b',done=False,early=True),
              dict(phase='playing',version=0,winner=None,actor='a',done=False)]
        compact=GraphLayout(2,512,1024,4096)
        with CudaVm(lower_program(program())) as vm:
            calls=DeviceRuleGraphCalls(vm,store,rows=3,lanes=2,seconds=60,compact_layout=compact);self.addCleanup(calls.close)
            pres=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30,seconds=60);self.addCleanup(pres.close)
            rules=DeviceFrozenRules(calls,pres,action_store=actions);self.addCleanup(rules.close)
            roots=store.put_device(_graph(data,d),t.arange(3,dtype=t.int64,device=d),t.tensor([True,True,False],device=d))
            actors=t.tensor([0,1,0],dtype=t.int64,device=d)
            teacher=DeviceBatchTeacherActions(rules,max_actions=4);self.addCleanup(teacher.close)
            step=calls.prepare_call('simulate_action');graph=t.cuda.CUDAGraph()
            with ExitStack() as stack:
                for name in ('cpu','numpy','item','tolist','__bool__'):
                    stack.enter_context(patch.object(t.Tensor,name,side_effect=AssertionError('Host teacher selection')))
                with t.cuda.graph(graph,capture_error_mode='thread_local'):
                    decision,chosen,errors=teacher.capture_device(roots,actors)
                    output=step.capture_device(roots,actors=actors,decision=decision,action_indices=chosen)
                    released=release_actions_device(decision,roots.active,roots.owners,protected=(roots,output))
            for _ in range(2):
                graph.replay();t.cuda.synchronize()
                self.assertEqual(errors.cpu().tolist(),[0,0,0],teacher.preview_statuses.cpu().tolist())
                self.assertEqual(chosen.cpu().tolist(),[2,0,-1])
                self.assertEqual(decision.counts.cpu().tolist(),[3,2,0])
                self.assertEqual(teacher.battle[0].cpu().tolist(),[True,False,True,False])
                self.assertEqual(output.errors.cpu().tolist(),[0,0,0])
                self.assertEqual(released.cpu().tolist(),[0,0,0])
                self.assertEqual(_decode_row(store,output,0,compact)['observed']['type'],'play_card')
                self.assertEqual(_decode_row(store,output,1,compact)['observed']['type'],'mulligan')
                self.assertEqual(_decode_row(store,roots,0,compact),data[0])
                self.assertEqual(int(actions.live.sum().cpu()),0)
                calls.release_device(output)
            roots.owners[0]+=10
            graph.replay();t.cuda.synchronize();self.assertNotEqual(int(errors[0].cpu()),0)
            calls.release_device(output);roots.owners[0]-=10
            graph.replay();t.cuda.synchronize();self.assertEqual(errors.cpu().tolist(),[0,0,0])
            calls.release_device(output)

    def test_unknown_kind_and_capacity_are_rejected(self):
        import torch as t
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_search.device_frozen_rules import DeviceFrozenRules
        from app.modules.card_game.rl.batched_search.device_batch_teacher import DeviceBatchTeacherActions
        from tests.test_duel_v2_device_frozen_rules import _store,_graph
        for width,kind in ((4,'unknown'),(2,'attack')):
            store=_store('cuda',slots=32);actions=_store('cuda',slots=32)
            self.addCleanup(store.close);self.addCleanup(actions.close)
            with CudaVm(lower_program(program())) as vm:
                calls=DeviceRuleGraphCalls(vm,store,rows=1,lanes=1,seconds=60,compact_layout=GraphLayout(1,512,1024,4096));self.addCleanup(calls.close)
                pres=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30);self.addCleanup(pres.close)
                rules=DeviceFrozenRules(calls,pres,action_store=actions);self.addCleanup(rules.close)
                roots=store.put_device(_graph([{'kind':kind}],vm.device),t.zeros(1,dtype=t.int64,device=vm.device),t.ones(1,dtype=t.bool,device=vm.device))
                teacher=DeviceBatchTeacherActions(rules,max_actions=width);self.addCleanup(teacher.close)
                graph=t.cuda.CUDAGraph()
                with t.cuda.graph(graph,capture_error_mode='thread_local'):
                    _,chosen,errors=teacher.capture_device(roots,t.zeros(1,dtype=t.int64,device=vm.device))
                graph.replay();t.cuda.synchronize()
                self.assertNotEqual(int(errors[0].cpu()),0);self.assertEqual(int(chosen[0].cpu()),-1)
                self.assertEqual(int(actions.live.sum().cpu()),0)


if __name__=='__main__':unittest.main()
