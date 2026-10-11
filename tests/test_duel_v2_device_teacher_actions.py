"""Internal raw teacher actions; fixtures do not qualify real search capacity."""
from pathlib import Path
import os,sys,unittest
from app.modules.card_game.rl.batched_search.device_teacher_actions import DeviceTeacherActions


def _no_public(state,actor):
    raise ValueError('Public observation must not run in this teacher plan')


def _choose(state,actor):
    shared=[1,2,3]
    actions=[{'type':'end_turn'}, {'type':'concede','unknown_left':shared,'unknown_right':shared}]
    return actions,state.get('chosen',1)


def _simulate(state,actor,action):
    out=dict(state);out['observed']=action;out['version']+=1
    return out


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
    from tests import test_duel_v2_device_frozen_rules as old
    path=os.environ.get('NTE_DEVICE_TEACHER_ACTIONS_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 export required')
    names=('acting_side','finished','terminal_value','rule_decision','apply_action','simulate_action','clean','sample_world','teacher_continue')
    entries={name:getattr(old,'_'+name) for name in names}
    entries.update(public_legal_decision=_no_public,rule_decision=_choose,simulate_action=_simulate)
    return capture_program(entries,allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,old.__name__),scope_character_ids=None,scope_card_ids=None)


class TeacherActionsContractTest(unittest.TestCase):
    def test_internal_only_and_no_fake_qualification(self):
        self.assertFalse(DeviceTeacherActions.root_policy_compatible)
        self.assertFalse(DeviceTeacherActions.supports_full_duel_rules)
        self.assertFalse(DeviceTeacherActions.host_rule_callbacks)
        with self.assertRaises(TypeError):DeviceTeacherActions(None)


class TeacherActionsCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_full_raw_order_concede_unknown_alias_and_checked_step_binding(self):
        import torch
        from contextlib import ExitStack
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_search.device_frozen_rules import DeviceFrozenRules
        from app.modules.card_game.rl.batched_search.device_action_leases import release_actions_device
        from tests.test_duel_v2_device_frozen_rules import _store,_graph
        from tests.test_duel_v2_device_rule_calls import _decode_row
        device=torch.device('cuda',torch.cuda.current_device());store=_store(device,slots=64);actions=_store(device,slots=64)
        self.addCleanup(store.close);self.addCleanup(actions.close)
        with CudaVm(lower_program(program())) as vm:
            layout=GraphLayout(2,128,256,1024)
            calls=DeviceRuleGraphCalls(vm,store,rows=3,lanes=2,seconds=60,compact_layout=layout);self.addCleanup(calls.close)
            pres=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30,seconds=60);self.addCleanup(pres.close)
            rules=DeviceFrozenRules(calls,pres,action_store=actions);self.addCleanup(rules.close)
            data=[dict(phase='playing',version=0,winner=None,actor='a',done=False,chosen=1) for _ in range(3)]
            owners=torch.arange(3,dtype=torch.int64,device=device);active=torch.tensor([True,False,True],device=device)
            roots=store.put_device(_graph(data,device),owners,active);actors=torch.zeros(3,dtype=torch.int64,device=device)
            raw=DeviceTeacherActions(rules);self.addCleanup(raw.close)
            step=calls.prepare_call('simulate_action');graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
            with ExitStack() as stack:
                for name in ('cpu','numpy','item','tolist','__bool__'):stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Host selector')))
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                    decision,chosen,errors=raw.capture_device(roots,actors)
                    output=step.capture_device(roots,actors=actors,decision=decision,action_indices=chosen)
                    released=release_actions_device(decision,active,owners,protected=(roots,output))
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(errors.cpu().tolist(),[0,0,0]);self.assertEqual(output.errors.cpu().tolist(),[0,0,0])
            self.assertEqual(decision.counts.cpu().tolist(),[2,0,2]);self.assertEqual(chosen.cpu().tolist(),[1,-1,1]);self.assertEqual(released.cpu().tolist(),[0,0,0])
            for row in (0,2):
                decoded=_decode_row(store,output,row,layout);self.assertEqual(decoded['observed']['type'],'concede')
                self.assertIs(decoded['observed']['unknown_left'],decoded['observed']['unknown_right'])
                self.assertEqual(decoded['observed']['unknown_left'],[1,2,3])
                self.assertEqual(_decode_row(store,roots,row,layout),data[row])
            self.assertEqual(int(actions.live.sum().cpu()),0)
            calls.release_device(output)

    def test_capacity_and_teacher_index_fail_closed(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_search.device_frozen_rules import DeviceFrozenRules
        from tests.test_duel_v2_device_frozen_rules import _store,_graph
        store=_store('cuda',slots=32);actions=_store('cuda',slots=32);self.addCleanup(store.close);self.addCleanup(actions.close)
        with CudaVm(lower_program(program())) as vm:
            calls=DeviceRuleGraphCalls(vm,store,rows=1,lanes=1,seconds=30,compact_layout=GraphLayout(1,128,256,1024));self.addCleanup(calls.close)
            pres=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30,seconds=30);self.addCleanup(pres.close)
            rules=DeviceFrozenRules(calls,pres,action_store=actions);self.addCleanup(rules.close)
            roots=store.put_device(_graph([{'chosen':5}],vm.device),torch.zeros(1,dtype=torch.int64,device=vm.device),torch.ones(1,dtype=torch.bool,device=vm.device))
            raw=DeviceTeacherActions(rules,max_actions=1);self.addCleanup(raw.close);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,capture_error_mode='thread_local'):decision,_,errors=raw.capture_device(roots,torch.zeros(1,dtype=torch.int64,device=vm.device))
            graph.replay();torch.cuda.synchronize();self.assertNotEqual(int(errors[0].cpu()),0);self.assertEqual(int(actions.live.sum().cpu()),0);self.assertFalse(bool(decision.valid.any().cpu()))


if __name__=='__main__':unittest.main()
