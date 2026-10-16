"""Disposable-state transition parity; CUDA scratch isolation is separate."""
import unittest
from copy import deepcopy
import json
import os
from pathlib import Path
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side,observe
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl.batched_duel.gpu_owned_transitions import scratch_apply_action,scratch_simulate_action
from app.modules.card_game.rl.league_rollout import clean


def scratch_apply_packet(packet):
    return scratch_apply_action(packet[0],packet[1],packet[2])


def scratch_simulate_packet(packet):
    return scratch_simulate_action(packet[0],packet[1],packet[2])


def aliases(value):
    paths={};stack=[((),value)]
    while stack:
        path,item=stack.pop()
        if not isinstance(item,(list,tuple,dict)):continue
        old=paths.setdefault(id(item),[]);old.append(path)
        if len(old)>1:continue
        pairs=item.items() if isinstance(item,dict) else enumerate(item)
        stack.extend((path+(str(k),),v) for k,v in pairs)
    return sorted(sorted(group) for group in paths.values() if len(group)>1)


class OwnedTransitionTest(unittest.TestCase):
    def test_canonical_multistep_values_and_aliases_both_transition_modes(self):
        # Use actual public and test preset construction, independent of model files.
        from tests.test_duel_v2_policy_projection import trace_states
        checked=0
        for state in trace_states():
            if state['phase']=='finished':continue
            actor=acting_side(state);view=observe(state,actor,include_previews=False)
            legal=[x['action'] for x in view['legal_actions'] if x['action']['type']!='concede']
            action=legal[(checked*17)%len(legal)];original=deepcopy(state)
            for formal,owned in ((apply_action,scratch_apply_action),(simulate_action,scratch_simulate_action)):
                expected=formal(state,actor,action);private=deepcopy(state)
                actual=owned(private,actor,deepcopy(action))
                self.assertEqual(actual,expected);self.assertEqual(aliases(actual),aliases(expected))
                self.assertEqual(state,original)
            checked+=1
        self.assertGreater(checked,200)

    def test_json_and_entity_counter_normalization_match_clone(self):
        state=new_game(seed=813,skip_mulligan=True)
        state=json.loads(json.dumps(state));state['next_entity']=0
        actor=acting_side(state);action={'type':'end_turn'}
        expected=apply_action(state,actor,action)
        actual=scratch_apply_action(deepcopy(state),actor,action)
        self.assertEqual(actual,expected);self.assertEqual(aliases(actual),aliases(expected))
        self.assertGreater(actual['next_entity'],0)

    def test_invalid_action_failure_only_consumes_private_sandbox(self):
        state=new_game(seed=99,skip_mulligan=True);original=deepcopy(state)
        for action in ({'type':'play_card','card_id':'missing'},None):
            for fn in (apply_action,scratch_apply_action):
                with self.assertRaises(ValueError):fn(deepcopy(state),'a',action)
            self.assertEqual(state,original)

    def test_conflicting_entity_ids_match_canonical_failure(self):
        state=new_game(seed=98,skip_mulligan=True)
        ids=list(state['sides']['a']['characters'])
        state['sides']['a']['characters'][ids[1]]['entity_id']=state['sides']['a']['characters'][ids[0]]['entity_id']
        for fn in (apply_action,scratch_apply_action):
            with self.assertRaises(ValueError):fn(deepcopy(state),'a',{'type':'end_turn'})


class OwnedTransitionCudaTest(unittest.TestCase):
    def test_actual_cuda_disposable_scratch_preserves_source_and_aliases(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from tests.test_duel_v2_policy_projection import trace_states
        packets=[];seen=set()
        for state in trace_states(max_steps=24):
            phase=state['phase'];actor=acting_side(state)
            actions=[e['action'] for e in observe(state,actor,include_previews=False)['legal_actions'] if e['action']['type']!='concede']
            if phase not in seen:
                packets.append([deepcopy(state),actor,actions[-1]]);seen.add(phase)
            if {'choice','playing','mulligan'}.issubset(seen):break
        self.assertEqual(seen,{'choice','playing','mulligan'})
        packets.append(json.loads(json.dumps(packets[-1])))
        packets[-1][0]['next_entity']=0
        path=os.environ.get('NTE_OWNED_TRANSITION_PROGRAM')
        if path:program=FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf-8'))
        else:program=capture_program({'scratch_apply_packet':scratch_apply_packet,'scratch_simulate_packet':scratch_simulate_packet},
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),scope_character_ids=None,scope_card_ids=None)
        layout=GraphLayout(len(packets),262144,524288,33554432)
        with CudaVm(lower_program(program),limits=VmLimits(steps=48000000),guard_standalone_entities=True,
                    constant_lookup=True,compact_frame_init=True,mutable_lookup_cache=True,frozen_class_lookup=True) as vm:
            source=vm.ingest(pack_states(packets,layout));scratch=vm.ingest(pack_states([{}]*len(packets),layout))
            workspace=vm.prepare_device_workspace(scratch)
            fields=('nodes','edges','payload','counts','roots');saved={n:getattr(source,n).clone() for n in fields}
            active=t.ones(len(packets),dtype=t.uint8,device=vm.device)
            errors=t.zeros(len(packets),dtype=t.int32,device=vm.device)
            deadline=t.full((len(packets),),vm.clock_ns()+int(120e9),dtype=t.int64,device=vm.device)
            refs=source.roots[:,None].contiguous()
            for entry,formal in (('scratch_apply_packet',apply_action),('scratch_simulate_packet',simulate_action)):
                vm.load_graph_into_device(workspace,source,active=active,errors=errors)
                call=vm.launch_into_device(workspace,entry,argument_refs=refs,active=active,gpu_deadlines=deadline)
                t.cuda.synchronize();self.assertFalse(bool(errors.any().cpu()))
                actual=call.decode();expected=[formal(*packet) for packet in packets]
                self.assertEqual(actual,expected)
                for x,y in zip(actual,expected):self.assertEqual(aliases(x),aliases(y))
                for name in fields:self.assertTrue(t.equal(getattr(source,name),saved[name]))
            # A failed disposable arena is discarded; the next load must reset it.
            invalid=deepcopy(packets)
            for packet in invalid:
                packet[0]['next_entity']=0
                packet[2]={'type':'play_card','card_id':'missing'}
            bad=vm.ingest(pack_states(invalid,layout))
            bad_saved={n:getattr(bad,n).clone() for n in fields}
            vm.load_graph_into_device(workspace,bad,active=active,errors=errors)
            failed=vm.launch_into_device(workspace,'scratch_apply_packet',argument_refs=bad.roots[:,None].contiguous(),active=active,gpu_deadlines=deadline)
            t.cuda.synchronize();self.assertTrue(bool((failed.statuses[:,0]!=0).all().cpu()))
            for name in fields:self.assertTrue(t.equal(getattr(bad,name),bad_saved[name]))
            vm.load_graph_into_device(workspace,source,active=active,errors=errors)
            again=vm.launch_into_device(workspace,'scratch_apply_packet',argument_refs=refs,active=active,gpu_deadlines=deadline)
            self.assertEqual(again.decode(),[apply_action(*packet) for packet in packets])


if __name__=='__main__':unittest.main()
