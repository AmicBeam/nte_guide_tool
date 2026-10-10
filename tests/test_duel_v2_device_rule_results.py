"""Actual formal metadata result decoding; zero full games/capacity evidence."""
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
import unittest


class DeviceRuleResultContractTest(unittest.TestCase):
    def test_reader_rejects_host_callback_or_dictionary(self):
        from app.modules.card_game.rl.batched_duel.device_rule_results import DeviceRuleResultReader
        for value in (None,{},lambda:None):
            with self.assertRaises(TypeError):DeviceRuleResultReader(value)
        self.assertFalse(DeviceRuleResultReader.supports_full_duel_rules)


class DeviceRuleResultCudaTest(unittest.TestCase):
    def test_actual_metadata_refs_capture_masks_and_errors_are_device_owned(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required; no metadata fallback')
        from app.modules.card_game.engine.duel_v2 import new_game,acting_side
        from app.modules.card_game.engine.duel_v2.state import finished
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.device_rule_results import DeviceRuleResultReader
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_device_vm import program
        states=[new_game(seed=111,skip_mulligan=True),new_game(seed=112),
                new_game(seed=113),new_game(seed=114)]
        states[1]['phase']='choice';states[1]['pending_choice']={'side':'b'}
        states[2]['phase']='finished';states[2]['winner']='a';states[2]['version']=(1<<63)-1
        states[3]['phase']='finished';states[3]['winner']='draw'
        with CudaVm(lower_program(program())) as vm:
            arena=vm.ingest(pack_states(states,GraphLayout(4,16384,32768,131072)))
            actor_call=vm.launch(arena,'actor');actor_call.check()
            finished_call=vm.launch(arena,'finished');finished_call.check()
            with_reader=DeviceRuleResultReader(vm)
            self.addCleanup(with_reader.close)
            output=with_reader.prepare_metadata_workspace(4)
            active=torch.ones(4,dtype=torch.uint8,device=vm.device)
            errors=torch.zeros(4,dtype=torch.int32,device=vm.device)
            inputs=dict(actor_refs=actor_call.outputs,finished_refs=finished_call.outputs,
                        input_errors=errors,active=active,output=output)
            with_reader.metadata_into_device(arena,**inputs);torch.cuda.synchronize()
            before=[getattr(arena,k).clone() for k in ('nodes','edges','payload','counts','roots')]
            graph=torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                for name in ('cpu','numpy','item','tolist','__bool__'):
                    stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError(name)))
                with torch.cuda.graph(graph):meta=with_reader.metadata_into_device(arena,**inputs)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(meta.errors.tolist(),[0]*4)
            self.assertEqual(meta.actors.tolist(),[{'a':0,'b':1,None:-1}[acting_side(s)] for s in states])
            self.assertEqual(meta.phases.tolist(),[1,2,3,3])
            self.assertEqual(meta.winners.tolist(),[-2,-2,0,-1])
            self.assertEqual(meta.finished.tolist(),[finished(s) for s in states])
            self.assertEqual(meta.versions.tolist(),[s['version'] for s in states])
            for name,value in zip(('nodes','edges','payload','counts','roots'),before):
                self.assertTrue(torch.equal(getattr(arena,name),value))
            errors[0]=108;active[1]=0
            actor_call.outputs[2]=-(1<<31) # Bad frozen output ref is not decoded as a side.
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(meta.errors.tolist(),[108,0,150,0])
            self.assertEqual(meta.actors.tolist(),[-1,-1,-1,-1])
            bad_output=replace(output,meta=replace(output.meta,versions=output.meta.versions[:1]))
            with self.assertRaises(ValueError):
                with_reader.metadata_into_device(arena,**{**inputs,'output':bad_output})
