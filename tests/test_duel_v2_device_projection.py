"""Actual device projection of public oracle fixtures, zero game executions."""
from copy import deepcopy
from types import SimpleNamespace
import unittest


class DeviceProjectionTest(unittest.TestCase):
    def test_actual_cuda_public_projection_capture_and_physical_action_refs(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        import numpy as np
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS,validate_deck
        from app.modules.card_game.engine.duel_v2 import new_game,observe
        from app.modules.card_game.rl.seven_lineup import encode
        from app.modules.card_game.rl.batched_search.evaluator import runtime_matrix_keys
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.gpu_rule_projection import GraphProjectionKernel
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_device_vm import program
        fixtures=[];expected=[]
        for index,key in enumerate(runtime_matrix_keys('seven')):
            deck=validate_deck(deepcopy(next(d for d in STARTER_DECKS if d['id']==key)))
            view=observe(new_game(seed=20261011+index,skip_mulligan=True,first_side='a',decks={'a':deck,'b':deepcopy(deck)}),'a',include_previews=False)
            actions=[entry['action'] for entry in view['legal_actions'] if entry['action']['type']!='concede']
            fixtures.append((actions,view));expected.append(encode(view,actions))
        with CudaVm(lower_program(program())) as vm, GraphProjectionKernel(runtime='seven') as projection:
            base=vm.ingest(pack_states(fixtures,GraphLayout(7,32768,65536,262144)))
            active=torch.ones(7,dtype=torch.uint8,device=vm.device)
            statuses=torch.zeros((7,4),dtype=torch.int32,device=vm.device)
            def project():
                arena=base.fork()
                call=SimpleNamespace(arena=arena,executor=vm,outputs=arena.roots,statuses=statuses)
                return projection.project_decision(call,active=active)
            project();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with patch.object(torch.Tensor,'__bool__',side_effect=AssertionError('Host validation')):
                with torch.cuda.graph(graph):result=project()
                graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(result.errors.tolist(),[0]*7)
            for row,(x,c) in enumerate(expected):
                np.testing.assert_array_equal(result.observations.observations[row].cpu().numpy(),x)
                np.testing.assert_array_equal(result.candidates.candidates[row,:len(c)].cpu().numpy(),c)
                self.assertEqual(result.action_refs[row,:len(c)].shape,(len(c),))
            from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import encode_tokens_device
            token_graph=torch.cuda.CUDAGraph()
            with patch('app.modules.card_game.rl.batched_duel.gpu_seven_candidates.encode_tokens_device',
                       side_effect=AssertionError('Dense candidate allocation during raw projection')):
                with torch.cuda.graph(token_graph):
                    token_call=SimpleNamespace(arena=base,executor=vm,outputs=base.roots,statuses=statuses)
                    raw=projection.project_tokens_device(token_call,active=active)
            token_graph.replay();torch.cuda.synchronize()
            dense,errors=encode_tokens_device(raw.candidates,device=vm.device)
            self.assertEqual(errors.tolist(),[0]*7)
            self.assertEqual(raw.errors.tolist(),[0]*7)
            self.assertTrue(torch.equal(raw.action_refs,result.action_refs))
            self.assertTrue(torch.equal(raw.candidates.counts,result.candidates.counts))
            self.assertTrue(torch.equal(dense.candidates,result.candidates.candidates))
            self.assertTrue(torch.equal(raw.observations.observations,result.observations.observations))
            self.assertFalse(hasattr(raw.candidates,'candidates'))
            active.zero_();graph.replay();torch.cuda.synchronize()
            self.assertEqual(result.errors.tolist(),[0]*7)
            self.assertFalse(bool(result.candidates.mask.any()))
