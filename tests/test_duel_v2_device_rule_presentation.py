"""Actual metadata/decision plans over oracle fixtures; zero full games."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import unittest


class DeviceRulePresentationContractTest(unittest.TestCase):
    def test_requires_actual_shared_gpu_calls(self):
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        for value in (None,{},lambda:None):
            with self.assertRaises(TypeError):
                DeviceRulePresentation(value,max_actions=800,max_output_bytes=1024)
        self.assertFalse(DeviceRulePresentation.supports_full_duel_rules)


class DeviceRulePresentationCudaTest(unittest.TestCase):
    def test_device_cohort_metadata_public_tokens_action_refs_and_release(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        import numpy as np
        from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game,acting_side,observe
        from app.modules.card_game.rl.seven_lineup import encode
        from app.modules.card_game.rl.batched_search.evaluator import runtime_matrix_keys
        from app.modules.card_game.rl.batched_duel.codec import pack_states,unpack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout,PackedStateBatch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,DeviceGraph
        from app.modules.card_game.rl.batched_duel.gpu_rule_program import capture_gpu_rule_program,verify_source_hashes
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from app.modules.card_game.rl.batched_duel.device_rule_calls import DeviceRuleGraphCalls
        from app.modules.card_game.rl.batched_duel.device_rule_presentation import DeviceRulePresentation
        from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import encode_tokens_device,RawCandidateTokens
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        fixture=os.environ.get('NTE_GPU_RULE_PRESENTATION_PROGRAM')
        if fixture:
            program=FrozenRuleProgram.from_json(Path(fixture).read_text());verify_source_hashes(program)
        elif sys.version_info[:2]==(3,10):
            program=capture_gpu_rule_program(['acting_side','finished','public_legal_decision','clean'],runtime='seven')
        else:self.fail('Explicit CPython3.10 frozen presentation fixture required')
        keys=runtime_matrix_keys('seven');decks={deck['id']:deck for deck in STARTER_DECKS}
        states=[new_game(seed=20261011+i,first_side='a',skip_mulligan=True,
                        decks={'a':deepcopy(decks[key]),'b':deepcopy(decks[keys[(i+1)%len(keys)]])})
                for i,key in enumerate(keys)]
        def pool(device):
            return CompactDeviceGraphStore(slots=32,node_pages=1024,edge_pages=1024,payload_pages=1024,
                node_page_bytes=4096,edge_page_bytes=4096,payload_page_bytes=4096,
                max_node_pages_per_slot=64,max_edge_pages_per_slot=64,max_payload_pages_per_slot=32,device=device)
        with CudaVm(lower_program(program),guard_standalone_entities=True) as vm:
            source=vm.ingest(pack_states(states,GraphLayout(len(states),16384,32768,131072)))
            with pool(vm.device) as roots_pool,pool(vm.device) as actions_pool:
                owners=torch.arange(len(states),dtype=torch.int64,device=vm.device)
                active=torch.ones(len(states),dtype=torch.bool,device=vm.device)
                roots=roots_pool.put_device(source,owners,active)
                with DeviceRuleGraphCalls(vm,roots_pool,rows=len(states),lanes=3,seconds=120,
                                         compact_layout=GraphLayout(3,16384,32768,131072)) as calls:
                    presentation=DeviceRulePresentation(calls,max_actions=800,max_output_bytes=1<<30,seconds=120)
                    self.addCleanup(presentation.close)
                    metadata_plan=presentation.prepare_metadata()
                    decision_plan=presentation.prepare_decision(actions_pool)
                    outer=CudaConditionalWhile(device=vm.device,max_ticks=2,seconds=300)
                    self.addCleanup(outer.close)
                    pending=torch.ones(1,dtype=torch.uint8,device=vm.device)
                    viewers=torch.tensor([0 if acting_side(s)=='a' else 1 for s in states],dtype=torch.int64,device=vm.device)
                    captured=[]
                    def body():
                        meta=metadata_plan.capture_device(roots)
                        decision=decision_plan.capture_device(roots,viewers)
                        captured.extend((meta,decision));pending.zero_()
                    outer.capture(pending,body,initialize=lambda:pending.fill_(1));outer.replay();torch.cuda.synchronize()
                    meta,decision=captured
                    self.assertEqual(meta.errors.tolist(),[0]*len(states))
                    self.assertEqual(decision.errors.tolist(),[0]*len(states))
                    self.assertEqual(outer.errors.item(),0)
                    self.assertEqual(metadata_plan.lanes.loop.errors.item(),0)
                    self.assertEqual(decision_plan.lanes.loop.errors.item(),0)
                    for row,state in enumerate(states):
                        view=observe(state,acting_side(state),include_previews=False)
                        legal=[entry['action'] for entry in view['legal_actions'] if entry['action']['type']!='concede']
                        x,c=encode(view,legal)
                        np.testing.assert_array_equal(decision.observations[row].cpu().numpy(),x)
                        # Explicit post-batch oracle decoding; no dense full-B allocation in the plan.
                        from dataclasses import fields
                        raw=RawCandidateTokens(**{f.name:getattr(decision.candidates.tokens,f.name)[row:row+1].contiguous()
                                                  for f in fields(RawCandidateTokens) if f.name!='request_ids'},request_ids=())
                        dense,errors=encode_tokens_device(raw,device=vm.device)
                        self.assertEqual(errors.item(),0)
                        np.testing.assert_array_equal(dense.candidates[0,:len(c)].cpu().numpy(),c)
                        self.assertEqual(decision.counts[row].item(),len(legal))
                    self.assertEqual(actions_pool.live.sum().item(),len(states))
                    presentation.release_decision_device(decision)
                    roots_pool.release_device(roots)
                    torch.cuda.synchronize()
                    self.assertEqual(actions_pool.live.sum().item(),0)
                    self.assertEqual(roots_pool.live.sum().item(),0)
