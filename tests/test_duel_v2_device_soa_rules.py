"""Real CUDA integration gates; absence of hardware never becomes a pass."""
import os
import sys
import unittest

import numpy as np
import torch


class SoACudaBoundaryTests(unittest.TestCase):
    def test_cpu_cannot_be_selected_as_cuda_backend(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        with self.assertRaisesRegex(RuntimeError,'Actual CUDA'):
            CudaSoARules(None,None,device='cpu')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        with self.assertRaisesRegex(RuntimeError,'Actual CUDA'):
            NativeSoACUDAQuery(None,None,device='cpu')


@unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
class SoACudaRuleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.batched_duel.native_array_program import (
            load_native_array_program,capture_native_array_program)
        path=os.environ.get('NTE_SOA_FROZEN_PROGRAM')
        if path:
            program=load_native_array_program(path)
        elif sys.version_info[:2]==(3,10):
            program=capture_native_array_program()
        else:
            raise unittest.SkipTest('Set NTE_SOA_FROZEN_PROGRAM to the packaged frozen program')
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.device_soa_storage import DeviceSoAStore
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules
        cls.schema=compile_execution_schema()
        cls.executor=CudaSoARules(lower_program(program),cls.schema,limits=VmLimits(steps=48000000))
        cls.store=DeviceSoAStore(cls.schema,32)
        cls.actions=CompactDeviceGraphStore(slots=32,node_pages=128,edge_pages=128,payload_pages=256,
            node_page_bytes=16384,edge_page_bytes=16384,payload_page_bytes=4096,
            max_node_pages_per_slot=16,max_edge_pages_per_slot=16,max_payload_pages_per_slot=32)
        cls.rules=DeviceSoARules(cls.executor,cls.store,cls.actions,rows=2)

    @classmethod
    def tearDownClass(cls):
        torch.cuda.synchronize()
        cls.rules.close();cls.actions.close();cls.executor.close()

    def test_captured_mixed_actors_physical_step_and_both_sample_guards(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game,acting_side,apply_action,legal_actions
        from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
        from app.modules.card_game.rl.information_search import sample_world
        from app.modules.card_game.rl import seven_runtime
        decks={deck['id']:deck for deck in STARTER_DECKS}
        params=[dict(seed=1016+i,first_side='ab'[i],skip_mulligan=bool(i),decks={
            'a':decks['zhenhong'],'b':decks['lingke']}) for i in range(2)]
        roots=self.rules.initialize_device(params)
        torch.cuda.synchronize()
        self.assertEqual(roots.errors.cpu().tolist(),[0,0])
        expected=[new_game(**param) for param in params]
        self.assertEqual(unpack_game_states(self.store.gather_device(roots).to_host()),expected)
        seeds=torch.tensor([[1016,0,0,0,1],[1017,0,0,0,1]],dtype=torch.int64,device='cuda')
        indices=torch.zeros(2,dtype=torch.int64,device='cuda')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            meta=self.rules.metadata_device(roots)
            decision=self.rules.decision_device(roots,meta.actors)
            stepped=self.rules.step_device(roots,meta.actors,decision,indices,physical=True)
            sampled=self.rules.sample_world_device(roots,meta.actors,seeds)
        graph.replay();torch.cuda.synchronize()
        actors=['ab'.index(acting_side(state)) for state in expected]
        from app.modules.card_game.rl.batched_duel.device_rule_results import DeviceRuleResultReader
        phases=[DeviceRuleResultReader.phase_codes.index(state['phase']) for state in expected]
        self.assertNotEqual(actors[0],actors[1])
        self.assertNotEqual(phases[0],phases[1])
        self.assertEqual(meta.actors.cpu().tolist(),actors)
        self.assertEqual(meta.phases.cpu().tolist(),phases)
        self.assertEqual(meta.errors.cpu().tolist(),[0,0])
        self.assertEqual(decision.errors.cpu().tolist(),[0,0])
        self.assertEqual(stepped.errors.cpu().tolist(),[0,0])
        self.assertEqual(sampled.errors.cpu().tolist(),[0,0])
        after=[];worlds=[]
        for i,state in enumerate(expected):
            actor=acting_side(state)
            legal=[action for action in legal_actions(state,actor) if action['type']!='concede']
            self.assertEqual(int(decision.counts[i].cpu()),len(legal))
            after.append(apply_action(state,actor,legal[0]))
            worlds.append(sample_world(state,actor,(1<<256)+1016+i,runtime=seven_runtime))
        self.assertEqual(unpack_game_states(self.store.gather_device(stepped).to_host()),after)
        self.assertEqual(unpack_game_states(self.store.gather_device(sampled).to_host()),worlds)
        # Parent states survive both a physical sandbox commit and determinize.
        self.assertEqual(unpack_game_states(self.store.gather_device(roots).to_host()),expected)

    def test_sample_outer_guard_projection_only_active_mulligan_rows(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game,acting_side
        from app.modules.card_game.rl.batched_duel.device_rule_results import DeviceRuleResultReader
        from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
        from app.modules.card_game.rl.information_search import sample_world
        from app.modules.card_game.rl import seven_runtime
        decks={deck['id']:deck for deck in STARTER_DECKS}
        mulligan=DeviceRuleResultReader.phase_codes.index('mulligan')
        cases=(((False,True),[1,0]),((True,True),[0,0]),((False,False),[1,1]))
        graph=None
        bound=None
        recorded=[]
        seeds=torch.tensor([[2026+i,0,0,0,1] for i in range(2)],dtype=torch.int64,device='cuda')
        for skips,outer_mask in cases:
            params=[dict(seed=2026+i,first_side='ab'[i],skip_mulligan=skips[i],decks={
                'a':decks['zhenhong'],'b':decks['lingke']}) for i in range(2)]
            expected=[new_game(**param) for param in params]
            phases=[DeviceRuleResultReader.phase_codes.index(state['phase']) for state in expected]
            self.assertEqual(outer_mask,[int(code==mulligan) for code in phases],skips)
            roots=self.rules.initialize_device(params)
            torch.cuda.synchronize()
            self.assertEqual(roots.errors.cpu().tolist(),[0,0],skips)
            original=self.rules._project
            def spy(call,active):
                recorded.append(active)
                return original(call,active)
            self.rules._project=spy
            sampled=None
            try:
                if graph is None:
                    from dataclasses import replace
                    bound=replace(roots,**{name:getattr(roots,name).clone()
                        for name in ('slots','generations','owners','active','errors')})
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        meta=self.rules.metadata_device(bound)
                        sampled=self.rules.sample_world_device(bound,meta.actors,seeds)
                    sampled_result=sampled
                else:
                    for name in ('slots','generations','owners','active','errors'):
                        getattr(bound,name).copy_(getattr(roots,name))
                    sampled=sampled_result
                graph.replay();torch.cuda.synchronize()
                masks=[mask.detach().cpu().tolist() for mask in recorded]
                self.assertEqual(len(masks),4,skips)
                self.assertEqual(masks[0],[1,1],skips)
                self.assertEqual(masks[1],masks[0],skips)
                self.assertEqual(masks[2],outer_mask,skips)
                self.assertEqual(masks[3],masks[2],skips)
                self.assertEqual(sampled.errors.cpu().tolist(),[0,0],skips)
                worlds=[]
                for i,state in enumerate(expected):
                    worlds.append(sample_world(state,acting_side(state),(1<<256)+2026+i,runtime=seven_runtime))
                self.assertEqual(unpack_game_states(self.store.gather_device(sampled).to_host()),worlds,skips)
            finally:
                self.rules._project=original
                self.rules.release_device(roots)
                if sampled is not None:
                    self.rules.release_device(sampled)

    def test_fixed_viewer_value_allows_empty_legal_actions(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game, apply_action
        from app.modules.card_game.rl import seven_runtime
        from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states, unpack_game_states
        decks={deck['id']:deck for deck in STARTER_DECKS}
        params=[dict(seed=2026+i,first_side='ab'[i],skip_mulligan=True,
                     decks={'a':decks['starter'],'b':decks['weave-rush']}) for i in range(2)]
        initial=[new_game(**p) for p in params]
        terminal=[apply_action(state,'ab'[i],{'type':'concede'}) for i,state in enumerate(initial)]
        for states,sides,terminal_values in ((initial,'ba',[0.,0.]),(terminal,'aa',[-1.,1.])):
            owners=torch.arange(2,dtype=torch.int64,device='cuda')
            active=torch.ones(2,dtype=torch.bool,device='cuda')
            packed=pack_game_states(states,self.schema).to_torch('cuda')
            roots=self.store.put_device(packed,owners,active)
            viewers=torch.tensor(['ab'.index(s) for s in sides],dtype=torch.int64,device='cuda')
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    value=self.rules.value_observation_device(roots,viewers)
                for _ in range(2):
                    graph.replay();torch.cuda.synchronize()
                    self.assertEqual(value.errors.cpu().tolist(),[0,0])
                    self.assertEqual(value.terminal.cpu().tolist(),[states is terminal]*2)
                    self.assertEqual(value.terminal_value.cpu().tolist(),terminal_values)
                    for row,side in enumerate(sides):
                        actions,(expected,_) = seven_runtime.decision(states[row],side)
                        self.assertEqual(actions,[])
                        np.testing.assert_array_equal(value.observations[row].cpu().numpy(),expected)
                    self.assertEqual(unpack_game_states(self.store.gather_device(roots).to_host()),states)
            finally:
                self.rules.release_device(roots)


if __name__ == '__main__':
    unittest.main()
