"""Real native-rule lane parity; this is not high-concurrency qualification."""
from dataclasses import fields
import os
import sys
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
class TiledSoARuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.batched_duel.native_array_program import load_native_array_program, capture_native_array_program
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
        from app.modules.card_game.rl.batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules
        path=os.environ.get('NTE_SOA_FROZEN_PROGRAM')
        if not path and sys.version_info[:2] != (3,10):
            raise unittest.SkipTest('Packaged frozen program required')
        program=load_native_array_program(path) if path else capture_native_array_program()
        cls.schema=compile_execution_schema()
        stack_runs=os.environ.get('NTE_SOA_STACK_RUNS','0')
        if stack_runs not in ('0','1'):raise ValueError('Explicit stack-run fixture flag required')
        cls.executor=CudaSoARules(lower_program(program),cls.schema,limits=VmLimits(steps=48000000),stack_runs=stack_runs=='1')
        cls.store=DeviceSparseSoAStore(cls.schema,32)
        cls.actions=CompactDeviceGraphStore(slots=32,node_pages=128,edge_pages=128,payload_pages=256,
            node_page_bytes=16384,edge_page_bytes=16384,payload_page_bytes=4096,
            max_node_pages_per_slot=16,max_edge_pages_per_slot=16,max_payload_pages_per_slot=32)
        cls.rules=DeviceTiledSoARules(cls.executor,cls.store,cls.actions,rows=3,work_rows=2,
                                    inference_rows=2,seconds=240)

    @classmethod
    def tearDownClass(cls):
        torch.cuda.synchronize()
        cls.rules.close();cls.actions.close();cls.store.close();cls.executor.close()

    def test_initialize_mixed_phases_tail_step_sample_and_fixed_viewer(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game, acting_side, apply_action
        from app.modules.card_game.rl import seven_runtime
        from app.modules.card_game.rl.information_search import sample_world
        from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
        from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
        from app.modules.card_game.rl.batched_search.device_action_leases import release_actions_device
        from scripts.validate_duel_v2_native_soa import compare_boundary
        decks={d['id']:d for d in STARTER_DECKS}
        params=[dict(seed=1700+i,first_side='aba'[i],skip_mulligan=bool(i),
                     decks={'a':decks[a],'b':decks[b]})
                for i,(a,b) in enumerate((('zhenhong','lingke'),('starter','weave-rush'),('quick-rush','murk')))]
        expected=[new_game(**p) for p in params]
        roots=self.rules.initialize_device(params)
        self.assertEqual(roots.errors.cpu().tolist(),[0,0,0])
        self.assertEqual(self.rules.inner.rows,2)
        self.assertEqual(self.rules.inner._shared_batch.rows,2)
        for left,right in zip(unpack_game_states(self.store.gather_device(roots).to_host()),expected):
            compare_boundary(left,right,'tiled native initialization')
        seeds=torch.tensor([[1700+i,0,0,0,1] for i in range(3)],dtype=torch.int64,device='cuda')
        choices=torch.zeros(3,dtype=torch.int64,device='cuda')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            meta=self.rules.metadata_device(roots)
            decision=self.rules.decision_device(roots,meta.actors)
            stepped=self.rules.step_device(roots,meta.actors,decision,choices,physical=True)
            sampled=self.rules.sample_world_device(roots,meta.actors,seeds)
            observed=self.rules.value_observation_device(roots,1-meta.actors)
        for replay in range(2):
            graph.replay();torch.cuda.synchronize()
            for result in (meta,decision,stepped,sampled,observed):
                self.assertEqual(result.errors.cpu().tolist(),[0,0,0],(replay,type(result).__name__))
            actual_step=unpack_game_states(self.store.gather_device(stepped).to_host())
            actual_world=unpack_game_states(self.store.gather_device(sampled).to_host())
            for row,state in enumerate(expected):
                actor=acting_side(state);legal,(x,c)=seven_runtime.decision(state,actor)
                self.assertEqual(int(decision.counts[row].cpu()),len(legal))
                np.testing.assert_array_equal(decision.observations[row].cpu().numpy(),x)
                _,(vx,_)=seven_runtime.decision(state,'b' if actor=='a' else 'a')
                np.testing.assert_array_equal(observed.observations[row].cpu().numpy(),vx)
                compare_boundary(actual_step[row],apply_action(state,actor,legal[0]),'tiled physical action')
                compare_boundary(actual_world[row],sample_world(state,actor,(1<<256)+1700+row,runtime=seven_runtime),
                                 'tiled immutable sampled world')
            for start in (0,2):
                stop=min(3,start+2)
                selected=encoder.RawCandidateTokens(**{f.name:getattr(decision.candidates.tokens,f.name)[start:stop].contiguous()
                    for f in fields(encoder.RawCandidateTokens) if f.name!='request_ids'})
                encoded,errors=encoder.encode_tokens_device(selected,device='cuda',
                    active=torch.ones(stop-start,dtype=torch.bool,device='cuda'),template_table=self.rules.templates)
                self.assertEqual(errors.cpu().tolist(),[0]*(stop-start))
                for row in range(start,stop):
                    _,(_,expected_c)=seven_runtime.decision(expected[row],acting_side(expected[row]))
                    np.testing.assert_array_equal(encoded.candidates[row-start,:len(expected_c)].cpu().numpy(),expected_c)
            for left,right in zip(unpack_game_states(self.store.gather_device(roots).to_host()),expected):
                compare_boundary(left,right,'physical roots survive tiled work')
            self.rules.release_device(stepped);self.rules.release_device(sampled)
            errors=release_actions_device(decision,roots.active,roots.owners)
            self.assertEqual(errors.cpu().tolist(),[0,0,0])
            self.assertEqual(int(self.store.live.sum().cpu()),3)
        seen=[q.root_visits.cpu().tolist() for q in self.rules._queues if int(q.total_rows.cpu()[0])]
        self.assertTrue(seen)
        self.assertTrue(all(all(v>0 for v in counts) for counts in seen))


if __name__ == '__main__':
    unittest.main()
