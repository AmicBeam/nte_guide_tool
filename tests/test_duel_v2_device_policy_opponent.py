"""Policy opponents: own observations, exact masked RNG, and device full32."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class PolicyOpponentTest(unittest.TestCase):
    def make(self, device='cpu', grouped=True):
        import torch as t
        from tests.test_duel_v2_device_search import TensorTreeFixture, FixtureInference, InjectiveFixtureKeys
        from app.modules.card_game.rl.batched_search.device_random import DevicePCG64
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        t.set_num_threads(1)
        b=4;rules=TensorTreeFixture(b,device,actions=3)
        bank=DevicePCG64([np.random.PCG64(913+i).state for i in range(b)],
            device=str(rules.device),max_rows=b,max_draws=3,max_choices=3,
            allow_cpu_reference=device=='cpu')
        stats=DeviceStatisticsPool(b*33,3,device=str(rules.device),max_batch=b,
            max_path_len=4,allow_cpu_reference=device=='cpu')
        search=DeviceGumbelSearch(rules,FixtureInference(),FixtureInference(),stats,
            InjectiveFixtureKeys(),rows=b,max_actions=3,simulations=32,max_depth=4,
            terminal_horizon=0,teacher_mode='network_value',opponent_mode='policy',
            random=bank,device=str(rules.device),allow_cpu_reference=device=='cpu',
            allow_synthetic=True,group_ready_leaves=grouped,reuse_interior_observation=True)
        roots=rules.roots();roots.active[2]=False
        seeds=t.zeros((b,5),dtype=t.int64,device=rules.device);seeds[:,4]=1
        ids=t.arange(b*2,dtype=t.int64,device=rules.device).reshape(b,2)
        return search,rules,bank,roots,seeds,ids

    def finish(self, search):
        for _ in range(2000):
            if not bool(search.unfinished_device.any()):return search.result_device()
            search.tick()
        self.fail('Finite fixture did not finish')

    def test_cpu_full32_matches_numpy_opponent_draws_and_grouping(self):
        import torch as t
        results=[]
        for grouped in (False,True):
            search,rules,bank,roots,seeds,ids=self.make(grouped=grouped)
            self.addCleanup(bank.close)
            generators=[np.random.Generator(np.random.PCG64(913+i)) for i in range(4)]
            draw_count=[0]*4
            categorical=bank.categorical
            def checked(probs,counts,active):
                result=categorical(probs,counts,active)
                for row in range(4):
                    if not active[row]:continue
                    n=int(counts[row]);p=probs[row,:n].numpy()
                    self.assertEqual(int(result.indices[row]),int(generators[row].choice(n,p=p)))
                    draw_count[row]+=1
                return result
            bank.categorical=checked
            def no_teacher(*args,**kwargs):raise AssertionError('Rule teacher called')
            rules.rule_decision_device=rules.teacher_continue_device=no_teacher
            search.initialize(roots,rules.viewer,seeds,ids,ids)
            actual=self.finish(search)
            self.assertEqual(actual.completed.tolist(),[32,32,0,32])
            self.assertEqual(actual.errors.tolist(),[0]*4)
            self.assertEqual(actual.complete.tolist(),[True,True,False,True])
            self.assertGreater(min(draw_count[0],draw_count[1],draw_count[3]),0)
            self.assertEqual(draw_count[2],0)
            self.assertEqual(bank.export_numpy_states(),tuple(g.bit_generator.state for g in generators))
            results.append([getattr(actual,n).clone() for n in ('actions','visits','policy')])
        for a,b in zip(*results):t.testing.assert_close(a,b)

    def test_opponent_queries_use_acting_player_model_and_observation(self):
        import torch as t
        search,rules,bank,roots,seeds,ids=self.make();self.addCleanup(bank.close)
        search.initialize(roots,rules.viewer,seeds,ids,ids)
        # Force one real opponent stage after allocating hypothetical worlds.
        search.tick()
        for _ in range(50):
            search.tick()
            from app.modules.card_game.rl.batched_search.device_search import OPPONENT
            if bool((search.phase==OPPONENT).any()):break
        captured=[];original=search._policy_prediction
        def predict(decision,actors,mask):
            expected=ids.gather(1,actors.clamp(0,1)[:,None])[:,0]
            seen=search.inference.predict_device
            def inspect(x,c,valid,model_ids,context,**kw):
                if bool(mask.any()):
                    t.testing.assert_close(model_ids[mask],expected[mask])
                    t.testing.assert_close(x[mask,2].long(),actors[mask])
                    captured.append(True)
                return seen(x,c,valid,model_ids,context,**kw)
            with patch.object(search.inference,'predict_device',side_effect=inspect):
                return original(decision,actors,mask)
        with patch.object(search,'_policy_prediction',side_effect=predict):search.tick()
        self.assertTrue(captured)

    def test_no_root_noise_still_resets_opponent_rng_each_real_decision(self):
        import torch as t
        from tests.test_duel_v2_device_games import RecordingSink, FixtureWorldSeeds, DeviceGameLoopTest
        from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
        search,rules,bank,roots,seeds,ids=self.make();self.addCleanup(bank.close)
        with self.assertRaisesRegex(ValueError,'stochastic search'):
            DeviceGameLoop(rules,search,RecordingSink(4,rules.device),FixtureWorldSeeds(),
                max_decisions=2,device='cpu',allow_cpu_reference=True,allow_synthetic=True)
        resets=[]
        class Plan:
            def reset_device(self,steps,actors,active):
                resets.append((steps.clone(),active.clone()))
                return t.zeros(4,dtype=t.int32)
        loop=DeviceGameLoop(rules,search,RecordingSink(4,rules.device),FixtureWorldSeeds(),
            max_decisions=2,random_plan=Plan(),device='cpu',allow_cpu_reference=True,allow_synthetic=True)
        helper=DeviceGameLoopTest()
        helper.finish_cpu(loop,roots,t.arange(4,dtype=t.int64),ids,t.ones((4,2),dtype=t.bool))
        self.assertEqual([int(steps[0]) for steps,active in resets],[0,1])
        self.assertTrue(all(not bool(active[2]) for steps,active in resets))

    def test_actual_cuda_full32_policy_opponents_match_cpu(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        ref,rules,bank,roots,seeds,ids=self.make();self.addCleanup(bank.close)
        ref.initialize(roots,rules.viewer,seeds,ids,ids);expected=self.finish(ref)
        actual,device_rules,device_bank,device_roots,device_seeds,device_ids=self.make('cuda')
        self.addCleanup(device_bank.close)
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        runner=CudaConditionalWhile(device=str(actual.device),max_ticks=3000,seconds=120)
        self.addCleanup(runner.close)
        actual.statistics.bind_capture_streams(runner.capture_stream,runner.body_stream)
        with actual.statistics.capture_scope():
            runner.capture(actual.unfinished_device,actual.tick,
                initialize=lambda:actual.initialize(device_roots,device_rules.viewer,
                    device_seeds,device_ids,device_ids))
        runner.replay();t.cuda.synchronize();got=actual.result_device()
        for name in ('actions','policy','visits','completed','complete','errors','information_keys'):
            t.testing.assert_close(getattr(got,name).cpu(),getattr(expected,name))
        self.assertEqual(device_bank.export_numpy_states(),bank.export_numpy_states())


class OriginalSearchAuditTest(unittest.TestCase):
    def test_original_decision_is_archived_before_guidance_and_audit_errors_block_commit(self):
        import torch as t
        from dataclasses import replace
        from tests.test_duel_v2_device_games import DeviceGameLoopTest, RecordingSink
        for reject in (False, True):
            helper=DeviceGameLoopTest()
            loop,rules,sink,roots,seeds,ids,train=helper.make(max_decisions=1)
            audit=RecordingSink(2,rules.device,capacity=0 if reject else 999)
            loop.search_audit_sink=audit
            class Guide:
                def capture_result(self,states,actors,result,model_ids,active):
                    self.original=result.actions.clone()
                    return replace(result,actions=t.zeros_like(result.actions))
            guide=Guide();loop.root_guide=guide
            result=helper.finish_cpu(loop,roots,seeds,ids,train)
            self.assertEqual(audit.begins.tolist(),[1,1]);self.assertEqual(audit.finishes.tolist(),[1,1])
            if reject:
                self.assertEqual(result.steps.tolist(),[0,0]);self.assertEqual(sink.appends.tolist(),[0,0])
                self.assertTrue(bool(result.errors.ne(0).all()))
            else:
                t.testing.assert_close(audit.actions,guide.original)
                self.assertEqual(sink.actions.tolist(),[0,0]);self.assertEqual(result.steps.tolist(),[1,1])


class PackedGuideQueueTest(unittest.TestCase):
    def test_actual_cuda_sparse_tail_preserves_inactive_and_foreign_rows(self):
        import torch as t
        if not t.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_search.device_zhenhong_guide import PackedGuideLanes
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        b=17;d=t.device('cuda',t.cuda.current_device())
        ids=t.arange(b,dtype=t.int64,device=d);on=t.zeros(b,dtype=t.bool,device=d)
        on[t.tensor([0,2,7,8,10,16],device=d)]=True
        errors=t.zeros(b,dtype=t.int32,device=d);errors[7]=99
        states=DeviceStateBatch(object(),ids+1,ids*2,ids+900,on,errors)
        queue=PackedGuideLanes(b,4,device=str(d),seconds=60);self.addCleanup(queue.close)
        output=t.full((b,),-999,dtype=t.int64,device=d)
        pending=t.ones(1,dtype=t.uint8,device=d)
        outer=CudaConditionalWhile(device=str(d),max_ticks=2,seconds=60);self.addCleanup(outer.close)
        def operation(local,rows,valid):
            values=(local.slots+local.owners).contiguous()
            queue.scatter_tensor_device(values,output)
        def body():
            queue.capture_dispatch(states,operation);pending.zero_()
        outer.capture(pending,body,initialize=lambda:pending.fill_(1))
        outer.replay();t.cuda.synchronize()
        expected=[-999]*b
        for i in (0,2,8,10,16):expected[i]=2*i+901
        self.assertEqual(output.cpu().tolist(),expected)
        self.assertEqual(int(queue.processed.cpu()[0]),5)
        self.assertEqual(int(queue.peak.cpu()[0]),4)
        self.assertEqual(int(queue.loop.counter.cpu()[0]),2)
