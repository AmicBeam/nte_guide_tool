"""Controlled GPU scratch dispatch; not real rules, games or 1600 acceptance."""
from contextlib import ExitStack
from unittest.mock import patch
import unittest


class DeviceRuleLanesContractTest(unittest.TestCase):
    def test_no_cpu_dispatch_fallback(self):
        from app.modules.card_game.rl.batched_duel.device_rule_lanes import DeviceRuleLanes
        with self.assertRaises(RuntimeError):DeviceRuleLanes(4,device='cpu')
        for rows,lanes in ((True,4),(0,4),(4,0)):
            with self.assertRaises(ValueError):DeviceRuleLanes(rows,lanes,device='cpu')
        self.assertFalse(DeviceRuleLanes.supports_full_duel_rules)


class DeviceRuleLanesCudaTest(unittest.TestCase):
    def test_nested_device_cohorts_preserve_partial_padding_errors_and_leases(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.device_rule_lanes import DeviceRuleLanes
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        device=torch.device('cuda',torch.cuda.current_device());rows=1633;pool=object()
        slots=torch.arange(rows,dtype=torch.int64,device=device)
        generations=torch.full((rows,),7,dtype=torch.int64,device=device)
        owners=torch.arange(rows,dtype=torch.int64,device=device)+11
        active=torch.ones(rows,dtype=torch.bool,device=device);active[14]=False
        errors=torch.zeros(rows,dtype=torch.int32,device=device);errors[3]=107
        states=DeviceStateBatch(pool,slots,generations,owners,active,errors)
        output=DeviceStateBatch(pool,slots.clone(),generations.clone(),owners.clone(),active.clone(),errors.clone())
        payload=torch.zeros((rows,3),dtype=torch.int64,device=device)
        lanes=DeviceRuleLanes(rows,32,device=device,seconds=30);self.addCleanup(lanes.close)
        outer=CudaConditionalWhile(device=device,max_ticks=3,seconds=30);self.addCleanup(outer.close)
        pending=torch.ones(1,dtype=torch.uint8,device=device)
        captures=[]
        def operation(local,ids,valid):
            captures.append('static capture')
            executable=local.active & (local.errors==0)
            # This is a controlled selector change, no rules are claimed.
            transformed=DeviceStateBatch(pool,torch.where(executable,local.slots+100,local.slots),
                local.generations,local.owners,local.active,local.errors)
            lanes.scatter_states_device(transformed,output)
            lanes.scatter_tensor_device(torch.stack((local.slots,local.owners,local.generations),dim=-1),payload)
        def body():
            lanes.capture_dispatch(states,operation)
            pending.zero_()
        with ExitStack() as stack:
            for name in ('cpu','numpy','item','tolist','__bool__'):
                stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError(name)))
            outer.capture(pending,body,initialize=lambda:pending.fill_(1))
            outer.replay()
        torch.cuda.synchronize()
        expected=torch.where(active & (errors==0),slots+100,slots)
        self.assertTrue(torch.equal(output.slots,expected))
        self.assertEqual(captures,['static capture'])
        self.assertTrue(torch.equal(output.owners,owners))
        self.assertTrue(torch.equal(output.generations,generations))
        self.assertTrue(torch.equal(payload,torch.stack((slots,owners,generations),dim=-1)))
        self.assertTrue(torch.equal(output.active,active))
        self.assertEqual(output.errors[3].item(),107)
        self.assertEqual(output.slots[0].item(),100) # Last partial cohort does not clobber zero.
        self.assertEqual(lanes.processed.item(),rows-2)
        self.assertEqual(lanes.peak.item(),32)
        self.assertEqual(lanes.loop.errors.item(),0)
        self.assertEqual(outer.errors.item(),0)
        self.assertEqual(lanes.cursor.item(),((rows+31)//32)*32)
        # A second replay reads new selector values; there is no CPU predicate.
        active.zero_();outer.replay();torch.cuda.synchronize()
        self.assertTrue(torch.equal(output.slots,slots))
        self.assertEqual(lanes.processed.item(),0)
        self.assertEqual(output.slots[3].item(),3) # Failed lease is retained for cleanup.
        self.assertEqual(captures,['static capture'])
