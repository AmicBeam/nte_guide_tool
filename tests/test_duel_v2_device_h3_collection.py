"""Source/contract and lifecycle fixtures, never actual CUDA qualification."""
from copy import deepcopy
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import Mock, patch

from app.modules.card_game.rl.batched_search.device_h3_collection import DeviceH3Collector, PROFILE, validate_h3_inputs
from tests.test_duel_v2_soa_training_jobs import models, jobs


class H3CollectionContractTest(unittest.TestCase):
    def fixtures(self):
        submitted=jobs(teacher_mode='covered_value',covered_value_source='/tmp/explicit-covered-scope')[:2]
        return submitted,models()

    def test_fixed_profile_and_actual_scope_loader(self):
        submitted,bank=self.fixtures()
        before=deepcopy(submitted)
        with patch('app.modules.card_game.rl.seven_runtime.value_precheck_ok',return_value=True) as precheck, \
             patch('app.modules.card_game.rl.seven_covered_value.load_scope',return_value=({'aux':1},{'passed':True})) as loader:
            metadata,aux,scope=validate_h3_inputs(submitted,bank)
        self.assertEqual(submitted,before)
        self.assertEqual(precheck.call_count,2)
        from pathlib import Path
        loader.assert_called_once_with(str(Path('/tmp/explicit-covered-scope').resolve()))
        self.assertEqual(metadata.search['terminal_horizon'],3)
        self.assertEqual(metadata.search['world_sampling'],'per_simulation')
        self.assertEqual((PROFILE['simulations'],PROFILE['gumbel_candidates'],PROFILE['opponent_mode']),(32,16,'rule'))

    def test_unqualified_policy_value_fails_before_scope_load_or_gpu(self):
        submitted,bank=self.fixtures()
        with patch('app.modules.card_game.rl.seven_runtime.value_precheck_ok',return_value=False), \
             patch('app.modules.card_game.rl.seven_covered_value.load_scope') as load:
            with self.assertRaisesRegex(ValueError,'policy value-precheck'):validate_h3_inputs(submitted,bank)
            load.assert_not_called()

    def test_stale_auxiliary_scope_is_not_repaired_or_bypassed(self):
        submitted,bank=self.fixtures()
        with patch('app.modules.card_game.rl.seven_runtime.value_precheck_ok',return_value=True), \
             patch('app.modules.card_game.rl.seven_covered_value.load_scope',side_effect=ValueError('stale scope')):
            with self.assertRaisesRegex(ValueError,'stale scope'):validate_h3_inputs(submitted,bank)

    def test_reduced_budget_legacy_world_sampling_and_network_teacher_rejected(self):
        for field,value in (('simulations',2),('terminal_horizon',0),('world_sampling','per_decision')):
            submitted,bank=self.fixtures();submitted[0][field]=value
            with self.assertRaises(ValueError):validate_h3_inputs(submitted,bank)
        submitted=jobs()[:2]
        with self.assertRaisesRegex(ValueError,'explicit qualified covered_value'):validate_h3_inputs(submitted,models())

    def test_large_batch_rejected_before_cuda_or_source_loading(self):
        with self.assertRaisesRegex(ValueError,'One to32'):validate_h3_inputs([{}]*33,{})

    def test_partial_constructor_failure_closes_arenas_and_keeps_diagnostics(self):
        collector=object.__new__(DeviceH3Collector)
        collector.seconds=1;collector.proof_seconds=1
        rules=NS(close=Mock(),physical_loop=object(),inner=NS(physical_loop=object()))
        def fail(metadata,models,auxiliary,owned):
            owned.append(rules)
            raise RuntimeError('capture failed')
        collector._collect=fail
        with patch('app.modules.card_game.rl.batched_search.device_h3_collection.validate_h3_inputs',
                return_value=(NS(model_versions=()),{},dict(models={}))):
            with self.assertRaisesRegex(RuntimeError,'capture failed'):
                collector.collect([dict(deadline=time.time()+30)],{})
        rules.close.assert_called_once();self.assertIsNone(rules.physical_loop)
        self.assertIsNone(rules.inner.physical_loop)
        self.assertEqual(collector.last_diagnostics['optimizer_steps'],0)

class H3ArchiveTest(unittest.TestCase):
    def test_final_statistics_are_stored_at_exact_physical_coordinates(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_h3_collection import _H3AuditSink
        zeros=lambda *shape:torch.zeros(shape,dtype=torch.float64)
        tape=NS(t=torch,device=torch.device('cpu'),rows=2,max_actions=3,
            begin_device=Mock(return_value=torch.zeros(2,dtype=torch.int32)),
            append_device=Mock(return_value=torch.zeros(2,dtype=torch.int32)),
            finish_device=Mock(return_value=torch.zeros(2,dtype=torch.int32)))
        result=NS(prior=zeros(2,3)+.25,visits=torch.ones((2,3),dtype=torch.int32),
            total=zeros(2,3)+.5,value=zeros(2)+.2,error=torch.zeros(2,dtype=torch.bool))
        statistics=NS(snapshot=Mock(return_value=result),
                      completed_q=Mock(return_value=NS(completed_q=zeros(2,3)+3,error=torch.zeros(2,dtype=torch.bool))))
        search=NS(_noise=zeros(2,3)+.8)
        audit=_H3AuditSink(tape,statistics,search,decisions=3)
        audit.begin_device()
        active=torch.tensor([True,False]);statistics.snapshot(None,None,active=active)
        roots=NS(owners=torch.tensor([0,1]));actors=torch.tensor([0,1]);steps=torch.tensor([1,0])
        physical=NS(information_keys=torch.arange(16,dtype=torch.int64).reshape(2,8),actions=torch.tensor([2,1]))
        audit.append_device(roots,actors,steps,physical,active)
        archive=audit.after_batch()
        self.assertEqual(archive['q'][0,1].tolist(),[3.,3.,3.])
        self.assertEqual(archive['mean'][0,1].tolist(),[.5,.5,.5])
        self.assertEqual(archive['q'][1,0].tolist(),[0.,0.,0.])
        self.assertEqual(audit.valid.tolist()[:6],[False,True,False,False,False,False])
        self.assertEqual(archive['value'][0,1],.2)
        self.assertFalse(archive['errors'].any())


if __name__=='__main__':unittest.main()
