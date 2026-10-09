"""Explicit hybrid device selection; synthetic binding does not prove CUDA execution."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.modules.card_game.rl.batched_search.training_rollout import rollout_from_config
from app.modules.card_game.rl.seven_round import rollout_contract

class StatisticsDeviceTest(unittest.TestCase):
    def test_hybrid_cuda_statistics_are_owned_and_bound_explicitly(self):
        class Pool:
            def __init__(self,*args,**kw):self.args=args;self.kw=kw;self.pool=SimpleNamespace(device='cuda')
            def close(self):pass
        backend=SimpleNamespace(rule_backend='resident_cpu')
        inference=SimpleNamespace(assert_frozen=lambda:None)
        cfg=dict(rule_backend='resident_cpu',device='cuda',search_roots=2,search_statistics_device='cuda')
        with patch('app.modules.card_game.rl.batched_search.statistics.BatchedStatistics',Pool):
            collector=rollout_from_config(cfg,{},backend=backend,inference=inference)
        self.assertEqual(collector.statistics.args,(66,2048))
        self.assertEqual(collector.statistics.kw['device'],'cuda')
        self.assertTrue(collector.own_statistics)
        self.assertFalse(collector.own_backend)
        self.assertEqual(rollout_contract(cfg)['search_statistics_device'],'cuda')
    def test_no_cpu_or_serial_substitution_and_default_unchanged(self):
        for cfg in (dict(rule_backend='serial_cpu',device='cuda',search_statistics_device='cuda'),
                    dict(rule_backend='resident_cpu',device='cpu',search_statistics_device='cuda'),
                    dict(rule_backend='resident_cpu',device='cuda',search_statistics_device='unknown')):
            with self.assertRaises(ValueError):rollout_contract(cfg)
        with self.assertRaises(ValueError):rollout_from_config(dict(device='cpu',search_statistics_device='cuda'),{})
        cfg=dict(rule_backend='resident_cpu',device='cuda',search_statistics_device='cuda')
        with self.assertRaises(ValueError):rollout_from_config(cfg,{},backend=SimpleNamespace(rule_backend='resident_cpu'),inference=SimpleNamespace(assert_frozen=lambda:None),statistics=SimpleNamespace(pool=SimpleNamespace(device='cpu')))
        self.assertIsNone(rollout_contract(dict(rule_backend='resident_cpu'))['search_statistics_device'])

if __name__=='__main__':unittest.main()
