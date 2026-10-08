"""Device statistics integrated with real traversal dependencies and cancellation."""
import gc
import unittest
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.batched_search.statistics import BatchedStatistics
from app.modules.card_game.rl.batched_search.nodes import BatchedNodePool
from app.modules.card_game.rl.batched_search.cooperative import cooperative_search,CooperativeScheduler,RootJob
from app.modules.card_game.rl.batched_search.traversal import search_traversal,RootEvaluation,InferenceQuery,InferenceResponse
from tests.test_duel_v2_batch_traversal import FakeBackend,drive_traversal
from tests.test_duel_v2_batch_cooperative import BulkFake,FakeInference


class StatisticsTraversalTest(unittest.TestCase):
    def compare(self,device):
        for mode in ('legacy','natural_wdl'):
            with self.subTest(mode=mode),BatchedStatistics(66,16,device=device,q_scale=mode) as statistics:
                backend=BulkFake();state=dict(phase='playing',actor='a',depth=0)
                args=(state,'a',{'a':('m','sha'),'b':('m','sha')})
                options=dict(backend=backend,seed=145,simulations=32,terminal_horizon=2,q_scale=mode,noise=True)
                expected=drive_traversal(search_traversal(*args,**options))
                actions,(x,c)=backend.decision(state,'a')
                query=InferenceQuery('cached','m','sha',x,c,True,'a')
                cache=RootEvaluation(state,'a',actions,query,InferenceResponse('cached',np.zeros(len(c)),np.array([.2,.1,.7]),'sha'))
                jobs=[RootJob('r'+str(i),'g'+str(i),1,cooperative_search(*args,**options,
                    statistics=statistics,root_evaluation=cache if i else None)) for i in range(2)]
                run=CooperativeScheduler(FakeInference()).run(jobs)
                for actual in run['results'].values():
                    np.testing.assert_array_equal(actual['visits'],expected['visits'])
                    np.testing.assert_allclose(actual['mean_values'],expected['mean_values'],atol=1e-12)
                    np.testing.assert_allclose(actual['pi'],expected['pi'],atol=1e-12)
                    self.assertEqual(actual['search_choice'],expected['search_choice'])
                    self.assertEqual(actual['simulations'],32)
                gc.collect();snapshot=statistics.stats()
                self.assertEqual(snapshot['active_nodes'],0)
                self.assertGreater(snapshot['operations']['backup'],0)

    def test_cpu_device_statistics_match_original_sequential_search(self):self.compare('cpu')

    def test_cuda_statistics_match_original_sequential_search(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA device required')
        self.compare('cuda')

    def test_late_backup_restores_prior_visits_and_values(self):
        class Clock:
            value=0.
            def __call__(self):return self.value
        clock=Clock();pool=BatchedNodePool(2,16);[ref]=pool.allocate([[.4,.6]],[.1],['root'])
        original=pool._apply_backup
        def late(flat):original(flat);clock.value=10.
        with patch.object(pool,'_apply_backup',side_effect=late):
            accepted=pool.backup_admitted([[(ref,1),(ref,1)]],[.7],[5.],clock)
        self.assertEqual(accepted,[False])
        snapshot=pool.snapshot([ref])
        self.assertEqual(snapshot['visits'].sum().item(),0)
        self.assertEqual(snapshot['total'].sum().item(),0.)

    def test_cancellation_finishes_read_only_snapshot_and_releases_nodes(self):
        with BatchedStatistics(33,16,device='cpu') as statistics:
            backend=BulkFake();inference=FakeInference();original=inference.predict
            def predict(requests):return original(requests)
            inference.predict=predict
            job=RootJob('r','g',1,cooperative_search(dict(phase='playing',actor='a',depth=0),'a',
                {'a':('m','sha'),'b':('m','sha')},backend=backend,seed=11,simulations=32,
                terminal_horizon=0,statistics=statistics))
            run=CooperativeScheduler(inference).run([job],cancelled=lambda:statistics.pool.active_nodes>0)
            self.assertFalse(run['results']['r']['complete'])
            gc.collect();self.assertEqual(statistics.stats()['active_nodes'],0)


if __name__=='__main__':unittest.main()
