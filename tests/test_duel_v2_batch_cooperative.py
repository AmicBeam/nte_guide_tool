"""Cross-root operation batching and ownership, no real games or GPU claims."""
import unittest
import numpy as np
from tests.test_duel_v2_batch_traversal import FakeBackend,drive_traversal
from app.modules.card_game.rl.batched_search.traversal import search_traversal
from app.modules.card_game.rl.batched_search.cooperative import (
    cooperative_search,CooperativeScheduler,RootJob,OperationQuery,OperationResponse)
from app.modules.card_game.rl.batched_search.inference import InferenceResult


class FakeInference:
    def __init__(self):self.sizes=[]
    def predict(self,requests):
        self.sizes.append(len(requests))
        return [InferenceResult(r.request_id,r.model_key,r.model_version,
            np.zeros(len(r.candidates)),np.array([.2,.1,.7])) for r in requests]


class BulkFake(FakeBackend):
    supports_gpu=False
    def __init__(self):super().__init__();self.batch_sizes=[]
    def resolve_operations(self,requests):
        self.batch_sizes.append(len(requests))
        return [OperationResponse(r.id,getattr(self,r.method)(*r.args,**dict(r.kwargs))) for r in requests]


class CooperativeTest(unittest.TestCase):
    def make(self,backend,count):
        jobs=[]
        for i in range(count):
            state=dict(phase='playing',actor='a',depth=0)
            generator=cooperative_search(state,'a',{'a':('m','sha'),'b':('m','sha')},
                backend=backend,seed=i,simulations=2,terminal_horizon=0,root_id='r'+str(i))
            jobs.append(RootJob('r'+str(i),'g'+str(i),1,generator))
        return jobs

    def test_1600_roots_are_suspended_together_and_accounted_once(self):
        backend=BulkFake();inference=FakeInference()
        run=CooperativeScheduler(inference).run(self.make(backend,1600))
        self.assertEqual(run['root_count'],1600)
        self.assertEqual(run['peak_outstanding_roots'],1600)
        self.assertEqual(len(run['results']),1600)
        self.assertIn(1600,backend.batch_sizes)
        self.assertIn(1600,inference.sizes)
        self.assertTrue(all(r['complete'] and r['simulations']==2 for r in run['results'].values()))

    def test_cooperative_and_direct_gumbel_results_equal(self):
        for mode in ('legacy','natural_wdl'):
            backend=BulkFake();state=dict(phase='playing',actor='a',depth=0)
            args=(state,'a',{'a':('m','sha'),'b':('m','sha')})
            kw=dict(backend=backend,seed=15,simulations=8,terminal_horizon=2,q_scale=mode)
            direct=drive_traversal(search_traversal(*args,**kw))
            run=CooperativeScheduler(FakeInference()).run([RootJob('r','g',1,cooperative_search(*args,**kw))])
            actual=run['results']['r']
            np.testing.assert_array_equal(actual['visits'],direct['visits'])
            np.testing.assert_allclose(actual['mean_values'],direct['mean_values'])
            np.testing.assert_allclose(actual['pi'],direct['pi'])
            self.assertEqual(actual['search_choice'],direct['search_choice'])

    def test_no_silent_cpu_fallback_or_relative_queue_clock(self):
        with self.assertRaises(RuntimeError):
            CooperativeScheduler(FakeInference()).run(self.make(FakeBackend(),1))
        run=CooperativeScheduler(FakeInference(),allow_cpu_oracle=True).run(self.make(FakeBackend(),1))
        self.assertTrue(run['cpu_oracle_opt_in'])
        with self.assertRaises(ValueError):cooperative_search(None,'a',{},time_limit=3.)

    def test_duplicate_game_ownership_and_corrupt_response_rejected(self):
        backend=BulkFake();jobs=self.make(backend,2)
        jobs[1]=RootJob('r2',jobs[0].game_id,2,jobs[1].generator)
        with self.assertRaises(ValueError):CooperativeScheduler(FakeInference()).run(jobs)
        for job in jobs:job.generator.close()
        backend=BulkFake();backend.resolve_operations=lambda requests:[OperationResponse('wrong',None)]*len(requests)
        with self.assertRaises(ValueError):CooperativeScheduler(FakeInference()).run(self.make(backend,1))

    def test_cancel_and_wave_bound_close_every_generator(self):
        jobs=self.make(BulkFake(),3)
        run=CooperativeScheduler(FakeInference()).run(jobs,cancelled=lambda:True)
        self.assertEqual(len(run['results']),3)
        self.assertTrue(all(not r['complete'] for r in run['results'].values()))
        self.assertTrue(all(j.generator.gi_frame is None for j in jobs))
        jobs=self.make(BulkFake(),3)
        with self.assertRaises(TimeoutError):CooperativeScheduler(FakeInference()).run(jobs,max_waves=1)
        self.assertTrue(all(j.generator.gi_frame is None for j in jobs))


if __name__=='__main__':unittest.main()
