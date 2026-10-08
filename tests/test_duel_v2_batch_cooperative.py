"""Cross-root operation batching and ownership, no real games or GPU claims."""
import unittest
from collections import deque
import numpy as np
from tests.test_duel_v2_batch_traversal import FakeBackend,drive_traversal
from app.modules.card_game.rl.batched_search.traversal import search_traversal
from app.modules.card_game.rl.batched_search.cooperative import (
    cooperative_search,CooperativeScheduler,RootJob,OperationQuery,OperationResponse)
from app.modules.card_game.rl.batched_search.cooperative import _operation
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
    def test_mixed_rule_operations_share_dispatch_without_changing_results(self):
        class Mixed(BulkFake):
            supports_mixed_operations=True
            def __init__(self):super().__init__();self.methods=[]
            def resolve_operations(self,queries):
                self.methods.append([q.method for q in queries])
                return super().resolve_operations(queries)
        backend=Mixed()
        def gen(method):
            return (yield from _operation(backend,method,dict(actor='a',phase='playing')))
        jobs=[RootJob('actor','a',1,gen('acting_side')),RootJob('done','b',1,gen('finished'))]
        result=CooperativeScheduler(FakeInference()).run(jobs)
        self.assertEqual(result['results'],{'actor':'a','done':False})
        self.assertEqual(backend.methods,[['acting_side','finished']])

    def test_stream_commits_and_refills_before_slow_root_finishes(self):
        backend=BulkFake();queue=deque();events=[]
        def gen(label,steps):
            for _ in range(steps):
                yield from _operation(backend,'acting_side',dict(actor='a'))
            return label
        queue.extend([RootJob('fast1','fast',1,gen('fast1',1)),
                      RootJob('slow1','slow',1,gen('slow1',5))])
        admitted=[]
        def admit():
            if not queue:return None
            job=queue.popleft();admitted.append(job);return job
        def completed(rows):
            for job,result in rows:
                events.append(result)
                if job.root_id=='fast1':queue.append(RootJob('fast2','fast',2,gen('fast2',1)))
        run=CooperativeScheduler(FakeInference(),max_roots=2).run_stream(admit,completed)
        self.assertEqual(events,['fast1','fast2','slow1'])
        self.assertEqual(run['peak_outstanding_roots'],2)
        self.assertEqual(run['root_count'],3)
        self.assertTrue(all(j.generator.gi_frame is None for j in admitted))
        self.assertEqual(run['dispatch_metrics']['rules.acting_side']['items'],7)

    def test_stream_1600_jobs_are_admitted_in_a_bounded_window(self):
        jobs=deque(self.make(BulkFake(),1600));done=[]
        run=CooperativeScheduler(FakeInference(),max_roots=16).run_stream(
            lambda:jobs.popleft() if jobs else None,lambda rows:done.extend(rows))
        self.assertEqual(run['root_count'],1600)
        self.assertEqual(len(done),1600)
        self.assertLessEqual(run['peak_outstanding_roots'],16)
        self.assertTrue(all(result['complete'] for _,result in done))

    def test_stream_cancellation_stops_admission_and_closes_pending_roots(self):
        backend=BulkFake();stop=[False];done=[]
        def gen(steps):
            for _ in range(steps):yield from _operation(backend,'acting_side',dict(actor='a'))
            return dict(complete=True)
        jobs=[RootJob('fast','fast',1,gen(1)),RootJob('slow','slow',1,gen(5)),
              RootJob('waiting','waiting',1,gen(1))]
        queue=deque(jobs)
        def completed(rows):
            done.extend(rows)
            if any(job.root_id=='fast' for job,_ in rows):stop[0]=True
        result=CooperativeScheduler(FakeInference(),max_roots=2).run_stream(
            lambda:queue.popleft() if queue else None,completed,cancelled=lambda:stop[0])
        self.assertEqual(result['root_count'],2)
        self.assertEqual([job.root_id for job,_ in done],['fast','slow'])
        self.assertFalse(done[-1][1]['complete'])
        self.assertTrue(all(job.generator.gi_frame is None for job in jobs[:2]))
        jobs[2].generator.close()  # Never admitted: still owned by the caller.

    def test_stream_failure_closes_owned_generators_and_rejects_stale_generation(self):
        backend=BulkFake();jobs=self.make(backend,2);queue=deque(jobs)
        jobs[1]=RootJob('r2',jobs[0].game_id,1,jobs[1].generator);queue=deque(jobs)
        with self.assertRaises(ValueError):
            CooperativeScheduler(FakeInference(),max_roots=2).run_stream(
                lambda:queue.popleft() if queue else None,lambda rows:None)
        self.assertTrue(all(j.generator.gi_frame is None for j in jobs))

    def test_stream_wave_guard_bounds_stalls_instead_of_entire_run(self):
        backend=BulkFake();done=[]
        def gen(steps):
            for _ in range(steps):yield from _operation(backend,'acting_side',dict(actor='a'))
            return True
        jobs=deque(RootJob(str(i),str(i),1,gen(1)) for i in range(4))
        result=CooperativeScheduler(FakeInference(),max_roots=1).run_stream(
            lambda:jobs.popleft() if jobs else None,lambda rows:done.extend(rows),max_waves=1)
        self.assertEqual(result['waves'],4)
        slow=RootJob('slow','slow',1,gen(3));queue=deque([slow])
        with self.assertRaises(TimeoutError):
            CooperativeScheduler(FakeInference(),max_roots=1).run_stream(
                lambda:queue.popleft() if queue else None,lambda rows:None,max_waves=1)
        self.assertIsNone(slow.generator.gi_frame)

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
