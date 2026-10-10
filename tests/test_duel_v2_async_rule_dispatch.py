"""Async submit/poll rule dispatch for resident CPU backends and the scheduler."""
from __future__ import annotations

import concurrent.futures
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.rl.batched_duel.process_backend import (
    ResidentRuleBackend,
    _worker_ops_batch,
)
from app.modules.card_game.rl.batched_search.cooperative import (
    CooperativeScheduler,
    OperationQuery,
    OperationResponse,
    RequestExpired,
    RootJob,
    _operation,
    cooperative_search,
)
from app.modules.card_game.rl.batched_search.inference import InferenceResult
from app.modules.card_game.rl.batched_search.traversal import InferenceQuery


class FakeInference:
    def __init__(self):
        self.sizes = []
        self.events = None

    def predict(self, requests):
        self.sizes.append(len(requests))
        if self.events is not None:
            self.events.append(('infer', len(requests)))
        return [
            InferenceResult(
                r.request_id, r.model_key, r.model_version,
                np.zeros(len(r.candidates), dtype=np.float32),
                np.array([0.2, 0.1, 0.7], dtype=np.float32),
            )
            for r in requests
        ]


class AsyncFakeBackend:
    supports_async_operations = True
    supports_mixed_operations = True
    supports_gpu = False

    def __init__(self):
        self.pending = []
        self.events = []
        self.allow_slow = threading.Event()
        self.cancelled = []
        self.idle_workers = 1

    def submit_operations(self, queries):
        self.pending.extend(queries)
        self.events.append(('submit', tuple(q.args[0]['name'] for q in queries)))

    def poll_operations(self, timeout=0.0, ids=None):
        if timeout and not self.allow_slow.is_set():
            self.allow_slow.wait(timeout=timeout)
        ready = []
        hold = []
        for query in self.pending:
            name = query.args[0]['name']
            if name == 'slow' and not self.allow_slow.is_set():
                hold.append(query)
                continue
            ready.append(OperationResponse(query.id, query.args[0].get('actor', 'a')))
            self.events.append(('ready', name, query.method))
        self.pending = hold
        self.idle_workers = 0 if self.pending else 1
        return ready

    def cancel_operations(self, ids):
        wanted = set(ids)
        self.cancelled.extend(ids)
        self.pending = [query for query in self.pending if query.id not in wanted]


class StatsFake:
    read_only_finalization = True

    def __init__(self, events):
        self.events = events

    def resolve_operations(self, queries):
        self.events.append(('stats', tuple(q.method for q in queries)))
        return [OperationResponse(query.id, 'snap') for query in queries]


class AsyncRuleDispatchTest(unittest.TestCase):
    def test_root_deadline_retires_queued_work_without_waiting_for_the_worker(self):
        from app.modules.card_game.rl.batched_search.traversal import _DeadlineReached
        backend = AsyncFakeBackend()
        deadline = time.monotonic() + 0.05
        def root():
            try:
                yield from _operation(backend, 'acting_side', {'name': 'slow'})
            except _DeadlineReached:
                return {'complete': False, 'reason': 'deadline'}
            raise AssertionError('Expired root resumed normally')
        result = CooperativeScheduler(FakeInference()).run(
            [RootJob('root', 'game', 1, root(), deadline)], max_waves=20)
        self.assertEqual(result['results']['root']['reason'], 'deadline')
        self.assertTrue(backend.cancelled)
        self.assertFalse(backend.pending)
        self.assertLess(result['waves'], 10)

    def test_slow_worker_does_not_block_fast_root_or_inference(self):
        backend = AsyncFakeBackend()
        stats = StatsFake(backend.events)
        inference = FakeInference()
        inference.events = backend.events

        def gen_fast():
            yield from _operation(backend, 'acting_side', {'name': 'fast', 'actor': 'a'})
            query = InferenceQuery(
                'inf-fast', 'm', 'sha', np.zeros(1), np.zeros((1, 1)), True, 'a',
            )
            yield query
            yield from _operation(stats, 'snapshot', 'node')
            backend.allow_slow.set()
            return 'fast'

        def gen_slow():
            yield from _operation(backend, 'acting_side', {'name': 'slow', 'actor': 'a'})
            return 'slow'

        run = CooperativeScheduler(inference, async_rule_dispatch=True).run([
            RootJob('fast', 'fast', 1, gen_fast()),
            RootJob('slow', 'slow', 1, gen_slow()),
        ])
        self.assertEqual(run['results']['fast'], 'fast')
        self.assertEqual(run['results']['slow'], 'slow')
        ready = [event for event in backend.events if event[0] in ('ready', 'infer', 'stats')]
        self.assertEqual(ready[0], ('ready', 'fast', 'acting_side'))
        self.assertEqual(ready[1][0], 'infer')
        self.assertEqual(ready[2], ('stats', ('snapshot',)))
        self.assertIn(('ready', 'slow', 'acting_side'), ready)
        self.assertLess(ready.index(('infer', 1)), ready.index(('ready', 'slow', 'acting_side')))
        self.assertLess(ready.index(('stats', ('snapshot',))), ready.index(('ready', 'slow', 'acting_side')))
        self.assertGreaterEqual(run['progress_metrics']['peak_pending_operations'], 1)
        self.assertIsNotNone(run['progress_metrics']['idle_workers'])

    def test_explicit_async_requires_capability_and_false_keeps_wave(self):
        class Bulk:
            supports_gpu = False
            def resolve_operations(self, queries):
                return [OperationResponse(q.id, 'a') for q in queries]

        def gen(backend):
            return (yield from _operation(backend, 'acting_side', {'actor': 'a'}))

        backend = Bulk()
        with self.assertRaises(RuntimeError):
            CooperativeScheduler(FakeInference(), async_rule_dispatch=True).run(
                [RootJob('r', 'g', 1, gen(backend))]
            )
        run = CooperativeScheduler(FakeInference(), async_rule_dispatch=False).run(
            [RootJob('r', 'g', 1, gen(backend))]
        )
        self.assertEqual(run['results']['r'], 'a')
        self.assertIs(run['async_rule_dispatch'], False)

    def test_scheduler_cancel_releases_outstanding_async_ops(self):
        backend = AsyncFakeBackend()

        def gen_fast():
            yield from _operation(backend, 'acting_side', {'name': 'fast', 'actor': 'a'})
            return dict(complete=True)

        def gen_slow():
            yield from _operation(backend, 'acting_side', {'name': 'slow', 'actor': 'a'})
            return dict(complete=True)

        jobs = [RootJob('fast', 'fast', 1, gen_fast()), RootJob('slow', 'slow', 1, gen_slow())]
        seen = [0]

        def cancelled():
            seen[0] += 1
            return seen[0] > 1

        run = CooperativeScheduler(FakeInference(), async_rule_dispatch=True).run(
            jobs, cancelled=cancelled,
        )
        self.assertTrue(backend.cancelled or not backend.pending)
        self.assertTrue(all(job.generator.gi_frame is None for job in jobs))
        self.assertEqual(len(run['results']), 2)


class ResidentAsyncDispatchTest(unittest.TestCase):
    def _games(self, count=2, seed=880):
        return [new_game(seed=seed + i, first_side='a', skip_mulligan=True) for i in range(count)]

    def test_submit_poll_matches_resolve_for_two_workers(self):
        games = self._games()
        with ResidentRuleBackend(workers=2, max_games=2, max_states_per_worker=40) as backend:
            refs = backend.import_states(games, ['g0', 'g1'], [1, 1])
            self.assertEqual([ref.worker for ref in refs], [0, 1])
            sync = backend.resolve_operations([
                OperationQuery('s0', backend, 'decision', (refs[0], 'a'), ()),
                OperationQuery('s1', backend, 'decision', (refs[1], 'a'), ()),
            ])
            backend.submit_operations([
                OperationQuery('a0', backend, 'decision', (refs[0], 'a'), ()),
                OperationQuery('a1', backend, 'decision', (refs[1], 'a'), ()),
            ])
            async_rows = backend.poll_operations(timeout=30.0, ids=['a0', 'a1'])
            self.assertEqual([row.id for row in async_rows], ['a0', 'a1'])
            for left, right in zip(sync, async_rows):
                self.assertEqual(left.value[0], right.value[0])
                np.testing.assert_allclose(left.value[1][0], right.value[1][0])
            worlds_sync = backend.resolve_operations([
                OperationQuery('w0', backend, 'sample_world', (refs[0], 'a', 11), ()),
                OperationQuery('w1', backend, 'sample_world', (refs[1], 'a', 12), ()),
            ])
            backend.submit_operations([
                OperationQuery('p0', backend, 'sample_world', (refs[0], 'a', 11), ()),
                OperationQuery('p1', backend, 'sample_world', (refs[1], 'a', 12), ()),
            ])
            worlds_async = backend.poll_operations(timeout=30.0, ids=['p0', 'p1'])
            self.assertEqual(worlds_sync[0].value.phase, worlds_async[0].value.phase)
            self.assertEqual(worlds_sync[1].value.phase, worlds_async[1].value.phase)
            self.assertEqual(backend.pending_operations, 0)
            self.assertEqual(backend.idle_workers, 2)

    def test_slow_independent_worker_cannot_block_fast_root(self):
        games = self._games()
        with ResidentRuleBackend(workers=2, max_games=2, max_states_per_worker=40) as backend:
            refs = backend.import_states(games, ['fast', 'slow'], [1, 1])
            slow_exec = backend._executors[1]
            original = slow_exec.submit
            held = []

            def gated(fn, *args, **kwargs):
                if fn is _worker_ops_batch:
                    fut = concurrent.futures.Future()
                    held.append((fut, fn, args, kwargs))
                    return fut
                return original(fn, *args, **kwargs)

            slow_exec.submit = gated
            try:
                backend.submit_operations([
                    OperationQuery('fast', backend, 'decision', (refs[0], 'a'), ()),
                    OperationQuery('slow', backend, 'decision', (refs[1], 'a'), ()),
                ])
                rows = backend.poll_operations(timeout=30.0)
                self.assertEqual({row.id for row in rows}, {'fast'})
                self.assertEqual(backend.pending_operations, 1)
                self.assertEqual(backend.idle_workers, 1)
                self.assertEqual(backend.ready_operations, 0)
            finally:
                for fut, fn, args, kwargs in held:
                    try:
                        real = original(fn, *args, **kwargs)
                        done, _ = concurrent.futures.wait([real], timeout=30)
                        if not done:
                            fut.set_exception(TimeoutError('slow worker did not finish'))
                        else:
                            fut.set_result(real.result())
                    except Exception as exc:
                        if not fut.done():
                            fut.set_exception(exc)
                slow_exec.submit = original
            rest = backend.poll_operations(timeout=30.0)
            self.assertEqual({row.id for row in rest}, {'slow'})
            self.assertEqual(backend.pending_operations, 0)

    def test_scheduler_parity_async_versus_sync_wave(self):
        policies = {'a': ('m', 'sha'), 'b': ('m', 'sha')}

        def run(flag):
            games = self._games(seed=910)
            with ResidentRuleBackend(workers=2, max_games=2, max_states_per_worker=80) as backend:
                refs = backend.import_states(games, ['g0', 'g1'], [1, 1])
                jobs = [
                    RootJob(
                        'r0', 'g0', 1,
                        cooperative_search(
                            refs[0], 'a', policies, backend=backend, seed=3,
                            simulations=2, terminal_horizon=0, root_id='r0',
                        ),
                    ),
                    RootJob(
                        'r1', 'g1', 1,
                        cooperative_search(
                            refs[1], 'a', policies, backend=backend, seed=4,
                            simulations=2, terminal_horizon=0, root_id='r1',
                        ),
                    ),
                ]
                return CooperativeScheduler(FakeInference(), async_rule_dispatch=flag).run(jobs)

        sync = run(False)
        async_run = run(True)
        for key in ('r0', 'r1'):
            left = sync['results'][key]
            right = async_run['results'][key]
            self.assertEqual(left['search_choice'], right['search_choice'])
            self.assertEqual(left['simulations'], right['simulations'])
            np.testing.assert_array_equal(left['visits'], right['visits'])
            np.testing.assert_allclose(left['pi'], right['pi'])
        self.assertTrue(async_run['results']['r0']['complete'])

    def test_terminal_expiry_does_not_allocate_and_late_results_release(self):
        game = new_game(seed=933, first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=1, max_games=1, max_states_per_worker=20) as backend:
            [ref] = backend.import_states([game], ['one'], [1])
            backend.submit_operations([
                OperationQuery('late', backend, 'sample_world', (ref, 'a', 10), (), deadline=0.0),
            ])
            [row] = backend.poll_operations(timeout=0.0)
            self.assertIsInstance(row.value, RequestExpired)
            self.assertEqual(backend.stats()['live_states'], 1)

            real_mono = time.monotonic
            start = real_mono()

            def shifted():
                return real_mono() + 10 ** 6

            backend.submit_operations([
                OperationQuery('world', backend, 'sample_world', (ref, 'a', 11), (), deadline=start + 60.0),
            ])
            until = real_mono() + 30.0
            while real_mono() < until and backend._pending_batches:
                fut = next(iter(backend._pending_batches))
                if fut.done():
                    break
                time.sleep(0.01)
            with patch('app.modules.card_game.rl.batched_duel.process_backend.time.monotonic', shifted):
                rows = backend.poll_operations(timeout=0.0)
            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0].value, RequestExpired)
            self.assertEqual(backend.stats()['live_states'], 1)

    def test_stale_foreign_duplicate_identities_rejected(self):
        game = new_game(seed=944, first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=1, max_games=1, max_states_per_worker=20) as backend:
            [ref] = backend.import_states([game], ['one'], [1])
            with self.assertRaises(ValueError):
                backend.submit_operations([
                    OperationQuery('foreign', object(), 'finished', (ref,), ()),
                ])
            backend.submit_operations([
                OperationQuery('dup', backend, 'finished', (ref,), ()),
            ])
            with self.assertRaises(ValueError):
                backend.submit_operations([
                    OperationQuery('dup', backend, 'finished', (ref,), ()),
                ])
            backend.poll_operations(timeout=0.0)
            with self.assertRaises(ValueError):
                backend.poll_operations(ids=['missing'])
            backend.release([ref])
            with self.assertRaises(ValueError):
                backend.submit_operations([
                    OperationQuery('stale', backend, 'finished', (ref,), ()),
                ])

    def test_cancel_and_close_release_without_leaks(self):
        game = new_game(seed=955, first_side='a', skip_mulligan=True)
        backend = ResidentRuleBackend(workers=1, max_games=1, max_states_per_worker=20)
        try:
            [ref] = backend.import_states([game], ['one'], [1])
            backend.submit_operations([
                OperationQuery('world', backend, 'sample_world', (ref, 'a', 13), ()),
            ])
            backend.cancel_operations(['world'])
            deadline = time.monotonic() + 30
            while backend.pending_operations and time.monotonic() < deadline:
                backend.poll_operations(timeout=0.05)
            self.assertEqual(backend.pending_operations, 0)
            self.assertEqual(backend.stats()['live_states'], 1)
            backend.submit_operations([
                OperationQuery('dec', backend, 'decision', (ref, 'a'), ()),
            ])
            procs = backend.owned_processes
            backend.close()
            self.assertTrue(all(not proc.is_alive() for proc in procs))
            with self.assertRaises(RuntimeError):
                backend.submit_operations([])
        finally:
            backend.close()

    def test_owner_thread_guard_rejects_foreign_thread(self):
        game = new_game(seed=966, first_side='a', skip_mulligan=True)
        with ResidentRuleBackend(workers=1, max_games=1, max_states_per_worker=20) as backend:
            [ref] = backend.import_states([game], ['one'], [1])
            errors = []

            def other():
                try:
                    backend.submit_operations([
                        OperationQuery('x', backend, 'finished', (ref,), ()),
                    ])
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
            self.assertTrue(errors)
            self.assertIsInstance(errors[0], RuntimeError)


if __name__ == '__main__':
    unittest.main()
