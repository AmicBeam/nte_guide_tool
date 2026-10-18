"""CPU orchestration fixtures only; never qualification of CUDA phase graphs."""
from copy import deepcopy
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import Mock

import torch

from app.modules.card_game.rl.batched_search.device_h3_segments import replay_h3_chunks, _progress


def result(steps=(0,), complete=(False,), errors=None, truncated=None):
    return NS(steps=torch.tensor(steps, dtype=torch.int64), complete=torch.tensor(complete),
        errors=torch.tensor(errors if errors is not None else [0]*len(steps), dtype=torch.int32),
        truncated=torch.tensor(truncated if truncated is not None else [False]*len(steps)))


class PersistentLoop:
    """Controlled chunk outcomes; no real rules or synthetic qualification."""
    def __init__(self, frames):
        self.frames = list(frames)
        self.current = result()
        self.calls = 0
        self.initialize = Mock(side_effect=AssertionError('Must not reinitialize between chunks'))
        self.capture = Mock(side_effect=AssertionError('Must not recapture between chunks'))
        self.chunk_search_ticks_device = torch.tensor(0)
        self._outer = NS(errors=torch.zeros(1, dtype=torch.int32))
        self._inner = NS(errors=torch.zeros(1, dtype=torch.int32))
        self._start_gate = NS(errors=torch.zeros(1, dtype=torch.int32))
        self._commit_gate = NS(errors=torch.zeros(1, dtype=torch.int32))
        self.arenas = object()
        self.prefix = []
        self.raw = []
        self.q = torch.zeros((2, 3))
        self.pcg = torch.tensor([31, 32, 33, 34, 1, 123])
        self.inside_root = 0
    def result_device(self): return self.current
    def replay_chunk(self):
        self.calls += 1
        current, ticks = self.frames.pop(0)
        before = int(self.current.steps[0])
        after = int(current.steps[0])
        self.chunk_search_ticks_device.fill_(ticks)
        self.inside_root += ticks
        self.pcg[0] += ticks
        if after > before:
            self.q[before].fill_(self.inside_root)
            self.prefix.append(dict(step=before, q=self.q[before].clone(), both_viewers=[before, before+10]))
            self.raw.append(dict(step=before, action={'type': 'fixture'}))
            self.inside_root = 0
        self.current = current


class H3SegmentsTests(unittest.TestCase):
    def collector(self, **changes):
        kwargs = dict(seconds=60, proof_seconds=30, max_decisions=800, chunk_search_ticks=1,
            on_chunk=None, last_diagnostics={}, _memory=Mock(return_value=123),
            executor=NS(torch=NS(cuda=NS(synchronize=Mock()))))
        kwargs.update(changes)
        return NS(**kwargs)

    def metadata(self, seconds=600):
        return NS(jobs=[dict(deadline=time.time()+seconds)])

    def test_pending_root_resumes_same_buffers_and_full_prefix_without_reset(self):
        loop = PersistentLoop([(result(), 1), (result((1,)), 1),
                               (result((1,)), 1), (result((2,), (True,)), 1)])
        arenas, q, pcg = loop.arenas, loop.q, loop.pcg
        callbacks = []
        collector = self.collector(on_chunk=callbacks.append)
        metadata = self.metadata()
        deadline = metadata.jobs[0]['deadline']
        final = replay_h3_chunks(collector, loop, metadata)
        self.assertTrue(final.complete.all())
        self.assertIs(loop.arenas, arenas)
        self.assertIs(loop.q, q)
        self.assertIs(loop.pcg, pcg)
        self.assertEqual(loop.pcg.tolist(), [35, 32, 33, 34, 1, 123])
        self.assertEqual([p['step'] for p in loop.prefix], [0, 1])
        self.assertEqual([p['q'].tolist() for p in loop.prefix], [[2.]*3, [2.]*3])
        self.assertEqual([p['both_viewers'] for p in loop.prefix], [[0, 10], [1, 11]])
        self.assertEqual([r['step'] for r in loop.raw], [0, 1])
        loop.initialize.assert_not_called(); loop.capture.assert_not_called()
        self.assertEqual(metadata.jobs[0]['deadline'], deadline)
        self.assertEqual(collector.last_diagnostics['chunk_replays'], 4)
        self.assertTrue(all(c['complete'] is False and c['optimizer_steps'] == 0 for c in callbacks))

    def test_actual_guard_does_not_become_planned_resume(self):
        loop = PersistentLoop([(result(), 1), (result((1,), (True,)), 1)])
        original = loop.replay_chunk
        def replay():
            original()
            loop._inner.errors.fill_(1)
        loop.replay_chunk = replay
        collector = self.collector()
        with self.assertRaisesRegex(RuntimeError, 'guard'):
            replay_h3_chunks(collector, loop, self.metadata())
        self.assertEqual(loop.calls, 1)
        self.assertFalse(collector.last_diagnostics['complete'])
        with self.assertRaisesRegex(RuntimeError, 'Prior'):
            replay_h3_chunks(collector, loop, self.metadata())
        self.assertEqual(loop.calls, 1)

    def test_error_truncation_capacity_and_step_reset_fail_closed(self):
        for bad in (result(errors=[128]), result(truncated=[True]), result((801,)), result((2,))):
            with self.subTest(bad=bad):
                loop = PersistentLoop([(bad, 1)])
                with self.assertRaises((ValueError, RuntimeError)):
                    replay_h3_chunks(self.collector(), loop, self.metadata())
                self.assertEqual(loop.calls, 1)
        previous = dict(steps=[2], complete_rows=[False])
        with self.assertRaisesRegex(ValueError, 'restarted'):
            _progress(result((1,)), previous, 800)
        with self.assertRaisesRegex(ValueError, 'restarted'):
            _progress(result((2,)), dict(steps=[2], complete_rows=[True]), 800)

    def test_original_deadline_and_memory_failure_prevent_replay(self):
        loop = PersistentLoop([])
        with self.assertRaises(TimeoutError):
            replay_h3_chunks(self.collector(), loop, self.metadata(seconds=10))
        self.assertEqual(loop.calls, 0)
        collector = self.collector(_memory=Mock(side_effect=MemoryError('budget')))
        with self.assertRaises(MemoryError):
            replay_h3_chunks(collector, loop, self.metadata())
        self.assertEqual(loop.calls, 0)

    def test_search_tick_overrun_never_exports_joint_labels(self):
        loop = PersistentLoop([(result((1,), (True,)), 2)])
        collector = self.collector()
        with self.assertRaisesRegex(ValueError, 'tick budget'):
            replay_h3_chunks(collector, loop, self.metadata())
        self.assertNotEqual(collector.last_diagnostics.get('complete'), True)


if __name__ == '__main__': unittest.main()
