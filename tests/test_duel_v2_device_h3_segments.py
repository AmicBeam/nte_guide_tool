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


class H3OwnedStorageCheckpointTests(unittest.TestCase):
    def fixture(self):
        backing=torch.arange(48,dtype=torch.int32)
        search=NS(phase=backing[:2],completed=backing[2:4],errors=torch.zeros(2,dtype=torch.int32),
            pcg=torch.arange(12,dtype=torch.int64).reshape(2,6),
            decision=NS(data=backing[4:20].reshape(4,4).t()),
            row_work=NS(plans={'policy':NS(buffer=backing[20:28])}))
        loop=NS(search=search,_ready=True,_captured=True,_errors=torch.zeros(2,dtype=torch.int32),
            steps=torch.zeros(2,dtype=torch.int64),teacher=NS(pending=backing[28:36]),
            _keepalive=[search.decision,backing[36:]],shared=backing)
        return dict(loop=loop,search=search)

    def test_full_storage_bits_alias_views_and_pending_root_roundtrip(self):
        import tempfile
        from pathlib import Path
        from app.modules.card_game.rl.batched_search.device_h3_segments import snapshot_h3_tensors,restore_h3_tensors
        from app.modules.card_game.rl.batched_search.pure_checkpoint import write_payload,read_payload
        original=self.fixture();original['search'].phase.fill_(4);original['search'].completed.fill_(17)
        snapshot=snapshot_h3_tensors(original,allow_cpu_reference=True)
        with tempfile.TemporaryDirectory()as root:
            path=Path(root)/'chunk'
            write_payload(path,snapshot,dict(metadata=dict(jobs=[])),kind='h3_cpu_fixture')
            saved=read_payload(path,dict(metadata=dict(jobs=[])),kind='h3_cpu_fixture')
            fresh=self.fixture(); before_id=id(fresh['search'].decision.data)
            restore_h3_tensors(fresh,saved,allow_cpu_reference=True)
            self.assertEqual(id(fresh['search'].decision.data),before_id)
            self.assertEqual(fresh['search'].completed.tolist(),[17,17])
            self.assertTrue(torch.equal(fresh['loop'].shared,original['loop'].shared))
            self.assertEqual(fresh['search'].decision.data.stride(),(1,4))
            fresh['loop'].shared[4]=999
            self.assertEqual(int(fresh['search'].decision.data[0,0]),999)
            self.assertIs(fresh['loop']._keepalive[0],fresh['search'].decision)
            self.assertTrue(fresh['loop']._ready)
            # Resume the same hypothetical root progress, without initialization.
            fresh['search'].completed.add_(1);original['search'].completed.add_(1)
            self.assertTrue(torch.equal(fresh['search'].completed,original['search'].completed))

    def test_missing_layout_stride_alias_and_initialized_flag_fail_before_writes(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import snapshot_h3_tensors,restore_h3_tensors
        original=self.fixture();snapshot=snapshot_h3_tensors(original,allow_cpu_reference=True)
        for change in ('missing','stride','alias','initialization'):
            with self.subTest(change=change):
                fresh=self.fixture();before=fresh['loop'].shared.clone()
                saved=deepcopy(snapshot)
                if change=='missing':saved['storage'].pop(next(iter(saved['storage'])))
                if change=='stride':saved['layout']['loop.search.decision.data']['stride']=[4,1]
                if change=='alias':fresh['search'].decision.data=fresh['search'].decision.data.clone()
                if change=='initialization':fresh['loop']._ready=False
                with self.assertRaisesRegex(ValueError,'layout/view/alias'):
                    restore_h3_tensors(fresh,saved,allow_cpu_reference=True)
                self.assertTrue(torch.equal(before,fresh['loop'].shared))

    def test_guard_and_semantic_error_prefix_cannot_be_saved(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import snapshot_h3_tensors
        for field in ('search','loop'):
            owned=self.fixture()
            (owned['search'].errors if field=='search'else owned['loop']._errors).fill_(256)
            with self.assertRaisesRegex(RuntimeError,'Failed'):
                snapshot_h3_tensors(owned,allow_cpu_reference=True)
        owned=self.fixture();owned['loop']._inner=NS(errors=torch.ones(1,dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError,'guard'):
            snapshot_h3_tensors(owned,allow_cpu_reference=True)


class H3ClockExclusionTests(unittest.TestCase):
    def test_only_conditional_guard_clock_is_nonpersistent(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import snapshot_h3_tensors,restore_h3_tensors
        Runner=type('CudaConditionalWhile',(NS,),{})
        old=NS(_outer=Runner(start=torch.tensor([999]),counter=torch.tensor([3]),errors=torch.zeros(1,dtype=torch.int32)),
            teacher=NS(start=torch.tensor([17])))
        fresh=NS(_outer=Runner(start=torch.tensor([111]),counter=torch.tensor([0]),errors=torch.zeros(1,dtype=torch.int32)),
            teacher=NS(start=torch.tensor([0])))
        saved=snapshot_h3_tensors(dict(loop=old),allow_cpu_reference=True)
        restore_h3_tensors(dict(loop=fresh),saved,allow_cpu_reference=True)
        self.assertEqual(fresh._outer.start.tolist(),[111])
        self.assertEqual(fresh._outer.counter.tolist(),[3])
        self.assertEqual(fresh.teacher.start.tolist(),[17])


class H3DurableSessionTests(unittest.TestCase):
    def test_pending_root_session_restores_new_buffers_and_rejects_failed_lineage(self):
        import tempfile
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search import device_h3_segments as module
        owner=H3OwnedStorageCheckpointTests()
        original=owner.fixture();original['search'].completed.fill_(19)
        snapshot,restore=module.snapshot_h3_tensors,module.restore_h3_tensors
        binding=dict(metadata=dict(jobs=[dict(deadline=time.time()+600)]),source='strict-fixture')
        with tempfile.TemporaryDirectory()as root, patch.object(module,'snapshot_h3_tensors',
                side_effect=lambda roots:snapshot(roots,allow_cpu_reference=True)), patch.object(module,'restore_h3_tensors',
                side_effect=lambda roots,data:restore(roots,data,allow_cpu_reference=True)):
            session=module.H3CheckpointSession(root,None,binding,max_chunks=1)
            path=session.save(original,[dict(version=0,phase='playing')],dict(steps=[0],complete_rows=[False]))
            loaded=module.H3CheckpointSession(root,None,binding,max_chunks=1)
            fresh=owner.fixture(); loaded.restore(fresh)
            self.assertTrue(loaded.restored)
            self.assertEqual(loaded.total,1)
            self.assertEqual(fresh['search'].completed.tolist(),[19,19])
            self.assertEqual(loaded.payload['initial'],[dict(version=0,phase='playing')])
            self.assertEqual(loaded.path,path)
            loaded.reject(RuntimeError('actual guard'))
            with self.assertRaisesRegex(RuntimeError,'rejected'):
                module.H3CheckpointSession(root,None,binding,max_chunks=1)


class H3ThreadOwnerExclusionTests(unittest.TestCase):
    def test_statistics_pool_rebinds_new_thread_but_keeps_semantic_flags_strict(self):
        from app.modules.card_game.rl.batched_search.device_h3_segments import snapshot_h3_tensors,restore_h3_tensors
        Pool=type('DeviceStatisticsPool',(NS,),{})
        old=Pool(_owner_thread=111,_stream=7,_ready=True,total=torch.tensor([3.]))
        fresh=Pool(_owner_thread=222,_stream=9,_ready=True,total=torch.tensor([0.]))
        saved=snapshot_h3_tensors(dict(statistics=old),allow_cpu_reference=True)
        restore_h3_tensors(dict(statistics=fresh),saved,allow_cpu_reference=True)
        self.assertEqual(fresh._owner_thread,222)
        self.assertEqual(fresh._stream,9)
        self.assertEqual(fresh.total.tolist(),[3.])
        fresh._ready=False
        with self.assertRaisesRegex(ValueError,'topology'):
            restore_h3_tensors(dict(statistics=fresh),saved,allow_cpu_reference=True)


if __name__ == '__main__': unittest.main()
