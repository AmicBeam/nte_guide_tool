"""Real CPU tape archives round-trip without pickle; no CUDA qualification."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from app.modules.card_game.engine.duel_v2.entities import PlayerEntity
from app.modules.card_game.rl.batched_search.pure_checkpoint import (
    save_history, load_history, continuation_receipt, read_payload, write_payload,
    batch_cache_path,
)
from app.modules.card_game.rl.batched_search.soa_training_results import _same_graph
from tests import test_duel_v2_device_pure_segments as fixtures


class PureCheckpointTests(unittest.TestCase):
    def fixture(self):
        helper = fixtures.PureSegmentsTests()
        self.addCleanup(helper.doCleanups)
        history, _ = helper.history()
        shared = ['typed', 2**127+7]
        player = PlayerEntity({'bag': shared}, entity_id='saved-player')
        history.initial[0].update(entity=player, same=player, alias=shared, pair=(True, None))
        history.final[0].update(entity=deepcopy(player))
        binding = dict(metadata=dict(jobs=[dict(seed=1, deadline=100), dict(seed=2, deadline=100)],
            model_versions=['a'*64, 'b'*64]), max_decisions=4, source=dict(rule='c'*64, image='d'*64, schema='e'*64))
        return history, binding

    def test_real_raw_value_audit_tensors_and_typed_alias_rng_roundtrip(self):
        history, binding = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'segment'
            save_history(path, history, binding)
            loaded = load_history(path, binding)
            _same_graph(loaded.initial, history.initial)
            _same_graph(loaded.final, history.final)
            self.assertIsInstance(loaded.initial[0]['entity'], PlayerEntity)
            self.assertIs(loaded.initial[0]['entity'], loaded.initial[0]['same'])
            self.assertIs(loaded.initial[0]['entity']['bag'], loaded.initial[0]['alias'])
            for name in ('raw', 'value', 'audit'):
                old, new = getattr(history, name), getattr(loaded, name)
                self.assertEqual(set(old), set(new))
                for field in old:
                    self.assertEqual(old[field].dtype, new[field].dtype)
                    self.assertTrue(torch.equal(old[field], new[field]), field)
            self.assertFalse(loaded.value['_eligible'][1].any())
            loaded.raw['_nodes'].zero_()
            self.assertFalse(torch.equal(loaded.raw['_nodes'], history.raw['_nodes']))
            with self.assertRaises(FileExistsError): save_history(path, history, binding)

    def test_corruption_and_identity_mismatch_rejected(self):
        history, binding = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'segment'; save_history(path, history, binding)
            changed = deepcopy(binding); changed['source']['rule'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_history(path, changed)
            content = bytearray((path/'arrays.npz').read_bytes()); content[-1] ^= 1
            (path/'arrays.npz').write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_history(path, binding)

    def test_new_budget_requires_explicit_deadline_only_receipt_and_preserves_original(self):
        history, old = self.fixture()
        new = deepcopy(old)
        for job in new['metadata']['jobs']: job['deadline'] = 200
        receipt = continuation_receipt(old, new, authorization='User explicitly authorized bounded continuation')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'segment'; save_history(path, history, old)
            original = (path/'manifest.json').read_bytes()
            with self.assertRaises(ValueError): load_history(path, new)
            loaded = load_history(path, new, continuation=receipt)
            self.assertEqual(loaded.steps, history.steps)
            self.assertEqual((path/'manifest.json').read_bytes(), original)
            self.assertEqual(batch_cache_path(root, old), batch_cache_path(root, new))
            wrong = deepcopy(new); wrong['metadata']['model_versions'][0] = 'f'*64
            with self.assertRaises(ValueError): load_history(path, wrong, continuation=receipt)

    def test_guard_prefix_and_object_arrays_never_saved(self):
        history, binding = self.fixture()
        history.raw['_game_errors'][0] = 256
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'bad'
            with self.assertRaisesRegex(ValueError, 'guard'):
                save_history(path, history, binding)
            self.assertFalse(path.exists())
            import numpy as np
            with self.assertRaisesRegex(ValueError, 'Numeric'):
                write_payload(Path(root)/'object', np.array([{}], dtype=object), binding, kind='episodes')

    def test_episode_cache_keeps_exact_numeric_samples_and_no_pickle(self):
        _, binding = self.fixture()
        episodes = [dict(complete=True, id='job', value_rows=[dict(x=torch.arange(7).numpy(), z=-1.)])]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'completed'
            write_payload(path, episodes, binding, kind='episodes')
            actual = read_payload(path, binding, kind='episodes')
            self.assertEqual(actual[0]['id'], 'job')
            self.assertEqual(actual[0]['value_rows'][0]['x'].tolist(), list(range(7)))
            data = json.loads((path/'manifest.json').read_text())
            self.assertEqual(data['schema'], 'pure_cuda_archive_v1')
            with self.assertRaises(ValueError): read_payload(path, binding, kind='history')


class PureBatchCacheTests(unittest.TestCase):
    def test_context_reuses_completed_batches_without_device_replay(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search.device_pure_collection import DevicePureCollector
        from app.modules.card_game.rl.batched_search.device_learning_context import DeviceValueCalibrationContext
        from app.modules.card_game.rl.batched_search.soa_pure_jobs import adapt_pure_jobs
        from app.modules.card_game.rl.batched_search.pure_checkpoint import binding_for
        from tests.test_duel_v2_soa_pure_jobs import calibration, models
        jobs, bank = calibration()[:9], models()
        collector = object.__new__(DevicePureCollector)
        collector.executor = NS(rule_source_sha256='a'*64, image=NS(identity='b'*64), schema=NS(identity='c'*64))
        collector.segment_decisions = 16
        collector.max_decisions = 800
        collector.seconds = collector.proof_seconds = 1
        collector.on_segment = None
        expected = [dict((k, j[k]) for k in ('id', 'seed', 'first', 'left', 'right')) | dict(complete=True) for j in jobs]
        with tempfile.TemporaryDirectory() as root:
            for start in (0, 8):
                batch = jobs[start:start+8]
                binding = binding_for(adapt_pure_jobs(batch, bank), collector.executor, 800)
                path = batch_cache_path(root, binding)
                write_payload(path/'completed', expected[start:start+8], binding, kind='episodes')
            with patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs', return_value=bank), \
                 patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment', side_effect=AssertionError('No duplicate GPU replay')):
                context = DeviceValueCalibrationContext(collector, cache_dir=root)
                self.assertEqual(context.collect(jobs), expected)
                self.assertEqual(context.last_diagnostics['completed_games'], 9)
                self.assertTrue(all(b['diagnostic']['reused_completed_batch'] for b in context.last_diagnostics['batches']))


class PurePersistentProcessTests(unittest.TestCase):
    def fixture(self):
        from dataclasses import make_dataclass
        from types import SimpleNamespace as NS
        import time
        helper = fixtures.PureSegmentsTests(); self.addCleanup(helper.doCleanups)
        history, _ = helper.history()
        metadata = make_dataclass('Metadata', ['jobs'])(tuple({'deadline': time.time()+600} for _ in range(2)))
        collector = NS(max_decisions=4, segment_decisions=1, seconds=1, proof_seconds=1,
            last_diagnostics={}, on_segment=None,
            executor=NS(rule_source_sha256='a'*64,image=NS(identity='b'*64),schema=NS(identity='c'*64)))
        return history, metadata, collector

    def test_clean_process_pause_resume_skips_unchanged_prefix_then_completed_cache(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search.device_pure_segments import collect_segments, PureSegmentPaused
        history, metadata, collector = self.fixture()
        def partial(owner, meta, models, owned, resume, **kwargs):
            self.assertIsNone(resume)
            save_history(kwargs['checkpoint_dir']/'segment-00000002', history, kwargs['checkpoint_binding'])
            return history, None, dict(steps=[1,1], complete_rows=[True,False])
        with tempfile.TemporaryDirectory() as root:
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment', side_effect=partial):
                with self.assertRaises(PureSegmentPaused) as paused:
                    collect_segments(collector,metadata,{},checkpoint_dir=root,max_segments=1)
            self.assertTrue(Path(paused.exception.checkpoint).is_dir())
            expected = [dict(complete=True,id='a'),dict(complete=True,id='b')]
            def complete(owner, meta, models, owned, resume, **kwargs):
                for key in history.raw: self.assertTrue(torch.equal(history.raw[key],resume.raw[key]),key)
                return None, expected, dict(steps=[1,2], complete_rows=[True,True])
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment', side_effect=complete) as gpu:
                self.assertEqual(collect_segments(collector,metadata,{},checkpoint_dir=root),expected)
                self.assertEqual(gpu.call_count,1)
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment',side_effect=AssertionError('Already complete')):
                self.assertEqual(collect_segments(collector,metadata,{},checkpoint_dir=root),expected)

    def test_failed_gpu_segment_cannot_reuse_old_clean_prefix(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search.device_pure_segments import collect_segments
        _, metadata, collector = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment',side_effect=RuntimeError('actual guard')):
                with self.assertRaisesRegex(RuntimeError,'actual guard'):
                    collect_segments(collector,metadata,{},checkpoint_dir=root)
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment') as gpu:
                with self.assertRaisesRegex(RuntimeError,'rejected'):
                    collect_segments(collector,metadata,{},checkpoint_dir=root)
                gpu.assert_not_called()


class PureNewBatchWindowTests(unittest.TestCase):
    def test_cached_batch_is_free_and_second_new_batch_never_starts(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search.device_pure_collection import DevicePureCollector
        from app.modules.card_game.rl.batched_search.device_learning_context import DeviceValueCalibrationContext
        from app.modules.card_game.rl.batched_search.device_pure_segments import PureSegmentPaused
        from app.modules.card_game.rl.batched_search.soa_pure_jobs import adapt_pure_jobs
        from app.modules.card_game.rl.batched_search.pure_checkpoint import binding_for
        from tests.test_duel_v2_soa_pure_jobs import calibration, models
        jobs, bank = calibration()[:17], models()
        collector = object.__new__(DevicePureCollector)
        collector.executor = NS(rule_source_sha256='a'*64, image=NS(identity='b'*64), schema=NS(identity='c'*64))
        collector.segment_decisions, collector.max_decisions = 16, 800
        collector.last_diagnostics = {}
        def episodes(batch):
            return [dict((k, j[k]) for k in ('id','seed','first','left','right')) | dict(complete=True) for j in batch]
        calls, gpu = [], []
        def collect(batch, loaded, **options):
            calls.append([j['id'] for j in batch])
            binding = binding_for(adapt_pure_jobs(batch, loaded), collector.executor, 800)
            completed = Path(options['checkpoint_dir'])/'completed'
            if completed.exists(): return read_payload(completed, binding, kind='episodes')
            gpu.append([j['id'] for j in batch])
            result = episodes(batch)
            write_payload(completed, result, binding, kind='episodes')
            return result
        collector.collect = collect
        with tempfile.TemporaryDirectory() as root:
            first = binding_for(adapt_pure_jobs(jobs[:8],bank),collector.executor,800)
            write_payload(batch_cache_path(root,first)/'completed',episodes(jobs[:8]),first,kind='episodes')
            context = DeviceValueCalibrationContext(collector,cache_dir=root,max_new_batches=1)
            with patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs',return_value=bank):
                with self.assertRaises(PureSegmentPaused): context.collect(jobs)
            self.assertEqual(len(calls),2)
            self.assertEqual(gpu,[[j['id'] for j in jobs[8:16]]])
            self.assertEqual(context.last_diagnostics['completed_games'],16)
            self.assertEqual(context.last_diagnostics['new_batches'],1)
            self.assertEqual(context.last_diagnostics['paused_before_batch'],16)
            self.assertFalse(context.last_diagnostics['batches'][-1]['attempted'])
            second = binding_for(adapt_pure_jobs(jobs[8:16],bank),collector.executor,800)
            self.assertTrue((batch_cache_path(root,second)/'completed').exists())
            # A later warm_value phase uses the same context/process budget.
            with patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs',return_value=bank):
                self.assertEqual(context.collect(jobs[:8]),episodes(jobs[:8]))
                with self.assertRaises(PureSegmentPaused):context.collect(jobs[16:])
            self.assertEqual(len(gpu),1)
            self.assertEqual(context.last_diagnostics['new_batches'],1)


class PureProcessBudgetTests(unittest.TestCase):
    def test_process_window_exhaustion_pauses_before_new_gpu_and_keeps_clean_checkpoint(self):
        from unittest.mock import patch
        import time
        from app.modules.card_game.rl.batched_search.device_pure_segments import collect_segments,PureSegmentPaused
        from app.modules.card_game.rl.batched_search.pure_checkpoint import binding_for
        helper=PurePersistentProcessTests();self.addCleanup(helper.doCleanups)
        history,metadata,collector=helper.fixture();collector._process_started=time.monotonic()
        with tempfile.TemporaryDirectory()as root:
            path=Path(root)/'segment-00000002'
            save_history(path,history,binding_for(metadata,collector.executor,collector.max_decisions))
            with patch('app.modules.card_game.rl.batched_search.device_pure_segments._segment')as gpu:
                with self.assertRaises(PureSegmentPaused)as paused:
                    collect_segments(collector,metadata,{},checkpoint_dir=root,process_seconds=1)
                gpu.assert_not_called()
            self.assertEqual(Path(paused.exception.checkpoint).resolve(),path.resolve())
            self.assertFalse((Path(root)/'REJECTED.json').exists())

    def test_unbounded_process_window_rejected(self):
        from app.modules.card_game.rl.batched_search.device_pure_segments import collect_segments
        for seconds in (0,151,True,float('nan')):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                collect_segments(None,None,{},checkpoint_dir='/tmp/unused-budget-fixture',process_seconds=seconds)


if __name__ == '__main__': unittest.main()
