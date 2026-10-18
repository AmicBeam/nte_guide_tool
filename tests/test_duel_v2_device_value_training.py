"""Bounded value stage tests; CPU fixtures never qualify actual CUDA execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch, Mock

from app.modules.card_game.rl.device_value_training import _check_jobs, run_value_training_smoke
from app.modules.card_game.rl.batched_search.device_pure_collection import DevicePureCollector
from app.modules.card_game.rl.batched_search.device_execution_contract import require_closed_device_driver, DeviceExecutionNotReady


class DeviceValueTrainingTest(unittest.TestCase):
    def job(self, seed=1):
        return dict(id='job-'+str(seed), seed=seed, pure=True, training=False, deadline=self.deadline)

    def setUp(self):
        self.deadline = time.time()+600

    def test_explicit_live_cuda_collector_required(self):
        with TemporaryDirectory() as folder:
            out=Path(folder)/'out'
            with self.assertRaises(TypeError):
                run_value_training_smoke(out, collector=SimpleNamespace(collect=Mock()), models={},
                    train_jobs=[self.job()], evaluation_jobs=[self.job(2)], deadline=self.deadline)
            self.assertFalse(out.exists())

    def test_original_job_deadline_and_safe_distinct_ids(self):
        for bad in (dict(self.job(), deadline=self.deadline+1), dict(self.job(), id='../escape'),
                    dict(self.job(), training=True), dict(self.job(), pure=False)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _check_jobs([bad], deadline=self.deadline)
        with self.assertRaises(ValueError):
            _check_jobs([self.job(), self.job()], deadline=self.deadline)

    def test_seed_overlap_fails_before_output_collection_and_update(self):
        collector=object.__new__(DevicePureCollector)
        collector.collect=Mock()
        with TemporaryDirectory() as folder:
            out=Path(folder)/'out'
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                run_value_training_smoke(out, collector=collector, models={},
                    train_jobs=[self.job()], evaluation_jobs=[self.job()], deadline=self.deadline)
            collector.collect.assert_not_called()
            self.assertFalse(out.exists())

    def test_incomplete_collection_never_updates_or_exports(self):
        collector=object.__new__(DevicePureCollector)
        collector.last_diagnostics={'steps':[3], 'natural_games':0}
        collector.collect=Mock(side_effect=RuntimeError('incomplete natural cohort'))
        with TemporaryDirectory() as folder, \
             patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs'), \
             patch('app.modules.card_game.rl.seven_runtime.update') as update, \
             patch('app.modules.card_game.rl.seven_runtime.export') as export:
            out=Path(folder)/'out'
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                run_value_training_smoke(out, collector=collector, models={},
                    train_jobs=[self.job()], evaluation_jobs=[self.job(2)], deadline=self.deadline)
            update.assert_not_called();export.assert_not_called()
            import json
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['optimizer_steps'],0)
            self.assertFalse(report['passed'])
            self.assertEqual(report['last_collection']['natural_games'],0)

    def test_large_pure_cohort_rejected_before_metadata_or_cuda_allocation(self):
        collector=object.__new__(DevicePureCollector)
        with patch('app.modules.card_game.rl.batched_search.device_pure_collection.adapt_pure_jobs') as adapt:
            with self.assertRaisesRegex(ValueError, 'small-batch'):
                collector.collect([self.job(i) for i in range(33)], {})
            adapt.assert_not_called()

    def test_value_collector_cannot_pass_formal_gumbel_gate(self):
        collector=object.__new__(DevicePureCollector)
        with self.assertRaises(DeviceExecutionNotReady):
            require_closed_device_driver(collector,purpose='training')

    def test_collector_closes_owned_partial_arenas_on_failure(self):
        collector=object.__new__(DevicePureCollector)
        collector.seconds,collector.proof_seconds=1,1
        collector.segment_decisions=None
        resource=SimpleNamespace(close=Mock(),physical_loop=object())
        def failed(metadata,models,owned):
            owned.append(resource)
            raise RuntimeError('capture failure')
        collector._collect=failed
        with patch('app.modules.card_game.rl.batched_search.device_pure_collection.adapt_pure_jobs', return_value=SimpleNamespace(reference_inference_keys=())):
            with self.assertRaisesRegex(RuntimeError,'capture failure'):
                collector.collect([self.job()],{})
        resource.close.assert_called_once()
        self.assertIsNone(resource.physical_loop)
        self.assertEqual(collector.last_diagnostics['optimizer_steps'],0)


class DeviceValueBootstrapContextTest(unittest.TestCase):
    def test_existing_warm_collect_routes_only_to_live_pure_gpu(self):
        from app.modules.card_game.rl.batched_search.device_learning_context import DeviceValueCalibrationContext
        from app.modules.card_game.rl import seven_round
        collector=object.__new__(DevicePureCollector)
        jobs=[dict(id='batch-'+str(i), seed=i, pure=True, training=False) for i in range(33)]
        collector.collect=Mock(side_effect=lambda batch,bank:[dict(j, complete=False, reason='fixture') for j in batch])
        context=DeviceValueCalibrationContext(collector)
        with patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs', return_value={'bank':1}), patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs'):
            results=context.collect(jobs)
        self.assertEqual(len(results),33)
        self.assertEqual(len(context.last_diagnostics['batches']),2)
        self.assertEqual([len(c.args[0]) for c in collector.collect.call_args_list],[32,1])
        with self.assertRaises(RuntimeError):context.map(lambda j:j,jobs)
        with self.assertRaises(ValueError):context.collect([dict(jobs[0],pure=False,training=True)])
        with TemporaryDirectory() as folder, patch.object(context,'collect',return_value=[{'complete':False,'reason':'fixture'}]) as live:
            with self.assertRaisesRegex(ValueError,'Incomplete phase'):
                seven_round.collect(context,[jobs[0]],folder)
            live.assert_called_once()

    def test_bootstrap_does_not_disable_formal_entry_gate(self):
        from app.modules.card_game.rl.batched_search.device_learning_context import DeviceValueCalibrationContext
        from app.modules.card_game.rl import seven_round
        collector=object.__new__(DevicePureCollector)
        context=DeviceValueCalibrationContext(collector)
        with TemporaryDirectory() as folder:
            with self.assertRaises(DeviceExecutionNotReady):
                seven_round.run(Path(folder)/'out',{},device_driver=context)
            self.assertFalse((Path(folder)/'out').exists())

class ValueSmokeUpdateContractTest(unittest.TestCase):
    def test_real_optimizer_export_restore_and_frozen_evaluation(self):
        # Actual CPU numerical update; only the CUDA trajectory ingress is a fixture.
        import numpy as np
        import torch
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.rl import seven_runtime as rt, seven_lineup
        from app.modules.card_game.rl.batched_search.soa_training_jobs import policy_source_alias
        torch.set_num_threads(1)
        deadline=time.time()+600
        with TemporaryDirectory() as folder:
            start=Path(folder)/'initial'
            build=next(d for d in STARTER_DECKS if d['id']=='starter')
            net=rt.create_network('starter',hidden=16,device='cpu')
            rt.export(net,start,'starter',build,dict(fixture=True))
            model=rt.model(start,'starter')
            alias=policy_source_alias(str(start),'starter')
            bank={alias:model}
            initial={n:w.copy() for n,w in model.weights.items()}
            def job(seed):
                return dict(id='fixture-'+str(seed),seed=seed,first='a',left='starter',right='starter',
                    pure=True,training=False,deadline=deadline,train_sides=['a','b'],
                    policies={s:(str(start),'starter')for s in ('a','b')})
            collector=object.__new__(DevicePureCollector)
            collector.executor=SimpleNamespace(device=torch.device('cpu'))
            collector.last_diagnostics={'complete':True,'fixture':True}
            seen=[]
            def collect(jobs,models):
                seen.append((jobs,models))
                if len(seen)==2:
                    for numeric in models.values():
                        for n,w in initial.items():
                            if not n.startswith(('wdl.','value_net.')):
                                np.testing.assert_array_equal(numeric.weights[n],w)
                return [dict(id=j['id'],winner='a',complete=True,decisions=1,searched=0,rows=[],
                    episode_actions=[dict(side='a',action={'type':'end_turn'})],value_rows=[
                        dict(x=np.ones(len(seven_lineup.all_features()),dtype=np.float32),z=1.,
                             value_actor=1.,side='a',key='starter',step=0)])for j in jobs]
            collector.collect=collect
            with patch('app.modules.card_game.rl.batched_search.soa_pure_jobs.adapt_pure_jobs'):
                result=run_value_training_smoke(Path(folder)/'out',collector=collector,models=bank,
                    train_jobs=[job(1)],evaluation_jobs=[job(2)],deadline=deadline)
            self.assertTrue(result['passed'])
            self.assertEqual(result['optimizer_steps'],1)
            self.assertEqual(result['formal_policy_updates'],0)
            self.assertEqual(len(seen),2)
            self.assertTrue(result['policy_arrays_unchanged'])
            self.assertTrue(result['numeric_restore_equal'])
            for n,w in model.weights.items():np.testing.assert_array_equal(initial[n],w)

class DevicePureRawExportTest(unittest.TestCase):
    def test_postbatch_raw_export_keeps_coordinates_and_entity_aliases(self):
        import torch
        from tests.test_duel_v2_device_raw_action_tape import DeviceRawActionTapeTest, _actions, _graph, _vectors
        from app.modules.card_game.rl.batched_search.device_pure_collection import _raw_records
        helper=DeviceRawActionTapeTest()
        raw=helper.tape()
        helper.begin(raw)
        graph=_graph(_actions(),'cpu')
        self.assertEqual(raw.append_device(graph,*_vectors(graph)).tolist(),[0,0])
        result=_raw_records(raw)
        self.assertEqual([(r['game'],r['step'],r['seed'],r['actor'])for r in result],[(0,0,0,0),(1,0,1,0)])
        from app.modules.card_game.engine.duel_v2.entities import PlayerEntity
        self.assertIs(type(result[0]['raw_action']['player']),PlayerEntity)
        self.assertIs(result[0]['raw_action']['player']['bag'],result[0]['raw_action']['also'])
        self.assertIs(result[1]['raw_action']['xs'][0],result[1]['raw_action']['xs'][1])


if __name__ == '__main__':
    unittest.main()
