"""Collection-boundary fixtures; these tests do not qualify a CUDA driver."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.modules.card_game.rl.batched_search.device_execution_contract import DeviceExecutionNotReady
from app.modules.card_game.rl.batched_search.device_learning_context import ClosedDeviceLearningContext
from app.modules.card_game.rl import seven_round


class ClosedDeviceLearningContextTest(unittest.TestCase):
    def test_q_profile_mismatch_rejects_before_any_device_dispatch(self):
        driver = SimpleNamespace(q_scale='mctx_default', collect=Mock(), evaluate_jobs=Mock())
        context = self.fixture(driver)
        for method in (context.collect, context.evaluate_jobs):
            for profile in ('natural_wdl', 'unknown', None):
                with self.assertRaisesRegex(ValueError, 'Q profile'):
                    method([dict(id='job', q_scale=profile)])
        driver.collect.assert_not_called()
        driver.evaluate_jobs.assert_not_called()
        job = dict(id='job', q_scale='mctx_default')
        driver.collect.return_value = [dict(job)]
        self.assertEqual(context.collect([job]), [job])

    def fixture(self, driver):
        with patch('app.modules.card_game.rl.batched_search.device_execution_contract.require_closed_device_driver',
                   return_value=driver):
            return ClosedDeviceLearningContext(driver)

    def test_missing_real_driver_rejected_before_output(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)/'training'
            with self.assertRaises(DeviceExecutionNotReady):
                seven_round.run_closed_device_learning(output, {}, device_driver=None)
            self.assertFalse(output.exists())

    def test_service_requires_collection_and_evaluation(self):
        for driver in (SimpleNamespace(collect=lambda jobs: []),
                       SimpleNamespace(evaluate_jobs=lambda jobs: [])):
            with self.assertRaises(ValueError):
                self.fixture(driver)

    def test_cpu_rule_map_and_foreign_collector_are_rejected(self):
        driver = SimpleNamespace(collect=Mock(), evaluate_jobs=Mock())
        context = self.fixture(driver)
        with self.assertRaises(RuntimeError):
            context.map(lambda job: job, [{}])
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                seven_round.collect(context, [{}], folder, rollout=object())
        driver.collect.assert_not_called()

    def test_missing_results_cannot_be_silently_zipped(self):
        driver = SimpleNamespace(collect=Mock(return_value=[]), evaluate_jobs=Mock(return_value=[]))
        context = self.fixture(driver)
        with TemporaryDirectory() as folder:
            output = Path(folder)/'batch'
            with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
                seven_round.collect(context, [{'id': 'job'}], output)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
            context.evaluate_jobs([{'id': 'job'}])

    def test_closed_search_uses_collector_and_post_batch_oracle(self):
        job = {'id': 'job'}
        game = {'complete': True}
        driver = SimpleNamespace(collect=Mock(), evaluate_jobs=Mock(), last_diagnostics={'roots': 2})
        context = self.fixture(driver)
        events = []
        def collect(*args, **kwargs):
            self.assertIs(kwargs['rollout'], context)
            events.append('device_batch_returned')
            return [game]
        def oracle(j, g):
            self.assertEqual(events, ['device_batch_returned'])
            self.assertIs(j, job); self.assertIs(g, game)
            events.append('oracle')
            return {'passed': True}
        with TemporaryDirectory() as folder, patch.object(seven_round, 'collect', side_effect=collect), \
             patch('app.modules.card_game.rl.batched_search.training_rollout.verify_training_replay', side_effect=oracle):
            self.assertEqual(seven_round.collect_search_batch(context, [job], folder, {'preflight': True}), [game])
            self.assertTrue((Path(folder)/'gpu-replay-check.json').exists())
        self.assertEqual(events, ['device_batch_returned', 'oracle'])

    def test_same_count_wrong_order_or_seed_is_rejected(self):
        jobs = [{'id': 'a', 'seed': 3}, {'id': 'b', 'seed': 4}]
        for results in ([jobs[0], jobs[0]], list(reversed(jobs)),
                        [{'id': 'a', 'seed': 9}, jobs[1]]):
            context = self.fixture(SimpleNamespace(collect=lambda _: results,
                                                   evaluate_jobs=lambda _: results))
            for call in (context.collect, context.evaluate_jobs):
                with self.assertRaisesRegex(ValueError, 'identity/order'):
                    call(jobs)


if __name__ == '__main__':
    unittest.main()
