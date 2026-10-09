"""Aggregate diagnostics tests with explicit CPU fixtures, not CUDA evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import unittest

from app.modules.card_game.rl.batched_search.training_rollout import (
    TrainingRollout, collect_training_games,
)
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl import seven_round, seven_runtime
from tests.test_duel_v2_gpu_training import FrozenInference, fixtures, jobs


class InferenceCounters(FrozenInference):
    def stats(self):
        return {'device': 'CPU_TEST_FIXTURE', 'calls': len(self.calls)}


class DiagnosticsTests(unittest.TestCase):
    def run_fixture(self, plan, diagnostics=None):
        inference = InferenceCounters()
        backend = ResidentRuleBackend(workers=1, max_games=len(plan), runtime=seven_runtime)
        games = collect_training_games(
            plan, inference=inference, backend=backend,
            models={'starter': SimpleNamespace(version='fixture-v1')},
            search_roots=len(plan), create_states=fixtures,
            allow_cpu_oracle=True, own_backend=True, diagnostics=diagnostics,
        )
        return games

    def test_counters_preserve_actual_fixture_trajectory(self):
        plan = jobs(2)
        baseline = self.run_fixture(plan)
        diagnostics = {}
        profiled = self.run_fixture(plan, diagnostics)
        self.assertEqual([g['final_state_sha256'] for g in baseline],
                         [g['final_state_sha256'] for g in profiled])
        self.assertEqual([g['episode_actions'] for g in baseline],
                         [g['episode_actions'] for g in profiled])
        self.assertGreater(diagnostics['inference']['calls'], 0)
        self.assertIn('parent_rpc_wall_ns', diagnostics['rules'])
        self.assertTrue(diagnostics['scheduler_dispatch'])
        self.assertIsNone(diagnostics['device_statistics'])
        self.assertIsNone(diagnostics['method_error'])
        json.dumps(diagnostics)

    def test_unavailable_counter_does_not_mask_collection_failure(self):
        def fail_stats():
            raise RuntimeError('counter unavailable')

        resource = SimpleNamespace(stats=fail_stats, close=lambda: closed.append(True))
        closed = []
        collector = TrainingRollout(
            inference=resource, backend=resource, models={}, own_backend=True,
        )
        plan = jobs(2)
        plan[1]['id'] = plan[0]['id']
        with self.assertRaisesRegex(ValueError, 'Duplicate training game'):
            collector.collect(plan)
        self.assertEqual(closed, [True])
        self.assertIn('Duplicate training game', collector.last_diagnostics['method_error'])
        self.assertIsNone(collector.last_diagnostics['inference'])
        self.assertIn('counter unavailable', collector.last_diagnostics['diagnostic_errors']['rules'])

    def test_round_persists_diagnostics_when_complete_phase_fails(self):
        collector = SimpleNamespace(last_diagnostics={'method_error': 'original failure'})
        with tempfile.TemporaryDirectory() as out, \
                patch('app.modules.card_game.rl.batched_search.training_rollout.models_for_jobs', return_value={}), \
                patch('app.modules.card_game.rl.batched_search.training_rollout.rollout_from_config', return_value=collector), \
                patch.object(seven_round, 'collect', side_effect=ValueError('original failure')):
            with self.assertRaisesRegex(ValueError, 'original failure'):
                seven_round.collect_search_batch(None, [dict(id='fixture')], out,
                    dict(rule_backend='resident_cpu', search_roots=1))
            self.assertEqual(json.loads((Path(out)/'search-diagnostics.json').read_text()),
                             collector.last_diagnostics)


if __name__ == '__main__':
    unittest.main()
