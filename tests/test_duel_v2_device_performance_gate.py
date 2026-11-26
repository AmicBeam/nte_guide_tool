"""Gate logic fixtures only; these tests are not actual CUDA evidence."""
from dataclasses import replace
import unittest
from app.modules.card_game.rl.batched_search.device_performance_gate import (
    ProbeBudget, SearchBatchMeasurement, assess_search_throughput, require_training_performance)


def measured(name='one', **updates):
    values=dict(batch_id=name,source_identity='frozen-source',scope='actual_duel_rules',roots=1600,
        distinct_initial_seeds=1600,peak_device_roots=1600,requested_min=32,requested_max=32,
        completed_min=32,completed_max=32,completed_total=51200,search_error_rows=0,game_error_rows=0,
        simulation_budget=32,candidate_cap=16,horizon=3,steady_seconds=20,setup_seconds=100,
        complete_policy_candidates=True,device_resident_search=True)
    values.update(updates);return SearchBatchMeasurement(**values)


class PerformanceGateTest(unittest.TestCase):
    def test_partial_search_never_reports_complete_decision_throughput(self):
        batch=measured(completed_min=8,completed_max=15,completed_total=19200,timed_out=True,search_error_rows=1600)
        self.assertEqual(batch.metrics()['complete_root_decisions_per_second'],0)
        gate=assess_search_throughput([batch],max_batch_seconds=60)
        self.assertFalse(gate['throughput_passed']);self.assertIn('one:completed',gate['reasons'])

    def test_unknown_threshold_and_small_or_synthetic_cohort_cannot_release_training(self):
        with self.assertRaisesRegex(RuntimeError,'threshold_not_set'):require_training_performance([measured(str(i)) for i in range(3)])
        for updates in (dict(scope='synthetic'),dict(roots=8,distinct_initial_seeds=8,peak_device_roots=8,completed_total=256),dict(host_rule_calls=1),dict(complete_policy_candidates=False)):
            batch=measured(**updates)
            self.assertFalse(assess_search_throughput([batch],max_batch_seconds=60,consecutive_batches=1)['throughput_passed'])

    def test_repeated_same_source_full_batches_require_steady_time_limit(self):
        batches=[measured(str(i)) for i in range(3)]
        result=require_training_performance(batches,max_batch_seconds=30)
        self.assertTrue(result['throughput_passed']);self.assertFalse(result['formal_learning_authorized'])
        self.assertFalse(assess_search_throughput(batches,max_batch_seconds=10)['throughput_passed'])
        self.assertFalse(assess_search_throughput([batches[0]]*3,max_batch_seconds=30)['throughput_passed'])
        self.assertFalse(assess_search_throughput([batches[0],batches[1],replace(batches[2],source_identity='other')],max_batch_seconds=30)['throughput_passed'])

    def test_invalid_measurements_rejected_and_both_clocks_bound_probe(self):
        for updates in (dict(steady_seconds=float('nan')),dict(completed_total=1),dict(horizon=True)):
            with self.assertRaises(ValueError):measured(**updates)
        budget=ProbeBudget(setup_seconds=30,execution_seconds=120)
        deadline=budget.deadline('execution',epoch=100,monotonic=20)
        self.assertFalse(budget.expired(deadline,epoch=219,monotonic=139))
        self.assertTrue(budget.expired(deadline,epoch=99,monotonic=140))
        self.assertTrue(budget.expired(deadline,epoch=220,monotonic=1))


if __name__=='__main__':unittest.main()
