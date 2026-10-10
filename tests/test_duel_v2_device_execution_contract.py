"""Evidence-schema fixtures only: passing these tests is not CUDA acceptance."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from app.modules.card_game.rl.batched_search.device_execution_contract import (
    SCHEMA, SEARCH_STAGES, DeviceSearchContract, audit_device_search,
)


class DeviceExecutionContractTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        artifacts = {}
        for key in ('trace', 'episodes', 'source_manifest'):
            path = self.root / (key + '.json')
            path.write_text('{"fixture_only":true}', encoding='utf-8')
            artifacts[key + '_artifact'] = dict(path=path.name,
                                               sha256=sha256(path.read_bytes()).hexdigest())
        ids = ['root-' + str(i) for i in range(1600)]
        self.report = dict(
            schema=SCHEMA, runtime='seven', algorithm='gumbel', q_scale='natural_wdl',
            simulations=32, gumbel_candidates=16, rule_device='cuda',
            control_plane='device_resident', source_kind='actual_cuda',
            cpu_rule_calls=0, host_root_steps=0, hot_path_d2h_bytes=0,
            device_errors=0, budget_failures=0, rule_parity_passed=True,
            search_parity_passed=True, replay_verified=True,
            peak_device_memory_bytes=1024, device_memory_budget_bytes=2048,
            search_roots=[dict(root_id=i, real_game=True, complete=True, legal_actions=40,
                               completed_simulations=32, policy_target_complete=True,
                               terminal_game_complete=True) for i in ids],
            dispatches=[dict(stage=stage, active_root_ids=ids, active_rows=1600,
                             device='cuda', event_completed=True, observed_on_device=True,
                             kernel='fixture_' + stage)
                        for stage in sorted(SEARCH_STAGES | {'search_driver'})],
            **artifacts,
        )

    def audit(self, report=None):
        return audit_device_search(self.report if report is None else report,
                                   artifact_root=self.root)

    def test_complete_record_structure_still_requires_independent_cuda_review(self):
        result = self.audit()
        self.assertTrue(result['evidence_consistent'])
        self.assertTrue(result['independent_cuda_review_required'])
        self.assertFalse(result['quality_approved'])
        self.assertEqual(result['observed_device_search_roots'], 1600)

    def test_inflight_peak_cannot_replace_actual_device_execution(self):
        self.report['peak_outstanding_roots'] = 1600
        for d in self.report['dispatches']:
            d['active_root_ids'] = d['active_root_ids'][:32]
            d['active_rows'] = 32
        result = self.audit()
        self.assertFalse(result['evidence_consistent'])
        self.assertIn('capacity:device_search_driver_roots', result['reasons'])
        self.assertEqual(result['observed_device_search_roots'], 32)

    def test_phase_divergence_permits_smaller_stage_batches_with_full_device_driver(self):
        for d in self.report['dispatches']:
            if d['stage'] != 'search_driver':
                d['active_root_ids'] = d['active_root_ids'][:32]
                d['active_rows'] = 32
        self.assertTrue(self.audit()['evidence_consistent'])

    def test_49_game_precheck_does_not_establish_1600_capacity(self):
        self.report['search_roots'] = self.report['search_roots'][:49]
        for d in self.report['dispatches']:
            d['active_root_ids'] = d['active_root_ids'][:49]
            d['active_rows'] = 49
        self.assertFalse(self.audit()['evidence_consistent'])

    def test_cpu_steps_d2h_and_missing_boundary_counters_reject(self):
        for key in ('cpu_rule_calls', 'host_root_steps', 'hot_path_d2h_bytes',
                    'device_errors', 'budget_failures'):
            for value in (1, None, False):
                with self.subTest(key=key, value=value):
                    report = deepcopy(self.report)
                    report[key] = value
                    self.assertIn('boundary:' + key, self.audit(report)['reasons'])

    def test_wrong_runtime_algorithm_scale_or_source_cannot_inherit_readiness(self):
        for key, value in (('algorithm', 'puct'), ('source_kind', 'synthetic'),
                           ('rule_device', 'cpu'), ('control_plane', 'host_cooperative'),
                           ('q_scale', 'legacy'), ('runtime', 'recovery'),
                           ('simulations', 16), ('gumbel_candidates', 8)):
            with self.subTest(key=key):
                report = deepcopy(self.report)
                report[key] = value
                self.assertIn('identity:' + key, self.audit(report)['reasons'])

    def test_shortcut_partial_search_and_incomplete_terminal_do_not_count(self):
        for key, value in (('completed_simulations', 31), ('legal_actions', 1),
                           ('policy_target_complete', False), ('terminal_game_complete', False),
                           ('complete', False), ('real_game', False)):
            with self.subTest(key=key):
                report = deepcopy(self.report)
                report['search_roots'][0][key] = value
                self.assertIn('roots:incomplete_budget_policy_or_game', self.audit(report)['reasons'])

    def test_duplicate_or_foreign_root_id_is_rejected(self):
        report = deepcopy(self.report)
        report['search_roots'][1]['root_id'] = report['search_roots'][0]['root_id']
        self.assertIn('roots:missing_or_duplicate', self.audit(report)['reasons'])
        report = deepcopy(self.report)
        report['dispatches'][0]['active_root_ids'][0] = 'unrelated-root'
        self.assertIn('dispatches:unobserved_or_invalid', self.audit(report)['reasons'])

    def test_missing_stage_and_unobserved_or_duplicate_dispatch_reject(self):
        for change in ('stage', 'observed', 'duplicate'):
            report = deepcopy(self.report)
            if change == 'stage':
                report['dispatches'] = [d for d in report['dispatches'] if d['stage'] != 'backup']
                expected = 'stage:backup'
            elif change == 'observed':
                report['dispatches'][0]['observed_on_device'] = False
                expected = 'dispatches:unobserved_or_invalid'
            else:
                d = report['dispatches'][0]
                d['active_root_ids'][1] = d['active_root_ids'][0]
                expected = 'dispatches:unobserved_or_invalid'
            self.assertIn(expected, self.audit(report)['reasons'])

    def test_artifacts_must_exist_match_sha_and_stay_inside_run(self):
        for name in ('../outside.json', '/tmp/outside.json', 'missing.json'):
            report = deepcopy(self.report)
            report['trace_artifact']['path'] = name
            self.assertIn('artifact:trace_artifact', self.audit(report)['reasons'])
        (self.root / 'trace.json').write_text('changed', encoding='utf-8')
        self.assertIn('artifact:trace_artifact', self.audit()['reasons'])

    def test_unmeasured_memory_and_missing_parity_are_rejected(self):
        self.report['peak_device_memory_bytes'] = 4096
        self.assertIn('capacity:unmeasured_or_over_budget_memory', self.audit()['reasons'])
        self.report['search_parity_passed'] = False
        self.assertIn('verification:search_parity_passed', self.audit()['reasons'])

    def test_contract_rejects_non_integer_or_legacy_limits(self):
        for kwargs in ({'concurrent_roots': True}, {'simulations': 0},
                       {'runtime': 'recovery'}, {'q_scale': 'legacy'}):
            with self.assertRaises(ValueError):
                DeviceSearchContract(**kwargs)


if __name__ == '__main__':
    unittest.main()
