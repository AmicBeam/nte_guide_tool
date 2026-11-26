"""Offline Mctx reference harness; missing JAX/Mctx is a skip, never a pass."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from app.modules.card_game.rl.batched_search.mctx_reference import (
    EXACT_GUMBEL_REQUIRED, compare_records, exact_claim_precondition, improved_policy,
    make_bandit_fixture, make_recorded_tree_fixture, mctx_sequence_of_considered_visits,
    natural_completed_q, plan_report, probe_official_mctx,
    select_final, sequential_halving_schedule,
)
from app.modules.card_game.rl.search_policy import completed_q as legacy_completed_q
from app.modules.card_game.rl.search_policy import visit_schedule


ROOT = Path(__file__).resolve().parents[1]


class MctxReferenceTest(unittest.TestCase):
    def test_missing_official_completed_q_cannot_pass_other_matching_axes(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search import mctx_reference
        payload = make_bandit_fixture()
        local = compare_records(payload, exact=True, run_official=False)['cases'][0]['device_semantics']
        fake = dict(status='ran', passed=None, visits=local['visits'],
                    policy=local['policy'], action=local['action'], completed_q=None)
        with patch.object(mctx_reference, 'run_official_mctx', return_value=fake):
            result = compare_records(payload, exact=True, run_official=True)
        self.assertFalse(result['passed'])
        self.assertFalse(result['official_comparison_passed'])
        self.assertIn('completed_q', {m['axis'] for m in result['cases'][0]['mismatches']})

    def test_import_is_lazy_and_avoids_website(self):
        import ast
        from app.modules.card_game.rl.batched_search import mctx_reference
        tree = ast.parse(Path(mctx_reference.__file__).read_text(encoding='utf8'))
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level.append(node.module or '')
        self.assertTrue(all('flask' not in name and 'jax' not in name and name != 'mctx'
                            and 'routes' not in name and 'auth' not in name
                            for name in top_level))
        # Other cases may already have explicitly loaded the optional oracle.
        # Module-scope AST above checks that importing this adapter does not.
        code = ('import sys; from app.modules.card_game.rl.batched_search.mctx_reference '
                'import plan_report; plan_report(); assert "jax" not in sys.modules; '
                'assert "mctx" not in sys.modules')
        process = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_missing_official_mctx_is_not_a_pass(self):
        status = probe_official_mctx()
        self.assertFalse(status.install_attempted)
        report = compare_records(make_bandit_fixture(), exact=True, run_official=True)
        official = report['cases'][0]['official_mctx']
        if status.available:
            self.assertEqual(official['status'], 'ran')
        else:
            self.assertFalse(status.available)
            self.assertTrue(status.skip_reason)
            self.assertIn(official['status'], ('skipped', 'unsupported', 'rejected'))
            self.assertFalse(official['passed'])
            self.assertFalse(official['install_attempted'])
        plan = plan_report()
        self.assertFalse(plan['passed'])
        self.assertFalse(plan['official_mctx']['passed'])
        self.assertFalse(plan['install_attempted'])
        self.assertNotEqual(plan['official_mctx']['status'], 'passed')

    def test_natural_completed_q_is_not_legacy_minmax(self):
        prior = np.array([0.75, 0.25])
        visits = np.array([2, 0])
        total = np.array([2.0, 0.0])
        legal = np.array([True, True])
        natural = natural_completed_q(prior, visits, total, 0.0, legal)
        legacy = legacy_completed_q(prior, visits, total, 0.0)
        np.testing.assert_allclose(legacy, [5.2, 0.0])
        np.testing.assert_allclose(natural, [5.2, 5.2 * (2.0 / 3.0)])
        self.assertFalse(np.allclose(natural, legacy))
        tree = compare_records(make_recorded_tree_fixture(), exact=True, run_official=True)
        q = tree['cases'][0]['device_semantics']['completed_q']
        np.testing.assert_allclose(q, natural)
        self.assertEqual(tree['cases'][0]['device_semantics']['action'], 0)
        official = tree['cases'][0]['official_mctx']
        if official['status'] != 'ran':
            self.assertFalse(official['passed'])

    def test_sequential_halving_matches_published_mctx_sequence(self):
        for n_legal, sims, cap in ((1, 8, 16), (3, 8, 3), (10, 32, 16), (20, 32, 16)):
            considered = min(n_legal, cap, sims)
            device = sequential_halving_schedule(n_legal, sims, cap)
            published = mctx_sequence_of_considered_visits(considered, sims)
            self.assertEqual(device, published)
            self.assertEqual(device, visit_schedule(considered, sims))

    def test_exact_claim_rejected_without_fixed_gumbel(self):
        payload = make_bandit_fixture()
        del payload['roots'][0]['gumbel']
        self.assertEqual(exact_claim_precondition(payload['roots'][0]), EXACT_GUMBEL_REQUIRED)
        report = compare_records(payload, exact=True, run_official=True)
        self.assertFalse(report['passed'])
        self.assertEqual(report['exact']['reject_reason'], EXACT_GUMBEL_REQUIRED)
        self.assertEqual(report['cases'][0]['exact']['status'], 'rejected')

    def test_same_noise_exact_replay_and_first_round_gumbel_topk(self):
        payload = make_bandit_fixture()
        report = compare_records(payload, exact=True, run_official=True, statistical_trials=5)
        local = report['cases'][0]['device_semantics']
        self.assertEqual(local['status'], 'ran')
        self.assertEqual(local['completed_simulations'], 8)
        self.assertEqual(sum(local['visits']), 8)
        self.assertEqual(local['schedule'], sequential_halving_schedule(3, 8, 3))
        prior = np.array([0.5, 0.3, 0.2])
        gumbel = np.array([0.10, 0.80, -0.20])
        first = set(np.argsort(np.log(prior) + gumbel)[-3:])
        self.assertEqual(set(np.flatnonzero(np.array(local['visits']) > 0)), first)
        self.assertGreater(local['visits'][0], local['visits'][2])
        self.assertAlmostEqual(sum(local['policy']), 1.0)
        self.assertEqual(local['action'], int(np.argmax(local['visits'])))
        stats = report['cases'][0]['statistical']
        self.assertTrue(stats['exact_same_noise_self_match'])
        self.assertIn('never satisfy an exact claim', stats['note'])
        self.assertIsInstance(stats['different_prng_action_agreement'], float)
        official = report['cases'][0]['official_mctx']
        if official['status'] != 'ran':
            self.assertFalse(official['passed'])
            self.assertFalse(report['passed'])

    def test_mismatch_detection_on_device_export_fields(self):
        payload = make_bandit_fixture()
        good = compare_records(payload, exact=True, run_official=False)
        self.assertTrue(good['torch_statistics_passed'])
        self.assertFalse(good['official_comparison_passed'])
        self.assertFalse(good['actual_device_export_compared'])
        visits = list(good['cases'][0]['device_semantics']['visits'])
        payload['roots'][0]['device_visits'] = visits
        payload['roots'][0]['device_policy'] = good['cases'][0]['device_semantics']['policy']
        payload['roots'][0]['device_action'] = good['cases'][0]['device_semantics']['action']
        matched = compare_records(payload, exact=True, run_official=False)
        self.assertTrue(matched['passed'])
        payload['roots'][0]['device_visits'] = [visits[0] + 3] + visits[1:]
        payload['roots'][0]['device_action'] = (payload['roots'][0]['device_action'] + 1) % 3
        broken = compare_records(payload, exact=True, run_official=False)
        self.assertFalse(broken['passed'])
        axes = {item['axis'] for item in broken['cases'][0]['mismatches']}
        self.assertIn('visits', axes)
        self.assertIn('final_action', axes)

    def test_hidden_and_reward_contracts_are_unsupported(self):
        hidden = make_bandit_fixture()
        hidden['roots'][0]['action_owner'] = 1
        hidden_report = compare_records(hidden, exact=True, run_official=True)
        self.assertNotEqual(hidden_report['cases'][0]['device_semantics']['status'], 'ran')
        self.assertIn('root_action_owner_is_not_fixed_viewer',
                      hidden_report['cases'][0]['device_semantics']['unsupported'])
        rewards = make_bandit_fixture()
        rewards['roots'][0]['transitions']['0:0']['reward'] = 0.25
        rewards['roots'][0]['transitions']['0:0']['discount'] = 1.0
        reward_report = compare_records(rewards, exact=True, run_official=True)
        unsupported = reward_report['cases'][0]['device_semantics']['unsupported']
        self.assertIn('process_rewards_are_not_device_gumbel_backup', unsupported)
        with self.assertRaises(ValueError):
            compare_records({'kind': 'recurrent_records', 'throughput_claim': 'cuda',
                             'roots': make_bandit_fixture()['roots']})

    def test_plan_script_does_not_install_or_claim_official_pass(self):
        script = ROOT / 'scripts' / 'validate_duel_v2_mctx_reference.py'
        proc = subprocess.run([sys.executable, str(script), '--plan'], cwd=str(ROOT),
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report['mode'], 'plan')
        self.assertFalse(report['passed'])
        self.assertFalse(report['install_attempted'])
        self.assertFalse(report['official_mctx']['passed'])
        self.assertFalse(report['official_mctx']['install_attempted'])
        self.assertIn(report['official_mctx']['status'], ('skipped', 'available_not_run'))
        self.assertEqual(report['timings']['scope'],
                         'host numeric replay and optional JAX CPU; not CUDA search throughput')

    def test_require_official_is_fail_closed_without_matching_mctx(self):
        script = ROOT / 'scripts' / 'validate_duel_v2_mctx_reference.py'
        proc = subprocess.run([sys.executable, str(script), '--run', '--require-official'],
                              cwd=str(ROOT), capture_output=True, text=True, check=False)
        report = json.loads(proc.stdout)
        self.assertFalse(report['install_attempted'])
        if not probe_official_mctx().available:
            self.assertFalse(report['passed'])
            self.assertFalse(report['official_summary']['passed'])
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(report['official_summary']['status'], 'incomplete')
        else:
            self.assertTrue(report['official_summary']['passed'])
            self.assertTrue(report['official_comparison_passed'])
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_select_final_is_max_visit_then_gumbel_score(self):
        prior = np.array([0.4, 0.4, 0.2])
        visits = np.array([3, 3, 1])
        total = np.array([1.5, 0.3, -0.5])
        legal = np.array([True, True, True])
        noise = np.array([0.0, 1.0, 5.0])
        action = select_final(prior, visits, total, 0.0, legal, noise)
        self.assertIn(action, (0, 1))
        self.assertNotEqual(action, 2)
        policy = improved_policy(prior, visits, total, 0.0, legal)
        self.assertAlmostEqual(policy.sum(), 1.0)


if __name__ == '__main__':
    unittest.main()
