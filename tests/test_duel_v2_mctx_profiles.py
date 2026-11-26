"""Q profiles: official comparison, real Torch statistics, explicit CUDA checks."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from app.modules.card_game.rl.mctx_q import completed_q
from app.modules.card_game.rl.batched_search.mctx_reference import (
    _prepare_payload, _compare_outputs, make_bandit_fixture,
    replay_device_gumbel_semantics, run_official_mctx, probe_official_mctx,
)


class MctxProfilesTest(unittest.TestCase):
    def test_cooperative_final_policy_matches_the_scalar_mctx_reference(self):
        from app.modules.card_game.rl.batched_search import traversal
        from app.modules.card_game.rl import recovery_search
        from app.modules.card_game.rl.information_search import Node
        node = Node.create(np.array([.75,.25]), -.2)
        node.visits[:] = [2,1]
        node.total[:] = [-.3,.6]
        for profile in ('natural_wdl','mctx_default'):
            np.testing.assert_allclose(traversal.completed_q(node,profile),recovery_search.completed_q(node,profile))
            np.testing.assert_allclose(traversal.improved_policy(node,profile),recovery_search.improved_policy(node,profile))

    def test_rescaling_excludes_padding_and_handles_equal_values(self):
        p = np.array([.75, 0., .25, 0.])
        n = np.array([2, 0, 0, 0])
        totals = np.array([2., 0., 0., 0.])
        legal = np.array([True, False, True, False])
        np.testing.assert_allclose(completed_q(p, n, totals, 0., legal), [5.2, 0., 0., 0.])
        np.testing.assert_allclose(completed_q(p, n, totals, 0., legal, q_scale='natural_wdl'),
                                   [5.2, 0., 5.2*2/3, 0.])
        np.testing.assert_array_equal(completed_q(p, n*0, totals*0, .2, legal), np.zeros(4))
        with self.assertRaises(ValueError):
            completed_q(p, n, totals*2, 0., legal)
        with self.assertRaises(ValueError):
            completed_q(p, n, totals, 0., legal, q_scale='unknown')

    def test_torch_complete_budget_matches_numpy_for_both_profiles(self):
        import torch
        torch.set_num_threads(1)
        from app.modules.card_game.rl.mctx_validation import verify_statistics
        report = verify_statistics()
        self.assertTrue(report['passed'])
        self.assertFalse(report['cuda_executed'])
        self.assertFalse(report['training_qualified'])
        self.assertEqual([r['q_scale'] for r in report['profiles']], ['natural_wdl', 'mctx_default'])
        self.assertEqual([c['simulations'] for c in report['profiles'][0]['cases']], [1]+[32]*8)
        json.dumps(report)

    def test_official_library_both_profiles_with_same_gumbel(self):
        if not probe_official_mctx().available:
            self.skipTest('Official JAX/Mctx optional oracle not installed')
        for profile in ('natural_wdl', 'mctx_default'):
            for noise in ([0., 0., 0.], [.1, .8, -.2]):
                payload = make_bandit_fixture()
                payload['roots'][0].update(q_scale=profile, gumbel=noise)
                root = _prepare_payload(payload)['roots'][0]
                local = replay_device_gumbel_semantics(root)
                official = run_official_mctx(root, exact=True)
                self.assertEqual(official['status'], 'ran', official)
                self.assertFalse(_compare_outputs(local, official, atol=1e-5, rtol=1e-5, require_all=True))

    def test_official_q_transform_on_compact_legal_axes(self):
        if not probe_official_mctx().available:
            self.skipTest('Official JAX/Mctx optional oracle not installed')
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        import mctx
        examples = [([.75, .25], [2, 0], [1.2, 0.], -.2),
                    ([.2, .3, .5], [2, 3, 1], [.4, .6, .2], .2),
                    ([.2, .3, .5], [2, 3, 1], [.4, .6+1e-10, .2], .2),
                    ([.5, .5], [0, 0], [0., 0.], .1)]
        with jax.experimental.enable_x64():
            for profile in ('natural_wdl', 'mctx_default'):
                for p, n, total, value in examples:
                    prior, visits, sums = np.array(p), np.array(n), np.array(total)
                    raw_q = np.divide(sums, visits, out=np.zeros_like(sums), where=visits > 0)
                    tree = SimpleNamespace(qvalues=lambda index: jnp.asarray(raw_q),
                        children_visits=jnp.asarray(visits[None]),
                        children_prior_logits=jnp.asarray(np.log(prior)[None]),
                        raw_values=jnp.asarray([value]))
                    expected = mctx.qtransform_completed_by_mix_value(tree, jnp.asarray(0),
                        rescale_values=profile == 'mctx_default')
                    actual = completed_q(prior, visits, sums, value, np.ones(len(prior), bool), q_scale=profile)
                    np.testing.assert_allclose(actual, np.asarray(expected), atol=1e-12, rtol=1e-12)

    def test_unknown_profile_rejected_before_replay(self):
        payload = make_bandit_fixture()
        payload['roots'][0]['q_scale'] = 'mctx_typo'
        with self.assertRaises(ValueError):
            _prepare_payload(payload)

    def test_mctx_profile_drives_complete_tensor_search(self):
        import torch
        torch.set_num_threads(1)
        from tests.test_duel_v2_device_search import (
            TensorTreeFixture, FixtureInference, InjectiveFixtureKeys, DeviceSearchTest,
        )
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        from app.modules.card_game.rl.information_search import Node
        from app.modules.card_game.rl import recovery_search, search_policy
        rows = 8
        fixture = TensorTreeFixture(rows)
        stats = DeviceStatisticsPool(rows*33, 3, device='cpu', q_scale='mctx_default',
            max_batch=rows, max_path_len=4, allow_cpu_reference=True)
        engine = DeviceGumbelSearch(fixture, FixtureInference(), FixtureInference(), stats,
            InjectiveFixtureKeys(), rows=rows, max_actions=3, simulations=32, max_depth=4,
            device='cpu', allow_cpu_reference=True, allow_synthetic=True, q_scale='mctx_default')
        helper = DeviceSearchTest()
        roots = helper.initialize(engine, fixture)
        result = helper.finish_cpu(engine)
        self.assertEqual(result.errors.tolist(), [0]*rows)
        self.assertEqual(result.completed.tolist(), [32]*rows)
        prior = search_policy.normalized_exp(np.asarray([.2, 0., -.1], np.float32).astype(np.float64))
        node = Node.create(prior, 0.)
        for visit in search_policy.visit_schedule(3, 32):
            a = recovery_search.choose_root(node, np.zeros(3), visit, 'mctx_default')
            node.visits[a] += 1
            node.total[a] += [-1., 1., 0.][a]
        for row in range(rows):
            np.testing.assert_array_equal(result.visits[row].numpy(), node.visits)
            np.testing.assert_allclose(result.policy[row].numpy(),
                recovery_search.improved_policy(node, 'mctx_default'), atol=1e-7)
        self.assertEqual(int(fixture.live.sum()), rows)
        self.assertTrue(torch.equal(fixture.data[roots.slots], torch.tensor([[0, -1]]*rows)))

    def test_plan_does_not_import_cuda_or_allocate_seeds(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root/'scripts/prepare_duel_v2_mctx.py')]
        done = subprocess.run(command, capture_output=True, text=True, cwd=root, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertFalse(plan['executed'])
        self.assertFalse(plan['training_started'])
        self.assertFalse(plan['cuda_qualified'])
        self.assertEqual(plan['profiles'], ['natural_wdl', 'mctx_default'])

    def test_cuda_request_without_cuda_fails_and_records_no_pass(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest('This rejection fixture requires no CUDA')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'failure'
            done = subprocess.run([sys.executable, str(root/'scripts/prepare_duel_v2_mctx.py'),
                                   '--verify-cuda', '--output', str(out)],
                                  capture_output=True, text=True, cwd=root, timeout=30)
            self.assertEqual(done.returncode, 2, done.stderr)
            result = json.loads((out/'report.json').read_text())
            self.assertFalse(result['passed'])
            self.assertIn('no CPU fallback', result['error'])

    def test_actual_cuda_graph_replay_for_both_profiles(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA is required; CPU is not a GPU pass')
        from app.modules.card_game.rl.mctx_validation import verify_statistics
        report = verify_statistics(device='cuda', capture=True)
        self.assertTrue(report['passed'])
        self.assertTrue(report['capture_replayed'])
        self.assertFalse(report['full_rule_search_qualified'])


if __name__ == '__main__':
    unittest.main()
