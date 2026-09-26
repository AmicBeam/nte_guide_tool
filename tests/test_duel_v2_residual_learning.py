import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from app.modules.card_game.rl import cross_grounded as grounded
from app.modules.card_game.rl.cross_lineup import all_features, CAND_DIM
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY
from scripts.check_duel_v2_residual_learning import fitted_gate, partitions


def candidates():
    c = np.zeros((4, CAND_DIM), dtype=np.float32)
    c[:, :5] = [[ACT_END, -1, -1, -1, -1], [ACT_ATTACK, 0, -1, 1, -1],
                [ACT_PLAY, 0, 0, 1, -1], [ACT_PLAY, 0, 1, 1, -1]]
    return c


@unittest.skipIf(torch is None, 'optional torch')
class ResidualLearningTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(27)
        from app.modules.card_game.rl.residual_policy import network
        self.net = network(32)
        self.x = np.random.default_rng(1).normal(size=len(all_features())).astype(np.float32)
        self.c = candidates()

    def forward(self, x=None, c=None):
        x = self.x if x is None else x
        c = self.c if c is None else c
        return self.net(torch.as_tensor(x[None]), torch.as_tensor(c[None]),
                        torch.ones((1, len(c)), dtype=torch.bool))[0][0].detach().numpy()

    def test_transform_matches_frozen_visible_features(self):
        actual = self.net.candidate_features(torch.as_tensor(self.x[None]), torch.as_tensor(self.c[None]))
        np.testing.assert_allclose(actual[0].detach().numpy(), grounded.transform(self.x, self.c))

    def test_permutation_padding_and_numeric_parity(self):
        from app.modules.card_game.rl.residual_policy import scores_numpy, wdl_numpy, tensor_shapes
        weights = {n: t.detach().numpy().copy() for n, t in self.net.state_dict().items()}
        self.assertEqual({n: a.shape for n, a in weights.items()}, tensor_shapes(32))
        base = self.forward()
        np.testing.assert_allclose(base, scores_numpy(self.x, self.c, weights), atol=2e-5, rtol=2e-5)
        order = [3, 1, 0, 2]
        np.testing.assert_allclose(self.forward(c=self.c[order]), base[order], atol=1e-6)
        c = np.pad(self.c, ((0, 3), (0, 0)))
        mask = torch.tensor([[True] * 4 + [False] * 3])
        padded, _ = self.net(torch.as_tensor(self.x[None]), torch.as_tensor(c[None]), mask)
        np.testing.assert_allclose(padded[0, :4].detach().numpy(), base, atol=1e-6)
        self.assertTrue(bool((padded[0, 4:] < -1e30).all()))
        for actor in (False, True):
            h = self.net.state_net(torch.as_tensor(self.x[None]))
            p = self.net.wdl(torch.cat((h, torch.tensor([[float(actor)]])), -1)).softmax(-1)
            np.testing.assert_allclose(p[0].detach().numpy(), wdl_numpy(self.x, actor, weights), atol=2e-5)

    def test_fit_same_candidates_to_different_actions_by_state(self):
        """Cannot pass by learning a global play bias or a card-only preference."""
        from app.modules.card_game.rl.residual_runtime import update
        rows = []
        for top in range(4):
            x = np.zeros(len(all_features()), dtype=np.float32)
            x[top] = 1.
            pi = np.full(4, .01, dtype=np.float32); pi[top] = .97
            rows.append(dict(x=x, c=self.c, pi=pi, z=(-1, 0, 1, 1)[top], value_actor=1))
        opt = torch.optim.Adam(self.net.parameters(), lr=.003)
        for _ in range(80):
            stats = update(self.net, opt, rows)
        for top, row in enumerate(rows):
            self.assertEqual(int(self.forward(x=row['x']).argmax()), top)
        self.assertLess(stats['policy_loss'], .3)
        self.assertTrue(all(stats['policy_gradient_norms'][name] > 0
                            for name in ('cand_net', 'score', 'state_net')))

    def test_roundtrip_restore_and_no_website_approval(self):
        from app.modules.card_game.rl import residual_runtime as rt
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        build = next(d for d in STARTER_DECKS if d['id'] == 'starter')
        with tempfile.TemporaryDirectory() as temp:
            rt.export(self.net, temp, 'starter', build, dict(initialization='test_only'))
            policy = rt.model(temp, 'starter')
            np.testing.assert_allclose(self.forward(), policy.scores(self.x, self.c), atol=2e-5)
            restored, _, _ = rt.restore(temp, 'starter')
            for name, tensor in self.net.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[name]))
            with self.assertRaisesRegex(ValueError, 'website'):
                policy.validate_human(build)
            path = Path(temp) / 'starter.json'
            m = json.loads(path.read_text()); m['schema'] = 'cross_five_grounded_wdl_v1'
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'identity'):
                rt.ResidualModel(temp, 'starter')

    def test_checkpoint_restores_adam_next_update_and_rejects_identity_change(self):
        from app.modules.card_game.rl import residual_runtime as rt
        from app.modules.card_game.rl import recovery_checkpoint as cp
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from copy import deepcopy
        build = next(d for d in STARTER_DECKS if d['id'] == 'starter')
        row = dict(x=self.x, c=self.c, pi=np.array([.1, .1, .7, .1]), z=1, value_actor=1)
        opt = torch.optim.Adam(self.net.parameters(), lr=.001)
        for _ in range(3): rt.update(self.net, opt, [row])
        with tempfile.TemporaryDirectory() as temp:
            rt.export(self.net, temp, 'starter', build, dict(test_only=True))
            path = Path(temp) / 'restore.pt'
            rng = np.random.default_rng(27)
            cp.save(path, self.net, opt, directory=temp, key='starter',
                    rng_state=rng.bit_generator.state, metadata=dict(hard_deadline='test_only'))
            restored, restored_opt, data = cp.restore(path, directory=temp, key='starter')
            control = deepcopy(self.net)
            control_opt = torch.optim.Adam(control.parameters()); control_opt.load_state_dict(deepcopy(opt.state_dict()))
            rt.update(control, control_opt, [row]); rt.update(restored, restored_opt, [row])
            self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in control.state_dict().items()))
            self.assertEqual(data['rng_state'], rng.bit_generator.state)
            data['algorithm_sha256'] = 'wrong'
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'identity'):
                cp.restore(path, directory=temp, key='starter')

    def test_cuda_request_never_falls_back_to_cpu_or_creates_output(self):
        if torch.cuda.is_available():
            self.skipTest('This negative check requires CUDA to be absent')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            result = subprocess.run([sys.executable, str(root / 'scripts/check_duel_v2_recovery_chain.py'),
                '--run', '--device', 'cuda', '--models', 'missing', '--value-precheck', 'missing',
                '--output', str(output), '--seconds', '300'], cwd=root, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CUDA requested but unavailable', result.stderr)
            self.assertFalse(output.exists())

    def test_preservation_is_exact_and_value_learning_cannot_erase_policy(self):
        from app.modules.card_game.rl import preserved_policy as preserved
        from app.modules.card_game.rl.cross_teacher_distillation import (
            verify_old_teacher, REPO_TEACHERS, project_observation, project_candidates, teacher_scores)
        from app.modules.card_game.rl.residual_runtime import update
        self.net, build = preserved.create_network('starter', hidden=32)
        teacher = verify_old_teacher(REPO_TEACHERS, 'starter')
        old_c, _ = project_candidates(self.c, teacher)
        expected = teacher_scores(teacher, project_observation(self.x, teacher), old_c)
        np.testing.assert_allclose(self.forward(), expected, atol=2e-4, rtol=2e-4)
        frozen = {name: value.clone() for name, value in self.net.legacy.state_dict().items()}
        opt = torch.optim.Adam(self.net.parameters(), lr=.003)
        for _ in range(10):
            update(self.net, opt, [dict(x=self.x, z=1, value_actor=1)], value_only=True)
        np.testing.assert_allclose(self.forward(), expected, atol=2e-4, rtol=2e-4)
        for name, value in frozen.items():
            self.assertTrue(torch.equal(value, self.net.legacy.state_dict()[name]))
        self.assertFalse(any(name.startswith('legacy') for name, _ in self.net.named_parameters()))
        with tempfile.TemporaryDirectory() as temp:
            preserved.export(self.net, temp, 'starter', build, dict(test_only=True))
            numeric = preserved.PreservedModel(temp, 'starter')
            np.testing.assert_array_equal(numeric.scores(self.x, self.c), expected)
            self.assertFalse(numeric.manifest['automatic_serving_approval'])
        unsupported = self.c.copy(); unsupported[-1, 2] = 110
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.forward(c=unsupported)


class ResidualGatesTest(unittest.TestCase):
    def test_value_precheck_requires_heldout_metrics_and_actual_weight_sha(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.recovery_runtime import value_precheck_ok
        from app.modules.card_game.rl.cross_lineup import identity
        policies = {'a': SimpleNamespace(manifest={'deck': 'starter'}, version='verified-sha')}
        report = dict(schema='recovery_value_precheck_v1', complete=True, passed=True,
                      heldout=True, rule_hash=identity(), models={'starter': 'verified-sha'},
                      metrics={'starter': dict(normal_games=8, brier=.5, log_loss=.9)})
        self.assertTrue(value_precheck_ok(report, policies))
        self.assertFalse(value_precheck_ok(True, policies))
        report['models']['starter'] = 'stale-sha'
        self.assertFalse(value_precheck_ok(report, policies))
        report['models']['starter'] = 'verified-sha'
        report['metrics']['starter']['brier'] = .8
        self.assertFalse(value_precheck_ok(report, policies))

    def test_more_plays_cannot_mask_wrong_actions(self):
        baseline = dict(exact=20, target_play_hit=10, attack_to_play=2, end_to_play=2)
        candidate = dict(n=100, kl=.1, confident_n=80, confident_accuracy=.9, exact=70,
                         target_play=30, target_play_hit=25, attack_n=30, attack_to_play=3,
                         end_n=20, end_to_play=1, play_top=60, end_top=10)
        self.assertFalse(fitted_gate(baseline, candidate)['passed'])
        candidate['attack_to_play'] = 1
        self.assertTrue(fitted_gate(baseline, candidate)['passed'])

    def test_paired_seeds_and_all_five_teams_are_kept_out_of_fit(self):
        games = []
        for seed in range(30):
            for first in ('a', 'b'):
                games.append(dict(seed=seed, first=first, rows=[dict(key=k, pi=[1.]) for k in
                    ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')]))
        train, held = partitions(games, [1, 2])
        train_seeds = {g['seed'] for g in train}; held_seeds = {g['seed'] for g in held}
        self.assertFalse(train_seeds & held_seeds)
        self.assertFalse(train_seeds & {1, 2})
        self.assertFalse(held_seeds & {1, 2})
        for seed in train_seeds:
            self.assertEqual({g['first'] for g in train if g['seed'] == seed}, {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
