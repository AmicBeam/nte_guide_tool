import unittest
import numpy as np
from unittest.mock import patch
from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from app.modules.card_game.rl.policy_equivalence import (
    representation_groups, grouped_target, irreducible_kl, canonical_root, representation_top_hit)
from scripts.check_duel_v2_value_calibration import probability_metrics


class LearningAuditTest(unittest.TestCase):
    def test_source_snapshot_excludes_credentials_unrelated_untracked_and_symlinks(self):
        from app.modules.card_game.rl.offline_sources import snapshot
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repo'; (root / 'app').mkdir(parents=True)
            names = ['app/shared.py', 'app/credentials.json', 'app/players.json', '.env', 'app/unrelated.py', 'app/owned.py']
            for name in names:
                (root / name).write_text('synthetic fixture')
            outside = Path(temp) / 'outside.py'; outside.write_text('outside fixture')
            (root / 'app/link.py').symlink_to(outside)
            tracked = b'app/shared.py\0app/credentials.json\0app/players.json\0.env\0app/link.py\0'
            with patch('app.modules.card_game.rl.offline_sources.subprocess.run',
                       return_value=SimpleNamespace(stdout=tracked)):
                manifest = snapshot(root, Path(temp) / 'out', ['app/owned.py'])
            self.assertEqual(set(manifest), {'app/shared.py', 'app/owned.py'})

    def test_full_hand_diagnostic_pins_hand_but_not_deck_order_or_real_flags(self):
        from app.modules.card_game.engine.duel_v2 import new_game, observe
        from app.modules.card_game.rl import full_hand_diagnostic as oracle
        state = new_game(seed=27, skip_mulligan=True)
        before = deepcopy(state)
        self.assertTrue(all(c.get('hidden') for c in observe(state, 'a')['sides']['b']['hand']))
        from app.modules.card_game.rl import cross_runtime
        privileged = oracle.for_viewer('a')
        _, (plain_x, plain_c) = cross_runtime.decision(state, 'b')
        _, (opponent_x, opponent_c) = privileged.decision(state, 'b')
        np.testing.assert_array_equal(plain_x, opponent_x)
        np.testing.assert_array_equal(plain_c, opponent_c)
        with self.assertRaises(ValueError): privileged.determinize(state, 'b', 99)
        view = oracle.diagnostic_view(state, 'a')
        self.assertTrue(all(c.get('card_id') for c in view['sides']['b']['hand']))
        world = oracle.determinize(state, 'a', 99)
        self.assertEqual(world['sides']['b']['hand'], state['sides']['b']['hand'])
        altered = deepcopy(state); altered['sides']['b']['deck'].reverse()
        other = oracle.determinize(altered, 'a', 99)
        self.assertEqual([c['card_id'] for c in world['sides']['b']['deck']],
                         [c['card_id'] for c in other['sides']['b']['deck']])
        self.assertEqual(state, before)
        self.assertTrue(all(c.get('hidden') for c in observe(state, 'a')['sides']['b']['hand']))

    def test_natural_q_scale_does_not_amplify_tiny_value_noise(self):
        from app.modules.card_game.rl.information_search import Node
        from app.modules.card_game.rl.recovery_search import improved_policy
        node = Node.create(np.array([.6, .4]), 0.)
        node.visits[:] = [4, 4]; node.total[:] = [0., .00004]
        self.assertGreater(improved_policy(node, 'legacy')[1], .98)
        self.assertAlmostEqual(improved_policy(node, 'natural_wdl')[0], .6, places=4)

    def test_legacy_recovery_traversal_matches_shared_search(self):
        from app.modules.card_game.rl.information_search import search as original
        from app.modules.card_game.rl.recovery_search import search as recovery
        class Runtime:
            @staticmethod
            def decision(state, side):
                return [{'type': 'wait'}, {'type': 'finish'}], (np.array([state.get('step', 0)], np.float32), np.eye(2, dtype=np.float32))
            @staticmethod
            def determinize(state, viewer, seed): return dict(state, latent=seed % 2)
            @staticmethod
            def predict_encoded(policy, x, c): return np.array([.5, 0.]), 0.
            @staticmethod
            def action_scores(policy, x, c): return np.array([.5, 0.])
        def act(state, side, action):
            state = deepcopy(state)
            state.update(phase='finished', winner='a' if action['type'] == 'finish' else 'b')
            return state
        state = dict(phase='playing', active_side='a', turn=1)
        with patch('app.modules.card_game.rl.information_search.apply_action', side_effect=act):
            for noise in (False, True):
                kwargs = dict(seed=3, simulations=32, runtime=Runtime, fast_simulation=False,
                              gumbel_candidates=16, terminal_horizon=3, noise=noise)
                old = original(state, 'a', {'a': None, 'b': None}, algorithm='gumbel', **kwargs)
                new = recovery(state, 'a', {'a': None, 'b': None}, q_scale='legacy', **kwargs)
                for field in ('pi', 'visits', 'mean_values', 'root_prior', 'root_gumbel'):
                    np.testing.assert_array_equal(old[field], new[field])
                self.assertEqual(old['search_choice'], new['search_choice'])
                natural = recovery(state, 'a', {'a': None, 'b': None}, q_scale='natural_wdl', **kwargs)
                self.assertEqual(natural['search_choice'], 1)
                self.assertTrue(natural['complete'])

    def test_identical_candidates_have_quantifiable_unlearnable_target(self):
        c = np.array([[1., 0.], [1., 0.], [2., 0.]])
        pi = np.array([.8, .1, .1])
        groups, mass = grouped_target(c, pi)
        self.assertEqual(groups, [[0, 1], [2]])
        np.testing.assert_allclose(mass, [.9, .1])
        expected = .8 * np.log(.8 / .45) + .1 * np.log(.1 / .45)
        self.assertAlmostEqual(irreducible_kl(c, pi), expected)
        self.assertTrue(representation_top_hit(c, pi, 1))
        self.assertFalse(representation_top_hit(c, pi, 2))
        self.assertAlmostEqual(irreducible_kl(c, [.45, .45, .1]), 0.)

    def test_root_canonicalization_ignores_action_order_but_not_state_or_semantics(self):
        c = np.array([[1., 0.], [1., 0.], [2., 0.]], dtype=np.float32)
        pi = np.array([.8, .1, .1]); x = np.array([.5], dtype=np.float32)
        a, p = canonical_root(x, c, pi)
        order = [2, 0, 1]
        b, q = canonical_root(x, c[order], pi[order])
        self.assertEqual(a, b); np.testing.assert_allclose(p, q)
        self.assertNotEqual(a, canonical_root(x + 1, c, pi)[0])
        self.assertEqual(representation_groups(c.copy()), [[0, 1], [2]])

    def test_wdl_metrics_and_game_weights(self):
        uniform = np.full((2, 3), 1 / 3)
        m = probability_metrics(uniform, [0, 2])
        self.assertAlmostEqual(m['brier'], 2 / 3)
        self.assertAlmostEqual(m['log_loss'], np.log(3))
        m = probability_metrics([[1, 0, 0], [0, 0, 1]], [0, 2])
        self.assertEqual(m['brier'], 0.); self.assertEqual(m['log_loss'], 0.)
        # Nine observations in the first game and one in the second: equal
        # game weight, not a 9:1 vote.
        p = [[1, 0, 0]] * 9 + [[0, 0, 1]]
        m = probability_metrics(p, [0] * 10, [1 / 9] * 9 + [1])
        self.assertAlmostEqual(m['brier'], 1.)
        with self.assertRaises(ValueError):
            probability_metrics([[.2, .2, .2]], [0])


if __name__ == '__main__':
    unittest.main()
