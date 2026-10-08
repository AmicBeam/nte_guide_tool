"""Deterministic unit tests for batched Gumbel search traversal generator.

Verifies:
1. Sequential yield of InferenceQuery and response validation (id, shape, finite, version).
2. Acting-side perspective handling and terminal value signs without ply-flipping.
3. Partial simulation completion and cancellation under deadline.
4. Legacy vs Natural WDL scale differences (raw_value writeback on short proofs).
5. Exact numerical equivalence with recovery_search.search on synthetic state trees.
6. CPUOracleBackend interface contract and GPU disclaimer.

Tests do NOT load neural network weights, execute real sampling, or run GPU kernels.
"""
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_search.traversal import (
    InferenceQuery,
    InferenceResponse,
    PolicyIdentity,
    Node,
    search_traversal,
    completed_q,
    improved_policy,
    choose_root,
    search_prior,
    info_key,
)
from app.modules.card_game.rl.batched_search.oracle_backend import CPUOracleBackend
from app.modules.card_game.rl import recovery_search


class FakeBackend:
    """Deterministic synthetic rules backend for testing traversal mechanics."""

    def __init__(self, branching=2, max_depth=3, terminal_depth=2, winner='a', alternate_actors=True):
        self.branching = branching
        self.max_depth = max_depth
        self.terminal_depth = terminal_depth
        self.winner = winner
        self.alternate_actors = alternate_actors
        self.clean_calls = 0

    def acting_side(self, state):
        return state.get('actor', 'a')

    def finished(self, state):
        return state.get('finished', False) or state.get('phase') == 'finished'

    def terminal_value(self, state, viewer):
        winner = state.get('winner')
        if winner == viewer:
            return 1.0
        elif winner is not None:
            return -1.0
        return 0.0

    def decision(self, state, side):
        actions = [{'type': 'play', 'idx': i} for i in range(self.branching)]
        depth = state.get('depth', 0)
        x = np.array([depth, 1.0 if side == 'a' else 0.0], dtype=np.float32)
        c = np.array([[i, depth] for i in range(self.branching)], dtype=np.float32)
        return actions, (x, c)

    def sample_world(self, state, viewer, seed):
        world = dict(state)
        world['seed'] = seed
        return world

    def step(self, state, side, action):
        depth = state.get('depth', 0) + 1
        is_finished = depth >= self.terminal_depth
        if self.alternate_actors:
            next_actor = 'b' if side == 'a' else 'a'
        else:
            next_actor = side
        return {
            'depth': depth,
            'actor': next_actor,
            'finished': is_finished,
            'winner': self.winner if is_finished else None,
            'phase': 'finished' if is_finished else 'playing',
        }

    def clean(self, state):
        self.clean_calls += 1
        return state


def drive_traversal(generator, response_provider=None, limit_steps=500):
    """Drive search_traversal generator with response_provider until StopIteration."""
    if response_provider is None:
        def response_provider(query):
            logits = np.zeros(len(query.c), dtype=np.float64)
            wdl = np.array([0.2, 0.1, 0.7], dtype=np.float64)  # value = 0.5
            return logits, wdl
    steps = 0
    try:
        query = next(generator)
        while steps < limit_steps:
            steps += 1
            resp = response_provider(query)
            query = generator.send(resp)
        raise RuntimeError("Exceeded step limit in driver")
    except StopIteration as stop:
        return stop.value


class CachedRootTest(unittest.TestCase):
    def test_cached_root_preserves_fixed_budget_and_rejects_foreign_identity(self):
        from dataclasses import replace
        from app.modules.card_game.rl.batched_search.traversal import RootEvaluation
        backend=FakeBackend();state=dict(phase='playing',actor='a',depth=0)
        policies={'a':('m','sha'),'b':('m','sha')}
        actions,(x,c)=backend.decision(state,'a')
        query=InferenceQuery('cached','m','sha',x,c,True,'a')
        cache=RootEvaluation(state,'a',actions,query,InferenceResponse(query.id,
            np.zeros(len(c),dtype=np.float64),np.array([.2,.1,.7]),'sha'))
        for mode in ('legacy','natural_wdl'):
            options=dict(backend=backend,seed=1972,simulations=32,terminal_horizon=2,q_scale=mode,noise=True)
            baseline=drive_traversal(search_traversal(state,'a',policies,**options))
            actual=drive_traversal(search_traversal(state,'a',policies,root_evaluation=cache,**options))
            for key in ('visits','mean_values','pi','root_prior'):np.testing.assert_array_equal(actual[key],baseline[key])
            for key in ('search_choice','value','simulations','worlds'):self.assertEqual(actual[key],baseline[key])
        for bad in (replace(cache,state=dict(state)),replace(cache,viewer='b'),
                    replace(cache,query=replace(query,model_version='foreign'))):
            with self.assertRaises(ValueError):next(search_traversal(state,'a',policies,backend=backend,seed=1,root_evaluation=bad))


class TestBatchedTraversal(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(branching=2, terminal_depth=2, winner='a')
        self.root_state = {'phase': 'playing', 'actor': 'a', 'depth': 0, 'game_id': 'root-test'}
        self.policies = {
            'a': PolicyIdentity('policy_a', 'hash_a_v1'),
            'b': PolicyIdentity('policy_b', 'hash_b_v1'),
        }

    def test_step_by_step_yield_and_query_identity(self):
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=2,
            gumbel_candidates=2, terminal_horizon=0, q_scale='natural_wdl',
        )
        # Root query
        q0 = next(gen)
        self.assertIsInstance(q0, InferenceQuery)
        self.assertTrue(q0.id.startswith('root-test:'))
        self.assertTrue(q0.id.endswith(':0'))
        self.assertEqual(q0.policy_key, 'policy_a')
        self.assertEqual(q0.model_version, 'hash_a_v1')
        self.assertIs(q0.value_actor, True)
        self.assertEqual(q0.observer_side, 'a')
        self.assertEqual(len(q0.c), 2)

        # Send root response
        resp_logits = np.array([0.1, -0.1], dtype=np.float64)
        resp_wdl = np.array([0.1, 0.2, 0.7], dtype=np.float64)  # value = 0.6
        q1 = gen.send(InferenceResponse(q0.id, resp_logits, resp_wdl, q0.model_version))

        # Interior / simulation query
        self.assertIsInstance(q1, InferenceQuery)
        self.assertEqual(q1.id, q0.id.rsplit(':', 1)[0] + ':1')
        self.assertIsInstance(q1.value_actor, bool)

        # Complete traversal
        def provider(query):
            logits = np.zeros(len(query.c), dtype=np.float64)
            wdl = np.array([0.2, 0.2, 0.6], dtype=np.float64)
            return InferenceResponse(query.id, logits, wdl, query.model_version)

        res = None
        curr_q = q1
        try:
            while True:
                curr_q = gen.send(provider(curr_q))
        except StopIteration as stop:
            res = stop.value

        self.assertIsNotNone(res)
        self.assertEqual(res['simulations'], 2)
        self.assertEqual(res['complete'], True)
        self.assertEqual(res['q_scale'], 'natural_wdl')
        self.assertIn('pi', res)
        self.assertIn('visits', res)
        self.assertIn('mean_values', res)
        self.assertIn('search_choice', res)
        self.assertIn('raw_choice', res)

    def test_query_id_mismatch_rejected(self):
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0,
        )
        q = next(gen)
        with self.assertRaises(ValueError):
            gen.send(InferenceResponse('wrong_query_id', np.zeros(len(q.c)), np.array([0.2, 0.2, 0.6])))

    def test_model_version_mismatch_rejected(self):
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0,
        )
        q = next(gen)
        with self.assertRaises(ValueError):
            gen.send(InferenceResponse(q.id, np.zeros(len(q.c)), np.array([0.2, 0.2, 0.6]), model_version='wrong_version'))

    def test_invalid_logits_shape_or_nan_rejected(self):
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0,
        )
        q = next(gen)
        # Wrong length
        with self.assertRaises(ValueError):
            gen.send(InferenceResponse(q.id, np.zeros(len(q.c) + 1), np.array([0.2, 0.2, 0.6])))

        gen2 = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0,
        )
        q2 = next(gen2)
        # NaN in logits
        nan_logits = np.array([np.nan] * len(q2.c))
        with self.assertRaises(ValueError):
            gen2.send(InferenceResponse(q2.id, nan_logits, np.array([0.2, 0.2, 0.6])))

    def test_invalid_wdl_rejected(self):
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0,
        )
        q = next(gen)
        # NaN in wdl
        with self.assertRaises(ValueError):
            gen.send(InferenceResponse(q.id, np.zeros(len(q.c)), np.array([np.nan, 0.2, 0.6])))

        # Out of [-1, 1] range for natural_wdl
        gen2 = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=1,
            terminal_horizon=0, q_scale='natural_wdl',
        )
        q2 = next(gen2)
        with self.assertRaises(ValueError):
            gen2.send(InferenceResponse(q2.id, np.zeros(len(q2.c)), 2.5))

    def test_cross_acting_side_signs_and_no_ply_negation(self):
        # When actor takes consecutive turns, value should not flip by ply count.
        backend_same_actor = FakeBackend(branching=2, terminal_depth=2, winner='a', alternate_actors=False)
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=backend_same_actor, seed=42, simulations=1,
            terminal_horizon=0, q_scale='natural_wdl',
        )
        res = drive_traversal(gen)
        self.assertEqual(res['simulations'], 1)
        # Terminal state reached at depth 2 with winner == 'a' (viewer 'a') -> terminal value is +1.0
        self.assertTrue((res['mean_values'] >= 0.0).all())

        # If winner is 'b' (viewer 'a' loses), terminal value is -1.0
        backend_loss = FakeBackend(branching=2, terminal_depth=1, winner='b', alternate_actors=False)
        gen_loss = search_traversal(
            self.root_state, 'a', self.policies,
            backend=backend_loss, seed=42, simulations=1,
            terminal_horizon=0, q_scale='natural_wdl',
        )
        res_loss = drive_traversal(gen_loss)
        self.assertTrue((res_loss['mean_values'] <= 0.0).all())

    def test_partial_completion_and_deadline(self):
        class ManualClock:
            now = 0.
            def __call__(self): return self.now
        clock = ManualClock()
        backend = FakeBackend(terminal_depth=1)
        sample = backend.sample_world
        calls = []
        def world(state, viewer, seed):
            calls.append(seed)
            if len(calls) == 2: clock.now = 5.
            return sample(state, viewer, seed)
        backend.sample_world = world
        gen = search_traversal(self.root_state, 'a', self.policies,
            backend=backend, seed=42, simulations=4, terminal_horizon=0,
            clock=clock, time_limit=5.)
        result = drive_traversal(gen)
        self.assertEqual(result['simulations'], 1)
        self.assertFalse(result['complete'])
        self.assertEqual(int(result['visits'].sum()), 1)

    def test_zero_completed_simulations_returns_prior(self):
        class ExpiredClock:
            def __call__(self):
                return 100.0

        clock = ExpiredClock()
        gen = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=4,
            terminal_horizon=0, clock=clock, deadline=10.0,
        )
        # Root evaluation still yields one query
        q0 = next(gen)
        resp_logits = np.array([2.0, -1.0], dtype=np.float64)
        expected_prior = search_prior(resp_logits)
        try:
            gen.send((resp_logits, np.array([0.3, 0.2, 0.5])))
        except StopIteration as stop:
            res = stop.value

        self.assertEqual(res['simulations'], 0)
        self.assertEqual(res['complete'], False)
        self.assertEqual(int(res['visits'].sum()), 0)
        np.testing.assert_allclose(res['visit_pi'], expected_prior)
        np.testing.assert_allclose(res['root_prior'], expected_prior)

    def test_legacy_vs_natural_wdl_scale(self):
        # In natural_wdl: leaf short proof does NOT overwrite tree[key].raw_value
        # In legacy: leaf short proof DOES overwrite tree[key].raw_value
        gen_natural = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=2,
            terminal_horizon=1, q_scale='natural_wdl',
        )
        res_nat = drive_traversal(gen_natural)
        self.assertEqual(res_nat['q_scale'], 'natural_wdl')

        gen_legacy = search_traversal(
            self.root_state, 'a', self.policies,
            backend=self.backend, seed=42, simulations=2,
            terminal_horizon=1, q_scale='legacy',
        )
        res_leg = drive_traversal(gen_legacy)
        self.assertEqual(res_leg['q_scale'], 'legacy')

    def test_numerical_equivalence_with_recovery_search(self):
        """Verify bitwise-identical search decisions against recovery_search on synthetic tree."""
        synthetic_backend = FakeBackend(branching=2, max_depth=3, terminal_depth=2, winner='a')
        root = {'phase': 'playing', 'actor': 'a', 'depth': 0, 'active_side': 'a'}
        seed = 12345
        simulations = 4

        # Fixed deterministic predictions
        fixed_logits = np.array([0.5, -0.5], dtype=np.float64)
        fixed_val = 0.4
        fixed_wdl = np.array([0.2, 0.2, 0.6], dtype=np.float64)  # wdl[2] - wdl[0] = 0.4

        class MockRuntime:
            def decision(self, state, side):
                return synthetic_backend.decision(state, side)

            def predict_encoded(self, policy, x, c):
                return fixed_logits.copy(), fixed_val

            def action_scores(self, policy, x, c):
                return fixed_logits.copy()

        mock_runtime = MockRuntime()
        policies = {'a': 'dummy_a', 'b': 'dummy_b'}

        # Run recovery_search.search with patched rules
        with patch.object(recovery_search.core, 'sample_world', side_effect=lambda state, viewer, seed, **kw: synthetic_backend.sample_world(state, viewer, seed)), \
             patch.object(recovery_search.core, 'simulate_action', side_effect=synthetic_backend.step), \
             patch.object(recovery_search.core, 'acting_side', side_effect=synthetic_backend.acting_side), \
             patch.object(recovery_search.core, 'clean', side_effect=synthetic_backend.clean), \
             patch.object(recovery_search.core, 'terminal_value', side_effect=synthetic_backend.terminal_value):
            ref_result = recovery_search.search(
                root, 'a', policies,
                seed=seed, runtime=mock_runtime, simulations=simulations,
                gumbel_candidates=2, terminal_horizon=0, max_depth=3,
                noise=False, q_scale='natural_wdl', teacher_mode='network',
            )

        # Run search_traversal generator driven with identical predictions
        gen = search_traversal(
            root, 'a', {'a': ('dummy_a', 'v1'), 'b': ('dummy_b', 'v1')},
            backend=synthetic_backend, seed=seed, simulations=simulations,
            gumbel_candidates=2, terminal_horizon=0, max_depth=3,
            noise=False, q_scale='natural_wdl',
        )

        def provider(query):
            return fixed_logits.copy(), fixed_wdl.copy()

        traversal_result = drive_traversal(gen, response_provider=provider)

        # Assert equivalence
        self.assertEqual(traversal_result['simulations'], ref_result['simulations'])
        self.assertEqual(traversal_result['nodes'], ref_result['nodes'])
        self.assertEqual(traversal_result['search_choice'], ref_result['search_choice'])
        self.assertEqual(traversal_result['raw_choice'], ref_result['raw_choice'])
        np.testing.assert_array_equal(traversal_result['visits'], ref_result['visits'])
        np.testing.assert_allclose(traversal_result['mean_values'], ref_result['mean_values'])
        np.testing.assert_allclose(traversal_result['pi'], ref_result['pi'])
        np.testing.assert_allclose(traversal_result['root_prior'], ref_result['root_prior'])

    def test_nonacting_leaf_keeps_root_observer_and_false_actor_context(self):
        backend = FakeBackend(terminal_depth=20)
        queries = []
        def provider(query):
            queries.append(query)
            value = np.array([.8, .1, .1]) if query.value_only else np.array([.1, .1, .8])
            return np.zeros(len(query.c)), value
        gen = search_traversal(self.root_state, 'a', self.policies, backend=backend,
            seed=3, simulations=1, max_depth=1, terminal_horizon=0)
        result = drive_traversal(gen, response_provider=provider)
        leaf = queries[-1]
        self.assertTrue(leaf.value_only)
        self.assertEqual(leaf.policy_key, 'policy_a')
        self.assertEqual(leaf.observer_side, 'a')
        self.assertIs(leaf.value_actor, False)
        self.assertEqual(len(leaf.c), 0)
        np.testing.assert_allclose(result['mean_values'][result['visits'] > 0], [-.7])

    def test_late_inference_does_not_finish_a_half_simulation(self):
        class Clock:
            now = 0.
            def __call__(self): return self.now
        clock = Clock()
        gen = search_traversal(self.root_state, 'a', self.policies, backend=self.backend,
            seed=3, simulations=2, terminal_horizon=0, clock=clock, time_limit=3.)
        q0 = next(gen)
        q1 = gen.send((np.zeros(len(q0.c)), np.array([.2,.2,.6])))
        clock.now = 3.
        with self.assertRaises(StopIteration) as stop:
            gen.send((np.zeros(len(q1.c)), np.array([.2,.2,.6])))
        result = stop.exception.value
        self.assertEqual(result['simulations'], 0)
        self.assertEqual(result['visits'].sum(), 0)
        self.assertEqual(result['stop_reason'], 'deadline')

    def test_cancellation_while_waiting_retains_only_previous_complete_work(self):
        from app.modules.card_game.rl.batched_search.traversal import TraversalCancelled
        gen = search_traversal(self.root_state, 'a', self.policies, backend=self.backend,
            seed=3, simulations=2, terminal_horizon=0)
        q0 = next(gen)
        gen.send((np.zeros(len(q0.c)), np.array([.2,.2,.6])))
        with self.assertRaises(StopIteration) as stop: gen.throw(TraversalCancelled())
        self.assertEqual(stop.exception.value['simulations'], 0)
        self.assertEqual(stop.exception.value['stop_reason'], 'cancelled')

    def test_same_game_root_epoch_ids_are_distinct_and_probabilities_valid(self):
        first = search_traversal(self.root_state, 'a', self.policies, backend=self.backend, seed=3)
        second = search_traversal(self.root_state, 'a', self.policies, backend=self.backend, seed=3)
        a = next(first); b = next(second)
        self.assertNotEqual(a.id, b.id)
        with self.assertRaises(ValueError):first.send((np.zeros(len(a.c)), np.array([-.1,1.2,-.1])))
        second.close()

    def test_cpu_oracle_backend_interface(self):
        backend = CPUOracleBackend(fast_simulation=True)
        # Check that all 7 required interface methods exist and are callable
        required_methods = [
            'acting_side', 'finished', 'terminal_value',
            'decision', 'sample_world', 'step', 'clean',
        ]
        for name in required_methods:
            self.assertTrue(hasattr(backend, name), f"Missing method {name}")
            self.assertTrue(callable(getattr(backend, name)), f"Method {name} is not callable")

        # Test terminal_value and finished on synthetic dictionary states
        self.assertTrue(backend.finished({'phase': 'finished'}))
        self.assertFalse(backend.finished({'phase': 'playing'}))
        self.assertEqual(backend.terminal_value({'winner': 'a'}, 'a'), 1.0)
        self.assertEqual(backend.terminal_value({'winner': 'b'}, 'a'), -1.0)
        self.assertEqual(backend.terminal_value({'winner': None}, 'a'), 0.0)

        # Verify documentation disclaims GPU execution
        self.assertEqual(backend.backend_kind, 'python_oracle')
        self.assertFalse(backend.supports_gpu)


if __name__ == '__main__':
    unittest.main()
