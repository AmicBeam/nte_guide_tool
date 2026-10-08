"""Tests for offline FP32 Torch batched inference bridge."""
import unittest
import numpy as np

from app.modules.card_game.rl import cross_lineup, recovery_policy, residual_policy
from app.modules.card_game.rl.batched_search.inference import (
    BatchedInference,
    InferenceRequest,
    InferenceResult,
    FEATURE_DIM,
    CANDIDATE_DIM,
)


class SyntheticNumericModel:
    """In-memory synthetic model satisfying strict recovery numeric contracts."""

    def __init__(self, key: str, hidden: int = 16, schema: str = recovery_policy.RESIDUAL_SCHEMA):
        import torch
        net = residual_policy.network(hidden=hidden, device='cpu')
        if schema == recovery_policy.RESIDUAL_SCHEMA:
            net = recovery_policy.attach(net)

        expected = residual_policy.tensor_shapes(hidden)
        if schema == recovery_policy.RESIDUAL_SCHEMA:
            expected.update(recovery_policy.extra_shapes(hidden))

        state = net.state_dict()
        self.weights = {name: state[name].detach().cpu().numpy().copy() for name in expected}
        for w in self.weights.values():
            w.flags.writeable = False

        self.hidden = hidden
        self.schema = schema
        self.separated = (schema == recovery_policy.RESIDUAL_SCHEMA)
        self.version = f'synthetic_ver_{key}'
        self.manifest = {
            'schema': schema,
            'deck': key,
            'hidden': hidden,
            'sha256': self.version,
            'features': list(cross_lineup.all_features()),
            'candidates': list(cross_lineup.candidate_names()),
            'wdl_order': ['loss', 'draw', 'win'],
            'value_context': ['viewer_is_actor'],
            'reward_mode': 'terminal_wdl_only',
            'automatic_serving_approval': False,
        }
        self.teacher = None

    def scores_only(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return residual_policy.scores_numpy(x, c, self.weights)

    scores = scores_only

    def wdl(self, x: np.ndarray, is_actor: bool) -> np.ndarray:
        w = recovery_policy.value_weights(self.weights) if self.separated else self.weights
        return residual_policy.wdl_numpy(x, is_actor, w)


def make_legal_observation(seed: int = 0) -> np.ndarray:
    """Generate legal float32 observation with front:1 index set properly."""
    rng = np.random.RandomState(seed)
    x = rng.randn(FEATURE_DIM).astype(np.float32) * 0.05
    front_index = cross_lineup.all_features().index('front:1')
    x[front_index] = 0.0
    return x


def make_legal_candidates(count: int, seed: int = 0) -> np.ndarray:
    """Generate legal float32 candidate actions with valid categorical slots."""
    c = np.zeros((count, CANDIDATE_DIM), dtype=np.float32)
    for i in range(count):
        c[i, 0] = float((seed + i) % 6)       # Action type in [0, 5]
        c[i, 1] = float((seed + i) % 4)       # Actor seat in [0, 3]
        c[i, 2] = -1.0                        # Card index (-1 = no card)
        c[i, 3] = 1.0                         # Target side
        c[i, 4] = float((seed + i) % 4)       # Target seat in [0, 3]
        c[i, 5:] = float(i) * 0.01            # Numeric features
    return c


class TestDuelV2BatchInference(unittest.TestCase):

    def test_expired_request_skips_forward_and_has_explicit_status(self):
        from unittest.mock import patch
        model=SyntheticNumericModel('expired',hidden=16)
        runner=BatchedInference({'expired':model},device='cpu')
        request=InferenceRequest('late','expired',model.version,make_legal_observation(),
            make_legal_candidates(2),True,deadline=0.)
        with patch.object(runner._networks['expired'].state_net,'forward',side_effect=AssertionError('Late forward')):
            [result]=runner.predict([request])
        self.assertTrue(result.expired);self.assertEqual(runner.stats()['rows'],0)

    def test_value_only_uses_independent_value_tower_and_no_policy_rows(self):
        model=SyntheticNumericModel('leaf',hidden=16)
        before={key:value.copy() for key,value in model.weights.items()}
        runner=BatchedInference({'leaf':model},device='cpu')
        x=make_legal_observation(92)
        empty=np.empty((0,CANDIDATE_DIM),dtype=np.float32)
        requests=[InferenceRequest(f'leaf-{actor}','leaf',model.version,x,empty,actor,True)
                  for actor in (False,True)]
        results=runner.predict(requests)
        for req,res in zip(requests,results):
            self.assertEqual(res.logits.shape,(0,))
            np.testing.assert_allclose(res.wdl,model.wdl(x,req.value_actor),atol=1e-5)
        for key,value in model.weights.items():
            np.testing.assert_array_equal(value,before[key])

    def test_padded_scalar_allowance_splits_without_dropping_candidates(self):
        model=SyntheticNumericModel('bounded',hidden=16)
        runner=BatchedInference({'bounded':model},device='cpu',max_batch_rows=10,
                                max_candidate_elements=8*CANDIDATE_DIM)
        x=make_legal_observation(93)
        requests=[InferenceRequest(f'row-{i}','bounded',model.version,x,
                                   make_legal_candidates(n),True)
                  for i,n in enumerate((1,4,8))]
        results=runner.predict(requests)
        self.assertEqual([len(r.logits) for r in results],[1,4,8])
        self.assertEqual(runner.stats()['batches'],2)
        for req,res in zip(requests,results):
            np.testing.assert_allclose(res.logits,model.scores(req.x,req.candidates),atol=1e-5)

    def test_no_torch_at_module_import(self):
        """Verifies that importing inference does not export or leak torch at module level."""
        import app.modules.card_game.rl.batched_search.inference as inf_mod
        self.assertNotIn('torch', inf_mod.__dict__)

    def test_synthetic_model_numeric_contract_and_stats(self):
        """Verifies initialization, schema verification, and initial stats."""
        model = SyntheticNumericModel('deck_synth', hidden=16)
        runner = BatchedInference({'deck_synth': model}, device='cpu', max_batch_rows=4, max_candidate_elements=16 * CANDIDATE_DIM)

        st = runner.stats()
        self.assertEqual(st['batches'], 0)
        self.assertEqual(st['rows'], 0)
        self.assertEqual(st['candidate_padding'], 0)
        self.assertGreater(st['weight_nbytes'], 0)
        self.assertEqual(st['device'], 'cpu')
        self.assertEqual(runner.max_batch_rows, 4)
        self.assertEqual(runner.max_candidate_elements, 16 * CANDIDATE_DIM)

    def test_policy_logits_and_wdl_alignment(self):
        """Verifies policy logits and WDL align with reference NumPy implementation."""
        model = SyntheticNumericModel('deck_a', hidden=16)
        runner = BatchedInference({'deck_a': model}, device='cpu', max_batch_rows=4, max_candidate_elements=16 * CANDIDATE_DIM)

        x = make_legal_observation(seed=10)
        c = make_legal_candidates(count=3, seed=10)

        req = InferenceRequest(
            request_id='r1',
            model_key='deck_a',
            model_version=model.version,
            x=x,
            candidates=c,
            value_actor=True,
        )

        results = runner.predict([req])
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertEqual(res.request_id, 'r1')
        self.assertEqual(res.model_key, 'deck_a')
        self.assertEqual(res.model_version, model.version)
        self.assertEqual(res.logits.shape, (3,))
        self.assertEqual(res.wdl.shape, (3,))

        # Verify numerical alignment with model reference NumPy implementations
        ref_scores = model.scores(x, c)
        ref_wdl = model.wdl(x, True)

        np.testing.assert_allclose(res.logits, ref_scores, atol=1e-5)
        np.testing.assert_allclose(res.wdl, ref_wdl, atol=1e-5)
        self.assertAlmostEqual(float(res.wdl.sum()), 1.0, places=5)

        # Verify returned arrays are independent copies
        self.assertFalse(res.logits.flags.owndata is False and res.logits.base is not None)
        orig_val = res.logits[0]
        res.logits[0] += 10.0
        self.assertNotEqual(res.logits[0], orig_val)
        self.assertEqual(model.scores(x, c)[0], ref_scores[0])

    def test_varying_candidate_padding(self):
        """Verifies padding varying-length candidates in a microbatch and tracking stats."""
        model = SyntheticNumericModel('deck_pad', hidden=16)
        runner = BatchedInference({'deck_pad': model}, device='cpu', max_batch_rows=8, max_candidate_elements=40 * CANDIDATE_DIM)

        x = make_legal_observation(seed=20)
        candidate_lengths = [1, 4, 7, 10]
        requests = []
        for i, count in enumerate(candidate_lengths):
            c = make_legal_candidates(count, seed=i)
            requests.append(InferenceRequest(
                request_id=f'pad_{i}',
                model_key='deck_pad',
                model_version=model.version,
                x=x,
                candidates=c,
                value_actor=bool(i % 2),
            ))

        results = runner.predict(requests)
        self.assertEqual(len(results), 4)

        for i, (res, req) in enumerate(zip(results, requests)):
            self.assertEqual(res.request_id, req.request_id)
            self.assertEqual(res.logits.shape, (candidate_lengths[i],))
            ref_scores = model.scores(x, req.candidates)
            np.testing.assert_allclose(res.logits, ref_scores, atol=1e-5)

        # Padding check: max is 10, padding = (10-1) + (10-4) + (10-7) + (10-10) = 9 + 6 + 3 + 0 = 18
        st = runner.stats()
        self.assertEqual(st['batches'], 1)
        self.assertEqual(st['rows'], 4)
        self.assertEqual(st['candidate_padding'], 18)

    def test_mixed_model_keys_and_order_preservation(self):
        """Verifies mixed model keys are evaluated in separate batches and returned in original order."""
        model_a = SyntheticNumericModel('deck_a', hidden=16)
        model_b = SyntheticNumericModel('deck_b', hidden=16)

        runner = BatchedInference(
            {'deck_a': model_a, 'deck_b': model_b},
            device='cpu',
            max_batch_rows=8,
            max_candidate_elements=16 * CANDIDATE_DIM,
        )

        x = make_legal_observation(seed=30)
        reqs = [
            InferenceRequest('a_1', 'deck_a', model_a.version, x, make_legal_candidates(2, seed=1), True),
            InferenceRequest('b_1', 'deck_b', model_b.version, x, make_legal_candidates(3, seed=2), False),
            InferenceRequest('a_2', 'deck_a', model_a.version, x, make_legal_candidates(4, seed=3), False),
            InferenceRequest('b_2', 'deck_b', model_b.version, x, make_legal_candidates(2, seed=4), True),
            InferenceRequest('a_3', 'deck_a', model_a.version, x, make_legal_candidates(1, seed=5), True),
        ]

        results = runner.predict(reqs)
        self.assertEqual(len(results), 5)

        expected_ids = ['a_1', 'b_1', 'a_2', 'b_2', 'a_3']
        expected_keys = ['deck_a', 'deck_b', 'deck_a', 'deck_b', 'deck_a']

        for res, expected_id, expected_key, req in zip(results, expected_ids, expected_keys, reqs):
            self.assertEqual(res.request_id, expected_id)
            self.assertEqual(res.model_key, expected_key)
            m = model_a if expected_key == 'deck_a' else model_b
            np.testing.assert_allclose(res.logits, m.scores(x, req.candidates), atol=1e-5)

        st = runner.stats()
        self.assertEqual(st['batches'], 2)  # 1 batch for deck_a (3 rows), 1 batch for deck_b (2 rows)
        self.assertEqual(st['rows'], 5)

    def test_microbatch_bounds(self):
        """Verifies microbatch slicing adheres to max_batch_rows and rejects roots exceeding candidate cap."""
        model = SyntheticNumericModel('deck_mb', hidden=16)
        runner = BatchedInference(
            {'deck_mb': model},
            device='cpu',
            max_batch_rows=2,
            max_candidate_elements=8 * CANDIDATE_DIM,
        )

        x = make_legal_observation(seed=40)
        reqs = [
            InferenceRequest(f'mb_{i}', 'deck_mb', model.version, x, make_legal_candidates(c_len, seed=i), True)
            for i, c_len in enumerate([2, 3, 4, 2, 5])
        ]

        results = runner.predict(reqs)
        self.assertEqual(len(results), 5)
        for i, res in enumerate(results):
            self.assertEqual(res.request_id, f'mb_{i}')

        st = runner.stats()
        # 5 rows chunked with max_batch_rows=2 -> 3 microbatches (2, 2, 1)
        self.assertEqual(st['batches'], 3)
        self.assertEqual(st['rows'], 5)

        # Single root exceeding candidate cap must fail clearly
        req_over_cap = InferenceRequest(
            'mb_overflow',
            'deck_mb',
            model.version,
            x,
            make_legal_candidates(9, seed=99),  # Entire root exceeds the scalar tensor cap
            True,
        )
        with self.assertRaises(ValueError) as ctx:
            runner.predict([req_over_cap])
        self.assertIn('max_candidate_elements', str(ctx.exception))

    def test_wdl_two_actor_values(self):
        """Verifies distinct WDL probabilities for actor True vs False (1 vs 0)."""
        model = SyntheticNumericModel('deck_wdl', hidden=16)
        runner = BatchedInference({'deck_wdl': model}, device='cpu')

        x = make_legal_observation(seed=50)
        c = make_legal_candidates(3, seed=50)

        req_actor_1 = InferenceRequest('act_1', 'deck_wdl', model.version, x, c, 1)
        req_actor_0 = InferenceRequest('act_0', 'deck_wdl', model.version, x, c, 0)
        req_actor_true = InferenceRequest('act_t', 'deck_wdl', model.version, x, c, True)
        req_actor_false = InferenceRequest('act_f', 'deck_wdl', model.version, x, c, False)

        results = runner.predict([req_actor_1, req_actor_0, req_actor_true, req_actor_false])

        res_1, res_0, res_t, res_f = results
        np.testing.assert_allclose(res_1.wdl, model.wdl(x, True), atol=1e-5)
        np.testing.assert_allclose(res_0.wdl, model.wdl(x, False), atol=1e-5)
        np.testing.assert_allclose(res_1.wdl, res_t.wdl, atol=1e-6)
        np.testing.assert_allclose(res_0.wdl, res_f.wdl, atol=1e-6)

        self.assertAlmostEqual(float(res_1.wdl.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(res_0.wdl.sum()), 1.0, places=5)

    def test_rejections_and_error_handling(self):
        """Verifies strict rejection of invalid shape, nonfinite values, mismatched version, and duplicate IDs."""
        model = SyntheticNumericModel('deck_err', hidden=16)
        runner = BatchedInference({'deck_err': model}, device='cpu', max_candidate_elements=16 * CANDIDATE_DIM)

        x_valid = make_legal_observation(seed=60)
        c_valid = make_legal_candidates(2, seed=60)

        # 1. Duplicate request_id
        req_dup1 = InferenceRequest('dup', 'deck_err', model.version, x_valid, c_valid, True)
        req_dup2 = InferenceRequest('dup', 'deck_err', model.version, x_valid, c_valid, False)
        with self.assertRaises(ValueError) as ctx:
            runner.predict([req_dup1, req_dup2])
        self.assertIn('Duplicate request_id', str(ctx.exception))

        # 2. Unknown model_key
        req_unknown = InferenceRequest('unk', 'deck_nonexistent', model.version, x_valid, c_valid, True)
        with self.assertRaises((KeyError, ValueError)):
            runner.predict([req_unknown])

        # 3. Model version mismatch
        req_bad_ver = InferenceRequest('ver', 'deck_err', 'wrong_version', x_valid, c_valid, True)
        with self.assertRaises(ValueError) as ctx:
            runner.predict([req_bad_ver])
        self.assertIn('version mismatch', str(ctx.exception).lower())

        # 4. Invalid observation shape
        x_bad_shape = np.zeros(FEATURE_DIM + 1, dtype=np.float32)
        req_bad_x = InferenceRequest('bx', 'deck_err', model.version, x_bad_shape, c_valid, True)
        with self.assertRaises(ValueError):
            runner.predict([req_bad_x])

        # 5. Non-finite observation
        x_nan = x_valid.copy()
        x_nan[0] = np.nan
        req_nan_x = InferenceRequest('nx', 'deck_err', model.version, x_nan, c_valid, True)
        with self.assertRaises(ValueError):
            runner.predict([req_nan_x])

        # 6. Invalid candidates shape
        c_empty = np.zeros((0, CANDIDATE_DIM), dtype=np.float32)
        req_c_empty = InferenceRequest('ce', 'deck_err', model.version, x_valid, c_empty, True)
        with self.assertRaises(ValueError):
            runner.predict([req_c_empty])

        c_bad_dim = np.zeros((2, CANDIDATE_DIM - 1), dtype=np.float32)
        req_c_dim = InferenceRequest('cd', 'deck_err', model.version, x_valid, c_bad_dim, True)
        with self.assertRaises(ValueError):
            runner.predict([req_c_dim])

        # 7. Non-finite candidates
        c_inf = c_valid.copy()
        c_inf[0, 0] = np.inf
        req_inf_c = InferenceRequest('ic', 'deck_err', model.version, x_valid, c_inf, True)
        with self.assertRaises(ValueError):
            runner.predict([req_inf_c])

        # 8. Invalid value_actor
        for bad_actor in (2, -1, 1.5, "true", None):
            req_bad_actor = InferenceRequest('ba', 'deck_err', model.version, x_valid, c_valid, bad_actor)
            with self.assertRaises(ValueError):
                runner.predict([req_bad_actor])

    def test_empty_requests(self):
        """Verifies predict on empty list returns empty list without error."""
        model = SyntheticNumericModel('deck_empty', hidden=16)
        runner = BatchedInference({'deck_empty': model}, device='cpu')
        res = runner.predict([])
        self.assertEqual(res, [])
        self.assertEqual(runner.stats()['batches'], 0)
        self.assertEqual(runner.stats()['rows'], 0)

    def test_cuda_device_if_available(self):
        """Explicitly tests true CUDA device when available, skipped if not available."""
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this environment")

        model = SyntheticNumericModel('deck_cuda', hidden=16)
        runner = BatchedInference({'deck_cuda': model}, device='cuda')

        x = make_legal_observation(seed=70)
        c = make_legal_candidates(3, seed=70)
        req = InferenceRequest('c1', 'deck_cuda', model.version, x, c, True)

        results = runner.predict([req])
        self.assertEqual(len(results), 1)
        self.assertEqual(runner.stats()['device'], 'cuda')

        ref_scores = model.scores(x, c)
        np.testing.assert_allclose(results[0].logits, ref_scores, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
