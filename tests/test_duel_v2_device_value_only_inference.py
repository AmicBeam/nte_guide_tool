"""Value-only empty-action inference. Not a 1600-batch or training-capacity claim."""
import inspect
import unittest
from unittest.mock import patch

from tests import test_duel_v2_compact_device_candidates as fixture
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.batched_search.device_memory_bounded_inference import (
    DeviceMemoryBoundInference,
)
from app.modules.card_game.rl.batched_search.device_value_only_inference import (
    DeviceValueOnlyInference,
)


class DeviceValueOnlyInferenceTest(unittest.TestCase):
    def _models(self):
        return {key: fixture.SyntheticSevenModel(key, 77 + i)
                for i, key in enumerate(seven_lineup.KEYS)}

    def _empty_args(self, bank, x, *, ids=None, context=None, value_only=None):
        t = bank.torch
        b = x.shape[0]
        candidates = t.empty((b, 0, bank.candidate_dim), dtype=t.float32, device=x.device)
        valid = t.empty((b, 0), dtype=t.bool, device=x.device)
        if ids is None:
            ids = t.arange(b, device=x.device, dtype=t.int64) % len(bank.keys)
        if context is None:
            context = t.ones((b, 1), dtype=t.float32, device=x.device)
        if value_only is None:
            value_only = t.ones(b, dtype=t.bool, device=x.device)
        return x, candidates, valid, ids, context, value_only

    def _assert_match(self, actual, expected):
        import torch as torch_mod
        self.assertEqual(tuple(actual.logits.shape), (expected.wdl.shape[0], 0))
        self.assertEqual(actual.logits.dtype, torch_mod.float32)
        self.assertEqual(expected.logits.shape, actual.logits.shape)
        self.assertTrue(torch_mod.equal(actual.logits, expected.logits))
        torch_mod.testing.assert_close(actual.wdl, expected.wdl, rtol=0, atol=0,
                                       equal_nan=True)
        self.assertTrue(torch_mod.equal(actual.invalid_rows, expected.invalid_rows))

    def test_cpu_seven_models_match_parent_wdl_invalid_and_empty_logits(self):
        import torch as t
        t.set_num_threads(1)
        cases = fixture._cases()
        models = self._models()
        self.assertEqual(len(models), 7)
        parent = DeviceMemoryBoundInference(models, device='cpu', allow_cpu_reference=True,
                                            max_rows=len(cases))
        child = DeviceValueOnlyInference(models, device='cpu', allow_cpu_reference=True,
                                         max_rows=len(cases))
        try:
            x = fixture._observations(cases, device='cpu')
            args = self._empty_args(child, x)
            value_only = args[-1].clone()
            value_only[1] = False
            ids = args[3].clone()
            ids[0] = 99
            ids[2] = -1
            context = args[4].clone()
            context[3, 0] = 0.5
            x_bad = x.clone()
            x_bad[4, 0] = float('nan')
            x_inf = x.clone()
            x_inf[5, 1] = float('inf')
            payloads = [
                self._empty_args(child, x),
                self._empty_args(child, x, value_only=value_only),
                self._empty_args(child, x, ids=ids),
                self._empty_args(child, x, context=context),
                self._empty_args(child, x_bad),
                self._empty_args(child, x_inf),
            ]
            references = [parent.predict_device(*payload[:-1], value_only=payload[-1])
                          for payload in payloads]
            orig_tower = child._tower
            seen = []

            def wrapped(features, model_ids, prefix):
                seen.append(prefix)
                if prefix != 'value_net':
                    raise AssertionError('candidate tower on value-only path: ' + prefix)
                return orig_tower(features, model_ids, prefix)

            outputs = []
            with patch.object(child, '_tower', wrapped), \
                    patch.object(child, '_features',
                                 side_effect=AssertionError('cand features')), \
                    patch.object(child, '_legacy',
                                 side_effect=AssertionError('legacy')), \
                    patch.object(child, '_aliases',
                                 side_effect=AssertionError('aliases')), \
                    patch.object(t.Tensor, 'cpu', side_effect=AssertionError('Hot-path CPU copy')), \
                    patch.object(t.Tensor, 'numpy', side_effect=AssertionError('Hot-path NumPy')), \
                    patch.object(t.Tensor, 'item', side_effect=AssertionError('Hot-path scalar')), \
                    patch.object(t.Tensor, 'tolist', side_effect=AssertionError('Hot-path list')), \
                    patch.object(t.Tensor, '__bool__', side_effect=AssertionError('Hot-path truth')):
                for x_row, candidates, valid, model_index, value_actor, mask in payloads:
                    actual = child.predict_device(
                        x_row, candidates, valid, model_index, value_actor, value_only=mask)
                    outputs.append(actual)
            for actual, expected in zip(outputs, references):
                self._assert_match(actual, expected)
            self.assertEqual(set(seen), {'value_net'})
            omitted = child.predict_device(*args[:-1])
            omitted_parent = parent.predict_device(*args[:-1])
            self.assertTrue(t.equal(omitted.invalid_rows, omitted_parent.invalid_rows))
            self.assertTrue(bool(omitted.invalid_rows.all()))
            source = inspect.getsource(DeviceValueOnlyInference.predict_device)
            self.assertNotIn('for i in range', source)
            self.assertNotIn('.item(', source)
            self.assertNotIn('.cpu(', source)
        finally:
            parent.close()
            child.close()

    def test_cpu_malformed_inputs_fallback_and_gradient_rejection(self):
        import torch as t
        t.set_num_threads(1)
        cases = fixture._cases()
        models = self._models()
        parent = DeviceMemoryBoundInference(models, device='cpu', allow_cpu_reference=True,
                                            max_rows=len(cases))
        child = DeviceValueOnlyInference(models, device='cpu', allow_cpu_reference=True,
                                         max_rows=len(cases))
        try:
            x = fixture._observations(cases, device='cpu')
            x_in, candidates, valid, ids, context, value_only = self._empty_args(child, x)
            with self.assertRaises(ValueError):
                DeviceValueOnlyInference(models, device='cpu')
            with self.assertRaises(ValueError):
                child.predict_device(x_in.to(t.float64), candidates, valid, ids, context,
                                     value_only=value_only)
            with self.assertRaises(ValueError):
                child.predict_device(x_in, candidates.to(t.float16), valid, ids, context,
                                     value_only=value_only)
            bad_valid = t.zeros((x_in.shape[0], 0), dtype=t.int32)
            with self.assertRaises(ValueError):
                child.predict_device(x_in, candidates, bad_valid, ids, context,
                                     value_only=value_only)
            with self.assertRaises(ValueError):
                child.predict_device(x_in, candidates, valid, ids.to(t.int32), context,
                                     value_only=value_only)
            with self.assertRaises(ValueError):
                child.predict_device(x_in, candidates, valid, ids, context[:, 0],
                                     value_only=value_only)
            with self.assertRaises(ValueError):
                child.predict_device(x_in, candidates, valid, ids, context,
                                     value_only=value_only[:, None])
            tiny = DeviceValueOnlyInference(models, device='cpu', allow_cpu_reference=True,
                                            max_rows=1)
            try:
                with self.assertRaises(ValueError):
                    tiny.predict_device(x_in, candidates, valid, ids, context,
                                        value_only=value_only)
            finally:
                tiny.close()
            with self.assertRaises(RuntimeError):
                child.predict_device(x_in.detach().requires_grad_(True), candidates, valid, ids,
                                     context, value_only=value_only)
            with self.assertRaises(RuntimeError):
                child.predict_device(x_in, candidates.detach().requires_grad_(True), valid, ids,
                                     context, value_only=value_only)
            with self.assertRaises(RuntimeError):
                child.predict_device(x_in, candidates, valid, ids,
                                     context.detach().requires_grad_(True), value_only=value_only)
            dense = t.zeros((x_in.shape[0], 3, child.candidate_dim), dtype=t.float32)
            dense_valid = t.ones((x_in.shape[0], 3), dtype=t.bool)
            dense_mask = t.zeros(x_in.shape[0], dtype=t.bool)
            expected = parent.predict_device(x_in, dense, dense_valid, ids, context,
                                             value_only=dense_mask)
            actual = child.predict_device(x_in, dense, dense_valid, ids, context,
                                          value_only=dense_mask)
            self.assertEqual(tuple(actual.logits.shape), (x_in.shape[0], 3))
            self.assertTrue(t.equal(actual.wdl, expected.wdl))
            self.assertTrue(t.equal(actual.invalid_rows, expected.invalid_rows))
            self.assertTrue(t.equal(actual.logits, expected.logits))
            with self.assertRaises(RuntimeError):
                child.predict_device(x_in.detach().requires_grad_(True), dense, dense_valid, ids,
                                     context, value_only=dense_mask)
        finally:
            parent.close()
            child.close()

    def test_actual_cuda_graph_capture_changing_inputs_ids_and_value_masks(self):
        import torch as t
        if not t.cuda.is_available():
            self.skipTest('Actual CUDA required')
        t.backends.cuda.matmul.allow_tf32 = False
        cases = fixture._cases()
        models = self._models()
        parent = DeviceMemoryBoundInference(models, device='cuda', max_rows=len(cases))
        child = DeviceValueOnlyInference(models, device='cuda', max_rows=len(cases))
        try:
            x = fixture._observations(cases, device='cuda')
            x_in, candidates, valid, ids, context, value_only = self._empty_args(child, x)
            x_before = x_in.clone()
            ids_before = ids.clone()
            mask_before = value_only.clone()
            expected = parent.predict_device(x_in, candidates, valid, ids, context,
                                             value_only=value_only)
            warm = t.cuda.Stream()
            warm.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(warm):
                for _ in range(3):
                    child.predict_device(x_in, candidates, valid, ids, context,
                                         value_only=value_only)
            t.cuda.current_stream().wait_stream(warm)
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                actual = child.predict_device(x_in, candidates, valid, ids, context,
                                              value_only=value_only)
            graph.replay()
            t.cuda.synchronize()
            self._assert_match(actual, expected)
            x_in.copy_(t.roll(x_before, 1, 0))
            ids.copy_(ids_before.flip(0))
            value_only.copy_(~mask_before)
            value_only[0] = True
            graph.replay()
            t.cuda.synchronize()
            changed = parent.predict_device(x_in, candidates, valid, ids, context,
                                            value_only=value_only)
            self._assert_match(actual, changed)
            self.assertTrue(t.equal(x_in, t.roll(x_before, 1, 0)))
            self.assertTrue(t.equal(ids, ids_before.flip(0)))
            self.assertFalse(t.equal(actual.wdl, expected.wdl) and t.equal(
                actual.invalid_rows, expected.invalid_rows))
        finally:
            parent.close()
            child.close()


if __name__ == '__main__':
    unittest.main()
