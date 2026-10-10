"""Numerical and capture-boundary tests; no real model or game-capacity claim."""
import unittest
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl import seven_policy, seven_lineup, seven_grounded
from app.modules.card_game.rl.batched_search.inference import BatchedInference, InferenceRequest
from app.modules.card_game.rl.batched_search.device_inference import DeviceSevenInference


class SyntheticSevenModel:
    def __init__(self, key, seed):
        import torch
        torch.manual_seed(seed)
        net = seven_policy.network(16, 'cpu')
        self.weights = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        self.hidden = 16; self.teacher = None; self.version = 'synthetic-' + key
        self.manifest = dict(schema=seven_policy.SCHEMA, deck=key, hidden=16,
                             sha256=self.version, features=seven_lineup.all_features(),
                             candidates=seven_lineup.candidate_names(),
                             wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
                             reward_mode='terminal_wdl_only', automatic_serving_approval=False)


def inputs():
    rng = np.random.default_rng(184920)
    x = rng.normal(0, .03, (4, len(seven_lineup.all_features()))).astype(np.float32)
    x[:, seven_grounded.FRONT_INDEX] = .1
    c = np.zeros((4, 5, seven_lineup.CAND_DIM), dtype=np.float32)
    for row in range(4):
        for i in range(5):
            c[row, i, :5] = (i % 4, i % 3, -1, 1, i % 3)
            c[row, i, 5:] = i * .01
    c[:, 4] = c[:, 2]
    mask = np.ones((4, 5), dtype=np.bool_); mask[2] = False
    return x, c, mask, np.asarray([0, 1, 0, 1], dtype=np.int64), np.asarray([[1], [0], [0], [1]], dtype=np.float32)


class DeviceInferenceTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.models = {'a': SyntheticSevenModel('a', 7), 'b': SyntheticSevenModel('b', 19)}

    def check_parity(self, device):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        bank = DeviceSevenInference(self.models, device=device, allow_cpu_reference=device == 'cpu')
        runner = BatchedInference(self.models, device=device)
        x, c, mask, ids, actor = inputs()
        args = [torch.as_tensor(a, device=device) for a in (x, c, mask, ids, actor)]
        value_only = torch.as_tensor([False, False, True, False], device=device)
        requests = [InferenceRequest(i, bank.keys[int(ids[i])],
                                     self.models[bank.keys[int(ids[i])]].version, x[i],
                                     c[i] if i != 2 else c[i, :0], bool(actor[i, 0]), i == 2)
                    for i in range(4)]
        expected = runner.predict(requests)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Hot-path CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('Hot-path NumPy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Hot-path scalar')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('Hot-path list')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Hot-path truth')):
            actual = bank.predict_device(*args, value_only=value_only)
        self.assertFalse(bool(actual.invalid_rows.any()))
        logits = actual.logits.detach().cpu().numpy(); wdl = actual.wdl.detach().cpu().numpy()
        for i, row in enumerate(expected):
            if i != 2:
                # Formal seven policy canonicalizes exact representations;
                # a tail GEMM tile may round duplicate physical rows differently.
                canonical = row.logits[seven_policy.alias_indices(c[i])]
                np.testing.assert_allclose(logits[i], canonical, atol=2e-5, rtol=2e-5)
                self.assertEqual(int(logits[i].argmax()), int(canonical.argmax()))
                self.assertEqual(logits[i, 4], logits[i, 2])
            np.testing.assert_allclose(wdl[i], row.wdl, atol=2e-5, rtol=2e-5)
        return bank, args, value_only

    def test_explicit_cpu_numerical_oracle_multiple_models_and_value_only(self):
        self.check_parity('cpu')

    def test_cpu_requires_explicit_reference_and_workspace_is_bounded(self):
        with self.assertRaises(ValueError):
            DeviceSevenInference(self.models, device='cpu')
        import torch
        bank = DeviceSevenInference(self.models, device='cpu', allow_cpu_reference=True,
                                    max_candidate_elements=1)
        args = [torch.as_tensor(a) for a in inputs()]
        with self.assertRaises(ValueError): bank.predict_device(*args)

    def test_invalid_device_model_rows_are_reported_without_host_fallback(self):
        import torch
        bank = DeviceSevenInference(self.models, device='cpu', allow_cpu_reference=True)
        args = [torch.as_tensor(a) for a in inputs()]; args[3][0] = 99
        row = bank.predict_device(*args, value_only=torch.as_tensor([False, False, True, False]))
        self.assertTrue(bool(row.invalid_rows[0]))
        self.assertFalse(bool(row.invalid_rows[1:].any()))

    def test_actual_cuda_device_output_and_graph_capture(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('Actual CUDA required')
        bank, args, value_only = self.check_parity('cuda')
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3): bank.predict_device(*args, value_only=value_only)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = bank.predict_device(*args, value_only=value_only)
        args[3].copy_(args[3].flip(0)); graph.replay(); torch.cuda.synchronize()
        expected = bank.predict_device(*args, value_only=value_only)
        torch.testing.assert_close(captured.logits, expected.logits)
        torch.testing.assert_close(captured.wdl, expected.wdl)
        self.assertFalse(bool(captured.invalid_rows.any()))


if __name__ == '__main__': unittest.main()
