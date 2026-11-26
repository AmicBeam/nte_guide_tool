"""Compact candidate wrapper tests. Not a 1600-game or training-capacity claim."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import seven_lineup, seven_policy
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder
from app.modules.card_game.rl.batched_search.compact_device_candidates import (
    CompactDeviceCandidates, _TOKEN_TENSORS,
)
from app.modules.card_game.rl.batched_search.device_inference import (
    DeviceSevenInference, DeviceInferenceResult,
)
from app.modules.card_game.rl.batched_search.device_info_key import (
    DeviceInformationKey, DeviceKeyResult,
)


class SyntheticSevenModel:
    def __init__(self, key, seed):
        import torch
        torch.manual_seed(seed)
        net = seven_policy.network(16, 'cpu')
        self.weights = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        self.hidden = 16
        self.teacher = None
        self.version = 'synthetic-' + key
        self.manifest = dict(
            schema=seven_policy.SCHEMA, deck=key, hidden=16, sha256=self.version,
            features=seven_lineup.all_features(), candidates=seven_lineup.candidate_names(),
            wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
            reward_mode='terminal_wdl_only', automatic_serving_approval=False,
        )


def _cases():
    cases = []
    for definition in STARTER_DECKS:
        if definition['id'] not in seven_lineup.KEYS:
            continue
        deck = validate_deck(deepcopy(definition))
        for skip in (False, True):
            view = observe(new_game(seed=7, first_side='a',
                                    decks={'a': deck, 'b': deepcopy(deck)},
                                    skip_mulligan=skip), 'a', include_previews=False)
            actions = [row['action'] for row in view['legal_actions']
                       if row['action']['type'] != 'concede']
            cases.append((view, actions))
    return cases


def _tokens(cases, device='cpu', capacity=None):
    import torch
    raw = encoder.extract_raw_tokens([view for view, _ in cases],
                                     [actions for _, actions in cases],
                                     capacity=capacity)
    values = {field.name: torch.as_tensor(getattr(raw, field.name), device=device).clone()
              for field in fields(raw) if field.name != 'request_ids'}
    return encoder.RawCandidateTokens(**values)


def _observations(cases, device='cpu'):
    import torch
    rows = [seven_lineup.encode(view, actions)[0] for view, actions in cases]
    return torch.as_tensor(np.stack(rows), device=device)


def _compact(tokens, scratch_rows=3, device='cpu'):
    table = encoder.get_card_template_table(tokens.int_tokens.device)
    return CompactDeviceCandidates(
        tokens, scratch_rows=scratch_rows, device=device,
        allow_cpu_reference=device == 'cpu', template_table=table,
    )


def _token_bytes(tokens):
    return sum(getattr(tokens, name).numel() * getattr(tokens, name).element_size()
               for name in _TOKEN_TENSORS)


class CompactDeviceCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def test_seven_initial_roots_chunked_encoding_matches_dense_and_formal(self):
        import torch
        capacity = 40
        tokens = _tokens(self.cases, capacity=capacity)
        compact = _compact(tokens, scratch_rows=3)
        self.assertEqual(compact.ndim, 3)
        self.assertEqual(compact.shape, (len(self.cases), capacity, encoder.CAND_DIM))
        self.assertGreater(capacity, 16)
        dense, errors = encoder.encode_tokens_device(
            tokens, device='cpu', allow_cpu_reference=True,
            template_table=compact.template_table)
        self.assertEqual(errors.tolist(), [0] * len(self.cases))
        encoded = []
        active = torch.ones(len(self.cases), dtype=torch.bool)
        for start, stop in compact._slices():
            packed, local_errors = compact.encode_slice_device(start, stop, active[start:stop])
            self.assertEqual(local_errors.tolist(), [0] * (stop - start))
            self.assertEqual(packed.request_ids, ())
            self.assertEqual(tuple(packed.candidates.shape),
                             (stop - start, capacity, encoder.CAND_DIM))
            encoded.append(packed.candidates)
        chunked = torch.cat(encoded, 0)
        self.assertTrue(torch.equal(chunked, dense.candidates))
        for i, (view, actions) in enumerate(self.cases):
            _, expected = seven_lineup.encode(view, actions)
            np.testing.assert_array_equal(chunked[i, :len(actions)].numpy(), expected)
            self.assertTrue(bool((chunked[i, len(actions):] == 0).all()))
        tokens.item_flags[0, 0, 0] = 1
        tokens.float_tokens[0, 0, 6] = tokens.turn_counts[0] + 26
        tokens.extra_ap[0, 0] = 26
        packed, errors = compact.encode_slice_device(0, 1, torch.ones(1, dtype=torch.bool))
        self.assertEqual(errors.tolist(), [0])
        expected_ratio = np.float32(26 / 10)
        self.assertEqual(packed.candidates[0, 0, encoder.CONTEXT_START + 1].numpy().view(np.uint32),
                         expected_ratio.view(np.uint32))
        self.assertEqual(packed.candidates[0, 0, -1].numpy().view(np.uint32),
                         expected_ratio.view(np.uint32))

    def test_hash_device_matches_dense_prefix_sha256(self):
        import torch
        tokens = _tokens(self.cases, capacity=40)
        compact = _compact(tokens, scratch_rows=2)
        obs = _observations(self.cases)
        active = torch.ones(len(self.cases), dtype=torch.bool)
        keys = DeviceInformationKey(device='cpu', allow_cpu_reference=True)
        dense, errors = encoder.encode_tokens_device(
            tokens, device='cpu', allow_cpu_reference=True,
            template_table=compact.template_table)
        expected = keys.hash(obs, dense.candidates, tokens.counts, active)
        actual = compact.hash_device(keys, obs, tokens.counts, active)
        self.assertEqual(errors.tolist(), [0] * len(self.cases))
        self.assertEqual(actual.errors.tolist(), [0] * len(self.cases))
        torch.testing.assert_close(actual.keys, expected.keys, atol=0, rtol=0)
        x, c, n = obs.numpy(), dense.candidates.numpy(), tokens.counts.numpy()
        for i in range(len(self.cases)):
            digest = hashlib.sha256(x[i].tobytes() + c[i, :int(n[i])].tobytes()).digest()
            np.testing.assert_array_equal(
                actual.keys[i].numpy(),
                np.frombuffer(digest, dtype='>u4').astype(np.int64))

    def test_clone_and_masked_copy_isolation(self):
        import torch
        tokens = _tokens(self.cases[:4], capacity=32)
        compact = _compact(tokens, scratch_rows=2)
        cloned = compact.clone()
        self.assertEqual(cloned.request_ids, ())
        before = cloned.tokens.int_tokens.clone()
        compact.tokens.int_tokens[0, 0, 0] = encoder.ACT_ATTACK
        self.assertTrue(torch.equal(cloned.tokens.int_tokens, before))
        dest = cloned.clone()
        src = compact
        mask = torch.tensor([True, False, True, False])
        errors = dest.copy_masked_device(src, mask)
        self.assertEqual(errors.tolist(), [0, 0, 0, 0])
        self.assertTrue(torch.equal(dest.tokens.int_tokens[0], src.tokens.int_tokens[0]))
        self.assertTrue(torch.equal(dest.tokens.int_tokens[1], before[1]))
        self.assertFalse(torch.equal(dest.tokens.int_tokens[0], before[0]))
        src.tokens.float_tokens[0, 0, 0] = 9.
        self.assertNotEqual(float(dest.tokens.float_tokens[0, 0, 0].detach()), 9.)
        with self.assertRaises(ValueError):
            dest.copy_masked_device(_compact(_tokens(self.cases[:3], capacity=32), 2), mask[:3])

    def test_full_legal_width_survives_low_scratch(self):
        import torch
        cases = [item for item in self.cases if len(item[1]) > 16]
        self.assertTrue(cases)
        tokens = _tokens(cases, capacity=32)
        compact = _compact(tokens, scratch_rows=1)
        self.assertEqual(compact.max_candidates, 32)
        self.assertGreater(compact.max_candidates, 16)
        shapes = []
        original = encoder.encode_tokens_device

        def wrapped(*args, **kwargs):
            packed, errors = original(*args, **kwargs)
            shapes.append(tuple(packed.candidates.shape))
            return packed, errors

        active = torch.ones(len(cases), dtype=torch.bool)
        with patch.object(encoder, 'encode_tokens_device', wrapped):
            packed, errors = compact.encode_slice_device(0, 1, active[:1])
            compact.hash_device(
                DeviceInformationKey(device='cpu', allow_cpu_reference=True),
                _observations(cases), tokens.counts, active)
        self.assertTrue(all(shape[0] == 1 and shape[1] == 32 and shape[2] == encoder.CAND_DIM
                            for shape in shapes))
        self.assertEqual(tuple(packed.candidates.shape), (1, 32, encoder.CAND_DIM))
        self.assertEqual(errors.tolist(), [0])
        self.assertEqual(int(packed.counts[0]), len(cases[0][1]))
        with self.assertRaises(ValueError):
            compact.encode_slice_device(0, 2, active[:2])

    def test_invalid_and_inactive_rows(self):
        import torch
        tokens = _tokens(self.cases[:5], capacity=32)
        tokens.int_tokens[1, 0, 2] = encoder.NUM_CARDS + 20
        tokens.float_tokens[2, 0, 3] = float('nan')
        tokens.counts[3] = tokens.counts[3] + 1
        compact = _compact(tokens, scratch_rows=2)
        active = torch.tensor([True, True, True, True, False])
        packed, errors = compact.encode_slice_device(0, 2, active[:2])
        self.assertEqual(errors.tolist(), [0, 1])
        self.assertTrue(bool((packed.candidates[1] == 0).all()))
        obs = _observations(self.cases[:5])
        keys = DeviceInformationKey(device='cpu', allow_cpu_reference=True)
        hashed = compact.hash_device(keys, obs, tokens.counts.clone(), active)
        self.assertEqual(hashed.errors[0].item(), 0)
        self.assertNotEqual(hashed.errors[1].item(), 0)
        self.assertNotEqual(hashed.errors[2].item(), 0)
        self.assertNotEqual(hashed.errors[3].item(), 0)
        self.assertEqual(hashed.errors[4].item(), 0)
        self.assertTrue(bool((hashed.keys[1:] == 0).all()))
        wrong_counts = tokens.counts.clone()
        wrong_counts[0] = tokens.counts[0] + 1
        mismatched = compact.hash_device(keys, obs, wrong_counts, active)
        self.assertEqual(mismatched.errors[0].item(), 1)
        self.assertTrue(bool((mismatched.keys[0] == 0).all()))

    def test_hot_path_has_no_host_tensor_reads_or_uuid(self):
        import torch
        tokens = _tokens(self.cases[:3], capacity=32)
        compact = _compact(tokens, scratch_rows=2)
        obs = _observations(self.cases[:3])
        active = torch.ones(3, dtype=torch.bool)
        mask = torch.tensor([True, False, True])
        other = compact.clone()

        class FakeKeys:
            device = compact.device

            def hash(self, observations, candidates, counts, on):
                b = observations.shape[0]
                return DeviceKeyResult(torch.zeros((b, 8), dtype=torch.int64),
                                       torch.zeros(b, dtype=torch.int32))

        class FakeInference:
            device = compact.device

            def predict_device(self, observations, candidates, valid, model_ids, value_actor):
                b, m = candidates.shape[:2]
                return DeviceInferenceResult(
                    torch.zeros((b, m), dtype=torch.float32),
                    torch.zeros((b, 3), dtype=torch.float32),
                    torch.zeros(b, dtype=torch.bool),
                )

        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')), \
             patch.object(encoder, 'uuid4', side_effect=AssertionError('Host request identity')):
            packed, errors = compact.encode_slice_device(0, 2, active[:2])
            compact.copy_masked_device(other, mask)
            compact.hash_device(FakeKeys(), obs, tokens.counts, active)
            compact.predict_device(
                FakeInference(), obs, tokens.mask, torch.zeros(3, dtype=torch.int64),
                torch.zeros((3, 1), dtype=torch.float32), active)
        self.assertEqual(errors.tolist(), [0, 0])
        self.assertEqual(packed.request_ids, ())

    def test_constructor_clone_do_not_materialize_dense_and_cpu_is_explicit(self):
        tokens = _tokens(self.cases[:2], capacity=32)
        with patch.object(encoder, 'encode_tokens_device',
                          side_effect=AssertionError('dense encode')):
            compact = _compact(tokens, scratch_rows=2)
            cloned = compact.clone()
        dense_bytes = 2 * 32 * encoder.CAND_DIM * 4
        self.assertEqual(compact.allocated_bytes, _token_bytes(tokens))
        self.assertLess(compact.allocated_bytes, dense_bytes)
        self.assertGreater(compact.template_allocated_bytes, 0)
        self.assertEqual(cloned.shape, compact.shape)
        self.assertFalse(CompactDeviceCandidates.supports_full_duel_rules)
        with self.assertRaises(ValueError):
            CompactDeviceCandidates(tokens, scratch_rows=2, device='cpu')
        raw = encoder.extract_raw_tokens([view for view, _ in self.cases[:1]],
                                         [actions for _, actions in self.cases[:1]])
        with self.assertRaises(ValueError):
            CompactDeviceCandidates(raw, scratch_rows=2, device='cpu',
                                    allow_cpu_reference=True)

    def test_predict_device_matches_chunked_inference_and_rejects_invalid_policy(self):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        cases = self.cases[:4]
        tokens = _tokens(cases, capacity=32)
        tokens.int_tokens[1, 0] = tokens.int_tokens[1, 2]
        tokens.float_tokens[1, 0] = tokens.float_tokens[1, 2]
        tokens.item_flags[1, 0] = tokens.item_flags[1, 2]
        tokens.mulligan_cards[1, 0] = tokens.mulligan_cards[1, 2]
        tokens.order_cards[1, 0] = tokens.order_cards[1, 2]
        tokens.select_cards[1, 0] = tokens.select_cards[1, 2]
        tokens.extra_ap[1, 0] = tokens.extra_ap[1, 2]
        compact = _compact(tokens, scratch_rows=2)
        obs = _observations(cases)
        models = {'a': SyntheticSevenModel('a', 7), 'b': SyntheticSevenModel('b', 19)}
        inference = DeviceSevenInference(models, device='cpu', allow_cpu_reference=True)
        ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        actor = torch.tensor([[1.], [0.], [0.], [1.]], dtype=torch.float32)
        active = torch.tensor([True, True, True, False])
        valid = tokens.mask.clone()
        actual = compact.predict_device(inference, obs, valid, ids, actor, active)
        dense, errors = encoder.encode_tokens_device(
            tokens, device='cpu', allow_cpu_reference=True,
            template_table=compact.template_table)
        expected = inference.predict_device(obs, dense.candidates, valid, ids, actor)
        self.assertEqual(errors[:3].tolist(), [0, 0, 0])
        torch.testing.assert_close(actual.logits[:3], expected.logits[:3], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual.wdl[:3], expected.wdl[:3], atol=2e-5, rtol=2e-5)
        self.assertEqual(float(actual.logits[1, 0]), float(actual.logits[1, 2]))
        self.assertFalse(bool(actual.invalid_rows[:3].any()))
        self.assertFalse(bool(actual.invalid_rows[3]))
        self.assertTrue(bool((actual.logits[3] == 0).all()))
        broken_valid = valid.clone()
        broken_valid[0, 0] = False
        rejected = compact.predict_device(inference, obs, broken_valid, ids, actor, active)
        self.assertTrue(bool(rejected.invalid_rows[0]))
        self.assertTrue(bool((rejected.logits[0] == 0).all()))
        shapes = []
        original = encoder.encode_tokens_device

        def wrapped(*args, **kwargs):
            packed, local = original(*args, **kwargs)
            shapes.append(tuple(packed.candidates.shape))
            return packed, local

        with patch.object(encoder, 'encode_tokens_device', wrapped):
            compact.predict_device(inference, obs, valid, ids, actor, active)
        self.assertTrue(all(shape[0] <= 2 and shape[1] == 32 for shape in shapes))

    def test_actual_cuda_1600_by_800_graph_hash_stays_compact(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; compact token wrapper only')
        view, actions = next(item for item in self.cases if len(item[1]) > 16)
        raw = _tokens([(view, actions)], device='cuda', capacity=800)
        values = {name: getattr(raw, name).repeat(1600, *([1] * (getattr(raw, name).ndim - 1)))
                  for name in _TOKEN_TENSORS}
        tokens = encoder.RawCandidateTokens(**values, request_ids=())
        table = encoder.get_card_template_table(tokens.int_tokens.device)
        compact = CompactDeviceCandidates(
            tokens, scratch_rows=32, device=tokens.int_tokens.device, template_table=table)
        torch.cuda.reset_peak_memory_stats()
        dense_bytes = 1600 * 800 * encoder.CAND_DIM * 4
        self.assertEqual(compact.shape, (1600, 800, encoder.CAND_DIM))
        self.assertLess(compact.allocated_bytes, dense_bytes // 8)
        self.assertEqual(compact.allocated_bytes, _token_bytes(tokens))
        obs = torch.as_tensor(seven_lineup.encode(view, actions)[0], device='cuda').repeat(1600, 1)
        active = torch.ones(1600, dtype=torch.bool, device='cuda')
        keys = DeviceInformationKey(device='cuda')
        self.addCleanup(keys.close)
        shapes = []
        original = encoder.encode_tokens_device

        def wrapped(*args, **kwargs):
            packed, errors = original(*args, **kwargs)
            shapes.append(tuple(packed.candidates.shape))
            return packed, errors

        with patch.object(encoder, 'encode_tokens_device', wrapped):
            actual = compact.hash_device(keys, obs, tokens.counts, active)
        self.assertTrue(shapes)
        self.assertTrue(all(shape[0] <= 32 and shape[1] == 800 and shape[2] == encoder.CAND_DIM
                            for shape in shapes))
        packed, errors = compact.encode_slice_device(0, 32, active[:32])
        self.assertFalse(bool(errors.any()))
        expected = keys.hash(obs[:32].contiguous(), packed.candidates.contiguous(),
                             tokens.counts[:32].contiguous(), active[:32].contiguous())
        torch.testing.assert_close(actual.keys[:32], expected.keys, atol=0, rtol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compact.hash_device(keys, obs, tokens.counts, active)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.keys[:32], expected.keys, atol=0, rtol=0)
        tokens.int_tokens[1, 0, 2] = encoder.NUM_CARDS + 20
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(captured.errors[:2].cpu().tolist(), [0, 1])
        self.assertTrue(bool((captured.keys[1] == 0).all()))
        peak = torch.cuda.max_memory_allocated()
        print('compact_candidate_primitive_bytes', compact.allocated_bytes,
              'torch_peak_allocated_bytes', peak, 'formal_games', 0, flush=True)
        self.assertLess(peak, 14 * 1024**3)


if __name__ == '__main__':
    unittest.main()

