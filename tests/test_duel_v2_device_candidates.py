"""Device token encoder parity/validation; tensor rows are not full games."""
from copy import deepcopy
from dataclasses import fields
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.batched_duel import gpu_seven_candidates as encoder


class DeviceCandidateTest(unittest.TestCase):
    def cases(self):
        cases = []
        for definition in STARTER_DECKS:
            if definition['id'] not in seven_lineup.KEYS:
                continue
            deck = validate_deck(deepcopy(definition))
            for skip in (False, True):
                view = observe(new_game(seed=7, first_side='a', decks={'a': deck, 'b': deepcopy(deck)},
                                        skip_mulligan=skip), 'a', include_previews=False)
                actions = [r['action'] for r in view['legal_actions'] if r['action']['type'] != 'concede']
                cases.append((view, actions))
        return cases

    def tokens(self, cases, device='cpu'):
        import torch
        raw = encoder.extract_raw_tokens([v for v, _ in cases], [a for _, a in cases])
        values = {field.name: torch.as_tensor(getattr(raw, field.name), device=device).clone()
                  for field in fields(raw) if field.name != 'request_ids'}
        return encoder.RawCandidateTokens(**values)

    def test_all_seven_rosters_and_mulligans_match_formal_numpy_encoding(self):
        cases = self.cases()
        tokens = self.tokens(cases)
        batch, errors = encoder.encode_tokens_device(tokens, device='cpu', allow_cpu_reference=True)
        self.assertEqual(errors.tolist(), [0]*len(cases))
        self.assertEqual(batch.request_ids, ())
        for i, (view, actions) in enumerate(cases):
            _, expected = seven_lineup.encode(view, actions)
            np.testing.assert_array_equal(batch.candidates[i, :len(actions)].numpy(), expected)

    def test_exact_ratio32_uses_formal_double_division_and_preserves_input(self):
        import torch
        cases = self.cases()[-1:]
        tokens = self.tokens(cases)
        tokens.item_flags[0, 0, 0] = 1
        tokens.float_tokens[0, 0, 6] = tokens.turn_counts[0] + 26
        tokens.extra_ap[0, 0] = 26
        before = tokens.float_tokens.clone()
        batch, errors = encoder.encode_tokens_device(tokens, device='cpu', allow_cpu_reference=True)
        self.assertEqual(errors.tolist(), [0])
        expected = np.float32(26 / 10)
        self.assertEqual(batch.candidates[0, 0, encoder.CONTEXT_START+1].numpy().view(np.uint32), expected.view(np.uint32))
        self.assertEqual(batch.candidates[0, 0, -1].numpy().view(np.uint32), expected.view(np.uint32))
        torch.testing.assert_close(tokens.float_tokens, before)

    def test_invalid_rows_zero_outputs_and_inactive_oob_does_not_corrupt_other_rows(self):
        import torch
        cases = [self.cases()[-1]]*5
        tokens = self.tokens(cases)
        tokens.int_tokens[1, 0, 2] = encoder.NUM_CARDS+20
        tokens.float_tokens[2, 0, 3] = float('nan')
        tokens.counts[3] = tokens.mask.shape[1]+1
        tokens.select_cards[4, 0, 0] = encoder.NUM_CARDS+1
        active = torch.tensor([True, True, True, True, False])
        before = tokens.int_tokens.clone()
        batch, errors = encoder.encode_tokens_device(tokens, device='cpu',
                                                      allow_cpu_reference=True, active=active)
        self.assertEqual(errors.tolist(), [0, 1, 1, 1, 0])
        self.assertTrue(bool(torch.isfinite(batch.candidates).all()))
        self.assertTrue(bool((batch.candidates[1:] == 0).all()))
        self.assertEqual(batch.counts[1:].tolist(), [0]*4)
        self.assertFalse(bool(batch.mask[1:].any()))
        torch.testing.assert_close(tokens.int_tokens, before)
        _, expected = seven_lineup.encode(*cases[0])
        np.testing.assert_array_equal(batch.candidates[0, :len(cases[0][1])].numpy(), expected)

    def test_device_api_rejects_host_tokens_and_cpu_requires_explicit_oracle(self):
        cases = self.cases()[-1:]
        tokens = self.tokens(cases)
        with self.assertRaises(ValueError):
            encoder.encode_tokens_device(tokens, device='cpu')
        raw = encoder.extract_raw_tokens([v for v, _ in cases], [a for _, a in cases])
        with self.assertRaises(ValueError):
            encoder.encode_tokens_device(raw, device='cpu', allow_cpu_reference=True)

    def test_encoder_hot_path_has_no_host_tensor_reads_or_request_uuid(self):
        import torch
        tokens = self.tokens(self.cases()[-1:])
        table = encoder.get_card_template_table(torch.device('cpu'))
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')), \
             patch.object(encoder, 'uuid4', side_effect=AssertionError('Host request identity')):
            _, errors = encoder.encode_tokens_device(tokens, device='cpu',
                allow_cpu_reference=True, template_table=table)
        self.assertEqual(errors.tolist(), [0])

    def test_actual_cuda_1600_token_rows_graph_capture_and_error_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; token encoder only')
        cases = [self.cases()[-1]]*1600
        tokens = self.tokens(cases, 'cuda')
        device = tokens.int_tokens.device
        table = encoder.get_card_template_table(device)
        encoded, errors = encoder.encode_tokens_device(tokens, device=device, template_table=table)
        _, expected = seven_lineup.encode(*cases[0])
        np.testing.assert_array_equal(encoded.candidates[0, :len(cases[0][1])].cpu().numpy(), expected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_errors = encoder.encode_tokens_device(tokens, device=device,
                                                                       template_table=table)
        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(bool(captured_errors.any()))
        tokens.int_tokens[1, 0, 2] = encoder.NUM_CARDS+20
        tokens.float_tokens[2, 0, 0] = float('inf')
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(captured_errors[:3].cpu().tolist(), [0, 1, 1])
        self.assertTrue(bool((captured.candidates[1:3] == 0).all()))
        self.assertTrue(bool(torch.isfinite(captured.candidates).all()))


if __name__ == '__main__':
    unittest.main()
