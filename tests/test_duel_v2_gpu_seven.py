"""Host-synthetic seven GPU encoder/schema tests. Not a 1600-game CUDA claim."""
from copy import deepcopy
import importlib.util
import unittest

import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.rl import seven_grounded, seven_lineup, seven_runtime
from app.modules.card_game.rl.batched_duel.gpu_rule_projection import (
    GraphProjectionKernel,
    intern_layout,
)
from app.modules.card_game.rl.batched_duel.gpu_schema_contract import (
    GPU_RUNTIME_CONTRACTS,
    GpuSchemaContractError,
    next_training_keys,
)

has_torch = importlib.util.find_spec('torch') is not None


def _decks():
    return {item['id']: validate_deck(deepcopy(item)) for item in STARTER_DECKS}


def _legal(view):
    return [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']


class GpuSevenContractTest(unittest.TestCase):
    def test_contract_keys_and_widths_match_next_training(self):
        keys = next_training_keys()
        seven = GPU_RUNTIME_CONTRACTS['seven']
        recovery = GPU_RUNTIME_CONTRACTS['recovery']
        self.assertEqual(keys, seven['keys'])
        self.assertEqual(keys, tuple(seven_lineup.KEYS))
        self.assertEqual(seven['obs_dim'], 3479)
        self.assertEqual(seven['cand_dim'], 1576)
        self.assertEqual(seven['prefix_obs_dim'], 2912)
        self.assertEqual(recovery['obs_dim'], 2420)
        self.assertEqual(recovery['cand_dim'], 254)
        self.assertEqual(recovery['keys'], ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk'))
        self.assertIn('collapse', seven['keys'])
        self.assertIn('lingke', seven['keys'])
        self.assertNotIn('heiyu', recovery['seats'])
        self.assertIn('heiyu', seven['seats'])
        self.assertEqual(seven['action_types'][7], 'exit_ultimate')
        self.assertEqual(seven_lineup.ACTION_TYPES.index('play_card'), 3)
        self.assertEqual(seven_lineup.ACTION_TYPES.index('mulligan'), 5)

    def test_intern_layout_keeps_recovery_identity(self):
        recovery_names, recovery_index = intern_layout()
        same_names, same_index = intern_layout('recovery')
        self.assertEqual(recovery_names, same_names)
        self.assertEqual(recovery_index, same_index)
        self.assertNotIn('heiyu', recovery_index)
        self.assertNotIn('dafutier', recovery_index)
        self.assertNotIn('lingke', recovery_index)
        seven_names, seven_index = intern_layout('seven')
        self.assertGreater(len(seven_names), len(recovery_names))
        start = seven_index[seven_lineup.SEATS[0]]
        for offset, seat in enumerate(seven_lineup.SEATS):
            self.assertEqual(seven_index[seat], start + offset)
        action_start = seven_index[seven_lineup.ACTION_TYPES[0]]
        for offset, action in enumerate(seven_lineup.ACTION_TYPES):
            self.assertEqual(seven_index[action], action_start + offset)
        with self.assertRaises(GpuSchemaContractError):
            intern_layout('nope')

    def test_seven_projection_source_remaps_action_codes(self):
        from app.modules.card_game.rl.batched_duel.gpu_seven_projection import emit_seven_projection_source
        source = emit_seven_projection_source()
        self.assertIn('CODE_PLAY', source)
        self.assertIn('if(code==CODE_PLAY){', source)
        self.assertNotIn('if(code==2){', source)
        self.assertIn('F_bottom_card_ids', source)
        self.assertIn('F_exit_ultimate', source)
        self.assertIn('F_hand_deck_placement', source)


@unittest.skipUnless(has_torch, 'PyTorch required for GPU encoder tests')
class GpuSevenEncoderTest(unittest.TestCase):
    def _actor_ratio_views(self):
        base=observe(new_game(seed=701,decks={'a':self.decks['lingke'],'b':self.decks['starter']},skip_mulligan=True),'a',include_previews=False)
        result=[]
        for actor in seven_lineup.SEATS:
            view=deepcopy(base)
            hero=next(h for h in view['sides']['a']['characters'] if h['id']=='lingke')
            hero['lingke_chain']={'character_id':actor,'count':1}
            meta=view['policy_state']['sides']['a']['characters']['lingke']
            meta.setdefault('flags',{})['mimic']={'id':actor}
            result.append(view)
        return result

    def test_actor_ratios_match_canonical_for_every_seat(self):
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import encode_observation_batch
        views=self._actor_ratio_views();expected=np.stack([seven_lineup.encode(v,[])[0] for v in views])
        actual=encode_observation_batch(views,device='cpu').observations.numpy()
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_actual_cuda_actor_ratios_match_canonical_for_every_seat(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import encode_observation_batch
        views=self._actor_ratio_views();expected=np.stack([seven_lineup.encode(v,[])[0] for v in views])
        actual=encode_observation_batch(views,device='cuda').observations.cpu().numpy()
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_device_observation_errors_stay_per_row_and_preserve_inputs(self):
        import torch
        from dataclasses import fields
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import extract_raw_observation_tokens, encode_observation_tokens_device, RawObservationTokens
        deck = self.decks['starter']
        view = observe(new_game(seed=7, decks={'a': deck, 'b': deepcopy(deck)}, skip_mulligan=True), 'a', include_previews=False)
        raw = extract_raw_observation_tokens([view, view])
        values = {field.name: torch.as_tensor(getattr(raw, field.name)).clone() for field in fields(raw) if field.name != 'request_ids'}
        values['char_seats'][1, 0, 0] = 999
        before = values['char_seats'].clone()
        device_tokens = RawObservationTokens(**values)
        encoded, errors = encode_observation_tokens_device(device_tokens, device='cpu', allow_cpu_reference=True)
        self.assertEqual(errors.tolist(), [0, 1])
        self.assertTrue(torch.equal(values['char_seats'], before))
        self.assertTrue(torch.isfinite(encoded.observations).all())
        self.assertEqual(encoded.request_ids, ())

    def test_actual_cuda_observation_encoding_capture_keeps_error_masks(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from dataclasses import fields
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import extract_raw_observation_tokens, encode_observation_tokens_device, RawObservationTokens
        deck = self.decks['starter']
        view = observe(new_game(seed=7, decks={'a': deck, 'b': deepcopy(deck)}, skip_mulligan=True), 'a', include_previews=False)
        raw = extract_raw_observation_tokens([view, view])
        values = {field.name: torch.as_tensor(getattr(raw, field.name), device='cuda').clone() for field in fields(raw) if field.name != 'request_ids'}
        tokens = RawObservationTokens(**values)
        encode_observation_tokens_device(tokens, device='cuda:0')
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            encoded, errors = encode_observation_tokens_device(tokens, device='cuda:0')
        graph.replay();torch.cuda.synchronize()
        self.assertEqual(errors.tolist(), [0, 0])
        values['global_tokens'][1, 0] = float('nan')
        graph.replay();torch.cuda.synchronize()
        self.assertEqual(errors.tolist(), [0, 1])
        self.assertTrue(torch.isfinite(encoded.observations).all())

    @classmethod
    def setUpClass(cls):
        cls.decks = _decks()

    def test_scalar_normalization_rounds_once_like_formal_encoder(self):
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import encode_observation_batch
        views = [observe(new_game(seed=19501+i, skip_mulligan=True,
                                  decks={'a': deck, 'b': deck}), 'a', include_previews=False)
                 for i, deck in enumerate(self.decks.values())]
        expected = np.stack([seven_lineup.encode(view, [])[0] for view in views])
        actual = encode_observation_batch(views, device='cpu').observations.numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0, 35], np.float32(26 / 10))

    def test_actual_cuda_exact_scalar_observation_normalization(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.gpu_seven_observation import encode_observation_batch
        views = [observe(new_game(seed=19501+i, skip_mulligan=True,
                                  decks={'a': deck, 'b': deck}), 'a', include_previews=False)
                 for i, deck in enumerate(self.decks.values())]
        expected = np.stack([seven_lineup.encode(view, [])[0] for view in views])
        actual = encode_observation_batch(views, device='cuda').observations.cpu().numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_recovery_encode_still_rejects_seven_only_roster(self):
        from app.modules.card_game.rl.batched_duel.gpu_observation import encode_single_observation
        view = observe(
            new_game(
                seed=9,
                skip_mulligan=True,
                decks={'a': self.decks['collapse'], 'b': self.decks['lingke']},
            ),
            'a',
            include_previews=False,
        )
        with self.assertRaisesRegex(ValueError, 'public characters'):
            encode_single_observation(view, device='cpu')
        with self.assertRaisesRegex(ValueError, 'public characters'):
            encode_single_observation(view, device='cpu', runtime='recovery')

    def test_seven_gpu_encode_matches_lineup_and_runtime(self):
        from app.modules.card_game.rl.batched_duel.gpu_candidates import encode_single
        from app.modules.card_game.rl.batched_duel.gpu_observation import encode_single_observation
        for seed, key_a, key_b in (
            (3, 'starter', 'starter'),
            (11, 'collapse', 'lingke'),
            (19, 'lingke', 'collapse'),
        ):
            state = new_game(
                seed=seed,
                skip_mulligan=True,
                decks={'a': self.decks[key_a], 'b': self.decks[key_b]},
            )
            side = 'a'
            view = observe(state, side, include_previews=False)
            actions = _legal(view)
            oracle_x, oracle_c = seven_lineup.encode(view, actions)
            runtime_actions, encoded = seven_runtime.decision(state, side)
            self.assertEqual(runtime_actions, actions)
            self.assertEqual(encoded[0].shape, (3479,))
            self.assertEqual(encoded[1].shape[1], 1576)
            gpu_x = encode_single_observation(view, device='cpu', runtime='seven')
            gpu_c = encode_single(view, actions, device='cpu', runtime='seven')
            self.assertEqual(gpu_x.observations.shape[-1], 3479)
            self.assertEqual(gpu_c.candidates.shape[-1], 1576)
            np.testing.assert_allclose(gpu_x.numpy()[0], oracle_x, atol=1e-6)
            np.testing.assert_allclose(gpu_c.candidates.detach().cpu().numpy()[0, :len(actions)], oracle_c, atol=1e-6)
            np.testing.assert_allclose(gpu_x.numpy()[0], encoded[0], atol=1e-6)
            np.testing.assert_allclose(gpu_c.candidates.detach().cpu().numpy()[0, :len(actions)], encoded[1], atol=1e-6)
            grounded = seven_grounded.transform(oracle_x, oracle_c)
            gpu_grounded = seven_grounded.transform(gpu_x.numpy()[0], gpu_c.candidates.detach().cpu().numpy()[0, :len(actions)])
            np.testing.assert_allclose(grounded, gpu_grounded, atol=1e-6)

    def test_seven_grounded_alias_indices_bind_equal_rows(self):
        from app.modules.card_game.rl.batched_duel.gpu_candidates import encode_single
        from app.modules.card_game.rl.batched_duel.gpu_observation import encode_single_observation
        from app.modules.card_game.rl.seven_policy import alias_indices
        state = new_game(
            seed=27,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['collapse']},
        )
        view = observe(state, 'a', include_previews=False)
        actions = _legal(view)
        oracle_x, oracle_c = seven_lineup.encode(view, actions)
        gpu_c = encode_single(view, actions, device='cpu', runtime='seven').candidates.detach().cpu().numpy()[0, :len(actions)]
        np.testing.assert_array_equal(alias_indices(oracle_c), alias_indices(gpu_c))
        encode_single_observation(view, device='cpu', runtime='seven')

    def test_exit_ultimate_and_ordered_bottom_cards(self):
        from app.modules.card_game.rl.batched_duel.gpu_candidates import encode_single
        from app.modules.card_game.rl.batched_duel.gpu_observation import encode_single_observation
        state = new_game(
            seed=41,
            skip_mulligan=True,
            decks={'a': self.decks['lingke'], 'b': self.decks['collapse']},
        )
        view = observe(state, 'a', include_previews=False)
        cards = [item['card_id'] for item in view['sides']['a']['hand'][:4]]
        while len(cards) < 4:
            cards.append('N01')
        view = deepcopy(view)
        view['pending_choice'] = {
            'kind': 'hand_deck_placement',
            'choices': [
                {'id': 'top0', 'card': {'card_id': cards[0]}},
                {'id': 'top1', 'card': {'card_id': cards[1]}},
                {'id': 'bot0', 'card': {'card_id': cards[2]}},
                {'id': 'bot1', 'card': {'card_id': cards[3]}},
            ],
        }
        actor = view['sides']['a']['characters'][0]['id']
        actions = [
            {'type': 'exit_ultimate', 'character_id': actor},
            {
                'type': 'choose_deck_order',
                'card_ids': ['top0', 'top1'],
                'bottom_card_ids': ['bot0', 'bot1'],
            },
            {
                'type': 'choose_deck_order',
                'card_ids': ['top0'],
                'bottom_card_id': 'bot1',
            },
        ]
        oracle_x, oracle_c = seven_lineup.encode(view, actions)
        self.assertEqual(oracle_c[0, 0], 7)
        self.assertNotEqual(tuple(oracle_c[1, 306:306 + 141 * 4].tolist()), tuple(oracle_c[1, 306 + 141 * 4:306 + 141 * 8].tolist()))
        gpu_x = encode_single_observation(view, device='cpu', runtime='seven')
        gpu_c = encode_single(view, actions, device='cpu', runtime='seven')
        np.testing.assert_allclose(gpu_x.numpy()[0], oracle_x, atol=1e-6)
        np.testing.assert_allclose(gpu_c.candidates.detach().cpu().numpy()[0], oracle_c, atol=1e-6)
        self.assertEqual(gpu_x.numpy()[0][seven_lineup.all_features().index('seven:choice:hand_deck_placement')], 1.0)

    def test_overflow_and_unknown_actions_raise(self):
        from app.modules.card_game.rl.batched_duel.gpu_candidates import encode_single
        from app.modules.card_game.rl.batched_duel.gpu_observation import encode_single_observation
        view = observe(
            new_game(seed=5, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']}),
            'a',
            include_previews=False,
        )
        with self.assertRaisesRegex(ValueError, 'Unencoded seven-team action|Unknown action type'):
            encode_single(view, [{'type': 'teleport'}], device='cpu', runtime='seven')
        with self.assertRaisesRegex(ValueError, 'Unknown action character|public characters'):
            encode_single(view, [{'type': 'attack', 'character_id': 'alien'}], device='cpu', runtime='seven')
        overflow = deepcopy(view)
        overflow['pending_choice'] = {
            'kind': 'hand_deck_placement',
            'choices': [{'id': 'c%s' % i, 'card': {'card_id': 'N01'}} for i in range(5)],
        }
        with self.assertRaisesRegex(ValueError, 'Deck order exceeds'):
            encode_single(
                overflow,
                [{'type': 'choose_deck_order', 'card_ids': ['c0', 'c1', 'c2', 'c3', 'c4'], 'bottom_card_ids': []}],
                device='cpu',
                runtime='seven',
            )
        bad = deepcopy(view)
        bad['sides']['a']['characters'][0]['id'] = 'not-a-hero'
        with self.assertRaisesRegex(ValueError, 'public characters|Unknown'):
            encode_single_observation(bad, device='cpu', runtime='seven')
        with self.assertRaisesRegex(ValueError, 'Candidate count .* exceeds explicit capacity'):
            encode_single(view, [{'type': 'end_turn'}] * 3, capacity=2, device='cpu', runtime='seven')

    def test_empty_candidates_use_full_seven_width(self):
        from app.modules.card_game.rl.batched_duel.gpu_candidates import encode_single
        view = observe(
            new_game(seed=8, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']}),
            'a',
            include_previews=False,
        )
        batch = encode_single(view, [], device='cpu', runtime='seven')
        self.assertEqual(tuple(batch.candidates.shape), (1, 0, 1576))

    def test_graph_projection_kernel_requires_actual_cuda(self):
        import torch
        if torch.cuda.is_available():
            kernel = GraphProjectionKernel(device='cuda', max_candidates=8, runtime='seven')
            self.addCleanup(kernel.close)
            self.assertEqual(kernel.runtime, 'seven')
            self.assertEqual(kernel.obs_dim, 3479)
            self.assertEqual(kernel.cand_dim, 1576)
            self.assertEqual(kernel.action_types[7], 'exit_ultimate')
            self.assertIn('heiyu', kernel.seats)
            return
        with self.assertRaisesRegex(RuntimeError, 'Actual CUDA required'):
            GraphProjectionKernel(device='cuda', max_candidates=8, runtime='seven')
        with self.assertRaisesRegex(RuntimeError, 'Actual CUDA required'):
            GraphProjectionKernel(device='cuda', max_candidates=8, runtime='recovery')


if __name__ == '__main__':
    unittest.main()
