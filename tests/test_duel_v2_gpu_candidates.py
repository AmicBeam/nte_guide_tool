"""Unit tests for GPU batch candidate action encoding in Duel V2.

Verifies 100% numerical identity with official cross_lineup.encode(view, actions)[1]
as an independent oracle.
Covers:
- Five preset decks (starter, weave-rush, quick-rush, zhenhong, murk) with legal actions
- Mulligan phase and mulligan card counting / scatter
- RF01 physical card identity and hand_face one-hot
- Card copy, expiry, derived override difference, and antique investment
- Choose actions with pending choices
- Summon aliases (summon_*) rewriting
- Jingu mark, set attack, and dynamic instant
- Empty candidates, mixed empty batches, and N=0
- Explicit capacity limits (strictly no silent truncation)
- Rejection of unknown actions, characters, cards, options, and targets
- Explicit device semantics: no CPU fallback when CUDA requested/defaulted, explicit CPU allowed
"""

import unittest
from copy import deepcopy
import importlib.util
import numpy as np

has_torch = importlib.util.find_spec('torch') is not None

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl import cross_lineup
from app.modules.card_game.rl.batched_duel.gpu_candidates import (
    CAND_DIM,
    CARD_IDS,
    CARD_INDEX,
    OPTION_IDS,
    PackedCandidateBatch,
    RawCandidateTokens,
    SCHEMA_SHA,
    SEAT_INDEX,
    encode_candidate_batch,
    encode_single,
    encode_tokens,
    extract_raw_tokens,
)


def _get_preset_decks():
    return {d['id']: validate_deck(deepcopy(d)) for d in STARTER_DECKS}


def _oracle_candidates(view, actions):
    """Independent oracle: official cross_lineup.encode candidate matrix."""
    _, oracle_c = cross_lineup.encode(view, actions)
    return oracle_c


@unittest.skipUnless(has_torch, "PyTorch required for gpu_candidates tests")
class TestGpuCandidateEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.decks = _get_preset_decks()

    def test_five_presets_legal_actions_match_oracle(self):
        """Test legal actions from all 5 presets match official oracle exactly."""
        preset_keys = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')

        for key in preset_keys:
            # 1. Playing phase (skip_mulligan=True)
            s_play = new_game(
                seed=42,
                skip_mulligan=True,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view_play = observe(s_play, 'a', include_previews=False)
            actions_play = [
                e['action']
                for e in view_play['legal_actions']
                if e['action']['type'] != 'concede'
            ]
            self.assertGreater(len(actions_play), 0, f"Preset {key} has no legal actions")

            batch_play = encode_candidate_batch([view_play], [actions_play], device='cpu')
            oracle_play = _oracle_candidates(view_play, actions_play)

            self.assertEqual(batch_play.batch_size, 1)
            self.assertEqual(batch_play.cand_dim, CAND_DIM)
            self.assertEqual(batch_play.counts.item(), len(actions_play))
            self.assertTrue(batch_play.mask[0].all().item())

            actual_c = batch_play.candidates[0].cpu().numpy()
            np.testing.assert_allclose(
                actual_c,
                oracle_play,
                atol=1e-5,
                err_msg=f"Mismatch on playing phase for preset {key}",
            )

            # 2. Mulligan phase (skip_mulligan=False)
            s_mull = new_game(
                seed=99,
                skip_mulligan=False,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view_mull = observe(s_mull, 'a', include_previews=False)
            actions_mull = [
                e['action']
                for e in view_mull['legal_actions']
                if e['action']['type'] != 'concede'
            ]
            self.assertGreater(len(actions_mull), 0)

            batch_mull = encode_candidate_batch([view_mull], [actions_mull], device='cpu')
            oracle_mull = _oracle_candidates(view_mull, actions_mull)

            actual_mull_c = batch_mull.candidates[0].cpu().numpy()
            np.testing.assert_allclose(
                actual_mull_c,
                oracle_mull,
                atol=1e-5,
                err_msg=f"Mismatch on mulligan phase for preset {key}",
            )

    def test_simultaneous_multigame_batch_alignment(self):
        """Test batched encoding across multiple games aligns IDs, counts, and masks."""
        preset_keys = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
        views = []
        actions_list = []
        oracles = []

        for seed_idx, key in enumerate(preset_keys):
            s = new_game(
                seed=100 + seed_idx,
                skip_mulligan=True,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view = observe(s, 'a', include_previews=False)
            acts = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
            views.append(view)
            actions_list.append(acts)
            oracles.append(_oracle_candidates(view, acts))

        batch = encode_candidate_batch(views, actions_list, device='cpu')
        self.assertEqual(batch.batch_size, 5)
        max_cands = max(len(a) for a in actions_list)
        self.assertEqual(batch.max_candidates, max_cands)

        for i in range(5):
            n_acts = len(actions_list[i])
            self.assertEqual(batch.counts[i].item(), n_acts)
            # Valid region matches oracle
            actual_slice = batch.candidates[i, :n_acts].cpu().numpy()
            np.testing.assert_allclose(actual_slice, oracles[i], atol=1e-5)
            # Padded region is strictly zero and masked out
            if n_acts < max_cands:
                self.assertFalse(batch.mask[i, n_acts:].any().item())
                padded_slice = batch.candidates[i, n_acts:].cpu().numpy()
                np.testing.assert_array_equal(padded_slice, 0.0)

    def test_mulligan_card_selection_and_scatter(self):
        """Test mulligan card scatter adds 0.5 per selected card including duplicates."""
        s = new_game(seed=7, skip_mulligan=False, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        hand = view['sides']['a']['hand']
        self.assertGreaterEqual(len(hand), 3)

        inst0 = hand[0]['instance_id']
        inst1 = hand[1]['instance_id']
        inst2 = hand[2]['instance_id']

        mull_actions = [
            {'type': 'mulligan', 'card_ids': []},                     # Keep all
            {'type': 'mulligan', 'card_ids': [inst0]},                # Replace 1
            {'type': 'mulligan', 'card_ids': [inst0, inst1, inst2]},  # Replace 3
        ]

        batch = encode_candidate_batch([view], [mull_actions], device='cpu')
        oracle = _oracle_candidates(view, mull_actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        # Base features for mulligan: [ACT_MULLIGAN (4), -1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        expected_base = np.array([4.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for act_idx in range(3):
            np.testing.assert_array_equal(actual_c[act_idx, :11], expected_base)

        # Selected card columns have 0.5
        c0_idx = CARD_INDEX[hand[0]['card_id']]
        self.assertAlmostEqual(actual_c[1, 11 + c0_idx], 0.5, places=5)
        expected=0.5*sum(card['card_id']==hand[0]['card_id'] for card in hand[:3])
        self.assertAlmostEqual(actual_c[2, 11 + c0_idx], expected, places=5)

    def test_option_ids_and_invalid_option_rejection(self):
        """Test all 5 valid option IDs one-hot encode properly and invalid option is rejected."""
        s = new_game(seed=12, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        hand_card = view['sides']['a']['hand'][0]
        inst_id = hand_card['instance_id']

        option_actions = [
            {'type': 'play_card', 'card_id': inst_id, 'option_id': opt}
            for opt in OPTION_IDS
        ]

        batch = encode_candidate_batch([view], [option_actions], device='cpu')
        oracle = _oracle_candidates(view, option_actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        option_start = 11 + len(CARD_IDS)
        for i, opt in enumerate(OPTION_IDS):
            self.assertAlmostEqual(actual_c[i, option_start + i], 1.0, places=5)
            # Other option bits are 0
            other_bits = [actual_c[i, option_start + j] for j in range(len(OPTION_IDS)) if j != i]
            self.assertTrue(all(b == 0.0 for b in other_bits))

        # Rejection of invalid option
        invalid_action = [{'type': 'play_card', 'card_id': inst_id, 'option_id': 'invalid_boost_ultra'}]
        with self.assertRaisesRegex(ValueError, "Unknown play option"):
            encode_candidate_batch([view], [invalid_action], device='cpu')

    def test_candidate_context_copy_expiry_derived_antique(self):
        """Test candidate context columns: copy, expires_in, derived_override, antique_investment."""
        s = new_game(seed=22, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        turn_count = view['sides']['a']['turn_count']

        # Inject context attributes into hand cards
        hand = view['sides']['a']['hand']
        c0 = hand[0]
        c0['copy'] = True
        c0['expires_turn'] = turn_count + 6
        c0['derived'] = not bool(CARDS[c0['card_id']].get('derived'))  # Explicitly invert this actual template
        c0['antique_investment'] = 3.75

        actions = [{'type': 'play_card', 'card_id': c0['instance_id']}]

        batch = encode_candidate_batch([view], [actions], device='cpu')
        oracle = _oracle_candidates(view, actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        context_start = CAND_DIM - len(CARD_IDS) - 4
        # copy=1.0, expires_in=(6/10)=0.6, derived_override=1.0, antique=3.75
        expected_context = np.array([1.0, 0.6, 1.0, 3.75], dtype=np.float32)
        np.testing.assert_allclose(actual_c[0, context_start:context_start + 4], expected_context, atol=1e-5)

    def test_rf01_physical_identity_one_hot(self):
        """Test hand_face original physical card one-hot column is set for RF01."""
        s = new_game(seed=33, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        hand = view['sides']['a']['hand']

        # Card with different physical face (RF01)
        c0 = hand[0]
        c0['hand_face'] = {'card_id': 'RF01'}
        # Card with identical physical face -> should NOT set physical bit
        c1 = hand[1]
        c1['hand_face'] = {'card_id': c1['card_id']}

        actions = [
            {'type': 'play_card', 'card_id': c0['instance_id']},
            {'type': 'play_card', 'card_id': c1['instance_id']},
        ]

        batch = encode_candidate_batch([view], [actions], device='cpu')
        oracle = _oracle_candidates(view, actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        physical_start = CAND_DIM - len(CARD_IDS)
        rf01_col = physical_start + CARD_INDEX['RF01']

        self.assertAlmostEqual(actual_c[0, rf01_col], 1.0, places=5)
        # c1 has identical face, so no physical bit set
        self.assertAlmostEqual(actual_c[1, rf01_col], 0.0, places=5)
        self.assertEqual(actual_c[1, physical_start:].sum(), 0.0)

    def test_choose_actions_with_pending_choices(self):
        """Test choose action encoding against pending choices."""
        s = new_game(seed=44, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)

        # Synthetic pending_choice in view
        view['pending_choice'] = {
            'kind': 'inspect_top',
            'choices': [
                {
                    'id': 'choice_1',
                    'card': {
                        'card_id': 'N01',
                        'revealed': True,
                        'copy': False,
                        'derived': False,
                    },
                },
                {
                    'id': 'choice_2',
                    'card': {
                        'card_id': 'Z01',
                        'revealed': False,
                        'copy': True,
                        'derived': True,
                        'antique_investment': 1.5,
                        'hand_face': {'card_id': 'RF01'},
                    },
                },
            ],
        }

        choose_actions = [
            {'type': 'choose', 'choice_id': 'choice_1'},
            {'type': 'choose', 'choice_id': 'choice_2'},
        ]

        batch = encode_candidate_batch([view], [choose_actions], device='cpu')
        oracle = _oracle_candidates(view, choose_actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        # Actor for choose action must be -1
        self.assertEqual(actual_c[0, 1], -1.0)
        self.assertEqual(actual_c[1, 1], -1.0)
        # Choice 1 revealed=1, Choice 2 revealed=0
        self.assertEqual(actual_c[0, 10], 1.0)
        self.assertEqual(actual_c[1, 10], 0.0)

    def test_summon_alias_rewriting(self):
        """Test summon_* character and target alias rewriting to 'summon'."""
        s = new_game(seed=55, skip_mulligan=True, decks={'a': self.decks['murk'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)

        actions = [
            {'type': 'attack', 'character_id': 'summon_10', 'target_id': 'b:player'},
            {'type': 'attack', 'character_id': 'anhunqu', 'target_id': 'b:summon_2'},
        ]

        batch = encode_candidate_batch([view], [actions], device='cpu')
        oracle = _oracle_candidates(view, actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        summon_idx = SEAT_INDEX['summon']
        self.assertEqual(actual_c[0, 1], float(summon_idx))
        self.assertEqual(actual_c[1, 4], float(summon_idx))

    def test_jingu_mark_and_set_attack_mode(self):
        """Test marked, mark_delta, set_attack, and dynamic instant features."""
        s = new_game(seed=66, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        hand = view['sides']['a']['hand']

        c0 = hand[0]
        c0['jingu_mark'] = {'delta': -2}
        c0['attack_mode'] = 'set'
        c0['instant'] = True

        actions = [{'type': 'play_card', 'card_id': c0['instance_id']}]

        batch = encode_candidate_batch([view], [actions], device='cpu')
        oracle = _oracle_candidates(view, actions)

        actual_c = batch.candidates[0].cpu().numpy()
        np.testing.assert_allclose(actual_c, oracle, atol=1e-5)

        mark_start = 11 + len(CARD_IDS) + len(OPTION_IDS)
        expected_marks = np.array([1.0, -2.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(actual_c[0, mark_start:mark_start + 4], expected_marks, atol=1e-5)

    def test_empty_candidates_and_zero_batches(self):
        """Test handling of empty candidate actions, zero batch, and masked zeros."""
        s = new_game(seed=77, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)

        # 1. Single view with empty actions
        batch_empty = encode_candidate_batch([view], [[]], device='cpu')
        self.assertEqual(batch_empty.candidates.shape, (1, 0, CAND_DIM))
        self.assertEqual(batch_empty.counts[0].item(), 0)

        # 2. Mixed batch: view 0 has 2 actions, view 1 has 0 actions
        act2 = [{'type': 'end_turn'}, {'type': 'end_turn'}]
        batch_mixed = encode_candidate_batch([view, view], [act2, []], device='cpu')
        self.assertEqual(batch_mixed.batch_size, 2)
        self.assertEqual(batch_mixed.max_candidates, 2)
        self.assertEqual(batch_mixed.counts[0].item(), 2)
        self.assertEqual(batch_mixed.counts[1].item(), 0)
        self.assertTrue(batch_mixed.mask[0].all().item())
        self.assertFalse(batch_mixed.mask[1].any().item())
        np.testing.assert_array_equal(batch_mixed.candidates[1].cpu().numpy(), 0.0)

        # 3. N=0 empty batch
        batch_n0 = encode_candidate_batch([], [], device='cpu')
        self.assertEqual(batch_n0.candidates.shape, (0, 0, CAND_DIM))
        self.assertEqual(batch_n0.counts.shape, (0,))

    def test_explicit_capacity_and_no_silent_truncation(self):
        """Test explicit capacity pads properly and raises when candidate count exceeds capacity."""
        s = new_game(seed=88, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)

        actions = [{'type': 'end_turn'}]

        # Capacity = 4: should pad to M=4
        batch = encode_candidate_batch([view], [actions], capacity=4, device='cpu')
        self.assertEqual(batch.max_candidates, 4)
        self.assertEqual(batch.counts[0].item(), 1)
        self.assertTrue(batch.mask[0, 0].item())
        self.assertFalse(batch.mask[0, 1:].any().item())

        # Exceeding capacity raises ValueError, never silently truncates
        with self.assertRaisesRegex(ValueError, "exceeds explicit capacity"):
            encode_candidate_batch([view], [[{'type': 'end_turn'}] * 5], capacity=4, device='cpu')

    def test_rejection_of_unknown_entities(self):
        """Test rejection of unknown action types, characters, cards, and targets."""
        s = new_game(seed=99, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)

        # Unknown action type
        with self.assertRaisesRegex(ValueError, "Unknown action type"):
            encode_candidate_batch([view], [[{'type': 'concede_spell'}]], device='cpu')

        # Unknown character ID
        with self.assertRaisesRegex(ValueError, "Unknown action character"):
            encode_candidate_batch([view], [[{'type': 'attack', 'character_id': 'alien'}]], device='cpu')

        # Unknown target
        with self.assertRaisesRegex(ValueError, "Unknown target character"):
            encode_candidate_batch(
                [view],
                [[{'type': 'attack', 'character_id': 'nanali', 'target_id': 'b:ghost'}]],
                device='cpu',
            )

        # Unknown card instance in hand
        with self.assertRaisesRegex(ValueError, "not found in viewer's hand"):
            encode_candidate_batch([view], [[{'type': 'play_card', 'card_id': 'fake_card_id'}]], device='cpu')

    def test_device_cuda_and_no_cpu_fallback(self):
        """Test that default/cuda device raises if CUDA unavailable (no silent fallback)."""
        import torch

        if not torch.cuda.is_available():
            # Must raise when no device specified or cuda specified
            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                encode_candidate_batch([], [], device=None)
            with self.assertRaisesRegex(RuntimeError, "CUDA device 'cuda' requested"):
                encode_candidate_batch([], [], device="cuda")
        else:
            # If actual CUDA is available, verify CUDA encoding matches CPU encoding
            s = new_game(seed=111, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
            view = observe(s, 'a', include_previews=False)
            actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']

            batch_cpu = encode_candidate_batch([view], [actions], device='cpu')
            batch_gpu = encode_candidate_batch([view], [actions], device='cuda')

            np.testing.assert_allclose(
                batch_gpu.candidates.cpu().numpy(),
                batch_cpu.candidates.numpy(),
                atol=1e-5,
            )

    def test_direct_raw_tokens_roundtrip(self):
        """Test extract_raw_tokens and encode_tokens direct API."""
        s = new_game(seed=122, skip_mulligan=True, decks={'a': self.decks['starter'], 'b': self.decks['starter']})
        view = observe(s, 'a', include_previews=False)
        actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']

        tokens = extract_raw_tokens([view], [actions], capacity=16)
        self.assertIsInstance(tokens, RawCandidateTokens)
        self.assertEqual(tokens.int_tokens.shape, (1, 16, 7))

        batch_from_tokens = encode_tokens(tokens, device='cpu')
        batch_direct = encode_candidate_batch([view], [actions], capacity=16, device='cpu')

        np.testing.assert_allclose(
            batch_from_tokens.candidates.numpy(),
            batch_direct.candidates.numpy(),
            atol=1e-6,
        )
        self.assertEqual(batch_from_tokens.schema_sha, SCHEMA_SHA)


if __name__ == '__main__':
    unittest.main()
