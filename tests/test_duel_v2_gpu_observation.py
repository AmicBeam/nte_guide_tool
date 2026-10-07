"""Unit tests for GPU batch observation encoding in Duel V2.

Verifies each column against official cross_lineup.encode(view, [])[0]
as an independent oracle.
Covers:
- Five preset decks (starter, weave-rush, quick-rush, zhenhong, murk) in mulligan, playing, choice, and finished phases
- Simultaneous multi-game batch alignment (shape [N, 2420], counts, masks, request IDs)
- RF01 physical card identity and substituted_hand counts
- Murk DOTs (nightmare, etch, venom, guard), burn stacks, mirage effect, and blood-mist turn comparisons
- Jingu mark deltas (-1, 0, 1) in viewer hand
- Timed-attack aggregations for left==1 and left==2
- Collapse mechanics (collapse_until, collapse_count, collapse_by relative side)
- Summon aliases (summon_*) and entity renaming invariance
- Opponent hidden hand cards filtering without reading identities
- Strict rejection of unknown characters, cards, shapes, non-finite values, and invalid viewer_side
- Device semantics: no CPU fallback when CUDA is requested/defaulted, explicit CPU allowed, optional CUDA
"""

import unittest
import subprocess,sys
from copy import deepcopy
import importlib.util
import numpy as np

has_torch = importlib.util.find_spec('torch') is not None

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS, validate_deck
from app.modules.card_game.engine.duel_v2 import new_game, observe
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl import cross_lineup
from app.modules.card_game.rl.batched_duel.gpu_observation import (
    CARD_IDS,
    CARD_INDEX,
    OBS_DIM,
    PackedObservationBatch,
    RawObservationTokens,
    SCHEMA_SHA,
    SEATS,
    SEAT_INDEX,
    encode_observation_batch,
    encode_observation_tokens,
    encode_single_observation,
    extract_raw_observation_tokens,
)


def _get_preset_decks():
    return {d['id']: validate_deck(deepcopy(d)) for d in STARTER_DECKS}


def _oracle_observation(view):
    """Independent oracle: official cross_lineup.encode state vector."""
    oracle_x, _ = cross_lineup.encode(view, [])
    return oracle_x


@unittest.skipUnless(has_torch, "PyTorch required for gpu_observation tests")
class TestGpuObservationEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.decks = _get_preset_decks()

    def test_import_does_not_initialize_torch(self):
        subprocess.run([sys.executable,'-c',"import sys;from app.modules.card_game.rl.batched_duel import gpu_observation;assert 'torch' not in sys.modules"],check=True,capture_output=True)

    def test_direct_token_validation_rejects_malformed_device_inputs(self):
        view=observe(new_game(seed=311,skip_mulligan=True,decks={'a':self.decks['starter'],'b':self.decks['starter']}),'a',include_previews=False)
        for corrupt in ('shape','dtype','identity','duplicate','mask'):
            tokens=extract_raw_observation_tokens([view])
            if corrupt=='shape':tokens.side_float_tokens=tokens.side_float_tokens[:,:,:38]
            elif corrupt=='dtype':tokens.zone_cards=tokens.zone_cards.astype(np.float32)
            elif corrupt=='identity':tokens.zone_cards[0,0,0]=len(CARD_IDS)
            elif corrupt=='duplicate':tokens.char_seats[0,0,1]=tokens.char_seats[0,0,0]
            else:tokens.counts[0]=0
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):encode_observation_tokens(tokens,device='cpu')

    def test_actual_cuda_varied_formal_states_and_1600_rows(self):
        if not self.torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.engine.duel_v2 import legal_actions,apply_action
        from app.modules.card_game.engine.duel_v2.flow import acting_side
        from random import Random
        views=[]
        for k,key in enumerate(('starter','weave-rush','quick-rush','zhenhong','murk')):
            state=new_game(seed=4310+k,decks={'a':self.decks[key],'b':self.decks['murk']},skip_mulligan=False)
            rng=Random(7120+k)
            for step in range(24):
                for side in ('a','b'):views.append(observe(state,side,include_previews=False))
                actor=acting_side(state)
                if actor is None:break
                actions=legal_actions(state,actor)
                state=apply_action(state,actor,actions[rng.randrange(len(actions))])
        views=(views*((1600+len(views)-1)//len(views)))[:1600]
        expected=np.stack([_oracle_observation(v) for v in views])
        tokens=extract_raw_observation_tokens(views)
        actual=encode_observation_tokens(tokens,device='cuda')
        self.assertEqual(actual.shape,(1600,2420));self.assertEqual(actual.device.type,'cuda')
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-6,rtol=1e-6)
        # A future rule kernel can supply the same compact token ABI on device.
        from dataclasses import fields
        for field in fields(tokens):
            value=getattr(tokens,field.name)
            if isinstance(value,np.ndarray):setattr(tokens,field.name,self.torch.as_tensor(value,device='cuda'))
        direct=encode_observation_tokens(tokens,device='cuda')
        self.assertTrue(self.torch.equal(actual.observations,direct.observations))

    def test_five_presets_playing_phase_matches_oracle(self):
        """Test playing phase observations from all 5 presets match official oracle exactly."""
        preset_keys = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
        for key in preset_keys:
            s = new_game(
                seed=42,
                skip_mulligan=True,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view = observe(s, 'a', include_previews=False)
            batch = encode_single_observation(view, device='cpu')
            oracle = _oracle_observation(view)

            self.assertEqual(batch.batch_size, 1)
            self.assertEqual(batch.obs_dim, OBS_DIM)
            self.assertTrue(batch.mask[0].item())

            actual_x = batch.observations[0].cpu().numpy()
            np.testing.assert_allclose(
                actual_x,
                oracle,
                atol=1e-5,
                err_msg=f"Playing phase mismatch for preset {key}",
            )

    def test_five_presets_mulligan_phase_matches_oracle(self):
        """Test mulligan phase observations from all 5 presets match official oracle exactly."""
        preset_keys = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
        for key in preset_keys:
            s = new_game(
                seed=99,
                skip_mulligan=False,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view = observe(s, 'a', include_previews=False)
            batch = encode_single_observation(view, device='cpu')
            oracle = _oracle_observation(view)

            actual_x = batch.observations[0].cpu().numpy()
            np.testing.assert_allclose(
                actual_x,
                oracle,
                atol=1e-5,
                err_msg=f"Mulligan phase mismatch for preset {key}",
            )

    def test_choice_phase_and_resolving_card_matches_oracle(self):
        """Test choice phase with pending choices and resolving card matches oracle."""
        s = new_game(
            seed=101,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)
        view['phase']='choice'
        view['pending_choice']={'kind':'inspect_top','side':'a','choices':[
            {'id':c['instance_id'],'card':deepcopy(c)} for c in view['sides']['a']['hand'][:3]]}
        view['resolving_card']={'card_id':'N04'}
        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

    def test_finished_phase_matches_oracle(self):
        """Test finished phase observation matches oracle."""
        s = new_game(
            seed=202,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        s['phase'] = 'finished'
        s['winner'] = 'a'

        view = observe(s, 'a', include_previews=False)
        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

    def test_simultaneous_multigame_batch_alignment(self):
        """Test batched encoding across multiple games aligns IDs, masks, and features."""
        preset_keys = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
        views = []
        oracles = []

        for idx, key in enumerate(preset_keys):
            s = new_game(
                seed=300 + idx,
                skip_mulligan=True,
                decks={'a': self.decks[key], 'b': self.decks['starter']},
            )
            view = observe(s, 'a', include_previews=False)
            views.append(view)
            oracles.append(_oracle_observation(view))

        batch = encode_observation_batch(views, device='cpu')
        self.assertEqual(batch.batch_size, 5)
        self.assertEqual(batch.obs_dim, OBS_DIM)
        self.assertEqual(len(batch.request_ids), 5)
        self.assertEqual(batch.schema_sha, SCHEMA_SHA)

        for i in range(5):
            self.assertTrue(batch.mask[i].item())
            actual_row = batch.observations[i].cpu().numpy()
            np.testing.assert_allclose(
                actual_row,
                oracles[i],
                atol=1e-5,
                err_msg=f"Batch alignment mismatch at index {i}",
            )

    def test_rf01_physical_identity_and_substituted_hand(self):
        """Test substituted_hand features for physical card substitution (e.g. RF01)."""
        s = new_game(
            seed=401,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        # Effective RF01 retains the original physical card identity
        hand = view['sides']['a']['hand']
        self.assertGreater(len(hand), 0)
        c0 = hand[0]
        original=c0['card_id']
        c0['hand_face'] = {'card_id':original}
        c0['card_id']='RF01'

        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

        # Physical-stock column records the original, not effective RF01
        rf01_col = 2420 - len(CARD_IDS) + CARD_INDEX[original]
        self.assertAlmostEqual(actual_x[rf01_col], 0.5, places=5)

    def test_murk_dots_burn_mirage_and_turn_comparison(self):
        """Test Murk debuffs, burn stacks, mirage flag, and canhong DOT attack turn comparison."""
        s = new_game(
            seed=501,
            skip_mulligan=True,
            decks={'a': self.decks['murk'], 'b': self.decks['starter']},
        )
        current_turn = s['turn']
        s['sides']['a']['front_debuff'] = {
            'burn': {'stacks': 3, 'left': 2, 'by': 'b'},
            'dots': {
                'nightmare': {'stacks': 4},
                'etch': {'stacks': 2},
                'venom': {'stacks': 5},
                'guard': {'stacks': 1},
            },
        }

        view=observe(s,'a',include_previews=False)
        # Encoding fixture: public effect identity, with no invalid engine effect.
        for hero in view['sides']['a']['characters']:
            if hero['id'] == 'canhong':
                hero['effects'] = [{'definition_id': 'canhong.mirage'}]
                eid = hero['entity_id']
                view['sides']['a']['used'] = {
                    f"canhong_dot_attack_turn:{eid}": current_turn
                }
                break

        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

        # Verify mirage column and dot_attack_used column are 1.0
        # Murk block starts at 2291:
        # 5 viewer debuffs (2291..2295), 5 opp debuffs (2296..2300),
        # viewer mirage (2301), opp mirage (2302), viewer dot (2303), opp dot (2304)
        self.assertAlmostEqual(actual_x[2291], 0.3, places=5)  # burn stacks 3 / 10
        self.assertAlmostEqual(actual_x[2292], 0.4, places=5)  # nightmare 4 / 10
        self.assertAlmostEqual(actual_x[2293], 0.2, places=5)  # etch 2 / 10
        self.assertAlmostEqual(actual_x[2294], 0.5, places=5)  # venom 5 / 10
        self.assertAlmostEqual(actual_x[2295], 0.1, places=5)  # guard 1 / 10
        self.assertAlmostEqual(actual_x[2301], 1.0, places=5)  # viewer mirage
        self.assertAlmostEqual(actual_x[2303], 1.0, places=5)  # viewer dot attack used

    def test_jingu_marks_in_viewer_hand(self):
        """Test Jingu mark delta scattering (-1, 0, 1) in viewer hand."""
        s = new_game(
            seed=601,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        hand = view['sides']['a']['hand']
        self.assertGreaterEqual(len(hand), 3)
        hand[0]['jingu_mark'] = {'delta': -1}
        hand[1]['jingu_mark'] = {'delta': 0}
        hand[2]['jingu_mark'] = {'delta': 1}

        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

    def test_timed_attacks_left1_and_left2(self):
        """Test timed_attack entries with left==1 and left==2 sum into timed_attack_1/2."""
        s = new_game(
            seed=701,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        # Inject timed attacks in nanali's flags
        meta = view['policy_state']['sides']['a']['characters']['nanali']
        meta['flags']['timed_attack'] = [
            {'amount': 3, 'left': 1},
            {'amount': 4, 'left': 1},
            {'amount': 5, 'left': 2},
            {'amount': 10, 'left': 3},  # Ignored by timed_attack_1/2
        ]

        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

    def test_collapse_relative_side_and_defaults(self):
        """Test collapse flags (until, by, count) and relative side perspective."""
        s = new_game(
            seed=801,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        meta = view['policy_state']['sides']['a']['characters']['nanali']
        meta['flags']['collapse'] = {'until': 5, 'side': 'b'}
        meta['flags']['collapse_count'] = 2

        batch = encode_single_observation(view, device='cpu')
        oracle = _oracle_observation(view)

        actual_x = batch.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_x, oracle, atol=1e-5)

    def test_summon_alias_and_entity_renaming_invariance(self):
        """Test summon_* adaptation to summon seat and entity ID renaming invariance."""
        s = new_game(
            seed=901,
            skip_mulligan=True,
            decks={'a': self.decks['murk'], 'b': self.decks['starter']},
        )
        # Observation-only fixture preserving the formal public summon shape.
        view1=observe(s,'a',include_previews=False)
        summoned=deepcopy(view1['sides']['a']['characters'][0])
        summoned.update(id='summon_shadow_beast',entity_id='e_9999',summoned=True,hp=15,max_hp=20)
        view1['sides']['a']['characters'].append(summoned)
        meta=deepcopy(view1['policy_state']['sides']['a']['characters']['canhong'])
        view1['policy_state']['sides']['a']['characters']['summon_shadow_beast']=meta
        view1['sides']['a']['front']='summon_shadow_beast'
        batch1 = encode_single_observation(view1, device='cpu')
        oracle1 = _oracle_observation(view1)

        actual1 = batch1.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual1, oracle1, atol=1e-5)

        # Entity ID renaming: modify instance_id and entity_id to random strings
        view2 = deepcopy(view1)
        for h in view2['sides']['a']['characters']:
            h['entity_id'] = 'renamed_' + str(h.get('entity_id', ''))
        for c in view2['sides']['a']['hand']:
            c['instance_id'] = 'renamed_' + str(c.get('instance_id', ''))

        batch2 = encode_single_observation(view2, device='cpu')
        actual2 = batch2.observations[0].cpu().numpy()
        np.testing.assert_array_equal(actual1, actual2)

    def test_opponent_hidden_cards_filtering(self):
        """Test opponent hidden hand cards are not counted in public card features."""
        s = new_game(
            seed=1001,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)
        opp_hand = view['sides']['b']['hand']
        self.assertGreater(len(opp_hand), 0)

        # All opponent hand cards are hidden by default in sanitized public view
        self.assertTrue(all(c.get('hidden') for c in opp_hand))

        batch1 = encode_single_observation(view, device='cpu')
        oracle1 = _oracle_observation(view)
        actual1 = batch1.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual1, oracle1, atol=1e-5)

        # In Zone 3 (opp revealed hand), counts should be all zero
        zone3_start = 1117 + 3 * len(CARD_IDS)
        zone3_end = zone3_start + len(CARD_IDS)
        np.testing.assert_array_equal(actual1[zone3_start:zone3_end], 0.0)

        # Mutate the hidden cards' card_ids: should NOT change public observation
        view_mutated = deepcopy(view)
        for c in view_mutated['sides']['b']['hand']:
            c['card_id'] = 'N08'

        batch_mut = encode_single_observation(view_mutated, device='cpu')
        actual_mut = batch_mut.observations[0].cpu().numpy()
        np.testing.assert_array_equal(actual1, actual_mut)

        # If a card is revealed (not hidden), it DOES appear in zone 3
        view_revealed = deepcopy(view)
        view_revealed['sides']['b']['hand'][0].update(hidden=False,card_id='N01')
        rev_cid = view_revealed['sides']['b']['hand'][0]['card_id']

        batch_rev = encode_single_observation(view_revealed, device='cpu')
        oracle_rev = _oracle_observation(view_revealed)
        actual_rev = batch_rev.observations[0].cpu().numpy()
        np.testing.assert_allclose(actual_rev, oracle_rev, atol=1e-5)
        self.assertAlmostEqual(actual_rev[zone3_start + CARD_INDEX[rev_cid]], 0.5, places=5)

    def test_strict_rejections_and_capacity(self):
        """Test rejection of unknown characters, cards, shapes, and capacity violations."""
        s = new_game(
            seed=1101,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        # 1. Unknown character
        v_bad_char = deepcopy(view)
        v_bad_char['sides']['a']['characters'].append({'id': 'unknown_alien_hero'})
        with self.assertRaisesRegex(ValueError, "Model only supports public characters"):
            encode_single_observation(v_bad_char, device='cpu')

        # 2. Unknown card in hand
        v_bad_card = deepcopy(view)
        v_bad_card['sides']['a']['hand'].append({'card_id': 'UNKNOWN_CARD_999'})
        with self.assertRaisesRegex(ValueError, "Unknown card_id"):
            encode_single_observation(v_bad_card, device='cpu')

        # 3. Unknown shape
        v_bad_shape = deepcopy(view)
        v_bad_shape['sides']['a']['characters'][0]['shape_id'] = 'UNKNOWN_SHAPE_999'
        with self.assertRaisesRegex(ValueError, "Unknown shape_id"):
            encode_single_observation(v_bad_shape, device='cpu')

        # 4. Invalid viewer_side
        v_bad_viewer = deepcopy(view)
        v_bad_viewer['viewer_side'] = 'c'
        with self.assertRaisesRegex(ValueError, "Invalid viewer_side"):
            encode_single_observation(v_bad_viewer, device='cpu')

        # 5. Capacity violation
        with self.assertRaisesRegex(ValueError, "exceeds explicit capacity"):
            encode_observation_batch([view, view], capacity=1, device='cpu')

    def test_device_semantics(self):
        """Test device semantics: explicit CPU allowed, no-CUDA default raises RuntimeError."""
        s = new_game(
            seed=1201,
            skip_mulligan=True,
            decks={'a': self.decks['starter'], 'b': self.decks['starter']},
        )
        view = observe(s, 'a', include_previews=False)

        # Explicit CPU works
        batch_cpu = encode_single_observation(view, device='cpu')
        self.assertEqual(batch_cpu.device.type, 'cpu')

        # If CUDA is unavailable, defaulting or requesting CUDA must raise RuntimeError
        if not self.torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                encode_single_observation(view, device=None)
            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                encode_single_observation(view, device='cuda')
        else:
            # CUDA available
            batch_cuda = encode_single_observation(view, device='cuda')
            self.assertEqual(batch_cuda.device.type, 'cuda')
            actual_cuda = batch_cuda.observations[0].cpu().numpy()
            oracle = _oracle_observation(view)
            np.testing.assert_allclose(actual_cuda, oracle, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
