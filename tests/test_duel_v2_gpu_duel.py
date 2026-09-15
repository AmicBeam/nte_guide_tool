"""Oracle diffs for the frozen-starter GPU duel. Requires torch."""
from __future__ import annotations

import importlib.util
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECK, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.rl.gpu_duel.catalog import (
    ACT_ATTACK, ACT_END, ACT_PLAY, N_SEATS, SEAT_INDEX, STARTER_LOCK, assert_gpu_deck, assert_starter_deck,
)

TORCH = importlib.util.find_spec('torch')


@unittest.skipUnless(TORCH, 'PyTorch is required for the GPU duel')
class GpuDuelOracleTest(unittest.TestCase):
    def setUp(self):
        from app.modules.card_game.rl.gpu_duel import GpuDuelEnv, python_mechan
        self.GpuDuelEnv = GpuDuelEnv
        self.python_mechan = python_mechan

    def _py(self, seed=7):
        return new_game(seed=seed, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})

    def _gpu_index(self, env, action):
        s = env.state
        n = int(s.legal_n[0])
        kind = action.get('type')
        side = int(s.active[0])
        if kind == 'end_turn':
            for index in range(n):
                if int(s.legal_type[0, index]) == ACT_END:
                    return index
        elif kind == 'attack':
            actor = SEAT_INDEX[action['character_id']]
            for index in range(n):
                if int(s.legal_type[0, index]) == ACT_ATTACK and int(s.legal_actor[0, index]) == actor:
                    return index
        elif kind == 'play_card':
            inst = int(str(action['card_id']).rsplit('-', 1)[-1])
            target = action.get('target_id')
            for index in range(n):
                if int(s.legal_type[0, index]) != ACT_PLAY:
                    continue
                slot = int(s.legal_hand[0, index])
                if int(s.hand_inst[0, side, slot]) != inst:
                    continue
                if target:
                    tside, tname = str(target).split(':', 1)
                    want_side = 0 if tside == 'a' else 1
                    want_seat = -1 if tname == 'player' else SEAT_INDEX[tname]
                    if int(s.legal_tside[0, index]) != want_side or int(s.legal_tseat[0, index]) != want_seat:
                        continue
                return index
        self.fail(f'no GPU legal slot for {action} among {n}')

    def test_starter_lock_rejects_other_seats(self):
        assert_starter_deck(STARTER_DECK)
        self.assertEqual(len(STARTER_LOCK), 64)
        self.assertEqual(N_SEATS, 6)
        rush = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        assert_gpu_deck(rush)
        with self.assertRaises(ValueError):
            assert_starter_deck({
                'id': 'x', 'name': 'x',
                'character_ids': ['nanali', 'zero', 'jiuyuan', 'xun'],
                'card_ids': list(STARTER_DECK['card_ids']),
            })

    def test_native_reset_accepts_weave_rush_and_cross(self):
        rush = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        env = self.GpuDuelEnv(2, device='cpu', deck=rush)
        env.reset(seeds=[1, 2], native=True)
        self.assertGreater(int(env.legal_count()[0]), 0)
        self.assertEqual(int(env.state.ch_present[0, 0, SEAT_INDEX['bohe']]), 1)
        self.assertEqual(int(env.state.ch_present[0, 0, SEAT_INDEX['nanali']]), 0)
        cross = self.GpuDuelEnv(2, device='cpu', matchup='cross')
        cross.reset(seeds=[3, 4], native=True)
        self.assertEqual(int(cross.state.ch_present[0, 0, SEAT_INDEX['nanali']]), 1)
        self.assertEqual(int(cross.state.ch_present[0, 1, SEAT_INDEX['bohe']]), 1)
        self.assertEqual(int(cross.state.ch_present[0, 1, SEAT_INDEX['nanali']]), 0)

    def test_pack_matches_python_opening(self):
        state = self._py()
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        self.assertEqual(env.snapshot(0), self.python_mechan(state))
        self.assertGreater(int(env.legal_count()[0]), 0)

    def test_native_opening_draws_for_first_player(self):
        env = self.GpuDuelEnv(2, device='cpu')
        env.reset(seeds=[1, 2], native=True)
        for index in range(2):
            first = int(env.state.active[index])
            self.assertEqual(int(env.state.deck_n[index, first]), 26)
            self.assertEqual(int(env.state.deck_n[index, 1 - first]), 27)

    def test_attack_matches_python(self):
        state = self._py()
        side = acting_side(state)
        attack = next(action for action in legal_actions(state, side) if action.get('type') == 'attack')
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        env.step([self._gpu_index(env, attack)])
        nxt = apply_action(state, side, attack)
        self.assertEqual(env.snapshot(0), self.python_mechan(nxt))

    def test_end_turn_matches_python(self):
        state = self._py()
        side = acting_side(state)
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        env.step([self._gpu_index(env, {'type': 'end_turn'})])
        nxt = apply_action(state, side, {'type': 'end_turn'})
        self.assertEqual(env.snapshot(0), self.python_mechan(nxt))

    def test_play_card_matches_python_when_present(self):
        state = self._py()
        side = acting_side(state)
        play = next((action for action in legal_actions(state, side) if action.get('type') == 'play_card'), None)
        if play is None:
            self.skipTest('opening hand has no playable card')
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        env.step([self._gpu_index(env, play)])
        nxt = apply_action(state, side, play)
        self.assertEqual(env.snapshot(0), self.python_mechan(nxt))

    def test_random_walk_stays_aligned(self):
        state = self._py(seed=11)
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        for _ in range(8):
            if state.get('phase') == 'finished':
                break
            side = acting_side(state)
            actions = [item for item in legal_actions(state, side) if item.get('type') != 'concede']
            if not actions:
                break
            action = actions[0]
            try:
                index = self._gpu_index(env, action)
            except AssertionError:
                break
            env.step([index])
            state = apply_action(state, side, action)
            self.assertEqual(env.snapshot(0), self.python_mechan(state))

    def test_native_reset_and_learner_step(self):
        import torch
        env = self.GpuDuelEnv(8, device='cpu')
        env.reset(seeds=list(range(8)), native=True)
        self.assertTrue((env.legal_count() > 0).all())
        self.assertTrue(((env.learner == 0) | (env.learner == 1)).all())
        from app.modules.card_game.rl.gpu_duel.observe import compact_observe
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, ppo_update, rollout_to_batch
        state, cand, _ = compact_observe(env.state)
        net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=16)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rollout = collect_rollout(env, net, horizon=2)
        ppo_update(net, opt, rollout_to_batch(rollout), epochs=1, minibatch=16)
        self.assertEqual(rollout['reward'].shape, (8, 2))

    def test_batch_reset_legal_and_step(self):
        env = self.GpuDuelEnv(8, device='cpu')
        env.reset(seeds=list(range(8)))
        counts = env.legal_count()
        self.assertTrue((counts > 0).all())
        env.step([0] * 8)
        done, _winner = env.outcome()
        self.assertEqual(int(done.sum()), int((env.state.phase == 3).sum()))

    def test_random_walk_longer_seeds(self):
        for seed in (3, 11, 21):
            state = self._py(seed=seed)
            env = self.GpuDuelEnv(1, device='cpu')
            env.reset_from_python([state])
            for _ in range(20):
                if state.get('phase') == 'finished':
                    break
                side = acting_side(state)
                actions = [item for item in legal_actions(state, side) if item.get('type') != 'concede']
                if not actions:
                    break
                action = actions[0]
                index = self._gpu_index(env, action)
                env.step([index])
                state = apply_action(state, side, action)
                self.assertEqual(env.snapshot(0), self.python_mechan(state), msg=f'seed={seed} action={action}')

    def test_compact_ppo_one_update(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.observe import compact_observe
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer, collect_rollout, ppo_update, rollout_to_batch
        env = self.GpuDuelEnv(4, device='cpu')
        env.reset(seeds=[0, 1, 2, 3])
        state, cand, _mask = compact_observe(env.state)
        net = CompactScorer(state.shape[-1], cand.shape[-1], hidden=32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rollout = collect_rollout(env, net, horizon=4)
        ppo_update(net, opt, rollout_to_batch(rollout), epochs=1, minibatch=16)
        self.assertGreater(rollout['state'].shape[0], 0)

    def test_cuda_batch_if_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        env = self.GpuDuelEnv(32, device='cuda')
        env.reset(seeds=list(range(32)))
        self.assertEqual(env.state.hp.device.type, 'cuda')
        env.step([0] * 32)
        self.assertTrue(int(env.legal_count().min()) >= 0)

    def test_encoder_observe_matches_v2encoder(self):
        import torch
        from app.modules.card_game.rl.encoding import V2Encoder
        from app.modules.card_game.rl.gpu_duel.encoder_obs import encoder_observe, reconstruct_observation
        state = self._py(seed=7)
        env = self.GpuDuelEnv(1, device='cpu')
        env.reset_from_python([state])
        viewer = int(env.state.active[0])
        recon = reconstruct_observation(env.state, 0, viewer)
        encoder = V2Encoder()
        expected = encoder.encode_observation(recon)
        got, cand, mask = encoder_observe(env.state)
        self.assertTrue(torch.allclose(got[0], torch.tensor(expected, dtype=torch.float32), atol=1e-6))
        self.assertEqual(int(mask[0].sum()), int((env.state.legal_type[0] >= 0).sum()))
        self.assertGreater(int(mask[0].sum()), 0)
        self.assertEqual(cand.shape[-1], encoder.action_dim)

    def test_encoder_policy_one_update(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.encoder_obs import encoder_observe
        from app.modules.card_game.rl.gpu_duel.policy import GpuCandidatePolicy
        from app.modules.card_game.rl.gpu_duel.ppo import collect_rollout, ppo_update, rollout_to_batch
        env = self.GpuDuelEnv(2, device='cpu')
        env.reset(seeds=[0, 1], native=True)
        state, cand, _mask = encoder_observe(env.state)
        net = GpuCandidatePolicy(state.shape[-1], cand.shape[-1], hidden=16)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rollout = collect_rollout(env, net, horizon=2, observe_fn=encoder_observe)
        ppo_update(net, opt, rollout_to_batch(rollout), epochs=1, minibatch=8)
        self.assertEqual(rollout['state'].shape[-1], state.shape[-1])
