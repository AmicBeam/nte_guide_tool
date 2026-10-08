"""Training-loop contracts. Optional SB3 test is skipped without contrib."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECK, STARTER_DECKS
from app.modules.card_game.rl.adapter import DuelAdapter
from app.modules.card_game.rl.budget import BudgetStop, CountingBackend, TrainBudget
from app.modules.card_game.rl.checkpoint import assert_compatible_identity, current_checkpoint_config, validate_checkpoint
from app.modules.card_game.rl.encoding import CARD_IDS, CHARACTER_IDS
from app.modules.card_game.rl.policies import RandomMaskedPolicy, VisibleEngineRulePolicy
from app.modules.card_game.rl.transfer import (
    apply_card_swaps, default_training_deck, parse_card_swaps, resolve_training_deck, similar_card_sources,
)
from app.modules.card_game.rl.rewards import (
    SHAPE_AWAKEN, SHAPE_COLLAPSE, SHAPE_DAMAGE, SHAPE_DOWN, SHAPE_HARMONY, TERMINAL_LOSS,
    shape_reward,
)
from tests.test_duel_v2_rl import FakeBackend, TinyEncoder


class ConcedeBackend(FakeBackend):
    def observe(self, state, side):
        view = super().observe(state, side)
        view['legal_actions'] = [{'action': {'type': 'end_turn'}}, {'action': {'type': 'concede'}}]
        return view


class BudgetAndPolicyTest(unittest.TestCase):
    def test_shaping_scores_harmony_awaken_and_dealt_damage_only(self):
        events = [
            {'seq': 1, 'type': 'harmony', 'side': 'a'},
            {'seq': 2, 'type': 'awaken', 'side': 'a'},
            {'seq': 3, 'type': 'combat', 'side': 'b', 'source': 'a:nanali',
             'before': {'hp': 8, 'shield': 0}, 'after': {'hp': 5, 'shield': 0}},
            {'seq': 4, 'type': 'combat', 'side': 'a', 'source': 'b:xun',
             'before': {'hp': 7, 'shield': 0}, 'after': {'hp': 4, 'shield': 0}},
            {'seq': 5, 'type': 'harmony', 'side': 'b'},
            {'seq': 6, 'type': 'down', 'side': 'b', 'source': 'xun', 'target': 'b:xun'},
            {'seq': 7, 'type': 'down', 'side': 'a', 'source': 'nanali', 'target': 'a:nanali'},
            {'seq': 8, 'type': 'collapse', 'side': 'b', 'after': {'collapse': True}},
            {'seq': 9, 'type': 'collapse', 'side': 'b', 'amount': 1,
             'before': {'collapse_count': 3}, 'after': {'collapse_count': 4}},
            {'seq': 10, 'type': 'collapse', 'side': 'a', 'after': {'collapse': True}},
        ]
        reward, parts = shape_reward(events, 'a', 0)
        self.assertAlmostEqual(
            reward,
            SHAPE_HARMONY + SHAPE_AWAKEN + SHAPE_DAMAGE * 3 + SHAPE_DOWN + SHAPE_COLLAPSE,
        )
        self.assertEqual(parts['harmony'], 1)
        self.assertEqual(parts['awaken'], 1)
        self.assertEqual(parts['damage_hp'], 3)
        self.assertEqual(parts['down'], 1)
        self.assertEqual(parts['collapse'], 1)
        none, _ = shape_reward(events, 'a', 10)
        self.assertEqual(none, 0.0)

    def test_adapter_step_adds_shaping_to_zero_mid_reward(self):
        class HarmonyBackend(FakeBackend):
            def new_game(self, *, seed=0, **options):
                state = super().new_game(seed=seed, **options)
                state['event_seq'] = 0
                state['events'] = []
                return state

            def apply_action(self, state, side, action):
                nxt = super().apply_action(state, side, action)
                seq = int(nxt.get('event_seq') or 0) + 1
                nxt['event_seq'] = seq
                nxt['events'] = list(nxt.get('events') or []) + [
                    {'seq': seq, 'type': 'harmony', 'side': side},
                ]
                return nxt

        adapter = DuelAdapter(HarmonyBackend([('a', None)] * 8), TinyEncoder(), max_game_actions=20)
        first = adapter.reset()
        self.assertEqual(first.reward, 0.0)
        nxt = adapter.step(0, revision=first.decision.revision)
        self.assertAlmostEqual(nxt.reward, SHAPE_HARMONY)
        self.assertEqual(nxt.info['shaping']['harmony'], 1)

    def test_terminal_step_reward_is_not_mixed_with_shaping(self):
        class FinishWithHarmony(FakeBackend):
            def new_game(self, *, seed=0, **options):
                state = super().new_game(seed=seed, **options)
                state['event_seq'] = 0
                state['events'] = []
                return state

            def apply_action(self, state, side, action):
                nxt = super().apply_action(state, side, action)
                seq = int(nxt.get('event_seq') or 0) + 1
                nxt['event_seq'] = seq
                nxt['events'] = list(nxt.get('events') or []) + [
                    {'seq': seq, 'type': 'harmony', 'side': side},
                ]
                return nxt

        adapter = DuelAdapter(
            FinishWithHarmony([('a', None), ('a', 'b')]), TinyEncoder(), max_game_actions=20)
        first = adapter.reset()
        nxt = adapter.step(0, revision=first.decision.revision)
        self.assertTrue(nxt.terminated)
        self.assertEqual(nxt.reward, TERMINAL_LOSS)
        self.assertEqual(nxt.info['terminal_reward'], TERMINAL_LOSS)
        self.assertAlmostEqual(nxt.info['episode_shaping'], SHAPE_HARMONY)
        self.assertEqual(nxt.info['episode_score']['terminal'], TERMINAL_LOSS)
        self.assertAlmostEqual(nxt.info['episode_score']['shaping'], SHAPE_HARMONY)

    def test_learner_never_receives_concede(self):
        adapter = DuelAdapter(
            ConcedeBackend([('a', None)] * 8), TinyEncoder(),
            max_game_actions=20, allow_concede=False, strict_isolation=True,
        )
        transition = adapter.reset()
        types = [action.get('type') for action in transition.decision.actions]
        self.assertIn('end_turn', types)
        self.assertNotIn('concede', types)

    def test_budget_stops_steps_not_last_game(self):
        budget = TrainBudget(max_raw_steps=4, max_games=3, max_seconds=30,
                             reserve_raw_steps=1, reserve_games=1, reserve_seconds=5)
        budget.consume_game()
        budget.consume_step()
        self.assertIsNone(budget.remaining_reason())
        budget.raw_steps = 3
        with self.assertRaises(BudgetStop) as exc:
            budget.consume_step()
        self.assertEqual(exc.exception.reason, 'max_raw_steps')

    def test_counting_backend_and_rule_policy(self):
        budget = TrainBudget(max_raw_steps=10, max_games=4, max_seconds=30)
        backend = CountingBackend(FakeBackend([('a', None), ('a', None)]), budget)
        backend.new_game(seed=1)
        self.assertEqual(budget.games, 1)
        backend.apply_action({'index': 0}, 'a', {'type': 'end_turn'})
        self.assertEqual(budget.raw_steps, 1)
        observation = {
            'viewer_side': 'a',
            'legal_actions': [
                {'action': {'type': 'end_turn'}, 'preview': {}},
                {'action': {'type': 'attack', 'character_id': 'nanali'}, 'preview': {'attack': 4, '攻击次数': 1, 'counter': 0}},
            ],
            'sides': {'a': {'hand': []}},
        }
        actions = (observation['legal_actions'][0]['action'], observation['legal_actions'][1]['action'])
        self.assertEqual(VisibleEngineRulePolicy()(observation, actions), 1)
        observation['legal_actions'].insert(0, {'action': {'type': 'mulligan', 'card_ids': []}})
        actions = tuple(item['action'] for item in observation['legal_actions'])
        self.assertEqual(VisibleEngineRulePolicy()(observation, actions), 0)
        rng = __import__('random').Random(0)
        self.assertIn(RandomMaskedPolicy(rng)(observation, actions), range(len(actions)))

    def test_thread_budget_counts_without_lost_increments(self):
        from concurrent.futures import ThreadPoolExecutor
        from app.modules.card_game.rl.budget import make_thread_counters
        budget = TrainBudget(max_raw_steps=10000, max_games=10000, max_seconds=60)
        budget.attach_shared(make_thread_counters())

        def work(_rank):
            for _ in range(100):
                budget.consume_step()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))
        self.assertEqual(budget.raw_steps, 800)

    def test_status_board_reads_artifact_dir(self):
        from app.modules.card_game.rl.status_board import list_run_snapshots, resolve_run, snapshot_run
        folder = Path(tempfile.mkdtemp())
        run = folder / 'duel-v2-rl-demo'
        run.mkdir()
        (run / 'status.json').write_text(json.dumps({
            'ok': True, 'phase': 'update', 'elapsed_seconds': 12, 'max_seconds': 60,
            'num_timesteps': 128, 'n_updates': 2, 'raw_steps': 40, 'games_started': 3, 'loss': 1.25,
        }), encoding='utf-8')
        (run / 'train.log').write_text('phase=update n_updates=2\n', encoding='utf-8')
        (run / 'metrics.jsonl').write_text(
            json.dumps({'event': 'rollout_end', 'train/loss': 1.25, 'env_steps_per_sec': 10}) + '\n',
            encoding='utf-8',
        )
        listed = list_run_snapshots(folder)
        self.assertEqual(listed[0]['name'], 'duel-v2-rl-demo')
        self.assertEqual(listed[0]['n_updates'], 2)
        snap = snapshot_run(resolve_run(folder, 'duel-v2-rl-demo'))
        self.assertEqual(snap['phase'], 'update')
        self.assertEqual(snap['loss_series'], [1.25])
        self.assertIsNone(resolve_run(folder, '../secrets'))
        from app.modules.card_game.rl.status_board import pid_alive
        self.assertTrue(pid_alive(os.getpid()))
        self.assertFalse(pid_alive(0))

    def test_default_training_deck_is_official_starter(self):
        deck = default_training_deck()
        self.assertEqual(deck['id'], STARTER_DECK['id'])
        self.assertEqual(deck['name'], STARTER_DECK['name'])
        self.assertEqual(list(deck['character_ids']), list(STARTER_DECK['character_ids']))
        self.assertEqual(list(deck['card_ids']), list(STARTER_DECK['card_ids']))
        self.assertEqual(list(CHARACTER_IDS)[:4], list(STARTER_DECK['character_ids']))
        self.assertIn('bohe', CHARACTER_IDS)
        self.assertIn('baicang', CHARACTER_IDS)
        self.assertEqual(set(deck['card_ids']) <= set(CARD_IDS), True)
        self.assertEqual(resolve_training_deck(None)['card_ids'], deck['card_ids'])
        self.assertEqual(resolve_training_deck('starter')['id'], 'starter')
        rush = resolve_training_deck('weave-rush')
        self.assertEqual(rush['id'], 'weave-rush')
        self.assertEqual(set(rush['character_ids']), {'bohe', 'baicang', 'zero', 'iloy'})
        self.assertEqual(set(rush['card_ids']) <= set(CARD_IDS), True)
        in_deck = set(deck['card_ids'])
        pair = None
        for card_id in CARD_IDS:
            if card_id not in in_deck or CARDS[card_id].get('derived'):
                continue
            match = next(
                (
                    other for other in CARD_IDS
                    if other not in in_deck
                    and not CARDS[other].get('derived')
                    and CARDS[other]['character_id'] == CARDS[card_id]['character_id']
                    and CARDS[other]['type'] == CARDS[card_id]['type']
                ),
                None,
            )
            if match:
                pair = (card_id, match)
                break
        self.assertIsNotNone(pair)
        owned, unused = pair
        swapped = apply_card_swaps(deck, parse_card_swaps(f'{owned}:{unused}'))
        self.assertNotIn(owned, swapped['card_ids'])
        self.assertIn(unused, swapped['card_ids'])
        sources = similar_card_sources([owned], [owned, unused])
        self.assertEqual(sources[unused], [owned])
        other = next(item for item in STARTER_DECKS if item['id'] not in ('starter', 'weave-rush'))
        with self.assertRaises(ValueError):
            resolve_training_deck(other['id'])

    def test_public_sample_and_asymmetric_eval_interfaces(self):
        from app.modules.card_game.content.duel_v2.catalog import PUBLIC_CHARACTER_IDS
        from app.modules.card_game.rl.capability import (
            compiled_public_support, encoder_public_support, matchup_decks,
            sample_public_deck,
        )
        from app.modules.card_game.rl.evaluate import _play_match
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        from app.modules.card_game.rl.transfer import resolve_opponent_deck
        support = encoder_public_support()
        self.assertTrue(support['ok'])
        self.assertEqual(support['character_count'], 6)
        self.assertEqual(support['kit_card_count'], 48)
        compiled = compiled_public_support()
        self.assertTrue(compiled['ok'])
        self.assertEqual(compiled['missing_ir_cards'],[])
        deck = sample_public_deck(21)
        self.assertEqual(set(deck['character_ids']) <= set(PUBLIC_CHARACTER_IDS), True)
        opponent = resolve_opponent_deck('weave-rush')
        self.assertEqual(opponent['id'], 'weave-rush')
        matchup = matchup_decks(learning_side='a', learner_deck=deck, opponent_deck=opponent)
        self.assertEqual(matchup['b']['id'], 'weave-rush')
        self.assertNotEqual(matchup['a']['card_ids'], matchup['b']['card_ids'])
        from app.modules.card_game.rl.budget import TrainBudget
        report = _play_match(
            TrainBudget(max_raw_steps=8, max_games=2, max_seconds=2),
            VisibleEngineRulePolicy(), VisibleEngineRulePolicy(),
            seed=3, learning_side='a', sample_opponent=True, deck=opponent,
        )
        self.assertEqual(report['deck']['id'], 'weave-rush')
        self.assertTrue(set(report['opponent_deck']['character_ids']) <= set(PUBLIC_CHARACTER_IDS))
        from app.modules.card_game.rl.capability import resolve_model_capability, capability_ready_manifest
        legacy = resolve_model_capability({}, 'starter')
        self.assertEqual(legacy['kind'], 'legacy_exact_mirror')
        self.assertFalse(legacy['human_public_custom'])
        with self.assertRaises(ValueError):
            resolve_model_capability({'capability': {'kind': 'legacy_exact_mirror', 'human_public_custom': True}}, 'starter')
        ready = resolve_model_capability({'capability': capability_ready_manifest('weave-rush')}, 'weave-rush')
        self.assertTrue(ready['human_public_custom'])

    def test_inspect_checkpoint_still_rejects_weights(self):
        config = current_checkpoint_config()
        config['trainable'] = True
        config['weights'] = 'model.zip'
        with self.assertRaises(ValueError):
            validate_checkpoint(config)
        assert_compatible_identity(current_checkpoint_config())


@unittest.skipUnless(
    importlib.util.find_spec('gymnasium') and importlib.util.find_spec('numpy'),
    'Optional Gymnasium/NumPy are not installed',
)
class GymTruncationTrainTest(unittest.TestCase):
    def test_training_path_pads_candidates_in_gym_not_adapter(self):
        from app.modules.card_game.rl.gym_compat import make_gym_env
        adapter = DuelAdapter(FakeBackend([('a', None)] * 8), TinyEncoder(),
                              max_game_actions=20, strict_isolation=False)
        adapter.mask_concede = True
        env = make_gym_env(adapter)
        observation, _info = env.reset()
        self.assertEqual(observation['candidates'].shape, (256, 1))
        self.assertGreater(float(observation['candidates'][0, 0]), 0)
        self.assertEqual(float(observation['candidates'][5, 0]), 0)
        mask = env.action_masks()
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])

    def test_opponent_window_cut_is_truncated_zero_reward(self):
        from app.modules.card_game.rl.gym_compat import make_gym_env
        adapter = DuelAdapter(FakeBackend([('a', None), ('b', None)]), TinyEncoder(), max_game_actions=1)
        env = make_gym_env(adapter)
        env.reset()
        _obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0)
        self.assertFalse(info.get('bootstrap_allowed'))

    def test_threaded_vec_env_steps_two_envs(self):
        import numpy as np
        from app.modules.card_game.rl.gym_compat import make_gym_env
        from app.modules.card_game.rl.thread_vec_env import ThreadedVecEnv

        def make_env():
            adapter = DuelAdapter(
                FakeBackend([('a', None)] * 80), TinyEncoder(),
                max_game_actions=40, strict_isolation=False,
            )
            return make_gym_env(adapter)

        vec = ThreadedVecEnv([make_env, make_env])
        try:
            obs = vec.reset()
            self.assertEqual(obs['candidates'].shape[0], 2)
            vec.step_async(np.array([0, 0]))
            obs, rewards, dones, infos = vec.step_wait()
            self.assertEqual(len(rewards), 2)
            self.assertEqual(len(dones), 2)
            self.assertEqual(len(infos), 2)
        finally:
            vec.close()

    def test_process_vec_env_steps_two_envs(self):
        import numpy as np
        from app.modules.card_game.rl.gym_compat import make_gym_env
        from app.modules.card_game.rl.process_vec_env import ProcessVecEnv

        def make_env():
            adapter = DuelAdapter(
                FakeBackend([('a', None)] * 80), TinyEncoder(),
                max_game_actions=40, strict_isolation=False,
            )
            return make_gym_env(adapter)

        vec = ProcessVecEnv([make_env, make_env])
        try:
            obs = vec.reset()
            self.assertEqual(obs['candidates'].shape[0], 2)
            vec.step_async(np.array([0, 0]))
            obs, rewards, dones, infos = vec.step_wait()
            self.assertEqual(len(rewards), 2)
            self.assertEqual(len(dones), 2)
            self.assertEqual(len(infos), 2)
            masks = vec.env_method('action_masks')
            self.assertEqual(len(masks), 2)
        finally:
            vec.close()


@unittest.skipUnless(
    importlib.util.find_spec('sb3_contrib') and importlib.util.find_spec('stable_baselines3')
    and importlib.util.find_spec('gymnasium') and importlib.util.find_spec('numpy'),
    'Optional SB3/Gymnasium are not installed',
)
class MaskableUpdateTest(unittest.TestCase):
    def test_one_update_save_load_masked_predict(self):
        import numpy as np
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from app.modules.card_game.rl.gym_compat import make_gym_env
        from app.modules.card_game.rl.checkpoint import load_training_checkpoint, write_training_checkpoint
        from app.modules.card_game.rl.policies import MaskablePolicy

        stages = [('a', None)] * 400
        adapter = DuelAdapter(FakeBackend(stages), TinyEncoder(), max_game_actions=300)
        env = ActionMasker(make_gym_env(adapter), lambda inner: inner.action_masks())
        model = MaskablePPO(
            'MultiInputPolicy', env, n_steps=8, batch_size=8, n_epochs=1,
            learning_rate=1e-3, verbose=0, device='cpu',
            policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])),
        )
        model.learn(total_timesteps=8, progress_bar=False)
        self.assertGreaterEqual(int(getattr(model, '_n_updates', 0)), 1)
        with tempfile.TemporaryDirectory() as folder:
            write_training_checkpoint(folder, model, {'n_updates': int(model._n_updates)})
            loaded, meta = load_training_checkpoint(folder, env=env, device='cpu')
            self.assertTrue(meta['trainable'])
            self.assertEqual(meta['weights'], 'model.zip')
            reset = env.reset()
            observation = reset[0] if isinstance(reset, tuple) else reset
            mask = env.action_masks()
            action, _ = loaded.predict(observation, action_masks=mask, deterministic=True)
            self.assertTrue(mask[int(np.asarray(action).item())])
            policy = MaskablePolicy(loaded, TinyEncoder(), action_capacity=256)
            index = policy({'public': 0, 'viewer_side': 'a'}, ({'type': 'end_turn'},))
            self.assertEqual(index, 0)
            self.assertTrue((Path(folder) / 'model.zip').is_file())
            payload = json.loads((Path(folder) / 'checkpoint.json').read_text(encoding='utf-8'))
            self.assertGreaterEqual(payload.get('n_updates', 0), 1)

    def test_candidate_scoring_policy_one_update(self):
        import numpy as np
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from app.modules.card_game.rl.gym_compat import make_gym_env
        from app.modules.card_game.rl.networks import CandidateScoringPolicy

        stages = [('a', None)] * 400
        adapter = DuelAdapter(FakeBackend(stages), TinyEncoder(), max_game_actions=300,
                              strict_isolation=False)
        env = ActionMasker(make_gym_env(adapter), lambda inner: inner.action_masks())
        model = MaskablePPO(
            CandidateScoringPolicy, env, n_steps=8, batch_size=8, n_epochs=1,
            learning_rate=1e-3, verbose=0, device='cpu',
            policy_kwargs=dict(hidden_dim=16),
        )
        model.learn(total_timesteps=8, progress_bar=False)
        self.assertGreaterEqual(int(getattr(model, '_n_updates', 0)), 1)
        device = next(model.policy.parameters()).device
        self.assertEqual(str(device), 'cpu')
        observation, _info = env.reset()
        mask = env.action_masks()
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        self.assertTrue(mask[int(np.asarray(action).item())])

    def test_freeze_generic_leaves_card_parameters_trainable(self):
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from app.modules.card_game.rl.gym_compat import make_gym_env
        from app.modules.card_game.rl.networks import CandidateScoringPolicy

        adapter = DuelAdapter(FakeBackend([('a', None)] * 80), TinyEncoder(), max_game_actions=40,
                              strict_isolation=False)
        env = ActionMasker(make_gym_env(adapter), lambda inner: inner.action_masks())
        model = MaskablePPO(
            CandidateScoringPolicy, env, n_steps=8, batch_size=8, n_epochs=1,
            learning_rate=1e-3, verbose=0, device='cpu',
            policy_kwargs=dict(hidden_dim=8),
        )
        policy = model.policy
        before = [param.detach().clone() for param in policy.generic_parameters()]
        policy.freeze_generic()
        self.assertTrue(all(not param.requires_grad for param in policy.generic_parameters()))
        self.assertTrue(any(param.requires_grad for param in policy.card_parameters()))
        for param in policy.card_parameters():
            if param.requires_grad:
                param.data.add_(0.25)
        self.assertTrue(all(
            (left == right).all().item()
            for left, right in zip(before, [param.detach() for param in policy.generic_parameters()])
        ))
