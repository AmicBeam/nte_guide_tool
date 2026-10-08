"""Frozen six-character encoder contracts, including historical copy/record fields.

Synthetic compatibility cases do not assert current Xun card effects.
"""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, STARTER_DECKS
from app.modules.card_game.rl.encoding import CHARACTER_IDS
from app.modules.card_game.rl import DuelAdapter, V2Backend, V2Encoder
from app.modules.card_game.rl.contracts import Outcome
from app.modules.card_game.rl.encoding import (
    ACTION_DIM, ACTION_LAYOUT, CARD_IDS, OBSERVATION_DIM, PUBLIC_CARD_WIDTH,
    _public_card_vector, card_identity_index, is_shared_action_vector, zero_card_identity,
)
from app.modules.card_game.rl.checkpoint import current_checkpoint_config, validate_checkpoint, read_checkpoint_metadata, write_checkpoint_metadata
from app.modules.card_game.rl.rewards import TERMINAL_LOSS, TERMINAL_WIN


def _copy_holder(card_id: str) -> str:
    owner = CARDS[card_id]['character_id']
    return next((cid for cid in CHARACTER_IDS if cid != owner), CHARACTER_IDS[0])


def card(cid='N03', instance='a-1', copied=False, expiry=21):
    result = dict(deepcopy(CARDS[cid]), card_id=cid, instance_id=instance, copy=copied)
    if copied:
        result.update(
            character_id=_copy_holder(cid),
            original_character_id=CARDS[cid]['character_id'],
            expires_turn=expiry,
        )
    return result


def projection():
    sides = {}
    for side in ('a', 'b'):
        characters = []
        for cid in STARTER_DECK['character_ids']:
            definition = CHARACTERS[cid]
            characters.append(dict(deepcopy(definition), hp=definition['max_hp'], shield=0,
                harmony=0, energy=0, awakened=False, growth=0, shape=None, shape_id=None,
                down_turns=0, next_attack_bonus=0, return_after_sortie=False, slow=None, burn=0, star=None))
        sides[side] = dict(name=side, hp=30, shield=0, ap=2, normal_attack_available=True,
            front=None, last_front=None, characters=characters, hand=[], hand_count=0,
            deck_count=20, discard=[], records=[], turn_count=20, fatigue=0, used={},
            harmonized={}, extra_flower=False)
    return dict(rules_version='duel_v2', version=1, phase='playing', active_side='a',
        viewer_side='a', is_my_turn=True, turn=40, winner=None, sides=sides,
        pending_choice=None, legal_actions=[], events=[], logs=[], resolving_card=None)


class EncodingTest(unittest.TestCase):
    def setUp(self):
        self.encoder = V2Encoder()
        self.view = projection()

    def test_dimensions_finite_and_shape_id(self):
        self.view['sides']['a']['characters'][0].update(shape_id='N07', shape='display can change')
        vector = self.encoder.encode_observation(self.view)
        self.assertEqual(len(vector), OBSERVATION_DIM)
        self.assertTrue(all(0 <= x <= 1 for x in vector))

    def test_legacy_public_copy_both_seats_and_relative_expiry(self):
        for side in ('a', 'b'):
            self.view['sides'][side]['hand'] = [card(copied=True)]
        self.view['sides']['b']['hand'].append({'hidden': True})
        before = self.encoder.encode_observation(self.view)
        self.view['sides']['b']['hand'][0]['expires_turn'] = 22
        self.assertNotEqual(before, self.encoder.encode_observation(self.view))
        a = _public_card_vector(card(copied=True, expiry=21), hidden=False, turn_count=20)
        b = _public_card_vector(card(copied=True, expiry=31), hidden=False, turn_count=30)
        self.assertEqual(a, b)
        self.assertEqual(len(a), PUBLIC_CARD_WIDTH)
        with self.assertRaises(ValueError):
            _public_card_vector(dict(card(), character_id='xun'), hidden=False)

    def test_legacy_record_without_instance_and_resolving(self):
        self.view['sides']['a']['records'] = [dict(card_id='N03', name=CARDS['N03']['name'], type='battle', character_id='nanali')]
        a = self.encoder.encode_observation(self.view)
        self.view['resolving_card'] = card('Z02')
        self.assertNotEqual(a, self.encoder.encode_observation(self.view))

    def test_legacy_public_counters_and_status_are_encoded(self):
        base = self.encoder.encode_observation(self.view)
        for key, value in [('last_front', 'zero'), ('turn_count', 21), ('fatigue', 1),
                           ('used', {'record': True, 'instant': True}), ('harmonized', {'zero': True}),
                           ('extra_flower', True)]:
            view = deepcopy(self.view)
            view['sides']['a'][key] = value
            with self.subTest(key=key):
                self.assertNotEqual(base, self.encoder.encode_observation(view))
        for key, value in [('next_attack_bonus', 2), ('return_after_sortie', True),
                           ('slow', {'amount': 2, 'end': 21}), ('burn', 2), ('star', 21)]:
            view = deepcopy(self.view)
            view['sides']['a']['characters'][0][key] = value
            with self.subTest(key=key):
                self.assertNotEqual(base, self.encoder.encode_observation(view))

    def test_ten_choices_and_true_kinds(self):
        for kind in ('discard', 'inspect_top', 'enemy_hand'):
            self.view['phase'] = 'choice'
            self.view['pending_choice'] = dict(side='a', kind=kind, choices=[
                dict(id=str(i), card=card('N03' if i % 2 else 'Z02', str(i))) for i in range(10)])
            self.assertEqual(len(self.encoder.encode_observation(self.view)), OBSERVATION_DIM)
            a = self.encoder.encode_action(self.view, {'type': 'choose', 'choice_id': '0'})
            b = self.encoder.encode_action(self.view, {'type': 'choose', 'choice_id': '1'})
            self.assertNotEqual(a, b)
            self.assertEqual(len(a), ACTION_DIM)
        self.view['pending_choice']['choices'].append(dict(id='overflow', card=card()))
        with self.assertRaises(ValueError):
            self.encoder.encode_observation(self.view)

    def test_actions_target_mulligan_copy_and_expiry(self):
        self.view['sides']['a']['hand'] = [card('N06', 'a-1'), card('N03', 'a-2', True), card('N03', 'a-3', True, 22)]
        actions = [dict(type='play_card', card_id='a-1', target_id='a:zero'),
                   dict(type='attack', character_id='nanali'), dict(type='ultimate', character_id='zero'),
                   dict(type='cycle', card_id='a-1'), dict(type='end_turn'), dict(type='concede')]
        for action in actions:
            self.assertEqual(len(self.encoder.encode_action(self.view, action)), ACTION_DIM)
        self.assertNotEqual(self.encoder.encode_action(self.view, dict(type='mulligan', card_ids=['a-1'])),
                            self.encoder.encode_action(self.view, dict(type='mulligan', card_ids=['a-2'])))
        self.assertNotEqual(self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-2')),
                            self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-3')))
        self.view['sides']['a']['hand'].append(card('Y03', 'a-4'))
        player = self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-4', target_id='a:player'))
        ally = self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-4', target_id='a:zero'))
        self.assertEqual(len(player), ACTION_DIM)
        self.assertNotEqual(player, ally)

    def test_generic_action_features_drop_card_identity(self):
        self.assertEqual(ACTION_LAYOUT['choice_card'].stop, ACTION_DIM)
        self.view['sides']['a']['hand'] = [card('N03', 'a-1'), card('N04', 'a-2')]
        play_a = self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-1'))
        play_b = self.encoder.encode_action(self.view, dict(type='play_card', card_id='a-2'))
        attack = self.encoder.encode_action(self.view, dict(type='attack', character_id='nanali'))
        end_turn = self.encoder.encode_action(self.view, dict(type='end_turn'))
        self.assertNotEqual(play_a, play_b)
        self.assertEqual(zero_card_identity(play_a), zero_card_identity(play_b))
        self.assertEqual(card_identity_index(play_a), CARD_IDS.index('N03') + 1)
        self.assertEqual(card_identity_index(attack), 0)
        self.assertTrue(is_shared_action_vector(attack))
        self.assertTrue(is_shared_action_vector(end_turn))
        self.assertFalse(is_shared_action_vector(play_a))
        self.assertTrue(is_shared_action_vector(play_a, overlap_card_indexes=(CARD_IDS.index('N03') + 1,)))

    def test_real_event_names_and_unknown(self):
        for event in ('turn', 'enter', 'resource', 'play', 'shape', 'revive', 'finish', 'flower', 'new_cosmetic_event'):
            self.view['events'] = [dict(type=event, amount=1, side='a')]
            self.assertEqual(len(self.encoder.encode_observation(self.view)), OBSERVATION_DIM)

    def test_legacy_encoder_rejects_unsupported_characters_and_unknown_cards(self):
        from app.modules.card_game.rl.transfer import assert_encoder_compatible
        # A valid current deck may still be outside this frozen encoder's seats.
        other = deepcopy(STARTER_DECK)
        other['character_ids'] = ['nanali', 'zero', 'jiuyuan', 'xun']
        other['card_ids'] = [key for owner in other['character_ids']
                             for key, definition in CARDS.items()
                             if definition['character_id'] == owner and not definition.get('derived')]
        with self.assertRaises(ValueError):
            assert_encoder_compatible(other)
        view = deepcopy(self.view)
        view['sides']['a']['hand'] = [card('A01')]
        with self.assertRaises(ValueError):
            self.encoder.encode_observation(view)
        from app.modules.card_game.engine.duel_v2 import new_game, acting_side
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row
        state = new_game(seed=4, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
        side = acting_side(state)
        state['sides'][side]['hand'][0]['card_id'] = 'X99'
        with self.assertRaisesRegex(ValueError, 'cannot represent'):
            pack_python_row(state)

    def test_hidden_hand_contents_ignored_and_rng_rejected(self):
        self.view['sides']['b']['hand'] = [{'hidden': True}]
        before = self.encoder.encode_observation(self.view)
        self.view['sides']['b']['hand'][0]['card_id'] = 'N01'
        self.assertEqual(before, self.encoder.encode_observation(self.view))
        self.view['rng'] = 123
        with self.assertRaises(ValueError):
            self.encoder.encode_observation(self.view)


class TinyEncoder:
    version = 'test'
    observation_dim = action_dim = 1
    def encode_observation(self, view):
        return [view['public'] / 10]
    def encode_action(self, view, action):
        return [0.5]


class FakeBackend:
    def __init__(self, stages):
        self.stages = stages
        self.calls = 0
    def new_game(self, *, seed=0, **options):
        return {'index': 0, 'hidden_secret': 'must not reach callbacks'}
    def current_player(self, state):
        return self.stages[state['index']][0]
    def outcome(self, state):
        winner = self.stages[state['index']][1]
        return Outcome(winner is not None, winner)
    def observe(self, state, side):
        return {'public': state['index'], 'viewer_side': side,
                'legal_actions': [] if self.outcome(state).terminated else [{'action': {'type': 'end_turn'}}]}
    def apply_action(self, state, side, action):
        self.calls += 1
        return dict(state, index=state['index'] + 1)


class AdapterTest(unittest.TestCase):
    def make(self, stages, **kwargs):
        return DuelAdapter(FakeBackend(stages), TinyEncoder(), **kwargs)

    def test_invalid_stale_and_defensive_copies(self):
        env = self.make([('a', None), ('a', None)])
        t = env.reset()
        t.decision.actions[0]['type'] = 'tampered'
        self.assertEqual(env.decision().actions[0]['type'], 'end_turn')
        for index in (-1, 1, True, 0.5):
            with self.assertRaises(ValueError):
                env.step(index, revision=t.decision.revision)
        with self.assertRaises(ValueError):
            env.step(0, revision=t.decision.revision - 1)
        self.assertEqual(env.backend.calls, 0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(0, revision=t.decision.revision)

    def test_sparse_rewards_both_seats_and_terminal_precedence(self):
        for seat in ('a', 'b'):
            for winner in ('a', 'b', 'draw'):
                env = self.make([(seat, None), (None, winner)], learning_side=seat, max_game_actions=1)
                t = env.reset()
                result = env.step(0, revision=t.decision.revision)
                self.assertEqual(
                    result.reward,
                    0 if winner == 'draw' else TERMINAL_WIN if winner == seat else TERMINAL_LOSS,
                )
                self.assertTrue(result.terminated)
                self.assertFalse(result.truncated)
                self.assertFalse(any(result.decision.mask))

    def test_truncation_keeps_candidates_but_prohibits_step(self):
        env = self.make([('a', None), ('a', None)], max_game_actions=1)
        t = env.reset()
        end = env.step(0, revision=t.decision.revision)
        self.assertTrue(end.truncated)
        self.assertTrue(end.info['bootstrap_allowed'])
        self.assertTrue(end.decision.mask[0])
        self.assertEqual(end.reward, 0)
        with self.assertRaises(RuntimeError):
            env.step(0, revision=end.decision.revision)

    def test_frozen_opponent_isolated_and_bounded(self):
        seen = []
        def opponent(view, actions):
            self.assertNotIn('hidden_secret', view)
            seen.append(view['viewer_side'])
            actions[0]['type'] = 'tampered'
            return 0
        env = self.make([('b', None), ('b', None), ('a', None)], opponent=opponent, max_opponent_actions=1)
        end = env.reset()
        self.assertTrue(end.truncated)
        self.assertFalse(end.info['bootstrap_allowed'])
        self.assertEqual(end.info['truncation_reason'], 'max_opponent_actions')
        self.assertEqual(env.backend.calls, 1)
        self.assertEqual(seen, ['b'])

    def test_empty_and_overflow_rejected(self):
        env = self.make([('a', None)])
        for entries in ([], [{'action': {'i': i}} for i in range(257)]):
            env.backend.observe = lambda state, side: {'public': 0, 'legal_actions': entries}
            with self.assertRaises(ValueError):
                env.reset()
        self.assertEqual(env.backend.calls, 0)

    def test_encoder_nonfinite_rejected(self):
        env = self.make([('a', None)])
        env.encoder.encode_observation = lambda view: [float('nan')]
        with self.assertRaises(ValueError):
            env.reset()


class CheckpointAndCliTest(unittest.TestCase):
    def test_metadata_roundtrip_and_incompatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            write_checkpoint_metadata(path)
            self.assertEqual(read_checkpoint_metadata(path), current_checkpoint_config())
        for key, value in [('encoder_version', 'old'), ('catalog_sha256', 'old'),
                           ('action_capacity', 1), ('schema_version', True), ('trainable', True),
                           ('card_ids', None), ('weights', 'model.zip')]:
            config = current_checkpoint_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint(config)

    def test_cli_never_touches_engine_or_training_libraries(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/train_duel_v2.py'
        spec = importlib.util.spec_from_file_location('duel_rl_cli_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with patch.object(V2Backend, 'new_game', side_effect=AssertionError('No games')), \
             patch.object(V2Backend, 'apply_action', side_effect=AssertionError('No steps')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['--check']), 0)
            with self.assertRaises(SystemExit) as help_exit:
                cli.main(['--help'])
            self.assertEqual(help_exit.exception.code, 0)
            for arguments in ([], ['--enable-training'], ['--train'],
                              ['--train', '--output', 'unused', '--freeze-generic'],
                              ['--train', '--output', 'unused', '--kl-coef', '0.1']):
                with self.assertRaises(SystemExit) as disabled:
                    cli.main(arguments)
                self.assertNotEqual(disabled.exception.code, 0)


@unittest.skipUnless(importlib.util.find_spec('gymnasium') and importlib.util.find_spec('numpy'),
                     'Optional Gymnasium/NumPy are not installed')
class OptionalGymTest(unittest.TestCase):
    def test_real_gym_type_and_candidate_observation(self):
        import gymnasium as gym
        from app.modules.card_game.rl.gym_compat import make_gym_env
        adapter = DuelAdapter(FakeBackend([('a', None), ('a', None)]), TinyEncoder(), max_game_actions=1)
        env = make_gym_env(adapter)
        self.assertIsInstance(env, gym.Env)
        observation, info = env.reset(seed=2)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(observation['candidates'].shape, (256, 1))
        with self.assertRaises(ValueError):
            env.step(0.5)
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(env.action_masks()[0])

    def test_reset_done_rejected_and_opponent_truncation_is_not_a_loss(self):
        from app.modules.card_game.rl.gym_compat import make_gym_env
        env = make_gym_env(DuelAdapter(FakeBackend([(None, 'draw')]), TinyEncoder()))
        with self.assertRaises(RuntimeError):
            env.reset()
        adapter = DuelAdapter(FakeBackend([('a', None), ('b', None)]), TinyEncoder(), max_game_actions=1)
        env = make_gym_env(adapter)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0)
        self.assertFalse(info.get('bootstrap_allowed'))
        self.assertTrue(env.observation_space.contains(observation))

    def test_maskable_predict_receives_candidates_and_mask(self):
        import numpy as np
        from app.modules.card_game.rl.policies import MaskablePolicy
        seen = []
        class FrozenModel:
            def predict(self, observation, *, action_masks, deterministic):
                seen.append((observation, action_masks, deterministic))
                return np.asarray(0), None
        policy = MaskablePolicy(FrozenModel(), TinyEncoder())
        self.assertEqual(policy({'public': 0}, ({'type': 'end_turn'},)), 0)
        self.assertEqual(seen[0][0]['candidates'].shape, (256, 1))
        self.assertEqual(seen[0][1].sum(), 1)
        self.assertTrue(seen[0][2])


class RealEngineSmokeTest(unittest.TestCase):
    def test_two_actions_only(self):
        # This is an API compatibility check, never a rollout or training episode.
        backend = V2Backend()
        encoder = V2Encoder()
        from app.modules.card_game.rl.transfer import default_training_deck
        deck = default_training_deck()
        state = backend.new_game(
            seed=7, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck},
        )
        before = deepcopy(state)
        view = backend.observe(state, 'a')
        self.assertEqual(len(encoder.encode_observation(view)), OBSERVATION_DIM)
        for entry in view['legal_actions']:
            self.assertEqual(len(encoder.encode_action(view, entry['action'])), ACTION_DIM)
        state2 = backend.apply_action(state, 'a', {'type': 'attack', 'character_id': 'nanali'})
        self.assertEqual(state, before)
        self.assertEqual(len(encoder.encode_observation(backend.observe(state2, 'a'))), OBSERVATION_DIM)
        ended = backend.apply_action(state2, 'a', {'type': 'concede'})
        self.assertTrue(backend.outcome(ended).terminated)
        self.assertEqual(len(encoder.encode_observation(backend.observe(ended, 'a'))), OBSERVATION_DIM)

    def test_encodes_weave_rush_mirror(self):
        backend = V2Backend()
        encoder = V2Encoder()
        rush = next(item for item in STARTER_DECKS if item['id'] == 'weave-rush')
        state = backend.new_game(
            seed=7, first_side='a', skip_mulligan=True, decks={'a': rush, 'b': rush},
        )
        view = backend.observe(state, 'a')
        self.assertEqual(len(encoder.encode_observation(view)), OBSERVATION_DIM)
        self.assertTrue(any(character['id'] == 'bohe' for character in view['sides']['a']['characters']))
        for entry in view['legal_actions']:
            self.assertEqual(len(encoder.encode_action(view, entry['action'])), ACTION_DIM)
        state2 = backend.apply_action(state, 'a', {'type': 'attack', 'character_id': 'bohe'})
        self.assertEqual(len(encoder.encode_observation(backend.observe(state2, 'a'))), OBSERVATION_DIM)

    def test_engine_states_convert_to_public_replay(self):
        from app.modules.card_game.engine.duel_v2.replay_log import (
            assemble_replay_game, build_match_payload, leak_markers,
        )
        from app.modules.card_game.rl.transfer import default_training_deck
        backend = V2Backend()
        deck = default_training_deck()
        opening = backend.new_game(
            seed=7, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck},
        )
        after = backend.apply_action(opening, 'a', {'type': 'attack', 'character_id': 'nanali'})
        payload = build_match_payload(opening, after, room_code='RTEST1', learning_side='a')
        self.assertEqual(leak_markers(payload), [])
        view = payload['views']['a']
        self.assertGreaterEqual(len(view['events']), 1)
        assembled = assemble_replay_game(view['opening_board'], view['events'])
        live = backend.observe(after, 'a')
        self.assertEqual(assembled['sides']['a']['hp'], live['sides']['a']['hp'])
        self.assertEqual(assembled['sides']['b']['hp'], live['sides']['b']['hp'])
        self.assertEqual(assembled['phase'], live['phase'])


if __name__ == '__main__':
    unittest.main()
