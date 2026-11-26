"""Controlled typed driver wiring; not compiled-rule or natural-game evidence."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_duel.record_soa import (
    compile_schema, pack_game_states, unpack_game_states,
)
from app.modules.card_game.rl.batched_duel.native_soa_reference import (
    NativeSoAReference, StateHandle,
)


def schema():
    return compile_schema({
        'Game': {'kind': 'record', 'capacity': 8, 'fields': {
            'phase': 'str', 'winner': 'str', 'turn': 'int', 'version': 'int',
            'depth': 'int', 'selected': 'int',
        }},
    }, max_strings=32, max_string_bytes=512)


class FakeSoACall:
    def __init__(self, rows, state=None):
        self.rows, self.state = rows, state

    def check(self):
        return True

    def decode(self, path=()):
        result = []
        for row in self.rows:
            value = row
            for step in path:
                value = value[step]
            result.append(value)
        return result

    def select_state(self, path=()):
        if path not in ((), (0,)):
            raise AssertionError(path)
        if self.state is None:
            raise AssertionError('typed state missing')
        return self.state


class FakeTypedExecutor:
    """Bounded fake typed executor. Never evidence of rule parity."""

    def __init__(self, schema):
        self.schema = schema
        self.calls = []

    def _public(self, game):
        actions = [dict(type='attack', character_id='zero', branch=i) for i in range(2)]
        view = dict(depth=game['depth'], selected=game.get('selected', -1),
                    legal_actions=[dict(action=a) for a in actions])
        return actions, view

    def call(self, entry, state, arguments, state_positions=(0,)):
        self.calls.append((entry, id(state.buffers['i32']), [row[0] for row in arguments],
                           state_positions, state.rows))
        if any(row[position] is not None for row in arguments for position in state_positions):
            raise AssertionError('Game graph passed as typed argument')
        games = unpack_game_states(state)
        output = []
        packed = None
        for game, row in zip(games, arguments):
            if entry == 'acting_side':
                output.append('a')
            elif entry == 'public_legal_decision':
                output.append(self._public(game))
            elif entry == 'sample_world_guard_views_packet':
                pair = self._public(game)
                output.append((None, ((pair, pair), (pair, pair))))
                packed = pack_game_states([deepcopy(game)], self.schema)
            elif entry in ('simulate_action', 'apply_action'):
                new = deepcopy(game)
                new['depth'] += 1
                if game['depth'] == 0:
                    new['selected'] = row[2]['branch']
                if new['depth'] == 2:
                    new.update(phase='finished', winner='a' if new['selected'] == 0 else 'b')
                output.append(None)
                packed = pack_game_states([new], self.schema)
            elif entry == 'clean':
                packed = pack_game_states([deepcopy(game)], self.schema)
                output.append(None)
            else:
                raise AssertionError(entry)
        return FakeSoACall(output, packed)


class ControlledPolicy:
    def scores(self, x, c):
        return np.zeros(len(c))

    def scores_value(self, x, c):
        return self.scores(x, c), 0.

    def wdl(self, x, is_actor):
        return np.array([.25, .5, .25])


class NativeSoAReferenceTest(unittest.TestCase):
    def setUp(self):
        self.schema = schema()
        packed = pack_game_states([
            dict(phase='playing', winner=None, turn=0, version=0, depth=0, selected=-1)
        ], self.schema)
        self.handle = StateHandle(packed)
        self.executor = FakeTypedExecutor(self.schema)
        self.runtime = NativeSoAReference(self.executor, schema=self.schema)

    def test_full32_driver_never_repacks_game_or_binds_teacher(self):
        from app.modules.card_game.rl import information_search, seven_lineup
        from app.modules.card_game.rl.batched_duel import native_soa_reference as module
        source = inspect.getsource(module.NativeSoAReference)
        self.assertNotIn('pack_game_states', source)
        self.assertNotIn('unpack_game_states', source)
        self.assertNotIn('new_game', source)
        def encode(view, actions):
            return (np.array([view['depth'], view['selected']], dtype=np.float32),
                    np.array([[a['branch']] for a in actions], dtype=np.float32).reshape(-1, 1))
        def forbidden(*args, **kwargs):
            raise AssertionError('Python rule fallback')
        with patch.object(seven_lineup, 'encode', encode), patch.multiple(information_search,
                acting_side=forbidden, apply_action=forbidden, simulate_action=forbidden,
                clean=forbidden, sample_world=forbidden, leaf_value=forbidden,
                short_terminal_value=forbidden):
            result = self.runtime.search(self.handle, 'a',
                                         {'a': ControlledPolicy(), 'b': ControlledPolicy()},
                                         seed=27, simulations=32, gumbel_candidates=2, max_depth=3)
        self.assertTrue(result['complete'])
        self.assertEqual(result['simulations'], 32)
        self.assertEqual(result['search_choice'], 0)
        self.assertEqual(self.runtime.rows['sample_world_guard_views_packet'], 32)
        self.assertGreater(self.runtime.rows['simulate_action'], 32)
        self.assertTrue(all(item is None for row in self.executor.calls for item in row[2]))
        self.assertEqual(self.handle['phase'], 'playing')
        self.assertIsNone(self.handle.get('winner'))
        self.assertEqual(unpack_game_states(self.handle.batch)[0]['depth'], 0)
        with self.assertRaises(ValueError):
            self.runtime.search(self.handle, 'a', {'a': ControlledPolicy(), 'b': ControlledPolicy()},
                                seed=1, teacher_mode='covered_expert', terminal_horizon=3)
        with self.assertRaises(KeyError):
            self.handle['sides']

    def test_owned_scratch_and_batch_bound(self):
        before = self.handle.batch.buffers['i32']
        advanced = self.runtime.apply(self.handle, 'a', dict(type='attack', character_id='zero', branch=0))
        self.assertIsNot(advanced.batch, self.handle.batch)
        self.assertIsNot(advanced.batch.buffers['i32'], before)
        self.assertEqual(self.handle['phase'], 'playing')
        self.assertEqual(unpack_game_states(advanced.batch)[0]['depth'], 1)
        with self.assertRaises(ValueError):
            NativeSoAReference(self.executor, max_batch_rows=0)
        from app.modules.card_game.rl.batched_duel.record_soa import RecordSoAError
        with self.assertRaises(RecordSoAError):
            self.runtime.call('acting_side', self.handle.batch,
                              [[{'sides': {}, 'rng': 0, 'next_entity': 1}]])


if __name__ == '__main__':
    unittest.main()
