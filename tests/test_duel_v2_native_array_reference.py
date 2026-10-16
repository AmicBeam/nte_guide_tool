"""Controlled driver wiring; not compiled-rule or natural-game evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.batched_duel.native_array_reference import NativeArrayReference


class ControlledExecutor:
    def __init__(self):
        self.calls = []

    def call(self, entry, rows):
        self.calls.append((entry, len(rows)))
        output = []
        for arguments in rows:
            state = arguments[0]
            if entry == 'acting_side':
                output.append('a')
            elif entry == 'public_legal_decision':
                actions = [dict(type='attack', character_id='zero', branch=i) for i in range(2)]
                view = dict(depth=state['depth'], selected=state.get('selected', -1),
                            legal_actions=[dict(action=a) for a in actions])
                output.append((actions, view))
            elif entry == 'sample_world_guard_views_packet':
                pair = self.call('public_legal_decision', [[state, arguments[1]]])[0]
                output.append((deepcopy(state), ((pair, pair), (pair, pair))))
            elif entry in ('simulate_action', 'apply_action'):
                new = deepcopy(state)
                new['depth'] += 1
                if state['depth'] == 0:
                    new['selected'] = arguments[2]['branch']
                if new['depth'] == 2:
                    new.update(phase='finished', winner='a' if new['selected'] == 0 else 'b')
                output.append(new)
            elif entry == 'clean':
                output.append(deepcopy(state))
            else:
                raise AssertionError(entry)
        return output


class ControlledPolicy:
    def scores(self, x, c):
        return np.zeros(len(c))

    def scores_value(self, x, c):
        return self.scores(x, c), 0.

    def wdl(self, x, is_actor):
        return np.array([.25, .5, .25])


class NativeArrayReferenceTest(unittest.TestCase):
    def test_full32_driver_never_falls_back_to_python_rules(self):
        from app.modules.card_game.rl import information_search, seven_lineup
        runtime = NativeArrayReference(ControlledExecutor())
        state = dict(depth=0, phase='playing', winner=None)
        def encode(view, actions):
            return (np.array([view['depth'], view['selected']], dtype=np.float32),
                    np.array([[a['branch']] for a in actions], dtype=np.float32).reshape(-1, 1))
        def forbidden(*args, **kwargs):
            raise AssertionError('Python rule fallback')
        with patch.object(seven_lineup, 'encode', encode), patch.multiple(information_search,
                acting_side=forbidden, apply_action=forbidden, simulate_action=forbidden,
                clean=forbidden, sample_world=forbidden, leaf_value=forbidden):
            result = runtime.search(state, 'a', {'a': ControlledPolicy(), 'b': ControlledPolicy()},
                                    seed=27, simulations=32, gumbel_candidates=2, max_depth=3)
        self.assertTrue(result['complete'])
        self.assertEqual(result['simulations'], 32)
        self.assertEqual(result['search_choice'], 0)
        self.assertEqual(runtime.rows['sample_world_guard_views_packet'], 32)
        self.assertGreater(runtime.rows['simulate_action'], 32)
        self.assertEqual(state, dict(depth=0, phase='playing', winner=None))

    def test_bounded_tail_dispatch_preserves_order_and_row_count(self):
        executor = ControlledExecutor()
        runtime = NativeArrayReference(executor, max_batch_rows=2)
        values = runtime.call('acting_side', [[dict(depth=i)] for i in range(5)])
        self.assertEqual(values, ['a']*5)
        self.assertEqual(executor.calls, [('acting_side', 2), ('acting_side', 2), ('acting_side', 1)])
        self.assertEqual(runtime.dispatches['acting_side'], 3)
        self.assertEqual(runtime.rows['acting_side'], 5)
        with self.assertRaises(ValueError):
            NativeArrayReference(executor, max_batch_rows=0)


if __name__ == '__main__':
    unittest.main()
