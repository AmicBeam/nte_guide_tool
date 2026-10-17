"""Public value parity only; these host tests do not qualify a GPU driver."""
import ast
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.engine.duel_v2 import new_game, observe, apply_action
from app.modules.card_game.engine.duel_v2.entities import needs_hydration
from app.modules.card_game.rl.batched_duel.gpu_cached_policy_projection import cached_policy_public_legal_decision
from app.modules.card_game.rl.batched_duel.gpu_value_projection import (
    cached_policy_value_observation, derive_value_observation, _ValueObserve,
)
from app.modules.card_game.rl.batched_duel.gpu_policy_projection import ProjectionSpecializationError
from app.modules.card_game.rl.seven_lineup import encode
from scripts.validate_duel_v2_native_soa import compare_boundary
from tests.test_duel_v2_policy_projection import trace_states


def changed_observe(state, side, *, include_previews=True):
    actions = legal_actions(state, 'a')
    return actions


class NoLegalAccess:
    def __deepcopy__(self, memo):
        return self

    def __iter__(self):
        raise AssertionError('Value encoding read legal_actions')

    def __len__(self):
        raise AssertionError('Value encoding read legal_actions')

    def __getitem__(self, key):
        raise AssertionError('Value encoding read legal_actions')


class ValueProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = []
        phases = set()
        for i, state in enumerate(trace_states(max_steps=48)):
            if i < 16 or state['phase'] not in phases:
                cls.states.append(deepcopy(state))
            phases.add(state['phase'])
            if {'mulligan', 'playing', 'choice'} <= phases and i >= 16:
                break
            if i >= 671:
                raise AssertionError('Bounded canonical trace did not reach choice')
        playing = next(s for s in cls.states if s['phase'] == 'playing')
        cls.states.append(apply_action(playing, playing['active_side'], {'type': 'concede'}))

    def test_both_viewers_all_value_fields_encoding_alias_and_rng_purity(self):
        phases = set()
        for state in self.states:
            phases.add(state['phase'])
            state['_test_alias'] = state['sides']['a']['hand']
            before = deepcopy(state)
            rng = state['rng']
            for side in ('a', 'b'):
                _, cached = cached_policy_public_legal_decision(state, side)
                canonical = observe(state, side, include_previews=False)
                value = cached_policy_value_observation(state, side)
                self.assertEqual(value['legal_actions'], [])
                cached['legal_actions'] = []
                compare_boundary(value, cached, 'all value-view fields and aliases')
                x, c = encode(value, [])
                xx, cc = encode(canonical, [])
                np.testing.assert_array_equal(x, xx)
                np.testing.assert_array_equal(c, cc)
                self.assertEqual(c.shape[0], 0)
                poisoned = deepcopy(value)
                poisoned['legal_actions'] = NoLegalAccess()
                px, pc = encode(poisoned, [])
                np.testing.assert_array_equal(px, x)
                np.testing.assert_array_equal(pc, c)
                other = 'b' if side == 'a' else 'a'
                self.assertTrue(all(card == {'hidden': True}
                                    for card in value['sides'][other]['hand']
                                    if card.get('hidden')))
                compare_boundary(state, before, 'parent state/RNG/types/aliases')
                self.assertEqual(state['rng'], rng)
                self.assertIs(state['_test_alias'], state['sides']['a']['hand'])
            self.assertFalse(needs_hydration(state))
        self.assertTrue({'mulligan', 'playing', 'choice', 'finished'} <= phases)

    def test_enumeration_absent_and_no_cache_between_calls_or_mutations(self):
        from app.modules.card_game.rl.batched_duel import gpu_value_projection as module
        state = new_game(seed=724, skip_mulligan=True)
        fn = module._value_observe
        with patch.dict(fn.__globals__, legal_actions=lambda *a: (_ for _ in ()).throw(AssertionError('enumerated'))):
            first = cached_policy_value_observation(state, 'a')
        hero = next(iter(state['sides']['a']['characters'].values()))
        hero['flags']['atk_buff'] = 3
        hero['hp'] -= 1
        changed = cached_policy_value_observation(state, 'a')
        _, expected = cached_policy_public_legal_decision(state, 'a')
        expected['legal_actions'] = []
        compare_boundary(changed, expected, 'fresh mutable rule-query inputs')
        self.assertNotEqual(first, changed)
        changed['sides']['a']['hand'].clear()
        self.assertTrue(cached_policy_value_observation(state, 'a')['sides']['a']['hand'])

    def test_canonical_shape_changes_fail_closed(self):
        with self.assertRaises(ProjectionSpecializationError):
            derive_value_observation(changed_observe)
        malformed = ast.parse('if include_previews:\n    legal_entries = []\nelse:\n    legal_entries = []').body[0]
        with self.assertRaises(ProjectionSpecializationError):
            _ValueObserve().visit(malformed)


if __name__ == '__main__':
    unittest.main()
