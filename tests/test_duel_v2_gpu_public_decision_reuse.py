"""Exact CPU source contracts for the optional frozen public decision entry."""
from copy import deepcopy
from unittest.mock import Mock, patch
import unittest

from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2 import projection
from app.modules.card_game.rl.batched_duel import gpu_rule_entries
from app.modules.card_game.rl.batched_duel.gpu_public_decision_reuse import public_legal_decision_reuse


class PublicDecisionReuseTest(unittest.TestCase):
    def test_full_order_observation_and_input_match_canonical(self):
        for first in ('a', 'b'):
            for skip in (False, True):
                state = new_game(seed=17, first_side=first, skip_mulligan=skip)
                for side in ('a', 'b'):
                    with self.subTest(first=first, skip=skip, side=side):
                        before = deepcopy(state)
                        expected = gpu_rule_entries.seven_public_legal_decision(state, side)
                        self.assertEqual(public_legal_decision_reuse(state, side), expected)
                        self.assertEqual(state, before)
        terminal = apply_action(new_game(seed=19, skip_mulligan=True), 'a', {'type': 'concede'})
        for side in ('a', 'b'):
            self.assertEqual(public_legal_decision_reuse(terminal, side),
                             gpu_rule_entries.seven_public_legal_decision(terminal, side))

    def test_one_legal_generation_instead_of_two(self):
        state = new_game(seed=17)
        calls = Mock(wraps=projection.legal_actions)
        with patch.object(projection, 'legal_actions', calls), \
             patch.object(gpu_rule_entries, 'canonical_legal_actions', calls):
            gpu_rule_entries.seven_public_legal_decision(state, 'a')
            self.assertEqual(calls.call_count, 2)
            calls.reset_mock()
            public_legal_decision_reuse(state, 'a')
            self.assertEqual(calls.call_count, 1)

    def test_nested_actions_do_not_alias_view_or_input(self):
        state = new_game(seed=17)
        before = deepcopy(state)
        actions, view = public_legal_decision_reuse(state, 'a')
        saved_view = deepcopy(view)
        actions[0]['diagnostic_only'] = True
        for action in actions:
            for value in action.values():
                if isinstance(value, list) and value:
                    value.append('diagnostic_only')
                    self.assertEqual(view, saved_view)
                    self.assertEqual(state, before)
                    return
        self.fail('Expected a nested mulligan selection fixture')


if __name__ == '__main__':
    unittest.main()
