"""Exact numerical battle previews for VisibleEngineRulePolicy. Not a 5x claim."""
from copy import deepcopy
from unittest.mock import patch
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game, observe
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.flow import apply_owned_action
from app.modules.card_game.engine.duel_v2.escalation import energy_cap
from app.modules.card_game.engine.duel_v2.projection import (
    RULE_PREVIEW_SCORE_FIELDS, prepared_preview_sandbox, preview, rule_preview_scores,
    rule_battle_preview,
)
from app.modules.card_game.engine.duel_v2.state import card_instance, hero
from app.modules.card_game.rl.covered_rollout import rule_decision
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
from app.modules.card_game.rl.league_rollout import clean


def _scores(value):
    return rule_preview_scores(value)


def _combat_actions(view, side):
    cards = {card['instance_id']: card for card in view['sides'][side]['hand']}
    found = []
    for entry in view['legal_actions']:
        action = entry['action']
        if action['type'] == 'attack' or (
                action['type'] == 'play_card' and cards.get(action.get('card_id'), {}).get('type') == 'battle'):
            found.append((action, entry.get('preview') or {}))
    return found


def _hold(state, card_id, side='a'):
    card = card_instance(state, card_id, side)
    state['sides'][side]['hand'] = [card]
    return card


class RulePreviewFastTest(unittest.TestCase):
    def game(self, seed=11, decks=None, skip_mulligan=True):
        selected = decks or STARTER_DECKS[0]
        return new_game(seed=seed, first_side='a', skip_mulligan=skip_mulligan,
                        decks={'a': selected, 'b': selected})

    def assert_untouched(self, state, original, action=None, action_original=None):
        self.assertEqual(state, original)
        if action is not None:
            self.assertEqual(action, action_original)

    def assert_score_match(self, state, side, action):
        original = deepcopy(state)
        action_original = deepcopy(action)
        full = preview(state, side, action)
        with prepared_preview_sandbox(state, side):
            fast = preview(state, side, action)
        self.assertEqual(fast, full)
        self.assertEqual(_scores(fast), _scores(full))
        with prepared_preview_sandbox(state, side):
            self.assertEqual(rule_battle_preview(state, side, action), _scores(full))
        self.assert_untouched(state, original, action, action_original)
        return full

    def test_starter7_scores_and_selected_action_match_full_preview(self):
        policy = VisibleEngineRulePolicy()
        checked = 0
        for index, deck in enumerate(STARTER_DECKS):
            other = STARTER_DECKS[(index + 1) % len(STARTER_DECKS)]
            state = new_game(seed=40 + index, first_side='a', skip_mulligan=True,
                             decks={'a': deck, 'b': other})
            for _ in range(8):
                if state['phase'] == 'finished':
                    break
                actor = acting_side(state)
                original = deepcopy(state)
                reference = observe(state, actor, include_previews=True)
                ref_legal = [entry['action'] for entry in reference['legal_actions']
                             if entry['action']['type'] != 'concede']
                legal, chosen = rule_decision(state, actor)
                self.assertEqual(legal, ref_legal)
                self.assertEqual(chosen, policy(reference, tuple(ref_legal)))
                self.assertEqual(legal[chosen], ref_legal[chosen])
                for action, full_preview in _combat_actions(reference, actor):
                    with prepared_preview_sandbox(state, actor):
                        fast = preview(state, actor, action)
                    self.assertEqual(_scores(fast), _scores(full_preview))
                    self.assertEqual(fast, full_preview)
                    with prepared_preview_sandbox(state, actor):
                        self.assertEqual(rule_battle_preview(state, actor, action), _scores(full_preview))
                    checked += 1
                self.assertEqual(state, original)
                state = clean(apply_action(state, actor, legal[chosen]))
        self.assertGreater(checked, 20)

    def test_teacher_combat_does_not_build_full_display_preview(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        view = observe(state, 'a', include_previews=True)
        expected = [entry['action'] for entry in view['legal_actions']
                    if entry['action']['type'] != 'concede']
        chosen = VisibleEngineRulePolicy()(view, tuple(expected))
        with patch('app.modules.card_game.engine.duel_v2.projection.preview',
                   side_effect=AssertionError('display preview in teacher')):
            legal, actual = rule_decision(state, 'a')
        self.assertEqual((legal, actual), (expected, chosen))

    def test_combat_battle_response_harmony_terminal_marks_and_hidden(self):
        cases = []

        combat = self.game()
        combat['sides']['b']['front'] = 'zero'
        cases.append((combat, {'type': 'attack', 'character_id': 'zero'}, 'combat'))

        battle = self.game()
        battle['sides']['a']['ap'] = 5
        battle['sides']['b']['front'] = 'zero'
        card = _hold(battle, 'N02')
        cases.append((battle, {'type': 'play_card', 'card_id': card['instance_id']}, 'battle'))

        hidden = self.game()
        hidden['sides']['a']['front'] = 'zero'
        hidden['sides']['b']['front'] = 'zero'
        hidden['sides']['b']['ap'] = 1
        hero(hidden, 'b', 'nanali')['base_attack'] = 9
        secret = card_instance(hidden, 'N02', 'b')
        hidden['sides']['b']['hand'] = [secret]
        cases.append((hidden, {'type': 'attack', 'character_id': 'zero'}, 'hidden'))

        revealed = deepcopy(hidden)
        revealed['sides']['b']['revealed_ids'] = [secret['instance_id']]
        cases.append((revealed, {'type': 'attack', 'character_id': 'zero'}, 'response'))

        harmony = self.game()
        harmony['sides']['a']['front'] = None
        harmony['sides']['a']['last_front'] = 'zero'
        hero(harmony, 'a', 'zero')['harmony'] = 2
        harmony['sides']['b']['front'] = 'nanali'
        cases.append((harmony, {'type': 'attack', 'character_id': 'nanali'}, 'harmony'))

        terminal = self.game()
        terminal['sides']['b']['front'] = 'zero'
        owner = hero(terminal, 'a', 'nanali')
        owner['energy'] = energy_cap(terminal, owner)
        terminal = apply_action(terminal, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        cases.append((terminal, {'type': 'attack', 'character_id': 'nanali'}, 'terminal'))

        marks = self.game()
        marks['sides']['a']['front'] = 'zero'
        marks['sides']['b']['front'] = 'nanali'
        marked = card_instance(marks, 'N01', 'b')
        marked['jingu_mark'] = {'delta': 1, 'source_entity_id': hero(marks, 'b', 'nanali')['entity_id']}
        marks['sides']['b']['hand'] = [marked]
        hidden_marks = deepcopy(marks)
        cases.append((hidden_marks, {'type': 'attack', 'character_id': 'zero'}, 'marks-hidden'))
        marks['sides']['b']['revealed_ids'] = [marked['instance_id']]
        cases.append((marks, {'type': 'attack', 'character_id': 'zero'}, 'marks-known'))

        flower = self.game()
        flower['sides']['a']['front'] = 'jiuyuan'
        flower['sides']['a']['extra_flower'] = True
        flower['sides']['b']['front'] = 'zero'
        cases.append((flower, {'type': 'attack', 'character_id': 'jiuyuan'}, 'flower'))

        seen = set()
        for state, action, label in cases:
            if action not in legal_actions(state, 'a'):
                self.fail(f'{label} fixture is not legal')
            full = self.assert_score_match(state, 'a', action)
            seen.add(label)
            original = deepcopy(state)
            legal, chosen = rule_decision(state, 'a')
            reference = observe(state, 'a', include_previews=True)
            ref_legal = [entry['action'] for entry in reference['legal_actions']
                         if entry['action']['type'] != 'concede']
            self.assertEqual(legal[chosen], ref_legal[VisibleEngineRulePolicy()(reference, tuple(ref_legal))])
            self.assertEqual(state, original)
            if label in ('combat', 'battle', 'response', 'harmony', 'terminal', 'flower'):
                self.assertTrue(any(key in full for key in RULE_PREVIEW_SCORE_FIELDS) or label == 'harmony')
        self.assertGreaterEqual(len(seen), 8)
        hidden_preview = preview(hidden, 'a', {'type': 'attack', 'character_id': 'zero'})
        revealed_preview = preview(revealed, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertNotEqual(_scores(hidden_preview), _scores(revealed_preview))
        hidden_mark_preview = preview(hidden_marks, 'a', {'type': 'attack', 'character_id': 'zero'})
        known_mark_preview = preview(marks, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertIn('jingu_mark', hidden_marks['sides']['b']['hand'][0])
        self.assertEqual(hidden_mark_preview.get('counter'), known_mark_preview.get('counter'))

    def test_mulligan_and_choose_skip_expensive_preview(self):
        mulligan = self.game(skip_mulligan=False)
        with patch('app.modules.card_game.engine.duel_v2.projection.preview', side_effect=AssertionError('preview')):
            legal, chosen = rule_decision(mulligan, 'a')
        self.assertEqual(legal[chosen]['type'], 'mulligan')

        choice = self.game()
        card = _hold(choice, 'N01')
        choice['phase'] = 'choice'
        choice['pending_choice'] = {
            'side': 'a', 'kind': 'inspect_top', 'prompt': '选择', 'cards': [card],
        }
        with patch('app.modules.card_game.engine.duel_v2.projection.preview', side_effect=AssertionError('preview')):
            legal, chosen = rule_decision(choice, 'a')
        self.assertEqual(legal[chosen]['type'], 'choose')

    def test_public_apply_action_independence_and_error_parity(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        action = {'type': 'attack', 'character_id': 'zero'}
        original = deepcopy(state)
        action_original = deepcopy(action)
        first = apply_action(state, 'a', action)
        second = apply_action(state, 'a', action)
        self.assertEqual(first, second)
        self.assert_untouched(state, original, action, action_original)
        owned = apply_owned_action(clone_state(state), 'a', action)
        self.assertEqual(owned, first)
        self.assert_untouched(state, original, action, action_original)

        illegal = {'type': 'attack', 'character_id': 'missing'}
        with self.assertRaises(ValueError) as public_error:
            apply_action(state, 'a', illegal)
        owned_state = clone_state(state)
        with self.assertRaises(ValueError) as owned_error:
            apply_owned_action(owned_state, 'a', illegal)
        self.assertEqual(str(public_error.exception), str(owned_error.exception))
        self.assertEqual(state, original)
        with self.assertRaises(ValueError):
            apply_action(state, 'z', action)
        with self.assertRaises(ValueError):
            apply_action(state, 'a', ['attack'])
        self.assertEqual(state, original)


if __name__ == '__main__':
    unittest.main()
