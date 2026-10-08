"""Asymmetric training contracts; no optimizer, GPU, or account data."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.capability import (
    sample_public_deck, training_matchup, preset_opponent_deck,
    capability_ready_manifest, resolve_model_capability,
)


class PublicTrainingTest(unittest.TestCase):
    def test_fixed_learner_and_all_public_opponent_cards(self):
        from app.modules.card_game.content.duel_v2.catalog import PUBLIC_CHARACTER_IDS
        seen_cards, seen_teams = set(), set()
        fixed = preset_opponent_deck('weave-rush')
        for seed in range(100):
            decks = training_matchup(seed, learner_deck=fixed, learning_side='b', sample_public=True)
            self.assertEqual(decks['b'], fixed)
            self.assertEqual(decks['a'], sample_public_deck(seed))
            self.assertTrue(set(decks['a']['character_ids']) <= PUBLIC_CHARACTER_IDS)
            seen_cards.update(decks['a']['card_ids'])
            seen_teams.add(frozenset(decks['a']['character_ids']))
        self.assertEqual(len(seen_cards), 48)
        self.assertEqual(len(seen_teams), 15)
        # Default must retain weave-rush mirrors, not silently switch the opponent.
        self.assertEqual(training_matchup(0, learner_deck=fixed)['a'], fixed)
        self.assertEqual(training_matchup(0, learner_deck=fixed)['b'], fixed)

    def test_evaluation_pairs_same_opponent_and_swaps_first_player(self):
        from app.modules.card_game.rl.budget import TrainBudget
        from app.modules.card_game.rl.evaluate import evaluate_strategies
        calls = []
        def match(*args, **kwargs):
            calls.append(kwargs)
            return {'winner':'draw', 'terminated':True, 'truncated':False}
        with TemporaryDirectory() as directory, patch('app.modules.card_game.rl.evaluate._play_match', side_effect=match):
            evaluate_strategies(Path(directory), TrainBudget(max_raw_steps=100, max_games=10, max_seconds=10),
                deck=preset_opponent_deck('starter'), sample_opponent=True,
                matches_per_seat=2, log=lambda _: None)
        self.assertEqual([c['learning_side'] for c in calls], ['a','a','b','b'])
        self.assertEqual([c['seed'] for c in calls[:2]], [c['seed'] for c in calls[2:]])
        self.assertTrue(all(c['sample_opponent'] for c in calls))
        self.assertTrue(all(c['deck']['id'] == 'starter' for c in calls))

    def test_capability_requires_complete_declaration(self):
        complete = capability_ready_manifest('starter')
        self.assertEqual(resolve_model_capability({'capability':complete}, 'starter'), complete)
        for key in ('validated_distribution', 'trained_distribution', 'encoder_kit_cards', 'bot_preset'):
            incomplete = deepcopy(complete)
            del incomplete[key]
            with self.assertRaises(ValueError):
                resolve_model_capability({'capability':incomplete}, 'starter')
