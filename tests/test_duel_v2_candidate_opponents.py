"""Candidate opponent schedules are deterministic pair lists, not a tournament."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

from app.modules.card_game.content.duel_v2.catalog import (
    CARDS,
    DECK_SIZE,
    MAX_COPIES,
    PER_CHARACTER,
    PUBLIC_CHARACTER_IDS,
)
from app.modules.card_game.rl.capability import (
    preset_opponent_deck,
    serving_deck_hash,
)
from app.modules.card_game.rl.candidate_opponents import (
    PUBLIC_PRESETS,
    build_opponent_schedule,
)


SERVING_DIR = Path('app/modules/card_game/engine/ai/models')


def _serving_build(preset_id):
    payload = json.loads((SERVING_DIR / f'{preset_id}.json').read_text(encoding='utf-8'))
    return payload['serving_build']


def _assert_legal_public_deck(test, deck):
    test.assertEqual(len(deck['character_ids']), 4)
    test.assertEqual(len(set(deck['character_ids'])), 4)
    test.assertTrue(set(deck['character_ids']) <= set(PUBLIC_CHARACTER_IDS))
    test.assertEqual(len(deck['card_ids']), DECK_SIZE)
    counts = Counter(deck['card_ids'])
    test.assertTrue(all(count <= MAX_COPIES for count in counts.values()))
    per_character = Counter()
    for card_id in deck['card_ids']:
        card = CARDS[card_id]
        test.assertFalse(card.get('derived'))
        test.assertIn(card['character_id'], deck['character_ids'])
        per_character[card['character_id']] += 1
    test.assertEqual(dict(per_character), {character_id: PER_CHARACTER for character_id in deck['character_ids']})


class CandidateOpponentsTest(unittest.TestCase):
    def test_import_stays_pure_python(self):
        self.assertNotIn('torch', Path('app/modules/card_game/rl/candidate_opponents.py').read_text())
        self.assertNotIn('advanced_model', Path('app/modules/card_game/rl/candidate_opponents.py').read_text())
        before = 'torch' in sys.modules
        module = sys.modules['app.modules.card_game.rl.candidate_opponents']
        self.assertFalse(hasattr(module, 'torch'))
        if not before:
            self.assertNotIn('torch', sys.modules)

    def test_repeatable_schedule_and_standalone_copies(self):
        first = build_opponent_schedule(5, seed=20260916)
        second = build_opponent_schedule(5, seed=20260916)
        self.assertEqual(first, second)
        self.assertNotEqual(first, build_opponent_schedule(5, seed=20260917))
        mutated = first[0]['deck']
        mutated['card_ids'] = list(mutated['card_ids'])
        mutated['card_ids'][0] = 'N01'
        third = build_opponent_schedule(5, seed=20260916)
        self.assertEqual(second, third)
        self.assertNotEqual(first[0]['deck']['card_ids'], third[0]['deck']['card_ids'])

    def test_exact_lineup_quotas_legal_cards_and_seeds(self):
        schedule = build_opponent_schedule(8, seed=11, random_fraction=0.25, rule_fraction=0.25)
        lineups = Counter(item['lineup'] for item in schedule)
        self.assertEqual(lineups['starter'], 8)
        self.assertEqual(lineups['weave-rush'], 8)
        self.assertEqual(lineups['public-random'], 5)  # 5/(16+5) approximates 25% with integer pairs.
        self.assertEqual(PUBLIC_PRESETS, ('starter', 'weave-rush'))
        self.assertEqual([item['pair'] for item in schedule], list(range(len(schedule))))
        official = {preset_id: preset_opponent_deck(preset_id) for preset_id in PUBLIC_PRESETS}
        serving = {preset_id: _serving_build(preset_id) for preset_id in PUBLIC_PRESETS}
        for item in schedule:
            _assert_legal_public_deck(self, item['deck'])
            self.assertEqual(item['build_sha256'], serving_deck_hash(item['deck']))
            self.assertEqual(item['seed'] % 2, 0)
            self.assertGreaterEqual(item['seed'], 0)
            self.assertLess(item['seed'], 1 << 31)
            self.assertIn(item['allocation'], ('official', 'serving', 'random', 'history'))
            self.assertIn(item['opponent_model'], ('learner', 'peer', 'rule'))
            if item['lineup'] in PUBLIC_PRESETS:
                self.assertEqual(item['deck']['character_ids'], official[item['lineup']]['character_ids'])
                if item['allocation'] == 'official':
                    self.assertEqual(item['deck']['card_ids'], official[item['lineup']]['card_ids'])
                elif item['allocation'] == 'serving':
                    self.assertEqual(item['deck']['card_ids'], serving[item['lineup']]['card_ids'])
            else:
                self.assertEqual(item['lineup'], 'public-random')
                self.assertEqual(item['allocation'], 'random')

    def test_lineage_and_model_are_independent(self):
        schedule = build_opponent_schedule(
            6,
            seed=3,
            random_fraction=0.2,
            rule_fraction=0.2,
            model_ids=('learner', 'peer'),
        )
        by_lineup = {}
        for item in schedule:
            by_lineup.setdefault(item['lineup'], Counter())[item['opponent_model']] += 1
        lineages = {(item['allocation'], item['opponent_model']) for item in schedule}
        self.assertGreaterEqual(len({allocation for allocation, _ in lineages}), 3)
        self.assertGreaterEqual(len({model for _, model in lineages}), 2)
        self.assertTrue(any(allocation != 'official' for allocation, model in lineages if model != 'rule'))
        self.assertTrue(any(model == 'rule' for allocation, model in lineages if allocation != 'random'))
        models = [item['opponent_model'] for item in schedule]
        self.assertGreater(models.count('learner') + models.count('peer'), models.count('rule'))
        self.assertEqual(models.count('rule'), 3)
        self.assertGreaterEqual(models.count('learner'), 1)
        self.assertGreaterEqual(models.count('peer'), 1)
        starter_models = [item['opponent_model'] for item in schedule if item['lineup'] == 'starter']
        public_models = [item['opponent_model'] for item in schedule if item['lineup'] == 'public-random']
        self.assertTrue(starter_models)
        self.assertTrue(public_models)
        allocations = Counter(
            item['allocation'] for item in schedule if item['lineup'] == 'starter'
        )
        counts = list(allocations.values())
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_random_team_zero_still_varies_cards_and_default_is_twenty_percent(self):
        fixed=build_opponent_schedule(8,seed=19,random_fraction=0)
        self.assertNotIn('public-random',{p['lineup'] for p in fixed})
        for key in PUBLIC_PRESETS:
            self.assertTrue(any(p['lineup']==key and p['allocation']=='random' for p in fixed))
        default=build_opponent_schedule(8,seed=19)
        self.assertEqual(sum(p['lineup']=='public-random' for p in default)/len(default),.2)
        for rate in (float('nan'),float('inf'),-float('inf')):
            for field in ('random_fraction','rule_fraction'):
                with self.assertRaises(ValueError):build_opponent_schedule(8,seed=19,**{field:rate})

    def test_invalid_rates_history_and_empty_models(self):
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, random_fraction=.999999999)
        with self.assertRaises(ValueError):
            build_opponent_schedule(10000, seed=1)
        with self.assertRaises(ValueError):
            build_opponent_schedule(0, seed=1)
        with self.assertRaises(ValueError):
            build_opponent_schedule(True, seed=1)
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, random_fraction=1)
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, rule_fraction=1.0)
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, random_fraction=-0.1)
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, model_ids=())
        with self.assertRaises(ValueError):
            build_opponent_schedule(4, seed=1, model_ids=('rule',))
        starter = deepcopy(preset_opponent_deck('starter'))
        swapped = deepcopy(starter)
        swapped['character_ids'] = list(reversed(swapped['character_ids']))
        with self.assertRaises(ValueError):
            build_opponent_schedule(2, seed=1, historical_builds={'starter': [swapped]})
        nonpublic = deepcopy(starter)
        nonpublic['character_ids'] = ['anhunqu', 'canhong', 'zaowu', 'lingke']
        nonpublic['card_ids'] = ['A01', 'A01', 'A02', 'A02', 'A03', 'A03', 'A08', 'A08',
                                 'C01', 'C01', 'C02', 'C02', 'C03', 'C03', 'C08', 'C08',
                                 'S01', 'S01', 'S03', 'S03', 'S05', 'S05', 'S07', 'S07',
                                 'K01', 'K01', 'K03', 'K03', 'K05', 'K05', 'K08', 'K08']
        with self.assertRaises(ValueError):
            build_opponent_schedule(2, seed=1, historical_builds={'starter': [nonpublic]})
        illegal = deepcopy(starter)
        illegal['card_ids'] = ['N02'] * 8 + illegal['card_ids'][8:]
        with self.assertRaises(ValueError):
            build_opponent_schedule(2, seed=1, historical_builds={'starter': [illegal]})

    def test_diversity_history_and_current_serving_z07(self):
        starter_history = deepcopy(preset_opponent_deck('starter'))
        starter_history['id'] = 'starter-history'
        starter_history['card_ids'] = [
            'N01' if card_id == 'N02' else card_id
            for card_id in starter_history['card_ids']
        ]
        schedule = build_opponent_schedule(
            9,
            seed=42,
            random_fraction=0.2,
            historical_builds={'starter': [starter_history]},
        )
        starter_pairs = [item for item in schedule if item['lineup'] == 'starter']
        allocations = Counter(item['allocation'] for item in starter_pairs)
        self.assertEqual(set(allocations), {'official', 'serving', 'random', 'history'})
        self.assertLessEqual(max(allocations.values()) - min(allocations.values()), 1)
        serving_cards = [item['deck']['card_ids'] for item in starter_pairs if item['allocation'] == 'serving']
        self.assertTrue(serving_cards)
        current = _serving_build('starter')
        for card_ids in serving_cards:
            self.assertEqual(card_ids, current['card_ids'])
            self.assertEqual(card_ids.count('Z07'), 2)
            self.assertEqual(card_ids.count('Z06'), 0)
        official_cards = preset_opponent_deck('starter')['card_ids']
        self.assertNotEqual(current['card_ids'], official_cards)
        history_cards = [item['deck']['card_ids'] for item in starter_pairs if item['allocation'] == 'history']
        self.assertTrue(history_cards)
        self.assertEqual(history_cards[0], starter_history['card_ids'])
        random_sets = {
            tuple(item['deck']['card_ids'])
            for item in starter_pairs
            if item['allocation'] == 'random'
        }
        self.assertGreaterEqual(len(random_sets), 2)
        public_teams = {
            tuple(item['deck']['character_ids'])
            for item in schedule
            if item['lineup'] == 'public-random'
        }
        self.assertTrue(public_teams)
        self.assertTrue(all(set(team) <= set(PUBLIC_CHARACTER_IDS) for team in public_teams))
        weave_serving = [item for item in schedule if item['lineup'] == 'weave-rush' and item['allocation'] == 'serving']
        self.assertTrue(weave_serving)
        self.assertEqual(weave_serving[0]['deck']['card_ids'], _serving_build('weave-rush')['card_ids'])


if __name__ == '__main__':
    unittest.main()
