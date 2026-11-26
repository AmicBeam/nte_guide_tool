"""User-selected public presets and future advanced-AI starts, without training."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.errors import RuleValidationError
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.ai import advanced_model
from app.modules.card_game.engine.application import v2_service
from app.modules.card_game.rl import next_training


STARTS = {
    'collapse': (['heiyu', 'dafutier', 'canhong', 'anhunqu'],
                 'A04 A04 A05 A05 A06 A06 A07 A08 C02 C02 C03 C03 C05 C05 C08 C08 '
                 'F03 F04 F06 F06 F07 F07 F08 F08 P01 P01 P02 P03 P05 P06 P08 P08'),
    'lingke': (['canhong', 'bohe', 'lingke', 'zaowu'],
               'C01 C01 C02 C02 C03 C03 C08 C08 S01 S01 S03 S03 S05 S05 S08 S08 '
               'K01 K01 K02 K02 K03 K05 K05 K08 M01 M01 M03 M03 M04 M04 M08 M08'),
}


class PresetStartsTest(unittest.TestCase):
    def test_official_presets_match_the_user_selected_teams_and_card_counts(self):
        presets = {d['id']: d for d in STARTER_DECKS}
        for key, (order, cards) in STARTS.items():
            with self.subTest(key=key):
                self.assertEqual(presets[key]['character_ids'], order)
                self.assertEqual(Counter(presets[key]['card_ids']), Counter(cards.split()))
                picked = v2_service._solo_ai_deck(SimpleNamespace(shaft_test_whitelisted=False), key)
                self.assertEqual(picked['character_ids'], order)
                self.assertEqual(Counter(picked['card_ids']), Counter(cards.split()))

    def test_next_round_has_seven_teams_with_exact_frozen_start_identities(self):
        request = next_training.plan(datetime(2026, 10, 8, 18, tzinfo=timezone.utc))
        self.assertEqual(request['roster'], ['starter', 'weave-rush', 'quick-rush', 'zhenhong',
                                           'murk', 'collapse', 'lingke'])
        self.assertEqual(request['learning_keys'], request['roster'])
        self.assertEqual(request['evaluation_shape'], [7, 7])
        self.assertEqual(len(request['cross_learning_schedule']), 42)
        self.assertEqual(len(request['mirror_learning_schedule']), 14)
        self.assertEqual(len(request['evaluation_cells']), 49)
        self.assertEqual(request['new_games_per_complete_round'], 56)
        self.assertFalse(request['automatically_start_training'])
        self.assertFalse(request['quality_approved'])
        self.assertFalse(request['preflight_passed'])
        self.assertEqual(request['starting_build_source']['quality_status'], 'manual_untrained_start')
        for key, (order, cards) in STARTS.items():
            build = request['builds'][key]
            self.assertEqual(build['character_ids'], order)
            self.assertEqual(Counter(build['card_ids']), Counter(cards.split()))
            self.assertEqual(build['card_ids'], sorted(build['card_ids']))
            self.assertEqual(request['build_identities'][key],
                             request['starting_build_source']['build_identities'][key])

    def test_start_drift_or_matrix_mismatch_is_rejected_but_card_row_order_is_not_identity(self):
        original = json.loads(next_training.PROFILE_PATH.read_text())
        for kind in ('cards', 'official', 'matrix', 'missing_identity', 'missing_starts', 'row_order'):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                spec = deepcopy(original)
                presets = deepcopy(STARTER_DECKS)
                if kind == 'cards':
                    cards = spec['starting_builds']['collapse']['card_ids']
                    cards[cards.index('F03')] = 'F04'
                elif kind == 'official':
                    cards = next(d for d in presets if d['id'] == 'collapse')['card_ids']
                    cards[cards.index('F03')] = 'F04'
                elif kind == 'matrix':
                    spec['evaluation_shape'] = [6, 6]
                elif kind == 'missing_identity':
                    spec['starting_build_source']['build_identities'].pop('lingke')
                elif kind == 'missing_starts':
                    spec.pop('starting_builds')
                    spec.pop('starting_build_source')
                else:
                    spec['starting_builds']['collapse']['card_ids'].reverse()
                path = Path(tmp) / 'request.json'
                path.write_text(json.dumps(spec))
                with patch.object(next_training, 'PROFILE_PATH', path), \
                     patch.object(next_training, 'STARTER_DECKS', presets):
                    if kind == 'row_order':
                        self.assertEqual(next_training.plan()['build_identities']['collapse'],
                                         original['starting_build_source']['build_identities']['collapse'])
                    else:
                        with self.assertRaises(ValueError):
                            next_training.plan()

    def test_existing_personal_lingke_build_is_not_overwritten_or_duplicated(self):
        personal = deepcopy(next(d for d in STARTER_DECKS if d['id'] == 'lingke'))
        personal['id'] = 'personal'
        personal['card_ids'][personal['card_ids'].index('K03')] = 'K08'
        original = deepcopy(personal)
        store = dict(active_id='personal', builds=[personal], starter_granted=True,
                     granted_preset_ids=['starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse'])
        self.assertTrue(v2_service._grant_starter_decks(store, SimpleNamespace(shaft_test_whitelisted=False)))
        self.assertEqual(store['builds'], [original])
        self.assertEqual(store['active_id'], 'personal')
        self.assertIn('lingke', store['granted_preset_ids'])
        self.assertFalse(v2_service._grant_starter_decks(store, SimpleNamespace(shaft_test_whitelisted=False)))

    def test_new_advanced_slots_stay_pending_without_their_own_models(self):
        with TemporaryDirectory() as tmp, patch.object(advanced_model.config, 'DUEL_AI_MODEL_DIR', Path(tmp)):
            self.assertEqual({r['id'] for r in advanced_model.pending_strategy_options()}, {'collapse', 'lingke'})
            self.assertTrue(all(r['pending'] for r in advanced_model.pending_strategy_options()))
            for key in STARTS:
                with self.subTest(key=key), self.assertRaisesRegex(RuleValidationError, '尚待训练'):
                    advanced_model.load_model(key)


if __name__ == '__main__':
    unittest.main()
