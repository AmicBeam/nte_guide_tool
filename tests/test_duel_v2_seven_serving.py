"""Real adopted exports retain research status and exact CPU inference semantics."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests.test_solo_room_flow import RoomFlowTestCase

ROOT = Path(__file__).resolve().parents[1] / 'app/modules/card_game/engine/ai/models/seven-20261024'
KEYS = ('starter', 'quick-rush', 'zhenhong', 'murk', 'collapse')


class SevenServingTest(unittest.TestCase):
    def test_exact_numeric_parity_and_real_search(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.engine.duel_v2 import new_game, observe, legal_actions, apply_action
        from app.modules.card_game.rl import seven_runtime, seven_lineup, recovery_search
        for i, key in enumerate(KEYS):
            with self.subTest(key=key):
                served = ai.load_model(key)
                research = seven_runtime.SevenModel(ROOT, key)
                self.assertFalse(served.manifest['validation']['approved'])
                self.assertTrue(served.serving_authorized)
                self.assertEqual(served.version, research.version)
                state = new_game(seed=24000+i, first_side='b',
                                 decks={'a': served.serving_deck, 'b': served.serving_deck})
                # Both learned mulligan roots and ordinary playing roots.
                for side in ('a', 'b'):
                    actions = [a for a in legal_actions(state, side) if a['type'] != 'concede']
                    if not actions:
                        continue
                    view = observe(state, side, include_previews=False)
                    x, c = seven_lineup.encode(view, actions)
                    np.testing.assert_array_equal(served.scores(x, c), research.scores(x, c))
                    self.assertEqual(served.select_public_action(view, actions), int(research.scores(x, c).argmax()))
                    state = apply_action(state, side, actions[served.select_public_action(view, actions)])
                state = new_game(seed=24000+i, first_side='b', skip_mulligan=True,
                                 decks={'a': served.serving_deck, 'b': served.serving_deck})
                state['ai_profile'] = dict(deck_id=key, version=served.version,
                                          build_sha256=served.serving_build_sha256)
                before = deepcopy(state)
                results = []
                search = recovery_search.search
                def record(*args, **kwargs):
                    self.assertIs(kwargs['runtime'], seven_runtime)
                    result = search(*args, **kwargs)
                    results.append(result)
                    return result
                with patch.object(recovery_search, 'search', side_effect=record):
                    selected = ai.choose_action(state, 'b')
                self.assertEqual(state, before)
                self.assertIn(selected, legal_actions(state, 'b'))
                self.assertEqual(len(results), 1)
                self.assertGreater(results[0]['simulations'], 0)
                self.assertLessEqual(results[0]['simulations'], 32)
                self.assertEqual(selected, results[0]['actions'][results[0]['search_choice']])

    def test_authorization_and_cache_fail_closed(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.engine.ai.seven_model import SevenServingModel
        from app.modules.card_game.rl import seven_lineup
        from app.errors import RuleValidationError
        for field in ('runtime_rule_hash', 'quality_approved', 'models', 'request'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / ROOT.name
                root.mkdir()
                for name in ('serving.json', 'murk.json', 'murk.npz'):
                    shutil.copyfile(ROOT / name, root / name)
                with patch.object(ai.config, 'DUEL_AI_MODEL_DIR', tmp):
                    ai.load_model('murk')
                    binding = json.loads((root / 'serving.json').read_text())
                    binding[field] = True if field == 'quality_approved' else {} if field == 'models' else ''
                    (root / 'serving.json').write_text(json.dumps(binding))
                    with self.assertRaises(RuleValidationError):
                        ai.load_model('murk')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('serving.json', 'murk.json', 'murk.npz'):
                shutil.copyfile(ROOT / name, root / name)
            with (root / 'murk.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'authorization'):
                SevenServingModel(root, 'murk')
        original = ai.load_model('murk')
        with patch.object(seven_lineup, 'identity', return_value='drift'):
            with self.assertRaises(RuleValidationError):
                ai.load_model('murk')
        self.assertIs(ai.load_model('murk'), original)

    def test_preserved_weave_and_pending_lingke(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        self.assertEqual(ai.load_model('weave-rush').version,
                         '29a25b318f0319bd96bb3eaa27e21df669cacbb43c1bf0f9e07e1e7e783833d1')
        options = {o['id']: o for o in ai.pending_strategy_options()}
        self.assertFalse(options['collapse']['pending'])
        self.assertTrue(options['lingke']['pending'])
        with self.assertRaises(ai.RuleValidationError):
            ai.load_model('lingke')

    def test_fresh_process_cpu_only(self):
        subprocess.run([sys.executable, '-c',
            'from app.modules.card_game.engine.ai.advanced_model import load_model; import sys; '
            f'[load_model(k) for k in {KEYS!r}]; assert "torch" not in sys.modules'],
            check=True, capture_output=True)


class SevenServingRoomTest(RoomFlowTestCase):
    def test_all_five_adoptions_bind_actual_weights_and_build(self):
        import importlib
        ai = importlib.import_module('app.modules.card_game.engine.ai.advanced_model')
        repo = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        token = self._issue_login_and_get_token('seven-adoption-test')
        for key in KEYS:
            with self.subTest(key=key):
                result = self._post('/api/duel-v2/start', {'mode': 'advanced', 'ai_deck': key}, token=token)
                game = storage.load(repo.room_by_code(result['room']['room_code']).id)['game']
                model = ai.load_model(key)
                self.assertEqual(game['ai_profile']['version'], model.version)
                self.assertEqual(game['ai_profile']['build_sha256'], model.serving_build_sha256)
                self.assertTrue(game['ai_profile']['experimental'])
                self._post('/api/duel-v2/leave', token=token)
