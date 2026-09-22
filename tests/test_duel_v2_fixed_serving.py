"""Authorization binding and real CPU policy dispatch for the three bundled models."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from app.modules.card_game.engine.ai.fixed_model import FixedServingModel
from app.modules.card_game.engine.ai import advanced_model as ai
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side

ROOT = Path(__file__).resolve().parents[1] / 'app/modules/card_game/engine/ai/models'
KEYS = ('starter', 'weave-rush', 'quick-rush')


class FixedServingTest(unittest.TestCase):
    def test_three_complete_cpu_games_with_learned_opening(self):
        for i, key in enumerate(KEYS):
            model = ai.load_model(key)
            self.assertTrue(model.serving_authorized)
            self.assertFalse(model.manifest['validation']['approved'])
            state = new_game(seed=301+i, decks={'a': model.serving_deck, 'b': model.serving_deck})
            state['ai_profile'] = dict(deck_id=key, version=model.version, build_sha256=model.serving_build_sha256)
            for _ in range(600):
                if state['phase'] == 'finished': break
                side = acting_side(state); action = ai.choose_action(state, side)
                state = apply_action(state, side, action); state['events'] = []; state.pop('_public_board', None)
            self.assertEqual(state['phase'], 'finished', key)

    def test_missing_or_misbound_authorization_rejected(self):
        for field in ('model_sha256', 'build_sha256', 'rule_hash', 'request'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                shutil.copy2(ROOT / 'starter.npz', path / 'starter.npz')
                manifest = json.loads((ROOT / 'starter.json').read_text())
                manifest['serving_authorization'][field] = ''
                (path / 'starter.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    FixedServingModel(path, 'starter')

    def test_weight_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for ext in ('json', 'npz'): shutil.copy2(ROOT / f'starter.{ext}', path / f'starter.{ext}')
            with (path / 'starter.npz').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                FixedServingModel(path, 'starter')

    def test_excluded_team_not_registered_or_installed(self):
        self.assertNotIn('zhenhong', ai.PRESETS)
        self.assertNotIn('midrange', ai.PRESETS)
        self.assertFalse((ROOT / 'midrange.json').exists())
        self.assertFalse((ROOT / 'midrange.npz').exists())
        self.assertFalse((ROOT / 'zhenhong.json').exists())
        self.assertEqual(ai.load_model('quick-rush').serving_deck['character_ids'], ['xiaozhi','zero','bohe','jiuyuan'])

    def test_test_only_human_build_remains_rejected(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        build = next(d for d in STARTER_DECKS if d['id'] == 'zhenhong')
        with self.assertRaisesRegex(ValueError, '公开角色'):
            ai.load_model('starter').validate_human(build)
