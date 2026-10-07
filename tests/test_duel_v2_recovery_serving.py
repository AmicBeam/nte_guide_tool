"""Real release bytes, explicit binding, information-set search and HTTP dispatch."""
from pathlib import Path
from copy import deepcopy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests.test_solo_room_flow import RoomFlowTestCase

ROOT = Path(__file__).resolve().parents[1] / 'app/modules/card_game/engine/ai/models/recovery-20261007'
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')


class RecoveryServingTest(unittest.TestCase):
    def test_release_identity_and_real_search_for_every_team(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.engine.duel_v2 import new_game, legal_actions
        from app.modules.card_game.rl import recovery_search
        from app.modules.card_game.engine.ai.recovery_model import SOURCE_RULE_HASH
        binding = json.loads((ROOT / 'serving.json').read_text())
        for i, key in enumerate(KEYS):
            with self.subTest(key=key):
                model = ai.load_model(key)
                self.assertEqual(model.version, binding['models'][key]['model_sha256'])
                self.assertEqual(model.manifest['rule_hash'], SOURCE_RULE_HASH)
                self.assertFalse(model.manifest['validation']['approved'])
                self.assertTrue(model.serving_authorized)
                state = new_game(seed=310+i, first_side='b', skip_mulligan=True,
                    decks={'a': ai.load_model('murk').serving_deck, 'b': model.serving_deck})
                state['ai_profile'] = dict(deck_id=key, version=model.version,
                    build_sha256=model.serving_build_sha256)
                recorded = []
                actual_search = recovery_search.search
                def record(*args, **kwargs):
                    result = actual_search(*args, **kwargs)
                    recorded.append(result)
                    return result
                before = deepcopy(state)
                with patch.object(recovery_search, 'search', side_effect=record):
                    selected = ai.choose_action(state, 'b')
                self.assertIn(selected, legal_actions(state, 'b'))
                self.assertEqual(state, before)
                self.assertEqual(len(recorded), 1)
                result = recorded[0]
                self.assertGreater(result['simulations'], 0)
                self.assertLessEqual(result['simulations'], 32)
                self.assertEqual(result['policy_source'], 'gumbel_q_natural_wdl')
                self.assertEqual(result['teacher_mode'], 'network')
                self.assertEqual(selected, result['actions'][result['search_choice']])
        self.assertEqual(ai.load_model('murk').version,
                         '235b19b3fc37ee1100aa67266ef0d1c452823486263fbdf73da5b830610d0694')

    def test_binding_mutation_and_weights_tampering_rejected(self):
        from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel
        for field in ('runtime_rule_hash', 'source_rule_hash', 'request', 'models'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('serving.json', 'murk.json', 'murk.npz'):
                    shutil.copyfile(ROOT / name, root / name)
                binding = json.loads((root / 'serving.json').read_text())
                binding[field] = {} if field == 'models' else ''
                (root / 'serving.json').write_text(json.dumps(binding))
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    RecoveryServingModel(root, 'murk')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('serving.json', 'murk.json', 'murk.npz'):
                shutil.copyfile(ROOT / name, root / name)
            with (root / 'murk.npz').open('ab') as f:
                f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'authorization'):
                RecoveryServingModel(root, 'murk')

    def test_cache_rechecks_serving_binding(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.errors import RuleValidationError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); release = root / ROOT.name; release.mkdir()
            for name in ('serving.json', 'murk.json', 'murk.npz'):
                shutil.copyfile(ROOT / name, release / name)
            with patch.object(ai.config, 'DUEL_AI_MODEL_DIR', str(root)):
                ai.load_model('murk')
                binding = json.loads((release / 'serving.json').read_text())
                binding['models']['murk']['build_sha256'] = 'changed'
                (release / 'serving.json').write_text(json.dumps(binding))
                with self.assertRaises(RuleValidationError):
                    ai.load_model('murk')

    def test_fresh_process_serving_imports_no_torch(self):
        script = "from app.modules.card_game.engine.ai.advanced_model import load_model; import sys; [load_model(k) for k in " + repr(KEYS) + "]; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)

    def test_stale_room_version_and_build_rejected(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.errors import RuleValidationError
        model = ai.load_model('murk'); state = new_game(seed=5)
        state['ai_profile'] = dict(deck_id='murk', version='old')
        with self.assertRaisesRegex(RuleValidationError, '版本已变更'):
            ai.choose_action(state)
        state['ai_profile'].update(version=model.version, build_sha256='changed')
        with self.assertRaisesRegex(RuleValidationError, '构筑版本'):
            ai.choose_action(state)


class RecoveryServingRoomTest(RoomFlowTestCase):
    def test_real_room_binds_new_weights_and_searches_one_operation(self):
        import importlib
        ai = importlib.import_module('app.modules.card_game.engine.ai.advanced_model')
        repository = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        search_module = importlib.import_module('app.modules.card_game.rl.recovery_search')
        token = self._issue_login_and_get_token('recovery-serving')
        recorded = []; actual_search = search_module.search
        def record(*args, **kwargs):
            result = actual_search(*args, **kwargs); recorded.append(result); return result
        result = self._post('/api/duel-v2/start', {'mode': 'advanced', 'ai_deck': 'murk'}, token=token)
        code = result['room']['room_code']
        state = storage.load(repository.room_by_code(code).id)['game']
        self.assertEqual(state['ai_profile']['version'], ai.load_model('murk').version)
        self.assertTrue(state['ai_profile']['experimental'])
        with patch.object(search_module, 'search', side_effect=record):
            for i in range(6):
                request = dict(room_code=code, expected_version=result['game']['version'], request_id=f'real-{i}')
                if result['room']['ai_pending']:
                    result = self._post('/api/duel-v2/ai-step', request, token=token)
                else:
                    actions = [item['action'] for item in result['game']['legal_actions']]
                    action = next(a for a in actions if a['type'] in ('mulligan', 'end_turn') and not a.get('card_ids'))
                    result = self._post('/api/duel-v2/action', dict(request, action=action), token=token)
                if recorded:
                    self.assertGreater(recorded[0]['simulations'], 0)
                    break
        self.assertTrue(recorded, 'HTTP bot operation must execute real recovery search')
        storage.flush(5)
