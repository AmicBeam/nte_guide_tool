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
    def test_short_trained_zhenhong_keeps_user_selected_yi_build(self):
        from collections import Counter
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
        from app.modules.card_game.engine.ai.advanced_model import load_model
        model = load_model('zhenhong')
        preset = next(d for d in STARTER_DECKS if d['id'] == 'zhenhong')
        self.assertEqual(model.serving_deck, validate_deck(preset))
        self.assertEqual(Counter(c for c in model.serving_deck['card_ids'] if c.startswith('I')),
                         Counter({'I01': 2, 'I03': 2, 'I04': 2, 'I08': 2}))
        original = model.manifest['build']
        self.assertEqual(Counter(c for c in original['card_ids'] if c.startswith('I')),
                         Counter({'I01': 2, 'I03': 2, 'I04': 2, 'I08': 2}))
        self.assertEqual([c for c in original['card_ids'] if not c.startswith('I')],
                         [c for c in model.serving_deck['card_ids'] if not c.startswith('I')])
        self.assertEqual(model.serving_build_sha256, model.manifest['build_sha256'])
        self.assertEqual(model.version,
                         'a3209030a44ed5d270f6974a12bf1ec681837e67cad0ce66440a70401bd5f07a')

    def test_per_model_training_source_requires_explicit_provenance(self):
        from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel
        for mutation in ('source_rule_hash', 'source_request', 'source_report',
                         'all_source_fields', 'invalid_rule_hash', 'wrong_rule_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('serving.json', 'zhenhong.json', 'zhenhong.npz'):
                    shutil.copyfile(ROOT / name, root / name)
                binding = json.loads((root / 'serving.json').read_text())
                pin = binding['models']['zhenhong']
                if mutation == 'all_source_fields':
                    for field in ('source_rule_hash', 'source_request', 'source_report'):
                        pin.pop(field)
                elif mutation in ('invalid_rule_hash', 'wrong_rule_hash'):
                    pin['source_rule_hash'] = 'not-a-hash' if mutation == 'invalid_rule_hash' else '0' * 64
                else:
                    pin.pop(mutation)
                (root / 'serving.json').write_text(json.dumps(binding))
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    RecoveryServingModel(root, 'zhenhong')

    def test_serving_build_override_requires_authorization_hash_and_original_roster(self):
        from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel
        for mutation in ('missing_request', 'missing_hash', 'missing_build', 'wrong_hash', 'reordered_roster'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('serving.json', 'zhenhong.json', 'zhenhong.npz'):
                    shutil.copyfile(ROOT / name, root / name)
                binding = json.loads((root / 'serving.json').read_text())
                pin = binding['models']['zhenhong']
                if mutation.startswith('missing_'):
                    pin.pop({'missing_request': 'serving_build_request', 'missing_hash': 'serving_build_sha256',
                             'missing_build': 'serving_build'}[mutation])
                elif mutation == 'wrong_hash':
                    pin['serving_build_sha256'] = 'changed'
                else:
                    from app.modules.card_game.rl.league_schema import build_hash
                    pin['serving_build']['character_ids'].reverse()
                    pin['serving_build_sha256'] = build_hash(pin['serving_build'])
                (root / 'serving.json').write_text(json.dumps(binding))
                with self.assertRaises(ValueError):
                    RecoveryServingModel(root, 'zhenhong')

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
                self.assertEqual(model.manifest['rule_hash'],
                                 binding['models'][key].get('source_rule_hash', SOURCE_RULE_HASH))
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

    def test_cached_release_rejects_live_rule_drift(self):
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.rl import cross_lineup
        from app.errors import RuleValidationError
        original = ai.load_model('murk')
        self.assertIs(ai.load_model('murk'), original)
        with patch.object(cross_lineup, 'identity', return_value='different-runtime'):
            with self.assertRaises(RuleValidationError) as failed:
                ai.load_model('murk')
            self.assertIn('authorization', str(failed.exception.__cause__))
        self.assertIs(ai.load_model('murk'), original)

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
    def test_public_player_can_start_all_five_recovery_opponents(self):
        import importlib
        ai = importlib.import_module('app.modules.card_game.engine.ai.advanced_model')
        repository = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        uid = 'five-public-recovery'
        token = self._issue_login_and_get_token(uid)
        self.assertFalse(self.dao_module.get_player_by_uid(uid).shaft_test_whitelisted)
        for key in KEYS:
            with self.subTest(key=key):
                model = ai.load_model(key)
                result = self._post('/api/duel-v2/start',
                                    {'mode': 'advanced', 'ai_deck': key}, token=token)
                self.assertEqual(result['room']['mode'], 'advanced')
                game = storage.load(repository.room_by_code(result['room']['room_code']).id)['game']
                self.assertTrue(game['ai_profile']['experimental'])
                self.assertEqual(game['ai_profile']['deck_id'], key)
                self.assertEqual(game['ai_profile']['version'], model.version)
                self.assertEqual(game['ai_profile']['build_sha256'], model.serving_build_sha256)
                self.assertEqual([hero['id'] for hero in result['game']['sides']['b']['characters']],
                                 model.serving_deck['character_ids'])
                self._post('/api/duel-v2/leave', token=token)

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
