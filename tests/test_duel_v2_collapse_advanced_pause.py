"""User-requested suspension wins over model presence and the random opponent pool."""
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.test_solo_room_flow import RoomFlowTestCase


class CollapseAdvancedPauseTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.token = self._issue_login_and_get_token('collapse-pause-test')
        self.model = importlib.import_module('app.modules.card_game.engine.ai.advanced_model')
        self.repo = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        self.player = self.dao_module.get_or_create_player('collapse-pause-test')[0]

    def test_files_and_warm_cache_cannot_reopen_collapse(self):
        with TemporaryDirectory() as directory, patch.object(self.model.config, 'DUEL_AI_MODEL_DIR', Path(directory)):
            for suffix in ('.json', '.npz'):
                (Path(directory) / ('collapse' + suffix)).write_bytes(b'present')
            with patch.dict(self.model._CACHE, {(str(Path(directory)), 'collapse'): object()}):
                option = next(o for o in self.model.pending_strategy_options() if o['id'] == 'collapse')
                self.assertTrue(option['pending'])
                self.assertIn('黑羽卡牌', option['unavailable_reason'])
                with self.assertRaisesRegex(self.model.RuleValidationError, '暂不开放'):
                    self.model.load_model('collapse')
            response = self.client.post('/api/duel-v2/start', json={'mode': 'advanced', 'ai_deck': 'collapse'},
                                        headers={'Authorization': 'Bearer ' + self.token})
            self.assertEqual(response.status_code, 400)
            self.assertIn('暂不开放', response.get_json()['error'])
            self.assertIsNone(self.repo.current_room(self.player))

    def test_random_never_attempts_suspended_strategy(self):
        with patch.object(self.model, 'load_model', side_effect=self.model.RuleValidationError('not ready')) as load:
            response = self.client.post('/api/duel-v2/start', json={'mode': 'advanced', 'ai_deck': 'random'},
                                        headers={'Authorization': 'Bearer ' + self.token})
            self.assertEqual(response.status_code, 400)
            self.assertFalse(any(call.args[0] == 'collapse' for call in load.call_args_list))
            self.assertIsNone(self.repo.current_room(self.player))
