"""Explicit experimental advanced opponents use frozen public observations."""
import unittest
from unittest.mock import patch

from tests.test_solo_room_flow import RoomFlowTestCase
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.ai.advanced_model import choose_action, load_model
from app.modules.card_game.engine.ai.experimental_cross_model import ExperimentalCrossModel
from app.modules.card_game.engine.duel_v2 import legal_actions, new_game


class ExperimentalModelTest(unittest.TestCase):
    def test_verified_models_choose_legal_action(self):
        decks = {item['id']: item for item in STARTER_DECKS}
        for key in ('zhenhong', 'murk'):
            with self.subTest(key=key):
                model = load_model(key)
                self.assertTrue(model.experimental)
                state = new_game(seed=901, first_side='b', skip_mulligan=True,
                                 decks={'a': decks['starter'], 'b': model.serving_deck})
                state['ai_profile'] = {'deck_id': key, 'version': model.version,
                                       'build_sha256': model.serving_build_sha256}
                self.assertIn(choose_action(state, 'b'), legal_actions(state, 'b'))

    def test_runtime_identity_change_rejects_archived_weights(self):
        contract = ExperimentalCrossModel.__init__.__globals__['cross_lineup']
        with patch.object(contract, 'identity', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'runtime has changed'):
                ExperimentalCrossModel('murk')


class ExperimentalAdvancedRoomTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.token = self._issue_login_and_get_token('experimental-advanced-test')

    def test_public_player_can_start_both_experimental_opponents(self):
        for key, members in (
            ('zhenhong', ['zhenhong', 'zero', 'iloy', 'yi']),
            ('murk', ['anhunqu', 'canhong', 'zaowu', 'adler']),
        ):
            with self.subTest(key=key):
                started = self._post('/api/duel-v2/start',
                                     {'mode': 'advanced', 'ai_deck': key}, token=self.token)
                self.assertEqual(started['room']['mode'], 'advanced')
                self.assertEqual([hero['id'] for hero in started['game']['sides']['b']['characters']], members)
                self.assertEqual(started['game']['sides']['b']['name'], '实验性高级人机')
                from app.modules.card_game.engine.application import v2_repository, v2_persistence
                room = v2_repository.room_by_code(started['room']['room_code'])
                game = v2_persistence.load(room.id)['game']
                self.assertTrue(game['ai_profile']['experimental'])
                self.assertEqual(game['ai_profile']['build_sha256'], load_model(key).serving_build_sha256)
                if key == 'zhenhong':
                    from collections import Counter
                    cards = game['sides']['b']['hand'] + game['sides']['b']['deck']
                    self.assertEqual(Counter(c['card_id'] for c in cards if c['character_id'] == 'yi'),
                                     Counter({'I01': 2, 'I03': 2, 'I04': 2, 'I08': 2}))
                self._post('/api/duel-v2/leave', token=self.token)


if __name__ == '__main__':
    unittest.main()
