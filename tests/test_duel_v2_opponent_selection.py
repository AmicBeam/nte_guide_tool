"""Opponent selection, mirror restrictions and creation retry coverage."""
from collections import Counter
from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_solo_room_flow import RoomFlowTestCase


class OpponentSelectionTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.service = importlib.import_module('app.modules.card_game.engine.application.v2_service')
        self.storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        self.repository = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        self.content = importlib.import_module('app.modules.card_game.content.duel_v2')
        self.token = self._issue_login_and_get_token('opponent-selection')
        self.player = self.dao_module.get_or_create_player('opponent-selection')[0]

    def tearDown(self):
        self.storage.flush(5)
        super().tearDown()

    def start(self, **kwargs):
        return self._post('/api/duel-v2/start', kwargs, token=self.token)

    def raw(self, result):
        room = self.repository.room_by_code(result['room']['room_code'])
        return self.storage.load(room.id)['game']

    def test_mirror_copies_custom_saved_cards_and_survives_changed_build(self):
        deck = deepcopy(self.content.STARTER_DECK)
        deck['card_ids'][deck['card_ids'].index('N02')] = 'N01'
        self._post('/api/duel-v2/build', deck, token=self.token)
        before = self._get('/api/duel-v2/catalog', token=self.token)['saved_build']
        result = self.start(mode='solo', ai_deck='mirror')
        game = self.raw(result)
        for side in ('a', 'b'):
            team = game['sides'][side]
            cards = [c['card_id'] for zone in ('hand', 'deck', 'discard') for c in team[zone]
                     if not self.content.CARDS[c['card_id']].get('derived')]
            self.assertEqual(Counter(cards), Counter(deck['card_ids']))
        self.assertEqual(list(game['sides']['a']['characters']), list(game['sides']['b']['characters']))
        self.assertEqual(game['ai_profile']['deck_id'], 'mirror')
        self.assertEqual(before, self._get('/api/duel-v2/catalog', token=self.token)['saved_build'])
        self.storage.flush(5)
        self.storage.clear_cache()
        with patch.object(self.service, '_selected_deck', side_effect=AssertionError('must resume original build')):
            again = self.start(mode='solo', ai_deck='mirror')
        self.assertEqual(result['room']['room_code'], again['room']['room_code'])
        self.assertEqual(again['game']['sides']['b']['name'], '镜像对局')
        replay = self._get('/api/duel-v2/replay/' + result['room']['room_code'], token=self.token)['replay']
        self.assertEqual(replay['opening_board']['sides']['b']['name'], '镜像对局')

    def test_mirror_rejects_excluded_card_but_accepts_test_character_without_it(self):
        self.player.shaft_test_whitelisted = True
        self.player.save()
        deck = deepcopy(self.content.STARTER_DECK)
        deck['character_ids'] = ['xun' if cid == 'nanali' else cid for cid in deck['character_ids']]
        deck['card_ids'] = [cid for cid in deck['card_ids'] if not cid.startswith('N')]
        deck['card_ids'] += ['X01', 'X02', 'X03', 'X04'] * 2
        self._post('/api/duel-v2/build', deck, token=self.token)
        response = self.client.post('/api/duel-v2/start', json={'mode': 'solo', 'ai_deck': 'mirror'},
                                    headers={'Authorization': 'Bearer ' + self.token})
        self.assertEqual(response.status_code, 400)
        self.assertIn('欣欣渐行', response.get_json()['error'])
        self.assertIsNone(self.repository.current_room(self.player))
        deck['card_ids'] = ['X05' if cid == 'X01' else cid for cid in deck['card_ids']]
        self._post('/api/duel-v2/build', deck, token=self.token)
        result = self.start(mode='solo', ai_deck='mirror')
        self.assertIn('xun', self.raw(result)['sides']['b']['characters'])

    def test_random_solo_defaults_to_public_pool_and_does_not_reroll(self):
        with patch('secrets.choice', side_effect=lambda choices: choices[-1]) as choose:
            first = self.start(mode='solo')
            selected = choose.call_args.args[0]
            self.assertEqual({d['id'] for d in selected}, {'starter', 'weave-rush', 'quick-rush', 'zhenhong'})
            calls = choose.call_count
            again = self.start(mode='solo', ai_deck='random')
            self.assertEqual(choose.call_count, calls)
        self.assertEqual(first['room']['room_code'], again['room']['room_code'])

    def test_mirror_restriction_uses_metadata_not_a_hardcoded_card(self):
        deck = deepcopy(self.content.STARTER_DECK)
        cid = deck['card_ids'][0]
        with patch.dict(self.content.CARDS[cid], training_excluded=True):
            response = self.client.post('/api/duel-v2/start',
                json={'mode': 'solo', 'ai_deck': 'mirror', 'deck': deck},
                headers={'Authorization': 'Bearer ' + self.token})
        self.assertEqual(response.status_code, 400)
        self.assertIn(self.content.CARDS[cid]['name'], response.get_json()['error'])
        self.assertIsNone(self.repository.current_room(self.player))

    def test_existing_mirror_requires_leaving_before_explicit_opponent_change(self):
        first = self.start(mode='solo', ai_deck='mirror')
        response = self.client.post('/api/duel-v2/start', json={'mode': 'solo', 'ai_deck': 'starter'},
                                    headers={'Authorization': 'Bearer ' + self.token})
        self.assertEqual(response.status_code, 409)
        again = self.start(mode='solo')
        self.assertEqual(first['room']['room_code'], again['room']['room_code'])

    def fake_model(self, key):
        from app.modules.card_game.rl.league_schema import SCHEMA
        from app.modules.card_game.engine.ai.advanced_model import preset
        deck = deepcopy(preset(key))
        return SimpleNamespace(version='selection-test', serving_deck=deck,
                               schema=SCHEMA, validate_human=lambda deck: None)

    def test_advanced_random_default_and_explicit_only_choose_loadable_models(self):
        from app.errors import RuleValidationError
        from app.modules.card_game.engine.duel_v2 import choose_action
        path = 'app.modules.card_game.engine.ai.advanced_model.'
        def load(key):
            if key == 'weave-rush':
                raise RuleValidationError('unavailable fixture')
            return self.fake_model(key)
        for options in ({}, {'ai_deck': 'random'}):
            with self.subTest(options=options), patch(path + 'load_model', side_effect=load) as loader, \
                    patch(path + 'choose_action', side_effect=choose_action), \
                    patch('secrets.choice', side_effect=lambda choices: choices[-1]) as choose:
                first = self.start(mode='advanced', **options)
                pool = choose.call_args.args[0]
                self.assertEqual([key for key, model in pool], ['starter', 'quick-rush'])
                self.assertNotIn('midrange', [call.args[0] for call in loader.call_args_list])
                self.assertEqual(self.raw(first)['ai_profile']['deck_id'], 'quick-rush')
                loader.reset_mock()
                choose.reset_mock()
                again = self.start(mode='advanced', ai_deck='random')
                self.assertEqual(first['room']['room_code'], again['room']['room_code'])
                loader.assert_not_called()
                choose.assert_not_called()
            self._post('/api/duel-v2/leave', token=self.token)

    def test_advanced_no_available_model_leaves_no_room(self):
        from app.errors import RuleValidationError
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model', side_effect=RuleValidationError('unavailable')):
            response = self.client.post('/api/duel-v2/start', json={'mode': 'advanced'},
                                        headers={'Authorization': 'Bearer ' + self.token})
        self.assertEqual(response.status_code, 400)
        self.assertIn('暂无可用', response.get_json()['error'])
        self.assertIsNone(self.repository.current_room(self.player))

    def test_retired_test_presets_are_unavailable_even_for_test_accounts(self):
        self.player.shaft_test_whitelisted = True
        self.player.save()
        catalog = self._get('/api/duel-v2/catalog', token=self.token)
        self.assertEqual({d['id'] for d in catalog['starter_decks']},
                         {'starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk'})
        self.assertEqual(len(catalog['saved_builds']), 5)
        for retired in ('requiem', 'sync-curse'):
            response = self.client.post('/api/duel-v2/start', json={'mode': 'solo', 'ai_deck': retired},
                                        headers={'Authorization': 'Bearer ' + self.token})
            self.assertEqual(response.status_code, 400)
            self.assertIsNone(self.repository.current_room(self.player))

    def test_test_account_can_face_murk_simple_ai(self):
        self.player.shaft_test_whitelisted = True
        self.player.save()
        result = self.start(mode='solo', ai_deck='murk')
        game = self.raw(result)
        self.assertEqual(list(game['sides']['b']['characters']), ['anhunqu', 'canhong', 'zaowu', 'adler'])
        self.assertEqual(game['sides']['b']['name'], '浊燃预组')
