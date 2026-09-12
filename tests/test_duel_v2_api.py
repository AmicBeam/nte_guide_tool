from __future__ import annotations

import importlib
import threading
from copy import deepcopy
from unittest.mock import patch

from tests.test_solo_room_flow import RoomFlowTestCase


class DuelV2ApiTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        self.repository = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        self.service = importlib.import_module('app.modules.card_game.engine.application.v2_service')
        self.a = self._issue_login_and_get_token('duel-v2-a')
        self.b = self._issue_login_and_get_token('duel-v2-b')
        self.deck = self._get('/api/duel-v2/catalog', token=self.a)['starter_deck']

    def tearDown(self):
        self.storage.flush(5)
        super().tearDown()

    def _pvp(self):
        result = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=self.a)
        self.code = result['room']['room_code']
        self._post('/api/duel-v2/join', {'room_code': self.code}, token=self.b)
        self._post('/api/duel-v2/ready', {'is_ready': True}, token=self.a)
        self._post('/api/duel-v2/ready', {'is_ready': True}, token=self.b)
        self._post('/api/duel-v2/room-start', token=self.a)
        for token in (self.a, self.b):
            view = self._get('/api/duel-v2/state', token=token)['game']
            action = next(entry['action'] for entry in view['legal_actions']
                          if entry['action']['type'] == 'mulligan' and not entry['action']['card_ids'])
            self._post('/api/duel-v2/action', {'room_code': self.code, 'request_id': f'mulligan-{view["viewer_side"]}',
                       'expected_version': view['version'], 'action': action}, token=token)
        view = self._get('/api/duel-v2/state', token=self.a)['game']
        token = self.a if view['active_side'] == 'a' else self.b
        return token, self._get('/api/duel-v2/state', token=token)['game']

    @staticmethod
    def _action(view, kind):
        return next(entry['action'] for entry in view['legal_actions'] if entry['action']['type'] == kind)

    def test_catalog_and_build_are_versioned_and_strict(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(catalog['rules_version'], 'duel_v2')
        self.assertEqual(len(catalog['characters']), 4)
        self.assertEqual(len(catalog['cards']), 32)
        self.assertEqual(catalog['starter_deck']['character_ids'], ['nanali', 'zero', 'jiuyuan', 'xun'])
        player = self.dao_module.get_or_create_player('duel-v2-a')[0]
        with self.db_module.atomic_transaction():
            self.dao_module.upsert_build(player, 'legacy', ['legacy-card'])
        legacy = self.dao_module.get_build(player)
        self._post('/api/duel-v2/build', self.deck, token=self.a)
        saved = self._get('/api/duel-v2/catalog', token=self.a)['saved_build']
        self.assertEqual(saved['card_ids'], self.deck['card_ids'])
        self.assertEqual(self.dao_module.get_build(player), legacy)
        invalid = deepcopy(self.deck)
        invalid['card_ids'] = invalid['card_ids'][:-1]
        self._post_error('/api/duel-v2/build', invalid, token=self.a)
        self.assertEqual(self._get('/api/duel-v2/catalog', token=self.a)['saved_build']['card_ids'], saved['card_ids'])
        invalid = deepcopy(self.deck)
        invalid['character_ids'] = ['nanali'] * 4
        self._post_error('/api/duel-v2/build', invalid, token=self.a)

    def test_pvp_ready_and_host_authorization(self):
        room = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=self.a)
        self._post_error('/api/duel-v2/room-start', token=self.a)
        self._post('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)
        self._post('/api/duel-v2/ready', {'is_ready': True}, token=self.a)
        ready = self._post('/api/duel-v2/ready', {'is_ready': True}, token=self.b)
        self.assertEqual(ready['room']['status'], 'ready')
        self._post_error('/api/duel-v2/room-start', token=self.b)
        started = self._post('/api/duel-v2/room-start', token=self.a)
        self.assertIsNotNone(started['game'])
        self.assertTrue(all('deck' not in person for person in started['room']['members']))

    def test_action_is_immediate_idempotent_and_versioned(self):
        token, view = self._pvp()
        payload = {'room_code': self.code, 'request_id': 'attack-once', 'expected_version': view['version'],
                   'action': self._action(view, 'attack')}
        result = self._post('/api/duel-v2/action', payload, token=token)['game']
        self.assertGreater(result['version'], view['version'])
        replay = self._post('/api/duel-v2/action', payload, token=token)
        self.assertTrue(replay['replayed_request'])
        self.assertEqual(replay['game']['version'], result['version'])
        self.assertEqual(replay['game']['sides'], result['sides'])
        self._post_error('/api/duel-v2/action', {**payload, 'room_code': self.code, 'request_id': 'stale'}, token=token, expected_status=409)
        self._post_error('/api/duel-v2/action', {**payload, 'action': {'type': 'end_turn'}}, token=token,
                         expected_status=409)
        self.assertTrue(self.storage.flush(5))
        self.storage.clear_cache()
        restored = self._get('/api/duel-v2/state', token=token)['game']
        self.assertEqual(restored['sides'], result['sides'])
        self.assertEqual(restored['version'], result['version'])
        self.assertTrue(self._post('/api/duel-v2/action', payload, token=token)['replayed_request'])

    def test_latest_state_visible_before_async_write_finishes(self):
        token, view = self._pvp()
        self.assertTrue(self.storage.flush(5))
        writing, release = threading.Event(), threading.Event()
        original = self.repository.write_snapshot

        def delayed(*args, **kwargs):
            writing.set()
            release.wait(3)
            return original(*args, **kwargs)

        try:
            with patch.object(self.repository, 'write_snapshot', side_effect=delayed):
                updated = self._post('/api/duel-v2/action', {
                    'room_code': self.code, 'request_id': 'write-delayed', 'expected_version': view['version'],
                    'action': self._action(view, 'attack'),
                }, token=token)['game']
                self.assertTrue(writing.wait(1))
                latest = self._get('/api/duel-v2/state', token=token)['game']
                self.assertEqual(latest['version'], updated['version'])
                self.assertEqual(latest['sides'], updated['sides'])
                release.set()
                self.assertTrue(self.storage.flush(5))
        finally:
            release.set()

    def test_other_user_cannot_read_or_apply_room_action(self):
        token, view = self._pvp()
        outsider = self._issue_login_and_get_token('duel-v2-outsider')
        self._get('/api/duel-v2/state', token=outsider, expected_status=404)
        self._post_error('/api/duel-v2/action', {
            'room_code': self.code, 'request_id': 'intruder', 'expected_version': view['version'],
            'action': self._action(view, 'attack'),
        }, token=outsider, expected_status=404)
        self.assertEqual(self._get('/api/duel-v2/state', token=token)['game']['version'], view['version'])

    def test_nonacting_player_can_concede_and_result_is_durable(self):
        active, view = self._pvp()
        other = self.b if active == self.a else self.a
        inactive_view = self._get('/api/duel-v2/state', token=other)['game']
        result = self._post('/api/duel-v2/action', {
            'room_code': self.code, 'request_id': 'concede-other-turn', 'expected_version': inactive_view['version'],
            'action': {'type': 'concede'},
        }, token=other)
        self.assertEqual(result['game']['phase'], 'finished')
        self.assertEqual(result['game']['winner'], view['active_side'])
        self.assertEqual(result['room']['status'], 'finished')
        self.storage.flush(5)
        self.storage.clear_cache()
        self.assertEqual(self._get('/api/duel-v2/state', token=active)['game']['phase'], 'finished')

    def test_solo_uses_rule_opponent_without_training(self):
        result = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        self.code = result['room']['room_code']
        view = result['game']
        action = next(entry['action'] for entry in view['legal_actions']
                      if entry['action']['type'] == 'mulligan' and not entry['action']['card_ids'])
        result = self._post('/api/duel-v2/action', {'room_code': self.code, 'request_id': 'solo-mulligan',
                            'expected_version': view['version'], 'action': action}, token=self.a)
        self.assertEqual(result['game']['active_side'], 'a')
        self.assertEqual(result['game']['phase'], 'playing')
        view = result['game']
        self._post('/api/duel-v2/action', {'room_code': self.code, 'request_id': 'solo-end', 'expected_version': view['version'],
                   'action': self._action(view, 'end_turn')}, token=self.a)
        self.assertEqual(self._get('/api/duel-v2/state', token=self.a)['game']['active_side'], 'a')

    def test_lobby_leave_and_legacy_snapshot_are_isolated(self):
        player = self.dao_module.get_or_create_player('duel-v2-a')[0]
        with self.db_module.atomic_transaction():
            legacy_room = self.dao_module.create_room(player, 'solo')
            self.dao_module.upsert_run(legacy_room, 'playing', {'legacy': 'keep-me'})
        room = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=self.a)
        self._post('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)
        self._post('/api/duel-v2/leave', token=self.a)
        self._get('/api/duel-v2/state', token=self.b, expected_status=404)
        self.assertEqual(self.dao_module.get_run(legacy_room)['snapshot'], {'legacy': 'keep-me'})

    def test_old_room_request_cannot_apply_to_new_room_at_same_version(self):
        first = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        delayed = {'room_code': first['room']['room_code'], 'request_id': 'old-concede',
                   'expected_version': first['game']['version'], 'action': {'type': 'concede'}}
        self._post('/api/duel-v2/leave', token=self.a)
        second = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        self.assertEqual(second['game']['version'], delayed['expected_version'])
        self._post_error('/api/duel-v2/action', delayed, token=self.a, expected_status=409)
        current = self._get('/api/duel-v2/state', token=self.a)
        self.assertEqual(current['game']['phase'], 'mulligan')
        self.assertEqual(current['room']['room_code'], second['room']['room_code'])

    def test_action_requires_room_identity(self):
        result = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        self._post_error('/api/duel-v2/action', {'request_id': 'missing-room',
                         'expected_version': result['game']['version'],
                         'action': {'type': 'concede'}}, token=self.a)
