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

    def test_catalog_grants_official_presets_once(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.b)
        official = [item['name'] for item in catalog['starter_decks']]
        names = [item['name'] for item in catalog['saved_builds']]
        self.assertEqual(names, official)
        self.assertEqual(official, ['创生预组'])
        first_ids = [item['id'] for item in catalog['saved_builds']]
        self.assertTrue(all(item_id not in ('starter', 'requiem', 'crimson') for item_id in first_ids))
        again = self._get('/api/duel-v2/catalog', token=self.b)
        self.assertEqual([item['id'] for item in again['saved_builds']], first_ids)
        deleted = self._post('/api/duel-v2/build-delete', {'id': first_ids[0]}, token=self.b)
        self.assertEqual(len(deleted['saved_builds']), len(first_ids) - 1)
        third = self._get('/api/duel-v2/catalog', token=self.b)
        self.assertEqual(len(third['saved_builds']), len(first_ids) - 1)
        self.assertNotIn(first_ids[0], [item['id'] for item in third['saved_builds']])

    def test_legacy_single_build_receives_remaining_presets(self):
        token = self._issue_login_and_get_token('duel-v2-legacy-presets')
        player = self.dao_module.get_or_create_player('duel-v2-legacy-presets')[0]
        starter = deepcopy(self.deck)
        starter['id'] = 'old-starter'
        with self.db_module.atomic_transaction():
            self.repository.save_build(player, starter)
        catalog = self._get('/api/duel-v2/catalog', token=token)
        expected = [item['name'] for item in catalog['starter_decks']]
        names = [item['name'] for item in catalog['saved_builds']]
        self.assertEqual(names, expected)
        self.assertEqual(catalog['saved_builds'][0]['id'], 'old-starter')
        self.assertEqual(catalog['active_build_id'], 'old-starter')
        self.assertEqual(expected, ['创生预组'])

    def test_complete_starter_presets_fills_already_granted_single_build(self):
        token = self._issue_login_and_get_token('duel-v2-seed-presets')
        player = self.dao_module.get_or_create_player('duel-v2-seed-presets')[0]
        starter = deepcopy(self.deck)
        with self.db_module.atomic_transaction():
            self.repository.save_build(player, {
                'active_id': starter['id'],
                'builds': [starter],
                'starter_granted': True,
            })
        store = self.service.complete_starter_presets(player)
        catalog = self._get('/api/duel-v2/catalog', token=token)
        expected = [item['name'] for item in catalog['starter_decks']]
        self.assertEqual([item['name'] for item in store['builds']], expected)
        self.assertEqual([item['name'] for item in catalog['saved_builds']], expected)
        self.assertEqual(catalog['active_build_id'], starter['id'])

    def test_catalog_and_build_are_versioned_and_strict(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(catalog['rules_version'], 'duel_v2')
        self.assertEqual({item['id'] for item in catalog['characters']}, {'nanali', 'zero', 'jiuyuan', 'iloy'})
        self.assertEqual(len(catalog['characters']), 4)
        self.assertEqual(len(catalog['cards']), 33)
        self.assertTrue(all(item['character_id'] in {'nanali', 'zero', 'jiuyuan', 'iloy'} for item in catalog['cards']))
        self.assertEqual(catalog['starter_deck']['character_ids'], ['nanali', 'iloy', 'zero', 'jiuyuan'])
        self.assertEqual([item['name'] for item in catalog['starter_decks']], ['创生预组'])
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

    def test_player_can_save_select_and_delete_multiple_builds(self):
        first = deepcopy(self.deck)
        first['name'] = '第一套'
        saved = self._post('/api/duel-v2/build', first, token=self.a)
        first_id = saved['build']['id']
        self.assertNotEqual(first_id, 'starter')
        second = deepcopy(self.deck)
        second['name'] = '第二套'
        second.pop('id', None)
        saved2 = self._post('/api/duel-v2/build', second, token=self.a)
        second_id = saved2['build']['id']
        self.assertNotEqual(first_id, second_id)
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        granted = len(catalog['starter_decks'])
        self.assertEqual(len(catalog['saved_builds']), granted + 2)
        self.assertEqual(catalog['active_build_id'], second_id)
        selected = self._post('/api/duel-v2/build-select', {'id': first_id}, token=self.a)
        self.assertEqual(selected['active_build_id'], first_id)
        self.assertEqual(selected['saved_build']['name'], '第一套')
        deleted = self._post('/api/duel-v2/build-delete', {'id': first_id}, token=self.a)
        self.assertEqual(len(deleted['saved_builds']), granted + 1)
        self.assertEqual(deleted['active_build_id'], deleted['saved_builds'][0]['id'])
        self.assertNotEqual(deleted['active_build_id'], first_id)
        self._post_error('/api/duel-v2/build-delete', {'id': first_id}, token=self.a, expected_status=404)

    def test_player_can_reorder_saved_builds(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        while len(catalog['saved_builds']) < 3:
            extra = deepcopy(self.deck)
            extra['name'] = f'加一套{len(catalog["saved_builds"])}'
            extra.pop('id', None)
            catalog = self._post('/api/duel-v2/build', extra, token=self.a)
        original = [item['id'] for item in catalog['saved_builds']]
        self.assertGreaterEqual(len(original), 3)
        reversed_ids = list(reversed(original))
        reordered = self._post('/api/duel-v2/build-reorder', {'ids': reversed_ids}, token=self.a)
        self.assertEqual([item['id'] for item in reordered['saved_builds']], reversed_ids)
        self.assertEqual(reordered['active_build_id'], catalog['active_build_id'])
        again = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual([item['id'] for item in again['saved_builds']], reversed_ids)
        self._post_error('/api/duel-v2/build-reorder', {'ids': original[1:]}, token=self.a)
        self._post_error('/api/duel-v2/build-reorder', {'ids': original + ['deck-missing']}, token=self.a)
        self._post_error('/api/duel-v2/build-reorder', {}, token=self.a)

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
        opponent = [item['id'] for item in view['sides']['b']['characters']]
        self.assertEqual(opponent, ['nanali', 'iloy', 'zero', 'jiuyuan'])
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

    def test_solo_leave_clears_current_room(self):
        self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        self._post('/api/duel-v2/leave', token=self.a)
        self._get('/api/duel-v2/state', token=self.a, expected_status=404)

    def test_lobby_leave_and_legacy_snapshot_are_isolated(self):
        player = self.dao_module.get_or_create_player('duel-v2-a')[0]
        with self.db_module.atomic_transaction():
            legacy_room = self.dao_module.create_room(player, 'solo')
            self.dao_module.upsert_run(legacy_room, 'playing', {'legacy': 'keep-me'})
        room = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=self.a)
        self._post('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)
        self._post('/api/duel-v2/leave', token=self.a)
        self._get('/api/duel-v2/state', token=self.a, expected_status=404)
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

    def _set_test_whitelist(self, player_uid: str, enabled: bool = True) -> None:
        models = importlib.import_module('app.models')
        models.Player.update(shaft_test_whitelisted=enabled).where(
            models.Player.player_uid == player_uid
        ).execute()

    def test_catalog_hides_test_characters_from_regular_and_invited_accounts(self):
        invited = self._issue_login_and_get_token('duel-v2-invited')
        models = importlib.import_module('app.models')
        models.Player.update(shaft_invited=True).where(
            models.Player.player_uid == 'duel-v2-invited'
        ).execute()
        for token in (self.a, invited):
            catalog = self._get('/api/duel-v2/catalog', token=token)
            names = {item['name'] for item in catalog['characters']}
            self.assertEqual(names, {'娜娜莉', '零', '九原', '伊洛伊'})
            self.assertNotIn('浔', names)
            self.assertEqual([item['name'] for item in catalog['starter_decks']], ['创生预组'])
            self.assertEqual([item['name'] for item in catalog['saved_builds']], ['创生预组'])

    def test_catalog_shows_test_characters_and_presets_to_whitelisted_accounts(self):
        token = self._issue_login_and_get_token('duel-v2-tester')
        self._set_test_whitelist('duel-v2-tester')
        catalog = self._get('/api/duel-v2/catalog', token=token)
        names = {item['name'] for item in catalog['characters']}
        self.assertEqual(len(catalog['characters']), 11)
        self.assertIn('浔', names)
        self.assertIn('残虹', names)
        self.assertEqual(len(catalog['cards']), 89)
        self.assertEqual(
            [item['name'] for item in catalog['starter_decks']],
            ['创生预组', '浊燃预组', '盈蓄预组', '覆纹预组'],
        )
        self.assertEqual(
            [item['name'] for item in catalog['saved_builds']],
            ['创生预组', '浊燃预组', '盈蓄预组', '覆纹预组'],
        )

    def test_regular_account_cannot_save_or_start_with_test_characters(self):
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        requiem = deepcopy(next(item for item in content.STARTER_DECKS if item['id'] == 'requiem'))
        error = self._post_error('/api/duel-v2/build', requiem, token=self.a)
        self.assertIn('尚未公开', error['error'])
        start_error = self._post_error('/api/duel-v2/start', {'mode': 'solo', 'deck': requiem}, token=self.a)
        self.assertIn('尚未公开', start_error['error'])

    def test_regular_account_cannot_join_room_using_test_characters(self):
        tester = self._issue_login_and_get_token('duel-v2-host-tester')
        self._set_test_whitelist('duel-v2-host-tester')
        catalog = self._get('/api/duel-v2/catalog', token=tester)
        requiem = next(item for item in catalog['saved_builds'] if item['name'] == '浊燃预组')
        self._post('/api/duel-v2/build-select', {'id': requiem['id']}, token=tester)
        room = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=tester)
        error = self._post_error('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)
        self.assertIn('尚未公开', error['error'])

    def test_whitelisted_account_receives_remaining_test_presets(self):
        token = self._issue_login_and_get_token('duel-v2-upgrade')
        first = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual([item['name'] for item in first['saved_builds']], ['创生预组'])
        self._set_test_whitelist('duel-v2-upgrade')
        upgraded = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual(
            [item['name'] for item in upgraded['saved_builds']],
            ['创生预组', '浊燃预组', '盈蓄预组', '覆纹预组'],
        )

    def test_regular_account_does_not_see_existing_test_character_builds(self):
        token = self._issue_login_and_get_token('duel-v2-hidden-builds')
        player = self.dao_module.get_or_create_player('duel-v2-hidden-builds')[0]
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        public = deepcopy(content.STARTER_DECK)
        public['id'] = 'deck-public'
        requiem = deepcopy(next(item for item in content.STARTER_DECKS if item['id'] == 'requiem'))
        requiem['id'] = 'deck-hidden'
        with self.db_module.atomic_transaction():
            self.repository.save_build(player, {
                'active_id': 'deck-hidden',
                'builds': [public, requiem],
                'starter_granted': True,
                'granted_preset_ids': ['starter', 'requiem', 'crimson', 'sync-curse'],
            })
        catalog = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual([item['id'] for item in catalog['saved_builds']], ['deck-public'])
        self.assertEqual(catalog['active_build_id'], 'deck-public')
        self.assertEqual(catalog['saved_build']['id'], 'deck-public')
        self._post_error('/api/duel-v2/build-select', {'id': 'deck-hidden'}, token=token, expected_status=404)
