from __future__ import annotations
from tests.duel_v2_test_decks import test_character_deck

import importlib
import json
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

    def test_haniya_resource_action_is_idempotent_and_reloadable(self):
        self._pvp()
        from tests.test_duel_v2_haniya import HaniyaTest
        from app.modules.card_game.engine.duel_v2.state import hero
        game = HaniyaTest().game()
        hero(game, 'a', 'haniya')['protagonist_aura'] = 4
        room = self.repository.room_by_code(self.code)
        self.storage.initialize(room, {'rules_version': 'duel_v2', 'game': game, 'requests': []})
        from app.modules.card_game.engine.application import v2_replay
        v2_replay.begin(room, game)
        body = dict(room_code=self.code, request_id='haniya-sortie', expected_version=0,
                    action={'type': 'attack', 'character_id': 'zero'})
        first = self._post('/api/duel-v2/action', body, token=self.a)['game']
        retry = self._post('/api/duel-v2/action', body, token=self.a)['game']
        self.assertEqual(first['version'], retry['version'])
        self.assertTrue(self.storage.flush(5))
        self.storage.clear_cache()
        loaded = self._get('/api/duel-v2/state', token=self.a)['game']
        self.assertEqual(loaded['sides']['b']['hp'], 24)
        h = next(h for h in loaded['sides']['a']['characters'] if h['id'] == 'haniya')
        self.assertEqual(h['protagonist_aura'], 0)

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

    def test_edgar_redraw_is_private_idempotent_and_reloadable(self):
        self._pvp()
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import card_instance
        deck = dict(id='edgar-test', name='埃德嘉', character_ids=['edgar','zero','jiuyuan','hathor'],
                    card_ids=[p+str(i).zfill(2) for p in ('E','Z','J','H') for i in range(1,5) for _ in range(2)])
        game = new_game(seed=71, first_side='a', skip_mulligan=True, decks={'a':deck,'b':deck})
        spell = card_instance(game,'E03','a')
        game['sides']['a']['hand'] = [spell] + [card_instance(game,'Z03','a') for _ in range(7)]
        room = self.repository.room_by_code(self.code)
        self.storage.initialize(room, {'rules_version':'duel_v2','game':game,'requests':[]})
        from app.modules.card_game.engine.application import v2_replay
        v2_replay.begin(room,game)
        body = dict(room_code=self.code, request_id='edgar-redraw-open', expected_version=0,
                    action={'type':'play_card','card_id':spell['instance_id']})
        view = self._post('/api/duel-v2/action',body,token=self.a)['game']
        self.assertEqual(len(view['pending_choice']['choices']),6)
        self.assertEqual(view['pending_choice']['max_select'],3)
        self.assertEqual(self._get('/api/duel-v2/state',token=self.b)['game']['pending_choice']['choices'],[])
        self.assertTrue(self.storage.flush(5));self.storage.clear_cache()
        view = self._get('/api/duel-v2/state',token=self.a)['game']
        choices = [c['id'] for c in view['pending_choice']['choices'][:2]]
        body = dict(room_code=self.code, request_id='edgar-redraw-finish', expected_version=view['version'],
                    action={'type':'choose_cards','card_ids':choices})
        result = self._post('/api/duel-v2/action',body,token=self.a)
        retried = self._post('/api/duel-v2/action',body,token=self.a)
        self.assertEqual(result['game']['version'],retried['game']['version'])
        self.assertEqual(result['game']['phase'],'playing')
        self.assertEqual(result['game']['sides']['a']['hand_count'],7)
        self.assertFalse(set(choices) & {c.get('instance_id') for c in result['game']['sides']['a']['hand']})

    def test_xun_rewind_retry_and_reload_preserve_version_and_removal(self):
        self._pvp()
        from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions
        from app.modules.card_game.engine.duel_v2.state import card_instance
        deck = dict(id='xun-test', name='浔', character_ids=['xun','nanali','zero','jiuyuan'],
                    card_ids=[p+str(i).zfill(2) for p in ('X','N','Z','J') for i in range(1,5) for _ in range(2)])
        game = new_game(seed=1, first_side='a', skip_mulligan=True, decks={'a':deck,'b':deck})
        game = apply_action(game,'a',{'type':'end_turn'})
        game = apply_action(game,'b',{'type':'end_turn'})
        card = card_instance(game,'X01','a'); game['sides']['a']['hand'] = [card]
        room = self.repository.room_by_code(self.code)
        self.storage.initialize(room, {'rules_version':'duel_v2','game':game,'requests':[]})
        from app.modules.card_game.engine.application import v2_replay
        v2_replay.begin(room,game)
        body = dict(room_code=self.code, request_id='rewind-once', expected_version=game['version'],
                    action={'type':'play_card','card_id':card['instance_id']})
        first = self._post('/api/duel-v2/action',body,token=self.a)
        retried = self._post('/api/duel-v2/action',body,token=self.a)
        self.assertEqual(first['game']['version'],game['version']+1)
        self.assertEqual(retried['game']['version'],first['game']['version'])
        self.assertEqual(first['game']['turn'],1)
        self.assertTrue(self.storage.flush(5));self.storage.clear_cache()
        restored = self.storage.load(room.id)['game']
        self.assertIn(card['entity_id'],restored['time_removed'])
        self.assertEqual(restored['version'],first['game']['version'])
        text=json.dumps(self._get('/api/duel-v2/state',token=self.b))
        self.assertNotIn('turn_snapshots',text);self.assertNotIn('time_removed',text)

    def test_build_text_roundtrip_is_draft_until_saved(self):
        before = self._get('/api/duel-v2/catalog', token=self.a)
        original = dict(self.deck, name="测试 # '组")
        text = self._post('/api/duel-v2/build-export', original, token=self.a)['text']
        self.assertIn('异能对决构筑 1', text)
        legacy = self._post('/api/duel-v2/build-import',
                            {'text': text.replace('异能对决构筑', '异象对决构筑')}, token=self.a)['build']
        self.assertCountEqual(legacy['card_ids'], original['card_ids'])
        self.assertTrue(all(line.strip() for line in text.split('\n')))
        imported = self._post('/api/duel-v2/build-import',
                              {'text': '\ufeff# 注释\r\n' + text.replace('\n', '\r\n')}, token=self.a)['build']
        self.assertEqual(imported['id'], '')
        self.assertEqual(imported['name'], original['name'])
        self.assertEqual(imported['character_ids'], original['character_ids'])
        self.assertCountEqual(imported['card_ids'], original['card_ids'])
        after = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(before['saved_builds'], after['saved_builds'])
        self.assertEqual(before['active_build_id'], after['active_build_id'])
        saved = self._post('/api/duel-v2/build', imported, token=self.a)
        self.assertEqual(len(saved['saved_builds']), len(before['saved_builds']) + 1)

    def test_build_text_rejects_invalid_or_hidden_content(self):
        from app.modules.card_game.engine.application.v2_build_text import format_build
        from app.modules.card_game.content import duel_v2
        text = format_build(self.deck)
        bad = ['# only comments', text + '\n卡牌 N01 2',
               text.replace('异能对决构筑 1', '异能对决构筑 2'),
               text.replace('卡牌 ' + self.deck['card_ids'][0] + ' 2', '卡牌 ' + self.deck['card_ids'][0] + ' -2'),
               text + '\n未知字段 abc', 'x' * 16385]
        hidden = test_character_deck('requiem')
        bad.append(format_build(hidden))
        for value in bad:
            with self.subTest(text=value[:60]):
                response = self.client.post('/api/duel-v2/build-import', json={'text': value},
                                            headers={'Authorization': 'Bearer ' + self.a})
                self.assertEqual(response.status_code, 400, response.get_json())
        response = self.client.post('/api/duel-v2/build-export', json=hidden,
                                    headers={'Authorization': 'Bearer ' + self.a})
        self.assertEqual(response.status_code, 400)

    def test_catalog_grants_official_presets_once(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.b)
        official = [item['name'] for item in catalog['starter_decks']]
        names = [item['name'] for item in catalog['saved_builds']]
        self.assertEqual(names, official)
        self.assertEqual(official, ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])
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
        self.assertEqual(expected, ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])

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
        self.assertEqual({item['id'] for item in catalog['characters']}, {'nanali', 'zero', 'jiuyuan', 'iloy', 'bohe', 'baicang', 'xiaozhi', 'haiyue', 'zhenhong', 'yi', 'anhunqu', 'canhong', 'zaowu', 'adler'})
        self.assertEqual(len(catalog['characters']), 14)
        self.assertEqual(len(catalog['cards']), 115)
        self.assertTrue(all(item['character_id'] in {'nanali', 'zero', 'jiuyuan', 'iloy', 'bohe', 'baicang', 'xiaozhi', 'haiyue', 'zhenhong', 'yi', 'anhunqu', 'canhong', 'zaowu', 'adler'} for item in catalog['cards']))
        self.assertEqual(catalog['starter_deck']['character_ids'], ['nanali', 'iloy', 'zero', 'jiuyuan'])
        self.assertEqual([item['name'] for item in catalog['starter_decks']], ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])
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
        from app.modules.card_game.content.duel_v2 import get_catalog
        self.assertIn(opponent, [d['character_ids'] for d in get_catalog(include_test_characters=False)['starter_decks']])
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
            self.assertEqual(names, {'娜娜莉', '零', '九原', '伊洛伊', '薄荷', '白藏', '小吱', '海月', '真红', '翳', '安魂曲', '残虹', '早雾', '阿德勒'})
            self.assertNotIn('浔', names)
            self.assertEqual([item['name'] for item in catalog['starter_decks']], ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])
            self.assertEqual([item['name'] for item in catalog['saved_builds']], ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])

    def test_catalog_shows_test_characters_and_presets_to_whitelisted_accounts(self):
        token = self._issue_login_and_get_token('duel-v2-tester')
        self._set_test_whitelist('duel-v2-tester')
        catalog = self._get('/api/duel-v2/catalog', token=token)
        names = {item['name'] for item in catalog['characters']}
        self.assertEqual(len(catalog['characters']), 19)
        self.assertIn('浔', names)
        self.assertIn('残虹', names)
        self.assertIn('阿德勒', names)
        self.assertIn('薄荷', names)
        self.assertIn('白藏', names)
        self.assertEqual(len(catalog['cards']), 157)
        self.assertEqual(
            [item['name'] for item in catalog['starter_decks']],
            ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'],
        )
        self.assertEqual(
            [item['name'] for item in catalog['saved_builds']],
            ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'],
        )

    def test_zhenhong_rule_opponent_uses_named_deck_and_resumes(self):
        token = self._issue_login_and_get_token('zhenhong-opponent-tester')
        self._set_test_whitelist('zhenhong-opponent-tester')
        view = self._post('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'zhenhong'}, token=token)
        self.assertEqual([h['id'] for h in view['game']['sides']['b']['characters']],
                         ['zhenhong', 'zero', 'iloy', 'yi'])
        self.assertEqual(view['game']['sides']['b']['name'], '真红预组')
        replay = self._get('/api/duel-v2/replay/' + view['room']['room_code'], token=token)['replay']
        self.assertEqual(replay['opening_board']['sides']['b']['name'], '真红预组')
        same = self._post('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'zhenhong'}, token=token)
        self.assertEqual(same['room']['room_code'], view['room']['room_code'])
        error = self._post_error('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'starter'}, token=token, expected_status=409)
        self.assertIn('更换人机预组', error['error'])

    def test_old_crimson_gift_migrates_when_catalog_is_loaded(self):
        from tests.test_duel_v2_preset_migrations import PresetMigrationTest
        token = self._issue_login_and_get_token('crimson-gift-migration')
        self._set_test_whitelist('crimson-gift-migration')
        player = self.dao_module.get_or_create_player('crimson-gift-migration')[0]
        store = PresetMigrationTest().store()
        store['starter_granted'] = True
        self.repository.save_build(player, store)
        catalog = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual(catalog['saved_build']['id'], 'gift')
        self.assertEqual(catalog['saved_build']['name'], '真红预组')
        self.assertNotIn('盈蓄预组', [b['name'] for b in catalog['saved_builds']])
        again = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual(catalog['saved_builds'], again['saved_builds'])

    def test_retired_crimson_cannot_be_selected_by_test_account(self):
        token = self._issue_login_and_get_token('retired-crimson-tester')
        self._set_test_whitelist('retired-crimson-tester')
        error = self._post_error('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'crimson'}, token=token)
        self.assertIn('不存在', error['error'])

    def test_zhenhong_public_account_can_save_and_start(self):
        from app.modules.card_game.content.duel_v2 import get_catalog
        deck = next(d for d in get_catalog(include_test_characters=False)['starter_decks'] if d['id']=='zhenhong')
        self._post('/api/duel-v2/build', deck, token=self.a)
        view=self._post('/api/duel-v2/start', {'mode':'solo','ai_deck':'zhenhong'}, token=self.a)
        for side in ('a','b'):
            self.assertEqual([h['id'] for h in view['game']['sides'][side]['characters']], ['zhenhong','zero','iloy','yi'])

    def test_solo_explicit_public_opponent_and_unknown_rejection(self):
        error = self._post_error('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'missing'}, token=self.a)
        self.assertIn('不存在', error['error'])
        view = self._post('/api/duel-v2/start', {'mode': 'solo', 'ai_deck': 'starter'}, token=self.a)
        self.assertEqual([h['id'] for h in view['game']['sides']['b']['characters']],
                         ['nanali', 'iloy', 'zero', 'jiuyuan'])

    def test_regular_account_cannot_save_or_start_with_test_characters(self):
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        requiem = test_character_deck('requiem')
        error = self._post_error('/api/duel-v2/build', requiem, token=self.a)
        self.assertIn('尚未公开', error['error'])
        start_error = self._post_error('/api/duel-v2/start', {'mode': 'solo', 'deck': requiem}, token=self.a)
        self.assertIn('尚未公开', start_error['error'])

    def test_regular_account_cannot_join_room_using_test_characters(self):
        tester = self._issue_login_and_get_token('duel-v2-host-tester')
        self._set_test_whitelist('duel-v2-host-tester')
        self._post('/api/duel-v2/build', test_character_deck('requiem'), token=tester)
        catalog = self._get('/api/duel-v2/catalog', token=tester)
        requiem = next(item for item in catalog['saved_builds'] if 'lingke' in item['character_ids'])
        self._post('/api/duel-v2/build-select', {'id': requiem['id']}, token=tester)
        room = self._post('/api/duel-v2/start', {'mode': 'pvp'}, token=tester)
        error = self._post_error('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)
        self.assertIn('尚未公开', error['error'])

    def test_public_account_already_has_the_murk_preset(self):
        token = self._issue_login_and_get_token('duel-v2-upgrade')
        first = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual([item['name'] for item in first['saved_builds']], ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'])
        self._set_test_whitelist('duel-v2-upgrade')
        upgraded = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual(
            [item['name'] for item in upgraded['saved_builds']],
            ['创生预组', '覆纹预组', '小吱预组', '真红预组', '浊燃预组'],
        )

    def test_regular_account_does_not_see_existing_test_character_builds(self):
        token = self._issue_login_and_get_token('duel-v2-hidden-builds')
        player = self.dao_module.get_or_create_player('duel-v2-hidden-builds')[0]
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        public = deepcopy(content.STARTER_DECK)
        public['id'] = 'deck-public'
        requiem = test_character_deck('requiem')
        requiem['id'] = 'deck-hidden'
        with self.db_module.atomic_transaction():
            self.repository.save_build(player, {
                'active_id': 'deck-hidden',
                'builds': [public, requiem],
                'starter_granted': True,
                'granted_preset_ids': ['starter', 'requiem', 'crimson', 'sync-curse', 'weave-rush', 'quick-rush', 'zhenhong'],
            })
        catalog = self._get('/api/duel-v2/catalog', token=token)
        self.assertEqual([item['id'] for item in catalog['saved_builds']], ['deck-public'])
        self.assertEqual(catalog['active_build_id'], 'deck-public')
        self.assertEqual(catalog['saved_build']['id'], 'deck-public')
        self._post_error('/api/duel-v2/build-select', {'id': 'deck-hidden'}, token=token, expected_status=404)

    def test_regular_account_can_save_start_and_join_weave_rush(self):
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        deck = deepcopy(next(item for item in content.STARTER_DECKS if item['id'] == 'weave-rush'))
        self._post('/api/duel-v2/build', deck, token=self.a)
        room = self._post('/api/duel-v2/start', {'mode': 'pvp', 'deck': deck}, token=self.a)
        self._post('/api/duel-v2/join', {'room_code': room['room']['room_code']}, token=self.b)

    def test_public_quick_rush_can_save_and_start_solo(self):
        from app.modules.card_game.content.duel_v2 import get_catalog
        deck = next(d for d in get_catalog(include_test_characters=False)['starter_decks'] if d['id'] == 'quick-rush')
        self._post('/api/duel-v2/build', deck, token=self.a)
        room = self._post('/api/duel-v2/start', {'mode': 'solo', 'deck': deck}, token=self.a)
        self.assertEqual([h['id'] for h in room['game']['sides']['a']['characters']], deck['character_ids'])

    def test_quick_rush_advanced_unavailable_does_not_create_room(self):
        from app.errors import RuleValidationError
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model',
                   side_effect=RuleValidationError('高级人机模型暂不可用：测试缺失模型')):
            result = self._post_error('/api/duel-v2/start',
                                      {'mode': 'advanced', 'ai_deck': 'quick-rush'}, token=self.a)
        self.assertIn('高级人机模型暂不可用', result['error'])
        player = self.dao_module.get_or_create_player('duel-v2-a')[0]
        self.assertIsNone(self.repository.current_room(player))

    def test_tutorial_start_returns_guided_level_one(self):
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(catalog['tutorial']['current_level'], 'tutorial_l01_board')
        started = self._post('/api/duel-v2/start', {'mode': 'tutorial'}, token=self.a)
        self.assertEqual(started['room']['mode'], 'tutorial')
        game = started['game']
        self.assertEqual(game['tutorial']['step_id'], 'l01_intro')
        self.assertTrue(game['tutorial']['mask'])
        self.assertIn('bench:a', game['tutorial']['spotlights'])
        self.assertEqual(len(game['sides']['a']['characters']), 2)
        ack = next(entry['action'] for entry in game['legal_actions'] if entry['action']['type'] == 'tutorial_ack')
        next_view = self._post('/api/duel-v2/action', {
            'room_code': started['room']['room_code'],
            'request_id': 'tut-ack-1',
            'expected_version': game['version'],
            'action': ack,
        }, token=self.a)['game']
        self.assertEqual(next_view['tutorial']['step_id'], 'l01_bench')
        self.assertFalse(game['tutorial'].get('can_skip_step'))

    def test_tutorial_start_replaces_existing_room_and_records_current_level(self):
        first = self._post('/api/duel-v2/start', {'mode': 'tutorial'}, token=self.a)
        first_code = first['room']['room_code']
        self.assertEqual(first['game']['tutorial']['scenario'], 'tutorial_l01_board')
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(catalog['tutorial']['current_level'], 'tutorial_l01_board')

        replay = self._post('/api/duel-v2/start', {
            'mode': 'tutorial',
            'scenario': 'tutorial_l01_board',
        }, token=self.a)
        self.assertNotEqual(replay['room']['room_code'], first_code)
        self.assertEqual(replay['game']['tutorial']['scenario'], 'tutorial_l01_board')
        self.assertEqual(replay['game']['tutorial']['step_id'], 'l01_intro')

        from app.models import DuelV2TutorialProgress, Player
        player = Player.get(Player.player_uid == 'duel-v2-a')
        row = DuelV2TutorialProgress.get(DuelV2TutorialProgress.player == player)
        row.completed_levels = '["tutorial_l01_board"]'
        row.current_level = 'tutorial_l02_hand'
        row.save()
        second = self._post('/api/duel-v2/start', {
            'mode': 'tutorial',
            'scenario': 'tutorial_l02_hand',
        }, token=self.a)
        self.assertEqual(second['game']['tutorial']['scenario'], 'tutorial_l02_hand')
        catalog = self._get('/api/duel-v2/catalog', token=self.a)
        self.assertEqual(catalog['tutorial']['current_level'], 'tutorial_l02_hand')
        titles = [item['title'] for item in catalog['tutorial']['levels'] if item['available']]
        self.assertIn('第二关 手牌', titles)

        back = self._post('/api/duel-v2/start', {
            'mode': 'tutorial',
            'scenario': 'tutorial_l01_board',
        }, token=self.a)
        self.assertNotEqual(back['room']['room_code'], second['room']['room_code'])
        self.assertEqual(back['game']['tutorial']['scenario'], 'tutorial_l01_board')
        self.assertEqual(back['game']['tutorial']['step_id'], 'l01_intro')


    def test_old_tutorial_graduates_unlock_new_escalation_lesson(self):
        from app.models import DuelV2TutorialProgress
        from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS
        self._get('/api/duel-v2/catalog', token=self.a)
        row = DuelV2TutorialProgress.get()
        row.completed_levels = json.dumps(list(CAMPAIGN_LEVELS[:11]))
        row.current_level = CAMPAIGN_LEVELS[10]
        row.completed = True
        row.save()
        tutorial = self._get('/api/duel-v2/catalog', token=self.a)['tutorial']
        self.assertFalse(tutorial['completed'])
        self.assertEqual(tutorial['current_level'], 'tutorial_l13_escalation')
        self.assertTrue(tutorial['levels'][11]['available'])
        started = self._post('/api/duel-v2/start', {'mode': 'tutorial'}, token=self.a)
        self.assertEqual(started['game']['turn'], 4)
        self.assertEqual(started['game']['first_side'], 'b')
        self.assertEqual(started['game']['tutorial']['level_index'], 12)
        game = started['game']
        for index in range(12):
            if game['phase'] == 'finished':
                break
            game = self._post('/api/duel-v2/action', {
                'room_code': started['room']['room_code'],
                'request_id': f'escalation-lesson-{index}',
                'expected_version': game['version'],
                'action': game['legal_actions'][0]['action'],
            }, token=self.a)['game']
        self.assertEqual(game['reason'], 'tutorial_complete')
        self.assertEqual(game['turn'], 6)
        progress = self._get('/api/duel-v2/catalog', token=self.a)['tutorial']
        self.assertFalse(progress['completed'])
        self.assertEqual(progress['current_level'], 'tutorial_l14_ranged')


    def test_old_twelve_lesson_graduate_completes_ranged_and_keeps_progress(self):
        from app.models import DuelV2TutorialProgress
        from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS
        self._get('/api/duel-v2/catalog', token=self.a)
        row = DuelV2TutorialProgress.get()
        row.completed_levels = json.dumps(list(CAMPAIGN_LEVELS[:-1]))
        row.current_level = 'tutorial_l13_escalation'
        row.completed = True
        row.save()
        progress = self._get('/api/duel-v2/catalog', token=self.a)['tutorial']
        self.assertFalse(progress['completed'])
        self.assertEqual(progress['current_level'], 'tutorial_l14_ranged')
        self.assertTrue(progress['levels'][-1]['available'])
        started = self._post('/api/duel-v2/start', {'mode': 'tutorial'}, token=self.a)
        game = started['game']
        self.assertEqual(game['tutorial']['level_index'], 13)
        for index in range(10):
            if game['phase'] == 'finished':
                break
            body = dict(room_code=started['room']['room_code'], request_id=f'ranged-{index}',
                        expected_version=game['version'], action=game['legal_actions'][0]['action'])
            game = self._post('/api/duel-v2/action', body, token=self.a)['game']
            retry = self._post('/api/duel-v2/action', body, token=self.a)['game']
            self.assertEqual(retry['version'], game['version'])
            self.assertTrue(self.storage.flush(5))
            self.storage.clear_cache()
            game = self._get('/api/duel-v2/state', token=self.a)['game']
        self.assertEqual(game['reason'], 'tutorial_complete')
        self.assertEqual(game['sides']['a']['front'], 'tutorial_bohe')
        progress = self._get('/api/duel-v2/catalog', token=self.a)['tutorial']
        self.assertTrue(progress['completed'])
        self.assertTrue(all(level['completed'] for level in progress['levels']))
