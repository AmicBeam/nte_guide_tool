from __future__ import annotations

import json

from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.presentation import replay_public_board
from app.modules.card_game.engine.application.v2_replay import (
    assemble_replay_game,
    leak_markers,
    public_replay_view,
)
from tests.test_solo_room_flow import RoomFlowTestCase


SECRET = 'SECRET-HIDDEN-CARD'


def _mark_hidden(state):
    for side in ('a', 'b'):
        for card in state['sides'][side]['deck']:
            card['name'] = SECRET
        if side == 'b':
            for card in state['sides'][side]['hand']:
                if not card.get('copy'):
                    card['name'] = SECRET
    state['rng'] = 123456789


class DuelV2ReplayApiTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.a = self._issue_login_and_get_token('duel-v2-a')
        self.b = self._issue_login_and_get_token('duel-v2-b')

    def tearDown(self):
        try:
            super().tearDown()
        except OSError:
            pass

    @staticmethod
    def _action(view, kind):
        return next(entry['action'] for entry in view['legal_actions'] if entry['action']['type'] == kind)

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

    def _solo_started(self):
        result = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=self.a)
        self.code = result['room']['room_code']
        view = result['game']
        action = next(entry['action'] for entry in view['legal_actions']
                      if entry['action']['type'] == 'mulligan' and not entry['action']['card_ids'])
        after = self._post('/api/duel-v2/action', {
            'room_code': self.code, 'request_id': 'replay-mulligan',
            'expected_version': view['version'], 'action': action,
        }, token=self.a)
        return after

    def test_solo_replay_records_public_events_and_reconstructs_board(self):
        after = self._solo_started()
        self.assertTrue(after['room']['has_replay'])
        self.assertEqual(after['room']['replay_code'], self.code)
        view = after['game']
        attack = self._post('/api/duel-v2/action', {
            'room_code': self.code, 'request_id': 'replay-attack',
            'expected_version': view['version'], 'action': self._action(view, 'attack'),
        }, token=self.a)
        replay = self._get(f'/api/duel-v2/replay/{self.code}', token=self.a)['replay']
        self.assertGreaterEqual(replay['event_count'], 1)
        self.assertTrue(any(event['type'] in ('attack', 'play', 'turn', 'draw', 'mulligan', 'enter')
                            or event.get('text') for event in replay['events']))
        dumped = json.dumps(replay, ensure_ascii=False)
        self.assertNotIn('legal_actions', dumped)
        self.assertNotIn('"deck":', dumped)
        self.assertNotIn('"rng"', dumped)
        self.assertNotIn('private_card', dumped)
        self.assertEqual(leak_markers(replay), [])
        assembled = assemble_replay_game(replay['opening_board'], replay['events'])
        live = attack['game']
        for side in ('a', 'b'):
            self.assertEqual(assembled['sides'][side]['hp'], live['sides'][side]['hp'])
            self.assertEqual(assembled['sides'][side]['ap'], live['sides'][side]['ap'])
            self.assertEqual(assembled['sides'][side]['front'], live['sides'][side]['front'])
            self.assertEqual(assembled['sides'][side]['hand_count'], live['sides'][side]['hand_count'])
            self.assertEqual(assembled['sides'][side]['deck_count'], live['sides'][side]['deck_count'])
        self.assertEqual(assembled['phase'], live['phase'])
        self.assertEqual(assembled['winner'], live['winner'])
        replayed = replay_public_board(replay['opening_board'], replay['events'])
        self.assertEqual(replayed['sides']['a']['hp'], live['sides']['a']['hp'])
        self.assertEqual(replayed['sides']['b']['hp'], live['sides']['b']['hp'])
        listed = self._get('/api/duel-v2/replays', token=self.a)['replays']
        row = next(item for item in listed if item['room_code'] == self.code)
        self.assertFalse(row['favorited'])
        self.assertEqual(len(row['characters_a']), 4)
        self.assertEqual(len(row['characters_b']), 4)
        self.assertTrue(all(item.get('id') and item.get('name') for item in row['characters_a']))
        self.assertTrue(any(item.get('avatar') for item in row['characters_a']))

    def test_import_training_replay_is_listed_and_fetchable(self):
        from app.modules.card_game.engine.duel_v2 import apply_action, new_game
        from app.modules.card_game.engine.duel_v2.replay_log import build_match_payload
        from app.modules.card_game.rl.transfer import default_training_deck
        deck = default_training_deck()
        opening = new_game(seed=7, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck})
        after = apply_action(opening, 'a', {'type': 'attack', 'character_id': 'nanali'})
        payload = build_match_payload(opening, after, room_code='IGNORE', learning_side='a')
        imported = self._post('/api/duel-v2/replay-import', payload, token=self.a)
        code = imported['replay']['room_code']
        self.assertNotEqual(code, 'IGNORE')
        self.assertGreaterEqual(imported['replay']['event_count'], 1)
        listed = self._get('/api/duel-v2/replays', token=self.a)['replays']
        self.assertTrue(any(item['room_code'] == code for item in listed))
        fetched = self._get(f'/api/duel-v2/replay/{code}', token=self.a)['replay']
        self.assertEqual(fetched['event_count'], imported['replay']['event_count'])
        self.assertTrue(fetched['favorited'])

    def test_replay_survives_leave_and_hides_from_outsiders(self):
        after = self._solo_started()
        code = after['room']['room_code']
        self._post('/api/duel-v2/leave', token=self.a)
        replay = self._get(f'/api/duel-v2/replay/{code}', token=self.a)['replay']
        self.assertEqual(replay['room_code'], code)
        listed = self._get('/api/duel-v2/replays', token=self.a)['replays']
        self.assertTrue(any(item['room_code'] == code for item in listed))
        outsider = self._issue_login_and_get_token('duel-v2-replay-outsider')
        self._get(f'/api/duel-v2/replay/{code}', token=outsider, expected_status=404)
        self.assertFalse(self._get('/api/duel-v2/replays', token=outsider)['replays'])

    def test_pvp_replay_is_viewer_specific(self):
        token, view = self._pvp()
        other = self.b if token == self.a else self.a
        self._post('/api/duel-v2/action', {
            'room_code': self.code, 'request_id': 'pvp-replay-attack',
            'expected_version': view['version'], 'action': self._action(view, 'attack'),
        }, token=token)
        mine = self._get(f'/api/duel-v2/replay/{self.code}', token=token)['replay']
        theirs = self._get(f'/api/duel-v2/replay/{self.code}', token=other)['replay']
        self.assertEqual(mine['viewer_side'], 'a' if token == self.a else 'b')
        self.assertEqual(theirs['viewer_side'], 'b' if token == self.a else 'a')
        my_hand = mine['opening_board']['sides'][mine['viewer_side']]['hand']
        self.assertTrue(my_hand)
        self.assertTrue(all(card.get('name') or card.get('card_id') for card in my_hand))
        their_view_of_me = theirs['opening_board']['sides'][mine['viewer_side']]['hand']
        self.assertTrue(all(card.get('hidden') or card.get('copy') for card in their_view_of_me))
        dumped = json.dumps(mine, ensure_ascii=False)
        self.assertNotIn('private_card', dumped)
        self.assertNotIn('"deck":', dumped)

    def _seed_finished_replays(self, player, count: int) -> list[str]:
        from app.modules.card_game.engine.application import v2_repository as repository

        codes = []
        lineup_a = [
            {'id': 'nanali', 'name': '娜娜莉', 'avatar': '/static/images/characters/avatar/娜娜莉.webp'},
            {'id': 'zero', 'name': '零', 'avatar': '/static/images/characters/avatar/鉴定师.webp'},
            {'id': 'jiuyuan', 'name': '九原', 'avatar': '/static/images/characters/avatar/九原.webp'},
            {'id': 'xun', 'name': '浔', 'avatar': '/static/images/characters/avatar/浔.webp'},
        ]
        lineup_b = [
            {'id': 'anhunqu', 'name': '安魂曲', 'avatar': '/static/images/characters/avatar/安魂曲.webp'},
            {'id': 'canhong', 'name': '残虹', 'avatar': '/static/images/characters/avatar/残虹.png'},
            {'id': 'zaowu', 'name': '早雾', 'avatar': '/static/images/characters/avatar/早雾.webp'},
            {'id': 'lingke', 'name': '灵可', 'avatar': '/static/images/characters/avatar/灵可.png'},
        ]
        for _ in range(count):
            room = repository.create_room(player, 'solo')
            payload = {
                'schema_version': 1,
                'room_code': room.room_code,
                'mode': 'solo',
                'status': 'finished',
                'winner': 'a',
                'player_a_id': player.id,
                'player_b_id': None,
                'name_a': '我方',
                'name_b': '练习对手',
                'last_seq': 1,
                'views': {
                    'a': {
                        'opening_board': {
                            'sides': {
                                'a': {'characters': lineup_a},
                                'b': {'characters': lineup_b},
                            }
                        },
                        'events': [{'seq': 1, 'type': 'turn'}],
                    }
                },
            }
            repository.upsert_replay(room, payload, {
                'status': 'finished',
                'winner': 'a',
                'player_a_id': player.id,
                'player_b_id': None,
                'name_a': '我方',
                'name_b': '练习对手',
                'mode': 'solo',
            })
            repository.set_status(room, 'closed')
            codes.append(room.room_code)
        return codes

    def test_replay_star_and_unfavorited_cap(self):
        token = self._issue_login_and_get_token('duel-v2-replay-star')
        player = self.dao_module.get_or_create_player('duel-v2-replay-star')[0]
        codes = self._seed_finished_replays(player, 31)
        oldest, newest = codes[0], codes[-1]
        listed = self._get('/api/duel-v2/replays', token=token)
        rows = listed['replays']
        self.assertEqual(listed['unfavorited_limit'], 30)
        self.assertEqual(listed['unfavorited_count'], 30)
        self.assertEqual(len(rows), 30)
        self.assertNotIn(oldest, [item['room_code'] for item in rows])
        self.assertIn(newest, [item['room_code'] for item in rows])
        self.assertTrue(all(item['characters_a'] and item['characters_b'] for item in rows))
        self._get(f'/api/duel-v2/replay/{oldest}', token=token, expected_status=404)
        starred = self._post('/api/duel-v2/replay-star', {
            'room_code': newest, 'favorited': True,
        }, token=token)
        self.assertEqual(len(starred['replays']), 31)
        self.assertTrue(next(item for item in starred['replays'] if item['room_code'] == newest)['favorited'])
        self.assertIn(oldest, [item['room_code'] for item in starred['replays']])
        self.assertEqual(starred['replays'][0]['room_code'], newest)
        recovered = self._get(f'/api/duel-v2/replay/{oldest}', token=token)['replay']
        self.assertEqual(recovered['room_code'], oldest)
        self.assertFalse(recovered['favorited'])
        outsider = self._issue_login_and_get_token('duel-v2-replay-star-out')
        self._post('/api/duel-v2/replay-star', {
            'room_code': newest, 'favorited': True,
        }, token=outsider, expected_status=404)


class DuelV2ReplayEngineTest(RoomFlowTestCase):
    def tearDown(self):
        try:
            super().tearDown()
        except OSError:
            pass

    def test_long_public_log_is_not_capped_at_observe_window(self):
        from app.modules.card_game.engine.application import v2_replay as replay_mod
        from app.modules.card_game.engine.application import v2_repository as repository

        token = self._issue_login_and_get_token('duel-v2-replay-long')
        started = self._post('/api/duel-v2/start', {'mode': 'solo'}, token=token)
        room = repository.current_room(self.dao_module.get_or_create_player('duel-v2-replay-long')[0])
        game = new_game(seed=7, skip_mulligan=True)
        game['sides']['a']['hp'] = 9999
        game['sides']['b']['hp'] = 9999
        opening_seq = int(game.get('event_seq') or 0)
        replay_mod.begin(room, game)
        for _ in range(800):
            if int(game.get('event_seq') or 0) - opening_seq > 220 or game.get('phase') == 'finished':
                break
            for side_id in ('a', 'b'):
                team = game['sides'][side_id]
                if len(team.get('deck') or []) < 4 and team.get('discard'):
                    team['deck'] = list(team.get('deck') or []) + list(team['discard'])
                    team['discard'] = []
            side = acting_side(game)
            if side is None:
                break
            game = apply_action(game, side, {'type': 'end_turn'})
        _mark_hidden(game)
        replay_mod.record(room, {'game': game})
        payload = self._get(f'/api/duel-v2/replay/{started["room"]["room_code"]}', token=token)['replay']
        self.assertGreater(len(payload['events']), 200)
        live = observe(game, 'a')
        self.assertLessEqual(len(live['presentation']['events']), 200)
        dumped = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(SECRET, dumped)
        self.assertNotIn('123456789', dumped)
        assembled = assemble_replay_game(payload['opening_board'], payload['events'])
        public_live = public_replay_view(game, 'a')
        self.assertEqual(assembled['sides']['a']['hp'], public_live['sides']['a']['hp'])
        self.assertEqual(assembled['sides']['b']['hp'], public_live['sides']['b']['hp'])
        self.assertEqual(assembled['turn'], public_live['turn'])
        self.assertEqual(leak_markers(payload), [])
        for card in assembled['sides']['b']['hand']:
            self.assertTrue(card.get('hidden') or card.get('copy'))
