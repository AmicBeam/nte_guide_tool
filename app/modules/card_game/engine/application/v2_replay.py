"""Persist and serve complete public V2 replay logs; never expose hidden info."""
from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime

from app.db import atomic_transaction, db
from app.errors import PersistenceError
from app.modules.card_game.engine.application import v2_repository as repository
from app.modules.card_game.engine.duel_v2.replay_log import (
    append_projected_events,
    assemble_replay_game,
    label_sides,
    leak_markers,
    public_replay_view,
    strip_hidden,
)

_lock = threading.RLock()
_cache: dict[tuple[str, int], dict] = {}


def _key(room_id: int) -> tuple[str, int]:
    return str(db.database), room_id


def _member_name(member) -> str:
    player = member.player
    return player.nickname or player.player_uid


def _names_for(room, members, game=None) -> tuple[str, str]:
    by_side = {entry.side: entry for entry in members}
    name_a = _member_name(by_side['a']) if 'a' in by_side else '我方'
    if room.mode in {'solo', 'advanced'}:
        name_b = ('高级人机' if room.mode == 'advanced' else
                  ((game or {}).get('ai_profile') or {}).get('name', '练习对手'))
    else:
        name_b = _member_name(by_side['b']) if 'b' in by_side else '对手'
    return name_a, name_b


def _player_ids(members) -> tuple[int | None, int | None]:
    by_side = {entry.side: entry.player_id for entry in members}
    return by_side.get('a'), by_side.get('b')


def _label_sides(view: dict, name_a: str, name_b: str) -> dict:
    return label_sides(view, name_a, name_b)


def _new_payload(room, game: dict, members) -> dict:
    name_a, name_b = _names_for(room, members, game)
    player_a_id, player_b_id = _player_ids(members)
    views = {}
    for member in members:
        opening = public_replay_view(game, member.side)
        _label_sides(opening, name_a, name_b)
        views[member.side] = {
            'opening_board': opening,
            'events': [],
        }
        setup = [event for event in (game.get('events') or []) if event.get('type') == 'initiative']
        if setup:
            _append_view(views[member.side], game, member.side, setup, name_a, name_b)
    return {
        'schema_version': 1,
        'room_code': room.room_code,
        'mode': room.mode,
        'status': 'playing' if game.get('phase') != 'finished' else 'finished',
        'winner': game.get('winner'),
        'player_a_id': player_a_id,
        'player_b_id': player_b_id,
        'name_a': name_a,
        'name_b': name_b,
        'last_seq': int(game.get('event_seq') or 0),
        'views': views,
        'updated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }


def _append_view(view: dict, game: dict, viewer: str, raw_events: list, name_a: str, name_b: str) -> None:
    append_projected_events(view, game, viewer, raw_events, name_a, name_b)


def _append_payload(payload: dict, room, game: dict, members) -> dict:
    name_a, name_b = _names_for(room, members, game)
    player_a_id, player_b_id = _player_ids(members)
    payload['name_a'] = name_a
    payload['name_b'] = name_b
    payload['player_a_id'] = player_a_id
    payload['player_b_id'] = player_b_id
    payload['winner'] = game.get('winner')
    payload['status'] = 'finished' if game.get('phase') == 'finished' else 'playing'
    payload['mode'] = room.mode
    payload['room_code'] = room.room_code
    last_seq = int(payload.get('last_seq') or 0)
    raw_new = [event for event in (game.get('events') or []) if int(event.get('seq') or 0) > last_seq]
    views = payload.setdefault('views', {})
    for member in members:
        view = views.get(member.side)
        if view is None:
            opening = public_replay_view(game, member.side)
            _label_sides(opening, name_a, name_b)
            views[member.side] = {'opening_board': opening, 'events': []}
            continue
        _append_view(view, game, member.side, raw_new, name_a, name_b)
    payload['last_seq'] = int(game.get('event_seq') or last_seq)
    payload['updated_at'] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    return payload


def drop_cache(room_id: int) -> None:
    with _lock:
        _cache.pop(_key(room_id), None)


def _trim_participants(payload: dict) -> None:
    from app.models import Player
    for player_id in (payload.get('player_a_id'), payload.get('player_b_id')):
        if not player_id:
            continue
        player = Player.get_or_none(Player.id == player_id)
        if player is None:
            continue
        for room_id in repository.trim_unfavorited_replays(player):
            drop_cache(room_id)


def _persist(room, payload: dict) -> None:
    meta = {
        'status': payload.get('status') or 'playing',
        'winner': payload.get('winner'),
        'player_a_id': payload.get('player_a_id'),
        'player_b_id': payload.get('player_b_id'),
        'name_a': payload.get('name_a') or '',
        'name_b': payload.get('name_b') or '',
    }
    with atomic_transaction():
        repository.upsert_replay(room, payload, meta)
        if payload.get('status') == 'finished':
            _trim_participants(payload)


def begin(room, game: dict) -> dict:
    members = repository.members(room)
    payload = _new_payload(room, game, members)
    with _lock:
        _cache[_key(room.id)] = payload
    return payload


def record(room, envelope: dict) -> dict:
    game = envelope.get('game')
    if not isinstance(game, dict):
        raise PersistenceError('对局快照不存在，无法记录回放。')
    members = repository.members(room)
    with _lock:
        payload = _cache.get(_key(room.id))
        if payload is None:
            stored = repository.get_replay(room.id)
            payload = deepcopy(stored['payload']) if stored else _new_payload(room, game, members)
        payload = _append_payload(payload, room, game, members)
        _cache[_key(room.id)] = payload
    _persist(room, payload)
    return payload


def _payload_for(room_id: int) -> dict | None:
    with _lock:
        cached = _cache.get(_key(room_id))
        if cached is not None:
            return deepcopy(cached)
    stored = repository.get_replay(room_id)
    if stored is None:
        return None
    with _lock:
        _cache[_key(room_id)] = deepcopy(stored['payload'])
    return deepcopy(stored['payload'])


def _viewer_side(payload: dict, player) -> str | None:
    if payload.get('player_a_id') == player.id:
        return 'a'
    if payload.get('player_b_id') == player.id:
        return 'b'
    return None


def available_for(room, player) -> bool:
    payload = _payload_for(room.id)
    return bool(payload and _viewer_side(payload, player) and payload.get('views'))


def get_replay(player, room_code: str) -> dict:
    from app.modules.card_game.engine.application.v2_service import DuelV2NotFound

    if not isinstance(room_code, str) or not room_code.strip():
        raise DuelV2NotFound('未找到这场对局回放。')
    row = repository.replay_by_code(room_code.strip())
    if row is None:
        raise DuelV2NotFound('未找到这场对局回放。')
    payload = _payload_for(row.room_id)
    if payload is None:
        payload = repository.replay_payload(row)
    viewer = _viewer_side(payload, player)
    if viewer is None:
        raise DuelV2NotFound('未找到这场对局回放。')
    view = (payload.get('views') or {}).get(viewer)
    if not view:
        raise DuelV2NotFound('未找到这场对局回放。')
    opening = strip_hidden(deepcopy(view['opening_board']))
    events = strip_hidden(deepcopy(view['events']))
    leaks = leak_markers({'opening_board': opening, 'events': events})
    if leaks:
        raise PersistenceError('回放日志含有隐藏信息，已拒绝返回。')
    members = [
        {'side': 'a', 'name': payload.get('name_a') or '我方', 'is_self': viewer == 'a'},
        {'side': 'b', 'name': payload.get('name_b') or '对手', 'is_self': viewer == 'b'},
    ]
    if not repository.replay_kept_for(player, row):
        raise DuelV2NotFound('未找到这场对局回放。')
    favorited = repository.is_replay_starred(player, row)
    return {
        'room': {
            'room_code': payload.get('room_code') or row.room_code,
            'mode': payload.get('mode') or row.mode,
            'status': payload.get('status') or row.status,
            'viewer_side': viewer,
            'has_replay': True,
            'replay_code': payload.get('room_code') or row.room_code,
            'members': members,
        },
        'replay': {
            'schema_version': 1,
            'room_code': payload.get('room_code') or row.room_code,
            'mode': payload.get('mode') or row.mode,
            'status': payload.get('status') or row.status,
            'winner': payload.get('winner'),
            'viewer_side': viewer,
            'name_a': payload.get('name_a') or '我方',
            'name_b': payload.get('name_b') or '对手',
            'favorited': favorited,
            'last_seq': payload.get('last_seq') or 0,
            'event_count': len(events),
            'opening_board': opening,
            'events': events,
            'updated_at': payload.get('updated_at') or (row.updated_at.isoformat() + 'Z' if row.updated_at else None),
        },
    }


def _lineup(view: dict, side: str) -> list[dict]:
    team = ((view or {}).get('opening_board') or {}).get('sides') or {}
    source = team.get(side) or {}
    rows = []
    for item in (source.get('characters') or [])[:4]:
        if not isinstance(item, dict):
            continue
        rows.append({
            'id': item.get('id') or '',
            'name': item.get('name') or '',
            'avatar': item.get('avatar') or '',
        })
    return rows


def _turn_count(view: dict) -> int:
    events = view.get('events') or []
    turns = sum(1 for event in events if isinstance(event, dict) and event.get('type') == 'turn')
    if turns:
        return turns
    opening = view.get('opening_board') or {}
    turn = int(opening.get('turn') or 0)
    for event in events:
        if not isinstance(event, dict):
            continue
        patch = event.get('patch') or {}
        if patch.get('turn') is not None:
            turn = int(patch['turn'] or 0)
    return turn


def _summarize_replay(row, player) -> dict:
    viewer = 'a' if row.player_a_id == player.id else 'b'
    payload = repository.replay_payload(row)
    view = (payload.get('views') or {}).get(viewer) or {}
    return {
        'room_code': row.room_code,
        'mode': row.mode,
        'status': row.status,
        'winner': row.winner,
        'viewer_side': viewer,
        'name_a': row.name_a,
        'name_b': row.name_b,
        'favorited': repository.is_replay_starred(player, row),
        'characters_a': _lineup(view, 'a'),
        'characters_b': _lineup(view, 'b'),
        'event_count': len(view.get('events') or []),
        'turn_count': _turn_count(view),
        'updated_at': row.updated_at.isoformat() + 'Z' if row.updated_at else None,
    }


def list_replays(player, limit: int = 12) -> dict:
    del limit
    rows = repository.list_visible_replays_for(player)
    items = [_summarize_replay(row, player) for row in rows]
    return {
        'replays': items,
        'unfavorited_limit': repository.UNFAVORITED_REPLAY_LIMIT,
        'unfavorited_count': sum(1 for item in items if not item.get('favorited')),
    }


def set_favorited(player, room_code: str, favorited: bool) -> dict:
    from app.modules.card_game.engine.application.v2_service import DuelV2NotFound

    if not isinstance(room_code, str) or not room_code.strip():
        raise DuelV2NotFound('未找到这场对局回放。')
    row = repository.replay_by_code(room_code.strip())
    if row is None:
        raise DuelV2NotFound('未找到这场对局回放。')
    payload = _payload_for(row.room_id) or repository.replay_payload(row)
    if _viewer_side(payload, player) is None:
        raise DuelV2NotFound('未找到这场对局回放。')
    with atomic_transaction():
        repository.set_replay_star(player, row, bool(favorited))
        for room_id in repository.trim_unfavorited_replays(player):
            drop_cache(room_id)
    return list_replays(player)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
