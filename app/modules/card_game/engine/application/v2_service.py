"""Authenticated application orchestration; combat remains in duel_v2."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from secrets import randbits

from app.db import atomic_transaction, db
from app.errors import PersistenceError, RuleValidationError
from app.modules.card_game.engine.application import v2_persistence as storage
from app.modules.card_game.engine.application import v2_repository as repository

_lobby_lock = threading.RLock()
_locks_guard = threading.Lock()
_room_locks: dict[tuple[str, int], threading.RLock] = {}
_RECEIPT_LIMIT = 128


class DuelV2NotFound(RuleValidationError):
    pass


class DuelV2Conflict(RuleValidationError):
    pass


def _content():
    from app.modules.card_game.content import duel_v2
    return duel_v2


def _engine():
    from app.modules.card_game.engine import duel_v2
    return duel_v2


def _room_lock(room_id: int):
    key = str(db.database), room_id
    with _locks_guard:
        return _room_locks.setdefault(key, threading.RLock())


def _validate_deck(payload: dict) -> dict:
    try:
        return _content().validate_deck(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise RuleValidationError(str(exc)) from exc


def _selected_deck(player, supplied: dict | None = None) -> dict:
    if supplied is not None:
        return _validate_deck(supplied)
    saved = repository.get_build(player)
    return _validate_deck(saved if saved is not None else deepcopy(_content().STARTER_DECK))


def catalog(player) -> dict:
    payload = _content().get_catalog()
    saved = repository.get_build(player)
    payload['saved_build'] = _validate_deck(saved) if saved is not None else None
    return payload


def save_build(player, payload: dict) -> dict:
    deck = _validate_deck(payload)
    with atomic_transaction():
        repository.save_build(player, deck)
    return {'build': deck}


def _require_room(player):
    room = repository.current_room(player)
    if room is None:
        raise DuelV2NotFound('当前没有新版房间，请先开始对局或加入房间。')
    return room


def _require_member(room, player):
    member = repository.member_for(room, player)
    if member is None:
        raise DuelV2NotFound('你已离开这个房间。')
    return member


def _response(room, player, envelope: dict | None = None, after_seq=None) -> dict:
    member = _require_member(room, player)
    members = repository.members(room)
    if envelope is None:
        envelope = storage.load(room.id)
    game = _engine().observe(envelope['game'], member.side, after_seq=after_seq) if envelope else None
    people = []
    for entry in members:
        name = entry.player.nickname or entry.player.player_uid
        people.append({'side': entry.side, 'name': name, 'nickname': name,
                       'is_host': entry.side == 'a', 'is_self': entry.player_id == player.id,
                       'is_ready': entry.is_ready})
        if game and entry.side in game['sides']:
            game['sides'][entry.side]['name'] = name
    if game and room.mode == 'solo' and 'b' in game['sides']:
        game['sides']['b']['name'] = '练习对手'
    return {'room': {'room_code': room.room_code, 'mode': room.mode, 'status': room.status,
                     'viewer_side': member.side, 'is_host': member.side == 'a',
                     'can_start': room.mode == 'pvp' and room.status == 'ready' and member.side == 'a',
                     'members': people}, 'game': game}


def get_state(player, after_seq=None) -> dict:
    room = _require_room(player)
    with _room_lock(room.id):
        room = repository.get_room(room.id)
        if room is None or room.status == 'closed':
            raise DuelV2NotFound('房间已经关闭。')
        return _response(room, player, after_seq=after_seq)


def _advance_ai(game: dict, mode: str) -> dict:
    if mode != 'solo':
        return game
    engine = _engine()
    for _ in range(32):
        if game.get('phase') == 'finished' or engine.acting_side(game) != 'b':
            return game
        action = engine.choose_action(game, 'b')
        version = game['version']
        updated = engine.apply_action(game, 'b', action)
        if updated['version'] <= version:
            raise PersistenceError('练习对手未能完成动作，请刷新后重试。')
        game = updated
    raise PersistenceError('练习对手动作超出单回合限制，请刷新后重试。')


def _initialize_game(room) -> dict:
    people = repository.members(room)
    decks = {entry.side: _validate_deck(repository.member_deck(entry)) for entry in people}
    if room.mode == 'solo':
        decks['b'] = deepcopy(_content().STARTER_DECK)
    game = _engine().new_game(seed=randbits(31), decks=decks)
    game = _advance_ai(game, room.mode)
    envelope = {'rules_version': 'duel_v2', 'game': game, 'requests': []}
    storage.initialize(room, envelope)
    repository.set_status(room, 'playing' if game['phase'] != 'finished' else 'finished')
    return envelope


def start(player, payload: dict) -> dict:
    mode = payload.get('mode', 'solo')
    if mode not in {'solo', 'pvp'}:
        raise RuleValidationError('请选择人机或双人模式。')
    deck = _selected_deck(player, payload.get('deck'))
    with _lobby_lock:
        existing = repository.current_room(player)
        if existing is not None and existing.status in {'waiting', 'ready', 'playing'}:
            if existing.mode != mode:
                raise DuelV2Conflict('请先退出当前房间，再切换模式。')
            return get_state(player)
        with atomic_transaction():
            room = repository.create_room(player, mode)
            repository.add_member(room, player, 'a', deck, ready=(mode == 'solo'))
            envelope = _initialize_game(room) if mode == 'solo' else None
        return _response(room, player, envelope)


def join(player, payload: dict) -> dict:
    code = payload.get('room_code', '')
    if not isinstance(code, str) or not re.fullmatch(r'[0-9A-Fa-f]{6}', code.strip()):
        raise RuleValidationError('请输入 6 位房间码。')
    with _lobby_lock:
        existing = repository.current_room(player)
        room = repository.room_by_code(code.strip())
        if room is None or room.status == 'closed':
            raise DuelV2NotFound('未找到这个新版房间。')
        if existing is not None and existing.status in {'waiting', 'ready', 'playing'}:
            if existing.id == room.id:
                return _response(room, player)
            raise DuelV2Conflict('请先退出当前房间。')
        if room.mode != 'pvp' or room.status not in {'waiting', 'ready'}:
            raise RuleValidationError('该房间当前不能加入。')
        with _room_lock(room.id), atomic_transaction():
            room = repository.get_room(room.id)
            if room is None or room.mode != 'pvp' or room.status not in {'waiting', 'ready'}:
                raise RuleValidationError('该房间当前不能加入。')
            if len(repository.members(room)) >= 2:
                raise RuleValidationError('房间已满。')
            repository.add_member(room, player, 'b', _selected_deck(player), False)
            repository.set_status(room, 'waiting')
        return _response(room, player)


def ready(player, payload: dict) -> dict:
    value = payload.get('is_ready', True)
    if not isinstance(value, bool):
        raise RuleValidationError('准备状态必须为 true 或 false。')
    room = _require_room(player)
    with _room_lock(room.id), atomic_transaction():
        room = repository.get_room(room.id)
        if room.mode != 'pvp' or room.status not in {'waiting', 'ready'}:
            raise RuleValidationError('当前房间不能调整准备状态。')
        member = _require_member(room, player)
        repository.set_ready(member, value, _selected_deck(player) if value else None)
        members = repository.members(room)
        repository.set_status(room, 'ready' if len(members) == 2 and all(m.is_ready for m in members) else 'waiting')
        return _response(room, player)


def start_room(player) -> dict:
    room = _require_room(player)
    with _room_lock(room.id), atomic_transaction():
        room = repository.get_room(room.id)
        member = _require_member(room, player)
        if member.side != 'a':
            raise RuleValidationError('只有房主可以开始。')
        if room.status == 'playing':
            return _response(room, player)
        members = repository.members(room)
        if room.mode != 'pvp' or room.status != 'ready' or len(members) != 2 or not all(m.is_ready for m in members):
            raise RuleValidationError('请等待双方准备完成。')
        envelope = _initialize_game(room)
        return _response(room, player, envelope)


def submit_action(player, payload: dict) -> dict:
    request_id = payload.get('request_id')
    expected = payload.get('expected_version')
    action = payload.get('action')
    if not isinstance(request_id, str) or not (1 <= len(request_id) <= 128):
        raise RuleValidationError('缺少有效的动作请求编号。')
    if type(expected) is not int or expected < 0:
        raise RuleValidationError('缺少有效的对局版本。')
    if not isinstance(action, dict) or not isinstance(action.get('type'), str):
        raise RuleValidationError('动作格式不正确。')
    try:
        encoded = json.dumps(action, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise RuleValidationError('动作内容必须为有效 JSON。') from exc
    if len(encoded) > 16000:
        raise RuleValidationError('动作内容过长。')
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    room = _require_room(player)
    requested_room = payload.get('room_code')
    if not isinstance(requested_room, str) or not requested_room:
        raise RuleValidationError('缺少动作所属房间，请刷新后再操作。')
    with _room_lock(room.id):
        room = repository.get_room(room.id)
        member = _require_member(room, player)
        if requested_room != room.room_code:
            raise DuelV2Conflict('该操作属于之前的房间，请刷新后再操作。')
        envelope = storage.load(room.id)
        if envelope is None:
            raise RuleValidationError('对局尚未开始。')
        for receipt in envelope.get('requests', []):
            if receipt['side'] == member.side and receipt['id'] == request_id:
                if receipt['fingerprint'] != fingerprint:
                    raise DuelV2Conflict('同一请求编号不能用于不同动作。')
                response = _response(room, player, envelope)
                response['replayed_request'] = True
                return response
        if room.status == 'closed':
            raise DuelV2NotFound('房间已经关闭。')
        if envelope['game']['version'] != expected:
            raise DuelV2Conflict('对局已更新，请刷新后再操作。')
        try:
            game = _engine().apply_action(envelope['game'], member.side, action)
            game = _advance_ai(game, room.mode)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuleValidationError(str(exc)) from exc
        receipts = [*envelope.get('requests', []),
                    {'side': member.side, 'id': request_id, 'fingerprint': fingerprint}][-_RECEIPT_LIMIT:]
        updated = {'rules_version': 'duel_v2', 'game': game, 'requests': receipts}
        storage.publish(room, updated, final=game['phase'] == 'finished')
        return _response(room, player, updated)


def leave(player) -> dict:
    room = _require_room(player)
    with _lobby_lock, _room_lock(room.id):
        room = repository.get_room(room.id)
        member = _require_member(room, player)
        envelope = storage.load(room.id)
        if envelope and envelope['game']['phase'] != 'finished':
            game = _engine().apply_action(envelope['game'], member.side, {'type': 'concede'})
            storage.publish(room, {**envelope, 'game': game}, final=True)
        with atomic_transaction():
            if room.status in {'waiting', 'ready'} and member.side == 'a':
                repository.clear_members(room)
            else:
                repository.remove_member(member)
            remaining = repository.members(room)
            if not remaining:
                repository.set_status(room, 'closed')
                repository.invalidate_run(room)
                storage.invalidate(room.id)
            elif room.status in {'waiting', 'ready'}:
                for other in remaining:
                    repository.set_ready(other, False)
                repository.set_status(room, 'waiting')
    return {'ok': True}
