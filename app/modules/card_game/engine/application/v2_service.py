"""Authenticated application orchestration; combat remains in duel_v2."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from secrets import randbits, token_hex

from app.db import atomic_transaction, db
from app.errors import PersistenceError, RuleValidationError
from app.modules.card_game.engine.application import v2_persistence as storage
from app.modules.card_game.engine.application import v2_replay as replay
from app.modules.card_game.engine.application import v2_repository as repository

_lobby_lock = threading.RLock()
_locks_guard = threading.Lock()
_room_locks: dict[tuple[str, int], threading.RLock] = {}
_RECEIPT_LIMIT = 128
_MAX_SAVED_BUILDS = 8
_MAX_BUILD_NAME = 16


class DuelV2NotFound(RuleValidationError):
    pass


class DuelV2Conflict(RuleValidationError):
    pass


def _content():
    from app.modules.card_game.content import duel_v2
    return duel_v2


def _solo_ai_deck() -> dict:
    content = _content()
    for preset in getattr(content, 'STARTER_DECKS', None) or []:
        if preset.get('id') == getattr(content, 'STARTER_DECK_ID', 'starter'):
            return _validate_deck(deepcopy(preset))
        if preset.get('name') == getattr(content, 'STARTER_DECK_NAME', '创生预组'):
            return _validate_deck(deepcopy(preset))
    return _validate_deck(deepcopy(content.STARTER_DECK))


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


def _clip_build_name(value) -> str:
    text = ' '.join(str(value or '').split())
    if not text:
        text = '自定义构筑'
    return text[:_MAX_BUILD_NAME]


def _new_build_id() -> str:
    return 'deck-' + token_hex(4)


def _normalize_store(raw) -> dict:
    granted = bool(isinstance(raw, dict) and raw.get('starter_granted'))
    if isinstance(raw, dict) and isinstance(raw.get('builds'), list):
        builds = []
        for item in raw.get('builds') or []:
            try:
                builds.append(_validate_deck(item))
            except (RuleValidationError, ValueError, TypeError, KeyError):
                continue
        active = raw.get('active_id')
        ids = {item['id'] for item in builds}
        if active not in ids:
            active = builds[0]['id'] if builds else None
        return {'active_id': active, 'builds': builds, 'starter_granted': granted}
    if isinstance(raw, dict) and raw.get('card_ids'):
        try:
            deck = _validate_deck(raw)
            return {'active_id': deck['id'], 'builds': [deck], 'starter_granted': granted}
        except (RuleValidationError, ValueError, TypeError, KeyError):
            pass
    return {'active_id': None, 'builds': [], 'starter_granted': granted}


def _preset_name(preset: dict) -> str:
    return str(preset.get('name') or '预组')


def _owned_preset_keys(store: dict) -> tuple[set[str], set[tuple[str, ...]]]:
    names = {item.get('name') for item in store['builds']}
    teams = {tuple(item.get('character_ids') or []) for item in store['builds']}
    return names, teams


def _grant_starter_decks(store: dict) -> bool:
    if store.get('starter_granted'):
        return False
    content = _content()
    presets = list(getattr(content, 'STARTER_DECKS', None) or [content.STARTER_DECK])
    owned_names, owned_teams = _owned_preset_keys(store)
    for preset in presets:
        if len(store['builds']) >= _MAX_SAVED_BUILDS:
            break
        name = _preset_name(preset)
        team = tuple(preset['character_ids'])
        if name in owned_names or team in owned_teams:
            continue
        store['builds'].append(_validate_deck({
            'id': _new_build_id(),
            'name': name,
            'character_ids': list(preset['character_ids']),
            'card_ids': list(preset['card_ids']),
        }))
        owned_names.add(name)
        owned_teams.add(team)
    if store['builds'] and not store.get('active_id'):
        store['active_id'] = store['builds'][0]['id']
    store['starter_granted'] = True
    return True


def _load_store(player) -> dict:
    with atomic_transaction():
        store = _normalize_store(repository.get_build(player))
        if _grant_starter_decks(store):
            repository.save_build(player, store)
        return store


def complete_starter_presets(player) -> dict:
    """Seed helper: keep existing decks and fill any missing official presets."""
    with atomic_transaction():
        store = _normalize_store(repository.get_build(player))
        store['starter_granted'] = False
        if _grant_starter_decks(store):
            repository.save_build(player, store)
        return store


def _public_store(store: dict) -> dict:
    active = next((item for item in store['builds'] if item['id'] == store.get('active_id')), None)
    return {
        'saved_build': deepcopy(active) if active else None,
        'saved_builds': deepcopy(store['builds']),
        'active_build_id': store.get('active_id'),
    }


def _selected_deck(player, supplied: dict | None = None) -> dict:
    if supplied is not None:
        return _validate_deck(supplied)
    store = _load_store(player)
    active = next((item for item in store['builds'] if item['id'] == store.get('active_id')), None)
    return _validate_deck(active if active is not None else deepcopy(_content().STARTER_DECK))


def catalog(player) -> dict:
    payload = _content().get_catalog()
    payload.update(_public_store(_load_store(player)))
    return payload


def save_build(player, payload: dict) -> dict:
    body = dict(payload or {})
    body['name'] = _clip_build_name(body.get('name'))
    if not body.get('id') or body.get('id') in ('starter', 'custom'):
        body['id'] = _new_build_id()
    deck = _validate_deck(body)
    with atomic_transaction():
        store = _load_store(player)
        existing = [item['id'] for item in store['builds']]
        if deck['id'] in existing:
            store['builds'] = [deck if item['id'] == deck['id'] else item for item in store['builds']]
        else:
            if len(store['builds']) >= _MAX_SAVED_BUILDS:
                raise RuleValidationError(f'最多保存 {_MAX_SAVED_BUILDS} 套构筑。')
            store['builds'].append(deck)
        store['active_id'] = deck['id']
        repository.save_build(player, store)
    return {'build': deck, **_public_store(store)}


def delete_build(player, payload: dict) -> dict:
    build_id = str((payload or {}).get('id') or '')
    if not build_id:
        raise RuleValidationError('缺少要删除的构筑。')
    with atomic_transaction():
        store = _load_store(player)
        kept = [item for item in store['builds'] if item['id'] != build_id]
        if len(kept) == len(store['builds']):
            raise DuelV2NotFound('没有这套构筑。')
        store['builds'] = kept
        if store.get('active_id') == build_id:
            store['active_id'] = kept[0]['id'] if kept else None
        repository.save_build(player, store)
    return {'ok': True, **_public_store(store)}


def select_build(player, payload: dict) -> dict:
    build_id = str((payload or {}).get('id') or '')
    with atomic_transaction():
        store = _load_store(player)
        if build_id not in {item['id'] for item in store['builds']}:
            raise DuelV2NotFound('没有这套构筑。')
        store['active_id'] = build_id
        repository.save_build(player, store)
    return {'ok': True, **_public_store(store)}


def reorder_builds(player, payload: dict) -> dict:
    raw_ids = (payload or {}).get('ids')
    if not isinstance(raw_ids, list):
        raise RuleValidationError('缺少构筑顺序。')
    ids = [str(item or '') for item in raw_ids]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise RuleValidationError('构筑顺序不完整。')
    with atomic_transaction():
        store = _load_store(player)
        current = [item['id'] for item in store['builds']]
        if set(ids) != set(current) or len(ids) != len(current):
            raise RuleValidationError('构筑顺序不完整。')
        by_id = {item['id']: item for item in store['builds']}
        store['builds'] = [by_id[item_id] for item_id in ids]
        repository.save_build(player, store)
    return {'ok': True, **_public_store(store)}


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
    has_replay = replay.available_for(room, player)
    favorited = False
    if has_replay:
        replay_row = repository.replay_by_code(room.room_code)
        favorited = bool(replay_row and repository.is_replay_starred(player, replay_row))
    return {'room': {'room_code': room.room_code, 'mode': room.mode, 'status': room.status,
                     'viewer_side': member.side, 'is_host': member.side == 'a',
                     'can_start': room.mode == 'pvp' and room.status == 'ready' and member.side == 'a',
                     'has_replay': has_replay,
                     'replay_code': room.room_code if has_replay else None,
                     'favorited': favorited,
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
        decks['b'] = _solo_ai_deck()
    game = _engine().new_game(seed=randbits(31), decks=decks)
    replay.begin(room, game)
    game = _advance_ai(game, room.mode)
    envelope = {'rules_version': 'duel_v2', 'game': game, 'requests': []}
    storage.initialize(room, envelope)
    replay.record(room, envelope)
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
        replay.record(room, updated)
        return _response(room, player, updated)


def list_replays(player) -> dict:
    return replay.list_replays(player)


def get_replay(player, room_code: str) -> dict:
    return replay.get_replay(player, room_code)


def _normalize_imported_replay(body: dict) -> dict:
    if not isinstance(body, dict):
        raise RuleValidationError('回放必须是 JSON 对象。')
    views = body.get('views')
    if isinstance(views, dict) and any(
        isinstance((views.get(side) or {}).get('opening_board'), dict) for side in ('a', 'b')
    ):
        return dict(body)
    source = body.get('replay') if isinstance(body.get('replay'), dict) else body
    opening = source.get('opening_board')
    events = source.get('events')
    if not isinstance(opening, dict) or not isinstance(events, list):
        raise RuleValidationError('回放需要 opening_board 与 events。')
    viewer = source.get('viewer_side') if source.get('viewer_side') in ('a', 'b') else 'a'
    return {
        'schema_version': 1,
        'name_a': source.get('name_a') or body.get('name_a') or '训练策略',
        'name_b': source.get('name_b') or body.get('name_b') or '规则AI',
        'winner': source.get('winner'),
        'learning_side': viewer,
        'source': source.get('source') or body.get('source') or 'rl-eval',
        'last_seq': source.get('last_seq') or 0,
        'views': {viewer: {'opening_board': opening, 'events': events}},
    }


def import_training_replay(player, body: dict) -> dict:
    payload = _normalize_imported_replay(body)
    leaks = replay.leak_markers(payload)
    if leaks:
        raise RuleValidationError('回放含有隐藏信息，已拒绝导入。')
    views = payload.get('views') or {}
    side = payload.get('learning_side') if payload.get('learning_side') in ('a', 'b') else 'a'
    if side not in views:
        side = 'a' if 'a' in views else 'b'
    if side not in views:
        raise RuleValidationError('回放缺少可观看的公开局面。')
    with atomic_transaction():
        room = repository.create_room(player, 'solo')
        repository.add_member(room, player, side, {}, True)
        repository.set_status(room, 'finished')
        stored = dict(payload)
        stored['room_code'] = room.room_code
        stored['mode'] = 'solo'
        stored['status'] = 'finished'
        stored['player_a_id'] = player.id if side == 'a' else None
        stored['player_b_id'] = player.id if side == 'b' else None
        stored['name_a'] = stored.get('name_a') or '训练策略'
        stored['name_b'] = stored.get('name_b') or '规则AI'
        stored['learning_side'] = side
        repository.upsert_replay(room, stored, {
            'mode': 'solo',
            'status': 'finished',
            'winner': stored.get('winner'),
            'player_a_id': stored.get('player_a_id'),
            'player_b_id': stored.get('player_b_id'),
            'name_a': stored['name_a'],
            'name_b': stored['name_b'],
        })
        row = repository.replay_by_code(room.room_code)
        if row is not None:
            repository.set_replay_star(player, row, True)
    replay.drop_cache(room.id)
    return replay.get_replay(player, room.room_code)


def star_replay(player, payload: dict) -> dict:
    room_code = str((payload or {}).get('room_code') or '').strip()
    favorited = (payload or {}).get('favorited')
    if favorited is None:
        favorited = True
    return replay.set_favorited(player, room_code, bool(favorited))


def leave(player) -> dict:
    room = _require_room(player)
    with _lobby_lock, _room_lock(room.id):
        room = repository.get_room(room.id)
        member = _require_member(room, player)
        envelope = storage.load(room.id)
        if envelope and envelope.get('game') and envelope['game'].get('phase') != 'finished':
            try:
                game = _engine().apply_action(envelope['game'], member.side, {'type': 'concede'})
                finished = {**envelope, 'game': game}
                storage.publish(room, finished, final=True)
                replay.record(room, finished)
            except (ValueError, RuleValidationError, PersistenceError):
                pass
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
