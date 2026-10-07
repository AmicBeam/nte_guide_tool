"""Authenticated application orchestration; combat remains in duel_v2."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from datetime import datetime
from secrets import randbits, token_hex

from app.db import atomic_transaction, db
from app.errors import PersistenceError, RuleValidationError
from app.models import DuelV2TutorialProgress
from app.modules.card_game.engine.application import v2_persistence as storage
from app.modules.card_game.engine.application import v2_replay as replay
from app.modules.card_game.engine.application import v2_repository as repository
from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS, LEVEL_META, load_scenario, next_level

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


def _solo_ai_deck(player=None, preset_id=None) -> dict:
    if preset_id not in (None, 'random'):
        if not isinstance(preset_id, str):
            raise RuleValidationError('请选择普通人机的对战预组。')
        selected = next((item for item in _allowed_presets(player) if item['id'] == preset_id), None)
        if selected is None:
            raise RuleValidationError('该人机预组不存在或尚未公开。')
        return _validate_deck(deepcopy(selected))
    from secrets import choice
    content = _content()
    presets = [preset for preset in getattr(content, 'STARTER_DECKS', None) or [content.STARTER_DECK]
               if not content.deck_uses_test_characters(preset)]
    if not presets:
        raise RuleValidationError('暂无可用的公开预组。')
    return _validate_deck(deepcopy(choice(presets)))


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


def _player_can_use_test_content(player) -> bool:
    return bool(player and getattr(player, 'shaft_test_whitelisted', False))


def _deck_visible(deck: dict, player) -> bool:
    if _player_can_use_test_content(player):
        return True
    return not _content().deck_uses_test_characters(deck)


def _require_visible_deck(player, deck: dict, *, joining: bool = False) -> dict:
    if _deck_visible(deck, player):
        return deck
    raise RuleValidationError('该房间使用尚未公开的角色。' if joining else '该构筑包含尚未公开的角色。')


def _require_public_deck(deck: dict, *, joining: bool = False) -> dict:
    if _content().deck_uses_test_characters(deck):
        raise RuleValidationError('该房间使用尚未公开的角色。' if joining else '高级人机仅支持公开角色构筑。')
    return deck


def _allowed_presets(player) -> list[dict]:
    content = _content()
    presets = list(getattr(content, 'STARTER_DECKS', None) or [content.STARTER_DECK])
    if _player_can_use_test_content(player):
        return presets
    return [item for item in presets if not content.deck_uses_test_characters(item)]


def _clip_build_name(value) -> str:
    text = ' '.join(str(value or '').split())
    if not text:
        text = '自定义构筑'
    return text[:_MAX_BUILD_NAME]


def _new_build_id() -> str:
    return 'deck-' + token_hex(4)


def _granted_preset_ids(raw) -> list[str]:
    if not isinstance(raw, dict) or not isinstance(raw.get('granted_preset_ids'), list):
        return []
    return [str(item) for item in raw.get('granted_preset_ids') or [] if item]


def _normalize_store(raw) -> dict:
    granted = bool(isinstance(raw, dict) and raw.get('starter_granted'))
    granted_preset_ids = _granted_preset_ids(raw)
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
        return {
            'active_id': active,
            'builds': builds,
            'starter_granted': granted,
            'granted_preset_ids': granted_preset_ids,
        }
    if isinstance(raw, dict) and raw.get('card_ids'):
        try:
            deck = _validate_deck(raw)
            return {
                'active_id': deck['id'],
                'builds': [deck],
                'starter_granted': granted,
                'granted_preset_ids': granted_preset_ids,
            }
        except (RuleValidationError, ValueError, TypeError, KeyError):
            pass
    return {
        'active_id': None,
        'builds': [],
        'starter_granted': granted,
        'granted_preset_ids': granted_preset_ids,
    }


def _preset_name(preset: dict) -> str:
    return str(preset.get('name') or '预组')


def _owned_preset_keys(store: dict) -> tuple[set[str], set[tuple[str, ...]]]:
    names = {item.get('name') for item in store['builds']}
    teams = {tuple(item.get('character_ids') or []) for item in store['builds']}
    return names, teams


def _grant_starter_decks(store: dict, player=None) -> bool:
    content = _content()
    all_presets = list(getattr(content, 'STARTER_DECKS', None) or [content.STARTER_DECK])
    allowed = _allowed_presets(player)
    from .v2_preset_migrations import retire_crimson
    migrated = retire_crimson(store, allowed)
    granted_ids = {str(item) for item in store.get('granted_preset_ids') or [] if item}
    changed = migrated
    if store.get('starter_granted') and not granted_ids:
        granted_ids = {str(preset.get('id') or '') for preset in all_presets}
        granted_ids.discard('')
        changed = True
    owned_names, owned_teams = _owned_preset_keys(store)
    for preset in allowed:
        preset_id = str(preset.get('id') or '')
        if not preset_id or preset_id in granted_ids:
            continue
        granted_ids.add(preset_id)
        changed = True
        if len(store['builds']) >= _MAX_SAVED_BUILDS:
            continue
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
        changed = True
    granted_list = sorted(granted_ids)
    if list(store.get('granted_preset_ids') or []) != granted_list:
        store['granted_preset_ids'] = granted_list
        changed = True
    if not store.get('starter_granted'):
        store['starter_granted'] = True
        changed = True
    return changed


def _visible_store(store: dict, player) -> dict:
    builds = [item for item in store['builds'] if _deck_visible(item, player)]
    active = store.get('active_id')
    ids = {item['id'] for item in builds}
    if active not in ids:
        active = builds[0]['id'] if builds else None
    return {
        'active_id': active,
        'builds': builds,
        'starter_granted': store.get('starter_granted'),
        'granted_preset_ids': list(store.get('granted_preset_ids') or []),
    }


def _load_store(player) -> dict:
    with atomic_transaction():
        store = _normalize_store(repository.get_build(player))
        changed = _grant_starter_decks(store, player)
        visible = _visible_store(store, player)
        if store.get('active_id') != visible.get('active_id'):
            store['active_id'] = visible.get('active_id')
            changed = True
        if changed:
            repository.save_build(player, store)
        return store


def complete_starter_presets(player) -> dict:
    """Seed helper: keep existing decks and fill any missing official presets."""
    with atomic_transaction():
        store = _normalize_store(repository.get_build(player))
        store['starter_granted'] = False
        store['granted_preset_ids'] = []
        if _grant_starter_decks(store, player):
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
        return _require_visible_deck(player, _validate_deck(supplied))
    store = _visible_store(_load_store(player), player)
    active = next((item for item in store['builds'] if item['id'] == store.get('active_id')), None)
    fallback = deepcopy((_allowed_presets(player) or [_content().STARTER_DECK])[0])
    return _validate_deck(active if active is not None else fallback)


def catalog(player) -> dict:
    payload = _content().get_catalog(include_test_characters=_player_can_use_test_content(player))
    payload.update(_public_store(_visible_store(_load_store(player), player)))
    payload['tutorial'] = tutorial_progress(player)
    return payload


def _progress_row(player) -> DuelV2TutorialProgress:
    row = DuelV2TutorialProgress.get_or_none(DuelV2TutorialProgress.player == player)
    if row is None:
        row = DuelV2TutorialProgress.create(player=player)
    return row


def _completed_levels(row) -> list[str]:
    try:
        payload = json.loads(row.completed_levels or '[]')
    except (TypeError, ValueError):
        payload = []
    return [str(item) for item in payload if item]


def tutorial_progress(player) -> dict:
    row = _progress_row(player)
    completed = _completed_levels(row)
    current = row.current_level or CAMPAIGN_LEVELS[0]
    campaign_complete = all(level in completed for level in CAMPAIGN_LEVELS)
    if current not in CAMPAIGN_LEVELS or (row.completed and not row.skipped and not campaign_complete):
        current = next_level(completed)
    playable = True
    try:
        load_scenario(current)
    except ValueError:
        playable = current in completed
    levels = []
    for index, meta in enumerate(LEVEL_META, start=1):
        scenario_id = meta['id']
        unlocked = scenario_id in completed or scenario_id == current or scenario_id == CAMPAIGN_LEVELS[0]
        available = False
        if unlocked:
            try:
                load_scenario(scenario_id)
                available = True
            except ValueError:
                available = False
        levels.append({
            'id': scenario_id,
            'title': meta['title'],
            'short': meta.get('short') or meta['title'],
            'index': index,
            'completed': scenario_id in completed,
            'unlocked': unlocked,
            'available': available,
        })
    return {
        'campaign_id': 'duel_v2',
        'current_level': current,
        'completed_levels': completed,
        'skipped': bool(row.skipped),
        'skipped_prompt': bool(row.skipped_prompt),
        'completed': bool(row.skipped or campaign_complete),
        'playable': playable,
        'level_count': len(CAMPAIGN_LEVELS),
        'levels': levels,
    }


def skip_tutorial(player, payload: dict | None = None) -> dict:
    body = payload or {}
    prompt_only = bool(body.get('skipped_prompt'))
    with atomic_transaction():
        row = _progress_row(player)
        row.skipped_prompt = True
        if not prompt_only:
            row.skipped = True
            row.completed = True
        row.updated_at = datetime.utcnow()
        row.save()
    if not prompt_only:
        try:
            room = repository.current_room(player)
            if room is not None and room.mode == 'tutorial':
                repository.set_status(room, 'closed')
                repository.invalidate_run(room)
                storage.invalidate(room.id)
        except Exception:
            pass
    return tutorial_progress(player)


def _enter_tutorial_level(player, scenario_id: str) -> None:
    if scenario_id not in CAMPAIGN_LEVELS:
        return
    with atomic_transaction():
        row = _progress_row(player)
        completed = _completed_levels(row)
        if scenario_id in completed:
            return
        index = CAMPAIGN_LEVELS.index(scenario_id)
        for prior in CAMPAIGN_LEVELS[:index]:
            if prior not in completed:
                completed.append(prior)
        row.completed_levels = json.dumps(completed, ensure_ascii=False)
        row.current_level = scenario_id
        row.completed = len(completed) >= len(CAMPAIGN_LEVELS)
        row.updated_at = datetime.utcnow()
        row.save()


def _close_tutorial_room(room) -> None:
    if room is None or room.mode != 'tutorial':
        return
    repository.set_status(room, 'closed')
    repository.invalidate_run(room)
    storage.invalidate(room.id)


def _mark_tutorial_level(player, scenario_id: str) -> None:
    with atomic_transaction():
        row = _progress_row(player)
        completed = _completed_levels(row)
        if scenario_id not in completed:
            completed.append(scenario_id)
        nxt = next_level(completed)
        row.completed_levels = json.dumps(completed, ensure_ascii=False)
        row.current_level = nxt
        row.completed = len(completed) >= len(CAMPAIGN_LEVELS)
        row.updated_at = datetime.utcnow()
        row.save()


def _allowed_tutorial_scenarios(player) -> set[str]:
    progress = tutorial_progress(player)
    allowed = set(progress['completed_levels'])
    allowed.add(progress['current_level'])
    allowed.add(CAMPAIGN_LEVELS[0])
    return {item for item in allowed if item in CAMPAIGN_LEVELS}


def import_build_text(player, payload: dict) -> dict:
    from . import v2_build_text
    deck = _require_visible_deck(player, v2_build_text.parse(payload.get('text')))
    deck['id'] = ''
    return {'build': deck}


def export_build_text(player, payload: dict) -> dict:
    from . import v2_build_text
    body = dict(payload)
    body['id'] = 'export'
    body['name'] = _clip_build_name(body.get('name'))
    deck = _require_visible_deck(player, _validate_deck(body))
    return {'text': v2_build_text.format_build(deck)}


def save_build(player, payload: dict) -> dict:
    body = dict(payload or {})
    body['name'] = _clip_build_name(body.get('name'))
    if not body.get('id') or body.get('id') in ('starter', 'custom'):
        body['id'] = _new_build_id()
    deck = _require_visible_deck(player, _validate_deck(body))
    with atomic_transaction():
        store = _load_store(player)
        existing = [item['id'] for item in store['builds']]
        visible_count = len(_visible_store(store, player)['builds'])
        if deck['id'] in existing:
            store['builds'] = [deck if item['id'] == deck['id'] else item for item in store['builds']]
        else:
            if visible_count >= _MAX_SAVED_BUILDS:
                raise RuleValidationError(f'最多保存 {_MAX_SAVED_BUILDS} 套构筑。')
            store['builds'].append(deck)
        store['active_id'] = deck['id']
        repository.save_build(player, store)
    return {'build': deck, **_public_store(_visible_store(store, player))}


def delete_build(player, payload: dict) -> dict:
    build_id = str((payload or {}).get('id') or '')
    if not build_id:
        raise RuleValidationError('缺少要删除的构筑。')
    with atomic_transaction():
        store = _load_store(player)
        visible_ids = {item['id'] for item in _visible_store(store, player)['builds']}
        if build_id not in visible_ids:
            raise DuelV2NotFound('没有这套构筑。')
        kept = [item for item in store['builds'] if item['id'] != build_id]
        store['builds'] = kept
        visible = _visible_store(store, player)
        store['active_id'] = visible.get('active_id')
        repository.save_build(player, store)
    return {'ok': True, **_public_store(visible)}


def select_build(player, payload: dict) -> dict:
    build_id = str((payload or {}).get('id') or '')
    with atomic_transaction():
        store = _load_store(player)
        visible = _visible_store(store, player)
        if build_id not in {item['id'] for item in visible['builds']}:
            raise DuelV2NotFound('没有这套构筑。')
        store['active_id'] = build_id
        repository.save_build(player, store)
    return {'ok': True, **_public_store(_visible_store(store, player))}


def reorder_builds(player, payload: dict) -> dict:
    raw_ids = (payload or {}).get('ids')
    if not isinstance(raw_ids, list):
        raise RuleValidationError('缺少构筑顺序。')
    ids = [str(item or '') for item in raw_ids]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise RuleValidationError('构筑顺序不完整。')
    with atomic_transaction():
        store = _load_store(player)
        visible = _visible_store(store, player)
        hidden = [item for item in store['builds'] if item['id'] not in {entry['id'] for entry in visible['builds']}]
        current = [item['id'] for item in visible['builds']]
        if set(ids) != set(current) or len(ids) != len(current):
            raise RuleValidationError('构筑顺序不完整。')
        by_id = {item['id']: item for item in visible['builds']}
        store['builds'] = [by_id[item_id] for item_id in ids] + hidden
        repository.save_build(player, store)
    return {'ok': True, **_public_store(_visible_store(store, player))}


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
    if game and room.mode in {'solo', 'advanced'} and 'b' in game['sides']:
        game['sides']['b']['name'] = (('实验性高级人机' if (envelope['game'].get('ai_profile') or {}).get('experimental') else '高级人机') if room.mode == 'advanced' else
                                    ((envelope['game'].get('ai_profile') or {}).get('name', '练习对手')))
    if game and room.mode == 'tutorial' and 'b' in game['sides']:
        game['sides']['b']['name'] = '对手'
    if game and room.mode == 'tutorial' and game.get('tutorial'):
        progress = tutorial_progress(player)
        nxt = None
        if game.get('phase') == 'finished' and game.get('winner') == member.side:
            candidate = progress.get('current_level')
            if candidate and candidate != game['tutorial'].get('scenario'):
                try:
                    load_scenario(candidate)
                    nxt = candidate
                except ValueError:
                    nxt = None
        game['tutorial']['next_scenario'] = nxt
        game['tutorial']['campaign_complete'] = bool(progress.get('completed'))
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
                     'ai_pending': bool(envelope and room.mode == 'advanced'
                                        and envelope['game']['phase'] != 'finished'
                                        and _engine().acting_side(envelope['game']) == 'b'),
                     'members': people}, 'game': game}


def get_state(player, after_seq=None) -> dict:
    room = _require_room(player)
    with _room_lock(room.id):
        room = repository.get_room(room.id)
        if room is None or room.status == 'closed':
            raise DuelV2NotFound('房间已经关闭。')
        return _response(room, player, after_seq=after_seq)


def _advance_ai(game: dict, mode: str) -> dict:
    engine = _engine()
    if mode == 'advanced':
        # Advanced actions are individually committed by submit_ai_step so
        # the client can present each result before requesting the next one.
        return game
    if mode == 'tutorial':
        from app.modules.card_game.engine.duel_v2.tutorial import run_scripted_opponent
        return run_scripted_opponent(game, engine.apply_action)
    if mode != 'solo':
        return game
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


def _initialize_game(room, scenario=None, ai_deck_id=None, ai_model=None, solo_deck=None) -> dict:
    if room.mode == 'tutorial':
        game = _engine().new_game(seed=0, scenario=scenario or 'tutorial_l01_board')
        game = _advance_ai(game, room.mode)
        envelope = {'rules_version': 'duel_v2', 'game': game, 'requests': []}
        storage.initialize(room, envelope)
        repository.set_status(room, 'playing' if game['phase'] != 'finished' else 'finished')
        return envelope
    people = repository.members(room)
    decks = {entry.side: _validate_deck(repository.member_deck(entry)) for entry in people}
    if room.mode == 'solo':
        decks['b'] = deepcopy(solo_deck) if solo_deck is not None else _solo_ai_deck()
    if room.mode == 'advanced':
        from app.modules.card_game.engine.ai.advanced_model import load_model, serving_deck, require_human_deck
        ai_model = ai_model if ai_model is not None else load_model(ai_deck_id)
        decks['a'] = _require_public_deck(_validate_deck(decks['a']))
        decks['b'] = _validate_deck(serving_deck(ai_model, ai_deck_id))
        require_human_deck(ai_model, decks['a'], ai_deck_id)
    game = _engine().new_game(seed=randbits(31), decks=decks)
    if room.mode == 'solo' and solo_deck is not None:
        game['ai_profile'] = {'kind': 'rule', 'deck_id': solo_deck['id'], 'name': solo_deck['name']}
        game['sides']['b']['name'] = solo_deck['name']
    if room.mode == 'advanced':
        model = ai_model
        from app.modules.card_game.engine.ai.advanced_model import model_build_hash
        game['ai_profile'] = {
            'deck_id': ai_deck_id,
            'version': model.version,
            'build_sha256': model_build_hash(model, decks['b']),
            'experimental': bool(getattr(model, 'experimental', False)),
            'capability': dict(getattr(model, 'capability', {}) or {}),
        }
        game['sides']['b']['name'] = '实验性高级人机' if getattr(model, 'experimental', False) else '高级人机'
    replay.begin(room, game)
    game = _advance_ai(game, room.mode)
    envelope = {'rules_version': 'duel_v2', 'game': game, 'requests': []}
    storage.initialize(room, envelope)
    replay.record(room, envelope)
    repository.set_status(room, 'playing' if game['phase'] != 'finished' else 'finished')
    return envelope


def start(player, payload: dict) -> dict:
    with _lobby_lock:
        return _start_locked(player, payload)


def _start_locked(player, payload: dict) -> dict:
    mode = payload.get('mode', 'solo')
    if mode not in {'solo', 'advanced', 'pvp', 'tutorial'}:
        raise RuleValidationError('请选择人机或双人模式。')
    requested_ai = payload.get('ai_deck', 'random')
    if mode in {'solo', 'advanced'} and not isinstance(requested_ai, str):
        raise RuleValidationError('请选择人机的对手套牌。')
    # Resume before choosing a random opponent or reading a changed saved build.
    existing = repository.current_room(player)
    if mode != 'tutorial' and existing is not None and existing.status in {'waiting', 'ready', 'playing'}:
        if existing.mode != mode:
            raise DuelV2Conflict('请先退出当前房间，再切换模式。')
        if mode in {'solo', 'advanced'} and requested_ai != 'random':
            active = storage.load(existing.id)
            profile = ((active or {}).get('game') or {}).get('ai_profile') or {}
            if profile.get('deck_id') != requested_ai:
                label = '高级人机预组' if mode == 'advanced' else '人机预组'
                raise DuelV2Conflict(f'请先离开当前对局，再更换{label}。')
        return get_state(player)
    ai_deck_id = None
    ai_model = None
    solo_deck = (_solo_ai_deck(player, payload['ai_deck'])
                 if mode == 'solo' and requested_ai not in ('random', 'mirror') else None)
    if mode == 'advanced':
        from app.modules.card_game.engine.ai.advanced_model import load_model, serving_deck
        ai_deck_id = requested_ai
        if ai_deck_id == 'random':
            from secrets import choice
            from app.modules.card_game.engine.ai.advanced_model import RANDOM_PRESETS, UNTRAINED_STRATEGIES
            available = []
            for candidate in RANDOM_PRESETS:
                if candidate in UNTRAINED_STRATEGIES:
                    continue
                try:
                    available.append((candidate, load_model(candidate)))
                except RuleValidationError:
                    continue
            if not available:
                raise RuleValidationError('暂无可用的高级人机，请稍后重试或选择普通人机。')
            ai_deck_id, ai_model = choice(available)
        else:
            ai_model = load_model(ai_deck_id)  # Pin the validated pair for this room.

    scenario = payload.get('scenario')
    if mode == 'tutorial':
        progress = tutorial_progress(player)
        scenario = str(scenario or progress['current_level'] or CAMPAIGN_LEVELS[0])
        if scenario not in _allowed_tutorial_scenarios(player):
            raise DuelV2Conflict('请按顺序进行新手教学。')
        try:
            load_scenario(scenario)
        except ValueError as exc:
            raise RuleValidationError(str(exc)) from exc
    if mode == 'advanced':
        player_deck_mode = payload.get('ai_player_deck', 'saved')
        if player_deck_mode not in ('mirror', 'saved'):
            raise RuleValidationError('请选择我方构筑。')
        deck = (_validate_deck(serving_deck(ai_model,ai_deck_id)) if player_deck_mode == 'mirror'
                else _selected_deck(player))
        deck = _require_public_deck(deck)
    else:
        deck = {} if mode == 'tutorial' else _selected_deck(player, payload.get('deck'))
    if mode == 'solo' and requested_ai == 'mirror':
        excluded = sorted({cid for cid in deck['card_ids'] if _content().CARDS[cid].get('training_excluded')})
        if excluded:
            names = '、'.join(_content().CARDS[cid]['name'] for cid in excluded)
            raise RuleValidationError(f'镜像对局不支持以下卡牌：{names}。请先调整当前构筑。')
        solo_deck = deepcopy(deck)
        solo_deck.update(id='mirror', name='镜像对局')
    existing = repository.current_room(player)
    if mode == 'tutorial':
        if existing is not None and existing.mode != 'tutorial' and existing.status in {'waiting', 'ready', 'playing'}:
            raise DuelV2Conflict('请先离开当前房间，再开始教学。')
        if existing is not None and existing.mode == 'tutorial':
            _close_tutorial_room(existing)
        _enter_tutorial_level(player, scenario)
    else:
        if existing is not None and existing.status == 'finished' and existing.mode == 'tutorial':
            _close_tutorial_room(existing)
    if mode == 'advanced':
        from app.modules.card_game.engine.ai.advanced_model import require_human_deck
        require_human_deck(ai_model, deck, ai_deck_id)
    with atomic_transaction():
        room = repository.create_room(player, mode)
        repository.add_member(room, player, 'a', deck, ready=(mode in {'solo', 'advanced', 'tutorial'}))
        envelope = _initialize_game(room, scenario=scenario, ai_deck_id=ai_deck_id, ai_model=ai_model, solo_deck=solo_deck) if mode in {'solo', 'advanced', 'tutorial'} else None
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
            people = repository.members(room)
            if len(people) >= 2:
                raise RuleValidationError('房间已满。')
            host = next((entry for entry in people if entry.side == 'a'), None)
            if host is not None:
                _require_visible_deck(player, _validate_deck(repository.member_deck(host)), joining=True)
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
    return _submit_action(player, payload)


def submit_ai_step(player, payload: dict) -> dict:
    return _submit_action(player, {**payload, 'action': {'type': '_ai_step'}}, ai_step=True)


def _submit_action(player, payload: dict, *, ai_step=False) -> dict:
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
        if ai_step:
            if room.mode != 'advanced' or member.side != 'a':
                raise RuleValidationError('当前房间不能请求高级人机操作。')
            if envelope['game']['phase'] == 'finished' or _engine().acting_side(envelope['game']) != 'b':
                return _response(room, player, envelope)
        try:
            if ai_step:
                from app.modules.card_game.engine.ai.advanced_model import choose_action
                action = choose_action(envelope['game'], 'b')
                game = _engine().apply_action(envelope['game'], 'b', action)
            else:
                game = _engine().apply_action(envelope['game'], member.side, action)
                game = _advance_ai(game, room.mode)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuleValidationError(str(exc)) from exc
        receipts = [*envelope.get('requests', []),
                    {'side': member.side, 'id': request_id, 'fingerprint': fingerprint}][-_RECEIPT_LIMIT:]
        updated = {'rules_version': 'duel_v2', 'game': game, 'requests': receipts}
        storage.publish(room, updated, final=game['phase'] == 'finished')
        if room.mode != 'tutorial':
            replay.record(room, updated)
        if room.mode == 'tutorial' and game.get('phase') == 'finished' and game.get('winner') == member.side:
            scenario = ((game.get('flags') or {}).get('tutorial') or {}).get('scenario')
            if scenario:
                _mark_tutorial_level(player, scenario)
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
