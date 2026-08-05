from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Blueprint, g, jsonify, redirect, render_template, request, url_for

from app.auth import login_with_code, token_required
from app.dao import is_tutorial_completed, mark_tutorial_completed, update_player_nickname, update_player_password
from app.modules.card_game.engine.game_service import (
    choose_cards,
    declaration_preview,
    declaration_previews,
    end_turn,
    get_catalog_payload,
    get_encyclopedia_payload,
    retreat,
    save_build,
)
from app.modules.card_game.engine.application.analytics_service import get_duel_analytics_payload
from app.modules.kongmu.service import get_kongmu_catalog_payload, plan_kongmu_layout
from app.modules.preteam.catalog import MAIN_CANDIDATES as PRETEAM_MAIN_CANDIDATES
from app.modules.preteam.catalog import TEAMMATES as PRETEAM_TEAMMATES
from app.modules.shaft.service import (
    ShaftAxisNameConflictError,
    backup_shaft_axis,
    create_shaft_axis_share,
    delete_shaft_axis,
    get_shared_shaft_axis,
    get_shaft_axis,
    get_shaft_catalog_payload,
    list_favorite_shaft_axes,
    list_my_shaft_axes,
    list_shaft_market,
    optional_player_from_token,
    publish_shaft_axis_snapshot,
    save_shaft_axis,
    set_shaft_axis_dislike,
    set_shaft_axis_favorite,
    set_shaft_axis_like,
)
from app.errors import AppError
from app.room_service import (
    create_or_resume_solo_room,
    create_room_for_player,
    get_current_room_run_state,
    get_current_room_state,
    join_room_by_code,
    reset_current_room_run,
    set_room_ready,
    start_room_game,
)
from app.utils.logger import get_logger

logger = get_logger('nte.routes')
main_bp = Blueprint('main', __name__)
_kongmu_plan_locks_guard = threading.Lock()
_kongmu_plan_locks: dict[str, threading.Lock] = {}
_shaft_market_search_guard = threading.Lock()
_shaft_market_search_times: dict[str, float] = {}
SHAFT_MARKET_SEARCH_INTERVAL_SECONDS = 3.0


def _request_source_key() -> str:
    forwarded_for = request.headers.get('X-Forwarded-For', '')
    if forwarded_for:
        return forwarded_for.split(',', 1)[0].strip() or 'unknown'
    return request.remote_addr or 'unknown'


def _optional_current_player():
    header = request.headers.get('Authorization', '')
    token = header.replace('Bearer ', '', 1).strip() if header.startswith('Bearer ') else ''
    return optional_player_from_token(token)


def _shaft_visitor_key(payload: dict[str, object] | None = None) -> str:
    return (
        request.headers.get('X-Shaft-Visitor-Key')
        or request.args.get('visitor_key')
        or str((payload or {}).get('visitor_key') or '')
    )


def _shaft_market_search_source_key(player) -> str:
    if player is not None:
        return f'player:{player.id}'
    visitor_key = _shaft_visitor_key().strip()
    if visitor_key:
        return f'visitor:{visitor_key[:128]}'
    return f'ip:{_request_source_key()}'


def _shaft_market_search_retry_after(source_key: str) -> float:
    now = time.monotonic()
    with _shaft_market_search_guard:
        last_search_at = _shaft_market_search_times.get(source_key)
        if last_search_at is not None:
            retry_after = SHAFT_MARKET_SEARCH_INTERVAL_SECONDS - (now - last_search_at)
            if retry_after > 0:
                return retry_after
        _shaft_market_search_times[source_key] = now
        if len(_shaft_market_search_times) > 2048:
            cutoff = now - SHAFT_MARKET_SEARCH_INTERVAL_SECONDS * 10
            stale_keys = [
                key for key, searched_at in _shaft_market_search_times.items()
                if searched_at < cutoff
            ]
            for key in stale_keys:
                _shaft_market_search_times.pop(key, None)
    return 0.0


def _request_int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _shaft_asset_version(filename: str) -> int:
    asset_path = Path(__file__).resolve().parent / 'modules' / 'shaft' / 'static' / filename
    try:
        return int(asset_path.stat().st_mtime)
    except OSError:
        return 0


def _acquire_kongmu_plan_lock(source_key: str) -> threading.Lock | None:
    with _kongmu_plan_locks_guard:
        lock = _kongmu_plan_locks.setdefault(source_key, threading.Lock())
    if not lock.acquire(blocking=False):
        return None
    return lock


@main_bp.get('/')
def default_page():
    return render_template('index.html')


@main_bp.get('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='images/brand/duel-icon.webp'))


@main_bp.get('/card-game')
def card_game_home():
    return render_template('card_game/index.html')


@main_bp.get('/login')
def login_page():
    return render_template('card_game/login.html')


@main_bp.get('/profile')
def profile_page():
    return render_template('card_game/profile.html')


@main_bp.get('/build')
def build_page():
    return render_template('card_game/build.html')


@main_bp.get('/codex')
def codex_page():
    return render_template('card_game/codex.html')


@main_bp.get('/analytics')
def analytics_page():
    return render_template('card_game/analytics.html')


@main_bp.get('/kongmu')
def kongmu_page():
    return render_template('kongmu/index.html')


@main_bp.get('/shaft')
@main_bp.get('/shaft/<page>')
def shaft_page(page: str = 'rotation'):
    if page == 'market':
        page = 'plaza'
    if page not in {'build', 'rotation', 'plaza'}:
        page = 'rotation'
    allow_anonymous_shaft_share = False
    share_token = str(request.args.get('share') or '').strip()
    if share_token:
        try:
            get_shared_shaft_axis(share_token)
            allow_anonymous_shaft_share = True
        except AppError:
            pass
    return render_template(
        'shaft/index.html',
        active_shaft_page=page,
        allow_anonymous_shaft_share=allow_anonymous_shaft_share,
        shaft_asset_version=_shaft_asset_version,
    )


@main_bp.get('/table')
def table_page():
    return render_template('card_game/table.html')


@main_bp.get('/preteam')
def preteam_page():
    return render_template(
        'preteam/index.html',
        main_candidates=PRETEAM_MAIN_CANDIDATES,
        teammates=PRETEAM_TEAMMATES,
    )


@main_bp.post('/api/auth/login')
def api_login():
    payload = request.get_json(silent=True) or {}
    try:
        result, error = login_with_code(str(payload.get('player_uid', '')).strip(), str(payload.get('code', '')).strip())
        if error:
            return jsonify({'error': error}), 400
        return jsonify(result)
    except Exception:
        logger.exception('api_login failed')
        return jsonify({'error': '登录过程中发生异常，请稍后重试。'}), 500


@main_bp.get('/api/me')
@token_required
def api_me():
    return jsonify({
        'player_uid': g.current_player.player_uid,
        'nickname': g.current_player.nickname or g.current_player.player_uid,
    })


@main_bp.post('/api/account/profile')
@token_required
def api_update_profile():
    payload = request.get_json(silent=True) or {}
    try:
        player = update_player_nickname(g.current_player, str(payload.get('nickname', '')))
        return jsonify({
            'player_uid': player.player_uid,
            'nickname': player.nickname or player.player_uid,
        })
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_update_profile failed')
        return jsonify({'error': '修改用户名时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/account/password')
@token_required
def api_update_password():
    payload = request.get_json(silent=True) or {}
    try:
        update_player_password(g.current_player, str(payload.get('password', '')))
        return jsonify({'ok': True})
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_update_password failed')
        return jsonify({'error': '修改密码时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/tutorial/status')
@token_required
def api_tutorial_status():
    try:
        scope = str(request.args.get('scope', ''))
        return jsonify({
            'scope': scope,
            'completed': is_tutorial_completed(g.current_player, scope),
        })
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_tutorial_status failed')
        return jsonify({'error': '读取教学状态时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/tutorial/complete')
@token_required
def api_tutorial_complete():
    payload = request.get_json(silent=True) or {}
    try:
        record = mark_tutorial_completed(g.current_player, str(payload.get('scope', '')))
        return jsonify({
            'scope': record.scope,
            'completed': record.completed,
        })
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_tutorial_complete failed')
        return jsonify({'error': '保存教学状态时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/catalog')
@token_required
def api_catalog():
    try:
        return jsonify(get_catalog_payload(g.current_player))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_catalog failed')
        return jsonify({'error': '读取图鉴与构筑信息时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/encyclopedia')
@token_required
def api_encyclopedia():
    return jsonify(get_encyclopedia_payload())


@main_bp.get('/api/analytics/balance')
@token_required
def api_balance_analytics():
    return jsonify(get_duel_analytics_payload())


@main_bp.get('/api/kongmu/catalog')
def api_kongmu_catalog():
    try:
        player = _optional_current_player()
        return jsonify(get_kongmu_catalog_payload(include_test_characters=bool(
            player and player.shaft_test_whitelisted
        )))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_kongmu_catalog failed')
        return jsonify({'error': '读取空幕数据时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/kongmu/plan')
def api_kongmu_plan():
    payload = request.get_json(silent=True) or {}
    source_key = _request_source_key()
    source_lock = _acquire_kongmu_plan_lock(source_key)
    if source_lock is None:
        return jsonify({'error': '同一来源已有空幕计算正在进行，请稍后再试。'}), 429
    try:
        player = _optional_current_player()
        return jsonify(plan_kongmu_layout(
            str(payload.get('character_id', '')),
            str(payload.get('cartridge_id', '')),
            include_test_characters=bool(player and player.shaft_test_whitelisted),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_kongmu_plan failed')
        return jsonify({'error': '计算空幕方案时发生异常，请稍后重试。'}), 500
    finally:
        source_lock.release()


@main_bp.get('/api/shaft/catalog')
def api_shaft_catalog():
    try:
        return jsonify(get_shaft_catalog_payload(player=_optional_current_player()))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_catalog failed')
        return jsonify({'error': '读取排轴数据时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/shaft/market')
def api_shaft_market():
    player = _optional_current_player()
    query_text = str(request.args.get('q', '')).strip()
    if query_text:
        retry_after = _shaft_market_search_retry_after(
            _shaft_market_search_source_key(player)
        )
        if retry_after > 0:
            return jsonify({
                'error': '搜索操作过于频繁，请稍候。',
                'code': 'shaft_market_search_rate_limited',
                'retry_after_ms': max(1, int(retry_after * 1000) + 1),
            }), 429
    try:
        return jsonify(list_shaft_market(
            character_ids=[value.strip() for value in request.args.getlist('character_id') if value.strip()][:4],
            sort=str(request.args.get('sort', 'dps')).strip(),
            query_text=query_text,
            page=_request_int_arg('page', 1),
            page_size=_request_int_arg('page_size', 20),
            player=player,
            visitor_key=_shaft_visitor_key(),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_market failed')
        return jsonify({'error': '读取排轴广场时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/shaft/axes/<int:axis_id>')
def api_shaft_axis(axis_id: int):
    try:
        return jsonify(get_shaft_axis(
            axis_id,
            player=_optional_current_player(),
            visitor_key=_shaft_visitor_key(),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '读取排轴详情时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/shaft/shared/<string:share_token>')
def api_shaft_shared_axis(share_token: str):
    try:
        return jsonify(get_shared_shaft_axis(
            share_token,
            player=_optional_current_player(),
            visitor_key=_shaft_visitor_key(),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_shared_axis failed')
        return jsonify({'error': '读取分享排轴时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes')
@token_required
def api_shaft_save_axis():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(save_shaft_axis(g.current_player, payload))
    except ShaftAxisNameConflictError as exc:
        return jsonify({
            'error': str(exc),
            'code': 'axis_name_conflict',
            'title': exc.title,
            'existing_axis_id': exc.axis_id,
        }), 409
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_save_axis failed')
        return jsonify({'error': '保存排轴时发生异常，请稍后重试。'}), 500


@main_bp.put('/api/shaft/axes/<int:axis_id>')
@token_required
def api_shaft_update_axis(axis_id: int):
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(save_shaft_axis(g.current_player, payload, axis_id=axis_id))
    except ShaftAxisNameConflictError as exc:
        return jsonify({
            'error': str(exc),
            'code': 'axis_name_conflict',
            'title': exc.title,
            'existing_axis_id': exc.axis_id,
        }), 409
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_update_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '更新排轴时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/publish')
@token_required
def api_shaft_publish_axis(axis_id: int):
    try:
        return jsonify(publish_shaft_axis_snapshot(g.current_player, axis_id))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_publish_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '上传排轴快照时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/backup')
@token_required
def api_shaft_backup_axis(axis_id: int):
    try:
        return jsonify(backup_shaft_axis(g.current_player, axis_id))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_backup_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '备份排轴时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/share')
@token_required
def api_shaft_share_axis(axis_id: int):
    try:
        return jsonify(create_shaft_axis_share(g.current_player, axis_id))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_share_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '生成分享链接时发生异常，请稍后重试。'}), 500


@main_bp.delete('/api/shaft/axes/<int:axis_id>')
@token_required
def api_shaft_delete_axis(axis_id: int):
    try:
        return jsonify(delete_shaft_axis(g.current_player, axis_id))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_delete_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '删除排轴时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/shaft/me/axes')
@token_required
def api_shaft_my_axes():
    try:
        return jsonify(list_my_shaft_axes(
            g.current_player,
            character_ids=[value.strip() for value in request.args.getlist('character_id') if value.strip()][:4],
            sort=str(request.args.get('sort', 'new')).strip(),
            query_text=str(request.args.get('q', '')).strip(),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_my_axes failed')
        return jsonify({'error': '读取我的排轴时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/shaft/me/favorites')
@token_required
def api_shaft_my_favorites():
    try:
        return jsonify(list_favorite_shaft_axes(
            g.current_player,
            character_ids=[value.strip() for value in request.args.getlist('character_id') if value.strip()][:4],
            sort=str(request.args.get('sort', 'new')).strip(),
            query_text=str(request.args.get('q', '')).strip(),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_my_favorites failed')
        return jsonify({'error': '读取收藏排轴时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/like')
@token_required
def api_shaft_like_axis(axis_id: int):
    try:
        return jsonify(set_shaft_axis_like(g.current_player, axis_id, True))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_like_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '点赞排轴时发生异常，请稍后重试。'}), 500


@main_bp.delete('/api/shaft/axes/<int:axis_id>/like')
@token_required
def api_shaft_unlike_axis(axis_id: int):
    try:
        return jsonify(set_shaft_axis_like(g.current_player, axis_id, False))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_unlike_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '取消点赞时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/dislike')
@token_required
def api_shaft_dislike_axis(axis_id: int):
    try:
        return jsonify(set_shaft_axis_dislike(g.current_player, axis_id, True))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_dislike_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '踩排轴时发生异常，请稍后重试。'}), 500


@main_bp.delete('/api/shaft/axes/<int:axis_id>/dislike')
@token_required
def api_shaft_undislike_axis(axis_id: int):
    try:
        return jsonify(set_shaft_axis_dislike(g.current_player, axis_id, False))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_undislike_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '取消踩时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/shaft/axes/<int:axis_id>/favorite')
def api_shaft_favorite_axis(axis_id: int):
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(set_shaft_axis_favorite(
            axis_id=axis_id,
            favorited=True,
            player=_optional_current_player(),
            visitor_key=_shaft_visitor_key(payload),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_favorite_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '收藏排轴时发生异常，请稍后重试。'}), 500


@main_bp.delete('/api/shaft/axes/<int:axis_id>/favorite')
def api_shaft_unfavorite_axis(axis_id: int):
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(set_shaft_axis_favorite(
            axis_id=axis_id,
            favorited=False,
            player=_optional_current_player(),
            visitor_key=_shaft_visitor_key(payload),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_shaft_unfavorite_axis failed axis_id=%s', axis_id)
        return jsonify({'error': '取消收藏时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/build/save')
@token_required
def api_save_build():
    payload = request.get_json(silent=True) or {}
    try:
        item_ids = payload.get('item_ids')
        if item_ids is None:
            item_ids = [
                *(payload.get('starter_item_ids') or []),
                *(payload.get('reserve_item_ids') or []),
            ]
        result = save_build(
            g.current_player,
            payload.get('character_id', ''),
            item_ids,
            payload.get('esper_card_ids', []),
        )
        return jsonify(result)
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_save_build failed')
        return jsonify({'error': '保存构筑时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/start')
@token_required
def api_start_game():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(create_or_resume_solo_room(g.current_player, payload))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_start_game failed')
        return jsonify({'error': '开始对局时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/game/state')
@token_required
def api_game_state():
    try:
        state = get_current_room_run_state(g.current_player)
        if state is None:
            if request.args.get('optional') == '1':
                return jsonify({'status': 'none', 'has_active_run': False})
            return jsonify({'error': '当前没有对局，请先开始。'}), 404
        return jsonify(state)
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_game_state failed')
        return jsonify({'error': '读取对局状态时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/game/declaration-previews')
@token_required
def api_declaration_previews():
    try:
        return jsonify(declaration_previews(g.current_player))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_declaration_previews failed')
        return jsonify({'error': '预读取检视候选时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/declaration-preview')
@token_required
def api_declaration_preview():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(declaration_preview(
            g.current_player,
            str(payload.get('card_instance_id', '')),
            str(payload.get('location_id', '')),
            str(payload.get('selected_target_instance_id', '')),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_declaration_preview failed')
        return jsonify({'error': '读取检视候选时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/choose-cards')
@token_required
def api_choose_cards():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(choose_cards(
            g.current_player,
            payload.get('card_instance_ids', []),
        ))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_choose_cards failed')
        return jsonify({'error': '选择卡牌时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/end-turn')
@token_required
def api_end_turn():
    payload = request.get_json(silent=True) or {}
    declaration_choices = payload.get('declaration_choices')
    if declaration_choices is not None and not isinstance(declaration_choices, list):
        declaration_choices = []
    planning_actions = payload.get('planning_actions')
    if planning_actions is not None and not isinstance(planning_actions, list):
        planning_actions = []
    try:
        return jsonify(end_turn(g.current_player, declaration_choices, planning_actions))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_end_turn failed')
        return jsonify({'error': '完成部署时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/retreat')
@token_required
def api_retreat():
    try:
        return jsonify(retreat(g.current_player))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_retreat failed')
        return jsonify({'error': '撤退时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/game/reset')
@token_required
def api_reset():
    try:
        reset_current_room_run(g.current_player)
        return jsonify({'ok': True})
    except Exception:
        logger.exception('api_reset failed')
        return jsonify({'error': '重置对局时发生异常，请稍后重试。'}), 500


@main_bp.get('/api/room/state')
@token_required
def api_room_state():
    try:
        room_state = get_current_room_state(g.current_player)
        if room_state is None:
            return jsonify({'error': '当前没有房间。'}), 404
        return jsonify(room_state)
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_room_state failed')
        return jsonify({'error': '读取房间状态时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/room/create')
@token_required
def api_create_room():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(create_room_for_player(g.current_player, str(payload.get('mode', 'solo'))))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_create_room failed')
        return jsonify({'error': '创建房间时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/room/join')
@token_required
def api_join_room():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(join_room_by_code(g.current_player, str(payload.get('room_code', ''))))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_join_room failed')
        return jsonify({'error': '加入房间时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/room/ready')
@token_required
def api_room_ready():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(set_room_ready(g.current_player, bool(payload.get('is_ready', True))))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_room_ready failed')
        return jsonify({'error': '更新准备状态时发生异常，请稍后重试。'}), 500


@main_bp.post('/api/room/start')
@token_required
def api_room_start():
    try:
        return jsonify(start_room_game(g.current_player))
    except AppError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('api_room_start failed')
        return jsonify({'error': '开始房间游戏时发生异常，请稍后重试。'}), 500
