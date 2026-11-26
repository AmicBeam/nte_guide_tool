"""Authenticated streaming endpoint for browser-owned Wiki conversations."""
import json
from pathlib import Path
import re
import urllib.parse

from flask import Response, current_app, g, jsonify, request, stream_with_context

from app.auth import token_required
from . import chat_history, chat_model, chat_quota, chat_service


def register(blueprint):
    @blueprint.context_processor
    def chat_assets():
        static = Path(__file__).parent / 'static'
        return {'guide_chat_asset_version': max((static / name).stat().st_mtime_ns
                for name in ('chat.js', 'chat_state.js', 'chat.css'))}

    @blueprint.get('/api/guide/chat/config')
    @token_required
    def chat_config():
        settings = current_app.config.get('GUIDE_CHAT_SETTINGS', {})
        return jsonify(available=chat_model.configured(settings), model=settings.get('model', ''),
                       context_gap_hours=6, max_images=2)

    @blueprint.post('/api/guide/chat')
    @token_required
    def chat():
        settings = dict(current_app.config.get('GUIDE_CHAT_SETTINGS', {}))
        if not chat_model.configured(settings):
            return jsonify(error={'code': 'chat_not_configured', 'message': '尚未配置对话模型，请联系管理员。'}), 503
        limit = 10 * 1024 * 1024
        if not request.is_json or (request.content_length or 0) > limit:
            return jsonify(error={'code': 'invalid_chat', 'message': '请求须为JSON且不超过10MB。'}), 400
        raw = request.stream.read(limit + 1)
        try:
            if len(raw) > limit:
                raise ValueError('本次图片和消息过大，请减少附图。')
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError('请求须为JSON对象。')
            history = chat_history.prepare(payload)
            path = str(payload.get('page', '/guide'))
            if not re.fullmatch(r'/guide(?:[/A-Za-z0-9_.?=&%#-]*)', path) or len(path) > 300:
                path = '/guide'
            parsed = urllib.parse.urlsplit(path)
            safe_query = [(key, value) for key, value in urllib.parse.parse_qsl(parsed.query)
                          if key in ('kind', 'stage', 'character', 'arc_type', 'rarity', 'element', 'q', 'sort')]
            path = parsed.path + ('?' + urllib.parse.urlencode(safe_query) if safe_query else '')
            title = payload.get('page_title', '')
            if not isinstance(title, str):
                raise ValueError('页面标题无效。')
            title = title[:160].replace('\n', ' ')
            from .infographic import editor_context
            editor = editor_context(payload.get('infographic'), parsed.path)
        except (ValueError, UnicodeError) as error:
            return jsonify(error={'code': 'invalid_chat', 'message': str(error)[:180]}), 400
        player = g.current_player
        if not chat_service.acquire(player.id):
            return jsonify(error={'code': 'chat_busy', 'message': '已有对话正在回答，或服务繁忙，请稍后重试。'}), 429
        try:
            retry_after = chat_quota.consume(player.id)
        except Exception:
            chat_service.release(player.id)
            raise
        if retry_after:
            chat_service.release(player.id)
            minutes = (retry_after + 59) // 60
            response = jsonify(error={'code': 'chat_rate_limited',
                'message': f'每个账号每小时最多发送10条消息，请在约{minutes}分钟后重试。',
                'retry_after': retry_after})
            response.status_code = 429
            response.headers['Retry-After'] = str(retry_after)
            return response
        reader = current_app.extensions.get('guide_axis_reader')
        logger = current_app.logger
        released = False

        def finish():
            nonlocal released
            if not released:
                released = True
                chat_service.release(player.id)

        def generate():
            try:
                for event in chat_service.events(settings, history, player, reader, path, title, **({'editor': editor} if editor else {})):
                    yield json.dumps(event, ensure_ascii=False) + '\n'
            except GeneratorExit:
                raise
            except Exception as error:
                # Log only the failure type, never prompts, attachments, credentials or upstream bodies.
                logger.warning('guide chat failed type=%s', type(error).__name__)
                message = str(error) if isinstance(error, chat_model.ChatModelError) else '对话暂未完成，请稍后重试。'
                yield json.dumps({'type': 'error', 'text': message}, ensure_ascii=False) + '\n'
            finally:
                finish()
        response = Response(stream_with_context(generate()), content_type='application/x-ndjson; charset=utf-8',
                            headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
        response.call_on_close(finish)
        return response

    @blueprint.after_request
    def chat_headers(response):
        if request.path.startswith('/api/guide/chat'):
            response.headers['Cache-Control'] = 'no-store'
        return response
