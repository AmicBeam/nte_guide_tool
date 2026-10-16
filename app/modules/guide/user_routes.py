"""Authenticated adapters for private cultivation data."""
from functools import wraps
import secrets
import traceback

from flask import current_app, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from app.auth import token_required
from app.db import atomic_transaction
from . import game_sync, user_data, service, presentation
from .game_provider import LoginDependencyError, ProviderError


def private_errors(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except (ProviderError, ValueError, LookupError, HTTPException):
            raise
        except Exception as error:
            reference = secrets.token_hex(6)
            frames = traceback.extract_tb(error.__traceback__)
            location = frames[-1] if frames else None
            # Never log exception text, request contents, credentials or locals.
            current_app.logger.error('guide private failure ref=%s endpoint=%s type=%s function=%s line=%s',
                reference, request.endpoint, type(error).__name__,
                location.name if location else '', location.lineno if location else 0)
            return jsonify(error={'code': 'private_data_unavailable',
                'message': f'同步或读取资料失败，请稍后重试。错误编号：{reference}'}), 500
    return wrapped


def register(blueprint):
    @blueprint.get('/guide/account')
    def account_page():
        characters = {item['id']: {
            'name': item['name'], 'avatar': item.get('assets', {}).get('avatar'),
            'element_name': item.get('element_name'),
            'awakens': {effect['name']: {'icon': effect.get('icon')}
                        for effect in item.get('awaken', [])},
            'portrait': item.get('assets', {}).get('fullbody') or item.get('assets', {}).get('portrait'),
            'skills': {skill['id']: service.ability_title(skill) for skill in item['abilities']},
            'skill_details': {skill['id'].casefold(): {
                'name': service.ability_title(skill), 'icon': skill.get('icon'), 'type': skill.get('type')
            } for skill in item['abilities']}
        } for item in service.catalog()['characters']}
        arcs = {item['id']: {'name': item['name'], 'image': item.get('image')}
                for item in service.catalog()['arcs']}
        return render_template('guide/account.html', characters=characters, arcs=arcs,
            equipment_images=presentation.equipment_images(),
            kinds=service.KINDS, counts={key: len(service.catalog()[key]) for key in service.KINDS})

    def body():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError('请求须为JSON对象。')
        return payload

    @blueprint.after_request
    def private_headers(response):
        if request.path.startswith('/api/guide/me/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @blueprint.errorhandler(LoginDependencyError)
    def dependency_unavailable(error):
        return jsonify(error={'code': 'login_dependency_unavailable', 'message': str(error)}), 503

    @blueprint.errorhandler(ProviderError)
    def unavailable(error):
        return jsonify(error={'code': 'game_service_unavailable', 'message': str(error)}), 502

    @blueprint.get('/api/guide/me/profiles')
    @token_required
    @private_errors
    def my_profiles():
        return jsonify(data=user_data.profiles(g.current_player))

    @blueprint.delete('/api/guide/me/profiles/<role_id>')
    @token_required
    @private_errors
    def delete_profile(role_id):
        with atomic_transaction():
            user_data.GuideGameProfile.delete().where(
                (user_data.GuideGameProfile.player == g.current_player) &
                (user_data.GuideGameProfile.role_id == role_id)).execute()
        return jsonify(ok=True)

    @blueprint.get('/api/guide/me/profiles/<role_id>/characters/<identifier>/prefill')
    @token_required
    @private_errors
    def character_prefill(role_id, identifier):
        record = service.detail('characters', identifier)
        profile = next((p for p in user_data.profiles(g.current_player) if p['role_id'] == role_id), None)
        saved = next((c for c in profile['characters'] if c['id'] == identifier), None) if profile else None
        if saved is None:
            raise LookupError('没有该角色的已同步数据。')
        return jsonify(data=user_data.prefill(record, saved))

    @blueprint.post('/api/guide/me/sync/sms')
    @token_required
    @private_errors
    def sync_sms():
        return jsonify(data=game_sync.send_sms(g.current_player, body()))

    @blueprint.post('/api/guide/me/sync/verify')
    @token_required
    @private_errors
    def sync_verify():
        return jsonify(data=game_sync.operate(g.current_player, body()))

    @blueprint.post('/api/guide/me/sync/select-role')
    @token_required
    @private_errors
    def sync_select():
        return jsonify(data=game_sync.operate(g.current_player, body(), select=True))
