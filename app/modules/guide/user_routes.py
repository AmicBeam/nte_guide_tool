"""Authenticated adapters for private cultivation data."""
from flask import g, jsonify, request
from app.auth import token_required
from app.db import atomic_transaction
from . import game_sync, user_data, service
from .game_provider import ProviderError


def register(blueprint):
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

    @blueprint.errorhandler(ProviderError)
    def unavailable(error):
        return jsonify(error={'code': 'game_service_unavailable', 'message': str(error)}), 502

    @blueprint.get('/api/guide/me/profiles')
    @token_required
    def my_profiles():
        return jsonify(data=user_data.profiles(g.current_player))

    @blueprint.delete('/api/guide/me/profiles/<role_id>')
    @token_required
    def delete_profile(role_id):
        with atomic_transaction():
            user_data.GuideGameProfile.delete().where(
                (user_data.GuideGameProfile.player == g.current_player) &
                (user_data.GuideGameProfile.role_id == role_id)).execute()
        return jsonify(ok=True)

    @blueprint.get('/api/guide/me/profiles/<role_id>/characters/<identifier>/prefill')
    @token_required
    def character_prefill(role_id, identifier):
        record = service.detail('characters', identifier)
        profile = next((p for p in user_data.profiles(g.current_player) if p['role_id'] == role_id), None)
        saved = next((c for c in profile['characters'] if c['id'] == identifier), None) if profile else None
        if saved is None:
            raise LookupError('没有该角色的已同步数据。')
        return jsonify(data=user_data.prefill(record, saved))

    @blueprint.post('/api/guide/me/sync/sms')
    @token_required
    def sync_sms():
        return jsonify(data=game_sync.send_sms(g.current_player, body()))

    @blueprint.post('/api/guide/me/sync/verify')
    @token_required
    def sync_verify():
        return jsonify(data=game_sync.operate(g.current_player, body()))

    @blueprint.post('/api/guide/me/sync/select-role')
    @token_required
    def sync_select():
        return jsonify(data=game_sync.operate(g.current_player, body(), select=True))
