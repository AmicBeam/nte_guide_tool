"""V2 HTTP adapter: authentication, request validation, application calls only."""
from functools import wraps

from flask import Blueprint, g, jsonify, request

from app.auth import token_required
from app.errors import AppError
from app.modules.card_game.engine.application import v2_service as service
from app.utils.logger import get_logger

blueprint = Blueprint('duel_v2_api', __name__, url_prefix='/api/duel-v2')
logger = get_logger('nte.duel_v2.routes')


def _api(func):
    @wraps(func)
    @token_required
    def wrapped(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            response = jsonify(result)
            response.headers['Cache-Control'] = 'private, no-store'
            response.headers['Vary'] = 'Authorization'
            return response
        except service.DuelV2NotFound as exc:
            return jsonify({'error': str(exc)}), 404
        except service.DuelV2Conflict as exc:
            return jsonify({'error': str(exc)}), 409
        except (AppError, ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception:
            logger.exception('V2 request failed path=%s', request.path)
            return jsonify({'error': '操作未完成，请刷新后重试。'}), 500
    return wrapped


def _body():
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError('请求内容必须为 JSON 对象。')
    return payload


@blueprint.get('/catalog')
@_api
def catalog():
    return service.catalog(g.current_player)


@blueprint.post('/build')
@_api
def save_build():
    return service.save_build(g.current_player, _body())


@blueprint.post('/build-import')
@_api
def import_build_text():
    return service.import_build_text(g.current_player, _body())


@blueprint.post('/build-export')
@_api
def export_build_text():
    return service.export_build_text(g.current_player, _body())


@blueprint.post('/build-delete')
@_api
def delete_build():
    return service.delete_build(g.current_player, _body())


@blueprint.post('/build-select')
@_api
def select_build():
    return service.select_build(g.current_player, _body())


@blueprint.post('/build-reorder')
@_api
def reorder_builds():
    return service.reorder_builds(g.current_player, _body())


@blueprint.post('/start')
@_api
def start():
    return service.start(g.current_player, _body())


@blueprint.get('/state')
@_api
def state():
    raw = request.args.get('after_seq')
    after_seq = None
    if raw not in (None, ''):
        try:
            after_seq = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError('after_seq 必须为整数。') from exc
    return service.get_state(g.current_player, after_seq=after_seq)


@blueprint.post('/join')
@_api
def join():
    return service.join(g.current_player, _body())


@blueprint.post('/ready')
@_api
def ready():
    return service.ready(g.current_player, _body())


@blueprint.post('/room-start')
@_api
def start_room():
    return service.start_room(g.current_player)


@blueprint.post('/action')
@_api
def action():
    return service.submit_action(g.current_player, _body())


@blueprint.post('/ai-step')
@_api
def ai_step():
    return service.submit_ai_step(g.current_player, _body())


@blueprint.get('/replays')
@_api
def list_replays():
    return service.list_replays(g.current_player)


@blueprint.get('/replay/<room_code>')
@_api
def get_replay(room_code):
    return service.get_replay(g.current_player, room_code)


@blueprint.post('/replay-star')
@_api
def star_replay():
    return service.star_replay(g.current_player, _body())


@blueprint.post('/replay-import')
@_api
def import_replay():
    return service.import_training_replay(g.current_player, _body())


@blueprint.post('/leave')
@_api
def leave():
    return service.leave(g.current_player)


@blueprint.post('/tutorial-skip')
@_api
def tutorial_skip():
    return service.skip_tutorial(g.current_player, _body())
