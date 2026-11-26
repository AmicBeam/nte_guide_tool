"""HTTP adapters for the CLI's online-axis reads, retaining shaft permissions."""
from flask import current_app, jsonify, request

from app.dao import get_player_by_token
from . import read_tools


def register(blueprint):
    def player():
        header = request.headers.get('Authorization', '')
        token = header[7:] if header.startswith('Bearer ') else ''
        value = get_player_by_token(token) if token else None
        if token and value is None:
            raise PermissionError('登录已失效，请重新登录。')
        return value

    def result(operation):
        try:
            return jsonify(operation())
        except PermissionError as error:
            return jsonify(error={'code': 'login_required', 'message': str(error)}), 401
        except (LookupError, ValueError) as error:
            return jsonify(error={'code': 'invalid_axis_query', 'message': str(error)}), 400
        except Exception:
            # Existing shaft errors may include only domain messages, but do not expose internals.
            return jsonify(error={'code': 'axis_unavailable', 'message': '未找到可访问的排轴，或排轴暂不可用。'}), 400

    @blueprint.get('/api/guide/axes')
    def query_axes():
        return result(lambda: read_tools.axes(current_app.extensions['guide_axis_reader'], player(),
            characters=request.args.getlist('character'), scope=request.args.get('scope', 'public'),
            query=request.args.get('q', ''), sort=request.args.get('sort', 'new'),
            page=int(request.args.get('page', 1)), limit=int(request.args.get('limit', 10))))

    @blueprint.get('/api/guide/axes/<int:axis_id>')
    def query_axis(axis_id):
        return result(lambda: read_tools.axis_detail(current_app.extensions['guide_axis_reader'], player(),
                         axis_id, request.args.get('section', 'all')))

    @blueprint.after_request
    def axis_headers(response):
        if request.path.startswith('/api/guide/axes'):
            response.headers['Cache-Control'] = 'no-store'
        return response
