from pathlib import Path

from flask import Blueprint, abort, g, jsonify
from app.auth import token_required


blueprint = Blueprint(
    'shaft_module',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/shaft',
)


@blueprint.get('/shaft/compare')
def removed_compare_page():
    abort(404)


@blueprint.get('/api/shaft/cli-runtime')
@token_required
def cli_runtime():
    from .service import get_shaft_catalog_payload

    javascript = Path(__file__).parent / 'static' / 'js'
    response = jsonify(
        catalog=get_shaft_catalog_payload(player=g.current_player),
        engine=(javascript / 'shaft_engine.js').read_text(encoding='utf-8'),
        self_check=(javascript / 'shaft_self_check.js').read_text(encoding='utf-8'),
    )
    response.cache_control.no_store = True
    return response
