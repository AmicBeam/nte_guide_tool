from pathlib import Path

from flask import Blueprint, abort, render_template, request


blueprint = Blueprint(
    'shaft_module',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/shaft',
)


@blueprint.get('/shaft/axes/<int:axis_id>/compare')
def compare_page(axis_id):
    if axis_id <= 0 or request.args.get('locked') != '1' or not request.args.get('character') or not request.args.get('guide_character', '').isdigit():
        abort(404)
    static = Path(__file__).parent / 'static'
    version = max(int(path.stat().st_mtime) for path in static.glob('js/shaft*.js'))
    return render_template('shaft/compare.html', compare_asset_version=version, axis_id=axis_id)


@blueprint.get('/shaft/compare')
def removed_compare_page():
    abort(404)
