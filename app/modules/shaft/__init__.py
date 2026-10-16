from pathlib import Path

from flask import Blueprint, render_template


blueprint = Blueprint(
    'shaft_module',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/shaft',
)


@blueprint.get('/shaft/compare')
def compare_page():
    static = Path(__file__).parent / 'static'
    version = max(int(path.stat().st_mtime) for path in static.glob('js/shaft*.js'))
    return render_template('shaft/compare.html', compare_asset_version=version)
