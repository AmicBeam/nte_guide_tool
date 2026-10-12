"""Static, offline-generated overlap reference tables."""
from pathlib import Path

from flask import Blueprint, render_template

blueprint = Blueprint(
    'overlap_module', __name__, template_folder='templates',
    static_folder='static', static_url_path='/static/overlap',
)


@blueprint.get('/overlap')
def index():
    root = Path(__file__).resolve().parent / 'static'
    version = max(int((root / name).stat().st_mtime) for name in (
        'js/overlap.js', 'css/overlap.css', 'data/tables.json',
    ))
    return render_template('overlap/index.html', asset_version=version)
