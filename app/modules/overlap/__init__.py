"""Offline reference tables and a browser-side two-character cycle planner."""
from pathlib import Path

from flask import Blueprint, render_template, request

blueprint = Blueprint(
    'overlap_module', __name__, template_folder='templates',
    static_folder='static', static_url_path='/static/overlap',
)


@blueprint.get('/overlap')
def index():
    root = Path(__file__).resolve().parent / 'static'
    version = max(int((root / name).stat().st_mtime) for name in (
        'js/overlap.js', 'js/reference_rows.js', 'js/pair.js', 'js/pair_engine.js', 'js/pair_multi_engine.js', 'js/pair_worker.js',
        'css/overlap.css', 'data/tables.json',
    ))
    template = 'overlap/pair.html' if request.args.get('page') == 'pair' else 'overlap/index.html'
    return render_template(template, asset_version=version)
