from flask import Blueprint, abort


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
