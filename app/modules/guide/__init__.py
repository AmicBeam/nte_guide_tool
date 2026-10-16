"""Public NTE reference, cultivation calculator and authenticated private imports."""
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request, send_file
from . import service, cultivation, presentation, arc_cultivation
from .challenge_calendar import is_current_challenge
from .rich_text import description_html

blueprint = Blueprint('guide_module', __name__, template_folder='templates',
                      static_folder='static', static_url_path='/static/guide')
blueprint.add_app_template_filter(description_html, 'guide_description_html')
blueprint.add_app_template_filter(is_current_challenge, 'guide_current_challenge')
blueprint.add_app_template_filter(presentation.material_display, 'guide_material_display')
blueprint.add_app_template_filter(presentation.kongmu_entry, 'guide_kongmu_entry')
blueprint.add_app_template_filter(presentation.overlap_entry, 'guide_overlap_entry')
blueprint.add_app_template_filter(service.label, 'guide_label')
blueprint.add_app_template_filter(service.ability_title, 'guide_ability_title')
blueprint.add_app_template_filter(service.effect_html, 'guide_effect_html')
blueprint.add_app_template_filter(service.character_name, 'guide_character_name')
blueprint.add_app_template_filter(service.equip_description, 'guide_equip_description')
blueprint.add_app_template_filter(service.display_scalar, 'guide_scalar')


def kind_check(kind):
    if kind not in service.KINDS:
        raise LookupError('未知分类；可选 characters、arcs、challenges、endless')


def integer(name, default, low, high):
    value = int(request.args.get(name, default))
    if not low <= value <= high:
        raise ValueError(f'{name} 须为 {low}–{high}')
    return value


def filters():
    return dict(query=request.args.get('q', '').strip(), element=request.args.get('element', ''),
                rarity=integer('rarity', 5, 1, 5) if request.args.get('rarity') else None,
                arc_type=request.args.get('arc_type', ''))


def get_record(kind, identifier):
    kind_check(kind)
    return service.detail(kind, identifier, request.args.get('section'), request.args.get('stage') or None)


@blueprint.errorhandler(ValueError)
def bad_request(error):
    return jsonify(error={'code': 'invalid_argument', 'message': str(error)}), 400


@blueprint.errorhandler(LookupError)
def missing(error):
    return jsonify(error={'code': 'not_found', 'message': str(error)}), 404


@blueprint.get('/guide')
def index():
    kind = request.args.get('kind', 'characters')
    kind_check(kind)
    items = service.search(kind, **filters())
    selected_ids = {item['id'] for item in items}
    result_count = len(items)
    if kind in ('characters', 'arcs'):
        items = service.filter_index_items(kind)
    counts = {key: len(service.catalog()[key]) for key in service.KINDS}
    return render_template('guide/index.html', kind=kind, kinds=service.KINDS, items=items, counts=counts, selected_ids=selected_ids, result_count=result_count,
        catalog_asset_version=(Path(__file__).parent / 'static/catalog_cards.css').stat().st_mtime_ns,
        arc_types=sorted({c['arc_type']['id']: c['arc_type'] for c in service.catalog()['characters']}.values(),
                         key=lambda arc: int(arc['id'])))


@blueprint.get('/guide/<kind>/<identifier>')
def entry(kind, identifier):
    record = get_record(kind, identifier)
    sections = service.page_sections(kind, record)
    profile = service.detail(kind, identifier) if kind == 'characters' else None
    art = profile.get('assets') if profile else None
    overview = profile if profile else (service.detail(kind, identifier) if kind == 'arcs' else None)
    arc_image = overview.get('image') if kind == 'arcs' else None
    return render_template('guide/detail.html', record=record, sections=sections, kind=kind,
                           kinds=service.KINDS, art=art, profile=profile, overview=overview, arc_image=arc_image,
                           material_preview=presentation.upgrade_material_preview(overview) if overview else [],
                           notes=service.schema()['notes'], recommended_arc=arc_cultivation.recommended(profile) if profile else None)


@blueprint.get('/guide/characters/<identifier>/axes/<int:axis_id>/compare')
def axis_compare_page(identifier, axis_id):
    if axis_id <= 0:
        raise LookupError('未找到排轴')
    record = service.detail('characters', identifier)
    static = Path(__file__).parent.parent / 'shaft' / 'static'
    version = max(int(path.stat().st_mtime) for path in static.glob('js/shaft*.js'))
    arc_rarities = {arc['name']: arc['rarity'] for arc in service.catalog()['arcs']}
    return render_template('shaft/compare.html', compare_asset_version=version,
                           axis_id=axis_id, record=record, arc_rarities=arc_rarities)


@blueprint.get('/guide/characters/<identifier>/cultivation')
def cultivation_page(identifier):
    record = service.detail('characters', identifier)
    config = service.catalog()['cultivation']
    values = arc_cultivation.defaults(record, config)
    values.update({key: request.args[key] for key in values if key in request.args})
    result, error = None, None
    if request.args.get('calculate') == '1':
        try:
            result = arc_cultivation.calculate(record, config, record['experience'], values)
        except ValueError as exc:
            error = str(exc)
    return render_template('guide/cultivation.html', record=record, values=values,
        caps=config['level_caps'][identifier], skills=cultivation.skill_options(record),
        result=result, error=error, arcs=arc_cultivation.choices(record), arc_configs=arc_cultivation.config()['arcs']), 400 if error else 200


@blueprint.get('/guide/characters/<identifier>/builds')
def character_builds_page(identifier):
    record = service.detail('characters', identifier)
    static = Path(__file__).parent / 'static'
    version = max(int((static / name).stat().st_mtime) for name in ['related_axes.js', 'related_axes.css'])
    return render_template('guide/builds.html', record=record, related_asset_version=version)


@blueprint.post('/api/guide/characters/<identifier>/cultivation')
def api_cultivation(identifier):
    record = service.detail('characters', identifier)
    config = service.catalog()['cultivation']
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError('请求须为 JSON 对象')
    values = arc_cultivation.defaults(record, config)
    values.update(payload)
    result = arc_cultivation.calculate(record, config, record['experience'], values)
    return jsonify(schema_version=1, kind='cultivation', data=result)


@blueprint.get('/guide/<kind>/<identifier>.md')
def markdown(kind, identifier):
    return Response(service.markdown_record(get_record(kind, identifier)), content_type='text/markdown; charset=utf-8')


@blueprint.get('/api/guide/schema')
def schema():
    return jsonify(service.schema())


@blueprint.get('/api/guide/search')
@blueprint.get('/api/guide/<kind>')
def listing(kind=None):
    if kind:
        kind_check(kind)
    limit = integer('limit', 20, 1, 100)
    offset = integer('offset', 0, 0, 1000000)
    rows = service.search(kind, **filters())
    return jsonify(schema_version=1, total=len(rows), limit=limit, offset=offset,
                   next_offset=offset + limit if offset + limit < len(rows) else None,
                   items=rows[offset:offset + limit])


@blueprint.get('/api/guide/<kind>/<identifier>')
def api_detail(kind, identifier):
    return jsonify(schema_version=1, kind=kind, data=get_record(kind, identifier))


@blueprint.get('/guide/cli.py')
def cli_download():
    return send_file(Path(__file__).parent / 'cli.py', mimetype='text/x-python', as_attachment=True,
                     download_name='nte-guide.py')


@blueprint.get('/llms.txt')
@blueprint.get('/guide/llms.txt')
def llms():
    text = '''# NTE Tools 攻略数据库

范围：角色、弧盘、轨外之境、全站深境。匿名只读；网页正文由服务端渲染。

- [网页入口](/guide)
- [接口与数值约定](/api/guide/schema)
- [角色列表](/api/guide/characters)
- [弧盘列表](/api/guide/arcs)
- [轨外之境赛期](/api/guide/challenges)
- [全站深境（每50层一组）](/api/guide/endless)
- [CLI 下载](/guide/cli.py)

检索：GET /api/guide/search?q=关键词&limit=20&offset=0
详情：GET /api/guide/{characters|arcs|challenges|endless}/{id}
纯文本：GET /guide/{characters|arcs|challenges|endless}/{id}.md
空幕：角色详情加 ?section=equip_slots；升级材料：详情加 ?section=materials。
等级经验：角色详情加 ?section=experience。
好感度与喜好礼物：角色详情加 ?section=affinity，包含赠送数值和各商店标价。
养成计算器：/guide/characters/{id}/cultivation；POST /api/guide/characters/{id}/cultivation 计算资源。
角色排轴与配装：/guide/characters/{id}/builds；固定当前角色筛选相关轴，再进入收益对比。
按字段读取：详情加 ?section=abilities（角色技能）或 ?section=effect（弧盘效果）。
按关卡读取：轨外之境详情加 ?stage=12；全站深境详情加 ?stage=50（实际层号）。
列表返回 total、next_offset 和摘要；详情返回 data。使用 ID 区分重名赛期。
数值数组保留等级/精炼顺序；敌人抗性为小数比例。日期缺失不代表当前开放。
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 search 黑羽
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 get characters 1042 --section abilities
'''
    return Response(text, content_type='text/plain; charset=utf-8')


from .user_routes import register as register_user_routes
register_user_routes(blueprint)



from .challenge_view import stage_selection
blueprint.add_app_template_filter(stage_selection, 'guide_stage_selection')
