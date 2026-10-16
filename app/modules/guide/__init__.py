"""Anonymous, server-rendered NTE reference and machine-readable API."""
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request, send_file
from . import service

blueprint = Blueprint('guide_module', __name__, template_folder='templates',
                      static_folder='static', static_url_path='/static/guide')
blueprint.add_app_template_filter(service.label, 'guide_label')
blueprint.add_app_template_filter(service.character_name, 'guide_character_name')
blueprint.add_app_template_filter(service.equip_description, 'guide_equip_description')
blueprint.add_app_template_filter(service.display_scalar, 'guide_scalar')


def kind_check(kind):
    if kind not in service.KINDS:
        raise LookupError('未知分类；可选 characters、arcs、challenges')


def integer(name, default, low, high):
    value = int(request.args.get(name, default))
    if not low <= value <= high:
        raise ValueError(f'{name} 须为 {low}–{high}')
    return value


def filters():
    return dict(query=request.args.get('q', '').strip(), element=request.args.get('element', ''),
                rarity=integer('rarity', 5, 1, 5) if request.args.get('rarity') else None)


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
    counts = {key: len(service.catalog()[key]) for key in service.KINDS}
    return render_template('guide/index.html', kind=kind, kinds=service.KINDS, items=items, counts=counts)


@blueprint.get('/guide/<kind>/<identifier>')
def entry(kind, identifier):
    record = get_record(kind, identifier)
    sections = service.page_sections(kind, record)
    art = service.detail(kind, identifier).get('assets') if kind == 'characters' else None
    return render_template('guide/detail.html', record=record, sections=sections, kind=kind,
                           kinds=service.KINDS, art=art, notes=service.schema()['notes'])


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

范围：角色、弧盘、轨外之境。匿名只读；网页正文由服务端渲染。

- [网页入口](/guide)
- [接口与数值约定](/api/guide/schema)
- [角色列表](/api/guide/characters)
- [弧盘列表](/api/guide/arcs)
- [轨外之境赛期](/api/guide/challenges)
- [CLI 下载](/guide/cli.py)

检索：GET /api/guide/search?q=关键词&limit=20&offset=0
详情：GET /api/guide/{characters|arcs|challenges}/{id}
纯文本：GET /guide/{characters|arcs|challenges}/{id}.md
空幕：角色详情加 ?section=equip_slots；升级材料：详情加 ?section=materials。
按字段读取：详情加 ?section=abilities（角色技能）或 ?section=effect（弧盘效果）。
按关卡读取：轨外之境详情加 ?stage=12。
列表返回 total、next_offset 和摘要；详情返回 data。使用 ID 区分重名赛期。
数值数组保留等级/精炼顺序；敌人抗性为小数比例。日期缺失不代表当前开放。
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 search 黑羽
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 get characters 1042 --section abilities
'''
    return Response(text, content_type='text/plain; charset=utf-8')
