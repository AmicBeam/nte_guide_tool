"""Public NTE reference, cultivation calculator and authenticated private imports."""
from pathlib import Path

from flask import Blueprint, Response, redirect, jsonify, render_template, request, send_file
from . import service, cultivation, presentation, arc_cultivation
from .challenge_calendar import is_current_challenge
from .rich_text import description_html
from .character_release import display_time

blueprint = Blueprint('guide_module', __name__, template_folder='templates',
                      static_folder='static', static_url_path='/static/guide')
blueprint.add_app_template_filter(description_html, 'guide_description_html')
blueprint.add_app_template_filter(display_time, 'guide_release_time')
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
        raise LookupError('未知分类；可选 characters、arcs、foods、permanent、dungeons、challenges、endless、diyboss、beginner')


def integer(name, default, low, high):
    value = int(request.args.get(name, default))
    if not low <= value <= high:
        raise ValueError(f'{name} 须为 {low}–{high}')
    return value


def filters():
    story = request.args.getlist('hide_story')[-1] if 'hide_story' in request.args else '0'
    if story not in ('0', '1'):
        raise ValueError('hide_story 须为0或1')
    return dict(query=request.args.get('q', '').strip(), element=request.args.get('element', ''),
                rarity=integer('rarity', 5, 1, 5) if request.args.get('rarity') else None,
                arc_type=request.args.get('arc_type', ''), function=request.args.get('function', ''), bonus=request.args.get('bonus', ''),
                hide_story=story == '1', material=request.args.get('material', ''), category=request.args.get('category', ''))


def get_record(kind, identifier, *, narrow=False):
    kind_check(kind)
    return service.detail(kind, identifier, request.args.get('section'), request.args.get('stage') or None,
                          request.args.get('ability_id') if narrow else None,
                          integer('level', 10, 1, 13) if narrow and 'level' in request.args else None,
                          skill_level=integer('skill_level', 10, 1, 13) if narrow and 'skill_level' in request.args else None,
                          hit_details=bool(integer('hit_details', 0, 0, 1)) if narrow and 'hit_details' in request.args else None,
                          encounter=request.args.get('encounter') if kind == 'dungeons' else None)


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
    selected_filters = filters()
    if kind == 'foods' and 'hide_story' not in request.args:
        selected_filters['hide_story'] = True
    items = service.search(kind, **selected_filters)
    selected_ids = {item['id'] for item in items}
    result_count = len(items)
    if kind in ('characters', 'arcs', 'foods'):
        items = service.filter_index_items(kind)
    from .article_routes import public_items
    authored = public_items(request.args.get('q', '').strip()) if kind == 'beginner' else []
    counts = {key: len(service.public_records(key)) for key in service.KINDS}
    if kind == 'dungeons':
        from . import dungeons
        material = request.args.get('material', '').casefold()
        category = request.args.get('category', '')
        records = [row for row in dungeons.records() if row['id'] in selected_ids]
        material_name = next((reward['name'] for row in records for reward in dungeons.rewards(row)
                              if reward['id'] == material), material)
        return render_template('guide/dungeons_index.html', kind=kind, kinds=service.KINDS, counts=counts,
            records=records, categories=list(dict.fromkeys(row['category'] for row in dungeons.records())),
            selected_material=material, material_name=material_name, selected_category=category)
    return render_template('guide/index.html', kind=kind, kinds=service.KINDS, items=items, counts=counts, selected_ids=selected_ids, result_count=result_count, authored_articles=authored,
        pending_releases=[item for item in items if item.get('content_released') is False],
        hide_story=selected_filters['hide_story'],
        food_groups=list({row['bonus_group']: row['bonus_name'] for row in service.catalog()['foods']}.items()),
        food_asset_version=max((Path(__file__).parent / 'static' / name).stat().st_mtime_ns for name in ('foods.js', 'foods.css')),
        catalog_asset_version=(Path(__file__).parent / 'static/catalog_cards.css').stat().st_mtime_ns,
        challenge_asset_version=(Path(__file__).parent / 'static/challenge_index.css').stat().st_mtime_ns,
        arc_types=sorted({c['arc_type']['id']: c['arc_type'] for c in service.catalog()['characters']}.values(),
                         key=lambda arc: int(arc['id'])))


@blueprint.get('/guide/<kind>/<identifier>')
def entry(kind, identifier):
    if kind == 'dungeons':
        from .dungeons import page_context
        return render_template('guide/dungeon.html', **page_context(service.detail(kind, identifier),
            request.args.get('stage'), request.args.get('material', ''), request.args.get('encounter')))
    profile = service.detail(kind, identifier) if kind == 'characters' else None
    record = profile if profile and profile.get('content_released') is False else get_record(kind, identifier)
    if kind == 'diyboss':
        return render_template('guide/diyboss.html', record=record,
                               configurations=service.catalog()['diyboss'])
    if kind == 'foods':
        return redirect('/guide?kind=foods&food=' + record['id'])
    if kind == 'characters':
        if profile.get('status') == 'preview' and not profile.get('reference_details'):
            return render_template('guide/preview.html', record=profile,
                                   material_preview=presentation.upgrade_material_preview(profile),
                                   pending_releases=[profile] if profile.get('content_released') is False else [])
    sections = service.page_sections(kind, record)
    if profile and profile.get('status') == 'preview':
        profile = {key: profile[key] for key in ('id', 'name', 'status', 'assets', 'rarity',
            'rarity_icon', 'element_name', 'element_icon', 'materials') if key in profile}
    art = profile.get('assets') if profile else None
    overview = profile if profile else (service.detail(kind, identifier) if kind == 'arcs' else None)
    arc_image = overview.get('image') if kind == 'arcs' else None
    from .article_routes import public_items
    attached = public_items(target_kind=kind, target_id=str(record['id']))
    return render_template('guide/detail.html', attached_articles=attached, record=record, sections=sections, kind=kind,
                           kinds=service.KINDS, art=art, profile=profile, overview=overview, arc_image=arc_image,
                           material_preview=presentation.upgrade_material_preview(overview) if overview else [],
                           pending_releases=[profile] if profile and profile.get('content_released') is False else [],
                           notes=service.schema()['notes'], recommended_arc=arc_cultivation.recommended(profile) if profile else None)


@blueprint.get('/api/guide/characters/<identifier>/release')
def character_release_state(identifier):
    return jsonify(data=service.character_release_status(identifier))


@blueprint.after_request
def dynamic_reference_cache(response):
    if request.path == '/guide' or request.path.startswith(('/guide/', '/api/guide/')):
        response.headers['Cache-Control'] = 'no-store'
    return response


@blueprint.get('/guide/characters/<identifier>/axes/<int:axis_id>/compare')
def axis_compare_page(identifier, axis_id):
    if axis_id <= 0:
        raise LookupError('未找到排轴')
    record = service.playable_character(identifier)
    static = Path(__file__).parent.parent / 'shaft' / 'static'
    version = max(int(path.stat().st_mtime) for path in static.glob('js/shaft*.js'))
    arc_rarities = {arc['name']: arc['rarity'] for arc in service.catalog()['arcs']}
    return render_template('shaft/compare.html', compare_asset_version=version,
                           axis_id=axis_id, record=record, arc_rarities=arc_rarities)


@blueprint.get('/guide/characters/<identifier>/cultivation')
def cultivation_page(identifier):
    record = service.cultivation_character(identifier)
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
    record = service.playable_character(identifier)
    static = Path(__file__).parent / 'static'
    version = max(int((static / name).stat().st_mtime) for name in ['related_axes.js', 'related_axes.css'])
    return render_template('guide/builds.html', record=record, related_asset_version=version)


@blueprint.post('/api/guide/characters/<identifier>/cultivation')
def api_cultivation(identifier):
    record = service.cultivation_character(identifier)
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
    return Response(service.markdown_record(get_record(kind, identifier, narrow=True), kind=kind), content_type='text/markdown; charset=utf-8')


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
    selected_filters = filters()
    rows = service.search(kind, **selected_filters)
    if kind in (None, 'beginner') and not any(selected_filters[key] for key in ('element', 'rarity', 'arc_type')):
        from .article_routes import public_items
        rows += public_items(selected_filters['query'])
    return jsonify(schema_version=1, total=len(rows), limit=limit, offset=offset,
                   next_offset=offset + limit if offset + limit < len(rows) else None,
                   items=rows[offset:offset + limit])


@blueprint.get('/api/guide/<kind>/<identifier>')
def api_detail(kind, identifier):
    return jsonify(schema_version=1, kind=kind, data=get_record(kind, identifier, narrow=True))


@blueprint.get('/guide/cli.py')
def cli_download():
    return send_file(Path(__file__).parent / 'cli.py', mimetype='text/x-python', as_attachment=True,
                     download_name='nte-guide.py')


@blueprint.get('/llms.txt')
@blueprint.get('/guide/llms.txt')
def llms():
    text = '''# NTE Tools 攻略数据库

范围：角色、弧盘、料理、常驻加成、副本、轨外之境、全站深境、通用攻略与自建攻略。匿名只读；网页正文由服务端渲染。

- [网页入口](/guide)
- [接口与数值约定](/api/guide/schema)
- [角色列表](/api/guide/characters)
- [弧盘列表](/api/guide/arcs)
- [料理列表](/api/guide/foods)
- [常驻加成（异象家具、噗咔特训）](/api/guide/permanent)
- [副本与材料掉落](/api/guide/dungeons)
- [轨外之境赛期](/api/guide/challenges)
- [全站深境（按期查询）](/api/guide/endless)
- [争锋赏宴（配置与自选强化）](/api/guide/diyboss)
- [攻略](/api/guide/beginner)
- [自建攻略](/api/guide/articles)
- [CLI 下载](/guide/cli.py)

检索：GET /api/guide/search?q=关键词&limit=20&offset=0
详情：GET /api/guide/{characters|arcs|foods|permanent|dungeons|challenges|endless|diyboss|beginner|articles}/{id}
纯文本：GET /guide/{characters|arcs|foods|permanent|dungeons|challenges|endless|diyboss|beginner|articles}/{id}.md
空幕：角色详情加 ?section=equip_slots；升级材料：详情加 ?section=materials。
等级经验：角色详情加 ?section=experience。
好感度与喜好礼物：角色详情加 ?section=affinity，包含赠送数值和各商店标价。
养成计算器：/guide/characters/{id}/cultivation；POST /api/guide/characters/{id}/cultivation 计算资源。
角色排轴与配装：/guide/characters/{id}/builds；固定当前角色筛选相关轴，再进入收益对比。
按字段读取：详情加 ?section=abilities（角色技能）或 ?section=effect（弧盘效果）。
按关卡读取：轨外之境详情加 ?stage=12；全站深境详情加 ?stage=50（实际层号）。
副本列表加 ?material=物品ID 或 category=副本类型；详情加 ?encounter=关卡ID&stage=难度。quantity中的约数与范围仅供估算。
列表返回 total、next_offset 和摘要；详情返回 data。使用 ID 区分重名赛期。
数值数组保留等级/精炼顺序；敌人抗性为小数比例。日期缺失不代表当前开放。
CLI：python nte-guide.py beginner --format markdown
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 search 黑羽
CLI：python nte-guide.py --base-url http://127.0.0.1:5001 get characters 1042 --section abilities
'''
    return Response(text, content_type='text/plain; charset=utf-8')


from .user_routes import register as register_user_routes
register_user_routes(blueprint)



from .challenge_view import stage_selection
blueprint.add_app_template_filter(stage_selection, 'guide_stage_selection')

from .axis_routes import register as register_axis_routes
from .chat_routes import register as register_chat_routes
register_axis_routes(blueprint)
register_chat_routes(blueprint)


from .article_routes import register as register_article_routes
register_article_routes(blueprint)

from .infographic import register as register_infographic_routes
register_infographic_routes(blueprint)

from .tool_routes import register as register_tool_routes
register_tool_routes(blueprint)

from .dungeons import register as register_dungeon_routes
register_dungeon_routes(blueprint)
