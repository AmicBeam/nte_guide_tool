"""Public editor assets and bounded browser-only configuration; no persistence."""
from copy import deepcopy
from pathlib import Path
import re

from flask import render_template
from . import arc_cultivation, presentation, service

MODULES = {'equipment': '弧盘与卡带', 'stats': '参考面板', 'radar': '能力雷达',
           'skills': '技能优先级', 'teams': '配队推荐', 'rotation': '操作与循环', 'advice': '养成与抽取建议'}
TEXT_LIMITS = {'title': 60, 'subtitle': 100, 'author': 40, 'version': 40, 'disclaimer': 160,
               'stats': 650, 'skills': 650, 'rotation': 1200, 'advice': 1000}
RADAR = ['爆发', '持续', '倾陷', '生存', '辅助', '机动']


def options(identifier):
    character = service.playable_character(identifier)
    data = service.catalog()
    characters = [{'id': c['id'], 'name': c['name'], 'element': c['element_name'],
                   'rarity': c['rarity'], 'assets': c['assets'], 'element_icon': c['element_icon'], 'rarity_icon': c['rarity_icon'],
                   'tags': [t['name'] for t in c['char_tags']]} for c in data['characters']]
    cores = [dict(id=key, **value) for key, value in presentation.equipment_images().items() if value['kind'] == 'core']
    return {'character_id': character['id'], 'characters': characters,
            'arcs': [{k: a[k] for k in ('id', 'name', 'image', 'rarity', 'type_id')} for a in arc_cultivation.choices(character)],
            'cores': cores, 'modules': MODULES, 'radar_labels': RADAR,
            'recommended_arc': (arc_cultivation.recommended(character) or {}).get('id', ''),
            'recommended_core': next(iter(presentation.kongmu_links().get(identifier, {}).get('recommendations', [])), {}).get('id', ''),
            'game_version': presentation.site_header()['game_version'] or ''}


def validate(config, available):
    """Validate imported/model-authored data against local IDs and bounded fields."""
    if not isinstance(config, dict):
        raise ValueError('一图流配置须为对象。')
    fields = set(TEXT_LIMITS) | {'orientation', 'layout', 'theme', 'modules', 'order', 'radar', 'plans', 'teams', 'portrait', 'art_scale', 'radar_labels', 'positions'}
    if set(config) != fields:
        raise ValueError('一图流配置字段不完整或包含未知字段。')
    for key, limit in TEXT_LIMITS.items():
        if not isinstance(config[key], str) or len(config[key]) > limit:
            raise ValueError(f'{key} 文字过长或格式无效。')
    for key, values in {'orientation': ('landscape', 'portrait'), 'layout': ('editorial', 'balanced', 'compact'),
                        'theme': ('night', 'paper', 'teal'), 'portrait': ('fullbody', 'portrait', 'avatar')}.items():
        if not isinstance(config[key], str) or config[key] not in values:
            raise ValueError(f'{key} 选项无效。')
    if type(config['art_scale']) not in (int, float) or not 0.7 <= config['art_scale'] <= 1.6:
        raise ValueError('立绘缩放须在0.7–1.6之间。')
    if not isinstance(config['modules'], dict) or set(config['modules']) != set(MODULES) or any(type(x) is not bool for x in config['modules'].values()):
        raise ValueError('模块开关无效。')
    if not isinstance(config['order'], list) or len(config['order']) != len(MODULES) or any(not isinstance(x, str) for x in config['order']) or set(config['order']) != set(MODULES):
        raise ValueError('模块顺序须包含每个模块且不重复。')
    if not isinstance(config['radar'], list) or len(config['radar']) != 6 or any(type(x) is not int or not 0 <= x <= 100 for x in config['radar']):
        raise ValueError('能力评分须为6个0–100整数。')
    if not isinstance(config['radar_labels'], list) or len(config['radar_labels']) != 6 or any(not isinstance(x, str) or not x.strip() or len(x) > 12 for x in config['radar_labels']):
        raise ValueError('雷达维度须为6个非空名称，每个最多12字。')
    if not isinstance(config['positions'], dict) or set(config['positions']) - set(MODULES):
        raise ValueError('板块位置包含未知模块。')
    for position in config['positions'].values():
        if not isinstance(position, dict) or set(position) != {'x', 'y'} or any(type(v) not in (int, float) or not 0 <= v <= 16000 for v in position.values()):
            raise ValueError('板块坐标须为0–16000的有限数值。')
    arc_ids = {a['id'] for a in available['arcs']} | {''}
    core_ids = {a['id'] for a in available['cores']} | {''}
    char_ids = {a['id'] for a in available['characters']} | {''}
    for key in ('plans', 'teams'):
        if not isinstance(config[key], list) or len(config[key]) != 2:
            raise ValueError('配装和配队各保留两个可选方案。')
    for plan in config['plans']:
        if not isinstance(plan, dict) or set(plan) != {'name', 'arc', 'core', 'main', 'sub', 'note'}:
            raise ValueError('配装方案字段无效。')
        if not isinstance(plan['arc'], str) or not isinstance(plan['core'], str) or plan['arc'] not in arc_ids or plan['core'] not in core_ids:
            raise ValueError('请选择当前角色适配的弧盘和已收录卡带。')
        for key, limit in [('name', 40), ('main', 200), ('sub', 200), ('note', 400)]:
            if not isinstance(plan[key], str) or len(plan[key]) > limit:
                raise ValueError('配装说明过长或格式无效。')
    for team in config['teams']:
        if not isinstance(team, dict) or set(team) != {'name', 'members', 'candidates', 'folded', 'note'}:
            raise ValueError('队伍字段无效。')
        if not isinstance(team['members'], list) or len(team['members']) != 4 or any(not isinstance(c, str) or c not in char_ids for c in team['members']):
            raise ValueError('队员须使用本站已收录的角色编号。')
        if not isinstance(team['candidates'], list) or len(team['candidates']) != 4 or not isinstance(team['folded'], list) or len(team['folded']) != 4 or any(type(v) is not bool for v in team['folded']):
            raise ValueError('每个格子须有候选列表及折叠开关。')
        for i, candidates in enumerate(team['candidates']):
            if not isinstance(candidates, list) or len(candidates) > len(available['characters']) or any(not isinstance(c, str) or c not in char_ids for c in candidates):
                raise ValueError('候选须为本站已收录角色，每格可加入全部角色。')
            selected = [c for c in [team['members'][i], *candidates] if c]
            if len(selected) != len(set(selected)):
                raise ValueError('同一格中的首选与候选不能重复。')
        if not isinstance(team['name'], str) or len(team['name']) > 40 or not isinstance(team['note'], str) or len(team['note']) > 500:
            raise ValueError('配队说明过长或格式无效。')
    return deepcopy(config)


def editor_context(raw, path):
    if raw is None:
        return None
    match = re.fullmatch(r'/guide/characters/([0-9]+)/infographic', path)
    if not match or not isinstance(raw, dict) or set(raw) != {'character_id', 'revision', 'config'} or raw['character_id'] != match[1]:
        raise ValueError('当前页面不支持一图流编辑。')
    if type(raw['revision']) is not int or not 0 <= raw['revision'] < 2**53:
        raise ValueError('一图流版本无效。')
    available = options(match[1])
    return {**raw, 'config': validate(raw['config'], available), 'options': available}


def register(blueprint):
    @blueprint.get('/guide/characters/<identifier>/infographic')
    def infographic_page(identifier):
        record = service.playable_character(identifier)
        data = options(identifier)
        static = Path(__file__).parent / 'static'
        version = max((static / name).stat().st_mtime_ns for name in ('infographic.js', 'infographic_state.js', 'infographic_canvas.js', 'infographic_board.js', 'infographic.css'))
        return render_template('guide/infographic.html', record=record, infographic_data=data, infographic_version=version)
