"""Local, read-only combat reference. No dependency on other business modules."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from markupsafe import Markup, escape

KINDS = {'characters': '角色', 'arcs': '弧盘', 'challenges': '轨外之境'}
LABELS = {
    'groups': '招式分组', 'hit_details': '详细命中数据', 'effect_id': '命中效果 ID',
    'hit_index': '命中项', 'repeat': '重复次数', 'harmony_gain': '基础环合',
    'energy_gain': '基础回能', 'stagger': '基础倾陷',
    'attack_values': '攻击倍率', 'defense_values': '防御倍率', 'hp_values': '生命倍率',
    'hit_detail_status': '命中数据状态',
    'assets': '角色图片', 'avatar': '头像', 'portrait': '立绘', 'fullbody': '高清立绘',
    'equip_slots': '空幕', 'materials': '升级材料', 'breakthrough': '角色突破',
    'skills': '技能升级', 'levels': '升级阶段', 'items': '所需材料',
    'items_id': '材料 ID', 'amount': '数量', 'cost_gold': '甲硬币',
    'require_breakthrough_level': '所需突破阶段', 'stage': '突破阶段',
    'owner_grid_count': '特化驱动格数', 'special_desc': '驱动特化',
    'slots': '格位矩阵', 'value': '数值', 'b_show_percent': '百分比',
    'Melee': '普通攻击', 'Skill': '变轨技能', 'UltraSkill': '极轨终结',
    'QTE': '援护技', 'Passive': '被动', 'Peculiarity': '特性', 'City': '都市技能',
    'param_rows': '技能倍率与参数', 'label': '参数名称', 'key': '参数 ID',
    'type': '类型代码', 'type_id': '类型 ID', 'ability_type': '技能类型',
    'ability_index': '技能序号', 'element': '属性代码', 'element_id': '属性 ID',
    'id_stats': '属性 ID', 'id_value': '参数编号', 'context': '补充说明',
    'previous_season_id': '前一期 ID', 'order': '排序', 'currency': '挑战积分',
    'initial_amount': '初始数量', 'rewards': '奖励', 'index': '序号',
    'required_seals': '所需印记', 'drop_id': '掉落 ID', 'only_half': '仅单半场',
    'spawn_type': '出场方式', 'spawn_time': '出场时间（秒）',
    'class_id': '敌人类型 ID', 'is_boss': '首领', 'attach': '附加说明',
    'stats': '属性数值', 'abilities': '技能', 'awaken': '觉醒', 'resonance': '共鸣',
    'nouns': '机制词条', 'effect': '弧盘效果', 'stages': '关卡', 'values': '数值',
    'name': '名称', 'id': 'ID', 'description': '说明', 'desc': '说明', 'simple_desc': '简述',
    'element_name': '属性', 'rarity': '星级', 'type_name': '类型', 'phases': '技能阶段',
    'params': '技能参数', 'attributes': '技能属性', 'level': '等级', 'hp': '生命', 'atk': '攻击',
    'def': '防御', 'attack': '攻击', 'defense': '防御', 'is_permanent': '常驻',
    'begin_date': '开始日期', 'end_date': '结束日期', 'stage_count': '关卡数',
    'maximum_time': '限时（秒）', 'objectives': '目标', 'halves': '半场', 'waves': '波次',
    'wave': '波次', 'monsters': '敌人', 'count': '数量', 'weaknesses': '弱点',
    'recommended_elements': '推荐属性', 'resistances': '抗性（小数比例）',
    'unbalance': '倾陷上限', 'unbalance_duration': '倾陷持续（秒）', 'text': '说明',
    'remaining_time': '剩余时间（秒）', 'b_is_percent': '百分比', 'title': '名称',
    'resonance_level': '共鸣等级', 'exclusive_character_ids': '专属角色',
    'exclusive_desc': '专属效果', 'aliases': '别名', 'ability_name': '异能',
    'char_tags': '定位', 'normal': '常规', 'cosmos': '光', 'nature': '灵',
    'incantation': '咒', 'chaos': '暗', 'psyche': '魂', 'lakshana': '相',
    'psychically': '异能', 'FirstHalf': '上半', 'SecondHalf': '下半',
}


@lru_cache(maxsize=1)
def catalog():
    data = json.loads((Path(__file__).parent / 'data/catalog.json').read_text(encoding='utf-8'))
    art = json.loads((Path(__file__).resolve().parents[2] / 'static/images/characters/index.json').read_text(encoding='utf-8'))
    bindings = {record['id']: record for record in art['characters']}
    for character in data['characters']:
        binding = bindings[str(character['id'])]
        character['element_icon'] = f'/static/images/elements/{character["element_name"]}.png'
        character['assets'] = {key: binding[key] for key in ('avatar', 'portrait', 'fullbody')}
    for arc in data['arcs']:
        arc['image'] = f'/static/images/arcs/{arc["id"]}.webp'
    return data


def label(key):
    return LABELS.get(str(key), str(key))


def summary(kind, record):
    identifier = str(record['id'])
    result = {k: record[k] for k in ('id', 'name', 'aliases', 'element_name', 'rarity',
        'type_name', 'image', 'element_icon', 'is_permanent', 'begin_date', 'end_date', 'stage_count') if k in record}
    result.update(kind=kind, url=f'/guide/{kind}/{identifier}',
                  json_url=f'/api/guide/{kind}/{identifier}',
                  markdown_url=f'/guide/{kind}/{identifier}.md')
    if kind == 'characters':
        result['assets'] = record['assets']
    if kind == 'challenges':
        result['description'] = record.get('resonance', {}).get('description', '')
    return result


def search(kind=None, query='', element='', rarity=None):
    result = []
    for group in ([kind] if kind else KINDS):
        for record in catalog()[group]:
            if element and element not in (record.get('element_name'), record.get('element_id'), record.get('element')):
                continue
            if rarity is not None and record.get('rarity') != rarity:
                continue
            # Full text includes skills, nouns, arc effects and enemy names.
            if query.casefold() not in json.dumps(record, ensure_ascii=False).casefold():
                continue
            result.append(summary(group, record))
    return result


def detail(kind, identifier, section=None, stage=None):
    record = next((r for r in catalog()[kind] if str(r['id']) == identifier), None)
    if record is None:
        raise LookupError('未找到条目')
    if stage is not None:
        if kind != 'challenges':
            raise ValueError('stage 仅适用于轨外之境')
        record = dict(record)
        record['stages'] = [x for x in record['stages'] if str(x['id']) == str(stage)]
        if not record['stages']:
            raise LookupError('未找到关卡')
    if section:
        if section not in record:
            raise LookupError('未找到指定字段')
        record = {'id': record['id'], 'name': record['name'], section: record[section]}
    return record


def markdown_value(value, depth=2):
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines += [f'{"#" * min(depth, 6)} {label(key)}', '', markdown_value(item, depth + 1)]
            else:
                lines.append(f'- {label(key)}: {scalar(item)}')
    elif isinstance(value, list):
        if all(not isinstance(x, (dict, list)) for x in value):
            lines.append(' / '.join(scalar(x) for x in value))
        else:
            for index, item in enumerate(value, 1):
                title = item.get('name') or item.get('title') or str(index) if isinstance(item, dict) else str(index)
                lines += [f'{"#" * min(depth, 6)} {title}', '', markdown_value(item, depth + 1)]
    else:
        lines.append(scalar(value))
    return '\n'.join(lines) + '\n'


def scalar(value):
    if value is None:
        return '未提供'
    if isinstance(value, bool):
        return '是' if value else '否'
    return str(value)


def markdown_record(record):
    return f'# {record["name"]}\n\n' + markdown_value(record)


def schema():
    return {'schema_version': 1, 'collections': KINDS,
        'list': '/api/guide/{collection}?q=&element=&rarity=&limit=20&offset=0',
        'search': '/api/guide/search?q={query}',
        'detail': '/api/guide/{collection}/{id}?section={field}&stage={stage_id}',
        'markdown': '/guide/{collection}/{id}.md',
        'hit_detail_fields': {'harmony_gain': 'HeterochromeAdd，单次命中基础环合',
            'energy_gain': 'ChargeAdd，单次命中基础终结回能', 'stagger': 'UnbalValue，单次命中基础倾陷',
            'repeat': '当前参数展示式中的重复次数，未知为 null，不推算技能实际总命中次数'},
        'character_sections': {'equip_slots': '空幕格位矩阵、驱动特化和加成属性',
            'materials': '突破与技能升级：每阶段所需材料、数量和突破条件'},
        'notes': ['空幕 slots 中 -1 为不可用格，0 为可装配格；矩阵按从上到下、从左到右排列。',
            '材料为各目标等级/阶段的单次消耗，不是累计消耗；items 为中文名称与数量，items_id/amount 保留原始配对。','数据为本地快照；请求与启动不会拉取外部数据。',
            '数值数组保留原始顺序：角色属性与弧盘属性按等级，技能参数按技能等级，弧盘效果按精炼档位。',
            '百分比保留原字段表示；敌人 resistances 为小数比例（0.3 = 30%）。',
            '说明中的 {0} 等占位符对应同条目 values 中的 id_value，不自动猜测替换。',
            'null 表示未提供；赛期日期不完整时不能据顺序推断当前开放赛期。',
            '本站提供数据查询，不自动生成配队强度结论。'],
        'labels': LABELS}


def page_sections(kind, record):
    preferred = {
        'characters': ('abilities', 'equip_slots', 'materials', 'awaken', 'resonance', 'nouns', 'stats', 'aliases'),
        'arcs': ('stats', 'effect', 'exclusive_desc', 'exclusive_character_ids', 'materials'),
        'challenges': ('resonance', 'stages', 'rewards'),
    }
    keys = preferred[kind] if len(record) > 3 else tuple(key for key in record if key not in ('id', 'name'))
    result = {key: record[key] for key in keys if key in record and record[key] is not None and record[key] != [] and record[key] != {}}
    if isinstance(result.get('nouns'), dict):
        result['nouns'] = list(result['nouns'].values())
    return result


def display_scalar(value):
    if isinstance(value, str) and value in ('Melee', 'Skill', 'UltraSkill', 'QTE', 'Passive', 'Peculiarity', 'City', 'cosmos', 'nature', 'incantation', 'chaos', 'psyche', 'lakshana', 'normal', 'psychically'):
        return label(value)
    return scalar(value)


def equip_description(equipment):
    description = equipment.get('desc') or ''
    stats = equipment.get('stats') or []
    if len(stats) == 1:
        description = description.replace('{0}', str(stats[0]['name'])).replace('{1}', str(stats[0]['value']))
    return description


def character_name(identifier):
    return next((c['name'] for c in catalog()['characters'] if str(c['id']) == str(identifier)), str(identifier))


def effect_html(template, parameters, level):
    values = {str(row['id_value']): row['values'][level - 1]
              for row in parameters if level <= len(row['values'])}
    text = re.sub(r'\{(\d+)\}', lambda m: str(values.get(m[1], m[0])), template)
    parts = []
    previous = 0
    for match in re.finditer(r'[+-]?\d+(?:\.\d+)?%?', text):
        parts.append(escape(text[previous:match.start()]))
        parts.append(Markup('<span class="guide-effect-number">') + escape(match[0]) + Markup('</span>'))
        previous = match.end()
    parts.append(escape(text[previous:]))
    return Markup('').join(parts)
