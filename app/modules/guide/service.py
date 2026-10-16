"""Local, read-only combat reference. No dependency on other business modules."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

KINDS = {'characters': '角色', 'arcs': '弧盘', 'challenges': '轨外之境'}
LABELS = {
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
    'resonance_level': '共鸣等级', 'exclusive_character_ids': '专属角色 ID',
    'exclusive_desc': '专属效果', 'aliases': '别名', 'ability_name': '异能',
    'char_tags': '定位', 'normal': '常规', 'cosmos': '光', 'nature': '灵',
    'incantation': '咒', 'chaos': '暗', 'psyche': '魂', 'lakshana': '相',
    'psychically': '异能', 'FirstHalf': '上半', 'SecondHalf': '下半',
}


@lru_cache(maxsize=1)
def catalog():
    return json.loads((Path(__file__).parent / 'data/catalog.json').read_text(encoding='utf-8'))


def label(key):
    return LABELS.get(str(key), str(key))


def summary(kind, record):
    identifier = str(record['id'])
    result = {k: record[k] for k in ('id', 'name', 'aliases', 'element_name', 'rarity',
        'type_name', 'is_permanent', 'begin_date', 'end_date', 'stage_count') if k in record}
    result.update(kind=kind, url=f'/guide/{kind}/{identifier}',
                  json_url=f'/api/guide/{kind}/{identifier}',
                  markdown_url=f'/guide/{kind}/{identifier}.md')
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
        'notes': ['数据为本地快照；请求与启动不会拉取外部数据。',
            '数值数组保留原始顺序：角色属性与弧盘属性按等级，技能参数按技能等级，弧盘效果按精炼档位。',
            '百分比保留原字段表示；敌人 resistances 为小数比例（0.3 = 30%）。',
            '说明中的 {0} 等占位符对应同条目 values 中的 id_value，不自动猜测替换。',
            'null 表示未提供；赛期日期不完整时不能据顺序推断当前开放赛期。',
            '本站提供数据查询，不自动生成配队强度结论。'],
        'labels': LABELS}


def page_sections(kind, record):
    preferred = {
        'characters': ('abilities', 'awaken', 'resonance', 'nouns', 'stats', 'aliases'),
        'arcs': ('effect', 'exclusive_desc', 'exclusive_character_ids', 'stats'),
        'challenges': ('resonance', 'stages', 'rewards'),
    }
    keys = preferred[kind] if len(record) > 3 else tuple(key for key in record if key not in ('id', 'name'))
    result = {key: record[key] for key in keys if key in record and record[key] is not None}
    if isinstance(result.get('nouns'), dict):
        result['nouns'] = list(result['nouns'].values())
    return result


def display_scalar(value):
    if isinstance(value, str) and value in ('cosmos', 'nature', 'incantation', 'chaos', 'psyche', 'lakshana', 'normal', 'psychically'):
        return label(value)
    return scalar(value)
