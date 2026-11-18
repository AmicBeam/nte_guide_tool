"""Local, read-only combat reference. No dependency on other business modules."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from markupsafe import Markup, escape
from .affinity import with_gift_costs
from .description_parameters import bind_description_parameters
from .challenge_calendar import is_released_challenge, period_sort_key
from . import reference_view, previews, character_release

KINDS = {'characters': '角色', 'arcs': '弧盘', 'foods': '料理', 'permanent': '常驻加成', 'dungeons': '副本', 'challenges': '轨外之境', 'endless': '全站深境', 'diyboss': '争锋赏宴', 'beginner': '攻略'}
BEGINNER_SECTIONS = ('intro', 'gameplay', 'currencies', 'events', 'wishes', 'closing')
LABELS = {
    'generic_stagger': '通用倾陷条', 'enemy_label': '敌人', 'special': '特殊挑战', 'enemies': '敌人',
    'unlock_hours': '活动开始后解锁小时数', 'difficulties': '挑战难度',
    'base_score': '基础积分', 'score_rate': '积分倍率', 'buff_groups': '自选强化',
    'targets': '额外挑战目标', 'target_score_limit': '额外挑战目标积分上限', 'unit': '单位',
    'category': '副本类型', 'required_level': '鉴别等级', 'stamina': '本性像素消耗',
    'encounters': '关卡', 'quantity': '单倍领取数量参考',
    'notes': '加成与升级说明', 'entries': '异象家具加成与升级',
    'max_level_name': '满级名称', 'max_level': '等级上限', 'unlock': '解锁条件',
    'next_level': '升至等级', 'upgrade_costs': '升至下一级要求',
    'required_puka_level': '所需噗卡计划等级', 'fons_cost': '升入本级方斯', 'total_fons': '累计方斯',
    'intro': '前言', 'gameplay': '一.游戏玩法讲解', 'currencies': '二.货币讲解及氪金建议',
    'events': '三.游戏活动讲解', 'wishes': '四.抽卡资源规划', 'closing': '结语',
    'status': '资料状态', 'preview': '预告', 'preview_image': '预告图', 'reference_details': '已确认资料',
    'rules': '楼层机制', 'enhanced_resistances': '强化抗性', 'node_first': '节点起始层', 'node_last': '关底',
    'reserve_character_count': '备选角色数', 'floor_slot_count': '角色栏位数',
    'buff_selection': '关底正向效果与反向效果', 'enabled': '开放选择', 'default_count': '默认正向效果数量', 'count_bonuses': '数量加成',
    'options': '可选组合', 'positive': '正向效果', 'negative': '反向效果', 'target': '目标', 'score': '挑战积分', 'floor': '层数',
    'floor_begin': '起始层', 'floor_end': '结束层', 'single': '敌人配置',
    'total_affinity': '从零到满好感所需好感值', 'required_gifts': '所需礼物份数',
    'full_affinity_total': '满好感总需求',
    'release_at': '资料开放时间（UTC+8）', 'content_released': '技能及后续资料已开放',
    'release_subject_id': '资料开放关联角色',
    'skills_included': '包含技能养成',
    'costs': '礼物花费', 'unit_price': '单价', 'total_cost': '总花费',
    'story_only': '剧情专属料理',
    'bonus_group': '具体加成分类', 'bonus_name': '加成类型', 'fons_price': '最低方斯标价',
    'sources': '获取途径', 'shop': '商店', 'limited': '限时商品', 'requires_unlock': '需要解锁',
    'card_segments': '卡片效果片段', 'effect_segments': '效果片段', 'highlight': '加成数值',
    'food_effect': '料理效果', 'quality': '品质',
    'function': '料理功能', 'function_name': '功能', 'cooldown': '使用冷却（秒）',
    'recommended': '推荐单价', 'percent': '百分比',
    'full_affinity_bonus': '满好感属性加成', 'bonuses': '属性加成', 'reward_texts': '等级奖励',
    'affinity': '好感度', 'available': '已提供好感度配置', 'gifts': '喜好礼物',
    'affinity_gain': '赠送好感值', 'default_gain': '基础好感值', 'exclusive': '专属礼物',
    'unlimited': '不受普通赠礼次数限制', 'offers': '购买渠道', 'shop_name': '商店',
    'shop_ids': '商店 ID', 'goods_ids': '商品 ID', 'currency_name': '货币', 'currency_id': '货币 ID',
    'price': '商店标价', 'discount_price': '配置折扣价', 'discount_time_limited': '限时折扣',
    'availability_limited': '限时商品', 'unlock_id': '解锁条件 ID', 'max_value': '本级好感值上限',
    'groups': '招式分组', 'hit_details': '详细命中数据', 'effect_id': '命中效果 ID',
    'hit_label': '变身名称', 'hit_description': '变身攻击说明',
    'form_ability_id': '变身技能 ID', 'details_only': '仅展示命中明细',
    'hit_index': '命中项', 'repeat': '重复次数', 'harmony_gain': '基础环合',
    'energy_gain': '基础回能', 'stagger': '基础倾陷',
    'frame_gain': '基础帧数', 'score_gain': '基础评分',
    'rating_table': '评分评级标准', 'rating': '评级', 'score_threshold': '分数门槛',
    'frame_multiplier': '帧数获取倍率', 'score_decay_per_second': '每秒掉分', 'rating_note': '评级说明',
    'attack_values': '攻击倍率', 'defense_values': '防御倍率', 'hp_values': '生命倍率',
    'hit_detail_status': '命中数据状态',
    'arc_type': '弧盘适配', 'icon': '图标',
    'assets': '角色图片', 'avatar': '头像', 'portrait': '立绘', 'fullbody': '高清立绘',
    'experience': '等级经验', 'required_exp': '经验需求',
    'equip_slots': '空幕', 'materials': '升级材料', 'breakthrough': '角色突破',
    'skills': '技能升级', 'levels': '升级阶段', 'items': '所需材料',
    'items_id': '材料 ID', 'amount': '数量', 'cost_gold': '甲硬币',
    'require_breakthrough_level': '所需突破阶段', 'stage': '突破阶段',
    'owner_grid_count': '特化驱动格数', 'special_desc': '驱动特化',
    'slots': '格位矩阵', 'value': '数值', 'b_show_percent': '百分比',
    'Melee': '普通攻击', 'Skill': '变轨技能', 'UltraSkill': '极轨终结',
    'QTE': '援护技', 'Passive': '被动', 'Peculiarity': '特性', 'City': '都市技能',
    'base_cooldown': '基础冷却（秒）',
    'param_rows': '技能倍率与参数', 'label': '参数名称', 'key': '参数 ID',
    'type': '类型代码', 'type_id': '类型 ID', 'ability_type': '技能类型',
    'ability_index': '技能序号', 'element': '属性代码', 'element_id': '属性 ID',
    'id_stats': '属性 ID', 'id_value': '参数编号', 'context': '补充说明',
    'previous_season_id': '前一期 ID', 'order': '排序', 'currency': '挑战积分',
    'initial_amount': '初始数量', 'rewards': '奖励', 'index': '序号',
    'required_seals': '所需印记', 'drop_id': '掉落 ID', 'only_half': '仅单半场',
    'spawn_type': '出场方式', 'spawn_time': '出场时间（秒）',
    'class_id': '敌人类型 ID', 'is_boss': '首领', 'attach': '附加说明',
    'image': '图片', 'swallowable': '可被鬼郎丸吞噬',
    'stats': '属性数值', 'abilities': '技能', 'awaken': '觉醒', 'resonance': '共鸣',
    'nouns': '机制词条', 'effect': '弧盘效果', 'stages': '关卡', 'values': '数值',
    'name': '名称', 'id': 'ID', 'description': '说明', 'desc': '说明', 'simple_desc': '简述',
    'element_name': '属性', 'rarity': '星级', 'type_name': '类型', 'phases': '技能阶段',
    'params': '技能参数', 'attributes': '技能属性', 'level': '等级', 'hp': '生命', 'atk': '攻击',
    'def': '防御', 'attack': '攻击', 'defense': '防御', 'is_permanent': '常驻',
    'season': '期数', 'begin_at': '开始时间', 'end_at': '结束时间',
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
    'psychically': '心灵', 'FirstHalf': '上半', 'SecondHalf': '下半',
}


@lru_cache(maxsize=1)
def catalog():
    data = json.loads((Path(__file__).parent / 'data/catalog.json').read_text(encoding='utf-8'))
    art = json.loads((Path(__file__).resolve().parents[2] / 'static/images/characters/index.json').read_text(encoding='utf-8'))
    bindings = {record['id']: record for record in art['characters']}
    metadata = json.loads((Path(__file__).parent / 'data/character_metadata.json').read_text(encoding='utf-8'))
    metadata_by_id = {record['id']: record for record in metadata['characters']}
    affinity = json.loads((Path(__file__).parent / 'data/affinity.json').read_text(encoding='utf-8'))
    affinity_by_id = {record['id']: record['affinity'] for record in affinity['characters']}
    passive_icons = json.loads((Path(__file__).parent / 'data/passive_icons.json').read_text(encoding='utf-8'))
    skill_icons = json.loads((Path(__file__).parent / 'data/skill_icons.json').read_text(encoding='utf-8'))
    for character in data['characters']:
        bind_description_parameters(character)
        for ability in character['abilities']:
            ability['icon'] = skill_icons[str(character['id'])].get(ability['id'])
        for section in ('awaken', 'resonance'):
            passive_bindings = {item['name']: item['icon'] for item in passive_icons[str(character['id'])][section]}
            for item in character[section]:
                item['icon'] = passive_bindings[item['name']]
        character['affinity'] = with_gift_costs(affinity_by_id[str(character['id'])])
        character['experience'] = data['character_experience']
        binding = bindings[str(character['id'])]
        character.update(metadata_by_id[str(character['id'])])
        rank = {4: 'A', 5: 'S'}[character['rarity']]
        character['rarity_icon'] = f'/static/images/rarities/YH_UI_star{rank}.webp'
        character['element_icon'] = f'/static/images/elements/{character["element_name"]}.png'
        character['assets'] = {key: binding[key] for key in ('avatar', 'portrait', 'fullbody')}
    for arc in data['arcs']:
        rank = {3: 'B', 4: 'A', 5: 'S'}[arc['rarity']]
        arc['rarity_icon'] = f'/static/images/rarities/YH_UI_star{rank}.webp'
        arc['image'] = f'/static/images/arcs/{arc["id"]}.webp'
    data['foods'] = json.loads((Path(__file__).parent / 'data/foods.json').read_text(encoding='utf-8'))
    data['permanent'] = json.loads((Path(__file__).parent / 'data/permanent_bonuses.json').read_text(encoding='utf-8'))
    data['endless'] = json.loads((Path(__file__).parent / 'data/endless.json').read_text(encoding='utf-8'))['groups']
    data['diyboss'] = json.loads((Path(__file__).parent / 'data/diyboss.json').read_text(encoding='utf-8'))
    data['beginner'] = json.loads((Path(__file__).parent / 'data/beginner.json').read_text(encoding='utf-8'))
    from . import dungeons
    data['dungeons'] = dungeons.records()
    monsters = json.loads((Path(__file__).parent / 'data/monster_metadata.json').read_text(encoding='utf-8'))
    for challenge in data['challenges'] + data['endless']:
        for stage in challenge['stages']:
            for half in stage['halves']:
                for wave in half['waves']:
                    for monster in wave['monsters']:
                        monster.update(monsters.get(str(monster['id']), {'image': monster.get('image'), 'swallowable': monster.get('swallowable')}))
    return data


def label(key):
    return LABELS.get(str(key), str(key))


def public_records(kind):
    records = catalog()[kind]
    if kind == 'characters':
        complete_ids = {str(record['id']) for record in records}
        records = records + [row for row in previews.characters() if str(row['id']) not in complete_ids]
        now = character_release.current_time()
        return [character_release.project(row, now) for row in records]
    if kind == 'arcs':
        now = character_release.current_time()
        return [character_release.project_arc(row, now) for row in records]
    if kind == 'endless':
        from .endless_seasons import season_records
        return season_records(records)
    if kind == 'challenges':
        return [record for record in records if is_released_challenge(record)]
    return records


def summary(kind, record):
    identifier = str(record['id'])
    result = {k: record[k] for k in ('id', 'name', 'aliases', 'element_name', 'rarity',
        'type_name', 'image', 'element_icon', 'rarity_icon', 'is_permanent', 'begin_date', 'end_date', 'stage_count', 'floor_begin', 'floor_end', 'season', 'begin_at', 'end_at', 'status', 'preview_image', 'release_at', 'content_released') if k in record}
    result.update(kind=kind, url=f'/guide/{kind}/{identifier}',
                  json_url=f'/api/guide/{kind}/{identifier}',
                  markdown_url=f'/guide/{kind}/{identifier}.md')
    if kind == 'foods':
        result['url'] = f'/guide?kind=foods&food={identifier}'
        result.update({key: record[key] for key in ('function', 'function_name', 'effect', 'quality', 'bonus_group', 'bonus_name', 'card_segments', 'effect_segments', 'description', 'sources', 'offers', 'fons_price', 'story_only')})
    if kind == 'characters':
        result['assets'] = record['assets']
    if kind == 'endless':
        result['description'] = record.get('description', '')
    if kind in ('beginner', 'permanent', 'dungeons', 'diyboss'):
        result['description'] = record['description']
    if kind == 'dungeons':
        result['category'] = record['category']
        result['stage_count'] = len(record['encounters'])
    if kind == 'challenges':
        result['description'] = record.get('resonance', {}).get('description', '')
    return result


def search(kind=None, query='', element='', rarity=None, arc_type='', function='', bonus='', hide_story=False,
           material='', category=''):
    result = []
    for group in ([kind] if kind else KINDS):
        records = public_records(group)
        if group in ('challenges', 'endless', 'diyboss'):
            records = sorted(records, key=lambda record: period_sort_key(record, group), reverse=True)
        for record in records:
            if category and record.get('category') != category:
                continue
            if material:
                from . import dungeons
                if group != 'dungeons' or not any(row['id'] in dungeons.material_index().get(material.casefold(), [])
                                                  for row in record['encounters']):
                    continue
            if element and element not in (record.get('element_name'), record.get('element_id'), record.get('element')):
                continue
            if rarity is not None and record.get('rarity') != rarity:
                continue
            if arc_type and str(record.get('type_id', '') if group == 'arcs' else record.get('arc_type', {}).get('id', '')) != arc_type:
                continue
            if function and record.get('function') != function:
                continue
            if bonus and record.get('bonus_group') != bonus:
                continue
            if hide_story and record.get('story_only'):
                continue
            # Full text includes skills, nouns, arc effects and enemy names.
            if query.casefold() not in json.dumps(record, ensure_ascii=False).casefold():
                continue
            result.append(summary(group, record))
    return result


def filter_index_items(kind):
    """Provide all character or arc cards for request-free browser filtering."""
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    return [{**summary(kind, record), 'arc_type_id': str(record['type_id'] if kind == 'arcs' else record.get('arc_type', {}).get('id', '')),
             'filter_text': '\n'.join(dict.fromkeys(strings(record))).casefold()}
            for record in public_records(kind)]


def playable_character(identifier):
    record = detail('characters', identifier)
    if record.get('status') == 'preview':
        raise LookupError('预告角色暂未开放此工具')
    return record


def character_release_status(identifier):
    identifier = '1074' if identifier == 'preview-exe' else identifier
    records = catalog()['characters'] + previews.characters()
    record = next((row for row in records if str(row['id']) == identifier), None)
    if record is None:
        raise LookupError('未找到角色')
    return character_release.release_status(record)


def cultivation_character(identifier):
    identifier = '1074' if identifier == 'preview-exe' else identifier
    record = playable_character(identifier)
    if record.get('content_released') is False:
        raw = next(c for c in catalog()['characters'] if str(c['id']) == identifier)
        record = {**record, 'abilities': [], 'materials': {
            'breakthrough': raw['materials']['breakthrough'], 'skills': []},
            'experience': raw['experience']}
    return record


def detail(kind, identifier, section=None, stage=None, ability_id=None, level=None, *, skill_level=None, hit_details=None, encounter=None):
    if kind == 'characters' and identifier == 'preview-exe':
        identifier = '1074'
    if kind == 'beginner' and section == 'progression':
        section = 'gameplay'
    record = next((r for r in public_records(kind) if str(r['id']) == identifier), None)
    if kind == 'dungeons':
        from . import dungeons
        record = dungeons.find(identifier)
        if encounter is not None:
            candidates = [row for row in record.get('encounters', []) if row['id'] == encounter]
            if not candidates:
                raise LookupError('未找到对应关卡')
            record = {**record, 'encounters': candidates}
        if stage is not None and 'encounters' in record:
            if len(record['encounters']) != 1:
                raise ValueError('请用encounter指定关卡后再选择stage难度')
            child = record['encounters'][0]
            stages = [row for row in child['stages'] if str(row['id']) == str(stage)]
            if not stages:
                raise LookupError('未找到副本难度')
            record = {**record, 'encounters': [{**child, 'stages': stages}]}
            stage = None
    if record is None and kind == 'endless':
        from .endless_seasons import legacy_record
        record = legacy_record(catalog()['endless'], identifier)
    if record is None:
        raise LookupError('未找到条目')
    if record.get('content_released') is False and (section in character_release.AFTER_SKILL_FIELDS or ability_id is not None):
        raise LookupError('角色技能及后续资料尚未开放，开放时间：' + character_release.display_time(record['release_at']))
    if stage is not None:
        if kind not in ('challenges', 'endless', 'dungeons', 'diyboss'):
            raise ValueError('stage 仅适用于轨外之境、全站深境、争锋赏宴或副本')
        record = dict(record)
        record['stages'] = [x for x in record['stages'] if str(x['id']) == str(stage)]
        if not record['stages']:
            raise LookupError('未找到关卡')
    selected = reference_view.select(record, kind, section, ability_id, level, skill_level, hit_details)
    if kind == 'characters':
        selected = {**selected, **{key: record[key] for key in ('release_at', 'content_released') if key in record}}
    if kind == 'beginner':
        selected = {**selected, **{key: record[key] for key in ('description', 'author', 'article_url', 'chapter_titles') if key in record}}
    return selected


def markdown_value(value, depth=2):
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'base_cooldown' and item is None:
                continue
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


def markdown_record(record, *, kind=None):
    if kind == 'foods':
        record = {('food_effect' if key == 'effect' else key): value for key, value in record.items()}
    chapters = record.get('chapter_titles') or {key: label(key) for key in BEGINNER_SECTIONS}
    if 'article_url' in record and any(key in record for key in chapters):
        lines = [f'# {record["name"]}', '', record.get('description', ''), '']
        if 'article_url' in record:
            caption = f'作者原帖（{record["author"]}）' if record.get('author') else '在哔哩哔哩观看'
            lines += [f'[{caption}]({record["article_url"]})', '']
        for key in chapters:
            if key in record:
                lines += [f'## {chapters[key]}', '']
                for block in record[key]:
                    if block['type'] == 'heading':
                        lines += ['### ' + block['text'], '']
                    elif block['type'] == 'formula':
                        lines += ['**' + block['text'].replace('*', '\\*') + '**', '']
                    elif block['type'] == 'link':
                        lines += [f'[{block["text"]}]({block["url"]})', '']
                    elif block['type'] == 'missing':
                        lines += ['> ' + block['text'], '']
                    elif block['type'] == 'video':
                        lines += [f'[{block["title"]}](https://www.bilibili.com/video/{block["bvid"]})', '']
                    elif block['type'] == 'image':
                        lines += [f'![{block["alt"]}]({block["src"]})', '']
                    else:
                        text = block['text']
                        for run in block.get('runs', []):
                            if 'src' in run:
                                text = text.replace(run['alt'], f'![{run["alt"].strip("[]")}]({run["src"]})')
                        lines += [text, '']
        return '\n'.join(lines)
    return f'# {record["name"]}\n\n' + markdown_value(record)


def schema():
    return {'schema_version': 1, 'collections': KINDS,
        'list': '/api/guide/{collection}?q=&element=&rarity=&limit=20&offset=0',
        'search': '/api/guide/search?q={query}',
        'detail': '/api/guide/{collection}/{id}?section={field}&stage={stage_id}&ability_id={ability_id}&level={level}',
        'reference_views': {'overview': 'section=overview返回机制和可读章节，不含所有等级参数或全部关卡。',
            'skill_level': '仅选择角色技能及关联觉醒、共鸣、词条的参数等级，不改变角色属性成长表。CLI默认10级，--all-levels省略此限制。',
            'hit_details': '0隐藏逐命中明细及回能、环合值、倾陷等资源参数行，1保留；不改写机制正文。CLI默认0，--hit-details设为1。公共API未传参数时保持完整数据。',
            'ability_id': '仅角色技能，可省略section或使用section=abilities；按准确技能ID只取一个技能。',
            'level': '可选1–13；values、attack_values等等级数组变为{level,max_level,value}，不足指定档时标明实际最高档。',
            'parameter_keys': '指定技能时，groups中与顶层param_rows相同的行改用parameter_keys引用，按key查找原行。'},
        'markdown': '/guide/{collection}/{id}.md',
        'diyboss_score': {'url': '/api/guide/diyboss/{id}/score', 'method': 'POST',
            'fields': ['stage', 'difficulty', 'buffs', 'targets'],
            'notes': '强化每行最多一项；目标可多选。总分=(基础积分+强化积分+目标积分)×难度倍率。'},
        'articles': {'list': '/api/guide/articles', 'detail': '/api/guide/articles/{uuid}',
                     'markdown': '/guide/articles/{uuid}.md', 'editor': '/guide/write'},
        'beginner': {'id': 'getting-started', 'cli': 'beginner --format markdown',
                     'sections': list(BEGINNER_SECTIONS)},
        'cultivation': {'url': '/api/guide/characters/{id}/cultivation', 'method': 'POST',
            'fields': ['current_level', 'target_level', 'current_ascended', 'target_ascended', 'current_skill_<index>', 'target_skill_<index>'],
            'skill_index': '按角色 materials.skills 顺序；技能等级不含觉醒和共鸣加成',
            'notes': '仅累计跨过阶段；经验书按突破区间配给，甲硬币包含使用费，不扣库存、不自动合成'},
        'hit_detail_fields': {'harmony_gain': 'HeterochromeAdd，单次命中基础环合',
            'energy_gain': 'ChargeAdd，单次命中基础终结回能', 'stagger': 'UnbalValue，单次命中基础倾陷',
            'frame_gain': '单次命中基础帧数，未含评级倍率；缺失字段不表示零',
            'score_gain': '单次命中基础评分，未含共鸣等修正；缺失字段不表示零',
            'repeat': '当前参数展示式中的重复次数，未知为 null，不推算技能实际总命中次数'},
        'character_sections': {'affinity': '好感度等级、文字奖励与解锁、满好感属性、喜好及专属礼物、赠送好感值与各商店标价；不表示玩家当前好感度或实时库存', 'experience': '角色1–80级配置及升入对应等级的经验需求；等级1不计入升级消耗',
            'equip_slots': '空幕格位矩阵、驱动特化和加成属性',
            'materials': '突破与技能升级：每阶段所需材料、数量和突破条件'},
        'notes': ['空幕 slots 中 -1 为不可用格，0 为可装配格；矩阵按从上到下、从左到右排列。',
            '副本amount为可确认的明示数量，null不表示零；quantity合并实测数量参考，约数与范围不保证每次产量。',
            '材料为各目标等级/阶段的单次消耗，不是累计消耗；items 为中文名称与数量，items_id/amount 保留原始配对。','数据为本地快照；请求与启动不会拉取外部数据。',
            '数值数组保留原始顺序：角色属性与弧盘属性按等级，技能参数按技能等级，弧盘效果按精炼档位。',
            '百分比保留原字段表示；敌人 resistances 为小数比例（0.3 = 30%）。',
            '说明中的 {0} 等占位符对应同条目 values 中的 id_value，不自动猜测替换。',
            'null 表示未提供；赛期日期不完整时不能据顺序推断当前开放赛期。',
            '本站提供数据查询，不自动生成配队强度结论。'],
        'labels': LABELS}


def page_sections(kind, record):
    if kind == 'characters' and record.get('content_released') is False:
        return {}
    if kind == 'characters' and record.get('status') == 'preview':
        confirmed = record.get('reference_details', {})
        return {key: confirmed[key] for key in ('abilities', 'awaken', 'resonance') if confirmed.get(key)}
    preferred = {
        'characters': ('abilities', 'awaken', 'resonance', 'nouns', 'equip_slots', 'affinity', 'aliases'),
        'arcs': ('effect', 'exclusive_desc', 'exclusive_character_ids'),
        'foods': ('effect', 'description', 'cooldown'),
        'challenges': ('resonance', 'stages', 'rewards'),
        'endless': ('stages',),
        'diyboss': ('stages',),
        'permanent': ('notes', 'entries', 'levels'),
        'dungeons': ('category', 'encounters', 'stages', 'rewards'),
        'beginner': tuple(record.get('chapter_titles', {})) or BEGINNER_SECTIONS,
    }
    keys = preferred[kind] if len(record) > 3 else tuple(key for key in record if key not in ('id', 'name'))
    result = {key: record[key] for key in keys if key in record and record[key] is not None and record[key] != [] and record[key] != {}}
    if kind in ('characters', 'arcs'):
        for key in ('stats', 'materials', 'experience'):
            result.pop(key, None)
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


def ability_title(ability):
    prefixes = {'melee': '普通攻击', 'skill': '变轨技能', 'ultraskill': '极轨终结',
                'qte': '援护技', 'peculiarity': '特性', 'city': '都市技能', 'passive': '天赋'}
    kind = ability.get('type')
    prefix = prefixes.get(kind, display_scalar(ability.get('type_name', '技能')))
    if kind in ('passive', 'city'):
        index = int(ability.get('ability_index') or 1)
        prefix += {1: '一', 2: '二', 3: '三', 4: '四'}.get(index, str(index))
    name = ability['name']
    return name if name.startswith((prefix + '：', prefix + ':')) else prefix + '：' + name
