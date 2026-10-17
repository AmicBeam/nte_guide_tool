"""Explicit, non-mutating projections for narrow public reference reads."""
LEVEL_FIELDS = {'values', 'attack_values', 'defense_values', 'hp_values'}


def ability_parameters(ability):
    """Reference identical parameter rows rather than sending them twice."""
    rows = {row.get('key'): row for row in ability.get('param_rows', []) if row.get('key')}
    groups = []
    for group in ability.get('groups', []):
        references, extra = [], []
        for row in group.get('param_rows', []):
            if row.get('key') in rows and rows[row['key']] == row:
                references.append(row['key'])
            else:
                extra.append(row)
        result = {key: value for key, value in group.items() if key != 'param_rows'}
        if references:
            result['parameter_keys'] = references
        if extra:
            result['param_rows'] = extra
        groups.append(result)
    return {**ability, 'groups': groups}


def at_level(value, level):
    if isinstance(value, list):
        return [at_level(item, level) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in LEVEL_FIELDS and isinstance(item, list) and item and all(
                isinstance(entry, (str, int, float, bool, type(None))) for entry in item):
            selected = min(level, len(item))
            result[key] = {'level': selected, 'max_level': len(item), 'value': item[selected - 1]}
        else:
            result[key] = at_level(item, level)
    return result


def overview(kind, record, level=10):
    result = {key: value for key, value in record.items() if key in (
        'id', 'name', 'aliases', 'element_name', 'rarity', 'arc_type', 'char_tags',
        'type_name', 'effect', 'exclusive_desc', 'exclusive_character_ids',
        'is_permanent', 'begin_date', 'end_date', 'resonance', 'description',
        'floor_begin', 'floor_end', 'stage_count')}
    result['available_sections'] = [key for key in record if key not in ('id', 'name')]
    if kind == 'characters':
        result['abilities'] = [{key: value for key, value in ability.items() if key in (
            'id', 'name', 'type_name', 'context', 'phases')} for ability in record['abilities']]
        for ability in result['abilities']:
            ability['phases'] = [{key: value for key, value in phase.items() if key != 'simple_desc'}
                                 for phase in ability.get('phases', [])]
        for key in ('awaken', 'nouns', 'equip_slots'):
            result[key] = at_level(record.get(key), level)
        result['read_hint'] = '机制概览不含技能倍率。精确参数请用section=abilities、ability_id指定技能、level指定等级；其余资料按available_sections读取。'
    if 'stages' in record:
        result['stages'] = [{key: value for key, value in stage.items() if key in ('id', 'name')}
                            for stage in record['stages']]
        result['read_hint'] = '此处仅列关卡目录；敌人和数值请用stage指定一关。'
    return result


def select(record, kind, section=None, ability_id=None, level=None):
    if level is not None and (isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 13):
        raise ValueError('level 须为1–13')
    if ability_id is not None:
        if kind != 'characters' or section not in (None, 'abilities'):
            raise ValueError('ability_id 仅适用于角色技能章节')
        abilities = [ability for ability in record['abilities'] if ability['id'] == ability_id]
        if not abilities:
            raise LookupError('未找到指定技能')
        record = {**record, 'abilities': [ability_parameters(ability) for ability in abilities]}
        section = 'abilities'
    if section == 'overview':
        return overview(kind, record, level if level is not None else 10)
    if section:
        if section not in record:
            raise LookupError('未找到指定字段')
        record = {'id': record['id'], 'name': record['name'], section: record[section]}
    if level is not None:
        record = at_level(record, level)
        record = {**record, 'requested_level': level,
                  'level_note': 'values等等级数组仅返回所选档；不足该等级时取已有最高档，level标明实际档位。'}
    return record
