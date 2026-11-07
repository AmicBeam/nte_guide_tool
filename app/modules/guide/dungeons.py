"""Offline dungeon references and explicit material acquisition links."""
from functools import lru_cache
import json
from pathlib import Path
from urllib.parse import urlencode

GROUPS = {
    'resources': ('胡迪尼的魔术舞台', '经验与甲硬币', '角色经验、弧盘经验与甲硬币。'),
    'skills': ('胡迪尼的诡计舞台', '技能材料', '异能升级所需的各类技能材料。'),
    'arcs': ('泡影罐头工厂', '弧盘突破', '五种类型的弧盘突破材料。'),
    'equipment': ('兔子洞', '空幕', '核心卡带、空幕经验材料与萝卜币。'),
    'weekly': ('异象巡礼', '异象巡礼', '挑战周本Boss，获取高级异能升级材料。'),
    'world': ('异象追猎', '异象追猎', '挑战地图Boss，获取角色突破材料。'),
}


def approximate(low, high=None):
    return '约' + str(low) + ('–' + str(high) if high is not None and high != low else '') + '个'


def quantity(identifier, category, level, amount):
    """Use observed ranges, without deriving means or unobserved levels."""
    material_ranges = {
        '技能材料': {30: {2: (3, 3), 1: (5, 5)},
                 40: {3: (1, 1), 2: (1, 2), 1: (2, 3)}, 45: {3: (1, 1), 2: (2, 2), 1: (3, 3)}},
        '弧盘突破': {30: {2: (3, 3), 1: (1, 6)},
                 40: {3: (1, 1), 2: (1, 2), 1: (1, 3)}, 45: {3: (1, 1), 2: (2, 2), 1: (3, 3)}},
    }
    if category in material_ranges and identifier.startswith(('skillupmaterial_', 'weaponbreakmaterial_')):
        tier = int(identifier.rsplit('_lv', 1)[-1])
        bounds = material_ranges[category].get(level, {}).get(tier)
        if bounds:
            return approximate(*bounds)
    if identifier.startswith(('characterupmaterial_', 'weaponupmaterial_')):
        tier = int(identifier.rsplit('_lv', 1)[-1])
        counts = ({30: [5, 7, 7], 40: [7, 7, 8], 45: [5, 4, 10]}
                  if identifier.startswith('character') else
                  {30: [7, 7, 8], 40: [7, 8, 9], 45: [10, 8, 10]})
        if category == '经验与甲硬币' and level in counts:
            return approximate(counts[level][tier - 1])
    if identifier.startswith('worldboss_material_'):
        bounds = {30: (6, 6), 40: (6, 7), 45: (7, 8)}.get(level)
        if bounds:
            return approximate(*bounds)
    if category == '异象巡礼' and identifier.startswith('weeklycloneboss_'):
        bounds = {30: (2, 3), 40: (2, 3), 45: (3, 3)}.get(level)
        if bounds:
            return approximate(*bounds)
    if category == '空幕':
        coins = {'equipment_orangeticket': {30: 40, 40: 120, 45: 200, 50: 280},
                 'equipment_purpleticket': {30: 280, 40: 240, 45: 160, 50: 80}}
        if level in coins.get(identifier, {}):
            return approximate(coins[identifier][level])
    return '×' + format(amount, ',') if amount is not None else '数量未确认'


@lru_cache(maxsize=1)
def encounters():
    root = Path(__file__).parent / 'data'
    rows = json.loads((root / 'dungeons.json').read_text(encoding='utf-8'))
    for row in rows:
        if not row['stages'] and row['category'] == '异象追猎':
            row['stages'] = [{'id': level, 'name': '鉴别等级 ' + str(level), 'required_level': level,
                              'stamina': None, 'rewards': row['rewards']} for level in (30, 40, 45)]
        row['stages'] = [{**stage, 'rewards': [{**reward,
            'quantity': quantity(reward['id'], row['category'], stage['required_level'], reward['amount'])}
            for reward in stage['rewards']]} for stage in row['stages']]
    return rows


@lru_cache(maxsize=1)
def records():
    return [{'id': key, 'name': name, 'category': category, 'description': description,
             'encounters': [row for row in encounters() if row['category'] == category]}
            for key, (name, category, description) in GROUPS.items()]


def find(identifier):
    record = next((row for row in records() + encounters() if row['id'] == identifier), None)
    if record is None:
        raise LookupError('未找到副本')
    return record


def group_for(encounter):
    return next(group for group in records() if group['category'] == encounter['category'])


def rewards(record):
    if 'encounters' in record:
        return [reward for encounter in record['encounters'] for reward in rewards(encounter)]
    return record.get('rewards', []) + [reward for stage in record['stages'] for reward in stage['rewards']]


@lru_cache(maxsize=1)
def material_index():
    result = {}
    for record in encounters():
        for reward in rewards(record):
            matches = result.setdefault(reward['id'].casefold(), [])
            if record['id'] not in matches:
                matches.append(record['id'])
    return result


def material_url(identifier):
    identifier = str(identifier).casefold()
    matches = material_index().get(identifier, [])
    if len(matches) == 1:
        encounter = find(matches[0])
        return '/guide/dungeons/' + group_for(encounter)['id'] + '?' + urlencode(
            {'encounter': encounter['id'], 'material': identifier}) + '#rewards'
    if matches:
        return '/guide?' + urlencode({'kind': 'dungeons', 'material': identifier})
    return None


def display_rewards(record, stage):
    result = []
    orange = []
    for reward in stage['rewards']:
        if record['category'] == '空幕' and reward['id'].endswith('_orange'):
            orange.append(reward)
        else:
            result.append(reward)
    if orange:
        bounds = {30: (0, 1), 40: (1, 1), 45: (1, 2), 50: (2, 3)}.get(stage['required_level'])
        result.insert(0, {'id': 'orange-cartridges', 'name': '金色卡带（' + ' / '.join(
            dict.fromkeys(row['name'] for row in orange)) + '）', 'amount': None,
            'items': [{'id': row['id'], 'name': row['name']} for row in orange],
            'quantity': approximate(*bounds) if bounds else '数量未确认'})
    return result


def page_context(record, stage=None, material='', encounter=None):
    if 'encounters' in record:
        group = record
        candidates = record['encounters']
        selected_encounter = next((row for row in candidates if row['id'] == encounter), None)
        if encounter and selected_encounter is None:
            raise LookupError('未找到对应关卡')
        if selected_encounter is None and material:
            selected_encounter = next((row for row in candidates if any(
                reward['id'] == material.casefold() for reward in rewards(row))), None)
        selected_encounter = selected_encounter or candidates[0]
    else:
        group, selected_encounter = group_for(record), record
    selected = next((row for row in selected_encounter['stages'] if str(row['id']) == str(stage)), None)
    if stage is not None and selected is None:
        raise LookupError('未找到副本难度')
    selected = selected or next((row for row in selected_encounter['stages'] if row['required_level'] == 45),
                                selected_encounter['stages'][-1])
    return {'record': group, 'selected_encounter': selected_encounter, 'selected_stage': selected,
            'selected_material': material.casefold(), 'rewards': display_rewards(selected_encounter, selected)}


def register(blueprint):
    from flask import redirect

    @blueprint.get('/guide/dungeons')
    def dungeon_directory():
        return redirect('/guide?kind=dungeons')
