"""Offline linkage of skill display parameters to individual damage effects."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess

TABLES = ('DT_GameplayAbilityTipsData', 'DT_SkillDamageData')


def table_rows(payload):
    return next(item['Rows'] for item in payload if item.get('Rows') is not None)


def load_tables(directory=None):
    result = {}
    for name in TABLES:
        if directory is not None:
            payload = json.loads((Path(directory) / f'{name}.json').read_text(encoding='utf-8'))
        else:
            data = subprocess.run(['gh', 'api', f'repos/mc-ctrl/NTEData/contents/DataTable/Skill/{name}.json',
                '-H', 'Accept: application/vnd.github.raw+json'], capture_output=True, check=True)
            payload = json.loads(data.stdout)
        result[name] = table_rows(payload)
    return result


def percentage(number):
    text = f'{Decimal(str(number)) * 100:f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text + '%'


def plain(value):
    return re.sub(r'<[^>]*>', '', str(value or '')).strip()


def display_text(value):
    return plain((value or {}).get('LocalizedString') or (value or {}).get('SourceString'))


def select_group(row, phases):
    label = row['label']
    # Explicit move names take precedence. A row mentioning several moves is shared.
    names = [(i, phase['name'].split('：')[-1].split(':')[-1].strip()) for i, phase in enumerate(phases)]
    matches = [(i, name) for i, name in names if name and name in label]
    if matches:
        longest = max(len(name) for _, name in matches)
        selected = [i for i, name in matches if len(name) == longest]
        # Different names in one label represent a shared rule, not just the longest name.
        distinct = [name for _, name in matches if not any(name != other and name in other for _, other in matches)]
        if len(distinct) == 1 and len(selected) == 1:
            return selected[0]
        return None
    key = row.get('key', '').lower()
    tests = []
    if 'airattack' in key or 'airatk' in key or '下落' in label:
        tests = ['下落', '滞空', '空中']
    elif 'perfectevade' in key or '极限反击' in label or '闪避反击' in label:
        tests = ['极限反击', '闪避反击']
    elif 'parry' in key or '承轨反击' in label:
        tests = ['承轨反击']
    elif '瞄准' in label or 'aimattack' in key:
        tests = ['瞄准']
    if tests:
        candidates = [i for i, phase in enumerate(phases) if any(t in phase['name'] for t in tests)]
        if len(candidates) == 1:
            return candidates[0]
        candidates = [i for i, phase in enumerate(phases) if phase['name'].startswith('普通攻击')
                      and tests not in (['承轨反击'], ['极限反击', '闪避反击'])
                      and any(t in phase.get('desc', '') for t in tests)]
        if len(candidates) == 1:
            return candidates[0]
        return None
    if re.match(r'^[一二三四五六七八九十\d]+段', label):
        candidates = [i for i, phase in enumerate(phases) if phase['name'].startswith('普通攻击')]
        if candidates:
            return candidates[0]
    for trigger in ('短按', '长按'):
        if trigger in label:
            candidates = [i for i, phase in enumerate(phases) if trigger in phase.get('desc', '')]
            if len(candidates) == 1:
                return candidates[0]
    if len(phases) == 1:
        return 0
    return None


def enrich_ability(ability, tables):
    tips = tables['DT_GameplayAbilityTipsData'].get(ability['id'])
    if tips is None and ability.get('param_rows'):
        raise ValueError(f'Missing skill tips: {ability["id"]}')
    tips = tips or {}
    bindings = {}
    for parameter in tips.get('GALevelUpArray') or []:
        key = (parameter.get('DescValue') or {}).get('Key')
        if key:
            bindings[key.casefold()] = parameter
    damages = tables['DT_SkillDamageData']
    for row in ability.get('param_rows') or []:
        parameter = bindings.get(row.get('key', '').casefold())
        row['hit_details'] = []
        if parameter is None:
            row['hit_detail_status'] = 'not_provided'
            continue
        up = parameter.get('LevelUpData') or {}
        effects = [effect for effect in up.get('DamageGEList') or [] if effect and effect != 'None']
        template = display_text(parameter.get('DescValue'))
        for index, effect in enumerate(effects):
            raw = damages.get(effect)
            if raw is None:
                raise ValueError(f'Missing damage effect: {effect}')
            # Multiplicity is recorded only for a single unambiguous placeholder occurrence.
            occurrences = list(re.finditer(r'\{' + str(index) + r'\}\s*%?(?:\s*[x×*]\s*(\d+))?', template))
            repeat = int(occurrences[0].group(1) or 1) if len(occurrences) == 1 else None
            detail = {'effect_id': effect, 'hit_index': index + 1, 'repeat': repeat,
                'harmony_gain': raw.get('HeterochromeAdd'),
                'energy_gain': raw.get('ChargeAdd'), 'stagger': raw.get('UnbalValue')}
            for field, source in [('attack_values', 'AtkRateBaseArray'), ('defense_values', 'DefRateBaseArray'), ('hp_values', 'HPRateBaseArray')]:
                detail[field] = [percentage(value) for value in raw.get(source) or []]
            row['hit_details'].append(detail)
        row['hit_detail_status'] = 'available' if effects else 'not_applicable'
    phases = ability.get('phases') or []
    groups = [{'name': p['name'], 'description': p.get('desc'), 'param_rows': []} for p in phases]
    shared = {'name': '共用与其他参数', 'description': None, 'param_rows': []}
    for row in ability.get('param_rows') or []:
        group = select_group(row, phases)
        (groups[group] if group is not None else shared)['param_rows'].append(row)
    # Damage rows without an unambiguous prose phase retain their own display label.
    other_groups = {}
    common = []
    for row in shared['param_rows']:
        if row['hit_details']:
            name = re.sub(r'[一二三四五六七八九十]+段', '', row['label'])
            name = re.sub(r'(?:基础)?(?:伤害)?倍率.*$', '', name).strip(' ：:') or row['label']
            other_groups.setdefault(name, {'name': name, 'description': None, 'param_rows': []})['param_rows'].append(row)
        else:
            common.append(row)
    groups.extend(other_groups.values())
    if common:
        shared['param_rows'] = common
        groups.append(shared)
    ability['groups'] = groups


def enrich_catalog(data, tables):
    for character in data['characters']:
        for ability in character['abilities']:
            enrich_ability(ability, tables)
    return data
