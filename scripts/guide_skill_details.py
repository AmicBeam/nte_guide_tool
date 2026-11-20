"""Offline linkage of skill display parameters to individual damage effects."""
from __future__ import annotations

from decimal import Decimal
import json
import math
from pathlib import Path
import re
import subprocess

TABLES = ('DT_GameplayAbilityTipsData', 'DT_SkillDamageData', 'DT_GameplayAbilityCostData')
LACRIMOSA_FORMS = Path(__file__).resolve().parents[1] / 'app/modules/guide/data/lacrimosa_forms.json'
LACRIMOSA_FORM_ROW = 'GA_Lacrimosa_Skill_stolen_damage'


def stolen_damage_row(damages):
    """Bind each learned form to its own player damage effects, never monster rates."""
    hits = []
    for form in json.loads(LACRIMOSA_FORMS.read_text(encoding='utf-8')):
        for index, effect in enumerate(form['effect_ids'], 1):
            raw = damages.get(effect)
            if raw is None or raw.get('GAName') != 'GA_Lacrimosa_Skill':
                raise ValueError(f'Missing or invalid stolen damage effect: {effect}')
            label = form['name']
            if len(form['effect_ids']) > 1:
                label += f' · 伤害项{index}'
            hits.append({
                'effect_id': effect, 'hit_index': len(hits) + 1,
                'hit_label': label, 'hit_description': form['description'],
                'form_ability_id': form['ability_id'], 'repeat': None,
                'harmony_gain': raw.get('HeterochromeAdd'),
                'energy_gain': raw.get('ChargeAdd'), 'stagger': raw.get('UnbalValue'),
                **{field: [percentage(v) for v in raw.get(source) or []]
                   for field, source in [('attack_values', 'AtkRateBaseArray'),
                                         ('defense_values', 'DefRateBaseArray'),
                                         ('hp_values', 'HPRateBaseArray')]},
            })
    return {'key': LACRIMOSA_FORM_ROW, 'label': '「恶魔的礼物」各变身伤害倍率',
            'values': [], 'details_only': True, 'hit_details': hits,
            'hit_detail_status': 'available'}


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
    key = row.get('key', '').lower()
    if re.fullmatch(r'ga_nanally_extraatk[1-3]_num', key):
        return next((i for i, phase in enumerate(phases)
                     if phase['name'] == '普通攻击：重拳出击！'), None)
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
    # Rebuild supplemental rows on every refresh, including repeated enrichment.
    ability['param_rows'] = [row for row in ability.get('param_rows') or []
                             if row.get('key') != LACRIMOSA_FORM_ROW]
    tips = tables['DT_GameplayAbilityTipsData'].get(ability['id'])
    if tips is None and ability.get('param_rows'):
        raise ValueError(f'Missing skill tips: {ability["id"]}')
    tips = tips or {}
    costs = tables.get('DT_GameplayAbilityCostData')
    if costs is not None:
        for parameter in tips.get('GALevelUpArray') or []:
            identifier = (parameter.get('LevelUpData') or {}).get('SkillCostName')
            if not identifier or identifier == 'None':
                continue
            value = costs[identifier]['GACost']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid skill energy cost: {identifier}')
            key = parameter['DescValue']['Key']
            rows = ability.setdefault('param_rows', [])
            levels = max((len(row.get('values') or []) for row in rows), default=1)
            row = next((row for row in rows if row['key'] == key), None)
            if row is None:
                row = {'key': key}
                rows.append(row)
            row.update(label=display_text(parameter['Desc']), values=[
                display_text(parameter['DescValue']).format(f'{value:g}')
            ] * max(1, levels))
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
    if ability['id'] == 'GA_Lacrimosa_Skill':
        ability['param_rows'].append(stolen_damage_row(damages))
    phases = ability.get('phases') or []
    if not phases and not ability.get('param_rows'):
        return
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
