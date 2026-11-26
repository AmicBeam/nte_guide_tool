#!/usr/bin/env python3
"""Refresh Elyms's normal character record from local skill and growth tables."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from refresh_guide_data import clean
from guide_skill_details import table_rows, display_text
from sync_guide_preview_details import build as build_skills
from sync_guide_rich_text import character_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/guide/data'
CID = '1090'
BASE = {'HPMaxBase': 1270, 'AtkBase': 83, 'DefBase': 75}
STAT_NAMES = {'HPMaxBase': '生命值', 'AtkBase': '攻击力', 'DefBase': '防御力',
              'CritBase': '暴击率', 'CritDamageBase': '暴击伤害'}


def assemble(directory, catalog, equipment):
    def table(name):
        return table_rows(json.loads((directory / (name + '.json')).read_text()))
    growth = table('DT_PlayerModifyPackData')
    breakthrough = table('DT_CharacterBreakthroughDataTable')
    config = table('DT_CharacterAbilityConfig')[CID]
    tips = table('DT_GameplayAbilityTipsData')
    details = build_skills(directory, CID, 'Elyms')
    record = dict(id=CID, name='曳尔沐', aliases=['Elyms'], element='Cosmos', element_id='COSMOS',
                  element_name='光', rarity=5, hp=BASE['HPMaxBase'], atk=BASE['AtkBase'],
                  **{'def': BASE['DefBase']}, **details)
    record['stats'] = []
    for prop, base in BASE.items():
        values = []
        for level in range(1, 81):
            stage = sum(cap < level for cap in (20, 30, 40, 50, 60, 70))
            amount = base
            for key in [f'elyms_lv_{level}', *([f'elyms_stage_{stage}'] if stage else [])]:
                row = growth[key]
                if row['ConditionArray']:
                    raise ValueError('Conditional character growth')
                for modifier in row['ModifyData']:
                    if modifier['PropName'] == prop:
                        if not modifier['ModifierOp'].endswith('MODIFY_MODOP_ADDITIVE'):
                            raise ValueError('Nonadditive growth')
                        amount += modifier['PropValue']
            values.append(int(amount))
        record['stats'].append({'id_stats': prop, 'name': STAT_NAMES[prop], 'values': values})
    for prop, value in [('CritBase', 5), ('CritDamageBase', 50)]:
        record['stats'].append({'id_stats': prop, 'name': STAT_NAMES[prop], 'values': [value] * 80})
    names = {}
    def collect(value):
        if isinstance(value, dict):
            if 'id' in value and 'name' in value:
                names[str(value['id']).casefold()] = value['name']
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(catalog)
    collect(json.loads((DATA / "character_previews.json").read_text()))
    names.update(gold='甲硬币', fons='方斯')
    def items(pairs):
        result = []
        for key, amount in pairs:
            if key.casefold() not in names:
                raise ValueError('Unknown cultivation material: ' + key)
            result.append({'id': key, 'name': names[key.casefold()], 'amount': amount})
        return result
    stages = []
    caps = []
    for stage in range(7):
        row = breakthrough[f'elyms_Breakthrough_{stage}']
        caps.append(row['MaxCharacterLevel'])
        if not stage: continue
        pairs = [(part.split(':')[0], int(part.split(':')[1])) for part in
                 (row['NeedItems'] + ',' + row['NeedGolds']).split(',') if part]
        stages.append({'level': str(row['MaxCharacterLevel']), 'stage': stage, 'items': items(pairs)})
    record['materials'] = {'breakthrough': stages, 'skills': []}
    by_id = {a['id']: a for a in record['abilities']}
    for section in ('CharacterAbilityList', 'PassiveAbilityList', 'PeculiarityList', 'CityList'):
        for binding in config[section]:
            identifier, raw = binding['Key'], binding['Value']
            if identifier not in by_id:
                if section != 'CityList':
                    raise ValueError('Missing skill text: ' + identifier)
                # The exported city strings are placeholders; never display them as real effects.
                a = {'id': identifier, 'name': '都市技能' + ('一' if raw['AbilityIndex'] == 1 else '二'),
                     'type_name': '都市技能', 'phases': [], 'groups': [], 'param_rows': [], 'icon': None}
                a['groups'] = [{'name': a['name'], 'description': '都市技能效果暂未提供。', 'param_rows': []}]
                record['abilities'].append(a); by_id[identifier] = a
            a = by_id[identifier]
            a['type'] = ('city' if section == 'CityList' else 'passive' if section == 'PassiveAbilityList'
                         else 'peculiarity' if section == 'PeculiarityList' else
                         {'GA_Elyms_Melee': 'melee', 'GA_Elyms_Skill': 'skill',
                          'GA_Elyms_UltraSkill': 'ultraskill', 'GA_Elyms_QTE': 'qte'}[identifier])
            a['type_name'] = {'melee':'Melee','skill':'Skill','ultraskill':'UltraSkill','qte':'QTE', 'passive':'Passive','city':'City','peculiarity':'Peculiarity'}[a['type']]
            a['ability_type'] = raw['AbilityType'].split('::')[-1]
            a['ability_index'] = raw['AbilityIndex']
            costs = raw['LevelsCostItems']
            if costs:
                record['materials']['skills'].append({'id': identifier, 'name': a['name'], 'levels': [
                    {'level': str(row['Level'] + 1), 'require_breakthrough_level': row['RequireTupoLevel'],
                     'items': items([(i['ID'], i['Number']) for i in row['CostItems']])} for row in costs]})
    cooldowns = table('CoolDwonCurveTable')
    for a in record['abilities']:
        if a['type'] in ('skill', 'ultraskill'):
            values = {key['Value'] for key in cooldowns[a['id']]['Keys']}
            if len(values) != 1: raise ValueError('Variable skill cooldown')
            a['base_cooldown'] = values.pop()
    record['nouns'] = {key.casefold(): {'id': key, 'title': clean(display_text(row['Title'])),
                         'desc': clean(display_text(row['Explanation']))}
                       for key, row in table('DT_NounsExplanation').items() if key.startswith('Elyms_')}
    record['equip_slots'] = deepcopy(equipment['equip_slots'])
    record['equip_slots']['desc'] = record['equip_slots'].pop('description_template')
    record['equip_slots']['type_name'] = '「卡带」属性'
    raw_rich = deepcopy(record)
    for a in raw_rich['abilities']:
        if a['id'] in tips:
            a['phases'] = [{'desc': row['Description'].get('LocalizedString', '')}
                           for row in tips[a['id']]['AbilityDescription'] if row['AbilityDesType'].endswith('ADT_DES')]
            a.pop('groups', None)
    for section in ('awaken', 'resonance'):
        for target, original in zip(raw_rich[section], table('1090AwakenEffect')[CID]['AwakenEffectStructList'][:6] if section == 'awaken'
                                   else table('1090AwakenEffect')[CID]['AwakenEffectStructList'][6:]):
            target['desc'] = original['Desc']['LocalizedString']
    for key, noun in raw_rich['nouns'].items():
        noun['desc'] = table('DT_NounsExplanation')[noun['id']]['Explanation']['LocalizedString']
    return record, caps, character_data(raw_rich)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path, required=True)
    parser.add_argument('--equipment-json', type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads((DATA / 'catalog.json').read_text())
    record, caps, rich = assemble(args.ntedata_dir, catalog, json.loads(args.equipment_json.read_text()))
    records = [c for c in catalog['characters'] if c['id'] != CID] + [record]
    catalog['characters'] = records; catalog['cultivation']['level_caps'][CID] = caps
    supplemental = json.loads((DATA / 'ntedata_characters.json').read_text(encoding='utf-8'))
    supplemental['characters'] = [c for c in supplemental['characters'] if c['id'] != CID] + [record]
    supplemental['level_caps'][CID] = caps
    outputs = {'catalog.json': catalog, 'ntedata_characters.json': supplemental}
    def load(name): return json.loads((DATA / name).read_text())
    for name in ('skill_icons.json', 'passive_icons.json', 'character_rich_text.json'):
        payload = load(name)
        if name == 'skill_icons.json': payload[CID] = {a['id']: a.get('icon') for a in record['abilities']}
        elif name == 'passive_icons.json': payload[CID] = {s: [{'name': a['name'], 'icon': a.get('icon')} for a in record[s]] for s in ('awaken','resonance')}
        else: payload['characters'][CID] = rich
        outputs[name] = payload
    previews = load('character_previews.json'); outputs['character_previews.json'] = [c for c in previews if c['id'] != CID]
    payload = load('character_preview_details.json');payload.pop(CID,None);outputs['character_preview_details.json'] = payload
    for name, data in outputs.items():
        target = DATA / name; temp = target.with_suffix('.tmp')
        pretty = name in ('skill_icons.json', 'passive_icons.json', 'character_previews.json',
                          'character_preview_details.json')
        formatting = {'indent': 2} if pretty else {'separators': (',', ':')}
        temp.write_text(json.dumps(data, ensure_ascii=False, **formatting)+'\n');temp.replace(target)


if __name__ == '__main__':main()
