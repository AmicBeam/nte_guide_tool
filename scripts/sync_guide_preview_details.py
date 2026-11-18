#!/usr/bin/env python3
"""Import verified skill and awakening sections for an incomplete character entry."""
import argparse
import json
from decimal import Decimal
from pathlib import Path
from guide_skill_details import table_rows, enrich_ability, display_text, percentage
from refresh_guide_data import clean

ROOT = Path(__file__).resolve().parents[1]


def build(directory, character_id, prefix, *, table_overrides=None):
    def table(name):
        if table_overrides and name in table_overrides:return table_overrides[name]
        return table_rows(json.loads((directory / (name + '.json')).read_text()))
    tips, damage = table('DT_GameplayAbilityTipsData'), table('DT_SkillDamageData')
    curves = table('DT_' + prefix + 'EffectFigure')
    def icon(raw, folder):
        name = raw.get('AssetPathName', '').split('.', 1)[0].rsplit('/', 1)[-1]
        relative = f'guide/{folder}/{name}.webp'
        return '/static/' + relative if name and (ROOT / 'app/modules/guide/static' / folder / (name + '.webp')).is_file() else None
    abilities = []
    for identifier, raw in tips.items():
        if not identifier.startswith('GA_' + prefix + '_') or not raw['AbilityDescription']:
            continue
        name = display_text(raw['Name'])
        ability = {'id': identifier, 'name': name,
                   'icon': icon(raw['icon'], 'skill-icons'),
                   'type_name': {'Melee': '普通攻击', 'Skill': '变轨技能', 'QTE': '援护技',
                                 'UltraSkill': '极轨终结', 'Passive1': '天赋', 'Passive2': '天赋',
                                 'Passive3': '特性'}.get(identifier.rsplit('_', 1)[-1], '技能'),
                   'phases': [], 'param_rows': []}
        for description in raw['AbilityDescription']:
            if not description['AbilityDesType'].endswith('ADT_DES'):
                continue
            ability['phases'].append({'name': clean(display_text(description['DescriptionTitle'])) or name,
                                      'desc': clean(display_text(description['Description']))})
        for parameter in raw['GALevelUpArray']:
            source = parameter['LevelUpData']
            values = []
            if source.get('DamageGEList'):
                arrays = [damage[key]['AtkRateBaseArray'] for key in source['DamageGEList']]
                if len({len(a) for a in arrays}) != 1:
                    raise ValueError('Inconsistent damage levels')
                values = [display_text(parameter['DescValue']).format(*[
                    percentage(a[level]).removesuffix('%') for a in arrays]) for level in range(len(arrays[0]))]
            elif source.get('EffectFigureDataList'):
                numbers = []
                for binding in source['EffectFigureDataList']:
                    keys = curves[binding.get('NameId', binding.get('NameID'))]['Keys']
                    if len(keys) != 1 or keys[0]['Time'] != 1:
                        raise ValueError('Nonconstant curve needs explicit level evaluation')
                    value = Decimal(str(keys[0]['Value'])) * (100 if binding['bIsPercent'] else 1)
                    numbers.append(format(value.normalize(), 'f'))
                values = [display_text(parameter['DescValue']).format(*numbers)]
            if not values:
                continue  # Unresolved cooldowns are omitted, never guessed.
            ability['param_rows'].append({'key': parameter['DescValue']['Key'],
                                         'label': display_text(parameter['Desc']), 'values': values})
        max_level = max((len(row['values']) for row in ability['param_rows']), default=1)
        for row in ability['param_rows']:
            if len(row['values']) == 1:
                row['values'] *= max_level
        tables = {'DT_GameplayAbilityTipsData': tips, 'DT_SkillDamageData': damage}
        if any(p.get('LevelUpData', {}).get('SkillCostName') not in (None, '', 'None')
               for p in raw['GALevelUpArray']):
            tables['DT_GameplayAbilityCostData'] = table('DT_GameplayAbilityCostData')
        enrich_ability(ability, tables)
        abilities.append(ability)
    awaken, resonance = [], []
    for raw in table(character_id + 'AwakenEffect')[character_id]['AwakenEffectStructList']:
        item = {'name': clean(display_text(raw['Title'])), 'desc': clean(display_text(raw['Desc'])),
                'icon': icon(raw['AwakenIcon'], 'passive-icons')}
        if raw.get('GEIdArray'):
            item['parameters'] = []
            for index, effect in enumerate(raw['GEIdArray']):
                hit = damage[effect]
                ability = next(a for a in abilities if a['id'] == hit['GAName'])
                item['parameters'].append({'id_value': index, 'ability_id': ability['id'],
                    'ability_name': ability['name'], 'formula': '追加伤害倍率',
                    'values': [percentage(v).removesuffix('%') for v in hit['AtkRateBaseArray']]})
        if raw['AwakenType'].endswith('Awaken_Resonance'):
            item['resonance_level'] = int(raw['EffectID'].split('_')[-1])
            resonance.append(item)
        else:
            awaken.append(item)
    if not abilities or len(awaken) != 6 or len(resonance) != 2:
        raise ValueError('Incomplete skill or awakening sections')
    return {'abilities': abilities, 'awaken': awaken, 'resonance': resonance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path, required=True)
    parser.add_argument('--character-id', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    path = ROOT / 'app/modules/guide/data/character_preview_details.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    records[args.character_id] = build(args.ntedata_dir, args.character_id, args.prefix)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


if __name__ == '__main__':
    main()
