#!/usr/bin/env python3
"""Build the local Endless Trial reference from an explicitly downloaded NTEData tree."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELEMENTS = ('normal', 'cosmos', 'nature', 'incantation', 'chaos', 'psyche', 'lakshana', 'psychically')


def rows(payload):
    return next(entry['Rows'] for entry in payload if entry.get('Type') == 'DataTable')


def text(value):
    return value.get('LocalizedString') or value.get('SourceString') or value.get('CultureInvariantString')


def floor_rules(payload, descriptions):
    """Keep per-floor conditions and the paired benefit/cost options together."""
    entries = {key.casefold(): value for key, value in descriptions[0]['StringTable']['KeysToEntries'].items()}
    result = {}
    for rule in rows(payload).values():
        options = []
        for option in rule['BuffSelected']:
            sides = {'positive': [], 'negative': []}
            for side in option['Buffs']:
                for effect in side['Value']['FloorBuff']:
                    identifier = effect['AssetPathName'].split('.')[0].rsplit('/', 1)[-1]
                    description = entries[(identifier + '_des').casefold()]
                    sides[side['Key'].lower()].append({'id': identifier, 'description': description,
                                                     'target': 'player' if side['Value']['IsPlayerBuff'] else 'monster'})
            options.append(sides)
        result[rule['FloorIndex']] = {
            'name': text(rule['LevelName']), 'spawn_id': rule['SpawnMonsterId'],
            'maximum_time': rule['TimeLimitSeconds'],
            'objectives': [{'text': text(task['TaskDesc']), 'score': task['TaskChallengeScore'],
                            'target': task['TargetValue'], 'type': task['Type'].split('::')[-1]} for task in rule['ChallengeTask']],
            'recommended_elements': [item.removeprefix('CHARACTER_ELEMENT_TYPE_').lower() for item in rule['RecommendAtkTypeArray']],
            'enhanced_resistances': [item.removeprefix('CHARACTER_ELEMENT_TYPE_').lower() for item in rule['EnhancedResistanceTypeArray']],
            'node_first': rule['IsNodeFirstFloor'], 'node_last': rule['IsNodeLastFloor'],
            'reserve_character_count': rule['ReserveCharacterCount'], 'floor_slot_count': rule['FloorSlotCount'],
            'buff_selection': {'enabled': rule['bEnableBuffSelection'], 'default_count': rule['DefaultPositiveBuffCount'],
                               'count_bonuses': [{'floor': int(bonus['Key'].removeprefix('Floor_')), 'count': bonus['Value']}
                                                 for bonus in rule['PositiveBuffCountBonusByFloor']], 'options': options}}
    if set(result) != set(range(1, 201)):
        raise ValueError('Expected complete floor rules 1–200')
    return result


def monster_identities(payload, bindings):
    config = rows(payload)
    result = {}
    for template, binding in bindings.items():
        record = config[template.rsplit('/', 1)[-1]]
        result[template] = {**binding, 'name': text(record['MonsterName']),
            'weaknesses': [item.removeprefix('CHARACTER_ELEMENT_TYPE_').lower() for item in record['WeaknessTypeArray']],
            'enhanced_resistances': [item.removeprefix('CHARACTER_ELEMENT_TYPE_').lower() for item in record['ResistanceTypeArray']]}
    return result


def build(spawns, packs, identities, rules=None):
    floors = rows(spawns)
    attributes = rows(packs)
    expected = {f'EndlessTrial_Floor_{floor}' for floor in range(1, 201)}
    if set(floors) != expected:
        raise ValueError('Expected exactly floors 1–200')
    stages = []
    for floor in range(1, 201):
        rule = rules[floor] if rules is not None else None
        spawn_id = rule['spawn_id'] if rule else f'EndlessTrial_Floor_{floor}'
        waves = []
        for index, wave in enumerate(floors[spawn_id]['SpawnWaveEntriesOverride'], 1):
            monsters = []
            for entry in wave['SpawnEntries']:
                template = entry['AITemplate']['AssetPathName'].split('.')[0]
                identity = identities[template]
                identifier = entry['AISettingsOverrideID']
                stats = attributes[identifier]
                # These packs provide final values; reject growth-based packs rather than guessing a formula.
                if any(stats[key] != 0 for key in ('HPMaxUp', 'HPMaxAdd', 'AtkUp', 'AtkAdd', 'DefUp', 'DefAdd')):
                    raise ValueError(f'Unsupported level growth in {identifier}')
                if not entry['bOverrideLevelType'] or entry['SpawnLevelTypeOverride'] != 'EAISpawnLevelType::PermanentLevel':
                    raise ValueError(f'Unresolved level on floor {floor}')
                resistances = {element: stats['DamageResist' + element.title() + 'Base'] for element in ELEMENTS}
                elemental = {key: resistances[key] for key in ELEMENTS[1:-1]}
                minimum = min(elemental.values())
                weaknesses = [key for key, value in elemental.items() if value == minimum] if minimum < max(elemental.values()) else []
                monsters.append({**identity, 'id': identifier, 'class_id': template.rsplit('/', 1)[-1],
                    'level': entry['LevelOverride'], 'count': entry['Count'], 'weaknesses': identity.get('weaknesses', weaknesses),
                    'stats': {'hp': stats['HPMaxBase'], 'attack': stats['AtkBase'], 'defense': stats['DefBase'],
                              'unbalance': stats['UnbalMax'], 'resistances': resistances}})
            if not monsters:
                raise ValueError(f'Empty floor {floor}')
            waves.append({'wave': index, 'spawn_type': '开场' if index == 1 else 'KillAll', 'spawn_time': wave['WaveDelayTime'], 'monsters': monsters})
        stage = {'id': floor, 'name': f"{rule['name']} · 第 {floor} 层" if rule else f'第 {floor} 层',
                 'maximum_time': rule['maximum_time'] if rule else None,
                 'objectives': rule['objectives'] if rule else [],
                 'halves': [{'id': 'single', 'recommended_elements': rule['recommended_elements'] if rule else [], 'waves': waves}]}
        if rule:
            stage['rules'] = {key: value for key, value in rule.items() if key not in ('name', 'spawn_id', 'maximum_time', 'objectives', 'recommended_elements')}
        stages.append(stage)
    return {'schema_version': 1, 'groups': [
        {'id': f'EndlessTrial_{start}_{start + 49}', 'name': f'全站深境 · {start}–{start + 49} 层',
         'description': '逐层敌人、强化抗性、挑战目标及关底增益与代价。',
         'floor_begin': start, 'floor_end': start + 49, 'stage_count': 50,
         'stages': stages[start - 1:start + 49]} for start in range(1, 201, 50)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', required=True, type=Path)
    parser.add_argument('--identities', type=Path, default=ROOT / 'app/modules/guide/data/endless_identities.json',
                        help='Verified template-path to name/icon/trait bindings')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/endless.json')
    args = parser.parse_args()
    rule_data = floor_rules(json.loads((args.ntedata_dir / 'DataAssets/DataAssetSet/EndlessTrial/DT_EndlessTrialFloor_1.json').read_text()),
                            json.loads((args.ntedata_dir / 'Text/PlayerSkillDes/EndlessTrail_Des.json').read_text()))
    data = build(json.loads((args.ntedata_dir / 'DataTable/CloneSystem/DT_EndlessTrialMonsterInfo.json').read_text()),
                 json.loads((args.ntedata_dir / 'DataTable/PackData/DT_MonsterPackData.json').read_text()),
                 monster_identities(json.loads((args.ntedata_dir / 'DataAssets/DataAssetSet/EndlessTrial/DT_EndlessTrialMonsterConfig.json').read_text()),
                                    json.loads(args.identities.read_text())), rule_data)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(args.output)
    print('Updated 200 floors in four groups')


if __name__ == '__main__':
    main()
