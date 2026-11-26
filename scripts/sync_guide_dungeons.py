#!/usr/bin/env python3
"""Explicitly import cultivation dungeons; hidden client counts remain unknown."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    'clones': 'CloneSystem/CloneSystemDataTable.json',
    'groups': 'Drop/Client/ClientDropGroupDataTable.json',
    'sequences': 'Drop/DropSequenceDataTable.json',
    'items': 'Inventory/DT_ItemConfig.json',
    'currencies': 'Inventory/DT_CapitalItemConfig.json',
    'equipment': 'Equipment/DT_Equipment.json',
    'sources': 'DT_OutputSourceRowData.json',
    'details': 'DT_OutputDetailSource.json',
}
CATEGORIES = {'Trial_Clone': '经验与甲硬币', 'Talent_Clone': '技能材料',
              'ArcDisk_Clone': '弧盘突破', 'AirCurtain_Clone': '空幕', 'Weekly_Clone': '异象巡礼'}


def text(value):
    return re.sub(r'<[^>]*>', '', (value or {}).get('LocalizedString') or
                  (value or {}).get('SourceString') or (value or {}).get('CultureInvariantString') or '').strip()


def build(tables):
    items = {k.casefold(): v for name in ('items', 'currencies', 'equipment')
             for k, v in tables[name].items()}
    sequences = {}
    for key, row in tables['sequences'].items():
        sequences.setdefault(key.rsplit('_', 1)[0].casefold(), []).append(row)
    groups = {}
    for key, row in tables['groups'].items():
        groups.setdefault(key.rsplit('_', 1)[0].casefold(), []).append(row)
    result = []
    for identifier, clone in tables['clones'].items():
        category = CATEGORIES.get(clone['CloneType'].split('::')[-1])
        name = text(clone.get('CloneName'))
        if not category or not name or identifier.endswith('_test'):
            continue
        stages = []
        for stage in clone['CloneSystemSubNodes']:
            drop_id = stage['DropID'].casefold()
            if drop_id in ('none', '0'):
                continue
            if drop_id not in groups:
                raise ValueError('Missing dungeon drop group: ' + drop_id)
            rewards = {}
            for group in groups[drop_id]:
                sequence_id = group['SequenceId'].casefold()
                if sequence_id not in sequences:
                    raise ValueError('Missing drop sequence: ' + sequence_id)
                entries = sequences[sequence_id]
                for entry in entries:
                    item_id = entry['ItemID'].casefold()
                    # Furniture progression is outside the main cultivation rewards.
                    if item_id.startswith('visions_furniture_'):
                        continue
                    item = items.get(item_id)
                    item_name = text(item.get('ItemName')) if item else ''
                    if not item_name:
                        raise ValueError('Missing reward name: ' + item_id)
                    amount = None
                    if (not group['HiddenCounts'] and not group['DropConditions'] and
                            len(entries) == 1 and group['SequenceWeight'] == 1 and
                            group['MinLevel'] == 0 and group['MaxLevel'] >= 100 and
                            entry['SequenceNumType'].endswith('_FIXED') and not entry['LimitLevel']):
                        amount = entry['Num'] * group['ModifyNum']
                        if not isinstance(amount, (int, float)) or amount < 0:
                            raise ValueError('Invalid fixed reward')
                    reward = {'id': item_id, 'name': item_name, 'amount': amount}
                    if item_id in rewards and rewards[item_id] != reward:
                        rewards[item_id]['amount'] = None
                    else:
                        rewards[item_id] = reward
            stages.append({'id': stage['DifficultyLevel'], 'name': '难度 ' + str(stage['DifficultyLevel']),
                           'required_level': stage['TeamLevel'], 'stamina': stage['StaminaCount'],
                           'rewards': list(rewards.values())})
        if not stages:
            continue
        if len({row['id'] for row in stages}) != len(stages):
            raise ValueError('Duplicate dungeon difficulty')
        result.append({'id': identifier, 'name': name, 'category': category,
                       'description': text(clone['CloneSystemSubNodes'][0]['CloneDesArray'][0].get('CloneDesBody'))
                           if clone['CloneSystemSubNodes'][0]['CloneDesArray'] else '',
                       'stages': stages})
    # World boss material bindings come from explicit acquisition references.
    world = {}
    source_rows = {k.casefold(): v for k, v in tables['sources'].items()}
    for item_id, item in items.items():
        if not item_id.startswith('worldboss_material_') or 'selectbox' in item_id:
            continue
        for source in source_rows.get(item.get('OutputSourceID', '').casefold(), {}).get('SourceArray', []):
            detail = tables['details'].get(source.strip(), {})
            if not source.casefold().startswith('worldboss_') or not source.casefold().endswith('_drop'):
                continue
            name = text(detail.get('ContentDesc'))
            if not name:
                raise ValueError('Missing world boss name')
            record = world.setdefault(source, {'id': source, 'name': name, 'category': '异象追猎',
                'description': '击败对应异象后领取角色突破材料。', 'stages': [], 'rewards': []})
            record['rewards'].append({'id': item_id, 'name': text(item['ItemName']), 'amount': None})
    result.extend(world.values())
    if not result or len({row['id'] for row in result}) != len(result):
        raise ValueError('Invalid dungeon catalog')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/dungeons.json')
    args = parser.parse_args()

    def load(pair):
        key, path = pair
        if args.ntedata_dir:
            candidates = (args.ntedata_dir / 'DataTable' / path, args.ntedata_dir / path,
                          args.ntedata_dir / Path(path).name)
            file = next((p for p in candidates if p.is_file()), None)
            if file is None:
                raise ValueError('Missing table: ' + path)
            data = json.loads(file.read_text())
        else:
            data = json.loads(subprocess.check_output(['gh', 'api', '-H',
                'Accept: application/vnd.github.raw+json', 'repos/mc-ctrl/NTEData/contents/DataTable/' + path]))
        return key, data[0]['Rows']

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = build(dict(executor.map(load, TABLES.items())))
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(f'Imported {len(records)} dungeons')


if __name__ == '__main__':
    main()
