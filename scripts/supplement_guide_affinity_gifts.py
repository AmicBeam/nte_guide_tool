#!/usr/bin/env python3
"""Import explicitly bound gifts while preserving missing affinity level state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def item_key(value):
    match = re.fullmatch(r'food_0*(\d+)', value, re.I)
    return 'food_' + str(int(match[1])) if match else value.casefold()


def supplement(data, role_id, gifts, strings):
    result = deepcopy(data)
    known = {item_key(gift['id']): gift for role in data['characters'] for gift in role['affinity']['gifts']}
    names = {key: gift['name'] for key, gift in known.items()}
    for key, value in strings.items():
        if key.casefold().endswith('_name') and value:
            names[item_key(key[:-5])] = value
    selected = []
    for ident, raw in gifts.items():
        matches = [row['LikeabilityValue'] for row in raw['SpeicalGiftDatas'] if row['RoleId'] == role_id]
        if not matches or raw['ExclusiveLikeabilityID'] not in ('None', role_id):
            continue
        if len(matches) != 1 or isinstance(matches[0], bool) or not isinstance(matches[0], int) or matches[0] < 0:
            raise ValueError('Invalid gift affinity binding')
        key = item_key(ident)
        if key not in names:
            raise ValueError('Missing gift name: ' + ident)
        selected.append({'id': ident, 'name': names[key], 'affinity_gain': matches[0],
            'default_gain': raw['DefaultLikeabilityValue'], 'unlimited': raw['bUnlimit'],
            'exclusive': raw['ExclusiveLikeabilityID'] == role_id,
            'offers': deepcopy(known.get(key, {}).get('offers', []))})
    if not selected:
        raise ValueError('No explicitly bound character gifts')
    role = next(role for role in result['characters'] if role['id'] == role_id)
    role['affinity']['gifts'] = sorted(selected, key=lambda gift: (gift['affinity_gain'], gift['id']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path, required=True)
    parser.add_argument('--character-id', required=True)
    args = parser.parse_args()
    path = ROOT / 'app/modules/guide/data/affinity.json'
    raw = json.loads((args.ntedata_dir / 'DT_LikeabilityGiftData.json').read_text())
    gifts = next(obj['Rows'] for obj in raw if 'Rows' in obj)
    strings = {}
    for filename in ('ST_Item.json', 'ST_food.json', 'ST_furniture.json'):
        for table in json.loads((args.ntedata_dir / filename).read_text()):
            strings.update(table.get('StringTable', {}).get('KeysToEntries', {}))
    result = supplement(json.loads(path.read_text()), args.character_id, gifts, strings)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


if __name__ == '__main__': main()
