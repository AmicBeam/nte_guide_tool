#!/usr/bin/env python3
"""Explicit offline import of foods grouped by their configured function."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from sync_guide_affinity import text
from sync_guide_arc_images import download

ROOT = Path(__file__).resolve().parents[1]
TABLES = {'foods': 'Food/DT_FoodItemData.json', 'types': 'Food/DT_ItemEffectTypeData.json'}
GROUPS = ('Attack', 'Defense', 'AddHPInstant', 'AddHPConstant', 'Revive', 'Explore', 'Special')


def build_foods(tables):
    result = []
    for identifier, row in tables['foods'].items():
        effect_type = row['ItemEffectType'].split('::')[-1]
        if effect_type not in tables['types'] or effect_type not in GROUPS:
            raise ValueError('Unknown food function')
        name, effect = text(row['ItemName']), text(row['UseContext'])
        if not name or not effect:
            raise ValueError('Missing food name or effect')
        result.append({'id': identifier, 'name': name, 'function': effect_type,
                       'function_name': text(tables['types'][effect_type]['EffectTypeName']),
                       'effect': effect, 'description': text(row['Description']),
                       'quality': row['ItemQuality'].split('_')[-1].lower(),
                       'cooldown': row['ItemCD'],
                       'image': f'/static/guide/foods/{identifier.lower()}.webp'})
    result.sort(key=lambda r: (GROUPS.index(r['function']), r['id']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path)
    args = parser.parse_args()
    def load(pair):
        key, path = pair
        raw = ((args.ntedata_dir / Path(path).name).read_bytes() if args.ntedata_dir else
               subprocess.check_output(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/DataTable/' + path,
                                        '-H', 'Accept: application/vnd.github.raw+json']))
        return key, next(x['Rows'] for x in json.loads(raw) if 'Rows' in x)
    tables = dict(map(load, TABLES.items()))
    foods = build_foods(tables)
    target = ROOT / 'app/modules/guide/static/foods'
    target.mkdir(parents=True, exist_ok=True)
    def icon(food):
        path = target / (food['id'].lower() + '.webp')
        if path.is_file():
            return
        asset = tables['foods'][food['id']]['ItemIcon']['AssetPathName'].split('.')[0]
        if not asset.startswith('/Game/'):
            raise ValueError('Invalid food icon')
        data = download('https://api.everness.info/data/assets/' + asset.removeprefix('/Game/UI/') + '.webp')
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError('Invalid food image')
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(icon, foods))
    path = ROOT / 'app/modules/guide/data/foods.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(foods, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    print(f'Synced {len(foods)} foods')


if __name__ == '__main__':
    main()
