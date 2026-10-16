#!/usr/bin/env python3
"""Sync local WebP icons and display qualities for guide materials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from sync_guide_arc_images import BASE, ROOT, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path)
    args = parser.parse_args()
    if args.index:
        index = json.loads(args.index.read_text())
    else:
        version = json.loads(download(BASE + '/manifest.json'))['nte']['latest']
        index = json.loads(download(f'{BASE}/nte/{version}/zh/item.json'))
    lookup = {str(row['id']).casefold(): row for row in index.values()}
    catalog = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())
    materials = {}

    def collect(value):
        if isinstance(value, dict):
            for item in value.get('items', []):
                materials[str(item['id']).casefold()] = item['name']
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for kind in ('characters', 'arcs'):
        for record in catalog[kind]:
            collect(record['materials'])
    for book in catalog['cultivation']['exp_books']:
        materials[book['id'].casefold()] = book['name']
    arc_config = json.loads((ROOT / 'app/modules/guide/data/arc_cultivation.json').read_text())
    for book in arc_config['exp_books']:
        materials[book['id'].casefold()] = book['name']
    materials.update(gold='甲硬币', fons='方斯')
    output = ROOT / 'app/static/images/materials'
    output.mkdir(parents=True, exist_ok=True)

    def sync(pair):
        identifier, name = pair
        row = lookup[{'gold': 'gold_1000', 'fons': 'fons_100'}.get(identifier, identifier)]
        if not re.fullmatch(r'[a-z0-9_-]+', identifier) or row['quality'] not in ('gray', 'green', 'blue', 'purple', 'orange', 'red'):
            raise ValueError('Invalid material binding')
        existing = ROOT / 'app/static/images/item' / (name + '.webp')
        if existing.is_file():
            icon = '/static/images/item/' + name + '.webp'
        else:
            if not row['icon'].startswith('/Game/'):
                raise ValueError('Invalid material icon')
            data = download(BASE + '/assets/nte/' + row['icon'].lstrip('/') + '.webp')
            if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
                raise ValueError('Invalid material WebP')
            target = output / (identifier + '.webp')
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)
            icon = '/static/images/materials/' + identifier + '.webp'
        return identifier, {'name': name, 'icon': icon, 'quality': row['quality']}
    with ThreadPoolExecutor(max_workers=6) as executor:
        records = dict(executor.map(sync, sorted(materials.items())))
    target = ROOT / 'app/modules/guide/data/material_metadata.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(f'Synced {len(records)} material bindings')


if __name__ == '__main__':
    main()
