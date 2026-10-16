#!/usr/bin/env python3
"""Explicit offline sync of character arc compatibility and role badges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from refresh_guide_data import clean

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://static.nanoka.cc'
ENUMS = {'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5'}


def table(local, remote):
    payload = json.loads(local.read_text(encoding='utf-8')) if local else json.loads(subprocess.run(
        ['gh', 'api', 'repos/mc-ctrl/NTEData/contents/' + remote,
         '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True).stdout)
    return next(item['Rows'] for item in payload if 'Rows' in item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--character-table', type=Path)
    parser.add_argument('--type-table', type=Path)
    args = parser.parse_args()
    characters = table(args.character_table, 'DataTable/Character/DT_Character.json')
    types = table(args.type_table, 'DataTable/Fork/DT_ForkTypeData.json')
    ids = [c['id'] for c in json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())['characters']]
    icons = {}

    def icon_binding(path, category):
        if not path.startswith('/Game/'):
            raise ValueError('Invalid icon path')
        filename = path.rsplit('/', 1)[-1] + '.webp'
        relative = f'images/{category}/{filename}'
        icons[relative] = BASE + '/assets/nte/' + path.lstrip('/') + '.webp'
        return '/static/' + relative

    def character(identifier):
        raw = characters[identifier]
        group = characters[identifier]['ElementData']['CharacterGroupType'].rsplit('_', 1)[-1]
        type_id = ENUMS[group]
        definition = types[type_id]
        icon = definition['ApplyTypeIcon']['AssetPathName'].split('.', 1)[0]
        arc = {'id': type_id, 'name': definition['TypeName']['LocalizedString'],
               'icon': icon_binding(icon, 'arc-types')}
        tags = []
        for tag in raw['PlayerViewTagArray']:
            item = {'name': clean(tag['TagText']['LocalizedString']),
                    'description': clean(tag['TagDesc'].get('LocalizedString') or tag['TagDesc'].get('SourceString'))}
            tag_icon = tag['TagIcon']['AssetPathName'].split('.', 1)[0]
            if tag_icon:
                item['icon'] = icon_binding(tag_icon, 'character-tags')
            tags.append(item)
        return {'id': identifier, 'arc_type': arc, 'char_tags': tags}

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(character, ids))

    def download(item):
        relative, url = item
        data = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
            '--max-time', '45', '--retry', '2', url], capture_output=True, check=True).stdout
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError('Invalid WebP badge')
        target = ROOT / 'app/static' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)
    for rank in ('A', 'S'):
        icon_binding(f'/Game/UI/UI/Gacha/YH_UI_star{rank}', 'rarities')
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(download, icons.items()))
    target = ROOT / 'app/modules/guide/data/character_metadata.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'characters': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)
    print(f'Synced {len(records)} characters and {len(icons)} badge icons')


if __name__ == '__main__':
    main()
