#!/usr/bin/env python3
"""Explicit offline sync of character awakening and resonance icons."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
from sync_guide_arc_images import BASE, download

ROOT = Path(__file__).resolve().parents[1]


def main():
    version = json.loads(download(BASE + '/manifest.json'))['nte']['latest']
    catalog = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())
    icons = {}

    def character(record):
        identifier = str(record['id'])
        raw = json.loads(download(f'{BASE}/nte/{version}/zh/character/{identifier}.json'))
        bindings = {}
        for section in ('awaken', 'resonance'):
            items = []
            for item in record[section]:
                matches = [entry for entry in raw[section] if entry['name'] == item['name']
                           and entry.get('resonance_level') == item.get('resonance_level')]
                if len(matches) != 1:
                    raise ValueError(f'Ambiguous passive: {identifier}/{section}/{item["name"]}')
                path = matches[0]['icon']
                if not re.fullmatch(r'/Game/[A-Za-z0-9_/-]+', path):
                    raise ValueError('Invalid passive icon')
                filename = path.rsplit('/', 1)[-1] + '.webp'
                icons[filename] = BASE + '/assets/nte/' + path.lstrip('/') + '.webp'
                items.append({'name': item['name'], 'icon': '/static/guide/passive-icons/' + filename})
            bindings[section] = items
        return identifier, bindings

    with ThreadPoolExecutor(max_workers=6) as executor:
        bindings = dict(executor.map(character, catalog['characters']))
    output = ROOT / 'app/modules/guide/static/passive-icons'
    output.mkdir(parents=True, exist_ok=True)

    missing = set()

    def save(entry):
        filename, url = entry
        target = output / filename
        if target.exists():
            cached = target.read_bytes()
            if cached[:4] == b'RIFF' and cached[8:12] == b'WEBP':
                return
        try:
            data = download(url)
        except subprocess.CalledProcessError as error:
            if b'404' not in (error.stderr or b''):
                raise
            missing.add(filename)
            return
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError('Invalid passive WebP')
        temporary = (output / filename).with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(output / filename)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(save, icons.items()))
    for character in bindings.values():
        for entries in character.values():
            for item in entries:
                if item['icon'].rsplit('/', 1)[-1] in missing:
                    item['icon'] = None
    target = ROOT / 'app/modules/guide/data/passive_icons.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(f'Synced {len(bindings)} characters, {len(icons) - len(missing)} icons; {len(missing)} unavailable')


if __name__ == '__main__':
    main()
