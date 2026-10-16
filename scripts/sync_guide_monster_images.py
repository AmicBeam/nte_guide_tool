#!/usr/bin/env python3
"""Explicitly bind each guide monster ID to its configured local WebP icon."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://static.nanoka.cc'
MODULE = ROOT / 'app/modules/guide'


def download(url):
    return subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
        '--max-time', '45', '--retry', '2', url], check=True, capture_output=True).stdout


def monsters(record):
    for stage in record['stages']:
        for half in stage['halves']:
            for wave in half['waves']:
                yield from wave['monsters']


def collect_bindings(catalog, records):
    expected = {str(monster['id']): monster['name'] for record in catalog['challenges'] for monster in monsters(record)}
    result = {}
    for record in records:
        for monster in monsters(record):
            identifier = str(monster['id'])
            if identifier not in expected:
                continue
            icon = monster.get('icon')
            if monster['name'] != expected[identifier] or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
                raise ValueError(f'Monster identity mismatch: {identifier}')
            if not isinstance(icon, str) or not re.fullmatch(r'/Game/[A-Za-z0-9_/-]+', icon):
                raise ValueError(f'Missing or invalid configured icon: {identifier}')
            if identifier in result and result[identifier] != icon:
                raise ValueError(f'Conflicting icons for monster ID: {identifier}')
            result[identifier] = icon
    if result.keys() != expected.keys():
        raise ValueError(f'Missing monster IDs: {sorted(expected.keys() - result.keys())}')
    return result


def main():
    catalog = json.loads((MODULE / 'data/catalog.json').read_text())
    metadata_path = MODULE / 'data/monster_metadata.json'
    previous = json.loads(metadata_path.read_text())
    names = {str(monster['id']): monster['name'] for record in catalog['challenges'] for monster in monsters(record)}
    version = json.loads(download(BASE + '/manifest.json'))['nte']['latest']
    urls = [f'{BASE}/nte/{version}/zh/abyss/{record["id"]}.json' for record in catalog['challenges']]
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda url: json.loads(download(url)), urls))
    bindings = collect_bindings(catalog, records)
    output = MODULE / 'static/monsters'
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    with tempfile.TemporaryDirectory(prefix='.sync-monsters-', dir=output) as temporary:
        temporary = Path(temporary)
        def sync(entry):
            identifier, icon = entry
            data = download(BASE + '/assets/nte/' + icon.lstrip('/') + '.webp')
            if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
                raise ValueError(f'Invalid WebP for monster ID: {identifier}')
            (temporary / f'{identifier}.webp').write_bytes(data)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(sync, bindings.items()))
        for identifier in sorted(bindings):
            # Preserve the established trait during migration from name-keyed data.
            old = previous.get(identifier, previous.get(names[identifier], {}))
            metadata[identifier] = {'image': f'/static/guide/monsters/{identifier}.webp',
                                    'swallowable': old.get('swallowable')}
        # All bindings and images must validate before replacing any live binding.
        for identifier in bindings:
            (temporary / f'{identifier}.webp').replace(output / f'{identifier}.webp')
        target = temporary / 'metadata.json'
        target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        target.replace(metadata_path)
    print(f'Synced {len(bindings)} monster ID bindings')


if __name__ == '__main__':
    main()
