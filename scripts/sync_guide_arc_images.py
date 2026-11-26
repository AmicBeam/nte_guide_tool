#!/usr/bin/env python3
"""Explicitly sync local WebP arc icons for the guide."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://static.nanoka.cc'


def download(url):
    return subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
        '--max-time', '45', '--retry', '2', url], capture_output=True, check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, help='Previously downloaded weapon index')
    args = parser.parse_args()
    if args.index:
        index = json.loads(args.index.read_text(encoding='utf-8'))
    else:
        version = json.loads(download(BASE + '/manifest.json'))['nte']['latest']
        index = json.loads(download(f'{BASE}/nte/{version}/weapon.json'))
    output = ROOT / 'app/static/images/arcs'
    output.mkdir(parents=True, exist_ok=True)
    def sync(record):
        identifier, icon = str(record['id']), record['icon']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or not icon.startswith('/Game/'):
            raise ValueError('Invalid arc icon binding')
        data = download(BASE + '/assets/nte/' + icon.lstrip('/') + '.webp')
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError(f'Invalid WebP: {identifier}')
        target = output / f'{identifier}.webp'
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(sync, index.values()))
    print(f'Synced {len(index)} arc WebP images')


if __name__ == '__main__':
    main()
