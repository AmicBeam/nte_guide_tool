#!/usr/bin/env python3
"""Explicit offline refresh of the public combat reference dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://static.nanoka.cc'
# Presentation and implementation assets do not belong in a text reference.
OMIT = {'bg_icon', 'color', 'icon', 'icon_gacha', 'element_icon', 'type_icon', 'card_icon', 'original_icon',
        'background', 'manual_background', 'asset', 'buffs', 'gameplay_tag',
        'show_detail_info', 'materials', 'material', 'route_ui_id', 'display_drop_id',
        'award_drop_id', 'pool_id', 'attribute_id'}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in OMIT}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        value = value.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n')
        value = re.sub(r'<br\s*/?>', '\n', value, flags=re.I)
        return html.unescape(re.sub(r'<[^>]*>', '', value)).strip()
    return value


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'NTE-Guide/1.0'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def fetch_curl(url):
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error',
        '--location', '--max-time', '45', '--retry', '2', url],
        check=True, capture_output=True)
    return json.loads(result.stdout)


def build(version, fetcher=fetch):
    if version == 'latest':
        version = fetcher(f'{BASE}/manifest.json')['nte']['latest']
    root = f'{BASE}/nte/{version}'
    characters = fetcher(f'{root}/character.json')
    weapons = fetcher(f'{root}/weapon.json')
    seasons = fetcher(f'{root}/zh/abyss/index.json')['seasons']
    jobs = [('characters', x, f'{root}/zh/character/{x}.json') for x in characters]
    jobs += [('arcs', x, f'{root}/zh/weapon/{x}.json') for x in weapons]
    jobs += [('challenges', x['id'], f'{root}/zh/abyss/{x["id"]}.json') for x in seasons]

    def load(job):
        kind, identifier, url = job
        raw = fetcher(url)
        if str(raw.get('id')) != str(identifier) or not raw.get('name'):
            raise ValueError(f'Invalid {kind} record: {identifier}')
        if kind == 'characters':
            keys = ('id', 'name', 'element', 'element_id', 'element_name', 'rarity',
                    'char_tags', 'ability_name', 'hp', 'atk', 'def', 'stats',
                    'abilities', 'awaken', 'resonance', 'nouns')
            raw = {k: raw[k] for k in keys if k in raw}
            if not raw.get('abilities'):
                raise ValueError(f'Missing character abilities: {identifier}')
        elif kind == 'arcs':
            keys = ('id', 'name', 'rarity', 'type_id', 'type_name', 'description',
                    'stats', 'effect', 'exclusive_character_ids', 'exclusive_desc')
            raw = {k: raw[k] for k in keys if k in raw}
            if not raw.get('effect'):
                raise ValueError(f'Missing arc effect: {identifier}')
        else:
            if len(raw.get('stages', [])) != raw.get('stage_count'):
                raise ValueError(f'Incomplete challenge stages: {identifier}')
        record = clean(raw)
        record['aliases'] = list(dict.fromkeys(str(characters[identifier][k])
            for k in ('en', 'zh', 'ja', 'ko') if characters[identifier].get(k))) if kind == 'characters' else []
        return kind, record

    result = {'schema_version': 1, 'characters': [], 'arcs': [], 'challenges': []}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for kind, record in executor.map(load, jobs):
            result[kind].append(record)
    for kind in ('characters', 'arcs', 'challenges'):
        if not result[kind]:
            raise ValueError(f'Empty collection: {kind}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='latest')
    parser.add_argument('--transport', choices=('urllib', 'curl'), default='urllib')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/catalog.json')
    args = parser.parse_args()
    data = build(args.version, fetch_curl if args.transport == 'curl' else fetch)
    # Do not replace the working dataset until every detail has validated.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(', '.join(f'{kind}: {len(data[kind])}' for kind in ('characters', 'arcs', 'challenges')))


if __name__ == '__main__':
    main()
