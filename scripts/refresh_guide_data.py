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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guide_skill_details import enrich_catalog, load_tables, table_rows
from guide_skill_cooldowns import CooldownData, enrich_cooldowns
from guide_cultivation_data import load_cultivation_data
from guide_challenge_schedule import apply_challenge_schedule

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://static.nanoka.cc'
# Presentation and implementation assets do not belong in a text reference.
OMIT = {'bg_icon', 'color', 'icon', 'icon_gacha', 'element_icon', 'type_icon', 'card_icon', 'original_icon',
        'background', 'manual_background', 'asset', 'buffs', 'gameplay_tag',
        'show_detail_info', 'route_ui_id', 'display_drop_id',
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


def build_challenges(version='latest', fetcher=fetch):
    """Refresh complete challenge records without touching character or arc data."""
    if version == 'latest':
        version = fetcher(f'{BASE}/manifest.json')['nte']['latest']
    root = f'{BASE}/nte/{version}/zh/abyss'
    seasons = fetcher(f'{root}/index.json')['seasons']
    if not seasons or len({row['id'] for row in seasons}) != len(seasons):
        raise ValueError('Empty or duplicate challenge seasons')

    def load(season):
        record = fetcher(f'{root}/{season["id"]}.json')
        stages = record.get('stages', [])
        if (record.get('id') != season['id'] or not record.get('name')
                or not stages or len(stages) != record.get('stage_count')
                or len({stage['id'] for stage in stages}) != len(stages)
                or not record.get('resonance', {}).get('description')):
            raise ValueError(f'Incomplete challenge: {season["id"]}')
        record = clean(record)
        record['aliases'] = []
        return record

    with ThreadPoolExecutor(max_workers=6) as executor:
        return apply_challenge_schedule(list(executor.map(load, seasons)))


def enrich_challenge_resonances(challenges, seasons, buffs):
    """Resolve each season's explicit buff binding against the current tables."""
    updates = []
    for record in challenges:
        season = seasons[record['id']]
        identifier = season['BuffID']
        buff = buffs[identifier]
        name = clean(buff['BuffName']['LocalizedString'])
        description = clean(buff['BuffDesc']['LocalizedString'])
        if not name or not description:
            raise ValueError(f'Missing challenge resonance: {record["id"]}')
        updates.append((record, {'id': identifier, 'name': name, 'description': description}))
    for record, resonance in updates:
        record['resonance'] = resonance
    return challenges


def load_challenge_table(relative, directory=None):
    relative = Path(relative)
    if directory is not None:
        path = directory / relative
        if not path.exists():
            path = directory / relative.name
        payload = json.loads(path.read_text(encoding='utf-8'))
    else:
        payload = json.loads(subprocess.run(['gh', 'api',
            'repos/mc-ctrl/NTEData/contents/' + relative.as_posix(),
            '-H', 'Accept: application/vnd.github.raw+json'],
            check=True, capture_output=True).stdout)
    return table_rows(payload)


def load_challenge_resonances(challenges, directory=None):
    root = 'DataAssets/DataAssetSet/Abyss/'
    return enrich_challenge_resonances(challenges,
        load_challenge_table(root + 'AbyssCloneSeasonDataTable.json', directory),
        load_challenge_table(root + 'DT_AbyssCloneBuffConfig.json', directory))


def enrich_challenge_monsters(challenges, levels, pools, packs):
    """Use exact season/stage/half/wave pool bindings and fixed attribute packs."""
    updates = []
    for record in challenges:
        configured = {row['LevelID']: row for row in levels[record['id']]['LevelConfigArray']}
        for stage in record['stages']:
            halves = {row['Key'].split('::')[-1]: row['Value']['CloneSpawnMonsterConfigArray']
                      for row in configured[stage['id']]['SpawnMonsterConfigMap']}
            for half in stage['halves']:
                waves = {row['Wave']: row for row in halves[half['id']]}
                for wave in half['waves']:
                    pool_id = waves[wave['wave']]['MonsterPoolID']
                    entries = pools[pool_id]['MonsterPoolArray']
                    if len(entries) != len(wave['monsters']):
                        raise ValueError(f'Changed monster lineup: {pool_id}')
                    for monster, entry in zip(wave['monsters'], entries):
                        class_id = entry['MonsterClass']['AssetPathName'].rsplit('.', 1)[-1].removesuffix('_C')
                        if (monster['class_id'].casefold() != class_id.casefold()
                                or monster['count'] != entry['MonsterCount']
                                or monster['level'] != entry['MonsterLevel']):
                            raise ValueError(f'Changed monster binding: {pool_id}')
                        pack = packs[entry['AttributeID']]
                        for prefix in ('HPMax', 'Atk', 'Def'):
                            if pack[prefix + 'Up'] or pack[prefix + 'Add']:
                                raise ValueError(f'Unsupported monster growth: {entry["AttributeID"]}')
                        stats = dict(monster['stats'])
                        for target, source in [('hp', 'HPMaxBase'), ('attack', 'AtkBase'),
                                               ('defense', 'DefBase'), ('unbalance', 'UnbalMax')]:
                            stats[target] = pack[source]
                        stats['resistances'] = {element: pack['DamageResist' + element.title() + 'Base']
                                                for element in stats['resistances']}
                        updates.append((monster, stats))
    for monster, stats in updates:
        monster['stats'] = stats
    return challenges


def load_challenge_monsters(challenges, directory=None):
    return enrich_challenge_monsters(challenges,
        load_challenge_table('DataAssets/DataAssetSet/Abyss/AbyssCloneLevelDataTable.json', directory),
        load_challenge_table('DataAssets/DataAssetSet/Abyss/DT_AbyssMonsterPool.json', directory),
        load_challenge_table('DataTable/PackData/DT_MonsterPackData.json', directory))


def build(version, fetcher=fetch):
    if version == 'latest':
        version = fetcher(f'{BASE}/manifest.json')['nte']['latest']
    root = f'{BASE}/nte/{version}'
    characters = fetcher(f'{root}/character.json')
    weapons = fetcher(f'{root}/weapon.json')
    seasons = fetcher(f'{root}/zh/abyss/index.json')['seasons']
    item_index = fetcher(f'{root}/zh/item.json')
    item_names = {str(item['id']).casefold(): item['name'] for item in item_index.values()}
    for currency, bundled_id in (('gold', 'gold_1000'), ('fons', 'fons_100')):
        if bundled_id in item_names:
            item_names[currency] = re.sub(r'\s*[x×]\s*[\d,]+$', '', item_names[bundled_id], flags=re.I)

    def material_names(value):
        if isinstance(value, dict):
            result = {key: material_names(child) for key, child in value.items()}
            if 'items_id' in value:
                ids, amounts = value['items_id'], value.get('amount', [])
                if len(ids) != len(amounts):
                    raise ValueError('Material IDs and quantities do not match')
                result['items'] = [{'id': ident, 'name': item_names.get(str(ident).casefold(), str(ident)),
                                    'amount': amount} for ident, amount in zip(ids, amounts)]
            return result
        if isinstance(value, list):
            return [material_names(child) for child in value]
        return value
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
                    'abilities', 'awaken', 'resonance', 'nouns', 'equip_slots', 'breakthrough')
            raw = {k: raw[k] for k in keys if k in raw}
            raw['materials'] = {
                'breakthrough': raw.pop('breakthrough', []),
                'skills': [{'id': a['id'], 'name': a['name'], 'levels': a['materials']}
                           for a in raw.get('abilities', []) if a.get('materials')],
            }
            if not raw.get('equip_slots') or not raw['materials']['breakthrough']:
                raise ValueError(f'Missing character equipment or breakthrough: {identifier}')
            if not raw.get('abilities'):
                raise ValueError(f'Missing character abilities: {identifier}')
        elif kind == 'arcs':
            keys = ('id', 'name', 'rarity', 'type_id', 'type_name', 'description',
                    'stats', 'effect', 'exclusive_character_ids', 'exclusive_desc', 'materials')
            raw = {k: raw[k] for k in keys if k in raw}
            if not raw.get('effect'):
                raise ValueError(f'Missing arc effect: {identifier}')
        else:
            if len(raw.get('stages', [])) != raw.get('stage_count'):
                raise ValueError(f'Incomplete challenge stages: {identifier}')
        record = clean(material_names(raw))
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
    apply_challenge_schedule(result['challenges'])
    return result


def character_experience(payload):
    """Keep the configured level keys and XP values without shifting levels."""
    rows = table_rows(payload)
    result = []
    for level in range(1, 81):
        value = rows[f'character_Upgrade_{level}']['NeedExp']
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'Invalid character experience at level {level}')
        result.append({'level': level, 'required_exp': value})
    return result


def load_character_experience(directory=None):
    if directory is not None:
        payload = json.loads((directory / 'DT_CharacterUpgradeDataTable.json').read_text(encoding='utf-8'))
    else:
        response = subprocess.run(['gh', 'api',
            'repos/mc-ctrl/NTEData/contents/DataTable/Character/DT_CharacterUpgradeDataTable.json',
            '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True)
        payload = json.loads(response.stdout)
    return character_experience(payload)


def merge_local_characters(data, supplemental):
    """Retain explicitly imported characters and Exe's locally synchronized skill data."""
    characters = data['characters']
    positions = {str(row['id']): index for index, row in enumerate(characters)}
    for record in supplemental['characters']:
        identifier = str(record['id'])
        if identifier in positions and identifier != '1074':
            continue
        caps = supplemental['level_caps'][identifier]
        if not record.get('stats') or not record.get('materials') or caps[-1] != 80:
            raise ValueError('Incomplete locally imported character')
        if identifier in positions:
            characters[positions[identifier]] = record
        else:
            positions[identifier] = len(characters)
            characters.append(record)
        data['cultivation']['level_caps'][identifier] = caps
    return data


def merge_local_arcs(data, supplemental):
    """Preserve independently imported arcs absent from the remote index."""
    present = {str(row['id']) for row in data['arcs']}
    for record in supplemental['arcs']:
        identifier = str(record['id'])
        if identifier in present:
            continue
        if not record.get('stats') or not record.get('effect') or len(record.get('materials', [])) != 6:
            raise ValueError('Incomplete locally imported arc')
        data['arcs'].append(record)
        present.add(identifier)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='latest')
    parser.add_argument('--ntedata-dir', type=Path, help='Local directory with skill and character cultivation tables; otherwise read with gh.')
    parser.add_argument('--transport', choices=('urllib', 'curl'), default='urllib')
    parser.add_argument('--challenges-only', action='store_true', help='Refresh only challenges in an existing catalog.')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/catalog.json')
    args = parser.parse_args()
    if args.challenges_only:
        data = json.loads(args.output.read_text(encoding='utf-8'))
        challenges = build_challenges(args.version, fetch_curl if args.transport == 'curl' else fetch)
        load_challenge_resonances(challenges, args.ntedata_dir)
        load_challenge_monsters(challenges, args.ntedata_dir)
        if not {row['id'] for row in data['challenges']} <= {row['id'] for row in challenges}:
            raise ValueError('Challenge refresh would remove existing seasons')
        data['challenges'] = challenges
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        temporary.replace(args.output)
        print(f'challenges: {len(challenges)}')
        return
    tables = load_tables(args.ntedata_dir)
    data = enrich_catalog(build(args.version, fetch_curl if args.transport == 'curl' else fetch), tables)
    load_challenge_resonances(data['challenges'], args.ntedata_dir)
    load_challenge_monsters(data['challenges'], args.ntedata_dir)
    data['character_experience'] = load_character_experience(args.ntedata_dir)
    data['cultivation'] = load_cultivation_data(args.ntedata_dir)
    enrich_cooldowns(data, CooldownData(args.ntedata_dir))
    local = ROOT / 'app/modules/guide/data/ntedata_characters.json'
    if local.exists():
        merge_local_characters(data, json.loads(local.read_text(encoding='utf-8')))
    local_arcs = ROOT / 'app/modules/guide/data/ntedata_arcs.json'
    if local_arcs.exists():
        merge_local_arcs(data, json.loads(local_arcs.read_text(encoding='utf-8')))
    # Do not replace the working dataset until every detail has validated.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(', '.join(f'{kind}: {len(data[kind])}' for kind in ('characters', 'arcs', 'challenges')))


if __name__ == '__main__':
    main()
