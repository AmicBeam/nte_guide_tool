#!/usr/bin/env python3
"""Import explicitly supplied BossDIY configurations and their enemy artwork offline."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'app/modules/guide'


def text(value):
    if not value:
        return ''
    if isinstance(value, str):
        return re.sub(r'<[^>]*>', '', value).strip()
    return text(value.get('LocalizedString') or value.get('SourceString') or value.get('CultureInvariantString'))


def rows(path):
    data = json.loads(path.read_text())
    result = next((item['Rows'] for item in data if item.get('Rows')), None)
    if not result:
        raise ValueError(f'Empty configuration: {path.name}')
    return result



def difficulty_monsters(stage_id, difficulty_id, tables):
    """Resolve the clone's actual spawns and difficulty-bound final stat packs."""
    if tables is None:
        raise ValueError('Missing clone, spawn, monster and stat-pack tables')
    lookup = {name: {key.casefold(): value for key, value in table.items()} for name, table in tables.items()}
    clone = lookup['clones'][stage_id.casefold()]
    node = next(n for n in clone['CloneSystemSubNodes'] if str(n['DifficultyLevel']) == str(difficulty_id))
    spawn = lookup['spawns'][node['SpawnID'].casefold()]
    result = []
    elements = ('normal', 'cosmos', 'nature', 'incantation', 'chaos', 'psyche', 'lakshana', 'psychically')
    for wave_index, wave in enumerate(spawn['SpawnWaveEntriesOverride'], 1):
        for entry in wave['SpawnEntries']:
            template = entry['AITemplate']['AssetPathName'].split('.')[0].rsplit('/', 1)[-1]
            key = entry['AIConfigOverrideID']
            static = lookup['monsters'][(key if key.lower() != 'none' else template).casefold()]
            if not static['bIsCloneDifficultyLevel'] or entry['bOverrideLevelType']:
                raise ValueError('Unsupported monster level binding: ' + template)
            level = static['CloneDifficultyLevelArray'][int(difficulty_id) - 1]
            pack_id = entry['AISettingsOverrideID']
            if pack_id.lower() == 'none':
                pack_id = level['MonsterClonePropModifyID']
            pack = lookup['packs'][pack_id.casefold()]
            if any(pack[key] != 0 for key in ('HPMaxUp', 'HPMaxAdd', 'AtkUp', 'AtkAdd', 'DefUp', 'DefAdd')):
                raise ValueError('Unsupported monster growth: ' + template)
            weak_types = [item.removeprefix('UNBAL_BAR_').lower() for item in static['UnbalBarTypeList']]
            if any(element not in elements for element in weak_types):
                raise ValueError('Unknown weakness type: ' + template)
            name = text(static['TextName']) or static['Comment'].removeprefix('自定义BOSS-')
            if not name or pack['HPMaxBase'] <= 0 or level['MonsterCloneLevel'] <= 0:
                raise ValueError('Missing monster identity or stats: ' + template)
            result.append({'id': template, 'name': name, 'wave': wave_index, 'count': entry['Count'],
                'level': level['MonsterCloneLevel'], 'weaknesses': [e for e in weak_types if e != 'normal'],
                'generic_stagger': 'normal' in weak_types,
                'stats': {'hp': pack['HPMaxBase'], 'attack': pack['AtkBase'], 'defense': pack['DefBase'],
                          'resistances': {e: pack['DamageResist' + e.title() + 'Base'] for e in elements}}})
    if not result:
        raise ValueError('Empty monster lineup: ' + stage_id)
    return result


def mechanism_targets(choices, options, descriptions, scores=None):
    entries = {key.casefold(): text(value) for key, value in descriptions.items()}
    targets = []
    for choice in choices:
        key = choice['OptionID']
        option = options[key]
        if not option['OptionType'].endswith('::MechanismBuff'):
            raise ValueError('Unexpected mechanism option type: ' + key)
        if option['OptionNode'] is not None:
            raise ValueError('New mechanism node needs an explicit parser: ' + key)
        description = entries.get(key.casefold())
        if not description:
            raise ValueError('Missing mechanism description: ' + key)
        score = (scores or {}).get(key)
        if score is not None and (type(score) not in (int, float) or score < 0):
            raise ValueError('Invalid mechanism score: ' + key)
        targets.append({'id': key, 'description': description, 'score': score})
    return targets

def build(stages, options, index, calendar=None, monster_tables=None, target_texts=None, target_scores=None):
    result, images = [], {}
    for identifier, row in stages.items():
        enemies = []
        for entry in row['DifficultyDisplayData']:
            display = entry['Value']
            asset = display['BossIcon']['AssetPathName'].split('.')[0]
            if not re.fullmatch(r'/Game/[A-Za-z0-9_/-]+', asset):
                raise ValueError('Invalid artwork binding')
            image_id = asset.rsplit('/', 1)[-1]
            images[image_id] = asset.removeprefix('/Game/') + '.png'
            enemy = {'name': text(display['Desc']), 'description': text(display.get('DetailDesc')),
                     'image': f'/static/guide/diyboss/{image_id}.webp'}
            if enemy not in enemies:
                enemies.append(enemy)
        if len(row.get('PreviewActorArray', [])) > 1 and row.get('Comment'):
            for enemy in enemies:
                enemy['name'] = row['Comment']
                enemy['description'] = ''
        groups = []
        for group in row['OptionChoices']:
            choices = []
            for key in group['Options']:
                option = options[key]
                node = option['OptionNode']
                if not node:
                    raise ValueError('Missing attribute option: ' + key)
                score = node.get('Score', node.get('score'))
                if score is None:
                    raise ValueError('Missing option score')
                is_time = option['OptionType'].endswith('::TimeComplete')
                value = node['LimitSeconds'] if is_time else node['AddPercent']
                choices.append({'id': key, 'value': value, 'unit': '秒' if is_time else '%', 'score': score})
            groups.append({'name': text(group['CategoryDesc']), 'options': choices})
        targets = mechanism_targets(row.get('MechanismOptionChoices', []), options, target_texts or {}, target_scores)
        unlock = row['UnlockNode']
        result.append({'id': identifier.removeprefix('DiyBossStage'), 'name': text(row['DescName']),
            'special': row.get('bSpecialHighDifficulty', False),
            'enemy_label': row.get('Comment') or '、'.join(e['name'] for e in enemies),
            'enemies': enemies, 'unlock': text(unlock.get('UnlockDesc')) or '活动开始时解锁',
            'unlock_hours': unlock['ActiveOpenedHours'],
            'difficulties': [{'id': d['Key'], 'name': text(d['Value']['Desc']),
                'base_score': d['Value']['BaseScore'], 'score_rate': d['Value']['ScoreRate'],
                'monsters': difficulty_monsters(identifier, d['Key'], monster_tables)}
                for d in row['DifficultyMap']],
            'buff_groups': groups, 'targets': targets, 'target_score_limit': row.get('MechanismScoreLimit'),
            'rewards': [{'score': d['Value']['LimitScore'], 'drop_id': d['Value']['DropID']}
                for d in row['ScoreAwardMap']]})
    if len({s['id'] for s in result}) != len(result):
        raise ValueError('Duplicate stage IDs')
    metadata = (calendar or {}).get(f'config-{index}', {})
    return {'id': f'config-{index}', 'name': f'争锋赏宴 · 配置{index}',
        'description': '、'.join(s['name'] for s in result), 'begin_at': None, 'end_at': None,
        'stage_count': len(result), 'stages': result, **metadata}, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, action='append', required=True,
                        help='Directory containing DT_DiyBossStage.json and DT_DiyBossOptions.json; chronological order')
    parser.add_argument('--ntedata-dir', type=Path, help='Local image tree; otherwise fetch configured artwork via gh')
    parser.add_argument('--art-ref', action='append', default=[], help='Additional Git refs for removed historical artwork')
    args = parser.parse_args()
    calendar = json.loads((MODULE / 'data/diyboss_calendar.json').read_text())
    target_scores = json.loads((MODULE / 'data/diyboss_target_scores.json').read_text())
    records, bindings = [], {}
    for index, folder in enumerate(args.snapshot, 1):
        monster_tables = {key: rows(folder / filename) for key, filename in {
            'clones': 'CloneSystemDataTable.json', 'spawns': 'DT_CloneMonsterConfig.json',
            'monsters': 'DT_MonsterStaticData_BigWorld.json', 'packs': 'DT_MonsterPackData.json'}.items()}
        text_file = folder / 'ST_UI_hpy.json'
        descriptions = json.loads(text_file.read_text())[0]['StringTable']['KeysToEntries'] if text_file.exists() else {}
        record, images = build(rows(folder / 'DT_DiyBossStage.json'), rows(folder / 'DT_DiyBossOptions.json'),
                               index, calendar, monster_tables, descriptions, target_scores)
        records.append(record)
        bindings.update(images)
    output = MODULE / 'static/diyboss'
    output.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        def sync(entry):
            name, path = entry
            if args.ntedata_dir:
                raw = (args.ntedata_dir / path).read_bytes()
            else:
                for ref in ['HEAD'] + args.art_ref:
                    response = subprocess.run(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/' + path + '?ref=' + ref], capture_output=True, check=False)
                    if response.returncode == 0:
                        raw = base64.b64decode(json.loads(response.stdout)['content'])
                        break
                else:
                    raise ValueError('Missing artwork: ' + name)
            with Image.open(BytesIO(raw)) as source:
                source.convert('RGBA').save(temporary / (name + '.webp'), 'WEBP', quality=90, method=6)
            with Image.open(temporary / (name + '.webp')) as image:
                image.load()
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(sync, bindings.items()))
        for name in bindings:
            (temporary / (name + '.webp')).replace(output / (name + '.webp'))
        target = temporary / 'diyboss.json'
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        target.replace(MODULE / 'data/diyboss.json')
    print(f'Imported {len(records)} configurations, {sum(r["stage_count"] for r in records)} stages, {len(bindings)} artworks')


if __name__ == '__main__':
    main()
