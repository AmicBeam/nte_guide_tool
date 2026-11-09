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


def build(stages, options, index):
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
        targets = []
        for choice in row.get('MechanismOptionChoices', []):
            key = choice['OptionID']
            node = options[key]['OptionNode']
            # Client exports currently omit these nodes. Preserve unknowns explicitly.
            if node is not None:
                raise ValueError('New mechanism node needs an explicit parser: ' + key)
            targets.append({'id': key, 'description': None, 'score': None})
        unlock = row['UnlockNode']
        result.append({'id': identifier.removeprefix('DiyBossStage'), 'name': text(row['DescName']),
            'special': row.get('bSpecialHighDifficulty', False),
            'enemy_label': row.get('Comment') or '、'.join(e['name'] for e in enemies),
            'enemies': enemies, 'unlock': text(unlock.get('UnlockDesc')) or '活动开始时解锁',
            'unlock_hours': unlock['ActiveOpenedHours'],
            'difficulties': [{'id': d['Key'], 'name': text(d['Value']['Desc']),
                'base_score': d['Value']['BaseScore'], 'score_rate': d['Value']['ScoreRate']}
                for d in row['DifficultyMap']],
            'buff_groups': groups, 'targets': targets, 'target_score_limit': row.get('MechanismScoreLimit'),
            'rewards': [{'score': d['Value']['LimitScore'], 'drop_id': d['Value']['DropID']}
                for d in row['ScoreAwardMap']]})
    if len({s['id'] for s in result}) != len(result):
        raise ValueError('Duplicate stage IDs')
    return {'id': f'config-{index}', 'name': f'争锋赏宴 · 配置{index}',
        'description': '、'.join(s['name'] for s in result), 'begin_at': None, 'end_at': None,
        'stage_count': len(result), 'stages': result}, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, action='append', required=True,
                        help='Directory containing DT_DiyBossStage.json and DT_DiyBossOptions.json; chronological order')
    parser.add_argument('--ntedata-dir', type=Path, help='Local image tree; otherwise fetch configured artwork via gh')
    parser.add_argument('--art-ref', action='append', default=[], help='Additional Git refs for removed historical artwork')
    args = parser.parse_args()
    records, bindings = [], {}
    for index, folder in enumerate(args.snapshot, 1):
        record, images = build(rows(folder / 'DT_DiyBossStage.json'), rows(folder / 'DT_DiyBossOptions.json'), index)
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
