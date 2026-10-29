"""Resolve base skill cooldowns from ability defaults and inherited NTEData curves."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_COOLDOWN_LABELS = {'冷却时间', '技能冷却时间', '基础冷却时间'}


def remote_json(path):
    return json.loads(subprocess.check_output([
        'gh', 'api', f'repos/mc-ctrl/NTEData/{path}',
        '-H', 'Accept: application/vnd.github.raw+json',
    ]))


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid base skill cooldown')
    return int(value) if value == int(value) else value


class CooldownData:
    def __init__(self, directory=None, fetcher=remote_json):
        self.directory = Path(directory).resolve() if directory is not None else None
        self.fetcher = fetcher

    @lru_cache(maxsize=None)
    def asset(self, path):
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or '..' in candidate.parts or not path.endswith('.json'):
            raise ValueError('Invalid ability asset path')
        if self.directory is None:
            return self.fetcher(f'contents/{path}')
        for file in (self.directory / path, self.directory / candidate.name):
            if file.is_file():
                if not file.resolve().is_relative_to(self.directory):
                    raise ValueError('Ability asset is outside the data directory')
                return json.loads(file.read_text(encoding='utf-8'))
        raise ValueError(f'Missing ability asset: {path}')

    @lru_cache(maxsize=1)
    def ability_index(self):
        if self.directory is not None:
            paths = [p.relative_to(self.directory).as_posix()
                     for p in self.directory.rglob('GA_*.json') if p.is_file()]
        else:
            entries = self.fetcher('contents/Blueprints/Abilities')
            player = next(entry for entry in entries if entry['name'] == 'Player')
            tree = self.fetcher(f'git/trees/{player["sha"]}?recursive=1')
            if tree.get('truncated'):
                raise ValueError('Incomplete player ability index')
            paths = ['Blueprints/Abilities/Player/' + entry['path']
                     for entry in tree['tree'] if entry['type'] == 'blob' and entry['path'].endswith('.json')]
        result = {}
        for path in paths:
            identifier = PurePosixPath(path).stem
            result.setdefault(identifier, []).append(path)
        return result

    def resolve(self, identifier):
        paths = self.ability_index().get(identifier, [])
        if not paths:
            return None
        if len(paths) != 1:
            raise ValueError(f'Ambiguous ability asset: {identifier}')
        path = paths[0]
        seen = set()
        while path:
            if path in seen or len(seen) >= 16:
                raise ValueError('Cyclic ability inheritance')
            seen.add(path)
            payload = self.asset(path)
            default = next((row for row in payload
                            if row.get('Name') == f'Default__{PurePosixPath(path).stem}_C'), None)
            if default is None:
                raise ValueError(f'Missing ability default: {path}')
            properties = default.get('Properties') or {}
            if 'CooldownDuration' in properties:
                return self.scalable_seconds(properties['CooldownDuration'])
            blueprint = next((row for row in payload if row.get('Type') == 'BlueprintGeneratedClass'), {})
            parent = (blueprint.get('Super') or {}).get('ObjectPath', '')
            path = parent.removeprefix('/Game/').split('.')[0] + '.json' if parent.startswith('/Game/') else None
        return None

    def scalable_seconds(self, duration):
        if not isinstance(duration, dict):
            raise ValueError('Invalid cooldown duration')
        curve = duration.get('Curve') or {}
        reference = (curve.get('CurveTable') or {}).get('ObjectPath', '')
        row_name = curve.get('RowName')
        if reference and row_name and row_name != 'None':
            if not reference.startswith('/Game/'):
                raise ValueError('Unsupported cooldown curve reference')
            path = reference.removeprefix('/Game/').split('.')[0] + '.json'
            payload = self.asset(path)
            rows = next(row['Rows'] for row in payload if isinstance(row.get('Rows'), dict))
            if row_name not in rows:
                raise ValueError(f'Missing cooldown curve: {row_name}')
            points = rows[row_name].get('Keys') or []
            values = {seconds(point['Value']) for point in points}
            if not values:
                return None
            if len(values) != 1:
                raise ValueError('Level-dependent cooldown requires an explicit level display')
            return seconds(next(iter(values)) * seconds(duration.get('Value', 1)))
        return seconds(duration['Value']) if 'Value' in duration else None


def enrich_cooldowns(data, source):
    for character in data['characters']:
        for ability in character['abilities']:
            if ability.get('type_name') not in ('Skill', 'UltraSkill', 'Peculiarity'):
                continue
            value = source.resolve(ability['id'])
            if value is None and ability.get('type_name') == 'Peculiarity':
                ability.pop('base_cooldown', None)
                continue
            ability['base_cooldown'] = value
            # Keep existing parameter rows consistent, including their group copies.
            for rows in [ability.get('param_rows') or [], *[g.get('param_rows') or [] for g in ability.get('groups') or []]]:
                for row in rows:
                    if row.get('label', '').strip() in BASE_COOLDOWN_LABELS:
                        row['values'] = [f'{value:g}秒' for _ in row.get('values') or []] if value is not None else []
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ntedata-dir', type=Path)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'app/modules/guide/data/catalog.json')
    args = parser.parse_args()
    data = json.loads(args.catalog.read_text(encoding='utf-8'))
    enrich_cooldowns(data, CooldownData(args.ntedata_dir))
    temporary = args.catalog.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(args.catalog)
    abilities = [a for c in data['characters'] for a in c['abilities'] if 'base_cooldown' in a]
    print(f'Cooldowns: {sum(a["base_cooldown"] is not None for a in abilities)}/{len(abilities)}')


if __name__ == '__main__':
    main()
