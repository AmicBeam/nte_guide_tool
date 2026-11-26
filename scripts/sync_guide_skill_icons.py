#!/usr/bin/env python3
"""Explicit offline sync of character skill title icons."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import tempfile
from sync_guide_arc_images import download

ROOT = Path(__file__).resolve().parents[1]


def table(local, remote):
    if local:
        data = local.read_bytes()
    else:
        metadata = json.loads(subprocess.run(
            ['gh', 'api', 'repos/mc-ctrl/NTEData/contents/DataTable/' + remote],
            capture_output=True, check=True).stdout)
        data = download(metadata['download_url'])
    return next(entry['Rows'] for entry in json.loads(data) if 'Rows' in entry)


def bindings(characters, tips, city):
    result, icons = {}, {}
    tips = {key.casefold(): value for key, value in tips.items()}
    city = {key.casefold(): value for key, value in city.items()}
    for character in characters:
        skills = {}
        for ability in character['abilities']:
            identifier = ability['id']
            row = city.get(identifier.casefold(), {}) if ability['type'] == 'city' else tips.get(identifier.casefold(), {})
            path = row.get('icon', {}).get('AssetPathName', '').split('.', 1)[0]
            if not path:
                skills[identifier] = None
                continue
            if not re.fullmatch(r'/Game/[A-Za-z0-9_/-]+', path):
                raise ValueError('Invalid skill icon path')
            filename = path.rsplit('/', 1)[-1] + '.webp'
            if filename in icons and icons[filename] != path:
                raise ValueError('Conflicting skill icon filenames')
            icons[filename] = path
            skills[identifier] = '/static/guide/skill-icons/' + filename
        result[str(character['id'])] = skills
    return result, icons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tips-table', type=Path)
    parser.add_argument('--city-table', type=Path)
    args = parser.parse_args()
    characters = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())['characters']
    result, icons = bindings(characters,
        table(args.tips_table, 'Skill/DT_GameplayAbilityTipsData.json'),
        table(args.city_table, 'Character/DT_CharacterAbilityCityGroupConfig.json'))
    output = ROOT / 'app/modules/guide/static/skill-icons'
    output.mkdir(parents=True, exist_ok=True)
    missing = set()

    def save(entry):
        filename, path = entry
        target = output / filename
        if target.exists():
            data = target.read_bytes()
            if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
                return
        try:
            original = path.removeprefix('/Game/') + '.png'
            with tempfile.TemporaryDirectory(prefix='guide-skill-') as directory:
                png = Path(directory) / 'icon.png'
                webp = Path(directory) / 'icon.webp'
                for attempt in range(3):
                    response = subprocess.run(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/' + original,
                                               '--jq', '.content'], capture_output=True)
                    if response.returncode == 0 or b'404' in response.stderr:
                        break
                response.check_returncode()
                png.write_bytes(base64.b64decode(response.stdout))
                subprocess.run(['cwebp', '-quiet', '-lossless', str(png), '-o', str(webp)],
                               check=True, capture_output=True)
                data = webp.read_bytes()
        except subprocess.CalledProcessError as error:
            if b'404' not in (error.stderr or b''):
                raise RuntimeError(f'Skill icon failed: {filename}, tool={error.cmd[0]}, code={error.returncode}') from None
            missing.add(filename)
            return
        if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
            raise ValueError('Invalid skill WebP')
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(save, icons.items()))
    for skills in result.values():
        for identifier, path in skills.items():
            if path and path.rsplit('/', 1)[-1] in missing:
                skills[identifier] = None
    target = ROOT / 'app/modules/guide/data/skill_icons.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    available = sum(bool(path) for skills in result.values() for path in skills.values())
    total = sum(len(skills) for skills in result.values())
    print(f'Updated {available}/{total} skill icon bindings; {len(missing)} assets unavailable')


if __name__ == '__main__':
    main()
