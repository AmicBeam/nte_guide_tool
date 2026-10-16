#!/usr/bin/env python3
"""Explicitly synchronize structured character descriptions for local display."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import html
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STYLE_TAGS = {'guang', 'ling', 'zhou', 'an', 'hun', 'xiang', 'numgreen', 'italic',
              'title', 'texthlt', 'textexpl', 'red'}


def normalized(value):
    return html.unescape(re.sub(r'<[^>]*>', '', value.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n'))).strip()


def runs(value):
    """Translate only known game markup into inert text/style/noun segments."""
    result, stack = [], []
    for part in re.split(r'(<[^>]*>)', value):
        if part.startswith('<'):
            if re.fullmatch(r'<br\s*/?>', part, re.I):
                result.append({'text': '\n'})
            elif part.startswith('</'):
                if stack:
                    stack.pop()
            else:
                match = re.match(r'<([\w]+)', part)
                tag = match[1].lower() if match else ''
                meta = {}
                if tag in STYLE_TAGS:
                    meta['style'] = tag
                elif tag == 'hot':
                    noun = re.search(r'param=[\"\']nouns&([\w]+)[\"\']', part, re.I)
                    if noun:
                        meta = {'style': 'textexpl', 'noun': noun[1].casefold()}
                stack.append(meta)
        elif part:
            current = {}
            for meta in stack:
                current.update(meta)
            result.append(dict(current, text=html.unescape(part.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n'))))
    return result


def character_data(raw):
    descriptions = {}
    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('desc', 'description') and isinstance(child, str):
                    descriptions[normalized(child)] = runs(child)
                elif isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for key in ('abilities', 'awaken', 'resonance', 'nouns'):
        collect(raw.get(key, []))
    nouns = {str(value.get('id', key)).casefold(): {'title': normalized(value['title']),
             'runs': runs(value['desc'])} for key, value in raw.get('nouns', {}).items()}
    return {'descriptions': descriptions, 'nouns': nouns}


def fetch(url):
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '45', url],
                            check=True, capture_output=True)
    return json.loads(result.stdout)


def build(characters, style_table):
    styles = next(x['Rows'] for x in style_table if 'Rows' in x)
    colors = {key.casefold(): value['TextStyle']['ColorAndOpacity']['SpecifiedColor']['Hex']
              for key, value in styles.items() if key.casefold() in STYLE_TAGS}
    if any(not re.fullmatch(r'[\da-fA-F]{6}', color) for color in colors.values()):
        raise ValueError('Invalid text color')
    return {'colors': colors, 'characters': {str(key): character_data(raw) for key, raw in characters.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--characters-json', type=Path, help='Local ID-to-character raw JSON map')
    parser.add_argument('--style-table', type=Path, help='Local DT_RichTextStyle_Ability.json')
    parser.add_argument('--version', default='latest')
    parser.add_argument('--output', type=Path, default=ROOT / 'app/modules/guide/data/character_rich_text.json')
    args = parser.parse_args()
    if args.characters_json:
        characters = json.loads(args.characters_json.read_text())
    else:
        version = fetch('https://static.nanoka.cc/manifest.json')['nte']['latest'] if args.version == 'latest' else args.version
        identifiers = [str(c['id']) for c in json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())['characters']]
        with ThreadPoolExecutor(max_workers=6) as pool:
            characters = dict(zip(identifiers, pool.map(lambda key: fetch(f'https://static.nanoka.cc/nte/{version}/zh/character/{key}.json'), identifiers)))
    if args.style_table:
        style_table = json.loads(args.style_table.read_text())
    else:
        result = subprocess.run(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/DataTable/UIStyle/DT_RichTextStyle_Ability.json',
                                 '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True)
        style_table = json.loads(result.stdout)
    data = build(characters, style_table)
    catalog = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())
    for character in catalog['characters']:
        rich = data['characters'][str(character['id'])]
        for ability in character['abilities']:
            for group in ability.get('groups', []):
                if group.get('description') and group['description'] not in rich['descriptions']:
                    raise ValueError(f'Unmatched skill description: {character["id"]} / {group["name"]}')
        for segments in list(rich['descriptions'].values()) + [n['runs'] for n in rich['nouns'].values()]:
            for run in segments:
                if run.get('noun') and run['noun'] not in rich['nouns']:
                    raise ValueError(f'Unresolved noun: {run["noun"]}')
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(args.output)
    print(f'Updated structured descriptions for {len(data["characters"])} characters')


if __name__ == '__main__':
    main()
