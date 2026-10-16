#!/usr/bin/env python3
"""Refresh local character links to their configured recommended cartridge."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def rows(path, remote):
    raw = path.read_bytes() if path else subprocess.run(
        ['gh', 'api', 'repos/mc-ctrl/NTEData/contents/DataTable/Equipment/' + remote,
         '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True).stdout
    return next(entry['Rows'] for entry in json.loads(raw) if 'Rows' in entry)


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def build(characters, cartridges, plans, equipment):
    result = {}
    for character in characters:
        identifier = str(character['id'])
        plan = plans.get(identifier)
        recommendations = []
        if plan and plan.get('CoreID'):
            core = plan['CoreID']
            family = normalized(core.rsplit('_', 1)[0])
            matches = [card for card in cartridges if normalized(card['id']) == family]
            if len(matches) != 1:
                raise ValueError(f'Unmapped recommended cartridge: {identifier}/{core}')
            card = matches[0]
            name = equipment[core]['ItemName']['LocalizedString']
            if name != card['name']:
                raise ValueError(f'Cartridge name mismatch: {identifier}/{core}')
            recommendations.append({'id': card['id'], 'name': name.strip('「」')})
        result[identifier] = {'character_id': identifier, 'recommendations': recommendations}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-table', type=Path)
    parser.add_argument('--equipment-table', type=Path)
    args = parser.parse_args()
    characters = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())['characters']
    cartridges = json.loads((ROOT / 'app/modules/kongmu/static/data/cartridges.json').read_text())['cartridges']
    result = build(characters, cartridges, rows(args.plan_table, 'DT_EquipmentPlanData.json'),
                   rows(args.equipment_table, 'DT_Equipment.json'))
    target = ROOT / 'app/modules/guide/data/kongmu_links.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(f'Updated {len(result)} character links')


if __name__ == '__main__':
    main()
