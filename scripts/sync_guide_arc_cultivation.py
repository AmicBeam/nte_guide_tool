#!/usr/bin/env python3
"""Explicit refresh of first recommended arcs and arc cultivation parameters."""
import argparse
import json
from pathlib import Path
from guide_cultivation_data import load_payload
from guide_skill_details import table_rows

ROOT = Path(__file__).resolve().parents[1]
PATHS = ['DataTable/Character/CultivationGuide/DT_CultivationGuideData.json',
         'DataTable/Fork/DT_ForkItemData.json', 'DataTable/Fork/DT_ForkUpgradeData.json',
         'DataTable/Fork/DT_ForkBreakthroughData.json', 'DataTable/Inventory/DT_ItemConfig.json']


def build(catalog, plans, forks, upgrades, breakthroughs, items):
    plans, forks, upgrades, breakthroughs, items = map(table_rows, (plans, forks, upgrades, breakthroughs, items))
    forks = {k.casefold(): v for k, v in forks.items()}
    upgrades = {k.casefold(): v for k, v in upgrades.items()}
    breakthroughs = {k.casefold(): v for k, v in breakthroughs.items()}
    arcs = {a['id'].casefold(): a for a in catalog['arcs']}
    recommendations, configs, experience = {}, {}, {}
    for character in catalog['characters']:
        entries = plans[str(character['id'])]['RecommendForkList']
        if not entries:
            continue
        arc = arcs[entries[0]['ForkItemId'].casefold()]
        recommendations[str(character['id'])] = {'id': arc['id'], 'name': arc['name']}
    for arc in catalog['arcs']:
        element = forks[arc['id'].casefold()]['ElementData']
        caps = [breakthroughs[f"{element['BreakthroughPackId']}_{i}".casefold()]['MaxForkLevel']
                for i in range(element['MaxBreakthrough'] + 1)]
        if caps != sorted(set(caps)) or len(arc['materials']) != len(caps) - 1:
            raise ValueError('Incomplete arc breakthrough data')
        pack = element['UpgradePackId'].casefold()
        if pack not in experience:
            experience[pack] = [upgrades[f'{pack}_{level}']['NeedExp'] for level in range(1, caps[-1] + 1)]
        if any(not isinstance(x, int) or x < 0 for x in experience[pack]):
            raise ValueError('Invalid arc experience')
        for stage, material in enumerate(arc['materials'], 1):
            raw = breakthroughs[f"{element['BreakthroughPackId']}_{stage}".casefold()]
            actual = {k.casefold(): int(v) for k, v in (s.split(':') for s in raw['NeedItems'].split(','))}
            expected = {i['id'].casefold(): i['amount'] for i in material['items']}
            if actual != expected or int(raw['NeedGolds'].split(':')[1]) != material['cost_gold']:
                raise ValueError('Arc material mismatch')
        configs[arc['id']] = {'level_caps': caps, 'experience_pack': pack}
    books = []
    for identifier, row in items.items():
        if identifier.casefold() in {f'weaponupmaterial_lv{i}' for i in range(1, 4)}:
            effect = row['ElementData']
            currency, amount = effect['CostGold'].split(':')
            if currency.casefold() != 'gold' or effect['EXP'] <= 0 or int(amount) < 0:
                raise ValueError('Invalid arc experience material')
            books.append({'id': identifier, 'name': row['ItemName']['LocalizedString'],
                          'exp': effect['EXP'], 'cost_gold': int(amount)})
    if not books:
        raise ValueError('No arc experience materials')
    return {'recommendations': recommendations, 'arcs': configs, 'experience': experience, 'exp_books': books}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'app/modules/guide/data/catalog.json').read_text())
    result = build(catalog, *(load_payload(path, args.tables_dir) for path in PATHS))
    target = ROOT / 'app/modules/guide/data/arc_cultivation.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(target)
    print(f"Updated {len(result['arcs'])} arcs and {len(result['recommendations'])} recommendations")


if __name__ == '__main__':
    main()
