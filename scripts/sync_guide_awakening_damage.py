"""Explicitly update the two awakening damage arrays from a local damage table."""
import argparse
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(table):
    rows = table[0]['Rows'] if isinstance(table, list) else table
    path = ROOT / 'app/modules/guide/data/awakening_damage.json'
    bindings = json.loads(path.read_text(encoding='utf-8'))
    for binding in bindings.values():
        values = rows[binding['damage_effect_id']]['AtkRateBaseArray']
        if len(values) != 13 or any(not isinstance(v, (int, float)) for v in values):
            raise ValueError('Expected 13 numeric damage levels')
        binding['values'] = [format(Decimal(str(v)) * 100, '.2f').rstrip('0').rstrip('.') for v in values]
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(bindings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--damage-table', type=Path, required=True)
    args = parser.parse_args()
    update(json.loads(args.damage_table.read_text(encoding='utf-8')))
