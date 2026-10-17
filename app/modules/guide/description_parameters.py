"""Display-only skill-level bindings for description placeholders."""
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import json
from pathlib import Path


def number(value):
    return format(Decimal(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP), 'f').rstrip('0').rstrip('.') or '0'


@lru_cache(maxsize=1)
def awakening_damage():
    return json.loads((Path(__file__).parent / 'data/awakening_damage.json').read_text(encoding='utf-8'))


def bind_description_parameters(character):
    identifier = str(character['id'])
    if identifier == '1004':
        ability_id, index = 'GA_Lacrimosa_Melee', 0
        ability = next(a for a in character['abilities'] if a['id'] == ability_id)
        row = next(r for r in ability['param_rows'] if r['key'] == 'GA_Lacrimosa_Dot_num')
        values = [number(Decimal(v.removesuffix('%')) * 4 / 3) for v in row['values']]
        formula = '噩梦伤害倍率'
    elif identifier in awakening_damage():
        binding = awakening_damage()[identifier]
        ability_id, index = binding['ability_id'], binding['awaken_index']
        ability = next(a for a in character['abilities'] if a['id'] == ability_id)
        values = binding['values']
        formula = '追加攻击伤害倍率' if identifier == '1010' else '额外范围伤害倍率'
    else:
        return
    character['awaken'][index]['parameters'] = [{
        'id_value': 0, 'ability_id': ability_id, 'ability_name': ability['name'],
        'values': values, 'formula': formula,
    }]
