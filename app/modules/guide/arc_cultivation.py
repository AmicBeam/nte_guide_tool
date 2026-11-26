"""Offline arc upgrade costs and combined character/arc plans."""
from functools import lru_cache
import json
from pathlib import Path

from . import cultivation, service


@lru_cache(maxsize=1)
def config():
    return json.loads((Path(__file__).parent / 'data/arc_cultivation.json').read_text(encoding='utf-8'))


def recommended(character):
    return config()['recommendations'].get(str(character['id']))


def choices(character):
    return [arc for arc in service.catalog()['arcs'] if str(arc['type_id']) == str(character['arc_type']['id'])]


def defaults(character, character_config):
    values = cultivation.defaults(character, character_config)
    first = recommended(character)
    values.update(include_arc=False, arc_id=first['id'] if first else '', arc_current_level=1,
                  arc_target_level=80, arc_current_ascended=False, arc_target_ascended=False)
    return values


def enabled(value):
    if value not in (False, True, '0', '1') or not isinstance(value, (str, int, bool)):
        raise ValueError('是否一并计算弧盘须为 0 或 1')
    return value in (True, '1')


def calculate_arc(arc, values):
    data = config()
    binding = data['arcs'][arc['id']]
    stages = [{'items': row['items'] + [{'id': 'gold', 'name': '甲硬币', 'amount': row['cost_gold']}]}
              for row in arc['materials']]
    record = {'id': arc['id'], 'abilities': [], 'materials': {'skills': [], 'breakthrough': stages}}
    settings = {'level_caps': {arc['id']: binding['level_caps']}, 'exp_books': data['exp_books']}
    xp = [{'level': i + 1, 'required_exp': amount}
          for i, amount in enumerate(data['experience'][binding['experience_pack']])]
    fields = {name: values.get('arc_' + name, default) for name, default in {
        'current_level': 1, 'target_level': binding['level_caps'][-1],
        'current_ascended': False, 'target_ascended': False}.items()}
    result = cultivation.calculate(record, settings, xp, fields)
    result.update(id=arc['id'], name=arc['name'])
    return result


def calculate(character, character_config, experience, values):
    result = cultivation.calculate(character, character_config, experience, values)
    if character.get('content_released') is False:
        result.update(skills_included=False, release_at=character['release_at'])
    result['state']['include_arc'] = enabled(values.get('include_arc', False))
    if not result['state']['include_arc']:
        return result
    arc = next((arc for arc in choices(character) if arc['id'] == values.get('arc_id')), None)
    if arc is None:
        raise ValueError('请选择该角色可装备的弧盘')
    arc_result = calculate_arc(arc, values)
    result['arc'] = arc_result
    result['state'].update({'arc_id': arc['id'], **{'arc_' + key: value for key, value in arc_result['state'].items()}})
    totals = {item['id'].casefold(): dict(item) for item in result['items']}
    for item in arc_result['items']:
        key = item['id'].casefold()
        if key not in totals:
            totals[key] = dict(item)
        else:
            totals[key]['amount'] += item['amount']
    books = sorted(character_config['exp_books'], key=lambda b: b['exp'], reverse=True) + sorted(config()['exp_books'], key=lambda b: b['exp'], reverse=True)
    order = {b['id'].casefold(): i for i, b in enumerate(books)}
    def sort_key(item):
        key = item['id'].casefold()
        return (0 if key == 'gold' else 1 if key in order else 2, order.get(key, 0), item['name'])
    result['items'] = sorted(totals.values(), key=sort_key)
    result['total_gold'] += arc_result['total_gold']
    result['breakdown'].extend({'name': f"弧盘 · {arc['name']} · {entry['name']}", 'items': entry['items']}
                               for entry in arc_result['breakdown'])
    return result
