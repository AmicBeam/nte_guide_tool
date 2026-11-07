"""Present archived Endless Trial floors as seasons without breaking old links."""
import json
from pathlib import Path


def _definitions():
    return json.loads((Path(__file__).parent / 'data/endless_seasons.json').read_text(encoding='utf-8'))


def season_records(groups):
    by_id = {group['id']: group for group in groups}
    records = []
    for definition in _definitions():
        selected = [by_id[key] for key in definition['group_ids']]
        stages = sorted((stage for group in selected for stage in group['stages']),
                        key=lambda stage: int(stage['id']))
        records.append({**{key: value for key, value in definition.items() if key != 'group_ids'},
                        'description': selected[0]['description'],
                        'floor_begin': stages[0]['id'], 'floor_end': stages[-1]['id'],
                        'stage_count': len(stages), 'stages': stages})
    return records


def legacy_record(groups, identifier):
    group = next((row for row in groups if row['id'] == identifier), None)
    if group is None:
        return None
    for definition, season in zip(_definitions(), season_records(groups)):
        if identifier in definition['group_ids']:
            return {**season, **group, 'season_id': season['id'],
                    'name': f"{season['name']} · {group['floor_begin']}–{group['floor_end']} 层"}
    return None
